import argparse
import json
import re
import sys
import time
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from contextlib import nullcontext
import random
import gc

# Import selectors
from din_selectors.din_selectors import (
    dispatch_select_din,
    select_din_topk_strength,
)

# Progress bar
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# NEW: dataset validators
from validator.dataset_validators import *
from src.bsr import *

# -----------------------------
# Utils
# -----------------------------
def vprint(*a, verbose=False, **k):
    if verbose:
        print(*a, **k)

def norm_text(s: str) -> str:
    return re.sub(r"\s+"," ", s.strip().lower())

def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            items = data
        else:
            # if dict-of-lists, take the first list value
            for _, v in data.items():
                if isinstance(v, list):
                    items = v
                    break
    else:
        raise ValueError("file must be .json or .jsonl")
    return items

def get_text(ex: Dict[str, Any]) -> str:
    return ex.get("context") or ex.get("passage") or ex.get("question") or ex.get("text") or ex.get("premises") or ""

def get_question(ex: Dict[str, Any]) -> str:
    return ex.get("question") or ex.get("input") or ex.get("conclusion") or ""

def extract_options_from_example(ex: Dict[str, Any], validator: Optional[BaseValidator] = None) -> Optional[List[str]]:
    # try structured options first
    if "options" in ex and isinstance(ex["options"], list) and ex["options"]:
        return ex["options"]
    if "choices" in ex and isinstance(ex["choices"], list) and ex["choices"]:
        return ex["choices"]
    # common A,B,C,D,E keys
    out = []
    for k in ["A","B","C","D","E","a","b","c","d","e"]:
        if k in ex and isinstance(ex[k], str) and ex[k].strip():
            out.append(f"{k.upper()}) {ex[k].strip()}")
    if out:
        return out
    # ask validator to do a best-effort extraction
    if validator is not None:
        return validator.default_options(ex)
    return None


# -----------------------------
# HF hidden -> DIN vec
# -----------------------------
@torch.no_grad()
def text_to_din_vec(
    text: str,
    tok: AutoTokenizer,
    hf_model: AutoModelForCausalLM,
    din_spec_signed_layers: Dict[int, List[int]],
    token_mean: bool = True
) -> np.ndarray:
    device = next(hf_model.parameters()).device
    enc = tok(text, return_tensors="pt", add_special_tokens=True).to(device)
    out = hf_model(**enc, output_hidden_states=True, use_cache=False)
    hidden_states: Tuple[torch.Tensor, ...] = out.hidden_states
    num_layers = len(hidden_states) - 1

    parts: List[np.ndarray] = []
    for layer_k, idx_list in din_spec_signed_layers.items():
        L = num_layers + layer_k if layer_k < 0 else layer_k
        if L < 0 or L >= len(hidden_states):
            continue
        H = hidden_states[L]  # (1, seq, d)
        H = H.mean(dim=1) if token_mean else H[:, -1, :]
        H = H.squeeze(0)      # (d,)
        inds = [i for i in idx_list if 0 <= i < H.shape[-1]]
        if not inds:
            continue
        v = H[inds].detach().float().cpu().numpy()
        parts.append(v)
    if not parts:
        return np.zeros([1], dtype=np.float32)
    return np.concatenate(parts, axis=0)

@torch.no_grad()
def batch_token_means(
    texts: List[str],
    tok: AutoTokenizer,
    hf_model: AutoModelForCausalLM,
    layer_signed: List[int],
    batch_size: int = 8,
    verbose: bool = False,
    desc: str = "HF forward",
) -> Dict[int, np.ndarray]:
    device = next(hf_model.parameters()).device
    mats: Dict[int, List[np.ndarray]] = {L: [] for L in layer_signed}

    enc0 = tok("dummy", return_tensors="pt").to(device)
    out0 = hf_model(**enc0, output_hidden_states=True, use_cache=False)
    num_layers = len(out0.hidden_states) - 1

    rng = range(0, len(texts), batch_size)
    if verbose:
        rng = tqdm(rng, total=(len(texts)+batch_size-1)//batch_size, desc=desc)

    for i in rng:
        batch = texts[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        out = hf_model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        for Ls in layer_signed:
            L = num_layers + Ls if Ls < 0 else Ls
            if L < 0 or L >= len(hs):
                continue
            H = hs[L]  # (B, seq, d)
            H = H.mean(dim=1)  # (B, d)
            mats[Ls].append(H.detach().float().cpu().numpy())

    out_dict: Dict[int, np.ndarray] = {}
    for Ls, parts in mats.items():
        if parts:
            out_dict[Ls] = np.concatenate(parts, axis=0)
        else:
            out_dict[Ls] = np.zeros((0, 0), dtype=np.float32)
    return out_dict


# -----------------------------
# DIN computation with pluggable selectors
# -----------------------------
def compute_din_from_corpora(
    tok: AutoTokenizer,
    hf_model: AutoModelForCausalLM,
    src_texts: List[str],
    tgt_texts: List[str],
    layer_signed: List[int],
    tau: float = 1.0,
    k_ratio: float = 0.05,
    batch_size: int = 8,
    verbose: bool = False,
    balance: str = "min",
    zs_mode: str = "pooled",
    seed: int = 42,
    primary_method: str = "same_sign",
    primary_kwargs: Optional[Dict[str, Any]] = None,
    fallback_method: str = "topk_strength",
    fallback_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[int, List[int]]:
    import random
    rng = np.random.default_rng(seed)
    random.seed(seed)

    primary_kwargs = dict(primary_kwargs or {})
    fallback_kwargs = dict(fallback_kwargs or {})

    # -------- balance samples --------
    S = list(src_texts); T = list(tgt_texts)
    if balance in ("min","tgt"):
        if balance == "min":
            m = min(len(S), len(T))
            if len(S) > m: S = random.sample(S, m)
            if len(T) > m: T = random.sample(T, m)
        elif balance == "tgt":
            if len(S) > len(T): S = random.sample(S, len(T))
    vprint(f"[DIN] After balance={balance}: nS={len(S)}, nT={len(T)}", verbose=verbose)

    # -------- token-mean mats per layer --------
    Hs_dict = batch_token_means(S, tok, hf_model, layer_signed, batch_size=batch_size,
                                verbose=verbose, desc="SRC hidden (token-mean)")
    Ht_dict = batch_token_means(T, tok, hf_model, layer_signed, batch_size=batch_size,
                                verbose=verbose, desc="TGT hidden (token-mean)")

    din_spec: Dict[int, List[int]] = {}

    for Ls in tqdm(layer_signed, desc="Select DIN per layer", disable=not verbose):
        Hs = Hs_dict[Ls]; Ht = Ht_dict[Ls]
        if Hs.size == 0 and Ht.size == 0:
            din_spec[Ls] = []
            vprint(f"[DIN][Layer {Ls}] EMPTY: no hidden.", verbose=verbose)
            continue
        if Hs.size == 0:
            d = Ht.shape[1]; Hs = np.zeros((0, d), dtype=np.float32)
        if Ht.size == 0:
            d = Hs.shape[1]; Ht = np.zeros((0, d), dtype=np.float32)
        d = Hs.shape[1]

        # -------- z-score statistics --------
        if zs_mode == "union_weighted":
            Hu = np.concatenate([Hs, Ht], axis=0)
            mu = Hu.mean(axis=0); sigma = Hu.std(axis=0) + 1e-6
        elif zs_mode == "union_equal":
            mu = 0.5 * (Hs.mean(axis=0) + Ht.mean(axis=0))
            sigma = np.sqrt(0.5 * (Hs.var(axis=0) + Ht.var(axis=0))) + 1e-6
        elif zs_mode == "pooled":
            muS = Hs.mean(axis=0); muT = Ht.mean(axis=0)
            varS = Hs.var(axis=0); varT = Ht.var(axis=0)
            mu = 0.5 * (muS + muT)
            sigma = np.sqrt(0.5 * (varS + varT)) + 1e-6
        else:
            raise ValueError(f"Unknown zs_mode={zs_mode}")

        zS = ((Hs - mu) / sigma).mean(axis=0) if Hs.shape[0] > 0 else np.zeros(d, dtype=np.float32)
        zT = ((Ht - mu) / sigma).mean(axis=0) if Ht.shape[0] > 0 else np.zeros(d, dtype=np.float32)

        # -------- primary selection --------
        chosen: List[int] = []
        try:
            if primary_method == "same_sign":
                chosen = dispatch_select_din("same_sign", zS=zS, zT=zT, tau=tau, k_ratio=k_ratio, **primary_kwargs)
            elif primary_method in {"topk_strength","intersect","rank_agg","soft_sign","stability_sign"}:
                kwargs = dict(k_ratio=k_ratio, **primary_kwargs)
                if primary_method in {"soft_sign","stability_sign"} and "tau" not in kwargs:
                    kwargs["tau"] = tau
                chosen = dispatch_select_din(primary_method, zS=zS, zT=zT, **kwargs)
            elif primary_method in {"maha","fisher","bootstrap"}:
                kwargs = dict(k_ratio=k_ratio, **primary_kwargs)
                chosen = dispatch_select_din(primary_method, Hs=Hs, Ht=Ht, **kwargs)
            else:
                raise ValueError(f"Unknown primary_method={primary_method}")
        except Exception as e:
            vprint(f"[DIN][Layer {Ls}] primary_method error: {e}", verbose=verbose)
            chosen = []

        # -------- fallback if needed --------
        if len(chosen) == 0:
            try:
                if fallback_method in {"same_sign","topk_strength","intersect","rank_agg","soft_sign","stability_sign"}:
                    kwargs = dict(k_ratio=k_ratio, **fallback_kwargs)
                    if fallback_method in {"same_sign","soft_sign","stability_sign"} and "tau" not in kwargs:
                        kwargs["tau"] = tau
                    chosen = dispatch_select_din(fallback_method, zS=zS, zT=zT, **kwargs)
                elif fallback_method in {"maha","fisher","bootstrap"}:
                    kwargs = dict(k_ratio=k_ratio, **fallback_kwargs)
                    chosen = dispatch_select_din(fallback_method, Hs=Hs, Ht=Ht, **kwargs)
                else:
                    chosen = select_din_topk_strength(zS, zT, k_ratio=k_ratio)
                vprint(f"[DIN][Layer {Ls}] Fallback ({fallback_method}) used: {len(chosen)}/{d}", verbose=verbose)
            except Exception as e:
                vprint(f"[DIN][Layer {Ls}] fallback_method error: {e}", verbose=verbose)
                chosen = select_din_topk_strength(zS, zT, k_ratio=k_ratio)
                vprint(f"[DIN][Layer {Ls}] Emergency fallback=topk_strength used.", verbose=verbose)

        din_spec[Ls] = [int(i) for i in chosen]

    vprint("[DIN] Done.", verbose=verbose)
    return din_spec


# -----------------------------
# Retrieval (cosine + optional MMR)
# -----------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + eps)
    return a_norm @ b_norm.T

def mmr(query_vec: np.ndarray, cand_vecs: np.ndarray, lam: float = 0.7, topk: int = 2) -> List[int]:
    sims = cosine_sim(query_vec[None, :], cand_vecs)[0]
    selected: List[int] = []
    cand = set(range(cand_vecs.shape[0]))
    for _ in range(min(topk, cand_vecs.shape[0])):
        if not selected:
            i = int(np.argmax(sims)); selected.append(i); cand.remove(i); continue
        div = np.array([max(cosine_sim(cand_vecs[j][None, :], cand_vecs[selected])[0]) for j in cand])
        rel = np.array([sims[j] for j in cand])
        score = lam * rel - (1 - lam) * div
        j_best = list(cand)[int(np.argmax(score))]
        selected.append(j_best); cand.remove(j_best)
    return selected


def label_to_bool(label: Any, validator=None) -> Optional[bool]:
    """
    Convert a dataset-specific label to boolean if applicable.
    Delegates to validator.normalize_label() when available.
    Returns True/False for binary tasks, else None.
    """
    if label is None:
        return None

    # Step 1: 让 validator 标准化
    if validator is not None:
        norm = validator.normalize_label(label)
    else:
        norm = str(label).strip().lower()

    if norm is None:
        return None
    s = str(norm).strip().lower()

    # Step 2: 根据 validator 类型决定映射逻辑
    if isinstance(validator, (ProntoQAValidator, BoolQValidator, StrategyQAValidator)):
        # 这些任务的 normalize_label 已经返回 "A"/"B"
        if s in {"a", "true", "yes"}:
            return True
        if s in {"b", "false", "no"}:
            return False
    # 其他任务返回 None，说明不适用布尔化
    return None


def exemplar_to_block(ex: Dict[str, Any], validator=None) -> str:
    """Render few-shot exemplar.
    - For prontoqa/folio, include Context.
    - For others (e.g., gsm8k), omit Context.
    - Include Question/Reasoning only when non-empty.
    """
    ctx = (ex.get("context") or ex.get("passage") or ex.get("text") or "").strip()
    q   = (ex.get("question") or ex.get("input") or ex.get("conclusion") or "").strip()
    rationale = (ex.get("rationale") or ex.get("reasoning") or ex.get("cot") or ex.get("response") or "").strip()

    vname = (getattr(validator, "name", "") or "").lower()

    parts = []
    if ctx and vname not in {"prontoqa", "folio"}:
        parts.append(f"Context:\n{ctx}")
    if q:
        parts.append(f"Question:\n{q}")
    if rationale:
        parts.append(f"Reasoning:\n{rationale}")

    return "\n".join(parts)



def build_prompt_with_exemplars(
    tokenizer,
    validator: BaseValidator,
    query_ctx: str,
    question: str,
    exemplars: List[Dict[str, Any]],
    icl_shots: int = 0,
    options: Optional[List[str]] = None,
) -> str:

    if validator.name in ["prontoqa", "folio"] and icl_shots >= 2 and not exemplars:
        # When no retrieved exemplars provided, prefix with EX1/EX2 as header
        fs_header = EX1 + "\n------\n" + EX2 + "\n------\n"
        msgs = [
            {"role": "system", "content": validator.system_instruction()},
            {"role": "user",   "content":
                fs_header +
                f"Context:\n{query_ctx}\n\n" +
                (f"Question:\n{question}\n\n" if question else "") +
                (("Options:\n" + "\n".join(options) + "\n\n") if options else "") +
                "Reasoning:"
            },
        ]
        if hasattr(tokenizer, "apple_chat_template"):
            return tokenizer.apple_chat_template(msgs, tokenize=False, add_generation_prompt=True)  
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    
    elif validator.name in ["prontoqa", "folio"] and exemplars:
        fs_text = "\n------\n".join([exemplar_to_block(e, validator) for e in exemplars]) + "\n------\n"
        msgs = [
            {"role": "system", "content": validator.system_instruction()},
            {"role": "user",   "content":
                fs_text +
                f"Context:\n{query_ctx}\n\n" +
                (f"Question:\n{question}\n\n" if question else "") +
                (("Options:\n" + "\n".join(options) + "\n\n") if options else "") +
                "Reasoning:"
            },
        ]
        if hasattr(tokenizer, "apple_chat_template"):
            return tokenizer.apple_chat_template(msgs, tokenize=False, add_generation_prompt=True)  
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    
    elif validator.name in ["gsm8k"] and exemplars:
        fs_text = "\n------\n".join([exemplar_to_block(e, validator) for e in exemplars]) + "\n------\n"
        msgs = [
            {"role": "system", "content": validator.system_instruction()},
            {"role": "user",   "content":
                fs_text +
                (f"Question:\n{question}\n\n" if question else "") +
                (("Options:\n" + "\n".join(options) + "\n\n") if options else "") +
                "Reasoning:"
            },
        ]
        if hasattr(tokenizer, "apple_chat_template"):
            return tokenizer.apple_chat_template(msgs, tokenize=False, add_generation_prompt=True)  
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # Otherwise delegate to validator
    return validator.build_prompt(
        tokenizer,
        query_context=query_ctx,
        question=question,
        exemplars=exemplars,
        icl_shots=icl_shots,
        options=options
    )


# -----------------------------
# vLLM inference
# -----------------------------
def run_vllm_generate(
    model_name: str,
    prompts: List[str],
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
    tensor_parallel_size: int = 1,
    dtype: str = "bfloat16",
    verbose: bool = False,
    llm: "LLM" = None,
):
    sp = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=None,
    )
    created_here = False
    if llm is None:
        created_here = True
        vprint(f"[vLLM] Loading model: {model_name}", verbose=verbose)
        llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            max_model_len=16384
        )
    vprint(f"[vLLM] Generating for {len(prompts)} prompts...", verbose=verbose)
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    dt = time.time() - t0
    vprint(f"[vLLM] Done. Elapsed: {dt:.2f}s", verbose=verbose)
    texts = [o.outputs[0].text for o in outputs]
    return texts, llm


# -----------------------------
# CLI helpers
# -----------------------------
def parse_layers(s: str) -> List[int]:
    if not s:
        return [-6,-5,-4,-3,-2,-1]
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_json_arg(s: Optional[str]) -> Dict[str, Any]:
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception as e:
        raise ValueError(f"Bad JSON for kwargs: {s}\nError: {e}")