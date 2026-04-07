#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    select_din_same_sign,
    select_din_topk_strength,
)

# tqdm (nice progress bars)
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

YES_SET = {"yes","y","true","correct","a"}
NO_SET  = {"no","n","false","incorrect","b"}

# =============================
# User-specified prompt schema
# =============================
SYSTEM_INSTR = (
    "You are a careful reasoner. Think step by step with concise chain-of-thought. "
    "Then on a new line, output exactly: 'Final answer: A' or 'Final answer: B'."
)

# Canonical A/B exemplars (used when icl_shots>=2)
EX1 = """Given a problem statement as contexts, the task is to answer a logical reasoning question. 
------
Context:
Jompuses are not shy. Jompuses are yumpuses. Each yumpus is aggressive. Each yumpus is a dumpus. Dumpuses are not wooden. Dumpuses are wumpuses. Wumpuses are red. Every wumpus is an impus. Each impus is opaque. Impuses are tumpuses. Numpuses are sour. Tumpuses are not sour. Tumpuses are vumpuses. Vumpuses are earthy. Every vumpus is a zumpus. Zumpuses are small. Zumpuses are rompuses. Max is a yumpus.

Question:
Is the following statement true or false? Max is sour.

Options:
A) True
B) False

Reasoning:
Max is a yumpus. Each yumpus is a dumpus. So Max is a dumpus. Dumpuses are wumpuses. So Max is a wumpus. Every wumpus is an impus. So Max is an impus. Impuses are tumpuses. So Max is a tumpus. Tumpuses are not sour. Therefore Max is not sour.

Final answer: B"""

EX2 = """Given a problem statement as contexts, the task is to answer a logical reasoning question. 
------
Context:
Every tumpus is not angry. Tumpuses are rompuses. Every numpus is not bright. Rompuses are not luminous. Rompuses are yumpuses. Yumpuses are transparent. Yumpuses are zumpuses. Each zumpus is not bitter. Zumpuses are impuses. Impuses are red. Each impus is a dumpus. Every dumpus is happy. Each dumpus is a vumpus. Vumpuses are bright. Every vumpus is a jompus. Jompuses are large. Each jompus is a wumpus. Stella is a yumpus.

Question:
Is the following statement true or false? Stella is bright.

Options:
A) True
B) False

Reasoning:
Stella is a yumpus. Yumpuses are zumpuses, so Stella is a zumpus. Zumpuses are impuses, so Stella is an impus. Each impus is a dumpus, thus Stella is a dumpus. Every dumpus is a vumpus; vumpuses are bright. Therefore Stella is bright.

Final answer: A"""

def extract_final_ab(s: str) -> Optional[str]:
    t = norm_text(s)
    m = re.search(r"final\s*answer\s*:\s*([ab])\b", t)
    if m: return m.group(1).upper()
    tf = re.search(r"\b(true|false)\b", t)
    if tf: return "A" if tf.group(1) == "true" else "B"
    yes_m = re.search(r"\b(yes|true|correct|a)\b", t)
    no_m  = re.search(r"\b(no|false|incorrect|b)\b", t)
    if yes_m and (not no_m or yes_m.start() < no_m.start()): return "A"
    if no_m  and (not yes_m or no_m.start()  < yes_m.start()): return "B"
    return None

def ab_to_bool(label_ab: Optional[str]) -> Optional[bool]:
    if label_ab is None: return None
    return True if label_ab.upper() == "A" else False if label_ab.upper() == "B" else None

def get_gold_bool(example: Dict[str, Any]) -> Optional[bool]:
    for k in ["answer", "label", "gold", "target"]:
        if k in example:
            return label_to_bool(example[k])
    return None

def build_messages_from_fields(context: str, question: str, options: list, icl_shots: int = 0) -> list:
    fewshot_block = ""
    if icl_shots >= 2:
        fewshot_block = EX1 + "\n------\n" + EX2 + "\n------\n"
    if not options or not isinstance(options, list) or len(options) < 2:
        options = ["A) True", "B) False"]
    options_str = "\n".join(options[:2])
    user_content = (
        f"{fewshot_block}"
        f"------\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Reasoning:"
    )
    return [
        {"role": "system", "content": SYSTEM_INSTR},
        {"role": "user", "content": user_content},
    ]

def render_chat_prompt(tokenizer, messages):
    if hasattr(tokenizer, "apple_chat_template"):
        return tokenizer.apple_chat_template(messages, tokenize=False, add_generation_prompt=True)  # type: ignore
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# -----------------------------
# Utils
# -----------------------------
def vprint(*a, verbose=False, **k):
    if verbose:
        print(*a, **k)

def norm_text(s: str) -> str:
    return re.sub(r"\s+"," ", s.strip().lower())

def label_to_bool(v) -> Optional[bool]:
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return bool(int(v))
    if isinstance(v, str):
        t = norm_text(v)
        if t in YES_SET: return True
        if t in NO_SET:  return False
    return None

def ab_from_bool(b: bool) -> str:
    return "A" if b else "B"

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
            for _, v in data.items():
                if isinstance(v, list):
                    items = v
                    break
    else:
        raise ValueError("file must be .json or .jsonl")
    return items

def get_text(ex: Dict[str, Any]) -> str:
    return ex.get("context") or ex.get("question") or ex.get("text") or ""

def get_question(ex: Dict[str, Any]) -> str:
    return ex.get("question") or ex.get("text") or ""

def load_din_spec(path: str) -> Dict[int, List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    spec: Dict[int, List[int]] = {}
    for k, v in raw.items():
        if isinstance(v, dict) and "indices" in v:
            spec[int(k)] = list(map(int, v["indices"]))
        elif isinstance(v, list):
            spec[int(k)] = list(map(int, v))
        else:
            raise ValueError(f"Bad DIN entry for key={k}: {v}")
    return spec

def save_din_spec(path: str, spec: Dict[int, List[int]]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)

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
                # zS/zT-based methods
                # supply tau if not present (for soft/stability)
                kwargs = dict(k_ratio=k_ratio, **primary_kwargs)
                if primary_method in {"soft_sign","stability_sign"} and "tau" not in kwargs:
                    kwargs["tau"] = tau
                chosen = dispatch_select_din(primary_method, zS=zS, zT=zT, **kwargs)
            elif primary_method in {"maha","fisher","bootstrap"}:
                # Hs/Ht-based methods
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
                    # ultimate safety net: pure top-k strength
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

# -----------------------------
# Prompt assembly
# -----------------------------
def exemplar_to_block(ex: Dict[str, Any]) -> str:
    ctx = get_text(ex)
    qes = ex.get('question')
    ans_bool = label_to_bool(ex.get("answer"))
    # ans_ab = ab_from_bool(ans_bool) if ans_bool is not None else "A"
    rationale = ex.get("rationale") or ex.get("reasoning") or ex.get("cot") or "By chaining the definitions, we deduce the property."
    return (
        # "Given a problem statement as contexts, the task is to answer a logical reasoning question.\n"
        # "------\n"
        # f"Context:\n{ctx}\n\n"
        f"Question:\n{qes}\n\n"
        f"Reasoning:\n{rationale}\n\n"
        # f"Final answer: {ans_ab}"
    )

def build_prompt_with_exemplars(tokenizer, query_ctx: str, question: str, exemplars: List[Dict[str, Any]], icl_shots: int = 0) -> str:
    options = ["A) True", "B) False"]
    if exemplars:
        fs_text = "\n------\n".join([exemplar_to_block(e) for e in exemplars]) + "\n------\n"
        context = fs_text + query_ctx
    else:
        context = query_ctx
    messages = build_messages_from_fields(context=context, question=question, options=options, icl_shots=icl_shots)
    return render_chat_prompt(tokenizer, messages)

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
# CLI
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--hf_embed_model", type=str, default=None)
    ap.add_argument("--pool_file", type=str, required=True)
    ap.add_argument("--query_file", type=str, required=True)

    ap.add_argument("--din_file", type=str, default=None)
    ap.add_argument("--compute_din", action="store_true")
    ap.add_argument("--src_file", type=str, default=None)
    ap.add_argument("--tgt_file", type=str, default=None)
    ap.add_argument("--layers", type=str,
                    default="-6,-5,-4,-3,-2,-1",
                    help="DIN layers (default: last 6)")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--k_ratio", type=float, default=0.05)

    # New: selector controls
    ap.add_argument("--primary_method", type=str, default="same_sign",
                    choices=["same_sign","topk_strength","intersect","rank_agg","soft_sign","maha","fisher","bootstrap","stability_sign"],
                    help="DIN selection strategy for the main path")
    ap.add_argument("--primary_kwargs", type=str, default=None,
                    help='JSON dict for primary selector kwargs (e.g., {"margin":0.2})')
    ap.add_argument("--fallback_method", type=str, default="topk_strength",
                    choices=["same_sign","topk_strength","intersect","rank_agg","soft_sign","maha","fisher","bootstrap","stability_sign"],
                    help="Fallback strategy when primary returns empty")
    ap.add_argument("--fallback_kwargs", type=str, default=None,
                    help='JSON dict for fallback selector kwargs')

    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--mmr_lambda", type=float, default=0.7)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--icl_shots", type=int, default=0, choices=[0,2], help="0 or 2")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)

    ap.add_argument("--save_jsonl", type=str, default=None)
    ap.add_argument("--save_din", type=str, default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--balance", type=str, default="min", choices=["none","min","tgt"])
    ap.add_argument("--zs_mode", type=str, default="pooled", choices=["union_weighted","union_equal","pooled"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with_random_control", action="store_true")
    ap.add_argument("--with_zero_shot", action="store_true")

    args = ap.parse_args()
    print(args)

    vprint(f"[Init] model_name={args.model_name}, hf_embed_model={args.hf_embed_model or args.model_name}", verbose=args.verbose)
    vprint(f"[Init] dtype={args.dtype}, device={args.device or 'auto'}, icl_shots={args.icl_shots}", verbose=args.verbose)

    embed_name = args.hf_embed_model or args.model_name
    tok = AutoTokenizer.from_pretrained(embed_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    vprint(f"[HF] Loading HF model for hidden states: {embed_name} on {device}", verbose=args.verbose)
    hf_model = AutoModelForCausalLM.from_pretrained(
        embed_name,
        torch_dtype=torch.bfloat16 if args.dtype=="bfloat16" else torch.float16,
        trust_remote_code=True,
        device_map='auto'
    )
    hf_model.eval()

    vprint(f"[Data] Loading pool from {args.pool_file}", verbose=args.verbose)
    pool = load_json_or_jsonl(args.pool_file)
    vprint(f"[Data] Loading queries from {args.query_file}", verbose=args.verbose)
    queries = load_json_or_jsonl(args.query_file)
    pool_texts = [get_text(ex) for ex in pool]

    # DIN spec
    if args.compute_din:
        assert args.src_file and args.tgt_file, "Using --compute_din requires --src_file and --tgt_file"
        vprint(f"[Data] Loading SRC from {args.src_file}", verbose=args.verbose)
        src = load_json_or_jsonl(args.src_file)
        vprint(f"[Data] Loading TGT from {args.tgt_file}", verbose=args.verbose)
        tgt = load_json_or_jsonl(args.tgt_file)
        src_texts = [get_text(ex) for ex in src]
        tgt_texts = [get_text(ex) for ex in tgt]
        layer_signed = parse_layers(args.layers)
        vprint(f"[DIN] layers={layer_signed}, tau={args.tau}, k_ratio={args.k_ratio}", verbose=args.verbose)

        din_spec = compute_din_from_corpora(
            tok, hf_model, src_texts, tgt_texts, layer_signed,
            tau=args.tau, k_ratio=args.k_ratio, batch_size=args.batch_size, verbose=args.verbose,
            balance=args.balance, zs_mode=args.zs_mode, seed=args.seed,
            primary_method=args.primary_method,
            primary_kwargs=parse_json_arg(args.primary_kwargs),
            fallback_method=args.fallback_method,
            fallback_kwargs=parse_json_arg(args.fallback_kwargs),
        )
        if args.save_din:
            save_din_spec(args.save_din, din_spec)
            vprint(f"[DIN] Saved DIN spec -> {args.save_din}", verbose=args.verbose)
    else:
        assert args.din_file, "No --compute_din: please provide --din_file"
        vprint(f"[DIN] Loading DIN spec from {args.din_file}", verbose=args.verbose)
        din_spec = load_din_spec(args.din_file)

    # Encode pool in DIN subspace
    def text_to_vec(text: str) -> np.ndarray:
        return text_to_din_vec(text, tok, hf_model, din_spec)

    vprint(f"[Vec] Encoding pool texts into DIN subspace...", verbose=args.verbose)
    cand_vecs = [text_to_vec(t) for t in tqdm(pool_texts, desc="Pool -> DIN vecs", disable=not args.verbose)]
    cand_vecs = np.stack(cand_vecs, axis=0)
    vprint(f"[Vec] Pool encoded: shape={cand_vecs.shape}", verbose=args.verbose)

    # Build prompts
    prompts_din: List[str] = []
    meta_din: List[Dict[str, Any]] = []
    vprint(f"[ICL] Building DIN prompts (topk={args.topk}, mmr_lambda={args.mmr_lambda})", verbose=args.verbose)
    for qi, q in enumerate(tqdm(queries, desc="Queries -> DIN prompts", disable=not args.verbose)):
        q_text = get_text(q)
        question = get_question(q)
        q_vec  = text_to_vec(q_text)
        sel    = mmr(q_vec, cand_vecs, lam=args.mmr_lambda, topk=args.topk)
        exemplars = [pool[i] for i in sel]
        prompt = build_prompt_with_exemplars(tok, q_text, question, exemplars, icl_shots=args.icl_shots)
        prompts_din.append(prompt)
        meta_din.append({"query_idx": qi, "selected_indices": sel})
    print(prompts_din[0])

    # Random control
    prompts_rand: List[str] = []
    meta_rand: List[Dict[str, Any]] = []
    if args.with_random_control:
        rng = np.random.default_rng(args.seed)
        vprint(f"[ICL] Building RANDOM-control prompts (topk={args.topk})", verbose=args.verbose)
        for qi, q in enumerate(tqdm(queries, desc="Queries -> RANDOM prompts", disable=not args.verbose)):
            q_text = get_text(q)
            question = get_question(q)
            k = min(args.topk, len(pool))
            sel_rand = rng.choice(len(pool), size=k, replace=False).tolist()
            exemplars = [pool[i] for i in sel_rand]
            prompt = build_prompt_with_exemplars(tok, q_text, question, exemplars, icl_shots=args.icl_shots)
            prompts_rand.append(prompt)
            meta_rand.append({"query_idx": qi, "selected_indices": sel_rand})
        print(prompts_rand[0])
    
    # Zero-shot group
    prompts_zero: List[str] = []
    meta_zero: List[Dict[str, Any]] = []
    if args.with_zero_shot:
        vprint(f"[ICL] Building ZERO-SHOT prompts", verbose=args.verbose)
        for qi, q in enumerate(tqdm(queries, desc="Queries -> ZERO-SHOT prompts", disable=not args.verbose)):
            q_text = get_text(q)
            question = get_question(q)
            prompt = build_prompt_with_exemplars(tok, q_text, question, exemplars=[], icl_shots=0)
            prompts_zero.append(prompt)
            meta_zero.append({"query_idx": qi, "selected_indices": []})
        print(prompts_zero[0])

    # Free HF model before vLLM
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)

    vprint(f"[vLLM] Creating shared engine ...", verbose=args.verbose)
    shared_llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=8192
    )

    gens_din, _ = run_vllm_generate(
        model_name=args.model_name,
        prompts=prompts_din,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        verbose=args.verbose,
        llm=shared_llm,
    )

    gens_zero = None
    if args.with_zero_shot:
        gens_zero, _ = run_vllm_generate(
            model_name=args.model_name,
            prompts=prompts_zero,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            verbose=args.verbose,
            llm=shared_llm,
        )

    gens_rand = None
    if args.with_random_control:
        gens_rand, _ = run_vllm_generate(
            model_name=args.model_name,
            prompts=prompts_rand,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            verbose=args.verbose,
            llm=shared_llm,
        )

    # Cleanup
    del shared_llm
    torch.cuda.empty_cache()

    # Evaluation & save
    def eval_group(group_name: str, gens: List[str], prompts: List[str], meta: List[Dict[str, Any]]):
        correct = 0; total = 0
        group_results = []
        for qi, (q, gen, prompt_str, m) in enumerate(zip(queries, gens, prompts, meta)):
            gold_bool = get_gold_bool(q)
            pred_ab   = extract_final_ab(gen)
            pred_bool = ab_to_bool(pred_ab)
            if pred_bool is None:
                pred_bool = label_to_bool(gen)  # lax fallback
            is_counted = (gold_bool is not None and pred_bool is not None)
            is_correct = (is_counted and (pred_bool == gold_bool))
            if is_counted:
                total += 1; 
                if is_correct: correct += 1
            rec = {
                "group": group_name, "idx": qi, "query": q,
                "prompt": prompt_str, "generation": gen, "retrieval": m,
                "gold_bool": gold_bool,
                "gold_ab": ("A" if gold_bool else "B") if gold_bool is not None else None,
                "pred_ab": pred_ab, "pred_bool": pred_bool,
                "correct": bool(is_correct) if is_counted else None,
                "counted_in_acc": bool(is_counted),
            }
            group_results.append(rec)
        acc = (correct / total) if total > 0 else 0.0
        return acc, correct, total, group_results

    acc_din, c_din, t_din, res_din = eval_group("din", gens_din, prompts_din, meta_din)

    acc_rand, c_rand, t_rand, res_rand = (0.0, 0, 0, [])
    if args.with_random_control and gens_rand is not None:
        acc_rand, c_rand, t_rand, res_rand = eval_group("random", gens_rand, prompts_rand, meta_rand)

    acc_zero, c_zero, t_zero, res_zero = (0.0, 0, 0, [])
    if args.with_zero_shot and gens_zero is not None:
        acc_zero, c_zero, t_zero, res_zero = eval_group("zero_shot", gens_zero, prompts_zero, meta_zero)

    print("\n[Summary]")
    print(f"  Top {args.topk} Demo")
    print(f"  DIN       : Accuracy = {acc_din:.4f}  ({c_din}/{t_din})")
    if args.with_random_control:
        print(f"  RANDOM    : Accuracy = {acc_rand:.4f}  ({c_rand}/{t_rand})")
    if args.with_zero_shot:
        print(f"  ZERO-SHOT : Accuracy = {acc_zero:.4f}  ({c_zero}/{t_zero})")

    results = res_din
    if args.with_random_control and res_rand:
        results += res_rand
    if args.with_zero_shot and res_zero:
        results += res_zero

    if args.save_jsonl:
        vprint(f"[Save] Writing generations -> {args.save_jsonl}", verbose=args.verbose)
        with open(args.save_jsonl, "w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
