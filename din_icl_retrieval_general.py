#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import time
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import gc

# Progress bar
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# ===== Project imports (fix the small typo from the original file) =====
from validator.dataset_validators import *  # noqa: F401,F403
from src.bsr import *  # noqa: F401,F403
from utils import *  # noqa: F401,F403

# ===== DIN selectors & utils (as in your project) =====
from din_selectors.din_selectors import (
    dispatch_select_din,
)

# =============================
# Built-in few-shot exemplars (for PrOntoQA-like A/B)
# =============================
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


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_name", type=str, default="prontoqa",
                    choices=["prontoqa","gsm8k","csqa","arc","mmlu","boolq","strategyqa","folio"],
                    help="Choose dataset validator to control prompt & evaluation.")
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--hf_embed_model", type=str, default=None)
    ap.add_argument("--pool_file", type=str, required=True)
    ap.add_argument("--query_file", type=str, required=True)

    ap.add_argument("--din_file", type=str, default=None)
    ap.add_argument("--compute_din", action="store_true")
    ap.add_argument("--src_file", type=str, default=None)
    ap.add_argument("--tgt_file", type=str, default=None)
    ap.add_argument("--layers", nargs="?", const="-6,-5,-4,-3,-2,-1",
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

    # Validator
    validator = get_validator(args.task_name)

    vprint(f"[Init] task={args.task_name}, model_name={args.model_name}, hf_embed_model={args.hf_embed_model or args.model_name}", verbose=args.verbose)
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
        device_map="auto",
        trust_remote_code=True
    )
    hf_model.eval()

    # Data
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
            with open(args.save_din, "w", encoding="utf-8") as f:
                json.dump(din_spec, f, ensure_ascii=False, indent=2)
            vprint(f"[DIN] Saved DIN spec -> {args.save_din}", verbose=args.verbose)
    else:
        assert args.din_file, "No --compute_din: please provide --din_file"
        vprint(f"[DIN] Loading DIN spec from {args.din_file}", verbose=args.verbose)
        with open(args.din_file, "r", encoding="utf-8") as f:
            din_spec = json.load(f)
        # ensure int keys
        din_spec = {int(k): list(map(int, v if isinstance(v, list) else v.get("indices", []))) for k, v in din_spec.items()}

    # Encode pool in DIN subspace
    def text_to_vec(text: str) -> np.ndarray:
        return text_to_din_vec(text, tok, hf_model, din_spec)

    vprint(f"[Vec] Encoding pool texts into DIN subspace...", verbose=args.verbose)
    cand_vecs = [text_to_vec(t) for t in tqdm(pool_texts, desc="Pool -> DIN vecs", disable=not args.verbose)]
    cand_vecs = np.stack(cand_vecs, axis=0)
    vprint(f"[Vec] Pool encoded: shape={cand_vecs.shape}", verbose=args.verbose)

    # Build prompts (DIN-retrieved)
    prompts_din: List[str] = []
    meta_din: List[Dict[str, Any]] = []
    vprint(f"[ICL] Building DIN prompts (topk={args.topk}, mmr_lambda={args.mmr_lambda})", verbose=args.verbose)
    for qi, q in enumerate(tqdm(queries, desc="Queries -> DIN prompts", disable=not args.verbose)):
        q_text = get_text(q)
        question = get_question(q)
        options = extract_options_from_example(q, validator=validator)
        q_vec  = text_to_vec(q_text)
        sel    = mmr(q_vec, cand_vecs, lam=args.mmr_lambda, topk=args.topk)
        exemplars = [pool[i] for i in sel]
        prompt = build_prompt_with_exemplars(tok, validator, q_text, question, exemplars, icl_shots=args.icl_shots, options=options)
        prompts_din.append(prompt)
        meta_din.append({"query_idx": qi, "selected_indices": sel})
    # print first prompt for sanity
    if prompts_din:
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
            options = extract_options_from_example(q, validator=validator)
            k = min(args.topk, len(pool))
            sel_rand = rng.choice(len(pool), size=k, replace=False).tolist()
            exemplars = [pool[i] for i in sel_rand]
            prompt = build_prompt_with_exemplars(tok, validator, q_text, question, exemplars, icl_shots=args.icl_shots, options=options)
            prompts_rand.append(prompt)
            meta_rand.append({"query_idx": qi, "selected_indices": sel_rand})
        if prompts_rand:
            print(prompts_rand[0])
    
    # Zero-shot group
    prompts_zero: List[str] = []
    meta_zero: List[Dict[str, Any]] = []
    if args.with_zero_shot:
        vprint(f"[ICL] Building ZERO-SHOT prompts", verbose=args.verbose)
        for qi, q in enumerate(tqdm(queries, desc="Queries -> ZERO-SHOT prompts", disable=not args.verbose)):
            q_text = get_text(q)
            question = get_question(q)
            options = extract_options_from_example(q, validator=validator)
            prompt = build_prompt_with_exemplars(tok, validator, q_text, question, exemplars=[], icl_shots=0, options=options)
            prompts_zero.append(prompt)
            meta_zero.append({"query_idx": qi, "selected_indices": []})
        if prompts_zero:
            print(prompts_zero[0])


    # Free HF model before vLLM
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

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

    # Cleanup shared engine
    del shared_llm
    torch.cuda.empty_cache()

    # ---------- Evaluation & save ----------
    def eval_group(group_name: str, gens: List[str], prompts: List[str], meta: List[Dict[str, Any]]):
        correct = 0; total = 0
        group_results = []
        for qi, (q, gen, prompt_str, m) in enumerate(zip(queries, gens, prompts, meta)):
            gold = validator.get_gold_label(q)
            pred = validator.parse_prediction(gen)
            ok = validator.is_correct(pred, gold)
            is_counted = (ok is not None)
            if is_counted:
                total += 1; 
                if ok: correct += 1
            rec = {
                "group": group_name, "idx": qi, "query": q,
                "prompt": prompt_str, "generation": gen, "retrieval": m,
                "gold": gold, "pred": pred,
                "correct": bool(ok) if is_counted else None,
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

    # Merge and save
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
