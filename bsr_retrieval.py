#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SET-BSR & Hidden-State Retrieval Baselines (ICL Example Selection)
=================================================================
This script provides:
  1) BSR (token-level recall) independent selection
  2) SET-BSR (greedy set coverage over token similarities)
  3) Hidden-state cosine retrieval (mean-pooled last layer)
  4) SET-HIDDEN (set coverage using token embeddings from any encoder)

It reads a candidate pool (JSONL) and a query file (JSONL), picks top-k
examples per query, and writes a JSONL with selections.

Example
-------
python set_bsr_and_hidden_state_retrieval.py \
  --pool_file pool.jsonl \
  --query_file queries.jsonl \
  --output_file selections.jsonl \
  --k 8 \
  --method set-bsr \
  --encoder microsoft/deberta-large-mnli \
  --text_keys source,input

Notes
-----
- "encoder" is any Hugging Face model that yields hidden states; default
  is DeBERTa-large-MNLI for BSR/SET-BSR. For hidden baselines you can
  choose e.g. sentence-transformers/all-mpnet-base-v2 or a causal LM.
- For causal LMs without a native [CLS], we use mean pooling over tokens.
- Token coverage uses cosine similarity and averages max-sim per query token.
- We ignore IDF weighting to keep the implementation simple and fast.

Output format (per query)
-------------------------
{
  "query_id": <id or index>,
  "selected_indices": [idx0, idx1, ...],
  "selected_texts": ["...", "...", ...],
  "scores": [score0, score1, ...],
  "method": "set-bsr" | "bsr" | "hidden-cos" | "set-hidden"
}

Pool/Query format
-----------------
Each line in JSONL is an object. The text fed to the encoder is the
concatenation of the fields listed by --text_keys, in order, if present.
We try common fallbacks: ["source", "input", "question", "premise", "sentence"].

Author: ChatGPT (GPT-5 Thinking)
"""

import argparse
import json
import math
import os
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# -----------------------------
# Utilities
# -----------------------------

DEFAULT_TEXT_KEYS = ["source", "input", "question", "premise", "sentence"]


def get_text_from_item(item: Dict, keys: List[str]) -> str:
    parts = []
    for k in keys:
        if k in item and isinstance(item[k], str):
            parts.append(item[k].strip())
    if parts:
        return " \n".join([p for p in parts if p])
    # Fallback over common keys if user-provided keys miss
    for k in DEFAULT_TEXT_KEYS:
        if k in item and isinstance(item[k], str) and item[k].strip():
            return item[k].strip()
    # Last resort: stringify item
    return json.dumps(item, ensure_ascii=False)


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


# -----------------------------
# Encoding (token embeddings)
# -----------------------------

def mean_pool_last_hidden(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # hidden_states: [B, L, D], attention_mask: [B, L]
    mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
    masked = hidden_states * mask
    denom = mask.sum(dim=1).clamp(min=1e-6)
    pooled = masked.sum(dim=1) / denom
    return pooled  # [B, D]


@torch.no_grad()
def encode_texts(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int,
    return_tokens: bool = False,
) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
    """
    Returns:
      seq_embs: [N, D] mean-pooled embeddings
      token_embs_list: list of [Li, D] per-text token embeddings if return_tokens
      attn_masks_list: list of [Li] attention masks (torch.float) if return_tokens
    """
    seq_embs = []
    token_embs_list = [] if return_tokens else None
    attn_masks_list = [] if return_tokens else None

    bs = 8
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(device)
        outputs = model(**enc, output_hidden_states=False)
        tok = outputs.last_hidden_state  # [B, L, D]
        pooled = mean_pool_last_hidden(tok, enc['attention_mask'])  # [B, D]
        seq_embs.append(pooled.detach().cpu())
        if return_tokens:
            # store per-example token states and masks (float mask)
            am = enc['attention_mask'].float()
            for bi in range(tok.size(0)):
                Li = int(am[bi].sum().item())
                token_embs_list.append(tok[bi, :Li, :].detach().cpu())
                attn_masks_list.append(am[bi, :Li].detach().cpu())

    seq_embs = torch.cat(seq_embs, dim=0) if seq_embs else torch.empty(0)
    return seq_embs, token_embs_list, attn_masks_list


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True).clamp(min=eps))


# -----------------------------
# Scoring functions
# -----------------------------

def cosine_scores(query_vec: torch.Tensor, pool_vecs: torch.Tensor) -> np.ndarray:
    """Cosine similarity between one query and all pool vectors."""
    q = l2_normalize(query_vec.unsqueeze(0))  # [1, D]
    P = l2_normalize(pool_vecs)               # [N, D]
    sims = (q @ P.t()).squeeze(0)            # [N]
    return sims.cpu().numpy()


def bsr_score(query_tok: torch.Tensor, cand_tok: torch.Tensor) -> float:
    """
    Token-level recall: average over query tokens of max cosine to any candidate token.
    query_tok: [Lq, D], cand_tok: [Lc, D]
    """
    Q = l2_normalize(query_tok, dim=-1)  # [Lq, D]
    C = l2_normalize(cand_tok, dim=-1)   # [Lc, D]
    sim = Q @ C.t()                      # [Lq, Lc]
    max_per_q = sim.max(dim=1).values    # [Lq]
    return float(max_per_q.mean().item())


def set_bsr_greedy(query_tok: torch.Tensor, pool_tok_list: List[torch.Tensor], k: int) -> Tuple[List[int], List[float]]:
    """
    Greedy maximization of set-level coverage: per query token keep current max sim
    across selected candidates; pick the candidate with largest marginal gain each step.
    Returns selected indices and their marginal gains.
    """
    Q = l2_normalize(query_tok, dim=-1)  # [Lq, D]
    Lq = Q.size(0)
    current = torch.full((Lq,), -1.0, dtype=torch.float32)  # start with -1 so first gains are non-negative

    # Precompute per-candidate max-per-q arrays for speed
    per_cand = []
    for C in pool_tok_list:
        Cn = l2_normalize(C, dim=-1)
        sim = Q @ Cn.t()            # [Lq, Lc]
        per_cand.append(sim.max(dim=1).values)  # [Lq]

    selected = []
    gains = []
    remaining = set(range(len(pool_tok_list)))

    for _ in range(min(k, len(pool_tok_list))):
        best_idx = -1
        best_gain = -1e9
        for idx in remaining:
            candidate = per_cand[idx]
            # marginal improvement over current (use clamp to handle -1 start)
            improved = torch.maximum(current, candidate)
            gain = float((improved - current).mean().item())
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx < 0:
            break
        # update
        current = torch.maximum(current, per_cand[best_idx])
        selected.append(best_idx)
        gains.append(best_gain)
        remaining.remove(best_idx)

    return selected, gains


# -----------------------------
# Main pipeline per query
# -----------------------------

def select_for_query(
    method: str,
    k: int,
    query_text: str,
    pool_texts: List[str],
    model,
    tokenizer,
    device: torch.device,
    max_length: int,
    cached_pool_seq: torch.Tensor,
    cached_pool_tok: Optional[List[torch.Tensor]],
) -> Tuple[List[int], List[float]]:

    if method in ("hidden-cos",):
        # Encode query to seq emb
        q_seq, _, _ = encode_texts(model, tokenizer, [query_text], device, max_length, return_tokens=False)
        sims = cosine_scores(q_seq[0], cached_pool_seq)
        order = np.argsort(-sims)[:k].tolist()
        return order, sims[order].tolist()

    # For token-based methods, we need token embeddings
    q_seq, q_tok_list, _ = encode_texts(model, tokenizer, [query_text], device, max_length, return_tokens=True)
    q_tok = q_tok_list[0]

    if method in ("bsr",):
        scores = []
        for c_tok in cached_pool_tok:
            scores.append(bsr_score(q_tok, c_tok))
        order = np.argsort(-np.array(scores))[:k].tolist()
        return order, [scores[i] for i in order]

    if method in ("set-bsr", "set-hidden"):
        sel, gains = set_bsr_greedy(q_tok, cached_pool_tok, k)
        return sel, gains

    raise ValueError(f"Unknown method: {method}")


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="SET-BSR & Hidden-State Retrieval Baselines")
    ap.add_argument('--pool_file', required=True, type=str)
    ap.add_argument('--query_file', required=True, type=str)
    ap.add_argument('--output_file', required=True, type=str)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--method', type=str, default='set-bsr',
                    choices=['bsr', 'set-bsr', 'hidden-cos', 'set-hidden'])
    ap.add_argument('--encoder', type=str, default='microsoft/deberta-large-mnli',
                    help='HF model id for embeddings (token-level & pooled)')
    ap.add_argument('--max_length', type=int, default=512)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--text_keys', type=str, default='',
                    help='Comma-separated fields to concatenate (e.g., "source,input").')
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    text_keys = [k.strip() for k in args.text_keys.split(',') if k.strip()]
    if not text_keys:
        text_keys = DEFAULT_TEXT_KEYS

    print(f"Loading encoder: {args.encoder}")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, use_fast=True)
    model = AutoModel.from_pretrained(args.encoder)
    model.to(device).eval()

    print("Loading pool & queries ...")
    pool = load_jsonl(args.pool_file)
    queries = load_jsonl(args.query_file)

    pool_texts = [get_text_from_item(x, text_keys) for x in pool]
    query_texts = [get_text_from_item(x, text_keys) for x in queries]

    print("Encoding pool (sequence-level & token-level) ...")
    pool_seq, pool_tok_list, _ = encode_texts(
        model, tokenizer, pool_texts, device, args.max_length,
        return_tokens=(args.method in ("bsr", "set-bsr", "set-hidden"))
    )

    results = []
    for qi, qtext in enumerate(query_texts):
        sel_idx, sel_scores = select_for_query(
            method=args.method,
            k=args.k,
            query_text=qtext,
            pool_texts=pool_texts,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=args.max_length,
            cached_pool_seq=pool_seq,
            cached_pool_tok=pool_tok_list if pool_tok_list is not None else [],
        )
        out = {
            "query_id": queries[qi].get("id", qi),
            "method": args.method,
            "selected_indices": sel_idx,
            "selected_texts": [pool_texts[i] for i in sel_idx],
            "scores": sel_scores,
        }
        results.append(out)

    print(f"Writing: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Done.")


if __name__ == '__main__':
    main()
