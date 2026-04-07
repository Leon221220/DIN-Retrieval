#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
grid_run_fallbacks.py
=====================
Benchmark different fallback DIN selection methods by repeatedly running
`din_icl_vllm_selectors.py` with fixed settings except for `--fallback_method`
(and its kwargs). For each run, we recompute DIN (`--compute_din`) to let the
fallback influence the final DIN set when the primary yields empty for a layer.

It parses the stdout of each run to extract the summary accuracies and writes a CSV.

Usage (mirrors your base command; edit paths/args as needed):
-------------------------------------------------------------
python -u grid_run_fallbacks.py \
  --runner din_icl_vllm_selectors.py \
  --model_name /userhome/hf/llama3.1-8B-Instruct \
  --pool_file /code/localdataset/gsm8k/train.jsonl \
  --query_file /code/localdataset/prontoqa-train/ProntoQA_dev_gpt-4.jsonl \
  --src_file /code/localdataset/gsm8k/train.jsonl \
  --tgt_file /code/localdataset/prontoqa-train/prontoqa_train_split_v2.jsonl \
  --tau 0.3 \
  --k_ratio 0.1 \
  --balance min \
  --zs_mode pooled \
  --dtype float16 \
  --tensor_parallel_size 2 \
  --max_new_tokens 4096 \
  --with_random_control \
  --with_zero_shot \
  --icl_shots 0 \
  --topk 2 \
  --layers "-6,-5,-4,-3,-2,-1" \
  --out_dir ./grid_fallback_runs \
  --summary_csv ./grid_fallback_runs/summary.csv \
  --verbose

By default:
- primary_method = same_sign
- fallback grid = [topk_strength, soft_sign, intersect, rank_agg, maha, fisher, bootstrap, stability_sign]
- You can tweak per-method kwargs via JSON strings, e.g. --soft_sign_kwargs '{"margin":0.2}'
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", type=str, default="din_icl_vllm_selectors.py",
                    help="Path to the integrated runner script.")
    ap.add_argument("--task_name", type=str, default="prontoqa",
                    choices=["prontoqa","gsm8k","csqa","arc","mmlu","boolq","strategyqa","folio"],
                    help="Choose dataset validator to control prompt & evaluation.")
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--pool_file", type=str, required=True)
    ap.add_argument("--query_file", type=str, required=True)
    ap.add_argument("--src_file", type=str, required=True)
    ap.add_argument("--tgt_file", type=str, required=True)

    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--k_ratio", type=float, default=0.05)
    ap.add_argument("--balance", type=str, default="min")
    ap.add_argument("--zs_mode", type=str, default="pooled")
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--with_random_control", action="store_true")
    ap.add_argument("--with_zero_shot", action="store_true")
    ap.add_argument("--icl_shots", type=int, default=0, choices=[0,2])
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--layers", type=str, default="-6,-5,-4,-3,-2,-1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype_vllm", type=str, default=None, help="Alias of --dtype (kept for clarity)")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--summary_csv", type=str, required=True)
    ap.add_argument("--verbose", action="store_true")

    # Optional per-method kwargs in JSON
    ap.add_argument("--soft_sign_kwargs", type=str, default=None)
    ap.add_argument("--intersect_kwargs", type=str, default=None)
    ap.add_argument("--rank_agg_kwargs", type=str, default=None)
    ap.add_argument("--maha_kwargs", type=str, default=None)
    ap.add_argument("--fisher_kwargs", type=str, default=None)
    ap.add_argument("--bootstrap_kwargs", type=str, default=None)
    ap.add_argument("--stability_sign_kwargs", type=str, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)

    return ap.parse_args()

# METHODS = [
#     "topk_strength",
#     "soft_sign",
#     "intersect",
#     "rank_agg",
#     "maha",
#     "fisher",
#     "bootstrap",
#     "stability_sign",
# ]

METHODS = [
    "topk_strength",
]

def as_json(s: Optional[str]) -> Optional[str]:
    if s is None: return None
    # Validate JSON to fail fast
    try:
        _ = json.loads(s)
    except Exception as e:
        raise ValueError(f"Invalid JSON: {s}\nError: {e}")
    return s

def build_cmd(base, fallback_method: str, out_dir: Path, args) -> Tuple[List[str], Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    din_out = out_dir / f"din_{fallback_method}.json"
    preds_out = out_dir / f"preds_{fallback_method}.jsonl"
    log_file = out_dir / f"log_{fallback_method}.txt"

    cmd = [
        sys.executable, "-u", args.runner,
        "--model_name", args.model_name,
        "--task_name", args.task_name,
        "--pool_file", args.pool_file,
        "--query_file", args.query_file,
        "--compute_din",
        "--src_file", args.src_file,
        "--tgt_file", args.tgt_file,
        f"--layers={args.layers}",
        "--tau", str(args.tau),
        "--k_ratio", str(args.k_ratio),
        "--balance", args.balance,
        "--zs_mode", args.zs_mode,
        "--dtype", args.dtype_vllm or args.dtype,
        "--tensor_parallel_size", str(args.tensor_parallel_size),
        "--max_new_tokens", str(args.max_new_tokens),
        "--topk", str(args.topk),
        "--icl_shots", str(args.icl_shots),
        "--primary_method", "same_sign",
        "--fallback_method", fallback_method,
        "--save_din", str(din_out),
        "--save_jsonl", str(preds_out),
        "--seed", str(args.seed),
        "--temperature", str(args.temperature)
    ]
    if args.with_random_control: cmd.append("--with_random_control")
    if args.with_zero_shot: cmd.append("--with_zero_shot")
    if args.verbose: cmd.append("--verbose")
    if args.device: cmd += ["--device", args.device]

    # attach fallback_kwargs if provided
    kw_map = {
        "soft_sign": args.soft_sign_kwargs,
        "intersect": args.intersect_kwargs,
        "rank_agg": args.rank_agg_kwargs,
        "maha": args.maha_kwargs,
        "fisher": args.fisher_kwargs,
        "bootstrap": args.bootstrap_kwargs,
        "stability_sign": args.stability_sign_kwargs,
    }
    kw_json = kw_map.get(fallback_method)
    if kw_json:
        cmd += ["--fallback_kwargs", as_json(kw_json)]

    return cmd, log_file, preds_out

def parse_summary(stdout: str) -> Dict[str, Optional[str]]:
    """
    Expect a block like:
    [Summary]
      Top K Demo
      DIN       : Accuracy = 0.4321  (123/284)
      RANDOM    : Accuracy = 0.4010  (114/284)
      ZERO-SHOT : Accuracy = 0.3200  (91/284)
    """
    res = {"acc_din": None, "acc_rand": None, "acc_zero": None}
    m = re.search(r"DIN\s*:\s*Accuracy\s*=\s*([0-9.]+)", stdout)
    if m: res["acc_din"] = m.group(1)
    m = re.search(r"RANDOM\s*:\s*Accuracy\s*=\s*([0-9.]+)", stdout)
    if m: res["acc_rand"] = m.group(1)
    m = re.search(r"ZERO-SHOT\s*:\s*Accuracy\s*=\s*([0-9.]+)", stdout)
    if m: res["acc_zero"] = m.group(1)
    return res

def run_once(cmd: List[str], log_file: Path, verbose: bool) -> Tuple[int, str]:
    if verbose:
        print("[RUN]", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in proc.stdout:
        lines.append(line)
        if verbose:
            print(line, end="")
    proc.wait()
    out = "".join(lines)
    log_file.write_text(out, encoding="utf-8")
    return proc.returncode, out

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for method in METHODS:
        cmd, log_file, preds_out = build_cmd([], method, out_dir, args)
        code, out = run_once(cmd, log_file, verbose=args.verbose)
        summ = parse_summary(out)
        row = {
            "fallback_method": method,
            "acc_din": summ["acc_din"],
            "acc_random": summ["acc_rand"],
            "acc_zero_shot": summ["acc_zero"],
            "preds_file": str(preds_out),
            "log_file": str(log_file),
        }
        rows.append(row)

    # Write CSV
    import csv
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fallback_method","acc_din","acc_random","acc_zero_shot","preds_file","log_file"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[Done] Wrote summary -> {summary_path}")
    for r in rows:
        print(f"  {r['fallback_method']}: DIN={r['acc_din']}  RAND={r['acc_random']}  ZERO={r['acc_zero_shot']}")

if __name__ == "__main__":
    main()
