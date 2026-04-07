import csv
import os
import shlex
import subprocess
from itertools import product
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.stats import ttest_ind
import pandas as pd

# 1) Base python and runner
BASE_CMD = "CUDA_VISIBLE_DEVICES=0 python -u ../src/grid_run_fallbacks.py"

# 2) Fixed args (shared across runs). You can copy-paste from your working command.
FIXED_KV = dict(
    runner="../din_icl_retrieval_general.py",
    model_name="/userhome/huggingface/llama3.1-8B-Instruct",
    pool_file="../../localdataset/gsm8k/train.jsonl",
    query_file="../../localdataset/folio/folio_v2_validation.jsonl",
    src_file="../../localdataset/gsm8k/train.jsonl",
    tgt_file="../../localdataset/folio/folio_v2_train.jsonl",
    dtype="float16",
    tensor_parallel_size="1",
    max_new_tokens="4096",
    with_random_control="",   # flags without value: keep empty string
    with_zero_shot="",        # same as above
    icl_shots="0",
    task_name="folio",
    balance="tgt",
    zs_mode="pooled"
)

# 3) Grid to sweep (edit to taste)
GRID = dict(
    seed=[42, 43, 44, 45, 46],  # 例如 5 个不同随机种子
    layers=["-4,-3,-2,-1"],
    temperature=[0.0],
    k_ratio=[0.07],
)

# 4) Output root for all runs (auto timestamped subfolder)
ROOT_OUT = Path("/code/icl/tools/grid_fallback_runs/llama3.1_8b/gsm8k_folio/significance_test")
ROOT_OUT.mkdir(parents=True, exist_ok=True)


# 5) Concurrency (set >1 if you want to run multiple jobs in parallel)
MAX_PARALLEL = 2  # keep 1 if GPU memory is tight

# ------------------ END USER CONFIG --------------

def kv_to_slug(k, v):
    s = str(v).replace(",", "_").replace("/", "_").replace(" ", "")
    s = s.replace(":", "").replace("-", "m").replace(".", "p")
    return f"{k}-{s}"

def build_arglist(kv: dict):
    args = []
    for k, v in kv.items():
        if v == "":  # val-less flags
            args.append(f"--{k}")
        elif "layers" in k:
            args.append(f"--{k}={shlex.quote(str(v))}")
        else:
            args.append(f"--{k} {shlex.quote(str(v))}")
    return " ".join(args)

def run_one(run_kv: dict, log_dir: Path):
    # Build command
    cmd = f"{BASE_CMD} {build_arglist(run_kv)}"
    print(f"[RUN] {cmd}")
    # Run
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out_lines = []
    for line in proc.stdout:
        print(line, end="")
        out_lines.append(line)
    ret = proc.wait()
    # Save raw log
    (log_dir / "stdout.log").write_text("".join(out_lines), encoding="utf-8")
    return ret, out_lines

def scrape_summary(lines):
    """Fallback: extract summary lines when summary_csv not present"""
    keep = []
    in_sum = False
    for ln in lines:
        if "[Summary]" in ln:
            in_sum = True
            keep.append(ln.strip())
            continue
        if in_sum:
            if ln.strip() == "":
                in_sum = False
            else:
                keep.append(ln.strip())
    return "\n".join(keep)

def main():
    # Cartesian product
    grid_keys = list(GRID.keys())
    grid_vals = [GRID[k] for k in grid_keys]
    combos = list(product(*grid_vals))
    print(f"[INFO] Total runs: {len(combos)}")

    master_rows = []
    for i, vals in enumerate(combos, 1):
        combo = dict(zip(grid_keys, vals))

        # Prepare paths
        slug = "__".join([kv_to_slug(k, v) for k, v in combo.items()])
        out_dir = ROOT_OUT / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        # Per-run summary CSV
        summary_csv = out_dir / "summary.csv"

        # Merge fixed + sweeped + mandatory paths
        run_kv = dict(FIXED_KV)
        run_kv.update(combo)
        run_kv["out_dir"] = str(out_dir)
        run_kv["summary_csv"] = str(summary_csv)
        # default fallbacks
        if "tau" not in run_kv: run_kv["tau"] = "0.3"
        if "k_ratio" not in run_kv: run_kv["k_ratio"] = "0.05"

        # Execute
        ret, out_lines = run_one(run_kv, out_dir)

        # Aggregate summary
        if summary_csv.exists():
            # Append CSV rows with run metadata
            import csv
            with open(summary_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    r2 = dict(r)
                    # add sweep metadata columns
                    for k, v in combo.items():
                        r2[f"hparam.{k}"] = v
                    master_rows.append(r2)
        else:
            # Fallback: scrape stdout summary
            (out_dir / "summary_scraped.txt").write_text(scrape_summary(out_lines), encoding="utf-8")
            master_rows.append({
                "run_slug": slug,
                "summary": scrape_summary(out_lines),
                **{f"hparam.{k}": v for k, v in combo.items()}
            })

    # Write master CSV
    master_csv = ROOT_OUT / "MASTER_summary.csv"
    if master_rows:
        # Collect union of keys
        keys = set()
        for r in master_rows: keys |= set(r.keys())
        keys = sorted(keys)
        with open(master_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(master_rows)
        print(f"[DONE] Master summary -> {master_csv}")
    else:
        print("[WARN] No rows aggregated.")

    # ========= Significance test (paired) =========

    df = pd.read_csv(master_csv)

    need_cols = {"hparam.seed", "acc_din", "acc_zero_shot"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"MASTER_summary.csv 缺少列: {missing}；请确保主程序写入 acc_din/acc_zero_shot 和 hparam.seed")

    # 只保留两组都存在的样本，并按 seed 对齐
    sub = df[["hparam.seed", "acc_din", "acc_zero_shot"]].dropna().drop_duplicates()
    sub = sub.sort_values("hparam.seed")

    x = sub["acc_din"].to_numpy(dtype=float)
    y = sub["acc_zero_shot"].to_numpy(dtype=float)

    if x.size != y.size or x.size == 0:
        raise ValueError("两组样本数量不一致或为空，请检查 acc_din / acc_zero_shot。")

    # 配对差值
    d = x - y
    n = d.size
    mean_d = float(d.mean())
    std_d  = float(d.std(ddof=1))

    # 配对 t 统计量（不依赖 SciPy）
    t_stat = mean_d / (std_d / np.sqrt(n)) if std_d > 0 else np.inf

    # 置换（符号翻转）检验：配对情况下更合适；两侧
    # 将每个差值随机乘以 +1/-1，计算均值，比较 |mean_d|
    rng = np.random.default_rng(2025)
    n_perm = 200000 if n <= 200 else 50000  # 样本大时降低置换次数以提速
    cnt = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n, replace=True)
        perm_mean = float((d * signs).mean())
        if abs(perm_mean) >= abs(mean_d):
            cnt += 1
    p_perm = (cnt + 1) / (n_perm + 1)

    # 效应量（配对 Cohen's dz）
    cohen_dz = mean_d / std_d if std_d > 0 else np.inf

    print("========== Significance (paired) ==========")
    print(f"n={n}")
    print(f"mean(DIN) = {x.mean():.6f}, mean(ZS) = {y.mean():.6f}")
    print(f"mean_diff  = {mean_d:.6f}  (DIN - ZS)")
    print(f"std_diff   = {std_d:.6f}")
    print(f"t_stat     = {t_stat:.4f}  (paired t, df={n-1})   # p值见置换检验")
    print(f"p_perm     = {p_perm:.6g}  (two-sided, sign-flip)")
    print(f"Cohen's dz = {cohen_dz:.3f}")
    print("Significant ✅" if p_perm < 0.05 else "Not Significant ❌")


if __name__ == "__main__":
    main()