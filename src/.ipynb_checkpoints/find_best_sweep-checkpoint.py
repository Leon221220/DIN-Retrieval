import csv
from pathlib import Path

ROOT = Path("/code/icl/tools/grid_fallback_runs/qwen2.5_32b/pronto_gsm8k/sweep/")  # 换成你的根目录
TOP_N = 20                                # ← 想看前几名改这里

rows = []
for p in ROOT.rglob("summary.csv"):
    with open(p, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            r["_path"] = str(p)
            def fget(k, default=0.0):
                try:
                    return float(r.get(k, ""))
                except Exception:
                    return default
            r["_acc_din"] = fget("acc_din")
            r["_acc_random"] = fget("acc_random")
            r["_acc_zero_shot"] = fget("acc_zero_shot")
            rows.append(r)

# 排序：优先 acc_din，其次 acc_zero_shot，再其次 acc_random
ranked = sorted(rows, key=lambda r: (-r["_acc_din"], -r["_acc_zero_shot"], -r["_acc_random"]))

if not ranked:
    print("❌ 没找到任何 summary.csv 或 acc_din 字段。")
else:
    print(f"\n=== 前 {TOP_N} 个最佳实验（按 acc_din 排序） ===")
    for i, r in enumerate(ranked[:TOP_N], start=1):
        print(f"[{i:02d}] acc_din={r['_acc_din']:.4f}, "
              f"acc_random={r['_acc_random']:.4f}, acc_zero_shot={r['_acc_zero_shot']:.4f}")
        print("     ", r["_path"])
    print()

    # # 保存汇总文件
    # out_csv = ROOT / "BEST_ranked_by_acc_din.csv"
    # keys = sorted(set().union(*[r.keys() for r in ranked]))
    # with open(out_csv, "w", encoding="utf-8", newline="") as f:
    #     wr = csv.DictWriter(f, fieldnames=keys)
    #     wr.writeheader()
    #     wr.writerows(ranked)
    # print(f"✅ 已保存汇总文件: {out_csv}")
