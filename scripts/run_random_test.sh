#!/usr/bin/env bash
set -euo pipefail

PY=../random_vs_din.py
MODEL=/userhome/huggingface/llama3.1-8B-Instruct

POOL=/code/icl/localdataset/gsm8k/train.jsonl
QUERY=/code/icl/localdataset/folio/folio_v2_validation.jsonl
SRC=/code/icl/localdataset/gsm8k/train.jsonl
TGT=/code/icl/localdataset/folio/folio_v2_train.jsonl

TAU=0.3
K_RATIO=0.10
BALANCE=tgt
ZS_MODE=pooled
TP=1
DTYPE=float16
MAX_NEW_TOKENS=4096
TOPK=1
TEMP=0.7
SHOTS=0
TASK=folio

# ===== 用10个不同seed重复跑 =====
for SEED in {52..62}; do
  OUTDIR=/code/icl/tools/random_runs/llama3.1_8b/$(date +"%Y%m%d_%H%M%S")_DIN_vs_VSDIN_seed${SEED}
  mkdir -p "$OUTDIR"
  echo "============================================================"
  echo "[Run $SEED] Results will be saved to: $OUTDIR"
  echo "============================================================"

  # ===== 1) DIN 正式跑法 =====
  echo "[Step 1/2] Running DIN-ICL (seed=$SEED)..."
  python -u "$PY" \
    --model_name "$MODEL" \
    --pool_file "$POOL" \
    --query_file "$QUERY" \
    --src_file "$SRC" \
    --tgt_file "$TGT" \
    --tau "$TAU" \
    --k_ratio "$K_RATIO" \
    --balance "$BALANCE" \
    --zs_mode "$ZS_MODE" \
    --verbose \
    --dtype "$DTYPE" \
    --tensor_parallel_size "$TP" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --icl_shots "$SHOTS" \
    --topk "$TOPK" \
    --temperature "$TEMP" \
    --task_name "$TASK" \
    --seed "$SEED" \
    --compute_din \
    --save_jsonl "$OUTDIR/din_pred.jsonl" \
    2>&1 | tee "$OUTDIR/din.log"

  # ===== 2) 随机子空间（VSDIN）对照 =====
  echo "[Step 2/2] Running VSDIN-ICL (seed=$SEED)..."
  python -u "$PY" \
    --model_name "$MODEL" \
    --pool_file "$POOL" \
    --query_file "$QUERY" \
    --src_file "$SRC" \
    --tgt_file "$TGT" \
    --tau "$TAU" \
    --k_ratio "$K_RATIO" \
    --balance "$BALANCE" \
    --zs_mode "$ZS_MODE" \
    --verbose \
    --dtype "$DTYPE" \
    --tensor_parallel_size "$TP" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --icl_shots "$SHOTS" \
    --topk "$TOPK" \
    --temperature "$TEMP" \
    --task_name "$TASK" \
    --seed "$SEED" \
    --compute_din \
    --randomize_din_subspace \
    --save_jsonl "$OUTDIR/vsdin_pred.jsonl" \
    2>&1 | tee "$OUTDIR/vsdin.log"

  echo "[Run $SEED Done] Results saved in $OUTDIR"
  echo
done

echo "[All 10 runs finished ✅]"
