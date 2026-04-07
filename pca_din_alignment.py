#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA visualization & distance analysis for DIN subspace vs. Random vs. Full
---------------------------------------------------------------------------
- Loads/Computes DIN per layer (compatible with your pipeline & args)
- Builds sentence reps (token-mean) for SRC/TGT in three spaces:
    1) DIN subspace (per-layer indices, concatenated)
    2) Random subspace(s) with same per-layer dimensionality as DIN
    3) Full hidden space (selected last-k layers concatenated)
- Computes & saves:
    * PCA 2D scatter plots (SRC vs TGT) for DIN / Random(1st trial) / Full
    * Distance metrics:
        - Cosine centroid distance
        - MMD with RBF kernel (median heuristic)
        - Mean pairwise cosine distance (with sampling for speed)
    * DDR ratios: DIN / Random, DIN / Full  (lower < 1 => better alignment)
- No external deps beyond numpy, torch, matplotlib, transformers, (your env)
"""

import argparse
import json
import os
import time
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.patches as mpatches

# ==== 新增：带解释度的 PCA ====
def pca_2d_with_var(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回:
      X2d: [N,2]
      evr: [2] 两个主成分解释率
      mu:  [D]
      comps: [2, D]
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    N = X.shape[0]
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:2, :]
    X2d = Xc @ comps.T
    # 解释率：奇异值平方 / 总方差(= Fro^2 / (N-1))
    total_var = (S**2).sum() / max(N-1, 1)
    var12 = (S[:2]**2) / max(N-1, 1)
    evr = var12 / (total_var + 1e-12)
    return X2d, evr, mu.squeeze(0), comps


# ==== 新增：出版级样式 ====
def set_pub_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "figure.autolayout": False,
    })

def _draw_group_scatter(ax, X_src, X_tgt, show_ellipse=True):
    """
    绘制SRC/TGT两类点：
      - 颜色更学术（色盲友好 + 高区分度）
      - 小点高透明度
      - 带质心与椭圆
    """
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors

    # === 对色盲友好调色方案 ===
    # SRC: 蓝灰调，TGT: 深红调（来自 ColorBrewer Set1 / Paul Tol's scheme）
    src_color = (0.3, 0.45, 0.7)   # muted blue
    tgt_color = (0.75, 0.35, 0.35) # muted red

    # 小点更学术：不完全透明，但不过分鲜艳
    s_kwargs = dict(s=10, alpha=0.6, linewidths=0)

    src = ax.scatter(X_src[:, 0], X_src[:, 1], label="SRC", color=src_color, **s_kwargs)
    tgt = ax.scatter(X_tgt[:, 0], X_tgt[:, 1], label="TGT", color=tgt_color, **s_kwargs)

    # === 质心标记 ===
    cs = X_src.mean(axis=0)
    ct = X_tgt.mean(axis=0)
    ax.scatter([cs[0]], [cs[1]], marker="x", s=60, linewidths=1.3, color=src_color)
    ax.scatter([ct[0]], [ct[1]], marker="+", s=70, linewidths=1.3, color=tgt_color)

    # === 椭圆（1σ区域）===
    if show_ellipse:
        for group, color in [(X_src, src_color), (X_tgt, tgt_color)]:
            if group.shape[0] >= 5:
                mu = group.mean(axis=0)
                cov = np.cov(group[:, 0], group[:, 1])
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                width, height = 2 * np.sqrt(vals)
                ell = mpatches.Ellipse(
                    xy=mu, width=width, height=height, angle=theta,
                    fill=False, lw=1.0, alpha=0.9, edgecolor=color
                )
                ax.add_patch(ell)

    # === 样式 ===
    ax.grid(True, which="major", linestyle="--", alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.15)
    ax.set_aspect("equal", adjustable="box")
    return src, tgt


def _format_axes(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def _cov_ellipse_params(X2: np.ndarray):
    """返回 2x2 协方差矩阵与均值，用于画 1σ 椭圆。"""
    mu = X2.mean(axis=0)
    C = np.cov(X2.T)
    # 特征分解
    vals, vecs = np.linalg.eigh(C)
    # 最大特征值对应的方向
    order = np.argsort(vals)[::-1]
    vals = vals[order]; vecs = vecs[:, order]
    # 1σ 半轴长度
    width, height = 2.0 * np.sqrt(vals)   # 2*sqrt(lambda) ≈ 1σ直径
    angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
    return mu, width, height, angle

def plot_two_spaces_one_figure(
    din_src_2d, din_tgt_2d,
    full_src_2d, full_tgt_2d,
    metrics: dict,
    save_path: str,
    evr_map: dict = None,
    panel_letters=("A", "B")
):
    """
    绘制两个空间 (DIN vs FULL) 的对比图。
    每个子图含：
      - 源域 / 目标域点云
      - 椭圆 (1σ)
      - 质心标记
      - 主成分解释率
      - 指标 (cos-cent, pair-cos)
    """
    set_pub_style()

    # 统一坐标范围
    all_x = np.concatenate([din_src_2d, din_tgt_2d, full_src_2d, full_tgt_2d], axis=0)
    x_min, x_max = all_x[:, 0].min(), all_x[:, 0].max()
    y_min, y_max = all_x[:, 1].min(), all_x[:, 1].max()
    pad_x = 0.05 * (x_max - x_min + 1e-6)
    pad_y = 0.05 * (y_max - y_min + 1e-6)

    # 创建并排的两个子图
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6), sharex=False, sharey=False)
    spaces = [
        ("DIN", din_src_2d, din_tgt_2d),
        ("FULL", full_src_2d, full_tgt_2d),
    ]

    handles = None
    for ax, (name, xs, xt), pl in zip(axes, spaces, panel_letters):
        _draw_group_scatter(ax, xs, xt, show_ellipse=True)
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)

        evr = (np.nan, np.nan)
        if evr_map and name in evr_map:
            evr = evr_map[name]

        ax.set_xlabel(f"PC1 ({(evr[0]*100):.1f}%)" if np.isfinite(evr[0]) else "PC1")
        ax.set_ylabel(f"PC2 ({(evr[1]*100):.1f}%)" if np.isfinite(evr[1]) else "PC2")

        # 标题：面板字母 + 指标信息
        m = metrics.get(name, {})
        ax.set_title(
            f"({pl}) {name}   cos-cent={m.get('cos', float('nan')):.3f}   "
            f"pair-cos={m.get('pair', float('nan')):.3f}",
            pad=4
        )

        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    # 外置图例
    fig.legend(handles, ["SRC", "TGT"], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    root, _ = os.path.splitext(save_path)
    fig.savefig(root + ".pdf", bbox_inches="tight")
    plt.close(fig)


# ==== 替换：更学术的三空间对比 ====
def plot_three_spaces_one_figure(
    din_src_2d, din_tgt_2d, full_src_2d, full_tgt_2d, rand_src_2d, rand_tgt_2d,
    metrics: dict, save_path: str, evr_map: dict = None, panel_letters=("A", "B", "C")
):
    """
    metrics 例:
      {"DIN":{"cos":..,"pair":..}, "FULL":{...}, "RAND":{...}}
    evr_map 例:
      {"DIN": (evr1, evr2), "FULL": (...), "RAND": (...)}
    """
    set_pub_style()
    # 统一坐标范围
    all_x = np.concatenate([
        din_src_2d, din_tgt_2d, full_src_2d, full_tgt_2d, rand_src_2d, rand_tgt_2d
    ], axis=0)
    x_min, x_max = all_x[:,0].min(), all_x[:,0].max()
    y_min, y_max = all_x[:,1].min(), all_x[:,1].max()
    pad_x = 0.05 * (x_max - x_min + 1e-6)
    pad_y = 0.05 * (y_max - y_min + 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(14.7, 4.6), sharex=False, sharey=False)
    spaces = [
        ("DIN",  din_src_2d,  din_tgt_2d),
        ("FULL", full_src_2d, full_tgt_2d),
        ("RAND", rand_src_2d, rand_tgt_2d),
    ]

    # 逐子图绘制
    handles = None
    for ax, (name, xs, xt), pl in zip(axes, spaces, panel_letters):
        _draw_group_scatter(ax, xs, xt, show_ellipse=True)
        # 轴限、标签（含解释率）
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)

        evr = (np.nan, np.nan)
        if evr_map and name in evr_map:
            evr = evr_map[name]
        ax.set_xlabel(f"PC1 ({(evr[0]*100):.1f}%)" if np.isfinite(evr[0]) else "PC1")
        ax.set_ylabel(f"PC2 ({(evr[1]*100):.1f}%)" if np.isfinite(evr[1]) else "PC2")

        # 标题：面板字母 + 简要指标
        m = metrics.get(name, {})
        ax.set_title(f"({pl}) {name}   cos-cent={m.get('cos', float('nan')):.3f}   "
                     f"pair-cos={m.get('pair', float('nan')):.3f}", pad=4)

        # 记录一个句柄用于外置图例
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    # 外置图例
    fig.legend(handles, ["SRC", "TGT"], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    root, _ = os.path.splitext(save_path)
    fig.savefig(root + ".pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Utilities (lightweight)
# -----------------------------
def ensure_dir(p):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

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
    return ex.get("context") or ex.get("passage") or ex.get("question") or ex.get("text") or ex.get("premises") or ""

def parse_layers(s: str) -> List[int]:
    if not s:
        return [-6,-5,-4,-3,-2,-1]
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------
# Hidden-state collection
# -----------------------------
@torch.no_grad()
def batch_token_means(
    texts: List[str],
    tok: AutoTokenizer,
    hf_model: AutoModelForCausalLM,
    layer_signed: List[int],
    batch_size: int = 8,
    max_len: int = 1024,
) -> Dict[int, np.ndarray]:
    """
    Returns {layer_signed: [N, d]} of token-mean hidden states for each requested layer index.
    layer_signed can be negative (e.g., -1 = last layer).
    """
    device = next(hf_model.parameters()).device
    out_dict: Dict[int, List[np.ndarray]] = {Ls: [] for Ls in layer_signed}

    # probe num layers
    enc0 = tok("dummy", return_tensors="pt").to(device)
    out0 = hf_model(**enc0, output_hidden_states=True, use_cache=False)
    num_layers = len(out0.hidden_states) - 1

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        out = hf_model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple of length num_layers+1
        for Ls in layer_signed:
            L = num_layers + Ls if Ls < 0 else Ls
            if L < 0 or L >= len(hs):
                continue
            H = hs[L]  # (B, seq, d)
            H = H.mean(dim=1)  # (B, d)
            out_dict[Ls].append(H.detach().float().cpu().numpy())

    mats: Dict[int, np.ndarray] = {}
    for Ls, parts in out_dict.items():
        if parts:
            mats[Ls] = np.concatenate(parts, axis=0)
        else:
            mats[Ls] = np.zeros((0, 0), dtype=np.float32)
    return mats

# -----------------------------
# DIN selection (re-implemented minimal, or load)
# -----------------------------
def select_din_same_sign(zS: np.ndarray, zT: np.ndarray, tau: float, k_ratio: float) -> List[int]:
    pos = np.where((zS > tau) & (zT > tau))[0]
    neg = np.where((zS < -tau) & (zT < -tau))[0]
    cand = np.concatenate([pos, neg], axis=0)
    d = zS.shape[0]
    K = max(1, int(round(k_ratio * d)))
    if cand.size == 0:
        # fallback: top by |zS|+|zT|
        scores = np.abs(zS) + np.abs(zT)
        return scores.argsort()[::-1][:K].tolist()
    scores = (np.abs(zS[cand]) + np.abs(zT[cand]))
    idx = scores.argsort()[::-1][:min(K, cand.size)]
    return cand[idx].tolist()

def compute_din_spec_from_hidden(
    Hs: Dict[int, np.ndarray],
    Ht: Dict[int, np.ndarray],
    tau: float = 1.0,
    k_ratio: float = 0.05,
    zs_mode: str = "pooled",
) -> Dict[int, List[int]]:
    """
    Given per-layer token-mean matrices for SRC/TGT, return {layer_signed: indices}
    """
    din: Dict[int, List[int]] = {}
    for Ls in Hs.keys():
        S = Hs[Ls]; T = Ht[Ls]
        if S.size == 0 and T.size == 0:
            din[Ls] = []; continue
        if S.size == 0:
            d = T.shape[1]; S = np.zeros((0, d), dtype=np.float32)
        if T.size == 0:
            d = S.shape[1]; T = np.zeros((0, d), dtype=np.float32)
        d = S.shape[1]

        if zs_mode == "union_weighted":
            U = np.concatenate([S, T], axis=0)
            mu = U.mean(axis=0); sigma = U.std(axis=0) + 1e-6
        elif zs_mode == "union_equal":
            mu = 0.5*(S.mean(axis=0) + T.mean(axis=0))
            sigma = np.sqrt(0.5*(S.var(axis=0) + T.var(axis=0))) + 1e-6
        elif zs_mode == "pooled":
            muS, muT = S.mean(axis=0), T.mean(axis=0)
            varS, varT = S.var(axis=0), T.var(axis=0)
            mu = 0.5*(muS + muT)
            sigma = np.sqrt(0.5*(varS + varT)) + 1e-6
        else:
            raise ValueError(f"Unknown zs_mode={zs_mode}")

        zS = ((S - mu) / sigma).mean(axis=0) if S.shape[0] > 0 else np.zeros(d, dtype=np.float32)
        zT = ((T - mu) / sigma).mean(axis=0) if T.shape[0] > 0 else np.zeros(d, dtype=np.float32)

        din[Ls] = select_din_same_sign(zS, zT, tau=tau, k_ratio=k_ratio)
    return din

# -----------------------------
# Vectorization in 3 spaces
# -----------------------------
def build_vectors_by_selection(
    H: Dict[int, np.ndarray],
    selection: Dict[int, Optional[List[int]]],  # None => full layer; [] => skip layer
) -> np.ndarray:
    """
    H[Ls] -> [N, d_l]. selection[Ls] -> indices in that layer (or None for full).
    Concatenate selected dims across layers => [N, sum_k dims_k]
    """
    parts = []
    N = None
    for Ls, mat in H.items():
        if mat.size == 0:
            continue
        if N is None:
            N = mat.shape[0]
        assert mat.shape[0] == N, "Mismatched sample size across layers"
        sel = selection.get(Ls, None)
        if sel is None:
            parts.append(mat)             # full layer
        else:
            if len(sel) == 0:             # skip layer
                continue
            sel_arr = np.array(sel, dtype=int)
            sel_arr = sel_arr[(sel_arr >= 0) & (sel_arr < mat.shape[1])]
            if sel_arr.size > 0:
                parts.append(mat[:, sel_arr])
    if not parts:
        return np.zeros((0,0), dtype=np.float32)
    return np.concatenate(parts, axis=1)

# -----------------------------
# PCA via SVD (no sklearn)
# -----------------------------
def pca_2d(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Center X, do SVD, project to top-2.
    Returns (X2d, mean, components[2, D])
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    # economy SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:2, :]             # [2, D]
    X2d = Xc @ comps.T            # [N, 2]
    return X2d, mu.squeeze(0), comps

# -----------------------------
# Distances
# -----------------------------
def cosine(u, v, eps=1e-8):
    nu = np.linalg.norm(u) + eps
    nv = np.linalg.norm(v) + eps
    return np.dot(u, v) / (nu * nv)

def cosine_centroid_distance(A: np.ndarray, B: np.ndarray) -> float:
    if A.size == 0 or B.size == 0:
        return float("nan")
    ca = A.mean(axis=0); cb = B.mean(axis=0)
    return 1.0 - float(cosine(ca, cb))

def rbf_kernel_matrix(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
    # ||x - y||^2 = |x|^2 + |y|^2 - 2x·y
    X2 = np.sum(X*X, axis=1, keepdims=True)
    Y2 = np.sum(Y*Y, axis=1, keepdims=True)
    XY = X @ Y.T
    d2 = X2 + Y2.T - 2.0*XY
    if gamma is None:
        # median heuristic over concatenated distances
        Z = np.concatenate([X, Y], axis=0)
        Z2 = np.sum(Z*Z, axis=1, keepdims=True)
        D2 = Z2 + Z2.T - 2.0*(Z @ Z.T)
        med = np.median(D2[np.triu_indices_from(D2, k=1)])
        if med <= 0:
            med = np.mean(D2)
        gamma = 1.0 / (2.0 * (med + 1e-8))
    K = np.exp(-gamma * np.maximum(d2, 0.0))
    return K

def mmd_rbf(X: np.ndarray, Y: np.ndarray) -> float:
    if X.size == 0 or Y.size == 0:
        return float("nan")
    Kxx = rbf_kernel_matrix(X, X)
    Kyy = rbf_kernel_matrix(Y, Y)
    Kxy = rbf_kernel_matrix(X, Y)
    n = X.shape[0]; m = Y.shape[0]
    # unbiased estimate
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    term_x = Kxx.sum() / (n*(n-1) + 1e-8)
    term_y = Kyy.sum() / (m*(m-1) + 1e-8)
    term_xy = 2.0 * Kxy.mean()
    return float(term_x + term_y - term_xy)

def mean_pairwise_cosine(A: np.ndarray, B: np.ndarray, max_pairs: int = 50000, seed: int = 42) -> float:
    if A.size == 0 or B.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    NA, NB = A.shape[0], B.shape[0]
    total = min(max_pairs, NA * NB)
    ia = rng.integers(0, NA, size=total)
    ib = rng.integers(0, NB, size=total)
    # normalize rows
    def norm_rows(M):
        n = np.linalg.norm(M, axis=1, keepdims=True) + 1e-8
        return M / n
    A_ = norm_rows(A); B_ = norm_rows(B)
    sims = np.sum(A_[ia] * B_[ib], axis=1)  # cosine
    return float(1.0 - sims.mean())         # distance

# -----------------------------
# Plotting
# -----------------------------
def scatter_src_tgt(X2d_src: np.ndarray, X2d_tgt: np.ndarray,
                    title: str, save_path: str,
                    evr: Tuple[float, float] = (np.nan, np.nan),
                    save_pdf_also: bool = True):
    set_pub_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    _draw_group_scatter(ax, X2d_src, X2d_tgt, show_ellipse=True)

    xlabel = f"PC1 ({evr[0]*100:.1f}%)" if np.isfinite(evr[0]) else "PC1"
    ylabel = f"PC2 ({evr[1]*100:.1f}%)" if np.isfinite(evr[1]) else "PC2"
    _format_axes(ax, xlabel, ylabel, title)

    # 图例放外侧，避免遮挡
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if save_pdf_also:
        root, _ = os.path.splitext(save_path)
        fig.savefig(root + ".pdf", bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_name", type=str, default="prontoqa")  # for future use
    ap.add_argument("--model_name", type=str, required=True, help="HF model for hidden states")
    ap.add_argument("--src_file", type=str, required=True)
    ap.add_argument("--tgt_file", type=str, required=True)
    ap.add_argument("--layers", type=str, default="-6,-5,-4,-3,-2,-1")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--k_ratio", type=float, default=0.05)
    ap.add_argument("--zs_mode", type=str, default="pooled", choices=["union_weighted","union_equal","pooled"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--n_src", type=int, default=None, help="optional limit")
    ap.add_argument("--n_tgt", type=int, default=None, help="optional limit")
    ap.add_argument("--balance", type=str, default="min", choices=["none","min","tgt"])
    ap.add_argument("--din_file", type=str, default=None, help="load DIN spec (json). If absent, compute.")
    ap.add_argument("--random_trials", type=int, default=5)
    ap.add_argument("--pairwise_max_pairs", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="./pca_din_outputs")
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    layer_signed = parse_layers(args.layers)

    # ----- Load data -----
    src_items = load_json_or_jsonl(args.src_file)
    tgt_items = load_json_or_jsonl(args.tgt_file)
    src_texts = [get_text(ex) for ex in src_items]
    tgt_texts = [get_text(ex) for ex in tgt_items]

    # balance
    import random
    if args.n_src is not None:
        src_texts = src_texts[:args.n_src]
    if args.n_tgt is not None:
        tgt_texts = tgt_texts[:args.n_tgt]
    if args.balance in ("min","tgt"):
        if args.balance == "min":
            m = min(len(src_texts), len(tgt_texts))
            if len(src_texts) > m: src_texts = random.sample(src_texts, m)
            if len(tgt_texts) > m: tgt_texts = random.sample(tgt_texts, m)
        elif args.balance == "tgt":
            if len(src_texts) > len(tgt_texts):
                src_texts = random.sample(src_texts, len(tgt_texts))

    print(f"[Data] nSRC={len(src_texts)}  nTGT={len(tgt_texts)}")

    # ----- HF model -----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # ----- Collect hidden -----
    print("[Hidden] SRC...")
    Hs = batch_token_means(src_texts, tok, model, layer_signed, batch_size=args.batch_size, max_len=args.max_len)
    print("[Hidden] TGT...")
    Ht = batch_token_means(tgt_texts, tok, model, layer_signed, batch_size=args.batch_size, max_len=args.max_len)

    # ----- DIN spec -----
    if args.din_file and os.path.exists(args.din_file):
        with open(args.din_file, "r", encoding="utf-8") as f:
            din_spec_raw = json.load(f)
        # normalize keys to int, values to list[int]
        din_spec = {int(k): list(map(int, v if isinstance(v, list) else v.get("indices", []))) for k, v in din_spec_raw.items()}
        print(f"[DIN] Loaded from {args.din_file}")
    else:
        print("[DIN] Computing DIN spec from hidden...")
        din_spec = compute_din_spec_from_hidden(Hs, Ht, tau=args.tau, k_ratio=args.k_ratio, zs_mode=args.zs_mode)
        save_path = os.path.join(args.out_dir, "din_spec_computed.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(din_spec, f, ensure_ascii=False, indent=2)
        print(f"[DIN] Saved -> {save_path}")

    # ----- Build selections -----
    # DIN selection: given
    sel_din: Dict[int, Optional[List[int]]] = {Ls: din_spec.get(Ls, []) for Ls in layer_signed}
    # FULL selection: None => full layer
    sel_full: Dict[int, Optional[List[int]]] = {Ls: None for Ls in layer_signed}

    # ----- Build vectors -----
    XS_din = build_vectors_by_selection(Hs, sel_din)
    XT_din = build_vectors_by_selection(Ht, sel_din)
    XS_full = build_vectors_by_selection(Hs, sel_full)
    XT_full = build_vectors_by_selection(Ht, sel_full)

    print(f"[Vec] DIN shapes  SRC={XS_din.shape}  TGT={XT_din.shape}")
    print(f"[Vec] FULL shapes SRC={XS_full.shape} TGT={XT_full.shape}")

    # ----- Random subspace trials (same per-layer dims as DIN) -----
    rng = np.random.default_rng(args.seed)
    rand_trials = []
    for t in range(args.random_trials):
        sel_rand: Dict[int, Optional[List[int]]] = {}
        for Ls in layer_signed:
            mat = Hs[Ls]
            if mat.size == 0:
                sel_rand[Ls] = []
                continue
            d = mat.shape[1]
            kL = len(sel_din.get(Ls, []))
            if kL <= 0:
                sel_rand[Ls] = []
                continue
            inds = rng.choice(d, size=kL, replace=False).tolist()
            sel_rand[Ls] = inds
        XS_r = build_vectors_by_selection(Hs, sel_rand)
        XT_r = build_vectors_by_selection(Ht, sel_rand)
        rand_trials.append((XS_r, XT_r))

    # ----- Distances -----
    def compute_all_dists(XS, XT, tag: str) -> Dict[str, float]:
        dists = {
            "cosine_centroid": cosine_centroid_distance(XS, XT),
            "mmd_rbf": mmd_rbf(XS, XT),
            "mean_pairwise_cosine": mean_pairwise_cosine(XS, XT, max_pairs=args.pairwise_max_pairs, seed=args.seed),
        }
        print(f"[Dist-{tag}] {dists}")
        return dists

    dist_din  = compute_all_dists(XS_din,  XT_din,  "DIN")
    dist_full = compute_all_dists(XS_full, XT_full, "FULL")

    dist_rand_list = []
    for i, (XS_r, XT_r) in enumerate(rand_trials):
        dist_rand_list.append(compute_all_dists(XS_r, XT_r, f"RAND#{i+1}"))
    # summarize random
    keys = list(dist_din.keys())
    dist_rand_mean = {k: float(np.mean([d[k] for d in dist_rand_list])) for k in keys}
    dist_rand_std  = {k: float(np.std([d[k] for d in dist_rand_list], ddof=0)) for k in keys}
    print(f"[Dist-RAND] mean={dist_rand_mean}  std={dist_rand_std}")

    # ----- DDR ratios -----
    ddr = {
        "DDR_din_over_rand": {k: float(dist_din[k] / (dist_rand_mean[k] + 1e-12)) for k in keys},
        "DDR_din_over_full": {k: float(dist_din[k] / (dist_full[k] + 1e-12)) for k in keys},
    }
    print(f"[DDR] {ddr}")

    # ----- PCA plots -----
    def pca_and_plot(XS, XT, title: str, save_name: str, out_dir: str):
        X = np.concatenate([XS, XT], axis=0)
        X2d, evr, _, _ = pca_2d_with_var(X)
        Xs2 = X2d[:XS.shape[0]]; Xt2 = X2d[XS.shape[0]:]
        path = os.path.join(out_dir, save_name)
        scatter_src_tgt(Xs2, Xt2, title=title, save_path=path, evr=tuple(evr.tolist()))
        return path, (Xs2, Xt2, evr)

    p_din,  (Xs2_din,  Xt2_din,  evr_din)  = pca_and_plot(XS_din,  XT_din,  "PCA in DIN subspace",   "pca_din.png",  args.out_dir)
    p_rand, (Xs2_rand, Xt2_rand, evr_rand) = pca_and_plot(rand_trials[0][0], rand_trials[0][1], "PCA in Random subspace (trial #1)", "pca_random.png", args.out_dir)
    p_full, (Xs2_full, Xt2_full, evr_full) = pca_and_plot(XS_full, XT_full, "PCA in Full hidden space", "pca_full.png", args.out_dir)

    
    X_din_all  = np.concatenate([XS_din,  XT_din], axis=0)
    X_full_all = np.concatenate([XS_full, XT_full], axis=0)
    X_rand0_S, X_rand0_T = rand_trials[0]  # 取第1个随机试验用于展示
    X_rand_all = np.concatenate([X_rand0_S, X_rand0_T], axis=0)

    # 各自空间内做 PCA -> 2D
    din_2d, _, _   = pca_2d(X_din_all)
    full_2d, _, _  = pca_2d(X_full_all)
    rand_2d, _, _  = pca_2d(X_rand_all)

    din_src_2d = din_2d[:XS_din.shape[0]]
    din_tgt_2d = din_2d[XS_din.shape[0]:]

    full_src_2d = full_2d[:XS_full.shape[0]]
    full_tgt_2d = full_2d[XS_full.shape[0]:]

    rand_src_2d = rand_2d[:X_rand0_S.shape[0]]
    rand_tgt_2d = rand_2d[X_rand0_S.shape[0]:]

    # 指标字典（从你已有的 dist 结果里拿）
    metrics_for_title = {
        "DIN":  {"cos": dist_din["cosine_centroid"],  "pair": dist_din["mean_pairwise_cosine"]},
        "FULL": {"cos": dist_full["cosine_centroid"], "pair": dist_full["mean_pairwise_cosine"]},
        "RAND": {"cos": dist_rand_mean["cosine_centroid"], "pair": dist_rand_mean["mean_pairwise_cosine"]},
    }

    evr_map = {"DIN": evr_din, "FULL": evr_full, "RAND": evr_rand}

    save_combo = os.path.join(args.out_dir, "pca_compare_DIN_FULL.png")
    plot_two_spaces_one_figure(
        Xs2_din, Xt2_din,
        Xs2_full, Xt2_full,
        {
            "DIN":  {"cos": dist_din["cosine_centroid"],  "pair": dist_din["mean_pairwise_cosine"]},
            "FULL": {"cos": dist_full["cosine_centroid"], "pair": dist_full["mean_pairwise_cosine"]},
        },
        save_combo,
        evr_map={"DIN": evr_din, "FULL": evr_full},
    )

    print(f"[Save] Combined figure -> {save_combo}")


    # ----- Save metrics -----
    result = {
        "args": vars(args),
        "shapes": {
            "din":  {"src": list(XS_din.shape),  "tgt": list(XT_din.shape)},
            "full": {"src": list(XS_full.shape), "tgt": list(XT_full.shape)},
            "rand_trial0": {"src": list(rand_trials[0][0].shape), "tgt": list(rand_trials[0][1].shape)} if rand_trials else {}
        },
        "distances": {
            "din": dist_din,
            "full": dist_full,
            "rand_mean": dist_rand_mean,
            "rand_std": dist_rand_std
        },
        "DDR": ddr,
        "plots": {
            "din": p_din,
            "random_trial1": p_rand,
            "full": p_full
        }
    }
    save_json = os.path.join(args.out_dir, "pca_alignment_summary.json")
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[Save] Summary -> {save_json}")
    print(f"[Save] Plots   -> {p_din}, {p_rand}, {p_full}")

if __name__ == "__main__":
    main()
