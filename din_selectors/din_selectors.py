#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
din_selectors.py
----------------
A collection of alternative DIN (Domain-Invariant Neuron) selection strategies.

Each selector returns a list[int] of chosen indices (0-based) for a given layer.
You can mix-and-match these methods in your pipeline, e.g., use one as a
fallback when the sign-consistent set is empty.

Minimal dependencies: numpy only.

Typical usage:
--------------
from din_selectors import (
    select_din_same_sign,
    select_din_topk_strength,
    select_din_intersect_topk,
    select_din_rank_agg,
    select_din_soft_sign,
    select_din_pooled_mahalanobis,
    select_din_fisher,
    select_din_bootstrap_freq,
    select_din_stability_sign,
    dispatch_select_din,
)

# Given zS, zT (dimension-wise z-scores for Source/Target), and hidden matrices Hs, Ht:
idx = select_din_same_sign(zS, zT, tau=1.0, k_ratio=0.05)
# or
idx = dispatch_select_din("rank_agg", zS=zS, zT=zT, k_ratio=0.05)

Author: yanjzh
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple, Dict, Literal

# -----------------------------
# Helpers
# -----------------------------

def _ensure_1d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr).reshape(-1)
    return arr

def _topk_indices(values: np.ndarray, k: int) -> List[int]:
    if k <= 0:
        return []
    k = min(k, values.shape[0])
    # argpartition faster than full sort
    part = np.argpartition(-values, k-1)[:k]
    # sort within top-k for determinism
    order = part[np.argsort(-values[part])]
    return order.astype(int).tolist()

def _calc_K(d: int, k_ratio: float) -> int:
    return max(int(np.floor(k_ratio * max(1, d))), 1)

def _strength_joint_abs(zS: np.ndarray, zT: np.ndarray) -> np.ndarray:
    """|zS| + |zT|"""
    return np.abs(zS) + np.abs(zT)

def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(None if seed is None else seed)

# -----------------------------
# 1) Strict same-sign, double-threshold (paper default)
# -----------------------------
def select_din_same_sign(
    zS: np.ndarray,
    zT: np.ndarray,
    tau: float = 1.0,
    k_ratio: float = 0.05,
) -> List[int]:
    """
    Select dims where zS and zT are both > +tau OR both < -tau.
    Rank by |zS|+|zT| and keep Top-K (K=floor(k_ratio*d)).
    If empty, returns empty list (caller may fallback to other methods).
    """
    zS = _ensure_1d(zS); zT = _ensure_1d(zT)
    d = zS.shape[0]
    K = _calc_K(d, k_ratio)
    mask = ((zS > tau) & (zT > tau)) | ((zS < -tau) & (zT < -tau))
    I = np.where(mask)[0]
    if I.size == 0:
        return []
    strength = _strength_joint_abs(zS[I], zT[I])
    order = _topk_indices(strength, K)
    return I[order].astype(int).tolist()

# -----------------------------
# 2) Pure Top-K by joint strength (fallback used in your code)
# -----------------------------
def select_din_topk_strength(
    zS: np.ndarray,
    zT: np.ndarray,
    k_ratio: float = 0.05,
) -> List[int]:
    """
    Ignore sign-consistency and thresholds.
    Pick Top-K by |zS|+|zT| over all dims.
    """
    zS = _ensure_1d(zS); zT = _ensure_1d(zT)
    d = zS.shape[0]
    K = _calc_K(d, k_ratio)
    strength_all = _strength_joint_abs(zS, zT)
    return _topk_indices(strength_all, K)

# -----------------------------
# 3) Intersect top-pct sets per domain
# -----------------------------
def select_din_intersect_topk(
    zS: np.ndarray,
    zT: np.ndarray,
    frac: float = 0.1,     # take top 10% per domain by |z|, then intersect; finally rank by |zS|+|zT|
    k_ratio: float = 0.05,
) -> List[int]:
    zS = _ensure_1d(zS); zT = _ensure_1d(zT)
    d = zS.shape[0]; K = _calc_K(d, k_ratio)
    m = max(1, int(np.floor(frac * d)))

    topS = _topk_indices(np.abs(zS), m)
    topT = _topk_indices(np.abs(zT), m)
    inter = np.intersect1d(np.array(topS), np.array(topT), assume_unique=False)
    if inter.size == 0:
        return []
    strength = _strength_joint_abs(zS[inter], zT[inter])
    order = _topk_indices(strength, K)
    return inter[order].astype(int).tolist()

# -----------------------------
# 4) Rank aggregation (Borda-like)
# -----------------------------
def select_din_rank_agg(
    zS: np.ndarray,
    zT: np.ndarray,
    k_ratio: float = 0.05,
    weights: Tuple[float, float] = (1.0, 1.0),
) -> List[int]:
    """
    Rank dims by |zS| and |zT| separately, convert ranks to scores, add with weights, pick Top-K.
    Encourages dims that are highly ranked in both domains.
    """
    zS = _ensure_1d(zS); zT = _ensure_1d(zT)
    d = zS.shape[0]; K = _calc_K(d, k_ratio)
    rS = np.argsort(-np.abs(zS))  # indices sorted by |zS| desc
    rT = np.argsort(-np.abs(zT))
    # inverse ranks as scores (d for best, 1 for worst)
    score = np.zeros(d, dtype=np.float64)
    inv_rank_S = np.empty(d, dtype=int); inv_rank_S[rS] = np.arange(d, 0, -1)
    inv_rank_T = np.empty(d, dtype=int); inv_rank_T[rT] = np.arange(d, 0, -1)
    score = weights[0]*inv_rank_S + weights[1]*inv_rank_T
    return _topk_indices(score.astype(float), K)

# -----------------------------
# 5) Soft sign-consistency with margin (tolerant alternative)
# -----------------------------
def select_din_soft_sign(
    zS: np.ndarray,
    zT: np.ndarray,
    tau: float = 1.0,
    margin: float = 0.2,
    k_ratio: float = 0.05,
) -> List[int]:
    """
    Require same sign, but allow one side to be slightly below tau if the other is strong.
    Conditions:
      - sign(zS) == sign(zT) != 0
      - min(|zS|, |zT|) >= tau - margin
    Rank by |zS|+|zT|.
    """
    zS = _ensure_1d(zS); zT = _ensure_1d(zT)
    d = zS.shape[0]; K = _calc_K(d, k_ratio)
    sgn = np.sign(zS) * np.sign(zT)
    mask = (sgn > 0) & (np.minimum(np.abs(zS), np.abs(zT)) >= (tau - margin))
    I = np.where(mask)[0]
    if I.size == 0:
        return []
    strength = _strength_joint_abs(zS[I], zT[I])
    order = _topk_indices(strength, K)
    return I[order].astype(int).tolist()

# -----------------------------
# 6) Pooled Mahalanobis distance on means (Hs/Ht matrices needed)
# -----------------------------
def select_din_pooled_mahalanobis(
    Hs: np.ndarray,  # shape [nS, d]
    Ht: np.ndarray,  # shape [nT, d]
    k_ratio: float = 0.05,
    eps: float = 1e-6,
) -> List[int]:
    """
    Score each dim by a 1D pooled Mahalanobis-like signal:
        score_k = |muS_k - muT_k| / sqrt( 0.5*(varS_k + varT_k) + eps )
    Then pick Top-K by *LOW* score to prefer dims with similar means across domains
    but non-trivial variance (domains agree).
    """
    if Hs.size == 0 and Ht.size == 0:
        return []
    if Hs.size == 0:
        Hs = np.zeros((0, Ht.shape[1]), dtype=np.float32)
    if Ht.size == 0:
        Ht = np.zeros((0, Hs.shape[1]), dtype=np.float32)

    muS, muT = Hs.mean(axis=0) if Hs.size else 0.0, Ht.mean(axis=0) if Ht.size else 0.0
    varS = Hs.var(axis=0) if Hs.size else 0.0
    varT = Ht.var(axis=0) if Ht.size else 0.0

    denom = np.sqrt(0.5*(varS + varT) + eps)
    score = np.abs(muS - muT) / (denom + eps)  # smaller is "more invariant"
    d = score.shape[0]; K = _calc_K(d, k_ratio)
    # pick smallest scores
    order = np.argpartition(score, K-1)[:K]
    order = order[np.argsort(score[order])]
    return order.astype(int).tolist()

# -----------------------------
# 7) Fisher-style variance ratio (prefer dims with small between-domain diff)
# -----------------------------
def select_din_fisher(
    Hs: np.ndarray,
    Ht: np.ndarray,
    k_ratio: float = 0.05,
    eps: float = 1e-6,
) -> List[int]:
    """
    Fisher-like criterion (inverted): prefer dims where between-domain mean diff is small
    relative to within-domain variance.
        score_k = ( (muS_k - muT_k)^2 ) / ( varS_k + varT_k + eps )
    Pick Top-K by *LOW* score to emphasize invariance.
    """
    if Hs.size == 0 and Ht.size == 0:
        return []
    if Hs.size == 0:
        Hs = np.zeros((0, Ht.shape[1]), dtype=np.float32)
    if Ht.size == 0:
        Ht = np.zeros((0, Hs.shape[1]), dtype=np.float32)

    muS, muT = Hs.mean(axis=0) if Hs.size else 0.0, Ht.mean(axis=0) if Ht.size else 0.0
    varS = Hs.var(axis=0) if Hs.size else 0.0
    varT = Ht.var(axis=0) if Ht.size else 0.0

    num = (muS - muT)**2
    den = (varS + varT + eps)
    score = num / den
    d = score.shape[0]; K = _calc_K(d, k_ratio)
    order = np.argpartition(score, K-1)[:K]
    order = order[np.argsort(score[order])]
    return order.astype(int).tolist()

# -----------------------------
# 8) Bootstrap frequency (stability selection)
# -----------------------------
def select_din_bootstrap_freq(
    Hs: np.ndarray,
    Ht: np.ndarray,
    base: Literal["same_sign","topk_strength","intersect","rank_agg","soft_sign","maha","fisher"] = "same_sign",
    iters: int = 50,
    sample_frac: float = 0.7,
    tau: float = 1.0,
    k_ratio: float = 0.05,
    seed: Optional[int] = 42,
) -> List[int]:
    """
    Re-sample S/T with replacement, re-compute z-scores (pooled), run a base method,
    count selection frequency for each dim, and finally take top-K by frequency.
    More robust to noise/config choices.
    """
    rng = _rng(seed)

    # compute baseline statistics once to get d
    d = Hs.shape[1] if Hs.size else (Ht.shape[1] if Ht.size else 0)
    if d == 0:
        return []

    freq = np.zeros(d, dtype=np.int32)
    nS = max(1, int(np.ceil(sample_frac * Hs.shape[0]))) if Hs.size else 0
    nT = max(1, int(np.ceil(sample_frac * Ht.shape[0]))) if Ht.size else 0

    for _ in range(iters):
        S_sub = Hs[rng.integers(0, Hs.shape[0], size=nS)] if Hs.size else np.zeros((0, d), dtype=np.float32)
        T_sub = Ht[rng.integers(0, Ht.shape[0], size=nT)] if Ht.size else np.zeros((0, d), dtype=np.float32)

        # pooled stats -> zS/zT
        muS, muT = (S_sub.mean(axis=0) if S_sub.size else 0.0), (T_sub.mean(axis=0) if T_sub.size else 0.0)
        varS, varT = (S_sub.var(axis=0) if S_sub.size else 0.0), (T_sub.var(axis=0) if T_sub.size else 0.0)
        mu = 0.5*(muS + muT)
        sigma = np.sqrt(0.5*(varS + varT)) + 1e-6
        zS = ((S_sub - mu)/sigma).mean(axis=0) if S_sub.size else np.zeros(d, dtype=np.float32)
        zT = ((T_sub - mu)/sigma).mean(axis=0) if T_sub.size else np.zeros(d, dtype=np.float32)

        if base == "same_sign":
            idx = select_din_same_sign(zS, zT, tau=tau, k_ratio=k_ratio)
        elif base == "topk_strength":
            idx = select_din_topk_strength(zS, zT, k_ratio=k_ratio)
        elif base == "intersect":
            idx = select_din_intersect_topk(zS, zT, frac=0.1, k_ratio=k_ratio)
        elif base == "rank_agg":
            idx = select_din_rank_agg(zS, zT, k_ratio=k_ratio)
        elif base == "soft_sign":
            idx = select_din_soft_sign(zS, zT, tau=tau, margin=0.2, k_ratio=k_ratio)
        elif base == "maha":
            idx = select_din_pooled_mahalanobis(S_sub, T_sub, k_ratio=k_ratio)
        elif base == "fisher":
            idx = select_din_fisher(S_sub, T_sub, k_ratio=k_ratio)
        else:
            raise ValueError(f"Unknown base={base}")

        for i in idx:
            if 0 <= i < d:
                freq[i] += 1

    K = _calc_K(d, k_ratio)
    return _topk_indices(freq.astype(float), K)

# -----------------------------
# 9) Sign-consistency with Gaussian jitter (robustness)
# -----------------------------
def select_din_stability_sign(
    zS: np.ndarray,
    zT: np.ndarray,
    tau: float = 1.0,
    k_ratio: float = 0.05,
    noise_sigma: float = 0.1,
    trials: int = 50,
    seed: Optional[int] = 42,
) -> List[int]:
    """
    Add small Gaussian noise to zS/zT across trials; count how often a dim meets the
    same-sign + threshold condition; take Top-K by frequency. This selects dims whose
    sign-consistency is stable under perturbations.
    """
    rng = _rng(seed)
    zS = _ensure_1d(zS); zT = _ensure_1d(zT)
    d = zS.shape[0]; K = _calc_K(d, k_ratio)

    freq = np.zeros(d, dtype=np.int32)
    for _ in range(trials):
        zSn = zS + rng.normal(0, noise_sigma, size=d)
        zTn = zT + rng.normal(0, noise_sigma, size=d)
        mask = ((zSn > tau) & (zTn > tau)) | ((zSn < -tau) & (zTn < -tau))
        I = np.where(mask)[0]
        for i in I:
            freq[i] += 1

    return _topk_indices(freq.astype(float), K)

# -----------------------------
# Dispatcher
# -----------------------------
def dispatch_select_din(
    method: Literal[
        "same_sign","topk_strength","intersect","rank_agg","soft_sign",
        "maha","fisher","bootstrap","stability_sign"
    ],
    zS: Optional[np.ndarray] = None,
    zT: Optional[np.ndarray] = None,
    Hs: Optional[np.ndarray] = None,
    Ht: Optional[np.ndarray] = None,
    **kwargs,
) -> List[int]:
    """
    Unified entry point. Provide required arrays depending on method:
      - Methods using zS/zT only: same_sign, topk_strength, intersect, rank_agg, soft_sign, stability_sign
      - Methods using Hs/Ht only: maha, fisher
      - bootstrap can use either (through base methods) but expects Hs/Ht to resample pooled z
    """
    if method == "same_sign":
        return select_din_same_sign(zS, zT, **kwargs)
    if method == "topk_strength":
        return select_din_topk_strength(zS, zT, **kwargs)
    if method == "intersect":
        return select_din_intersect_topk(zS, zT, **kwargs)
    if method == "rank_agg":
        return select_din_rank_agg(zS, zT, **kwargs)
    if method == "soft_sign":
        return select_din_soft_sign(zS, zT, **kwargs)
    if method == "maha":
        if Hs is None or Ht is None:
            raise ValueError("method='maha' requires Hs and Ht")
        return select_din_pooled_mahalanobis(Hs, Ht, **kwargs)
    if method == "fisher":
        if Hs is None or Ht is None:
            raise ValueError("method='fisher' requires Hs and Ht")
        return select_din_fisher(Hs, Ht, **kwargs)
    if method == "bootstrap":
        if Hs is None or Ht is None:
            raise ValueError("method='bootstrap' requires Hs and Ht")
        return select_din_bootstrap_freq(Hs, Ht, **kwargs)
    if method == "stability_sign":
        return select_din_stability_sign(zS, zT, **kwargs)
    raise ValueError(f"Unknown method={method}")

# -----------------------------
# Quick self-test
# -----------------------------
if __name__ == "__main__":
    d = 16
    rng = np.random.default_rng(0)
    zS = rng.standard_normal(d)
    zT = zS + rng.normal(0, 0.2, size=d)  # partially aligned
    Hs = rng.standard_normal((64, d))
    Ht = Hs + rng.normal(0, 0.2, size=(64, d))

    print("same_sign:", dispatch_select_din("same_sign", zS=zS, zT=zT, tau=0.5, k_ratio=0.25))
    print("topk_strength:", dispatch_select_din("topk_strength", zS=zS, zT=zT, k_ratio=0.25))
    print("intersect:", dispatch_select_din("intersect", zS=zS, zT=zT, frac=0.3, k_ratio=0.25))
    print("rank_agg:", dispatch_select_din("rank_agg", zS=zS, zT=zT, k_ratio=0.25))
    print("soft_sign:", dispatch_select_din("soft_sign", zS=zS, zT=zT, tau=0.7, margin=0.3, k_ratio=0.25))
    print("maha:", dispatch_select_din("maha", Hs=Hs, Ht=Ht, k_ratio=0.25))
    print("fisher:", dispatch_select_din("fisher", Hs=Hs, Ht=Ht, k_ratio=0.25))
    print("bootstrap:", dispatch_select_din("bootstrap", Hs=Hs, Ht=Ht, base="same_sign", tau=0.5, k_ratio=0.25, iters=10))
    print("stability_sign:", dispatch_select_din("stability_sign", zS=zS, zT=zT, tau=0.6, k_ratio=0.25, noise_sigma=0.1, trials=20))
