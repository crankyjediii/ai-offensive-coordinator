"""Probabilistic and regression metrics, with game-clustered paired bootstrap intervals.

Brier normalization: binary = mean (p - y)^2; multiclass = mean over rows of the sum over classes.
"""

from __future__ import annotations

import numpy as np


def log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-row log loss. p: (N, K) probabilities, y: (N,) int labels."""
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    if p.shape[1] == 2:
        return (p[:, 1] - onehot[:, 1]) ** 2
    return ((p - onehot) ** 2).sum(1)


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    """Fixed equal-width bins for a binary probability; denominators always reported."""
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        n = int(m.sum())
        out.append({"bin_lo": round(float(lo), 2), "bin_hi": round(float(hi), 2), "n": n,
                    "mean_pred": float(p[m].mean()) if n else None,
                    "observed_rate": float(y[m].mean()) if n else None})
    return out


def calibration_in_the_large_and_slope(p: np.ndarray, y: np.ndarray) -> dict:
    from coordinator.models.calibration import fit_platt
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    a, b = fit_platt(logit, y)
    return {"mean_pred": float(p.mean()), "observed_rate": float(y.mean()), "slope": a, "intercept": b}


def cluster_bootstrap_mean(values: np.ndarray, clusters: np.ndarray, reps: int = 1000, seed: int = 0) -> dict:
    """Mean with a game-clustered percentile bootstrap interval (95%)."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(reps, len(uniq)))
    boot = sums[idx].sum(1) / counts[idx].sum(1)
    return {"mean": float(values.mean()), "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5)), "n_rows": int(len(values)), "n_clusters": int(len(uniq))}


def paired_difference(loss_a: np.ndarray, loss_b: np.ndarray, clusters: np.ndarray, reps: int = 1000) -> dict:
    """Clustered interval for mean(loss_a - loss_b) on identical rows (negative favors a)."""
    d = cluster_bootstrap_mean(loss_a - loss_b, clusters, reps)
    base = float(loss_b.mean())
    d["relative_change"] = d["mean"] / base if base else None
    return d
