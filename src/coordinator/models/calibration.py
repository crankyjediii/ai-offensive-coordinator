"""Post-hoc calibrators fitted on a temporally held-out calibration block."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Multiclass temperature scaling (Guo et al. 2017)."""
    def nll(log_t: float) -> float:
        p = softmax(logits / np.exp(log_t))
        return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())
    res = minimize_scalar(nll, bounds=(-2.0, 2.0), method="bounded")
    return float(np.exp(res.x))


def fit_platt(logit: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration p = sigmoid(a * logit + b)."""
    y = y.astype(float)

    def nll(ab: np.ndarray) -> float:
        p = np.clip(sigmoid(ab[0] * logit + ab[1]), 1e-12, 1 - 1e-12)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    res = minimize(nll, x0=np.array([1.0, 0.0]), method="L-BFGS-B")
    return float(res.x[0]), float(res.x[1])


def conformal_quantile(abs_residuals: np.ndarray, coverage: float) -> float:
    """Split-conformal residual quantile with the finite-sample correction."""
    n = len(abs_residuals)
    if n == 0:
        return float("nan")
    k = int(np.ceil((n + 1) * coverage))  # k-th smallest residual
    if k > n:
        return float("inf")
    return float(np.sort(abs_residuals)[k - 1])
