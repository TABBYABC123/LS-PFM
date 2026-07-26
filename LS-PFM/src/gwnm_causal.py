"""Auxiliary losses, pseudo anomaly generation, discriminators, and metrics."""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


class MeasureDiscriminator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DynamicMeasureDiscriminator(nn.Module):
    """Variable-node dynamic discriminator.

    Unlike the old discriminator that flattens N*D, this pools nodes first and
    therefore supports different node counts across clusters.
    """

    def __init__(self, hidden_dim: int, rnn_hidden: int = 64, dropout: float = 0.05):
        super().__init__()
        self.gru = nn.GRU(hidden_dim, rnn_hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_hidden, 1)

    def forward(self, h_dyn: torch.Tensor) -> torch.Tensor:
        # h_dyn: [B,T,D,C]
        h_pool = h_dyn.mean(dim=2)  # [B,T,C]
        out, _ = self.gru(h_pool)
        return self.fc(self.dropout(out[:, -1])).squeeze(-1)


def measure_feature(mu: torch.Tensor, sigma: torch.Tensor, token_z: Optional[torch.Tensor] = None, feature: str = "mu") -> torch.Tensor:
    feature = feature.lower().strip()
    if feature == "mu":
        return mu
    if feature in {"musigma", "mu_sigma"}:
        return torch.cat([mu, sigma], dim=-1)
    if feature == "token_mean":
        if token_z is None:
            raise ValueError("token_mean feature requires token_z.")
        return token_z.mean(dim=1)
    raise ValueError(f"Unknown feature: {feature}")


def _pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = (x * x).sum(dim=1, keepdim=True)
    y_norm = (y * y).sum(dim=1, keepdim=True).T
    return (x_norm + y_norm - 2.0 * x @ y.T).clamp_min(0.0)


def gaussian_mmd(x: torch.Tensor, y: torch.Tensor, bandwidth: Optional[float] = None, unbiased: bool = False) -> torch.Tensor:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"MMD inputs must be 2-D, got {tuple(x.shape)} and {tuple(y.shape)}")
    if x.shape[0] == 0 or y.shape[0] == 0:
        return x.new_tensor(0.0)
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    xx = _pairwise_sq_dist(x, x)
    yy = _pairwise_sq_dist(y, y)
    xy = _pairwise_sq_dist(x, y)
    if bandwidth is None or bandwidth <= 0:
        with torch.no_grad():
            bw = torch.median(xy.detach().flatten()).clamp_min(1e-3)
    else:
        bw = x.new_tensor(float(bandwidth)).clamp_min(1e-6)
    gamma = 1.0 / (2.0 * bw)
    k_xx = torch.exp(-gamma * xx)
    k_yy = torch.exp(-gamma * yy)
    k_xy = torch.exp(-gamma * xy)
    if unbiased and x.shape[0] > 1 and y.shape[0] > 1:
        n, m = x.shape[0], y.shape[0]
        k_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (n * (n - 1))
        k_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (m * (m - 1))
        k_xy = k_xy.mean()
        return (k_xx + k_yy - 2.0 * k_xy).clamp_min(0.0)
    return (k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()).clamp_min(0.0)


def mmd_from_outputs(out_a: Dict[str, torch.Tensor], out_b: Dict[str, torch.Tensor], feature: str = "mu", bandwidth: float = 0.0) -> torch.Tensor:
    fa = measure_feature(out_a["mu"], out_a["sigma"], out_a.get("token_z"), feature=feature)
    fb = measure_feature(out_b["mu"], out_b["sigma"], out_b.get("token_z"), feature=feature)
    return gaussian_mmd(fa, fb, bandwidth=bandwidth)


def pseudo_anomaly_windows(x: torch.Tensor, max_seg: int = 8, sigma: float = 0.5, mode: str = "mixed") -> torch.Tensor:
    """Generate RRC-style pseudo anomalies for [B,D,T] windows.

    The old version used only temporal permutation + jitter. That can make
    reconstruction error large while prototype/flow evidence becomes smaller,
    which then makes pseudo calibration drop the MPFM components. This mixed
    generator creates more realistic time-series disturbances:
    spike, drop, level shift, flatline, neighbor-inconsistency, and permutation.
    """
    if x.ndim != 3:
        raise ValueError(f"x must be [B,D,T], got {tuple(x.shape)}")
    b, d, t = x.shape
    out = x.clone()
    mode = str(mode or "mixed").lower().strip()
    if t <= 2:
        return out + torch.randn_like(out) * float(sigma)

    max_seg = max(2, min(int(max_seg), t))
    amp_base = max(float(sigma), 1e-3)

    def _permute_one(vec: torch.Tensor) -> torch.Tensor:
        if t <= 2:
            return vec
        nseg = int(torch.randint(2, max_seg + 1, (1,), device=x.device).item())
        if nseg >= t:
            return vec[torch.randperm(t, device=x.device)]
        split = torch.randperm(t - 1, device=x.device)[: nseg - 1] + 1
        split = torch.sort(split).values.detach().cpu().numpy().tolist()
        parts = torch.tensor_split(torch.arange(t, device=x.device), split)
        order = torch.randperm(len(parts), device=x.device).detach().cpu().numpy().tolist()
        idx = torch.cat([parts[j] for j in order], dim=0)
        return vec[idx]

    if mode in {"permute", "permute_jitter", "old"}:
        for bi in range(b):
            for di in range(d):
                out[bi, di] = _permute_one(out[bi, di])
        return out + torch.randn_like(out) * amp_base

    for bi in range(b):
        seg_len = int(torch.randint(2, max_seg + 1, (1,), device=x.device).item())
        start_hi = max(1, t - seg_len + 1)
        st = int(torch.randint(0, start_hi, (1,), device=x.device).item())
        ed = min(t, st + seg_len)

        # Usually perturb only a subset of nodes. This better matches local
        # base-station anomalies and creates graph-neighbor inconsistency.
        if d <= 1:
            nodes = torch.arange(d, device=x.device)
        else:
            node_count = int(torch.randint(1, max(2, min(d, max(2, d // 2))) + 1, (1,), device=x.device).item())
            nodes = torch.randperm(d, device=x.device)[:node_count]

        op = int(torch.randint(0, 6, (1,), device=x.device).item())
        sign = 1.0 if int(torch.randint(0, 2, (1,), device=x.device).item()) == 1 else -1.0
        amp = amp_base * (1.0 + 3.0 * torch.rand((), device=x.device, dtype=x.dtype))

        if op == 0:  # spike / burst
            out[bi, nodes, st:ed] = out[bi, nodes, st:ed] + sign * amp
        elif op == 1:  # drop / degradation
            factor = 0.05 + 0.45 * torch.rand((), device=x.device, dtype=x.dtype)
            out[bi, nodes, st:ed] = out[bi, nodes, st:ed] * factor
        elif op == 2:  # level shift
            out[bi, nodes, st:] = out[bi, nodes, st:] + sign * amp
        elif op == 3:  # flatline / frozen counter
            src = max(0, st - 1)
            out[bi, nodes, st:ed] = out[bi, nodes, src:src + 1]
        elif op == 4:  # temporal order anomaly
            for di in nodes.tolist():
                out[bi, di] = _permute_one(out[bi, di])
        else:  # neighbor inconsistency: perturb one node more strongly
            one = nodes[:1]
            out[bi, one, st:ed] = out[bi, one, st:ed] + sign * (amp * 1.5)

    # small jitter keeps pseudo samples from being trivially discrete.
    out = out + torch.randn_like(out) * (0.1 * amp_base)
    return out


def pseudo_margin_loss(score_normal: torch.Tensor, score_pseudo: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    return F.relu(float(margin) + score_normal - score_pseudo).mean()


def fit_robust_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"median": 0.0, "scale": 1.0}
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = 1.4826 * mad if mad > EPS else float(np.std(x) + EPS)
    return {"median": med, "scale": max(float(scale), EPS)}


def apply_robust_stats(x: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - float(stats["median"])) / max(float(stats["scale"]), EPS)


def pot_threshold(init_score: np.ndarray, q: float = 1e-3, level: float = 0.98, fallback_quantile: float = 0.995) -> float:
    """Dependency-light POT-style threshold.

    Fits a GPD by method of moments on calibration exceedances. Falls back to a
    high quantile when the fit is numerically unstable.
    """
    x = np.asarray(init_score, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return float(np.quantile(x, fallback_quantile)) if len(x) else 0.0
    u = float(np.quantile(x, level))
    excess = x[x > u] - u
    nt = len(excess)
    n = len(x)
    if nt < 5:
        return float(np.quantile(x, fallback_quantile))
    mean = float(np.mean(excess))
    var = float(np.var(excess))
    if mean <= EPS or var <= EPS:
        return float(np.quantile(x, fallback_quantile))
    # Method-of-moments GPD parameters.
    xi = 0.5 * (1.0 - (mean * mean) / var)
    beta = 0.5 * mean * (1.0 + (mean * mean) / var)
    if not np.isfinite(xi) or not np.isfinite(beta) or beta <= EPS:
        return float(np.quantile(x, fallback_quantile))
    p = max(float(q), 1e-8)
    ratio = max((n * p) / max(nt, 1), 1e-8)
    try:
        if abs(xi) < 1e-6:
            z = u + beta * np.log(1.0 / ratio)
        else:
            z = u + (beta / xi) * (ratio ** (-xi) - 1.0)
    except Exception:
        z = np.nan
    if not np.isfinite(z):
        z = float(np.quantile(x, fallback_quantile))
    return float(max(z, u))


def adjust_predicts(score: np.ndarray, label: np.ndarray, threshold: Optional[float] = None, pred: Optional[np.ndarray] = None, calc_latency: bool = False):
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    if pred is None:
        predict = score > float(threshold)
    else:
        predict = np.asarray(pred).astype(bool).copy()
    actual = label > 0
    anomaly_state = False
    latency = 0
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            j = i
            while j >= 0 and actual[j]:
                if not predict[j]:
                    predict[j] = True
                    latency += 1
                j -= 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict.astype(int), latency / (anomaly_count + 1e-8)
    return predict.astype(int)


def confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray, score: Optional[np.ndarray] = None) -> Dict[str, float]:
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    out = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "false_alarm_rate": float(fp / (fp + tn + EPS)),
        "accuracy": float((tp + tn) / max(len(y_true), 1)),
    }
    if score is not None and len(np.unique(y_true)) > 1:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, score))
        except Exception:
            out["roc_auc"] = 0.0
        try:
            out["average_precision"] = float(average_precision_score(y_true, score))
        except Exception:
            out["average_precision"] = 0.0
    else:
        out["roc_auc"] = 0.0
        out["average_precision"] = 0.0
    return out
