"""
Data utilities for Residual-GWNM.

Expected CSV format:
- one datetime column, default: datetime
- one column per base station / cell / node

The code keeps the GW-NM advantage that node count D can differ across clusters.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

EPS = 1e-8


@dataclass
class LoadedSeries:
    datetime: np.ndarray
    values: np.ndarray          # [N, D]
    columns: List[str]
    frame: pd.DataFrame


@dataclass
class WindowedSeries:
    windows: np.ndarray         # [M, D, T]
    time_features: np.ndarray   # [M, T, F]
    starts: np.ndarray          # [M]
    ends: np.ndarray            # [M]
    anchors: Optional[np.ndarray] = None


class StandardNormalizer:
    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        self.mean = mean
        self.std = std

    def fit(self, x: np.ndarray) -> "StandardNormalizer":
        x = np.asarray(x, dtype=np.float64)
        self.mean = np.nanmean(x, axis=0)
        self.std = np.nanstd(x, axis=0)
        self.std = np.where(self.std < EPS, 1.0, self.std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardNormalizer must be fitted before transform().")
        return ((np.asarray(x, dtype=np.float64) - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def to_dict(self) -> Dict[str, list]:
        return {"type": "standard", "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: Dict[str, list]) -> "StandardNormalizer":
        return cls(np.asarray(payload["mean"], dtype=np.float64), np.asarray(payload["std"], dtype=np.float64))


class RobustNormalizer:
    def __init__(self, median: Optional[np.ndarray] = None, iqr: Optional[np.ndarray] = None):
        self.median = median
        self.iqr = iqr

    def fit(self, x: np.ndarray) -> "RobustNormalizer":
        x = np.asarray(x, dtype=np.float64)
        self.median = np.nanmedian(x, axis=0)
        q75 = np.nanpercentile(x, 75, axis=0)
        q25 = np.nanpercentile(x, 25, axis=0)
        self.iqr = q75 - q25
        self.iqr = np.where(self.iqr < EPS, 1.0, self.iqr)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.median is None or self.iqr is None:
            raise RuntimeError("RobustNormalizer must be fitted before transform().")
        return ((np.asarray(x, dtype=np.float64) - self.median) / self.iqr).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def to_dict(self) -> Dict[str, list]:
        return {"type": "robust", "median": self.median.tolist(), "iqr": self.iqr.tolist()}

    @classmethod
    def from_dict(cls, payload: Dict[str, list]) -> "RobustNormalizer":
        return cls(np.asarray(payload["median"], dtype=np.float64), np.asarray(payload["iqr"], dtype=np.float64))


def make_normalizer(kind: str):
    kind = kind.lower().strip()
    if kind == "standard":
        return StandardNormalizer()
    if kind == "robust":
        return RobustNormalizer()
    raise ValueError(f"Unknown normalizer: {kind}")


def normalizer_from_dict(payload: Dict[str, list]):
    if payload["type"] == "standard":
        return StandardNormalizer.from_dict(payload)
    if payload["type"] == "robust":
        return RobustNormalizer.from_dict(payload)
    raise ValueError(f"Unknown normalizer payload type: {payload.get('type')}")


def load_time_series(csv_path: str, datetime_col: str = "datetime") -> LoadedSeries:
    df = pd.read_csv(csv_path)
    if datetime_col not in df.columns:
        raise ValueError(f"{csv_path} does not contain `{datetime_col}`.")
    df = df.copy()
    df[datetime_col] = df[datetime_col].astype(str)
    df = df.sort_values(datetime_col).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c != datetime_col]
    if not feature_cols:
        raise ValueError(f"{csv_path} contains no feature columns.")
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].interpolate(method="linear", limit_direction="both")
    df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0.0)
    return LoadedSeries(
        datetime=df[datetime_col].to_numpy().astype(str),
        values=df[feature_cols].to_numpy(dtype=np.float32),
        columns=list(feature_cols),
        frame=df,
    )



def preprocess_values(values: np.ndarray, mode: str = "none", rolling: int = 96) -> np.ndarray:
    """Causal preprocessing for RRC-style network time series.

    This function is intentionally placed before normalization/windowing.
    Supported modes:
    - none/raw: keep the filled raw values.
    - log1p: variance-stabilizing transform for non-negative traffic counters.
    - diff: first-order difference on raw values.
    - logdiff: first-order difference on log1p values.
    - log_residual/residual: log1p value minus causal rolling-median baseline.
    - residual_diff: first-order difference of log_residual.

    rolling is the baseline window length. For 15-minute sampling, 96 is about
    one day. The baseline uses shift(1), so the current point is not used to
    estimate its own baseline.
    """
    mode = str(mode or "none").lower().strip()
    x = np.asarray(values, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if mode in {"none", "raw"}:
        return x.astype(np.float32)

    if mode in {"log", "log1p", "log_residual", "residual", "logdiff", "residual_diff"}:
        base = np.log1p(np.clip(x, 0.0, None))
    else:
        base = x.copy()

    if mode in {"log", "log1p"}:
        return base.astype(np.float32)

    if mode == "diff":
        out = np.zeros_like(base)
        out[1:] = x[1:] - x[:-1]
        return out.astype(np.float32)

    if mode == "logdiff":
        out = np.zeros_like(base)
        out[1:] = base[1:] - base[:-1]
        return out.astype(np.float32)

    if mode in {"log_residual", "residual", "residual_diff"}:
        w = max(3, int(rolling))
        min_periods = max(3, min(w, w // 8 if w >= 8 else 3))
        trend = (
            pd.DataFrame(base)
            .rolling(window=w, min_periods=min_periods)
            .median()
            .shift(1)
            .ffill()
            .bfill()
            .to_numpy(dtype=np.float64)
        )
        residual = base - trend
        if mode == "residual_diff":
            out = np.zeros_like(residual)
            out[1:] = residual[1:] - residual[:-1]
            return out.astype(np.float32)
        return residual.astype(np.float32)

    raise ValueError(
        f"Unknown preprocess mode: {mode}. Choose from none, log1p, diff, logdiff, log_residual, residual_diff."
    )


def _coerce_adj_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Pure square matrix.
    if df.shape[0] == df.shape[1]:
        return df.copy()
    # Square matrix with index/station-name column.
    if df.shape[1] == df.shape[0] + 1:
        out = df.iloc[:, 1:].copy()
        out.index = df.iloc[:, 0].astype(str).tolist()
        return out
    raise ValueError(f"Cannot coerce adjacency shape {df.shape} to a square matrix.")


def load_adj_matrix(adj_path: str, columns: List[str]) -> np.ndarray:
    df = pd.read_csv(adj_path)
    df = _coerce_adj_dataframe(df)

    # If adjacency columns carry station names, try to align to time-series columns.
    str_cols = [str(c) for c in df.columns]
    if set(columns).issubset(set(str_cols)):
        df.columns = str_cols
        if set(columns).issubset(set([str(x) for x in df.index])):
            df.index = [str(x) for x in df.index]
            df = df.loc[columns, columns]
        else:
            df = df[columns]
            if df.shape[0] != len(columns):
                raise ValueError("Adjacency row count does not match selected columns.")
    else:
        if df.shape[0] != len(columns) or df.shape[1] != len(columns):
            raise ValueError(
                f"Adjacency shape {df.shape} does not match number of stations {len(columns)}."
            )

    adj = df.to_numpy(dtype=np.float32)
    adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
    return adj.astype(np.float32)



def build_correlation_adjacency(values: np.ndarray, min_abs_corr: float = 0.05) -> np.ndarray:
    """Build a similarity adjacency from time-series values when an adj CSV is missing.

    This is a safe fallback for target clusters whose physical adjacency file is
    unavailable. It uses absolute Pearson correlation across nodes, clips invalid
    values, thresholds tiny correlations, and keeps self-loops.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"values must be [T,D], got {x.shape}")
    n = x.shape[1]
    if n <= 0:
        raise ValueError("values has zero feature columns.")
    if n == 1:
        return np.ones((1, 1), dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    std = x.std(axis=0)
    valid = std > EPS
    corr = np.eye(n, dtype=np.float64)
    if valid.sum() >= 2:
        c = np.corrcoef(x[:, valid], rowvar=False)
        c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        c = np.abs(c)
        idx = np.where(valid)[0]
        for ii, i in enumerate(idx):
            for jj, j in enumerate(idx):
                corr[i, j] = c[ii, jj]
    corr = np.clip(corr, 0.0, 1.0)
    corr[corr < float(min_abs_corr)] = 0.0
    np.fill_diagonal(corr, 1.0)
    return corr.astype(np.float32)

def adjacency_to_distance(adj: np.ndarray, mode: str = "auto") -> np.ndarray:
    """Convert an adjacency/similarity matrix to a normalized distance matrix."""
    a = np.asarray(adj, dtype=np.float32).copy()
    n = a.shape[0]
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("adj must be a square matrix.")
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    if mode == "auto":
        finite = a[np.isfinite(a)]
        finite = finite[finite != 0]
        if len(finite) == 0:
            mode = "similarity"
        else:
            # Values in [0,1] normally indicate connectivity/similarity.
            mode = "similarity" if finite.min() >= 0 and finite.max() <= 1.5 else "distance"

    if mode == "similarity":
        sim = a.copy()
        sim = np.maximum(sim, 0.0)
        if sim.max() > 0:
            sim = sim / sim.max()
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0.0)
    elif mode == "distance":
        dist = np.maximum(a, 0.0)
        np.fill_diagonal(dist, 0.0)
    else:
        raise ValueError("adj_mode must be auto, similarity, or distance.")

    if n > 1:
        mask = ~np.eye(n, dtype=bool)
        mean = np.mean(dist[mask])
        if mean > EPS:
            dist = dist / mean
    return dist.astype(np.float32)


def normalize_adjacency_from_distance(distance: np.ndarray, self_loop: float = 1.0) -> np.ndarray:
    """Build a row-normalized physical graph from a distance matrix."""
    d = np.asarray(distance, dtype=np.float32)
    sim = np.exp(-d)
    np.fill_diagonal(sim, self_loop)
    denom = sim.sum(axis=1, keepdims=True)
    denom = np.where(denom < EPS, 1.0, denom)
    return (sim / denom).astype(np.float32)


def build_station_meta(columns: List[str], distance: np.ndarray) -> np.ndarray:
    """Variable-node station metadata independent of fixed node count."""
    d = np.asarray(distance, dtype=np.float32)
    n = len(columns)
    idx = np.linspace(0.0, 1.0, n, dtype=np.float32) if n > 1 else np.zeros(1, dtype=np.float32)
    deg = np.exp(-d).sum(axis=1)
    deg = deg / max(float(deg.max()), EPS)
    if n > 1:
        mask = ~np.eye(n, dtype=bool)
        avg_dist = d.sum(axis=1) / max(n - 1, 1)
        min_dist = np.where(np.eye(n, dtype=bool), np.inf, d).min(axis=1)
        max_dist = d.max(axis=1)
        avg_dist = avg_dist / max(float(np.mean(d[mask])), EPS)
        min_dist = np.nan_to_num(min_dist, posinf=0.0)
        min_dist = min_dist / max(float(np.max(min_dist)), EPS)
        max_dist = max_dist / max(float(np.max(max_dist)), EPS)
    else:
        avg_dist = min_dist = max_dist = np.zeros(1, dtype=np.float32)
    meta = np.stack([idx, deg.astype(np.float32), avg_dist.astype(np.float32), min_dist.astype(np.float32), max_dist.astype(np.float32)], axis=-1)
    return meta.astype(np.float32)


def build_temporal_features(datetime_values: np.ndarray) -> np.ndarray:
    """Return cyclic time features [N, 6].

    Features: hour sin/cos, minute-of-day sin/cos, day-of-week sin/cos.
    If parsing fails, returns zeros.
    """
    dt = pd.to_datetime(pd.Series(datetime_values.astype(str)), errors="coerce")
    if dt.isna().all():
        return np.zeros((len(datetime_values), 6), dtype=np.float32)
    dt = dt.ffill().bfill()
    hour = dt.dt.hour.to_numpy(dtype=np.float32)
    minute = dt.dt.minute.to_numpy(dtype=np.float32)
    tod = hour * 60.0 + minute
    dow = dt.dt.dayofweek.to_numpy(dtype=np.float32)
    feats = np.stack([
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * tod / 1440.0),
        np.cos(2 * np.pi * tod / 1440.0),
        np.sin(2 * np.pi * dow / 7.0),
        np.cos(2 * np.pi * dow / 7.0),
    ], axis=-1)
    return feats.astype(np.float32)


def make_windows(values: np.ndarray, datetimes: np.ndarray, window: int, stride: int = 1) -> WindowedSeries:
    values = np.asarray(values, dtype=np.float32)
    time_feats = build_temporal_features(datetimes)
    n, d = values.shape
    if n < window:
        raise ValueError(f"Series length {n} is shorter than window {window}.")
    starts = np.arange(0, n - window + 1, stride, dtype=np.int64)
    ends = starts + window - 1
    xs = np.stack([values[s:s + window].T for s in starts], axis=0).astype(np.float32)
    ts = np.stack([time_feats[s:s + window] for s in starts], axis=0).astype(np.float32)
    return WindowedSeries(xs, ts, starts, ends, anchors=None)


def make_point_representative_windows(
    values: np.ndarray,
    datetimes: np.ndarray,
    window: int,
    mode: str = "lead",
    stride: int = 1,
) -> WindowedSeries:
    win = make_windows(values, datetimes, window=window, stride=stride)
    if mode == "lead":
        anchors = win.starts.copy()
    elif mode == "center":
        anchors = win.starts + window // 2
    elif mode == "end":
        anchors = win.ends.copy()
    else:
        raise ValueError("mode must be lead, center, or end for point representative windows.")
    win.anchors = anchors.astype(np.int64)
    return win


def assign_window_scores_to_points(
    length: int,
    anchors: np.ndarray,
    scores: Dict[str, np.ndarray],
    fill: str = "interpolate",
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    anchors = np.asarray(anchors, dtype=np.int64)
    valid = (anchors >= 0) & (anchors < length)
    anchors = anchors[valid]
    for name, arr in scores.items():
        arr = np.asarray(arr, dtype=np.float64)[valid]
        point = np.full(length, np.nan, dtype=np.float64)
        point[anchors] = arr
        if fill == "none" or fill == "nan":
            pass
        elif fill == "edge":
            s = pd.Series(point).ffill().bfill()
            point = s.to_numpy(dtype=np.float64)
        elif fill == "interpolate":
            s = pd.Series(point).interpolate(method="linear", limit_direction="both").ffill().bfill()
            point = s.to_numpy(dtype=np.float64)
        else:
            raise ValueError("fill must be none, nan, edge, or interpolate.")
        out[name] = point
    return out


def aggregate_window_scores_to_points(
    length: int,
    starts: np.ndarray,
    ends: np.ndarray,
    scores: Dict[str, np.ndarray],
    aggregation: str = "mean",
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name, score in scores.items():
        buckets = [[] for _ in range(length)]
        for s, e, v in zip(starts, ends, score):
            for i in range(max(0, int(s)), min(length, int(e) + 1)):
                buckets[i].append(float(v))
        point = np.full(length, np.nan, dtype=np.float64)
        for i, vals in enumerate(buckets):
            if not vals:
                continue
            if aggregation == "mean":
                point[i] = float(np.mean(vals))
            elif aggregation == "max":
                point[i] = float(np.max(vals))
            elif aggregation == "end":
                point[i] = float(vals[-1])
            else:
                raise ValueError("aggregation must be mean, max, or end.")
        point = pd.Series(point).interpolate(method="linear", limit_direction="both").ffill().bfill().to_numpy(dtype=np.float64)
        out[name] = point
    return out


def load_ground_truth(gt_path: str, datetime_col: str = "datetime", label_col: str = "label") -> pd.DataFrame:
    gt = pd.read_csv(gt_path)
    if datetime_col not in gt.columns or label_col not in gt.columns:
        raise ValueError(f"{gt_path} must contain `{datetime_col}` and `{label_col}`.")
    gt = gt[[datetime_col, label_col]].copy()
    gt[datetime_col] = gt[datetime_col].astype(str)
    gt[label_col] = pd.to_numeric(gt[label_col], errors="coerce").fillna(0).astype(int)
    return gt.sort_values(datetime_col).reset_index(drop=True)


def align_labels_to_series(series: LoadedSeries, gt_path: str, datetime_col: str = "datetime", label_col: str = "label") -> np.ndarray:
    gt = load_ground_truth(gt_path, datetime_col=datetime_col, label_col=label_col)
    left = pd.DataFrame({datetime_col: series.datetime.astype(str)})
    merged = left.merge(gt, on=datetime_col, how="left")
    return merged[label_col].fillna(0).astype(int).to_numpy()


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
