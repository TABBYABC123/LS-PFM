"""Small evaluation helpers used by the fixed-best reproduction script."""
from __future__ import annotations

import math
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from .gwnm_data import (
    adjacency_to_distance,
    build_correlation_adjacency,
    build_station_meta,
    load_adj_matrix,
    load_time_series,
    make_normalizer,
    make_point_representative_windows,
    make_windows,
    normalize_adjacency_from_distance,
    preprocess_values,
)


def subsample_windows(win, max_windows: int):
    if max_windows is None or int(max_windows) <= 0 or len(win.windows) <= int(max_windows):
        return win
    max_windows = int(max_windows)
    idx = np.linspace(0, len(win.windows) - 1, num=max_windows).round().astype(np.int64)
    win.windows = win.windows[idx]
    win.time_features = win.time_features[idx]
    win.starts = win.starts[idx]
    win.ends = win.ends[idx]
    if win.anchors is not None:
        win.anchors = win.anchors[idx]
    return win


def resolve_preprocess_args(args, ckpt_args: Dict) -> Tuple[str, int]:
    mode = str(getattr(args, "preprocess_mode", "auto") or "auto").lower().strip()
    if mode == "auto":
        mode = str(ckpt_args.get("preprocess_mode", "none"))
    rolling = int(getattr(args, "preprocess_rolling", 0) or 0)
    if rolling <= 0:
        rolling = int(ckpt_args.get("preprocess_rolling", 96))
    return mode, rolling


def load_target_windows(csv_path: str, adj_path: str, args, normalizer=None, fit_norm: bool = False, point: bool = False):
    series = load_time_series(csv_path, datetime_col=args.datetime_col)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    train_args = ckpt["args"]
    pp_mode, pp_rolling = resolve_preprocess_args(args, train_args)
    values_proc = preprocess_values(series.values, mode=pp_mode, rolling=pp_rolling)
    if normalizer is None:
        normalizer = make_normalizer(args.normalizer)
        fit_norm = True
    values = normalizer.fit_transform(values_proc) if fit_norm else normalizer.transform(values_proc)

    window = int(train_args["window"])
    if point:
        win = make_point_representative_windows(
            values,
            series.datetime,
            window=window,
            mode=args.test_window_mode,
            stride=args.test_stride,
        )
    else:
        win = make_windows(values, series.datetime, window=window, stride=int(train_args.get("stride", 1)))
        win = subsample_windows(win, args.max_support_windows)

    if adj_path and os.path.exists(adj_path):
        adj_raw = load_adj_matrix(adj_path, series.columns)
    else:
        print(f"[WARN] adjacency file not found for {csv_path}: {adj_path}. Using correlation adjacency fallback.")
        adj_raw = build_correlation_adjacency(values_proc)
    dist = adjacency_to_distance(adj_raw, mode=args.adj_mode)
    adj = normalize_adjacency_from_distance(dist)
    meta = build_station_meta(series.columns, dist)
    return series, normalizer, win, torch.from_numpy(dist).float(), torch.from_numpy(adj).float(), torch.from_numpy(meta).float()


def reconstruction_anchor_scores(
    x: torch.Tensor,
    x_rec: torch.Tensor,
    mode: str = "end",
    steps: int = 1,
    topk_fraction: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.shape != x_rec.shape or x.ndim != 3:
        raise ValueError(f"x and x_rec must have equal [B,D,T] shapes, got {tuple(x.shape)} and {tuple(x_rec.shape)}")
    t = x.shape[-1]
    steps = min(max(1, int(steps)), t)
    mode = str(mode).lower().strip()
    if mode == "lead":
        start = 0
    elif mode == "center":
        start = max(0, min(t - steps, t // 2 - steps // 2))
    elif mode == "end":
        start = t - steps
    else:
        raise ValueError("mode must be lead, center, or end for anchor reconstruction scores.")
    err2 = (x[..., start : start + steps] - x_rec[..., start : start + steps]).pow(2)
    local = torch.sqrt(err2.mean(dim=(1, 2)) + 1e-8)
    flat = err2.flatten(1)
    frac = min(max(float(topk_fraction), 1.0 / max(flat.shape[1], 1)), 1.0)
    k = min(flat.shape[1], max(1, int(math.ceil(flat.shape[1] * frac))))
    topk = torch.sqrt(torch.topk(flat, k=k, dim=1, largest=True).values.mean(dim=1) + 1e-8)
    return local, topk


def rolling_median_score(x: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1:
        return x
    s = pd.Series(np.asarray(x, dtype=np.float64))
    return s.rolling(window=window, center=True, min_periods=1).median().to_numpy()


def suppress_short_runs(pred: np.ndarray, min_len: int) -> np.ndarray:
    min_len = int(min_len)
    out = np.asarray(pred, dtype=np.int64).copy()
    if min_len <= 1 or out.size == 0:
        return out
    i = 0
    while i < len(out):
        if out[i] == 0:
            i += 1
            continue
        j = i
        while j < len(out) and out[j] == 1:
            j += 1
        if j - i < min_len:
            out[i:j] = 0
        i = j
    return out


def event_peak_candidates(score: np.ndarray, pred: np.ndarray, mode: str = "none", radius: int = 10) -> np.ndarray:
    mode = str(mode or "none").lower().strip()
    pred = np.asarray(pred, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if mode == "none" or pred.size == 0:
        return pred.copy()
    out = np.zeros_like(pred)
    if mode == "run":
        i = 0
        while i < len(pred):
            if pred[i] == 0:
                i += 1
                continue
            j = i
            while j < len(pred) and pred[j] == 1:
                j += 1
            out[i + int(np.argmax(score[i:j]))] = 1
            i = j
        return out
    if mode == "nms":
        radius = max(0, int(radius))
        candidates = np.flatnonzero(pred)
        order = candidates[np.argsort(score[candidates])[::-1]]
        blocked = np.zeros(len(pred), dtype=bool)
        for idx in order:
            if blocked[idx]:
                continue
            out[idx] = 1
            blocked[max(0, idx - radius) : min(len(pred), idx + radius + 1)] = True
        return out
    raise ValueError("event_peak_mode must be none, run, or nms.")
