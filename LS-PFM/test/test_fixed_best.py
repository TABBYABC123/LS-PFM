#!/usr/bin/env python3
"""Reproduce fixed-best Cluster1 anomaly detection from the saved Cluster3 model.

This script performs no parameter search. It loads the packaged best.pt,
generates score components, applies the selected fixed config, and writes the
adjusted anomaly-detection result.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_utils import (  # noqa: E402
    event_peak_candidates,
    load_target_windows,
    reconstruction_anchor_scores,
    rolling_median_score,
    suppress_short_runs,
)
from src.gwnm_data import align_labels_to_series, assign_window_scores_to_points, load_adj_matrix, save_json  # noqa: E402
from src.gwnm_model import reconstruction_score  # noqa: E402
from src.physical_topology_mpfm_gwnm import (  # noqa: E402
    PhysicalTopologyMPFMResidualGWNM,
    adapt_private_on_support,
    init_private_from_support,
)
from src.physical_topology_utils import physical_topology_from_adj  # noqa: E402
from src.topology_mpfm_gwnm import topology_roughness_score  # noqa: E402


PRESETS: Dict[str, Dict[str, object]] = {
    "shielding": {
        "description": "Cluster1 shielding/original test set",
        "test_csv": ROOT / "data" / "cluster1" / "RRC_ConnMean_cluster1_test.csv",
        "config_file": ROOT / "configs" / "shielding_fixed_best_config.json",
        "config_name": "original_best_adjusted",
        "prefix": "cluster1_original",
    },
    "decomming": {
        "description": "Cluster1 decomming/decommission test set",
        "test_csv": ROOT / "data" / "cluster1" / "RRC_ConnMean_cluster1_test_6_decommssion.csv",
        "config_file": ROOT / "configs" / "decomming_fixed_best_config.json",
        "config_name": "decommission_best_adjusted",
        "prefix": "cluster1_decommission",
    },
}

PRESET_ALIASES = {
    "original": "shielding",
    "decommission": "decomming",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run fixed-best anomaly detection.")
    p.add_argument("--preset", choices=[*PRESETS.keys(), *PRESET_ALIASES.keys(), "all"], default="decomming")
    p.add_argument("--ckpt_path", default=str(ROOT / "checkpoints" / "best.pt"))

    p.add_argument("--out_dir", default=str(ROOT / "outputs" / "test_fixed_best"))
    p.add_argument("--target_train_csv", default=str(ROOT / "data" / "cluster1" / "RRC_ConnMean_cluster1_train.csv"))
    p.add_argument("--target_adj_csv", default=str(ROOT / "data" / "cluster1" / "cluster1_adj.csv"))
    p.add_argument("--ground_truth_csv", default=str(ROOT / "data" / "cluster1" / "cluster1_ground_truth.csv"))
    p.add_argument("--base_score_csv", default=None, help="Optional component-score CSV; skips model inference when set.")


    p.add_argument("--datetime_col", default="datetime")
    p.add_argument("--label_col", default="label")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--normalizer", choices=["standard", "robust"], default="robust")
    p.add_argument("--adj_mode", choices=["auto", "similarity", "distance"], default="auto")
    p.add_argument(
        "--preprocess_mode",
        choices=["auto", "none", "log1p", "diff", "logdiff", "log_residual", "residual_diff"],
        default="none",
    )
    p.add_argument("--preprocess_rolling", type=int, default=0)
    p.add_argument("--test_window_mode", choices=["lead", "center", "end"], default="end")
    p.add_argument("--test_stride", type=int, default=1)
    p.add_argument("--fill_unavailable", choices=["none", "nan", "interpolate", "edge"], default="interpolate")
    p.add_argument("--max_support_windows", type=int, default=0)
    p.add_argument("--max_test_windows", type=int, default=0)

    p.add_argument("--adapt_private_steps", type=int, default=20)
    p.add_argument("--adapt_private_lr", type=float, default=2e-3)
    p.add_argument("--adapt_lambda_flow", type=float, default=0.05)
    p.add_argument("--adapt_lambda_ortho", type=float, default=0.01)
    p.add_argument("--adapt_lambda_smooth", type=float, default=0.02)
    p.add_argument("--adapt_lambda_anchor", type=float, default=0.2)
    p.add_argument("--kmeans_iters", type=int, default=25)

    p.add_argument("--flow_t_star", type=float, default=0.5)
    p.add_argument("--flow_mc", type=int, default=1)
    p.add_argument("--flow_deterministic", type=int, default=1)
    p.add_argument("--rec_local_steps", type=int, default=1)
    p.add_argument("--rec_topk_fraction", type=float, default=0.2)

    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--torch_num_threads", type=int, default=1)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def require_file(path: str | Path, description: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{description} not found: {path}")



def canonical_preset(name: str) -> str:
    return PRESET_ALIASES.get(str(name), str(name))


def load_preset_config(preset_name: str, args: argparse.Namespace) -> Tuple[str, Dict, str]:
    preset = PRESETS[preset_name]
    config_file = Path(preset["config_file"])
    require_file(config_file, f"{preset_name} fixed config")
    with open(config_file, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if "config_name" not in payload or "config" not in payload:
        raise ValueError(f"{config_file} must contain config_name and config.")
    return str(payload["config_name"]), payload["config"], str(config_file.resolve())


def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x)
    q75, q25 = np.nanpercentile(x, [75, 25])
    scale = max(float(q75 - q25), float(np.nanstd(x)) * 0.1, 1e-8)
    return np.nan_to_num((x - med) / scale, nan=0.0, posinf=0.0, neginf=0.0)


def require_columns(df: pd.DataFrame, cols: Iterable[str], context: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "false_alarm_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
    }


def pred_metrics(labels: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    y = labels.astype(bool)
    p = pred.astype(bool)
    tp = int(np.sum(y & p))
    fp = int(np.sum(~y & p))
    fn = int(np.sum(y & ~p))
    tn = int(np.sum(~y & ~p))
    return metrics_from_counts(tp, fp, fn, tn)


def label_runs(labels: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start = None
    for i, value in enumerate(labels.astype(bool)):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(labels)))
    return runs


def adjusted_metrics(labels: np.ndarray, pred_event: np.ndarray, runs: List[Tuple[int, int]]) -> Tuple[Dict[str, float], np.ndarray]:
    y = labels.astype(bool)
    p = pred_event.astype(bool).copy()
    latency_total = 0
    hit_count = 0
    for start, end in runs:
        hit = np.flatnonzero(p[start:end])
        if len(hit):
            first = int(hit[0])
            latency_total += first
            hit_count += 1
            p[start:end] = True
    out = pred_metrics(y.astype(int), p.astype(int))
    out["latency"] = float(latency_total / hit_count) if hit_count else 0.0
    return out, p.astype(int)


def build_model_from_ckpt(ckpt, args: argparse.Namespace, meta_dim: int, time_dim: int) -> PhysicalTopologyMPFMResidualGWNM:
    a = ckpt["args"]
    topo_nodes = int(a.get("topo_nodes_resolved", a.get("topo_nodes", 4)))
    model = PhysicalTopologyMPFMResidualGWNM(
        window_len=int(a["window"]),
        meta_dim=meta_dim,
        time_dim=time_dim,
        hidden_dim=int(a.get("hidden_dim", 96)),
        latent_dim=int(a.get("latent_dim", 16)),
        graph_layers=int(a.get("graph_layers", 1)),
        dropout=float(a.get("dropout", 0.05)),
        adaptive_rank=int(a.get("adaptive_rank", 16)),
        use_temporal_embedding=bool(int(a.get("use_temporal_embedding", 1))),
        use_residual_branch=bool(int(a.get("use_residual_branch", 1))),
        use_adaptive_graph=bool(int(a.get("use_adaptive_graph", 1))),
        num_domains=int(ckpt.get("num_domains", a.get("num_domains", 1))),
        k_shared=int(a.get("k_shared", 16)),
        k_private=int(a.get("k_private", 8)),
        tau=float(a.get("tau", 0.5)),
        flow_hidden=int(a.get("flow_hidden", 128)),
        flow_depth=int(a.get("flow_depth", 3)),
        topo_nodes=topo_nodes,
        topo_layers=int(a.get("topo_layers", 1)),
    ).to(args.device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model


def target_topology(adj_path: str, columns, topo_nodes: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    raw = load_adj_matrix(adj_path, columns)
    assign, topo_adj, info = physical_topology_from_adj(raw, topo_nodes=topo_nodes)
    return torch.from_numpy(assign).float(), torch.from_numpy(topo_adj).float(), info


@torch.no_grad()
def score_window_set(
    model: PhysicalTopologyMPFMResidualGWNM,
    windows_np: np.ndarray,
    time_np: np.ndarray,
    meta: torch.Tensor,
    adj: torch.Tensor,
    topo_assign: torch.Tensor,
    topo_adj: torch.Tensor,
    private_target: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    model.eval()
    windows = torch.from_numpy(windows_np).float()
    time_feat = torch.from_numpy(time_np).float()
    meta_d = meta.to(args.device)
    adj_d = adj.to(args.device)
    topo_assign_d = topo_assign.to(args.device)
    topo_adj_d = topo_adj.to(args.device)
    private_target = private_target.to(args.device)
    outs: Dict[str, list] = {
        "proto_dist_score": [],
        "proto_rec_score": [],
        "flow_score": [],
        "rec_score": [],
        "rec_local_score": [],
        "rec_topk_score": [],
        "topo_rough_score": [],
    }

    for i in range(0, len(windows), args.batch_size):
        xb = windows[i : i + args.batch_size].to(args.device)
        tb = time_feat[i : i + args.batch_size].to(args.device)
        out = model(
            xb,
            meta=meta_d,
            adj=adj_d,
            time_feat=tb,
            topo_assign=topo_assign_d,
            topo_adj=topo_adj_d,
            domain_idx=0,
            private_override=private_target,
        )
        ctx = model.mpfm.domain_context(0, private_override=private_target).detach()
        flow_err = model.mpfm.flow_error(
            out["h_topo"],
            ctx,
            topo_adj_d,
            t_star=args.flow_t_star,
            mc_samples=args.flow_mc,
            deterministic=bool(args.flow_deterministic),
        )
        rec = reconstruction_score(xb.detach().cpu(), out["x_rec"].detach().cpu())
        rec_local, rec_topk = reconstruction_anchor_scores(
            xb.detach().cpu(),
            out["x_rec"].detach().cpu(),
            mode=args.test_window_mode,
            steps=args.rec_local_steps,
            topk_fraction=args.rec_topk_fraction,
        )
        rough = topology_roughness_score(out["h_topo"].detach(), topo_adj_d)
        outs["proto_dist_score"].append(out["proto_min_dist"].detach().cpu())
        outs["proto_rec_score"].append(out["proto_proto_rec"].detach().cpu())
        outs["flow_score"].append(flow_err.detach().cpu())
        outs["rec_score"].append(rec.detach().cpu())
        outs["rec_local_score"].append(rec_local.detach().cpu())
        outs["rec_topk_score"].append(rec_topk.detach().cpu())
        outs["topo_rough_score"].append(rough.detach().cpu())

    return {key: torch.cat(value, dim=0).numpy() for key, value in outs.items()}


def generate_component_frame(preset_name: str, args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict]:
    preset = PRESETS[preset_name]
    require_file(args.ckpt_path, "checkpoint")
    require_file(args.target_train_csv, "target train CSV")
    require_file(args.target_adj_csv, "target adjacency CSV")
    require_file(preset["test_csv"], "test CSV")
    require_file(args.ground_truth_csv, "ground-truth CSV")

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    target_series, normalizer, target_win, _, target_adj, target_meta = load_target_windows(
        args.target_train_csv,
        args.target_adj_csv,
        args,
        normalizer=None,
        fit_norm=True,
        point=False,
    )
    test_series, _, test_win, _, test_adj, test_meta = load_target_windows(
        str(preset["test_csv"]),
        args.target_adj_csv,
        args,
        normalizer=normalizer,
        fit_norm=False,
        point=True,
    )

    if int(args.max_test_windows) > 0 and len(test_win.windows) > int(args.max_test_windows):
        idx = np.linspace(0, len(test_win.windows) - 1, num=int(args.max_test_windows)).round().astype(np.int64)
        test_win.windows = test_win.windows[idx]
        test_win.time_features = test_win.time_features[idx]
        test_win.starts = test_win.starts[idx]
        test_win.ends = test_win.ends[idx]
        if test_win.anchors is not None:
            test_win.anchors = test_win.anchors[idx]

    model = build_model_from_ckpt(ckpt, args, meta_dim=target_meta.shape[1], time_dim=target_win.time_features.shape[-1])
    topo_nodes = int(ckpt["args"].get("topo_nodes_resolved", ckpt["args"].get("topo_nodes", 4)))
    target_topo_assign, target_topo_adj, _ = target_topology(args.target_adj_csv, target_series.columns, topo_nodes)
    test_topo_assign, test_topo_adj, _ = target_topology(args.target_adj_csv, test_series.columns, topo_nodes)

    support_windows = torch.from_numpy(target_win.windows).float()
    support_time = torch.from_numpy(target_win.time_features).float()
    private_target, support_stats = init_private_from_support(
        model,
        support_windows,
        support_time,
        target_meta,
        target_adj,
        target_topo_assign,
        target_topo_adj,
        batch_size=args.batch_size,
        device=args.device,
        k_private=int(ckpt["args"].get("k_private", 8)),
        kmeans_iters=args.kmeans_iters,
    )
    for param in model.parameters():
        param.requires_grad_(False)
    private_target = adapt_private_on_support(
        model,
        private_target.to(args.device),
        support_stats["h_topo"].to(args.device),
        target_topo_adj.to(args.device),
        steps=args.adapt_private_steps,
        lr=args.adapt_private_lr,
        lambda_flow=args.adapt_lambda_flow,
        lambda_ortho=args.adapt_lambda_ortho,
        lambda_smooth=args.adapt_lambda_smooth,
        lambda_anchor=args.adapt_lambda_anchor,
    ).detach()

    test_components_win = score_window_set(
        model,
        test_win.windows,
        test_win.time_features,
        test_meta,
        test_adj,
        test_topo_assign,
        test_topo_adj,
        private_target,
        args,
    )
    point_scores = assign_window_scores_to_points(
        len(test_series.values),
        test_win.anchors,
        test_components_win,
        fill=args.fill_unavailable,
    )
    labels = align_labels_to_series(
        test_series,
        args.ground_truth_csv,
        datetime_col=args.datetime_col,
        label_col=args.label_col,
    )

    frame = pd.DataFrame({"datetime": test_series.datetime.astype(str)})
    for key, value in point_scores.items():
        frame[key] = value
    frame["label"] = labels
    info = {
        "checkpoint": str(Path(args.ckpt_path).resolve()),
        "preset": preset_name,
        "test_csv": str(Path(preset["test_csv"]).resolve()),
        "target_train_csv": str(Path(args.target_train_csv).resolve()),
        "target_adj_csv": str(Path(args.target_adj_csv).resolve()),
        "rows": int(len(frame)),
        "label_positive_points": int(labels.sum()),
    }
    return frame, info


def build_tuned_score(df: pd.DataFrame, config: Dict) -> np.ndarray:
    weights = config.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Config must contain a non-empty weights dictionary.")
    require_columns(df, weights.keys(), context="component score frame")

    if str(config.get("score_norm", "robust_z")).lower().strip() != "robust_z":
        raise ValueError("Only score_norm=robust_z is supported by this reproduction script.")

    score = np.zeros(len(df), dtype=np.float64)
    for column, weight in weights.items():
        score += float(weight) * robust_z(df[column].to_numpy(dtype=np.float64))
    if int(config.get("smooth_window", 1)) > 1:
        score = rolling_median_score(score, int(config["smooth_window"]))
    return score


def resolve_threshold(score: np.ndarray, config: Dict) -> float:
    finite = score[np.isfinite(score)]
    if len(finite) == 0:
        raise ValueError("Tuned score has no finite values.")
    mode = str(config.get("threshold_mode", "quantile")).lower().strip()
    if mode == "fixed":
        if "threshold" not in config:
            raise ValueError("threshold_mode=fixed requires threshold in config.")
        return float(config["threshold"])
    if mode == "quantile":
        return float(np.quantile(finite, float(config["threshold_quantile"])))
    raise ValueError(f"Unsupported threshold_mode={mode!r}.")


def materialize_fixed_result(
    df: pd.DataFrame,
    config_name: str,
    config: Dict,
    out_dir: Path,
    prefix: str,
    info: Dict,
) -> Dict:
    require_columns(df, ["label"], context="component score frame")
    labels = df["label"].fillna(0).astype(int).to_numpy()
    runs = label_runs(labels)
    score = build_tuned_score(df, config)
    threshold = resolve_threshold(score, config)
    pred_raw = suppress_short_runs((score >= threshold).astype(int), int(config.get("min_anomaly_run", 1)))
    pred_event = event_peak_candidates(
        score,
        pred_raw,
        mode=str(config.get("event_peak_mode", "none")),
        radius=int(config.get("event_peak_radius", 10)),
    )
    adjusted, pred_adjusted = adjusted_metrics(labels, pred_event, runs)

    scores_csv = out_dir / f"{prefix}_{config_name}_scores.csv"
    metrics_json = out_dir / f"{prefix}_{config_name}_metrics.json"
    out = df.copy()
    out["anomaly_score_tuned"] = score
    out["pred_label_adjusted"] = pred_adjusted
    out["pred_label"] = pred_adjusted

    payload = {
        **config,
        "config_name": config_name,
        "threshold_used": float(threshold),
        "threshold_mode_used": str(config.get("threshold_mode", "quantile")),
        "adjusted": adjusted,
        "adjusted_f1": float(adjusted["f1"]),
        "adjusted_precision": float(adjusted["precision"]),
        "adjusted_recall": float(adjusted["recall"]),
        "adjusted_fp": int(adjusted["fp"]),
        "adjusted_fn": int(adjusted["fn"]),
        "adjusted_latency": float(adjusted.get("latency", 0.0)),
        "output_csv": str(scores_csv.resolve()),
        "run_info": info,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(scores_csv, index=False)
    save_json(json_safe(payload), str(metrics_json))
    return payload


def run_one(preset_name: str, args: argparse.Namespace) -> Dict:
    preset = PRESETS[preset_name]
    config_name, config, config_source = load_preset_config(preset_name, args)

    out_dir = Path(args.out_dir).resolve() / preset_name
    prefix = str(preset["prefix"])


    if args.base_score_csv:
        df = pd.read_csv(args.base_score_csv)
        info = {
            "preset": preset_name,
            "base_score_csv": str(Path(args.base_score_csv).resolve()),
            "component_source": "provided_base_score_csv",
            "config_source": config_source,
        }
    else:
        print(f"[model] scoring preset={preset_name}")
        df, info = generate_component_frame(preset_name, args)
        base_csv = out_dir / f"{prefix}_base_components.csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(base_csv, index=False)
        info["base_components_csv"] = str(base_csv.resolve())
        info["component_source"] = "model_inference"
        info["config_source"] = config_source

    result = materialize_fixed_result(
        df,
        config_name,
        config,
        out_dir,
        prefix,
        info,
    )
    print(f"[done] {preset_name}: adjusted_f1={result['adjusted_f1']:.6f} csv={result['output_csv']}")
    return result


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if int(args.torch_num_threads) > 0:
        torch.set_num_threads(int(args.torch_num_threads))
    requested = canonical_preset(args.preset)
    presets = list(PRESETS) if requested == "all" else [requested]

    summary = {
        "preset": args.preset,
        "canonical_presets": presets,
        "results": {},
    }
    for preset_name in presets:
        summary["results"][preset_name] = run_one(preset_name, args)

    summary_path = Path(args.out_dir).resolve() / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(json_safe(summary), str(summary_path))
    print(f"[summary] {summary_path}")


if __name__ == "__main__":
    main()

