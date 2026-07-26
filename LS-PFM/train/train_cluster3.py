#!/usr/bin/env python3
"""Train the physical-topology MPFM-GWNM model on the packaged Cluster3 data."""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.gwnm_causal import pseudo_anomaly_windows
from src.gwnm_data import (
    adjacency_to_distance,
    build_correlation_adjacency,
    build_station_meta,
    load_adj_matrix,
    load_time_series,
    make_normalizer,
    make_windows,
    normalize_adjacency_from_distance,
    preprocess_values,
    save_json,
)
from src.topology_mpfm_gwnm import (
    prototype_usage_loss,
    topology_smoothness_loss,
    topology_spread_loss,
)
from src.physical_topology_mpfm_gwnm import PhysicalTopologyMPFMResidualGWNM
from src.physical_topology_utils import (
    parse_topo_nodes,
    physical_topology_from_adj,
    resolve_read_path,
    resolve_write_path,
    save_topology_info,
)


@dataclass
class DomainPack:
    name: str
    series_len: int
    num_nodes: int
    loader: DataLoader
    meta: torch.Tensor
    adj: torch.Tensor
    dist: torch.Tensor
    topo_assign: torch.Tensor
    topo_adj: torch.Tensor
    topology_info: Dict
    normalizer_payload: Dict


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def resolve_csv_list(s: str) -> str:
    return ",".join(resolve_read_path(x.strip(), ROOT) for x in s.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train physical-topology latent MPFM-GWNM.")
    p.add_argument("--source_train_csvs", default=str(ROOT / "data" / "cluster3" / "RRC_ConnMean_cluster3_train.csv"))
    p.add_argument("--source_adj_csvs", default=str(ROOT / "data" / "cluster3" / "cluster3_adj.csv"))
    p.add_argument("--save_dir", default=str(ROOT / "outputs" / "train_cluster3"))
    p.add_argument("--datetime_col", default="datetime")

    p.add_argument("--window", type=int, default=15)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--normalizer", choices=["standard", "robust"], default="robust")
    p.add_argument("--adj_mode", choices=["auto", "similarity", "distance"], default="auto")
    p.add_argument("--preprocess_mode", choices=["none", "log1p", "diff", "logdiff", "log_residual", "residual_diff"], default="none")
    p.add_argument("--preprocess_rolling", type=int, default=96)

    p.add_argument("--hidden_dim", type=int, default=96)
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--graph_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--adaptive_rank", type=int, default=16)
    p.add_argument("--use_temporal_embedding", type=int, default=1)
    p.add_argument("--use_residual_branch", type=int, default=1)
    p.add_argument("--use_adaptive_graph", type=int, default=1)

    p.add_argument("--topo_nodes", default="auto", help="auto derives K from source adjacency; integer forces K.")
    p.add_argument("--topo_max_auto", type=int, default=8)
    p.add_argument("--topo_layers", type=int, default=1)

    p.add_argument("--k_shared", type=int, default=16)
    p.add_argument("--k_private", type=int, default=8)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--flow_hidden", type=int, default=128)
    p.add_argument("--flow_depth", type=int, default=3)

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_train_windows", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--num_workers", type=int, default=0)

    # Defaults are deliberately close to run_train_eval_windows.bat, except
    # preprocess_mode defaults to none so the F1-optimized eval path can be used
    # without a train/eval preprocessing mismatch.
    p.add_argument("--lambda_rec", type=float, default=1.0)
    p.add_argument("--lambda_proto", type=float, default=0.3)
    p.add_argument("--lambda_flow", type=float, default=0.8)
    p.add_argument("--lambda_nce", type=float, default=0.03)
    p.add_argument("--lambda_ortho", type=float, default=0.01)
    p.add_argument("--lambda_hard", type=float, default=0.0)
    p.add_argument("--lambda_pseudo_score", type=float, default=0.5)
    p.add_argument("--lambda_topo_smooth", type=float, default=0.03)
    p.add_argument("--lambda_topo_spread", type=float, default=0.03)
    p.add_argument("--lambda_proto_usage", type=float, default=0.02)
    p.add_argument("--lambda_proto_smooth", type=float, default=0.01)
    p.add_argument("--pseudo_margin", type=float, default=1.0)
    p.add_argument("--pseudo_max_seg", type=int, default=8)
    p.add_argument("--pseudo_jitter_sigma", type=float, default=0.5)
    p.add_argument("--pseudo_mode", choices=["mixed", "permute_jitter"], default="mixed")
    p.add_argument("--topo_spread_margin", type=float, default=0.05)
    p.add_argument("--hard_beta", type=float, default=0.5)
    p.add_argument("--hard_topk", type=int, default=8)

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--torch_num_threads", type=int, default=1)
    return p.parse_args()


def load_domain(csv_path: str, adj_path: str, args: argparse.Namespace, domain_idx: int, topo_nodes: int) -> DomainPack:
    series = load_time_series(csv_path, datetime_col=args.datetime_col)
    values_proc = preprocess_values(series.values, mode=args.preprocess_mode, rolling=args.preprocess_rolling)
    normalizer = make_normalizer(args.normalizer)
    values_norm = normalizer.fit_transform(values_proc)
    win = make_windows(values_norm, series.datetime, window=args.window, stride=args.stride)
    if int(args.max_train_windows) > 0 and len(win.windows) > int(args.max_train_windows):
        n = int(args.max_train_windows)
        idx = np.linspace(0, len(win.windows) - 1, num=n).round().astype(np.int64)
        win.windows = win.windows[idx]
        win.time_features = win.time_features[idx]
        win.starts = win.starts[idx]
        win.ends = win.ends[idx]
    if adj_path and os.path.exists(adj_path):
        adj_raw = load_adj_matrix(adj_path, series.columns)
    else:
        print(f"[WARN] adjacency file not found for {csv_path}: {adj_path}. Using correlation adjacency fallback.")
        adj_raw = build_correlation_adjacency(values_proc)
    dist = adjacency_to_distance(adj_raw, mode=args.adj_mode)
    adj = normalize_adjacency_from_distance(dist)
    meta = build_station_meta(series.columns, dist)
    topo_assign, topo_adj, topology_info = physical_topology_from_adj(adj_raw, topo_nodes=topo_nodes)
    topology_info["domain_name"] = f"domain{domain_idx}:{os.path.basename(csv_path)}"
    topology_info["source_adj"] = adj_path
    loader = DataLoader(
        TensorDataset(torch.from_numpy(win.windows).float(), torch.from_numpy(win.time_features).float()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return DomainPack(
        name=f"domain{domain_idx}:{os.path.basename(csv_path)}",
        series_len=len(series.values),
        num_nodes=win.windows.shape[1],
        loader=loader,
        meta=torch.from_numpy(meta).float(),
        adj=torch.from_numpy(adj).float(),
        dist=torch.from_numpy(dist).float(),
        topo_assign=torch.from_numpy(topo_assign).float(),
        topo_adj=torch.from_numpy(topo_adj).float(),
        topology_info=topology_info,
        normalizer_payload=normalizer.to_dict(),
    )


def save_checkpoint(path: str, model: PhysicalTopologyMPFMResidualGWNM, args: argparse.Namespace, domains: List[DomainPack], best_loss: float, epoch: int) -> None:
    payload = {
        "format": "physical_topology_mpfm_residual_gwnm_v1",
        "model_state": model.state_dict(),
        "args": vars(args),
        "num_domains": len(domains),
        "domain_names": [d.name for d in domains],
        "source_normalizers": [d.normalizer_payload for d in domains],
        "topology_info": [d.topology_info for d in domains],
        "best_loss": float(best_loss),
        "epoch": int(epoch),
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    args.source_train_csvs = resolve_csv_list(args.source_train_csvs)
    args.source_adj_csvs = resolve_csv_list(args.source_adj_csvs)
    args.save_dir = resolve_write_path(args.save_dir, ROOT)
    set_seed(args.seed)
    if int(args.torch_num_threads) > 0:
        torch.set_num_threads(int(args.torch_num_threads))
    os.makedirs(args.save_dir, exist_ok=True)

    csvs = split_csv_list(args.source_train_csvs)
    adjs = split_csv_list(args.source_adj_csvs)
    if len(csvs) != len(adjs):
        raise ValueError("--source_train_csvs and --source_adj_csvs must have the same number of comma-separated paths.")
    if not csvs:
        raise ValueError("At least one source domain is required.")

    first_series = load_time_series(csvs[0], datetime_col=args.datetime_col)
    first_adj = load_adj_matrix(adjs[0], first_series.columns) if adjs[0] and os.path.exists(adjs[0]) else build_correlation_adjacency(first_series.values)
    topo_nodes = parse_topo_nodes(args.topo_nodes, first_adj, max_auto=args.topo_max_auto)
    args.topo_nodes_resolved = int(topo_nodes)
    save_json(vars(args), os.path.join(args.save_dir, "train_args.json"))

    domains = [load_domain(c, a, args, i, topo_nodes=topo_nodes) for i, (c, a) in enumerate(zip(csvs, adjs))]
    save_topology_info(os.path.join(args.save_dir, "topology_info.json"), {"topo_nodes": topo_nodes, "domains": [d.topology_info for d in domains]})

    meta_dim = domains[0].meta.shape[1]
    time_dim = next(iter(domains[0].loader))[1].shape[-1]
    model = PhysicalTopologyMPFMResidualGWNM(
        window_len=args.window,
        meta_dim=meta_dim,
        time_dim=time_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        graph_layers=args.graph_layers,
        dropout=args.dropout,
        adaptive_rank=args.adaptive_rank,
        use_temporal_embedding=bool(args.use_temporal_embedding),
        use_residual_branch=bool(args.use_residual_branch),
        use_adaptive_graph=bool(args.use_adaptive_graph),
        num_domains=len(domains),
        k_shared=args.k_shared,
        k_private=args.k_private,
        tau=args.tau,
        flow_hidden=args.flow_hidden,
        flow_depth=args.flow_depth,
        topo_nodes=topo_nodes,
        topo_layers=args.topo_layers,
    ).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Resolved physical topology nodes: {topo_nodes}")
    print("Loaded source domains:")
    for i, d in enumerate(domains):
        print(f"  [{i}] {d.name} | length={d.series_len} | physical_nodes={d.num_nodes} | topo_nodes={topo_nodes} | batches={len(d.loader)}")
    print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best = float("inf")
    prev_shared = None
    prev_private = None
    for epoch in range(args.epochs):
        model.train()
        if epoch > 0:
            prev_shared = model.mpfm.shared.detach().clone()
            prev_private = model.mpfm.private.detach().clone()
        totals = {k: 0.0 for k in ["loss", "rec", "proto", "flow", "nce", "ortho", "hard", "pseudo", "topo_smooth", "topo_spread", "usage", "proto_smooth"]}
        n_batches = 0
        for domain_idx, dpack in enumerate(domains):
            meta = dpack.meta.to(args.device)
            adj = dpack.adj.to(args.device)
            topo_assign = dpack.topo_assign.to(args.device)
            topo_adj = dpack.topo_adj.to(args.device)
            bar = tqdm(dpack.loader, desc=f"Epoch {epoch+1}/{args.epochs} D{domain_idx}", ncols=130, leave=False)
            for xb, tb in bar:
                xb = xb.to(args.device)
                tb = tb.to(args.device)
                out = model(xb, meta=meta, adj=adj, time_feat=tb, topo_assign=topo_assign, topo_adj=topo_adj, domain_idx=domain_idx)
                ctx = model.mpfm.domain_context(domain_idx)

                loss_rec = F.mse_loss(out["x_rec"], xb)
                loss_proto = (out["h_topo"] - out["proto_h_hat"]).pow(2).mean()
                loss_flow = 0.5 * model.mpfm.flow_loss(out["h_topo"], ctx, topo_adj) + 0.5 * model.mpfm.flow_loss(out["proto_h_hat"], ctx, topo_adj)
                loss_nce = model.mpfm.info_nce_loss(out["h_topo"], domain_idx=domain_idx)
                loss_ortho = model.mpfm.orthogonal_loss()
                loss_hard = model.mpfm.hard_drift_loss(
                    domain_idx,
                    prev_shared,
                    None if prev_private is None else prev_private[domain_idx],
                    topo_adj=topo_adj,
                    beta=args.hard_beta,
                    topk=args.hard_topk,
                )
                if float(args.lambda_pseudo_score) > 0:
                    xp = pseudo_anomaly_windows(xb, max_seg=args.pseudo_max_seg, sigma=args.pseudo_jitter_sigma, mode=args.pseudo_mode)
                    out_p = model(xp, meta=meta, adj=adj, time_feat=tb, topo_assign=topo_assign, topo_adj=topo_adj, domain_idx=domain_idx)
                    rec_n = (out["x_rec"] - xb).pow(2).flatten(1).mean(dim=1)
                    rec_p = (out_p["x_rec"] - xp).pow(2).flatten(1).mean(dim=1)
                    score_n = 0.5 * rec_n + out["proto_min_dist"] + out["proto_proto_rec"]
                    score_p = 0.5 * rec_p + out_p["proto_min_dist"] + out_p["proto_proto_rec"]
                    loss_pseudo = F.relu(float(args.pseudo_margin) + score_n - score_p).mean()
                else:
                    loss_pseudo = loss_rec.new_tensor(0.0)

                loss_topo_smooth = topology_smoothness_loss(out["h_topo"], topo_adj)
                loss_topo_spread = topology_spread_loss(out["h_topo"], margin=args.topo_spread_margin)
                loss_usage = prototype_usage_loss(out["proto_alpha"])
                loss_proto_smooth = model.mpfm.prototype_smoothness_loss(topo_adj)

                loss = (
                    args.lambda_rec * loss_rec
                    + args.lambda_proto * loss_proto
                    + args.lambda_flow * loss_flow
                    + args.lambda_nce * loss_nce
                    + args.lambda_ortho * loss_ortho
                    + args.lambda_hard * loss_hard
                    + args.lambda_pseudo_score * loss_pseudo
                    + args.lambda_topo_smooth * loss_topo_smooth
                    + args.lambda_topo_spread * loss_topo_spread
                    + args.lambda_proto_usage * loss_usage
                    + args.lambda_proto_smooth * loss_proto_smooth
                )
                optim.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optim.step()

                vals = {
                    "loss": loss.item(),
                    "rec": loss_rec.item(),
                    "proto": loss_proto.item(),
                    "flow": loss_flow.item(),
                    "nce": loss_nce.item(),
                    "ortho": loss_ortho.item(),
                    "hard": loss_hard.item(),
                    "pseudo": loss_pseudo.item(),
                    "topo_smooth": loss_topo_smooth.item(),
                    "topo_spread": loss_topo_spread.item(),
                    "usage": loss_usage.item(),
                    "proto_smooth": loss_proto_smooth.item(),
                }
                for k, v in vals.items():
                    totals[k] += float(v)
                n_batches += 1
                bar.set_postfix({k: f"{totals[k]/max(n_batches,1):.4f}" for k in ["loss", "rec", "proto", "flow", "pseudo"]})

        avg = {k: v / max(n_batches, 1) for k, v in totals.items()}
        save_json(avg, os.path.join(args.save_dir, f"epoch_{epoch+1:03d}_loss.json"))
        save_checkpoint(os.path.join(args.save_dir, "latest.pt"), model, args, domains, avg["loss"], epoch + 1)
        if avg["loss"] < best:
            best = avg["loss"]
            save_checkpoint(os.path.join(args.save_dir, "best.pt"), model, args, domains, best, epoch + 1)
        print(f"Epoch {epoch+1}: {json.dumps(avg, ensure_ascii=False)} | best={best:.6f}")

    print("Training finished. Best checkpoint:", os.path.join(args.save_dir, "best.pt"))


if __name__ == "__main__":
    main()
