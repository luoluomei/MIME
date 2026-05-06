#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Dict, Any

from mime_steal import MIMEConfig, run_mime
from mime_steal.utils import as_serializable


def parse_args():
    parser = argparse.ArgumentParser(description="Run standard MIME GNN model extraction.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name: CoCS, CoraFull, Arxiv, Products, Amazon-ratings, Squirrel")
    parser.add_argument("--c", type=int, required=True, help="Query budget multiplier. Total budget = C * number of classes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--root", type=str, default="./data", help="Dataset root directory.")
    parser.add_argument("--device", type=str, default=None, help="Device, e.g., cuda, cuda:0, cpu. Default: auto.")
    parser.add_argument("--prior-ratio", type=float, default=0.10, help="Visible attributed subgraph ratio.")
    parser.add_argument("--eval-scope", type=str, default="visible", choices=["visible", "full"], help="Evaluation scope.")
    parser.add_argument("--output", type=str, default="results/mime_results.csv", help="CSV output path.")

    # Common optional knobs for faster debugging or ablation-free reproduction.
    parser.add_argument("--victim-epochs", type=int, default=200)
    parser.add_argument("--dgi-epochs", type=int, default=200)
    parser.add_argument("--epochs-per-round", type=int, default=100)
    parser.add_argument("--final-epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lap-lambda", type=float, default=5e-4)
    return parser.parse_args()


def append_csv(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    config = MIMEConfig(
        root=args.root,
        prior_ratio=args.prior_ratio,
        device=args.device,
        eval_scope=args.eval_scope,
        victim_epochs=args.victim_epochs,
        dgi_epochs=args.dgi_epochs,
        epochs_per_round=args.epochs_per_round,
        final_epochs=args.final_epochs,
        hidden_dim=args.hidden_dim,
        lap_lambda=args.lap_lambda,
    )
    result = as_serializable(run_mime(dataset=args.dataset, c=args.c, seed=args.seed, config=config))
    append_csv(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved result to: {args.output}")


if __name__ == "__main__":
    main()
