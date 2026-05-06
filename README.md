# MIME: Minimal Information Model Extraction for GNNs

This repository contains a modular implementation of **MIME** for partial-view, zero-prior, hard-label GNN model extraction. It keeps only the standard MIME pipeline:

1. label-free attributed-graph bootstrapping with DGI,
2. entropy-diversity query selection under a node-level hard-label budget,
3. topology-regularized surrogate refinement with a Laplacian prediction penalty.

The code is refactored from the original standalone runner into reusable Python modules. It removes baseline methods and extra experimental sweeps, so each run only executes MIME on one dataset, one query budget, and one random seed.

## Repository structure

```text
mime_github/
├── README.md
├── requirements.txt
├── run_mime.py
└── mime_steal/
    ├── __init__.py
    ├── config.py          # MIMEConfig dataclass
    ├── data.py            # dataset loading, splitting, visible subgraph construction
    ├── mime.py            # public run_mime(...) API
    ├── models.py          # GCN and DGI models
    ├── query.py           # k-center, entropy, exploration split
    ├── train.py           # victim training, surrogate refinement, evaluation
    └── utils.py           # seeds, metrics, serialization utilities
```

## Installation

Create an environment with PyTorch and PyTorch Geometric. The exact PyG install command depends on your CUDA/PyTorch version; see the official PyG instructions if needed.

```bash
pip install -r requirements.txt
```

For OGB datasets such as Arxiv and Products, `ogb` is required.

## Supported datasets

The command-line loader supports:

- `CoCS`
- `CoraFull`
- `Arxiv`
- `Products`
- `Amazon-ratings`
- `Squirrel`

For `Products`, the code uses the front-200K induced subgraph by OGB node order as the working graph.

## Quick start

Run MIME on CoCS with a query budget of `20C`, where `C` is the number of classes:

```bash
python run_mime.py --dataset CoCS --c 20 --seed 42
```

Run on Arxiv using CUDA and save results to a custom file:

```bash
python run_mime.py \
  --dataset Arxiv \
  --c 20 \
  --seed 42 \
  --device cuda \
  --output results/arxiv_c20_seed42.csv
```

Run Products with the front-200K working graph:

```bash
python run_mime.py --dataset Products --c 20 --seed 42 --device cuda
```

## Python API

```python
from mime_steal import MIMEConfig, run_mime

config = MIMEConfig(
    root="./data",
    prior_ratio=0.10,
    device="cuda",
    eval_scope="visible",  # or "full"
)

result = run_mime(dataset="CoCS", c=20, seed=42, config=config)
print(result)
```

The result dictionary includes accuracy, fidelity, number of queried nodes, runtime breakdown, and memory usage.

## Evaluation scope

The default setting is:

```bash
--eval-scope visible
```

This evaluates the surrogate on unqueried nodes inside the attacker-visible partial attributed graph, so the surrogate is never given the provider-side full graph during evaluation.

For compatibility with older transductive experimental runners, you can use:

```bash
--eval-scope full
```

This evaluates the learned surrogate weights on the full graph test mask. The attack still queries only nodes from the attacker-visible partial graph.

## Main command-line arguments

```bash
python run_mime.py \
  --dataset CoCS \
  --c 20 \
  --seed 42 \
  --root ./data \
  --device cuda \
  --prior-ratio 0.10 \
  --eval-scope visible
```

Optional training knobs:

```bash
--victim-epochs 200
--dgi-epochs 200
--epochs-per-round 100
--final-epochs 200
--hidden-dim 128
--lap-lambda 5e-4
```

## Protocol summary

- Victim split: 60% victim-training nodes and 40% held-out test nodes.
- Visible prior: sampled from the victim-training portion according to `--prior-ratio`.
- Oracle: each query returns one top-1 victim prediction for one real node in the visible partial graph.
- Budget: total queries = `c * number_of_classes`.
- Surrogate: 2-layer GCN.
- Initial query batch: greedy k-center coverage on DGI representations.
- Later query batches: exploration coverage plus entropy-diversity selection.
- Surrogate loss: hard-label cross-entropy plus prediction-level Laplacian regularization.

## Example output

Each run appends one row to the output CSV and prints a JSON object like:

```json
{
  "dataset": "CoCS",
  "seed": 42,
  "c": 20,
  "budget": 300,
  "visible_nodes": 1833,
  "queried_nodes": 300,
  "eval_scope": "visible",
  "accuracy": 0.91,
  "fidelity": 0.92,
  "pretrain_time": 12.34,
  "query_time": 0.56,
  "train_time": 78.90,
  "total_time": 100.12
}
```

## Notes

This repository intentionally keeps only the standard MIME attack. It does not include Random, CEGA, GNNSteal, EffGNN, OnStealing, label-flip defenses, domain-shift sweeps, or multi-seed table runners.
