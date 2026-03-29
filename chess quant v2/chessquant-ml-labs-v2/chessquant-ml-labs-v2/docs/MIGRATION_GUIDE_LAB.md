# Migration Guide

This guide assumes your current ML LABS root still resembles the handoff shape:
- `chessquant_ml/`
- `data/`
- `models/`
- `exports/`
- `experiments/`

That earlier scaffold matched the current fetch/enrich/features/train/export-web pipeline. The updated `features.py`, `train.py`, and `export_web.py` already assume a stable feature contract and browser export path. fileciteturn1file0 fileciteturn1file6

## Recommended migration strategy

### Safe changes now
- move package under `src/`
- add `configs/`
- add `outputs/`
- add `scripts/`
- add `tests/`
- keep your current CLI, pipeline, training, and export code in place

### Avoid on day 1
- renaming the package
- deleting legacy `models/` or `exports/` before confirming nothing points to them
- changing every path constant in one giant refactor

## Target state

```text
project/
├── configs/
├── data/
├── notebooks/
├── outputs/
├── scripts/
├── src/chessquant_ml/
└── tests/
```

## Compatibility strategy

For the first pass:
- keep writing production artifacts where your working code expects them
- also mirror research outputs into `outputs/`
- once stable, repoint defaults from legacy dirs to `outputs/`

## Splitter rationale

The handoff says session grouping and cumulative session P/L are already core features and targets, and grouped CV was already added in training. Because this is time-ordered personal game history, split logic should be explicit and testable to avoid leakage across sessions. fileciteturn1file0 fileciteturn1file6
