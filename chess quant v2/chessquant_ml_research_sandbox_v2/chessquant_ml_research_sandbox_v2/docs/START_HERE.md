# ChessQuant Research Sandbox - Start Here

## Install

After this cleanup, you can install dependencies directly in the sandbox:

```powershell
poetry install
```

## First run

```powershell
poetry run python scripts/train_lab.py --config configs/experiments/peak_binary_xgb.yaml
poetry run python scripts/evaluate_policy.py backtest-policy --run-dir outputs/models/latest --policy configs/policies/balanced.yaml
poetry run python scripts/visualize_lab.py policy-report --run-dir $(Get-Content outputs/models/latest) --policy configs/policies/balanced.yaml --output-dir outputs/reports/policy_report_peak
```

## Cleanup policy

- Keep only stable code and docs inside the repo.
- Do not keep ad-hoc patch folders after applying them.
- Generated reports belong in `outputs/`.
- Dataset copies belong in `data/`.
