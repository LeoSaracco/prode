# Testing

Run the full ML safety suite:

```powershell
python -m pytest tests
```

Useful focused runs:

```powershell
python -m pytest tests/test_poisson_model.py
python -m pytest tests/test_group_simulator.py
python -m pytest tests/test_rolling_features.py
python -m pytest tests/test_report_generation.py
python -m pytest tests/test_runtime_predictions.py
```

Report-specific checks:

```powershell
python -m pytest tests/test_prediction_policy.py tests/test_poisson_model.py tests/test_report_generation.py
.\run_all.bat /reports
```

Expected report invariants:

- The PDF must show the dynamic feature count from `models/model_metadata.json` (`27 variables` in the current model), not a hardcoded `21 variables`.
- A predicted winner must never be rendered with `0-0`.
- `exact_most_likely_scoreline` remains the pure modal scoreline, while `outcome_scoreline` is the communicated scoreline conditioned on predicted outcome and xG volume.
- Group-stage report validation should include varied scorelines such as `2-0`, `2-1`, `1-2` or `3-0` when supported by xG and defensive weakness.
