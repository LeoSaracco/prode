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

