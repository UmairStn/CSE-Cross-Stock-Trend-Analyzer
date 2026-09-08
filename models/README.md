# Model artifacts

Generated XGBoost models and their metadata are written here by `cse-analyzer train`.
They are ignored by Git by default because binary/generated artifacts are not source code.

Each training run creates one model per horizon `h`:

- `cse_next_day_regressor.json` + `.metadata.json` (h = 1, the default)
- `cse_next_day_regressor_h7.json` + `.metadata_h7.json` (weekly)
- `cse_next_day_regressor_h30.json` + `.metadata_h30.json` (monthly)

The metadata files record the horizon, target definition, features, data split
sizes, symbols, hyperparameters, and evaluation metrics.

To reproduce them, fetch data and run the training commands described in the root README.
