# Model artifacts

Generated XGBoost models and their metadata are written here by `cse-analyzer train`.
They are ignored by Git by default because binary/generated artifacts are not source code.

Each training run creates:

- `cse_next_day_regressor.json` — XGBoost model
- `cse_next_day_regressor.metadata.json` — features, data split sizes, symbols, hyperparameters, and metrics

To reproduce them, fetch data and run the training command described in the root README.
