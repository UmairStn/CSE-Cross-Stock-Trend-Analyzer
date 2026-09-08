# Methodology

## Research question

Can one model learn short-horizon technical patterns shared across a diverse set of Colombo Stock Exchange securities and estimate the cumulative close-to-close return over the next 1, 7, or 30 trading sessions?

The implementation uses gradient-boosted trees (`XGBRegressor`) because they capture nonlinear interactions, require little feature scaling, and work well on tabular data. One model is trained per horizon; the features are identical, only the label window differs. This choice does not establish that the market is predictably profitable.

## Data pipeline

1. Convert canonical CSE symbols such as `JKH.N0000` to Yahoo's `JKH-N0000.CM` format.
2. Download daily unadjusted Open, High, Low, Close, and Volume bars.
3. Reject empty histories, invalid schemas, and histories shorter than 200 rows.
4. Sort by date and remove rows with non-positive volume. Yahoo can forward-fill inactive or unavailable CSE observations; retaining these would create artificial zero returns.
5. Generate indicators using only the current and previous rows.
6. Compute the target as the cumulative close-to-close return over the next `h` trading rows: `Close(t+h) / Close(t) - 1`.
7. Remove rows affected by an absolute daily return above 40%, including lag and target-window neighbours.
8. Drop indicator warm-up rows with missing feature values.

## Feature definitions

Let `C`, `O`, `H`, and `L` be close, open, high, and low.

- `Open_Pct = O / C - 1`
- `High_Pct = H / C - 1`
- `Low_Pct = L / C - 1`
- `Rel_Volume = Volume / rolling_mean_20(Volume)`
- `RSI_3`: three-session Relative Strength Index
- `Return_Lag1` and `Return_Lag2`: one- and two-row lags of close return
- `SMA_3_Pct = SMA_3 / C - 1`
- `EMA_3_Pct = EMA_3 / C - 1`
- `MACD_Pct = (EMA_12 - EMA_26) / C`
- `MACD_Hist_Pct = (MACD - MACD_signal_9) / C`

Normalization allows securities at different nominal price and volume levels to share one model. Company identity is intentionally absent.

## Target and contamination controls

`Target_Return(t, h) = Close(t + h) / Close(t) - 1` is the cumulative return over the next `h` trading rows. The final `h` rows per security have no complete target and are excluded.

Unadjusted corporate actions can resemble extraordinary market returns. A jump contaminates:

- its current return,
- later rows that include it as a lag,
- and any row whose target window (the next `h` rows) contains the jump day.

The pipeline removes these neighborhoods using a configurable-in-code 40% threshold; for multi-day horizons the target-window check widens to the full horizon. A production system should prefer authoritative corporate-action adjustments rather than this heuristic.

## Validation

Each stock is prepared and split independently: its earliest 80% becomes training data and its latest 20% becomes test data. Assets are concatenated only after this split. This avoids a random row split and guarantees that a stock's own test observations follow its training observations.

**Label embargo.** The last `h` rows of each training slice are dropped because their target windows reach into the test period — without the embargo, test-period closes would appear in training labels.

**Overlapping labels.** For `h > 1`, consecutive targets share most of their window (two adjacent 7-day targets share 6 of 7 days). Test rows are therefore highly correlated with each other, which makes metrics look more stable than the effective sample size justifies. The zero-return baseline is reported per horizon so the model is always compared against a like-for-like naïve alternative; a purged walk-forward design would be stronger still.

There remains a cross-sectional calendar caveat: because assets have unequal histories, one asset's test dates can overlap another asset's training dates. The model contains no direct cross-asset features, but a strict global-date or walk-forward split would be stronger for production-grade research.

Reported metrics are:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R²
- Directional accuracy
- Zero-return baseline MAE

Directional accuracy alone is not profitability. Evaluation does not currently model spreads, fees, liquidity, execution delay, or position sizing.

## Reproducibility

Use `--train-end YYYY-MM-DD` to freeze the information set and `--horizon h` to select the label window. Every trained model receives adjacent JSON metadata containing its horizon, target definition, features, hyperparameters, assets, row counts, cutoff, creation time, and metrics. The loader rejects a model whose embedded feature names differ from the application's feature schema.
