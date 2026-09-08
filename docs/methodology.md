# Methodology

## Research question

Can one model learn short-horizon technical patterns shared across a diverse set of Colombo Stock Exchange securities and estimate the next observed session's close-to-close return?

The implementation uses gradient-boosted trees (`XGBRegressor`) because they capture nonlinear interactions, require little feature scaling, and work well on tabular data. This choice does not establish that the market is predictably profitable.

## Data pipeline

1. Convert canonical CSE symbols such as `JKH.N0000` to Yahoo's `JKH-N0000.CM` format.
2. Download daily unadjusted Open, High, Low, Close, and Volume bars.
3. Reject empty histories, invalid schemas, and histories shorter than 200 rows.
4. Sort by date and remove rows with non-positive volume. Yahoo can forward-fill inactive or unavailable CSE observations; retaining these would create artificial zero returns.
5. Generate indicators using only the current and previous rows.
6. Shift the current return backward one row to create the next-session target.
7. Remove rows affected by an absolute daily return above 40%, including lag and target neighbors.
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

`Target_Return` is the following observed row's close return. The final row has no known target and is excluded.

Unadjusted corporate actions can resemble extraordinary market returns. A jump contaminates:

- its current return,
- later rows that include it as a lag, and
- the preceding row whose target equals that return.

The pipeline removes this neighborhood using a configurable-in-code 40% threshold. A production system should prefer authoritative corporate-action adjustments rather than this heuristic.

## Validation

Each stock is prepared and split independently: its earliest 80% becomes training data and its latest 20% becomes test data. Assets are concatenated only after this split. This avoids a random row split and guarantees that a stock's own test observations follow its training observations.

There remains a cross-sectional calendar caveat: because assets have unequal histories, one asset's test dates can overlap another asset's training dates. The model contains no direct cross-asset features, but a strict global-date or walk-forward split would be stronger for production-grade research.

Reported metrics are:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R²
- Directional accuracy
- Zero-return baseline MAE

Directional accuracy alone is not profitability. Evaluation does not currently model spreads, fees, liquidity, execution delay, or position sizing.

## Reproducibility

Use `--train-end YYYY-MM-DD` to freeze the information set. Every trained model receives adjacent JSON metadata containing its features, hyperparameters, assets, row counts, cutoff, creation time, and metrics. The loader rejects a model whose embedded feature names differ from the application's feature schema.
