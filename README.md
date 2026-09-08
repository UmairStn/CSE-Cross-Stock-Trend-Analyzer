# CSE Stock Analyzer

A reproducible machine-learning research project for analyzing Colombo Stock Exchange (CSE) securities and estimating their **future returns over 1, 7, and 30 trading sessions** from daily price, volume, momentum, and trend signals.

> **Project status:** research/portfolio project — not a live trading system and not financial advice.

## What problem does this solve?

**Sri Lankan equity analysis is manual and time-constrained.** The CSE lists roughly 300 companies across the ASPI and S&P SL20, and an investor or analyst following it faces three recurring friction points:

1. **Screening burden.** Reviewing hundreds of tickers by hand — checking charts, volume, momentum — before a market open takes hours, so most coverage collapses to a handful of familiar large caps.
2. **Inconsistent process.** Judgement-based reading of RSI, moving averages, and volume spikes varies day to day and person to person; nothing is standardized or auditable.
3. **No local tooling.** Global screener platforms cover CSE thinly, and dedicated local analytics are scarce or closed.

**What this project provides instead:**

- **A daily screening shortlist, in seconds.** One command scores every ticker the same way, ranking them by expected return — e.g. "of these 3 banks, HNB looks strongest (+0.25%), COMB next (+0.12%), JKH flat (−0.03%)" — so attention goes to the shortlist instead of the whole board.
- **Standardized, auditable analysis.** Every ticker passes through the identical pipeline: symbol normalization → data validation → the same 11 features → the same model. Same inputs always produce the same score, and every result can be traced back through committed code and per-model metadata.
- **Time-horizon choice.** Separate 1-, 7-, and 30-session models match different decisions: a day trader wants tomorrow's expected move, a swing trader the coming week, a position trader the coming month — from one CLI.
- **Coverage of the whole board, not just big names.** The model is **cross-sectional**: it uses no company ID, only pattern-of-price-and-volume inputs, so it scores any CSE stock with sufficient history — including smaller tickers it never trained on.
- **Honest evaluation by default.** Every model is benchmarked against a zero-return baseline with a chronological, embargoed split — so the project itself tells you how much (or little) signal there is, rather than hiding it.

In short: **it turns a manual, judgement-driven review into a reproducible, ranked, multi-horizon screen for the whole CSE — and is upfront that it is a research tool, not a trading signal.**

### How it works, end to end

1. Normalize CSE ticker symbols.
2. Download daily OHLCV history from Yahoo Finance.
3. Remove non-trading/forward-filled observations.
4. Generate scale-independent technical features.
5. Train one cross-sectional XGBoost regression model per horizon (1/7/30 sessions) across many companies.
6. Evaluate each chronologically against a baseline, and predict expected returns.

See [methodology](docs/methodology.md) for the full technical detail.

## Highlights

- 54-symbol, sector-diverse CSE training universe
- Multi-horizon prediction: 1, 7, and 30 trading sessions (one model per horizon)
- Shared feature pipeline for training and inference
- Per-security chronological train/test split with label embargo to reduce future leakage
- Filtering around implausibly large split/scrip-related price jumps
- MAE, MSE, R², directional accuracy, and zero-return baseline
- Model metadata saved beside each trained artifact
- Fetch, train, and predict CLI
- Offline unit tests—CI does not depend on Yahoo Finance availability

## Architecture

```mermaid
flowchart LR
    A[CSE symbols] --> B[Yahoo Finance OHLCV]
    B --> C[Validation and non-trading-row removal]
    C --> D[Technical feature engineering]
    D --> E[Per-stock chronological split + embargo]
    E --> F[XGBoost regressor, one per horizon]
    F --> G[Evaluation + model metadata]
    F --> H[Multi-horizon return prediction]
```

## Model methodology

### Target

The continuous target is the **cumulative close-to-close return over the next `h` trading sessions** (`h` = 1, 7, or 30):

```text
Target_Return(t, h) = Close(t + h) / Close(t) - 1
```

One XGBoost model is trained per horizon. This is a **regression** task, not a price-level forecast or classifier. `UP`/`DOWN` is derived from the sign of the predicted return for presentation.

The same 11 features feed every horizon; only the label window changes. Longer horizons are harder: daily-noise features lose meaning as the window grows, and overlapping label windows make test metrics optimistic (see [methodology](docs/methodology.md)).

### Features

| Group | Features |
|---|---|
| Intraday price position | `Open_Pct`, `High_Pct`, `Low_Pct` |
| Volume | `Rel_Volume` (relative to a rolling 20-session mean) |
| Momentum | `RSI_3`, `Return_Lag1`, `Return_Lag2` |
| Trend | `SMA_3_Pct`, `EMA_3_Pct` |
| MACD | `MACD_Pct`, `MACD_Hist_Pct` |

Price-derived indicators are normalized by closing price, and no asset identifier is included. See [the methodology](docs/methodology.md) for preprocessing and validation details.

## Repository structure

```text
.
├── src/cse_stock_analyzer/   # installable application package
├── tests/                    # offline tests and synthetic fixtures
├── data/raw/                 # downloaded OHLCV CSVs (gitignored, see below)
├── models/                   # trained model + metadata (generated, ignored)
├── reports/                  # experiment reports and charts
├── docs/                     # methodology and limitations
├── .github/workflows/        # CI: ruff + pytest on every push
├── pyproject.toml            # package, dependencies, and tool configuration
├── LICENSE                   # MIT
└── README.md
```

## Getting started

### Requirements

- Python 3.10 or newer
- Internet access only when downloading current market data

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/cse-stock-analyzer.git
cd cse-stock-analyzer
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install the project:

```bash
pip install -e .

# Include test/lint tools when developing
pip install -e ".[dev]"
```

## Usage

All commands work as either `cse-analyzer ...` or `python -m cse_stock_analyzer ...`.

### 1. Fetch market data

Fetch the complete configured universe:

```bash
cse-analyzer fetch --period 5y
```

Fetch selected securities:

```bash
cse-analyzer fetch JKH.N0000 HNB.N0000 COMB.N0000 --period 5y
```

Short symbols such as `JKH` are normalized to `JKH.N0000`. Existing files are retained unless `--force` is supplied.

### 2. Train and evaluate

Train on locally available data in `data/raw/` (CSVs are not committed to
the repository — see [dataset](#dataset) below). Default horizon is 1 day:

```bash
cse-analyzer train
```

Train the weekly and monthly models too:

```bash
cse-analyzer train --horizon 7
cse-analyzer train --horizon 30
```

Any positive horizon works — each gets its own model and metadata file.

For reproducible historical experiments, specify an inclusive data cutoff:

```bash
cse-analyzer train --train-end 2025-12-31
```

The default cutoff is 2026-01-31 because Yahoo's CSE feed forward-fills ghost
rows after that date (see [data considerations](#data-considerations)); pass
`--train-end none` to disable it once the feed is live again.

Outputs (per horizon `h`):

- `models/cse_next_day_regressor.json` + `.metadata.json` (h = 1, the default)
- `models/cse_next_day_regressor_h7.json` + `.metadata_h7.json`
- `models/cse_next_day_regressor_h30.json` + `.metadata_h30.json`

The metadata contains the horizon, target definition, feature names, included assets, row counts, model parameters, timestamp, cutoff, and evaluation results. Do not copy sample metrics into a portfolio claim—run the command and report the generated results.

### 3. Predict

Use locally downloaded data where available, otherwise fetch recent history.
Default horizon is 1 day; add `--horizon` for the weekly/monthly models:

```bash
cse-analyzer predict JKH.N0000 HNB.N0000
```

```bash
cse-analyzer predict JKH.N0000 HNB.N0000 --horizon 7
```

Example output shape:

```json
[
  {
    "symbol": "JKH.N0000",
    "observation_date": "2025-12-31",
    "horizon_days": 7,
    "predicted_return": 0.0066,
    "predicted_percent": 0.66,
    "direction": "UP"
  }
]
```

`predicted_percent` is the cumulative close-to-close return expected over the
next `horizon_days` trading sessions. This example illustrates the schema
only; it is not a current forecast.

### Dataset

Price history is **not included** in the repository — CSVs are ignored by
Git (see `.gitignore`). Download the 54-symbol universe before your first
training run:

```bash
cse-analyzer fetch --period 5y
```

`SOFT.N0000` has no Yahoo Finance history, so in practice this yields ~53
csv files (2021-08 onward, ~4 MB). Yahoo's CSE coverage is uneven — some
symbols may be missing or short; the fetcher skips and reports them rather
than failing. See [data/README.md](data/README.md) for dataset properties
and attribution.

## Testing and code quality

```bash
pytest
ruff check .
```

Tests use deterministic synthetic price series and therefore work without network access.

## Data considerations

Yahoo Finance CSE symbols generally follow this conversion:

```text
JKH.N0000  ->  JKH-N0000.CM
```

Coverage can be patchy. A successful HTTP response does not guarantee complete, current, or corporate-action-adjusted data. The fetcher deliberately uses unadjusted prices (`auto_adjust=False`) and the training pipeline removes neighborhoods around daily returns greater than 40%, which are treated as probable split/scrip artifacts.

Read [Data and limitations](docs/data-and-limitations.md) before interpreting results.

## Limitations and responsible use

- Technical indicators carry weak and non-stationary predictive signals.
- Results can vary materially by cutoff, universe, corporate actions, and data quality.
- The current pipeline does not include fundamentals, disclosures, macroeconomics, news, transaction costs, liquidity, slippage, taxes, or portfolio risk constraints.
- A chronological holdout is useful but is not a complete walk-forward backtest.
- Predicted returns are research outputs—not buy, sell, or hold recommendations.
- Historical performance does not guarantee future performance.

## Roadmap

- Walk-forward and rolling-window validation
- ASPI/S&P SL20 market-regime and relative-strength features
- Corporate-action-aware adjusted data source
- Feature-importance and SHAP reports
- Transaction-cost-aware strategy backtesting
- Optional FastAPI service and dashboard after the research pipeline is validated

## Technology

Python, pandas, NumPy, scikit-learn, XGBoost, yfinance, pytest, and Ruff.

## License and attribution

The source code is released under the MIT License (see [LICENSE](LICENSE)). Market data remains subject to the terms of its original provider and exchange; an open-source code license does not grant redistribution rights for third-party data.

## Disclaimer

This repository is provided for educational and research purposes only. It does not constitute financial, investment, legal, or tax advice. Independently verify all data and consult a qualified professional before making investment decisions.
