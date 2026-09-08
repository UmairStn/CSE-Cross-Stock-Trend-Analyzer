"""Command-line interface for data collection, training, and prediction."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import DATA_DIR, DEFAULT_PERIOD, DEFAULT_TRAIN_END, METADATA_PATH, MODEL_PATH
from .data import fetch_history, load_history, save_history
from .model import load_model, predict_latest, train
from .universe import CSE_UNIVERSE, normalize_symbol, to_yahoo_symbol


def _data_path(data_dir: Path, symbol: str) -> Path:
    return data_dir / f"{to_yahoo_symbol(symbol)}.csv"


def fetch_command(args: argparse.Namespace) -> int:
    symbols = args.symbols or [symbol for symbol, _ in CSE_UNIVERSE]
    failures = 0
    for raw_symbol in symbols:
        symbol = normalize_symbol(raw_symbol)
        path = _data_path(args.data_dir, symbol)
        if path.exists() and not args.force:
            print(f"SKIP {symbol}: {path} already exists")
            continue
        try:
            frame = fetch_history(symbol, args.period)
            path = save_history(frame, symbol, args.data_dir)
            print(f"OK   {symbol}: {len(frame)} rows -> {path}")
            time.sleep(0.4)
        except Exception as error:
            failures += 1
            print(f"FAIL {symbol}: {error}")
    return 1 if failures == len(symbols) else 0


def train_command(args: argparse.Namespace) -> int:
    csv_paths = list(args.data_dir.glob("*.csv"))
    train_end = None if str(args.train_end).lower() in {"none", "off"} else args.train_end
    metadata = train(csv_paths, args.model, args.metadata, train_end)
    print(json.dumps(metadata["metrics"], indent=2))
    print(f"Model: {args.model}\nMetadata: {args.metadata}")
    return 0


def predict_command(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    records = []
    for raw_symbol in args.symbols:
        symbol = normalize_symbol(raw_symbol)
        path = _data_path(args.data_dir, symbol)
        frame = load_history(path) if path.exists() else fetch_history(symbol, args.period).reset_index()
        records.append(predict_latest(model, frame, symbol))
    print(json.dumps(records, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cse-analyzer", description="CSE stock research toolkit")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch_parser = commands.add_parser("fetch", help="download daily OHLCV data")
    fetch_parser.add_argument("symbols", nargs="*")
    fetch_parser.add_argument("--period", default=DEFAULT_PERIOD)
    fetch_parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    fetch_parser.add_argument("--force", action="store_true")
    fetch_parser.set_defaults(handler=fetch_command)

    train_parser = commands.add_parser("train", help="train and evaluate the return model")
    train_parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    train_parser.add_argument("--model", type=Path, default=MODEL_PATH)
    train_parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    train_parser.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    train_parser.set_defaults(handler=train_command)

    predict_parser = commands.add_parser("predict", help="predict next-session returns")
    predict_parser.add_argument("symbols", nargs="+", help="e.g. JKH.N0000 HNB.N0000")
    predict_parser.add_argument("--period", default=DEFAULT_PERIOD)
    predict_parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    predict_parser.add_argument("--model", type=Path, default=MODEL_PATH)
    predict_parser.set_defaults(handler=predict_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
