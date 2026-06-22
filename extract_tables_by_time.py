#!/usr/bin/env python3
"""Extract METHOD/P@100R/R@100P tables from run logs in a time window.

Scans results directories named:
  results_YYYY-MM-DD_HH-MM-SS

For each run in [start, end], parses the latest table rows from output.log:
  Standard
  Weighted
  Weighted+Prob

Writes a consolidated text file that can be consumed by:
  plot_r100p_from_extracted_tables.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple


DIR_RE = re.compile(r"^results_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})$")
ROW_RE = {
    "Standard": re.compile(r"^Standard\s*\|.*$", re.MULTILINE),
    "Weighted": re.compile(r"^Weighted\s*\|.*$", re.MULTILINE),
    "Weighted+Prob": re.compile(r"^Weighted\+Prob\s*\|.*$", re.MULTILINE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract METHOD/P@100R/R@100P tables from output.log in a time window"
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Path to results root directory (default: results)",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start timestamp (YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End timestamp (YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--out-txt",
        default=None,
        help=(
            "Output text path. If omitted, writes to "
            "results/tables_<start>_to_<end>.txt"
        ),
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help=(
            "Keep only the first N runs per MAX_SAMPLES (earliest by timestamp). "
            "If omitted, keeps all runs in the window."
        ),
    )
    return parser.parse_args()


def parse_dir_timestamp(name: str) -> Optional[datetime]:
    match = DIR_RE.match(name)
    if not match:
        return None
    return datetime.strptime(
        f"{match.group(1)} {match.group(2).replace('-', ':')}",
        "%Y-%m-%d %H:%M:%S",
    )


def sanitize_ts(ts: str) -> str:
    return ts.replace(" ", "_").replace(":", "-")


def _load_max_samples(hparam_path: str) -> Optional[int]:
    try:
        with open(hparam_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        val = data.get("MAX_SAMPLES")
        return int(val) if val is not None else None
    except Exception:
        return None


def extract_latest_rows(log_text: str) -> Optional[Tuple[str, str, str]]:
    std_matches = ROW_RE["Standard"].findall(log_text)
    w_matches = ROW_RE["Weighted"].findall(log_text)
    wp_matches = ROW_RE["Weighted+Prob"].findall(log_text)

    if not std_matches or not w_matches or not wp_matches:
        return None

    return std_matches[-1], w_matches[-1], wp_matches[-1]


def collect_tables(results_root: str, start_dt: datetime, end_dt: datetime) -> List[dict]:
    runs: List[dict] = []

    for name in sorted(os.listdir(results_root)):
        ts = parse_dir_timestamp(name)
        if ts is None:
            continue
        if not (start_dt <= ts <= end_dt):
            continue

        run_dir = os.path.join(results_root, name)
        hparams_path = os.path.join(run_dir, "hyperparameters.json")
        max_samples = _load_max_samples(hparams_path)
        log_path = os.path.join(run_dir, "output.log")
        if not os.path.isfile(log_path):
            continue

        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            continue

        rows = extract_latest_rows(text)
        if rows is None:
            continue

        runs.append(
            {
                "timestamp": ts,
                "dir_name": name,
                "max_samples": max_samples,
                "standard": rows[0],
                "weighted": rows[1],
                "weighted_prob": rows[2],
            }
        )

    runs.sort(key=lambda r: r["timestamp"])
    return runs


def limit_runs_per_max_samples(runs: List[dict], max_runs: Optional[int]) -> List[dict]:
    if max_runs is None:
        return runs
    if max_runs <= 0:
        raise ValueError("--max-runs must be a positive integer")

    grouped: Dict[int, List[dict]] = defaultdict(list)
    # Use -1 as bucket if MAX_SAMPLES unavailable in a run.
    for run in runs:
        grouped[int(run["max_samples"]) if run["max_samples"] is not None else -1].append(run)

    selected: List[dict] = []
    for key in sorted(grouped.keys()):
        bucket = sorted(grouped[key], key=lambda r: r["timestamp"])
        selected.extend(bucket[:max_runs])

    selected.sort(key=lambda r: r["timestamp"])
    return selected


def write_tables(out_path: str, start_str: str, end_str: str, runs: List[dict]) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Extracted METHOD/P@100R/R@100P tables from results folder\n")
        f.write(f"Time window: {start_str} -> {end_str}\n")
        f.write(f"Tables found: {len(runs)}\n\n")

        for run in runs:
            ts = run["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] {run['dir_name']}\n")
            f.write("=" * 95 + "\n")
            f.write("METHOD                         | ACCURACY   | P@100R       | R@100P     | AUC-PR    \n")
            f.write("-" * 95 + "\n")
            f.write(run["standard"] + "\n")
            f.write(run["weighted"] + "\n")
            f.write(run["weighted_prob"] + "\n")
            f.write("=" * 95 + "\n\n")


def main() -> None:
    args = parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S")
    if end_dt < start_dt:
        raise ValueError("--end must be >= --start")

    out_txt = args.out_txt
    if out_txt is None:
        out_txt = os.path.join(
            args.results_root,
            f"tables_{sanitize_ts(args.start)}_to_{sanitize_ts(args.end)}.txt",
        )

    runs = collect_tables(args.results_root, start_dt, end_dt)
    runs = limit_runs_per_max_samples(runs, args.max_runs)
    write_tables(out_txt, args.start, args.end, runs)

    print(f"Wrote: {out_txt}")
    print(f"Tables found: {len(runs)}")


if __name__ == "__main__":
    main()
