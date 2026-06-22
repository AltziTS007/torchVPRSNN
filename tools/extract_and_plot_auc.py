import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

#!/usr/bin/env python3
"""Extract AUC from Weighted+Prob PR images and plot AUC vs MAX_SAMPLES.

This script combines:
- OCR extraction from Weighted+Prob/pr_curve.png
- consistency checking of hyperparameters (only MAX_SAMPLES may differ)
- summary CSV/TXT export
- mean+std errorbar plot generation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


DIR_RE = re.compile(r"^results_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})$")
AUC_RE = re.compile(r"AUC\s*[:=]?\s*([01](?:\.\d+)?)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract AUC from results images and create the final AUC plot"
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Path to results root directory",
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
        "--tries",
        type=int,
        default=10,
        help="How many earliest runs to keep per MAX_SAMPLES",
    )
    parser.add_argument(
        "--out-csv",
        default="auc_values.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--out-txt",
        default="auc_summary.txt",
        help="Output text summary path",
    )
    parser.add_argument(
        "--out-plot",
        default="auc_pr_vs_places.png",
        help="Output plot image path",
    )
    parser.add_argument(
        "--weighted-dir",
        default="Weighted+Prob",
        help="Folder name inside each run containing pr_curve.png",
    )
    parser.add_argument(
        "--pr-image",
        default="pr_curve.png",
        help="PR curve image filename",
    )
    parser.add_argument(
        "--strict-params",
        action="store_true",
        help="Fail if any selected run differs in hyperparameters other than MAX_SAMPLES",
    )
    return parser.parse_args()


def _parse_dir_timestamp(name: str) -> Optional[datetime]:
    m = DIR_RE.match(name)
    if not m:
        return None
    return datetime.strptime(
        f"{m.group(1)} {m.group(2).replace('-', ':')}", "%Y-%m-%d %H:%M:%S"
    )


def _load_hparams(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _normalize_params_for_compare(hparams: dict) -> dict:
    normalized = dict(hparams)
    normalized.pop("MAX_SAMPLES", None)
    return normalized


def _find_param_mismatches(selected_runs: List[dict]) -> List[str]:
    if not selected_runs:
        return []

    first = selected_runs[0]
    baseline = _normalize_params_for_compare(first["hparams"])
    mismatch_lines: List[str] = []

    for run in selected_runs[1:]:
        current = _normalize_params_for_compare(run["hparams"])
        if current == baseline:
            continue

        keys = sorted(set(baseline.keys()) | set(current.keys()))
        diffs = [k for k in keys if baseline.get(k) != current.get(k)]
        mismatch_lines.append(
            f"{run['timestamp']} {run['dir_name']} differs in keys: {', '.join(diffs)}"
        )

    return mismatch_lines


def _ocr_auc_from_image(image_path: str) -> Tuple[Optional[float], str]:
    try:
        from PIL import Image
    except Exception as e:
        return None, f"PIL unavailable: {e}"

    try:
        import pytesseract
    except Exception as e:
        return None, f"pytesseract unavailable: {e}"

    try:
        img = Image.open(image_path)
    except Exception as e:
        return None, f"cannot open image: {e}"

    passes = [
        {"config": "--psm 6", "transform": "raw"},
        {"config": "--psm 11", "transform": "raw"},
        {"config": "--psm 6", "transform": "grayscale"},
    ]

    for p in passes:
        work = img.convert("L") if p["transform"] == "grayscale" else img
        try:
            text = pytesseract.image_to_string(work, config=p["config"])
        except Exception as e:
            return None, f"tesseract execution failed: {e}"

        m = AUC_RE.search(text)
        if m:
            try:
                return float(m.group(1)), "ok"
            except ValueError:
                pass

    return None, "AUC text not found"


def _mean_std(values: List[float]) -> Tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def _plot_from_rows(rows: List[dict], out_plot: str) -> None:
    grouped: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        if row["status"] != "ok" or not row["auc"]:
            continue
        grouped[int(row["max_samples"])].append(float(row["auc"]))

    if not grouped:
        print("No valid AUC values found. Plot was not generated.")
        return

    x = sorted(grouped.keys())
    y = []
    yerr = []
    for ms in x:
        m, s = _mean_std(grouped[ms])
        y.append(m)
        yerr.append(s)

    fig, ax = plt.subplots(figsize=(7.68, 4.61), dpi=150)
    fig.patch.set_facecolor("#d9d9d9")
    ax.set_facecolor("#d9d9d9")

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="-o",
        color="#1f77b4",
        ecolor="#ff7f0e",
        elinewidth=2,
        capsize=4,
        capthick=1.5,
        markersize=4,
        linewidth=2,
    )

    ax.set_xlabel("Number of places", fontsize=20)
    ax.set_ylabel("AUC PR", fontsize=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(min(x) - 25, max(x) + 25)
    ax.tick_params(axis="both", labelsize=16, width=1.2, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_plot) or ".", exist_ok=True)
    plt.savefig(out_plot)
    print(f"Saved plot: {out_plot}")


def main() -> None:
    args = parse_args()
    if args.tries <= 0:
        raise ValueError("--tries must be a positive integer")

    start_dt = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S")
    if end_dt < start_dt:
        raise ValueError("--end must be >= --start")

    all_runs: List[dict] = []
    for name in sorted(os.listdir(args.results_root)):
        ts = _parse_dir_timestamp(name)
        if ts is None or not (start_dt <= ts <= end_dt):
            continue

        run_dir = os.path.join(args.results_root, name)
        hparams_path = os.path.join(run_dir, "hyperparameters.json")
        hparams = _load_hparams(hparams_path)
        if not hparams or "MAX_SAMPLES" not in hparams:
            continue

        all_runs.append(
            {
                "timestamp": ts,
                "dir_name": name,
                "run_dir": run_dir,
                "max_samples": int(hparams["MAX_SAMPLES"]),
                "hparams": hparams,
            }
        )

    grouped: Dict[int, List[dict]] = defaultdict(list)
    for run in all_runs:
        grouped[run["max_samples"]].append(run)
    for ms in grouped:
        grouped[ms].sort(key=lambda r: r["timestamp"])

    selected_runs: List[dict] = []
    for ms in sorted(grouped.keys()):
        chosen = grouped[ms][: args.tries]
        for idx, run in enumerate(chosen, start=1):
            run["run_index_within_max_samples"] = idx
            selected_runs.append(run)

    selected_runs.sort(key=lambda r: r["timestamp"])

    if not selected_runs:
        raise RuntimeError("No runs found in the selected time window.")

    # Build output folder name from baseline hyperparameters and tries:
    # <N_EXC>-<BATCH_SIZE>-<EPOCHS>-<t_steps>-<tries>
    baseline_hparams = selected_runs[0]["hparams"]
    n_exc = baseline_hparams.get("N_EXC", "na")
    batch_size = baseline_hparams.get("BATCH_SIZE", "na")
    epochs = baseline_hparams.get("EPOCHS", "na")
    t_steps = baseline_hparams.get("t_steps", "na")
    run_folder_name = f"{n_exc}-{batch_size}-{epochs}-{t_steps}-{args.tries}"
    output_dir = os.path.join(args.results_root, run_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Keep CLI compatibility, but force artifacts inside the derived output folder.
    out_csv_path = os.path.join(output_dir, os.path.basename(args.out_csv))
    out_txt_path = os.path.join(output_dir, os.path.basename(args.out_txt))
    out_plot_path = os.path.join(output_dir, os.path.basename(args.out_plot))

    mismatch_lines = _find_param_mismatches(selected_runs)
    if mismatch_lines and args.strict_params:
        raise RuntimeError(
            "Hyperparameter mismatch detected beyond MAX_SAMPLES. "
            "Re-run without --strict-params to continue and report mismatches."
        )

    rows: List[dict] = []
    for run in selected_runs:
        image_path = os.path.join(run["run_dir"], args.weighted_dir, args.pr_image)
        if not os.path.isfile(image_path):
            rows.append(
                {
                    "timestamp": run["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "results_dir": run["run_dir"],
                    "max_samples": run["max_samples"],
                    "run_index_within_max_samples": run["run_index_within_max_samples"],
                    "auc": "",
                    "status": "missing_image",
                }
            )
            continue

        auc, status = _ocr_auc_from_image(image_path)
        rows.append(
            {
                "timestamp": run["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "results_dir": run["run_dir"],
                "max_samples": run["max_samples"],
                "run_index_within_max_samples": run["run_index_within_max_samples"],
                "auc": "" if auc is None else f"{auc:.6f}",
                "status": status,
            }
        )

    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "results_dir",
                "max_samples",
                "run_index_within_max_samples",
                "auc",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("Combined AUC extraction + plotting summary\n")
        f.write(f"window: {start_dt} -> {end_dt}\n")
        f.write(f"tries per MAX_SAMPLES: {args.tries}\n")
        f.write(f"selected runs: {len(selected_runs)}\n")
        f.write(f"output_dir: {output_dir}\n")
        f.write(f"csv: {out_csv_path}\n")
        f.write(f"plot: {out_plot_path}\n\n")

        if mismatch_lines:
            f.write("Parameter consistency check: MISMATCH FOUND\n")
            f.write("Only MAX_SAMPLES should differ, but these runs differ on other keys:\n")
            for line in mismatch_lines:
                f.write(f"- {line}\n")
            f.write("\n")
        else:
            f.write("Parameter consistency check: OK (only MAX_SAMPLES differs)\n\n")

        by_ms: Dict[int, List[dict]] = defaultdict(list)
        for row in rows:
            by_ms[int(row["max_samples"])].append(row)

        for ms in sorted(by_ms):
            subset = by_ms[ms]
            ok_vals = [float(r["auc"]) for r in subset if r["status"] == "ok" and r["auc"]]
            f.write(f"MAX_SAMPLES={ms}: selected={len(subset)}, extracted={len(ok_vals)}\n")
            if ok_vals:
                m, s = _mean_std(ok_vals)
                f.write(f"  mean_auc={m:.6f}, std_auc={s:.6f}\n")
            failures = [r for r in subset if r["status"] != "ok"]
            if failures:
                f.write("  failures:\n")
                for r in failures:
                    f.write(
                        f"    run#{r['run_index_within_max_samples']} {r['timestamp']} status={r['status']}\n"
                    )
            f.write("\n")

    manifest_path = os.path.join(output_dir, "results_manifest.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("Result files in this folder\n")
        f.write("- auc_values.csv: per-run extracted AUC values and statuses\n")
        f.write("- auc_summary.txt: run-window summary, parameter-check status, per-MAX_SAMPLES stats\n")
        f.write("- auc_pr_vs_places.png: mean AUC vs MAX_SAMPLES with std error bars\n")

    _plot_from_rows(rows, out_plot_path)

    print(f"Output directory: {output_dir}")
    print(f"Wrote CSV: {out_csv_path}")
    print(f"Wrote TXT: {out_txt_path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
