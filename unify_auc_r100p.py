#!/usr/bin/env python3
"""Unified workflow to produce AUC-PR and R@100P artifacts for one time window.

This script orchestrates:
1) extract_and_plot_auc.py
2) extract_tables_by_time.py
3) plot_r100p_from_extracted_tables.py

It can also produce a side-by-side combined figure from the two generated plots.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unified AUC-PR + R@100P plotting pipeline for a time window"
    )
    parser.add_argument("--results-root", default="results", help="Results root directory")
    parser.add_argument("--start", required=True, help="Start timestamp (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--end", required=True, help="End timestamp (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument(
        "--tries",
        type=int,
        default=10,
        help="Keep first N runs per MAX_SAMPLES (used by AUC and table extraction)",
    )

    parser.add_argument("--auc-csv", default="auc_values.csv", help="AUC CSV basename")
    parser.add_argument("--auc-txt", default="auc_summary.txt", help="AUC summary basename")
    parser.add_argument("--auc-plot", default="auc_pr_vs_places.png", help="AUC plot basename")

    parser.add_argument(
        "--tables-txt",
        default=None,
        help="Optional full path for extracted tables txt; defaults to results/tables_<start>_to_<end>.txt",
    )
    parser.add_argument(
        "--r100p-plot",
        default="r100p_boxplot_from_extracted_tables.png",
        help="R@100P boxplot basename",
    )

    parser.add_argument(
        "--combined-plot",
        default="unified_auc_r100p.png",
        help="Combined side-by-side figure basename",
    )
    parser.add_argument(
        "--skip-combined",
        action="store_true",
        help="Skip creating combined side-by-side figure",
    )
    parser.add_argument(
        "--strict-params",
        action="store_true",
        help="Pass through strict hyperparameter consistency check to AUC script",
    )

    return parser.parse_args()


def sanitize_ts(ts: str) -> str:
    return ts.replace(" ", "_").replace(":", "-")


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _infer_output_dir_from_table(table_file: str, tries: int) -> str | None:
    # Reuse existing helper from plot script to keep folder naming consistent.
    import plot_r100p_from_extracted_tables as r100p_plot

    return r100p_plot._infer_output_dir(table_file, tries_override=tries)


def create_combined_plot(auc_plot: str, r100p_plot: str, combined_out: str) -> None:
    auc_img = mpimg.imread(auc_plot)
    r100p_img = mpimg.imread(r100p_plot)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    axes[0].imshow(auc_img)
    axes[0].set_title("AUC PR vs Number of Places")
    axes[0].axis("off")

    axes[1].imshow(r100p_img)
    axes[1].set_title("R@100P Boxplot")
    axes[1].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(combined_out) or ".", exist_ok=True)
    plt.savefig(combined_out)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S")
    if end_dt < start_dt:
        raise ValueError("--end must be >= --start")
    if args.tries <= 0:
        raise ValueError("--tries must be a positive integer")

    tables_txt = args.tables_txt
    if tables_txt is None:
        tables_txt = os.path.join(
            args.results_root,
            f"tables_{sanitize_ts(args.start)}_to_{sanitize_ts(args.end)}.txt",
        )

    # 1) AUC extraction + plot
    auc_cmd = [
        sys.executable,
        "extract_and_plot_auc.py",
        "--results-root",
        args.results_root,
        "--start",
        args.start,
        "--end",
        args.end,
        "--tries",
        str(args.tries),
        "--out-csv",
        args.auc_csv,
        "--out-txt",
        args.auc_txt,
        "--out-plot",
        args.auc_plot,
    ]
    if args.strict_params:
        auc_cmd.append("--strict-params")
    run_cmd(auc_cmd)

    # 2) Extract tables with same selection policy
    run_cmd(
        [
            sys.executable,
            "extract_tables_by_time.py",
            "--results-root",
            args.results_root,
            "--start",
            args.start,
            "--end",
            args.end,
            "--max-runs",
            str(args.tries),
            "--out-txt",
            tables_txt,
        ]
    )

    # 3) R@100P boxplot from extracted tables
    run_cmd(
        [
            sys.executable,
            "plot_r100p_from_extracted_tables.py",
            "--in-file",
            tables_txt,
            "--tries",
            str(args.tries),
            "--out",
            args.r100p_plot,
        ]
    )

    # 4) Combined figure (optional)
    output_dir = _infer_output_dir_from_table(tables_txt, args.tries)
    if output_dir is None:
        output_dir = "."

    auc_plot_path = os.path.join(output_dir, os.path.basename(args.auc_plot))
    r100p_plot_path = os.path.join(output_dir, os.path.basename(args.r100p_plot))
    combined_out = os.path.join(output_dir, os.path.basename(args.combined_plot))

    if not args.skip_combined:
        if os.path.isfile(auc_plot_path) and os.path.isfile(r100p_plot_path):
            create_combined_plot(auc_plot_path, r100p_plot_path, combined_out)
            print(f"Saved combined plot: {combined_out}")
        else:
            print("Skipped combined plot: source plots not found.")

    print("Unified workflow complete.")
    print(f"Tables file: {tables_txt}")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()
