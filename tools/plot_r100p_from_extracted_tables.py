import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

#!/usr/bin/env python3
"""Recreate R@100P boxplot from extracted summary tables file."""

import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt


ROW_RE = re.compile(
    r"^(Standard|Weighted|Weighted\+Prob)\s*\|\s*([0-9]+\.[0-9]+)\s*%\s*\|\s*([0-9]+\.[0-9]+)\s*%\s*\|\s*([0-9]+\.[0-9]+)\s*%\s*\|\s*([0-9]+\.[0-9]+)\s*%",
    re.MULTILINE,
)


def load_values(table_file: str):
    with open(table_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    values = defaultdict(list)
    for method, acc, p100r, r100p, auc in ROW_RE.findall(text):
        # File stores percentages (e.g., 58.51), plot expects 0-1 like the reference image.
        values[method].append(float(r100p) / 100.0)

    return values


def _infer_output_dir(table_file: str, tries_override=None):
    """Infer results output folder: results/<N_EXC>-<BATCH_SIZE>-<EPOCHS>-<t_steps>-<tries>."""
    with open(table_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    run_dirs = re.findall(r"\]\s+(results_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", text)
    if not run_dirs:
        return None

    # Preserve order while removing duplicates.
    uniq = list(dict.fromkeys(run_dirs))

    # Infer tries from table entries when not provided.
    tries = tries_override if tries_override is not None else len(uniq)

    # Try resolving hyperparameters.json near the table file first.
    candidate_paths = [
        os.path.join("results", uniq[0], "hyperparameters.json"),
        os.path.join(os.path.dirname(table_file), uniq[0], "hyperparameters.json"),
    ]

    hparams = None
    for p in candidate_paths:
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as hf:
                hparams = json.load(hf)
            break

    if not hparams:
        return None

    n_exc = hparams.get("N_EXC", "na")
    batch_size = hparams.get("BATCH_SIZE", "na")
    epochs = hparams.get("EPOCHS", "na")
    t_steps = hparams.get("t_steps", "na")
    folder = f"{n_exc}-{batch_size}-{epochs}-{t_steps}-{tries}"
    return os.path.join("results", folder)


def main():
    parser = argparse.ArgumentParser(description="Plot R@100P from extracted table file")
    parser.add_argument("--in-file", required=True, help="Path to extracted tables text file")
    parser.add_argument(
        "--out",
        default="r100p_boxplot.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--tries",
        type=int,
        default=None,
        help="Tries value for output folder naming (defaults to inferred run count in table file).",
    )
    args = parser.parse_args()

    values = load_values(args.in_file)

    ordered_methods = ["Standard", "Weighted", "Weighted+Prob"]
    labels = [m for m in ordered_methods if values.get(m)]
    data = [values[m] for m in labels]

    if not data:
        raise RuntimeError("No R@100P rows found in input table file.")

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=150)
    bp = ax.boxplot(data, patch_artist=True, tick_labels=labels, showfliers=True)

    # Color palette aligned with your reference image feel.
    color_map = {
        "Standard": "#1f77b4",
        "Weighted": "#ff7f0e",
        "Weighted+Prob": "#d62728",
    }

    for patch, lbl in zip(bp["boxes"], labels):
        patch.set_facecolor(color_map.get(lbl, "#888888"))
        patch.set_alpha(0.95)
        patch.set_linewidth(0.9)

    for key in ("whiskers", "caps", "medians"):
        for artist in bp[key]:
            artist.set_color("black")
            artist.set_linewidth(0.8)

    ax.set_ylabel("R@100P", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_facecolor("#f2f2f2")
    ax.grid(axis="y", alpha=0.25, linestyle="-")

    # Tilt and color x labels to mimic the provided chart style.
    label_colors = {
        "Standard": "#1f77b4",
        "Weighted": "#ff7f0e",
        "Weighted+Prob": "#d62728",
    }
    for tick in ax.get_xticklabels():
        txt = tick.get_text()
        tick.set_rotation(25)
        tick.set_ha("right")
        tick.set_color(label_colors.get(txt, "black"))
        tick.set_fontsize(11)

    output_dir = _infer_output_dir(args.in_file, tries_override=args.tries)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, os.path.basename(args.out))
    else:
        out_path = args.out

    plt.tight_layout()
    plt.savefig(out_path)

    out_dir = os.path.dirname(out_path) or "."
    summary_path = os.path.join(out_dir, "r100p_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("R@100P boxplot summary\n")
        f.write(f"source_table_file: {args.in_file}\n")
        f.write(f"plot_file: {out_path}\n")
        for m in labels:
            vals = values[m]
            mean_v = sum(vals) / len(vals)
            f.write(f"{m}: n={len(vals)}, mean={mean_v:.4f}\n")

    manifest_path = os.path.join(out_dir, "results_manifest.txt")
    if not os.path.exists(manifest_path):
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write("Result files in this folder\n")
            f.write("- r100p_boxplot.png: R@100P distribution boxplot\n")
            f.write("- r100p_summary.txt: source table and per-method summary stats\n")

    print(f"Saved plot to: {out_path}")
    print(f"Saved summary to: {summary_path}")
    for m in labels:
        vals = values[m]
        mean_v = sum(vals) / len(vals)
        print(f"{m}: n={len(vals)}, mean={mean_v:.4f}")


if __name__ == "__main__":
    main()
