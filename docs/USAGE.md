# Usage / Quickstart

This quickstart covers the most common flows: training, evaluation, and visualization.

1) Prepare the Nordland dataset

- Place datasets into `nordland_clean/data/<season>` directories. Example:
  - `nordland_clean/data/spring`
  - `nordland_clean/data/fall`
  - `nordland_clean/data/summer`

2) Activate environment

```bash
conda activate vpr_snn
```

3) Run the end-to-end experiment

```bash
python core/snn_model.py
```

This runs an unsupervised STDP training loop and evaluation. By default the script will:
- Create a timestamped `results/` subdirectory
- Save `hyperparameters.json` and `output.log`
- Store weight visualizations and STDP monitor plots

4) Quick configuration

- Open `snn_model.py` and edit the `params` dictionary near the top of `main()` for:
  - `EPOCHS`, `BATCH_SIZE`, `DEVICE`, `N_IN`, `N_EXC`, `t_steps`, etc.

5) Running on GPU

- Ensure `torch.cuda.is_available()` returns `True` and `params['DEVICE']` will be set automatically to `cuda`.

6) Inspecting results

- The results tree will contain method-specific subfolders (e.g., `Standard/`, `Weighted/`) with plots:
  - `distance_matrix.png`
  - `pr_curve.png`
  - `recall_at_n.png`
  - qualitative result images

7) Reproducing a saved run

- Each run saves `hyperparameters.json` and weight images. To reproduce, load the saved weights and re-run the evaluation utilities in `vprsnn_evaluation.py`.

8) Helpful scripts

- `verify_refactor.py` contains minimal checks and example usage of `RateEncoder` and `torchVPRSNN`.

9) Post-run analysis scripts

- `extract_tables_by_time.py`
  - Extracts METHOD/P@100R/R@100P summary rows from `results_*/output.log`
    within a specific timestamp window.
  - Output format is compatible with `plot_r100p_from_extracted_tables.py`.
  - If `--out-txt` is omitted, it writes to:
    - `results/tables_<start>_to_<end>.txt`
  - Example:

```bash
python tools/extract_tables_by_time.py \
  --start "2026-04-01 14:12:24" \
  --end "2026-04-02 02:49:35"
```

- `plot_r100p_from_extracted_tables.py`
  - Recreates the R@100P boxplot from an extracted tables text file.
  - Uses the same output-folder logic as `extract_and_plot_auc.py`:
    - saves into `results/<N_EXC>-<BATCH_SIZE>-<EPOCHS>-<t_steps>-<tries>/`
    - `tries` is inferred from the table file run count (or set explicitly with `--tries`).
  - Example:

```bash
python tools/plot_r100p_from_extracted_tables.py \
  --in-file results/tables_2026-03-26_22-40-59_to_2026-03-27_00-36-08.txt \
  --out r100p_boxplot.png
```

  - Typical two-step workflow:

```bash
python tools/extract_tables_by_time.py \
  --start "2026-04-01 14:12:24" \
  --end "2026-04-02 02:49:35"

python tools/plot_r100p_from_extracted_tables.py \
  --in-file results/tables_2026-04-01_14-12-24_to_2026-04-02_02-49-35.txt \
  --out r100p_boxplot_from_extracted_tables.png
```

  - Files written in the output folder:
    - `r100p_boxplot.png`
    - `r100p_summary.txt`
    - `results_manifest.txt`

- `extract_and_plot_auc.py`
  - End-to-end utility that:
    - selects runs in a time window,
    - keeps the first `N` tries per `MAX_SAMPLES`,
    - OCR-extracts AUC from `Weighted+Prob/pr_curve.png`,
    - checks that hyperparameters match except `MAX_SAMPLES`,
    - writes CSV/TXT summaries and the final AUC-vs-places figure.
  - Output folder:
    - `results/<N_EXC>-<BATCH_SIZE>-<EPOCHS>-<t_steps>-<tries>/`
    - output filenames come from `--out-csv`, `--out-txt`, `--out-plot` basenames.
  - Example:

```bash
python tools/extract_and_plot_auc.py \
  --start "2026-03-26 19:58:55" \
  --end "2026-03-27 21:44:03" \
  --tries 10 \
  --out-csv auc_values.csv \
  --out-txt auc_summary.txt \
  --out-plot auc_pr_vs_places.png
```

  - Files written in the output folder:
    - `auc_values.csv`
    - `auc_summary.txt`
    - `auc_pr_vs_places.png`
    - `results_manifest.txt`

- `unify_auc_r100p.py`
  - One-command workflow that unifies:
    - AUC extraction + AUC-vs-places plot
    - table extraction for METHOD/P@100R/R@100P
    - R@100P boxplot generation
    - optional side-by-side combined figure
  - Uses the same `--tries` policy for both AUC and R@100P so both views are aligned.
  - Example:

```bash
python tools/unify_auc_r100p.py \
  --start "2026-04-01 14:12:24" \
  --end "2026-04-02 02:49:35" \
  --tries 10
```

  - Files written in `results/<N_EXC>-<BATCH_SIZE>-<EPOCHS>-<t_steps>-<tries>/`:
    - `auc_values.csv`
    - `auc_summary.txt`
    - `auc_pr_vs_places.png`
    - `r100p_boxplot_from_extracted_tables.png`
    - `unified_auc_r100p.png` (unless `--skip-combined`)

Notes:
- `extract_and_plot_auc.py` uses OCR and requires `pytesseract` and a system `tesseract-ocr` installation.
- Use `--strict-params` to fail fast if parameters differ across runs (other than `MAX_SAMPLES`).

If you want a targeted example (e.g., only evaluate without training), ask and a short utility script can be added to `scripts/`.
