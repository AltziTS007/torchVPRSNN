#!/usr/bin/env python3
"""
汇总消融实验结果，生成对比表格
"""
import os
import re
from pathlib import Path

# 消融实验目录映射
ABLATION_DIRS = {
    "Baseline": "nordland_100_2026-09-02_17-41-28",
    "no_wta (关闭 WTA)": "ablation_no_wta_2026-09-04_12-25-12",
    "no_homeo (关闭稳态)": "ablation_no_homeo_2026-09-04_13-32-59",
    "no_norm (关闭归一化)": "ablation_no_norm_2026-09-04_13-34-42",
    "spillover (模拟溢出)": "ablation_spillover_2026-09-04_13-36-02",
}

RESULTS_BASE = Path("E:\\QQDownloads\\牛马\\snn\\torchVPRSNN\\results")


def extract_table_metrics(log_text):
    """从最终汇总表中提取四种方法的指标"""
    metrics = {}

    # 匹配表格中每行: Standard | xx% | xx% | xx% | xx% | N
    table_pattern = re.compile(
        r"(Standard|Weighted\+Prob|Weighted|SeqAgg\(k=5,product\))\s*\|\s*"
        r"([\d.]+)\s*%\s*\|\s*([\d.]+)\s*%\s*\|\s*([\d.]+)\s*%\s*\|\s*([\d.]+)\s*%\s*\|\s*(\d+)"
    )
    for m in table_pattern.finditer(log_text):
        method = m.group(1)
        metrics[method] = {
            "Accuracy": float(m.group(2)),
            "P@100R": float(m.group(3)),
            "R@100P": float(m.group(4)),
            "AUC-PR": float(m.group(5)),
            "N": int(m.group(6)),
        }

    return metrics


def extract_max_recall(log_text, method_label):
    """从 Probability-Based Metrics 提取 Max Recall（k=1 时的最佳单帧）"""
    # 匹配 Max Recall: xx%, P@MaxR: xx% 行
    pattern = re.compile(r"Max Recall: ([\d.]+)%")
    matches = pattern.findall(log_text)
    if matches:
        return float(matches[0])  # 第一个 Max Recall 是单帧评估
    return None


def extract_k5_product(log_text):
    """从 aggregation sweep 中提取 k=5, product rule 的 R@100P"""
    # 匹配 k= 5 行后的 R@100P
    pattern = re.compile(r"k=\s*5\s*\(N=(\d+)\):\s*R@100P\s*=\s*([\d.]+)%")
    m = pattern.search(log_text)
    if m:
        return float(m.group(2))
    return None


def main():
    print("=" * 100)
    print("消融实验对比汇总")
    print("=" * 100)

    all_results = {}
    for label, dirname in ABLATION_DIRS.items():
        log_path = RESULTS_BASE / dirname / "output.log"
        if not log_path.exists():
            print(f"[WARNING] 未找到日志: {log_path}")
            continue

        text = log_path.read_text(encoding="utf-8")
        metrics = extract_table_metrics(text)
        all_results[label] = metrics

    # 打印详细表格
    print("\n详细指标对比")
    print("-" * 100)
    header = f"{'实验配置':<28} | {'Accuracy':>8} | {'W+Prob Acc':>10} | {'W+Prob R@100P':>13} | {'SeqAgg(k=5)':>11} | {'N':>4}"
    print(header)
    print("-" * 100)

    for label, metrics in all_results.items():
        std = metrics.get("Standard", {})
        wp = metrics.get("Weighted+Prob", {})
        seq = metrics.get("SeqAgg(k=5,product)", {})
        n = seq.get("N", wp.get("N", 0))

        print(
            f"{label:<28} | {std.get('Accuracy', 'N/A'):>7.1f}% | {wp.get('Accuracy', 'N/A'):>9.1f}% "
            f"| {wp.get('R@100P', 'N/A'):>12.1f}% | {seq.get('R@100P', 'N/A'):>10.1f}% | {n:>4}"
        )

    # 打印消融效果总结
    print("\n" + "=" * 100)
    print("消融效果总结（W+Prob R@100P 对比）")
    print("=" * 100)

    baseline_wp_r100p = all_results.get("Baseline", {}).get("Weighted+Prob", {}).get("R@100P")
    if baseline_wp_r100p is not None:
        print(f"\nBaseline W+Prob R@100P: {baseline_wp_r100p:.1f}%\n")
        for label in list(ABLATION_DIRS.keys())[1:]:
            wp_r100p = all_results.get(label, {}).get("Weighted+Prob", {}).get("R@100P")
            if wp_r100p is not None:
                delta = wp_r100p - baseline_wp_r100p
                sign = "+" if delta >= 0 else ""
                print(f"  {label:<28}: {wp_r100p:>6.1f}%  ({sign}{delta:.1f}%)")

    # 保存 CSV
    import csv
    csv_path = RESULTS_BASE / "ablation_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Experiment", "Standard_Acc", "Standard_R100P",
                         "Weighted_Acc", "Weighted_R100P",
                         "W+Prob_Acc", "W+Prob_R100P", "W+Prob_AUC",
                         "SeqAgg5_Acc", "SeqAgg5_R100P", "SeqAgg5_AUC", "N"])
        for label, metrics in all_results.items():
            std = metrics.get("Standard", {})
            wt = metrics.get("Weighted", {})
            wp = metrics.get("Weighted+Prob", {})
            seq = metrics.get("SeqAgg(k=5,product)", {})
            writer.writerow([
                label,
                std.get("Accuracy", ""), std.get("R@100P", ""),
                wt.get("Accuracy", ""), wt.get("R@100P", ""),
                wp.get("Accuracy", ""), wp.get("R@100P", ""), wp.get("AUC-PR", ""),
                seq.get("Accuracy", ""), seq.get("R@100P", ""), seq.get("AUC-PR", ""),
                seq.get("N", wp.get("N", "")),
            ])
    print(f"\nCSV 已保存: {csv_path}")


if __name__ == "__main__":
    main()
