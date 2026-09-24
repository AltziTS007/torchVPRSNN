#!/usr/bin/env python3
"""
汇总多种子实验结果，计算均值和标准差，与论文对比
"""

import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path

def extract_metrics_from_log(log_path):
    """从 output.log 中提取关键指标"""
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    metrics = {}
    
    # 提取准确率
    acc_match = re.search(r'Accuracy: ([\d.]+)%', content)
    if acc_match:
        metrics['Accuracy'] = float(acc_match.group(1))
    
    # 提取 R@100P (Raw)
    r100p_match = re.search(r'R@100P \(Raw\): ([\d.]+)%', content)
    if r100p_match:
        metrics['R@100P_Raw'] = float(r100p_match.group(1))
    
    # 提取 Probability-Based Metrics
    prob_match = re.search(r'--- Probability-Based Metrics ---\n.*?Max Recall: ([\d.]+)%, P@MaxR: ([\d.]+)%', 
                          content, re.DOTALL)
    if prob_match:
        metrics['MaxRecall_Prob'] = float(prob_match.group(1))
        metrics['P@MaxR_Prob'] = float(prob_match.group(2))
    
    # 从最终汇总表提取三种分配策略的 R@100P
    # 汇总表格式: METHOD | ACCURACY | P@100R | R@100P | AUC-PR | N
    std_match = re.search(r'Standard\s+\|\s+([\d.]+)\s*%\s*\|\s*([\d.]+)\s*%\s*\|\s*([\d.]+)\s*%\s*\|\s*([\d.]+)\s*%\s*\|\s*\d+', content)
    if std_match:
        metrics['Standard_Acc'] = float(std_match.group(1))
        metrics['Standard_P100R'] = float(std_match.group(2))
        metrics['Standard_R100P'] = float(std_match.group(3))
        metrics['Standard_AUC'] = float(std_match.group(4))
    
    wt_match = re.search(r'Weighted\s+\|\s+([\d.]+)\s*%\s*\|\s*([\d.]+)\s*%\s*\|\s*([\d.]+)\s*%\s*\|\s*([\d.]+)\s*%\s*\|\s*\d+', content)
    if wt_match:
        metrics['Weighted_Acc'] = float(wt_match.group(1))
        metrics['Weighted_P100R'] = float(wt_match.group(2))
        metrics['Weighted_R100P'] = float(wt_match.group(3))
        metrics['Weighted_AUC'] = float(wt_match.group(4))
    
    wp_match = re.search(r'Weighted\+Prob\s+\|\s+([\d.]+)\s*%\s*\|\s*([\d.]+)\s*%\s*\|\s*([\d.]+)\s*%\s*\|\s*([\d.]+)\s*%\s*\|\s*\d+', content)
    if wp_match:
        metrics['WeightedProb_Acc'] = float(wp_match.group(1))
        metrics['WeightedProb_P100R'] = float(wp_match.group(2))
        metrics['WeightedProb_R100P'] = float(wp_match.group(3))
        metrics['WeightedProb_AUC'] = float(wp_match.group(4))
    
    seq_match = re.search(r'SeqAgg\(k=5,product\)\s+\|\s+([\d.]+)\s*%\s*\|\s*([\d.]+)\s*%\s*\|\s*([\d.]+)\s*%\s*\|\s*([\d.]+)\s*%\s*\|\s*\d+', content)
    if seq_match:
        metrics['SeqAgg5_Acc'] = float(seq_match.group(1))
        metrics['SeqAgg5_P100R'] = float(seq_match.group(2))
        metrics['SeqAgg5_R100P'] = float(seq_match.group(3))
        metrics['SeqAgg5_AUC'] = float(seq_match.group(4))
    
    # 提取 SeqAgg sweep 中 k=5 的 R@100P
    sweep_match = re.search(r'k=\s*5 \(N=\d+\): R@100P =\s+([\d.]+)%', content)
    if sweep_match:
        metrics['Sweep_k5_R100P'] = float(sweep_match.group(1))
    
    return metrics

def main():
    results_dir = Path('results')
    
    # 找到所有 seed_* 目录
    seed_dirs = sorted([d for d in results_dir.iterdir() if d.name.startswith('seed_')])
    
    print(f"找到 {len(seed_dirs)} 个 seed 实验目录\n")
    
    all_metrics = []
    seeds = []
    
    for seed_dir in seed_dirs:
        log_path = seed_dir / 'output.log'
        if not log_path.exists():
            print(f"警告: {seed_dir.name} 缺少 output.log，跳过")
            continue
        
        # 提取 seed 编号
        seed_num = int(seed_dir.name.split('_')[1])
        seeds.append(seed_num)
        
        # 提取指标
        metrics = extract_metrics_from_log(log_path)
        metrics['seed'] = seed_num
        all_metrics.append(metrics)
        
        print(f"Seed {seed_num}: Acc={metrics.get('Accuracy', float('nan')):.2f}%, "
              f"Std={metrics.get('Standard_R100P', float('nan')):.2f}%, "
              f"Wt={metrics.get('Weighted_R100P', float('nan')):.2f}%, "
              f"W+P={metrics.get('WeightedProb_R100P', float('nan')):.2f}%, "
              f"Seq5={metrics.get('SeqAgg5_R100P', float('nan')):.2f}%")
    
    if not all_metrics:
        print("没有提取到任何数据")
        return
    
    # 转换为 DataFrame
    df = pd.DataFrame(all_metrics)
    
    print("\n" + "="*80)
    print("统计汇总 (15 seeds)")
    print("="*80)
    
    # 计算均值和标准差
    cols = ['Accuracy', 'Standard_R100P', 'Weighted_R100P', 'WeightedProb_R100P', 'SeqAgg5_R100P']
    
    stats = []
    for col in cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            stats.append({
                'Metric': col,
                'Mean': mean_val,
                'Std': std_val,
                'Min': min_val,
                'Max': max_val
            })
    
    stats_df = pd.DataFrame(stats)
    
    # 格式化输出
    print("\n你的实验结果 (均值 ± 标准差):")
    print("-" * 80)
    for _, row in stats_df.iterrows():
        print(f"{row['Metric']:25s}: {row['Mean']:6.2f}% ± {row['Std']:5.2f}%  "
              f"(min: {row['Min']:6.2f}%, max: {row['Max']:6.2f}%)")
    
    # 与论文对比
    print("\n" + "="*80)
    print("与论文 Table I 对比")
    print("="*80)
    
    paper_results = {
        'Accuracy': (93.80, 0.93),
        'Standard_R100P': (44.13, 2.97),
        'Weighted_R100P': (54.13, 5.56),
        'WeightedProb_R100P': (77.93, 3.97),
        'SeqAgg5_R100P': (100.00, 0.00)
    }
    
    print(f"\n{'Metric':<25s} | {'你的结果':^20s} | {'论文结果':^20s} | {'差异':^15s}")
    print("-" * 80)
    
    for _, row in stats_df.iterrows():
        metric = row['Metric']
        your_mean = row['Mean']
        your_std = row['Std']
        
        if metric in paper_results:
            paper_mean, paper_std = paper_results[metric]
            diff = your_mean - paper_mean
            
            print(f"{metric:<25s} | {your_mean:6.2f}% ± {your_std:5.2f}% | "
                  f"{paper_mean:6.2f}% ± {paper_std:5.2f}% | {diff:+6.2f}%")
    
    # 保存到 CSV
    output_csv = 'multi_seed_statistics.csv'
    df.to_csv(output_csv, index=False)
    print(f"\n详细数据已保存到: {output_csv}")
    
    # 保存统计摘要
    stats_output = 'statistics_summary.txt'
    with open(stats_output, 'w', encoding='utf-8') as f:
        f.write("多种子统计结果 (15 seeds)\n")
        f.write("="*80 + "\n\n")
        
        for _, row in stats_df.iterrows():
            f.write(f"{row['Metric']}: {row['Mean']:.2f}% ± {row['Std']:.2f}%\n")
            f.write(f"  Min: {row['Min']:.2f}%, Max: {row['Max']:.2f}%\n\n")
        
        f.write("\n与论文对比\n")
        f.write("="*80 + "\n\n")
        
        for _, row in stats_df.iterrows():
            metric = row['Metric']
            if metric in paper_results:
                paper_mean, paper_std = paper_results[metric]
                f.write(f"{metric}:\n")
                f.write(f"  你的: {row['Mean']:.2f}% ± {row['Std']:.2f}%\n")
                f.write(f"  论文: {paper_mean:.2f}% ± {paper_std:.2f}%\n")
                f.write(f"  差异: {row['Mean'] - paper_mean:+.2f}%\n\n")
    
    print(f"统计摘要已保存到: {stats_output}")

if __name__ == '__main__':
    main()
