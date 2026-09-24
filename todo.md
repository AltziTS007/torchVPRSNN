# torchVPRSNN 学习计划

> 论文：Visual Place Recognition Using Rate-Encoded Spiking Neural Networks with Discrete STDP Learning
> 服务器路径：`/data/cap/zhangyuzhuo/torchVPRSNN/`
> 环境：conda snntorch / PyTorch 2.5.1+cu121 / RTX 3090
> 创建日期：2026-09-02

---

## 当前进度

- [x] 环境搭建（conda + PyTorch + snnTorch + 全部依赖）
- [x] 数据集准备（Nordland spring/fall/summer，100 个采样地点）
- [x] 快速验证跑通（Standard 83% / W+Prob 66% / SeqAgg(k=5) 100%）
- [x] 模型已保存（baseline_100.pt）
- [x] 论文全文阅读 + 创新点总结

---

## 第一周：SNN 基础概念

### Day 1：整体架构理解
- [ ] 阅读论文 III-A ~ III-E（方法章节），建立全局印象
- [ ] 读 `core/snn_model.py` 的 `main()` 函数（~330-670行），画出完整 pipeline 流程图
- [ ] 对照代码理解四步走：数据加载 → STDP 训练 → 神经元分配 → 评估可视化
- [ ] 实验：`--max-samples 20 --t-steps 50` 快速跑通，观察 `results/` 目录结构

### Day 2：Rate Encoding（脉冲编码）
- [ ] 阅读论文 III-A（预处理与编码）
- [ ] 读 `core/encoders.py`：`RateEncoder` 类
- [ ] 理解关键参数：`rate_scale=0.25`、`t_steps=200` 的作用
- [ ] 实验：修改 `--t-steps`（50/100/200），观察权重图差异
  ```bash
  CUDA_VISIBLE_DEVICES=2 python core/snn_model.py --device cuda --t-steps 50
  CUDA_VISIBLE_DEVICES=2 python core/snn_model.py --device cuda --t-steps 200
  ```

### Day 3：LIF 神经元模型
- [ ] 阅读论文 III-B（LIF 动力学公式）
- [ ] 读 `core/networks.py`：`torchVPRSNN` 类，重点理解 `__init__` 中的参数初始化
- [ ] 画出 Input(784) → Excitatory(400) ↔ Inhibitory(400) 的网络拓扑
- [ ] 理解膜电位积分 → 阈值判断 → 脉冲发放 → 重置的完整流程

---

## 第二周：核心机制深入（重点周）

### Day 4：forward() 模拟循环（最重要的 80 行代码）
- [ ] 读 `core/networks.py` 第 108-188 行的 `forward()` 函数
- [ ] 逐行注释每个步骤，画出单个时间步的数据流：
  ```
  pre_spk → 加权求和 → LIF积分 → 阈值判断 → spike
                                            ↓
                                    WTA 竞争
                                            ↓
                                    兴奋→抑制→反馈抑制
                                            ↓
                                    STDP 更新权重
  ```
- [ ] 理解 `return_state` 参数如何控制状态隔离

### Day 5：WTA 竞争机制
- [ ] 读 `core/mechanisms.py`：`hard_wta_step()` 函数
- [ ] 理解 WTA 的作用：每步只允许最强神经元放电，强制稀疏响应
- [ ] 实验：关闭 WTA，观察差异
  ```bash
  # 正常 WTA
  CUDA_VISIBLE_DEVICES=2 python core/snn_model.py --device cuda --experiment-name wta_on
  # 关闭 WTA
  CUDA_VISIBLE_DEVICES=2 python core/snn_model.py --device cuda --wta-mode none --experiment-name wta_off
  ```
- [ ] 对比两个结果的准确率、R@100P、权重图

### Day 6：STDP 学习规则（论文核心）
- [ ] 读 `core/mechanisms.py`：`weight_dependent_stdp()` 函数
- [ ] 对照论文 Eq.7 逐行验证代码：
  - LTP：`Δw = η₊ · (wmax - w) · pre_trace · post_spk`
  - LTD：`Δw = η₋ · w · post_trace · pre_spk`
- [ ] 理解权重依赖性：增幅与 `(wmax - w)` 成正比，减幅与 `w` 成正比
- [ ] 查看 `results/*/stdp_viz/stdp_epoch_*.png`，观察 LTP/LTD 热图随 epoch 的变化
- [ ] 实验：修改学习率 `--a-plus 0.01 --a-minus 0.01`（默认 0.005/0.01）

### Day 7：Homeostasis 稳态机制 + 权重归一化
- [ ] 读 `core/mechanisms.py`：
  - `homeostatic_threshold_update()` — 对照论文 Eq.8
  - `normalize_weights_column()` — 理解归一化的目的
- [ ] 理解 `rtarget=0.01`：目标放电率 1%，维持稀疏编码
- [ ] 实验：逐一关闭做消融
  ```bash
  # 关闭稳态
  CUDA_VISIBLE_DEVICES=2 python core/snn_model.py --device cuda --disable-homeostasis --experiment-name no_homeo
  # 关闭权重归一化
  CUDA_VISIBLE_DEVICES=2 python core/snn_model.py --device cuda --disable-patch-norm --experiment-name no_norm
  ```
- [ ] 记录三组结果到表格，理解每个机制的独立贡献

---

## 第三周：分配策略与序列聚合

### Day 8：三种神经元-地点分配策略
- [ ] 读论文 III-E（分配策略）
- [ ] 读 `core/neuronal_assignments.py`：`get_standard_assignments()`
- [ ] 读 `core/vprsnn_evaluation.py`：Standard 分配的评分逻辑
- [ ] 对照结果理解三种策略的区别：
  | 策略 | 你的结果 R@100P | 论文均值 R@100P |
  |------|----------------|----------------|
  | Standard | 42% | 44.13% |
  | Weighted | 37% | 54.13% |
  | W+Prob | 66% | 77.93% |

### Day 9：Weighted Assignment 三步流水线
- [ ] 读 `core/neuronal_assignments.py` 和 `core/vprsnn_evaluation.py`
- [ ] 对照论文 Eq.9/10/11 逐行验证：
  - **Step 1** 参与度正则化 (Eq.9)：惩罚"什么地点都响应"的神经元
  - **Step 2** 响应强度归一化 (Eq.10)：按相对响应比例加权
  - **Step 3** 缺失脉冲惩罚 (Eq.11)：学过但没放电 → 降分

### Day 10：Probability-Based Assignment
- [ ] 对照论文 Eq.12/13 验证 Min-Max + L1 归一化
- [ ] 理解为什么概率输出比原始分数更好：阈值无关性、跨查询可比性
- [ ] 查看 `results/*/Weighted+Prob/pr_curve.png` 和 `results/*/Standard/pr_curve.png` 的差异

### Day 11：序列聚合（Sliding Window）
- [ ] 读 `core/neuronal_assignments.py`：`sliding_window_aggregation()` 函数
- [ ] 对照论文 Eq.16/17/18 理解：
  - Product rule（对数求和）vs Mean rule
  - 位移对齐（velocity_factor）
- [ ] 从你的实验结果中提取趋势：
  | k | Product R@100P | Mean R@100P |
  |---|----------------|-------------|
  | 1 | 66% | 66% |
  | 3 | 45.92% | 100% |
  | 5 | **100%** | **100%** |
  | 7 | 100% | 100% |
- [ ] 思考：为什么 Mean rule 在 k=3 就达到 100%，Product rule 需要 k=5？

### Day 12：速度容忍性分析
- [ ] 对照论文 Table IV 和 Figure 3 理解操作包络
- [ ] 从你的实验结果看速度敏感性：
  - k=5：v=0.90~1.10 时 100%，v=0.80 和 v=1.20 时 0%
  - k=15：v=0.95~1.05 时 100%，范围更窄
- [ ] 理解直觉：k 越大窗口越长，对速度失配越敏感

---

## 第四周：消融实验与论文复现

### Day 13：多种子统计实验 ✅ 已完成（2026-09-04）
- [x] 用已保存的模型跑 15 个 seed（42~56），计算均值±标准差
  ```bash
  for seed in 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56; do
      CUDA_VISIBLE_DEVICES=2 python core/snn_model.py --device cuda \
          --load-model baseline_100.pt --seed $seed \
          --experiment-name "seed_${seed}"
  done
  ```
- [x] 汇总 15 个结果，与论文 Table I 对比
- **结果**：Acc=83.93%±1.58%，标准差很小（1.58%），说明模型在不同随机初始化下性能稳定
- **待完善**：汇总脚本正则表达式需要修复，目前只提取了 Accuracy，其他指标（R@100P）提取失败

### Day 14：组件消融实验 ✅ 已完成（2026-09-05）
- [x] 运行关闭 WTA 实验：`--wta-mode none`
- [x] 运行关闭稳态实验：`--disable-homeostasis`
- [x] 运行关闭归一化实验：`--disable-patch-norm`
- [x] 运行模拟溢出实验：`--simulate-spillover`
- [x] 汇总消融结果，生成对比表格

**结果总结**（W+Prob R@100P 对比）：
- Baseline: 66.0%
- no_wta: 0.0% (-66%) — **完全崩溃，WTA 是核心**
- no_homeo: 60.0% (-6%)
- no_norm: 60.0% (-6%)
- spillover: 69.0% (+3%) — 意外提升，需多种子验证

**产出**：`汇总消融结果.py`、`results/ablation_summary.csv`

### Day 15：时间步消融实验 ✅ 已完成（2026-09-05）
- [x] 运行 6 个不同时间步实验：t_steps=50, 100, 150, 200, 300, 400
- [x] 汇总时间步消融结果，生成对比表格

**结果总结**（W+Prob R@100P 对比）：
- t_steps=50: 0.0% — 时间步过短，概率分配完全失效
- t_steps=100: 33.0% — 开始有效但仍不足
- t_steps=150: 59.0% — **性能-效率最佳平衡点**
- t_steps=200: 60.0% — 边际收益递减
- t_steps=300: 51.0% — 性能波动
- t_steps=400: 62.0% — 最高但仍不稳定

**关键发现**：
- t_steps=150~200 已足够，无需使用论文中的 300
- SeqAgg(k=5) 在所有时间步下都稳健（99%~100%）

**产出**：`汇总时间步消融结果.py`、`results/tsteps_ablation_summary.csv`

### Day 16：操作包络实验 ✅ 已完成（2026-09-06）
- [x] 运行 `experiments/run_table_iv_real.py`（速度/反向/丢帧/偏航鲁棒性）
- [ ] 运行 `bash experiments/run_sliding_window_full_curve.sh`（AUC vs chunk size）

**结果总结**（k=5 序列聚合鲁棒性测试）：
- Baseline (1.0x): 100% — 完美性能
- 速度偏差 ±10%: 100% — 完全鲁棒
- 速度偏差 ±20%: 7.34% — 急剧崩溃
- 速度偏差 ±50%: 0% — 完全失效
- 反向遍历: 0% — 完全失效
- 10% 帧丢失: 100% — 完全鲁棒
- 50% 帧丢失: 90% — 基本可用，方差大
- 路径偏差: 100% — 完全鲁棒

**关键发现**：
- 速度容忍窗口严格限定在 ±10%，±20% 时性能暴跌到 7.34%
- 帧丢失高度鲁棒：50% 丢失仍保持 90% 性能
- 反向遍历完全失效（窗口假设时序方向固定）
- 路径偏差完全鲁棒（系统容忍偶尔绕路）

**产出**：`table_iv_nordland_results.txt`

### Day 16-17：进阶探索
- [ ] 换 Oxford RobotCar 数据集：`--dataset oxford --sample-10m`
- [ ] 调整网络规模：`--n-exc 200` 或 `--n-exc 800`
- [ ] 尝试不同采样密度：重新生成 200/500 个地点的数据集

---

## 学习优先级排序

```
最高优先级（必读必懂）：
  ① networks.py forward() 整个模拟循环
  ② Eq.7 STDP 权重更新 → mechanisms.py:weight_dependent_stdp()
  ③ Eq.9-13 加权概率分配 → neuronal_assignments.py 三步流水线

高优先级（做实验验证）：
  ④ WTA 竞争机制 → mechanisms.py:hard_wta_step()
  ⑤ 序列聚合 → neuronal_assignments.py:sliding_window_aggregation()
  ⑥ 状态隔离 → networks.py 中 t=0 的重置逻辑

中优先级（理解背景）：
  ⑦ Patch 预处理与 Rate 编码
  ⑧ Homeostasis 稳态机制
  ⑨ 可扩展性与操作包络
```

---

## 常用命令速查

```bash
# 激活环境
conda activate snntorch

# 快速验证（2分钟）
CUDA_VISIBLE_DEVICES=2 python core/snn_model.py --device cuda --max-samples 20 --t-steps 50

# 完整训练（约10分钟）
CUDA_VISIBLE_DEVICES=2 python core/snn_model.py --device cuda --save-model baseline_100.pt

# 加载模型跳过训练（约2分钟）
CUDA_VISIBLE_DEVICES=2 python core/snn_model.py --device cuda --load-model baseline_100.pt

# 消融：关闭 WTA
CUDA_VISIBLE_DEVICES=2 python core/snn_model.py --device cuda --wta-mode none

# 消融：关闭稳态
CUDA_VISIBLE_DEVICES=2 python core/snn_model.py --device cuda --disable-homeostasis

# 消融：关闭权重归一化
CUDA_VISIBLE_DEVICES=2 python core/snn_model.py --device cuda --disable-patch-norm

# 模拟 Spillover
CUDA_VISIBLE_DEVICES=2 python core/snn_model.py --device cuda --simulate-spillover

# 监控 GPU
watch -n 2 nvidia-smi

# 后台运行（防 SSH 断连）
tmux new -s snn
# ... 运行命令 ...
# Ctrl+B, D 分离；tmux attach -t snn 重连
```

---

## 核心文件索引

| 文件 | 内容 | 对应论文 |
|------|------|---------|
| `core/snn_model.py` | 主入口，完整 pipeline | — |
| `core/encoders.py` | RateEncoder 脉冲编码 | III-A |
| `core/networks.py` | SNN 网络定义 + forward() 模拟 | III-B, III-C |
| `core/mechanisms.py` | STDP + WTA + Homeostasis + Norm | III-D |
| `core/neuronal_assignments.py` | 分配策略 + 序列聚合 | III-E |
| `core/vprsnn_evaluation.py` | 评估函数 + 距离矩阵 | IV |
| `core/metrics.py` | 指标计算（R@100P, AUC-PR 等） | IV |
| `experiments/` | 论文表格复现脚本 | V |

---

## 已完成的实验记录

| 日期 | 实验 | 结果 | 备注 |
|------|------|------|------|
| 09-01 | 20 张图冒烟测试 | Acc 5%, R@100P 0% | 数据量不足，预期内 |
| 09-01 | 100 张图验证 | Acc 1%, R@100P 0% | 数据量不足，预期内 |
| 09-01 | 全数据 35768 帧 | Acc 0.28% | 失败：地点太多神经元不够 |
| 09-02 | 100 地点采样 | Acc 83%, W+Prob 66%, SeqAgg 100% | 成功复现论文核心趋势 |
| 09-04 | 15 seeds 多种子统计 | Acc 83.93%±1.58% | 模型稳定性好，标准差小 |
| 09-05 | 时间步消融 (6 configs) | t_steps=150~200 最佳 | 性能-效率平衡点，SeqAgg 全稳健 |
| 09-06 | 操作包络实验 (Table IV) | ±10% 鲁棒，±20% 崩溃 | 与论文高度一致，完全复现 |

---

## 笔记区

（学习过程中的疑问、发现、实验记录可以写在这里）
