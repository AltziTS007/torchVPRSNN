# 创新阶段工作记录：序列聚合的方向/速度鲁棒性

> 阶段切换：2026-09-24 起，从「论文复现」转入「创新改进」
> 主攻方向：改进速度补偿序列聚合，解决**反向遍历完全失效**，并扩展速度容差
> 对照论文：Tsanko et al., *Visual Place Recognition Using Rate-Encoded Spiking Neural Networks with Discrete STDP Learning*（RA-L）
> 基线代码：`core/neuronal_assignments.py` → `sliding_window_aggregation()`
> 基线协议：`experiments/run_table_iv_real.py`（Nordland, k=5, product）

---

## 0. 创新目标（一句话）

在**不改动 STDP 训练与神经元分配**的前提下，只改推理端序列聚合，使得：

| 条件 | 当前（论文/复现） | 目标 |
|------|-------------------|------|
| 正向恒速 1.0× | 100% R@100P | 保持 100% |
| **反向遍历** | **0%** | 恢复到接近正向水平 |
| 速度 ±10% | 100% | 保持 |
| 速度 ±20% | ~7%（悬崖） | 明显改善（争取 ≥50%，理想 ≥90%） |
| 丢帧 / 绕路 | 100% / 90% | 不劣化 |

评估仍统一用 **R@100P**，协议与 Table IV 对齐，便于直接对比。

---

## 1. 当前「正向」实验是怎么做的（对比基线）

### 1.1 端到端主流程（`core/snn_model.py`）

正向主实验不是只跑聚合，而是完整 SNN-VPR 流水线：

```text
Nordland 数据
  train (Reference): nordland_clean/data/spring + fall
  test  (Query):     nordland_clean/data/summer
        │
        ▼
预处理：灰度 → 28×28 → 7×7 patch Min-Max 归一化
        │
        ▼
Rate 编码（Poisson）：t_steps=200（Table IV 脚本里用 150），rate_scale=0.25
  输出 spk_in: [T, B, 784]
        │
        ▼
STDP 无监督训练 120 epoch（Input→Exc 400，Hard WTA + 稳态 + 列 L2 归一化）
  权重 w_in_exc: [784, 400]，可存 baseline_100.pt
        │
        ▼
校准：get_standard_assignments() / get_training_spike_counts()
  S_R: [400, n_classes]  训练集各神经元对各地点的总脉冲数
        │
        ▼
查询评估：evaluate_vpr() 在 summer 上逐帧推理（默认状态隔离）
  S_Q: [n_query, 400]    每帧兴奋性神经元脉冲计数
        │
        ▼
分配策略
  Standard  → 按 argmax 分配打分
  Weighted  → Eq.9-11：参与度正则 + 响应强度归一 + 缺失脉冲惩罚
  W+Prob    → 在 Weighted 分数上做 Min-Max + L1，得到 prob_scores [n_query, n_classes]
        │
        ▼
序列聚合 sliding_window_aggregation(prob_scores, targets, k, rule, velocity_factor)
  k ∈ {1,3,5,7,10,15}，rule ∈ {product, mean}
        │
        ▼
指标：Accuracy / P@100R / R@100P / AUC-PR
```

**关键约定（正向隐含假设）**：

1. 查询帧按**时间顺序**排列，对应地点编号**单调递增**且近似 `targets[i] ≈ targets[i-1] + 1`（恒速、1 帧 ≈ 1 个地点）。
2. 每帧查询之间**状态隔离**（膜电位/突触 trace 清零），脉冲计数 `S_Q` 独立。
3. 序列聚合只作用在 **prob_scores** 上，**不重跑 SNN**。
4. 主结果里 product、k=5 是论文选定配置（速度鲁棒性与精度的平衡点）。

### 1.2 序列聚合算法（正向实现细节）

代码：`core/neuronal_assignments.py` 的 `sliding_window_aggregation()`。

**Step A — 断序检测（sequence break）**

```python
# 当前实现：只把「严格 +1」当作连续
if targets_t[i] != targets_t[i-1] + 1:
    resets.append(i)
```

合法窗口要求：当前位置到本段起点至少 `k` 帧（丢掉段首 `k-1` 帧）。

**Step B — 速度补偿位移对齐**

对窗口 `[i-k+1, i]` 中的第 `m` 帧（`m=0` 最早，`m=k-1` 为当前帧）：

```python
raw_shift = (k - 1 - m) * velocity_factor   # 默认 v=1.0
shift = round_half_away(raw_shift)          # 四舍五入到整数格
# 将该帧的类别维向左/右平移 shift 格，对齐到「当前帧」的地点坐标
```

直觉：更早的帧对应更小的地点编号，需要向右平移才能和当前帧对齐；`velocity_factor` 放大/缩小每步的空间间隔。

**Step C — 融合**

- **product**：对齐后的 log 概率求和 → exp → 归一化（对错位极敏感）
- **mean**：对齐后的概率取平均 → 归一化（更钝，小数据上更早到 100%）

**Step D — 输出**

返回合法帧的 `agg_scores` 与 `valid_targets`，再算 `calculate_r_at_100p`。

### 1.3 正向基线数字（创新前后都要用这套比）

| 来源 | 配置 | R@100P |
|------|------|--------|
| 本复现 Table IV | 正向 k=5 product v=1.0，10 seeds (16–25) | **100.00% ± 0.00** |
| 本复现核心实验 | W+Prob k=1 | 66%（单次） |
| 本复现核心实验 | SeqAgg k=5 product | 100% |
| 论文 Table IV | 正向同配置 | 100.00% ± 0.00 |
| 论文 Table V | k=1 / k=5 product | 77.93% / 100% |

复现结果文件：`table_iv_nordland_results.txt`、`results/nordland_100_2026-09-02_17-41-28/`。

### 1.4 正向操作包络（与反向对照用）

| 条件 | 本复现 R@100P | 论文 Table IV |
|------|---------------|---------------|
| Baseline 1.0× | 100% | 100% |
| 速度 +10% / −10% | 100% / 100% | 100% / 100% |
| 速度 +20% / −20% | 9.27% / 5.42% | 5.83% / 0.00% |
| 速度 ±50% | 0% | 0% |
| **反向遍历** | **0%** | **0%** |
| 丢帧 10% / 50% | 100% / 90% | 100% / 90% |
| 路径偏差 | 100% | 100% |

---

## 2. 反向实验做了吗？怎么做的？

### 2.1 状态

**已经做过**，作为 Table IV「操作包络」的一列，不是单独完整研究。

- 脚本：`experiments/run_table_iv_real.py`（Nordland）与 `run_table_iv_oxford.py`（Oxford 同款逻辑）
- 时间：2026-09-06（见 `已做.md` 第十二节）
- 种子：16–25 共 10 个（每个 seed **从头 STDP 训练 120 epoch**，再评估）
- 结果：`Reverse traversal | 0.00 | ± 0.00`（见 `table_iv_nordland_results.txt`）

### 2.2 反向是怎么仿真的（非常重要，创新实验要沿用同一协议）

**没有单独的「反向行驶数据集」**。做法是：在**已经算好的正向查询概率**上把时间顺序整体反转，再丢给同一个 `sliding_window_aggregation`：

```python
# run_table_iv_real.py 约 103–107 行
targets_rev = list(reversed(targets))           # 99,98,...,0
prob_scores_rev = torch.flip(prob_scores, dims=[0])  # 同步反转行序
agg_rev, vt_rev, _ = sliding_window_aggregation(
    prob_scores_rev, targets_rev,
    k=5, rule='product', velocity_factor=1.0   # ← 仍用正向对齐逻辑
)
results['reverse'] = calculate_r_at_100p(...) if len(vt_rev) > 0 else 0.0
```

含义：

1. **SNN 前向、STDP、分配、prob_scores 都与正向共用**（每帧查询仍是夏天原图的响应）。
2. 反向只改变「查询序列的时间顺序」——等价于机器人沿同一路径倒着走时，看到的帧序反过来。
3. 标签变成递减序列：`99, 98, 97, …, 0`。
4. 聚合函数**没有**针对递减序列做任何适配，`velocity_factor` 仍为 +1.0。

因此当前 0% 的含义是：**「正向定义的滑动窗聚合，在递减地点序列上失效」**，不是 SNN 认不出地方。

### 2.3 反向 0% 的根因（创新切入点，已定位）

根因全部在聚合层，可对照 `neuronal_assignments.py`：

| # | 根因 | 机制 | 后果 |
|---|------|------|------|
| R1 | 断序检测只认 `targets[i]==targets[i-1]+1` | 递减序列每一帧都被判为 reset | 合法窗口数 ≈ 0，R@100P 记 0 |
| R2 | 位移符号写死为正向 `+(k-1-m)*v` | 反向时空间偏移应取反 | 即使修好 R1，对齐仍错位，product 把概率乘到错误地点 |
| R3 | （次要）velocity_factor 语义与方向耦合 | 反向未定义 `v` 的符号 | 后续扩展变速/掉头时要一并设计 |

**证据**：`len(vt_rev)==0` 时脚本直接写 0.0；即使窗口非空，R2 也会让检索崩溃。创新修复必须 **R1+R2 同时改**。

### 2.4 与正向对比一览（给导师/自己看的表）

| 项目 | 正向实验 | 反向实验（现状） |
|------|----------|------------------|
| 数据帧 | summer 0→99 顺序 | 同一套 prob_scores 整行 flip |
| targets | 0,1,…,99 | 99,98,…,0 |
| SNN / 分配 | 完整流水线 | **共用正向结果**（不重跑 SNN） |
| 聚合参数 | k=5, product, v=1.0 | **相同**（未改方向） |
| 合法窗口 | N≈96 | ≈0 或全部错位 |
| R@100P | 100% | 0% |
| 评估脚本 | `snn_model.py` sweep + Table IV | 仅 Table IV 一列 |

### 2.5 创新实验协议约定（避免前后不可比）

1. **聚合层修复**评价时：固定同一份 `prob_scores`（或同一模型多 seed 的 prob_scores），只比 `sliding_window_aggregation` 新旧实现 → 消融最干净。
2. **系统级**评价时：仍跑 `run_table_iv_real.py` 式 10-seed Table IV，输出同格式 `table_iv_*.txt`。
3. 反向协议保持「flip 时间序」不变，除非明确做「真·反向采集」扩展实验（Nordland 无此数据，不必强行做）。
4. 主指标：**R@100P**；辅：合法窗口数 N、Acc、latency。

---

## 3. 改进方案（分层，按风险递增）

| 层级 | 内容 | 预期 | 状态 |
|------|------|------|------|
| **L1** | 断序兼容 `±1` 连续；检测到连续递减则内部视为反向，对齐 shift 取反 | 反向 0% → 接近正向 100% | 待做 |
| **L2** | 双向假设：正/反各聚合一次，取一致性更高者或概率混合 | 无需预知行驶方向 | 待做 |
| **L3** | 多速度假设 `{0.8…1.2}` 并行，窗内一致性选 v 或加权 | ±20% 悬崖变缓 | 待做 |
| **L4** | 自适应速度估计 + 软（分数）位移对齐 | 论文 future work 完整版 | 可选 |

实现落点：优先新增 `sliding_window_aggregation_v2()` 或给旧函数加 `direction='auto'|'forward'|'backward'`，**保持旧函数可复现 Table IV**。

---

## 4. 待做事项（Todo）

### 阶段 A：协议固化与可测基线（先做）

- [ ] A1. 从现有 10-seed Table IV / 单次 `nordland_100_*` 结果中，导出**固定的一份** `prob_scores` + `targets` 缓存（`.pt`/`.npz`），作为聚合层单测输入
- [ ] A2. 写最小复现脚本：只调 `sliding_window_aggregation`，打印正向 N、R@100P 与反向 N、R@100P（不训网）
- [ ] A3. 确认反向 0% 的直接证据：合法窗口数、断序次数（在 A2 里打印 `len(resets)` / `len(valid_indices)`）

### 阶段 B：L1 方向自适应聚合（核心）

- [ ] B1. 修改断序逻辑：连续判定支持 `+1` 与 `-1` 两种 run
- [ ] B2. 对齐 shift：反向 run 使用 `raw_shift = -(k-1-m)*velocity_factor`（或等价地对类别轴做镜像后再用旧公式）
- [ ] B3. 单测：同一 prob_scores 正向 vs flip 后反向，R@100P 应对称接近
- [ ] B4. 跑 10-seed Table IV，对比旧 reverse=0% 与新 reverse=?
- [ ] B5. 确认正向 baseline 仍为 100%（回归）

### 阶段 C：L2 未知方向（双向假设）

- [ ] C1. 实现 `direction='auto'`：双向聚合，按窗内一致性（如 max-prob 熵 / 与单帧 top1 一致率）选方向
- [ ] C2. 在「正向 / 反向 / 中途掉头」三种仿真序列上测 auto
- [ ] C3. 报告 auto 相对 oracle-direction 的差距

### 阶段 D：L3 速度容差（与反向正交，可并行）

- [ ] D1. 多 v 网格并行聚合，选窗内一致性最高的 v
- [ ] D2. 重点测 ±20%（当前 ~7%）是否变缓；不破坏 ±10% 的 100%
- [ ] D3. 记录聚合延迟开销（论文强调 SeqAgg 仅 +0.20 ms）

### 阶段 E：汇报与文档

- [ ] E1. 更新 `new.md` 进展日志（每次实验后追加）
- [ ] E2. 整理「正向 vs 反向修复前后」对比图（R@100P 条形图 + 操作包络）
- [ ] E3. 给郭老师的进展说明（突出：根因、L1 效果、是否支持未知方向）

---

## 5. 进展日志

### 2026-09-24

- **阶段切换**：复现 → 创新（序列聚合方向/速度鲁棒性）。
- **盘点结论**：
  - 正向全流程与 Table IV 协议已复现；正向 k=5 product = 100% R@100P（10 seeds）。
  - **反向实验已做过**（Table IV 一列，`run_table_iv_real.py` 中 flip 时间序），10 seeds 均为 **0%**；未单独研究。
  - 反向仿真方式：共用正向 `prob_scores`，仅反转帧序；聚合未改方向逻辑。
- **根因确认**（代码级）：
  1. 断序检测只识别 `+1` 递增 → 反向几乎无合法窗口；
  2. 位移对齐符号写死正向 → 反向必错位。
- **决定**：先做 L1（方向自适应）+ 双向可测基线（阶段 A/B），L2/L3 随后。
- **产出**：本文件 `new.md`。

---

## 6. 相关文件索引

| 文件 | 用途 |
|------|------|
| `core/neuronal_assignments.py` | `sliding_window_aggregation`（待改） |
| `core/snn_model.py` | 正向全流程 + k sweep + 速度敏感性 |
| `experiments/run_table_iv_real.py` | 操作包络（含 reverse 列）10-seed |
| `table_iv_nordland_results.txt` | 创新前基线表（必须保留） |
| `results/nordland_100_2026-09-02_17-41-28/` | 正向核心一次完整结果 |
| `results/seed_42_*` … `seed_56_*` | 多种子评估（聚合层可复用其 log/缓存） |
| `已做.md` / `todo.md` | 复现阶段归档（只读对照） |
| **`new.md`** | **本创新阶段工作记录（主更新文件）** |

---

## 7. 笔记区

（实验命令、失败原因、和导师讨论结论写在下面）

