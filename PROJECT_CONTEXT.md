# 染色体倒位异常检测项目 — 上下文文档

> **用途**：每次开新对话或切换 agent 时，将此文件作为背景提供给 Claude，避免重复解释项目背景。
> **维护**：每次完成重要实验后更新"实验结果"与"当前状态"两节。

---

## 一、项目目标

在染色体倒位异常检测任务中，实现对**正常/异常的可靠判别**，并尽可能提升对**未见异常类型**的泛化能力。

这不是普通的闭集分类器任务。

---

## 二、核心困难（必须理解）

1. **异常类型少且长尾**：训练集中异常类别数量有限，且分布极不均衡。
2. **存在未见异常类型**：测试阶段会出现训练阶段从未见过的异常类别。
3. **单图信息不足**：倒位异常需要通过同源染色体配对比较才能判断，单张图不够。
4. **正常分布不唯一**：不同染色体（1号、2号、X号...）各有自己的正常形态，正常分布是 chromosome-conditioned 的，不是全局统一的。

---

## 三、核心转变（贯穿整个项目）

项目主线从"异常分类"转向了"**正常分布建模**"：

- **不去学**"异常长什么样"（因为异常类型不完整）
- **转为学**"正常同源 pair 应该长什么样"

---

## 四、数据协议

- 训练集、验证集、测试集已按**病例隔离**原则划分（同一病例不跨集）
- 评估必须区分三类能力：
  - 正常/异常检测能力
  - 已见异常类型的识别能力
  - 未见异常类型的泛化能力
- 不能只报总体准确率（会掩盖未见异常的真实失败）

---

## 五、已尝试的方向与结论

### 方向一：单条染色体监督学习（已证明不足）

| 实验 | 设定 | 结论 |
|------|------|------|
| N1 | 单图二分类（ResNet18/50，加减 chr_id） | 效果弱，异常召回偏低。单图信息不足以稳定抓到倒位异常 |
| N2 | 只在已见异常类型上做闭集多分类 | 效果很好（ResNet50 很强）。**但这不是主任务**——没有正常样本、没有未见异常。意义在于证明图像里有可学信号 |
| N3 | 单图原型/度量学习 | 比普通分类更接近主线，但仍缺少同源对照信息。问题不在于分类头不好，而在于输入信息不够 |

**核心结论**：仅靠单条染色体图像无法稳定识别倒位异常，转向 pair 路线是实验逼出来的。

---

### 方向二：配对输入监督学习（有改善，但未彻底解决）

| 实验 | 设定 | 关键指标 | 结论 |
|------|------|----------|------|
| P1 | 基础配对分类基线 | — | 比单图路线合理，但未解决未见异常问题 |
| P4/P5 | Siamese + pair contrastive loss + abnormal side 预测 | — | 让模型学习"pair 应该相似"，对已见异常有帮助，未见异常仍不稳 |
| P6 | 平衡采样 Siamese 配对基线 | AUPRC=0.2478, F1=0.2647, Recall_abn=0.1731 | 召回率不够，说明监督式 pair 分类对异常检测偏弱 |
| P10 | 配对结构化头 | AUPRC=0.1983, F1=0.2500, Recall_abn=0.1538 | 加结构化输出头无根本改善，监督信号仍围绕已知异常类别 |
| P11 | 对应关系建模 + 多原型度量 | AUPRC=0.2694, F1=0.2500, Recall_abn=0.1635 | 接近 P12 思想，但仍属有异常监督范式 |

---

### 方向三：轻量增强与中间改进（已证明无效）

尝试过：MixStyle、style consistency、balanced supervised contrastive、某些附加 pair loss、order-aware 设计、hybrid pair 结构。

**核心结论**：
- 病例间风格域偏移不明显，MixStyle 等无关键增益
- 小损失项有时改善验证集，但无法稳定转移到测试集
- 任务瓶颈不在于缺一个小损失项，而在于建模范式是否正确
- **不要再做轻量化改进**，这一判断已被实验支持

---

### 方向四：只用正常样本训练的异常检测（**当前主线**）

#### P12：核心模型（当前最优主干）

**设定**：
- 训练时只用正常的同源染色体对（normal-only）
- 输入是一对同源染色体
- 显式保留 chromosome id 条件
- 通过多原型度量空间学习"正常 pair manifold"

**为什么有效（四个缺一不可的条件）**：
1. **Pair 输入**：利用"病例内天然对照"
2. **Chromosome 条件化**：不同染色体正常分布不同，必须分开建模
3. **多原型度量空间**：正常分布不是单峰的，多个 prototype 比单一中心合理
4. **只训练正常分布**：避免模型去背有限的异常类别标签

**P12 的本质**：
> 一个 pair-aware、chromosome-conditioned、normal-only 的正常结构分布建模器，不是二分类器。

---

## 六、P12 模型结构（技术细节）

### 整体架构

```
一对同源染色体图像 + chromosome_id
        ↓
pair-aware encoder (CorrespondenceIntervalPairClassifier)
        ↓
256维 pair embedding
        ↓
chromosome-specific prototype matching (MultiPrototypeMetricModel)
        ↓
anomaly score（到本染色体最近 normal prototype 的距离）
```

### Pair Encoder 的关键设计

按 forward 顺序：

1. **左右图分别提 backbone 特征**：同一 ResNet + `1x1 conv + BN + GELU` 投影到 192 维
2. **全局差异特征**：左右各做全局平均池化后取绝对差（`global_diff`）
3. **序列化**：特征图在宽度方向平均后转为 token 序列，送入 Transformer，保留沿染色体轴向的顺序信息
4. **对应关系建模（核心）**：
   - `direct_corr`：左右按正常顺序的对应关系
   - `reverse_corr`：左右按反向顺序的对应关系
   - `corr_delta = reverse_corr - direct_corr`（倒位倾向的直接编码）
5. **区段级证据提取**：相关矩阵经卷积 + attention 得到 `corr_vec`（哪一段最可疑）
6. **结构统计量**（显式编码）：
   - `1 - direct_diag_mean`（正向对齐差异度）
   - `1 - reverse_diag_mean`（反向对齐差异度）
   - `reverse_gain = relu(reverse_diag_mean - direct_diag_mean)`（反向比正向更优的量）
   - 以及 direct/reverse corr 的均值和最大值
7. **Token 级差异**：正向和反向 token 差异融合得到 `token_vec`
8. **Chromosome id 注入**：`chr_embedding(chr_idx)` 拼入 fused feature（16维可学习 embedding）
9. **最终 embedding**：所有特征拼接后经 MLP（Linear → LayerNorm → GELU → Dropout）输出 256 维向量

### Chromosome id 被使用了两次

- **第一次**：在 pair encoder 内部，影响 embedding 形成
- **第二次**：在 prototype 层，`prototypes[chr_idx]` 选择当前染色体对应的原型集合

### Prototype Head

- 每种染色体各有 `num_prototypes=4` 个可学习 normal prototype（不共享）
- Anomaly score = 当前 embedding 到本染色体 4 个 prototype 的最小余弦距离
- 距离越大 = 越不正常

---

## 七、实验结果完整表

### P12 主干后处理消融（全部基于同一 P12 checkpoint）

| 方法 | 阈值策略 | F1 | Precision_abn | Recall_abn | Balanced Acc | 备注 |
|------|----------|------|------|------|------|------|
| 原始 anomaly score | 验证集全局最佳 | 0.2796 | 0.1855 | 0.5673 | 0.7253 | 基线，直接用默认阈值 |
| 原始 anomaly score | 训练正常样本第99百分位全局阈值 | 0.4803 | 0.4400 | 0.5288 | 0.7487 | train-normal quantile 有明显价值 |
| **按染色体 z-score 标准化** | **验证集全局最佳** | **0.5490** | **0.5600** | **0.5385** | **0.7593** | **当前最高 F1** |
| 按染色体 z-score 标准化 | 按染色体 q99 阈值 | 0.4459 | 0.3333 | 0.6731 | 0.8050 | 最高召回 + 最高 Balanced Acc |
| 按染色体稳健标准化（MAD） | 验证集全局最佳 | 0.5289 | 0.4638 | 0.6154 | 0.7910 | 接近 z-score 版本 |
| 按染色体稳健标准化（MAD） | 按染色体 q99 阈值 | 0.4459 | 0.3333 | 0.6731 | 0.8050 | 与 z-score 版本结果相同 |

### P12 Embedding + Memory Bank Scorer

| 方法 | k | 阈值策略 | F1 | Precision_abn | Recall_abn | Balanced Acc | 备注 |
|------|---|----------|------|------|------|------|------|
| 余弦距离 | 1 | 验证集全局最佳 | 0.5302 | 0.5135 | 0.5481 | 0.7619 | 很强，说明 P12 embedding 本身有用 |
| 余弦距离 | 3 | 验证集全局最佳 | 0.5302 | 0.5135 | 0.5481 | 0.7619 | 与 k=1 基本一致 |
| 余弦距离 | 3 | 按染色体 quantile 阈值 | 0.3578 | 0.2306 | 0.7981 | 0.8367 | 召回和 balanced acc 很高 |
| 欧氏距离 | 1 | 验证集全局最佳 | 0.4959 | 0.4296 | 0.5865 | 0.7750 | 能工作，不如余弦 |
| 余弦/欧氏距离 | 1 | 按染色体 quantile 阈值 | 0.0856 | 0.0447 | 1.0000 | 0.5000 | **崩溃**，原因见下 |

**Memory bank k=1 + chr-conditioned 崩溃原因**：训练正常样本在 memory bank 中与自身匹配（self-match），导致训练分数接近 0，估出的 q95/q99 阈值近零，测试时几乎全报异常。**修复方案**：leave-one-out 评分（排除自身匹配）。这不是方向错了，是实现细节问题。

### 失败实验

| 实验 | 关键指标 | 失败原因 |
|------|----------|----------|
| P13 | 整体变差 | 重新训练的 checkpoint 本身比 P12 差，后处理无法救回 |
| P14 | AUPRC=0.0602, F1=0.0247, Recall=0.0288 | 加入染色体监督对比目标，把模型从"正常性建模"拉偏到"染色体身份区分"，表示空间被毁 |
| P15 | 结果很差 | 通用 frozen 自然图像特征 + patch memory bank，没有 pair-aware 表征，没有 chr-conditioned 建模，冻结特征不适合细粒度染色体结构判断 |

**P14 的关键教训**：在这个任务里，增强 chromosome discrimination ≠ 增强 anomaly detection。目标函数如果错了，会毁掉整个表示空间。

---

## 八、消融验证链条总结

| 问题 | 验证结论 |
|------|----------|
| 单图 vs 配对输入 | 配对输入有明显价值，单图信息不足 |
| 异常监督 vs 正常分布建模 | normal-only 路线更接近主任务，能泛化到未见异常 |
| 轻量损失与风格增强 | 不是当前瓶颈，无法解决根本问题 |
| 增强 chromosome discrimination | 会伤害 anomaly detection，目标函数必须服务正常性建模 |
| 全局阈值 vs 按染色体阈值 | 正常分布确实是 chr-conditioned 的，按染色体分阈值效果显著更好 |

---

## 九、当前最佳 Operating Points

### Operating Point A：追求最高 F1
- **方法**：P12 输出按染色体 z-score 标准化 + 验证集全局最佳阈值
- **结果**：F1=0.5490, Precision=0.5600, Recall=0.5385, Balanced Acc=0.7593

### Operating Point B：追求更高召回与平衡准确率
- **方法**：P12 输出按染色体 z-score 标准化 + 每个染色体的训练正常样本 q99 作为阈值
- **结果**：F1=0.4459, Precision=0.3333, Recall=0.6731, Balanced Acc=0.8050

---

## 十、当前主线方向（不要偏离）

### 原则
- **不再另起炉灶**：不回到普通分类、不用通用 frozen anomaly detection
- **保持 P12 主干**：pair-aware + chromosome-conditioned + normal-only 是正确建模范式
- **改进方向在 scorer/calibration 层**，不在 backbone

### 优先级排序

**第一优先级**：修复 memory bank 的 self-match 问题
- 具体：train split 做 leave-one-out，禁止与自身匹配
- 重新评估 cosine k=1/k=3 + chr-conditioned 阈值

**第二优先级**：`num_prototypes` 超参数消融
- 当前固定 num_prototypes=4，从未系统对比
- 建议：1 / 2 / 4 / 8（/ 16 如果资源允许）
- 回答：正常 manifold 需要几个 mode 才够表达

**第三优先级**：P12 embedding + KMeans / GMM 统计化建模
- 在已验证有效的 embedding 上继续改 scorer
- 回答：多原型 normal manifold 统计化是否比 learnable prototype 更强

**不建议做**：Haar-like + AdaBoost 等手工特征路线
- 与当前成功机制不一致（不 pair-aware，不 chr-conditioned）
- 如果确实要做，只能定位为"传统手工方法对照实验"，不是主线升级

---

## 十一、尚未完成的实验

- 独立 `prototype bank` 后处理实验（只有工具文件 `src/utils/prototype_bank.py`，没有实验脚本和结果）
- KMeans / GMM 统计化 manifold 建模
- `num_prototypes` 超参数消融（1 / 2 / 4 / 8）
- Memory bank self-match 修复后的重新评估

---

## 十二、核心代码位置

| 模块 | 路径 |
|------|------|
| Pair encoder | `src/models/correspondence_interval_pair_model.py` → `CorrespondenceIntervalPairClassifier` |
| Prototype metric head | `src/models/multi_prototype_metric.py` → `MultiPrototypeMetricModel` |
| Prototype bank 工具 | `src/utils/prototype_bank.py` |

---

## 十三、一句话总结（报告用）

> 本项目将染色体倒位异常检测任务建模为"chromosome-conditioned 正常结构分布学习"问题，而非传统异常类别分类问题。当前最优方法 P12 基于同源染色体配对输入、仅用正常样本训练，在测试集上达到 F1=0.5490；进一步结合按染色体划分的 scorer 校准后，Balanced Accuracy 可提升至 0.8050。
