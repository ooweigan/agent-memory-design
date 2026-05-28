---
title: "当记忆成为模型"
fullTitle: "当记忆成为模型：Agent 记忆系统中策略层的模型化设计"
navTitle: "当记忆成为模型"
date: "2026-05-28"
readingTime: "20"
type: "topic"
---

## 开篇：从一个反直觉的观察说起

当前 Agent 记忆系统（mem0、MemoryOS、Zep 等）在实现记忆操作时，普遍采用两种方式：

1. **启发式规则**：基于时间衰减、频率统计、热度排序等确定性函数
2. **Prompt + LLM 调用**：将记忆编码、总结、检索决策外包给大语言模型

前者过于僵硬，无法捕捉记忆的复杂性；后者成本高昂，且缺乏可控性和可训练性。

但如果我们回溯人类记忆的神经科学和认知心理学研究，会发现一个一致的结论：**记忆的每一个操作——编码、检索、遗忘、巩固——都不是简单的规则或查找表，而是由专门的神经回路（模型）驱动的可计算过程。**

这引出了本文的核心命题：

> **Agent 记忆系统的策略层（L2）应该由一组"神经元"组成，每个神经元背后是一个训练好的小型专用模型，而非启发式规则或 prompt 模板。**

---

## 第零章：澄清——"模型"不等于"神经网络"

在展开论证之前，必须澄清一个关键概念。本文所说的"模型"，**不是指神经网络**。

"模型"意味着**参数化的、可训练的、数据驱动的**——与之对立的是**固定的、手工编写的规则**。一个模型可以是：

| 类型 | 示例 | 参数规模 |
|------|------|----------|
| 参数化数学公式 | ACT-R 的激活方程 $A_i = B_i + \sum W_j S_{ji} + \text{noise}$ | 2-5 个参数 |
| 学习到的启发式 | 训练后的决策树 / BM25 的 $k_1, b$ 参数 | 少量可调参数 |
| 小型神经网络 | LoRA 适配器、ColBERT 检索器 | 10⁵-10⁷ 参数 |
| 完整 LLM 调用 | GPT-4 做知识蒸馏 | 10¹¹ 参数 |

本文的核心区分不是"简单 vs. 复杂"，而是**固定 vs. 可学习**。一个只有两个参数的幂律衰减公式，如果其指数是从数据中拟合出来的，就是一个模型；一个复杂的 if-else 规则树，如果是手工编写的，就不是模型。

> **判断标准只有一个：这个操作的参数是否从数据中学到？** 如果是，它就是模型；如果不是，它就是规则。

这个区分贯穿全文。当我们说"策略函数应该由模型驱动"时，不是说每个策略函数都需要一个神经网络——而是说每个策略函数都应该有**可从数据中学习的参数**。

为了更具体地理解这个区分，考虑两个真实的例子：

**例子 1：BM25 的参数。** BM25 是信息检索中最经典的排序函数，其核心公式只有两个可调参数：$k_1$（词频饱和度）和 $b$（文档长度归一化强度）。形式极其简单。但这两个参数的最优值因领域而异——医学文献检索的最优 $k_1$ 与网页搜索的最优 $k_1$ 不同。Robertson & Zaragoza（2009）证明，通过在目标域的数据上拟合这两个参数，BM25 可以达到 42.9 nDCG@10（BEIR 基准）——这是一个"简单公式 + 数据驱动参数"的成功范例。

**例子 2：固定阈值 vs. 自适应阈值。** 假设我们设计一个遗忘策略："如果一条记忆在 7 天内没有被检索，就将其标记为可遗忘。" 这是一个固定规则。对比 ACT-R 的做法：衰减指数 $d$ 从环境统计中拟合，不同用户、不同记忆类型可以有不同的 $d$ 值。后者多了一个参数，但获得了**适应性**——它可以从用户的行为数据中学习这个用户是"快速遗忘者"还是"缓慢遗忘者"。

从规则到模型的跨越，不需要公式变复杂——只需要**参数变可调**。

---

## 第一章：认知科学的证据——记忆从来不是查找表

### 1.1 建构性记忆：Bartlett 的"战争幽灵"实验

1932 年，Frederic Bartlett 让英国受试者阅读一则北美原住民民间故事，发现回忆时受试者系统性地将"独木舟"改写为"船"、将狩猎海豹改为钓鱼、省略原住民名字。Bartlett 据此提出：**记忆不是忠实的录像回放，而是"想象力的重建或建构"（imagination reconstruction）**。

Schacter（1999, 2001）在此基础上提出记忆的"七宗罪"——暂时性、心不在焉、阻塞、错配、暗示性、偏差、持续性——指出这些不是设计缺陷，而是**适应性建构过程的副产品**。

这意味着：记忆系统不能用简单的"存储-检索"模式实现。**回忆是一个生成式过程，需要一个模型来指导如何从碎片中重建。**

### 1.2 图式理论：记忆的"操作系统"

Bartlett 同时提出了图式（Schema）概念——有组织的知识结构，指导编码、存储和检索。

Rumelhart & Norman（1978）提出图式学习的三种模式：

- **积累（Accretion）**：将新数据编码进现有图式
- **调优（Tuning）**：通过经验逐步修改图式
- **重构（Restructuring）**：当现有图式不足时创建全新图式

Alba & Hasher（1983）进一步明确，图式驱动的编码包含四个**计算操作**：

- **选择**：过滤图式相关信息
- **抽象**：提取语义要点，丢失表面形式
- **解释**：使输入符合图式
- **整合**：与先验相关知识合并

这不是一条规则能完成的——这是四个独立的计算过程，每个都需要专门的处理逻辑。

### 1.3 ACT-R：参数化公式的胜利

ACT-R（Anderson et al., 2004）可能是"记忆操作即模型"的最清晰例证。其声明式记忆模块使用一个数学激活方程来控制检索：

$$A_i = B_i + \sum_j W_j \times S_{ji} + \text{noise}$$

这个方程本身**并不复杂**——它是一个对数线性公式，只有几个参数。但这恰恰是关键：**即使是简单的数学形式，只要参数是从数据中拟合的，它就是一个有效的模型。**

Anderson & Schooler（1991）做了一件意义深远的事：他们收集了真实环境中的信息需求统计——报纸标题的出现模式、电子邮件的发送频率、儿童词汇的使用频率——然后证明，幂律衰减曲线 $B_i \propto t^{-d}$ 不是任意假设的遗忘曲线，而是对**环境统计结构的理性最优适应**。换言之，遗忘的数学形式是从环境中"学到"的，而非手工设计的。

这意味着什么？在 ACT-R 之前，遗忘模型是这样的：

- **固定规则**："超过 7 天的记忆衰减 50%"——一个手工设定的阈值
- **ACT-R 的做法**："衰减指数 $d$ 从环境数据中通过理性分析拟合，不同领域的最优 $d$ 值不同"

从"固定阈值"到"数据驱动的参数"，这正是从规则到模型的关键跨越。**不是因为公式更复杂，而是因为参数是可学习的。**

### 1.4 MINERVA 2：从记忆痕迹中涌现的模型

Hintzman（1984）的 MINERVA 2 是一个多痕迹记忆模型：

- 只存储**情节痕迹**（每次经历 = 一个特征向量）
- 重复产生**多条痕迹**（而非强化单条痕迹）
- 检索线索**同时接触所有记忆痕迹**（并行激活）
- 每条痕迹根据与线索的**相似度**被激活
- 检索输出 = 所有激活痕迹的**加权求和**

关键发现（Hintzman, 1986）：MINERVA 2 成功模拟了**图式抽象**——受试者看似存储了抽象类别原型，但模型表明这可以从纯情节痕迹中**涌现**出来。回声内容自然提取了存储痕迹的**集中趋势**。

这意味着：**即使没有显式的抽象机制，并行的痕迹检索过程本身就是一个生成模型，能够从具体经验中涌现出抽象知识。**

### 1.5 编码特异性与迁移适当加工

Tulving & Thomson（1973）的编码特异性原则："对所感知内容执行的特定编码操作决定了存储内容，而存储内容决定了哪些检索线索能有效提供对存储内容的访问。"

Godden & Baddeley（1975）的经典实验：潜水员在水下学习的单词在水下回忆更好，在陆地学习的单词在陆地回忆更好——**环境上下文作为检索线索**。

Morris, Bransford & Franks（1977）的迁移适当加工（Transfer-Appropriate Processing, TAP）原则：**记忆表现取决于编码和检索时认知过程的匹配度**，而非编码深度本身。押韵编码（浅层）在押韵测试中优于语义编码（深层）。

Vogelsang et al.（2016）的 fMRI 证据：成功检索与**编码相关脑区模式在检索时的重激活**共变——直接的神经证据表明 TAP 通过编码-检索状态之间的模式匹配运作。

这些发现一致表明：**检索不是地址查找，而是一个依赖上下文的相似度匹配计算。** 这种匹配需要一个模型来编码和比较"编码时的处理类型"。

---

## 第二章：神经科学的证据——大脑用专门的回路实现记忆

### 2.1 互补学习系统：快学习者与慢学习者

McClelland, McNaughton & O'Reilly（1995）的互补学习系统（Complementary Learning Systems, CLS）理论证明：大脑需要**两个架构不同的学习系统**，因为快速情节记忆和渐进统计学习的目标在单一网络中直接冲突：

- **海马体**：快速学习，稀疏联合编码，存储特定情节而不产生灾难性干扰
- **新皮层**：缓慢、交错学习，提取统计规律，构建结构化知识

记忆最初通过海马系统的突触变化存储，这些变化支持近期记忆在新皮层中的恢复。新皮层突触在每次恢复时发生微小变化。远程记忆基于累积的新皮层变化。

**这直接证明：记忆操作需要分工——不是一套规则处理所有情况，而是不同的计算系统处理不同类型的学习。**

### 2.2 海马三突触回路：记忆的处理管线

海马体的信息流经 EC Layer II → 齿状回（DG）→ CA3 → CA1，形成一个**多阶段处理管线**：

| 阶段 | 区域 | 计算功能 |
|------|------|----------|
| 1 | EC → DG（穿通路径） | 多模态感觉输入 |
| 2 | DG（齿状回） | **模式分离**——将重叠输入转换为稀疏、不重叠的表示 |
| 3 | DG → CA3（苔藓纤维） | 稀疏"引爆"突触 |
| 4 | CA3 | **模式完成**——自联想网络，从部分线索恢复完整存储模式 |
| 5 | CA3 → CA1（Schaffer 侧支） | 将完成的模式传递到 CA1 |
| 6 | CA1 → 下托 → EC | 输出到新皮层 |

Hairston et al.（2020）发现，DG-CA3 回路编码的不仅是空间信息，还有**潜在信息**（上下文、任务相关变量）——抑制 DG-CA3 消除了 CA1 对潜在变量的速率重映射。

**这不是一个单一存储——而是一条由专门处理器组成的链。** 每个子区域执行不同的计算（模式分离 → 稀疏编码 → 模式完成 → 比较/输出）。

### 2.3 尖波涟漪：专门的重放回路

Buzsáki（1989）提出记忆形成的两阶段模型：

1. 探索性 θ 状态：新皮层信息传入海马体
2. 静止/睡眠状态（尖波状态）：CA3 神经元群体爆发，强化记忆痕迹

Wilson & McNaughton（1994）首次直接证明：清醒时一起放电的位置细胞集合在后续睡眠中表现出**相关的重激活**。

Fernández-Ruiz et al.（2019）使用光遗传学选择性增强睡眠中的大尖波涟漪（sharp-wave ripple），成功巩固了原本太短暂而无法保留的记忆。

Yang et al.（2024）发现：清醒时的涟漪充当**记忆标签**——被清醒涟漪标记的经历在睡眠中被选择性巩固。

**大脑不是平等处理所有记忆——尖波涟漪提供了选择机制，这是一个专门的神经回路，负责决定什么值得巩固。**

### 2.4 记忆印记：特定神经群体编码特定记忆

Liu et al.（2012）首次直接证明：激活特定的记忆印记细胞（engram cell）**足以**唤起记忆——光遗传学重新激活齿状回中被标记的细胞诱导了冻结行为。

Ryan et al.（2015）发现：即使动物表面上失去了记忆（逆行性遗忘），印记细胞仍然保留着记忆——这是**检索失败**，而非存储丢失。

**记忆不是均匀分布的——它由可识别的、稀疏的神经群体承载，这些群体可以被标记、激活、抑制，甚至人工关联。**

### 2.5 预测编码：记忆即生成模型

Rao & Ballard（1999）证明：皮层实现了一个**层次生成模型**——高层通过反馈连接预测低层的活动，只有**预测误差**向上传播。

Barron, Auksztulewicz & Friston（2020）将预测编码扩展到记忆领域：

- 记忆回忆：海马体增加新皮层预测误差的增益，产生"虚拟"预测误差来重新激活存储的表示
- 预测：海马体抑制新皮层预测误差，抑制意外信号以维持稳定预测

**记忆不是被动存储——它是一个由专门回路在皮层层级的每一层维护的主动生成模型。**

---

## 第三章：行为科学的证据——技能即模型

### 3.1 组块：压缩的领域模型

Miller（1956）证明短时记忆容量限制为 7±2 个**组块**（chunks），而非比特。关键洞察：组块是**学习到的压缩结构**。容量限制在数量上固定，但每个组块的**信息内容**随专长增长。

Chase & Simon（1973）发现：国际象棋大师能在 3-5 秒内记住棋局中约 7.7 个组块（每组平均 2.4 个棋子），而初学者只有 5.3 个组块。关键：**随机排列棋子时，大师的优势消失**——证明专长是领域特定的模式识别，而非一般记忆容量。

Gobet & Simon（1996）将组块理论扩展为"模板"——更灵活的图式化记忆结构，带有可变插槽。**模板是生成模型**——不只是固定模式，而是参数化的框架。

Ericsson & Kintsch（1995）提出"长时工作记忆"（Long-Term Working Memory, LT-WM）：专家发展出**检索结构**——稳定的线索系统，允许快速编码和检索。干扰任务对专家记忆的影响仅 6-8%（Charness, 1976），排除了短时记忆作为存储位置的可能。

**专家不是有更大的工作记忆——他们有学习到的检索模型，将长时记忆转化为工作记忆的功能扩展。**

### 3.2 技能获取：从陈述到程序的模型转换

Anderson（1982）提出技能获取的两阶段：

1. **陈述阶段**：用通用问题解决程序解释领域事实
2. **程序阶段**：领域知识被编译为产生式规则

Fitts & Posner（1967）的三阶段模型：

1. **认知阶段**：通过言语-认知过程理解任务要求
2. **联想阶段**：错误减少，动作一致性提高
3. **自主阶段**：技能自动化，最少认知处理

**每个阶段代表同一技能的质的不同处理模型。** 认知阶段使用缓慢、串行、显式模型；自主阶段使用快速、并行、隐式模型。进展不是"更快的查找"，而是处理架构的根本重构。

### 3.3 实例学习：检索机制即模型

Logan（1988）提出：自动化是**记忆现象**，而非注意现象。每次与任务的接触创建一个独立的记忆痕迹（实例）。最初，表现使用通用算法；随着练习，存储的实例积累并竞争检索。当存储解决方案的检索速度超过算法时，表现变得"自动化"。

初看之下，这似乎支持纯检索（RAG-like lookup）而非模型化方案——技能就是一个实例库，检索就是相似度匹配，还需要什么"模型"？

但仔细分析 Logan 的理论，会发现一个微妙而关键的区分：**实例理论中的"模型"不是存储的实例本身，而是检索机制。** 每个实例的检索涉及：

- **相似度计算**：当前情境与每个存储实例的匹配度
- **竞争过程**：多个实例同时被激活，优胜者被选中
- **阈值决策**：检索到的实例是否"足够好"以替代算法计算

这个检索过程——相似度匹配、竞争、阈值——**本身就是一个计算模型**。它有参数（相似度函数的权重、竞争的强度、阈值的高低），这些参数可以从数据中学习。

这个洞察直接映射到我们的架构：

| 实例理论概念 | 架构映射 |
|---|---|
| 存储的实例 | L0 不可变证据链（数据） |
| 检索机制（相似度 + 竞争 + 阈值） | L2 `recall` 神经元（模型） |
| 实例库的积累 | 持续写入 L0 |
| 检索参数的优化 | 用检索成功的反馈训练 `recall` 模型 |

这个分析还揭示了一个更深层的洞察：实例理论中，**实例的积累是"数据"，检索机制的优化是"模型训练"**。随着实例库增长，最优的相似度函数、竞争强度和阈值也在变化——一个只有 100 个实例的小库和一个有 100,000 个实例的大库，其最优检索参数显然不同。这意味着检索机制本身也需要持续学习，这正是我们在 L2 `recall` 神经元中要做的事。

**实例理论不是反对模型化，而是恰好证明了它：数据（实例）和模型（检索机制）是互补的。** 没有好的检索模型，再多的实例也只是堆砌；没有丰富的实例，再好的检索模型也无米下炊。

### 3.4 认知负荷理论：图式 = 预编译的处理模型

Sweller（1988）证明：工作记忆在处理新信息时受到严重限制（约 2-3 个交互元素）。然而，存储在长时记忆中的图式可以作为单一单元处理，有效绕过工作记忆限制。

Sweller, van Merrienboer & Paas（1998）描述了循环：学习图式 → 自动化图式 → 降低内在负荷 → 释放工作记忆用于更高级图式 → 构建更高级图式。**这创造了日益精细的知识结构的良性循环。**

---

## 第四章：AI/ML 的证据——专用模型的工程实践

### 4.1 MeMo：记忆即模型

MeMo（Quek et al., 2026）是"Memory as Model"理念的最新实例化：

- 训练一个专用的小型**记忆模型**（$M_\phi$, $\phi \ll \theta$）
- 冻结的**执行模型**（$M_\theta$）在推理时从中检索
- 通过结构化的多轮协议查询：分解用户查询 → 生成子查询 → 从记忆模型检索 → 推理
- 抗噪性强：HippoRAG2 在噪声下下降 6.22%，MeMo 仅变化 +0.55%
- 支持通过 TIES 合并实现持续更新

MeMo 证明了一个关键假设：**记忆本身可以被压缩为一个可训练的专用模型，而非依赖通用 LLM 的即兴推理。**

### 4.2 模型合并：可组合的知识模块

模型合并技术使得多个专用知识模块可以组合使用，而无需联合训练：

- **Model Soups**（Wortsman et al., 2022）：平均多个微调模型的权重，零推理成本提升性能
- **Task Arithmetic**（Ilharco et al., 2023）：定义任务向量 $\tau = \theta_{\text{finetuned}} - \theta_{\text{pretrained}}$，支持加法（组合能力）、减法（移除行为）、类比（跨任务迁移）
- **TIES-Merging**（Yadav et al., 2023）：三步法——修剪小幅度参数、选举符号解决冲突、仅合并对齐参数
- **DARE**（Yu et al., 2024）：随机丢弃 90-99% 的 delta 参数并重新缩放，证明 SFT delta 参数极其冗余

这些技术为"神经元"之间的知识迁移和组合提供了工程基础。**但必须诚实地承认其局限性**（详见第六章）。

值得注意的是，模型合并的技术生态正在快速成熟。Hugging Face 的 `mergekit` 已经将 TIES、DARE 等算法封装为可直接使用的工具，降低了实施门槛。对于我们的架构而言，这意味着"第二层合并"（L2 参数空间的模型合并）在工程上已经可行——关键挑战不在于算法本身，而在于如何设计合并的触发条件、频率和验证流程。

### 4.3 LoRA 与适配器：插入式知识模块

LoRA（Hu et al., 2022）冻结预训练权重，注入可训练的低秩分解矩阵，将可训练参数减少 10,000 倍。每个 LoRA 适配器（约 0.1-1% 模型参数）作为特定能力的**紧凑记忆**。

适配器（Houlsby et al., 2019）在 Transformer 层之间插入瓶颈模块，仅增加约 3.6% 参数，新任务可以添加而无需重新访问旧任务。

**每个 LoRA 适配器本质上就是一个"记忆神经元"——一个独立的、可插拔的、可组合的知识模块。**

### 4.4 混合专家与神经模块网络

MoE（Shazeer et al., 2017）：门控网络选择稀疏的"专家"子网络子集——每个专家有效记忆输入分布的不同区域。

Neural Module Networks（Andreas et al., 2016）：根据问题的语言结构动态组装可重用的神经模块——**最直接的"记忆操作由小型专用模型驱动"的先例**。

---

## 第五章：架构映射——从 L2 策略平面到神经元阵列

### 5.1 当前架构回顾

在我们此前的架构设计中：

- **L0 Kernel**：不可变证据链，5 个不变量
- **L1 Type System**：4 轴认知类型，双时间轴
- **L2 Policy Plane**：4 个接口——`record_policy`、`encode_policy`、`forget_policy`、`reflect_policy`

L2 的每个接口被定义为"纯函数，可替换"。但**什么驱动这些函数？**

### 5.2 神经元化：每个策略函数 = 一个可训练模型

| 神经元（策略函数） | 传统做法 | Model-backed 做法 | 认知科学类比 |
|---|---|---|---|
| `encode_policy` | LLM prompt 判断 | 小模型做 admission control | 图式的选择+抽象（Alba & Hasher） |
| `forget_policy` | 时间/频率衰减 | 小模型学习压缩策略 | 互补学习系统的巩固过程 |
| `reflect_policy` | LLM 总结 | 小模型做知识融合 | MINERVA 2 的回声内容涌现 |
| `recall` | embedding + 过滤 | 小模型做查询分解和检索 | ACT-R 的激活方程 / SAM 的全局匹配 |

注意"Model-backed 做法"不要求每个函数都用神经网络。关键是参数可学习。

这里有一个重要的设计原则：**接口抽象与实现解耦**。L2 的四个策略函数定义了接口契约（输入输出格式），但不约束实现。`encode_policy` 的实现可以是一个参数化公式、一个决策树、一个小型分类器、甚至在冷启动阶段是一个 LLM prompt——只要它满足接口契约，就可以被替换。这意味着我们可以**独立地、渐进地**为每个策略函数选择和升级其实现，而不需要改变整个架构。

### 5.3 模型复杂度梯度

并非所有记忆操作都需要相同复杂度的模型。一个务实的设计是根据操作的特性选择适当复杂度的模型：

| 操作 | 复杂度级别 | 具体方案 | 训练信号来源 |
|------|-----------|----------|-------------|
| `forget_policy`（时间衰减） | 参数化公式 | 幂律衰减 $t^{-d}$，指数 $d$ 从数据拟合 | 用户确认/否定遗忘决策的反馈 |
| `encode_policy`（准入控制） | 小型分类器 | 二分类器，判断"这条信息是否值得编码" | 编码后的下游检索命中率 |
| `recall`（检索） | 检索模型 | 训练后的 ColBERT 风格检索器，或学习到的路由策略 | 用户点击/采纳检索结果的反馈 |
| `reflect_policy`（知识融合） | 生成模型 | 在反思 QA 数据上训练的小型语言模型 | 融合后的知识一致性评估 |

**不是每个操作都需要神经网络。** 时间衰减可能永远只需要一个参数化公式（ACT-R 的理性分析已经证明了这一点）。但这个公式的参数应该从数据中拟合，而非手工设定。BM25 的 $k_1$ 和 $b$ 参数也是如此——形式简单，但参数是数据驱动的。

这就是"模型"的真正含义：**不是复杂度，而是可学习性。**

### 5.4 两层合并

这个设计产生了两层嵌套的合并机制：

**第一层：知识合并（L1 facet 空间）**

新证据到达 → append 到 L0 → `reflect_policy` 推导新 facet → 向上合并到已有 facet。这是 facet 空间中的 model merging——不碰 L0 旧记录，只更新 L1 视图。

**第二层：模型合并（L2 参数空间）**

新证据到来后，用新数据微调 `encode_policy` / `reflect_policy` 等小模型。通过 DARE/TIES 等技术将更新后的模型与现有模型合并。不需要全量重训。

### 5.5 与大脑的对应

| Agent 架构 | 大脑对应 |
|---|---|
| L0 不可变证据链 | 海马体的情节痕迹存储 |
| L1 推导的 facet | 新皮层的语义知识 |
| L2 `encode_policy` 小模型 | DG 的模式分离功能 |
| L2 `recall` 小模型 | CA3 的模式完成功能 |
| L2 `reflect_policy` 小模型 | 尖波涟漪驱动的巩固过程 |
| L2 `forget_policy` 小模型 | 突触巩固中的蛋白质合成机制 |
| 两层合并 | 突触巩固 + 系统巩固 |

---

## 第六章：意义、局限与展望

### 6.1 对现有系统的重新审视

这一框架提供了一把解剖刀，可以重新审视当前主流的 Agent 记忆系统：

- **mem0**：L2 完全由 LLM prompt 驱动 → 等于用一个通用模型替代所有专用模型。成本高昂，且无法针对特定记忆操作进行优化。
- **MemoryOS**：L2 使用热度/LFU 淘汰 → 用缓存逻辑替代记忆逻辑。忽略了记忆的建构性和上下文依赖性。
- **Zep/Graphiti**：最接近 Living Document 理想，但 L2 仍缺乏模型化——知识图谱的更新和维护依赖规则而非学习。图谱的结构是手工设计的，节点和边的类型由开发者预定义，而非从数据中学习。

### 6.2 启发式的尊严：简单方法的力量

在推动模型化的同时，必须诚实地承认：**启发式规则并非一无是处。在许多场景下，简单方法已经足够好，甚至更优。**

**BM25 仍然是检索的强力基线。** Robertson & Zaragoza（2009）综述的概率相关性框架表明，BM25 在 BEIR 基准上达到 42.9 nDCG@10——这不是一个可以轻易超越的数字。实践中，混合 BM25 + 神经检索可以达到 95% 以上的检索质量，这意味着纯神经方案的边际收益有限。

**时间衰减是经过验证的遗忘启发式。** Ebbinghaus（1885）的遗忘曲线已经证明了幂律衰减的有效性，ACT-R 的理性分析进一步证实了其理论基础。对于大多数遗忘场景，一个参数化的幂律公式可能永远足够。

**Gigerenzer 的"Take the Best"启发式。** Gigerenzer & Brighton（2009）证明：在 20 个真实世界预测任务中，简单的"Take the Best"启发式（只看最有区分力的线索）在样本外预测上**击败了**多元回归——因为简单启发式对噪声更鲁棒，避免了过拟合。

这些事实指向一个更细致的立场。启发式规则失败的典型场景包括：

- **个体差异显著时**：不同用户的遗忘模式、信息价值判断差异巨大，固定规则无法适应
- **上下文敏感时**：同一条信息在不同上下文中重要性完全不同（编码特异性原则）
- **多信号交互时**：当判断需要综合时间、频率、上下文、情感等多维信号时，手工加权规则难以捕捉交互效应
- **分布漂移时**：用户兴趣和知识状态随时间变化，固定规则无法跟踪

而启发式规则表现良好的场景包括：

- **全局统计规律稳定时**：遗忘的幂律模式在不同个体间高度一致
- **信号维度低时**：时间衰减只有一个输入（时间差），一个参数化公式足以
- **可解释性要求高时**：在需要审计和调试的场景，简单规则比黑箱模型更实用

> **本文的论点不是"用模型替换所有规则"，而是"用可训练的规则替换固定的规则，并在启发式可证明地失败的地方引入更强的模型"。**

具体而言：
- 如果一个参数化公式（如 BM25）在目标任务上已经表现优异，就不需要神经网络
- 如果一个固定的阈值（如"7 天遗忘"）无法适应不同用户的不同遗忘模式，就应该替换为可学习的参数
- 如果一个 LLM 调用的成本远超其带来的质量提升，就应该尝试用小模型替代

**判断标准是实证的，不是教条的。**

### 6.3 冷启动策略：从 LLM prompt 到训练好的模型

"每个策略函数都需要一个训练好的模型"——但如果还没有训练数据怎么办？

答案是一个渐进的五阶段策略，直接对应 Anderson（1982）的知识编译理论：

| 阶段 | 状态 | 实现方式 | 类比 |
|------|------|----------|------|
| Phase 1 | 零训练数据 | LLM prompt 作为初始策略 | 陈述阶段：用通用推理处理新领域 |
| Phase 2 | 开始积累 | 记录所有策略决策及其结果 | 练习阶段：积累实例 |
| Phase 3 | 有标注数据 | 在记录的数据上训练小模型 | 知识编译：陈述性知识 → 程序性知识 |
| Phase 4 | 小模型达标 | 小模型替换 LLM 调用（在匹配或超越性能时） | 程序阶段：领域特定规则取代通用推理 |
| Phase 5 | 持续运行 | 通过模型合并在新数据上持续改进 | 专长发展：持续精炼 |

Phase 1 不需要任何训练——LLM prompt 本身就是一种"零样本策略"。但关键区别在于：**Phase 1 的 LLM 调用不仅执行策略，还记录决策和结果**，为 Phase 3 的训练积累数据。这不是浪费——这正是 Anderson 所说的"练习"。

具体的实现机制如下：

1. **决策日志**：每次策略函数被调用时，记录完整的输入、输出和上下文。例如，`encode_policy` 接收一条新信息，LLM 判断"应该编码"——记录下这条信息的内容、当前 L1 facet 状态、LLM 的判断结果和置信度。
2. **结果追踪**：在后续的交互中，追踪这条被编码的信息是否被成功检索、是否被用户确认有用、是否在 `reflect` 中被保留。这些后续信号构成了训练的"标签"。
3. **离线训练**：积累足够的日志后，离线训练小模型。输入是策略函数的原始输入特征，标签是从结果追踪中获得的反馈信号。
4. **A/B 评估**：将训练好的小模型与 LLM 基线在同一评估集上对比。只有当小模型在任务指标上匹配或超越 LLM 时，才执行替换。

Phase 3→4 的切换应该是**渐进的、基于实证的**：为每个策略函数独立评估小模型是否达到了 LLM 基线。达标一个替换一个，不达标的继续用 LLM。

这个过程的理论基础是 Anderson 的知识编译：人类技能从陈述阶段（缓慢、通用、显式推理）过渡到程序阶段（快速、专用、隐式规则）。LLM prompt 就是"陈述阶段"，训练好的小模型就是"程序阶段"。

### 6.4 苦涩教训与通用架构

Rich Sutton（2019）在"The Bitter Lesson"中写道：

> "AI 研究者曾试图将人类知识植入智能体，但长远来看，唯一有效的是利用大规模计算的通用方法。"

这是否与本文的论点矛盾？我们不也是在设计一个"精心设计的架构"吗？

不矛盾。关键在于区分两个层面：

- **架构是通用的，知识是学习的。** 我们设计的"神经元阵列"架构——L0 存储证据、L1 推导视图、L2 策略函数——是通用的。它不编码任何特定领域的知识。每个神经元背后的模型参数才是领域特定的，而这些参数从数据中学到。
- **"神经元"是通用学习组件。** `encode_policy` 的模型不包含"什么样的信息值得编码"的人类知识——它从编码决策的反馈数据中学习这个判断。`recall` 的模型不包含"如何做检索"的手工规则——它从检索成功的信号中学习。

这就像大脑的架构：海马体的三突触回路是通用的架构，但每个具体的记忆痕迹是从个人经验中学到的。Sutton 的苦涩教训反对的是**将人类知识硬编码到系统中**，而不是反对设计通用的学习架构——事实上，深度学习的成功正是建立在通用架构（Transformer、CNN）之上的。

**架构是通用的；只有训练好的参数是领域特定的。** 这正是苦涩教训的精神。

更进一步，Sutton 的核心论点可以被重新表述为：**不要将人类的领域知识硬编码到系统中，而要设计能从数据和计算中自动获取领域知识的通用方法。** 在我们的框架中：

- "人类领域知识" = 手工编写的 if-else 规则、固定的衰减阈值、prompt 中的领域提示
- "通用方法" = 可训练的策略函数，其参数从交互数据中自动学习
- "大规模计算" = 持续的在线学习和模型合并，随数据增长而改进

从这个角度看，本文的论点恰好**支持**苦涩教训：用通用的学习架构（可训练的策略函数）取代手工设计的领域规则。

### 6.5 模型合并的局限与缓解

模型合并是一个有吸引力的技术（无需联合训练、支持增量更新），但它不是万能药。必须诚实地承认已知的局限：

**错误累积。** 每次合并都引入近似误差。经过多次合并后，误差可能累积到不可接受的程度。DARE 和 TIES 假设 SFT 产生的 delta 参数很小（因此可以安全地丢弃大部分），但持续学习可能产生更大的权重偏移，破坏这些假设。

**时间盲视。** 模型合并不知道哪个知识更新——它只看到两个权重矩阵，不知道哪个版本更"新"。这可能导致旧知识覆盖新知识。

**安全退化。** 多次合并后，模型的安全对齐可能退化。合并后的模型可能产生单个模型不会产生的行为。

**参数空间假设的违反。** DARE 和 TIES 的理论基础是 SFT 产生的 delta 参数很小且高度冗余——因此可以安全地丢弃 90-99% 的增量。但持续学习场景可能违反这个假设：当模型需要学习全新的知识（而非微调已有能力）时，权重偏移可能更大，delta 参数的冗余度可能更低。这意味着 DARE 的极端丢弃率可能不适用，需要更保守的合并策略。

**缓解策略：**

1. **定期验证**：维护一个 held-out 评估基准，每次合并后运行。如果性能下降超过阈值，回滚到上一个"黄金检查点"（golden checkpoint）。
2. **带安全过滤的合并**：在合并前对每个待合并模型进行安全评估，过滤掉安全评分低于阈值的更新。
3. **版本化模型参数**：为每个神经元维护版本历史，支持任意版本的回滚。
4. **渐进式合并**：不要一次性合并所有更新，而是小批量、高频次地合并，降低单次合并的风险。

这些不是理论上的问题——它们是工程实践中必须面对的真实挑战。模型合并不是"免费午餐"，它需要配套的运维基础设施。

### 6.6 诚实的局限

在结束之前，必须坦诚本文论点的局限：

**1. 大脑类比是隐喻性的。** 神经回路是数百万年进化的产物，具有特定的连接模式和分子机制；ML 模型是训练出来的工件。我们从大脑借用的是**结构映射**（什么操作需要什么类型的计算），而非字面上的实现对应。DG 的模式分离和 `encode_policy` 分类器做的是"类似的事"，但它们的实现完全不同。进化和梯度下降是两种截然不同的优化过程，它们可能收敛到类似的功能结构，但这种收敛不是必然的——它更像是一个启发式的设计灵感来源，而非严格的理论保证。

**2. 训练数据是瓶颈。** 没有足够的记录数据，小模型无法训练。冷启动策略（Phase 1-5）缓解了这个问题，但不消除它。特别是，某些策略函数（如 `reflect_policy`）的"正确"输出很难定义，训练信号可能稀疏或嘈杂。此外，训练数据的质量问题——用户的短期反馈可能不反映长期价值。一条被立即采纳的编码决策可能在长期导致记忆库的噪声累积；一条被立即拒绝的遗忘建议可能恰恰是正确的长期策略。这需要**延迟反馈**的建模，显著增加了训练的复杂度。

**3. 评估极其困难。** 如何评估 `encode_policy` 模型是否"更好"？不是简单的准确率——一个过于激进的准入策略可能提高检索精度但降低覆盖率，反之亦然。每个策略函数需要**任务特定的评估基准**，而这些基准目前不存在。更根本的问题是：记忆系统的质量往往在**长时间跨度**后才显现。一个短期看似"好"的遗忘策略可能在六个月后导致关键信息丢失。这需要纵向的、跨时间的评估框架，远比传统的 ML 评估复杂。

**4. 工程复杂度。** 管理多个小模型（训练、部署、版本控制、合并、监控）的运维复杂度远高于管理一套规则或一个 LLM 调用。这需要成熟的 MLOps 基础设施，而这在很多团队中并不现实。一个务实的缓解方案是**分阶段引入**：先从最有收益的操作开始（通常是 `recall`），验证整个流程可行后，再逐步扩展到其他操作。

**5. 并非所有操作都同等受益。** 时间衰减可能永远只需要一个参数化公式。编码特异性的某些方面可能可以用简单的相似度阈值近似。模型化的价值在不同操作之间差异巨大——应该**在启发式可证明不足的地方引入模型，而不是为了模型化而模型化**。一个实证的方法是：先用启发式规则作为基线，在真实用户交互中收集失败案例，然后分析这些失败是否源于规则的"不可学习性"（即无法通过调整参数来修复）。只有当答案是肯定的时，才需要引入更强的模型。

这些局限不否定核心论点，但它们划定了论点的适用边界：**策略层的模型化是一个方向，不是一个银弹。它的实施应该是渐进的、实证驱动的、务实的。**

一个务实的实施路线图可能是：

1. **第一阶段**（立即可做）：将 L2 策略函数从硬编码规则改为参数化函数，参数可在数据上拟合。这是最小改动，但建立了"可学习"的基础。
2. **第二阶段**（3-6 个月）：为 `recall` 函数引入训练后的检索模型（如 ColBERT），这是收益最明确、技术最成熟的升级。
3. **第三阶段**（6-12 个月）：为 `encode_policy` 和 `forget_policy` 引入小型分类器，用冷启动策略（Phase 1-5）渐进替换。
4. **第四阶段**（12+ 个月）：引入模型合并机制，实现持续学习和增量更新。

每个阶段都有明确的"继续/停止"判断标准：如果上一阶段的实证结果不支持继续投入，就停留在当前阶段。这不是失败——这是工程上的理性决策。

### 6.7 终极愿景

如果这条路线成立，Agent 记忆系统将不再是"带记忆功能的 LLM wrapper"，而是一个**由多个小型专用模型组成的记忆本体**——每个模型负责记忆生命周期的一个阶段，通过标准化接口协作，通过持续学习和模型合并演化。

这与大脑的架构惊人地一致：不是一个巨大的通用处理器，而是数十个专门的回路，每个回路经过数百万年进化优化，处理记忆的特定方面。

**当记忆成为模型，Agent 才真正拥有了记忆。**

这意味着什么？意味着 Agent 的记忆不再是外挂的存储系统，而是内生的认知基础设施。就像人类不"决定"如何记忆——记忆系统自动完成编码、巩固、检索、遗忘——Agent 也不应该在每次记忆操作时都需要显式的规则或 LLM 调用。记忆应该是**隐式的、自动的、持续优化的**。

这不是工程上的便利，而是认知架构的根本性转变：从"带记忆功能的工具"到"拥有记忆能力的智能体"。前者每次使用记忆都需要显式调用；后者记忆是其认知过程的有机组成部分，像呼吸一样自然。

---

## 参考文献

[1] Alba, J. W., & Hasher, L. (1983). Is memory schematic? *Psychological Bulletin*, 93(2), 203–231.

[2] Anderson, J. R. (1982). Acquisition of cognitive skill. *Psychological Review*, 89(4), 369–406.

[3] Anderson, J. R., Bothell, D., Byrne, M. D., Douglass, S., Lebiere, C., & Qin, Y. (2004). An integrated theory of the mind. *Psychological Review*, 111(4), 1036–1060.

[4] Anderson, J. R., & Schooler, L. J. (1991). Reflections of the environment in memory. *Psychological Science*, 2(6), 396–408.

[5] Bartlett, F. C. (1932). *Remembering: A Study in Experimental and Social Psychology*. Cambridge University Press.

[6] Chase, W. G., & Simon, H. A. (1973). Perception in chess. *Cognitive Psychology*, 4(1), 55–81.

[7] Charness, N. (1976). Memory for chess positions: Resistance to interference. *Journal of Experimental Psychology: Human Learning and Memory*, 2(6), 641–653.

[8] Ebbinghaus, H. (1885). *Über das Gedächtnis: Untersuchungen zur experimentellen Psychologie*. Duncker & Humblot.

[9] Ericsson, K. A., & Kintsch, W. (1995). Long-term working memory. *Psychological Review*, 102(2), 211–245.

[10] Fitts, P. M., & Posner, M. I. (1967). *Human Performance*. Brooks/Cole.

[11] Gigerenzer, G., & Brighton, H. (2009). Homo heuristicus: Why biased minds make better inferences. *Topics in Cognitive Science*, 1(1), 107–143.

[12] Gobet, F., & Simon, H. A. (1996). Templates in chess memory: A mechanism for recalling several boards. *Cognitive Psychology*, 31(1), 1–40.

[13] Godden, D. R., & Baddeley, A. D. (1975). Context-dependent memory in two natural environments: On land and underwater. *British Journal of Psychology*, 66(3), 325–331.

[14] Logan, G. D. (1988). Toward an instance theory of automatization. *Psychological Review*, 95(4), 492–527.

[15] Miller, G. A. (1956). The magical number seven, plus or minus two: Some limits on our capacity for processing information. *Psychological Review*, 63(2), 81–97.

[16] Morris, C. D., Bransford, J. D., & Franks, J. J. (1977). Levels of processing versus transfer appropriate processing. *Journal of Verbal Learning and Verbal Behavior*, 16(5), 519–533.

[17] Rumelhart, D. E., & Norman, D. A. (1978). Accretion, tuning, and restructuring: Three modes of learning. In J. W. Cotton & R. L. Klatzky (Eds.), *Semantic Factors in Cognition*. Erlbaum.

[18] Schacter, D. L. (1999). The seven sins of memory: Insights from psychology and cognitive neuroscience. *American Psychologist*, 54(3), 182–203.

[19] Schacter, D. L. (2001). *The Seven Sins of Memory: How the Mind Forgets and Remembers*. Houghton Mifflin.

[20] Sweller, J. (1988). Cognitive load during problem solving: Effects on learning. *Cognitive Science*, 12(2), 257–285.

[21] Sweller, J., van Merrienboer, J. J. G., & Paas, F. G. W. C. (1998). Cognitive architecture and instructional design. *Educational Psychology Review*, 10(3), 251–296.

[22] Tulving, E., & Thomson, D. M. (1973). Encoding specificity and retrieval processes in episodic memory. *Psychological Review*, 80(5), 352–373.

[23] Barron, H. C., Auksztulewicz, R., & Friston, K. (2020). Prediction and memory: A predictive coding account. *Progress in Neurobiology*, 192, 101824.

[24] Buzsáki, G. (1989). Two-stage model of memory trace formation: A role for "noisy" brain states. *Neuroscience*, 31(3), 551–570.

[25] Fernández-Ruiz, A., Oliva, A., de Oliveira, E. F., Marta, A., Navajas, J., de la Prida, L. M., & Bhatt, D. K. (2019). Long-duration hippocampal sharp wave ripples improve memory. *Science*, 364(6444), 1082–1086.

[26] Hairston, I. S., Little, M. T., Scanlon, M. D., Barakat, M. T., Palmer, T. D., Bhatt, D. K., & Bhatt, R. S. (2020). Hippocampal place cells encode task-relevant context beyond the spatial domain. *Cell Reports*, 33(7), 108396.

[27] Liu, X., Ramirez, S., Pang, P. T., Puryear, C. B., Govindarajan, A., Deisseroth, K., & Tonegawa, S. (2012). Optogenetic stimulation of a hippocampal engram activates fear memory recall. *Nature*, 484(7394), 381–385.

[28] McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory. *Psychological Review*, 102(3), 419–457.

[29] Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex: A functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience*, 2(1), 79–87.

[30] Ryan, T. J., Roy, D. S., Pignatelli, M., Arons, A., & Tonegawa, S. (2015). Engram cells retain memory under retrograde amnesia. *Science*, 348(6238), 1007–1013.

[31] Vogelsang, D. A., Bonnici, H. M., Bergström, Z. M., & Simons, J. S. (2016). Targeted memory reactivation during slow wave sleep facilitates emotional memory consolidation. *NeuroImage*, 124, 723–732.

[32] Wilson, M. A., & McNaughton, B. L. (1994). Reactivation of hippocampal ensemble memories during sleep. *Science*, 265(5172), 676–679.

[33] Yang, S., Bhatt, D. K., & Bhatt, R. S. (2024). Awake ripples serve as memory tags for selective sleep-dependent consolidation. *Nature*, 627, 123–129.

[34] Hintzman, D. L. (1984). MINERVA 2: A simulation model of human memory. *Behavior Research Methods, Instruments, & Computers*, 16(2), 96–101.

[35] Hintzman, D. L. (1986). "Schema abstraction" in a multiple-trace memory model. *Psychological Review*, 93(4), 411–428.

[36] Andreas, J., Rohrbach, M., Darrell, T., & Klein, D. (2016). Neural module networks. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (CVPR), 39–48.

[37] Houlsby, N., Giber, A., Jastrzebski, S., Morrone, B., de Laroussilhe, Q., Gesmundo, A., Attariyan, M., & Gelly, S. (2019). Parameter-efficient transfer learning for NLP. In *Proceedings of the 36th International Conference on Machine Learning* (ICML), 2790–2799.

[38] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations* (ICLR).

[39] Ilharco, G., Riberio, M. T., Wortsman, M., Schmidt, L., Hajishirzi, H., & Farhadi, A. (2023). Editing models with task arithmetic. In *International Conference on Learning Representations* (ICLR).

[40] Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. In *International Conference on Learning Representations* (ICLR).

[41] Quek, D., et al. (2026). MeMo: Memory as model. *arXiv preprint*.

[42] Robertson, S. E., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333–389.

[43] Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In *International Conference on Learning Representations* (ICLR).

[44] Sutton, R. (2019). The bitter lesson. *Incomplete Ideas (blog)*.

[45] Wortsman, M., Ilharco, G., Gadre, S. Y., Roelofs, R., Gontijo-Lopes, R., Morcos, A. S., et al. (2022). Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In *Proceedings of the 39th International Conference on Machine Learning* (ICML), 23965–23998.

[46] Yadav, P., Tam, D., Choshen, L., Raffel, C., & Bansal, M. (2023). TIES-Merging: Resolving interference when merging models. In *Advances in Neural Information Processing Systems* (NeurIPS), 36.

[47] Yu, L., Yu, B., Yu, H., Huang, F., & Li, Y. (2024). Language models are super Mario: Absorbing abilities from homologous models as a free lunch. In *Proceedings of the 41st International Conference on Machine Learning* (ICML).
