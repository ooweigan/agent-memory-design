---
number: 3
title: "统一价值函数"
fullTitle: "统一价值函数——记与忘的同一把尺"
date: "2026-05-24"
readingTime: "10"
navTitle: "统一价值函数"
---

### 3.1 问题形式化：记忆管理的本质是受限资源下的序贯决策



Agent 的存储是有限的，LLM 的上下文窗口是有限的，每次记忆操作（写入、检索、归纳）的计算预算是有限的。三个有限叠加，意味着记忆管理不是一个"存数据"的问题，而是一个**序贯决策问题**（sequential decision problem）：在每一个时间点，面对新到达的信息流，系统必须决定是否写入、以何种形式写入、以及淘汰哪些旧信息——所有这些决策的累积效果，决定了 Agent 在未来任务中的表现。



这个决策问题可以用 Markov Decision Process（MDP）的形式化语言精确描述。Sutton & Barto 在其经典教材 *Reinforcement Learning: An Introduction* 中给出了 MDP 的标准五元组定义：状态空间 $\mathcal{S}$、动作空间 $\mathcal{A}$、转移函数 $\mathcal{P}$、回报函数 $\mathcal{R}$、折扣因子 $\gamma$ [[31]](#ref-31)。映射到记忆管理场景：




| MDP 要素 | 记忆管理对应 |
| --- | --- |
| **状态 $s_t$** | 当前记忆库 $\mathcal{M}_t$ 的完整快照（所有 facet 的内容、置信度、证据链） |
| **动作 $a_t$** | 四类策略决策：是否写入（record）、如何编码（encode）、是否淘汰（forget）、是否反思（reflect） |
| **转移 $\mathcal{P}(s_{t+1} \mid s_t, a_t)$** | 写入/淘汰/反思后记忆库的新状态——由 L0 的五条不变量保证确定性 |
| **回报 $R(a_t, s_t)$** | Agent 在任务上的终端表现变化——这是 $R_{\text{task}}$ 的增量 |
| **折扣 $\gamma$** | 未来回报的衰减——越远期的预测越不确定，权重越低 |



形式化目标函数为：



$$\max_{\pi} \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R(a_t, s_t) \;\middle|\; \mathcal{M}_0 \right] \quad \text{s.t.} \quad |\mathcal{M}_t| \leq C,\ \sum_t \text{cost}(a_t) \leq B$$



用自然语言说：**在固定的存储容量 $C$ 和总计算成本预算 $B$ 下，让智能体未来做出的决策获得最高的折扣累积任务回报。** 一条记忆该不该留，只看它对完成终极目标的**边际价值**（marginal value）是否大于它占用的资源成本。



这里的"保留价值"概念，其理论根基可以追溯到信息论。Shannon 在 1948 年的奠基论文中定义了信息熵 $H(X) = -\sum_x p(x) \log_2 p(x)$，它度量的是一个随机变量的"惊奇程度"——越不可预测的事件包含越多的信息量 [[24]](#ref-24)。一条记忆的价值，粗略地说，可以用它对未来决策的**条件熵减少量**来度量：给定这条记忆 $m$，Agent 对未来最优动作的不确定性减少了多少？这就是互信息（Mutual Information）$I(\text{Action}; m \mid \text{context})$ 的含义。如果一条记忆对 Agent 的未来行为没有任何影响——无论有没有它，Agent 都会做出同样的决策——那它的互信息为零，保留价值为零。



但互信息只度量了"相关性"，没有度量"功利性"。一只猫闯进会议室，对于"会议内容"这个随机变量具有极高的互信息（它确实减少了对"会议发生了什么"的不确定性），但对于一个跟踪项目进度的 Agent 来说，这段记忆的任务回报贡献为零。因此，价值函数必须是**条件于任务回报的互信息**——不是 $I(X; m)$，而是 $I(R_{\text{task}}; m \mid \text{context})$。这个区分将在第 4 章展开讨论。



在 MDP 框架下，记忆管理策略 $\pi$ 就是一个从状态到动作的映射：给定当前记忆库的快照，决定做什么。最优策略 $\pi^*$ 是使累积回报最大化的那个策略。然而，直接求解这个 MDP 是不可能的——状态空间是所有可能 facet 组合的幂集，维度爆炸。因此，我们不追求全局最优，而是将问题分解为四个可独立近似的子决策：**记什么（admission control）、怎么记（encoding depth）、什么时候记（event-driven trigger）、什么时候挤（value-sorted forgetting）**。每个子决策共享同一个价值标尺，但在不同时间尺度上运作。






#### 3.1.1 有界最优性：在约束下做到最好



这里需要引入一个重要概念区分。Herbert Simon 在 1956 年提出了"有限理性"（bounded rationality）：真实世界的决策者不具备无限的计算资源来找到全局最优解，而是采用"满意即止"（satisficing）策略 [[25]](#ref-25)。Stuart Russell 在 1995 年进一步形式化了"有界最优性"（bounded optimality）：给定计算资源约束，最优的不是那个做出最优决策的程序，而是在约束内用最有效方式逼近最优的程序 [[26]](#ref-26)。



映射到记忆管理：我们不期望系统在每条信息到达时都精确计算其边际价值（这本身就需要解决整个 MDP），而是设计一个**在给定计算预算内最大化长期回报的近似策略**。3.2-3.5 节的四个子决策就是这个近似策略的具体设计——每一个都用启发式而非精确求解来平衡计算成本与决策质量。











### 3.2 "记什么"：基于增量价值的准入控制



智能体时刻都在产生观察和思考，不可能全部写入长期记忆。"记什么"就是在问：**这条新信息值得进入长期记忆吗？**



答案的理论基础来自 Kolmogorov 复杂度与最小描述长度（Minimum Description Length, MDL）原则。Rissanen 在 1978 年提出的 MDL 原则指出：给定数据集 $D$，最好的模型（假说）$H$ 是使 $L(H) + L(D \mid H)$ 最小化的那个——即模型自身的描述长度加上数据在模型下的编码长度之和最小 [[27]](#ref-27)。直觉上，一个好的模型应该既简洁又能很好地解释数据。映射到记忆系统：一条新记忆 $m$ 值得写入，当且仅当它的加入能减少记忆库对未来观察的总编码长度——即 $m$ 提供了"压缩增益"。



具体地，设当前记忆库为 $\mathcal{M}$，新观察为 $o$。定义写入 $m$ 的边际价值为：



$$\Delta(m, \mathcal{M}) = \underbrace{L(o \mid \mathcal{M})}_{\text{不加 }m\text{ 时的编码长度}} - \underbrace{L(o \mid \mathcal{M} \cup \{m\})}_{\text{加了 }m\text{ 后的编码长度}} - \underbrace{L(m)}_{\text{记忆自身的存储成本}}$$



只有当 $\Delta(m, \mathcal{M}) > 0$ 时，写入才是值得的。这个标准同时考虑了三个因素：



- **新颖性与冗余比**：如果 $m$ 与 $\mathcal{M}$ 中已有记忆高度冗余，则 $L(o \mid \mathcal{M}) \approx L(o \mid \mathcal{M} \cup \{m\})$，边际价值趋零。工程上可以用新信息与已有记忆库的语义相似度来近似——重复度越高，增量信息越低。
- **任务相关性**：与当前目标、活跃计划或高频检索模式高度相关的信息，未来调用概率极高。这是 $R_{\text{task}}$ 驱动的最直接体现：一条信息的价值最终取决于它是否帮助 Agent 在任务中做出更好的决策。
- **重要度启发**：来自发言者的强调（"这一点很关键"）、结构位置（开场白、总结陈词）、情绪标记等，都会赋予信息更高的先验重要性。这些是通用信号，在没有足够任务反馈时提供冷启动的入场分。



Schmidhuber 在 2009 年提出的"压缩进步"（compression progress）理论进一步阐明了这一点：好奇心（curiosity）的本质是**最大化压缩进步的速率**——智能体倾向于关注那些能让自己找到新规律、从而提升压缩能力的观察 [[28]](#ref-28)。一条记忆的价值不仅在于它对当前任务的直接贡献，还在于它开启了对未来更高效压缩的可能性——即它是一个"可被进一步压缩的原材料"。



准入机制不是设一个固定门槛，而是**只有当新信息的保留价值大于当前记忆库中最低价值条目的分数时，才触发写入**。这个条件同时解决了"记什么"和"挤什么"——当新信息价值足够高时，它自动触发对最低价值旧条目的淘汰。记与挤的决策用同一把尺，避免了准入标准与淘汰标准不一致导致的系统振荡。



从强化学习的角度，这个准入控制对应的是一个**exploration-exploitation 权衡**。Auer 等人在 2002 年提出的 UCB（Upper Confidence Bound）算法给出了一个优雅的解决方案：选择具有最高置信上界的动作 [[32]](#ref-32)。映射到记忆准入：对每条候选记忆，维护一个价值估计和一个不确定性范围。当记忆的价值估计高度不确定时（新类型的信息、从未被检索验证过），给予更高的探索奖励——倾向于先写入，留待未来验证。随着该类型信息被反复检索或忽略，不确定性下降，价值估计收敛，准入决策变得更加确定。这避免了"只记已知重要的东西"的陷阱——系统需要偶尔写入一些"不确定是否重要"的信息，才能发现新的重要模式。






#### 3.2.1 预测编码视角下的准入判断



预测编码（predictive coding）理论为准入控制提供了另一个视角。Rao & Ballard 在 1999 年的开创性论文中提出：大脑不是一个被动的记录装置，而是一个主动的预测机器 [[33]](#ref-33)。大脑持续生成对下一刻感官输入的预测，只有**预测误差**（prediction error）——实际输入与预测之间的差异——才会被向上传播到更高层的处理区域。预测准确的部分被静默"压缩"，只有意外部分消耗认知资源。



映射到记忆准入：Agent 维护一个基于当前记忆的"世界模型"，对每条新到达的信息生成预测。如果信息与预测高度一致（低预测误差），说明这条信息已被现有记忆充分编码，其边际价值低。如果信息出乎意料（高预测误差），说明现有记忆不足以解释世界，这条信息值得写入——因为它代表了世界模型需要更新的地方。



Karl Friston 的自由能原理（Free Energy Principle, 2010）将这一思想推向更一般的理论框架 [[34]](#ref-34)。自由能 $\mathcal{F}$ 是一个上界，度量了 Agent 内部模型与真实世界分布之间的 KL 散度 $D_{\text{KL}}[q(\theta) \| p(\theta \mid o)]$。最小化自由能等价于同时做两件事：（1）更新内部模型使其更好地匹配观察（perceptual inference，即"记"）；（2）主动选择行动来减少未来观察的不确定性（active inference，即"选择性记"）。记忆准入控制在自由能框架下，就是：只写入那些能有效降低自由能的观察——即那些能帮助内部模型更好地预测世界的信号。








#### 3.2.2 信用分配：一条记忆的未来价值如何归属



Sutton 在 1988 年提出的信用分配（credit assignment）问题指出：在时间序列决策中，一个动作的回报往往延迟到来，如何将最终回报正确地归因到沿途的每个动作，是一个根本性难题 [[35]](#ref-35)。TD（Temporal Difference）学习和 eligibility traces 提供了解决方案：每个状态-动作对维护一个"资格迹"，记录它在最近决策中的参与程度，回报到达时按资格迹分配。



映射到记忆系统：当一条记忆在三个月后的一次关键决策中被召回并发挥作用时，这个"功劳"不应该只归于这条记忆本身，而应该沿着它的引用链回传——它所依据的原始观察、它所属的 reflective facet、以及当初决定写入它的 record_policy 都分享了这份功劳。这种回传机制让策略的训练信号不是"这条记忆用了几次"这么粗糙，而是"这条记忆在什么情境下、通过什么路径、对什么任务产生了多少增量回报"——一个精确的信用分配图。











### 3.3 "怎么记"：编码深度的决策



同样一条信息"Hans 要求把付款周期从 45 天改为 30 天"，系统可以选择不同深度的编码：



- **浅层记录（原文/embedding）**：完整保存原始表述。保真度最高，但存储成本高，未来检索时也需要更多 token 来消费。
- **结构化摘要**：压缩为 "{主体: Hans, 属性: 付款周期, 旧值: 45天, 新值: 30天, 状态: 待确认}"。大幅降低存储成本，保留核心语义，但丢失了原文措辞的精确性。
- **反思/归因**：如果系统识别出这是 Hans 过去半年内第三次调整付款条款，它可以向上派生一条 reflective facet："Hans 有在交货前重新谈判付款条款的模式，可能与他的现金流周期有关。"这是价值密度最高的形式——它提供了跨实例的模式识别，能在未来相似场景中直接替代多次原始查询。



选择哪种深度，其理论基础是 Shannon 的**率失真理论**（Rate-Distortion Theory, 1959）[[29]](#ref-29)。率失真理论回答的核心问题是：**给定一个可容忍的失真（distortion）上限 $D$，信源编码所需的最小比特率 $R(D)$ 是多少？** 或者反过来：**给定一个比特率预算 $R$，能达到的最小失真是多少？**



在记忆编码场景中：




| 率失真概念 | 记忆编码对应 |
| --- | --- |
| 信源（Source） | 原始观察/交互流 |
| 编码（Encoding） | 选择的记录形式（原文、摘要、反思） |
| 比特率 $R$ | 每条记忆占用的 token/存储量 |
| 失真 $D$ | 编码后信息对原始观察的"损失"——在未来任务中导致次优决策的概率 |
| 率失真函数 $R(D)$ | 达到失真水平 $D$ 所需的最小编码量 |



率失真理论告诉我们：存在一个理论下界 $R(D)$，任何编码方案都无法在低于这个比特率的同时保持失真低于 $D$。浅层记录的 $R$ 高但 $D$ 低；深层压缩的 $R$ 低但 $D$ 高。最优编码深度选择就是在这个 tradeoff 曲线上找到与当前存储预算和任务容忍度对应的切点。



Tishby 等人在 1999 年提出的**信息瓶颈**（Information Bottleneck）方法进一步将这个框架推广到"保留与任务相关的信息、丢弃无关信息"的场景 [[30]](#ref-30)。信息瓶颈的目标是找到一个压缩表示 $T$（对应我们的"编码后记忆"），使得：



$$\min_{p(t \mid x)} \left[ I(X; T) - \beta \cdot I(T; Y) \right]$$



其中 $X$ 是原始输入（观察），$Y$ 是目标变量（任务回报），$T$ 是压缩表示（记忆编码），$\beta$ 是权衡参数。直觉上：最小化表示的复杂度 $I(X; T)$（存储成本），同时最大化表示对任务的信息量 $I(T; Y)$（任务价值）。$\beta$ 越大，系统越倾向于保留任务相关信息而丢弃其他；$\beta$ 越小，系统越倾向于保留更多原始信息。



这个框架精确地形式化了"怎么记"的决策：**编码深度由信息瓶颈的 $\beta$ 参数控制，而 $\beta$ 本身由外部任务目标 $R_{\text{task}}$ 驱动。** 客服场景下 $\beta$ 较大（只保留与问题解决相关的信息，其余可压缩），科研场景下 $\beta$ 较小（意外发现可能有长远价值，保留更多原始细节）。



这里有一个重要的工程分工：**提取深度与反思深度是两件不同的事。** 单次 LLM 调用（Extract 阶段）适合做模式识别和分类——把"付款周期 30 天"从邮件原文中抽取出来，标注为 semantic facet。但多步归因——"这是第三次调整，可能存在谈判模式"——不应该在 Extract 阶段做，应该在 Reflect 阶段做。把浅层提取和深层反思分开，既控制了单次调用的成本和延迟，也让"压缩即智能"有了明确的工程节奏：日常增量运行浅层提取，夜间或触发条件满足时运行深层反思。这两阶段各有不同的深度边界，分工清晰。








### 3.4 "什么时候记"：事件驱动的价值触发机制



在超长会议或持续对话中，Agent 不应该匀速记笔记——它应该在**关键瞬间突然提笔**。记录动作的触发是一个门控函数：当前信息的预估价值超过动态阈值时，打开写入窗口。



触发信号的理论基础同样可以从预测编码中找到。Rao & Ballard 的模型表明，神经元只在预测误差超过某个阈值时才发放脉冲——这在神经科学中被称为"surprise-gated encoding" [[33]](#ref-33)。映射到 Agent 记忆：



触发信号通常包括：



- **惊奇/预测误差（surprise）**：当实际内容与基于现有记忆的预测严重不符时，说明出现了高信息量事件。信息论中，surprise 可以用 $-\log p(o \mid \mathcal{M})$ 来量化——概率越低的事件，信息量越大。比如老师突然推翻前面结论，或客户说出完全出乎意料的诉求。
- **目标/意图相关性**：如果 Agent 带着明确的任务目标（如"会后写技术方案"），那么任何与任务语义相关的信息会立刻拉高价值分，触发记录，哪怕它表面平淡。
- **结构性转折点**：话题切换、总结陈词、明确列举"三个重点"等，是天然的记录时机。这些信号可以用 discourse structure analysis 来自动检测。
- **工作记忆缓冲区满**：类似人类听完一个子话题后集中整理，Agent 在语义段落结束或内部缓冲区饱和时，触发一次批量记录与抽象。这借鉴了认知心理学中 Baddeley 的工作记忆模型——语音环路（phonological loop）的容量有限，当满时触发向长期记忆的转写 [[36]](#ref-36)。



动态阈值随认知负荷和已记信息密度调节——当前正在高强度推理时，阈值升高，只记最关键的信息；当前处于低负荷观察状态时，阈值降低，更密集地记录。这避免了"要么全记要么不记"的粗暴二元选择。



从强化学习的角度，这个触发机制本质上是一个**bandit 问题**：在每个时间步，系统选择"记"或"不记"，"记"的动作有即时成本（计算、存储），"不记"有潜在的延迟成本（未来任务中缺失这条信息的风险）。UCB 算法的启示是：对于不确定性高的信号（首次出现的信息类型、从未验证过的模式），倾向于"记"——这对应了"宁可多记不可漏记"的保守策略。随着经验积累，触发阈值收敛到更加精确的水平。








### 3.5 "什么时候挤"：基于统一价值排序的遗忘



"挤"就是记忆淘汰。在容量超限或注意力预算紧张时，必须舍弃一部分。目标函数要求：**永远淘汰当前记忆库中"保留价值"最低的条目。**



保留价值可以用多因素加权来近似：



$$\text{Score}(m) = w_1 \cdot \text{Recency} + w_2 \cdot \text{Importance} + w_3 \cdot \text{Relevance} - w_4 \cdot \text{Redundancy}$$



- **Recency（新近性）**：刚访问过的更可能再被访问，但权重不宜过高，否则系统会陷入"只记最近的事"的短视。这个启发来自认知心理学中的"近因效应"（recency effect）——但长期记忆系统需要对抗这种偏差。
- **Importance（重要性）**：写入时赋予的长期先验重要度。
- **Relevance（与当前情境的相关度）**：当前任务激活的记忆，其未来效用会骤升。
- **冗余度惩罚**：与强记忆高度重叠的弱记忆，其独立价值打折。



从信息论角度，遗忘的目标可以重新表述为：**在存储预算 $C$ 的约束下，选择保留哪个子集 $\mathcal{M}' \subseteq \mathcal{M}$，使得 $I(R_{\text{task}}; \mathcal{M}' \mid \text{context})$ 最大化**——即保留的那个子集对任务回报的信息量最大。这是一个组合优化问题，在一般情况下是 NP-hard 的（子集选择问题），但贪心算法——每次淘汰对任务回报互信息贡献最小的那条记忆——在次模性（submodularity）条件下有 $(1 - 1/e)$ 的近似保证 [[37]](#ref-37)。



Rate-distortion 理论再次提供了理论支撑 [[29]](#ref-29)。遗忘本质上是一种有损压缩：将记忆库从 $\mathcal{M}$ 压缩到 $\mathcal{M}'$（$|\mathcal{M}'| < |\mathcal{M}|$），目标是在给定的存储预算下最小化任务相关信息的失真。



### 3.6 协同流水线：价值驱动记忆系统的完整运行时



将以上四个决策整合为一条闭环流水线：



1. **实时估价**：对每个新语块计算惊喜度（$-\log p(o \mid \mathcal{M})$）、任务相关性（$I(R_{\text{task}}; o \mid \text{context})$ 的近似）、与已有记忆的冗余度（$1 - \max_{m \in \mathcal{M}} \text{sim}(o, m)$），形成"记录紧迫性"分数。
2. **时机决策**：分数突破动态阈值时，打开写入窗口。
3. **内容与形式联合选择**：在窗口内摘取增量价值最高的子片段，立刻决定编码深度——首次出现的关键事实以结构化摘要写入；与刚记过的某条记忆高度关联则触发归并与更新；极其重要的顿悟生成高阶反思 facet。
4. **检索反馈自适应**：在后续任务中，被频繁命中的记忆，其所属的记录模式、触发阈值和内容过滤器都会得到正强化。这让未来的"什么时候记、记什么、怎么记"更贴合 Agent 的个性化任务需求。这也解释了为什么在同一堂课上每个人的笔记本都不同——各自的目标函数在持续塑造这条流水线。



Ng 等人在 1999 年的**reward shaping**理论提供了反馈自适应的理论保证 [[38]](#ref-38)。他们证明：如果中间奖励（shaping reward）是基于势函数（potential function）$\Phi$ 的形式 $F(s, s') = \gamma \Phi(s') - \Phi(s)$，那么 shaping 不会改变最优策略——它只是加速了学习。映射到记忆系统：检索频率、用户反馈等中间信号可以作为 reward shaping 加速策略学习，但它们的权重必须满足势函数条件，否则会引入偏差。这在工程上意味着：短期反馈信号应该以"增量"而非"绝对值"的形式参与价值更新——不是"这条记忆被检索了 5 次所以价值是 5"，而是"这次检索使它的价值从 $v$ 增加到 $v + \delta$"。






#### 3.6.1 多时间尺度的协同



四个子决策在不同时间尺度上运作，但共享同一个价值标尺，这形成了一个多时间尺度的自适应系统：




| 决策 | 时间尺度 | 触发信号 | 理论对应 |
| --- | --- | --- | --- |
| Admission（记什么） | 实时（每条信息到达） | 边际价值 > 淘汰阈值 | MDL / 信息瓶颈 |
| Encoding（怎么记） | 实时（写入时一次性决策） | 率失真切点 | Rate-distortion / 信息瓶颈 |
| Trigger（什么时候记） | 亚秒级（门控决策） | 预测误差 > 动态阈值 | 预测编码 / UCB |
| Forgetting（什么时候挤） | 事件驱动（容量压力时） | 库容量 > 阈值 | 子模优化 / 率失真 |



这种多时间尺度结构与认知心理学中的记忆巩固（memory consolidation）理论吻合。Atkinson & Shiffrin 的多存储模型（1968）提出感觉记忆（[[39]](#ref-39)，每一级有不同的编码深度和淘汰速率。Agent 记忆系统的四级决策管道在工程上复现了这种层级结构，但将其从"固定的时间窗口"泛化为"由价值函数驱动的动态门控"——信息不是因为"存在超过 30 秒"才被巩固到长期记忆，而是因为"它的保留价值超过了当前阈值"。











### 3.7 本章小结



统一价值函数是记忆系统的调度核心。它用同一把尺量"记"与"忘"，保证了系统在有限资源下的稳定收敛。四个策略——记什么、怎么记、什么时候记、什么时候挤——都是从同一价值标尺派生的具体决策面。



本章的理论贡献在于将四个决策面锚定到成熟的信息论与强化学习框架：



- **"记什么"** 的理论根基是 Kolmogorov 复杂度/MDL（$L(H) + L(D \mid H)$ 最小化）和 Schmidhuber 的压缩进步理论（好奇心 = 最大化压缩进步）。
- **"怎么记"** 的理论根基是 Shannon 的率失真理论（在给定失真约束下最小化编码率）和 Tishby 的信息瓶颈方法（保留任务相关信息、压缩无关信息）。
- **"什么时候记"** 的理论根基是预测编码（surprise-gated encoding）和 UCB 探索-利用权衡。
- **"什么时候挤"** 的理论根基是子模函数优化（贪心淘汰的近似保证）和率失真理论（遗忘 = 有损压缩）。



四个决策面通过统一的价值标尺协同工作——这把标尺的校准信号来自哪里？如果信号来自记忆系统内部（"被检索次数多就是重要"），系统会陷入自指循环——策略训练数据是记忆本身，记忆又被策略塑造，最终向"自己觉得重要的方向"漂移。这是第 4 章要解决的核心问题。

## 参考文献

[24] Shannon, C.E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379–423. [PDF](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)

[25] Simon, H.A. (1956). Rational choice and the structure of the environment. *Psychological Review*, 63(2), 129–138. [PDF](https://pages.ucsd.edu/~mckenzie/Simon1956PsychReview.pdf)

[26] Russell, S. (1995). Rationality and intelligence. *Proceedings of IJCAI*. [PDF](https://people.eecs.berkeley.edu/~russell/papers/ijcai95-cnt.pdf)

[27] Rissanen, J. (1978). Modeling by shortest data description. *Automatica*, 14(5), 465–471. [PDF](https://homepages.cwi.nl/~paulv/course-kc/mdlintro.pdf)

[28] Schmidhuber, J. (2009). Driven by compression progress. *Cognitive Science Monographs*. [arXiv](https://arxiv.org/abs/0812.4360)

[29] Shannon, C.E. (1959). Coding theorems for a discrete source with a fidelity criterion. *IRE National Convention Record*, Part 4, 142–163. [PDF](https://web.stanford.edu/class/ee376a/files/2017-18/lecture_12.pdf)

[30] Tishby, N., Pereira, F.C. & Bialek, W. (1999). The information bottleneck method. *Proceedings of the 37th Allerton Conference*. [arXiv](https://arxiv.org/abs/physics/0004057)

[31] Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. [Online](http://incompleteideas.net/book/the-book-2nd.html)

[32] Auer, P., Cesa-Bianchi, N. & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning*, 47(2–3), 235–256. [DOI](https://doi.org/10.1023/A:1013689704352)

[33] Rao, R.P.N. & Ballard, D.H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*, 2(1), 79–87. [DOI](https://doi.org/10.1038/1828)

[34] Friston, K. (2010). The free-energy principle: A unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2666703)

[35] Sutton, R.S. (1988). Learning to predict by the methods of temporal differences. *Machine Learning*, 3(1), 9–44. [DOI](https://doi.org/10.1007/BF00115009)

[36] Baddeley, A.D. (1992). Working memory. *Science*, 255(5044), 556–559. [DOI](https://doi.org/10.1126/science.1736359)

[37] Nemhauser, G.L., Wolsey, L.A. & Fisher, M.L. (1978). An analysis of approximations for maximizing submodular set functions—I. *Mathematical Programming*, 14(1), 265–294. [DOI](https://doi.org/10.1007/BF01588971)

[38] Ng, A.Y., Harada, D. & Russell, S. (1999). Policy invariance under reward transformations. *ICML*. [PDF](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)

[39] Atkinson, R.C. & Shiffrin, R.M. (1968). Human memory: A proposed system and its control processes. In K.W. Spence & J.T. Spence (Eds.), *The Psychology of Learning and Motivation*, Vol. 2. Academic Press. [DOI](https://doi.org/10.1016/S0079-7421(08)60422-3)

