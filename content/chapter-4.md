---
number: 4
title: "外部锚定与压缩边界"
fullTitle: "外部锚定与压缩边界——R_task 为什么必须从外面来"
date: "2026-05-24"
readingTime: "8"
navTitle: "外部锚定与压缩边界"
---

### 4.1 自指循环问题：如果价值信号来自记忆系统内部



第 3 章我们建立了一套价值驱动的记忆管理流水线，但有意回避了一个问题：**价值函数本身用什么信号来训练和校准？**



一个直觉的回答是"用记忆的使用频率"——被检索次数多的就是重要的，被遗忘的就是不重要的。这个回答在工程上很诱人，因为使用频率是系统内部可观测的信号，不需要外部标注。但它会导致一个致命的问题：**自指循环**。



假设记忆系统的 L2 策略由"facet 被检索到的次数"来驱动——经常被 recall 的 facet 获得更高的保留价值，从而更不容易被淘汰。这看起来很合理。但考虑长期运行的动态：哪些 facet 会被检索到，取决于 Agent 当前在面对什么问题；Agent 当前面对什么问题，取决于它之前记得什么。这就形成了一个闭合环路——**策略的训练数据是记忆本身，记忆又被策略塑造**。



Charles Goodhart 在 1975 年提出的 **Goodhart 定律**精确地描述了这个陷阱："当一个度量指标成为目标时，它就不再是一个好的度量指标"（When a measure becomes a target, it ceases to be a good measure）[[40]](#ref-40)。Donald Campbell 在 1979 年独立提出了几乎相同的观察，被称为 **Campbell 定律**："任何用于社会决策的量化指标，越是被用于决策，就越容易受到腐蚀压力和扭曲" [[41]](#ref-41)。



两条定律指向同一个认识论陷阱。映射到记忆系统：



- **度量** = "facet 被检索到的次数"
- **目标** = "保留有价值的记忆"
- **腐蚀** = 系统向"更容易被检索到"而非"对任务真正重要"的方向优化



在这个闭环里，系统没有任何来自外部的信号来纠正方向。一个早期被高估的 facet（比如一次偶然的多次检索）会获得更高的保留价值，从而更不容易被淘汰，从而在投影时更优先被看到，从而更频繁地被检索——这是一个正反馈强化循环（positive feedback loop）。而一个实际上对业务至关重要但恰好没有被检索到的 facet，会因"使用频率低"被衰减、被遗忘，永远失去被重新发现的机会。



这就是自指循环的灾难性后果：系统不是向"对任务真正重要"的方向优化，而是向"自己觉得自己重要"的方向漂移。通用信息论指标——新颖性、惊异度、信息增益——也面临同样的问题：它们能找出"统计上特殊"的事件，但无法区分"业务上重要"和"业务上无关"的特殊。会议室里突然闯进一只猫，新颖性极高；但如果 Agent 的任务是跟踪项目进度，这段记忆毫无价值。通用指标能给信息打分，但不能给信息打上正确的功利主义烙印。



这个困境在哲学上可以追溯到休谟的"是-应该"问题（is-ought problem）：从"事实如此"推不出"应该如此"。记忆系统内部可观测的所有信号（检索频率、时间衰减、语义相似度）都是"是什么"的描述，而"什么是有价值的"是一个"应该"的判断。要从描述跨越到规范，必须引入一个外部的、非描述性的锚点——这就是 $R_{\text{task}}$。






#### 4.1.1 贝叶斯视角下的自指循环



用贝叶斯决策理论的语言来表达，自指循环的问题更加清晰 [[42]](#ref-42)。一个理性的贝叶斯 Agent 在做记忆管理决策时，应该计算：



$$P(\text{valuable} \mid \text{signal}) = \frac{P(\text{signal} \mid \text{valuable}) \cdot P(\text{valuable})}{P(\text{signal})}$$



其中先验 $P(\text{valuable})$ 代表"在没有任何信号时，一条记忆有多大概率是有价值的"。当价值信号全部来自系统内部时，先验本身就是系统之前的决策产物——系统之前倾向于保留某类信息，导致当前对"有价值"的先验分布被扭曲。这是一个**先验漂移**（prior drift）问题：先验不是固定的，而是被系统的行动不断修改，最终可能漂移到与真实价值分布严重偏离的状态。



要防止先验漂移，需要定期用外部观察来"校准"先验——即用 $R_{\text{task}}$ 的实际观测来更新 $P(\text{valuable})$。这就是外部锚定的贝叶斯解释：$R_{\text{task}}$ 不是替代内部信号，而是为内部信号提供了一个校准基准（calibration anchor），防止先验在闭环中漂移。











### 4.2 Rtask 外锚：打破循环的唯一方式



打破自指循环的唯一方法，是在闭环上打一个外部的锚——**用一个不属于记忆系统内部的信号来定义"什么是好的记忆"**。这个信号就是 **$R_{\text{task}}$**：业务任务的终端回报。



- 对客服 Agent，$R_{\text{task}}$ 是问题的**一次解决率**——用户不需要为同一问题再次联系。
- 对编程 Agent，$R_{\text{task}}$ 是**交付功能的正确率**和**需求覆盖完整度**。
- 对会议纪要 Agent，$R_{\text{task}}$ 是**关键决策的召回率与准确率**——纪要中捕捉到的决策要点与实际发生的重要决策的对比。



$R_{\text{task}}$ 的关键属性是**外生性**（exogeneity）：它不是记忆系统自己计算出来的，不依赖于任何一条 facet 的内容，不由 L2 策略自己定义。它是业务世界对 Agent 工作的评价，在时间和因果链上都位于记忆系统的外部。正因为它是外生的，它不会参与自指循环——策略不能通过操纵记忆来改变 $R_{\text{task}}$，$R_{\text{task}}$ 是铁面无私的审计。



从因果推断的角度，$R_{\text{task}}$ 在因果图（causal graph）中是一个**外生节点**：记忆系统的操作可以影响中间变量（facet 的内容、检索的结果），但不能直接影响 $R_{\text{task}}$——$R_{\text{task}}$ 是由用户行为、业务环境等外部因素决定的。这个因果独立性正是 Goodhart 定律要求的"好指标"的条件：度量指标必须独立于被优化的过程。



这给"记什么、怎么记、什么时候记"提供了一个认识论上干净的优化目标：**一条记忆操作是好的，当且仅当在给定当前记忆状态下，它在未来导致了更高的 $R_{\text{task}}$。** 这个判断不由 Agent 自己说了算，由业务结果说了算。



在**逆强化学习**（Inverse Reinforcement Learning, IRL）的框架下，$R_{\text{task}}$ 可以被视为"环境的真实回报函数"。Ng & Russell 在 2000 年的开创性论文中研究了如何从观察到的最优行为中反推回报函数 [[43]](#ref-43)。映射到记忆系统：如果我们观察到某个垂类业务中表现最好的 Agent 的记忆管理策略，可以用 IRL 来反推出隐含的 $R_{\text{task}}$——它不一定能被显式表述，但可以从"什么记忆策略导致了好的业务结果"中推断出来。这为 $R_{\text{task}}$ 的工程化提供了另一条路径：不是显式定义回报函数，而是从成功案例中学习。








### 4.3 引理：通用最优记忆系统不存在——以及它为何反证了我们的架构



从 $R_{\text{task}}$ 的外生性可以直接推出一个看似自相矛盾的引理：



> **不存在一个通用的"最优"记忆系统。** 每个垂类业务的 $R_{\text{task}}$ 都不同。客服场景的最优记忆策略（优先记住客户的投诉历史和偏好条款）和编程场景的最优记忆策略（优先记住函数契约的变更历史和测试覆盖率）没有任何理由是一致的。如果把任何一个 $R_{\text{task}}$ 写死在记忆系统的内核里——比如把"recency × relevance"的权重固定为客服场景的黄金比例——那么这个内核在编程场景下就不再是最优的。



这个引理可以进一步用**无免费午餐定理**（No Free Lunch Theorem, NFL）来支撑。Wolpert & Macready 在 1997 年证明：在所有可能的问题分布上，没有任何一个优化算法优于随机搜索 [[44]](#ref-44)。映射到记忆管理：在所有可能的 $R_{\text{task}}$ 分布上，没有任何一个固定的记忆策略优于随机策略。这听起来虚无主义，但它的实际含义是：**策略的有效性完全取决于它与具体 $R_{\text{task}}$ 的匹配程度**——不存在一个"放之四海而皆准"的记忆策略。



这个引理乍一看似乎在反对"通用记忆组件"的提法。但仔细读，它恰恰是通用组件最坚实的论证。**正因为 $R_{\text{task}}$ 不可通用，L0 和 L1 才必须对 $R_{\text{task}}$ 完全无知，L2 才必须是可插拔的策略平面。** 内核对业务价值函数一无所知，反而能容纳所有业务价值函数。这就是 PostgreSQL 之于业务应用、LSM 之于 KV 引擎的位置感：内核只承诺不变量和算子，不替用户决定什么是"重要的数据"。



从 Solomonoff 归纳法和 AIXI 的角度看，这个引理有更深层的含义。Marcus Hutter 在 2005 年提出的 AIXI 模型是一个理论上的通用最优 Bayesian Agent：它用 Solomonoff 先验（基于 Kolmogorov 复杂度的先验分布）来对所有可能的环境假设加权，然后选择在所有环境下最大化期望回报的动作 [[45]](#ref-45)。AIXI 是不可计算的（Kolmogorov 复杂度不可计算），但它给出了一个理论上限。AIXI 的"最优性"依赖于一个关键假设：**环境是静态的、回报函数是固定的**。当回报函数随场景变化时（客服 vs. 编程），AIXI 需要重新求解——这正是我们说"$R_{\text{task}}$ 不可通用"的理论表达。



AIXI 还提供了一个深刻的洞察：**压缩 = 智能**。Solomonoff 先验 $P(M) = 2^{-K(M)}$ 对更简洁的环境假设赋予更高概率——这等价于说"越能压缩观察的模型越可能是对的"。映射到记忆系统：一个好的记忆系统应该倾向于保留那些能最大化压缩未来观察的记忆——这正是第 3 章 MDL 原则的 AIXI 推广。








### 4.4 Rtask 的代理化：当真实回报是稀疏且延迟的



$R_{\text{task}}$ 外锚在理论上干净，但在工程上有一个棘手的问题：**真实 $R_{\text{task}}$ 往往是稀疏和延迟的。** 客服的一次解决率要到用户好几天没再联系才知道。编程的交付正确率要经过多次迭代和测试才知道。会议纪要的关键决策召回率可能要等到几个月后项目复盘才知道。



而记忆系统的四个策略——record_policy、encode_policy、forget_policy、reflect_policy——必须在毫秒级做出。直接拿延迟 $R_{\text{task}}$ 做实时门控，等价于让 Agent 凭借"几个月后的反馈"决定眼前该不该记一句话，在工程上不可执行。



**预期完美信息价值**（Expected Value of Perfect Information, EVPI）为这个问题提供了理论度量 [[46]](#ref-46)。EVPI 定义为：在做出决策之前，拥有完美信息（即知道真实的 $R_{\text{task}}$）所能带来的期望回报增量。形式化为：



$$\text{EVPI} = \mathbb{E}_{\theta}\left[\max_a Q(a, \theta)\right] - \max_a \mathbb{E}_{\theta}\left[Q(a, \theta)\right]$$



其中 $\theta$ 是未知的 $R_{\text{task}}$ 参数。EVPI 量化了"如果我们现在就知道 $R_{\text{task}}$ 的确切值，决策质量能提升多少"——这也是我们因等待 $R_{\text{task}}$ 而付出的"延迟代价"的上界。当 EVPI 很高时（即不确定性大且决策对 $R_{\text{task}}$ 敏感），系统应更积极地探索；当 EVPI 很低时（即已有足够信息），系统可以利用已有估计做决策。



必须引入**写时刻可计算的代理目标**（proxy objective），让它在真实 $R_{\text{task}}$ 到达之前充当近似。三层代理目标构成一个从即时到长期的递进结构：




| 代理层 | 时延 | 信号来源 | 角色 | 理论对应 |
| --- | --- | --- | --- | --- |
| **即时启发式** | 毫秒级 | 语义相关度、惊异度、与已有 facet 的冗余度 | 冷启动默认入场分 | MDL / 预测编码 |
| **短期反馈** | 秒~分钟 | 用户当下的纠正/采纳/点赞、facet 被 recall 命中的实际频率 | 在线校准 | Reward shaping / TD learning |
| **长期 $R_{\text{task}}$** | 小时~天 | 业务终端指标（解决率、正确率、召回率） | 离线最终监督信号 | IRL / 策略梯度 |



三层通过加权汇总（或 learned aggregator）得到写时刻可用的代理 $\hat{R}_{\text{task}}$。即时启发式提供冷启动分数；短期反馈在交互过程中在线修正；长期 $R_{\text{task}}$ 到达时做离线校准——用真实业务回报回传梯度，更新前两层的权重。这不是完美的信用分配，但它是让 $R_{\text{task}}$ 从"理论正确"推进到"工程可执行"的唯一路径。






#### 4.4.1 代理目标的信用分配困境



三层代理目标面临的核心难题是**时间信用分配**（temporal credit assignment）[[35]](#ref-35)。当 $R_{\text{task}}$ 最终到达时——比如客服问题被一次性解决——这个结果应该归因到记忆系统的哪些决策？是三天前写入的那条 reflective facet 发挥了作用？还是昨天的一次成功检索？还是今天 prompt 装填策略恰好把正确的 facet 组合在一起？



Sutton 的 TD($\lambda$) 方法通过 eligibility traces 给出了一个优雅的近似 [[35]](#ref-35)：每个决策维护一个随时间衰减的"资格值"，回报到达时按资格值分配。映射到记忆系统：每次记忆操作（写入、检索、遗忘、反思）都维护一个 eligibility trace，当 $R_{\text{task}}$ 到达时，按所有操作的 eligibility trace 值加权分配功劳。这使得"三个月前写入的一条记忆在今天发挥了作用"能够被正确地归因到写入决策——尽管时间间隔很长，但只要这条记忆在中间过程中被反复检索（保持了较高的 eligibility），它的贡献就不会被遗忘。



另一个重要的工程选择是采用 **actor-critic** 架构：四个策略接口是 actor（做出决策），一个独立的 critic 网络评估"当前记忆状态的质量"（类似于价值函数 $V(\mathcal{M})$）。Critic 的训练信号来自 $R_{\text{task}}$，但它提供了一个中间的、时间密集的评估信号——即使 $R_{\text{task}}$ 还没到达，critic 也可以基于当前记忆状态估算未来的回报期望。这大大加速了学习：策略不需要等到 $R_{\text{task}}$ 到达才获得反馈，每一步都有 critic 的即时评估。











### 4.5 "压缩即智能"的工程边界：向上派生而非向下擦除



第 2 章的 Reflective 类型和第 3 章的编码深度选择，都指向同一个原则：**压缩不是被动的存储腾挪，而是主动的模式归纳。** 多条关于 Hans 付款条款的具体观察，被压缩为一条 reflective facet "Hans 有在交货前重新谈判付款条款的模式"——这条压缩后的认知不仅占更少的存储空间，更重要的是它提供了泛化能力：未来 Hans 的任何新邮件，系统都可以从这条模式出发做预测，而不需要每次重新检索五封原始邮件。



Schmidhuber 的压缩进步理论和 Hutter 的 AIXI 模型为"压缩 = 智能"提供了最有力的理论支撑 [[45]](#ref-45)[[28]](#ref-28)。两者的核心论点是：**智能的本质是发现并利用世界中的可压缩结构**。一个能用 $n$ 比特编码 $2n$ 比特观察的系统，本质上"理解"了观察中 $n$ 比特的规律。压缩率越高，理解越深。



但"压缩"这个词本身有一个危险的工程歧义：它听起来像是"删掉底层证据，只保留高阶规则"。这个歧义一旦变成工程实现，就会正面撞上 L0 的两条不变量——永不拒写（底层证据被变相拒写）和 Memory 是派生（高阶规则失去了可追溯的证据链）。一旦规则被反例挑战——比如 Hans 接下来五次交易都没有调整付款条款——系统无法回到原始证据重新判断"这个模式是否还成立"。Living Document 就死了。



Shannon 的率失真理论在这里划出了精确的理论边界 [[29]](#ref-29)。率失真函数 $R(D)$ 定义了在失真不超过 $D$ 的前提下编码所需的最小比特率。但关键是：**$R(D)$ 不等于删除**。率失真理论中的"压缩"是指用更少的比特来表示信源，但表示仍然存在——原始信息的"影子"在压缩表示中没有消失，只是以一种有损但可控的方式保留。记忆系统中的压缩应该严格类比这个语义：



- **Rate-distortion 压缩**：保留一个有损但保留关键结构的表示。原始数据的"影子"还在。
- **物理删除**：原始数据消失，不可逆。这不是压缩，是信息销毁。



正确的工程语义只有一种：



> **压缩 = 在已有证据集之上派生一条新的 reflective-type facet。新 facet 的证据集是它所归纳的底层 facet 的引用。底层 facet 留在 L0 中不动。投影时默认只展开到 reflective 层，需要时下钻。**



这意味着"注意力瓶颈"和"检索信噪比"这两个真实约束，不是靠物理删除底层信息来解决的，而是靠**读路径的层级式投影**解决的：注意力紧张时只看高阶；预算充裕或发现矛盾时深挖底层。写路径只增不删，读路径按价值取舍——这条非对称是整个架构得以"既会遗忘又可逆"的关键。






#### 4.5.1 有界最优性视角下的工程边界



Russell 的有界最优性（bounded optimality）框架 [[26]](#ref-26) 给出了工程边界的另一层含义：在有限的计算资源约束下，最优的压缩策略不是"尽可能多地压缩"（这需要指数级的计算来找到最优编码），而是"在给定计算预算内找到压缩收益最大的那些操作"。



这直接对应了 reflect_policy 的设计原则：不是"对所有记忆都做反思压缩"，而是"在可用计算预算下，选择那些压缩进步（compression progress）最大的记忆子集来做反思"。一条 reflective facet 的生成需要 LLM 调用——有 token 成本和延迟成本。只有当预期的压缩进步——即加入这条 reflective facet 后，对未来同类事件的预测损失下降幅度——大于其生成成本时，反思操作才值得执行。



用 Kolmogorov 复杂度的语言表述：设 $\mathcal{M}$ 为当前记忆库，$K(\mathcal{M})$ 为其描述长度。一个反思操作将 $\mathcal{M}$ 更新为 $\mathcal{M}'$，其价值为：



$$\Delta_{\text{compress}} = K(\mathcal{M}) - K(\mathcal{M}') + L(\mathcal{M}' \mid \mathcal{M})$$



其中 $L(\mathcal{M}' \mid \mathcal{M})$ 是从 $\mathcal{M}$ 推导出 $\mathcal{M}'$ 所需的额外描述长度（即反思操作本身的成本）。只有当 $\Delta_{\text{compress}} > 0$ 时，反思才是值得的——这是"压缩 = 智能"在工程上的精确判据。








#### 4.5.2 压缩与不变量的共存



三层架构的分离使得压缩与 L0 不变量的共存成为可能：




| 层级 | 与压缩的关系 |
| --- | --- |
| **L0（内核）** | 对压缩完全无知。Record 不可变，永不拒写。底层证据永远保留。 |
| **L1（类型系统）** | 提供 Reflective 类型作为"压缩的载体"。一条 reflective facet 的证据集引用底层 facet。 |
| **L2（策略平面）** | reflect_policy 决定何时、对什么做压缩。压缩的优化目标由 $R_{\text{task}}$ 驱动。 |



这种分离保证了：（1）压缩不会破坏底层证据（L0 不变量）；（2）压缩的形式可以随类型系统扩展（L1 的灵活性）；（3）压缩的时机和目标由策略决定（L2 的可替换性）。三层各自独立，共同使得"压缩 = 智能"从一个理论口号变成一条可工程化的路径。











### 4.6 本章小结



本章完成了三个核心论证。



**第一，记忆系统的价值函数必须由外部业务目标 $R_{\text{task}}$ 锚定**——否则系统会陷入 Goodhart 定律和 Campbell 定律所描述的自指循环，向"自己觉得重要"的方向漂移。$R_{\text{task}}$ 的外生性——它在因果链上独立于记忆系统的操作——是打破循环的唯一方式。贝叶斯决策理论进一步揭示了自指循环的本质是**先验漂移**，而 $R_{\text{task}}$ 充当了校准锚点的角色。由此推出的引理"通用最优记忆系统不存在"看似反对通用组件，实则印证了它：正因为 $R_{\text{task}}$ 不可通用，L0/L1 必须对 $R_{\text{task}}$ 完全无知，L2 必须可插拔。NFL 定理和 AIXI 模型从不同角度为这一结论提供了理论支撑。



**第二，$R_{\text{task}}$ 的稀疏性和延迟性要求三层代理目标**——即时启发式（MDL/预测编码）、短期反馈（reward shaping/TD learning）、长期 $R_{\text{task}}$（IRL/策略梯度）构成从冷启动到离线校准的完整代理链。EVPI 量化了延迟代价，eligibility traces 解决了时间信用分配，actor-critic 架构提供了时间密集的中间评估。



**第三，"压缩即智能"在工程上必须被翻译为"向上派生而非向下擦除"**——压缩是 reflective facet 的累积，不是 episodic/semantic facet 的删除。率失真理论划出了压缩的精确边界，Kolmogorov 复杂度给出了反思操作的工程判据，有界最优性框架指导了计算预算的分配。三层架构的分离保证了压缩与 L0 不变量的共存。



这三个结论——外部锚定、代理化、向上派生——共同构成了记忆系统从理论通向工程的桥。

## 参考文献

[25] Simon, H.A. (1956). Rational choice and the structure of the environment. *Psychological Review*, 63(2), 129–138. [PDF](https://pages.ucsd.edu/~mckenzie/Simon1956PsychReview.pdf)

[26] Russell, S. (1995). Rationality and intelligence. *Proceedings of IJCAI*. [PDF](https://people.eecs.berkeley.edu/~russell/papers/ijcai95-cnt.pdf)

[28] Schmidhuber, J. (2009). Driven by compression progress. *Cognitive Science Monographs*. [arXiv](https://arxiv.org/abs/0812.4360)

[29] Shannon, C.E. (1959). Coding theorems for a discrete source with a fidelity criterion. *IRE National Convention Record*, Part 4, 142–163. [PDF](https://web.stanford.edu/class/ee376a/files/2017-18/lecture_12.pdf)

[35] Sutton, R.S. (1988). Learning to predict by the methods of temporal differences. *Machine Learning*, 3(1), 9–44. [DOI](https://doi.org/10.1007/BF00115009)

[40] Goodhart, C.A.E. (1975). Problems of monetary management: The UK experience. In A.S. Courakis (Ed.), *Monetary Policy and Financial Activity*. Oxford University Press. [Wikipedia](https://en.wikipedia.org/wiki/Goodhart%27s_law)

[41] Campbell, D.T. (1979). Assessing the impact of planned social change. *Evaluation and Program Planning*, 2(1), 67–90. [Wikipedia](https://en.wikipedia.org/wiki/Campbell%27s_law)

[42] Russell, S. & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson. [Online](https://aima.cs.berkeley.edu/)

[43] Ng, A.Y. & Russell, S. (2000). Algorithms for inverse reinforcement learning. *ICML*. [PDF](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf)

[44] Wolpert, D.H. & Macready, W.G. (1997). No free lunch theorems for optimization. *IEEE Transactions on Evolutionary Computation*, 1(1), 67–82. [DOI](https://doi.org/10.1109/4235.585893)

[45] Hutter, M. (2005). *Universal Artificial Intelligence: Sequential Decisions Based on Algorithmic Probability*. Springer. [DOI](https://doi.org/10.1007/978-3-540-68677-4_4)

[46] Expected Value of Perfect Information (EVPI). UC Berkeley CS 188 Textbook. [Online](https://inst.eecs.berkeley.edu/~cs188/textbook/vpis/vpi.html)

