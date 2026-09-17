# 推荐系统里的双塔、Transformer、BERT

不是三个可以互相替换的「SOTA 模型」，而是漏斗里三层经常一起在线的东西。
下面按 **召回要独立性、精排要交叉、BERT 其实是两条产品线** 来写。

配套代码：`python3 -m recsys_models`（stdlib，无 PyTorch）。会话默认
`coffee → cake → ramen`。

## 1. 先把层放对

| 层 | 候选规模 | 延迟预算 | 能用的结构 | 不能用的结构 |
|---|---|---|---|---|
| 召回 | 百万～千万 | 十毫秒级 | 双塔 + ANN；用户塔可以是均值 / 多兴趣 / SASRec / 文本 BERT | DIN、交叉 Transformer、交叉 BERT：要对每个候选跑一遍用户侧 |
| 粗排/精排 | 百～千 | 几十毫秒 | DIN、BST、WDL/DeepFM、多目标 MLP、交叉 Transformer | 再扫一遍全库 |
| 重排 | 几十 | 更宽 | 多样性、生态、规则、可选 LLM listwise | 当召回用 |

工业上「最重要」的模型，是 **还在扛流量的那一层**：

- 2016 至今：YouTube DNN / DSSM 双塔召回没有被换掉，被换掉的是用户塔内部（从 lookup+均值 → 序列 Transformer → 多兴趣 → 多模态）。
- 2018 至今：精排的主线是 **候选感知**（DIN 的 target attention，BST 把 candidate 拼进序列）。
- BERT：搜索和内容场景里，**语义双塔** 的用量远大于 BERT4Rec；序列 MLM 是另一条线。

2023 以后的生成式检索（TIGER、OneRec）和 LLM 重排是往漏斗上叠，不是把双塔从召回位换走。

## 2. 双塔：为什么召回非它不可

### 2.1 公式

用户塔 \(u = f_\theta(user, history, context)\)，物品塔 \(v = g_\phi(item, side)\)，
分值 \(s = \langle \bar u, \bar v \rangle / \tau\)（余弦 + 温度）。

训练常用 InfoNCE / sampled softmax：

\[
\mathcal{L} = -\log \frac{e^{s(u,v^+)}}{e^{s(u,v^+)} + \sum_{v^-} e^{s(u,v^-)}}
\]

负例一般要 **混**：in-batch（便宜、有偏）、全局随机（校准流行度）、难负例（同品类未点、ANN 近邻未点）。温度太小、难负太脏，都会把表征推到坍缩或把流行度学成捷径。

### 2.2 那条不能破的约束

**两个塔在内积之前不能看对方。** \(u\) 对全库是同一个向量。这样 \(v\) 才能离线算完写入 Faiss/ScaNN/HNSW，在线只跑用户塔 + 近邻搜索。

一旦把候选拼进用户塔（target attention、cross-encoder），索引失效，复杂度从 \(O(d)\) 近邻变成 \(O(N)\) 精排。

### 2.3 用户塔内部可以很深

「双塔」说的是 **打分形状**，不是「两边都是一层 embedding」：

- YouTube DNN：观看序列均值/sum pooling 再 MLP。
- 均值双塔：本仓库默认。无序，长期兴趣稳，当下意图弱。`coffee,cake,ramen` → `latte`。
- 多兴趣 MIND/ComiRec：一个用户多个 \(u\)，ANN 查多次再融合，缓解「均值被多数类淹没」。
- **SASRec 用户塔**：因果 Transformer 的最后一个 hidden 当 \(u\)，仍然 \(\langle u, v \rangle\)，仍然可以进 ANN。本仓库里它会把同一会话打成 `sushi`。
- 语义 BERT 双塔：塔输入是标题/类目/图文，不是 item id。冷启靠文本。

外卖里用户塔还会拼：地理、时段、价格带、配送、端、曝光位置。物品塔拼：类目、SPU、商家、文本、图像。

### 2.4 本例在算什么

会话两下咖啡厅、一下拉面。均值 \(u\) 偏 cafe，召回第一是 `latte`。
这不是 bug：召回这一层本来就该把「长期像咖啡厅的人」的物料召回来。当下想吃日料，是精排或「用户塔改成序列编码器」的事。

InfoNCE 在本例里：正例 `latte` vs 易负 `burger` 很轻松；vs 难负 `sushi` loss 明显更大。只喂随机负例，模型会停在「用户不是西式」，分不开咖啡厅和日料。

## 3. Transformer：先分清「进用户塔」还是「进精排」

把「推荐里的 Transformer」说成一个模型，是最常见的简化。至少两件完全不同的事：

### 3.1 SASRec：序列双塔（召回形状）

因果 mask，位置 \(i\) 只看 \(\le i\)。用 **最后一个 hidden** 当用户向量，再对物品 embedding 点积。

- 有顺序、有 recency，最后一次点击 ramen 会主导。
- **对所有候选仍是同一个 \(u\)**。所以它是双塔，不是精排交叉。
- 和 BERT4Rec 比：训练目标是 next-item（自回归），不是 MLM。

本仓库会打印最后一行 attention：质量集中在 `ramen` 上，全库第一变成 `sushi`。

### 3.2 DIN / BST：候选感知（精排形状）

DIN（阿里 2018）：对 **当前候选** 做 target attention，历史里和候选像的点击权重大。

\[
u_i = \sum_t \alpha(h_t, v_i)\, h_t, \quad \alpha = \mathrm{softmax}(a(h_t, v_i))
\]

论文里 \(a(\cdot)\) 是 MLP\([h, v, h-v, h \odot v]\)；本仓库用缩放余弦，几何一样。

于是：

- 打 `latte` 时，注意力在 `coffee/cake`；
- 打 `sushi` 时，注意力在 `ramen`；
- 打 `burger` 时，历史对不上，分数低。

**一个请求里，用户向量随候选变。** 这才能表达「这个人既像咖啡厅也像日料」，代价是不能 ANN。BST 把 candidate 接到行为序列末尾再跑 Transformer，是同一思想的序列实现。

精排还可以继续叠：多目标（点击/转化/停留）、位置偏差、交叉特征、序列长度到百级的行为。这些都发生在召回截断之后。

### 3.3 对照（同一会话）

| | 用户向量几个 | 能否 ANN | 本例第一 |
|---|---|---|---|
| 均值双塔 | 1 | 能 | latte（多数类） |
| SASRec 用户塔 | 1 | 能 | sushi（最后一跳） |
| DIN | 每个候选 1 个 | 不能 | 两路兴趣都能打高分，在召回子集上重排 |

漏斗正确用法：双塔或 SASRec 用户塔取出 top-N，DIN/BST 只对这 N 条做候选感知。本仓库 `recall_then_rank` 就是这个形状。

## 4. BERT：语义双塔 ≠ BERT4Rec

### 4.1 BERT4Rec（序列 MLM）

把 item id 当成 token，随机 mask，双向 Transformer 填洞。相对 SASRec：

- 训练信号更密（每个被 mask 的位置都是样本），不只是最后一个 next-item。
- 双向能用 **未来点击** 解释中间洞。本例 cloze `coffee → [MASK] → ramen`：因果只能看 coffee → `latte`；双向看到右边拉面 → `sushi`。
- **推断下一跳** 时通常把 `[MASK]` 放句尾。句尾没有未来，双向并不比单向多看一格。这时它更像会话 bag，本例会回到 cafe（`latte`）。不要用句尾预测的线上指标去理解「双向一定更强」。

### 4.2 语义双塔 BERT（工业里更大的那条）

塔的输入是 **文本/多模态**，不是点击 id：

- 物品塔：标题、类目、属性、图像 encoder，离线入库。新商品没有点击也能检索。
- 用户塔：Query、历史标题拼接、画像文本。
- 训练：点击/下单对上 InfoNCE，和 id 双塔同一套损失。
- 本例冷启 `udon`（tokens `jp, noodle`）从未出现在会话里，但文本空间已经靠近拉面。

交叉编码器（把 query 和 item 拼成一句过 BERT）精度更高，复杂度和 DIN 同类，只能精排/重排。

搜索、内容推荐、同款、标题改写后的召回，这条线往往比 BERT4Rec 更值钱。Id 塔解决「点过像什么」，文本塔解决「长什么样」。

## 5. 训练目标和评测不要串层

| 模型 | 典型损失 | 线上近似指标 | 常见作假 |
|---|---|---|---|
| 双塔 | InfoNCE / sampled softmax | Recall@K、HitRate | 只用随机负例；把精排 AUC 当召回好坏 |
| SASRec | next-item CE | 序列 HitRate/NDCG | 用随机负例评 next-item，虚高 |
| DIN/BST | 点分 BCE / 多目标 | AUC、GAUC、分桶校准 | 不按请求做 GAUC，被热门单子带着走 |
| BERT4Rec | MLM | 补洞准确率 ≠ 下一跳 | 用 MLM 开发集选精排模型 |
| 语义双塔 | 文本 InfoNCE | 冷启 Recall、搜索 NDCG | 只用点击热门标题，学成流行度 |

## 6. 外卖/到店场景里这三家怎么放

- **召回并行多路**：id 双塔（长期+短期）、实时序列塔（最近十几次点击）、语义塔（query/类目/菜名）、地理/商家召回。每路都是「可索引」的形状。
- **融合截断**：每路 top 几百，去重后进粗排。
- **精排**：DIN/BST + 交叉特征 + 多目标（曝光→点击→下单→复购）。这里才吃「当前商家 × 用户刚才点了辣」这种交叉。
- **实时**：双塔物品向量可以小时级更新；用户塔必须用新鲜序列（见仓库里另一份 freshness 讨论）。精排更吃实时行为。
- **不要**：用精排网扫全库；把 BERT4Rec 当语义检索；指望均值双塔跟上「刚点了拉面」。

## 7. 和本仓库代码的对应

| 对象 | 文件 | 要点 |
|---|---|---|
| 均值双塔 + 长期画像混合 | `models.TwoTower` | `user_embedding(..., long_term=)` |
| InfoNCE | `losses.infonce_loss` | 易负 vs 难负 |
| SASRec 用户塔 | `models.CausalTransformer` | 因果 mask + recency，一个 \(u\) |
| DIN | `din.TargetAttentionRanker` | 每个候选不同 \(u_i\) |
| BERT4Rec | `models.Bert4Rec` | 句尾 mask 与 cloze |
| 语义双塔 | `semantic.BertDualEncoder` | 标题 word-piece + 冷启 udon |
| 漏斗 | `pipeline.recall_then_rank` | 双塔 top-4 再精排 |

跑 `python3 -m recsys_models` 会按上面的顺序把同一会话算一遍，并打印 attention 和 InfoNCE 数字。
