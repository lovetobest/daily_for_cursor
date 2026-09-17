"""CLI walkthrough: funnel, two-tower, Transformer, BERT, DIN, pipeline."""

from __future__ import annotations

from recsys_models.catalog import CATALOG, COLD_START_ID, DEFAULT_HISTORY, item_embedding
from recsys_models.din import TargetAttentionRanker
from recsys_models.losses import infonce_loss
from recsys_models.models import Bert4Rec, CausalTransformer, ScoredItem, TwoTower
from recsys_models.pipeline import recall_then_rank
from recsys_models.semantic import BertDualEncoder
from recsys_models.vectors import cosine, format_vector, l2_normalize


def run(history: tuple[str, ...], *, top_k: int = 4) -> None:
    _header(history)
    _funnel_map()
    _two_tower(history, top_k)
    _sasrec(history, top_k)
    _din(history, top_k)
    _bert(history, top_k)
    _pipeline(history)
    _how_to_choose()


def _header(history: tuple[str, ...]) -> None:
    print("推荐系统里最常用的三类模型：双塔、Transformer、BERT")
    print("=" * 72)
    print("同一用户、同一会话，看它们各自在算什么、能服务什么、不能服务什么。")
    print()
    print("会话:", " → ".join(history))
    cats = " → ".join(CATALOG[i].category for i in history)
    print("品类:", cats)
    print("冲突: 前两下是咖啡厅，最后一下是拉面 —— 长期兴趣 vs 当下意图。")
    print()


def _funnel_map() -> None:
    print("-" * 72)
    print("0. 先放进漏斗里，不要把三个模型当同一层的替代品")
    print()
    print("  曝光前要过四层，算力差几个数量级：")
    print()
    print("    物料百万/千万")
    print("        │  召回 recall     双塔 / SASRec用户塔 / 语义BERT双塔")
    print("        │  每次请求只跑用户塔 + ANN（Faiss/ScaNN/HNSW）")
    print("        ▼  几百条")
    print("       粗排/精排 rank     DIN / BST / 交叉Transformer / 多目标MLP")
    print("        │  用户×候选 交叉，候选感知，不能直接 ANN")
    print("        ▼  几十条")
    print("       重排 re-rank      多样性、新鲜度、生态、LLM listwise（可选）")
    print()
    print("  双塔解决的是「怎么从一千万里取出五百」。")
    print("  Transformer/DIN 解决的是「这五百里谁真正该排前面」。")
    print("  BERT 有两条互不替代的产品线：语义双塔（召回）和 BERT4Rec（序列 MLM）。")
    print()


def _two_tower(history: tuple[str, ...], k: int) -> None:
    model = TwoTower()
    user = model.user_embedding(history)
    ranked = model.rank(history)
    print("-" * 72)
    print("1. 双塔（DSSM / YouTube DNN）—— 召回的默认答案")
    print()
    print("  结构:")
    print("    用户塔  f(user, history, context) → u     在线算")
    print("    物品塔  g(item, side-info, text)  → v     离线算完写入 ANN")
    print("    分值    s = cos(u, v) / τ")
    print()
    print("  关键约束: 两个塔在打分之前不准看对方。u 对所有物品是同一个向量。")
    print("  否则无法把物品向量预计算进索引，召回就会退化成「对全库跑一遍精排」。")
    print()
    print(f"  本例用户向量 (均值池化，无序) = {format_vector(l2_normalize(user))}")
    print("  轴: [日料, 咖啡厅, 西式]。两下咖啡厅压过一下拉面，所以偏向 cafe。")
    print()
    _print_rank("  全库余弦召回", ranked, k)
    print("  顺序不敏感：coffee,ramen 与 ramen,coffee 得到同一个 u。这是优点（稳定）")
    print("  也是缺点（丢失当下意图）。")
    print()
    print("  训练: InfoNCE / sampled softmax。正样本=点击/下单，负样本= in-batch")
    print("  + 全局随机 + 难负例（同品类未点）。温度 τ 太小容易表征坍缩。")
    pos = item_embedding("latte")
    easy_neg = item_embedding("burger")
    hard_neg = item_embedding("sushi")
    loss_easy, p_easy = infonce_loss(user, pos, [easy_neg], temperature=0.4)
    loss_hard, p_hard = infonce_loss(user, pos, [hard_neg], temperature=0.4)
    print(
        f"  本例 InfoNCE  τ=0.4   正例=latte"
        f"  vs burger(易负) loss={loss_easy:.3f}  P+= {p_easy[0]:.3f}"
    )
    print(
        f"                         vs sushi (难负) loss={loss_hard:.3f}  P+= {p_hard[0]:.3f}"
    )
    print("  难负例更接近用户向量，loss 更大 —— 线上一般要混难负，否则只学会「不是西式」。")
    print()
    cafe = (0.05, 1.00, 0.05)
    mixed = model.user_embedding(history, long_term=cafe, long_term_weight=0.7)
    print(
        "  工业用户塔几乎从不会只均值会话。长期画像 ⊕ 短期序列 ⊕ 场景"
        f"（本例 0.7×咖啡画像 + 0.3×会话 → {format_vector(l2_normalize(mixed))}）"
    )
    print("  外卖场景还会拼：地理位置、时段、价格带、配送、曝光位置偏差。")
    print()


def _sasrec(history: tuple[str, ...], k: int) -> None:
    model = CausalTransformer()
    hidden, weights = model.encode(history)
    ranked = model.rank(history)
    print("-" * 72)
    print("2. Transformer（SASRec）—— 序列进用户塔，仍可以是双塔")
    print()
    print("  SASRec / BST / 行为序列 Transformer 不是「取代双塔」，而是:")
    print("    (a) 用户塔里的序列编码器（召回，ANN 友好）← 本段")
    print("    (b) 候选拼进序列的精排网（DIN/BST，ANN 不友好）← 下一段")
    print()
    print("  因果 self-attention: 位置 i 只能看 ≤ i。最后一格的 hidden 当 u。")
    print("  再对物品向量做点积，形状仍是 <u, v>，所以仍可进 ANN。")
    print()
    print("  注意力（行=query 位置，列=history；上三角被 causal mask 打掉）:")
    _print_attn(history, weights)
    print()
    print("  最后一行看最后一次点击 ramen，recency bias 把质量压到最后一列。")
    print(f"  用户向量 u_last = {format_vector(hidden[-1])}")
    print()
    _print_rank("  用这一个 u_last 打全库（序列双塔）", ranked, k)
    print("  和均值双塔的差别: 当下意图赢了，sushi/hotpot 超过 latte。")
    print("  和 DIN 的差别: 对 latte、对 sushi，用的是同一个 u_last。")
    print()


def _din(history: tuple[str, ...], k: int) -> None:
    model = TargetAttentionRanker()
    print("-" * 72)
    print("3. DIN / BST —— 候选感知精排，这才是 Transformer 真正贵的地方")
    print()
    print("  双塔/SASRec:  u = f(user, history)             一次")
    print("  DIN:          u_i = f(user, history | item_i)   每个候选一次")
    print()
    print("  看候选时，历史里「和候选像」的点击权重大。咖啡厅候选会回看 coffee/cake，")
    print("  日料候选会回看 ramen。一个用户可以同时对两路兴趣打出高分。")
    print()
    for cand in ("latte", "sushi", "burger"):
        _user, attn = model.attend(history, cand)
        bits = "  ".join(f"{h}={w:.2f}" for h, w in zip(history, attn))
        print(
            f"  候选 {cand:<7}  历史注意力 {bits}  "
            f"score={model.score(history, cand):.3f}"
        )
    print()
    _print_rank("  DIN 全库精排（每个候选一个 u_i）", model.rank(history), k)
    print("  latte 和 sushi 都能打出接近 1 的分 —— 两路兴趣同时成立。")
    print("  分数扎堆是 target attention 的副作用：注意力一塌到匹配点击，")
    print("  cos(u_i, c) ≈ 1。论文后面接 MLP 交叉，就是为了在「都匹配」的")
    print("  候选之间再拉开。双塔做不到这种「按候选回看历史」。")
    print("  BST 把 candidate 接到序列末尾再跑 Transformer，是同一件事的序列版。")
    print("  复杂度 O(|history| × |candidates| × d)，只扛得住几百条，扛不住一千万。")
    print()


def _bert(history: tuple[str, ...], k: int) -> None:
    seq = Bert4Rec()
    dual = BertDualEncoder()
    print("-" * 72)
    print("4. BERT 在推荐里是两条产品线，不要混成一个「BERT 模型」")
    print()
    print("  A. BERT4Rec —— 把点击序列当成句子，用 MLM / cloze 训练")
    print("     双向: [MASK] 能看左右。因果 Transformer 不能看未来点击。")
    print("     推断下一跳时常把 [MASK] 放在句尾；句尾没有未来，结构上接近双塔均值。")
    print()
    _print_rank("  BERT4Rec 句尾 [MASK] 下一跳", seq.rank(history), k)
    left, right = (history[0],), (history[-1],)
    print(f"  cloze  {left[0]} → [MASK] → {right[0]}")
    _print_rank("    因果（只能看 coffee）", seq.cloze(left, right, causal=True), 3)
    _print_rank("    双向（还能看 ramen）", seq.cloze(left, right, causal=False), 3)
    print("  双向补洞是 BERT4Rec 相对 SASRec 的训练密度优势；线上句尾预测未必更强。")
    print()
    print("  B. 语义双塔 BERT —— 用标题/类目文本做塔，这才是工业里更大的 BERT")
    print("     用户塔: 历史标题 word-piece 双向编码")
    print("     物品塔: 商品标题编码，离线写入 ANN（冷启、内容、搜索都靠它）")
    print()
    user_text = dual.user_embedding(history)
    print(f"  文本用户向量 = {format_vector(l2_normalize(user_text))}")
    _print_rank("  语义双塔召回", dual.rank(history), k)
    cold = CATALOG[COLD_START_ID]
    cold_score = cosine(user_text, dual.item_embedding(COLD_START_ID))
    burger_score = cosine(user_text, dual.item_embedding("burger"))
    print(
        f"  冷启 {cold.item_id} {cold.name!r} tokens={list(cold.tokens)}"
        f"  cos={cold_score:.3f}  vs burger={burger_score:.3f}"
    )
    print("  没有点击过 udon，但「jp+noodle」已经靠近拉面。这是 id 塔做不到的。")
    print("  交叉编码器（query 和 item 拼成一句过 BERT）更准，O(N) 全库算不起，")
    print("  只配做精排/重排，和 DIN 是同一层的事。")
    print()


def _pipeline(history: tuple[str, ...]) -> None:
    sasrec = CausalTransformer()
    din = TargetAttentionRanker()
    via_seq = recall_then_rank(history, ranker=sasrec, recall_k=4)
    via_din = recall_then_rank(history, ranker=din, recall_k=4)
    print("-" * 72)
    print("5. 串起来: 双塔召回 → 序列/DIN 精排（线上长这样）")
    print()
    _print_rank("  召回 top-4（均值双塔）", via_seq.recalled, 4)
    _print_rank("  精排: 只对这 4 条跑 SASRec（同一个 u_last）", via_seq.reranked, 4)
    _print_rank("  精排: 只对这 4 条跑 DIN（每个候选不同 u_i）", via_din.reranked, 4)
    print("  召回保证覆盖（咖啡厅 + 日料都进来），精排决定当下点谁。")
    print("  漏斗切错层是最常见的误用: 用 DIN 扫全库会超时，用双塔当精排会少交叉。")
    print()


def _how_to_choose() -> None:
    print("-" * 72)
    print("6. 怎么选（以及 2024–2026 还在用什么）")
    print()
    print("  要扫全库 / 要缓存物品向量          → 双塔。用户塔可以是均值、")
    print("                                      多兴趣 (MIND)、或 SASRec。")
    print("  要冷启、搜索、内容理解              → 语义 BERT/CLIP 双塔。")
    print("  要吃「这个候选 × 这段历史」的交叉   → DIN / BST / 交叉 Transformer。")
    print("  要下一跳、会话内顺序                → SASRec 用户塔或 BERT4Rec。")
    print()
    print("  生成式检索 (TIGER / OneRec) 和 LLM listwise 重排在往上叠，")
    print("  但赚钱的底座仍是: 双塔召回 + 深度精排。双塔没过时，过时的是")
    print("  「用户塔只做 embedding lookup + 均值」这一层实现。")
    print()
    print("  更完整的文字版: docs/recsys-models.md")
    print()


def _print_rank(title: str, ranked: list[ScoredItem], k: int) -> None:
    print(f"  {title}")
    for i, row in enumerate(ranked[:k], start=1):
        print(
            f"    {i}. {row.item.item_id:<8} {row.score:7.3f}  "
            f"{row.item.category:<9} {row.item.name}"
        )


def _print_attn(history: tuple[str, ...], weights: list[list[float]]) -> None:
    col_w = 8
    header = " " * 10 + "".join(f"{h:>{col_w}}" for h in history)
    print("   " + header)
    for i, row in enumerate(weights):
        cells = "".join(
            f"{'—':>{col_w}}" if j > i else f"{w:{col_w}.2f}"
            for j, w in enumerate(row)
        )
        print(f"    q[{history[i]:<6}]{cells}")
