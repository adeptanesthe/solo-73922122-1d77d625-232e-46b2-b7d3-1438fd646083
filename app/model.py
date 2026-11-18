"""质检模型：合成训练数据 + 固定种子的随机森林（5 棵树，深度不超过 3）。

模型输入为 4 种原料克数除以 1000 后的 float32 比例，输出为“合格”概率。
本模块同时负责把训练好的森林导出为与 sklearn 完全同语义的树结构描述，
供 z3 求解器按原分类器的真实阈值语义编码。
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ---- 固定超参（题目要求：5 棵树、树深不超过 3、固定种子） ----
TRAIN_SEED = 20241117
N_TREES = 5
MAX_DEPTH = 3
N_FEATURES = 4
TOTAL_GRAMS = 1000
PROBA_THRESHOLD = 0.7          # 合格概率阈值
# 每条实测记录的训练权重 = 合成记录权重的 20 倍
MEASURED_WEIGHT = 20.0
SYNTHETIC_WEIGHT = 1.0
# 当班实测批次允许的记录条数
MIN_MEASURED_RECORDS = 5
MAX_MEASURED_RECORDS = 30
# 概率复核的浮点容差：远小于该森林可取概率值之间的最小间隔（约 1e-4 量级），
# 仅用于吸收 float64 求和顺序造成的尾数差异。
PROBA_EPS = 1e-9

FEATURE_NAMES = ["再生PP", "再生PE", "矿物填料", "功能助剂"]


def make_synthetic_data(seed: int = TRAIN_SEED, n_samples: int = 6000):
    """生成确定性的合成训练数据（克数比例的 Dirichlet 采样 + 含噪质检规则）。"""
    rng = np.random.default_rng(seed)
    X = rng.dirichlet(alpha=[2.5, 2.0, 1.6, 1.2], size=n_samples).astype(np.float32)
    # 合成质检规则：再生PP/再生PE 偏多、矿物填料偏少时更易合格，含交互项与噪声。
    score = (
        0.95 * X[:, 0]
        + 0.55 * X[:, 1]
        - 0.65 * X[:, 2]
        + 0.20 * X[:, 3]
        + 0.35 * X[:, 0] * X[:, 1]
        - 0.25 * X[:, 2] * X[:, 3]
    )
    noise = rng.normal(0.0, 0.03, size=n_samples)
    # 阈值取中位数，保证类别大致均衡；整个过程由固定种子决定，完全可复现。
    thr = float(np.quantile(score + noise, 0.5))
    y = (score + noise >= thr).astype(int)
    return X, y


def validate_measured_batch(records) -> list[str]:
    """校验当班实测批次：5~30 条、每条 4 个整数克数且总重 1000、标签为 0/1。"""
    errs: list[str] = []
    if not isinstance(records, (list, tuple)):
        return ["records 必须是数组"]
    n = len(records)
    if not (MIN_MEASURED_RECORDS <= n <= MAX_MEASURED_RECORDS):
        errs.append(f"实测记录条数须在 {MIN_MEASURED_RECORDS}~{MAX_MEASURED_RECORDS} 之间（当前 {n}）")
        return errs
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            errs.append(f"第 {idx + 1} 条记录必须是对象")
            continue
        grams = rec.get("grams")
        label = rec.get("label")
        if not isinstance(grams, (list, tuple)) or len(grams) != N_FEATURES:
            errs.append(f"第 {idx + 1} 条：用量必须是 {N_FEATURES} 个整数的数组")
            continue
        bad_gram = any(
            (not isinstance(g, int)) or isinstance(g, bool) or g < 0 or g > TOTAL_GRAMS
            for g in grams
        )
        if bad_gram:
            errs.append(f"第 {idx + 1} 条：每种原料用量须为 0~{TOTAL_GRAMS} 的整数")
        elif sum(grams) != TOTAL_GRAMS:
            errs.append(f"第 {idx + 1} 条：总重须为 {TOTAL_GRAMS} 克（当前 {sum(grams)}）")
        if not isinstance(label, int) or isinstance(label, bool) or label not in (0, 1):
            errs.append(f"第 {idx + 1} 条：标签必须为 0 或 1")
    return errs


def train_model(seed: int = TRAIN_SEED, measured_records: list[dict] | None = None):
    """按固定规格与种子训练随机森林。

    measured_records 为 None 时仅使用固定合成训练集（启动基线，与历史版本完全一致）；
    提供实测批次时，与固定合成训练集合并训练：合成记录权重 1，实测记录权重 20。
    随机森林超参（5 棵树、深度 ≤ 3、random_state=固定种子）始终不变，
    因此相同输入下重训结果确定可复现。
    """
    X, y = make_synthetic_data(seed)
    weights = np.full(len(X), SYNTHETIC_WEIGHT, dtype=np.float64)
    if measured_records:
        xm = np.array(
            [[g / TOTAL_GRAMS for g in rec["grams"]] for rec in measured_records],
            dtype=np.float32,
        )
        ym = np.array([int(rec["label"]) for rec in measured_records], dtype=np.int64)
        wm = np.full(len(measured_records), MEASURED_WEIGHT, dtype=np.float64)
        X = np.vstack([X, xm])
        y = np.concatenate([y, ym])
        weights = np.concatenate([weights, wm])
    clf = RandomForestClassifier(
        n_estimators=N_TREES, max_depth=MAX_DEPTH, random_state=seed
    )
    clf.fit(X, y, sample_weight=weights)
    return clf


# ---------------------------------------------------------------------------
# 森林结构导出（与 sklearn 阈值语义严格一致）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Node:
    """内部节点：feature 上按 threshold 二分；left/right 为子节点下标。"""
    feature: int
    threshold: float
    left: int
    right: int


@dataclass(frozen=True)
class Leaf:
    """叶子节点：合格概率（与 sklearn predict_proba 完全一致的精确有理数）。"""
    prob1: Fraction


@dataclass(frozen=True)
class TreeSpec:
    nodes: tuple  # 下标即节点 id，元素为 Node 或 Leaf
    root: int = 0


@dataclass(frozen=True)
class ForestSpec:
    trees: tuple
    n_trees: int
    max_depth: int
    seed: int
    threshold: float = PROBA_THRESHOLD


def _leaf_prob1_exact(value_row: np.ndarray) -> Fraction:
    """从 tree_.value 的一行提取类别 1 概率，保持与 sklearn 相同的数值语义。

    sklearn 的 Tree.predict_proba 就是把 value 行归一化；无论该版本存储的是
    计数还是比例，prob1 = v1 / (v0 + v1) 都与 predict_proba 一致。
    计数为整数值浮点时取整数分数，否则直接取该 float64 的精确有理值，
    两种情形都与 sklearn 实际参与平均的数值严格相等。
    """
    v0, v1 = float(value_row[0]), float(value_row[1])
    total = v0 + v1
    if total <= 0.0:  # 理论上不会出现（叶子必有训练样本），防御性处理
        return Fraction(0)
    if v0.is_integer() and v1.is_integer():
        return Fraction(int(v1), int(total))
    return Fraction(v1) / Fraction(total)


def extract_forest(clf: RandomForestClassifier, seed: int = TRAIN_SEED) -> ForestSpec:
    assert len(clf.estimators_) == N_TREES
    trees = []
    for est in clf.estimators_:
        t = est.tree_
        assert t.max_depth <= MAX_DEPTH
        nodes = []
        for nid in range(t.node_count):
            left, right = int(t.children_left[nid]), int(t.children_right[nid])
            if left == right:  # sklearn 用 left == right（均为 -1）表示叶子
                nodes.append(Leaf(prob1=_leaf_prob1_exact(t.value[nid][0])))
            else:
                nodes.append(
                    Node(
                        feature=int(t.feature[nid]),
                        threshold=float(t.threshold[nid]),
                        left=left,
                        right=right,
                    )
                )
        trees.append(TreeSpec(nodes=tuple(nodes)))
    return ForestSpec(
        trees=tuple(trees), n_trees=N_TREES, max_depth=MAX_DEPTH, seed=seed
    )


def predict_proba_qualified(clf: RandomForestClassifier, grams: list[int]) -> float:
    """用原分类器对整数克数配方复算合格概率（输入为克数/1000 的 float32）。"""
    x = np.array([[g / TOTAL_GRAMS for g in grams]], dtype=np.float32)
    return float(clf.predict_proba(x)[0, 1])


def is_qualified(proba: float) -> bool:
    """统一的合格判定：概率 >= 0.7（含 1e-9 浮点容差）。"""
    return proba >= PROBA_THRESHOLD - PROBA_EPS


# ---------------------------------------------------------------------------
# 纯 Python 复算器：用整数克数 + 预计算阈值截断，逐步走树。
# 用于自测时独立验证 z3 编码与 sklearn 语义一致。
# ---------------------------------------------------------------------------

def cutoff_left(threshold: float) -> int:
    """最大的整数克数 g∈[0,1000]，使 float64(float32(g/1000)) <= threshold。

    sklearn 树内部以 float32 特征与 float64 阈值比较；该截断把浮点比较
    精确翻译成整数比较 g <= K。若没有任何 g 满足（阈值 < 0）返回 -1；
    若全部满足（阈值 >= 1）返回 TOTAL_GRAMS。
    """
    if np.float32(0.0) > threshold:
        return -1
    if np.float32(1.0) <= threshold:
        return TOTAL_GRAMS
    lo, hi = 0, TOTAL_GRAMS  # 不变式：lo 满足，hi 不满足（由上面两支保证）
    # 注意 hi=1000 可能也满足（threshold>=1 已排除），lo=0 必满足
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if np.float32(mid / TOTAL_GRAMS) <= threshold:
            lo = mid
        else:
            hi = mid
    return lo


def forest_proba_rational(forest: ForestSpec, grams: list[int]) -> Fraction:
    """按整数克数走树，返回森林平均合格概率的精确有理值。"""
    total = Fraction(0)
    for tree in forest.trees:
        nid = tree.root
        while True:
            node = tree.nodes[nid]
            if isinstance(node, Leaf):
                total += node.prob1
                break
            if grams[node.feature] <= cutoff_left(node.threshold):
                nid = node.left
            else:
                nid = node.right
    return total / forest.n_trees
