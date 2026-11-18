"""FastAPI 入口：质检模型 + 配方求解服务。

启动时以固定种子训练随机森林（5 棵树、深度 ≤ 3）作为基线模型，并预生成经
求解器实测的可行 / 无解样例。当班实测批次可经 /api/model/update 整体替换并重训，
新候选分类器、z3 树结构与样例全部就绪后才原子切换；失败时旧版继续可用。
求解结果一律用同一快照内的原分类器复算概率后返回。
"""
from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .model import (
    FEATURE_NAMES,
    MAX_DEPTH,
    MEASURED_WEIGHT,
    N_TREES,
    PROBA_THRESHOLD,
    TOTAL_GRAMS,
    TRAIN_SEED,
    Leaf,
)
from .samples import build_samples  # noqa: F401  # 保持模块公共入口兼容
from .solver import SolveInput, solve
from .versioning import ModelRegistry, ModelUpdateError

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="再生塑料配方试配筛选", version="2.0.0")

# ---- 启动即训练基线（固定种子，秒级完成），并构建求解用森林规范与样例 ----
_t0 = time.perf_counter()
REGISTRY = ModelRegistry()
BASELINE = REGISTRY.load_baseline()
STARTUP_SECONDS = time.perf_counter() - _t0


def _depth(tree, nid: int) -> int:
    node = tree.nodes[nid]
    if isinstance(node, Leaf):
        return 0
    return 1 + max(_depth(tree, node.left), _depth(tree, node.right))


@app.get("/api/meta")
def meta() -> dict:
    snap = REGISTRY.current()
    forest = snap.forest
    return {
        "version": snap.version,
        "feature_names": FEATURE_NAMES,
        "total_grams": TOTAL_GRAMS,
        "proba_threshold": PROBA_THRESHOLD,
        "n_trees": N_TREES,
        "max_depth": MAX_DEPTH,
        "tree_depths": [_depth(t, 0) for t in forest.trees],
        "train_seed": TRAIN_SEED,
        "measured_weight": MEASURED_WEIGHT,
        "n_measured": snap.n_measured,
        "created_at": snap.created_at,
        "train_seconds": snap.train_seconds,
        "startup_seconds": round(STARTUP_SECONDS, 3),
    }


@app.get("/api/model/status")
def model_status() -> dict:
    """当前模型版次、所用实测批次、更新在途标志与上次更新结果。"""
    return REGISTRY.status()


@app.post("/api/model/update")
def model_update(payload: dict) -> dict:
    """整体替换当班实测批次并按原规格/种子重训；成功才切换，失败旧版保留。"""
    if REGISTRY.is_updating():
        raise HTTPException(status_code=409, detail="已有模型更新任务在执行，请稍后再试。")
    records = payload.get("records") if isinstance(payload, dict) else None
    try:
        snap = REGISTRY.update_with_measured_batch(records if records is not None else [])
    except ModelUpdateError as exc:
        # 输入类失败返回 400；在途冲突返回 409；其余重训失败统一 422
        msg = str(exc)
        status = 409 if "已有模型更新任务" in msg else 400
        if status == 400 and "输入校验失败" not in msg:
            status = 422
        raise HTTPException(status_code=status, detail=msg)
    return {
        "status": "success",
        "message": f"新版本 v{snap.version} 已发布：分类器、z3 树结构与样例均已就绪。",
        "model": REGISTRY.status(),
    }


@app.post("/api/solve")
def api_solve(payload: dict) -> dict:
    def _int_list(key):
        v = payload.get(key)
        return list(v) if isinstance(v, (list, tuple)) else []

    def _bool_list(key):
        v = payload.get(key)
        return [bool(x) for x in v] if isinstance(v, (list, tuple)) else []

    def _as_ints(key):
        out = []
        for x in _int_list(key):
            if isinstance(x, bool):
                out.append(x)  # 校验阶段会拒绝
            elif isinstance(x, int):
                out.append(x)
            elif isinstance(x, float) and x.is_integer():
                out.append(int(x))
            else:
                out.append(x)  # 原样传入，由 validate 报错
        return out

    inp = SolveInput(
        original=_as_ints("original"),
        lower=_as_ints("lower"),
        upper=_as_ints("upper"),
        locked=_bool_list("locked"),
    )
    # 整次求解只取一次快照：森林约束、原分类器复算、版次来自同一模型快照
    snap = REGISTRY.current()
    result = solve(snap.forest, snap.clf, inp)
    out = result.to_dict()
    out["model_version"] = snap.version
    out["model_created_at"] = snap.created_at
    return out


@app.get("/api/samples")
def api_samples() -> dict:
    snap = REGISTRY.current()
    data = dict(snap.samples)
    data["model_version"] = snap.version
    return data


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/update", include_in_schema=False)
def update_page():
    return FileResponse(STATIC_DIR / "update.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
