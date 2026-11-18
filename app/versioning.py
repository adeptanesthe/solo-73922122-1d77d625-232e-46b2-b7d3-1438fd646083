"""模型版次管理：原子快照、整体替换实测批次、失败保留旧版。

一次“当班试配模型更新”的完整流程：
  1. 校验实测批次（5~30 条、整数克数、总重 1000、0/1 标签）；
  2. 与固定合成训练集合并，按原规格（5 棵树、深度 ≤ 3）和原种子重训随机森林，
     合成记录权重 1、实测记录权重 20；
  3. 导出供 z3 求解的树结构 ForestSpec；
  4. 用真实模型 + 求解器重新实测可行 / 无解样例；
以上四步全部成功后，才把新 ModelSnapshot 原子切换为服务运行期的当前版本；
任一步失败则旧版快照（连同其所用实测批次）继续可用，当前版本指针不变。

求解请求在整个处理期间只读取一次快照引用，因此其森林约束、原分类器复算
与版次标识必然来自同一模型快照，更新期间旧版求解照常完成。
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .model import (
    MEASURED_WEIGHT,
    extract_forest,
    train_model,
    validate_measured_batch,
)
from .samples import build_samples


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ModelSnapshot:
    """一份不可变的模型快照：分类器、z3 树结构、样例与版次信息同生共死。"""

    version: int
    clf: object
    forest: object
    samples: dict
    measured_batch: tuple          # 本快照所用实测批次（基线为空元组），整体替换语义
    created_at: str
    train_seconds: float
    n_measured: int
    label_counts: dict             # {"0": x, "1": y}
    measured_weight: float

    def public_info(self) -> dict:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "n_measured": self.n_measured,
            "label_counts": self.label_counts,
            "measured_weight": self.measured_weight,
            "measured_batch": [dict(rec) for rec in self.measured_batch],
            "train_seconds": round(self.train_seconds, 3),
        }


@dataclass
class _UpdateReport:
    state: str                     # "success" | "failed"
    version: int | None = None
    error: str = ""
    at: str = ""
    n_records: int = 0


class ModelRegistry:
    """线程安全的当前快照持有者；同一时刻只允许一个更新任务。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._snapshot: ModelSnapshot | None = None
        self._updating = False
        self._last_update: _UpdateReport | None = None

    # ---- 启动基线：仅合成数据（与历史版本逐字节同规格同种子） ----
    def load_baseline(self) -> ModelSnapshot:
        t0 = time.perf_counter()
        clf = train_model()
        forest = extract_forest(clf)
        samples = build_samples(forest, clf)
        snap = ModelSnapshot(
            version=1,
            clf=clf,
            forest=forest,
            samples=samples,
            measured_batch=(),
            created_at=_now_iso(),
            train_seconds=time.perf_counter() - t0,
            n_measured=0,
            label_counts={"0": 0, "1": 0},
            measured_weight=MEASURED_WEIGHT,
        )
        with self._lock:
            self._snapshot = snap
        return snap

    def current(self) -> ModelSnapshot:
        with self._lock:
            assert self._snapshot is not None, "基线快照尚未加载"
            return self._snapshot

    def is_updating(self) -> bool:
        with self._lock:
            return self._updating

    def status(self) -> dict:
        with self._lock:
            snap = self._snapshot
            last = self._last_update
            return {
                "updating": self._updating,
                "current": snap.public_info() if snap else None,
                "last_update": (
                    {
                        "state": last.state,
                        "version": last.version,
                        "error": last.error,
                        "at": last.at,
                        "n_records": last.n_records,
                    }
                    if last
                    else None
                ),
            }

    def update_with_measured_batch(self, records: list[dict]) -> ModelSnapshot:
        """整体替换实测批次并重训；全部产物就绪后原子切换。

        任何异常都不会改动当前快照；返回新快照，失败时抛出 ModelUpdateError。
        """
        errs = validate_measured_batch(records)
        if errs:
            raise ModelUpdateError("输入校验失败：" + "；".join(errs))
        # 规整成不可变的内部记录（拷贝，避免调用方后续修改）
        batch = tuple(
            {"grams": tuple(int(g) for g in rec["grams"]), "label": int(rec["label"])}
            for rec in records
        )
        with self._lock:
            if self._updating:
                raise ModelUpdateError("已有模型更新任务在执行，请等待其完成后再提交。")
            self._updating = True
            old_snapshot = self._snapshot
        try:
            t0 = time.perf_counter()
            clf = train_model(measured_records=list(batch))
            forest = extract_forest(clf)
            # 候选分类器、z3 树结构、可行/无解样例全部准备好后才允许切换
            samples = build_samples(forest, clf)
            label_counts = {"0": 0, "1": 0}
            for rec in batch:
                label_counts[str(rec["label"])] += 1
            new_snap = ModelSnapshot(
                version=old_snapshot.version + 1,
                clf=clf,
                forest=forest,
                samples=samples,
                measured_batch=batch,        # 整体替换：新批次完全取代上次实测批次
                created_at=_now_iso(),
                train_seconds=time.perf_counter() - t0,
                n_measured=len(batch),
                label_counts=label_counts,
                measured_weight=MEASURED_WEIGHT,
            )
            with self._lock:
                self._snapshot = new_snap   # 单次引用赋值即原子发布
                self._last_update = _UpdateReport(
                    state="success",
                    version=new_snap.version,
                    at=new_snap.created_at,
                    n_records=len(batch),
                )
            return new_snap
        except Exception as exc:  # 失败：旧版及其实测批次原样保留
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            with self._lock:
                self._snapshot = old_snapshot
                self._last_update = _UpdateReport(
                    state="failed",
                    version=old_snapshot.version,
                    error=detail,
                    at=_now_iso(),
                    n_records=len(batch),
                )
            raise ModelUpdateError(f"模型更新失败，旧版继续可用：{detail}") from exc
        finally:
            with self._lock:
                self._updating = False


class ModelUpdateError(Exception):
    """模型更新失败（输入不合法、重训或样例构建失败、已有更新在途）。"""
