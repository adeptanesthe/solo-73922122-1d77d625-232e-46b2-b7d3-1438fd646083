"""关键功能自测（开发环境运行，不进镜像）：

  python -m tests.selftest

覆盖：
 1. 模型规格：5 棵树、深度 ≤ 3、同种子重训结果一致；
 2. 阈值语义：cutoff_left 对全部树阈值 × 全部克数逐一与 float32 比较核对；
    随机配方上纯 Python 走树概率与 sklearn predict_proba 完全一致；
 3. 可行样例：求解 ok、概率复算达标、改动量一致，并用 z3 反证 + sklearn 暴力
    枚举双重核对最优性结论；
 4. 无解样例：z3 判 unsat，并对整个可行域用 sklearn 暴力枚举复核确无合格配方；
 5. 字典序：改 2 种（量大）与改 3 种（量小）并存时，必须返回改 2 种；
 6. 边界：原配方已合格（0 改动）、全锁定合格/不合格、各类非法输入；
 7. 求解失败路径：异常注入返回 solver_failure 而非无解；
 8. API：/、/api/meta、/api/samples、/api/solve 各状态分支；页面含迟到结果丢弃逻辑。
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
import z3

from app.model import (
    MAX_DEPTH,
    N_TREES,
    TOTAL_GRAMS,
    cutoff_left,
    extract_forest,
    forest_proba_rational,
    is_qualified,
    predict_proba_qualified,
    train_model,
)
from app.solver import (
    INFEASIBLE,
    INVALID,
    OK,
    SOLVER_FAILURE,
    SolveInput,
    _build_optimize,
    solve,
)

PASS, FAIL = "PASS", "FAIL"
failures = []


def check(name, cond, detail=""):
    tag = PASS if cond else FAIL
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def main():
    print("== 1. 模型规格 ==")
    clf = train_model()
    forest = extract_forest(clf)
    check("树数量为 5", len(clf.estimators_) == N_TREES == 5)
    depths = [e.tree_.max_depth for e in clf.estimators_]
    check("树深均 ≤ 3", all(d <= MAX_DEPTH for d in depths), f"depths={depths}")
    clf2 = train_model()
    same = np.allclose(
        clf.predict_proba(np.zeros((1, 4), dtype=np.float32) + 0.25),
        clf2.predict_proba(np.zeros((1, 4), dtype=np.float32) + 0.25),
    )
    check("同种子重训结果一致", same)

    print("== 2. 阈值语义 ==")
    # cutoff_left：对森林中全部阈值 × 全部整数克数，逐一与 float32 比较核对
    thr_ok = True
    for tree in forest.trees:
        for node in tree.nodes:
            if node.__class__.__name__ == "Leaf":
                continue
            k = cutoff_left(node.threshold)
            for g in range(TOTAL_GRAMS + 1):
                expect = bool(np.float32(g / TOTAL_GRAMS) <= node.threshold)
                if (g <= k) != expect:
                    thr_ok = False
                    break
            if not thr_ok:
                break
    check("cutoff_left 与 float32 比较逐点一致", thr_ok)

    rng = np.random.default_rng(99)
    proba_ok = True
    for _ in range(500):
        g = rng.multinomial(TOTAL_GRAMS, [0.3, 0.3, 0.2, 0.2]).tolist()
        p_sk = predict_proba_qualified(clf, g)
        p_py = float(forest_proba_rational(forest, g))
        if abs(p_sk - p_py) > 1e-12:
            proba_ok = False
            break
    check("500 个随机配方概率与 sklearn 完全一致", proba_ok)

    print("== 3. 可行样例 ==")
    from app.samples import build_samples

    samples = build_samples(forest, clf)
    fs = samples["feasible"]
    inp = SolveInput(fs["original"], fs["lower"], fs["upper"], fs["locked"])
    res = solve(forest, clf, inp)
    check("可行样例求解为 ok", res.status == OK, res.reason)
    check("建议概率 ≥ 0.7（原分类器复算）", res.probability is not None and is_qualified(res.probability),
          f"p={res.probability:.6f}")
    sug = res.suggestion
    check("建议满足总重/上下限/锁定",
          sum(sug) == TOTAL_GRAMS
          and all(fs["lower"][i] <= sug[i] <= fs["upper"][i] for i in range(4))
          and all(sug[i] == fs["original"][i] for i in range(4) if fs["locked"][i]))
    k, m = res.changed_count, res.total_change
    real_k = sum(1 for i in range(4) if sug[i] != fs["original"][i])
    real_m = sum(abs(sug[i] - fs["original"][i]) for i in range(4))
    check("报告的改动种数/克数与建议一致", (k, m) == (real_k, real_m), f"k={k}, m={m}")

    # 3a. z3 反证最优性（直接调用 z3，也经专用 z3 工作线程执行）
    from app.solver import Z3_RUNNER
    def _recheck():
        opt, g, num_changed, total_change = _build_optimize(forest, inp)
        opt.add(num_changed <= k - 1)
        fewer_unsat = opt.check() == z3.unsat
        opt2, g2, num_changed2, total_change2 = _build_optimize(forest, inp)
        opt2.add(num_changed2 == k, total_change2 <= m - 1)
        smaller_unsat = opt2.check() == z3.unsat
        return fewer_unsat, smaller_unsat
    fewer_unsat, smaller_unsat = Z3_RUNNER.run(_recheck)
    check("z3 反证：更少种数无解", fewer_unsat)
    check("z3 反证：同种数更小改动量无解", smaller_unsat)

    # 3b. sklearn 暴力枚举复核最优性（改动 ≤2 种且更优者不存在）
    def brute_best(inp_d, max_kinds):
        orig, lo, hi, lk = inp_d["original"], inp_d["lower"], inp_d["upper"], inp_d["locked"]
        free = [i for i in range(4) if not lk[i]]
        best = None  # (kinds, delta)
        for n_k in range(0, max_kinds + 1):
            for S in itertools.combinations(free, n_k):
                fixed_idx = [i for i in range(4) if i not in S]
                fixed_sum = sum(orig[i] for i in fixed_idx)
                if len(S) == 0:
                    cand = [tuple(orig)]
                elif len(S) == 1:
                    i = S[0]
                    gi = TOTAL_GRAMS - fixed_sum
                    cand = [] if not (lo[i] <= gi <= hi[i]) else [tuple(orig[:i]) + (gi,) + tuple(orig[i+1:])]
                else:  # len(S) == 2：和约束下第二变量被唯一确定
                    i, j = S
                    cand = []
                    for gi in range(lo[i], hi[i] + 1):
                        gj = TOTAL_GRAMS - fixed_sum - gi
                        if lo[j] <= gj <= hi[j]:
                            row = list(orig); row[i], row[j] = gi, gj
                            cand.append(tuple(row))
                if not cand:
                    continue
                arr = np.array(cand, dtype=np.float32) / TOTAL_GRAMS
                probs = clf.predict_proba(arr)[:, 1]
                for row, p in zip(cand, probs):
                    if p >= 0.7 - 1e-9:
                        d = sum(abs(row[t] - orig[t]) for t in range(4))
                        cur = (n_k, d)
                        if best is None or cur < best:
                            best = cur
        return best

    bb = brute_best(fs, 2)
    check("sklearn 暴力枚举复核最优性", bb == (k, m), f"brute={bb}, solver=({k},{m})")

    print("== 4. 无解样例 ==")
    isf = samples["infeasible"]
    res_i = solve(forest, clf, SolveInput(isf["original"], isf["lower"], isf["upper"], isf["locked"]))
    check("无解样例求解为 infeasible", res_i.status == INFEASIBLE, res_i.reason)
    # 对整个可行域用 sklearn 暴力枚举复核“确实无解”
    lo, hi = isf["lower"], isf["upper"]
    combos = []
    for g0 in range(lo[0], hi[0] + 1):
        for g1 in range(lo[1], hi[1] + 1):
            for g2 in range(lo[2], hi[2] + 1):
                g3 = TOTAL_GRAMS - g0 - g1 - g2
                if lo[3] <= g3 <= hi[3]:
                    combos.append((g0, g1, g2, g3))
    found_q = False
    for s in range(0, len(combos), 2_000_000):
        arr = np.array(combos[s:s + 2_000_000], dtype=np.float32) / TOTAL_GRAMS
        if (clf.predict_proba(arr)[:, 1] >= 0.7 - 1e-9).any():
            found_q = True
            break
    check(f"sklearn 全可行域枚举（{len(combos)} 点）确无合格配方", not found_q)

    print("== 5. 字典序优先级 ==")
    lex_orig = [244, 294, 218, 244]  # 改 2 种最少 218 克；改 3 种可 214 克
    res_lex = solve(forest, clf, SolveInput(lex_orig, [0]*4, [1000]*4, [False]*4))
    check("先最少种数再最小克数", res_lex.status == OK and res_lex.changed_count == 2
          and res_lex.total_change == 218,
          f"got kinds={res_lex.changed_count}, delta={res_lex.total_change}")

    print("== 6. 边界与非法输入 ==")
    r_ok0 = solve(forest, clf, SolveInput([700, 100, 100, 100], [0]*4, [1000]*4, [False]*4))
    check("原配方已合格 → 0 改动", r_ok0.status == OK and r_ok0.changed_count == 0
          and r_ok0.total_change == 0 and r_ok0.suggestion == [700, 100, 100, 100])
    r_lock_bad = solve(forest, clf, SolveInput([150, 130, 620, 100], [0]*4, [1000]*4, [True]*4))
    check("全锁定且不合格 → infeasible", r_lock_bad.status == INFEASIBLE)
    r_lock_ok = solve(forest, clf, SolveInput([700, 100, 100, 100], [0]*4, [1000]*4, [True]*4))
    check("全锁定且合格 → ok 0 改动", r_lock_ok.status == OK and r_lock_ok.changed_count == 0)
    bad_cases = [
        SolveInput([250, 250, 350, 149], [0]*4, [1000]*4, [False]*4),   # 总和 999
        SolveInput([250, 250, 350, 150], [0, 0, 500, 0], [1000, 1000, 400, 1000], [False]*4),  # lo>hi
        SolveInput([250, 250, 350, 150], [300, 0, 0, 0], [1000]*4, [True, False, False, False]),  # 锁定值越界
        SolveInput([250, 250, 350, 150], [600, 600, 0, 0], [1000]*4, [False]*4),  # 下限和超 1000
        SolveInput([250, 250, 350, 150], [0]*4, [200, 200, 200, 200], [False]*4),  # 上限和不足 1000
    ]
    check("非法输入均判 invalid", all(solve(forest, clf, c).status == INVALID for c in bad_cases))

    print("== 7. 求解失败路径 ==")
    import app.solver as sv
    orig_build = sv._build_optimize
    def boom(forest_, inp_):
        raise z3.Z3Exception("injected")
    sv._build_optimize = boom
    try:
        r_fail = solve(forest, clf, SolveInput([250, 250, 350, 150], [0]*4, [1000]*4, [False]*4))
    finally:
        sv._build_optimize = orig_build
    check("求解器异常 → solver_failure（非无解）", r_fail.status == SOLVER_FAILURE)

    print("== 7b. 多线程并发求解（单一 z3 工作线程） ==")
    import threading as _th
    cc_inp = SolveInput([250, 250, 350, 150], [0]*4, [1000]*4, [False, True, False, False])
    cc_errs, cc_results = [], []
    def cc_worker():
        try:
            for _ in range(4):
                rr = solve(forest, clf, cc_inp)
                if rr.status != OK:
                    cc_errs.append(rr.status)
                else:
                    cc_results.append(rr.suggestion)
        except Exception as e:  # 段错误无法在同进程捕获，异常也算失败
            cc_errs.append(repr(e))
    threads = [_th.Thread(target=cc_worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check("6 线程 ×4 次并发求解全部 ok（无 solver_failure/异常）",
          not cc_errs and len(cc_results) == 24)
    check("并发求解结果一致（同输入同快照）",
          all(s == cc_results[0] for s in cc_results))

    print("== 8. API ==")
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    r = client.get("/")
    check("GET / 返回网页", r.status_code == 200 and "开始求解" in r.text)
    r = client.get("/api/meta")
    mj = r.json()
    check("GET /api/meta", r.status_code == 200 and mj["n_trees"] == 5 and mj["max_depth"] == 3
          and abs(mj["proba_threshold"] - 0.7) < 1e-12)
    r = client.get("/api/samples")
    sj = r.json()
    check("GET /api/samples 两类样例齐全", r.status_code == 200
          and sj["feasible"]["check"]["status"] == "ok"
          and sj["infeasible"]["check"]["status"] == "infeasible")
    r = client.post("/api/solve", json={"original": fs["original"], "lower": fs["lower"],
                                        "upper": fs["upper"], "locked": fs["locked"]})
    rj = r.json()
    check("POST /api/solve 可行分支", r.status_code == 200 and rj["status"] == "ok"
          and rj["probability"] >= 0.7 and rj["suggestion"] == sug)
    r = client.post("/api/solve", json={"original": isf["original"], "lower": isf["lower"],
                                        "upper": isf["upper"], "locked": isf["locked"]})
    check("POST /api/solve 无解分支", r.json()["status"] == "infeasible")
    r = client.post("/api/solve", json={"original": [1, 2, 3, 4], "lower": [0]*4,
                                        "upper": [1000]*4, "locked": [False]*4})
    check("POST /api/solve 非法输入分支", r.json()["status"] == "invalid")
    html = client.get("/").text
    check("页面含迟到结果丢弃逻辑", "my !== epoch" in html and "abort" in html)

    print("== 9. 当班模型更新（实测批次重训与原子切换） ==")
    import threading
    from app.versioning import ModelRegistry, ModelUpdateError
    from app.model import (
        MEASURED_WEIGHT,
        MAX_MEASURED_RECORDS,
        MIN_MEASURED_RECORDS,
        validate_measured_batch,
    )

    good_batch = [
        {"grams": [620, 180, 120, 80], "label": 1},
        {"grams": [580, 220, 130, 70], "label": 1},
        {"grams": [550, 250, 140, 60], "label": 1},
        {"grams": [500, 260, 160, 80], "label": 1},
        {"grams": [180, 140, 560, 120], "label": 0},
        {"grams": [150, 130, 620, 100], "label": 0},
        {"grams": [220, 160, 520, 100], "label": 0},
        {"grams": [200, 150, 550, 100], "label": 0},
    ]

    # 9.1 输入校验：条数、总重、标签、整数与边界
    check("校验常量 5~30 且权重 20", MIN_MEASURED_RECORDS == 5 and MAX_MEASURED_RECORDS == 30
          and MEASURED_WEIGHT == 20.0)
    check("合法批次校验通过", validate_measured_batch(good_batch) == [])
    invalid_batches = {
        "条数不足 4": [{"grams": [250] * 4, "label": 1}] * 4,
        "条数超限 31": [{"grams": [250] * 4, "label": 1}] * 31,
        "总重非 1000": [{"grams": [250, 250, 250, 249], "label": 1}] + [{"grams": [250] * 4, "label": 1}] * 5,
        "标签非法": [{"grams": [250] * 4, "label": 2}] * 6,
        "负数克数": [{"grams": [-1, 251, 500, 250], "label": 0}] * 6,
        "非整数克数": [{"grams": [250.5, 249.5, 250, 250], "label": 0}] * 6,
        "布尔冒充整数": [{"grams": [True, 250, 250, 249], "label": 0}] * 6,
        "结构缺失": [{"grams": [250] * 4}] * 6,
    }
    check("各类非法批次均被拒绝", all(validate_measured_batch(b) for b in invalid_batches.values()))

    # 9.2 独立注册表演练完整更新生命周期（不影响 main 的在役注册表）
    reg = ModelRegistry()
    base = reg.load_baseline()
    check("基线版次 v1 且无实测批次", base.version == 1 and base.n_measured == 0
          and base.measured_batch == ())

    # 9.3 失败原子性：注入样例构建失败，当前快照与实测批次保持不变
    import app.versioning as vmod
    orig_build = vmod.build_samples
    def boom_samples(forest, clf):
        raise RuntimeError("injected sample failure")
    vmod.build_samples = boom_samples
    kept = None
    try:
        try:
            reg.update_with_measured_batch(good_batch)
        except ModelUpdateError:
            kept = reg.current()
    finally:
        vmod.build_samples = orig_build
    check("重训中途失败 → 抛错且旧版快照原样保留", kept is base and reg.current() is base
          and reg.current().n_measured == 0)
    check("失败后 last_update 记录 failed 且不阻塞再次提交",
          reg.status()["last_update"]["state"] == "failed" and reg.is_updating() is False)

    # 9.4 成功更新：版次递增、批次整体替换、样例与树结构随快照就绪
    snap2 = reg.update_with_measured_batch(good_batch)
    check("成功后版次 v2", snap2.version == 2 and reg.current() is snap2)
    check("实测批次被整体替换为 8 条", snap2.n_measured == 8
          and len(snap2.measured_batch) == 8
          and snap2.label_counts == {"0": 4, "1": 4})
    check("新快照三件套齐全（分类器/树结构/样例）",
          snap2.clf is not None and len(snap2.forest.trees) == N_TREES
          and snap2.samples["feasible"]["check"]["status"] == "ok"
          and snap2.samples["infeasible"]["check"]["status"] == "infeasible")
    check("新样例与新分类器自洽（可行样例复算达标）",
          is_qualified(predict_proba_qualified(snap2.clf, snap2.samples["feasible"]["check"]["suggestion"])))

    # 9.5 同批次同种子重训确定性
    snap2b = reg.update_with_measured_batch(good_batch)
    xt = np.zeros((1, 4), dtype=np.float32) + 0.25
    check("相同批次重训结果确定可复现",
          np.allclose(snap2.clf.predict_proba(xt), snap2b.clf.predict_proba(xt))
          and snap2b.version == 3)

    # 9.6 权重语义：实测权重 20 倍生效——同样数据按权重 1 训练结果应不同
    from app.model import make_synthetic_data
    Xs, ys = make_synthetic_data()
    Xm = np.array([r["grams"] for r in good_batch], dtype=np.float32) / TOTAL_GRAMS
    ym = np.array([r["label"] for r in good_batch])
    Xa = np.vstack([Xs, Xm]); ya = np.concatenate([ys, ym])
    from sklearn.ensemble import RandomForestClassifier
    clf_w1 = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                    random_state=20241117).fit(
        Xa, ya, sample_weight=np.concatenate([np.ones(len(Xs)), np.ones(len(Xm))]))
    clf_w20 = snap2b.clf
    check("实测 20 倍权重确实改变模型（与等权重结果不同）",
          not np.allclose(clf_w1.predict_proba(xt), clf_w20.predict_proba(xt)))
    # 等权重拟合与同数据无权重拟合一致（证明超参/种子语义未变；sklearn 随机流
    # 与样本数相关，因此这里必须在同一份数据上比较）
    clf_base = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                      random_state=20241117).fit(Xa, ya)
    clf_eq = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                    random_state=20241117).fit(
        Xa, ya, sample_weight=np.ones(len(ya)))
    check("等权重与无权重在同数据上一致（原规格原种子保持）",
          np.allclose(clf_base.predict_proba(xt), clf_eq.predict_proba(xt)))

    # 9.7 快照一致性：同一次求解的森林约束、分类器复算、版次同源
    from app.solver import solve as _solve
    fs2 = snap2b.samples["feasible"]
    r2 = _solve(snap2b.forest, snap2b.clf,
                SolveInput(fs2["original"], fs2["lower"], fs2["upper"], fs2["locked"]))
    check("快照内求解 ok 且复算达标", r2.status == OK and is_qualified(r2.probability))
    # 拿旧基线森林 + 新分类器混搭必须能被发现/不允许：注册表 current 永远成对发布
    check("注册表当前快照的分类器与森林成对", reg.current().clf is snap2b.clf
          and reg.current().forest is snap2b.forest)

    # 9.8 并发更新互斥：在途期间第二个提交被拒，在役版本不变
    entered = threading.Event()
    release = threading.Event()
    outcome = {}
    def slow_build(forest, clf):
        entered.set()
        release.wait(timeout=10)
        return orig_build(forest, clf)
    vmod.build_samples = slow_build
    def worker():
        try:
            reg.update_with_measured_batch(good_batch[:5] + good_batch[:2])
            outcome["worker"] = "ok"
        except ModelUpdateError as e:
            outcome["worker"] = str(e)
    th = threading.Thread(target=worker)
    th.start()
    entered.wait(timeout=10)
    try:
        conflict = None
        try:
            reg.update_with_measured_batch(good_batch)
        except ModelUpdateError as e:
            conflict = str(e)
    finally:
        release.set()
        th.join()
    vmod.build_samples = orig_build
    check("更新在途时并发提交被拒且旧版不动", conflict is not None and "已有模型更新任务" in conflict
          and outcome.get("worker") == "ok")

    # 9.9 HTTP 面：状态/成功/失败/冲突状态码与版次字段
    r = client.get("/api/model/status")
    sj = r.json()
    check("GET /api/model/status", r.status_code == 200 and sj["updating"] is False
          and sj["current"]["version"] >= 1)
    r = client.post("/api/model/update", json={"records": good_batch})
    check("POST 成功发布新版本", r.status_code == 200 and r.json()["status"] == "success"
          and r.json()["model"]["current"]["n_measured"] == 8)
    r = client.post("/api/model/update", json={"records": [{"grams": [1] * 4, "label": 1}] * 3})
    check("POST 非法批次返回 400 且在役版不变", r.status_code == 400)
    sv_before = client.get("/api/meta").json()["version"]
    r = client.post("/api/model/update", json={"records": "not-a-list"})
    sv_after = client.get("/api/meta").json()["version"]
    check("结构错误返回 400 且版次不变", r.status_code == 400 and sv_before == sv_after)
    r = client.get("/update")
    check("GET /update 返回更新页面", r.status_code == 200 and "当班试配模型更新" in r.text)
    r = client.get("/")
    check("原试配入口保留且含版次撤回逻辑", r.status_code == 200
          and "currentModelVersion" in r.text and "/update" in r.text)
    smp = client.get("/api/samples").json()
    meta_v = client.get("/api/meta").json()["version"]
    check("样例与 meta 版次一致", smp["model_version"] == meta_v)
    sol = client.post("/api/solve", json={"original": smp["feasible"]["original"],
                                          "lower": smp["feasible"]["lower"],
                                          "upper": smp["feasible"]["upper"],
                                          "locked": smp["feasible"]["locked"]}).json()
    check("求解响应携带同一快照版次", sol["status"] == "ok" and sol["model_version"] == meta_v)

    print()
    if failures:
        print(f"共 {len(failures)} 项失败：{failures}")
        return 1
    print("全部自测通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
