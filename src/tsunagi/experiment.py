"""python -m tsunagi.experiment  -> results/report.md, results/results.json"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .bugs import CATEGORICAL_BUGS, NUMERIC_BUGS
from .contract import Contract
from .serve import Service, make_data

TRIALS = 200
N_TRAIN, N_CAL, BATCH = 250, 150, 100


def take(cols, idx):
    return {k: v[idx] for k, v in cols.items()}


def main():
    cols, y = make_data(0)
    n = len(y)
    tr = np.arange(N_TRAIN)
    cal = np.arange(N_TRAIN, N_TRAIN + N_CAL)
    te = np.arange(N_TRAIN + N_CAL, n)
    contract = Contract.fit(take(cols, tr))
    rng = np.random.default_rng(1)
    cal_batches = [take(cols, rng.choice(cal, 100, replace=False)) for _ in range(100)]
    chosen = contract.calibrate_range(cal_batches)
    res = {"false_alarm": {}, "bugs": {}, "out_frac_chosen": chosen}
    models = {"rf": Service(take(cols, tr), y[tr], kind="rf"), "lr": Service(take(cols, tr), y[tr], kind="lr")}
    svc = models["rf"]

    for size in (50, BATCH):
        fa = 0
        for _ in range(TRIALS):
            idx = rng.choice(te, size, replace=False)
            if contract.check(take(cols, idx)):
                fa += 1
        res["false_alarm"][size] = fa / TRIALS

    groups = {"top3": svc.top_numeric(3), "bottom3": svc.bottom_numeric(3)}
    for gname, targets in groups.items():
        for bname, fn in NUMERIC_BUGS.items():
            det = right = 0
            drops, total = {"rf": [], "lr": []}, 0
            for t in range(TRIALS):
                idx = rng.choice(te, BATCH, replace=False)
                clean = take(cols, idx)
                tgt = targets[t % 3]
                oth = targets[(t + 1) % 3] if bname == "swap_values" else [k for k in svc.num if k != tgt][int(rng.integers(len(svc.num) - 1))]
                bad = fn(clean, rng, tgt, oth)
                viol = contract.check(bad)
                total += 1
                det += bool(viol)
                names = {tgt, tgt + "_v2", oth}
                right += any(x.column in names for x in viol)
                for mk, m in models.items():
                    drops[mk].append(float((m.predict(clean) == y[idx]).mean() - (m.predict(bad) == y[idx]).mean()))
            res["bugs"][f"{gname}|{bname}"] = {"detected": det / total, "flagged_right_column": right / total,
                                             "acc_drop_rf": float(np.mean(drops["rf"])), "acc_drop_lr": float(np.mean(drops["lr"]))}
    for bname, fn in CATEGORICAL_BUGS.items():
        det = right = 0; drops = {"rf": [], "lr": []}
        for t in range(TRIALS):
            idx = rng.choice(te, BATCH, replace=False)
            clean = take(cols, idx)
            bad = fn(clean, rng, "size_tier", "texture_tier")
            viol = contract.check(bad)
            det += bool(viol); right += any(x.column == "size_tier" for x in viol)
            for mk, m in models.items():
                drops[mk].append(float((m.predict(clean) == y[idx]).mean() - (m.predict(bad) == y[idx]).mean()))
        res["bugs"][f"cat|{bname}"] = {"detected": det / TRIALS, "flagged_right_column": right / TRIALS,
                                      "acc_drop_rf": float(np.mean(drops["rf"])), "acc_drop_lr": float(np.mean(drops["lr"]))}
    out = Path("results"); out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    L = ["# tsunagi results\n",
         f"Random forest and logistic regression on breast-cancer data (30 numeric + 2 derived categorical). Rows: {N_TRAIN} train (fit contract and models), "
         f"{N_CAL} calibration (chose the range-check threshold: {chosen}), {len(te)} test. {TRIALS} trials per cell, serving batches of {BATCH} rows.\n",
         f"False-alarm rate on clean test batches: batch 50 = {res['false_alarm'][50]:.1%}, batch {BATCH} = {res['false_alarm'][BATCH]:.1%}\n",
         f"Top-3 important numeric columns: {groups['top3']}; bottom-3: {groups['bottom3']}\n",
         "| bug | target | detected | flagged right column | acc drop (forest) | acc drop (logistic) |", "|---|---|---|---|---|---|"]
    for k, v in res["bugs"].items():
        g, b = k.split("|")
        L.append(f"| {b} | {g} | {v['detected']:.0%} | {v['flagged_right_column']:.0%} | {v['acc_drop_rf']:+.3f} | {v['acc_drop_lr']:+.3f} |")
    (out / "report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
