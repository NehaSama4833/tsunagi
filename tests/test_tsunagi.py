import numpy as np
from tsunagi.contract import Contract
from tsunagi import bugs
from tsunagi.serve import make_data

rng = np.random.default_rng(0)


def base():
    return {"a": rng.normal(10, 2, 400), "b": rng.uniform(0, 1, 400),
            "c": np.array(rng.choice(["x", "y", "z"], 400), dtype=object)}


def kinds(v):
    return {x.kind for x in v}


def test_clean_batches_from_same_distribution_raise_few_alarms():
    tr = base(); k = Contract.fit(tr)
    alarms = sum(bool(k.check({n: v[:200] for n, v in base().items()})) for _ in range(100))
    assert alarms <= 8  # fixed thresholds incl. range check; may be flaky for heavy tails


def test_hand_built_violations_are_named_correctly():
    tr = base(); k = Contract.fit(tr)
    t = {n: v[:200].copy() for n, v in tr.items()}
    assert kinds(k.check({n: v for n, v in t.items() if n != "a"})) == {"missing_column"}
    assert "extra_column" in kinds(k.check({**t, "d": t["a"]}))
    assert "range" in kinds(k.check({**t, "a": t["a"] * 100}))
    s = t["b"].copy(); s[:60] = np.nan
    assert "nulls" in kinds(k.check({**t, "b": s}))
    assert "dtype" in kinds(k.check({**t, "b": np.array([f"{x:.3f}" for x in t["b"]], dtype=object)}))
    cc = t["c"].copy(); cc[:30] = "w"
    assert "unseen_category" in kinds(k.check({**t, "c": cc}))
    f = t["b"].copy(); f[:80] = -1
    assert "constant_fill" in kinds(k.check({**t, "b": f}))


def test_swapping_two_columns_is_caught_by_distribution_checks():
    tr = base(); k = Contract.fit(tr)
    t = {n: v[:200].copy() for n, v in tr.items()}
    swapped = bugs.swap_values(t, rng, "a", "b")
    assert {x.column for x in k.check(swapped)} >= {"a", "b"}


def test_bugs_do_not_mutate_input():
    cols, _ = make_data(0)
    snapshot = {k: v.copy() for k, v in cols.items()}
    for fn in bugs.NUMERIC_BUGS.values():
        fn(cols, rng, "mean_radius", "mean_texture")
    assert all((cols[k] == snapshot[k]).all() for k in cols)


def test_numeric_strings_with_units_become_nan_and_flag_dtype_and_nulls():
    tr = base(); k = Contract.fit(tr)
    t = {n: v[:200].copy() for n, v in tr.items()}
    bad = bugs.stringify_with_units(t, rng, "a", "b")
    assert {"dtype", "nulls"} <= kinds(k.check(bad))
