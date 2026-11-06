"""Planted serving-side bugs. Each takes (cols, rng, target, other) and returns a modified copy."""
from __future__ import annotations
import numpy as np


def _copy(cols):
    return {k: v.copy() for k, v in cols.items()}


def rename(cols, rng, t, o):
    c = _copy(cols); c[t + "_v2"] = c.pop(t); return c


def drop_column(cols, rng, t, o):
    c = _copy(cols); c.pop(t); return c


def swap_values(cols, rng, t, o):
    c = _copy(cols); c[t], c[o] = c[o].copy(), c[t].copy(); return c


def unit_scale_up(cols, rng, t, o):
    c = _copy(cols); c[t] = c[t] * 100.0; return c


def unit_scale_down(cols, rng, t, o):
    c = _copy(cols); c[t] = c[t] / 100.0; return c


def inject_nulls(cols, rng, t, o):
    c = _copy(cols); m = rng.random(len(c[t])) < 0.10
    c[t] = c[t].astype(float); c[t][m] = np.nan; return c


def stringify(cols, rng, t, o):
    c = _copy(cols); c[t] = np.array([f"{x:.4f}" for x in c[t]], dtype=object); return c


def stringify_with_units(cols, rng, t, o):
    c = _copy(cols); c[t] = np.array([f"{x:.4f}mm" for x in c[t]], dtype=object); return c


def constant_fill(cols, rng, t, o):
    c = _copy(cols); m = rng.random(len(c[t])) < 0.30
    c[t] = c[t].astype(float); c[t][m] = -1.0; return c


def duplicate_column(cols, rng, t, o):
    c = _copy(cols); c[t] = c[o].copy(); return c


def mild_shift(cols, rng, t, o):
    c = _copy(cols); x = c[t].astype(float); c[t] = x + 0.5 * x.std(); return c


def new_category(cols, rng, t, o):
    c = _copy(cols); m = rng.random(len(c[t])) < 0.20
    c[t] = c[t].astype(object); c[t][m] = "unknown_tier"; return c


NUMERIC_BUGS = {"rename": rename, "drop_column": drop_column, "swap_values": swap_values,
                "unit_x100": unit_scale_up, "unit_div100": unit_scale_down, "nulls_10pct": inject_nulls,
                "stringified": stringify, "strings_with_units": stringify_with_units,
                "const_fill_30pct": constant_fill, "duplicate_column": duplicate_column,
                "shift_0.5std": mild_shift}
CATEGORICAL_BUGS = {"new_category_20pct": new_category}
