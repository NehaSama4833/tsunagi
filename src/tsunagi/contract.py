"""A data contract learned from training data and checked against serving batches.
Columns are dicts of name -> numpy array (float with NaN for numeric, object/str for categorical).
All thresholds are fixed constants chosen before any experiment."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

NULL_TOL = 0.02          # extra null rate allowed over training
OUT_FRAC = 0.01          # fraction outside widened training range
RANGE_PAD = 0.10         # widen [min, max] by this fraction of the range
SHIFT_STD = 0.5          # |median shift| in training std units
SPREAD = (0.5, 2.0)      # allowed batch/train std ratio
UNSEEN_FRAC = 0.01       # fraction of unseen categories
MODE_EXTRA = 0.20        # extra mass on the most frequent value (constant fill)


@dataclass
class Violation:
    kind: str
    column: str
    detail: str = ""


def _is_str(v) -> bool:
    return isinstance(v, str)


def _numeric(arr) -> np.ndarray:
    """Float view of an array; unparseable entries become NaN."""
    if arr.dtype != object:
        return arr.astype(float)
    out = np.empty(len(arr))
    for i, x in enumerate(arr):
        try:
            out[i] = float(x)
        except (TypeError, ValueError):
            out[i] = np.nan
    return out


@dataclass
class ColumnSpec:
    kind: str                        # "num" or "cat"
    null_rate: float
    stats: dict = field(default_factory=dict)


@dataclass
class Contract:
    columns: dict[str, ColumnSpec]
    out_frac: float = OUT_FRAC

    @classmethod
    def fit(cls, cols: dict[str, np.ndarray]) -> "Contract":
        specs = {}
        for name, a in cols.items():
            if a.dtype == object and any(_is_str(x) for x in a if x is not None) and not _all_numeric_strings(a):
                vals = [x for x in a if x is not None]
                uniq, cnt = np.unique(vals, return_counts=True)
                specs[name] = ColumnSpec("cat", 1 - len(vals) / len(a),
                                         {"values": set(uniq.tolist()), "mode_frac": cnt.max() / len(vals)})
            else:
                x = _numeric(a)
                ok = x[~np.isnan(x)]
                mode_frac = np.unique(ok, return_counts=True)[1].max() / len(ok)
                specs[name] = ColumnSpec("num", float(np.isnan(x).mean()),
                                         {"min": ok.min(), "max": ok.max(), "median": float(np.median(ok)),
                                          "std": float(ok.std()) or 1.0, "mode_frac": float(mode_frac)})
        return cls(specs)

    def calibrate_range(self, batches, target=0.05, grid=(0.01, 0.02, 0.05, 0.10, 0.15, 0.25)) -> float:
        """Pick the smallest out-of-range fraction threshold whose false-alarm rate (any violation)
        on the given CLEAN batches is <= target. Uses rows that are neither train nor test."""
        for g in grid:
            self.out_frac = g
            if np.mean([bool(self.check(b)) for b in batches]) <= target:
                return g
        return grid[-1]

    def check(self, cols: dict[str, np.ndarray]) -> list[Violation]:
        v: list[Violation] = []
        for name in self.columns:
            if name not in cols:
                v.append(Violation("missing_column", name))
        for name in cols:
            if name not in self.columns:
                v.append(Violation("extra_column", name))
        for name, spec in self.columns.items():
            if name not in cols:
                continue
            a = cols[name]
            if spec.kind == "num":
                v += self._check_num(name, spec, a)
            else:
                v += self._check_cat(name, spec, a)
        return v

    def _check_num(self, name, spec, a):
        out = []
        x = _numeric(a)
        if a.dtype == object and any(_is_str(t) for t in a):
            out.append(Violation("dtype", name, "numeric column arrived as strings"))
        nulls = float(np.isnan(x).mean())
        if nulls > spec.null_rate + NULL_TOL:
            out.append(Violation("nulls", name, f"{nulls:.2f} vs {spec.null_rate:.2f}"))
        ok = x[~np.isnan(x)]
        if len(ok) < 5:
            return out
        s = spec.stats
        pad = RANGE_PAD * (s["max"] - s["min"])
        frac_out = float(np.mean((ok < s["min"] - pad) | (ok > s["max"] + pad)))
        if frac_out > self.out_frac:
            out.append(Violation("range", name, f"{frac_out:.2f} outside"))
        shift = abs(float(np.median(ok)) - s["median"]) / s["std"]
        if shift > SHIFT_STD:
            out.append(Violation("shift", name, f"{shift:.2f} std"))
        ratio = float(ok.std()) / s["std"]
        if not (SPREAD[0] <= ratio <= SPREAD[1]):
            out.append(Violation("spread", name, f"{ratio:.2f}x"))
        mode = np.unique(ok, return_counts=True)[1].max() / len(ok)
        if mode > s["mode_frac"] + MODE_EXTRA:
            out.append(Violation("constant_fill", name, f"{mode:.2f} on one value"))
        return out

    def _check_cat(self, name, spec, a):
        out = []
        vals = [x for x in a if x is not None]
        nulls = 1 - len(vals) / len(a)
        if nulls > spec.null_rate + NULL_TOL:
            out.append(Violation("nulls", name, f"{nulls:.2f}"))
        if vals and not all(_is_str(x) for x in vals):
            out.append(Violation("dtype", name, "non-string in categorical column"))
        unseen = float(np.mean([x not in spec.stats["values"] for x in vals])) if vals else 0.0
        if unseen > UNSEEN_FRAC:
            out.append(Violation("unseen_category", name, f"{unseen:.2f}"))
        return out


def _all_numeric_strings(a) -> bool:
    try:
        for x in a:
            if x is not None:
                float(x)
        return True
    except (TypeError, ValueError):
        return False
