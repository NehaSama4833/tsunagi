# tsunagi — 繋ぎ

**繋ぎ (tsunagi)** means "connection, a link". A model is trained on one table and served from another place that builds the table differently. This project learns a **data contract** from training data and checks serving batches against it, then measures which planted serving bugs it catches, what it falsely flags, and, separately, which bugs actually hurt a model.

## Setup of the experiment
Breast-cancer data (30 numeric columns + 2 derived categorical), 250 rows to fit the contract and the models (random forest, logistic regression), 150 rows to calibrate one threshold, 169 test rows. Serving batches of 100 rows, 200 trials per cell. 12 planted bugs (rename, drop, swap two columns' values, units x100 and /100, 10% nulls, numbers as strings, strings with units, 30% constant fill, duplicated column, 0.5-std shift, new category) applied to the 3 most and 3 least important columns.

## What happened
- **First run was broken, and it is kept as evidence.** With a range-check threshold I picked in advance, clean batches were flagged 94-98% of the time (`results/first_run_uncalibrated_report.md`): small-sample tails often exceed the training min/max. I fixed it by choosing the threshold on a **separate calibration set** (not the test rows).
- **After calibration:** false alarms on clean test batches **0.0% at batch size 100**, but **28.5% at batch size 50**. The calibration does not transfer to a smaller batch.
- **Detection:** structural and type bugs (rename, drop, swap, units, nulls, strings, constant fill, new category) were flagged in 99-100% of trials. Duplicated column 98-99%. The mild 0.5-std shift only **62%** (important columns) and **36%** (unimportant ones).
- **Detection is not harm.** The random forest lost at most about 1 accuracy point under any bug, even units x100. The logistic regression lost **59 points** on a x100 unit error and **45 points** on swapped values in important columns, but 0 points on dropped or renamed columns (the missing column is imputed). Many flagged bugs were harmless for these models; a contract alarm says "different", not "broken".

Full table: `results/report.md`.

## Caveats
One small dataset; bugs were written by the same author as the checks; "flagged right column" credits any flag on the affected column. Thresholds other than the range check are fixed constants chosen up front and not tuned. Not compared with Great Expectations, TFDV, pandera or Evidently, which exist and do more.

## Run
```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q            # 5 tests
.venv/bin/python -m tsunagi.experiment   # ~1 minute
```
