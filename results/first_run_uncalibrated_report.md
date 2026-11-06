# tsunagi results

Model: random forest on breast-cancer data (30 numeric + 2 derived categorical), 300 train rows; 200 trials per cell, serving batches of 200 rows from 269 held-out rows.

False-alarm rate on clean held-out batches: batch 50 = 98.0%, batch 200 = 94.5%

Top-3 important numeric columns: ['worst_perimeter', 'worst_radius', 'worst_area']; bottom-3: ['mean_compactness', 'mean_symmetry', 'symmetry_error']

| bug | target | detected | flagged right column | mean accuracy drop |
|---|---|---|---|---|
| rename | top3 | 100% | 100% | +0.009 |
| drop_column | top3 | 100% | 100% | +0.009 |
| swap_values | top3 | 100% | 100% | -0.004 |
| unit_x100 | top3 | 100% | 100% | +0.001 |
| unit_div100 | top3 | 100% | 100% | +0.010 |
| nulls_10pct | top3 | 100% | 100% | +0.001 |
| stringified | top3 | 100% | 100% | +0.000 |
| strings_with_units | top3 | 100% | 100% | +0.009 |
| const_fill_30pct | top3 | 100% | 100% | +0.003 |
| duplicate_column | top3 | 100% | 92% | +0.009 |
| shift_0.5std | top3 | 98% | 70% | +0.003 |
| rename | bottom3 | 100% | 100% | +0.002 |
| drop_column | bottom3 | 100% | 100% | +0.003 |
| swap_values | bottom3 | 100% | 100% | +0.004 |
| unit_x100 | bottom3 | 100% | 100% | +0.000 |
| unit_div100 | bottom3 | 100% | 100% | +0.003 |
| nulls_10pct | bottom3 | 100% | 100% | +0.000 |
| stringified | bottom3 | 100% | 100% | +0.000 |
| strings_with_units | bottom3 | 100% | 100% | +0.003 |
| const_fill_30pct | bottom3 | 100% | 100% | +0.001 |
| duplicate_column | bottom3 | 100% | 100% | +0.000 |
| shift_0.5std | bottom3 | 98% | 78% | +0.000 |
| new_category_20pct | cat | 100% | 100% | +0.000 |