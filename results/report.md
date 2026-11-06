# tsunagi results

Random forest and logistic regression on breast-cancer data (30 numeric + 2 derived categorical). Rows: 250 train (fit contract and models), 150 calibration (chose the range-check threshold: 0.05), 169 test. 200 trials per cell, serving batches of 100 rows.

False-alarm rate on clean test batches: batch 50 = 28.5%, batch 100 = 0.0%

Top-3 important numeric columns: ['worst_perimeter', 'worst_radius', 'worst_area']; bottom-3: ['mean_symmetry', 'smoothness_error', 'mean_compactness']

| bug | target | detected | flagged right column | acc drop (forest) | acc drop (logistic) |
|---|---|---|---|---|---|
| rename | top3 | 100% | 100% | +0.008 | +0.000 |
| drop_column | top3 | 100% | 100% | +0.007 | +0.000 |
| swap_values | top3 | 100% | 100% | -0.011 | +0.449 |
| unit_x100 | top3 | 100% | 100% | -0.010 | +0.588 |
| unit_div100 | top3 | 100% | 100% | +0.011 | +0.025 |
| nulls_10pct | top3 | 100% | 100% | +0.001 | +0.000 |
| stringified | top3 | 100% | 100% | +0.000 | +0.000 |
| strings_with_units | top3 | 100% | 100% | +0.008 | +0.000 |
| const_fill_30pct | top3 | 100% | 100% | +0.004 | +0.012 |
| duplicate_column | top3 | 98% | 98% | +0.010 | +0.044 |
| shift_0.5std | top3 | 62% | 62% | -0.008 | +0.000 |
| rename | bottom3 | 100% | 100% | -0.004 | -0.002 |
| drop_column | bottom3 | 100% | 100% | -0.004 | -0.002 |
| swap_values | bottom3 | 100% | 100% | -0.002 | +0.202 |
| unit_x100 | bottom3 | 100% | 100% | -0.002 | +0.430 |
| unit_div100 | bottom3 | 100% | 100% | -0.004 | +0.003 |
| nulls_10pct | bottom3 | 99% | 99% | -0.000 | -0.000 |
| stringified | bottom3 | 100% | 100% | +0.000 | +0.000 |
| strings_with_units | bottom3 | 100% | 100% | -0.004 | -0.002 |
| const_fill_30pct | bottom3 | 100% | 100% | -0.001 | +0.120 |
| duplicate_column | bottom3 | 99% | 99% | -0.003 | +0.191 |
| shift_0.5std | bottom3 | 36% | 36% | -0.004 | +0.000 |
| new_category_20pct | cat | 100% | 100% | +0.000 | -0.001 |