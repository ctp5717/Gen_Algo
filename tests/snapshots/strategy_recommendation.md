# Strategy Recommendation Report

## Overall Confidence
Medium (63)

Confidence Medium (63). Median fitness 0.50, 66.7% positive folds; worst fold -0.30 and downside deviation 0.00.

### Confidence Factors
Folds: median 0.50, worst -0.30, positive 66.7%.
| Factor | Score |
|---|---|
| Median Fitness | 33.3 |
| Consistency | 66.7 |
| Tail (worst fold) | 85.0 |
| Downside Deviation | 100.0 |

## Asset Summary
Stars: BBB; Stalwarts: AAA; All assets have ≥3 qualifying fold(s).

## Parameter Summary
Parameters appear stable.

## Asset Performance Matrix
| Ticker | Performance | Consistency | Class | Samples |
|---|---|---|---|---|
| BBB | 1.20 | 100.0% | Stars | 3 |
| AAA | 0.50 | 66.7% | Stalwarts | 3 |

Legend: Stars ≥1.0 perf & ≥70% consistency; Stalwarts 0.0–1.0 perf & ≥60% consistency; Gambles ≥1.0 perf & <50% consistency; Drags <0.0 perf & <50% consistency

## Parameter Stability
No unstable parameters detected.

## SRE Config
### Category Cutoffs
- High: ≥80
- Medium: ≥50
### Weights
- median: 0.35
- consistency: 0.35
- tail: 0.15
- downside: 0.15
### Asset Class Thresholds
- Stars: ≥1.0 perf & ≥70% consistency
- Stalwarts: 0.0–1.0 perf & ≥60% consistency
- Gambles: ≥1.0 perf & <50% consistency
- Drags: <0.0 perf & <50% consistency

### Stability Regularizer
- Enabled: False
- Alpha: 0.1
- Genes: rsi_period

### Logging
- ENV: —
- IS_PROD: False
- LOG_UNKNOWN_COLUMNS_ON_SUCCESS: True (dev default)