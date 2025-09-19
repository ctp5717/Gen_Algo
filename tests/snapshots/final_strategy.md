# Final Strategy
## Overview
Confidence: Medium (71)
Fold selection: Elite/Viable
Recency weighting: enabled
Weighting scheme: risk_adjusted — weights ∝ (performance / volatility) × consistency (cap 0.35, floor 0.02)
## Recommended Parameters
| Gene | Value | Stability | Distribution |
| --- | --- | --- | --- |
| alpha | 10 | Stable |  10.000 ┤ 10.000 ┼ 10.000 ┼ 11.000 ┤  11.000 |
| beta | 1.500 | Stable |   1.500 ┤  1.500 ┼  1.500 ┼  1.600 ┤   1.600 |
## Asset Allocation
| Ticker | Class | Performance | Consistency | Volatility | Weight |
| --- | --- | ---: | ---: | ---: | ---: |
| AAA | Stars | 1.100 | 80.0% | 0.1000 | 0.5000 |
| BBB | Stalwarts | 0.650 | 70.0% | 0.1000 | 0.5000 |
| **Total** | | | | | 1.000000 |

Note: displayed weights are rounded for readability; the internal sum remains exactly 1.0.
### Derivation
| Ticker | Raw Weight | Performance | Consistency | Volatility |
| --- | ---: | ---: | ---: | ---: |
| AAA | 880.000000 | 1.100 | 80.0% | 0.1000 |
| BBB | 455.000000 | 0.650 | 70.0% | 0.1000 |
## Excluded Assets
- CCC: class=Gambles not in INCLUDE_CLASSES

## Confidence & SRE Summary
Inherited confidence: Medium (71).
FSS stability classifications use relative coefficient of variation (RCV; IQR/median) while SRE reports coefficient of variation (CoV), so labels may diverge.
## Notes
Weight cap further relaxed to 0.500 for feasibility.
## Configuration
```json
{
  "ASSET_WEIGHTS_OVERRIDE": {},
  "FOLD_DECAY_RATE": 0.139,
  "INCLUDE_CLASSES": [
    "Stars",
    "Stalwarts"
  ],
  "MAX_WEIGHT_CAP": 0.35,
  "MIN_ASSET_CONSISTENCY": 60.0,
  "MIN_CONFIDENCE_FOR_FINAL": 60,
  "MIN_WEIGHT_FLOOR": 0.02,
  "MULTIMODAL_MIN_CLUSTER_WEIGHT": 0.2,
  "MULTIMODAL_MIN_SEPARATION": 0.75,
  "PARAM_RCV_DDOF": 0,
  "PARAM_RCV_UNSTABLE": 0.5,
  "PARAM_RCV_WATCHLIST": 0.35,
  "PARAM_SENSITIVITY_THRESHOLD": 0.15,
  "PARAM_VALUE_DECIMALS": {
    "default": 3
  },
  "SHOW_PARAM_DISTS": true,
  "SHOW_RECENCY_HALFLIFE": true,
  "SHRINK_TO_EQUAL": 0.0,
  "USE_RECENCY_WEIGHTING": true,
  "WEIGHTING_SCHEME": "risk_adjusted",
  "WEIGHT_SENSITIVITY_RATIO_THRESHOLD": 0.25,
  "WEIGHT_SENSITIVITY_THRESHOLD": 0.05
}
```
