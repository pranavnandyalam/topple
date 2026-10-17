## waypoint-v1/standard: 400 seeds (0..399)

protocol_hash dd14af4aa4ef · complete · darwin\-arm64 arm64\-sme · onnxruntime 1\.31\.0

| # | Submission | Score [95% CI] | Success | Falls | Errors | Median energy (J) | Mean saturation |
|---|---|---|---|---|---|---|---|
| 1 | waypoint\_pid\_example | 80.24 [78.99, 81.35] | 0.932 | 27 | 0 | 10.62 | 0.013 |
| 2 | balance\_in\_place | 19.89 [19.80, 19.97] | 0.000 | 0 | 0 | 3.66 | 0.005 |
| 3 | spin | 17.45 [17.30, 17.60] | 0.000 | 0 | 0 | 16.38 | 0.009 |
| 4 | creeper | 60.65 [60.36, 60.93] | 0.000 | 2 | 0 | 9.33 | 0.006 |
| 5 | gates\_only | 68.86 [67.90, 69.73] | 0.005 | 35 | 0 | 10.97 | 0.013 |
| 6 | reckless | 71.26 [68.77, 73.66] | 0.738 | 105 | 0 | 9.81 | 0.089 |

Paired score differences (seed-paired bootstrap, 95% CI):

| A − B | Mean | 95% CI | Resamples |
|---|---|---|---|
| balance\_in\_place − waypoint\_pid\_example | -60.36 | [-61.48, -59.10] | 10000 (seed 0) |
| spin − waypoint\_pid\_example | -62.80 | [-63.92, -61.55] | 10000 (seed 0) |
| spin − balance\_in\_place | -2.44 | [-2.60, -2.27] | 10000 (seed 0) |
| creeper − waypoint\_pid\_example | -19.59 | [-20.74, -18.31] | 10000 (seed 0) |
| creeper − balance\_in\_place | +40.77 | [+40.48, +41.05] | 10000 (seed 0) |
| creeper − spin | +43.21 | [+42.89, +43.51] | 10000 (seed 0) |
| gates\_only − waypoint\_pid\_example | -11.38 | [-11.70, -11.03] | 10000 (seed 0) |
| gates\_only − balance\_in\_place | +48.98 | [+48.01, +49.85] | 10000 (seed 0) |
| gates\_only − spin | +51.41 | [+50.45, +52.30] | 10000 (seed 0) |
| gates\_only − creeper | +8.21 | [+7.22, +9.13] | 10000 (seed 0) |
| reckless − waypoint\_pid\_example | -8.99 | [-11.37, -6.65] | 10000 (seed 0) |
| reckless − balance\_in\_place | +51.37 | [+48.91, +53.76] | 10000 (seed 0) |
| reckless − spin | +53.81 | [+51.35, +56.20] | 10000 (seed 0) |
| reckless − creeper | +10.60 | [+8.14, +12.98] | 10000 (seed 0) |
| reckless − gates\_only | +2.40 | [+0.06, +4.70] | 10000 (seed 0) |
