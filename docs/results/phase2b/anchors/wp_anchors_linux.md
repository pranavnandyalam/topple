## waypoint-v1/standard: 400 seeds (0..399)

protocol_hash dd14af4aa4ef · complete · linux\-x86\_64 avx2 · onnxruntime 1\.31\.0

| # | Submission | Score [95% CI] | Success | Falls | Errors | Median energy (J) | Mean saturation |
|---|---|---|---|---|---|---|---|
| 1 | waypoint\_pid\_example | 80.38 [79.18, 81.47] | 0.935 | 26 | 0 | 10.72 | 0.012 |
| 2 | balance\_in\_place | 19.88 [19.79, 19.96] | 0.000 | 0 | 0 | 3.67 | 0.005 |
| 3 | spin | 17.43 [17.28, 17.58] | 0.000 | 0 | 0 | 16.41 | 0.009 |
| 4 | creeper | 60.66 [60.37, 60.94] | 0.000 | 2 | 0 | 9.33 | 0.006 |
| 5 | gates\_only | 68.97 [68.05, 69.81] | 0.005 | 34 | 0 | 11.06 | 0.013 |
| 6 | reckless | 71.01 [68.51, 73.45] | 0.733 | 107 | 0 | 9.78 | 0.090 |

Paired score differences (seed-paired bootstrap, 95% CI):

| A − B | Mean | 95% CI | Resamples |
|---|---|---|---|
| balance\_in\_place − waypoint\_pid\_example | -60.51 | [-61.59, -59.29] | 10000 (seed 0) |
| spin − waypoint\_pid\_example | -62.95 | [-64.04, -61.74] | 10000 (seed 0) |
| spin − balance\_in\_place | -2.45 | [-2.61, -2.29] | 10000 (seed 0) |
| creeper − waypoint\_pid\_example | -19.73 | [-20.85, -18.50] | 10000 (seed 0) |
| creeper − balance\_in\_place | +40.78 | [+40.49, +41.06] | 10000 (seed 0) |
| creeper − spin | +43.23 | [+42.92, +43.54] | 10000 (seed 0) |
| gates\_only − waypoint\_pid\_example | -11.41 | [-11.73, -11.07] | 10000 (seed 0) |
| gates\_only − balance\_in\_place | +49.10 | [+48.17, +49.94] | 10000 (seed 0) |
| gates\_only − spin | +51.54 | [+50.61, +52.40] | 10000 (seed 0) |
| gates\_only − creeper | +8.31 | [+7.36, +9.21] | 10000 (seed 0) |
| reckless − waypoint\_pid\_example | -9.37 | [-11.72, -7.09] | 10000 (seed 0) |
| reckless − balance\_in\_place | +51.13 | [+48.63, +53.56] | 10000 (seed 0) |
| reckless − spin | +53.58 | [+51.08, +56.01] | 10000 (seed 0) |
| reckless − creeper | +10.35 | [+7.87, +12.78] | 10000 (seed 0) |
| reckless − gates\_only | +2.04 | [-0.29, +4.29] | 10000 (seed 0) |
