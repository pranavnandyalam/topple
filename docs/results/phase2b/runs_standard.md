## waypoint-v1/standard: 400 seeds (0..399)

protocol_hash dd14af4aa4ef · complete · darwin\-arm64 arm64\-sme · onnxruntime 1\.31\.0

| # | Submission | Score [95% CI] | Success | Falls | Errors | Median energy (J) | Mean saturation |
|---|---|---|---|---|---|---|---|
| 1 | waypoint\_pid\_example | 80.24 [78.99, 81.35] | 0.932 | 27 | 0 | 10.62 | 0.013 |
| 2 | creeper | 60.65 [60.36, 60.93] | 0.000 | 2 | 0 | 9.33 | 0.006 |
| 3 | balance\_in\_place | 19.89 [19.80, 19.97] | 0.000 | 0 | 0 | 3.66 | 0.005 |
| 4 | spin | 17.45 [17.30, 17.60] | 0.000 | 0 | 0 | 16.38 | 0.009 |
| 5 | gates\_only | 68.86 [67.90, 69.73] | 0.005 | 35 | 0 | 10.97 | 0.013 |
| 6 | reckless | 71.26 [68.77, 73.66] | 0.738 | 105 | 0 | 9.81 | 0.089 |
| 7 | ppo\_wp\_s0 | 81.95 [80.27, 83.56] | 0.823 | 71 | 0 | 24.53 | 0.144 |
| 8 | ppo\_wp\_s1 | 77.34 [75.16, 79.37] | 0.755 | 98 | 0 | 28.91 | 0.194 |
| 9 | ppo\_wp\_s2 | 79.22 [77.41, 80.93] | 0.772 | 91 | 0 | 29.76 | 0.178 |
| 10 | ppo\_wpft\_s0 | 80.50 [78.36, 82.52] | 0.805 | 73 | 0 | 10.89 | 0.073 |
| 11 | ppo\_wpft\_s1 | 79.23 [76.91, 81.37] | 0.810 | 76 | 0 | 9.48 | 0.063 |
| 12 | ppo\_wpft\_s2 | 77.66 [75.57, 79.57] | 0.825 | 69 | 0 | 13.87 | 0.088 |

Paired score differences (seed-paired bootstrap, 95% CI):

| A − B | Mean | 95% CI | Resamples |
|---|---|---|---|
| creeper − waypoint\_pid\_example | -19.59 | [-20.74, -18.31] | 10000 (seed 0) |
| balance\_in\_place − waypoint\_pid\_example | -60.36 | [-61.48, -59.10] | 10000 (seed 0) |
| balance\_in\_place − creeper | -40.77 | [-41.05, -40.48] | 10000 (seed 0) |
| spin − waypoint\_pid\_example | -62.80 | [-63.92, -61.55] | 10000 (seed 0) |
| spin − creeper | -43.21 | [-43.51, -42.89] | 10000 (seed 0) |
| spin − balance\_in\_place | -2.44 | [-2.60, -2.27] | 10000 (seed 0) |
| gates\_only − waypoint\_pid\_example | -11.38 | [-11.70, -11.03] | 10000 (seed 0) |
| gates\_only − creeper | +8.21 | [+7.22, +9.13] | 10000 (seed 0) |
| gates\_only − balance\_in\_place | +48.98 | [+48.01, +49.85] | 10000 (seed 0) |
| gates\_only − spin | +51.41 | [+50.45, +52.30] | 10000 (seed 0) |
| reckless − waypoint\_pid\_example | -8.99 | [-11.37, -6.65] | 10000 (seed 0) |
| reckless − creeper | +10.60 | [+8.14, +12.98] | 10000 (seed 0) |
| reckless − balance\_in\_place | +51.37 | [+48.91, +53.76] | 10000 (seed 0) |
| reckless − spin | +53.81 | [+51.35, +56.20] | 10000 (seed 0) |
| reckless − gates\_only | +2.40 | [+0.06, +4.70] | 10000 (seed 0) |
| ppo\_wp\_s0 − waypoint\_pid\_example | +1.70 | [-0.20, +3.55] | 10000 (seed 0) |
| ppo\_wp\_s0 − creeper | +21.29 | [+19.57, +22.92] | 10000 (seed 0) |
| ppo\_wp\_s0 − balance\_in\_place | +62.06 | [+60.38, +63.65] | 10000 (seed 0) |
| ppo\_wp\_s0 − spin | +64.50 | [+62.81, +66.12] | 10000 (seed 0) |
| ppo\_wp\_s0 − gates\_only | +13.08 | [+11.27, +14.79] | 10000 (seed 0) |
| ppo\_wp\_s0 − reckless | +10.69 | [+8.06, +13.34] | 10000 (seed 0) |
| ppo\_wp\_s1 − waypoint\_pid\_example | -2.90 | [-5.11, -0.80] | 10000 (seed 0) |
| ppo\_wp\_s1 − creeper | +16.69 | [+14.50, +18.71] | 10000 (seed 0) |
| ppo\_wp\_s1 − balance\_in\_place | +57.45 | [+55.28, +59.50] | 10000 (seed 0) |
| ppo\_wp\_s1 − spin | +59.89 | [+57.71, +61.94] | 10000 (seed 0) |
| ppo\_wp\_s1 − gates\_only | +8.48 | [+6.38, +10.47] | 10000 (seed 0) |
| ppo\_wp\_s1 − reckless | +6.08 | [+3.32, +8.89] | 10000 (seed 0) |
| ppo\_wp\_s1 − ppo\_wp\_s0 | -4.61 | [-7.34, -1.85] | 10000 (seed 0) |
| ppo\_wp\_s2 − waypoint\_pid\_example | -1.02 | [-3.03, +0.92] | 10000 (seed 0) |
| ppo\_wp\_s2 − creeper | +18.57 | [+16.72, +20.35] | 10000 (seed 0) |
| ppo\_wp\_s2 − balance\_in\_place | +59.34 | [+57.52, +61.05] | 10000 (seed 0) |
| ppo\_wp\_s2 − spin | +61.77 | [+59.96, +63.49] | 10000 (seed 0) |
| ppo\_wp\_s2 − gates\_only | +10.36 | [+8.44, +12.16] | 10000 (seed 0) |
| ppo\_wp\_s2 − reckless | +7.96 | [+5.24, +10.59] | 10000 (seed 0) |
| ppo\_wp\_s2 − ppo\_wp\_s0 | -2.73 | [-4.21, -1.28] | 10000 (seed 0) |
| ppo\_wp\_s2 − ppo\_wp\_s1 | +1.88 | [-0.97, +4.71] | 10000 (seed 0) |
| ppo\_wpft\_s0 − waypoint\_pid\_example | +0.25 | [-1.99, +2.45] | 10000 (seed 0) |
| ppo\_wpft\_s0 − creeper | +19.84 | [+17.71, +21.88] | 10000 (seed 0) |
| ppo\_wpft\_s0 − balance\_in\_place | +60.61 | [+58.49, +62.62] | 10000 (seed 0) |
| ppo\_wpft\_s0 − spin | +63.05 | [+60.90, +65.06] | 10000 (seed 0) |
| ppo\_wpft\_s0 − gates\_only | +11.63 | [+9.46, +13.72] | 10000 (seed 0) |
| ppo\_wpft\_s0 − reckless | +9.24 | [+6.42, +12.07] | 10000 (seed 0) |
| ppo\_wpft\_s0 − ppo\_wp\_s0 | -1.45 | [-4.28, +1.31] | 10000 (seed 0) |
| ppo\_wpft\_s0 − ppo\_wp\_s1 | +3.16 | [+1.37, +4.95] | 10000 (seed 0) |
| ppo\_wpft\_s0 − ppo\_wp\_s2 | +1.28 | [-1.47, +3.99] | 10000 (seed 0) |
| ppo\_wpft\_s1 − waypoint\_pid\_example | -1.02 | [-3.35, +1.21] | 10000 (seed 0) |
| ppo\_wpft\_s1 − creeper | +18.57 | [+16.26, +20.73] | 10000 (seed 0) |
| ppo\_wpft\_s1 − balance\_in\_place | +59.34 | [+57.01, +61.50] | 10000 (seed 0) |
| ppo\_wpft\_s1 − spin | +61.78 | [+59.47, +63.92] | 10000 (seed 0) |
| ppo\_wpft\_s1 − gates\_only | +10.37 | [+8.09, +12.48] | 10000 (seed 0) |
| ppo\_wpft\_s1 − reckless | +7.97 | [+5.21, +10.76] | 10000 (seed 0) |
| ppo\_wpft\_s1 − ppo\_wp\_s0 | -2.72 | [-5.61, +0.10] | 10000 (seed 0) |
| ppo\_wpft\_s1 − ppo\_wp\_s1 | +1.89 | [+0.12, +3.75] | 10000 (seed 0) |
| ppo\_wpft\_s1 − ppo\_wp\_s2 | +0.01 | [-2.89, +2.82] | 10000 (seed 0) |
| ppo\_wpft\_s1 − ppo\_wpft\_s0 | -1.27 | [-2.97, +0.42] | 10000 (seed 0) |
| ppo\_wpft\_s2 − waypoint\_pid\_example | -2.58 | [-4.71, -0.51] | 10000 (seed 0) |
| ppo\_wpft\_s2 − creeper | +17.01 | [+14.91, +18.93] | 10000 (seed 0) |
| ppo\_wpft\_s2 − balance\_in\_place | +57.77 | [+55.68, +59.70] | 10000 (seed 0) |
| ppo\_wpft\_s2 − spin | +60.21 | [+58.14, +62.14] | 10000 (seed 0) |
| ppo\_wpft\_s2 − gates\_only | +8.80 | [+6.74, +10.74] | 10000 (seed 0) |
| ppo\_wpft\_s2 − reckless | +6.40 | [+3.64, +9.19] | 10000 (seed 0) |
| ppo\_wpft\_s2 − ppo\_wp\_s0 | -4.29 | [-6.98, -1.65] | 10000 (seed 0) |
| ppo\_wpft\_s2 − ppo\_wp\_s1 | +0.32 | [-1.48, +2.17] | 10000 (seed 0) |
| ppo\_wpft\_s2 − ppo\_wp\_s2 | -1.56 | [-4.29, +1.07] | 10000 (seed 0) |
| ppo\_wpft\_s2 − ppo\_wpft\_s0 | -2.84 | [-4.43, -1.24] | 10000 (seed 0) |
| ppo\_wpft\_s2 − ppo\_wpft\_s1 | -1.57 | [-3.20, +0.13] | 10000 (seed 0) |
