## waypoint-v1/clean: 400 seeds (0..399)

protocol_hash 7c109cb5c525 · complete · darwin\-arm64 arm64\-sme · onnxruntime 1\.31\.0

| # | Submission | Score [95% CI] | Success | Falls | Errors | Median energy (J) | Mean saturation |
|---|---|---|---|---|---|---|---|
| 1 | waypoint\_pid\_example | 83.68 [83.54, 83.81] | 1.000 | 0 | 0 | 9.82 | 0.001 |
| 2 | ppo\_wp\_s0 | 87.13 [86.19, 87.94] | 0.983 | 7 | 0 | 26.81 | 0.148 |
| 3 | ppo\_wp\_s1 | 84.38 [82.82, 85.73] | 0.960 | 16 | 0 | 33.32 | 0.178 |
| 4 | ppo\_wp\_s2 | 85.62 [84.53, 86.55] | 0.978 | 9 | 0 | 35.07 | 0.156 |
| 5 | ppo\_wpft\_s0 | 88.72 [87.54, 89.73] | 0.955 | 12 | 0 | 10.96 | 0.045 |
| 6 | ppo\_wpft\_s1 | 87.69 [86.22, 88.98] | 0.965 | 14 | 0 | 9.62 | 0.040 |
| 7 | ppo\_wpft\_s2 | 85.80 [84.69, 86.76] | 0.970 | 12 | 0 | 13.54 | 0.049 |

Paired score differences (seed-paired bootstrap, 95% CI):

| A − B | Mean | 95% CI | Resamples |
|---|---|---|---|
| ppo\_wp\_s0 − waypoint\_pid\_example | +3.46 | [+2.51, +4.27] | 10000 (seed 0) |
| ppo\_wp\_s1 − waypoint\_pid\_example | +0.70 | [-0.84, +2.04] | 10000 (seed 0) |
| ppo\_wp\_s1 − ppo\_wp\_s0 | -2.76 | [-4.47, -1.14] | 10000 (seed 0) |
| ppo\_wp\_s2 − waypoint\_pid\_example | +1.94 | [+0.86, +2.87] | 10000 (seed 0) |
| ppo\_wp\_s2 − ppo\_wp\_s0 | -1.51 | [-2.62, -0.50] | 10000 (seed 0) |
| ppo\_wp\_s2 − ppo\_wp\_s1 | +1.24 | [-0.48, +3.04] | 10000 (seed 0) |
| ppo\_wpft\_s0 − waypoint\_pid\_example | +5.04 | [+3.87, +6.05] | 10000 (seed 0) |
| ppo\_wpft\_s0 − ppo\_wp\_s0 | +1.59 | [+0.21, +2.96] | 10000 (seed 0) |
| ppo\_wpft\_s0 − ppo\_wp\_s1 | +4.34 | [+2.90, +5.88] | 10000 (seed 0) |
| ppo\_wpft\_s0 − ppo\_wp\_s2 | +3.10 | [+1.64, +4.57] | 10000 (seed 0) |
| ppo\_wpft\_s1 − waypoint\_pid\_example | +4.02 | [+2.56, +5.29] | 10000 (seed 0) |
| ppo\_wpft\_s1 − ppo\_wp\_s0 | +0.56 | [-1.07, +2.10] | 10000 (seed 0) |
| ppo\_wpft\_s1 − ppo\_wp\_s1 | +3.32 | [+1.82, +4.89] | 10000 (seed 0) |
| ppo\_wpft\_s1 − ppo\_wp\_s2 | +2.07 | [+0.33, +3.74] | 10000 (seed 0) |
| ppo\_wpft\_s1 − ppo\_wpft\_s0 | -1.03 | [-2.57, +0.46] | 10000 (seed 0) |
| ppo\_wpft\_s2 − waypoint\_pid\_example | +2.12 | [+1.03, +3.08] | 10000 (seed 0) |
| ppo\_wpft\_s2 − ppo\_wp\_s0 | -1.34 | [-2.65, +0.00] | 10000 (seed 0) |
| ppo\_wpft\_s2 − ppo\_wp\_s1 | +1.42 | [-0.24, +3.16] | 10000 (seed 0) |
| ppo\_wpft\_s2 − ppo\_wp\_s2 | +0.18 | [-1.23, +1.60] | 10000 (seed 0) |
| ppo\_wpft\_s2 − ppo\_wpft\_s0 | -2.92 | [-4.22, -1.61] | 10000 (seed 0) |
| ppo\_wpft\_s2 − ppo\_wpft\_s1 | -1.90 | [-3.29, -0.40] | 10000 (seed 0) |
