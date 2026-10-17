## waypoint-v1/standard: 50 seeds (3000..3049)

protocol_hash 8d2229df811a · complete · darwin\-arm64 arm64\-sme · onnxruntime 1\.31\.0

| # | Submission | Score [95% CI] | Success | Falls | Errors | Median energy (J) | Mean saturation |
|---|---|---|---|---|---|---|---|
| 1 | waypoint\_pid\_example | 78.30 [73.90, 82.01] | 0.880 | 6 | 0 | 9.55 | 0.015 |
| 2 | creeper | 60.90 [60.26, 61.54] | 0.000 | 0 | 0 | 8.87 | 0.005 |
| 3 | ppo\_wp\_s0 | 79.94 [73.83, 85.07] | 0.800 | 10 | 0 | 22.61 | 0.138 |
| 4 | ppo\_wp\_s1 | 81.15 [75.77, 85.42] | 0.840 | 8 | 0 | 30.82 | 0.187 |
| 5 | ppo\_wp\_s2 | 79.42 [73.85, 84.10] | 0.760 | 12 | 0 | 26.41 | 0.168 |
| 6 | ppo\_wpft\_s0 | 89.75 [88.03, 90.89] | 0.980 | 1 | 0 | 11.14 | 0.039 |
| 7 | ppo\_wpft\_s1 | 87.75 [84.67, 90.09] | 0.940 | 3 | 0 | 8.95 | 0.027 |
| 8 | ppo\_wpft\_s2 | 84.58 [81.45, 86.96] | 0.940 | 3 | 0 | 14.01 | 0.073 |

Paired score differences (seed-paired bootstrap, 95% CI):

| A − B | Mean | 95% CI | Resamples |
|---|---|---|---|
| creeper − waypoint\_pid\_example | -17.40 | [-21.11, -13.00] | 10000 (seed 0) |
| ppo\_wp\_s0 − waypoint\_pid\_example | +1.64 | [-5.21, +7.80] | 10000 (seed 0) |
| ppo\_wp\_s0 − creeper | +19.04 | [+12.93, +24.20] | 10000 (seed 0) |
| ppo\_wp\_s1 − waypoint\_pid\_example | +2.84 | [-3.80, +8.98] | 10000 (seed 0) |
| ppo\_wp\_s1 − creeper | +20.24 | [+14.71, +24.64] | 10000 (seed 0) |
| ppo\_wp\_s1 − ppo\_wp\_s0 | +1.20 | [-5.31, +8.12] | 10000 (seed 0) |
| ppo\_wp\_s2 − waypoint\_pid\_example | +1.12 | [-4.18, +5.64] | 10000 (seed 0) |
| ppo\_wp\_s2 − creeper | +18.52 | [+12.91, +23.27] | 10000 (seed 0) |
| ppo\_wp\_s2 − ppo\_wp\_s0 | -0.52 | [-3.76, +3.41] | 10000 (seed 0) |
| ppo\_wp\_s2 − ppo\_wp\_s1 | -1.73 | [-8.11, +4.36] | 10000 (seed 0) |
| ppo\_wpft\_s0 − waypoint\_pid\_example | +11.44 | [+7.38, +15.88] | 10000 (seed 0) |
| ppo\_wpft\_s0 − creeper | +28.85 | [+26.94, +30.26] | 10000 (seed 0) |
| ppo\_wpft\_s0 − ppo\_wp\_s0 | +9.80 | [+4.45, +15.96] | 10000 (seed 0) |
| ppo\_wpft\_s0 − ppo\_wp\_s1 | +8.60 | [+3.87, +14.26] | 10000 (seed 0) |
| ppo\_wpft\_s0 − ppo\_wp\_s2 | +10.33 | [+5.43, +15.84] | 10000 (seed 0) |
| ppo\_wpft\_s1 − waypoint\_pid\_example | +9.45 | [+4.85, +14.25] | 10000 (seed 0) |
| ppo\_wpft\_s1 − creeper | +26.85 | [+23.76, +29.26] | 10000 (seed 0) |
| ppo\_wpft\_s1 − ppo\_wp\_s0 | +7.81 | [+1.68, +14.55] | 10000 (seed 0) |
| ppo\_wpft\_s1 − ppo\_wp\_s1 | +6.61 | [+1.32, +12.66] | 10000 (seed 0) |
| ppo\_wpft\_s1 − ppo\_wp\_s2 | +8.33 | [+2.73, +14.48] | 10000 (seed 0) |
| ppo\_wpft\_s1 − ppo\_wpft\_s0 | -2.00 | [-4.50, -0.33] | 10000 (seed 0) |
| ppo\_wpft\_s2 − waypoint\_pid\_example | +6.28 | [+1.73, +11.07] | 10000 (seed 0) |
| ppo\_wpft\_s2 − creeper | +23.68 | [+20.52, +26.15] | 10000 (seed 0) |
| ppo\_wpft\_s2 − ppo\_wp\_s0 | +4.64 | [-1.52, +11.46] | 10000 (seed 0) |
| ppo\_wpft\_s2 − ppo\_wp\_s1 | +3.44 | [-1.94, +9.60] | 10000 (seed 0) |
| ppo\_wpft\_s2 − ppo\_wp\_s2 | +5.16 | [-0.54, +11.35] | 10000 (seed 0) |
| ppo\_wpft\_s2 − ppo\_wpft\_s0 | -5.17 | [-7.63, -3.43] | 10000 (seed 0) |
| ppo\_wpft\_s2 − ppo\_wpft\_s1 | -3.17 | [-3.67, -2.71] | 10000 (seed 0) |
