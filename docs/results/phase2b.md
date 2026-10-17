# Phase 2b results: `waypoint-v1`

Shipped `ppo_wpft_s1` (finetune) as `submissions/ppo_waypoint`. E_ref 21.4 J.

![driver and ppo_waypoint on one course](phase2/waypoint.gif)

## Headline: steps to the first eligible checkpoint

Env steps until a selection eval first passed the `waypoint` gate (seeds 0, 1, 2). The fine-tune arm also pays for its balance pretraining: 5.01M, 5.01M, 5.01M steps.

| Arm | First eligible | Median | Median incl. pretraining |
|---|---|---|---|
| From scratch | 2.51M, 3.01M, 4.01M | 3.01M | — |
| Fine-tune | 2.51M, 3.01M, 5.01M | 3.01M | 8.03M |

## Fresh evaluation: `waypoint-v1/standard` seeds 0..399

Item 6 (docked ≥ 60%, paired PPO − driver lower bound > −5, above the creeper, fell ≤ driver + 20 pp): **5 of 6 runs pass.**

| Run | Arm | Score | Docked | Fell (by cause) | PPO − driver [95% CI] | Item 6 |
|---|---|---|---|---|---|---|
| `ppo_wp_s0` | scratch | 81.95 | 88.5% | 17.8% (push 64, self 7) | +1.70 [-0.20, +3.55] | pass |
| `ppo_wp_s1` | scratch | 77.34 | 82.2% | 24.5% (push 83, self 15) | -2.90 [-5.11, -0.80] | FAIL |
| `ppo_wp_s2` | scratch | 79.22 | 85.2% | 22.8% (push 82, self 9) | -1.02 [-3.03, +0.92] | pass |
| `ppo_wpft_s0` | finetune | 80.50 | 80.5% | 18.2% (push 61, self 12) | +0.25 [-1.99, +2.45] | pass |
| `ppo_wpft_s1` | finetune | 79.23 | 81.0% | 19.0% (push 62, self 14) | -1.02 [-3.35, +1.21] | pass |
| `ppo_wpft_s2` | finetune | 77.66 | 82.5% | 17.2% (push 57, self 12) | -2.58 [-4.71, -0.51] | pass |
| `waypoint_pid_example` | baseline | 80.24 | 93.2% | 6.8% (push 27) |  |  |
| `creeper` | baseline | 60.65 | 0.0% | 0.5% (push 2) |  |  |

- From scratch: score 79.50 ± 2.32, fell 21.7% ± 3.5%; median energy 2.72× the driver's.
- Fine-tune: score 79.13 ± 1.42, fell 18.2% ± 0.9%; median energy 1.03× the driver's.
- Better arm (higher mean fresh score): scratch.

## Anchors and PPO: score parts on standard 0..399

| Entry | Score | Course /55 | Speed /25 | Alive /15 | Battery /5 | Docked | Fell | Median t_finish (s) |
|---|---|---|---|---|---|---|---|---|
| `waypoint_pid_example` | 80.24 | 53.41 | 10.39 | 14.32 | 2.13 | 93.2% | 6.8% | 13.34 |
| `ppo_wpft_s1 (shipped)` | 79.23 | 50.32 | 14.17 | 12.71 | 2.03 | 81.0% | 19.0% | 9.77 |
| `creeper` | 60.65 | 42.97 | 0.00 | 14.97 | 2.71 | 0.0% | 0.5% | 20.00 |
| `gates_only` | 68.86 | 52.56 | 0.06 | 14.21 | 2.04 | 0.5% | 8.8% | 20.00 |
| `reckless` | 71.26 | 47.81 | 9.46 | 12.05 | 1.93 | 73.8% | 26.2% | 12.50 |
| `balance_in_place` | 19.89 | 0.82 | 0.00 | 15.00 | 4.06 | 0.0% | 0.0% | 20.00 |
| `spin` | 17.45 | 1.27 | 0.00 | 15.00 | 1.18 | 0.0% | 0.0% | 20.00 |

Exchange rate: a fall costs about 55.9 points (the shipped policy's mean score of docked survivors minus fallers), so each extra 1% of falls costs about 0.56; finishing 1 s faster on a docked run earns 25/(20 − 5) = 1.67.

## Report only: `waypoint-v1/clean` seeds 0..399

No pushes: every fall here is self-caused.

| Entry | Score | Docked | Fell |
|---|---|---|---|
| `ppo_wp_s0` | 87.13 | 98.2% | 1.8% (self 7) |
| `ppo_wp_s1` | 84.38 | 96.0% | 4.0% (self 16) |
| `ppo_wp_s2` | 85.62 | 97.8% | 2.2% (self 9) |
| `ppo_wpft_s0` | 88.72 | 95.5% | 3.0% (self 12) |
| `ppo_wpft_s1` | 87.69 | 96.5% | 3.5% (self 14) |
| `ppo_wpft_s2` | 85.80 | 97.0% | 3.0% (self 12) |
| `waypoint_pid_example` | 83.68 | 100.0% | 0.0% |

## `wp_eval_v1` (50 public seeds)

The leaderboard protocol.

| Entry | Score | Docked | Fell |
|---|---|---|---|
| `ppo_wp_s0` | 79.94 | 88.0% | 20.0% (push 9, self 1) |
| `ppo_wp_s1` | 81.15 | 90.0% | 16.0% (push 6, self 2) |
| `ppo_wp_s2` | 79.42 | 86.0% | 24.0% (push 11, self 1) |
| `ppo_wpft_s0` | 89.75 | 98.0% | 2.0% (push 1) |
| `ppo_wpft_s1` | 87.75 | 94.0% | 6.0% (push 3) |
| `ppo_wpft_s2` | 84.58 | 94.0% | 6.0% (push 3) |
| `waypoint_pid_example` | 78.30 | 88.0% | 12.0% (push 6) |
| `creeper` | 60.90 | 0.0% | 0.0% |

## Shipping (§8.8)

1. `ppo_wpft_s1`: ships

`submissions/ppo_waypoint` = `ppo_wpft_s1` (finetune), step 5,505,024, onnx sha256 `f3a05d71b891969a2e7e6f532c79a2400fcdeddce3777d9ded612c7c2cf0cb70`.

## Fine-tune sanity (item 7)

Each fine-tune source after `extend_obs`, on `balance-v1/clean` seeds 0..19 with 5 zero features appended and +100 rad on wheel_pos (features 4, 5); full-10 s survival:

- `ppo_wpft_s0` from `runs/ppo_s0` step 3,506,176: 100.0%
- `ppo_wpft_s1` from `runs/ppo_s1` step 1,753,088: 100.0%
- `ppo_wpft_s2` from `runs/ppo_s2` step 5,013,504: 100.0%

## Pilot and budget

Pilot `ppo_waypoint.yaml` seed 0 at `52b28bd`: 6,012,928 steps, cpu 2376 s, so waypoint `max_cpu_s` = 7129. Budget rule: late − mid +0.66 [-0.78, +2.03], so total_timesteps stays 6M.

## Selection evals (seeds 2000..2199)

### `ppo_wp_s0` (scratch, best 6.01M)

| Step | Score | Gates | Docked | Fell | Median t_finish | Eligible |
|---|---|---|---|---|---|---|
| 0.51M | 12.39 | 0.03 | 0.0% | 67.0% | 20.00 | no |
| 1.02M | 40.83 | 1.56 | 0.0% | 8.0% | 20.00 | no |
| 1.51M | 67.73 | 4.14 | 37.5% | 4.0% | 20.00 | no |
| 2.02M | 77.17 | 4.77 | 88.0% | 8.5% | 14.34 | no |
| 2.51M | 78.80 | 4.74 | 89.0% | 13.5% | 12.28 | yes |
| 3.01M | 76.37 | 4.54 | 83.5% | 23.0% | 11.60 | no |
| 3.51M | 80.16 | 4.66 | 88.0% | 17.0% | 10.59 | yes |
| 4.01M | 81.24 | 4.72 | 88.5% | 15.5% | 10.34 | yes |
| 4.51M | 80.96 | 4.66 | 88.5% | 18.0% | 9.86 | yes |
| 5.01M | 80.67 | 4.64 | 88.0% | 14.0% | 10.00 | yes |
| 5.51M | 81.34 | 4.70 | 88.5% | 20.0% | 9.69 | yes |
| 6.01M | 81.37 | 4.68 | 88.5% | 19.0% | 9.67 | yes |

### `ppo_wp_s1` (scratch, best 5.01M)

| Step | Score | Gates | Docked | Fell | Median t_finish | Eligible |
|---|---|---|---|---|---|---|
| 0.51M | 7.05 | 0.02 | 0.0% | 93.5% | 20.00 | no |
| 1.02M | 25.07 | 0.31 | 0.0% | 2.5% | 20.00 | no |
| 1.51M | 47.48 | 2.29 | 0.5% | 2.0% | 20.00 | no |
| 2.02M | 70.66 | 4.36 | 46.5% | 5.5% | 20.00 | no |
| 2.51M | 76.70 | 4.71 | 82.0% | 8.5% | 13.80 | no |
| 3.01M | 78.63 | 4.75 | 86.5% | 9.5% | 12.33 | yes |
| 3.51M | 78.05 | 4.65 | 85.0% | 18.5% | 11.12 | no |
| 4.01M | 79.22 | 4.66 | 85.5% | 19.0% | 10.47 | yes |
| 4.51M | 76.23 | 4.52 | 81.5% | 25.5% | 10.76 | no |
| 5.01M | 79.75 | 4.67 | 86.5% | 21.5% | 10.09 | yes |
| 5.51M | 77.86 | 4.53 | 82.5% | 26.0% | 9.80 | no |
| 6.01M | 78.03 | 4.54 | 82.0% | 25.5% | 9.40 | no |

### `ppo_wp_s2` (scratch, best 6.01M)

| Step | Score | Gates | Docked | Fell | Median t_finish | Eligible |
|---|---|---|---|---|---|---|
| 0.51M | 4.99 | 0.01 | 0.0% | 100.0% | 20.00 | no |
| 1.02M | 31.79 | 0.67 | 0.0% | 0.5% | 20.00 | no |
| 1.51M | 69.59 | 4.33 | 51.0% | 4.5% | 19.87 | no |
| 2.02M | 71.97 | 4.47 | 61.5% | 10.0% | 18.23 | no |
| 2.51M | 75.41 | 4.62 | 84.0% | 22.5% | 13.23 | no |
| 3.01M | 77.02 | 4.62 | 85.0% | 24.5% | 11.45 | no |
| 3.51M | 77.85 | 4.62 | 85.5% | 24.0% | 10.83 | no |
| 4.01M | 79.92 | 4.68 | 87.5% | 22.0% | 9.94 | yes |
| 4.51M | 79.37 | 4.68 | 86.5% | 23.5% | 10.41 | yes |
| 5.01M | 79.57 | 4.66 | 87.0% | 24.5% | 10.01 | yes |
| 5.51M | 77.58 | 4.58 | 83.5% | 26.5% | 10.43 | no |
| 6.01M | 80.17 | 4.69 | 88.0% | 22.0% | 10.02 | yes |

### `ppo_wpft_s0` (finetune, best 6.01M)

| Step | Score | Gates | Docked | Fell | Median t_finish | Eligible |
|---|---|---|---|---|---|---|
| 0.51M | 54.47 | 2.77 | 0.0% | 0.5% | 20.00 | no |
| 1.02M | 69.36 | 4.20 | 35.0% | 4.5% | 20.00 | no |
| 1.51M | 77.75 | 4.74 | 84.0% | 7.5% | 14.09 | no |
| 2.02M | 77.09 | 4.57 | 73.0% | 12.5% | 12.41 | no |
| 2.51M | 82.02 | 4.74 | 88.5% | 10.0% | 10.52 | yes |
| 3.01M | 83.28 | 4.75 | 88.5% | 11.5% | 9.70 | yes |
| 3.51M | 80.46 | 4.62 | 81.0% | 12.0% | 10.41 | yes |
| 4.01M | 80.83 | 4.61 | 80.5% | 15.0% | 9.79 | yes |
| 4.51M | 82.79 | 4.67 | 85.5% | 14.0% | 9.48 | yes |
| 5.01M | 82.63 | 4.67 | 84.5% | 13.0% | 9.41 | yes |
| 5.51M | 78.78 | 4.51 | 79.0% | 18.5% | 10.16 | no |
| 6.01M | 83.45 | 4.70 | 85.0% | 12.5% | 9.25 | yes |

### `ppo_wpft_s1` (finetune, best 5.51M)

| Step | Score | Gates | Docked | Fell | Median t_finish | Eligible |
|---|---|---|---|---|---|---|
| 0.51M | 39.46 | 1.34 | 0.0% | 1.0% | 20.00 | no |
| 1.02M | 67.86 | 4.00 | 20.5% | 1.5% | 20.00 | no |
| 1.51M | 73.06 | 4.43 | 57.5% | 5.5% | 18.31 | no |
| 2.02M | 77.32 | 4.67 | 83.5% | 9.5% | 14.12 | no |
| 2.51M | 77.39 | 4.63 | 80.5% | 12.5% | 13.33 | no |
| 3.01M | 79.80 | 4.63 | 85.0% | 14.5% | 11.29 | yes |
| 3.51M | 81.17 | 4.67 | 85.5% | 14.0% | 10.48 | yes |
| 4.01M | 83.10 | 4.73 | 88.5% | 11.5% | 9.98 | yes |
| 4.51M | 82.75 | 4.72 | 88.5% | 11.5% | 10.48 | yes |
| 5.01M | 82.99 | 4.71 | 87.0% | 13.5% | 9.79 | yes |
| 5.51M | 83.71 | 4.72 | 88.5% | 12.0% | 9.56 | yes |
| 6.01M | 82.31 | 4.70 | 85.0% | 14.5% | 9.89 | yes |

### `ppo_wpft_s2` (finetune, best 6.01M)

| Step | Score | Gates | Docked | Fell | Median t_finish | Eligible |
|---|---|---|---|---|---|---|
| 0.51M | 47.02 | 2.03 | 0.0% | 1.0% | 20.00 | no |
| 1.02M | 63.92 | 3.76 | 11.5% | 9.5% | 20.00 | no |
| 1.51M | 75.18 | 4.62 | 78.5% | 10.0% | 14.69 | no |
| 2.02M | 75.65 | 4.58 | 77.0% | 14.5% | 13.05 | no |
| 2.51M | 74.64 | 4.58 | 77.0% | 15.0% | 13.73 | no |
| 3.01M | 77.08 | 4.62 | 84.5% | 15.5% | 12.30 | no |
| 3.51M | 78.27 | 4.71 | 86.5% | 13.5% | 12.37 | no |
| 4.01M | 76.98 | 4.67 | 85.0% | 13.5% | 12.78 | no |
| 4.51M | 78.12 | 4.68 | 85.5% | 13.0% | 12.31 | no |
| 5.01M | 79.60 | 4.70 | 86.5% | 13.5% | 11.18 | yes |
| 5.51M | 79.18 | 4.67 | 85.0% | 14.0% | 11.32 | yes |
| 6.01M | 81.03 | 4.70 | 88.5% | 11.5% | 10.93 | yes |
