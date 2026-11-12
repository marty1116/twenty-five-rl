# Twenty-Five evaluation leaderboard

> Held-out duplicate-deal evaluation against the fixed `baseline` suite: random, greedy, conservative, counting.
> The 95% interval bootstraps complete deal clusters; candidate matchups below do not affect the leaderboard.

| Policy | Games | Wins | Win rate (95% CI) | Avg score | Avg diff | Tricks/round | 2P BT rating | Latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 500 | 161 | 0.322 (0.285-0.363) | 14.43 | -6.50 | 1.28 | 1574 | 0.018 |
| conservative | 500 | 141 | 0.282 (0.246-0.318) | 13.79 | -8.19 | 1.10 | 1518 | 0.028 |
| ppo_best_250 | 500 | 125 | 0.250 (0.214-0.287) | 13.18 | -8.95 | 1.07 | 1524 | 0.293 |
| ppo_final_250 | 500 | 119 | 0.238 (0.215-0.262) | 12.72 | -9.29 | 1.07 | 1534 | 0.311 |
| random | 500 | 114 | 0.228 (0.195-0.260) | 12.30 | -10.34 | 0.99 | 1479 | 0.001 |
| counting | 500 | 109 | 0.218 (0.186-0.246) | 11.98 | -10.50 | 1.01 | 1521 | 0.061 |
| lowest | 500 | 63 | 0.126 (0.090-0.164) | 9.76 | -14.01 | 0.74 | 1350 | 0.016 |

## 4-player focal policy versus homogeneous opponent table

| Policy | random | lowest | greedy | conservative | counting | ppo_final_250 | ppo_best_250 |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | - | 0.410 | 0.090 | 0.130 | 0.210 | 0.170 | 0.130 |
| lowest | 0.130 | - | 0.090 | 0.130 | 0.130 | 0.130 | 0.050 |
| greedy | 0.370 | 0.650 | - | 0.250 | 0.410 | 0.250 | 0.330 |
| conservative | 0.370 | 0.410 | 0.170 | - | 0.330 | 0.250 | 0.330 |
| counting | 0.170 | 0.370 | 0.090 | 0.250 | - | 0.210 | 0.210 |
| ppo_final_250 | 0.330 | 0.450 | 0.170 | 0.210 | 0.250 | - | 0.250 |
| ppo_best_250 | 0.330 | 0.530 | 0.170 | 0.170 | 0.370 | 0.210 | - |

## Two-player head-to-head matrix

| Policy | random | lowest | greedy | conservative | counting | ppo_final_250 | ppo_best_250 |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | - | 0.676 | 0.372 | 0.436 | 0.452 | 0.412 | 0.432 |
| lowest | 0.324 | - | 0.188 | 0.292 | 0.288 | 0.248 | 0.272 |
| greedy | 0.628 | 0.812 | - | 0.560 | 0.544 | 0.576 | 0.580 |
| conservative | 0.564 | 0.708 | 0.440 | - | 0.488 | 0.472 | 0.492 |
| counting | 0.548 | 0.712 | 0.456 | 0.512 | - | 0.468 | 0.496 |
| ppo_final_250 | 0.588 | 0.752 | 0.424 | 0.528 | 0.532 | - | 0.500 |
| ppo_best_250 | 0.568 | 0.728 | 0.420 | 0.508 | 0.504 | 0.500 | - |

## Diagnostics

- Requested and completed games per policy: 500
- Multiplayer matchup games per matrix cell: 100
- Cyclic two-player matchups: none detected
- Evaluation deal seeds: [9001, 9002, 9003, 9004, 9005, 1009004, 1009005, 1009006, 1009007, 1009008, 2009007, 2009008, 2009009, 2009010, 2009011, 3009010, 3009011, 3009012, 3009013, 3009014, 4009013, 4009014, 4009015, 4009016, 4009017, 5009016, 5009017, 5009018, 5009019, 5009020, 6009019, 6009020]
- Invalid actions and mask disagreements: zero unless listed in machine-readable output.
