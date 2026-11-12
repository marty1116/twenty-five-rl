# Twenty-Five evaluation leaderboard

> Held-out duplicate-deal evaluation against the fixed `baseline` suite: random, greedy, conservative, counting.
> The 95% interval bootstraps complete deal clusters; candidate matchups below do not affect the leaderboard.

| Policy | Games | Wins | Win rate (95% CI) | Avg score | Avg diff | Tricks/round | 2P BT rating | Latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 80 | 24 | 0.300 (0.200-0.375) | 13.00 | -8.00 | 1.15 | 1586 | 0.009 |
| conservative | 80 | 20 | 0.250 (0.175-0.325) | 12.75 | -9.50 | 1.07 | 1549 | 0.013 |
| counting | 80 | 18 | 0.225 (0.150-0.300) | 11.88 | -10.75 | 0.93 | 1524 | 0.030 |
| random | 80 | 16 | 0.200 (0.150-0.250) | 11.88 | -11.00 | 0.99 | 1502 | 0.001 |
| lowest | 80 | 10 | 0.125 (0.025-0.225) | 10.38 | -13.75 | 0.76 | 1339 | 0.008 |

## 4-player focal policy versus homogeneous opponent table

| Policy | random | lowest | greedy | conservative | counting |
|---|---:|---:|---:|---:|---:|
| random | - | 0.850 | 0.050 | 0.050 | 0.450 |
| lowest | 0.250 | - | 0.000 | 0.050 | 0.050 |
| greedy | 0.250 | 0.900 | - | 0.250 | 0.500 |
| conservative | 0.250 | 0.450 | 0.200 | - | 0.450 |
| counting | 0.200 | 0.250 | 0.200 | 0.250 | - |

## Two-player head-to-head matrix

| Policy | random | lowest | greedy | conservative | counting |
|---|---:|---:|---:|---:|---:|
| random | - | 0.725 | 0.375 | 0.400 | 0.500 |
| lowest | 0.275 | - | 0.175 | 0.225 | 0.275 |
| greedy | 0.625 | 0.825 | - | 0.575 | 0.550 |
| conservative | 0.600 | 0.775 | 0.425 | - | 0.525 |
| counting | 0.500 | 0.725 | 0.450 | 0.475 | - |

## Diagnostics

- Requested and completed games per policy: 80
- Multiplayer matchup games per matrix cell: 20
- Cyclic two-player matchups: none detected
- Evaluation deal seeds: [9001, 9002, 9003, 9004, 9005]
- Invalid actions and mask disagreements: zero unless listed in machine-readable output.
