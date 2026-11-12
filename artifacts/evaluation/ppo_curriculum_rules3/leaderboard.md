# Twenty-Five evaluation leaderboard

> Held-out duplicate-deal evaluation against the fixed `baseline` suite: random, greedy, conservative, counting.
> The 95% interval bootstraps complete deal clusters; candidate matchups below do not affect the leaderboard.

| Policy | Games | Wins | Win rate (95% CI) | Avg score | Avg diff | Tricks/round | 2P BT rating | Latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 1000 | 309 | 0.309 (0.282-0.337) | 14.30 | -7.24 | 1.17 | 1580 | 0.019 |
| conservative | 1000 | 276 | 0.276 (0.248-0.305) | 14.26 | -8.02 | 1.13 | 1536 | 0.028 |
| counting | 1000 | 237 | 0.237 (0.208-0.265) | 13.29 | -9.40 | 1.02 | 1528 | 0.058 |
| rules3_final | 1000 | 228 | 0.228 (0.202-0.254) | 12.84 | -9.53 | 1.06 | 1534 | 0.282 |
| rules3_best | 1000 | 220 | 0.220 (0.196-0.244) | 12.72 | -9.86 | 1.03 | 1517 | 0.353 |
| random | 1000 | 183 | 0.183 (0.161-0.207) | 11.81 | -11.31 | 0.93 | 1456 | 0.001 |
| lowest | 1000 | 124 | 0.124 (0.098-0.151) | 10.18 | -13.76 | 0.77 | 1349 | 0.017 |

## 4-player focal policy versus homogeneous opponent table

| Policy | random | lowest | greedy | conservative | counting | rules3_final | rules3_best |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | - | 0.530 | 0.180 | 0.090 | 0.170 | 0.200 | 0.180 |
| lowest | 0.210 | - | 0.130 | 0.160 | 0.090 | 0.130 | 0.120 |
| greedy | 0.330 | 0.700 | - | 0.210 | 0.250 | 0.330 | 0.330 |
| conservative | 0.350 | 0.660 | 0.300 | - | 0.250 | 0.250 | 0.290 |
| counting | 0.310 | 0.370 | 0.240 | 0.330 | - | 0.170 | 0.240 |
| rules3_final | 0.250 | 0.670 | 0.250 | 0.210 | 0.330 | - | 0.220 |
| rules3_best | 0.260 | 0.660 | 0.260 | 0.290 | 0.370 | 0.210 | - |

## Two-player head-to-head matrix

| Policy | random | lowest | greedy | conservative | counting | rules3_final | rules3_best |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | - | 0.652 | 0.374 | 0.360 | 0.364 | 0.402 | 0.410 |
| lowest | 0.348 | - | 0.174 | 0.278 | 0.288 | 0.246 | 0.270 |
| greedy | 0.626 | 0.826 | - | 0.590 | 0.592 | 0.550 | 0.574 |
| conservative | 0.640 | 0.722 | 0.410 | - | 0.518 | 0.512 | 0.540 |
| counting | 0.636 | 0.712 | 0.408 | 0.482 | - | 0.496 | 0.528 |
| rules3_final | 0.598 | 0.754 | 0.450 | 0.488 | 0.504 | - | 0.520 |
| rules3_best | 0.590 | 0.730 | 0.426 | 0.460 | 0.472 | 0.480 | - |

## Diagnostics

- Requested and completed games per policy: 1000
- Multiplayer matchup games per matrix cell: 100
- Cyclic two-player matchups: none detected
- Evaluation deal seeds: [15001, 15002, 15003, 15004, 15005, 1015004, 1015005, 1015006, 1015007, 1015008, 2015007, 2015008, 2015009, 2015010, 2015011, 3015010, 3015011, 3015012, 3015013, 3015014, 4015013, 4015014, 4015015, 4015016, 4015017, 5015016, 5015017, 5015018, 5015019, 5015020, 6015019, 6015020, 6015021, 6015022, 6015023, 7015022, 7015023, 7015024, 7015025, 7015026, 8015025, 8015026, 8015027, 8015028, 8015029, 9015028, 9015029, 9015030, 9015031, 9015032, 10015031, 10015032, 10015033, 10015034, 10015035, 11015034, 11015035, 11015036, 11015037, 11015038, 12015037, 12015038, 12015039]
- Invalid actions and mask disagreements: zero unless listed in machine-readable output.
