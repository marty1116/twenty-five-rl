# Twenty-Five evaluation leaderboard

> Held-out duplicate-deal evaluation against the fixed `baseline` suite: random, greedy, conservative, counting.
> The 95% interval bootstraps complete deal clusters; candidate matchups below do not affect the leaderboard.

| Policy | Games | Wins | Win rate (95% CI) | Avg score | Avg diff | Tricks/round | 2P BT rating | Latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 1000 | 294 | 0.294 (0.266-0.320) | 13.65 | -7.71 | 1.20 | 1558 | 0.018 |
| conservative | 1000 | 266 | 0.266 (0.240-0.292) | 13.68 | -8.54 | 1.09 | 1537 | 0.028 |
| rules3_3000_best | 1000 | 248 | 0.248 (0.220-0.276) | 12.68 | -9.42 | 1.05 | 1534 | 0.282 |
| rules3_3000_final | 1000 | 244 | 0.244 (0.218-0.268) | 12.66 | -9.49 | 1.05 | 1535 | 0.280 |
| counting | 1000 | 242 | 0.242 (0.218-0.266) | 12.99 | -9.41 | 1.04 | 1529 | 0.060 |
| random | 1000 | 185 | 0.185 (0.159-0.210) | 11.57 | -11.56 | 0.92 | 1473 | 0.001 |
| lowest | 1000 | 115 | 0.115 (0.087-0.143) | 9.74 | -14.24 | 0.74 | 1335 | 0.017 |

## 4-player focal policy versus homogeneous opponent table

| Policy | random | lowest | greedy | conservative | counting | rules3_3000_final | rules3_3000_best |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | - | 0.410 | 0.200 | 0.090 | 0.170 | 0.290 | 0.210 |
| lowest | 0.160 | - | 0.040 | 0.160 | 0.120 | 0.120 | 0.120 |
| greedy | 0.460 | 0.800 | - | 0.210 | 0.460 | 0.300 | 0.300 |
| conservative | 0.450 | 0.490 | 0.200 | - | 0.280 | 0.380 | 0.300 |
| counting | 0.340 | 0.470 | 0.160 | 0.160 | - | 0.210 | 0.280 |
| rules3_3000_final | 0.220 | 0.590 | 0.240 | 0.130 | 0.380 | - | 0.210 |
| rules3_3000_best | 0.300 | 0.630 | 0.250 | 0.170 | 0.290 | 0.250 | - |

## Two-player head-to-head matrix

| Policy | random | lowest | greedy | conservative | counting | rules3_3000_final | rules3_3000_best |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | - | 0.690 | 0.396 | 0.384 | 0.404 | 0.442 | 0.410 |
| lowest | 0.310 | - | 0.170 | 0.294 | 0.304 | 0.206 | 0.208 |
| greedy | 0.604 | 0.830 | - | 0.542 | 0.542 | 0.510 | 0.512 |
| conservative | 0.616 | 0.706 | 0.458 | - | 0.522 | 0.520 | 0.522 |
| counting | 0.596 | 0.696 | 0.458 | 0.478 | - | 0.512 | 0.526 |
| rules3_3000_final | 0.558 | 0.794 | 0.490 | 0.480 | 0.488 | - | 0.510 |
| rules3_3000_best | 0.590 | 0.792 | 0.488 | 0.478 | 0.474 | 0.490 | - |

## Diagnostics

- Requested and completed games per policy: 1000
- Multiplayer matchup games per matrix cell: 100
- Cyclic two-player matchups: none detected
- Evaluation deal seeds: [25001, 25002, 25003, 25004, 25005, 1025004, 1025005, 1025006, 1025007, 1025008, 2025007, 2025008, 2025009, 2025010, 2025011, 3025010, 3025011, 3025012, 3025013, 3025014, 4025013, 4025014, 4025015, 4025016, 4025017, 5025016, 5025017, 5025018, 5025019, 5025020, 6025019, 6025020, 6025021, 6025022, 6025023, 7025022, 7025023, 7025024, 7025025, 7025026, 8025025, 8025026, 8025027, 8025028, 8025029, 9025028, 9025029, 9025030, 9025031, 9025032, 10025031, 10025032, 10025033, 10025034, 10025035, 11025034, 11025035, 11025036, 11025037, 11025038, 12025037, 12025038, 12025039]
- Invalid actions and mask disagreements: zero unless listed in machine-readable output.
