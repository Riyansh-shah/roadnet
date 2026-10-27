# Results

All experiments use test seeds 100+ (calibration used seeds 0-39).

## E1: leave-one-out ablation (10 rooms, 3 units, seeds 100-199)

Times and escort dose: mean [95% bootstrap CI] over the 67 seeds every method rescued.

| method | rescued % | casualty % | time to find (s) | time to rescue (s) | escort smoke dose | bumps | unit exposure | ms/tick mean | ms/tick p95 |
|---|---|---|---|---|---|---|---|---|---|
| FULL (all three) | 80 | 18 | 41.4 [35.3, 48.2] | 55.3 [48.8, 61.9] | 0.077 [0.020, 0.170] | 0.21 | 13.71 | 26.6 | 174.7 |
| without fuzzy | 83 | 15 | 40.7 [34.9, 46.6] | 54.9 [48.9, 60.9] | 0.133 [0.025, 0.273] | 0.34 | 13.96 | 25.0 | 174.0 |
| without Bayes | 75 | 19 | 61.9 [47.9, 79.3] | 103.6 [87.6, 122.5] | 0.589 [0.256, 0.954] | 0.12 | 6.43 | 25.0 | 150.5 |
| without GA | 80 | 17 | 48.1 [39.9, 56.5] | 62.4 [54.1, 70.3] | 0.069 [0.026, 0.139] | 0.27 | 12.96 | 8.2 | 50.0 |
| B0 (none) | 73 | 22 | 62.4 [49.5, 75.7] | 102.2 [87.7, 117.2] | 0.548 [0.280, 0.890] | 0.09 | 6.65 | 6.6 | 41.4 |

Paired comparison with FULL on the same seeds: seeds where only one of the two rescued the person (exact McNemar p), and mean differences on seeds both rescued (other minus FULL, 95% bootstrap CI; a CI that excludes 0 is a clear effect).

| method | only FULL rescued | only this rescued | McNemar p | extra time to find (s) | extra escort smoke dose |
|---|---|---|---|---|---|
| without fuzzy | 1 | 4 | 0.38 | -1.6 [-5.2, 1.6] | 0.055 [-0.003, 0.161] |
| without Bayes | 7 | 2 | 0.18 | 22.1 [9.8, 36.3] | 0.542 [0.230, 0.914] |
| without GA | 2 | 2 | 1.00 | 4.2 [-1.6, 10.6] | -0.010 [-0.048, 0.022] |
| B0 (none) | 9 | 2 | 0.07 | 21.1 [11.0, 31.7] | 0.455 [0.218, 0.753] |

## E2: scaling (seeds 100-149)

Times are means over the rescued runs.

| rooms | units | method | rescued % | time to find (s) | time to rescue (s) | ms/tick p95 |
|---|---|---|---|---|---|---|
| 6 | 1 | FULL | 82 | 38.0 | 50.4 | 49.8 |
| 6 | 1 | noGA | 80 | 36.1 | 47.7 | 12.9 |
| 6 | 2 | FULL | 82 | 21.2 | 33.2 | 87.7 |
| 6 | 2 | noGA | 80 | 24.6 | 36.2 | 15.2 |
| 6 | 3 | FULL | 82 | 20.7 | 32.5 | 99.6 |
| 6 | 3 | noGA | 82 | 23.0 | 35.0 | 17.1 |
| 6 | 4 | FULL | 82 | 17.2 | 29.3 | 113.3 |
| 6 | 4 | noGA | 82 | 19.7 | 31.6 | 16.9 |
| 6 | 6 | FULL | 82 | 17.3 | 29.2 | 150.9 |
| 6 | 6 | noGA | 82 | 21.2 | 33.2 | 20.4 |
| 10 | 1 | FULL | 80 | 95.0 | 109.2 | 127.0 |
| 10 | 1 | noGA | 70 | 103.3 | 118.5 | 37.8 |
| 10 | 2 | FULL | 78 | 59.3 | 74.6 | 159.6 |
| 10 | 2 | noGA | 78 | 61.2 | 75.5 | 46.2 |
| 10 | 3 | FULL | 78 | 51.9 | 66.0 | 170.0 |
| 10 | 3 | noGA | 80 | 54.1 | 68.4 | 53.5 |
| 10 | 4 | FULL | 84 | 44.2 | 59.1 | 201.5 |
| 10 | 4 | noGA | 80 | 46.8 | 61.0 | 49.1 |
| 10 | 6 | FULL | 84 | 40.9 | 55.1 | 228.0 |
| 10 | 6 | noGA | 78 | 41.8 | 56.4 | 57.2 |
| 16 | 1 | FULL | 84 | 136.2 | 156.4 | 217.1 |
| 16 | 1 | noGA | 82 | 137.6 | 157.2 | 111.9 |
| 16 | 2 | FULL | 88 | 90.5 | 111.3 | 272.0 |
| 16 | 2 | noGA | 88 | 83.5 | 104.0 | 125.4 |
| 16 | 3 | FULL | 90 | 69.7 | 89.7 | 309.9 |
| 16 | 3 | noGA | 90 | 68.1 | 88.2 | 145.0 |
| 16 | 4 | FULL | 90 | 60.0 | 78.9 | 340.2 |
| 16 | 4 | noGA | 90 | 68.3 | 88.4 | 147.5 |
| 16 | 6 | FULL | 90 | 51.9 | 71.5 | 375.0 |
| 16 | 6 | noGA | 90 | 55.2 | 75.0 | 163.8 |

## E3: allocation objective at t = 30 s (expected find time; lower is better)

| rooms | units | problems | GA better than greedy | GA gain over greedy % | random vs greedy % | GA ms | greedy ms |
|---|---|---|---|---|---|---|---|
| 10 | 3 | 60 | 95% | 13.6 [9.4, 19.0] | +158 | 245 | 0.4 |
| 16 | 6 | 78 | 96% | 11.1 [9.4, 13.1] | +178 | 385 | 0.5 |
