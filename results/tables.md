# Results

All experiments use test seeds 100+ (calibration used seeds 0-39).

## E1: leave-one-out ablation (10 rooms, 3 drones, seeds 100-199)

Times and escort dose: mean [95% bootstrap CI] over the 59 seeds every method rescued.

| method | rescued % | casualty % | time to find (s) | time to rescue (s) | escort smoke dose | bumps | drone smoke exposure | ms/tick mean | ms/tick p95 |
|---|---|---|---|---|---|---|---|---|---|
| FULL (all three) | 73 | 23 | 38.8 [32.5, 45.4] | 65.6 [55.8, 75.7] | 0.329 [0.140, 0.607] | 0.30 | 15.09 | 18.1 | 112.0 |
| without fuzzy | 75 | 21 | 36.8 [31.3, 42.7] | 63.6 [54.7, 72.7] | 0.383 [0.168, 0.690] | 0.33 | 14.86 | 17.1 | 119.3 |
| without Bayes | 66 | 29 | 54.4 [43.4, 66.1] | 94.7 [79.4, 110.3] | 1.277 [0.226, 3.058] | 0.16 | 6.23 | 18.7 | 122.5 |
| without GA | 72 | 25 | 42.5 [34.7, 50.6] | 69.3 [58.4, 79.8] | 0.409 [0.161, 0.740] | 0.28 | 13.57 | 5.7 | 31.0 |
| B0 (none) | 62 | 31 | 61.9 [47.4, 78.1] | 106.0 [85.2, 129.4] | 0.950 [0.370, 1.728] | 0.13 | 6.66 | 4.8 | 31.1 |

Paired comparison with FULL on the same seeds: seeds where only one of the two rescued the person (exact McNemar p), and mean differences on seeds both rescued (other minus FULL, 95% bootstrap CI; a CI that excludes 0 is a clear effect).

| method | only FULL rescued | only this rescued | McNemar p | extra time to find (s) | extra escort smoke dose |
|---|---|---|---|---|---|
| without fuzzy | 0 | 2 | 0.50 | -2.1 [-5.9, 1.2] | 0.115 [0.033, 0.224] |
| without Bayes | 8 | 1 | 0.04 | 14.2 [5.6, 23.4] | 0.888 [0.098, 2.311] |
| without GA | 2 | 1 | 1.00 | 6.5 [1.0, 12.2] | 0.222 [0.032, 0.494] |
| B0 (none) | 11 | 0 | <0.01 | 20.9 [9.8, 33.5] | 0.925 [0.275, 1.785] |

## E2: scaling (seeds 100-149)

Times are means over the rescued runs.

| rooms | drones | method | rescued % | time to find (s) | time to rescue (s) | ms/tick p95 |
|---|---|---|---|---|---|---|
| 6 | 1 | FULL | 74 | 37.6 | 57.1 | 36.8 |
| 6 | 1 | noGA | 76 | 38.6 | 56.4 | 7.1 |
| 6 | 2 | FULL | 74 | 20.6 | 37.2 | 67.2 |
| 6 | 2 | noGA | 72 | 24.9 | 41.8 | 10.1 |
| 6 | 3 | FULL | 74 | 19.6 | 35.9 | 71.5 |
| 6 | 3 | noGA | 74 | 20.8 | 37.6 | 12.1 |
| 6 | 4 | FULL | 76 | 17.2 | 33.6 | 80.4 |
| 6 | 4 | noGA | 74 | 20.1 | 36.9 | 13.9 |
| 6 | 6 | FULL | 74 | 17.3 | 33.3 | 98.9 |
| 6 | 6 | noGA | 76 | 18.8 | 35.3 | 15.2 |
| 10 | 1 | FULL | 56 | 84.2 | 112.1 | 59.8 |
| 10 | 1 | noGA | 60 | 92.6 | 120.8 | 21.8 |
| 10 | 2 | FULL | 68 | 53.8 | 84.1 | 104.0 |
| 10 | 2 | noGA | 64 | 52.6 | 82.2 | 21.4 |
| 10 | 3 | FULL | 68 | 42.7 | 71.9 | 67.2 |
| 10 | 3 | noGA | 66 | 50.9 | 80.7 | 16.0 |
| 10 | 4 | FULL | 70 | 38.9 | 69.1 | 54.6 |
| 10 | 4 | noGA | 70 | 41.8 | 71.3 | 16.5 |
| 10 | 6 | FULL | 72 | 34.3 | 64.4 | 93.0 |
| 10 | 6 | noGA | 68 | 39.1 | 69.1 | 37.6 |
| 16 | 1 | FULL | 70 | 134.0 | 174.7 | 120.4 |
| 16 | 1 | noGA | 72 | 132.6 | 175.4 | 60.4 |
| 16 | 2 | FULL | 84 | 77.6 | 119.6 | 175.3 |
| 16 | 2 | noGA | 80 | 78.0 | 119.7 | 86.7 |
| 16 | 3 | FULL | 82 | 63.5 | 104.6 | 213.2 |
| 16 | 3 | noGA | 86 | 65.7 | 107.8 | 99.9 |
| 16 | 4 | FULL | 84 | 62.8 | 104.7 | 221.9 |
| 16 | 4 | noGA | 86 | 67.9 | 110.1 | 114.3 |
| 16 | 6 | FULL | 86 | 49.8 | 92.0 | 259.3 |
| 16 | 6 | noGA | 86 | 54.8 | 96.7 | 117.0 |

## E3: allocation objective at t = 30 s (expected find time; lower is better)

| rooms | drones | problems | GA better than greedy | GA gain over greedy % | random vs greedy % | GA ms | greedy ms |
|---|---|---|---|---|---|---|---|
| 10 | 3 | 62 | 92% | 12.6 [8.9, 17.1] | +158 | 190 | 0.3 |
| 16 | 6 | 79 | 97% | 11.1 [9.5, 13.2] | +177 | 305 | 0.4 |
