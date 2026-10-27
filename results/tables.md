# Results

Every scenario is winnable: an all-knowing rescuer that starts 120 s late can still save the person (`navsim/scenario.py`). Test scenarios are #100 and up; calibration used #0-39.

**Success** = the person reaches the exit alive. **Efficiency** = best possible rescue time / actual rescue time (0 if the person is lost); the best possible time is that of an all-knowing rescuer starting at t = 0. Values are means with [95% bootstrap CI].

## Headline: the final system (10 rooms, 3 drones, scenarios #100-199)

| success | efficiency | time to find | best possible find | time to rescue | best possible rescue | person health at exit |
|---|---|---|---|---|---|---|
| **99%** (99/100) | **69 [66, 72] %** | 39.6 [34.3, 45.1] s | 22.2 s | 65.8 [58.6, 73.6] s | 44.4 s | 1.00 |

## Evidence for the design (same 100 scenarios)

Times and escort smoke dose are over the 85 scenarios every version rescued.

| version | success % | efficiency % | time to find (s) | time to rescue (s) | escort smoke dose |
|---|---|---|---|---|---|
| Final system | 99 | 69 [66, 72] | 38.3 [33.3, 43.5] | 65.0 [57.1, 73.0] | 0.253 [0.145, 0.386] |
| + fuzzy cost (rejected) | 99 | 68 [64, 71] | 38.7 [33.6, 44.1] | 65.1 [57.7, 72.8] | 0.222 [0.134, 0.336] |
| without Bayes | 95 | 46 [42, 50] | 63.5 [50.3, 77.7] | 108.5 [92.3, 126.4] | 0.469 [0.247, 0.764] |
| without GA (greedy) | 98 | 67 [63, 69] | 41.4 [34.8, 48.0] | 67.8 [59.2, 77.3] | 0.236 [0.137, 0.350] |
| neither (sweep) | 89 | 44 [40, 49] | 64.0 [52.1, 77.5] | 106.8 [91.4, 123.9] | 0.682 [0.292, 1.200] |

Paired with the final system on the same scenarios (other minus final; a CI that excludes 0 is a clear effect; McNemar p is for the scenarios only one of the two rescued):

| version | only final rescued | only this rescued | McNemar p | efficiency difference (points) | extra time to find (s) | extra escort smoke dose |
|---|---|---|---|---|---|---|
| + fuzzy cost (rejected) | 1 | 1 | 1.00 | -1.1 [-4.3, 2.1] | 2.2 [-1.3, 6.4] | -0.025 [-0.063, 0.005] |
| without Bayes | 5 | 1 | 0.22 | -22.9 [-27.0, -18.7] | 24.0 [13.8, 36.5] | 0.222 [0.084, 0.400] |
| without GA (greedy) | 1 | 0 | 1.00 | -2.5 [-5.5, 0.5] | 2.9 [-0.7, 6.7] | -0.009 [-0.042, 0.023] |
| neither (sweep) | 11 | 1 | <0.01 | -24.8 [-29.4, -20.2] | 24.7 [15.5, 35.2] | 0.414 [0.109, 0.879] |

## Scaling (scenarios #100-149)

Times are over the rescued runs.

| rooms | drones | allocation | success % | efficiency % | time to find (s) | ms/tick p95 |
|---|---|---|---|---|---|---|
| 6 | 1 | GA | 98 | 54 | 47.7 | 56.3 |
| 6 | 1 | greedy | 98 | 55 | 51.3 | 11.4 |
| 6 | 2 | GA | 100 | 71 | 26.1 | 51.8 |
| 6 | 2 | greedy | 100 | 69 | 28.9 | 8.2 |
| 6 | 3 | GA | 100 | 78 | 21.2 | 75.9 |
| 6 | 3 | greedy | 100 | 75 | 24.6 | 11.9 |
| 6 | 4 | GA | 100 | 80 | 19.8 | 59.2 |
| 6 | 4 | greedy | 100 | 76 | 24.2 | 9.6 |
| 6 | 6 | GA | 100 | 82 | 18.7 | 76.1 |
| 6 | 6 | greedy | 100 | 76 | 23.9 | 12.5 |
| 10 | 1 | GA | 96 | 45 | 87.2 | 55.5 |
| 10 | 1 | greedy | 94 | 41 | 109.6 | 17.3 |
| 10 | 2 | GA | 100 | 62 | 54.1 | 91.0 |
| 10 | 2 | greedy | 100 | 62 | 51.2 | 22.6 |
| 10 | 3 | GA | 98 | 69 | 39.8 | 103.4 |
| 10 | 3 | greedy | 96 | 65 | 44.5 | 25.3 |
| 10 | 4 | GA | 100 | 71 | 37.4 | 106.3 |
| 10 | 4 | greedy | 98 | 68 | 40.9 | 25.6 |
| 10 | 6 | GA | 98 | 73 | 33.2 | 118.1 |
| 10 | 6 | greedy | 96 | 70 | 38.3 | 29.6 |
| 16 | 1 | GA | 90 | 38 | 142.6 | 127.0 |
| 16 | 1 | greedy | 100 | 44 | 151.1 | 71.5 |
| 16 | 2 | GA | 100 | 65 | 76.3 | 157.3 |
| 16 | 2 | greedy | 98 | 60 | 81.5 | 57.5 |
| 16 | 3 | GA | 100 | 69 | 64.1 | 170.4 |
| 16 | 3 | greedy | 100 | 70 | 63.1 | 61.3 |
| 16 | 4 | GA | 100 | 75 | 57.1 | 159.0 |
| 16 | 4 | greedy | 100 | 73 | 57.4 | 73.6 |
| 16 | 6 | GA | 100 | 78 | 49.7 | 195.0 |
| 16 | 6 | greedy | 100 | 76 | 54.9 | 76.3 |

## GA allocation quality at t = 30 s (expected time to find; lower is better)

| rooms | drones | problems | GA better than greedy | GA gain over greedy % | random vs greedy % | GA ms | greedy ms |
|---|---|---|---|---|---|---|---|
| 10 | 3 | 54 | 85% | 9.7 [7.5, 11.8] | +172 | 137 | 0.2 |
| 16 | 6 | 63 | 100% | 12.0 [10.9, 13.2] | +168 | 221 | 0.3 |

## A* heuristic (same sampled planning queries, final system)

h(n) = octile distance to the goal. Every step costs at least its length, so h never overestimates (admissible) and A* still returns optimal paths.

| rooms | nodes expanded with no heuristic (Dijkstra) | nodes expanded by A* | saving |
|---|---|---|---|
| 6 | 210,540 | 69,534 | **67%** |
| 10 | 567,932 | 179,109 | **68%** |
| 16 | 1,221,828 | 328,524 | **73%** |
