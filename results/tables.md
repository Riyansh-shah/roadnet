# Results

Every scenario is winnable: an all-knowing rescuer that starts 120 s late can still save the person (`navsim/scenario.py`). Test scenarios are #100 and up; calibration used #0-39.

**Success** = the person reaches the exit alive. **Efficiency** = best possible rescue time / actual rescue time (0 if the person is lost); the best possible time is that of an all-knowing rescuer starting at t = 0. Values are means with [95% bootstrap CI].

## Headline: the full system (10 rooms, 3 drones, scenarios #100-199)

| success | efficiency | time to find | best possible find | time to rescue | best possible rescue | person health at exit |
|---|---|---|---|---|---|---|
| **99%** (99/100) | **68 [64, 71] %** | 41.8 [36.0, 47.9] s | 22.2 s | 67.9 [60.1, 76.1] s | 44.4 s | 1.00 |

## Why each component is needed (same 100 scenarios)

Times and escort smoke dose are over the 84 scenarios every version rescued.

| version | success % | efficiency % | time to find (s) | time to rescue (s) | escort smoke dose |
|---|---|---|---|---|---|
| Full system | 99 | 68 [64, 71] | 38.5 [33.5, 44.0] | 64.5 [57.0, 72.2] | 0.219 [0.123, 0.327] |
| without fuzzy A* | 99 | 69 [66, 72] | 37.9 [32.7, 43.2] | 64.1 [56.2, 71.9] | 0.249 [0.144, 0.389] |
| without Bayes | 98 | 51 [48, 55] | 58.0 [46.4, 71.1] | 97.0 [83.2, 112.4] | 0.313 [0.175, 0.481] |
| without GA | 97 | 65 [62, 68] | 41.7 [35.5, 48.2] | 67.7 [58.8, 76.9] | 0.231 [0.136, 0.353] |
| none of the three | 89 | 44 [40, 49] | 63.7 [50.9, 77.0] | 105.5 [88.8, 124.3] | 0.680 [0.282, 1.174] |

Paired with the full system on the same scenarios (other minus full; a CI that excludes 0 is a clear effect; McNemar p is for the scenarios only one of the two rescued):

| version | only full rescued | only this rescued | McNemar p | efficiency difference (points) | extra time to find (s) | extra escort smoke dose |
|---|---|---|---|---|---|---|
| without fuzzy A* | 1 | 1 | 1.00 | 1.1 [-2.1, 4.4] | -2.2 [-6.3, 1.4] | 0.025 [-0.005, 0.067] |
| without Bayes | 2 | 1 | 1.00 | -16.6 [-20.3, -12.6] | 16.6 [7.4, 27.3] | 0.204 [0.074, 0.340] |
| without GA | 2 | 0 | 0.50 | -3.0 [-7.0, 0.9] | 2.2 [-3.7, 7.8] | 0.024 [-0.012, 0.071] |
| none of the three | 11 | 1 | <0.01 | -23.7 [-28.2, -19.5] | 24.4 [14.8, 35.3] | 0.444 [0.121, 0.898] |

## Scaling (scenarios #100-149)

Times are over the rescued runs.

| rooms | drones | allocation | success % | efficiency % | time to find (s) | ms/tick p95 |
|---|---|---|---|---|---|---|
| 6 | 1 | GA | 100 | 58 | 48.1 | 54.7 |
| 6 | 1 | greedy | 100 | 55 | 52.9 | 11.4 |
| 6 | 2 | GA | 100 | 74 | 25.0 | 83.9 |
| 6 | 2 | greedy | 100 | 71 | 27.0 | 14.7 |
| 6 | 3 | GA | 100 | 79 | 21.3 | 87.8 |
| 6 | 3 | greedy | 100 | 75 | 25.1 | 16.5 |
| 6 | 4 | GA | 98 | 77 | 20.7 | 111.7 |
| 6 | 4 | greedy | 100 | 77 | 23.1 | 18.2 |
| 6 | 6 | GA | 100 | 81 | 19.3 | 130.9 |
| 6 | 6 | greedy | 100 | 78 | 21.5 | 19.9 |
| 10 | 1 | GA | 94 | 45 | 86.5 | 70.9 |
| 10 | 1 | greedy | 98 | 43 | 98.7 | 28.7 |
| 10 | 2 | GA | 98 | 61 | 51.6 | 146.4 |
| 10 | 2 | greedy | 96 | 60 | 54.4 | 37.7 |
| 10 | 3 | GA | 98 | 67 | 42.4 | 154.4 |
| 10 | 3 | greedy | 96 | 64 | 47.2 | 49.7 |
| 10 | 4 | GA | 100 | 70 | 39.3 | 186.4 |
| 10 | 4 | greedy | 98 | 69 | 41.8 | 44.8 |
| 10 | 6 | GA | 100 | 74 | 33.7 | 184.4 |
| 10 | 6 | greedy | 96 | 71 | 36.9 | 53.7 |
| 16 | 1 | GA | 90 | 42 | 123.8 | 154.7 |
| 16 | 1 | greedy | 94 | 43 | 130.2 | 68.6 |
| 16 | 2 | GA | 100 | 63 | 78.2 | 242.8 |
| 16 | 2 | greedy | 100 | 60 | 86.9 | 95.2 |
| 16 | 3 | GA | 100 | 70 | 64.7 | 254.3 |
| 16 | 3 | greedy | 100 | 68 | 68.4 | 122.9 |
| 16 | 4 | GA | 100 | 74 | 59.5 | 232.6 |
| 16 | 4 | greedy | 100 | 69 | 67.7 | 138.7 |
| 16 | 6 | GA | 98 | 76 | 50.3 | 309.9 |
| 16 | 6 | greedy | 100 | 77 | 52.0 | 145.3 |

## GA allocation quality at t = 30 s (expected time to find; lower is better)

| rooms | drones | problems | GA better than greedy | GA gain over greedy % | random vs greedy % | GA ms | greedy ms |
|---|---|---|---|---|---|---|---|
| 10 | 3 | 57 | 93% | 9.0 [7.5, 10.6] | +176 | 235 | 0.3 |
| 16 | 6 | 71 | 100% | 12.3 [10.3, 15.2] | +174 | 366 | 0.5 |

## A* heuristic (same sampled queries, full system)

h(n) = c_min x octile distance, c_min = cheapest cell cost on the grid (admissible and consistent; paths stay optimal).

| rooms | nodes with no heuristic (Dijkstra) | h = octile | h = c_min x octile |
|---|---|---|---|
| 6 | 228,803 (100%) | 65% | **35%** |
| 10 | 539,075 (100%) | 65% | **31%** |
| 16 | 1,276,250 (100%) | 69% | **31%** |
