# Results

Every scenario is winnable: an all-knowing rescuer that starts 120 s late can still save the person (`navsim/scenario.py`). Test scenarios are #100 and up; calibration used #0-39.

**Success** = the person reaches the exit alive. **Efficiency** = best possible rescue time / actual rescue time (0 if the person is lost); the best possible time is that of an all-knowing rescuer starting at t = 0. Values are means with [95% bootstrap CI].

## Headline: the final system (10 rooms, 3 drones, scenarios #100-199)

| success | efficiency | time to find | best possible find | time to rescue | best possible rescue | person health at exit |
|---|---|---|---|---|---|---|
| **100%** (100/100) | **68 [65, 71] %** | 42.3 [37.1, 48.0] s | 22.6 s | 68.1 [60.8, 75.5] s | 45.1 s | 1.00 |

## Evidence for the design (same 100 scenarios)

Times and escort smoke dose are over the 94 scenarios every version rescued.

| version | success % | efficiency % | time to find (s) | time to rescue (s) | escort smoke dose |
|---|---|---|---|---|---|
| Final system | 100 | 68 [65, 71] | 39.9 [34.9, 45.4] | 65.7 [58.7, 73.6] | 0.221 [0.149, 0.301] |
| + fuzzy cost (rejected) | 98 | 66 [63, 70] | 41.6 [35.6, 48.2] | 67.6 [59.8, 76.1] | 0.262 [0.160, 0.388] |
| without Bayes | 98 | 70 [66, 74] | 37.8 [32.8, 42.8] | 63.7 [56.7, 70.6] | 0.210 [0.141, 0.280] |
| without GA (greedy) | 99 | 66 [63, 68] | 42.2 [36.5, 47.8] | 68.1 [59.5, 77.0] | 0.267 [0.184, 0.358] |
| neither (sweep) | 99 | 68 [64, 71] | 41.0 [35.1, 47.2] | 67.0 [58.9, 75.5] | 0.284 [0.164, 0.442] |

Paired with the final system on the same scenarios (other minus final; a CI that excludes 0 is a clear effect; McNemar p is for the scenarios only one of the two rescued):

| version | only final rescued | only this rescued | McNemar p | efficiency difference (points) | extra time to find (s) | extra escort smoke dose |
|---|---|---|---|---|---|---|
| + fuzzy cost (rejected) | 2 | 0 | 0.50 | -1.6 [-5.2, 1.5] | 1.4 [-2.0, 5.2] | 0.036 [-0.019, 0.112] |
| without Bayes | 2 | 0 | 0.50 | 2.0 [-0.7, 4.8] | -3.2 [-6.8, 0.4] | -0.012 [-0.053, 0.036] |
| without GA (greedy) | 1 | 0 | 1.00 | -2.1 [-4.7, 0.6] | 3.2 [0.3, 6.5] | 0.057 [0.016, 0.106] |
| neither (sweep) | 1 | 0 | 1.00 | -0.0 [-3.3, 3.4] | -0.5 [-5.1, 4.1] | 0.071 [-0.001, 0.169] |

## Evidence with a big fire (the 4-cell fire burned 150 s before the drones arrived)

| version | success % | efficiency % | efficiency vs final (points) | only final rescued | only this rescued | McNemar p |
|---|---|---|---|---|---|---|
| Final system | 99 | 67 | – | 0 | 0 | – |
| + fuzzy cost (rejected) | 99 | 68 | 0.8 [-2.6, 4.2] | 1 | 1 | 1.00 |
| without Bayes | 95 | 67 | -0.4 [-4.6, 3.6] | 5 | 1 | 0.22 |
| without GA (greedy) | 98 | 65 | -1.8 [-5.4, 1.6] | 2 | 1 | 1.00 |
| neither (sweep) | 96 | 66 | -1.3 [-5.4, 2.5] | 4 | 1 | 0.38 |

## Bayesian map vs coverage sweep as detection gets less reliable (scenarios #100-199)

pd0 = chance that one look spots the person up close (it falls further with distance and smoke).

| pd0 | Bayes: success % | sweep: success % | only Bayes rescued | only sweep rescued | McNemar p | Bayes: efficiency % | sweep: efficiency % |
|---|---|---|---|---|---|---|---|
| 0.95 | 100 | 98 | 2 | 0 | 0.500 | 68 | 70 |
| 0.70 | 100 | 95 | 5 | 0 | 0.062 | 65 | 66 |
| 0.50 | 99 | 89 | 11 | 1 | <0.01 | 63 | 60 |

## Scaling (scenarios #100-149)

Times are over the rescued runs.

| rooms | drones | allocation | success % | efficiency % | time to find (s) | ms/tick p95 |
|---|---|---|---|---|---|---|
| 6 | 1 | GA | 96 | 51 | 49.2 | 2.5 |
| 6 | 1 | greedy | 94 | 49 | 55.0 | 1.1 |
| 6 | 2 | GA | 100 | 65 | 32.2 | 7.0 |
| 6 | 2 | greedy | 98 | 64 | 31.7 | 2.0 |
| 6 | 3 | GA | 100 | 77 | 23.2 | 9.3 |
| 6 | 3 | greedy | 100 | 74 | 25.5 | 2.5 |
| 6 | 4 | GA | 100 | 80 | 20.1 | 9.7 |
| 6 | 4 | greedy | 100 | 76 | 23.1 | 3.2 |
| 6 | 6 | GA | 100 | 81 | 19.6 | 31.7 |
| 6 | 6 | greedy | 100 | 75 | 23.2 | 5.1 |
| 10 | 1 | GA | 90 | 39 | 99.2 | 9.1 |
| 10 | 1 | greedy | 94 | 43 | 86.7 | 2.9 |
| 10 | 2 | GA | 98 | 60 | 55.0 | 11.0 |
| 10 | 2 | greedy | 98 | 58 | 56.7 | 3.9 |
| 10 | 3 | GA | 100 | 67 | 44.7 | 10.8 |
| 10 | 3 | greedy | 100 | 66 | 47.9 | 4.2 |
| 10 | 4 | GA | 100 | 74 | 35.9 | 9.9 |
| 10 | 4 | greedy | 100 | 69 | 44.6 | 5.0 |
| 10 | 6 | GA | 100 | 78 | 30.9 | 18.9 |
| 10 | 6 | greedy | 98 | 71 | 38.2 | 6.6 |
| 16 | 1 | GA | 90 | 40 | 140.2 | 11.7 |
| 16 | 1 | greedy | 90 | 45 | 119.9 | 5.9 |
| 16 | 2 | GA | 98 | 63 | 77.3 | 14.4 |
| 16 | 2 | greedy | 98 | 62 | 80.7 | 6.8 |
| 16 | 3 | GA | 98 | 66 | 69.8 | 14.6 |
| 16 | 3 | greedy | 98 | 69 | 64.4 | 6.8 |
| 16 | 4 | GA | 98 | 72 | 58.2 | 14.1 |
| 16 | 4 | greedy | 96 | 71 | 57.3 | 7.6 |
| 16 | 6 | GA | 100 | 77 | 51.1 | 18.9 |
| 16 | 6 | greedy | 98 | 74 | 57.2 | 8.3 |

## GA allocation quality at t = 30 s (expected time to find; lower is better)

| rooms | drones | problems | GA better than greedy | GA gain over greedy % | random vs greedy % | GA ms | greedy ms |
|---|---|---|---|---|---|---|---|
| 10 | 3 | 58 | 95% | 9.2 [7.7, 10.7] | +184 | 144 | 0.2 |
| 16 | 6 | 74 | 100% | 10.3 [9.3, 11.2] | +171 | 211 | 0.3 |

## A* heuristic (same sampled planning queries, final system)

h(n) = octile distance to the goal. Every step costs at least its length, so h never overestimates (admissible) and A* still returns optimal paths.

| rooms | nodes expanded with no heuristic (Dijkstra) | nodes expanded by A* | saving |
|---|---|---|---|
| 6 | 194,215 | 50,070 | **74%** |
| 10 | 518,961 | 131,736 | **75%** |
| 16 | 1,215,084 | 272,247 | **78%** |
