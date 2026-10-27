# Drone search and rescue in a burning building: Bayesian search, GA allocation and A*

![demo](figs/demo_seed3.gif)

## The problem

K drones enter a single-floor building with N rooms through **one door, which is also the only way out**. One person is
somewhere inside, unconscious, at a random spot, and loses health in smoke. The drones must **find the person and lead
them back out through that door**, while the fire spreads, smoke diffuses and debris blocks doors and corridor sections.
The drones know the floor plan, but not the fire, the blockages or where the person is. All they have is a short-range
noisy sensor, and they make a decision every simulated second.

**Every scenario is winnable.** `navsim/scenario.py` keeps a scenario only if an all-knowing rescuer could still save the person
even when it **starts 120 s late**. That rescuer knows the fire, the debris and where the person is. Without this check,
25% of random fires made the rescue impossible for anyone.

## The method: three parts, each kept because it measurably helps

| Part | Question it answers | How |
|---|---|---|
| **Bayesian search map** (`navsim/belief.py`) | *Where is the person probably?* | P(person) over every free cell. A look that finds nobody multiplies a cell's probability by (1 − p_detect), and p_detect falls with distance and smoke, so rooms glanced at through smoke keep part of their probability and are searched again. The same module keeps a Bayes filter for "is this door blocked?" (debris never clears; bumping into it is certain) and a Kalman estimate of smoke. |
| **Genetic algorithm** (`navsim/ga.py`) | *Which drone searches which rooms, in what order?* | Chromosome = permutation of rooms + K−1 separators. Fitness = **expected time to find the person = Σ P(room) × arrival time at the room**. Uses order crossover, inversion and swap mutation, tournament selection and elitism, vectorised over the population, and re-runs when the beliefs change. |
| **A\*** (`navsim/astar.py`) | *How does each drone get there, and how does it lead the person out?* | 8-connected grid, cells believed on fire or blocked are avoided, with the heuristic below. |

## Heuristics

- **A\* heuristic:** `f(n) = g(n) + h(n)`, with `h(n) = octile distance to the goal = max(dr, dc) + (√2 − 1)·min(dr, dc)`.
  Every step costs at least its own length, so `h` never overestimates (it is admissible) and never drops by more than a
  step's cost (it is consistent). A\* therefore returns optimal paths. **Measured: A\* expands 67-73% fewer nodes than Dijkstra**
  (no heuristic) on the same planning queries (fig 8).
- **Search heuristic (GA fitness):** expected time to find = Σ P(room) × arrival time. It sends drones first to rooms that are
  both likely and close.

## Metrics

| Metric | Definition |
|---|---|
| **Success** (primary) | The person reaches the exit alive. Over many missions: success rate. |
| **Efficiency** | Best possible rescue time ÷ our rescue time (0 if the person is lost). The best possible time is that of the all-knowing rescuer starting at t = 0, so 100% is unbeatable. |
| **POS(t)** | Bayesian probability that the search so far would have found the person. It is shown live in the app. |
| Node saving | A\* nodes expanded vs Dijkstra on the same queries. |

## Results

Test scenarios #100-199 (calibration used #0-39). Full tables are in `results/tables.md`.

**Final system, 10 rooms, 3 drones:** **99% success (99/100)**, **efficiency 69% [66, 72]**. The person is found in 39.6 s
(the best possible is 22.2 s) and is outside in 65.8 s (the best possible is 44.4 s).

**Evidence for the design** (same 100 scenarios; differences are paired, final system as reference, 95% CI):

| version | success | efficiency | vs final: efficiency | vs final: time to find |
|---|---|---|---|---|
| **Final system** (Bayes + GA + A\*) | 99% | 69% | – | – |
| without Bayes | 95% | 46% | **−22.9 points [−27.0, −18.7]** | **+24.0 s [13.8, 36.5]** |
| without GA (greedy allocation) | 98% | 67% | −2.5 [−5.5, 0.5] | +2.9 s [−0.7, 6.7] |
| neither (plain room-by-room sweep) | 89% | 44% | **−24.8 [−29.4, −20.2]** | **+24.7 s [15.5, 35.2]** |
| + fuzzy cost (tested, rejected) | 99% | 68% | −1.1 [−4.3, 2.1] | +2.2 s [−1.3, 6.4] |

- **Bayes is essential:** without it the search is about 24 s slower and efficiency drops by 23 points.
- **The GA is better than greedy allocation:** on its own objective it wins in 85-100% of allocation problems, by 10-12%. Across
  all 750 scaling missions it raises efficiency by **+2.0 points [+0.9, +3.2]**, and it finds the person faster in 13 of the 15
  building/drone combinations. In the single 10-room test above, the difference alone isn't significant.
- **Fuzzy logic was tried and left out.** A Mamdani fuzzy cell cost (4 inputs, 20 rules; `navsim/fuzzy.py`, fig 2) was built
  and tested: as a path cost, under heavy smoke (300 s pre-burn), as a "when to leave a room" rule, and with drones that
  overheat. None of these gave a measurable gain on winnable scenarios, so the final system doesn't use it.
- **Scaling (fig 6):** in 10 rooms, going from 1 to 3 drones cuts the time to find from 87 s to 40 s. More drones keep helping, with
  diminishing returns.
- **Real time (fig 7):** averaged over missions, the 95th-percentile decision time per 1 s tick is at most 195 ms (16 rooms,
  6 drones). The worst single mission's 95th percentile was 452 ms, still inside the 1 s budget.

| Figure | Shows |
|---|---|
| `figs/fig1_mission.png` | one mission: entry, find, rescue |
| `figs/fig2_fuzzy.png` | the fuzzy cost that was tested and rejected |
| `figs/fig3_bayes.png` | the Bayesian P(person) map draining as the drones search |
| `figs/fig4_ablation.png` | evidence for the design (table above) |
| `figs/fig5_ga.png` | GA convergence; GA vs greedy vs random allocation |
| `figs/fig6_scaling.png` | 1-6 drones in 6, 10 and 16 rooms |
| `figs/fig7_realtime.png` | decision time per tick against the 1 s budget |
| `figs/fig8_heuristic.png` | A\* nodes expanded vs Dijkstra |

## Presentation app

```
.venv\Scripts\python demo.py              (add --fullscreen for full screen)
```

- **Left:** *Reality* (smoke and fire, blocked doors, the trapped person) and *What the drones believe* (the probability map,
  and each drone's GA plan as outlined rooms with their visiting order, e.g. D2·1 = drone 2's first room).
- **Middle, success metrics:** live charts of search progress POS(t) and the person's health. At the end there is a **report card**
  (rescued alive or not, time to find and to rescue next to the best possible, efficiency). Also a **heuristics** panel (the A\*
  formula and its measured node saving, and the GA fitness) and an event feed.
- **Right:** scenario ◀ ▶ and Random, Rooms − + (4-20), Drones − + (1-8), Start/Pause, Step, Reset, Speed, view toggles, and
  **Run 20 missions**, a benchmark that reports success rate and mean efficiency and saves `results/benchmark.csv`.
- **Keys:** space start/pause · → step · r reset · n random scenario · f full screen.

## Run and test

```
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python selftest.py                 # 15 checks: A*, heuristic, Bayes, GA, oracle, episodes
.venv\Scripts\python tests_ui.py                 # clicks every control of the app headless, checks frame time
.venv\Scripts\python experiments.py --jobs 8     # results/*.csv (about 15 min on 8 cores)
.venv\Scripts\python figures.py                  # figs/*.png and results/tables.md
.venv\Scripts\python demo.py --scenario 3 --save figs/demo_seed3.gif   # record a mission
```

## Files

```
demo.py              presentation app (and GIF recorder)
navsim/scenario.py   winnable-scenario generation and the all-knowing reference rescuer
navsim/building.py   floor plan: corridor, N rooms, doors, furniture, one entrance/exit
navsim/world.py      ground truth: fire spread, smoke, debris, person health
navsim/sensor.py     line-of-sight sensor with noisy readings and smoke-dependent detection
navsim/belief.py     Bayesian search map, blockage filter, smoke estimate
navsim/ga.py         GA allocation (and the greedy/sweep baselines)
navsim/astar.py      A* with the octile heuristic, Dijkstra
navsim/mission.py    the real-time loop and the drone state machine
navsim/fuzzy.py      the fuzzy cost that was tested and rejected
navsim/config.py     all constants; METHODS = final system and the evaluation variants
```

## Notes on method and honesty

- Constants were tuned on scenarios #0-39 only; every reported number comes from #100 and up.
- The all-knowing rescuer is a fair best case: it may move diagonally and only needs to reach the person's side. The
  winnable check is conservative: it starts 120 s late and moves in 4 directions only.
- How the design got here (see the git history): three exits → one; a misleading 73% success rate (a quarter of the scenarios
  were impossible) → winnable scenarios; fuzzy cost included → tested and removed; a blockage belief that "forgot" blocked
  doors → the correct absorbing model; a slow, unresponsive app → step buttons and region blitting.
- Simplifications: 2-D single floor, 0.5 m grid, perfect communication between drones, a stationary person, and one smoke/heat
  value per cell.
