# Drone search and rescue in a burning building: Bayesian search, GA allocation and A*

![demo](figs/demo_seed3.gif)

## The problem

K drones enter a single-floor building with N rooms through **one door, which is also the only way out**. One person is
somewhere inside, unconscious, at a random spot, and loses health in smoke. The drones must **find the person and lead
them back out through that door**. A fire starts in one room as a **2 × 2 block of cells (4 cells)** just as the drones
arrive, then spreads cell by cell while smoke diffuses through the building. Burning cells cannot be entered. The drones
know the floor plan, but not where the fire is or where the person is. All they have is a short-range noisy sensor, and
they make a decision every simulated second.

**Every scenario is winnable.** `navsim/scenario.py` keeps a scenario only if an all-knowing rescuer could still save the person
even when it **starts 120 s late**. That rescuer knows the fire and where the person is. In an earlier version (a bigger
fire plus falling debris), a quarter of random scenarios were impossible for anyone. The check makes every test fair.

## The method: three parts, each kept because it measurably helps

| Part | Question it answers | How |
|---|---|---|
| **Bayesian search map** (`navsim/belief.py`) | *Where is the person probably?* | P(person) over every free cell. A look that finds nobody multiplies a cell's probability by (1 − p_detect), and p_detect falls with distance and smoke, so rooms glanced at through smoke keep part of their probability and are searched again. The same module keeps a Kalman estimate of the smoke in each cell. A Bayes filter for blocked doors also exists, but it is only active if falling debris is switched on (`Config.debris`, off by default). |
| **Genetic algorithm** (`navsim/ga.py`) | *Which drone searches which rooms, in what order?* | Chromosome = permutation of rooms + K−1 separators. Fitness = **expected time to find the person = Σ P(room) × arrival time at the room**. Uses order crossover, inversion and swap mutation, tournament selection and elitism, vectorised over the population, and re-runs when the beliefs change. |
| **A\*** (`navsim/astar.py`) | *How does each drone get there, and how does it lead the person out?* | 8-connected grid, cells believed to be on fire are avoided, with the heuristic below. |

## Heuristics

### A\* and the octile distance

The building is a grid of 0.5 m cells. A drone can move in **8 directions**: a straight step (up, down, left, right) costs 1
and a diagonal step costs √2 ≈ 1.414. A\* always expands the cell with the smallest

    f(n) = g(n) + h(n)      g(n) = distance flown from the start to n,   h(n) = estimated distance from n to the goal

and our estimate `h(n)` is the **octile distance**:

    h(n) = max(dx, dy) + (√2 − 1) · min(dx, dy)        dx, dy = how many cells apart n and the goal are, across and down

**What it means:** it is the length of the shortest route on an *open* grid when you may move straight and diagonally. You
take as many diagonal steps as possible (each covers one cell across *and* one cell down), then finish with straight steps.

**Example:** the goal is 5 cells across and 3 down (dx = 5, dy = 3). Take 3 diagonal steps (3 × 1.414 = 4.24) and then 2
straight steps (2), so h = 6.24 cells = 3.1 m. For comparison, the straight-line distance is 5.83 (too optimistic, because a
drone can't fly at that angle on the grid) and the 4-direction "Manhattan" distance is 8 (too high, because it forbids
diagonals). Octile is the exact answer on an empty grid. The app shows this as a diagram (ⓘ How it works).

**Why it's correct:** walls and fire can only make the real route longer than the open-grid route, so `h` never
overestimates the true remaining distance (it is **admissible**). It also never drops by more than the cost of one step
(it is **consistent**), so the first path A\* finds to the goal is a shortest one. **Measured: A\* expands about 70% fewer
cells than Dijkstra** (no heuristic, h = 0) on the same planning queries (fig 8).

### Search heuristic (the GA's fitness)

Expected time to find the person = Σ P(room) × the time the assigned drone reaches that room. Minimising it sends drones
first to rooms that are both likely and close.

## Methodology, step by step

1. **Model:** a random building (corridor, N rooms, furniture, one entrance/exit), a fire starting as 4 cells, smoke diffusion,
   and a person at a random spot. Every scenario is checked to be winnable.
2. **Sense:** each drone sees cells within 2 m in its line of sight. Detection of the person is less likely far away and in smoke.
3. **Believe (Bayes):** update the shared P(person) map after every look, and estimate the smoke.
4. **Allocate (GA):** split the rooms between the drones and order them to minimise the expected time to find. Re-plan when the
   map changes or a drone runs out of rooms.
5. **Move (A\*):** every drone follows an A\* path to its next room, re-planning every few seconds or when its path is no longer valid.
6. **Rescue:** the nearest drone flies to the person and leads them out along an A\* path to the entrance.
7. **Evaluate:** run each scenario with the final system and with simpler versions (without Bayes, without GA, with a fuzzy
   cost added), on scenarios that were not used for tuning. Compare success and efficiency with paired confidence intervals.

## Metrics

| Metric | Definition |
|---|---|
| **Success** (primary) | The person reaches the exit alive. Over many missions: success rate. |
| **Efficiency** | Best possible rescue time ÷ our rescue time (0 if the person is lost). The best possible time is that of the all-knowing rescuer starting at t = 0, so 100% is unbeatable. |
| **POS(t)** | Bayesian probability that the search so far would have found the person. It is shown live in the app. |
| Node saving | A\* nodes expanded vs Dijkstra on the same queries. |

## Results

Test scenarios #100-199 (calibration used #0-39). Full tables are in `results/tables.md`.

**Final system, 10 rooms, 3 drones:** **100% success (100/100)**, **efficiency 68% [65, 71]**. The person is found in 42.3 s
(the best possible is 22.6 s) and is outside in 68.1 s (the best possible is 45.1 s).

**What each part contributes** (same scenarios, paired, 95% CI; a CI that excludes 0 is a real effect):

| part | evidence | result |
|---|---|---|
| **Bayesian search map** | Final system vs a coverage sweep that looks everywhere once, as one look gets less reliable (fig 9) | When one look spots the person up close only half the time, **Bayes rescues 99/100 and the sweep 89/100** (McNemar p < 0.01). At 0.7 it is 100 vs 95 (p = 0.06); at our default 0.95, 100 vs 98 (no real difference). The sweep never comes back to a room it has "seen"; the Bayesian map keeps the leftover probability and sends a drone back. |
| **GA allocation** | GA vs greedy allocation (each free drone takes the best P/distance room) | Over all 750 scaling missions: **+1.7 efficiency points [+0.6, +2.9]**. In the 10-room test it finds the person **3.2 s sooner [0.3, 6.5]**. On the allocation problems themselves it beats greedy in **95-100%** of cases, by 9-10%. |
| **A\* heuristic** | A\* with the octile heuristic vs Dijkstra on the same queries (fig 8) | **74-78% fewer cells expanded**, with the same optimal paths |
| *Fuzzy cost (rejected)* | Final system vs final system + fuzzy cell cost | Efficiency −1.6 points [−5.2, 1.5]; no measurable gain in any test (also under heavy smoke, as a "leave the room" rule, and with drones that overheat), so it is left out |

The full comparison table (with time to find, time to rescue and escort smoke for every version) is in `results/tables.md` and
fig 4. With a **bigger fire** (the same 4-cell start, but burning for 150 s before the drones arrive), all versions stay within
a few points of each other on efficiency. The variants without the Bayesian map lose a few more people (95-96% vs 99%), but at
100 scenarios that difference isn't significant.

- **Scaling (fig 6):** in 10 rooms, going from 1 to 3 drones cuts the time to find from 99 s to 45 s. More drones keep helping, with
  diminishing returns.
- **Real time (fig 7):** a typical decision tick takes 4-12 ms. A tick with a GA re-plan takes 73-184 ms, well inside the 1 s tick.

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
| `figs/fig9_detection.png` | Bayesian map vs coverage sweep as detection gets less reliable |

## Presentation app

```
.venv\Scripts\python demo.py              (add --fullscreen for full screen)
```

- **Left:** *Reality* (fire and smoke, the drones, the trapped person) and *What the drones believe* (the probability map,
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
navsim/world.py      ground truth: fire spread from a 4-cell start, smoke, person health (debris optional, off)
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
  doors → the correct absorbing model; a slow, unresponsive app → step buttons and region blitting; a fire with a 150 s head
  start plus falling debris → a fire that starts as 4 cells, with no debris.
- With the debris gone, the Bayesian map no longer makes the search *faster*. Its earlier speed advantage came largely
  from coping with noisy "is this door blocked?" readings. What it still adds is *reliability* when a single look can miss
  the person (fig 9). We report this as it is.
- Simplifications: 2-D single floor, 0.5 m grid, perfect communication between drones, a stationary person, and one smoke/heat
  value per cell.
