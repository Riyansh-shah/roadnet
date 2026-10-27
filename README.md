# Real-time search and rescue in a burning building: Fuzzy A*, Bayesian search and GA allocation

**Task.** K rescue units enter a single-floor building with N rooms through **one entrance**. One person is
somewhere inside at a random spot. The person is unconscious and doesn't move, but loses health in smoke. The units
must **find the person and guide them out** by the safest of three exits. Meanwhile the fire spreads, smoke diffuses,
and doors and corridor sections get blocked by debris. The units know the floor plan, but not the fire, the blockages or
where the person is. A short-range noisy sensor is all they have, and they decide every simulated second (1 tick = 1 s).

## Soft-computing methods and the job each one does

| Method | Where | Job |
|---|---|---|
| **Bayesian inference** | `navsim/belief.py` | A shared **P(person) map** over every free cell. Each look that finds nobody multiplies the cell's probability by (1 − p_detect), and p_detect falls with distance and smoke (Bayesian search theory), so areas missed in smoke keep part of their probability and get searched again. Also a **binary Bayes filter (log-odds) for P(blocked)** per cell, and a **1-D Kalman filter** for smoke per cell. |
| **Fuzzy A\*** | `navsim/fuzzy.py`, `navsim/astar.py` | A **Mamdani FIS** (4 inputs: hazard, P(blocked), uncertainty, resilience; 20 rules; centroid defuzzification) turns the belief into a per-cell traversal cost, and **A\*** plans on it every few ticks. Protected units tolerate smoke but avoid probably-blocked doors. The escorted civilian (resilience = 0.5 × health) gets much more smoke-averse routes. For real-time use the FIS is compiled into a 4-D lookup table (41×41×11×11) that is interpolated in under 1 ms. |
| **Genetic algorithm** | `navsim/ga.py` | Decides **which areas each unit searches and in what order** (a multiple-TSP). Chromosome = a permutation of areas plus K−1 delimiters. Fitness = **expected time to find the person**, Σ P(area) × arrival time, evaluated for the whole population at once with numpy. It uses order crossover, inversion and swap mutation, tournament selection and elitism, and re-runs (warm-started) when the belief changes. |

## Mission loop (`navsim/mission.py`)

Each tick:

1. The world evolves.
2. Every unit senses.
3. The shared belief is updated.
4. The fuzzy cost grid is rebuilt.
5. Areas are reallocated by the GA when needed: periodically, when a door's believed state flips, or when a unit runs out of areas.
6. Each unit picks a goal and moves one cell. If it runs into a blockage it didn't know about, it bumps and the belief is corrected.

On detection, the nearest unit walks to the person and leads them to an exit, planning with the person's own resilience. The other units leave.

## Ablation (leave one out)

| id | planner | belief | allocation |
|---|---|---|---|
| `FULL` | Fuzzy A\* | Bayes | GA |
| `noFuzzy` | plain A\* (distance) | Bayes | GA |
| `noBayes` | Fuzzy A\* | last reading only + coverage search | GA |
| `noGA` | Fuzzy A\* | Bayes | greedy P / cost |
| `B0` | plain A\* | last reading only + coverage search | nearest area first |

The coverage-search baseline is not a straw man. It walks to every cell of an area until 95% of the area has been
looked at, and it re-sweeps the whole building if the person is still missing. What it lacks is any memory that
a glance through smoke might have missed someone.

## Run it

```
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python selftest.py                     # 13 sanity checks
.venv\Scripts\python demo.py --seed 3                # live animation (space = pause)
.venv\Scripts\python demo.py --seed 3 --save figs/demo.gif
.venv\Scripts\python experiments.py --jobs 8         # E1-E3 -> results/*.csv (about 15 min on 8 cores)
.venv\Scripts\python figures.py                      # figs/*.png + results/tables.md
```

## Files

```
navsim/config.py     every constant; METHODS = the ablation table
navsim/building.py   floor plan generator: corridor, N rooms, doors, furniture, 3 exits, search areas
navsim/world.py      ground truth: fire spread, smoke diffusion, debris, person health
navsim/sensor.py     line-of-sight sensor: noisy blockage/smoke readings, distance- and smoke-dependent detection
navsim/belief.py     Bayesian belief (search map, log-odds, Kalman) and the raw-reading baseline
navsim/fuzzy.py      Mamdani FIS from scratch + compiled lookup table + cost grid
navsim/astar.py      8-connected A* (multi-goal) and Dijkstra
navsim/ga.py         GA allocation, greedy baselines, vectorised fitness
navsim/mission.py    the real-time loop and unit state machine
navsim/render.py     map drawing shared by the demo and the figures
```

## Results

RESULTS_PLACEHOLDER

## Calibration and honesty notes

- The fire, debris and health constants and the fuzzy rule base were tuned on **seeds 0-39**. All reported experiments
  use **seeds 100 and up**.
- The first fuzzy rule base made protected units avoid smoke as strongly as a civilian would. That made them detour
  around the corridor where the person was, and it lost rescues. The rules were redesigned so that units tolerate smoke and the
  civilian does not. This is documented in the git history.
- Some scenarios can't be won by any method: the person is cut off by fire on every route. They count as failures
  for every method alike.
- Simplifications: a 2-D single floor, a 0.5 m grid, perfect communication between units, a stationary person, and a
  smoke/heat field reduced to one hazard value per cell.
