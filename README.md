# Real-time search and rescue in a burning building: Fuzzy A*, Bayesian search and GA allocation

**Task.** K rescue units enter a single-floor building with N rooms through **one entrance**. One person is
somewhere inside at a random spot. The person is unconscious and doesn't move, but loses health in smoke. The units
must **find the person and guide them out** by the safest of three exits. Meanwhile the fire spreads, smoke diffuses,
and doors and corridor sections get blocked by debris. The units know the floor plan, but not the fire, the blockages or
where the person is. A short-range noisy sensor is all they have, and they decide every simulated second (1 tick = 1 s).

## Soft-computing methods and the job each one does

| Method | Where | Job |
|---|---|---|
| **Bayesian inference** | `navsim/belief.py` | A shared **P(person) map** over every free cell. Each look that finds nobody multiplies the cell's probability by (1 − p_detect), and p_detect falls with distance and smoke (Bayesian search theory), so areas missed in smoke keep part of their probability and get searched again. Also a **binary Bayes filter (log-odds) for P(blocked)** per cell. Its predict step is the world's transition model: a door can become blocked, more likely in smoke, but never clears. Bumping into a blockage is a certain observation, so it stays at P = 1. And a **1-D Kalman filter** for smoke per cell, whose variance grows while a cell goes unseen. |
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

Full tables are in `results/tables.md`, figures in `figs/`. Test seeds 100-199, 10 rooms, 3 units, 95% bootstrap CIs.

| | rescued alive | time to find* | time to rescue* | civilian smoke dose during escort* |
|---|---|---|---|---|
| **FULL** (Fuzzy A\* + Bayes + GA) | 80% | 41 s | 55 s | 0.077 |
| without fuzzy | 83% | 41 s | 55 s | 0.133 |
| without Bayes | 75% | 62 s | 104 s | 0.589 |
| without GA | 80% | 48 s | 62 s | 0.069 |
| B0 (none of the three) | 73% | 62 s | 102 s | 0.548 |

\* mean over the 67 seeds that every method rescued (paired comparison).

What the data supports:
- **Bayes is the method that matters most.** Removing it makes the team find the person **22 s later
  (95% CI 10-36 s)**, and the escort smoke dose is about 8x higher. A single glance through smoke can miss the person, and only
  the Bayesian map keeps that possibility alive and sends units back.
- **The GA optimises its objective clearly.** On the allocation problems the team faces at t = 30 s, it beats greedy
  allocation in **95-96% of cases, by 11-14%** in expected find time (random allocation is +160-180% worse). It converges in
  about 40 generations and takes 250-400 ms. In whole missions this gives **4 s faster finds, but the CI (-1.6 to 10.6 s)
  includes zero**, so the end-to-end gain isn't significant at 100 seeds.
- **Fuzzy A\*'s measurable effect is on the escort.** The civilian breathes **42% less smoke** while being led out
  (0.077 vs 0.133; the paired CI just touches zero). It doesn't change how fast the person is found or how many survive
  (1 vs 4 discordant seeds, McNemar p = 0.38).
- **Scaling (fig 6):** going from 1 to 2-3 units roughly halves the time to find. Beyond that the returns diminish.
- **Real time (fig 7):** the 95th-percentile decision time per tick stays under 400 ms even with 16 rooms and 6 units,
  well inside the 1 s tick. The mean is 10-50 ms.
- About 18% of scenarios are lost by every method. In those, the person is cut off or overcome by fire before anyone can reach them.

| Figure | Shows |
|---|---|
| `figs/fig1_mission.png` | one mission: entry, find, rescue (seed picked by a fixed rule: the first test seed found after >= 30 s) |
| `figs/fig2_fuzzy.png` | membership functions and the penalty surfaces for a unit vs the civilian |
| `figs/fig3_bayes.png` | the Bayesian P(person) map draining as the units search |
| `figs/fig4_ablation.png` | leave-one-out ablation |
| `figs/fig5_ga.png` | GA convergence and GA vs greedy vs random |
| `figs/fig6_scaling.png` | 1-6 units in 6, 10 and 16 rooms |
| `figs/fig7_realtime.png` | decision time per tick against the 1 s budget |
| `figs/demo_seed3.gif` | the live demo, recorded |

## Calibration and honesty notes

- The fire, debris and health constants and the fuzzy rule base were tuned on **seeds 0-39**. All reported experiments
  use **seeds 100 and up**.
- The first fuzzy rule base made protected units avoid smoke as strongly as a civilian would. That made them detour
  around the corridor where the person was, and it lost rescues. The rules were redesigned so that units tolerate smoke and the
  civilian does not. This is documented in the git history.
- A first version let the blockage belief drift back to its prior. Units then forgot doors they knew were blocked and
  walked into them about 11 times per mission. It was replaced by the correct absorbing transition model; all
  results below come from the corrected version.
- Some scenarios can't be won by any method: the person is cut off by fire on every route. They count as failures
  for every method alike.
- Simplifications: a 2-D single floor, a 0.5 m grid, perfect communication between units, a stationary person, and a
  smoke/heat field reduced to one hazard value per cell.
