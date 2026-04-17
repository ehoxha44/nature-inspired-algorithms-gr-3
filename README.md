# Smart TV Schedule Optimizer

A **Genetic Algorithm** solver for the Smart TV Scheduling combinatorial optimisation problem.
Enhanced with a **Dynamic Programming seed**, four fine-grained mutation operators,
stagnation-adaptive mutation, and a post-GA local search refinement.

Developed as part of the Nature-Inspired Algorithms course — Group 3.

## Project Structure

```
tv_scheduler/          Python package (solver)
  models.py            Data classes (Program, PriorityBlock, PreferenceInterval, ProblemInstance)
  decoder.py           Permutation chromosome → valid schedule (O(n log n))
  fitness.py           Scoring: base scores + bonuses − switch penalties
  ga.py                Genetic Algorithm engine (OX1 crossover, 4 mutation ops, elitism, stagnation)
  main.py              CLI entry-point, DP seed, local search, I/O helpers

data/
  input/               Problem instances (JSON) – 16 instances
  output/              Best schedules found (JSON)

```

## Algorithm Overview

| Component | Approach |
|---|---|
| Encoding | Integer permutation (priority-order) |
| Initial seed | **Weighted interval scheduling DP** (O(n log n)) — near-optimal relaxed solution |
| Crossover | Order Crossover (OX1) — preserves relative ordering |
| Mutation (4 ops) | Adjacent swap · Short inversion · Or-opt-1 · Full swap — all fine-grained |
| Selection | Tournament (k=5) |
| Elitism | Top 10 chromosomes preserved each generation |
| Stagnation | Mutation rate ×3 after 30 stagnant generations |
| Post-GA | Randomised local search (adjacent swap, or-opt-1) |

## Quick Start

```bash
# Single instance
python -m tv_scheduler.main data/input/kosovo.json

# Override parameters
python -m tv_scheduler.main data/input/kosovo.json \
    --generations 400 --population 150 --mutation-rate 0.25

# Batch-run all inputs
python -m tv_scheduler.main --all data/input/ --generations 300
```

## Results (default parameters: pop=100, gen=200, seed=42)

Output files are saved as JSON in `data/output/`.

| Instance | Score | Programs | Time |
|---|---|---|---|
| youtube_premium | 53 998 | 644 | 165 s |
| youtube_gold | 33 156 | 587 | 134 s |
| uk | 11 317 | 140 | 61 s |
| france | 10 831 | 177 | 35 s |
| singapore | 6 494 | 99 | 22 s |
| spain | 6 439 | 90 | 23 s |
| australia | 5 469 | 56 | 39 s |
| usa | 5 907 | 77 | 612 s |
| canada | 5 369 | 67 | 58 s |
| usa_synthetic | 3 318 | 33 | 69 s |
| uk_synthetic | 2 266 | 26 | 7 s |
| china | 2 405 | 35 | 116 s |
| netherlands | 1 894 | 25 | 1 s |
| kosovo | 1 814 | 23 | 1 s |
| croatia | 2 208 | 25 | 1 s |
| germany | 967 | 12 | 0.5 s |

## GA Parameters

| Flag | Default | Description |
|---|---|---|
| `--generations` | 200 | GA generations |
| `--population` | 100 | Population size |
| `--mutation-rate` | 0.20 | Base mutation probability |
| `--crossover-rate` | 0.85 | Crossover probability |
| `--tournament-size` | 5 | Tournament size |
| `--elitism` | 10 | Elite chromosomes per generation |
| `--stagnation` | 30 | Stagnant generations before mutation boost |
| `--seed` | 42 | RNG seed (-1 = non-deterministic) |
