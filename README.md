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
| Post-GA | **Guided local search** (attractiveness-biased moves) + optional classic LS (`--classic-ls`) |
| Time / optimality | Optional `--max-time-sec` hard cap; early stop when score reaches a **DP relaxation upper bound** (provable for this decoder objective) |

## Quick Start

```bash
# Single instance
python -m tv_scheduler.main data/input/kosovo.json

# Override parameters
python -m tv_scheduler.main data/input/kosovo.json \
    --generations 400 --population 150 --mutation-rate 0.25

# Batch-run all inputs
python -m tv_scheduler.main --all data/input/ --generations 300

# Cap wall time at 5 minutes per instance (GA + guided LS)
python -m tv_scheduler.main data/input/france.json --max-time-sec 300
```

## Per-instance parameter tuning (Optuna)

Requires `optuna` (`pip install -r requirements.txt`).

For **each** JSON instance in `data/input/`, `scripts/optuna_per_instance_tuning.py`:

1. Runs an **Optuna** study (default **6** trials in **fast** mode, **24** with `--no-fast`). Each trial is one GA + guided LS solve with a wall-time cap (default **40 s** fast, **300 s** full).
2. Writes **`data/output/parameter-tuning-configs/<instance>.json`** with **only** `instance` and `adjustments`. The `adjustments` object holds the chosen GA hyperparameters (as returned by Optuna) plus **`best_score`** from the study — no `trials`, no manifest, no Optuna dump.
3. Runs exactly **`--runs`** solves (default **10**) per instance using those best hyperparameters. Each **`data/output/parameter-tuning/<instance>/run_XX.json`** must contain **exactly** these keys (validator / same as `data/output/spain.json`): `instance`, `total_score`, `programs_scheduled`, `elapsed_seconds`, `schedule`, `convergence`. Any other top-level JSON shape for runs is invalid.

   `data/output/parameter-tuning/<instance>/run_00.json` … `run_09.json`

```bash
# Fast defaults: --fast, 6 Optuna trials, 40s cap, 10 runs/instance
python scripts/optuna_per_instance_tuning.py --input-dir data/input --output-dir data/output

# Heavier search
python scripts/optuna_per_instance_tuning.py --no-fast --optuna-trials 24 --time-limit-sec 300
```

Legacy multi-instance preset study: `scripts/parameter_study.py`.

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
| `--max-time-sec` | *(none)* | Hard wall time (seconds) for GA + local search |
| `--classic-ls` | off | Use unguided randomised LS instead of guided LS |
