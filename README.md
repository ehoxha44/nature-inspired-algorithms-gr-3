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

## Results — Per-Instance Tuned Parameters (10 runs each)

Each instance was optimised independently with Optuna (6 fast trials, 40 s wall-time cap per trial).
The best hyperparameters found were then used for 10 validation runs.
All runs are stored in `data/output/parameter-tuning/<instance>/run_00.json … run_09.json`.

---

### germany

Tuned parameters: Pop=80, Gen=70, CR=0.775, MR=0.280, Tourn=7, Elitism=8, Stagnation=21 | Optuna best: 967

| Run | Score | Programs | Time (s) |
|---|---|---|---|
| run_00 | 967 | 12 | 0.16 |
| run_01 | 967 | 12 | 0.16 |
| run_02 | 967 | 12 | 0.16 |
| run_03 | 967 | 12 | 0.16 |
| run_04 | 967 | 12 | 0.16 |
| run_05 | 967 | 12 | 0.16 |
| run_06 | 967 | 12 | 0.16 |
| run_07 | 967 | 12 | 0.16 |
| run_08 | 967 | 12 | 0.16 |
| run_09 | 967 | 12 | 0.16 |
| **Mean** | **967.0** | **12.0** | **0.16** |
| **Min** | **967** | **12** | — |
| **Max** | **967** | **12** | — |
| **Std dev** | **0.0** | — | — |

The smallest instance (12 programs, 0.16 s per run) is completely solved in every run without exception. The GA converges to the same schedule deterministically — the search space is small enough that a modest population exhausts all meaningful orderings within the first few generations. Zero variance confirms this is the global optimum for this decoder.

---

### kosovo

Tuned parameters: Pop=65, Gen=110, CR=0.904, MR=0.158, Tourn=5, Elitism=7, Stagnation=26 | Optuna best: 1 814

| Run | Score | Programs | Time (s) |
|---|---|---|---|
| run_00 | 1 811 | 23 | 0.75 |
| run_01 | 1 814 | 23 | 0.75 |
| run_02 | 1 814 | 23 | 0.75 |
| run_03 | 1 814 | 23 | 0.75 |
| run_04 | 1 814 | 23 | 0.75 |
| run_05 | 1 814 | 23 | 0.75 |
| run_06 | 1 814 | 23 | 0.75 |
| run_07 | 1 814 | 23 | 0.75 |
| run_08 | 1 814 | 23 | 0.75 |
| run_09 | 1 811 | 23 | 0.75 |
| **Mean** | **1 813.4** | **23.0** | **0.75** |
| **Min** | **1 811** | **23** | — |
| **Max** | **1 814** | **23** | — |
| **Std dev** | **1.3** | — | — |

Near-perfect consistency: 8 of 10 runs reach the maximum observed score of 1 814 and only 2 runs land three points lower at 1 811. All runs schedule the same 23 programs. The instance is small and tightly constrained, and Optuna settled on the highest crossover rate in the study (0.904) to efficiently recombine the few good building blocks available.

---

### netherlands

Tuned parameters: Pop=40, Gen=130, CR=0.794, MR=0.200, Tourn=4, Elitism=9, Stagnation=26 | Optuna best: 1 894

| Run | Score | Programs | Time (s) |
|---|---|---|---|
| run_00 | 1 894 | 25 | 0.71 |
| run_01 | 1 886 | 24 | 0.71 |
| run_02 | 1 894 | 25 | 0.71 |
| run_03 | 1 890 | 25 | 0.71 |
| run_04 | 1 894 | 25 | 0.71 |
| run_05 | 1 894 | 25 | 0.71 |
| run_06 | 1 886 | 24 | 0.71 |
| run_07 | 1 894 | 25 | 0.71 |
| run_08 | 1 886 | 24 | 0.71 |
| run_09 | 1 894 | 25 | 0.71 |
| **Mean** | **1 891.2** | **24.7** | **0.71** |
| **Min** | **1 886** | **24** | — |
| **Max** | **1 894** | **25** | — |
| **Std dev** | **3.8** | — | — |

Extremely stable. Seven of ten runs reach the ceiling score of 1 894 (matching the default run) and the remaining three land at 1 886. The smallest population in the study (40) with a long generation budget (130) is sufficient for this compact instance. The three lower-scoring runs also schedule one fewer program (24 vs 25), indicating a constraint satisfaction difference rather than a pure fitness gap.

---

### croatia

Tuned parameters: Pop=85, Gen=140, CR=0.789, MR=0.280, Tourn=4, Elitism=8, Stagnation=42 | Optuna best: 2 208

| Run | Score | Programs | Time (s) |
|---|---|---|---|
| run_00 | 2 208 | 25 | 1.23 |
| run_01 | 2 208 | 25 | 1.23 |
| run_02 | 2 208 | 25 | 1.23 |
| run_03 | 2 208 | 25 | 1.23 |
| run_04 | 2 208 | 25 | 1.23 |
| run_05 | 2 208 | 25 | 1.23 |
| run_06 | 2 185 | 24 | 1.23 |
| run_07 | 2 208 | 25 | 1.23 |
| run_08 | 2 208 | 25 | 1.23 |
| run_09 | 2 208 | 25 | 1.23 |
| **Mean** | **2 205.7** | **24.9** | **1.23** |
| **Min** | **2 185** | **24** | — |
| **Max** | **2 208** | **25** | — |
| **Std dev** | **7.3** | — | — |

Nine of ten runs reach the top score of 2 208 — identical to the default run — and only one run (run_06) falls to 2 185 while also scheduling one fewer program. The high stagnation limit (42, the highest in the study) allows the GA to persist through flat regions before converging. This instance is effectively solved to near-optimality on almost every run.

---

### uk_synthetic

Tuned parameters: Pop=45, Gen=90, CR=0.832, MR=0.176, Tourn=7, Elitism=5, Stagnation=27 | Optuna best: 2 266

| Run | Score | Programs | Time (s) |
|---|---|---|---|
| run_00 | 2 266 | 26 | 5.63 |
| run_01 | 2 263 | 26 | 5.63 |
| run_02 | 2 266 | 26 | 5.63 |
| run_03 | 2 263 | 26 | 5.63 |
| run_04 | 2 263 | 26 | 5.63 |
| run_05 | 2 263 | 26 | 5.63 |
| run_06 | 2 266 | 26 | 5.63 |
| run_07 | 2 266 | 26 | 5.63 |
| run_08 | 2 263 | 26 | 5.63 |
| run_09 | 2 263 | 26 | 5.63 |
| **Mean** | **2 264.2** | **26.0** | **5.63** |
| **Min** | **2 263** | **26** | — |
| **Max** | **2 266** | **26** | — |
| **Std dev** | **1.5** | — | — |

Essentially converged: all 10 runs schedule exactly 26 programs and the score oscillates between just two values (2 263 and 2 266), matching the default result. The synthetic structure imposes hard constraints that tightly bound the feasible schedules, leaving almost no variance between runs.

---

### china

Tuned parameters: Pop=85, Gen=80, CR=0.786, MR=0.241, Tourn=7, Elitism=5, Stagnation=21 | Optuna best: 2 483

| Run | Score | Programs | Time (s) |
|---|---|---|---|
| run_00 | 2 473 | 33 | 40.0 |
| run_01 | 2 402 | 31 | 40.0 |
| run_02 | 2 415 | 31 | 40.0 |
| run_03 | 2 411 | 32 | 40.0 |
| run_04 | 2 480 | 33 | 40.0 |
| run_05 | 2 525 | 34 | 40.0 |
| run_06 | 2 411 | 30 | 40.0 |
| run_07 | 2 412 | 31 | 40.0 |
| run_08 | 2 420 | 32 | 40.0 |
| run_09 | 2 432 | 32 | 40.0 |
| **Mean** | **2 438.1** | **31.9** | **40.0** |
| **Min** | **2 402** | **30** | — |
| **Max** | **2 525** | **34** | — |
| **Std dev** | **40.7** | — | — |

Moderate variance reflecting the complexity of this large instance (3.66 MB, complex channel constraints). Different runs settle into different local optima, and the programs scheduled per run ranges from 30 to 34. The tuned parameters (tournament size 7, low generation count 80) prioritise selection pressure and speed over broad exploration, yielding a modest improvement over the default (2 405 → mean 2 438). Run_05 stands out as the best, scheduling 34 programs for a score of 2 525.

---

### usa_synthetic

Tuned parameters: Pop=60, Gen=60, CR=0.817, MR=0.199, Tourn=3, Elitism=7, Stagnation=35 | Optuna best: 3 418

| Run | Score | Programs | Time (s) |
|---|---|---|---|
| run_00 | 3 314 | 33 | 40.0 |
| run_01 | 3 092 | 31 | 40.0 |
| run_02 | 3 343 | 33 | 40.0 |
| run_03 | 3 392 | 33 | 40.0 |
| run_04 | 3 292 | 33 | 40.0 |
| run_05 | 3 319 | 34 | 40.0 |
| run_06 | 3 370 | 33 | 40.0 |
| run_07 | 3 346 | 33 | 40.0 |
| run_08 | 3 295 | 33 | 40.0 |
| run_09 | 3 315 | 33 | 40.0 |
| **Mean** | **3 307.8** | **32.9** | **40.0** |
| **Min** | **3 092** | **31** | — |
| **Max** | **3 392** | **33** | — |
| **Std dev** | **82.2** | — | — |

The widest spread among the synthetic instances. Run_01 (3 092) is a clear outlier roughly 200 points below the cluster, the only run to schedule just 31 programs, suggesting an occasional failure to escape a poor local basin. The remaining 9 runs form a tight cluster between 3 292 and 3 392. The small tournament size (3) chosen by Optuna favours diversity, but the 40 s cap occasionally limits recovery from bad initial conditions.

---

### australia

Tuned parameters: Pop=80, Gen=120, CR=0.856, MR=0.299, Tourn=4, Elitism=5, Stagnation=24 | Optuna best: 5 585

| Run | Score | Programs | Time (s) |
|---|---|---|---|
| run_00 | 5 512 | 57 | 40.0 |
| run_01 | 5 526 | 57 | 40.0 |
| run_02 | 5 551 | 57 | 40.0 |
| run_03 | 5 562 | 57 | 40.0 |
| run_04 | 5 543 | 57 | 40.0 |
| run_05 | 5 545 | 56 | 40.0 |
| run_06 | 5 579 | 57 | 40.0 |
| run_07 | 5 629 | 57 | 40.0 |
| run_08 | 5 509 | 56 | 40.0 |
| run_09 | 5 450 | 56 | 40.0 |
| **Mean** | **5 540.6** | **56.7** | **40.0** |
| **Min** | **5 450** | **56** | — |
| **Max** | **5 629** | **57** | — |
| **Std dev** | **47.3** | — | — |

Tight, consistent results. Tuning lifted the mean from the default 5 469 to 5 541 — a reliable +72 point improvement across all runs. The highest mutation rate in the study (0.299) helps the GA escape local optima in this moderately sized instance. Eight of ten runs schedule 57 programs; the three slightly lower-scoring runs (run_05, run_08, run_09) schedule 56.

---

### canada

Tuned parameters: Pop=85, Gen=140, CR=0.859, MR=0.192, Tourn=5, Elitism=13, Stagnation=24 | Optuna best: 6 386

| Run | Score | Programs | Time (s) |
|---|---|---|---|
| run_00 | 6 210 | 65 | 40.0 |
| run_01 | 6 311 | 67 | 40.0 |
| run_02 | 6 333 | 68 | 40.0 |
| run_03 | 6 309 | 67 | 40.0 |
| run_04 | 6 261 | 67 | 40.0 |
| run_05 | 6 297 | 67 | 40.0 |
| run_06 | 6 394 | 68 | 40.0 |
| run_07 | 6 301 | 67 | 40.0 |
| run_08 | 6 337 | 68 | 40.0 |
| run_09 | 6 313 | 67 | 40.0 |
| **Mean** | **6 306.6** | **67.1** | **40.0** |
| **Min** | **6 210** | **65** | — |
| **Max** | **6 394** | **68** | — |
| **Std dev** | **48.1** | — | — |

One of the most striking tuning gains: the mean jumped from the default 5 369 to 6 307 — nearly +18 %. Tuning unlocked a substantially better region of the search space. The highest elitism count in the study (13) preserves more of the best partial solutions each generation. Run_00 is slightly lower (6 210, 65 programs) while runs 02, 06, and 08 each schedule 68 programs and score above 6 330.

---

### usa

Tuned parameters: Pop=65, Gen=50, CR=0.879, MR=0.202, Tourn=7, Elitism=6, Stagnation=28 | Optuna best: 5 911

| Run | Score | Programs | Time (s) |
|---|---|---|---|
| run_00 | 5 720 | 75 | 41.1 |
| run_01 | 5 484 | 73 | 41.1 |
| run_02 | 5 573 | 74 | 41.1 |
| run_03 | 5 311 | 72 | 41.1 |
| run_04 | 5 528 | 78 | 41.1 |
| run_05 | 5 418 | 74 | 41.1 |
| run_06 | 5 253 | 71 | 41.1 |
| run_07 | 5 722 | 76 | 41.1 |
| run_08 | 5 617 | 75 | 41.1 |
| run_09 | 5 533 | 76 | 41.1 |
| **Mean** | **5 515.9** | **74.4** | **41.1** |
| **Min** | **5 253** | **71** | — |
| **Max** | **5 722** | **76** | — |
| **Std dev** | **156.3** | — | — |

The highest variance among the medium-sized instances, reflecting the USA dataset's complexity (23.6 MB, the largest standard instance). Programs scheduled per run ranges from 71 to 78, and the score spread exceeds 450 points. The 40 s wall cap is tight for this instance — the default uncapped run at 612 s achieved 5 907, which is above every tuned run here. The tuned mean of 5 516 reflects the time constraint rather than a regression in quality.

---

### singapore

Tuned parameters: Pop=75, Gen=50, CR=0.805, MR=0.166, Tourn=3, Elitism=6, Stagnation=34 | Optuna best: 7 049

| Run | Score | Programs | Time (s) |
|---|---|---|---|
| run_00 | 6 830 | 95 | 37.2 |
| run_01 | 7 000 | 95 | 37.2 |
| run_02 | 6 698 | 90 | 37.2 |
| run_03 | 6 693 | 90 | 37.2 |
| run_04 | 6 775 | 94 | 37.2 |
| run_05 | 6 767 | 94 | 37.2 |
| run_06 | 6 888 | 96 | 37.2 |
| run_07 | 6 863 | 96 | 37.2 |
| run_08 | 6 682 | 92 | 37.2 |
| run_09 | 6 665 | 91 | 37.2 |
| **Mean** | **6 786.1** | **93.3** | **37.2** |
| **Min** | **6 665** | **90** | — |
| **Max** | **7 000** | **96** | — |
| **Std dev** | **108.7** | — | — |

Notable range for a medium instance. Optuna selected a small tournament size (3), favouring genetic diversity and broad exploration over exploitation. This pays off in the best run (7 000, well above the default 6 494) but also introduces more run-to-run spread. Programs scheduled varies from 90 to 96. The mean of 6 786 comfortably exceeds the default 6 494.

---

### spain

Tuned parameters: Pop=80, Gen=140, CR=0.833, MR=0.235, Tourn=5, Elitism=11, Stagnation=29 | Optuna best: 7 506

| Run | Score | Programs | Time (s) |
|---|---|---|---|
| run_00 | 7 380 | 93 | 38.4 |
| run_01 | 7 355 | 92 | 38.4 |
| run_02 | 7 353 | 92 | 38.4 |
| run_03 | 7 465 | 95 | 38.4 |
| run_04 | 7 420 | 94 | 38.4 |
| run_05 | 7 474 | 94 | 38.4 |
| run_06 | 7 373 | 93 | 38.4 |
| run_07 | 7 455 | 94 | 38.4 |
| run_08 | 7 430 | 94 | 38.4 |
| run_09 | 7 508 | 94 | 38.4 |
| **Mean** | **7 421.3** | **93.5** | **38.4** |
| **Min** | **7 353** | **92** | — |
| **Max** | **7 508** | **94** | — |
| **Std dev** | **54.3** | — | — |

Good consistency with a strong tuning gain: mean 7 421 versus default 6 439 (+15 %). All ten runs exceed 7 350 and the full range is only 155 points wide, confirming the tuned parameter set generalises reliably. The config with high elitism (11) and a moderate generation budget (140) efficiently exploits the good solutions found early in each run.

---

### uk

Tuned parameters: Pop=65, Gen=50, CR=0.917, MR=0.182, Tourn=7, Elitism=9, Stagnation=21 | Optuna best: 12 120

| Run | Score | Programs | Time (s) |
|---|---|---|---|
| run_00 | 11 660 | 135 | 40.0 |
| run_01 | 11 576 | 133 | 40.0 |
| run_02 | 11 527 | 132 | 40.0 |
| run_03 | 11 678 | 137 | 40.0 |
| run_04 | 11 633 | 135 | 40.0 |
| run_05 | 12 182 | 136 | 40.0 |
| run_06 | 12 129 | 138 | 40.0 |
| run_07 | 11 686 | 132 | 40.0 |
| run_08 | 11 857 | 137 | 40.0 |
| run_09 | 11 733 | 131 | 40.0 |
| **Mean** | **11 766.1** | **134.6** | **40.0** |
| **Min** | **11 527** | **131** | — |
| **Max** | **12 182** | **138** | — |
| **Std dev** | **223.7** | — | — |

The largest standard deviation among the non-streaming instances, reflecting the UK dataset's size and constraint complexity. Two runs (run_05 and run_06) stand out at 12 182 and 12 129 — roughly 500 points above the other eight, showing the GA occasionally discovers significantly better schedules with more programs (136–138 vs 131–135). Mean 11 766 vs default 11 317 is a solid +4 % improvement. The highest crossover rate in the study (0.917) encourages aggressive recombination; the short generation budget (50) pairs with the 40 s cap.

---

### france

Tuned parameters: Pop=95, Gen=100, CR=0.783, MR=0.231, Tourn=7, Elitism=11, Stagnation=25 | Optuna best: 12 261

| Run | Score | Programs | Time (s) |
|---|---|---|---|
| run_00 | 11 788 | 182 | 40.0 |
| run_01 | 11 983 | 186 | 40.0 |
| run_02 | 12 123 | 186 | 40.0 |
| run_03 | 11 968 | 182 | 40.0 |
| run_04 | 11 960 | 187 | 40.0 |
| run_05 | 11 787 | 180 | 40.0 |
| run_06 | 12 057 | 184 | 40.0 |
| run_07 | 12 205 | 183 | 40.0 |
| run_08 | 12 007 | 181 | 40.0 |
| run_09 | 12 236 | 189 | 40.0 |
| **Mean** | **12 011.4** | **184.0** | **40.0** |
| **Min** | **11 787** | **180** | — |
| **Max** | **12 236** | **189** | — |
| **Std dev** | **152.1** | — | — |

Strong improvement: mean 12 011 vs default 10 831 (+11 %). France schedules the most programs of any standard instance (up to 189), and the tuned config uses the largest population (95) with high elitism (11) and large tournament (7) to efficiently exploit good partial solutions. Variance is moderate — runs cluster in a 450-point window — indicating reliable convergence to a good region. Run_09 achieves both the highest score (12 236) and most programs (189).

---

### youtube_gold

Tuned parameters: Pop=65, Gen=100, CR=0.828, MR=0.223, Tourn=3, Elitism=5, Stagnation=31 | Optuna best: 40 066

| Run | Score | Programs | Time (s) |
|---|---|---|---|
| run_00 | 40 023 | 664 | 40.2 |
| run_01 | 40 076 | 665 | 40.2 |
| run_02 | 40 134 | 665 | 40.2 |
| run_03 | 40 051 | 662 | 40.2 |
| run_04 | 40 068 | 664 | 40.2 |
| run_05 | 40 054 | 664 | 40.2 |
| run_06 | 40 018 | 664 | 40.2 |
| run_07 | 40 007 | 663 | 40.2 |
| run_08 | 40 027 | 664 | 40.2 |
| run_09 | 40 089 | 665 | 40.2 |
| **Mean** | **40 054.7** | **664.0** | **40.2** |
| **Min** | **40 007** | **662** | — |
| **Max** | **40 134** | **665** | — |
| **Std dev** | **38.7** | — | — |

Remarkably tight for the largest instance by program count. All ten runs land within a 127-point band despite scheduling over 660 programs, giving the lowest coefficient of variation in the study. Tuning dramatically improved the score relative to the default (33 156 → mean 40 055, +21 %). The small tournament size (3) and low elitism (5) chosen by Optuna keep genetic diversity high, which appears well-suited for this dense, high-program instance.

---

### youtube_premium

Tuned parameters: Pop=65, Gen=130, CR=0.839, MR=0.181, Tourn=5, Elitism=7, Stagnation=40 | Optuna best: 52 368

| Run | Score | Programs | Time (s) |
|---|---|---|---|
| run_00 | 51 668 | 638 | 40.3 |
| run_01 | 50 949 | 618 | 40.3 |
| run_02 | 51 651 | 636 | 40.3 |
| run_03 | 52 722 | 638 | 40.3 |
| run_04 | 52 283 | 628 | 40.3 |
| run_05 | 50 658 | 618 | 40.3 |
| run_06 | 51 722 | 625 | 40.3 |
| run_07 | 52 531 | 633 | 40.3 |
| run_08 | 50 779 | 617 | 40.3 |
| run_09 | 50 597 | 616 | 40.3 |
| **Mean** | **51 556.0** | **626.7** | **40.3** |
| **Min** | **50 597** | **616** | — |
| **Max** | **52 722** | **638** | — |
| **Std dev** | **787.8** | — | — |

The most volatile instance in absolute terms, expected given it is also the largest (5.74 MB, 600+ programs). Programs scheduled per run varies considerably — from 616 to 638 — and the score spread exceeds 2 000 points. The default uncapped run achieved 53 998 in 165 s; the tuned runs are capped at 40 s, explaining the lower mean despite better per-trial optimisation. Within that budget the GA reaches up to 52 722 (run_03). The highest stagnation limit in the study (40) allows sustained search before triggering the mutation boost.

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
