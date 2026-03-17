from __future__ import annotations

import argparse
import bisect
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .models import Program, ProblemInstance, PriorityBlock, PreferenceInterval
from .decoder import decode_chromosome
from .fitness import compute_score, compute_attractiveness
from .ga import GeneticAlgorithm, GAConfig


def load_instance(filepath: str) -> ProblemInstance:
    with open(filepath, encoding="utf-8") as fh:
        data = json.load(fh)

    priority_blocks = [
        PriorityBlock(
            start=b["start"],
            end=b["end"],
            allowed_channels=set(b["allowed_channels"]),
        )
        for b in data.get("priority_blocks", [])
    ]

    time_preferences = [
        PreferenceInterval(
            start=p["start"],
            end=p["end"],
            genre=p["preferred_genre"],
            bonus=p["bonus"],
        )
        for p in data.get("time_preferences", [])
    ]

    programs: List[Program] = []
    for ch in data["channels"]:
        ch_id = ch["channel_id"]
        for prog in ch["programs"]:
            programs.append(
                Program(
                    program_id=prog["program_id"],
                    channel_id=ch_id,
                    start=prog["start"],
                    end=prog["end"],
                    genre=prog["genre"],
                    score=prog["score"],
                )
            )

    return ProblemInstance(
        opening_time=data["opening_time"],
        closing_time=data["closing_time"],
        min_duration=data["min_duration"],
        max_consecutive_genre=data["max_consecutive_genre"],
        channels_count=data["channels_count"],
        switch_penalty=data["switch_penalty"],
        termination_penalty=data["termination_penalty"],
        priority_blocks=priority_blocks,
        time_preferences=time_preferences,
        programs=programs,
    )


def filter_programs(programs: List[Program], instance: ProblemInstance) -> List[Program]:
    O, E = instance.opening_time, instance.closing_time
    return [p for p in programs if p.start >= O and p.end <= E]


def dp_optimal_seed(
    programs: List[Program], instance: ProblemInstance
) -> List[int]:
    n = len(programs)
    if n == 0:
        return []

    eff: List[float] = []
    for p in programs:
        violates_block = any(
            blk.overlaps_program(p.start, p.end) and not blk.allows(p.channel_id)
            for blk in instance.priority_blocks
        )
        eff.append(0.0 if violates_block else compute_attractiveness(p, instance))

    sort_order = sorted(range(n), key=lambda i: programs[i].end)
    s_end   = [programs[sort_order[k]].end   for k in range(n)]
    s_start = [programs[sort_order[k]].start for k in range(n)]
    s_eff   = [eff[sort_order[k]]            for k in range(n)]

    def pred(k: int) -> int:
        return bisect.bisect_right(s_end, s_start[k], lo=0, hi=k) - 1

    dp = [0.0] * (n + 1)
    for k in range(n):
        p_k = pred(k)
        dp[k + 1] = max(dp[k], s_eff[k] + dp[p_k + 1])

    selected_sorted_indices: List[int] = []
    dp_idx = n
    while dp_idx >= 1:
        k   = dp_idx - 1
        p_k = pred(k)
        if s_eff[k] > 0 and s_eff[k] + dp[p_k + 1] >= dp[dp_idx - 1]:
            selected_sorted_indices.append(k)
            dp_idx = p_k + 1
        else:
            dp_idx -= 1

    selected_orig = {sort_order[k] for k in selected_sorted_indices}

    selected_part = [sort_order[k] for k in reversed(selected_sorted_indices)]
    remaining     = sorted(
        (i for i in range(n) if i not in selected_orig),
        key=lambda i: eff[i], reverse=True
    )
    return selected_part + remaining


def build_seed_chromosomes(
    programs: List[Program], instance: ProblemInstance
) -> List[List[int]]:
    n = len(programs)
    indices = list(range(n))

    attract = [compute_attractiveness(programs[i], instance) for i in indices]

    seed_dp       = dp_optimal_seed(programs, instance)
    seed_attract  = sorted(indices, key=lambda i: attract[i], reverse=True)
    seed_time     = sorted(indices, key=lambda i: programs[i].start)
    seed_duration = sorted(indices, key=lambda i: programs[i].duration)

    return [seed_dp, seed_attract, seed_time, seed_duration]


def local_search_chromosome(
    chromosome: List[int],
    fitness_fn: Callable[[List[int]], float],
    max_no_improve: int = 150,
    rng: Optional[random.Random] = None,
) -> Tuple[List[int], float]:
    if rng is None:
        rng = random.Random()

    best     = chromosome[:]
    best_fit = fitness_fn(best)
    n        = len(best)
    no_imp   = 0

    while no_imp < max_no_improve:
        move = rng.randint(0, 2)
        c    = best[:]

        if move == 0:
            if n < 2:
                no_imp += 1
                continue
            i = rng.randrange(n - 1)
            c[i], c[i + 1] = c[i + 1], c[i]

        elif move == 1:
            if n < 2:
                no_imp += 1
                continue
            seg = rng.randint(2, min(4, n))
            i   = rng.randrange(n - seg + 1)
            c[i : i + seg] = c[i : i + seg][::-1]

        else:
            if n < 2:
                no_imp += 1
                continue
            i    = rng.randrange(n)
            gene = c.pop(i)
            j    = rng.randrange(n)
            c.insert(j, gene)

        fit = fitness_fn(c)
        if fit > best_fit:
            best     = c
            best_fit = fit
            no_imp   = 0
        else:
            no_imp += 1

    return best, best_fit


def save_json_output(
    schedule: List[Program],
    score: float,
    convergence: List[float],
    instance_name: str,
    output_path: str,
    elapsed_sec: float,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    result = {
        "instance": instance_name,
        "total_score": score,
        "programs_scheduled": len(schedule),
        "elapsed_seconds": round(elapsed_sec, 2),
        "schedule": [
            {"channel_id": p.channel_id, "program_id": p.program_id}
            for p in schedule
        ],
        "convergence": convergence,
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)


def solve(
    input_path: str,
    output_dir: str,
    config: GAConfig,
    verbose: bool = True,
) -> Tuple[float, int]:
    instance_name = Path(input_path).stem

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Instance : {instance_name}")
        print(f"{'=' * 60}")

    instance = load_instance(input_path)
    programs = filter_programs(instance.programs, instance)

    if verbose:
        print(
            f"Programs : {len(programs)} valid "
            f"(from {len(instance.programs)} total across "
            f"{instance.channels_count} channels)"
        )
        print(
            f"Window   : {instance.opening_time} – {instance.closing_time} min  |  "
            f"D={instance.min_duration}  R={instance.max_consecutive_genre}  "
            f"S={instance.switch_penalty}  T={instance.termination_penalty}"
        )

    if not programs:
        print("  No valid programs found – writing empty schedule.")
        return 0.0, 0

    seeds = build_seed_chromosomes(programs, instance)

    def fitness_fn(chromosome: List[int]) -> float:
        sched = decode_chromosome(chromosome, programs, instance)
        return compute_score(sched, instance)

    ga = GeneticAlgorithm(config)
    t0 = time.perf_counter()

    if verbose:
        print(
            f"\nGA params: pop={config.population_size}  "
            f"gen={config.num_generations}  "
            f"cx={config.crossover_rate}  mut={config.mutation_rate}  "
            f"elite={config.elitism_count}  stagnation_limit={config.stagnation_limit}  "
            f"seed={config.seed}\n"
        )

    best_chrom, best_fitness, convergence = ga.run(
        chromosome_length=len(programs),
        fitness_fn=fitness_fn,
        seed_chromosomes=seeds,
        verbose=verbose,
        log_interval=max(1, config.num_generations // 10),
    )

    ls_budget = max(100, min(500, len(programs) * 2))
    if verbose:
        print(f"\n  Local search ({ls_budget} max trials) …")

    ls_rng = random.Random(config.seed)
    best_chrom, best_fitness = local_search_chromosome(
        best_chrom, fitness_fn,
        max_no_improve=ls_budget,
        rng=ls_rng,
    )
    elapsed = time.perf_counter() - t0

    best_schedule = decode_chromosome(best_chrom, programs, instance)
    best_schedule.sort(key=lambda p: p.start)

    verified_score = compute_score(best_schedule, instance)

    out_json = os.path.join(output_dir, f"{instance_name}.json")
    save_json_output(
        best_schedule, verified_score, convergence,
        instance_name, out_json, elapsed
    )

    if verbose:
        print(f"\n  Score    : {verified_score:.1f}")
        print(f"  Programs : {len(best_schedule)}")
        print(f"  Time     : {elapsed:.1f}s")
        print(f"  Saved    : {out_json}")

    return verified_score, len(best_schedule)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tv_scheduler",
        description="Smart TV Schedule Optimizer – Genetic Algorithm + DP seed + Local Search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("input", nargs="?",
                   help="Path to a single JSON input file.")
    p.add_argument("--all", "-A", metavar="DIR", dest="input_dir",
                   help="Process every *.json file inside DIR.")

    p.add_argument("--output-dir", "-o", default="data/output",
                   help="Directory where schedule files are written.")

    p.add_argument("--generations",     "-g", type=int,   default=200)
    p.add_argument("--population",      "-p", type=int,   default=100)
    p.add_argument("--crossover-rate",  "-c", type=float, default=0.85)
    p.add_argument("--mutation-rate",   "-m", type=float, default=0.20)
    p.add_argument("--tournament-size", "-t", type=int,   default=5)
    p.add_argument("--elitism",         "-e", type=int,   default=10)
    p.add_argument("--stagnation",            type=int,   default=30,
                   help="Generations without improvement before mutation boost.")
    p.add_argument("--seed",            "-s", type=int,   default=42,
                   help="Random seed (use -1 for non-deterministic).")
    p.add_argument("--quiet",           "-q", action="store_true",
                   help="Suppress per-generation progress output.")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args   = parser.parse_args(argv)

    if args.input is None and args.input_dir is None:
        parser.print_help()
        return 1

    seed = None if args.seed == -1 else args.seed
    config = GAConfig(
        population_size  = args.population,
        num_generations  = args.generations,
        crossover_rate   = args.crossover_rate,
        mutation_rate    = args.mutation_rate,
        tournament_size  = args.tournament_size,
        elitism_count    = args.elitism,
        stagnation_limit = args.stagnation,
        seed             = seed,
    )

    verbose = not args.quiet
    summary: List[Tuple[str, float, int]] = []

    if args.input_dir:
        input_files = sorted(Path(args.input_dir).glob("*.json"))
        if not input_files:
            print(f"No JSON files found in {args.input_dir}", file=sys.stderr)
            return 1
        for fp in input_files:
            score, n = solve(str(fp), args.output_dir, config, verbose=verbose)
            summary.append((fp.stem, score, n))
    else:
        score, n = solve(args.input, args.output_dir, config, verbose=verbose)
        summary.append((Path(args.input).stem, score, n))

    if len(summary) > 1 or verbose:
        print(f"\n{'=' * 60}")
        print(f"{'Instance':<25} {'Score':>10} {'Programs':>10}")
        print(f"{'-' * 60}")
        for name, sc, np in summary:
            print(f"{name:<25} {sc:>10.1f} {np:>10}")
        total = sum(s for _, s, _ in summary)
        print(f"{'-' * 60}")
        print(f"{'TOTAL':.<25} {total:>10.1f}")
        print(f"{'=' * 60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
