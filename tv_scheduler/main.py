from __future__ import annotations

import argparse
import bisect
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import Program, ProblemInstance, PriorityBlock, PreferenceInterval
from .decoder import decode_chromosome
from .fitness import (
    compute_attractiveness,
    compute_score,
    relaxed_interval_upper_bound,
)
from .ga import GeneticAlgorithm, GAConfig


@dataclass
class SolveResult:
    score: float
    programs_scheduled: int
    elapsed_seconds: float
    convergence: List[float]
    stop_reason: str
    relaxation_upper_bound: float
    hit_upper_bound: bool
    schedule: List[Dict[str, Any]] = field(default_factory=list)

    def as_output_json(self, instance_name: str) -> Dict[str, Any]:
        """Same schema as ``save_json_output`` / ``data/output/<instance>.json``."""
        return {
            "instance": instance_name,
            "total_score": self.score,
            "programs_scheduled": self.programs_scheduled,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "schedule": self.schedule,
            "convergence": self.convergence,
        }


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
    time_deadline: Optional[float] = None,
    score_upper_bound: Optional[float] = None,
) -> Tuple[List[int], float]:
    if rng is None:
        rng = random.Random()

    best     = chromosome[:]
    best_fit = fitness_fn(best)
    n        = len(best)
    no_imp   = 0

    while no_imp < max_no_improve:
        if time_deadline is not None and time.perf_counter() >= time_deadline:
            break
        if score_upper_bound is not None and best_fit >= score_upper_bound - 1e-5:
            break

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


def guided_local_search_chromosome(
    chromosome: List[int],
    fitness_fn: Callable[[List[int]], float],
    attract: List[float],
    max_no_improve: int = 500,
    rng: Optional[random.Random] = None,
    time_deadline: Optional[float] = None,
    score_upper_bound: Optional[float] = None,
) -> Tuple[List[int], float]:
    """
    Local search biased toward moving high-attractiveness genes earlier in the
    permutation (decoder processes left-to-right).
    """
    if rng is None:
        rng = random.Random()

    best     = chromosome[:]
    best_fit = fitness_fn(best)
    n        = len(best)
    no_imp   = 0

    if n < 2:
        return best, best_fit

    while no_imp < max_no_improve:
        if time_deadline is not None and time.perf_counter() >= time_deadline:
            break
        if score_upper_bound is not None and best_fit >= score_upper_bound - 1e-5:
            break

        c   = best[:]
        r   = rng.random()

        if r < 0.48:
            i = rng.randrange(1, n)
            if attract[c[i]] > attract[c[i - 1]] + 1e-9:
                c[i], c[i - 1] = c[i - 1], c[i]
            else:
                j = rng.randrange(n - 1)
                c[j], c[j + 1] = c[j + 1], c[j]

        elif r < 0.72 and n >= 4:
            back  = list(range(max(1, n // 2), n))
            front = list(range(0, n // 2))
            i_hi  = max(back, key=lambda idx: attract[c[idx]])
            j_lo  = min(front, key=lambda idx: attract[c[idx]])
            if attract[c[i_hi]] > attract[c[j_lo]] + 1e-9:
                c[i_hi], c[j_lo] = c[j_lo], c[i_hi]
            else:
                i = rng.randrange(n - 1)
                c[i], c[i + 1] = c[i + 1], c[i]

        elif r < 0.88:
            seg = rng.randint(2, min(4, n))
            i   = rng.randrange(n - seg + 1)
            c[i : i + seg] = c[i : i + seg][::-1]

        else:
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


def solve_detailed(
    input_path: str,
    output_dir: str,
    config: GAConfig,
    verbose: bool = True,
    write_output: bool = True,
    time_limit_sec: Optional[float] = None,
    use_guided_local_search: bool = True,
) -> SolveResult:
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
        if verbose:
            print("  No valid programs found – writing empty schedule.")
        return SolveResult(
            score=0.0,
            programs_scheduled=0,
            elapsed_seconds=0.0,
            convergence=[],
            stop_reason="no_programs",
            relaxation_upper_bound=0.0,
            hit_upper_bound=False,
            schedule=[],
        )

    t0 = time.perf_counter()
    deadline = (t0 + time_limit_sec) if time_limit_sec is not None else None

    upper = relaxed_interval_upper_bound(programs, instance)
    seeds = build_seed_chromosomes(programs, instance)
    attract = [compute_attractiveness(programs[i], instance) for i in range(len(programs))]

    def fitness_fn(chromosome: List[int]) -> float:
        sched = decode_chromosome(chromosome, programs, instance)
        return compute_score(sched, instance)

    ga = GeneticAlgorithm(config)

    if verbose:
        tl = f"  wall≤{time_limit_sec:.0f}s" if time_limit_sec else ""
        print(
            f"\nGA params: pop={config.population_size}  "
            f"gen={config.num_generations}  "
            f"cx={config.crossover_rate}  mut={config.mutation_rate}  "
            f"elite={config.elitism_count}  stagnation_limit={config.stagnation_limit}  "
            f"seed={config.seed}  relax_UB={upper:.1f} {tl}\n"
        )

    best_chrom, best_fitness, convergence = ga.run(
        chromosome_length=len(programs),
        fitness_fn=fitness_fn,
        seed_chromosomes=seeds,
        verbose=verbose,
        log_interval=max(1, config.num_generations // 10),
        deadline=deadline,
        score_upper_bound=upper,
    )

    if time_limit_sec is not None:
        ls_budget = max(500, min(12_000, len(programs) * 40))
    else:
        ls_budget = max(100, min(500, len(programs) * 2))

    if verbose:
        mode = "Guided local search" if use_guided_local_search else "Local search"
        print(f"\n  {mode} ({ls_budget} max idle steps) …")

    ls_rng = random.Random(config.seed if config.seed is not None else 0)

    if use_guided_local_search:
        best_chrom, best_fitness = guided_local_search_chromosome(
            best_chrom,
            fitness_fn,
            attract,
            max_no_improve=ls_budget,
            rng=ls_rng,
            time_deadline=deadline,
            score_upper_bound=upper,
        )
    else:
        best_chrom, best_fitness = local_search_chromosome(
            best_chrom,
            fitness_fn,
            max_no_improve=ls_budget,
            rng=ls_rng,
            time_deadline=deadline,
            score_upper_bound=upper,
        )

    elapsed = time.perf_counter() - t0

    best_schedule = decode_chromosome(best_chrom, programs, instance)
    best_schedule.sort(key=lambda p: p.start)

    verified_score = compute_score(best_schedule, instance)
    hit_ub = verified_score >= upper - 1e-5
    if hit_ub:
        stop_reason = "relaxation_upper_bound"
    elif time_limit_sec is not None and elapsed >= time_limit_sec - 0.02:
        stop_reason = "time_limit"
    else:
        stop_reason = "completed"

    if write_output:
        out_json = os.path.join(output_dir, f"{instance_name}.json")
        save_json_output(
            best_schedule, verified_score, convergence,
            instance_name, out_json, elapsed
        )

    if verbose:
        print(f"\n  Score    : {verified_score:.1f}")
        print(f"  Programs : {len(best_schedule)}")
        print(f"  Time     : {elapsed:.1f}s  stop={stop_reason}")
        if write_output:
            print(f"  Saved    : {out_json}")

    sched_json = [
        {"channel_id": p.channel_id, "program_id": p.program_id}
        for p in best_schedule
    ]

    return SolveResult(
        score=verified_score,
        programs_scheduled=len(best_schedule),
        elapsed_seconds=elapsed,
        convergence=convergence,
        stop_reason=stop_reason,
        relaxation_upper_bound=upper,
        hit_upper_bound=hit_ub,
        schedule=sched_json,
    )


def solve(
    input_path: str,
    output_dir: str,
    config: GAConfig,
    verbose: bool = True,
    write_output: bool = True,
) -> Tuple[float, int]:
    r = solve_detailed(
        input_path,
        output_dir,
        config,
        verbose=verbose,
        write_output=write_output,
        time_limit_sec=None,
        use_guided_local_search=True,
    )
    return r.score, r.programs_scheduled


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
    p.add_argument(
        "--max-time-sec",
        type=float,
        default=None,
        help="Hard wall-clock limit (seconds) for GA + local search on this run.",
    )
    p.add_argument(
        "--classic-ls",
        action="store_true",
        help="Use randomised local search instead of attractiveness-guided LS.",
    )

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
            res = solve_detailed(
                str(fp),
                args.output_dir,
                config,
                verbose=verbose,
                write_output=True,
                time_limit_sec=args.max_time_sec,
                use_guided_local_search=not args.classic_ls,
            )
            summary.append((fp.stem, res.score, res.programs_scheduled))
    else:
        res = solve_detailed(
            args.input,
            args.output_dir,
            config,
            verbose=verbose,
            write_output=True,
            time_limit_sec=args.max_time_sec,
            use_guided_local_search=not args.classic_ls,
        )
        summary.append((Path(args.input).stem, res.score, res.programs_scheduled))

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
