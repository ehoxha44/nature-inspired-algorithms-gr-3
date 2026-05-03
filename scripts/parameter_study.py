#!/usr/bin/env python3
"""
Parameter study: ≥3 GA configurations × N runs per instance.

Optional Optuna phase fits hyperparameters on a small reference instance
(`kosovo.json`); the best trial becomes preset "C_Optuna".
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Repo root on sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tv_scheduler.ga import GAConfig
from tv_scheduler.main import solve


def _preset_a() -> GAConfig:
    """Lower budget — faster, less exploration."""
    return GAConfig(
        population_size=80,
        num_generations=150,
        crossover_rate=0.80,
        mutation_rate=0.15,
        tournament_size=4,
        elitism_count=8,
        stagnation_limit=25,
        seed=42,
    )


def _preset_b() -> GAConfig:
    """Baseline aligned with README defaults."""
    return GAConfig(
        population_size=100,
        num_generations=200,
        crossover_rate=0.85,
        mutation_rate=0.20,
        tournament_size=5,
        elitism_count=10,
        stagnation_limit=30,
        seed=42,
    )


def _preset_c_manual() -> GAConfig:
    """High capacity — more search pressure without Optuna."""
    return GAConfig(
        population_size=150,
        num_generations=350,
        crossover_rate=0.90,
        mutation_rate=0.28,
        tournament_size=6,
        elitism_count=12,
        stagnation_limit=35,
        seed=42,
    )


def _config_from_optuna_trial(trial: Any) -> GAConfig:
    return GAConfig(
        population_size=trial.suggest_int("population_size", 70, 160, step=5),
        num_generations=trial.suggest_int("num_generations", 120, 420, step=10),
        crossover_rate=trial.suggest_float("crossover_rate", 0.75, 0.95),
        mutation_rate=trial.suggest_float("mutation_rate", 0.12, 0.35),
        tournament_size=trial.suggest_int("tournament_size", 3, 8),
        elitism_count=trial.suggest_int("elitism_count", 5, 16),
        stagnation_limit=trial.suggest_int("stagnation_limit", 15, 45),
        seed=42,
    )


def run_optuna_phase(
    reference_json: str,
    output_dir: str,
    n_trials: int,
) -> Tuple[GAConfig, Dict[str, Any]]:
    import optuna

    def objective(trial: optuna.Trial) -> float:
        cfg = _config_from_optuna_trial(trial)
        score, _ = solve(
            reference_json,
            output_dir,
            cfg,
            verbose=False,
            write_output=False,
        )
        return -float(score)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_trial
    cfg = GAConfig(
        population_size=int(best.params["population_size"]),
        num_generations=int(best.params["num_generations"]),
        crossover_rate=float(best.params["crossover_rate"]),
        mutation_rate=float(best.params["mutation_rate"]),
        tournament_size=int(best.params["tournament_size"]),
        elitism_count=int(best.params["elitism_count"]),
        stagnation_limit=int(best.params["stagnation_limit"]),
        seed=42,
    )
    meta = {
        "best_value": -study.best_value,
        "n_trials": n_trials,
        "reference_instance": Path(reference_json).stem,
        "best_params": best.params,
    }
    return cfg, meta


def _run_study(
    instances: List[Path],
    presets: Dict[str, GAConfig],
    runs: int,
    output_dir: str,
    base_seed: int,
) -> Dict[str, Any]:
    raw_rows: List[Dict[str, Any]] = []
    combo_keys = list(presets.keys())

    for ci, combo_name in enumerate(combo_keys):
        base_cfg = presets[combo_name]
        for ii, inst in enumerate(instances):
            for r in range(runs):
                seed = base_seed + ci * 10_000 + ii * 100 + r
                cfg = GAConfig(**{**asdict(base_cfg), "seed": seed})
                score, nprog = solve(
                    str(inst),
                    output_dir,
                    cfg,
                    verbose=False,
                    write_output=False,
                )
                raw_rows.append(
                    {
                        "combo": combo_name,
                        "instance": inst.stem,
                        "run": r,
                        "seed": seed,
                        "score": score,
                        "programs": nprog,
                    }
                )

    # Aggregate: mean / std / best per (combo, instance)
    from statistics import mean, stdev

    agg: Dict[str, Dict[str, Any]] = {}
    for row in raw_rows:
        key = f"{row['combo']}::{row['instance']}"
        if key not in agg:
            agg[key] = {"scores": [], "programs": []}
        agg[key]["scores"].append(row["score"])
        agg[key]["programs"].append(row["programs"])

    summary_table: List[Dict[str, Any]] = []
    for key, data in sorted(agg.items()):
        combo, inst = key.split("::", 1)
        sc = data["scores"]
        summary_table.append(
            {
                "combo": combo,
                "instance": inst,
                "mean_score": mean(sc),
                "std_score": stdev(sc) if len(sc) > 1 else 0.0,
                "best_score": max(sc),
                "mean_programs": mean(data["programs"]),
            }
        )

    return {
        "presets": {k: asdict(v) for k, v in presets.items()},
        "runs_per_instance": runs,
        "instances": [p.stem for p in instances],
        "raw": raw_rows,
        "summary": summary_table,
    }


def _markdown_table(summary: List[Dict[str, Any]]) -> str:
    lines = [
        "| Kombinimi | Instanca | Mesatarja | Std | Më i miri |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['combo']} | {row['instance']} | "
            f"{row['mean_score']:.1f} | {row['std_score']:.2f} | {row['best_score']:.1f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="GA parameter study + optional Optuna fit")
    parser.add_argument(
        "--input-dir",
        default=str(_ROOT / "data" / "input"),
        help="Directory with instance JSON files",
    )
    parser.add_argument("--runs", type=int, default=10, help="Runs per instance per combo")
    parser.add_argument(
        "--output-dir",
        default=str(_ROOT / "data" / "output"),
        help="Unused for schedule files when write_output=False; still passed to solve",
    )
    parser.add_argument(
        "--results-json",
        default=str(_ROOT / "data" / "output" / "parameter_study_results.json"),
    )
    parser.add_argument("--base-seed", type=int, default=7_000)
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=30,
        help="If >0, run Optuna on reference instance to build preset C_Optuna (requires optuna)",
    )
    parser.add_argument(
        "--optuna-reference",
        default=str(_ROOT / "data" / "input" / "kosovo.json"),
        help="Small instance for Optuna search",
    )
    parser.add_argument(
        "--no-optuna",
        action="store_true",
        help="Use fixed high-budget preset C instead of Optuna",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    instances = sorted(input_dir.glob("*.json"))
    if not instances:
        print(f"No JSON in {input_dir}", file=sys.stderr)
        return 1

    presets: Dict[str, GAConfig] = {
        "A_ekonomik": _preset_a(),
        "B_standard": _preset_b(),
    }

    optuna_meta: Dict[str, Any] | None = None
    if args.no_optuna or args.optuna_trials <= 0:
        presets["C_intensiv"] = _preset_c_manual()
    else:
        try:
            cfg_c, optuna_meta = run_optuna_phase(
                args.optuna_reference,
                args.output_dir,
                args.optuna_trials,
            )
            presets["C_Optuna"] = cfg_c
        except ImportError:
            print("optuna not installed — using manual C_intensiv", file=sys.stderr)
            presets["C_intensiv"] = _preset_c_manual()

    print("Presets:", list(presets.keys()))
    if optuna_meta:
        print(
            f"Optuna best on {optuna_meta['reference_instance']}: "
            f"score={optuna_meta['best_value']:.1f} params={optuna_meta['best_params']}"
        )

    report = _run_study(instances, presets, args.runs, args.output_dir, args.base_seed)
    report["optuna"] = optuna_meta

    out_path = Path(args.results_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    md_path = out_path.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# Parameter study — summary table\n\n")
        fh.write(_markdown_table(report["summary"]))
        fh.write("\n")

    print(f"\nWrote {out_path}")
    print(f"Wrote {md_path}")
    print("\n" + _markdown_table(report["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
