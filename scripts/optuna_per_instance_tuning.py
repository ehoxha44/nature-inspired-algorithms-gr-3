#!/usr/bin/env python3
"""
Per-instance Optuna tuning + validation schedules.

* ``parameter-tuning-configs/<instance>.json`` — **only**
  ``{ "instance", "adjustments" }`` where ``adjustments`` holds the chosen GA
  hyperparameters plus ``best_score`` from the Optuna study (no trial dumps).

* ``parameter-tuning/<instance>/run_<MM>.json`` — exactly ``--runs`` files
  per instance. **Strictly** the same six keys as ``data/output/<instance>.json``
  (validator): ``instance``, ``total_score``, ``programs_scheduled``,
  ``elapsed_seconds``, ``schedule``, ``convergence`` — via
  ``SolveResult.as_output_json`` only.

Use ``--fast`` (default) for a quicker search; ``--no-fast`` for a heavier study.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tv_scheduler.ga import GAConfig
from tv_scheduler.main import solve_detailed

_VALIDATOR_KEYS = frozenset(
    {"instance", "total_score", "programs_scheduled", "elapsed_seconds", "schedule", "convergence"}
)


def _suggest_config(trial: Any, *, fast: bool) -> GAConfig:
    if fast:
        return GAConfig(
            population_size=trial.suggest_int("population_size", 40, 95, step=5),
            num_generations=trial.suggest_int("num_generations", 40, 140, step=10),
            crossover_rate=trial.suggest_float("crossover_rate", 0.76, 0.92),
            mutation_rate=trial.suggest_float("mutation_rate", 0.14, 0.30),
            tournament_size=trial.suggest_int("tournament_size", 3, 7),
            elitism_count=trial.suggest_int("elitism_count", 5, 14),
            stagnation_limit=trial.suggest_int("stagnation_limit", 18, 42),
            seed=trial.suggest_int("trial_seed", 0, 2**30 - 1),
        )
    return GAConfig(
        population_size=trial.suggest_int("population_size", 55, 145, step=5),
        num_generations=trial.suggest_int("num_generations", 90, 320, step=10),
        crossover_rate=trial.suggest_float("crossover_rate", 0.72, 0.94),
        mutation_rate=trial.suggest_float("mutation_rate", 0.12, 0.34),
        tournament_size=trial.suggest_int("tournament_size", 3, 8),
        elitism_count=trial.suggest_int("elitism_count", 5, 16),
        stagnation_limit=trial.suggest_int("stagnation_limit", 16, 48),
        seed=trial.suggest_int("trial_seed", 0, 2**30 - 1),
    )


def _config_from_frozen_params(params: Dict[str, Any], seed: int) -> GAConfig:
    return GAConfig(
        population_size=int(params["population_size"]),
        num_generations=int(params["num_generations"]),
        crossover_rate=float(params["crossover_rate"]),
        mutation_rate=float(params["mutation_rate"]),
        tournament_size=int(params["tournament_size"]),
        elitism_count=int(params["elitism_count"]),
        stagnation_limit=int(params["stagnation_limit"]),
        seed=seed,
    )


def run_optuna_on_instance(
    instance_path: Path,
    scratch_output: str,
    n_trials: int,
    time_limit_sec: float,
    fast: bool,
) -> Tuple[Dict[str, Any], Optional[float]]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        cfg = _suggest_config(trial, fast=fast)
        res = solve_detailed(
            str(instance_path),
            scratch_output,
            cfg,
            verbose=False,
            write_output=False,
            time_limit_sec=time_limit_sec,
            use_guided_local_search=True,
        )
        trial.set_user_attr("programs", res.programs_scheduled)
        trial.set_user_attr("stop_reason", res.stop_reason)
        return float(res.score)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    try:
        return dict(study.best_params), float(study.best_value)
    except ValueError:
        return {}, None


def build_adjustments_config(
    instance_stem: str,
    best_params: Dict[str, Any],
    best_value: Optional[float],
) -> Dict[str, Any]:
    adj: Dict[str, Any] = dict(best_params)
    if best_value is not None:
        adj["best_score"] = best_value
    return {"instance": instance_stem, "adjustments": adj}


def run_best_validation_runs(
    instance_path: Path,
    scratch_output: str,
    best_params: Dict[str, Any],
    n_runs: int,
    time_limit_sec: float,
    base_seed: int,
    inst_run_dir: Path,
    instance_stem: str,
) -> int:
    """
    Writes ``run_00.json`` … — **only** keys in ``_VALIDATOR_KEYS``.
    """
    inst_run_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0

    for r in range(n_runs):
        seed = base_seed + r * 19 + hash(instance_path.stem) % 9_991
        cfg = _config_from_frozen_params(best_params, seed)
        res = solve_detailed(
            str(instance_path),
            scratch_output,
            cfg,
            verbose=False,
            write_output=False,
            time_limit_sec=time_limit_sec,
            use_guided_local_search=True,
        )
        fname = f"run_{r:02d}.json"
        out_path = inst_run_dir / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = res.as_output_json(instance_stem)
        if frozenset(payload.keys()) != _VALIDATOR_KEYS:
            raise RuntimeError(
                f"Internal error: run output keys {sorted(payload.keys())!r} != {_VALIDATOR_KEYS!r}"
            )
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        n_ok += 1
    return n_ok


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Optuna tuning (minimal config) + validator-format run_*.json",
    )
    ap.add_argument("--input-dir", default=str(_ROOT / "data" / "input"))
    ap.add_argument(
        "--output-dir",
        default=str(_ROOT / "data" / "output"),
        help="Parent directory; writes parameter-tuning-configs/ and parameter-tuning/",
    )
    ap.add_argument(
        "--fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fast search (default): smaller ranges / fewer trials unless overridden.",
    )
    ap.add_argument(
        "--optuna-trials",
        type=int,
        default=None,
        help="Optuna trials per instance (default: 6 if --fast, else 24)",
    )
    ap.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Validation runs per instance (best Optuna params, default 10)",
    )
    ap.add_argument(
        "--time-limit-sec",
        type=float,
        default=None,
        help="Wall time per solve (default: 40 if --fast, else 300)",
    )
    ap.add_argument("--base-seed", type=int, default=42_000)
    args = ap.parse_args()

    try:
        import optuna  # noqa: F401
    except ImportError:
        print("Install optuna:  pip install optuna", file=sys.stderr)
        return 1

    fast = bool(args.fast)
    n_trials = args.optuna_trials if args.optuna_trials is not None else (6 if fast else 24)
    time_limit = args.time_limit_sec if args.time_limit_sec is not None else (40.0 if fast else 300.0)

    input_dir = Path(args.input_dir)
    instances = sorted(input_dir.glob("*.json"))
    if not instances:
        print(f"No JSON in {input_dir}", file=sys.stderr)
        return 1

    out_parent = Path(args.output_dir).resolve()
    cfg_dir = out_parent / "parameter-tuning-configs"
    run_dir = out_parent / "parameter-tuning"
    scratch = str(run_dir / "_scratch")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    Path(scratch).mkdir(parents=True, exist_ok=True)

    for inst in instances:
        print(f"\n=== {inst.stem}  (fast={fast}, trials={n_trials}, cap={time_limit}s) ===")

        best_params, best_value = run_optuna_on_instance(
            inst,
            scratch,
            n_trials=n_trials,
            time_limit_sec=time_limit,
            fast=fast,
        )

        cfg_path = cfg_dir / f"{inst.stem}.json"
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(
                build_adjustments_config(inst.stem, best_params, best_value),
                fh,
                indent=2,
            )

        inst_run = run_dir / inst.stem
        inst_run.mkdir(parents=True, exist_ok=True)
        n_written = 0
        if best_params:
            n_written = run_best_validation_runs(
                inst,
                scratch,
                best_params,
                n_runs=args.runs,
                time_limit_sec=time_limit,
                base_seed=args.base_seed,
                inst_run_dir=inst_run,
                instance_stem=inst.stem,
            )
        else:
            print(f"  skip validation (no completed trials): {inst.stem}", file=sys.stderr)

        print(f"  configs   → {cfg_path}  (instance + adjustments only)")
        print(f"  schedules → {inst_run}/run_*.json ({n_written} files, keys={sorted(_VALIDATOR_KEYS)})")

    print(f"\nDone.\n  {cfg_dir}/\n  {run_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
