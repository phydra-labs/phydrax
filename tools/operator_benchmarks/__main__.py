from __future__ import annotations

import argparse
import json
from typing import cast

import jax.random as jr

import phydrax as phx

from .matrix import run_benchmark_matrix, save_benchmark_artifacts
from .runner import run_operator_benchmark
from .scenarios import (
    green_function_scenario,
    periodic_burgers_scenario,
    split_operator_scenario,
    standard_operator_benchmarks,
)
from .uq import (
    OperatorUQBenchmarkProfile,
    run_operator_uq_suite,
    save_operator_uq_artifacts,
)
from .v2 import (
    ComparisonMode,
    load_family_parity_evidence,
    OperatorBenchmarkProtocol,
    run_operator_benchmark_v2,
    save_benchmark_v2_artifacts,
    standard_operator_benchmark_ladders,
)


def _comma_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _seed_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _comma_tuple(value))


def _float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in _comma_tuple(value))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic and uncertainty-aware neural-operator benchmarks."
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--minimum-delta", type=float, default=0.0)
    parser.add_argument("--relative-minimum-delta", type=float, default=None)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--benchmark-profile",
        choices=("smoke", "shortlist", "decision"),
        default="shortlist",
    )
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--split-seed", type=int, default=1729)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--architectures", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--commit-identity", default="working-tree")
    parser.add_argument("--uq", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--posterior-samples", type=int, default=32)
    parser.add_argument("--skip-laplace", action="store_true")
    parser.add_argument("--v2", action="store_true")
    parser.add_argument(
        "--comparison",
        choices=("native", "capacity", "compute", "pareto"),
        default=None,
    )
    parser.add_argument("--learning-rates", default="")
    parser.add_argument("--target-parameters", type=int, default=0)
    parser.add_argument("--compute-budget", type=int, default=0)
    parser.add_argument("--size-scales", default="0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--sample-fractions", default="")
    parser.add_argument("--difficulty", choices=("all", "easy", "hard"), default="all")
    parser.add_argument("--ladders", default="")
    parser.add_argument("--parity-evidence", default="")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--checkpoint-directory", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--train-sensor-dropout", type=float, default=0.0)
    arguments = parser.parse_args()
    benchmark_profile = "smoke" if arguments.quick else arguments.benchmark_profile
    benchmark_quick = benchmark_profile == "smoke"
    comparison = cast(
        ComparisonMode,
        arguments.comparison
        or {
            "smoke": "native",
            "shortlist": "compute",
            "decision": "pareto",
        }[benchmark_profile],
    )
    if arguments.seeds:
        seeds = _seed_tuple(arguments.seeds)
    elif arguments.v2 and benchmark_quick:
        seeds = (0,)
    elif arguments.v2:
        seeds = (0, 1, 2, 3, 4)
    else:
        seeds = (0, 1, 2)
    if arguments.steps is not None:
        steps = int(arguments.steps)
    elif arguments.v2:
        steps = {
            "smoke": 1,
            "shortlist": 300,
            "decision": 1000,
        }[benchmark_profile]
    else:
        steps = 20

    if arguments.v2:
        requested_ladders = set(_comma_tuple(arguments.ladders))
        ladders = standard_operator_benchmark_ladders(
            quick=benchmark_quick,
            profile=benchmark_profile,
        )
        if requested_ladders:
            ladders = tuple(
                ladder for ladder in ladders if ladder.name in requested_ladders
            )
        if not ladders:
            raise ValueError("No benchmark-v2 difficulty ladders were selected.")
        architectures = _comma_tuple(arguments.architectures)
        if not architectures and benchmark_profile != "smoke":
            architectures = (
                "constant",
                "nearest_neighbor",
                "pod_linear_rom",
                "deeponet",
                "fno",
                "cno",
                "fno_p4_augmented",
                "laplace",
                "sfno",
                "cochain_pointwise",
                "cochain_no_harmonic",
                "cochain_neural_operator",
            )
        learning_rates = (
            _float_tuple(arguments.learning_rates)
            if arguments.learning_rates
            else ((1e-3,) if benchmark_quick else (3e-4, 1e-3, 3e-3))
        )
        protocol = OperatorBenchmarkProtocol(
            seeds=seeds,
            comparison=comparison,
            steps=steps,
            learning_rates=learning_rates,
            repeats=arguments.repeats,
            validation_interval=arguments.validation_interval,
            patience=(
                (
                    None
                    if benchmark_quick
                    else (20 if benchmark_profile == "decision" else 10)
                )
                if arguments.patience is None
                else (None if arguments.patience == 0 else arguments.patience)
            ),
            minimum_delta=arguments.minimum_delta,
            relative_minimum_delta=(
                1e-4
                if arguments.relative_minimum_delta is None
                else arguments.relative_minimum_delta
            ),
            target_parameters=(
                None if arguments.target_parameters == 0 else arguments.target_parameters
            ),
            compute_budget=(
                None if arguments.compute_budget == 0 else arguments.compute_budget
            ),
            size_scales=_float_tuple(arguments.size_scales),
            sample_fractions=(
                _float_tuple(arguments.sample_fractions)
                if arguments.sample_fractions
                else ((0.25, 0.5, 1.0) if benchmark_profile == "decision" else (1.0,))
            ),
            normalize=not arguments.no_normalize,
            split_seed=arguments.split_seed,
            quick=benchmark_quick,
            profile=benchmark_profile,
            commit_identity=arguments.commit_identity,
            checkpoint_directory=(
                arguments.checkpoint_directory
                or (
                    f"{arguments.output}/checkpoints"
                    if benchmark_profile == "decision" and arguments.output
                    else None
                )
            ),
            resume=arguments.resume,
            sensor_training_dropout=arguments.train_sensor_dropout,
        )
        family_parity = (
            ()
            if not arguments.parity_evidence
            else load_family_parity_evidence(arguments.parity_evidence)
        )
        result = run_operator_benchmark_v2(
            ladders,
            protocol=protocol,
            architecture_names=None if not architectures else architectures,
            family_parity=family_parity,
            difficulty=None if arguments.difficulty == "all" else arguments.difficulty,
        )
        if arguments.output:
            paths = save_benchmark_v2_artifacts(arguments.output, result)
            summary = {
                "artifacts": [str(path) for path in paths],
                "scenarios": len(result.audits),
                "trials": len(result.trials),
                "selected_runs": len(result.results),
                "scenario_promotions": sum(
                    report.promoted for report in result.promotions
                ),
                "portfolio_promotions": sum(
                    report.promoted for report in result.portfolio_promotions
                ),
                "difficulty_audits_passed": sum(
                    audit.passed for audit in result.difficulty_audits
                ),
                "complete_pareto_points": sum(
                    point.complete
                    for front in result.pareto_fronts
                    for point in front.points
                ),
                "nondominated_pareto_points": sum(
                    point.nondominated is True
                    for front in result.pareto_fronts
                    for point in front.points
                ),
            }
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return

    if arguments.uq:
        case_count = 24
        periodic = split_operator_scenario(
            periodic_burgers_scenario(
                train_resolution=arguments.resolution,
                test_resolution=arguments.resolution + arguments.resolution // 2,
                num_cases=case_count,
                rollout_steps=2,
            ),
            seed=arguments.split_seed,
            train_fraction=0.4,
            validation_fraction=0.4,
        )
        point_cloud = split_operator_scenario(
            green_function_scenario(
                source_points=max(64, 2 * arguments.resolution),
                query_points=max(32, 2 * arguments.resolution),
                num_cases=case_count,
            ),
            seed=arguments.split_seed,
            train_fraction=0.4,
            validation_fraction=0.4,
        )
        suite = run_operator_uq_suite(
            (
                OperatorUQBenchmarkProfile(periodic, "fno"),
                OperatorUQBenchmarkProfile(point_cloud, "deeponet"),
            ),
            seeds=seeds,
            steps=steps,
            learning_rate=arguments.learning_rate,
            repeats=arguments.repeats,
            alpha=arguments.alpha,
            quick=arguments.quick,
            validation_interval=arguments.validation_interval,
            patience=None if arguments.patience == 0 else arguments.patience,
            minimum_delta=arguments.minimum_delta,
            fit_projection_laplace=not arguments.skip_laplace,
            posterior_samples=arguments.posterior_samples,
            commit_identity=arguments.commit_identity,
        )
        if arguments.output:
            save_operator_uq_artifacts(arguments.output, suite)
        print(json.dumps(suite.to_dict(), indent=2, sort_keys=True))
        return

    if arguments.matrix:
        architectures = _comma_tuple(arguments.architectures)
        scenarios = standard_operator_benchmarks(quick=arguments.quick)
        if arguments.split:
            scenarios = tuple(
                split_operator_scenario(scenario, seed=arguments.split_seed)
                for scenario in scenarios
            )
        matrix = run_benchmark_matrix(
            scenarios,
            seeds=seeds,
            architecture_names=None if not architectures else architectures,
            steps=steps,
            learning_rate=arguments.learning_rate,
            repeats=arguments.repeats,
            quick=arguments.quick,
            validation_interval=arguments.validation_interval,
            patience=None if arguments.patience == 0 else arguments.patience,
            minimum_delta=arguments.minimum_delta,
            commit_identity=arguments.commit_identity,
        )
        if arguments.output:
            save_benchmark_artifacts(arguments.output, matrix)
        print(json.dumps(matrix.to_dict(), indent=2, sort_keys=True))
        return

    scenario = periodic_burgers_scenario(
        train_resolution=arguments.resolution,
        test_resolution=arguments.resolution + arguments.resolution // 2,
        num_cases=4,
    )
    model = phx.nn.operator.architectures.FNO(
        width=12,
        depth=2,
        n_modes=(min(6, arguments.resolution // 2),),
        key=jr.key(0),
    )
    _, result = run_operator_benchmark(
        model,
        scenario,
        steps=steps,
        learning_rate=arguments.learning_rate,
        repeats=arguments.repeats,
        validation_interval=arguments.validation_interval,
        patience=None if arguments.patience == 0 else arguments.patience,
        minimum_delta=arguments.minimum_delta,
        architecture="fno",
        family="spectral",
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
