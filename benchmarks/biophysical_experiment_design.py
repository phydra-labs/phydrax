from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
from _runtime import capture_environment, measure_repeated

import phydrax as phx


def _candidate_panel(count: int) -> tuple[phx.uq.ExperimentalDesignCandidate, ...]:
    return tuple(
        phx.uq.ExperimentalDesignCandidate(
            f"synthetic-candidate-{index:02d}",
            f"condition-{index:02d}",
            0.75 + 0.25 * (index % 3),
            "synthetic-assay",
            "synthetic-finite-model-predictive-v1",
            setup_id=f"setup-{index // 4}",
            setup_cost=0.5,
            diversity_group=f"condition-family-{index % 4}",
            mandatory_control=index == 0,
        )
        for index in range(count)
    )


def _finite_model_channel(count: int) -> jnp.ndarray:
    candidate = jnp.arange(count, dtype=float)[:, None]
    model = jnp.arange(3, dtype=float)[None, :]
    success = 0.5 + 0.32 * jnp.sin(0.71 * candidate + 1.37 * model)
    return jnp.stack((1.0 - success, success), axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--budget", type=float, default=6.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 4 <= args.candidates <= 16:
        raise ValueError("candidates must lie in [4, 16] for exact batch enumeration")
    if not 2 <= args.batch_size <= args.candidates:
        raise ValueError("batch-size must lie in [2, candidates]")
    if args.budget <= 0.0:
        raise ValueError("budget must be positive")

    candidates = _candidate_panel(args.candidates)
    conditional = _finite_model_channel(args.candidates)
    model_probabilities = jnp.asarray([0.35, 0.40, 0.25])
    evaluate = lambda: phx.uq.exact_finite_expected_utility(
        conditional,
        model_probabilities,
        candidates=candidates,
        model_ids=("synthetic-model-a", "synthetic-model-b", "synthetic-model-c"),
        utility_target="model_discrimination",
    )
    utility, utility_elapsed = measure_repeated(
        evaluate,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    constraints = phx.uq.ExperimentalBatchConstraints(
        args.budget,
        args.batch_size,
        mutually_exclusive_candidate_groups=(
            ("synthetic-candidate-01", "synthetic-candidate-02"),
        ),
        minimum_diversity_groups=2,
        maximum_per_diversity_group=1,
    )
    coordinate = jnp.arange(args.candidates, dtype=float)
    separation = jnp.abs(coordinate[:, None] - coordinate[None, :])
    redundancy = jnp.exp(-separation / 2.0) - jnp.eye(args.candidates)
    select = lambda: phx.uq.select_experimental_batch(
        candidates,
        utility,
        constraints,
        objective_id="synthetic-finite-model-information",
        model_ids=("synthetic-model-a", "synthetic-model-b", "synthetic-model-c"),
        analysis_id="biophysical-experiment-design-benchmark-v1",
        pairwise_redundancy=redundancy,
        redundancy_weight=0.05,
    )
    plan, selection_elapsed = measure_repeated(
        select,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    replay = phx.uq.evaluate_retrospective_design(
        candidates,
        constraints,
        utility,
        random_key=jr.key(92),
        space_filling_distances=separation,
        uncertainty_scores=jnp.max(conditional, axis=-1).var(axis=-1),
        domain_heuristic_scores=jnp.linspace(0.0, 1.0, args.candidates),
        realized_utility=lambda selected: sum(
            float(utility.expected_utility[index])
            for index, candidate in enumerate(candidates)
            if candidate.candidate_id in selected
        ),
        metric_id="synthetic-realized-model-ambiguity-reduction",
        model_ids=("synthetic-model-a", "synthetic-model-b", "synthetic-model-c"),
        analysis_id="biophysical-experiment-design-benchmark-v1",
        objective_id="synthetic-retrospective-replay",
        proposed_pairwise_redundancy=redundancy,
        proposed_redundancy_weight=0.05,
    )

    payload = {
        "environment": capture_environment().to_dict(),
        "candidate_count": args.candidates,
        "maximum_batch_size": args.batch_size,
        "budget": args.budget,
        "utility_execution_seconds": utility_elapsed.to_seconds_dict(),
        "selection_execution_seconds": selection_elapsed.to_seconds_dict(),
        "estimator": {
            "method_id": utility.method_id,
            "approximation": utility.approximation,
            "error_basis": utility.error_basis,
            "utility_target": utility.utility_target,
            "candidate_content_ids": list(utility.candidate_content_ids),
            "prediction_source_ids": list(utility.prediction_source_ids),
            "model_ids": list(utility.model_ids),
            "maximum_standard_error": float(jnp.max(utility.estimator_standard_error)),
            "maximum_bias_bound": float(jnp.max(utility.estimator_bias_bound)),
        },
        "selected_candidate_ids": list(plan.selected_candidate_ids),
        "planned_total_cost": plan.planned_total_cost,
        "plan_id": plan.plan_id,
        "retrospective": {
            "evaluation_kind": replay.evaluation_kind,
            "strategy_ids": list(replay.strategy_ids),
            "comparison_basis": replay.comparison_basis,
            "planned_total_costs": [float(value) for value in replay.planned_total_costs],
            "selected_batch_sizes": [int(value) for value in replay.selected_batch_sizes],
            "cost_normalized_realized_utility": [
                float(value) for value in replay.cost_normalized_realized_utility
            ],
            "matched_planned_total_cost": replay.matched_planned_total_cost,
            "matched_batch_size": replay.matched_batch_size,
            "realized_utility": [float(value) for value in replay.realized_utility],
            "evaluation_id": replay.evaluation_id,
        },
        "scientific_claim": "none; synthetic software/design benchmark",
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
