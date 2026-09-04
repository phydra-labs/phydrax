#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr

from phydrax.applications.skeletal_muscle.motor_units import (
    commit_fuglevand_winter_patla_1993,
    FuglevandWinterPatla1993Plan,
    FuglevandWinterPatla1993QualificationPlan,
    FuglevandWinterPatla1993RandomInput,
)


def _trial(prepared, *, seed: int, steps: int):
    state = prepared.initialize()
    key = jr.key(seed)
    force = []
    scores = []
    for _ in range(steps):
        random_input = FuglevandWinterPatla1993RandomInput(
            key,
            state.random_step,
            stream_id=prepared.plan.random_stream_id,
        )
        candidate = prepared.evaluate(
            state,
            0.7 * prepared.maximum_excitation,
            5.0,
            random_input,
        )
        if not bool(candidate.evidence.successful):
            raise RuntimeError(
                f"motor-unit qualification trial failed with status {int(candidate.evidence.status)}"
            )
        state = commit_fuglevand_winter_patla_1993(candidate, state)
        force.append(prepared.force(state).total_force_arbitrary)
        scores.append(candidate.evidence.normal_scores)
    return jnp.stack(force), jnp.stack(scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/skeletal_motor_units_qualification.json"),
    )
    arguments = parser.parse_args()
    prepared = FuglevandWinterPatla1993Plan(
        120,
        event_capacity_per_unit=4,
        random_stream_id="qualification/fuglevand-1993/discharge",
    ).prepare()
    force, scores = _trial(prepared, seed=arguments.seed, steps=arguments.steps)
    replay_force, replay_scores = _trial(
        prepared, seed=arguments.seed, steps=arguments.steps
    )
    sample_mask = jnp.ones_like(scores, dtype=bool)
    evidence = FuglevandWinterPatla1993QualificationPlan().evaluate(
        prepared,
        scores,
        sample_mask,
        force,
        replay_force,
    )
    payload = {
        "model_id": prepared.plan.model_id,
        "source_doi": "10.1152/jn.1993.70.6.2470",
        "replay_scores_exact": bool(jnp.array_equal(scores, replay_scores)),
        "replay_force_exact": bool(evidence.replay_exact),
        "normal_score_mean": float(evidence.normal_score_mean),
        "normal_score_standard_deviation": float(
            evidence.normal_score_standard_deviation
        ),
        "force_mean_arbitrary": float(evidence.force_mean_arbitrary),
        "force_coefficient_of_variation": float(
            evidence.force_coefficient_of_variation
        ),
        "event_topology_gradient_supported": False,
        "valid": bool(evidence.valid),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["valid"] or not payload["replay_scores_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
