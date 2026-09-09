#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Numerical split refinement against the unchanged dense Shorten small oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.skeletal_muscle.cellular import ShortenFastTwitchModel
from phydrax.applications.skeletal_muscle.fibers import (
    PrescribedFiberStimulusSchedule,
    Shorten2007FiberReaction,
    SkeletalFiberBundlePlan,
    StructuredFiberResponsePlan,
)


def qualify(substep_counts: tuple[int, ...]) -> dict[str, object]:
    model = ShortenFastTwitchModel()
    node_count = 3
    duration = 0.08
    diffusion = 0.5
    positions = (
        jnp.zeros((1, node_count, 3)).at[0, :, 0].set(jnp.linspace(0.0, 1.0, node_count))
    )
    stimulus = PrescribedFiberStimulusSchedule(
        jnp.asarray([0.0]),
        jnp.asarray([0.16]),
        jnp.asarray([150.0]),
        jnp.zeros((1, 1, node_count), dtype=bool).at[0, 0, 0].set(True),
    )
    dense = SkeletalFiberBundlePlan(
        ("oracle-fiber",),
        node_count,
        jnp.asarray([1.0]),
        jnp.asarray([diffusion]),
        stimulus,
        relative_tolerance=1e-9,
        absolute_tolerance=1e-11,
        maximum_step_ms=duration,
    ).prepare(model)
    dense_source = dense.initialize()
    oracle = eqx.filter_jit(dense.candidate)(dense_source, duration)
    oracle.candidate_state.values.block_until_ready()
    reference = np.asarray(oracle.candidate_state.values)
    rows = []
    for count in substep_counts:
        runtime = StructuredFiberResponsePlan(
            ("oracle-fiber",),
            positions,
            stimulus,
            jnp.linspace(0.0, 1.0, count + 1),
            geometry_source_id="manufactured-uniform-1mm-world-x-small-oracle",
            relative_tolerance=1e-9,
            absolute_tolerance=1e-11,
            maximum_step_ms=duration,
        ).prepare(
            Shorten2007FiberReaction(model), jnp.full((1, node_count - 1), diffusion)
        )
        source = runtime.initialize()
        path = runtime.linear_geometry_path(source, source.node_positions_mm)
        candidate = eqx.filter_jit(runtime.candidate)(source, duration, path)
        candidate.candidate_state.values.block_until_ready()
        error = np.asarray(candidate.candidate_state.values) - reference
        scaled = error / np.maximum(1.0, np.abs(reference))
        rows.append(
            {
                "substeps": count,
                "substep_ms": duration / count,
                "prepared_id": runtime.prepared_id,
                "successful": bool(candidate.evidence.successful),
                "maximum_scaled_state_error": float(np.max(np.abs(scaled))),
                "maximum_voltage_error_mV": float(np.max(np.abs(error[..., 0]))),
                "maximum_diffusion_relative_residual": float(
                    jnp.max(candidate.evidence.diffusion_relative_residual)
                ),
                "maximum_diffusion_balance_error_mV_mm": float(
                    jnp.max(candidate.evidence.diffusion_balance_error)
                ),
                "minimum_diffusion_pivot_mm": float(
                    jnp.min(candidate.evidence.minimum_diffusion_pivot)
                ),
                # Nonnegative CN RHS is sufficient for a monotone diffusion update.
                # This bound is numerical and is not a biological/AP accuracy bound.
                "cn_monotonicity_sufficient_step_bound_ms": (1.0 / (node_count - 1)) ** 2
                / diffusion,
            }
        )
    errors = [row["maximum_scaled_state_error"] for row in rows]
    # Compare above the explicit adaptive-reaction/dense-oracle tolerance floor.
    refinement = all(
        fine <= 0.8 * coarse + 1e-7
        for coarse, fine in zip(errors[:-1], errors[1:], strict=True)
    )
    successful = (
        bool(oracle.evidence.successful)
        and all(row["successful"] for row in rows)
        and refinement
        and errors[-1] < 2e-3
    )
    return {
        "source_id": model.model_id,
        "source_url": model.source_url,
        "dense_prepared_id": dense.prepared_id,
        "dense_successful": bool(oracle.evidence.successful),
        "numerical_contract": "Strang independent local Kvaerno5 and midpoint-geometry lumped-FEM Crank-Nicolson",
        "scope": "Small stationary isometric-Shorten oracle; no coupled or physiological validation.",
        "refinement_with_tolerance_floor": refinement,
        "all_successful": successful,
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    payload = qualify((1, 2) if arguments.smoke else (1, 2, 4, 8))
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(text)
    print(text, end="")
    if not payload["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
