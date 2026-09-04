#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp

from phydrax.applications.skeletal_muscle import continuum
from phydrax.discretization import (
    CellMesh,
    MixedFiniteElementConstraintPlan,
    PressureGaugePolicy,
)


def _material(activation):
    fibers = continuum.UniformFiberArchitecturePlan("qualification-x-fibers").prepare(
        jnp.asarray((1.0, 0.0, 0.0))
    )
    return continuum.EngelhardtGasam2025Plan("gasam-2025-qualification").prepare(
        continuum.EngelhardtGasam2025Parameters.published_multiload_fit(),
        fibers,
        activation,
    )


def qualify():
    material = _material(0.7)
    deformation = jnp.asarray(
        ((1.03, 0.02, 0.0), (0.0, 0.99, 0.01), (0.0, 0.0, 0.981))
    )
    rate = jnp.asarray(
        ((0.01, 0.002, 0.0), (0.0, -0.004, 0.0), (0.0, 0.0, -0.006))
    )
    point = continuum.GasamQualificationPlan().evaluate(material, deformation, rate)

    volumes = jnp.asarray(
        ((1.0e-6, 0.0, 0.0, 0.0), (0.5e-6, 0.5e-6, 0.0, 0.0), (0.25e-6,) * 4)
    )
    mesh_power = continuum.affine_mesh_power_evidence(
        material,
        volumes,
        volumes > 0.0,
        deformation,
        rate,
    )

    mesh = CellMesh.from_tetrahedra(
        jnp.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        ),
        jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
    )
    passive = _material(0.0)
    mixed = passive.prepare_qualified_mixed(
        MixedFiniteElementConstraintPlan(mesh, PressureGaugePolicy("mean-zero"))
    )
    manufactured = continuum.solve_manufactured_rest(mixed).commit()

    rollback_candidate = material.propose_activation(1.1)
    rollback = rollback_candidate.commit()
    rollback_valid = (
        rollback.rollback_applied
        and not rollback.committed
        and bool(rollback.state.activation == material.state.activation)
    )
    passed = (
        bool(point.valid)
        and bool(mesh_power.valid)
        and manufactured.committed
        and rollback_valid
    )
    return {
        "maturity": "qualified-source-fidelity",
        "passed": passed,
        "source": (
            "Engelhardt et al. 2025, DOI 10.1002/cnm.70036, "
            "Eqs. 15-16, 20, 25-27, Table 5"
        ),
        "scope": (
            "exact-mixed prescribed-activation GASAM; "
            "local smooth-branch AD/stability only"
        ),
        "point": {
            "passed": bool(point.valid),
            "objectivity_energy_error": float(point.objectivity_energy_error),
            "objectivity_stress_error_pa": float(point.objectivity_stress_error),
            "stress_gradient_error_pa": float(point.stress_gradient_error),
            "tangent_jvp_error_pa_per_s": float(point.tangent_jvp_error),
            "power_error_w_per_m3": float(point.power_error),
            "active_fiber_stress_increment_pa": float(
                point.active_fiber_stress_increment_pa
            ),
            "minimum_sampled_acoustic_value_pa": float(
                point.minimum_acoustic_value_pa
            ),
            "global_active_stability_claimed": (
                point.active_global_stability_claimed
            ),
        },
        "mesh_power": {
            "passed": bool(mesh_power.valid),
            "active_cell_counts": [
                int(value) for value in mesh_power.active_cell_counts
            ],
            "maximum_energy_error_j": float(jnp.max(mesh_power.energy_errors_j)),
            "maximum_power_error_w": float(jnp.max(mesh_power.power_errors_w)),
        },
        "mixed_manufactured_rest": {
            "passed": manufactured.committed,
            "pair": list(mixed.qualification.pair_names),
            "inf_sup_constant": float(mixed.qualification.inf_sup_constant),
            "final_residual_norm": float(manufactured.evidence.final_residual_norm),
        },
        "failure_rollback": {"passed": rollback_valid},
        "tendon_aponeurosis": {
            "implemented": False,
            "source_blocker": (
                "No single tendon/aponeurosis source was found that supplies both an independently "
                "identified parameter set for the same reference configuration and a validated "
                "muscle-interface traction/power transfer law; Engelhardt et al. 2025 models tendon "
                "attachments through boundary constraints rather than a constitutive interface."
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Qualify the source-complete skeletal GASAM continuum route."
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = qualify()
    payload = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
