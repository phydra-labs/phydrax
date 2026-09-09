#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Run implementation consistency checks; never label these biological validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp

from phydrax.applications.skeletal_muscle.continuum import (
    HeidlaufRoehrle2014Parameters,
    HeidlaufRoehrle2014Plan,
    HeidlaufRoehrle2014QualificationPlan,
    HeidlaufRoehrle2014StressInput,
    UniformFiberArchitecturePlan,
)
from phydrax.discretization import (
    CellMesh,
    MixedFiniteElementConstraintPlan,
    PressureGaugePolicy,
)


def qualify():
    source_id = "qualification-prescribed-gamma"
    architecture = UniformFiberArchitecturePlan("qualification-2014-x").prepare(
        jnp.asarray((1.0, 0.0, 0.0))
    )
    material = HeidlaufRoehrle2014Plan("qualification-2014", source_id).prepare(
        HeidlaufRoehrle2014Parameters.published_table_2(),
        architecture,
        HeidlaufRoehrle2014StressInput(0.7, jnp.zeros(8, dtype=jnp.uint32), source_id),
    )
    rate = jnp.asarray(((0.03, 0.01, 0.0), (-0.01, -0.02, 0.01), (0.0, 0.0, 0.01)))
    plan = HeidlaufRoehrle2014QualificationPlan()
    points = []
    for stretch in (0.8, 1.0, 1.25):
        deformation = jnp.diag(jnp.asarray((stretch, stretch**-0.5, stretch**-0.5)))
        point = plan.evaluate(material, deformation, 7200.0, rate)
        points.append(
            {
                "stretch": stretch,
                "passive_gradient_relative_error": float(
                    point.passive_gradient_relative_error
                ),
                "objectivity_relative_error": float(point.objectivity_relative_error),
                "tangent_difference_relative_error": float(
                    point.tangent_difference_relative_error
                ),
                "active_power_relative_error": float(point.active_power_relative_error),
                "pressure_adjoint_relative_error": float(
                    point.pressure_adjoint_relative_error
                ),
                "passed": bool(point.valid),
            }
        )
    passive = material.with_commit(
        material.propose_active_stress(
            HeidlaufRoehrle2014StressInput(0.0, jnp.ones(8, dtype=jnp.uint32), source_id)
        ).commit()
    )
    mesh = CellMesh.from_tetrahedra(
        jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
    )
    origin = 2 * material.parameters.c10_pa + 4 * material.parameters.c01_pa
    mixed = passive.prepare_qualified_mixed(
        MixedFiniteElementConstraintPlan(mesh, PressureGaugePolicy("mean-zero")),
        pressure_origin_pa=origin,
    )
    rest = mixed.problem.residual(mixed.problem.state_space.zeros())
    rest_norm = jnp.sqrt(sum(jnp.vdot(value, value).real for value in rest))
    rejected = material.propose_active_stress(
        HeidlaufRoehrle2014StressInput(
            float("nan"), jnp.ones(8, dtype=jnp.uint32), source_id
        )
    ).commit()
    return {
        "scope": "source-material and native mixed-FE implementation consistency only",
        "physiological_validation": False,
        "integrated_2014_cell_model": False,
        "source": "doi:10.3389/fphys.2014.00498; Eq.1-2, Table2",
        "points": points,
        "mixed_rest_residual_norm": float(rest_norm),
        "assembled_inf_sup_constant": float(mixed.inf_sup.inf_sup_constant),
        "rollback_applied": bool(rejected.rollback_applied),
        "passed": all(point["passed"] for point in points)
        and float(rest_norm) < 0.01
        and bool(rejected.rollback_applied),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = qualify()
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n", encoding="utf-8")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
