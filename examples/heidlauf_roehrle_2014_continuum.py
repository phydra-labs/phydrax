#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prescribed 2014 source stress and native mixed-FE manufactured rest.

This driver does not simulate the source cell model or reproduce physiological
traces. It demonstrates one independent mechanical force owner and rollback.
"""

from __future__ import annotations

import hashlib
import json

import jax.numpy as jnp
import numpy as np

from phydrax.applications.skeletal_muscle.continuum import (
    HeidlaufRoehrle2014Parameters,
    HeidlaufRoehrle2014Plan,
    HeidlaufRoehrle2014StressInput,
    UniformFiberArchitecturePlan,
)
from phydrax.discretization import (
    CellMesh,
    MixedFiniteElementConstraintPlan,
    PressureGaugePolicy,
)
from phydrax.nonlinear import NewtonKrylov, NonlinearTermination


def run():
    def source(gamma, sample):
        digest = hashlib.sha256(
            f"manufactured-protocol:{sample}:{gamma}".encode()
        ).digest()
        token = jnp.asarray(np.frombuffer(digest, dtype=">u4").astype(np.uint32))
        return HeidlaufRoehrle2014StressInput(gamma, token, "manufactured-protocol")

    parameters = HeidlaufRoehrle2014Parameters.published_table_2()
    architecture = UniformFiberArchitecturePlan("manufactured-x").prepare(
        jnp.asarray((1.0, 0.0, 0.0))
    )
    material = HeidlaufRoehrle2014Plan(
        "manufactured-2014", "manufactured-protocol"
    ).prepare(
        parameters,
        architecture,
        source(0.0, 0),
    )
    mesh = CellMesh.from_tetrahedra(
        jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
    )
    origin = 2 * parameters.c10_pa + 4 * parameters.c01_pa
    mixed = material.prepare_qualified_mixed(
        MixedFiniteElementConstraintPlan(mesh, PressureGaugePolicy("mean-zero")),
        pressure_origin_pa=origin,
    )
    solved = NewtonKrylov().solve(
        mixed.problem.as_nonlinear_problem(),
        mixed.problem.state_space.zeros(),
        termination=NonlinearTermination(),
    )
    evaluation = mixed.evaluate(solved.state)
    accepted = material.propose_active_stress(source(0.65, 1)).commit(
        successful=solved.successful
    )
    active = material.with_commit(accepted)
    rejected = active.propose_active_stress(source(float("nan"), 2)).commit()
    rolled_back = active.with_commit(rejected)
    response = active.evaluate(jnp.eye(3), origin)
    residual_norm = jnp.sqrt(
        sum(jnp.vdot(value, value).real for value in evaluation.residual)
    )
    passed = (
        bool(solved.successful)
        and bool(evaluation.valid)
        and bool(accepted.committed)
        and bool(rejected.rollback_applied)
        and bool(
            rolled_back.state.normalized_active_stress
            == active.state.normalized_active_stress
        )
    )
    return {
        "scope": "prescribed-normalized-stress material; manufactured numerical example only",
        "source": "doi:10.3389/fphys.2014.00498; Eq.1-2, Table2",
        "force_owner": response.evidence.force_owner,
        "prepared_id": material.prepared_id,
        "numeric_revision": active.numeric_revision().revision_id,
        "pressure_origin_pa": float(origin),
        "mixed_rest_residual_norm": float(residual_norm),
        "assembled_inf_sup_constant": float(mixed.inf_sup.inf_sup_constant),
        "active_nominal_stress_pa": float(response.active_nominal_stress_pa),
        "rollback_applied": bool(rejected.rollback_applied),
        "passed": passed,
    }


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)
