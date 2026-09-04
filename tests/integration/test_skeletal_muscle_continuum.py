#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp

from phydrax.applications.skeletal_muscle.continuum import (
    EngelhardtGasam2025Parameters,
    EngelhardtGasam2025Plan,
    solve_manufactured_rest,
    UniformFiberArchitecturePlan,
)
from phydrax.discretization import (
    CellMesh,
    MixedFiniteElementConstraintPlan,
    PressureGaugePolicy,
)


def test_exact_mixed_taylor_hood_manufactured_rest_solves_and_commits():
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
    architecture = UniformFiberArchitecturePlan("manufactured-x-fibers").prepare(
        jnp.asarray((1.0, 0.0, 0.0))
    )
    material = EngelhardtGasam2025Plan("manufactured-rest").prepare(
        EngelhardtGasam2025Parameters.published_multiload_fit(),
        architecture,
        0.0,
    )
    mixed_plan = MixedFiniteElementConstraintPlan(
        mesh,
        PressureGaugePolicy("mean-zero"),
    )
    qualified = material.prepare_qualified_mixed(mixed_plan)
    candidate = solve_manufactured_rest(qualified)
    commit = candidate.commit()

    assert qualified.prepared.spaces.pair_names == ("taylor-hood",)
    assert qualified.prepared.problem.block_dependency_graph() == (
        (True, True),
        (True, False),
    )
    assert bool(qualified.qualification.valid)
    assert commit.committed
    assert bool(commit.evidence.valid)
    assert (
        commit.evidence.final_residual_norm
        <= commit.evidence.initial_residual_norm
    )
    assert jnp.linalg.norm(commit.state[0]) == 0.0
    assert jnp.linalg.norm(commit.state[1]) == 0.0
