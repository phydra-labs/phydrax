#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.solid_mechanics._mixed_hyperelastic import (
    MixedHyperelasticLaw,
    MixedHyperelasticModel,
    prepare_mixed_hyperelastic_problem,
)
from phydrax.discretization import CellBlock, CellMesh
from phydrax.discretization.fem._mixed_constraint import (
    mixed_inf_sup_diagnostic,
    MixedFiniteElementConstraintPlan,
    MixedPressureStabilization,
    PressureGaugePolicy,
)


def _triangle_mesh():
    return CellMesh.from_triangles(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        jnp.asarray(((0, 1, 2),), dtype=jnp.int32),
    )


def _quadrilateral_mesh():
    return CellMesh(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        (
            CellBlock(
                "quad",
                "quadrilateral",
                jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
                global_ids=jnp.asarray((0,), dtype=jnp.int64),
            ),
        ),
    )


def _model(*, bulk_modulus=None):
    def isochoric_energy(deformation_bar):
        dimension = deformation_bar.shape[0]
        return jnp.sum(deformation_bar * deformation_bar) - dimension

    return MixedHyperelasticModel(
        MixedHyperelasticLaw(
            isochoric_energy,
            lambda deformation: jnp.linalg.det(deformation) - 1.0,
            bulk_modulus=bulk_modulus,
            minimum_jacobian=1.0e-8,
        )
    )


def test_mean_zero_and_pinned_gauges_remove_only_the_constant_pressure_mode():
    pressure = jnp.asarray((2.0, 4.0, 7.0, -1.0))
    mean_zero = PressureGaugePolicy("mean-zero")
    pinned = PressureGaugePolicy("pinned", pinned_dof=2)
    mean_projected = mean_zero.project(pressure)
    pin_projected = pinned.project(pressure)

    np.testing.assert_allclose(jnp.sum(mean_projected), 0.0, atol=1e-12)
    np.testing.assert_allclose(pin_projected[2], 0.0, atol=0.0)
    np.testing.assert_allclose(
        pin_projected[:, None] - pin_projected[None, :],
        pressure[:, None] - pressure[None, :],
        atol=0.0,
    )
    assert bool(mean_zero.evidence(mean_projected).valid)
    assert bool(pinned.evidence(pin_projected).valid)


def test_inf_sup_evidence_distinguishes_stable_and_unstable_spaces():
    stable_constraint = jnp.asarray(((1.0, 0.0), (-1.0, 0.0)))
    unstable_constraint = jnp.zeros((2, 2))
    gauge = PressureGaugePolicy("mean-zero")

    stable = mixed_inf_sup_diagnostic(
        stable_constraint,
        stable_constraint.T,
        gauge,
        formulation="exact",
    )
    unstable = mixed_inf_sup_diagnostic(
        unstable_constraint,
        unstable_constraint.T,
        gauge,
        formulation="exact",
    )

    assert stable.numerical_rank == 1
    assert stable.pressure_nullity == 1
    assert stable.gauge_resolves_nullspace
    assert stable.inf_sup_constant > 0.0
    assert stable.stable
    assert stable.locking_safe
    assert not unstable.stable
    assert not unstable.locking_safe


def test_mixed_plan_requires_physical_gauge_policy_and_refuses_unverified_stabilization():
    mesh = _triangle_mesh()

    with pytest.raises(ValueError, match="requires an explicit gauge"):
        MixedFiniteElementConstraintPlan(mesh, PressureGaugePolicy("none"))
    with pytest.raises(ValueError, match="must use the explicit no-gauge"):
        MixedFiniteElementConstraintPlan(
            mesh,
            PressureGaugePolicy("mean-zero"),
            bulk_modulus=100.0,
        )
    with pytest.raises(ValueError, match="refuses unverified pressure stabilization"):
        MixedFiniteElementConstraintPlan(
            mesh,
            PressureGaugePolicy("mean-zero"),
            stabilization=MixedPressureStabilization(
                "pressure-laplacian", coefficient=1.0e-3
            ),
        )


def test_taylor_hood_exact_problem_prepares_coupled_blocks_gauge_and_locking_evidence():
    plan = MixedFiniteElementConstraintPlan(
        _triangle_mesh(),
        PressureGaugePolicy("mean-zero"),
    )
    prepared = prepare_mixed_hyperelastic_problem(_model(), plan)
    zero = prepared.problem.state_space.zeros()
    pressure = jnp.arange(zero[1].size, dtype=zero[1].dtype) + 3.0
    evaluation = prepared.evaluate((zero[0], pressure))

    assert prepared.spaces.pair_names == ("taylor-hood",)
    assert prepared.spaces.displacement_degree == 2
    assert prepared.spaces.pressure_degree == 1
    assert prepared.spaces.stabilization_absent
    assert prepared.spaces.stabilization_refused
    assert prepared.spaces.locking_safe
    assert prepared.inf_sup.stable
    assert prepared.inf_sup.locking_safe
    assert prepared.pressure_operator is None
    assert prepared.problem.block_dependency_graph() == (
        (True, True),
        (True, False),
    )
    np.testing.assert_allclose(jnp.sum(evaluation.gauged_pressure), 0.0, atol=1e-12)
    assert bool(evaluation.gauge.valid)
    assert bool(evaluation.finite)


def test_q2_q1_finite_bulk_problem_has_physical_pressure_block_without_gauge_or_stabilization():
    bulk_modulus = 250.0
    plan = MixedFiniteElementConstraintPlan(
        _quadrilateral_mesh(),
        PressureGaugePolicy("none"),
        bulk_modulus=bulk_modulus,
    )
    prepared = prepare_mixed_hyperelastic_problem(
        _model(bulk_modulus=bulk_modulus),
        plan,
    )
    zero = prepared.problem.state_space.zeros()
    pressure_direction = (jnp.zeros_like(zero[0]), jnp.ones_like(zero[1]))
    pressure_image = prepared.problem.block_linearization_operator(zero).mv(
        pressure_direction
    )[1]

    assert prepared.spaces.pair_names == ("q2-q1",)
    assert prepared.spaces.displacement_degree == 2
    assert prepared.spaces.pressure_degree == 1
    assert prepared.spaces.stabilization_absent
    assert prepared.pressure_operator is not None
    assert prepared.problem.block_dependency_graph() == (
        (True, True),
        (True, True),
    )
    assert jnp.linalg.norm(pressure_image) > 0.0
    assert prepared.gauge.mode == "none"
    assert prepared.inf_sup.stable
