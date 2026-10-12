#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.discretization.mpm import (
    MPMRuntimeState,
    MPMVolumetricStabilizationKind,
)
from phydrax.equations import MaterialPointArguments
from phydrax.solver import PreparedImplicitMPMDynamics


def _problem(
    kind: MPMVolumetricStabilizationKind,
    sparse: bool = False,
) -> tuple[PreparedImplicitMPMDynamics, MPMRuntimeState, MaterialPointArguments]:
    bounds = jnp.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=jnp.float64)
    grid_plan = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(8, periodic=True, endpoint=False),
            phx.discretization.UniformAxisSpec(8, periodic=True, endpoint=False),
        ),
        axis_names=("x", "y"),
    )
    grid = grid_plan.prepare(bounds)
    position = jnp.asarray(
        [[0.27, 0.31], [0.34, 0.36], [0.43, 0.38], [0.36, 0.52]],
        dtype=jnp.float64,
    )
    volume = jnp.asarray([0.01, 0.015, 0.012, 0.009], dtype=jnp.float64)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(4, dtype=jnp.int32), volume, ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(2)
    ).prepare(particles)
    storage = None
    if sparse:
        index_space = grid_plan.prepare_index_space(bounds)
        storage = phx.discretization.BlockSparseMPMNodalStoragePlan(
            phx.discretization.SparseBlockTopologyPlan(
                index_space, (4, 4), 4, layout=index_space.vertices()
            )
        )
    material = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2)
    arguments = MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 200.0)
    )
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR("volumetric-implicit", material),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(
            volumetric_stabilization=phx.discretization.MPMVolumetricStabilizationPlan(
                kind
            )
        ),
        phx.discretization.MPMParticleDomainPlan(
            bounds, periodic=(True, True), support_margin=0.0
        ),
        nodal_storage=storage,
    )
    state = compiled.initialize_state(
        position, jnp.zeros_like(position), volume, arguments
    )
    return PreparedImplicitMPMDynamics(compiled.dynamics), state, arguments


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "block-sparse"])
def test_implicit_b_bar_accepts_rest(sparse: bool) -> None:
    implicit, state, arguments = _problem("b_bar", sparse)
    result = implicit.step_detailed(state, 0.001, arguments)

    assert bool(result.successful)
    assert bool(result.diagnostics.tangent_successful)
    assert float(result.diagnostics.residual_norm) < 1.0e-10
    np.testing.assert_allclose(
        result.accepted_state.particles.deformation_gradient,
        state.particles.deformation_gradient,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert float(result.accepted_state.time) == 0.001


@pytest.mark.parametrize("kind", ["b_bar", "f_bar"], ids=["b-bar", "f-bar"])
@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "block-sparse"])
def test_implicit_stabilized_residual_directional_tangent(
    kind: MPMVolumetricStabilizationKind, sparse: bool
) -> None:
    implicit, state, arguments = _problem(kind, sparse)
    prepared = implicit.prepare_residual(state, 0.002, arguments)
    phase = jnp.arange(prepared.initial_velocity.size, dtype=jnp.float64).reshape(
        prepared.initial_velocity.shape
    )
    velocity = prepared.initial_velocity + 0.025 * jnp.sin(0.31 * phase)
    direction = jnp.cos(0.17 * phase)

    def residual(value: Array) -> Array:
        return prepared.evaluate(value)[0]

    value, tangent = jax.jvp(residual, (velocity,), (direction,))
    epsilon = jnp.asarray(1.0e-5, dtype=velocity.dtype)
    plus, plus_valid = prepared.evaluate(velocity + epsilon * direction)
    minus, minus_valid = prepared.evaluate(velocity - epsilon * direction)
    finite_difference = (plus - minus) / (2.0 * epsilon)

    assert bool(plus_valid & minus_valid)
    assert bool(jnp.all(jnp.isfinite(value)))
    assert bool(jnp.all(jnp.isfinite(tangent)))
    np.testing.assert_allclose(tangent, finite_difference, rtol=2.0e-6, atol=2.0e-9)


def test_implicit_b_bar_changes_nonuniform_volumetric_tangent() -> None:
    b_bar, state, arguments = _problem("b_bar")
    ordinary, ordinary_state, ordinary_arguments = _problem("none")
    stabilized_residual = b_bar.prepare_residual(state, 0.002, arguments)
    ordinary_residual = ordinary.prepare_residual(
        ordinary_state, 0.002, ordinary_arguments
    )
    phase = jnp.arange(stabilized_residual.initial_velocity.size, dtype=jnp.float64)
    direction = jnp.sin(0.37 * phase).reshape(stabilized_residual.initial_velocity.shape)
    _, stabilized_tangent = jax.jvp(
        lambda value: stabilized_residual.evaluate(value)[0],
        (stabilized_residual.initial_velocity,),
        (direction,),
    )
    _, ordinary_tangent = jax.jvp(
        lambda value: ordinary_residual.evaluate(value)[0],
        (ordinary_residual.initial_velocity,),
        (direction,),
    )

    assert float(jnp.linalg.norm(stabilized_tangent - ordinary_tangent)) > 1.0e-5


@pytest.mark.parametrize("kind", ["b_bar", "f_bar"], ids=["b-bar", "f-bar"])
def test_implicit_stabilization_plan_mismatch_refused(
    kind: MPMVolumetricStabilizationKind,
) -> None:
    implicit, _, _ = _problem(kind)
    with pytest.raises(ValueError, match="volumetric stabilization plans must match"):
        PreparedImplicitMPMDynamics(implicit.explicit, phx.solver.ImplicitMPMMethodPlan())


def test_implicit_f_bar_rejects_inverted_initial_physical_deformation() -> None:
    implicit, state, arguments = _problem("f_bar")
    physical = state.particles.physical_deformation_gradient
    if physical is None:
        raise RuntimeError("F-bar initialization must provide physical deformation.")
    inverted = physical.at[0, 0, 0].set(-1.0)
    state = eqx.tree_at(
        lambda value: value.particles.physical_deformation_gradient, state, inverted
    )
    prepared = implicit.prepare_residual(state, 0.001, arguments)
    _, valid = prepared.evaluate(prepared.initial_velocity)

    assert not bool(valid)


def test_commercial_b_bar_route_superset_includes_cell_gradient_derivatives() -> None:
    implicit, state, _ = _problem("b_bar")
    dynamics = implicit.explicit
    plan = phx.solver.MPMRouteSupersetPlan(
        dynamics.splat, dynamics=dynamics, minimum_margin=1.0e-12
    )
    position = state.particles.position
    base = plan.build(position, state.assignment_input)
    cell = base.base_routes.cell_average_gradients
    if cell is None:
        raise RuntimeError("B-bar route superset must include cell-average gradients.")
    cotangents = (
        jnp.ones_like(base.base_routes.stencil.weights),
        jnp.ones_like(base.base_routes.weight_gradients),
        jnp.ones_like(base.base_routes.route_offsets),
        jnp.ones_like(cell),
    )
    direction = jnp.asarray(
        [[0.001, -0.002], [-0.003, 0.001], [0.002, 0.001], [0.001, 0.002]],
        dtype=position.dtype,
    )
    result = plan.linearize(
        base,
        position,
        state.particles.deformation_gradient,
        state.assignment_input,
        direction,
        jnp.zeros_like(state.particles.deformation_gradient),
        None,
        cotangents,
    )

    assert bool(result.successful)
    assert result.primal_cell_average_gradients is not None
    assert result.cell_average_gradient_jvp is not None
    np.testing.assert_array_equal(result.primal_cell_average_gradients, cell)
    forward = (
        jnp.vdot(result.weight_jvp, cotangents[0])
        + jnp.vdot(result.gradient_jvp, cotangents[1])
        + jnp.vdot(result.offset_jvp, cotangents[2])
        + jnp.vdot(result.cell_average_gradient_jvp, cotangents[3])
    )
    reverse = jnp.vdot(result.position_transpose, direction)
    np.testing.assert_allclose(forward, reverse, rtol=1.0e-10, atol=1.0e-12)
