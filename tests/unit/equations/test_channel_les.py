#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.equations._channel_les import (
    channel_les_filter,
    compile_channel_les,
    CompiledChannelLESDynamics,
)


def _base_channel(*, wall_velocity=0.0):
    space = phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.FourierBasisPlan(6),
            phx.discretization.ChebyshevBasisPlan(9),
            phx.discretization.FourierBasisPlan(4),
        ),
        axis_names=("x", "y", "z"),
        field_name="velocity",
    ).prepare(
        (
            phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi),
            phx.discretization.AxisDomain.interval(-1.0, 1.0),
            phx.discretization.AxisDomain.periodic(0.0, 4.0 * jnp.pi),
        )
    )
    walls = (wall_velocity, 0.0, 0.0)
    stokes = phx.discretization.ChannelStokesPlan(
        space,
        0.05,
        lower_wall_velocity=walls,
        upper_wall_velocity=walls,
    )
    base = phx.equations.compile_channel_flow(
        phx.equations.IncompressibleFlowProblem(3, 0.05),
        stokes,
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.PaddingDealiasingPlan(2)
        ),
    )
    return space, base


def _prepared_model(base, model):
    provenance = phx.equations.LESParameterProvenance(
        channel_les_filter(base.discretization),
        base.discretization.prepared_id,
        "wall-resolved-channel",
        source_kind="user",
        evidence_ids=(),
    )
    return model.prepare(provenance)


def _channel_les(coefficient=0.5, model_type=phx.equations.WALELESPlan):
    space, base = _base_channel()
    return (
        space,
        base,
        compile_channel_les(
            base,
            _prepared_model(base, model_type(coefficient)),
        ),
    )


def _vortex_state(space, dynamics, amplitude=0.02):
    x = space.axes[0].nodes[:, None, None]
    y = space.axes[1].nodes[None, :, None]
    envelope = (1.0 - y**2) ** 2
    streamwise = -4.0 * y * (1.0 - y**2) * jnp.sin(x)
    wall_normal = -envelope * jnp.cos(x)
    physical = jnp.zeros(space.physical_shape + (3,))
    physical = physical.at[..., 0].set(
        amplitude * jnp.broadcast_to(streamwise, space.physical_shape)
    )
    physical = physical.at[..., 1].set(
        amplitude * jnp.broadcast_to(wall_normal, space.physical_shape)
    )
    return dynamics.project_state(physical)


def test_channel_les_filter_widths_are_resolved_anisotropic_and_noncommuting():
    _, _, dynamics = _channel_les()
    geometry = dynamics.filter_geometry
    widths = geometry.directional_widths
    retained = dynamics.discretization

    assert widths.shape == (
        1,
        dynamics.spatial_method.dealiasing.evaluation.physical_shape[1],
        1,
        3,
    )
    np.testing.assert_allclose(
        geometry.streamwise_width,
        retained.axes[0].length / retained.axes[0].physical_count,
    )
    np.testing.assert_allclose(
        geometry.spanwise_width,
        retained.axes[2].length / retained.axes[2].physical_count,
    )
    assert not np.isclose(
        float(geometry.streamwise_width), float(geometry.spanwise_width)
    )
    assert np.ptp(np.asarray(geometry.wall_normal_widths)) > 0.0
    assert np.all(np.asarray(widths) > 0.0)
    assert dynamics.model.provenance.resolved_filter.commutation_status == "unmodeled"
    assert float(geometry.noncommutation_evidence) > 0.0

    constant = jnp.ones(dynamics.spatial_method.dealiasing.evaluation.physical_shape)
    evidence = geometry.wall_normal_scale_commutator(constant)
    assert jnp.max(jnp.abs(evidence)) > 0.0


def test_channel_les_uses_all_mixed_velocity_derivatives():
    space, _, dynamics = _channel_les()
    state = _vortex_state(space, dynamics, amplitude=1.0)
    evaluated = dynamics.evaluate_subgrid(state)
    grid = dynamics.spatial_method.dealiasing.evaluation
    x = grid.axes[0].nodes[:, None, None]
    y = grid.axes[1].nodes[None, :, None]
    expected = {
        (0, 0): -4.0 * y * (1.0 - y**2) * jnp.cos(x),
        (0, 1): (-4.0 + 12.0 * y**2) * jnp.sin(x),
        (1, 0): (1.0 - y**2) ** 2 * jnp.sin(x),
        (1, 1): 4.0 * y * (1.0 - y**2) * jnp.cos(x),
    }
    for indices, exact in expected.items():
        np.testing.assert_allclose(
            np.asarray(evaluated.velocity_gradient[..., indices[0], indices[1]]),
            np.asarray(jnp.broadcast_to(exact, grid.physical_shape)),
            atol=2e-5,
            rtol=2e-5,
        )
    np.testing.assert_allclose(evaluated.velocity_gradient[..., 2], 0.0, atol=2e-5)
    np.testing.assert_allclose(evaluated.velocity_gradient[..., :, 2], 0.0, atol=2e-5)


def test_wale_has_cubic_near_wall_scaling_on_manufactured_gradients():
    _, _, dynamics = _channel_les(coefficient=0.6)
    count = dynamics.filter_geometry.wall_normal_widths.size
    distances = jnp.zeros((count,)).at[:4].set(jnp.asarray((0.0, 1.0e-4, 2.0e-4, 4.0e-4)))
    gradient = jnp.zeros((1, count, 1, 3, 3))
    gradient = gradient.at[..., 0, 0].set(distances[None, :, None])
    gradient = gradient.at[..., 0, 1].set(1.0)
    result = dynamics.model.evaluate(
        phx.equations.AlgebraicLESInputs(
            gradient,
            dynamics.filter_geometry.filter_scale,
        )
    )
    equivalent_squared = dynamics.filter_geometry.filter_scale.equivalent_width**2
    normalized = result.kinematic_viscosity / equivalent_squared

    assert float(normalized[0, 0, 0]) == 0.0
    np.testing.assert_allclose(
        np.asarray(normalized[0, 1:4, 0] / normalized[0, 1, 0]),
        np.asarray((1.0, 8.0, 64.0)),
        rtol=2e-5,
    )


@pytest.mark.parametrize(
    "model_type",
    (
        phx.equations.SmagorinskyLESPlan,
        phx.equations.WALELESPlan,
        phx.equations.VremanLESPlan,
        phx.equations.AMDLESPlan,
    ),
)
def test_channel_les_accepts_exactly_bound_prepared_algebraic_models(model_type):
    _, base = _base_channel()
    compiled = compile_channel_les(base, _prepared_model(base, model_type(0.2)))
    assert isinstance(compiled, CompiledChannelLESDynamics)
    assert compiled.base_compilation_id == base.compilation_id
    assert compiled.les_prepared_id == compiled.model.prepared_id
    assert compiled.state_shape == base.state_shape
    assert compiled.source_hash == base.source_hash

    wrong_provenance = phx.equations.LESParameterProvenance(
        channel_les_filter(base.discretization),
        "different-grid",
        "wall-resolved-channel",
        source_kind="user",
        evidence_ids=(),
    )
    with pytest.raises(ValueError, match="retained channel grid"):
        compile_channel_les(base, model_type(0.2).prepare(wrong_provenance))


def test_zero_coefficient_channel_les_matches_no_les_rhs_and_step():
    space, base = _base_channel()
    zero = compile_channel_les(
        base,
        _prepared_model(base, phx.equations.WALELESPlan(0.0)),
    )
    state = _vortex_state(space, base)
    np.testing.assert_allclose(
        np.asarray(zero.nonlinear(0.0, state, None)),
        np.asarray(base.nonlinear(0.0, state, None)),
        atol=0.0,
        rtol=0.0,
    )
    restriction = zero.explicit_restriction(state)
    assert restriction.diffusive_rate == 0.0
    assert restriction.advective_rate > 0.0
    assert bool(restriction.active)

    times = jnp.asarray((0.0, 5.0e-4))
    baseline = phx.solver.solve_channel_sbdf2(base, state, times)
    modeled = phx.solver.solve_channel_sbdf2(zero, state, times)
    np.testing.assert_allclose(
        modeled.velocity, baseline.velocity, atol=2e-11, rtol=2e-11
    )
    np.testing.assert_allclose(
        modeled.pressure, baseline.pressure, atol=2e-11, rtol=2e-11
    )


def test_channel_les_stress_work_energy_and_explicit_restriction_are_consistent():
    space, _, dynamics = _channel_les()
    state = _vortex_state(space, dynamics)
    evaluated = dynamics.evaluate_subgrid(state)
    local_transfer = -jnp.sum(
        evaluated.specific_deviatoric_stress * evaluated.velocity_gradient,
        axis=(-2, -1),
    )
    np.testing.assert_allclose(
        np.asarray(evaluated.energy_transfer),
        np.asarray(local_transfer),
        rtol=2e-5,
        atol=2e-9,
    )
    assert jnp.min(evaluated.energy_transfer) >= -1e-9

    ledger = dynamics.energy_ledger(state)
    assert bool(ledger.finite)
    assert ledger.molecular_dissipation >= 0.0
    assert ledger.subgrid_transfer >= -1e-9
    np.testing.assert_allclose(
        ledger.resolved_energy_rate,
        ledger.wall_power - ledger.molecular_dissipation - ledger.subgrid_transfer,
    )
    np.testing.assert_allclose(ledger.wall_power, 0.0, atol=2e-7)

    restriction = dynamics.explicit_restriction(state)
    assert bool(restriction.finite)
    assert bool(restriction.active)
    assert restriction.maximum_kinematic_viscosity > 0.0
    assert restriction.diffusive_rate > 0.0
    assert bool(restriction.permits(0.5 * restriction.maximum_step))
    assert not bool(restriction.permits(2.0 * restriction.maximum_step))


def test_channel_les_is_jittable_and_has_a_state_jvp():
    space, _, dynamics = _channel_les()
    state = _vortex_state(space, dynamics)
    compiled_rhs = jax.jit(lambda value: dynamics.nonlinear(0.0, value, None))(state)
    tangent = 0.1 * state
    value, derivative = jax.jvp(
        lambda candidate: dynamics.nonlinear(0.0, candidate, None),
        (state,),
        (tangent,),
    )

    assert compiled_rhs.shape == dynamics.state_shape
    assert value.shape == dynamics.state_shape
    assert derivative.shape == dynamics.state_shape
    assert jnp.all(jnp.isfinite(compiled_rhs))
    assert jnp.all(jnp.isfinite(derivative))


def test_short_channel_les_step_preserves_walls_divergence_and_is_deterministic():
    space, _, dynamics = _channel_les()
    initial = _vortex_state(space, dynamics, amplitude=0.01)
    times = jnp.asarray((0.0, 2.0e-4))
    first = phx.solver.solve_channel_sbdf2(dynamics, initial, times)
    second = phx.solver.solve_channel_sbdf2(dynamics, initial, times)
    final = dynamics.reconstruct_state(first.velocity[-1])
    diagnostics = dynamics.state_diagnostics(first.velocity[-1])

    assert bool(first.successful)
    assert bool(diagnostics.successful)
    assert bool(diagnostics.finite)
    np.testing.assert_allclose(first.velocity, second.velocity, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(final[:, 0], 0.0, atol=2e-9)
    np.testing.assert_allclose(final[:, -1], 0.0, atol=2e-9)
    assert diagnostics.divergence_norm < dynamics.stokes_plan.constraint_tolerance
    assert diagnostics.wall_residual < dynamics.stokes_plan.constraint_tolerance
