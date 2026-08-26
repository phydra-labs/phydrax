#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _cell_grid(shape):
    dimension = len(shape)
    return phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(count) for count in shape),
        axis_names=tuple("xyz"[:dimension]),
    ).prepare(jnp.asarray([[0.0] * dimension, [1.0] * dimension]))


def test_portable_fd_checkpoint_roundtrips_fields_auxiliary_and_identity(tmp_path):
    plan = phx.discretization.FDCheckpointPlan(
        ("grid-id", "operator-id"),
        "ssprk3",
        boundary_program_id="boundary-id",
        amr_trace_id="amr-trace",
        partition_id="partition-id",
    )
    path = tmp_path / "state.phydrax"
    fields = {
        "pressure": jnp.arange(8.0).reshape((2, 4)),
        "pressure_split_0": jnp.ones((2, 4)),
    }
    auxiliary = {
        "amr_active": jnp.asarray([True, False]),
        "pml_velocity_split": jnp.zeros((3, 2)),
    }

    phx.discretization.write_fd_checkpoint(
        path,
        plan,
        0.75,
        fields,
        auxiliary=auxiliary,
        metadata={"step": 12},
    )
    checkpoint = phx.discretization.read_fd_checkpoint(path, plan)

    np.testing.assert_allclose(checkpoint.time, 0.75)
    np.testing.assert_allclose(checkpoint.field("pressure"), fields["pressure"])
    np.testing.assert_allclose(
        checkpoint.auxiliary_value("pml_velocity_split"),
        auxiliary["pml_velocity_split"],
    )
    assert checkpoint.plan_id == plan.plan_id

    incompatible = phx.discretization.FDCheckpointPlan(
        ("different-grid",),
        "ssprk3",
    )
    with pytest.raises(ValueError, match="incompatible"):
        phx.discretization.read_fd_checkpoint(path, incompatible)


def test_boundary_halo_and_transfer_actions_have_exact_discrete_vjps():
    boundary = phx.discretization.CellGhostBoundary(
        0,
        "dirichlet",
        "neumann",
        0.25,
        lower_width=2,
        upper_width=2,
    )
    values = jnp.asarray([1.0, 2.0, 4.0, 7.0])
    lower = jnp.asarray(0.3)
    upper = jnp.asarray(-0.2)
    cotangent = jnp.linspace(-1.0, 1.0, 8)
    adjoint = phx.discretization.FDActionAdjointPlan(
        boundary.fill,
        action_id="boundary-fill",
    )

    report = adjoint.identity_report(
        (values, lower, upper),
        0,
        jnp.asarray([0.2, -0.1, 0.4, 0.3]),
        cotangent,
    )

    assert report.passed
    assert report.residual < 1e-12

    fine = _cell_grid((8,))
    coarse = _cell_grid((4,))
    transfer = phx.discretization.StructuredTransferPlan(fine, coarse)
    restriction, _ = transfer.prepare(
        fine.field_space("fine").vector_space,
        coarse.field_space("coarse").vector_space,
    )
    transfer_adjoint = phx.discretization.FDActionAdjointPlan(
        restriction.mv,
        action_id="restriction",
    )
    transfer_report = transfer_adjoint.identity_report(
        (jnp.arange(8.0),),
        0,
        jnp.linspace(0.1, 0.8, 8),
        jnp.linspace(-0.5, 0.5, 4),
    )

    assert transfer_report.passed


def test_checkpointed_time_discrete_adjoint_matches_closed_form_gradient():
    steps = 20
    dt = 0.01
    parameter = jnp.asarray(0.7)
    initial = jnp.asarray([1.2, -0.4])
    plan = phx.discretization.CheckpointedFDAdjointPlan(
        lambda time, state, step_size, rate: state + step_size * rate * state,
        steps,
        checkpointing="recompute",
    )

    result = plan.value_and_gradient(
        initial,
        parameter,
        0.0,
        dt,
        lambda final, rate: 0.5 * jnp.sum(final**2),
    )
    amplification = (1.0 + dt * parameter) ** steps
    expected_initial = amplification**2 * initial
    expected_parameter = (
        jnp.sum(initial**2) * steps * dt * (1.0 + dt * parameter) ** (2 * steps - 1)
    )

    np.testing.assert_allclose(
        result.initial_gradient,
        expected_initial,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        result.parameter_gradient,
        expected_parameter,
        rtol=2e-12,
        atol=2e-12,
    )


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_structured_cochain_bridge_satisfies_boundary_of_boundary_identity(dimension):
    bridge = phx.discretization.StructuredCochainBridge(_cell_grid((3,) * dimension))
    values = jnp.arange(bridge.cochain.cell_counts[0], dtype=float)

    first = bridge.exterior_derivative(0, values)

    if dimension > 1:
        second = bridge.exterior_derivative(1, first)
        np.testing.assert_allclose(second, 0.0, rtol=0.0, atol=0.0)
    components = bridge.unpack(0, values)
    np.testing.assert_allclose(bridge.pack(0, components), values)


def test_prepared_maxwell_preserves_constraints_and_material_gradients():
    bridge = phx.discretization.StructuredCochainBridge(_cell_grid((3, 3, 3)))
    degree_zero = bridge.cochain.cell_counts[0]
    degree_one = bridge.cochain.cell_counts[1]
    degree_two = bridge.cochain.cell_counts[2]
    permittivity = 1.0 + 0.2 * (jnp.arange(degree_one, dtype=float) + 1.0) / degree_one
    permeability = 1.0 + 0.1 * (jnp.arange(degree_two, dtype=float) + 1.0) / degree_two
    electric = jnp.sin(jnp.arange(degree_one, dtype=float) / 7.0)
    magnetic = bridge.exterior_derivative(1, electric)
    charge = bridge.codifferential(1, permittivity * electric)
    current = bridge.exterior_derivative(
        0,
        jnp.cos(jnp.arange(degree_zero, dtype=float) / 5.0),
    )

    def current_source(time, coordinates, args):
        del coordinates
        return args["amplitude"] * jnp.cos(time) * current

    maxwell = phx.solver.CompatibleMaxwellPlan(
        bridge,
        constitutive=phx.solver.maxwell.DiagonalMaxwellConstitutivePlan(
            permittivity=permittivity,
            permeability=permeability,
        ),
        current_source=current_source,
    ).prepare()
    state = maxwell.pack(permittivity * electric, magnetic, charge)
    step_size = 0.1 * maxwell.stable_dt
    diagnostics = maxwell.diagnostics(
        0.0,
        state,
        {"amplitude": jnp.asarray(0.3)},
        step_size=step_size,
    )
    stepped = maxwell.leapfrog_step(
        0.0,
        state,
        step_size,
        {"amplitude": jnp.asarray(0.3)},
    )

    np.testing.assert_allclose(
        maxwell.electric_constraint(stepped),
        maxwell.electric_constraint(state),
        rtol=0.0,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        maxwell.magnetic_constraint(stepped),
        0.0,
        rtol=0.0,
        atol=2e-11,
    )
    assert diagnostics.electric_constraint_linf < 2e-11
    assert diagnostics.magnetic_constraint_linf < 2e-11
    assert diagnostics.gauss_rate_linf < 2e-11
    assert jnp.abs(diagnostics.power_balance_residual) < 2e-11
    np.testing.assert_allclose(diagnostics.step_fraction, 0.1)
    assert jnp.isfinite(maxwell.energy(stepped))

    def material_energy(epsilon):
        prepared = phx.solver.CompatibleMaxwellPlan(
            bridge,
            constitutive=phx.solver.maxwell.DiagonalMaxwellConstitutivePlan(
                permittivity=epsilon,
                permeability=permeability,
            ),
        ).prepare()
        displacement = epsilon * electric
        state_ = prepared.pack(
            displacement,
            magnetic,
            bridge.codifferential(1, displacement),
        )
        return prepared.energy(state_)

    energy_gradient = jax.grad(material_energy)(permittivity)
    np.testing.assert_allclose(
        energy_gradient,
        0.5 * bridge.cochain.hodge_stars[1] * electric**2,
        rtol=2e-12,
        atol=2e-12,
    )

    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="stable_dt"):
        invalid = maxwell.leapfrog_step(
            0.0,
            state,
            1.01 * maxwell.stable_dt,
            {"amplitude": jnp.asarray(0.3)},
        )
        jax.block_until_ready(invalid.primary.electric_displacement)


def test_elastic_energy_and_incompressible_projection_are_compatible():
    bridge = phx.discretization.StructuredCochainBridge(_cell_grid((3, 3, 3)))

    elasticity = phx.solver.CompatibleElasticityDynamics(
        bridge,
        wave_speed=1.3,
    )
    displacement = jnp.sin(jnp.arange(bridge.cochain.cell_counts[0], dtype=float) / 7.0)
    velocity = jnp.cos(jnp.arange(bridge.cochain.cell_counts[0], dtype=float) / 5.0)
    elastic_state = elasticity.pack(displacement, velocity)
    elastic_drift = elasticity.drift(elastic_state)
    energy_gradient = jax.grad(
        lambda displacement_, velocity_: elasticity.energy(
            elasticity.pack(displacement_, velocity_)
        ),
        argnums=(0, 1),
    )(displacement, velocity)
    energy_rate = jnp.vdot(energy_gradient[0], elastic_drift.displacement) + jnp.vdot(
        energy_gradient[1], elastic_drift.velocity
    )
    np.testing.assert_allclose(energy_rate, 0.0, rtol=0.0, atol=2e-9)

    projection = phx.solver.CompatibleIncompressibleProjection(bridge)
    raw_velocity = jnp.sin(jnp.arange(bridge.cochain.cell_counts[1], dtype=float) / 3.0)
    projected = eqx.filter_jit(projection.project)(raw_velocity)

    assert jnp.linalg.norm(projected.divergence_before) > 1e-3
    assert jnp.linalg.norm(projected.divergence_after) < 1e-9
