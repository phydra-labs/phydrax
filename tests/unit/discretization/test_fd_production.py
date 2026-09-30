#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from math import comb
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._array_archive import (
    ArrayArchiveCorruptionError,
    read_array_archive,
    write_array_archive,
)


def _cell_grid(shape: Any) -> Any:
    dimension = len(shape)
    return phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(count) for count in shape),
        axis_names=tuple(f"axis{axis}" for axis in range(dimension)),
    ).prepare(jnp.asarray([[0.0] * dimension, [1.0] * dimension]))


def _cosine_envelope(time: Any, args: Any) -> Any:
    del args
    return jnp.cos(time)


def test_portable_fd_checkpoint_roundtrips_fields_auxiliary_and_identity(
    tmp_path: Any,
) -> None:
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


def test_fd_checkpoint_refuses_nonfinite_or_nonscalar_runtime_state(
    tmp_path: Any,
) -> None:
    plan = phx.discretization.FDCheckpointPlan(("grid-id",), "ssprk3")
    fields = {"state": jnp.ones((4,))}

    with pytest.raises(Exception, match="finite real scalar"):
        phx.discretization.write_fd_checkpoint(
            tmp_path / "vector-time.phydrax",
            plan,
            jnp.asarray((0.0, 1.0)),
            fields,
        )
    with pytest.raises(Exception, match="finite inexact"):
        phx.discretization.write_fd_checkpoint(
            tmp_path / "nonfinite-state.phydrax",
            plan,
            0.0,
            {"state": jnp.asarray((0.0, jnp.nan, 1.0, 2.0))},
        )

    valid_path = phx.discretization.write_fd_checkpoint(
        tmp_path / "valid.phydrax",
        plan,
        0.0,
        fields,
    )
    manifest, arrays = read_array_archive(valid_path)
    manifest.pop("arrays")
    arrays["time"] = np.asarray(np.nan)
    corrupt_path = write_array_archive(
        tmp_path / "corrupt.phydrax",
        manifest=manifest,
        arrays=arrays,
    )
    with pytest.raises(ArrayArchiveCorruptionError, match="invalid runtime state"):
        phx.discretization.read_fd_checkpoint(corrupt_path, plan)


def test_fd_production_scenario_1() -> None:
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
    for dimension in [1, 2, 3, 4]:
        bridge = phx.discretization.StructuredCochainBridge(_cell_grid((3,) * dimension))
        values = jnp.arange(bridge.cochain.cell_counts[0], dtype="float64")

        first = bridge.exterior_derivative(0, values)

        if dimension > 1:
            second = bridge.exterior_derivative(1, first)
            np.testing.assert_allclose(second, 0.0, rtol=0.0, atol=0.0)
        components = bridge.unpack(0, values)
        np.testing.assert_allclose(bridge.pack(0, components), values)
    with pytest.raises(ValueError, match="maximum_entities"):
        phx.discretization.StructuredCochainBridge(
            _cell_grid((3, 3, 3, 3)),
            resources=phx.discretization.StructuredCochainResourcePolicy(
                maximum_entities=10,
            ),
        )


def test_prepared_maxwell_preserves_constraints_and_material_gradients() -> None:
    bridge = phx.discretization.StructuredCochainBridge(_cell_grid((3, 3, 3)))
    degree_zero = bridge.cochain.cell_counts[0]
    degree_one = bridge.cochain.cell_counts[1]
    degree_two = bridge.cochain.cell_counts[2]
    permittivity = (
        1.0 + 0.2 * (jnp.arange(degree_one, dtype="float64") + 1.0) / degree_one
    )
    permeability = (
        1.0 + 0.1 * (jnp.arange(degree_two, dtype="float64") + 1.0) / degree_two
    )
    electric = jnp.sin(jnp.arange(degree_one, dtype="float64") / 7.0)
    magnetic = bridge.exterior_derivative(1, electric)
    charge = -bridge.codifferential(1, permittivity * electric)
    current = bridge.exterior_derivative(
        0,
        jnp.cos(jnp.arange(degree_zero, dtype="float64") / 5.0),
    )

    source = phx.solver.maxwell.MaxwellElectricCurrentSourcePlan(
        jnp.arange(degree_one),
        current,
        envelope=_cosine_envelope,
        control_key="amplitude",
    )
    maxwell = phx.solver.CompatibleMaxwellPlan(
        bridge,
        constitutive=phx.solver.maxwell.DiagonalMaxwellConstitutivePlan(
            permittivity=permittivity,
            permeability=permeability,
        ),
        sources=(source,),
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
    # ty: ignore[no-matching-overload]
    np.testing.assert_allclose(diagnostics.step_fraction, 0.1)
    assert jnp.isfinite(maxwell.energy(stepped))

    def material_energy(epsilon: Any) -> Any:
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
        0.5 * bridge.cochain.hodge_diagonal(1) * electric**2,
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


def test_elastic_energy_and_incompressible_projection_are_compatible() -> None:
    bridge = phx.discretization.StructuredCochainBridge(_cell_grid((3, 3, 3)))

    elasticity = phx.solver.CompatibleElasticityDynamics(
        bridge,
        wave_speed=1.3,
    )
    displacement = jnp.sin(
        jnp.arange(bridge.cochain.cell_counts[0], dtype="float64") / 7.0
    )
    velocity = jnp.cos(jnp.arange(bridge.cochain.cell_counts[0], dtype="float64") / 5.0)
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
    raw_velocity = jnp.sin(
        jnp.arange(bridge.cochain.cell_counts[1], dtype="float64") / 3.0
    )
    projected = eqx.filter_jit(projection.project)(raw_velocity)

    assert jnp.linalg.norm(projected.divergence_before) > 1e-3
    assert jnp.linalg.norm(projected.divergence_after) < 1e-9


def test_structured_polynomial_differential_has_oriented_integrals() -> None:
    bridge = phx.discretization.StructuredCochainBridge(_cell_grid((3, 4)))
    x, y = bridge.grid.structured_axes
    scalar = x.point_coordinates[:, None] * y.point_coordinates[None, :]
    differential = bridge.exterior_derivative(0, bridge.pack(0, (scalar,)))
    dx, dy = bridge.unpack(1, differential)
    np.testing.assert_allclose(
        dx, x.interval_widths[:, None] * y.point_coordinates[None, :], atol=1e-15
    )
    np.testing.assert_allclose(
        dy, x.point_coordinates[:, None] * y.interval_widths[None, :], atol=1e-15
    )
    np.testing.assert_allclose(
        bridge.directional_exterior_derivative(0, bridge.pack(0, (scalar,)), 0),
        bridge.pack(1, (dx, jnp.zeros_like(dy))),
        atol=1e-15,
    )
    np.testing.assert_allclose(bridge.exterior_derivative(1, differential), 0, atol=1e-15)


@pytest.mark.parametrize("degree", [-1, 3])
def test_structured_unpack_refuses_outside_degree(degree: int) -> None:
    bridge = phx.discretization.StructuredCochainBridge(_cell_grid((2, 2)))
    with pytest.raises(ValueError, match="degree"):
        bridge.unpack(degree, jnp.zeros((1,), dtype=jnp.float64))


def test_structured_top_degree_hodge_recovers_constant_density() -> None:
    bridge = phx.discretization.StructuredCochainBridge(_cell_grid((2, 3)))
    density_integrals = bridge.cochain.primal_measures[2] * 2.5
    np.testing.assert_allclose(
        bridge.hodge_star(2, density_integrals), jnp.full((6,), 2.5), atol=1e-14
    )


@pytest.mark.parametrize("dimension", [1, 2, 3, 4])
def test_periodic_structured_complex_has_torus_betti_numbers(dimension: int) -> None:
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(2, periodic=True)
            for _ in range(dimension)
        ),
        axis_names=tuple(f"axis{axis}" for axis in range(dimension)),
    ).prepare(jnp.asarray([[0.0] * dimension, [1.0] * dimension], dtype=jnp.float64))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    complex_ = bridge.hilbert_complex()
    policy = phx.linalg.MaterializationPolicy(max_entries=10_000)
    matrices = tuple(
        np.asarray(phx.linalg.materialize(operator, policy))
        for operator in complex_.differentials
    )
    ranks = (0,) + tuple(np.linalg.matrix_rank(matrix) for matrix in matrices) + (0,)
    for degree, count in enumerate(bridge.cell_counts):
        assert count - ranks[degree] - ranks[degree + 1] == comb(dimension, degree)


def test_four_dimensional_flux_proxy_integrates_analytic_divergence() -> None:
    bridge = phx.discretization.StructuredCochainBridge(_cell_grid((2, 2, 2, 2)))
    components = []
    for axis in range(4):
        orientation = tuple(direction for direction in range(4) if direction != axis)
        block = bridge.orientations[3].index(orientation)
        shape = bridge.orientation_shapes[3][block]
        reshape = [1] * 4
        reshape[axis] = shape[axis]
        coordinate = bridge.grid.structured_axes[axis].point_coordinates.reshape(reshape)
        components.append(jnp.broadcast_to(coordinate, shape))
    flux = bridge.pack_normal_flux(tuple(components))
    np.testing.assert_allclose(
        bridge.exterior_derivative(3, flux),
        4.0 * bridge.cochain.primal_measures[4],
        atol=1e-14,
    )
