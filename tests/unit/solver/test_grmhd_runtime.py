#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.equations._relativistic_eos import GammaLawEOS
from phydrax.equations._relativistic_mhd import IdealValenciaGRMHDSystem
from phydrax.metrix._adm_exchange import ADMGridGeometry
from phydrax.metrix._spacetime_conventions import RelativityConvention
from phydrax.solver._grmhd_ct import (
    GRMHDConstrainedTransportPlan,
    GRMHDCTState,
    GRMHDVectorPotentialGauge,
)
from phydrax.solver._grmhd_runtime import (
    GRMHDRunStatus,
    GRMHDSSPRK3Plan,
    GRMHDState,
)
from phydrax.units import KILOGRAM, METER, SECOND


def _scale():
    return RelativityScaleContract(
        DimensionalScaleContract(METER, KILOGRAM, SECOND), 1, 1, 1, 1
    )


def _periodic_grid(dimension, count):
    names = tuple("xyz"[:dimension])
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True)
            for _ in range(dimension)
        ),
        axis_names=names,
    ).prepare(jnp.stack((jnp.zeros(dimension), jnp.ones(dimension))))
    return grid, phx.discretization.StructuredCochainBridge(grid)


def _geometry(grid, scale, convention):
    shape = grid.shape
    identity = jnp.broadcast_to(jnp.eye(3), shape + (3, 3))
    return ADMGridGeometry(
        jnp.ones(shape),
        jnp.zeros(shape + (3,)),
        identity,
        identity,
        jnp.ones(shape),
        jnp.zeros(shape + (3, 3)),
        jnp.ones(shape, dtype=bool),
        jnp.ones(shape, dtype=bool),
        chart_id="minkowski-cartesian",
        convention_id=convention.convention_id,
        scale_id=scale.scale_id,
        topology_id=grid.topology.topology_id,
        snapshot_token=jnp.asarray(0, dtype=jnp.int32),
        geometry_lineage_id="minkowski-grid",
    )


def test_vector_potential_ct_preserves_discrete_divergence_and_faraday_balance():
    _, bridge = _periodic_grid(2, 4)
    gauge = GRMHDVectorPotentialGauge("weyl")
    plan = GRMHDConstrainedTransportPlan(bridge, gauge=gauge)
    coordinates = bridge.cochain.coordinates[0]
    potential = jnp.sin(2.0 * jnp.pi * coordinates[:, 0]) * jnp.sin(
        2.0 * jnp.pi * coordinates[:, 1]
    )
    state = plan.initialize(vector_potential=potential)
    initial_divergence = plan.magnetic_divergence(state.magnetic_flux)
    magnetic_scale = jnp.maximum(jnp.max(jnp.abs(state.magnetic_flux)), 1.0)
    divergence_atol = 256.0 * jnp.finfo(state.magnetic_flux.dtype).eps * magnetic_scale
    np.testing.assert_allclose(initial_divergence, 0.0, atol=float(divergence_atol))
    assert float(jnp.max(jnp.abs(initial_divergence))) <= max(
        plan.divergence_tolerance, float(divergence_atol)
    )

    electromotive = jnp.cos(2.0 * jnp.pi * coordinates[:, 0])
    rate = plan.rate(state, electromotive)
    step = jnp.asarray(0.03)
    after = GRMHDCTState(
        state.magnetic_flux + step * rate.magnetic_rate,
        state.vector_potential + step * rate.vector_potential_rate,
        state.gauge_scalar,
    )
    ledger = plan.defects(state, after, step * electromotive)
    np.testing.assert_allclose(
        ledger.divergence_after,
        ledger.divergence_before,
        atol=float(divergence_atol),
    )
    np.testing.assert_allclose(ledger.faraday_balance_defect, 0.0, atol=2.0e-7)
    np.testing.assert_allclose(ledger.vector_potential_defect, 0.0, atol=2.0e-7)
    assert bool(ledger.physically_valid)
    assert bool(ledger.qualified)


def test_generalized_lorenz_gauge_has_fixed_cochain_shapes():
    _, bridge = _periodic_grid(3, 2)
    gauge = GRMHDVectorPotentialGauge(
        "generalized_lorenz", propagation_speed=0.8, damping_rate=0.1
    )
    plan = GRMHDConstrainedTransportPlan(bridge, gauge=gauge)
    potential = jnp.zeros((plan.vector_potential_size,))
    scalar = jnp.linspace(-0.1, 0.1, plan.gauge_scalar_size)
    state = plan.initialize(vector_potential=potential, gauge_scalar=scalar)
    electromotive = jnp.zeros((plan.vector_potential_size,))
    rate = plan.rate(state, electromotive)
    assert state.magnetic_flux.shape == (
        bridge.cochain.cell_counts[plan.layout.magnetic_degree],
    )
    assert rate.vector_potential_rate.shape == potential.shape
    assert rate.gauge_scalar_rate.shape == scalar.shape
    np.testing.assert_allclose(rate.faraday_defect, 0.0, atol=2.0e-7)


def test_bounded_grid_requires_boundary_aware_uct():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(4, periodic=False),),
        axis_names=("x",),
    ).prepare(jnp.asarray(((0.0,), (1.0,))))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    with pytest.raises(ValueError, match="boundary-aware UCT"):
        GRMHDConstrainedTransportPlan(bridge)


def test_atomic_ssprk_acceptance_and_rejected_ledger_are_material_ct_consistent():
    grid, bridge = _periodic_grid(2, 2)
    scale = _scale()
    convention = RelativityConvention.canonical()
    eos = GammaLawEOS(scale, 4.0 / 3.0, minimum_density=1.0e-15)
    system = IdealValenciaGRMHDSystem(
        eos,
        scale,
        convention=convention,
        pressure_ceiling=1.0e5,
        recovery_iterations=28,
        enthalpy_iterations=28,
    )
    ct = GRMHDConstrainedTransportPlan(bridge)
    runtime = GRMHDSSPRK3Plan(system, ct, cfl=0.2)
    geometry = _geometry(grid, scale, convention)
    primitive = jnp.zeros(grid.shape + (8,))
    primitive = primitive.at[..., 0].set(1.0)
    primitive = primitive.at[..., 1].set(0.05)
    primitive = primitive.at[..., 4].set(0.1)
    primitive = primitive.at[..., 5].set(0.2)
    primitive = primitive.at[..., 6].set(-0.1)
    conserved = system.primitive_to_conserved(primitive, geometry)
    magnetic_flux = ct.pack_densitized_face_flux(
        tuple(primitive[..., 5 + axis] for axis in range(2))
    )
    state = runtime.initialize(
        conserved,
        geometry,
        magnetic_flux=magnetic_flux,
        step_size=1.0e-4,
    )

    accepted = runtime.advance(state, 0.0, 1.0e-4, geometry)
    assert bool(accepted.accepted)
    assert int(accepted.status) == int(GRMHDRunStatus.SUCCESS)
    np.testing.assert_allclose(
        accepted.state.material_state, state.material_state, atol=5.0e-6
    )
    np.testing.assert_allclose(
        accepted.state.constrained_transport.magnetic_flux,
        state.constrained_transport.magnetic_flux,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        accepted.attempted_ledger.material_balance_defect, 0.0, atol=5.0e-6
    )
    np.testing.assert_allclose(
        accepted.attempted_ledger.faraday_balance_defect, 0.0, atol=2.0e-7
    )

    rejected = runtime.advance(state, 0.0, 10.0, geometry)
    assert not bool(rejected.accepted)
    assert int(rejected.status) == int(GRMHDRunStatus.STABILITY_LIMIT_EXCEEDED)
    np.testing.assert_array_equal(rejected.state.material_state, state.material_state)
    np.testing.assert_array_equal(
        rejected.state.constrained_transport.magnetic_flux,
        state.constrained_transport.magnetic_flux,
    )
    assert float(rejected.state.time) == float(state.time)
    np.testing.assert_allclose(rejected.accepted_ledger.material_state_change, 0.0)
    np.testing.assert_allclose(rejected.accepted_ledger.magnetic_flux_change, 0.0)
    np.testing.assert_allclose(rejected.accepted_ledger.edge_electromotive_integral, 0.0)

    rate = runtime.rate(
        state.time,
        state.material_state,
        state.constrained_transport,
        geometry,
    )
    perturbed = GRMHDState(
        state.material_state.at[0, 0, 0].add(0.01),
        state.constrained_transport,
        jnp.asarray(1.0e-4),
        jnp.asarray(1.0e-4),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(int(GRMHDRunStatus.SUCCESS), dtype=jnp.int32),
    )
    adversarial_ledger = runtime._ledger(
        state,
        perturbed,
        jnp.asarray(1.0e-4),
        (rate, rate, rate),
        jnp.asarray(False),
    )
    assert float(jnp.max(jnp.abs(adversarial_ledger.material_balance_defect))) > 0.0
    assert not bool(adversarial_ledger.qualified)

    mixed_active = geometry.active.at[0, 0].set(False)
    mixed_geometry = eqx.tree_at(lambda value: value.active, geometry, mixed_active)
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="every cell active",
    ):
        rejected_rate = runtime.rate(
            state.time,
            state.material_state,
            state.constrained_transport,
            mixed_geometry,
        )
        jax.block_until_ready(rejected_rate.material_rate)
