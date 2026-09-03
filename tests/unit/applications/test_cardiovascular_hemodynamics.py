#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.cardiovascular.circulation._components import Resistance
from phydrax.applications.cardiovascular.hemodynamics._domain import (
    compare_lbm_mac,
    FixedWallLumenRegion,
    HemodynamicsScaling,
    HemodynamicsValidityLimits,
    PoiseuillePipeReference,
    WomersleyPipeReference,
)
from phydrax.applications.cardiovascular.hemodynamics._fixed_wall_lbm import (
    FixedWallLBMPlan,
)
from phydrax.applications.cardiovascular.hemodynamics._ports import (
    CirculationPortBinding,
    FlowTerminalPort,
    PressureTerminalPort,
    terminal_balance_evidence,
    TerminalDirection,
    TerminalFace,
    TerminalPortValues,
)
from phydrax.applications.cardiovascular.hemodynamics._rheology import (
    CarreauYasudaRheology,
    NewtonianRheology,
)
from phydrax.discretization import (
    D2Q9,
    D3Q19,
    LatticeBoltzmannLinkOwner,
    LatticeBoltzmannPlan,
    TensorGridPlan,
    UniformCellAxisSpec,
)


def _discretization(shape=(6, 4, 4), *, cell_size=1.0):
    grid = TensorGridPlan(
        tuple(UniformCellAxisSpec(count) for count in shape),
        axis_names=("x", "y", "z")[: len(shape)],
    ).prepare(
        jnp.asarray(
            (
                tuple(0.0 for _ in shape),
                tuple(cell_size * count for count in shape),
            )
        )
    )
    lattice = D3Q19() if len(shape) == 3 else D2Q9()
    return LatticeBoltzmannPlan(grid, lattice).prepare()


def _prepared_workflow(*, limits=None, lumen_mask=None):
    discretization = _discretization()
    scaling = HemodynamicsScaling(
        1.0,
        1.0,
        1.06,
        reference_velocity_mm_per_ms=0.02,
    )
    mask = np.ones((6, 4, 4), dtype=bool) if lumen_mask is None else lumen_mask
    lumen = FixedWallLumenRegion(mask)
    component = Resistance("terminal_resistance", 1.0)
    terminals = (
        FlowTerminalPort(
            "inlet",
            TerminalFace("x", "lower", TerminalDirection.INTO_LUMEN),
            CirculationPortBinding(component, "inlet"),
        ),
        PressureTerminalPort(
            "outlet",
            TerminalFace("x", "upper", TerminalDirection.OUT_OF_LUMEN),
            CirculationPortBinding(component, "outlet"),
        ),
    )
    plan = FixedWallLBMPlan(
        discretization,
        scaling,
        lumen,
        terminals,
        NewtonianRheology(0.004, maximum_shear_rate_per_ms=1.0),
        limits=limits,
    )
    return plan.prepare()


def test_static_lumen_mask_compiles_stationary_halfway_wall_links():
    mask = np.ones((6, 4, 4), dtype=bool)
    mask[:, 0, :] = False
    prepared = _prepared_workflow(lumen_mask=mask)
    owner = np.asarray(prepared.boundary.topology.owner)

    assert np.any(owner == int(LatticeBoltzmannLinkOwner.HALFWAY))
    assert prepared.boundary.geometry.fluid_count == int(np.sum(mask))
    assert prepared.scope.wall_motion_supported is False


def test_hemodynamics_scaling_roundtrips_every_coupled_quantity():
    scaling = HemodynamicsScaling(
        0.25,
        0.01,
        1.06,
        reference_velocity_mm_per_ms=0.5,
        maximum_lattice_mach=0.05,
    )

    velocity = jnp.asarray((0.1, -0.2, 0.3))
    viscosity = jnp.asarray((1.0e-3, 4.0e-3))
    density = jnp.asarray((1.04, 1.06))
    pressure = jnp.asarray((-2.0, 0.0, 12.0))
    flow = jnp.asarray((-4.0, 7.5))
    shear = jnp.asarray((0.0, 0.25))
    mass = jnp.asarray((1.0, 20.0))
    momentum = jnp.asarray((1.0, -2.0, 3.0))
    power = jnp.asarray((-0.3, 0.7))

    np.testing.assert_allclose(
        scaling.physical_velocity(scaling.lattice_velocity(velocity)), velocity
    )
    np.testing.assert_allclose(
        scaling.physical_kinematic_viscosity(
            scaling.lattice_kinematic_viscosity(viscosity)
        ),
        viscosity,
    )
    np.testing.assert_allclose(
        scaling.physical_density(scaling.lattice_density(density)), density
    )
    np.testing.assert_allclose(
        scaling.density_gauge_pressure(scaling.pressure_density(pressure)), pressure
    )
    np.testing.assert_allclose(
        scaling.physical_flow_rate(scaling.lattice_flow_rate(flow)), flow
    )
    np.testing.assert_allclose(
        scaling.physical_shear_rate(scaling.lattice_shear_rate(shear)), shear
    )
    np.testing.assert_allclose(scaling.physical_mass(scaling.lattice_mass(mass)), mass)
    np.testing.assert_allclose(
        scaling.physical_momentum(scaling.lattice_momentum(momentum)), momentum
    )
    np.testing.assert_allclose(
        scaling.physical_power(scaling.lattice_power(power)), power
    )
    assert len(scaling.quantity_spec_ids) == 10


def test_scaling_and_rheology_refuse_outside_validity_envelopes():
    with pytest.raises(ValueError, match="Mach limit"):
        HemodynamicsScaling(
            1.0,
            1.0,
            1.06,
            reference_velocity_mm_per_ms=0.2,
            maximum_lattice_mach=0.1,
        )

    rheology = CarreauYasudaRheology(0.056, 0.0035, 3313.0, 0.3568, 2.0)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="validity envelope"):
        invalid = rheology.dynamic_viscosity(jnp.asarray(11.0))
        jax.block_until_ready(invalid)

    discretization = _discretization()
    too_fast_relaxation = HemodynamicsScaling(
        1.0,
        0.01,
        1.06,
        reference_velocity_mm_per_ms=0.01,
    )
    component = Resistance("invalid_scale_terminal", 1.0)
    terminals = (
        FlowTerminalPort(
            "inlet",
            TerminalFace("x", "lower", TerminalDirection.INTO_LUMEN),
            CirculationPortBinding(component, "inlet"),
        ),
        PressureTerminalPort(
            "outlet",
            TerminalFace("x", "upper", TerminalDirection.OUT_OF_LUMEN),
            CirculationPortBinding(component, "outlet"),
        ),
    )
    with pytest.raises(ValueError, match="relaxation rates"):
        FixedWallLBMPlan(
            discretization,
            too_fast_relaxation,
            FixedWallLumenRegion(np.ones((6, 4, 4), dtype=bool)),
            terminals,
            NewtonianRheology(0.004),
        )


def test_carreau_yasuda_newtonian_limit_and_shear_thinning():
    constant = CarreauYasudaRheology(0.004, 0.004, 100.0, 0.4, 2.0)
    newtonian = NewtonianRheology(0.004)
    shear = jnp.asarray((0.0, 1.0e-3, 0.1, 1.0))
    np.testing.assert_allclose(
        constant.dynamic_viscosity(shear),
        newtonian.dynamic_viscosity(shear),
        rtol=0.0,
        atol=1.0e-14,
    )

    thinning = CarreauYasudaRheology(0.056, 0.0035, 3313.0, 0.3568, 2.0)
    values = np.asarray(thinning.dynamic_viscosity(shear))
    assert np.all(np.diff(values) <= 0.0)
    assert values[-1] >= thinning.minimum_dynamic_viscosity_kpa_ms
    assert values[0] == pytest.approx(thinning.maximum_dynamic_viscosity_kpa_ms)


def test_terminal_measurements_close_outlet_volume_and_power_balances():
    prepared = _prepared_workflow()
    shape = prepared.discretization.grid.shape
    velocity = jnp.zeros(shape + (3,)).at[..., 0].set(0.01)
    pressure = jnp.zeros(shape).at[0, :, :].set(12.0).at[-1, :, :].set(10.0)
    measured = prepared.terminal_measurements.measure(pressure, velocity)
    values = TerminalPortValues(
        jnp.asarray((12.0, 10.0)),
        jnp.asarray((0.16, 0.16)),
    )
    evidence = terminal_balance_evidence(
        prepared.terminal_measurements,
        measured,
        values,
        storage_volume_change_mm3=0.0,
        time_step_ms=1.0,
        flow_relative_tolerance=1.0e-12,
        pressure_absolute_tolerance_kpa=jnp.asarray((0.1, 0.2)),
        volume_relative_tolerance=1.0e-12,
        power_relative_tolerance=1.0e-12,
    )

    np.testing.assert_allclose(measured.outward_flow_mm3_per_ms, (-0.16, 0.16))
    np.testing.assert_allclose(measured.directed_flow_mm3_per_ms, (0.16, 0.16))
    np.testing.assert_allclose(evidence.pressure_residual_kpa, (0.0, 0.0))
    np.testing.assert_allclose(evidence.pressure_tolerance_kpa, (0.1, 0.2))
    np.testing.assert_array_equal(evidence.pressure_balanced, (True, True))
    assert bool(evidence.passed)
    assert float(evidence.measured_power_into_lumen) == pytest.approx(0.32)

    mismatched = terminal_balance_evidence(
        prepared.terminal_measurements,
        measured,
        TerminalPortValues(jnp.asarray((11.5, 10.0)), jnp.asarray((0.16, 0.12))),
        storage_volume_change_mm3=0.0,
        time_step_ms=1.0,
        flow_relative_tolerance=1.0e-3,
        pressure_absolute_tolerance_kpa=jnp.asarray((0.1, 0.2)),
        volume_relative_tolerance=1.0e-12,
        power_relative_tolerance=1.0e-3,
    )
    assert not bool(mismatched.passed)
    np.testing.assert_allclose(mismatched.pressure_residual_kpa, (0.5, 0.0))
    np.testing.assert_array_equal(mismatched.pressure_balanced, (False, True))


def test_poiseuille_and_womersley_references_recover_expected_limits():
    pipe = PoiseuillePipeReference(1.5, 20.0, 0.8, 0.004)
    radius = np.linspace(0.0, 1.5, 4001)
    velocity = np.asarray(pipe.axial_velocity(radius))
    integrated_flow = np.trapezoid(2.0 * np.pi * radius * velocity, radius)
    assert integrated_flow == pytest.approx(float(pipe.flow_rate_mm3_per_ms), rel=2.0e-7)
    assert velocity[0] == pytest.approx(float(pipe.centerline_velocity_mm_per_ms))
    assert velocity[-1] == pytest.approx(0.0, abs=1.0e-14)

    gradient = 0.02
    womersley = WomersleyPipeReference(1.0, gradient, 0.004, 1.0, 1.0e-6)
    steady = PoiseuillePipeReference(1.0, 1.0, gradient, 0.004)
    samples = jnp.asarray((0.0, 0.25, 0.75, 1.0))
    np.testing.assert_allclose(
        womersley.axial_velocity(samples, 0.0),
        steady.axial_velocity(samples),
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    assert womersley.womersley_number < 0.02


def test_lbm_mac_comparison_keeps_routes_distinct_and_auditable():
    coordinate = jnp.linspace(0.0, 1.0, 128)
    profile = 1.0 - coordinate**2
    mac_velocity = jnp.stack(
        (profile, jnp.zeros_like(profile), jnp.zeros_like(profile)), axis=-1
    )
    lbm_velocity = 0.995 * mac_velocity
    mac_pressure = 10.0 - coordinate
    lbm_pressure = 1.002 * mac_pressure
    evidence = compare_lbm_mac(
        lbm_velocity,
        mac_velocity,
        lbm_pressure,
        mac_pressure,
        jnp.ones_like(coordinate),
        velocity_relative_tolerance=0.01,
        pressure_relative_tolerance=0.01,
    )

    assert bool(evidence.passed)
    assert evidence.lbm_route_id != evidence.mac_route_id
    assert float(evidence.velocity_relative_l2) == pytest.approx(0.005, rel=1.0e-5)
    assert float(evidence.pressure_relative_l2) == pytest.approx(0.002, rel=1.0e-5)


def test_fixed_wall_d3q19_candidate_commit_checkpoint_and_fail_closed_state():
    prepared = _prepared_workflow(
        limits=HemodynamicsValidityLimits(
            maximum_relative_mass_balance_defect=1.0e-5,
            maximum_relative_momentum_change=0.25,
            maximum_terminal_flow_relative_defect=1.0e-6,
            maximum_terminal_power_relative_defect=1.0e-8,
        )
    )
    state = prepared.initialize_state()
    values = TerminalPortValues(jnp.zeros((2,)), jnp.zeros((2,)))
    candidate = prepared.candidate(state, values)
    committed = prepared.commit(state, candidate)

    assert bool(candidate.evidence.successful)
    assert int(committed.step_index) == 1
    assert prepared.scope.wall_motion_supported is False
    assert prepared.scope.fluid_structure_interaction_supported is False
    assert prepared.scope.curved_wall_accuracy_supported is False
    assert prepared.scope.clinical_use_supported is False
    restored = prepared.restore(prepared.checkpoint(committed))
    np.testing.assert_array_equal(restored.populations, committed.populations)

    excessive = prepared.candidate(
        committed,
        TerminalPortValues(jnp.zeros((2,)), jnp.asarray((1.0, 0.0))),
    )
    rejected = prepared.commit(committed, excessive)
    assert not bool(excessive.evidence.successful)
    np.testing.assert_array_equal(rejected.populations, committed.populations)
    assert int(rejected.step_index) == int(committed.step_index)


@pytest.mark.parametrize(
    ("pressure", "flow"),
    (
        (jnp.asarray((jnp.nan, 0.0)), jnp.zeros((2,))),
        (jnp.zeros((2,)), jnp.asarray((jnp.inf, 0.0))),
        (jnp.zeros((2,)), jnp.asarray((32.0, 0.0))),
    ),
)
def test_invalid_port_iterates_return_rejected_candidate_before_native_boundary(
    pressure, flow
):
    prepared = _prepared_workflow()
    state = prepared.initialize_state()

    candidate = prepared.candidate(state, TerminalPortValues(pressure, flow))
    committed = prepared.commit(state, candidate)

    assert not bool(candidate.evidence.port_iterate_admissible)
    assert not bool(candidate.evidence.successful)
    assert np.all(np.isfinite(np.asarray(candidate.state.populations)))
    np.testing.assert_array_equal(committed.populations, state.populations)
    assert int(committed.step_index) == int(state.step_index)


def test_pressure_controlled_inflow_is_rejected_during_planning():
    discretization = _discretization()
    scaling = HemodynamicsScaling(
        1.0,
        1.0,
        1.06,
        reference_velocity_mm_per_ms=0.01,
    )
    component = Resistance("pressure_inflow_terminal", 1.0)
    terminal = PressureTerminalPort(
        "unsupported_pressure_inlet",
        TerminalFace("x", "lower", TerminalDirection.INTO_LUMEN),
        CirculationPortBinding(component, "inlet"),
    )

    with pytest.raises(ValueError, match="Pressure-controlled inflow"):
        FixedWallLBMPlan(
            discretization,
            scaling,
            FixedWallLumenRegion(np.ones((6, 4, 4), dtype=bool)),
            (terminal,),
            NewtonianRheology(0.004),
        )


def test_fixed_wall_plan_refuses_non_d3q19_lattice():
    discretization = _discretization((6, 4))
    scaling = HemodynamicsScaling(
        1.0,
        1.0,
        1.06,
        reference_velocity_mm_per_ms=0.01,
    )
    component = Resistance("wrong_lattice_terminal", 1.0)
    terminals = (
        FlowTerminalPort(
            "inlet",
            TerminalFace("x", "lower", TerminalDirection.INTO_LUMEN),
            CirculationPortBinding(component, "inlet"),
        ),
    )
    with pytest.raises(ValueError, match="D3Q19"):
        FixedWallLBMPlan(
            discretization,
            scaling,
            FixedWallLumenRegion(np.ones((6, 4, 2), dtype=bool)),
            terminals,
            NewtonianRheology(0.004),
        )
