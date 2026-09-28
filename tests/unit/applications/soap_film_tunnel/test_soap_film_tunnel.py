#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from phydrax.applications.soap_film_tunnel import (
    PreparedSoapFilmTunnel,
    SheddingStatus,
    SoapFilmInflow,
    SoapFilmTunnelGeometry,
    SoapFilmTunnelPlan,
    SoapFilmTunnelResult,
    SoapFilmWireKind,
)
from phydrax.interfacial_transport import (
    AdsorptionKinetics,
    FilmStepStatus,
    LangmuirSurfactantLaw,
    PlugFlowNonlinearStage,
)
from phydrax.nonlinear import NonlinearStatus


_requires_meshcore = pytest.mark.skipif(
    not meshcore_available(), reason="channel triangulation requires meshcore"
)

_DENSITY, _THICKNESS, _CONCENTRATION = 1000.0, 2e-6, 1.6e-6
_GRAVITY, _DRAG = 9.81, 0.02
_TERMINAL = _DENSITY * _THICKNESS * _GRAVITY / _DRAG
_DIAMETER = 1e-3
_STEP = eqx.filter_jit(lambda prepared, state, step_size: prepared.step(state, step_size))


def _law() -> LangmuirSurfactantLaw:
    return LangmuirSurfactantLaw(0.072, 298.15, 4e-6)


def _plan(
    geometry: SoapFilmTunnelGeometry, *, wire_kind: SoapFilmWireKind
) -> SoapFilmTunnelPlan:
    return SoapFilmTunnelPlan(
        geometry,
        _law(),
        SoapFilmInflow(
            velocity_m_s=_TERMINAL,
            thickness_m=_THICKNESS,
            surface_concentration_mol_m2=_CONCENTRATION,
        ),
        density_kg_m3=_DENSITY,
        air_drag_coefficient_kg_m2_s=_DRAG,
        gravity_m_s2=_GRAVITY,
        surface_shear_viscosity_n_s_m=5e-9,
        surface_diffusivity_m2_s=1e-9,
        wire_kind=wire_kind,
    )


def test_tunnel_plan_refuses_inflow_outside_positive_tension_support() -> None:
    law = _law()
    inflow = SoapFilmInflow(
        velocity_m_s=_TERMINAL,
        thickness_m=_THICKNESS,
        surface_concentration_mol_m2=0.9999
        * float(law.maximum_surface_concentration_mol_m2),
    )
    assert float(law.evaluate(inflow.surface_concentration_mol_m2).surface_tension_n_m) < 0
    with pytest.raises(ValueError, match="positive surface tension"):
        SoapFilmTunnelPlan(
            SoapFilmTunnelGeometry(4e-3, 2e-3, mesh_size_m=5e-4),
            law,
            inflow,
            density_kg_m3=_DENSITY,
            air_drag_coefficient_kg_m2_s=_DRAG,
        )


@pytest.fixture(scope="module")
def empty_channel() -> PreparedSoapFilmTunnel:
    geometry = SoapFilmTunnelGeometry(4e-3, 2e-3, mesh_size_m=5e-4)
    return _plan(geometry, wire_kind="free-slip").prepare()


@pytest.fixture(scope="module")
def cylinder_channel() -> PreparedSoapFilmTunnel:
    geometry = SoapFilmTunnelGeometry(
        6 * _DIAMETER,
        4 * _DIAMETER,
        mesh_size_m=0.5 * _DIAMETER,
        obstacle_diameter_m=_DIAMETER,
        obstacle_center_m=(2 * _DIAMETER, 2 * _DIAMETER),
        rim_segments=16,
    )
    return _plan(geometry, wire_kind="no-slip").prepare()


@pytest.mark.meshcore
@_requires_meshcore
def test_terminal_uniform_flow_stays_uniform_with_analytic_film_mach(
    empty_channel: PreparedSoapFilmTunnel,
) -> None:
    state = empty_channel.initial_state()
    step_size = 0.1 * 5e-4 / _TERMINAL
    result = empty_channel.run(state, jnp.asarray(step_size), 4)
    assert bool(result.all_accepted)
    velocity = np.asarray(empty_channel.velocity(result.state))
    np.testing.assert_allclose(velocity[:, 0], _TERMINAL, rtol=1e-10)
    np.testing.assert_allclose(velocity[:, 1:], 0.0, atol=1e-10 * _TERMINAL)
    np.testing.assert_allclose(
        np.asarray(empty_channel.thickness(result.state)), _THICKNESS, rtol=1e-10
    )
    np.testing.assert_allclose(float(result.state.time_s), 4 * step_size, rtol=1e-14)
    evidence = result.evidence
    flux = _THICKNESS * _TERMINAL * 2e-3
    np.testing.assert_allclose(evidence.inflow_volume_rate_m3_s, flux, rtol=1e-10)
    np.testing.assert_allclose(evidence.outflow_volume_rate_m3_s, flux, rtol=1e-10)
    elasticity = float(_law().gibbs_elasticity(_CONCENTRATION))
    mach = _TERMINAL / np.sqrt(2.0 * elasticity / (_DENSITY * _THICKNESS))
    np.testing.assert_allclose(evidence.film_mach_number, mach, rtol=1e-10)
    np.testing.assert_allclose(float(empty_channel.scales.film_mach_number), mach)
    assert bool(
        jnp.all(
            evidence.terminal_nonlinear_stage
            == int(PlugFlowNonlinearStage.SURFACTANT)
        )
    )
    assert bool(jnp.all(evidence.converged))


@pytest.mark.meshcore
@_requires_meshcore
def test_rejected_step_keeps_state_and_time(
    empty_channel: PreparedSoapFilmTunnel,
) -> None:
    state = empty_channel.initial_state()
    result = _STEP(empty_channel, state, 10.0 * 5e-4 / _TERMINAL)
    assert int(result.evidence.status) == FilmStepStatus.COURANT_LIMIT
    assert float(result.state.time_s) == 0.0
    assert np.array_equal(
        np.asarray(result.state.film.momentum_kg_m_s),
        np.asarray(state.film.momentum_kg_m_s),
    )


@pytest.mark.meshcore
@_requires_meshcore
def test_courant_rejection_reports_attempted_candidate_diagnostics(
    empty_channel: PreparedSoapFilmTunnel,
) -> None:
    coordinates = np.asarray(empty_channel.flow.plan.surface.coordinates)
    velocity = np.zeros_like(coordinates)
    velocity[:, 0] = _TERMINAL * (
        0.5 + coordinates[:, 0] / float(empty_channel.plan.geometry.length_m)
    )
    state = empty_channel.initial_state(velocity)

    result = _STEP(empty_channel, state, 10.0 * 5e-4 / _TERMINAL)

    assert int(result.evidence.status) == FilmStepStatus.COURANT_LIMIT
    area = empty_channel.flow.plan.surface.vertex_area
    candidate_thickness = result.film.candidate_state.liquid_volume_m3 / area
    candidate_speed = jnp.linalg.norm(
        empty_channel.flow.velocity(result.film.candidate_state), axis=1
    )
    np.testing.assert_allclose(
        result.evidence.minimum_thickness_m, jnp.min(candidate_thickness)
    )
    np.testing.assert_allclose(result.evidence.maximum_speed_m_s, jnp.max(candidate_speed))
    assert not np.isclose(
        float(jnp.min(candidate_thickness)),
        float(jnp.min(empty_channel.thickness(result.state))),
    )


@pytest.mark.meshcore
@_requires_meshcore
def test_nonfinite_step_preserves_attempted_candidate_diagnostics(
    empty_channel: PreparedSoapFilmTunnel,
) -> None:
    state = empty_channel.initial_state()

    result = _STEP(empty_channel, state, jnp.nan)

    assert int(result.evidence.status) == FilmStepStatus.INADMISSIBLE_INPUT
    assert np.isnan(float(result.evidence.minimum_thickness_m))
    assert np.isnan(float(result.evidence.maximum_speed_m_s))
    assert np.isnan(float(result.evidence.film_mach_number))
    np.testing.assert_array_equal(
        result.state.film.liquid_volume_m3, state.film.liquid_volume_m3
    )


@pytest.mark.meshcore
@_requires_meshcore
def test_soluble_exchange_failure_reaches_tunnel_evidence() -> None:
    geometry = SoapFilmTunnelGeometry(4e-3, 2e-3, mesh_size_m=5e-4)
    kinetics = AdsorptionKinetics(1e-5, 1.0, 4e-6)
    prepared = SoapFilmTunnelPlan(
        geometry,
        _law(),
        SoapFilmInflow(
            velocity_m_s=_TERMINAL,
            thickness_m=_THICKNESS,
            surface_concentration_mol_m2=1e-7,
            dissolved_concentration_mol_m3=1.0,
        ),
        density_kg_m3=_DENSITY,
        air_drag_coefficient_kg_m2_s=_DRAG,
        gravity_m_s2=_GRAVITY,
        kinetics=kinetics,
        wire_kind="free-slip",
        maximum_iterations=1,
        transport_scheme="donor-cell",
    ).prepare()

    result = _STEP(prepared, prepared.initial_state(), 0.1 * 5e-4 / _TERMINAL)

    assert int(result.evidence.status) == FilmStepStatus.SOLVE_FAILED
    assert (
        int(result.evidence.terminal_nonlinear_stage)
        == PlugFlowNonlinearStage.SURFACTANT
    )
    assert (
        int(result.evidence.nonlinear_status)
        == NonlinearStatus.MAXIMUM_STEPS_REACHED
    )
    assert int(result.evidence.nonlinear_iterations) == 1
    assert not bool(result.evidence.converged)


@pytest.mark.meshcore
@_requires_meshcore
def test_cylinder_channel_balances_fluxes_and_holds_no_slip(
    cylinder_channel: PreparedSoapFilmTunnel,
) -> None:
    channel = cylinder_channel.channel
    center = np.asarray((2 * _DIAMETER, 2 * _DIAMETER))
    rim = np.asarray(channel.mesh.vertices)[np.asarray(channel.rim_vertices), :2]
    np.testing.assert_allclose(
        np.linalg.norm(rim - center, axis=1), 0.5 * _DIAMETER, rtol=1e-12
    )
    state = cylinder_channel.initial_state()
    step_size = 0.05 * (np.pi * _DIAMETER / 16) / _TERMINAL
    volume = float(jnp.sum(state.film.liquid_volume_m3))
    surfactant = float(state.film.total_surfactant_mol())
    result = cylinder_channel.run(state, jnp.asarray(step_size), 6)
    assert bool(result.all_accepted)
    evidence = result.evidence
    np.testing.assert_allclose(
        float(jnp.sum(result.state.film.liquid_volume_m3)) - volume,
        step_size
        * float(
            jnp.sum(evidence.inflow_volume_rate_m3_s - evidence.outflow_volume_rate_m3_s)
        ),
        rtol=1e-9,
        atol=1e-14 * volume,
    )
    np.testing.assert_allclose(
        float(result.state.film.total_surfactant_mol()) - surfactant,
        step_size
        * float(
            jnp.sum(
                evidence.inflow_surfactant_rate_mol_s
                - evidence.outflow_surfactant_rate_mol_s
            )
        ),
        rtol=1e-9,
        atol=1e-14 * surfactant,
    )
    impulse = step_size * float(jnp.max(jnp.abs(evidence.obstacle_force_n)))
    assert float(jnp.max(jnp.abs(evidence.momentum_residual_n_s))) <= 1e-9 * impulse
    velocity = np.asarray(cylinder_channel.velocity(result.state))
    held = np.asarray(cylinder_channel.rim | cylinder_channel.wires)
    assert np.max(np.abs(velocity[held])) == 0.0
    free = ~np.asarray(
        cylinder_channel.rim
        | cylinder_channel.wires
        | cylinder_channel.inlet
        | cylinder_channel.outlet
    )
    assert np.max(np.linalg.norm(velocity[free], axis=1)) > 0.5 * _TERMINAL
    # The film drags the cylinder downstream.
    assert float(evidence.obstacle_force_n[-1, 0]) > 0.0


@pytest.mark.meshcore
@_requires_meshcore
def test_strouhal_estimate_counts_hysteretic_lift_periods(
    cylinder_channel: PreparedSoapFilmTunnel,
) -> None:
    state = cylinder_channel.initial_state()
    record = cylinder_channel.run(state, jnp.asarray(1e-6), 6)
    time = np.linspace(0.0, 0.05, 10001)
    frequency = 200.0
    dynamic = 0.5 * _DENSITY * _THICKNESS * _TERMINAL**2 * _DIAMETER
    rng = np.random.default_rng(4)
    lift = 0.3 * dynamic * np.sin(2 * np.pi * frequency * time)
    lift = lift + 0.01 * dynamic * rng.standard_normal(time.size)
    force = np.stack((np.full_like(time, 1.2 * dynamic), lift, 0.0 * time), axis=1)

    def with_record(status: np.ndarray, force: np.ndarray) -> SoapFilmTunnelResult:
        return eqx.tree_at(
            lambda result: (
                result.evidence.time_s,
                result.evidence.status,
                result.evidence.obstacle_force_n,
            ),
            record,
            (jnp.asarray(time), jnp.asarray(status), jnp.asarray(force)),
        )

    accepted = np.zeros(time.size, dtype=np.int32)
    estimate = cylinder_channel.strouhal(with_record(accepted, force), transient_s=0.005)
    assert estimate.status is SheddingStatus.SHEDDING
    np.testing.assert_allclose(float(estimate.frequency_hz), frequency, rtol=0.01)
    np.testing.assert_allclose(
        float(estimate.strouhal_number), frequency * _DIAMETER / _TERMINAL, rtol=0.01
    )
    assert np.isfinite(float(estimate.strouhal_standard_error))
    assert float(estimate.strouhal_standard_error) < 0.01
    np.testing.assert_allclose(float(estimate.mean_drag_coefficient), 1.2, rtol=1e-12)
    reference = cylinder_channel.cylinder_wake_reference(0.1)
    assert float(reference.blockage_corrected_strouhal_number) > float(
        reference.unconfined_strouhal_number
    )
    assert float(reference.corrected_reynolds_number) > float(
        cylinder_channel.scales.reynolds_number
    )
    quiet = force * np.asarray((1.0, 1e-4, 0.0))
    assert (
        cylinder_channel.strouhal(with_record(accepted, quiet), transient_s=0.005).status
        is SheddingStatus.NO_SHEDDING
    )
    rejected = accepted.copy()
    rejected[-1] = int(FilmStepStatus.SOLVE_FAILED)
    assert (
        cylinder_channel.strouhal(with_record(rejected, force), transient_s=0.005).status
        is SheddingStatus.REJECTED_STEPS
    )


@pytest.mark.meshcore
@_requires_meshcore
def test_909_vertex_recurrence_keeps_bounded_nonlinear_work() -> None:
    geometry = SoapFilmTunnelGeometry(
        12 * _DIAMETER,
        6 * _DIAMETER,
        mesh_size_m=0.4 * _DIAMETER,
        obstacle_diameter_m=_DIAMETER,
        obstacle_center_m=(3 * _DIAMETER, 3 * _DIAMETER),
        rim_segments=32,
    )
    prepared = _plan(geometry, wire_kind="no-slip").prepare()
    assert prepared.flow.plan.surface.topology.num_vertices == 909
    points = np.asarray(prepared.flow.plan.surface.coordinates)
    velocity = np.zeros_like(points)
    velocity[:, 0] = _TERMINAL
    velocity[:, 1] = 0.1 * _TERMINAL * np.exp(
        -(((points[:, 0] - 4 * _DIAMETER) / _DIAMETER) ** 2)
        - ((points[:, 1] - 3 * _DIAMETER) / _DIAMETER) ** 2
    )
    step_size = 0.1 * np.pi * _DIAMETER / (32 * _TERMINAL)
    result = prepared.run(prepared.initial_state(velocity), jnp.asarray(step_size), 32)

    assert bool(result.all_accepted)
    assert int(jnp.max(result.evidence.nonlinear_iterations)) <= 3
    np.testing.assert_allclose(
        float(result.state.time_s), 32 * step_size, rtol=1e-14
    )
