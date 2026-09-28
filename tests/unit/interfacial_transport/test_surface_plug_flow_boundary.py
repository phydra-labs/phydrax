#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.interfacial_transport as it
from tests._support.film_meshes import icosphere, planar_grid


_LENGTH, _WIDTH = 4e-3, 2e-3
_THICKNESS, _CONCENTRATION = 5e-6, 2e-6
_DENSITY, _GRAVITY, _DRAG = 1000.0, 9.81, 0.05
_STEP = eqx.filter_jit(lambda prepared, state, step_size: prepared.step(state, step_size))


def _law() -> it.LangmuirSurfactantLaw:
    return it.LangmuirSurfactantLaw(0.072, 298.15, 4e-6)


def _channel() -> tuple[it.PreparedFilmSurface, dict[str, np.ndarray]]:
    surface = it.prepare_film_surface(planar_grid(12, 6, _LENGTH, _WIDTH))
    x, y = np.asarray(surface.coordinates[:, 0]), np.asarray(surface.coordinates[:, 1])
    boundary = np.asarray(surface.topology.boundary_vertices)
    tolerance = 1e-12
    parts = {
        "inlet": np.flatnonzero(boundary & (x < tolerance)),
        "outlet": np.flatnonzero(boundary & (x > _LENGTH - tolerance)),
        "walls": np.flatnonzero(boundary & ((y < tolerance) | (y > _WIDTH - tolerance))),
    }
    return surface, parts


def _tunnel(wall_kind: it.PlugFlowBoundaryKind) -> tuple[
    it.PreparedSurfacePlugFlow, dict[str, np.ndarray], float
]:
    surface, parts = _channel()
    terminal = _DENSITY * _THICKNESS * _GRAVITY / _DRAG
    boundary = it.PlugFlowBoundary(
        surface,
        {"inflow": parts["inlet"], "outflow": parts["outlet"], wall_kind: parts["walls"]},
        inflow_velocity_m_s=(terminal, 0.0, 0.0),
        inflow_thickness_m=_THICKNESS,
        inflow_surface_concentration_mol_m2=_CONCENTRATION,
    )
    prepared = it.SurfacePlugFlowPlan(
        surface,
        _law(),
        density_kg_m3=_DENSITY,
        viscosity_pa_s=1e-3,
        surface_diffusivity_m2_s=1e-9,
        air_drag_coefficient_kg_m2_s=_DRAG,
        gravity_m_s2=(_GRAVITY, 0.0, 0.0),
        boundary=boundary,
    ).prepare()
    return prepared, parts, terminal


def test_terminal_uniform_flow_through_open_edges_stays_uniform() -> None:
    prepared, parts, terminal = _tunnel("free-slip")
    state = prepared.initial_state(_THICKNESS, _CONCENTRATION, (terminal, 0.0, 0.0))
    step_size = 0.3 * (_LENGTH / 12) / terminal
    for _ in range(5):
        result = _STEP(prepared, state, step_size)
        assert int(result.status) == it.FilmStepStatus.ACCEPTED
        state = result.state
    velocity = np.asarray(prepared.velocity(state))
    np.testing.assert_allclose(velocity[:, 0], terminal, rtol=1e-12)
    np.testing.assert_allclose(velocity[:, 1:], 0.0, atol=1e-12 * terminal)
    area = np.asarray(prepared.plan.surface.vertex_area)
    np.testing.assert_allclose(
        np.asarray(state.liquid_volume_m3) / area, _THICKNESS, rtol=1e-12
    )
    exchange = result.plug_flow.boundary
    inflow = -float(jnp.sum(exchange.volume_outflow_m3[parts["inlet"]])) / step_size
    outflow = float(jnp.sum(exchange.volume_outflow_m3[parts["outlet"]])) / step_size
    np.testing.assert_allclose(inflow, _THICKNESS * terminal * _WIDTH, rtol=1e-12)
    np.testing.assert_allclose(outflow, inflow, rtol=1e-12)


def test_no_slip_walls_hold_rest_and_close_every_ledger() -> None:
    prepared, parts, terminal = _tunnel("no-slip")
    state = prepared.initial_state(_THICKNESS, _CONCENTRATION, (terminal, 0.0, 0.0))
    assert float(jnp.max(jnp.abs(prepared.velocity(state)[parts["walls"]]))) == 0.0
    step_size = 0.3 * (_LENGTH / 12) / terminal
    for _ in range(4):
        result = _STEP(prepared, state, step_size)
        assert int(result.status) == it.FilmStepStatus.ACCEPTED
        evidence = result.plug_flow
        scale = float(jnp.max(jnp.abs(evidence.external_impulse_n_s)))
        np.testing.assert_allclose(
            evidence.momentum_change_n_s,
            evidence.external_impulse_n_s + evidence.boundary_impulse_n_s,
            rtol=0.0,
            atol=1e-10 * scale,
        )
        volume = float(jnp.sum(result.state.liquid_volume_m3))
        assert abs(float(result.evidence.liquid_volume_residual_m3)) <= 1e-13 * volume
        total = float(result.state.total_surfactant_mol())
        assert abs(float(evidence.surfactant_residual_mol)) <= 1e-13 * total
        state = result.state
    velocity = np.asarray(prepared.velocity(state))
    assert np.max(np.abs(velocity[parts["walls"]])) == 0.0
    interior = np.setdiff1d(
        np.arange(velocity.shape[0]),
        np.concatenate((parts["walls"], parts["inlet"], parts["outlet"])),
    )
    # The wires retard the film, so the flow no longer carries the inflow flux.
    assert np.min(velocity[interior, 0]) < terminal
    drag = -np.sum(np.asarray(evidence.boundary.constraint_force_n)[parts["walls"]], axis=0)
    assert drag[0] > 0.0


def test_film_at_rest_pulls_its_no_slip_frame_inward_with_twice_the_tension() -> None:
    surface, parts = _channel()
    frame = np.concatenate((parts["inlet"], parts["outlet"], parts["walls"]))
    law = _law()
    prepared = it.SurfacePlugFlowPlan(
        surface,
        law,
        density_kg_m3=_DENSITY,
        boundary=it.PlugFlowBoundary(surface, {"no-slip": frame}),
    ).prepare()
    state = prepared.initial_state(_THICKNESS, _CONCENTRATION)
    result = _STEP(prepared, state, 1e-4)
    assert int(result.status) == it.FilmStepStatus.ACCEPTED
    force = -np.asarray(result.plug_flow.boundary.constraint_force_n)
    tension = 2.0 * float(law.evaluate(_CONCENTRATION).surface_tension_n_m)
    np.testing.assert_allclose(
        np.sum(force[parts["outlet"], 0]), -tension * _WIDTH, rtol=1e-12
    )
    np.testing.assert_allclose(
        np.sum(force[parts["inlet"], 0]), tension * _WIDTH, rtol=1e-12
    )
    np.testing.assert_allclose(
        np.sum(force, axis=0), 0.0, atol=1e-12 * tension * _WIDTH
    )
    velocity = np.asarray(prepared.velocity(result.state))
    assert np.max(np.abs(velocity[frame])) == 0.0
    # Uniform tension leaves only summation roundoff on free vertices.
    assert np.max(np.abs(velocity)) < 1e-12


def test_boundary_declarations_fail_closed() -> None:
    surface, parts = _channel()
    with pytest.raises(ValueError, match="no declared kind"):
        it.PlugFlowBoundary(surface, {"inflow": parts["inlet"], "no-slip": parts["walls"]})
    with pytest.raises(ValueError, match="boundary"):
        it.PlugFlowBoundary(surface, {"no-slip": np.arange(surface.topology.num_vertices)})
    with pytest.raises(ValueError, match="bordered"):
        it.PlugFlowBoundary(it.prepare_film_surface(icosphere(1)), {})
    open_channel = it.PlugFlowBoundary(
        surface,
        {"inflow": parts["inlet"], "outflow": parts["outlet"], "no-slip": parts["walls"]},
        inflow_thickness_m=_THICKNESS,
    )
    with pytest.raises(ValueError, match="dissolved"):
        it.SurfacePlugFlowPlan(
            surface,
            _law(),
            density_kg_m3=_DENSITY,
            kinetics=it.AdsorptionKinetics(1.0, 1e-3, 4e-6),
            boundary=open_channel,
        )


def test_inflow_must_satisfy_positive_tension_support() -> None:
    surface, parts = _channel()
    law = _law()
    boundary = it.PlugFlowBoundary(
        surface,
        {
            "inflow": parts["inlet"],
            "outflow": parts["outlet"],
            "no-slip": parts["walls"],
        },
        inflow_thickness_m=_THICKNESS,
        inflow_surface_concentration_mol_m2=0.9999
        * float(law.maximum_surface_concentration_mol_m2),
    )
    with pytest.raises(ValueError, match="positive surface tension"):
        it.SurfacePlugFlowPlan(
            surface,
            law,
            density_kg_m3=_DENSITY,
            boundary=boundary,
        )
