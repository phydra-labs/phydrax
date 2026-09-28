#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.interfacial_transport as it
from tests._support.film_meshes import planar_grid


_GAS_CONSTANT = 6.02214076e23 * 1.380649e-23
_CAPACITY = 4e-6
_STEP = eqx.filter_jit(lambda prepared, state, step_size: prepared.step(state, step_size))
_ADVECT = eqx.filter_jit(
    lambda prepared, state, step_size, velocity: prepared.step(
        state, step_size, tangential_velocity_m_s=velocity
    )
)


def _law() -> it.LangmuirSurfactantLaw:
    return it.LangmuirSurfactantLaw(0.072, 298.15, _CAPACITY)


def _zero_tension_concentration(law: it.LangmuirSurfactantLaw) -> float:
    capacity = float(law.maximum_surface_concentration_mol_m2)
    return capacity * (
        1.0 - np.exp(-float(law.clean_surface_tension_n_m / law.surface_pressure_scale_n_m))
    )


@pytest.fixture(scope="module")
def surface() -> it.PreparedFilmSurface:
    return it.prepare_film_surface(planar_grid(10, 10, 1e-3, 1e-3))


def test_gibbs_elasticity_matches_analytic_derivative_and_jvp() -> None:
    law = _law()
    concentration = jnp.asarray((0.0, 1e-6, 2.5e-6, 3.9e-6))
    scale = _GAS_CONSTANT * 298.15 * _CAPACITY
    np.testing.assert_allclose(
        law.gibbs_elasticity(concentration),
        scale * concentration / (_CAPACITY - concentration),
        rtol=1e-12,
    )
    derivative = jax.vmap(jax.grad(law.surface_tension))(concentration)
    np.testing.assert_allclose(
        law.gibbs_elasticity(concentration), -concentration * derivative, rtol=1e-12
    )
    _, tangent = jax.jvp(law.surface_tension, (concentration,), (jnp.ones(4),))
    np.testing.assert_allclose(tangent, law.tension_derivative(concentration), rtol=1e-12)


def test_langmuir_state_at_capacity_is_refused() -> None:
    law = _law()
    with pytest.raises(eqx.EquinoxRuntimeError, match="Langmuir"):
        jax.block_until_ready(law.surface_tension(jnp.asarray((1e-6, _CAPACITY))))
    assert not bool(law.evaluate(jnp.asarray(_CAPACITY)).admissible)


def test_capacity_violating_input_is_rejected_by_transport(
    surface: it.PreparedFilmSurface,
) -> None:
    kinetics = it.AdsorptionKinetics(1e-5, 1.0, _CAPACITY)
    prepared = it.SymmetricFilmSurfactantPlan(
        surface, _law(), surface_diffusivity_m2_s=1e-9, kinetics=kinetics
    ).prepare()
    state = prepared.initial_state(1e-6, 1.01 * _CAPACITY, 1.0)
    result = _STEP(prepared, state, 1e-3)
    assert int(result.status) == it.FilmStepStatus.INADMISSIBLE_INPUT
    np.testing.assert_array_equal(
        result.state.surfactant_amount_mol, state.surfactant_amount_mol
    )


def test_insoluble_candidate_at_positive_tension_edge_is_accepted(
    surface: it.PreparedFilmSurface,
) -> None:
    law = _law()
    zero_tension = _zero_tension_concentration(law)
    concentration = 0.999 * zero_tension
    prepared = it.SymmetricFilmSurfactantPlan(
        surface, law, surface_diffusivity_m2_s=0.0
    ).prepare()
    state = prepared.initial_state(1e-6, concentration)
    result = _STEP(prepared, state, 1e-3)
    assert int(result.status) == it.FilmStepStatus.ACCEPTED
    assert bool(result.evidence.surface_state_admissible)
    assert float(result.evidence.minimum_surface_tension_n_m) > 0.0


def test_insoluble_nonpositive_tension_rejects_atomically(
    surface: it.PreparedFilmSurface,
) -> None:
    law = _law()
    zero_tension = _zero_tension_concentration(law)
    concentration = 0.5 * (zero_tension + _CAPACITY)
    assert concentration < _CAPACITY
    assert float(law.evaluate(concentration).surface_tension_n_m) < 0.0
    prepared = it.SymmetricFilmSurfactantPlan(
        surface, law, surface_diffusivity_m2_s=0.0
    ).prepare()
    state = prepared.initial_state(1e-6, concentration)
    result = _STEP(prepared, state, 1e-3)
    assert int(result.status) == it.FilmStepStatus.NONPOSITIVE_TENSION
    assert not bool(result.evidence.surface_state_admissible)
    assert float(result.evidence.maximum_coverage) < 1.0
    np.testing.assert_array_equal(
        result.state.surfactant_amount_mol, state.surfactant_amount_mol
    )


def test_soluble_kinetics_candidate_must_also_satisfy_law_support(
    surface: it.PreparedFilmSurface,
) -> None:
    law = _law()
    zero_tension = _zero_tension_concentration(law)
    kinetics = it.AdsorptionKinetics(1e-5, 1.0, _CAPACITY)
    prepared = it.SymmetricFilmSurfactantPlan(
        surface,
        law,
        surface_diffusivity_m2_s=0.0,
        kinetics=kinetics,
    ).prepare()
    state = prepared.initial_state(1e-6, 0.5 * (zero_tension + _CAPACITY), 1.0)
    result = _STEP(prepared, state, 1e-9)
    assert int(result.status) == it.FilmStepStatus.NONPOSITIVE_TENSION
    assert not bool(result.evidence.surface_state_admissible)
    np.testing.assert_array_equal(
        result.state.surfactant_amount_mol, state.surfactant_amount_mol
    )
    np.testing.assert_array_equal(
        result.state.dissolved_amount_mol, state.dissolved_amount_mol
    )


def test_insoluble_diffusion_conserves_interfacial_amount_and_smooths(
    surface: it.PreparedFilmSurface,
) -> None:
    prepared = it.SymmetricFilmSurfactantPlan(
        surface, _law(), surface_diffusivity_m2_s=1e-9
    ).prepare()
    rng = np.random.default_rng(5)
    count = surface.topology.num_vertices
    state = prepared.initial_state(1e-6, _CAPACITY * rng.uniform(0.1, 0.6, count))
    total = float(state.total_surfactant_mol())
    spread = float(jnp.ptp(prepared.plan.surface_concentration(state)))
    for _ in range(3):
        result = _STEP(prepared, state, 20.0)
        assert int(result.status) == it.FilmStepStatus.ACCEPTED
        state = result.state
    assert abs(float(state.total_surfactant_mol()) - total) <= 1e-14 * total
    assert float(jnp.ptp(prepared.plan.surface_concentration(state))) < 0.5 * spread


def test_adsorption_conserves_bulk_plus_interfaces_and_reaches_isotherm(
    surface: it.PreparedFilmSurface,
) -> None:
    kinetics = it.AdsorptionKinetics(1e-5, 1.0, _CAPACITY)
    prepared = it.SymmetricFilmSurfactantPlan(
        surface, _law(), surface_diffusivity_m2_s=1e-9, kinetics=kinetics
    ).prepare()
    state = prepared.initial_state(1e-5, 0.0, 5.0)
    total = float(state.total_surfactant_mol())
    first = _STEP(prepared, state, 0.1)
    assert float(first.evidence.transferred_to_interfaces_mol) > 0.0
    for _ in range(40):
        result = _STEP(prepared, state, 0.5)
        assert int(result.status) == it.FilmStepStatus.ACCEPTED
        state = result.state
    assert abs(float(state.total_surfactant_mol()) - total) <= 1e-13 * total
    concentration = prepared.plan.surface_concentration(state)
    bulk = state.dissolved_amount_mol / state.liquid_volume_m3
    equilibrium = (
        kinetics.adsorption_rate_m_s
        * bulk
        / (
            kinetics.desorption_rate_s_inv
            + kinetics.adsorption_rate_m_s * bulk / _CAPACITY
        )
    )
    np.testing.assert_allclose(concentration, equilibrium, rtol=1e-6)


def test_uniform_concentration_has_zero_marangoni_force(
    surface: it.PreparedFilmSurface,
) -> None:
    plan = it.SymmetricFilmSurfactantPlan(surface, _law(), surface_diffusivity_m2_s=0.0)
    prepared = plan.prepare()
    uniform = prepared.initial_state(1e-6, 2e-6)
    # Roundoff of Gamma = N / A only; a physical force here is ~ sigma * dx = 7e-6 N.
    assert float(jnp.max(jnp.abs(plan.marangoni_force(uniform)))) < 1e-17
    x = np.asarray(surface.coordinates[:, 0])
    graded = prepared.initial_state(1e-6, 1e-6 + 1e-3 * x)
    force = plan.marangoni_force(graded)
    # Higher concentration lowers tension, so the net force points to -x.
    assert float(jnp.sum(force[:, 0])) < 0.0


def test_advection_conserves_content_and_refuses_courant_violation(
    surface: it.PreparedFilmSurface,
) -> None:
    prepared = it.SymmetricFilmSurfactantPlan(
        surface, _law(), surface_diffusivity_m2_s=0.0
    ).prepare()
    x = np.asarray(surface.coordinates[:, 0])
    state = prepared.initial_state(1e-6 * (1.0 + x / 1e-3), 1e-6 * (1.0 + x / 1e-3))
    velocity = jnp.broadcast_to(jnp.asarray((1e-3, 0.0, 0.0)), (x.size, 3))
    result = _ADVECT(prepared, state, 0.02, velocity)
    assert int(result.status) == it.FilmStepStatus.ACCEPTED
    assert float(result.evidence.courant_number) <= 1.0
    np.testing.assert_allclose(
        float(result.state.total_surfactant_mol()),
        float(state.total_surfactant_mol()),
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        float(jnp.sum(result.state.liquid_volume_m3)),
        float(jnp.sum(state.liquid_volume_m3)),
        rtol=1e-14,
    )
    rejected = _ADVECT(prepared, state, 2.0, velocity)
    assert int(rejected.status) == it.FilmStepStatus.COURANT_LIMIT
    np.testing.assert_array_equal(rejected.state.liquid_volume_m3, state.liquid_volume_m3)


def test_insoluble_nonfinite_volume_rejects_before_positivity(
    surface: it.PreparedFilmSurface,
) -> None:
    prepared = it.SymmetricFilmSurfactantPlan(
        surface, _law(), surface_diffusivity_m2_s=0.0
    ).prepare()
    state = prepared.initial_state(1e-6, 1e-6)
    invalid = it.SymmetricFilmSurfactantState(
        state.liquid_volume_m3.at[0].set(jnp.nan),
        state.surfactant_amount_mol,
        None,
        topology_id=state.topology_id,
        geometry_revision=state.geometry_revision,
    )

    result = _STEP(prepared, invalid, 1e-3)

    assert int(result.status) == it.FilmStepStatus.NONFINITE
    assert not bool(result.evidence.finite)
    np.testing.assert_array_equal(
        result.state.liquid_volume_m3, invalid.liquid_volume_m3
    )


def test_compressive_transport_crossing_capacity_rejects_without_commit(
    surface: it.PreparedFilmSurface,
) -> None:
    prepared = it.SymmetricFilmSurfactantPlan(
        surface, _law(), surface_diffusivity_m2_s=0.0
    ).prepare()
    state = prepared.initial_state(1e-6, 0.99 * _CAPACITY)
    center = np.asarray((0.5e-3, 0.5e-3, 0.0))
    velocity = -1.5 * (np.asarray(surface.coordinates) - center)
    result = _ADVECT(prepared, state, 0.02, velocity)
    assert float(result.evidence.courant_number) < 1.0
    assert int(result.status) == it.FilmStepStatus.CAPACITY_EXCEEDED
    assert float(result.evidence.maximum_coverage) > 1.0
    assert not bool(result.evidence.surface_state_admissible)
    np.testing.assert_allclose(
        float(result.candidate_state.total_surfactant_mol()),
        float(state.total_surfactant_mol()),
        rtol=1e-14,
    )
    np.testing.assert_array_equal(
        result.state.liquid_volume_m3, state.liquid_volume_m3
    )
    np.testing.assert_array_equal(
        result.state.surfactant_amount_mol, state.surfactant_amount_mol
    )
