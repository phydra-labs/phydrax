"""Passive Kr/Xe ledger through evaluated total-gas steps."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from tests._support.nuclear_fuel_gas import (
    advance_gas,
    DAY,
    fission_gas_model,
    PRODUCTION,
)


pytestmark = pytest.mark.strict_jax


def test_passive_release_uses_boundary_donor_not_whole_retained_mixture() -> None:
    model = fission_gas_model(mode_count=40)
    radius = 5.0e-6
    boundary_kr = 1.0e26
    grain_xe = 1.0e24
    weights = 3.0 / (np.arange(1, model.mode_count + 1, dtype=np.float64) * np.pi) ** 2
    tail = (1.0 - float(np.sum(2.0 * weights))) * grain_xe
    grain_modes = jnp.full((model.mode_count,), 2.0 * grain_xe, dtype=jnp.float64)
    total = eqx.tree_at(
        lambda s: (
            s.intragranular_modes,
            s.intragranular_unresolved,
            s.boundary_gas,
            s.bubble_gas_atoms,
            s.bubble_vacancies,
            s.produced,
        ),
        model.initial_state((), grain_radius=radius),
        (
            grain_modes,
            jnp.asarray(tail),
            jnp.asarray(boundary_kr),
            jnp.asarray(boundary_kr / (1.5 / radius * 2.0e13)),
            jnp.asarray(1.0e8),
            jnp.asarray(boundary_kr + grain_xe),
        ),
    )
    passive = eqx.tree_at(
        lambda s: (
            s.intragranular_modes,
            s.intragranular_unresolved,
            s.boundary_gas,
            s.produced,
        ),
        model.initial_species_state(()),
        (
            jnp.stack([jnp.zeros_like(grain_modes), grain_modes], axis=-1),
            jnp.asarray([0.0, tail]),
            jnp.asarray([boundary_kr, 0.0]),
            jnp.asarray([boundary_kr, grain_xe]),
        ),
    )
    candidate = advance_gas(
        model,
        total,
        temperature=1500.0,
        previous_temperature=1500.0,
        burnup=0.0,
        time_step=0.0,
        fission_rate=0.0,
        production=0.0,
    )
    traced = model.advance_species(
        passive,
        total,
        candidate,
        species_production_rate=np.asarray([0.0, 0.0]),
        time_step=0.0,
    )
    assert bool(traced.admissible)
    assert float(traced.released_increment[0]) > 0.0
    assert abs(float(traced.released_increment[1])) <= 1.0e-12 * grain_xe
    np.testing.assert_allclose(
        float(jnp.sum(traced.released_increment)),
        float(candidate.released_increment),
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(traced.retained_total + traced.state.released),
        np.asarray(passive.produced),
        rtol=1.0e-12,
    )


def test_changing_yields_preserve_passive_modes_reservoirs_and_species_ledgers() -> None:
    model = fission_gas_model(mode_count=12)
    total = model.initial_state((), grain_radius=5.0e-6)
    passive = model.initial_species_state(())
    rates = (
        np.asarray(
            [[0.05, 0.95], [0.80, 0.20], [0.20, 0.80], [0.60, 0.40]],
            dtype=np.float64,
        )
        * PRODUCTION
    )
    temperature = 1100.0
    for step, production in enumerate(rates):
        candidate = advance_gas(
            model,
            total,
            temperature=temperature,
            previous_temperature=temperature,
            burnup=40.0 + 0.5 * step,
            time_step=2.0 * DAY,
        )
        traced = model.advance_species(
            passive,
            total,
            candidate,
            species_production_rate=production,
            time_step=2.0 * DAY,
        )
        assert bool(traced.admissible)
        np.testing.assert_allclose(
            np.asarray(traced.retained_total + traced.state.released),
            np.asarray(passive.produced) + production * 2.0 * DAY,
            rtol=1.0e-12,
        )
        np.testing.assert_allclose(
            np.asarray(jnp.sum(traced.state.intragranular_modes, axis=-1)),
            np.asarray(candidate.state.intragranular_modes),
            rtol=1.0e-12,
        )
        np.testing.assert_allclose(
            np.asarray(jnp.sum(traced.state.hbs_pore_gas)),
            np.asarray(candidate.state.hbs_pore_gas),
            rtol=1.0e-12,
        )
        np.testing.assert_allclose(
            np.asarray(jnp.sum(traced.state.hbs_boundary_gas)),
            np.asarray(candidate.state.hbs_boundary_gas),
            rtol=1.0e-12,
        )
        total = candidate.state
        passive = traced.state
