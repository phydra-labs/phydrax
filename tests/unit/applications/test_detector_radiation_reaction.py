#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Radiation reaction on detector tracks in a constant field.

Reference: planar Landau–Lifshitz cooling in a uniform magnetic field,
``γ(t) = coth(τ ω_B² t + arccoth γ₀)`` with ``τ = q²/(6π ε₀ m c³)`` and
``ω_B = |q| B / m`` (Landau & Lifshitz, *The Classical Theory of Fields*, §76).
"""

from __future__ import annotations

import math
from fractions import Fraction

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax import ElectromagneticScaleContract, units
from phydrax.applications import detector
from phydrax.discretization.pic import (
    PIC_CODE_RELATIVITY,
    RadiationReactionFlag,
    RadiationReactionModel,
    RadiationReactionPlan,
    RadiationReactionTables,
    RelativisticPushPlan,
)
from phydrax.units import CHARGE, UnitDefinition


_CHARGE = -0.1
_MASS = 0.1
_TAU = _CHARGE**2 / (6.0 * math.pi * _MASS)


def _scale() -> ElectromagneticScaleContract:
    return ElectromagneticScaleContract.code_units(
        PIC_CODE_RELATIVITY.dimensional_scale,
        UnitDefinition("code_charge", CHARGE, "phydrax:pic-code"),
        gravitational_constant=1,
        speed_of_light=1,
        reduced_planck_constant=Fraction(1, 10**8),
        boltzmann_constant=1,
        elementary_charge=1,
        electron_mass=1,
        vacuum_permittivity=1,
        constant_set_id="detector-radiation-reaction-test",
    )


def _conditions() -> detector.DetectorConditions:
    return detector.DetectorConditions(
        magnetic_field=jnp.asarray([0.0, 0.0, 1.0]),
        electric_field=jnp.zeros(3),
        momentum_unit=units.JOULE,
        length_unit=units.METER,
        time_unit=units.SECOND,
        geometry_id="vacuum-solenoid",
        material_id="vacuum",
        field_id="uniform-axial",
        alignment_id="nominal",
        calibration_id="nominal",
        validity_interval=(0, 1),
    )


def _bank(
    conditions: detector.DetectorConditions, gamma: tuple[float, float], charge: float
) -> detector.TransportTrackBank:
    """One event with two tracks moving perpendicular to ``B``."""
    speed = [_MASS * math.sqrt(value**2 - 1.0) for value in gamma]
    return detector.TransportTrackBank(
        event_ids=jnp.asarray([3]),
        track_ids=jnp.asarray([[0, 1]]),
        parent_track_ids=jnp.full((1, 2), -1),
        pdg_ids=jnp.asarray([[11, 11]]),
        positions=jnp.zeros((1, 2, 3)),
        momenta=jnp.asarray([[[speed[0], 0.0, 0.0], [0.0, speed[1], 0.0]]]),
        rest_energies=jnp.full((1, 2), _MASS),
        charges=jnp.asarray([[_CHARGE, charge]]),
        active=jnp.asarray([[True, True]]),
        conditions_id=conditions.conditions_id,
    )


def _reaction(
    model: RadiationReactionModel,
    *,
    maximum_chi: float = 1.0e-2,
    tables: RadiationReactionTables | None = None,
) -> RadiationReactionPlan:
    return RadiationReactionPlan(
        model,
        _scale(),
        _CHARGE,
        _MASS,
        tables=tables,
        maximum_chi=maximum_chi,
        minimum_gamma=1.0,
    )


def _gamma(momenta: np.ndarray) -> np.ndarray:
    return np.sqrt(1.0 + np.sum((momenta / _MASS) ** 2, axis=-1))


def test_radiated_energy_history_equals_track_energy_loss_and_planar_cooling() -> None:
    conditions = _conditions()
    steps, dt = 200, 0.01
    plan = detector.ChargedPropagationPlan(
        conditions,
        step_size=dt,
        step_count=steps,
        pusher=RelativisticPushPlan(_scale().relativity, method="boris"),
        radiation_reaction=_reaction("landau-lifshitz"),
    )
    tracks = _bank(conditions, (50.0, 20.0), _CHARGE)
    result = detector.propagate_charged_tracks(plan, tracks)
    assert bool(jnp.all(result.accepted))
    np.testing.assert_array_equal(result.radiation_flags_history, 0)
    radiated = np.asarray(result.radiated_energy_history).sum(axis=0)[0]
    initial = _gamma(np.asarray(tracks.momenta)[0])
    final = _gamma(np.asarray(result.tracks.momenta)[0])
    np.testing.assert_allclose(radiated, _MASS * (initial - final), rtol=1e-12)
    exact = 1.0 / np.tanh(_TAU * steps * dt + np.arctanh(1.0 / initial))
    np.testing.assert_allclose(final, exact, rtol=2e-3)


def test_propagation_without_radiation_reaction_radiates_nothing() -> None:
    conditions = _conditions()
    plan = detector.ChargedPropagationPlan(
        conditions,
        step_size=0.01,
        step_count=10,
        pusher=RelativisticPushPlan(_scale().relativity, method="boris"),
    )
    tracks = _bank(conditions, (50.0, 20.0), _CHARGE)
    result = detector.propagate_charged_tracks(plan, tracks)
    np.testing.assert_array_equal(result.radiated_energy_history, 0.0)
    np.testing.assert_allclose(
        _gamma(np.asarray(result.tracks.momenta)[0]), [50.0, 20.0], rtol=1e-12
    )


def test_unsupported_reaction_is_not_committed_and_is_flagged() -> None:
    conditions = _conditions()
    # χ = γ β |q| B ħ/(m²c²) = 5e-6 at γ = 50 exceeds the declared bound.
    plan = detector.ChargedPropagationPlan(
        conditions,
        step_size=0.01,
        step_count=3,
        pusher=RelativisticPushPlan(_scale().relativity, method="boris"),
        radiation_reaction=_reaction("landau-lifshitz-reduced", maximum_chi=3.0e-6),
    )
    tracks = _bank(conditions, (50.0, 20.0), _CHARGE)
    result = detector.propagate_charged_tracks(plan, tracks)
    np.testing.assert_array_equal(result.accepted, [[False, True]])
    assert RadiationReactionFlag.CHI_EXCEEDED in RadiationReactionFlag(
        int(result.radiation_flags_history[0, 0, 0])
    )
    np.testing.assert_array_equal(result.radiated_energy_history[:, 0, 0], 0.0)
    np.testing.assert_array_equal(
        result.tracks.momenta[0, 0], np.asarray(tracks.momenta)[0, 0]
    )


def test_radiation_reaction_refuses_foreign_species_units_and_missing_keys() -> None:
    conditions = _conditions()
    pusher = RelativisticPushPlan(_scale().relativity, method="boris")
    plan = detector.ChargedPropagationPlan(
        conditions,
        step_size=0.01,
        step_count=2,
        pusher=pusher,
        radiation_reaction=_reaction("landau-lifshitz-reduced"),
    )
    with pytest.raises(ValueError, match="differ in charge or mass"):
        detector.propagate_charged_tracks(plan, _bank(conditions, (50.0, 20.0), 0.1))
    stochastic = _reaction(
        "stochastic-fokker-planck",
        tables=RadiationReactionTables(maximum_chi=1.0),
        maximum_chi=1.0,
    )
    with pytest.raises(ValueError, match="radiation_key"):
        detector.ChargedPropagationPlan(
            conditions,
            step_size=0.01,
            step_count=2,
            pusher=pusher,
            radiation_reaction=stochastic,
        )
    keyed = detector.ChargedPropagationPlan(
        conditions,
        step_size=0.01,
        step_count=2,
        pusher=pusher,
        radiation_reaction=stochastic,
        radiation_key=jr.key(0),
    )
    assert bool(
        jnp.all(
            detector.propagate_charged_tracks(
                keyed, _bank(conditions, (50.0, 20.0), _CHARGE)
            ).accepted
        )
    )
    with pytest.raises(ValueError, match="speed of light"):
        detector.ChargedPropagationPlan(
            conditions,
            step_size=0.01,
            step_count=2,
            radiation_reaction=RadiationReactionPlan(
                "landau-lifshitz-reduced",
                ElectromagneticScaleContract.code_units(
                    PIC_CODE_RELATIVITY.dimensional_scale,
                    UnitDefinition("code_charge", CHARGE, "phydrax:pic-code"),
                    gravitational_constant=1,
                    speed_of_light=2,
                    reduced_planck_constant=1,
                    boltzmann_constant=1,
                    elementary_charge=1,
                    electron_mass=1,
                    vacuum_permittivity=1,
                    constant_set_id="detector-radiation-reaction-test",
                ),
                _CHARGE,
                _MASS,
                maximum_chi=1.0,
                minimum_gamma=1.0,
            ),
        )
