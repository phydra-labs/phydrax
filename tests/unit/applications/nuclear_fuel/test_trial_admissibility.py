"""The prepared rod's Newton trial domain refuses unphysical solver trials."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.applications.nuclear_fuel import FuelRodState, PreparedFuelRod
from tests._support.nuclear_fuel import prepare_low_power_startup_rod


def _axial_strain_fuel(strain: float) -> Callable[[FuelRodState], FuelRodState]:
    def perturb(state: FuelRodState) -> FuelRodState:
        return eqx.tree_at(
            lambda record: record.fields.axial_strain_fuel,
            state,
            jnp.full_like(state.fields.axial_strain_fuel, strain),
        )

    return perturb


def _axial_strain_cladding(strain: float) -> Callable[[FuelRodState], FuelRodState]:
    def perturb(state: FuelRodState) -> FuelRodState:
        return eqx.tree_at(
            lambda record: record.fields.axial_strain_cladding,
            state,
            jnp.full_like(state.fields.axial_strain_cladding, strain),
        )

    return perturb


def _negative_species_positive_total(state: FuelRodState) -> FuelRodState:
    # Canonical He, Ar, Kr, Xe, N2 axis: a positive total cannot hide -1 mol Ar.
    return eqx.tree_at(
        lambda record: record.fields.free_moles,
        state,
        jnp.asarray([2.0, -1.0, 0.0, 0.0, 0.0], dtype=jnp.float64),
    )


@pytest.fixture(scope="module")
def prepared() -> PreparedFuelRod:
    with jax.enable_x64():
        return prepare_low_power_startup_rod()


def test_fabricated_state_is_an_admissible_trial(prepared: PreparedFuelRod) -> None:
    with jax.enable_x64():
        problem = prepared.nonlinear_problem
        unknown = prepared.pack_unknown(prepared.initial_state())
        assert bool(problem.trial_valid(unknown, prepared.nonlinear_template.args))


# Axial stretch 1 + strain: -1 makes the deformation gradient singular and -2
# inverts its orientation.
@pytest.mark.parametrize(
    "perturb",
    [
        _axial_strain_fuel(-1.0),
        _axial_strain_fuel(-2.0),
        _axial_strain_cladding(-1.0),
        _axial_strain_cladding(-2.0),
        _negative_species_positive_total,
    ],
    ids=[
        "fuel-singular",
        "fuel-inverted",
        "cladding-singular",
        "cladding-inverted",
        "negative-species-positive-total",
    ],
)
def test_unphysical_trial_is_outside_the_newton_trial_domain(
    prepared: PreparedFuelRod, perturb: Callable[[FuelRodState], FuelRodState]
) -> None:
    with jax.enable_x64():
        problem = prepared.nonlinear_problem
        unknown = prepared.pack_unknown(perturb(prepared.initial_state()))
        assert not bool(problem.trial_valid(unknown, prepared.nonlinear_template.args))
