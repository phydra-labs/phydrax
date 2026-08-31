#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.discretization.lattice_boltzmann import (
    SpeciesLatticeBoltzmannState,
    SpeciesLedger,
    ThermalEnergyLedger,
    ThermalLatticeBoltzmannState,
)
from phydrax.solver import (
    ReactiveLocalStepResult,
    ReactiveSpeciesCouplingSchedulePlan,
    ReactiveSpeciesLatticeBoltzmannState,
)


class _NoopReaction:
    def step(self, species_amount, sensible_energy, step_size, args, /):
        del step_size, args
        return ReactiveLocalStepResult(
            species_amount,
            sensible_energy,
            jnp.zeros(species_amount.shape[:-1]),
            jnp.zeros_like(sensible_energy),
            jnp.zeros_like(sensible_energy),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(True),
        )


class _FailingReaction:
    def step(self, species_amount, sensible_energy, step_size, args, /):
        del step_size, args
        return ReactiveLocalStepResult(
            2.0 * species_amount,
            sensible_energy + 1.0,
            jnp.ones(species_amount.shape[:-1]),
            jnp.zeros_like(sensible_energy),
            jnp.zeros_like(sensible_energy),
            jnp.asarray(2, dtype=jnp.int32),
            jnp.asarray(False),
        )


def _state():
    thermal_populations = jnp.full((2, 2, 9), 2.0 / 9.0)
    species_populations = jnp.stack(
        (
            jnp.full((2, 2, 9), 1.0 / 9.0),
            jnp.full((2, 2, 9), 0.5 / 9.0),
        ),
        axis=-2,
    )
    thermal = ThermalLatticeBoltzmannState(
        thermal_populations,
        ThermalEnergyLedger(
            jnp.asarray(8.0),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
        ),
        jnp.asarray(True),
        jnp.asarray(0, dtype=jnp.int32),
        "reactive-thermal-test",
    )
    species = SpeciesLatticeBoltzmannState(
        species_populations,
        SpeciesLedger(
            jnp.asarray((4.0, 2.0)),
            jnp.zeros((2,)),
            jnp.zeros((2,)),
            jnp.zeros((2,)),
            jnp.zeros((2,)),
            jnp.asarray((6.0,)),
            jnp.zeros((1,)),
            jnp.zeros((1,)),
            jnp.zeros((1,)),
        ),
        jnp.asarray(True),
        jnp.asarray(0, dtype=jnp.int32),
        "reactive-species-test",
    )
    return ReactiveSpeciesLatticeBoltzmannState(
        thermal,
        species,
        jnp.zeros((2, 2)),
        jnp.asarray((6.0,)),
        jnp.asarray(True),
    )


def _identity_transport(thermal, species, args):
    del args
    return thermal, species, jnp.asarray(True)


def test_reactive_strang_schedule_preserves_noop_state():
    state = _state()
    result = ReactiveSpeciesCouplingSchedulePlan(reaction_substeps=2).advance(
        state,
        jnp.asarray(0.1),
        _NoopReaction(),
        _identity_transport,
    )

    assert result.successful
    assert result.diagnostics.maximum_iterations == 1
    np.testing.assert_array_equal(
        result.accepted_state.thermal.populations,
        state.thermal.populations,
    )
    np.testing.assert_array_equal(
        result.accepted_state.species.populations,
        state.species.populations,
    )


def test_reactive_failure_rolls_back_all_coupled_fields_atomically():
    state = _state()
    result = ReactiveSpeciesCouplingSchedulePlan().advance(
        state,
        jnp.asarray(0.1),
        _FailingReaction(),
        _identity_transport,
    )

    assert not result.successful
    np.testing.assert_array_equal(
        result.accepted_state.thermal.populations,
        state.thermal.populations,
    )
    np.testing.assert_array_equal(
        result.accepted_state.species.populations,
        state.species.populations,
    )
    np.testing.assert_array_equal(
        result.accepted_state.reaction_extent,
        state.reaction_extent,
    )
