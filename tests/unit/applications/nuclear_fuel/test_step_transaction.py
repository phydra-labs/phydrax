"""Accepted-state rollback for a real coupled, increment-limited thermal attempt."""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import config as pxconfig
from phydrax.applications.nuclear_fuel import (
    FUEL_GAS_SPECIES,
    FuelAcceptancePolicy,
    FuelGeometryReduction,
    FuelRodResult,
    FuelRodState,
)
from phydrax.applications.nuclear_fuel._acceptance import (
    FuelStepStatus,
    physical_increments,
)
from phydrax.applications.nuclear_fuel._step import evidence_prototype, FuelStepMethod
from phydrax.solver import FixedStepResult
from tests._support.assertions import assert_tree_equal
from tests._support.nuclear_fuel import prepare_low_power_startup_rod


# A 1e-4 K temperature-increment limit refuses the first 1 s step, during which
# the prescribed wall heats from 300 K towards 305 K.
_TEMPERATURE_INCREMENT_LIMIT = 1e-4  # [K]


@pytest.fixture(
    scope="module", params=("layered-radial", "axisymmetric-rz"), ids=("gps", "rz")
)
def increment_limited_attempt(
    request: pytest.FixtureRequest,
) -> tuple[FuelRodState, FixedStepResult]:
    reduction: FuelGeometryReduction = request.param
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        prepared = prepare_low_power_startup_rod(reduction=reduction)
        initial = prepared.initial_state()
        method = FuelStepMethod(
            prepared,
            policy=FuelAcceptancePolicy(
                temperature_increment=_TEMPERATURE_INCREMENT_LIMIT
            ),
        )
        attempt = method.step(
            jnp.asarray(0, dtype=jnp.int32), initial.time, initial, jnp.asarray(1.0), None
        )
        return initial, attempt


def test_refused_increment_keeps_every_accepted_history(
    increment_limited_attempt: tuple[FuelRodState, FixedStepResult],
) -> None:
    initial, attempt = increment_limited_attempt
    assert not bool(attempt.successful)
    assert_tree_equal(attempt.accepted_state, initial)


def test_refused_candidate_reports_increment_limit_and_retains_its_heating(
    increment_limited_attempt: tuple[FuelRodState, FixedStepResult],
) -> None:
    _, attempt = increment_limited_attempt
    assert attempt.evidence is not None
    assert int(attempt.evidence.status) == FuelStepStatus.INCREMENT_LIMIT
    assert not bool(attempt.evidence.accepted)
    candidate = attempt.candidate_state
    assert isinstance(candidate, FuelRodState)
    assert float(jnp.max(candidate.fields.temperature_cladding)) > 300.01


def test_release_fraction_increment_propagates_invalid_production() -> None:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        previous = prepare_low_power_startup_rod().initial_state()
        krypton = FUEL_GAS_SPECIES.index("Kr")

        def release_increment(produced: float, released: float) -> float:
            candidate = eqx.tree_at(
                lambda state: (state.gas.produced_moles, state.gas.released_moles),
                previous,
                (
                    previous.gas.produced_moles.at[krypton].set(produced),
                    previous.gas.released_moles.at[krypton].set(released),
                ),
            )
            return float(physical_increments(previous, candidate)[2])

        assert release_increment(0.0, 0.0) == 0.0
        assert release_increment(4.0, 1.0) == 0.25
        assert np.isnan(release_increment(float("nan"), 1.0))
        assert release_increment(-4.0, 1.0) == 0.25


def test_result_refuses_attempts_of_another_history() -> None:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        prepared = prepare_low_power_startup_rod()
        attempt = evidence_prototype(prepared, FuelAcceptancePolicy())
        foreign = dataclasses.replace(attempt, history_id="another-history")
        # The identity check precedes any use of the runtime result.
        with pytest.raises(ValueError, match="history IDs"):
            FuelRodResult(
                (prepared.initial_state(),),
                (attempt, foreign),
                None,  # ty: ignore[invalid-argument-type]
                prepared.prepared_id,
                bundle_id=prepared.admission.bundle.bundle_id,
                history_id=prepared.admission.history.history_id,
            )
