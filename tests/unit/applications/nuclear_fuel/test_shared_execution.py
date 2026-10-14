"""Different scientific rods execute one numerical artifact without identity loss."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import pytest

from phydrax import config as pxconfig
from phydrax.applications.nuclear_fuel import (
    FuelNumericalExecution,
    FuelRodSimulation,
    FuelRodState,
    PreparedFuelRod,
)
from phydrax.applications.nuclear_fuel._step import FuelStepMethod
from phydrax.solver import FixedStepResult
from tests._support.assertions import assert_tree_close, assert_tree_values_equal
from tests._support.nuclear_fuel import (
    prepare_rod,
    synthetic_model_bundle,
    synthetic_rod_specification,
    wall_temperature_history,
)


_STEP_INDEX = 0
_TIME = 0.0  # [s]
_TIME_STEP = 0.2  # [s]


# Prepared rods are immutable; each float64 preparation is shared in-process.
@functools.cache
def execution_rod(
    *,
    pellet_radius: float = 4.095e-3,
    density_fraction: float = 0.95,
    scattering_cross_section_barn: float = 5.0,
    fuel_cells: int = 2,
) -> PreparedFuelRod:
    """Constant 100 W/m at a 300 K wall; arguments change only numerical inputs.

    Pellet radius, density, and scattering cross section are array data of one
    execution structure; the fuel cell count changes its layout.
    """
    specification = synthetic_rod_specification(
        pellet_radius=pellet_radius, density_fraction=density_fraction
    )
    history, coolant = wall_temperature_history(
        times=(0.0, 1.0), linear_power=(100.0, 100.0), wall_temperature=(300.0, 300.0)
    )
    return prepare_rod(
        specification,
        synthetic_model_bundle(
            mode_count=2, scattering_cross_section_barn=scattering_cross_section_barn
        ),
        history,
        coolant,
        reduction="layered-radial",
        fuel_cells=fuel_cells,
        cladding_cells=2,
    )


def _second_rod() -> PreparedFuelRod:
    return execution_rod(
        pellet_radius=4.02e-3, density_fraction=0.96, scattering_cross_section_barn=6.0
    )


@dataclass(frozen=True)
class _SharedStep:
    prepared: PreparedFuelRod
    shared: FixedStepResult


@pytest.fixture(scope="module")
def shared_steps() -> tuple[_SharedStep, _SharedStep]:
    """One step of two distinct rods through one execution built from the first."""
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        first, second = execution_rod(), _second_rod()
        owner = FuelNumericalExecution(first)
        index = jnp.asarray(_STEP_INDEX, dtype=jnp.int32)
        time = jnp.asarray(_TIME, dtype=jnp.float64)
        dt = jnp.asarray(_TIME_STEP, dtype=jnp.float64)

        def shared_step(prepared: PreparedFuelRod) -> _SharedStep:
            binding = owner.bind(prepared)
            return _SharedStep(
                prepared, binding.step(index, time, prepared.initial_state(), dt)
            )

        return shared_step(first), shared_step(second)


def test_distinct_rods_keep_distinct_scientific_identities(
    shared_steps: tuple[_SharedStep, _SharedStep],
) -> None:
    first, second = shared_steps
    assert first.prepared.prepared_id != second.prepared.prepared_id
    assert (
        first.prepared.admission.bundle.radial_power.law_id
        != second.prepared.admission.bundle.radial_power.law_id
    )
    for case in shared_steps:
        result, prepared = case.shared, case.prepared
        assert result.evidence is not None
        assert isinstance(result.accepted_state, FuelRodState)
        assert isinstance(result.candidate_state, FuelRodState)
        assert result.accepted_state.prepared_id == prepared.prepared_id
        assert result.candidate_state.prepared_id == prepared.prepared_id
        assert result.evidence.prepared_id == prepared.prepared_id
        assert result.evidence.bundle_id == prepared.admission.bundle.bundle_id
        assert result.evidence.history_id == prepared.admission.history.history_id


def test_shared_execution_solves_each_rod_with_its_own_numerical_inputs(
    shared_steps: tuple[_SharedStep, _SharedStep],
) -> None:
    first, second = (case.shared.candidate_state for case in shared_steps)
    assert isinstance(first, FuelRodState) and isinstance(second, FuelRodState)
    assert float(jnp.max(first.fields.temperature_fuel)) != float(
        jnp.max(second.fields.temperature_fuel)
    )


@pytest.mark.parametrize("index", [0, 1], ids=["template-rod", "bound-rod"])
def test_shared_step_matches_the_case_bound_step(
    shared_steps: tuple[_SharedStep, _SharedStep], index: int
) -> None:
    case = shared_steps[index]
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        native = eqx.filter_jit(FuelStepMethod(case.prepared).step)(
            jnp.asarray(_STEP_INDEX, dtype=jnp.int32),
            jnp.asarray(_TIME, dtype=jnp.float64),
            case.prepared.initial_state(),
            jnp.asarray(_TIME_STEP, dtype=jnp.float64),
            None,
        )
    shared = case.shared
    assert shared.evidence is not None and native.evidence is not None
    assert bool(shared.successful) == bool(native.successful)
    assert bool(shared.terminal_failure) == bool(native.terminal_failure)
    assert int(shared.evidence.status) == int(native.evidence.status)
    assert_tree_close(
        shared.candidate_state, native.candidate_state, rtol=2e-8, atol=1e-10
    )
    assert_tree_close(shared.accepted_state, native.accepted_state, rtol=2e-8, atol=1e-10)


def test_shared_execution_refuses_a_rod_of_different_layout() -> None:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        owner = FuelNumericalExecution(execution_rod())
        with pytest.raises(
            ValueError, match="layout, topology, or scientific role mismatch"
        ):
            owner.bind(execution_rod(fuel_cells=3))


def test_binding_refuses_a_checkpoint_of_another_rod() -> None:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        prepared = execution_rod()
        binding = FuelNumericalExecution(prepared).bind(_second_rod())
        with pytest.raises(ValueError, match="different prepared FE identity"):
            binding.step(
                jnp.asarray(_STEP_INDEX, dtype=jnp.int32),
                jnp.asarray(_TIME, dtype=jnp.float64),
                prepared.initial_state(),
                jnp.asarray(_TIME_STEP, dtype=jnp.float64),
            )


def test_shared_execution_retains_independent_case_checkpoints(
    tmp_path: Path,
) -> None:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        first, second = execution_rod(), _second_rod()
        owner = FuelNumericalExecution(first)
        runs = tuple(
            FuelRodSimulation(
                prepared,
                time_step=_TIME_STEP,
                checkpoint_directory=tmp_path / name,
                numerical_execution=owner,
            )
            for prepared, name in ((first, "first"), (second, "second"))
        )
        checkpoints = []
        for prepared, run in zip((first, second), runs, strict=True):
            state, attempt = run.runtime.step(
                run.runtime.initial_state(prepared.initial_state())
            )
            assert bool(attempt.successful)
            checkpoints.append(run.runtime.checkpoint(state))
        restored = tuple(
            run.runtime.resume(checkpoint)
            for run, checkpoint in zip(runs, checkpoints, strict=True)
        )
        run_ids = tuple(run.runtime.run_id for run in runs)
        for run in runs:
            run.close()
    assert run_ids[0] != run_ids[1]
    for prepared, checkpoint, resumed in zip(
        (first, second), checkpoints, restored, strict=True
    ):
        assert isinstance(checkpoint.accepted_state, FuelRodState)
        assert checkpoint.accepted_state.prepared_id == prepared.prepared_id
        assert_tree_values_equal(resumed.accepted_state, checkpoint.accepted_state)
