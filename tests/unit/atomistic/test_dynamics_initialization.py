"""Initialization status, current-cell admission and required-field finiteness."""

from __future__ import annotations

from collections.abc import Iterator

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import config as pxconfig
from phydrax.atomistic._barostat import (
    apply_isotropic_monte_carlo_barostat,
    IsotropicMonteCarloBarostatPlan,
)
from phydrax.atomistic._dynamics import (
    AtomisticDynamicsPlan,
    AtomisticDynamicsStatus,
    AtomisticStepRejectionReason,
    PreparedAtomisticDynamics,
    VelocityVerletPlan,
)
from phydrax.atomistic._thermal import BAOABLangevinPlan
from phydrax.atomistic._thermodynamic import PreparedThermodynamicStateTable
from tests._support.atomistic_dynamics_fixtures import (
    atomistic_system,
    ExternalFieldTerm,
    measure,
    periodic_cell,
    prepare_potential,
    row,
    table,
)


@pytest.fixture(autouse=True)
def _double_precision() -> Iterator[None]:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        yield


POSITIONS = jnp.asarray([[0.1, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.3, 0.0]])


def _nonperiodic(
    term: ExternalFieldTerm,
) -> tuple[PreparedAtomisticDynamics, PreparedThermodynamicStateTable]:
    system = atomistic_system([0, 1, 2], [1.0] * 3)
    potential = prepare_potential(system, (term,))
    dynamics = AtomisticDynamicsPlan(
        system, potential, VelocityVerletPlan(1.0e-3)
    ).prepare()
    return dynamics, table(dynamics, row(measure(system), "nve"))


def _tether() -> ExternalFieldTerm:
    return ExternalFieldTerm("tether", "harmonic", [0.0, 0.0, 0.0], stiffness=1.0)


def test_unrequested_nan_stress_does_not_reject_prime_or_rebase() -> None:
    # Baseline (.tmp/force_cache_finiteness_probe.py): finite forces, NaN
    # unrequested stress, and prime/rebase rejected as NONFINITE.
    dynamics, thermodynamic = _nonperiodic(_tether())
    assert not dynamics.strain_response
    state = dynamics.initialize_state(
        POSITIONS,
        thermodynamic,
        velocity=jnp.zeros_like(POSITIONS),
        key=jax.random.key(0),
    )
    assert bool(jnp.all(jnp.isnan(state.force.stress)))
    primed = dynamics.prime_state(state, thermodynamic)
    assert bool(primed.successful), int(primed.rejection_reasons)
    assert int(primed.rejection_reasons) == 0
    rebased = dynamics.rebase_thermodynamic_state(state, thermodynamic, 0)
    assert bool(rebased.successful)
    # A genuinely nonfinite required field is still refused.
    poisoned = eqx.tree_at(
        lambda value: value.force.forces, state, state.force.forces.at[1, 0].set(jnp.nan)
    )
    assert not bool(dynamics.force_cache_finite(poisoned.force))


def test_monte_carlo_volume_move_without_strain_response_is_admitted() -> None:
    system = atomistic_system([0, 1], [1.0, 1.0], cell=periodic_cell(4.0 * np.eye(3)))
    potential = prepare_potential(system, (ExternalFieldTerm("zero", "zero", [0.0]),))
    dynamics = AtomisticDynamicsPlan(
        system, potential, BAOABLangevinPlan(0.01, 1.0), strain_response=False
    ).prepare()
    thermodynamic = table(
        dynamics, row(measure(system), "npt", temperature=1.0, pressure=1.0)
    )
    positions = jnp.asarray([[0.5, 1.0, 1.0], [2.5, 1.0, 1.0]])
    state = dynamics.initialize_state(
        positions,
        thermodynamic,
        velocity=jnp.zeros_like(positions),
        key=jax.random.key(1),
    )
    move = apply_isotropic_monte_carlo_barostat(
        dynamics, state, thermodynamic, IsotropicMonteCarloBarostatPlan(0.05)
    )
    assert bool(move.successful), int(move.rejection_reasons)
    assert not int(move.rejection_reasons) & int(AtomisticStepRejectionReason.NONFINITE)
    np.testing.assert_allclose(float(move.energy_after), 0.0)


def test_failed_potential_returns_an_explicitly_failed_initial_state() -> None:
    # The linear field with a NaN coefficient makes the scalar energy NaN.
    dynamics, thermodynamic = _nonperiodic(
        ExternalFieldTerm("broken", "linear", [np.nan, 0.0, 0.0])
    )
    evaluation = eqx.filter_jit(
        lambda positions: dynamics.initialize_state_detailed(
            positions,
            thermodynamic,
            velocity=jnp.zeros_like(positions),
            key=jax.random.key(2),
        )
    )(POSITIONS)
    assert not bool(evaluation.successful)
    reasons = int(evaluation.rejection_reasons)
    assert reasons & int(AtomisticStepRejectionReason.POTENTIAL)
    state = evaluation.state
    assert int(state.last_status) == int(AtomisticDynamicsStatus.INVALID_INITIAL_STATE)
    assert int(state.last_rejection_reasons) == reasons
    assert not bool(state.force.successful)
    # The actual configuration is kept, not replaced by a substitute state.
    np.testing.assert_array_equal(
        np.asarray(state.kinematics.positions), np.asarray(POSITIONS)
    )
    # No transition advances a failed initial state.
    step = dynamics.step_detailed(state, thermodynamic)
    assert not bool(step.successful)
    assert int(step.rejection_reasons) & int(AtomisticStepRejectionReason.STALE_FORCE)
    assert int(step.accepted_state.step_index) == 0
    # The admission-only form still raises on the same input.
    with pytest.raises(eqx.EquinoxRuntimeError, match="not admissible"):
        dynamics.initialize_state(
            POSITIONS,
            thermodynamic,
            velocity=jnp.zeros_like(POSITIONS),
            key=jax.random.key(2),
        )


def test_current_cell_vectors_preserve_the_unwrapped_configuration() -> None:
    system = atomistic_system([0, 1], [1.0, 1.0], cell=periodic_cell(4.0 * np.eye(3)))
    potential = prepare_potential(system, (ExternalFieldTerm("zero", "zero", [0.0]),))
    dynamics = AtomisticDynamicsPlan(
        system, potential, BAOABLangevinPlan(0.01, 1.0)
    ).prepare()
    thermodynamic = table(dynamics, row(measure(system), "nvt", temperature=1.0))
    # A later NPT stage left a 5.5 cell; the packet carries unwrapped positions.
    current = 5.5 * np.eye(3)
    unwrapped = jnp.asarray([[6.0, 1.0, -0.5], [2.5, 12.0, 1.0]])
    evaluation = dynamics.initialize_state_detailed(
        unwrapped,
        thermodynamic,
        velocity=jnp.zeros_like(unwrapped),
        key=jax.random.key(3),
        cell_vectors=current,
    )
    assert bool(evaluation.successful), int(evaluation.rejection_reasons)
    state = evaluation.state
    np.testing.assert_array_equal(np.asarray(state.cell_vectors), current)
    wrapped = np.asarray(state.kinematics.positions)
    images = np.asarray(state.kinematics.image_counts)
    assert np.all((wrapped >= 0.0) & (wrapped < 5.5))
    np.testing.assert_allclose(
        wrapped + images @ current, np.asarray(unwrapped), atol=1e-12
    )
    np.testing.assert_array_equal(images, [[1, 0, -1], [0, 2, 0]])
    # A nonperiodic system has no cell to bind.
    nonperiodic, nonperiodic_table = _nonperiodic(_tether())
    with pytest.raises(ValueError, match="no cell_vectors"):
        nonperiodic.initialize_state_detailed(
            POSITIONS,
            nonperiodic_table,
            velocity=jnp.zeros_like(POSITIONS),
            key=jax.random.key(4),
            cell_vectors=np.eye(3),
        )
    with pytest.raises(ValueError, match="cell_vectors must have shape"):
        dynamics.initialize_state_detailed(
            unwrapped,
            thermodynamic,
            velocity=jnp.zeros_like(unwrapped),
            key=jax.random.key(5),
            cell_vectors=np.eye(2),
        )
