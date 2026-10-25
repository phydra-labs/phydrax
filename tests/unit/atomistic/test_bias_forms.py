"""Conservative bias terms: metric forms, full-matrix umbrellas and cell response."""

from __future__ import annotations

from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import config as pxconfig
from phydrax.atomistic import AtomisticThermodynamicStatePlan, PreparedAtomisticSystem
from phydrax.atomistic._barostat import (
    apply_isotropic_monte_carlo_barostat,
    IsotropicMonteCarloBarostatPlan,
)
from phydrax.atomistic._cell_dynamics import MTKBarostatPlan, MTKNPTPlan
from phydrax.atomistic._thermal import BAOABLangevinPlan, NoseHooverChainParameters
from phydrax.atomistic._thermodynamic import AtomisticBiasRowBinding
from phydrax.atomistic.sampling import (
    AtomisticBiasTerm,
    CollectiveVariableGeometry,
    CollectiveVariableKind,
    CollectiveVariableMetric,
    FlatBottomBiasForm,
    HarmonicBiasForm,
    PreparedBiasedDynamics,
    WallBiasForm,
    WallSide,
)
from phydrax.atomistic.sampling._bias import bind_bias_row
from tests._support.atomistic_bundles import bundle_state
from tests._support.atomistic_dynamics_fixtures import (
    atomistic_system,
    measure,
    periodic_cell,
    prepare_potential,
    row,
    table,
)
from tests._support.atomistic_sampling_fixtures import (
    bias_term,
    biased_dynamics,
    cv_program,
    initial_bindings,
    start,
    zero_field,
)


@pytest.fixture(autouse=True)
def _double_precision() -> Iterator[None]:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        yield


KAPPA, V0 = 1.0, 50.0
POSITIONS = jnp.asarray([[0.5, 1.0, 1.0], [2.5, 1.0, 1.0]])


def _volume_umbrella(system: PreparedAtomisticSystem) -> AtomisticBiasTerm:
    variables = cv_program(
        system, CollectiveVariableKind.CELL_VOLUME, np.zeros((0,), np.int64)
    )
    return bias_term(system, variables, HarmonicBiasForm(np.asarray([V0]), KAPPA))


def test_volume_umbrella_stress_is_tension_positive() -> None:
    # Baseline defect 4 (stress omitted): sym(dV_b/d strain)/V = kappa (V - V0) I.
    system = atomistic_system([0, 1], [1.0, 1.0], cell=periodic_cell(4.0 * np.eye(3)))
    potential = prepare_potential(system, (zero_field(), _volume_umbrella(system)))
    evaluation = potential.evaluate(
        POSITIONS,
        bundle_state(potential, POSITIONS),
        compute_stress=True,
        term_states=initial_bindings(potential),
    )
    assert bool(evaluation.successful)
    expected = KAPPA * (64.0 - V0)
    np.testing.assert_allclose(
        np.asarray(evaluation.stress), expected * np.eye(3), atol=1e-10
    )
    np.testing.assert_allclose(float(evaluation.energy), 0.5 * KAPPA * (64.0 - V0) ** 2)
    # Pressure contribution has the opposite sign of the tensile stress.
    assert float(-jnp.trace(evaluation.stress) / 3.0) < 0.0


def test_monte_carlo_volume_trial_includes_the_bias() -> None:
    # Baseline defect 4 (MC omitted the umbrella: E_after 0 vs 62.07).
    system = atomistic_system([0, 1], [1.0, 1.0], cell=periodic_cell(4.0 * np.eye(3)))
    dynamics = biased_dynamics(
        system, _volume_umbrella(system), BAOABLangevinPlan(0.01, 1.0)
    )
    thermodynamic = table(
        dynamics, row(measure(system), "npt", temperature=1.0, pressure=1.0)
    )
    state = start(dynamics, thermodynamic, POSITIONS, seed=1)
    move = apply_isotropic_monte_carlo_barostat(
        dynamics, state, thermodynamic, IsotropicMonteCarloBarostatPlan(0.05)
    )
    before, after = float(move.volume_before), float(move.volume_after)
    np.testing.assert_allclose(
        float(move.energy_before), 0.5 * KAPPA * (before - V0) ** 2
    )
    np.testing.assert_allclose(float(move.energy_after), 0.5 * KAPPA * (after - V0) ** 2)


def test_integrator_owned_cell_dynamics_admits_a_conservative_bias() -> None:
    # Baseline defect 4 (the biased runtime refused integrator-owned MTK cells).
    system = atomistic_system([0, 1], [1.0, 1.0], cell=periodic_cell(4.0 * np.eye(3)))
    integrator = MTKNPTPlan(
        0.005,
        MTKBarostatPlan(0.5),
        NoseHooverChainParameters(0.5),
        NoseHooverChainParameters(0.5),
    )
    dynamics = biased_dynamics(system, _volume_umbrella(system), integrator)
    thermodynamic = table(
        dynamics, row(measure(system), "npt", temperature=1.0, pressure=1.0)
    )
    runtime = PreparedBiasedDynamics(dynamics, thermodynamic, physical_force_groups=(0,))
    state = runtime.initialize(start(dynamics, thermodynamic, POSITIONS))
    # V = 64 in the reference cell: tensile stress kappa (V - V0) I = 14 I.
    np.testing.assert_allclose(
        np.asarray(state.dynamics.force.stress),
        KAPPA * (64.0 - V0) * np.eye(3),
        atol=1e-10,
    )
    stepped = runtime.step(state)
    assert bool(stepped.successful)
    accepted = stepped.accepted_state.dynamics
    assert int(accepted.step_index) == 1
    np.testing.assert_allclose(
        np.asarray(accepted.force.stress),
        KAPPA * (float(jnp.linalg.det(accepted.cell_vectors)) - V0) * np.eye(3),
        atol=1e-8,
    )


def test_full_matrix_umbrella_requires_psd_and_uses_metric_displacement() -> None:
    with pytest.raises(ValueError, match="semidefinite"):
        HarmonicBiasForm(np.asarray([0.0, 0.0]), np.asarray([[1.0, 2.0], [2.0, 1.0]]))
    form = HarmonicBiasForm(np.asarray([0.1, 3.0]), np.asarray([[2.0, 0.5], [0.5, 1.0]]))
    assert form.stiffness_minimum_eigenvalue > 0.0
    metrics = (CollectiveVariableMetric(), CollectiveVariableMetric(period=2.0 * np.pi))
    value = jnp.asarray([0.3, -3.0])
    evaluation = form.evaluate(value, metrics, form.center, None, jnp.int32(0))
    delta = np.asarray([0.2, (-3.0 - 3.0 + np.pi) % (2.0 * np.pi) - np.pi])
    np.testing.assert_allclose(
        float(evaluation.energy), 0.5 * delta @ np.asarray(form.stiffness) @ delta
    )
    assert bool(evaluation.successful)


def test_walls_require_a_periodic_branch_and_have_the_declared_continuity() -> None:
    periodic = (CollectiveVariableMetric(period=2.0 * np.pi),)
    wall = WallBiasForm(np.asarray([1.0]), 3.0, WallSide.UPPER, exponent=4.0)
    with pytest.raises(ValueError, match="branch"):
        wall.evaluate(jnp.asarray([2.0]), periodic, jnp.zeros((0,)), None, jnp.int32(0))
    branched = WallBiasForm(
        np.asarray([1.0]),
        3.0,
        WallSide.UPPER,
        exponent=4.0,
        branch_start=np.asarray([-np.pi]),
    )
    evaluation = branched.evaluate(
        jnp.asarray([2.0]), periodic, jnp.zeros((0,)), None, jnp.int32(0)
    )
    np.testing.assert_allclose(float(evaluation.energy), 3.0 / 4.0)
    flat = FlatBottomBiasForm(np.asarray([0.0]), np.asarray([1.0]), 2.0)
    plain = (CollectiveVariableMetric(),)
    slope = jax.grad(
        lambda x: (
            flat.evaluate(
                jnp.stack([x]), plain, jnp.zeros((0,)), None, jnp.int32(0)
            ).energy
        )
    )
    assert float(slope(jnp.asarray(1.0))) == 0.0
    np.testing.assert_allclose(float(slope(jnp.asarray(1.5))), 1.0)


def test_nonempty_bias_rows_bind_distinct_hamiltonians() -> None:
    # Baseline defect 5: nonempty table bias IDs were refused.
    system = atomistic_system([0, 1, 2], [1.0] * 3)
    variables = cv_program(system, CollectiveVariableKind.DISTANCE, [0, 1])
    term = bias_term(system, variables, HarmonicBiasForm(np.asarray([1.0]), 5.0))
    dynamics = biased_dynamics(system, term, BAOABLangevinPlan(0.01, 1.0))
    rows = tuple(
        AtomisticThermodynamicStatePlan(
            measure(system),
            ensemble="nvt",
            temperature=1.0,
            bias=AtomisticBiasRowBinding({term.name: np.asarray([center])}),
        )
        for center in (1.0, 1.5)
    )
    thermodynamic = table(dynamics, *rows)
    assert thermodynamic.bias_ids[0] != thermodynamic.bias_ids[1]
    assert thermodynamic.requires_cross_evaluation
    assert np.asarray(thermodynamic.hamiltonian_index).tolist() == [0, 1]
    assert len(set(thermodynamic.potential_ids)) == 2
    geometry = CollectiveVariableGeometry(
        jnp.asarray([[0.0, 0, 0], [1.2, 0, 0], [0, 2.0, 0]])
    )
    prepared = next(t for t in dynamics.potential.terms if t.term_id == term.term_id)
    defaults = initial_bindings(dynamics.potential)
    # Each row realizes its own umbrella: 0.5 k (1.2 - c)^2 = 0.1 and 0.225.
    for index, center in enumerate((1.0, 1.5)):
        bound = bind_bias_row((prepared,), thermodynamic.bias_rows[index], defaults)
        energy, _, ok = prepared.bias(geometry, bound[term.name])
        assert bool(ok)
        np.testing.assert_allclose(
            float(energy), 0.5 * 5.0 * (1.2 - center) ** 2, rtol=1e-12
        )
