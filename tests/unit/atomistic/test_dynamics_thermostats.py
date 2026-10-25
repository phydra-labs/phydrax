#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""In-schedule thermostats: NHC, CSVR, deterministic rescaling, GLE, DOF and RNG."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax import config as pxconfig
from phydrax.atomistic._constraints import DistanceConstraintPlan
from phydrax.atomistic._rollout import AtomisticRolloutPlan, AtomisticTrajectoryPlan
from phydrax.atomistic._stochastic_runtime import (
    apply_generalized_langevin,
    GeneralizedLangevinPlan,
)
from phydrax.atomistic._thermal import (
    apply_baoab_ornstein_uhlenbeck,
    BAOAB_OPERATOR_ID,
    BAOABLangevinPlan,
    CSVR_FIRST_OPERATOR_ID,
    NoseHooverChainParameters,
    NoseHooverChainPlan,
    stable_particle_normals,
    stochastic_velocity_rescaling_scale,
    StochasticVelocityRescalingPlan,
    VelocityRescalingPlan,
)
from phydrax.atomistic._topology import MolecularTopologyPlan
from tests._support.atomistic_dynamics_fixtures import (
    assert_tree_equal,
    atomistic_system,
    ExternalFieldTerm,
    measure,
    periodic_cell,
    prepare_dynamics,
    row,
    step_detailed,
    table,
)


HARMONIC_MASSES = np.asarray([1.0, 2.0, 1.5, 0.5])
GAS_MASSES = np.asarray([1.0, 2.0, 0.5, 1.5, 1.0, 3.0, 0.75, 1.25])
# Discrete GLE half-step propagator with one auxiliary channel:
# T T^T = 0.45 I and S S^T = 0.55 I, so T T^T + S S^T = I (discrete FDT).
GLE_TRANSITION = np.asarray([[0.6, 0.3], [-0.3, 0.6]])
GLE_NOISE = np.sqrt(0.55) * np.eye(2)


@pytest.fixture(autouse=True)
def _double_precision() -> Iterator[None]:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        yield


# --- fixtures ------------------------------------------------------------------


def _harmonic_runtime(
    integrator: Any, temperatures: tuple[float, ...]
) -> tuple[Any, Any, Any]:
    """Four tethered oscillators ``k/2 |x|^2`` (k = 1) started at twice the target."""
    system = atomistic_system(list(range(4)), HARMONIC_MASSES)
    term = ExternalFieldTerm("tether", "harmonic", np.zeros((4, 3)), stiffness=1.0)
    dynamics = prepare_dynamics(system, (term,), integrator)
    phase_space = measure(system)
    thermodynamic = table(
        dynamics, *(row(phase_space, "nvt", temperature=value) for value in temperatures)
    )
    rng = np.random.default_rng(21)
    positions = rng.normal(size=(4, 3))
    velocity = (
        rng.normal(size=(4, 3))
        * np.sqrt(2.0 * temperatures[0] / HARMONIC_MASSES)[:, None]
    )
    state = dynamics.initialize_state(
        jnp.asarray(positions),
        thermodynamic,
        velocity=jnp.asarray(velocity),
        key=jr.key(21),
    )
    return dynamics, thermodynamic, state


def _ideal_gas_runtime(
    integrator: Any,
    *,
    temperature: float,
    ids: Any = None,
    masses: np.ndarray = GAS_MASSES,
    positions: np.ndarray | None = None,
    velocity: np.ndarray | None = None,
    seed: int = 31,
) -> tuple[Any, Any, Any]:
    """Free particles: a zero field, no neighborhood, no cell."""
    count = masses.shape[0]
    system = atomistic_system(list(range(count)) if ids is None else ids, masses)
    term = ExternalFieldTerm("ideal-gas", "zero", np.zeros((count, 3)))
    dynamics = prepare_dynamics(system, (term,), integrator)
    thermodynamic = table(dynamics, row(measure(system), "nvt", temperature=temperature))
    rng = np.random.default_rng(seed)
    position = rng.uniform(-2.0, 2.0, size=(count, 3)) if positions is None else positions
    speed = rng.normal(size=(count, 3)) if velocity is None else velocity
    state = dynamics.initialize_state(
        jnp.asarray(position),
        thermodynamic,
        velocity=jnp.asarray(speed),
        key=jr.key(seed),
    )
    return dynamics, thermodynamic, state


def _kinetic(momenta: Any, masses: np.ndarray) -> float:
    value = np.asarray(momenta)
    return float(0.5 * np.sum(value * value / masses[:, None]))


def _chain_energy(
    parameters: NoseHooverChainParameters, chain: Any, dof: int, thermal: float
) -> float:
    """``sum Q v^2/2 + N_f kT xi_1 + kT sum_{k>1} xi_k`` with Q_1 = N_f kT tau^2."""
    tau_squared = parameters.time_constant**2
    masses = np.full((parameters.chain_length,), thermal * tau_squared)
    masses[0] = dof * thermal * tau_squared
    velocities = np.asarray(chain.velocities)
    positions = np.asarray(chain.positions)
    weights = np.ones_like(positions)
    weights[0] = dof
    return float(
        0.5 * np.sum(masses * velocities * velocities)
        + thermal * np.sum(weights * positions)
    )


# --- Nose-Hoover chains --------------------------------------------------------


def test_nose_hoover_chain_conserves_extended_energy_for_every_suzuki_order() -> None:
    with pytest.raises(ValueError, match="1, 3 or 5"):
        NoseHooverChainParameters(0.5, yoshida_suzuki_order=2)
    for order in (1, 3, 5):
        chain = NoseHooverChainParameters(0.5, chain_length=3, yoshida_suzuki_order=order)
        dynamics, thermodynamic, state = _harmonic_runtime(
            NoseHooverChainPlan(0.01, chain), (1.0,)
        )
        initial = float(state.energy.initial_extended_energy)
        worst = 0.0
        for _ in range(300):
            evaluation = step_detailed(dynamics, state, thermodynamic)
            assert bool(evaluation.successful)
            state = evaluation.accepted_state
            worst = max(worst, abs(float(state.energy.extended_energy) - initial))
        # Second-order splitting error ~ (omega dt)^2/8 ~ 2.5e-5 per mode
        # (omega <= sqrt(2), dt = 0.01) plus ~(dt/tau)^2/8 = 5e-5 from the chain;
        # 1e-3 relative keeps a >10x margin and still excludes an O(dt) drift.
        assert worst < 1.0e-3 * abs(initial)
        # The chain did real work (started at T = 2 T_target), so conservation is
        # not trivial.
        assert abs(float(state.energy.thermostat_heat)) > 1.0e-2
        units = dynamics.system.plan.units
        expected = (
            _kinetic(state.kinematics.momenta, HARMONIC_MASSES)
            + float(state.force.potential_energy)
            + _chain_energy(
                chain,
                state.controllers.particle_chain,
                thermodynamic.degrees_of_freedom,
                units.boltzmann_constant * 1.0,
            )
        )
        # Ledger identity H = K + U + E_chain with an independent host chain energy.
        np.testing.assert_allclose(state.energy.extended_energy, expected, rtol=1e-12)


# --- stochastic velocity rescaling ---------------------------------------------


def test_csvr_exact_propagator_samples_the_canonical_mean_kinetic_energy() -> None:
    temperature = 1.5
    dynamics, thermodynamic, state = _ideal_gas_runtime(
        StochasticVelocityRescalingPlan(0.05, 0.1), temperature=temperature
    )
    steps, burn_in, blocks = 20_000, 2_000, 18
    result = AtomisticRolloutPlan(
        dynamics,
        thermodynamic,
        AtomisticTrajectoryPlan(steps, include_initial=False),
    ).rollout(state)
    assert bool(result.successful)
    assert bool(np.all(np.asarray(result.trajectory.valid)))
    kinetic = np.asarray(result.trajectory.energies[:, 0])
    assert kinetic.shape == (steps,)
    means = kinetic[burn_in:].reshape((blocks, -1)).mean(axis=1)
    mean = float(means.mean())
    standard_error = float(means.std(ddof=1) / np.sqrt(blocks))
    thermal = dynamics.system.plan.units.boltzmann_constant * temperature
    target = 0.5 * thermodynamic.degrees_of_freedom * thermal
    # Without forces K is the exact CSVR Markov chain with Gamma(N_f/2, kT)
    # stationary law; its correlation time is ~tau/dt = 2 steps, so 1000-step
    # blocks are independent. Declared interval: |mean - N_f kT/2| <= 4 SE
    # (two-sided t_17 tail ~1e-3). Power precondition: 4 SE < kT/2, so a
    # one-degree-of-freedom miscount would be detected.
    assert 4.0 * standard_error < 0.5 * thermal
    assert abs(mean - target) <= 4.0 * standard_error


# --- deterministic velocity rescaling ------------------------------------------


def test_deterministic_rescaling_hits_the_target_on_due_steps_only() -> None:
    plan = VelocityRescalingPlan(0.01, interval=2)
    assert plan.canonical is False
    assert plan.stochastic is False
    temperature = 0.7
    dynamics, thermodynamic, state = _ideal_gas_runtime(plan, temperature=temperature)
    initial = float(dynamics.diagnostics(state, thermodynamic).temperature)
    assert abs(initial - temperature) > 1e-3
    first = step_detailed(dynamics, state, thermodynamic)
    assert bool(first.successful)
    after_first = first.accepted_state
    # Step 1 is not due (interval 2): free flight leaves K bitwise unchanged.
    np.testing.assert_allclose(
        dynamics.diagnostics(after_first, thermodynamic).temperature, initial, rtol=1e-14
    )
    assert int(after_first.controllers.rescale_events) == 0
    second = step_detailed(dynamics, after_first, thermodynamic)
    assert bool(second.successful)
    after_second = second.accepted_state
    # K' = s^2 K with s^2 = N_f kT / (2K): T(K') = T_target up to a few roundings.
    np.testing.assert_allclose(
        dynamics.diagnostics(after_second, thermodynamic).temperature,
        temperature,
        rtol=1e-13,
    )
    assert int(after_second.controllers.rescale_events) == 1
    # The injected energy is reservoir heat (noncanonical control).
    np.testing.assert_allclose(
        float(after_second.energy.reservoir_heat),
        float(after_second.energy.kinetic_energy)
        - float(after_first.energy.kinetic_energy),
        rtol=1e-12,
        atol=1e-14,
    )

    system = atomistic_system(
        list(range(4)), GAS_MASSES[:4], cell=periodic_cell(4.0 * np.eye(3))
    )
    periodic = prepare_dynamics(
        system, (ExternalFieldTerm("ideal-gas", "zero", np.zeros((4, 3))),), plan
    )
    with pytest.raises(ValueError, match="cannot execute ensembles"):
        table(periodic, row(measure(system), "npt", temperature=0.7, pressure=1.0))


# --- generalized Langevin ------------------------------------------------------


def test_generalized_langevin_is_an_addressed_in_schedule_method() -> None:
    with pytest.raises(ValueError, match="discrete FDT"):
        GeneralizedLangevinPlan(0.01, 0.9 * np.eye(2), 0.1 * np.eye(2))
    plan = GeneralizedLangevinPlan(0.01, GLE_TRANSITION, GLE_NOISE)
    assert plan.auxiliary_count == 1
    # Residual of an exact-in-reals identity evaluated in float64.
    assert float(plan.covariance_residual()) <= 1e-15
    dynamics, thermodynamic, state = _ideal_gas_runtime(plan, temperature=1.0)
    count = GAS_MASSES.shape[0]
    assert state.controllers.gle.memory.shape == (count, 3, 1)
    np.testing.assert_array_equal(state.controllers.gle.memory, 0.0)

    units = dynamics.system.plan.units
    system = dynamics.system

    def half_step(phase: int) -> Any:
        return apply_generalized_langevin(
            plan,
            state.random_key,
            system.plan.particle_ids,
            jnp.zeros((count, 3)),
            state.controllers.gle,
            system.plan.masses,
            system.mobile_mask,
            jnp.zeros((), dtype=jnp.int32),
            jnp.asarray(1.0),
            phase=phase,
            boltzmann_constant=units.boltzmann_constant,
            kinetic_to_energy=units.kinetic_to_energy,
        )

    first, again, second = half_step(0), half_step(0), half_step(1)
    assert bool(first.successful) and bool(second.successful)
    assert_tree_equal(first, again)
    # The two halves of one step draw at distinct operator addresses.
    assert not np.allclose(np.asarray(first.momenta), np.asarray(second.momenta))
    assert not np.allclose(
        np.asarray(first.memory.memory), np.asarray(second.memory.memory)
    )

    def run(current: Any) -> Any:
        for _ in range(3):
            evaluation = step_detailed(dynamics, current, thermodynamic)
            assert bool(evaluation.successful)
            current = evaluation.accepted_state
        return current

    finished = run(state)
    assert finished.controllers.gle.memory.shape == (count, 3, 1)
    assert np.any(np.asarray(finished.controllers.gle.memory) != 0.0)
    assert_tree_equal(finished, run(state))


# --- degrees of freedom --------------------------------------------------------


def test_degrees_of_freedom_and_center_of_mass_restriction() -> None:
    topology = MolecularTopologyPlan(constraints=[[0, 1]], constraint_distances=[1.0])
    masses = np.asarray([1.0, 2.0, 1.0, 1.0])
    system = atomistic_system(list(range(4)), masses, topology=topology)
    constraints = DistanceConstraintPlan().prepare(system)
    term = ExternalFieldTerm("ideal-gas", "zero", np.zeros((4, 3)))
    dynamics = prepare_dynamics(
        system,
        (term,),
        StochasticVelocityRescalingPlan(0.01, 0.1),
        constraints=constraints,
    )
    center_of_mass = measure(system, restriction="center-of-mass")
    thermodynamic = table(dynamics, row(center_of_mass, "nvt", temperature=1.0))
    assert thermodynamic.momentum_restriction == "center-of-mass"
    assert thermodynamic.degrees_of_freedom == (
        3 * 4 - constraints.independent_constraint_count - 3
    )
    assert thermodynamic.degrees_of_freedom == 8
    rng = np.random.default_rng(41)
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 1.0]]
    )
    state = dynamics.initialize_state(
        jnp.asarray(positions),
        thermodynamic,
        velocity=jnp.asarray(rng.normal(size=(4, 3))),
        key=jr.key(41),
    )
    boltzmann = dynamics.system.plan.units.boltzmann_constant
    for current in (state, step_detailed(dynamics, state, thermodynamic).accepted_state):
        # COM removal and equal-and-opposite RATTLE corrections: |P| at roundoff.
        np.testing.assert_allclose(
            np.sum(np.asarray(current.kinematics.momenta), axis=0), 0.0, atol=1e-13
        )
        kinetic = _kinetic(current.kinematics.momenta, masses)
        np.testing.assert_allclose(
            dynamics.diagnostics(current, thermodynamic).temperature,
            2.0 * kinetic / (8 * boltzmann),
            rtol=1e-12,
        )

    for method in (
        BAOABLangevinPlan(0.01, 1.0),
        GeneralizedLangevinPlan(0.01, GLE_TRANSITION, GLE_NOISE),
    ):
        stochastic = prepare_dynamics(system, (term,), method, constraints=constraints)
        with pytest.raises(ValueError, match="Per-particle stochastic thermostats"):
            table(stochastic, row(center_of_mass, "nvt", temperature=1.0))

    pinned = atomistic_system(
        list(range(4)), masses, mobile_mask=np.asarray([True, True, True, False])
    )
    with pytest.raises(ValueError, match="every active atom\\s+to be mobile"):
        measure(pinned, restriction="center-of-mass")


# --- warm rebase ---------------------------------------------------------------


def test_warm_rebase_rescales_momenta_and_books_external_work() -> None:
    chain = NoseHooverChainParameters(0.5)
    dynamics, thermodynamic, state = _harmonic_runtime(
        NoseHooverChainPlan(0.01, chain), (1.0, 2.0)
    )
    for _ in range(5):
        state = step_detailed(dynamics, state, thermodynamic).accepted_state
    assert np.any(np.asarray(state.controllers.particle_chain.velocities) != 0.0)
    rebased = dynamics.rebase_thermodynamic_state(state, thermodynamic, 1)
    assert bool(rebased.successful)
    after = rebased.accepted_state
    assert int(after.thermodynamic_state_index) == 1
    # p' = sqrt(T'/T) p with T'/T = 2 exactly: one correctly rounded product.
    np.testing.assert_allclose(
        after.kinematics.momenta,
        np.asarray(state.kinematics.momenta) * np.sqrt(2.0),
        rtol=1e-15,
    )
    assert_tree_equal(after.controllers.particle_chain, state.controllers.particle_chain)
    # Same Hamiltonian row: the force cache is retained.
    np.testing.assert_array_equal(after.force.forces, state.force.forces)
    jump = float(after.energy.extended_energy) - float(state.energy.extended_energy)
    assert abs(jump) > 1e-3
    np.testing.assert_allclose(
        float(after.energy.external_work) - float(state.energy.external_work),
        jump,
        rtol=1e-12,
        atol=1e-14,
    )
    thermal = dynamics.system.plan.units.boltzmann_constant * 2.0
    expected = (
        _kinetic(after.kinematics.momenta, HARMONIC_MASSES)
        + float(after.force.potential_energy)
        + _chain_energy(
            chain,
            after.controllers.particle_chain,
            thermodynamic.degrees_of_freedom,
            thermal,
        )
    )
    # Extended energy at the new row with the chain inertia rebound to T'.
    np.testing.assert_allclose(after.energy.extended_energy, expected, rtol=1e-12)

    stale = eqx.tree_at(
        lambda value: value.force.configuration_revision,
        state,
        state.force.configuration_revision - 1,
    )
    refused = dynamics.rebase_thermodynamic_state(stale, thermodynamic, 1)
    assert not bool(refused.successful)
    assert_tree_equal(refused.accepted_state, stale)


# --- addressed noise -----------------------------------------------------------


def test_thermostat_noise_depends_only_on_its_address() -> None:
    key = jr.key_data(jr.key(17)).astype(jnp.uint32)
    ids = jnp.asarray([10, 20, 30])
    permutation = np.asarray([2, 0, 1], dtype=np.int64)
    step = jnp.asarray(3, dtype=jnp.int32)

    def normals(particle_ids: Any, **changes: Any) -> np.ndarray:
        address = {
            "step_index": step,
            "operator_id": BAOAB_OPERATOR_ID,
            "realization_id": 0,
        }
        address.update(changes)
        return np.asarray(
            stable_particle_normals(
                address.pop("key", key),
                particle_ids,
                address.pop("step_index"),
                dtype=jnp.float64,
                **address,
            )
        )

    base = normals(ids)
    np.testing.assert_array_equal(normals(ids[permutation]), base[permutation])
    # A spectator particle does not shift any other particle's draw.
    np.testing.assert_array_equal(normals(jnp.asarray([10, 20, 30, 40]))[:3], base)
    for change in (
        {"step_index": jnp.asarray(4, dtype=jnp.int32)},
        {"operator_id": BAOAB_OPERATOR_ID + 1},
        {"realization_id": 1},
        {"key": jr.key_data(jr.key(18)).astype(jnp.uint32)},
    ):
        assert not np.allclose(normals(ids, **change), base)

    plan = BAOABLangevinPlan(0.1, 1.0)
    masses = jnp.asarray([1.0, 2.0, 3.0])
    momenta = jnp.asarray([[0.1, -0.2, 0.3], [0.0, 0.5, -0.1], [0.2, 0.2, 0.2]])
    mobile = jnp.ones((3,), dtype=jnp.bool_)

    def baoab(order: np.ndarray) -> np.ndarray:
        return np.asarray(
            apply_baoab_ornstein_uhlenbeck(
                plan,
                key,
                ids[order],
                momenta[order],
                masses[order],
                mobile[order],
                step,
                jnp.asarray(1.0),
                boltzmann_constant=1.0,
                kinetic_to_energy=1.0,
            ).momenta
        )

    identity = np.arange(3)
    np.testing.assert_array_equal(baoab(permutation), baoab(identity)[permutation])
    csvr = StochasticVelocityRescalingPlan(0.1, 0.5)
    scales = [
        float(
            stochastic_velocity_rescaling_scale(
                csvr,
                key,
                step,
                jnp.asarray(
                    _kinetic(np.asarray(momenta)[order], np.asarray(masses)[order])
                ),
                9,
                jnp.asarray(1.0),
                jnp.asarray(0.05),
                operator_id=CSVR_FIRST_OPERATOR_ID,
            ).scale
        )
        for order in (identity, permutation)
    ]
    # Global (scalar-addressed) draws; K differs only by summation order.
    assert scales[0] == pytest.approx(scales[1], rel=1e-14)

    # Whole-step identity across storage orders of the same stable IDs.
    ordered_ids = np.asarray([10, 20, 30])
    gas_masses = np.asarray([1.0, 2.0, 3.0])
    rng = np.random.default_rng(51)
    positions = rng.uniform(-1.0, 1.0, size=(3, 3))
    velocity = rng.normal(size=(3, 3))
    for method, exact in (
        (BAOABLangevinPlan(0.1, 1.0), True),
        (StochasticVelocityRescalingPlan(0.1, 0.5), False),
    ):
        results = []
        for order in (identity, permutation):
            dynamics, thermodynamic, state = _ideal_gas_runtime(
                method,
                temperature=1.0,
                ids=ordered_ids[order],
                masses=gas_masses[order],
                positions=positions[order],
                velocity=velocity[order],
                seed=52,
            )
            evaluation = dynamics.step_detailed(state, thermodynamic)
            assert bool(evaluation.successful)
            results.append(np.asarray(evaluation.accepted_state.kinematics.momenta))
        if exact:
            # Per-particle draws addressed by stable ID: elementwise identical.
            np.testing.assert_array_equal(results[1], results[0][permutation])
        else:
            # One global scale; only the K summation order differs.
            np.testing.assert_allclose(results[1], results[0][permutation], rtol=1e-13)
