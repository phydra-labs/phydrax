#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bundle-backed pressure, MTK cell dynamics and Monte Carlo cell ownership."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import product
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax import config as pxconfig
from phydrax.atomistic._barostat import (
    apply_isotropic_monte_carlo_barostat,
    IsotropicMonteCarloBarostatPlan,
    MONTE_CARLO_VOLUME_OPERATOR_ID,
)
from phydrax.atomistic._cell_dynamics import (
    AtomisticCellChart,
    MTKBarostatPlan,
    MTKNPHPlan,
    MTKNPTPlan,
)
from phydrax.atomistic._constraints import DistanceConstraintPlan
from phydrax.atomistic._dynamics import (
    AtomisticDynamicsPlan,
    AtomisticStepRejectionReason,
    CAPACITY_REJECTIONS,
    retry_atomistic_step_with_capacity,
    VelocityVerletPlan,
)
from phydrax.atomistic._rollout import AtomisticRolloutPlan, AtomisticTrajectoryPlan
from phydrax.atomistic._thermal import (
    BAOABLangevinPlan,
    NoseHooverChainParameters,
    NoseHooverChainPlan,
)
from phydrax.atomistic._topology import MolecularTopologyPlan
from tests._support.atomistic_bundles import role_capacity
from tests._support.atomistic_dynamics_fixtures import (
    assert_tree_equal,
    atomistic_system,
    DirectedPairTerm,
    ExternalFieldTerm,
    measure,
    observed_role_storage,
    periodic_cell,
    prepare_dynamics,
    prepare_potential,
    ROLE_CAPACITIES,
    row,
    step_detailed,
    table,
)


Reason = AtomisticStepRejectionReason

EDGE = 2.0
EXPONENTIAL_CUTOFF = 2.4
PAIR_POSITIONS = np.asarray([[0.1, 1.0, 1.0], [1.7, 1.1, 0.9]])
TRICLINIC = np.asarray([[3.0, 0.0, 0.0], [0.3, 3.1, 0.0], [0.2, -0.25, 3.2]])
ORTHORHOMBIC = np.diag([3.0, 3.1, 3.2])
# Capacity-only failures may carry the potential/nonfinite flags an overflow
# causes; any other flag is a scientific cause and is never repairable.
REPAIRABLE = CAPACITY_REJECTIONS | int(Reason.POTENTIAL | Reason.NONFINITE)


@pytest.fixture(autouse=True)
def _double_precision() -> Iterator[None]:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        yield


# --- fixtures ------------------------------------------------------------------


def _exponential_term() -> DirectedPairTerm:
    """``E = sum_directed exp(-|d|)`` inside 2.4 (the self-image diagnostic term)."""
    return DirectedPairTerm("edge-exponential", EXPONENTIAL_CUTOFF, profile="exponential")


def _smooth_term(cutoff: float = 2.0) -> DirectedPairTerm:
    """Soft C3 repulsion ``10 (1 - d/r_c)^4`` per unordered pair."""
    return DirectedPairTerm("smooth-repulsion", cutoff, profile="smooth", strength=10.0)


def _cubic_exponential_runtime(atoms: int) -> tuple[Any, Any]:
    system = atomistic_system(
        list(range(atoms)), [1.0] * atoms, cell=periodic_cell(EDGE * np.eye(3))
    )
    dynamics = prepare_dynamics(
        system, (_exponential_term(),), VelocityVerletPlan(1.0e-3)
    )
    return dynamics, table(dynamics, row(measure(system), "nve"))


def _lattice_state(
    dynamics: Any, thermodynamic: Any, vectors: np.ndarray, *, seed: int, speed: float
) -> Any:
    """Eight atoms on a jittered 2x2x2 lattice with seeded Gaussian velocities."""
    rng = np.random.default_rng(seed)
    corners = np.asarray(list(product((0.25, 0.75), repeat=3)))
    fractional = corners + rng.uniform(-0.04, 0.04, size=corners.shape)
    velocity = speed * rng.normal(size=corners.shape)
    return dynamics.initialize_state(
        jnp.asarray(fractional @ vectors),
        thermodynamic,
        velocity=jnp.asarray(velocity),
        key=jr.key(seed),
    )


def _mtk_npt_runtime(
    vectors: np.ndarray, chart: AtomisticCellChart
) -> tuple[Any, Any, Any]:
    system = atomistic_system(list(range(8)), [1.0] * 8, cell=periodic_cell(vectors))
    integrator = MTKNPTPlan(
        0.005,
        MTKBarostatPlan(0.5, chart=chart),
        NoseHooverChainParameters(0.5),
        NoseHooverChainParameters(0.5),
    )
    dynamics = prepare_dynamics(system, (_smooth_term(),), integrator)
    thermodynamic = table(
        dynamics, row(measure(system), "npt", temperature=0.25, pressure=0.2)
    )
    state = _lattice_state(dynamics, thermodynamic, vectors, seed=5, speed=0.5)
    return dynamics, thermodynamic, state


def _monte_carlo_runtime(
    *, role_capacities: tuple[Any, ...] = ROLE_CAPACITIES, seed: int = 0
) -> tuple[Any, Any, tuple[Any, ...], Any]:
    """BAOAB NPT pair in the cubic ``L = 2`` cell; the cell is owned by MC moves."""
    system = atomistic_system([0, 1], [1.0, 1.0], cell=periodic_cell(EDGE * np.eye(3)))
    dynamics = prepare_dynamics(
        system,
        (_exponential_term(),),
        BAOABLangevinPlan(0.01, 1.0),
        role_capacities=role_capacities,
    )
    # beta P dV ~ 32 u dominates: expansions are mostly rejected and
    # compressions accepted, so short move sequences see both decisions.
    states = (row(measure(system), "npt", temperature=1.0, pressure=4.0),)
    thermodynamic = table(dynamics, *states)
    positions = jnp.asarray(PAIR_POSITIONS)
    state = dynamics.initialize_state(
        positions,
        thermodynamic,
        velocity=jnp.zeros_like(positions),
        key=jr.key(seed),
    )
    return dynamics, thermodynamic, states, state


def _log_volume_draw(
    random_key: Any, counter: Any, plan: IsotropicMonteCarloBarostatPlan
) -> float:
    """The documented draw address: key, realization, operator, move counter, 0."""
    key = jr.wrap_key_data(jnp.asarray(random_key, dtype=jnp.uint32))
    key = jr.fold_in(key, jnp.asarray(plan.realization_id, dtype=jnp.uint32))
    key = jr.fold_in(key, jnp.uint32(MONTE_CARLO_VOLUME_OPERATOR_ID))
    key = jr.fold_in(key, jnp.asarray(counter, dtype=jnp.uint32))
    return float(
        jr.uniform(
            jr.fold_in(key, jnp.uint32(0)),
            (),
            minval=-plan.maximum_log_volume_change,
            maxval=plan.maximum_log_volume_change,
            dtype=jnp.float64,
        )
    )


def _finite_difference_pressure(
    dynamics: Any, state: Any, thermodynamic: Any, *, step: float = 1.0e-5
) -> float:
    """Central ``-dE/dV`` under isotropic scaling at fixed fractional coordinates.

    Integer image counts are held fixed and the bundle is refreshed (pure) at
    each scaled geometry before the energy-only evaluation.
    """
    cell = dynamics.system.cell
    vectors = state.cell_vectors
    images = state.kinematics.image_counts
    fractional = cell.fractional_with_vectors(state.kinematics.positions, vectors)
    controls = thermodynamic.state_at_replica(state.thermodynamic_state_index).controls
    volume = abs(float(np.linalg.det(np.asarray(vectors))))

    def energy(scale: float) -> float:
        scaled = vectors * scale
        positions = cell.cartesian_with_vectors(fractional, scaled)
        neighborhoods = dynamics.refresh_neighborhoods(
            state.neighborhoods, positions, cell_vectors=scaled, image_counts=images
        )
        evaluation = dynamics.energy_configuration(
            positions,
            neighborhoods,
            unwrapped_positions=positions + images.astype(positions.dtype) @ scaled,
            image_counts=images,
            species=state.species,
            cell_vectors=scaled,
            controls=controls,
        )
        assert bool(evaluation.successful)
        return float(evaluation.energy)

    plus, minus = energy(1.0 + step), energy(1.0 - step)
    return -(plus - minus) / (volume * ((1.0 + step) ** 3 - (1.0 - step) ** 3))


def _cutoff_clearance(positions: np.ndarray, vectors: np.ndarray, cutoff: float) -> float:
    """Smallest ``| |d| - cutoff |`` over every directed image route (|n_k| <= 3)."""
    clearance = np.inf
    for shift in product(range(-3, 4), repeat=3):
        translation = np.asarray(shift, dtype=np.float64) @ vectors
        for source, receiver in product(range(positions.shape[0]), repeat=2):
            if source == receiver and not any(shift):
                continue
            length = np.linalg.norm(positions[receiver] - positions[source] + translation)
            clearance = min(clearance, abs(length - cutoff))
    return float(clearance)


# --- pressure ------------------------------------------------------------------


def test_self_image_and_boundary_pair_pressure_is_minus_volume_derivative() -> None:
    dynamics, thermodynamic = _cubic_exponential_runtime(1)
    assert dynamics.strain_response
    positions = jnp.asarray([[0.3, 0.4, 0.5]])
    state = dynamics.initialize_state(
        positions, thermodynamic, velocity=jnp.zeros_like(positions), key=jr.key(0)
    )
    # Six face images at |d| = L = 2 lie inside 2.4; edge images (2.83) do not.
    np.testing.assert_allclose(
        state.force.potential_energy, 6.0 * np.exp(-EDGE), rtol=1e-13
    )
    # Self routes have displacement n L independent of x: the gradient cancels exactly
    # (atol one ulp-scale bound on exp(-2) contributions).
    np.testing.assert_allclose(state.force.forces, 0.0, atol=1e-15)
    diagnostics = dynamics.diagnostics(state, thermodynamic)
    analytic = 2.0 * np.exp(-EDGE) / EDGE**2
    assert analytic == pytest.approx(0.0676676416, rel=1e-9)
    # E(V) = 6 exp(-V^(1/3)) so -dE/dV = 2 exp(-L)/L^2; one float64 strain
    # derivative of six exponentials: rtol 1e-12 is ~1e4 ulps of headroom.
    np.testing.assert_allclose(diagnostics.pressure, analytic, rtol=1e-12)
    # Cubic symmetry: the configurational tensor is isotropic.
    np.testing.assert_allclose(
        diagnostics.pressure_tensor, analytic * np.eye(3), rtol=1e-12, atol=1e-15
    )
    # Central difference: O(h^2) truncation ~1e-10 relative, roundoff
    # eps*E/(V h) ~1e-11 relative, so rtol 1e-7 bounds both with margin.
    np.testing.assert_allclose(
        diagnostics.pressure,
        _finite_difference_pressure(dynamics, state, thermodynamic),
        rtol=1e-7,
    )

    dynamics, thermodynamic = _cubic_exponential_runtime(2)
    # The pair's interacting minimum image crosses the x boundary (|dx| = 0.4).
    # Precondition: every image route keeps >= 1e-3 from the cutoff, so the
    # 1e-5 strain probes of the difference never toggle a route.
    assert _cutoff_clearance(PAIR_POSITIONS, EDGE * np.eye(3), EXPONENTIAL_CUTOFF) > 1e-3
    positions = jnp.asarray(PAIR_POSITIONS)
    state = dynamics.initialize_state(
        positions, thermodynamic, velocity=jnp.zeros_like(positions), key=jr.key(0)
    )
    assert float(jnp.max(jnp.abs(state.force.forces))) > 1e-3
    diagnostics = dynamics.diagnostics(state, thermodynamic)
    reference = _finite_difference_pressure(dynamics, state, thermodynamic)
    assert abs(reference) > 1e-3
    # Same central-difference bound as above.
    np.testing.assert_allclose(diagnostics.pressure, reference, rtol=1e-7)


def test_ideal_gas_pressure_tensor_is_the_kinetic_momentum_flux() -> None:
    masses = np.asarray([1.0, 2.0, 0.5, 1.5])
    system = atomistic_system(list(range(4)), masses, cell=periodic_cell(TRICLINIC))
    dynamics = prepare_dynamics(
        system,
        (ExternalFieldTerm("ideal-gas", "zero", np.zeros((4, 3))),),
        VelocityVerletPlan(0.01),
    )
    assert dynamics.strain_response
    thermodynamic = table(dynamics, row(measure(system), "nve"))
    rng = np.random.default_rng(3)
    positions = rng.uniform(0.0, 1.0, size=(4, 3)) @ TRICLINIC
    momenta = rng.normal(size=(4, 3))
    state = dynamics.initialize_state(
        jnp.asarray(positions),
        thermodynamic,
        momentum=jnp.asarray(momenta),
        key=jr.key(1),
    )
    np.testing.assert_array_equal(np.asarray(state.force.strain_derivative), 0.0)
    kinetic = dynamics.system.plan.units.kinetic_to_energy
    volume = abs(np.linalg.det(TRICLINIC))
    expected = kinetic * np.einsum("ni,nj,n->ij", momenta, momenta, 1.0 / masses) / volume
    # A 4-term float64 sum and one division: rtol 1e-13.
    for current in (state, step_detailed(dynamics, state, thermodynamic).accepted_state):
        diagnostics = dynamics.diagnostics(current, thermodynamic)
        np.testing.assert_allclose(diagnostics.volume, volume, rtol=1e-14)
        np.testing.assert_allclose(
            diagnostics.pressure_tensor, expected, rtol=1e-13, atol=1e-15
        )
        np.testing.assert_allclose(
            diagnostics.pressure, np.trace(expected) / 3.0, rtol=1e-13
        )


# --- MTK -----------------------------------------------------------------------


def _nph_extended_errors(
    step_size: float, steps: int, stride: int
) -> tuple[np.ndarray, float, float]:
    """Samples of ``H(t) - H(0)`` every ``stride`` steps, ``H(0)`` and ``|dV|/V``."""
    vectors = 3.0 * np.eye(3)
    system = atomistic_system(list(range(8)), [1.0] * 8, cell=periodic_cell(vectors))
    integrator = MTKNPHPlan(step_size, MTKBarostatPlan(0.5, inertia_temperature=1.0))
    dynamics = prepare_dynamics(system, (_smooth_term(),), integrator)
    thermodynamic = table(dynamics, row(measure(system), "nph", pressure=0.2))
    state = _lattice_state(dynamics, thermodynamic, vectors, seed=11, speed=0.5)
    initial = float(state.energy.initial_extended_energy)
    volume = abs(float(np.linalg.det(np.asarray(state.cell_vectors))))
    errors = []
    for index in range(steps):
        evaluation = step_detailed(dynamics, state, thermodynamic)
        assert bool(evaluation.successful)
        state = evaluation.accepted_state
        if (index + 1) % stride == 0:
            errors.append(float(state.energy.extended_energy) - initial)
    final_volume = abs(float(np.linalg.det(np.asarray(state.cell_vectors))))
    return np.asarray(errors), initial, abs(final_volume - volume) / volume


def test_mtk_nph_extended_energy_error_is_second_order() -> None:
    coarse, initial, coarse_change = _nph_extended_errors(0.01, 40, 1)
    fine, fine_initial, _ = _nph_extended_errors(0.005, 80, 2)
    # Identical seeded initial conditions; only the step size differs.
    assert initial == pytest.approx(fine_initial, rel=1e-15)
    # The cell genuinely moves (barostat force nonzero), so K_cell + P V take part.
    assert coarse_change > 1e-6
    coarse_max = float(np.max(np.abs(coarse)))
    fine_max = float(np.max(np.abs(fine)))
    # Splitting error ~ (omega dt)^2 / 8 * K with omega ~ 2, dt = 0.01: ~1e-5
    # relative; 1e-3 relative is a 100x bound that still excludes O(dt) drift.
    assert coarse_max < 1e-3 * abs(initial)
    assert fine_max < coarse_max
    # Fixed-time error of a symmetric second-order splitting is C dt^2 (1 + O(dt^2));
    # halving dt at identical sample times gives 4 within [3, 5], which excludes
    # first order (2) and fourth order (16).
    assert 3.0 < coarse_max / fine_max < 5.0


def test_mtk_npt_charts_preserve_their_cell_subspace() -> None:
    dynamics, thermodynamic, state = _mtk_npt_runtime(
        TRICLINIC, AtomisticCellChart.ISOTROPIC
    )
    initial_vectors = np.asarray(state.cell_vectors)
    for _ in range(10):
        evaluation = step_detailed(dynamics, state, thermodynamic)
        assert bool(evaluation.successful)
        state = evaluation.accepted_state
    assert state.controllers.cell.momentum.shape == ()
    vectors = np.asarray(state.cell_vectors)
    scale = (np.linalg.det(vectors) / np.linalg.det(initial_vectors)) ** (1.0 / 3.0)
    assert abs(scale - 1.0) > 1e-9
    # dH/dt = H G^T with G = g I: H(t) = s H(0); exact flows leave ulp-level noise.
    np.testing.assert_allclose(vectors, scale * initial_vectors, rtol=1e-12, atol=1e-13)

    dynamics, thermodynamic, state = _mtk_npt_runtime(
        ORTHORHOMBIC, AtomisticCellChart.DIAGONAL
    )
    for _ in range(10):
        evaluation = step_detailed(dynamics, state, thermodynamic)
        assert bool(evaluation.successful)
        state = evaluation.accepted_state
    momentum = np.asarray(state.controllers.cell.momentum)
    off_diagonal = ~np.eye(3, dtype=bool)
    # The diagonal projection writes exact zeros off the diagonal.
    np.testing.assert_array_equal(momentum[off_diagonal], 0.0)
    assert np.any(np.abs(np.diag(momentum)) > 0.0)
    vectors = np.asarray(state.cell_vectors)
    # A diagonal generator keeps a diagonal lattice diagonal (ulp-level atol).
    np.testing.assert_allclose(vectors[off_diagonal], 0.0, atol=1e-14)
    assert not np.allclose(np.diag(vectors), np.diag(ORTHORHOMBIC), rtol=1e-10, atol=0.0)

    dynamics, thermodynamic, state = _mtk_npt_runtime(
        TRICLINIC, AtomisticCellChart.TRICLINIC_SYMMETRIC
    )
    for _ in range(10):
        evaluation = step_detailed(dynamics, state, thermodynamic)
        assert bool(evaluation.successful)
        state = evaluation.accepted_state
        momentum = np.asarray(state.controllers.cell.momentum)
        # Symmetrized forces and scalar chain scalings keep p_g exactly symmetric.
        np.testing.assert_array_equal(momentum, momentum.T)
    assert np.any(np.abs(momentum[off_diagonal]) > 0.0)


def test_one_cell_owner_refusals() -> None:
    dynamics, thermodynamic, state = _mtk_npt_runtime(
        TRICLINIC, AtomisticCellChart.ISOTROPIC
    )
    barostat = IsotropicMonteCarloBarostatPlan(0.1)
    with pytest.raises(ValueError, match="one cell-evolution owner"):
        apply_isotropic_monte_carlo_barostat(dynamics, state, thermodynamic, barostat)
    with pytest.raises(ValueError, match="does not already own the cell"):
        AtomisticRolloutPlan(
            dynamics,
            thermodynamic,
            AtomisticTrajectoryPlan(2),
            barostat=barostat,
            barostat_interval=1,
        )

    baoab, baoab_table, _, _ = _monte_carlo_runtime()
    with pytest.raises(ValueError, match="exactly one cell owner"):
        AtomisticRolloutPlan(baoab, baoab_table, AtomisticTrajectoryPlan(2))
    with pytest.raises(ValueError, match="cannot execute ensembles"):
        table(baoab, row(measure(baoab.system), "nph", pressure=1.0))

    with pytest.raises(ValueError, match="inertia_temperature"):
        MTKNPHPlan(0.01, MTKBarostatPlan(0.5))

    system = atomistic_system(
        [0, 1],
        [1.0, 1.0],
        cell=periodic_cell(6.0 * np.eye(3)),
        topology=MolecularTopologyPlan(constraints=[[0, 1]], constraint_distances=[1.0]),
    )
    potential = prepare_potential(system, (_smooth_term(),))
    constraints = DistanceConstraintPlan().prepare(system)
    for method in (
        MTKNPTPlan(
            0.005,
            MTKBarostatPlan(0.5),
            NoseHooverChainParameters(0.5),
            NoseHooverChainParameters(0.5),
        ),
        MTKNPHPlan(0.005, MTKBarostatPlan(0.5, inertia_temperature=1.0)),
    ):
        with pytest.raises(ValueError, match="Constrained MTK"):
            AtomisticDynamicsPlan(system, potential, method, constraints=constraints)


# --- Monte Carlo cell moves ----------------------------------------------------


def test_monte_carlo_barostat_counter_acceptance_and_failure_atomicity() -> None:
    dynamics, thermodynamic, _, state = _monte_carlo_runtime()
    assert thermodynamic.scaled_entity_count == 2
    values = thermodynamic.state_at_replica(0)
    beta, pressure = float(values.beta), float(values.pressure)
    # |ln V'/V| <= 0.1 keeps twelve moves well inside the 25% stencil margin.
    plan = IsotropicMonteCarloBarostatPlan(0.1)
    decisions = {True: 0, False: 0}
    for move in range(12):
        evaluation = apply_isotropic_monte_carlo_barostat(
            dynamics, state, thermodynamic, plan
        )
        assert bool(evaluation.successful)
        after = evaluation.accepted_state
        assert after.controllers.move_counter.dtype == jnp.uint32
        assert int(after.controllers.move_counter) == move + 1
        volume = abs(float(np.linalg.det(np.asarray(state.cell_vectors))))
        proposed = volume * np.exp(
            _log_volume_draw(state.random_key, state.controllers.move_counter, plan)
        )
        np.testing.assert_allclose(evaluation.volume_before, volume, rtol=1e-14)
        # det(H e^{u/3}) = V e^u up to three roundings: rtol 1e-12.
        np.testing.assert_allclose(evaluation.volume_after, proposed, rtol=1e-12)
        energy_after = dynamics.energy_state(evaluation.candidate_state, thermodynamic)
        assert bool(energy_after.successful)
        # ln alpha = -beta (dU + P dV) + (N + 1) ln(V'/V), N = 2 scaled atoms,
        # recomputed from an independent energy-only evaluation; rtol 1e-10
        # covers the energy-path vs force-path summation order.
        expected = -beta * (
            float(energy_after.energy)
            - float(state.force.potential_energy)
            + pressure * (proposed - volume)
        ) + 3.0 * np.log(proposed / volume)
        np.testing.assert_allclose(
            evaluation.log_acceptance_probability, expected, rtol=1e-10, atol=1e-12
        )
        accepted = bool(evaluation.accepted)
        decisions[accepted] += 1
        if accepted:
            assert (
                int(after.controllers.accepted_moves)
                == int(state.controllers.accepted_moves) + 1
            )
            assert (
                int(after.configuration_revision) == int(state.configuration_revision) + 1
            )
            np.testing.assert_array_equal(
                after.cell_vectors, evaluation.candidate_state.cell_vectors
            )
        else:
            # A rejection commits the counter and nothing else.
            assert_tree_equal(
                eqx.tree_at(
                    lambda value: value.controllers.move_counter,
                    after,
                    state.controllers.move_counter,
                ),
                state,
            )
        state = after
    assert decisions[True] > 0 and decisions[False] > 0

    stale = eqx.tree_at(
        lambda value: value.force.configuration_revision,
        state,
        state.force.configuration_revision - 1,
    )
    evaluation = apply_isotropic_monte_carlo_barostat(
        dynamics, stale, thermodynamic, plan
    )
    assert not bool(evaluation.successful)
    assert int(evaluation.rejection_reasons) & int(Reason.STALE_FORCE)
    assert_tree_equal(evaluation.accepted_state, stale)

    _, _, _, probe = _monte_carlo_runtime()
    exact = observed_role_storage(probe)
    shrinking = IsotropicMonteCarloBarostatPlan(0.6)
    # Deterministic host search for a run key whose first move shrinks the cell
    # by e^{-0.15} or more linearly: more image routes than the probed storage.
    seed = next(
        value
        for value in range(256)
        if _log_volume_draw(jr.key_data(jr.key(value)), 0, shrinking) < -0.45
    )
    dynamics, thermodynamic, states, state = _monte_carlo_runtime(
        role_capacities=(exact, role_capacity(4096, 512)), seed=seed
    )
    evaluation = apply_isotropic_monte_carlo_barostat(
        dynamics, state, thermodynamic, shrinking
    )
    reasons = int(evaluation.rejection_reasons)
    assert not bool(evaluation.successful)
    assert reasons & CAPACITY_REJECTIONS
    assert not reasons & ~REPAIRABLE
    assert_tree_equal(evaluation.accepted_state, state)
    # The overflow is attributed to the resources: the strain of the failed
    # evaluation is a consequence, not a separate scientific cause.
    assert dynamics.strain_response
    assert not reasons & int(Reason.STRAIN_UNAVAILABLE)
    evidence = evaluation.candidate_state.neighborhoods.views[0].evidence
    assert bool(jnp.any(evidence.route_overflow | evidence.degree_overflow))

    # Host growth to the declared next level from the candidate's resource
    # evidence keeps the accepted state, key and move counter; the replayed move
    # draws the same volume and matches a run declared at that storage.
    grown_bundle = dynamics.bundle.grow_capacities(
        evaluation.candidate_state.neighborhoods
    )
    grown, grown_table, rebound = dynamics.rebind_hamiltonian(
        dynamics.potential.with_neighborhood_bundle(grown_bundle),
        state,
        thermodynamic,
        states,
    )
    assert grown.bundle.role_levels == (1,)
    primed = grown.prime_state(rebound, grown_table)
    assert bool(primed.successful)
    np.testing.assert_array_equal(primed.accepted_state.random_key, state.random_key)
    np.testing.assert_array_equal(
        primed.accepted_state.controllers.move_counter, state.controllers.move_counter
    )
    replayed = apply_isotropic_monte_carlo_barostat(
        grown, primed.accepted_state, grown_table, shrinking
    )
    assert bool(replayed.successful)
    np.testing.assert_array_equal(replayed.volume_after, evaluation.volume_after)
    reference_dynamics, reference_table, _, reference_state = _monte_carlo_runtime(
        seed=seed
    )
    reference = apply_isotropic_monte_carlo_barostat(
        reference_dynamics, reference_state, reference_table, shrinking
    )
    assert bool(reference.successful)
    assert bool(replayed.accepted) == bool(reference.accepted)
    np.testing.assert_array_equal(replayed.volume_after, reference.volume_after)
    # Equal storage; only padded-buffer summation order may differ: atol 1e-12
    # on the O(10) log acceptance.
    np.testing.assert_allclose(
        replayed.log_acceptance_probability,
        reference.log_acceptance_probability,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        replayed.accepted_state.controllers.move_counter,
        reference.accepted_state.controllers.move_counter,
    )
    np.testing.assert_array_equal(
        replayed.accepted_state.cell_vectors, reference.accepted_state.cell_vectors
    )


def test_capacity_only_replay_reproduces_the_reference_step() -> None:
    vectors = 6.0 * np.eye(3)
    positions = jnp.asarray([[2.0, 3.0, 3.0], [4.2, 3.0, 3.0]])
    velocity = jnp.asarray([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])

    def runtime(
        role_capacities: tuple[Any, ...],
    ) -> tuple[Any, Any, tuple[Any, ...], Any]:
        system = atomistic_system([0, 1], [1.0, 1.0], cell=periodic_cell(vectors))
        dynamics = prepare_dynamics(
            system,
            (_smooth_term(1.5),),
            BAOABLangevinPlan(1.0, 0.1),
            role_capacities=role_capacities,
        )
        states = (row(measure(system), "nvt", temperature=0.01),)
        thermodynamic = table(dynamics, *states)
        state = dynamics.initialize_state(
            positions, thermodynamic, velocity=velocity, key=jr.key(9)
        )
        return dynamics, thermodynamic, states, state

    reference_dynamics, reference_table, _, reference_state = runtime(ROLE_CAPACITIES)
    # The pair starts 2.2 apart, outside the 1.5 + 0.3 search radius, and closes
    # to ~1.2 in one step: the rebuilt epoch has two directed routes.
    exact = observed_role_storage(reference_state)
    reference = reference_dynamics.step_detailed(reference_state, reference_table)
    assert bool(reference.successful)

    dynamics, thermodynamic, states, state = runtime((exact, role_capacity(4096, 512)))
    first = dynamics.step_detailed(state, thermodynamic)
    reasons = int(first.rejection_reasons)
    assert not bool(first.successful)
    assert reasons & CAPACITY_REJECTIONS
    assert not reasons & ~REPAIRABLE
    # The failed attempt consumed nothing: same key, step and every other leaf.
    assert_tree_equal(first.accepted_state, state)
    # The candidate keeps its exact overflow evidence; the potential failure and
    # missing strain it causes are downstream, never a scientific strain cause.
    evidence = first.candidate_state.neighborhoods.views[0].evidence
    assert bool(jnp.any(evidence.route_overflow | evidence.degree_overflow))
    assert dynamics.strain_response
    assert not reasons & int(Reason.STRAIN_UNAVAILABLE)
    # A scientific cause alongside the overflow is never replayed: a stale force
    # cache refuses at the declared storage without growth or state change.
    stale = eqx.tree_at(
        lambda value: value.force.configuration_revision,
        state,
        state.force.configuration_revision - 1,
    )
    same, same_table, refused = retry_atomistic_step_with_capacity(
        dynamics, stale, thermodynamic, states
    )
    assert same is dynamics and same_table is thermodynamic
    assert not bool(refused.successful)
    refused_reasons = int(refused.rejection_reasons)
    assert refused_reasons & CAPACITY_REJECTIONS
    assert refused_reasons & int(Reason.STALE_FORCE)
    assert_tree_equal(refused.accepted_state, stale)

    grown, grown_table, replayed = retry_atomistic_step_with_capacity(
        dynamics, state, thermodynamic, states
    )
    assert bool(replayed.successful)
    assert grown.bundle.role_levels == (1,)
    assert grown_table.table_id != thermodynamic.table_id
    accepted = replayed.accepted_state
    assert int(accepted.step_index) == int(state.step_index) + 1
    np.testing.assert_array_equal(accepted.random_key, state.random_key)
    np.testing.assert_array_equal(accepted.time, reference.accepted_state.time)
    # Same physical and stochastic attempt at equal storage; only padded-buffer
    # summation order may differ: atol 1e-12 on O(1) coordinates and momenta.
    np.testing.assert_allclose(
        accepted.kinematics.positions,
        reference.accepted_state.kinematics.positions,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        accepted.kinematics.momenta,
        reference.accepted_state.kinematics.momenta,
        rtol=0.0,
        atol=1e-12,
    )


def test_final_thermostat_after_an_overflowed_boundary_is_replayed() -> None:
    """``T(dt/2) K D K T(dt/2)``: the closing chain half-step consumes the
    overflowed boundary's invalid forces and is that boundary's consequence;
    an opening chain failure stays an independent, never-replayed cause."""
    vectors = 6.0 * np.eye(3)
    positions = jnp.asarray([[2.0, 3.0, 3.0], [4.2, 3.0, 3.0]])
    velocity = jnp.asarray([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])

    def runtime(
        role_capacities: tuple[Any, ...],
    ) -> tuple[Any, Any, tuple[Any, ...], Any]:
        system = atomistic_system([0, 1], [1.0, 1.0], cell=periodic_cell(vectors))
        # A long chain time constant (Q_1 = N_f kT tau^2) barely scales the
        # momenta, so the pair closes into the search radius in one step.
        dynamics = prepare_dynamics(
            system,
            (_smooth_term(1.5),),
            NoseHooverChainPlan(1.0, NoseHooverChainParameters(10.0)),
            role_capacities=role_capacities,
        )
        states = (row(measure(system), "nvt", temperature=0.5 / 6.0),)
        thermodynamic = table(dynamics, *states)
        state = dynamics.initialize_state(
            positions, thermodynamic, velocity=velocity, key=jr.key(9)
        )
        return dynamics, thermodynamic, states, state

    reference_dynamics, reference_table, _, reference_state = runtime(ROLE_CAPACITIES)
    exact = observed_role_storage(reference_state)
    reference = reference_dynamics.step_detailed(reference_state, reference_table)
    assert bool(reference.successful)

    dynamics, thermodynamic, states, state = runtime((exact, role_capacity(4096, 512)))
    first = dynamics.step_detailed(state, thermodynamic)
    reasons = int(first.rejection_reasons)
    assert not bool(first.successful)
    assert reasons & CAPACITY_REJECTIONS
    assert not reasons & ~REPAIRABLE
    assert not reasons & int(Reason.THERMOSTAT)
    assert_tree_equal(first.accepted_state, state)

    # An opening half-step failure (non-finite chain position: finite scale,
    # finite momenta) precedes every force boundary: it is reported alongside
    # the overflow and refuses without growth or state change.
    broken = eqx.tree_at(
        lambda value: value.controllers.particle_chain.positions,
        state,
        state.controllers.particle_chain.positions.at[0].set(jnp.nan),
    )
    same, same_table, refused = retry_atomistic_step_with_capacity(
        dynamics, broken, thermodynamic, states
    )
    assert same is dynamics and same_table is thermodynamic
    assert not bool(refused.successful)
    refused_reasons = int(refused.rejection_reasons)
    assert refused_reasons & CAPACITY_REJECTIONS
    assert refused_reasons & int(Reason.THERMOSTAT)
    assert_tree_equal(refused.accepted_state, broken)

    grown, _, replayed = retry_atomistic_step_with_capacity(
        dynamics, state, thermodynamic, states
    )
    assert bool(replayed.successful)
    assert grown.bundle.role_levels == (1,)
    accepted = replayed.accepted_state
    assert int(accepted.step_index) == int(state.step_index) + 1
    np.testing.assert_array_equal(accepted.random_key, state.random_key)
    # Same attempt at equal storage; only padded-buffer summation order may
    # differ: atol 1e-12 on O(1) coordinates, momenta and chain variables.
    expected = reference.accepted_state
    for read in (
        lambda value: value.kinematics.positions,
        lambda value: value.kinematics.momenta,
        lambda value: value.controllers.particle_chain.positions,
        lambda value: value.controllers.particle_chain.velocities,
    ):
        np.testing.assert_allclose(read(accepted), read(expected), rtol=0.0, atol=1e-12)


# --- priming and replay --------------------------------------------------------


def test_prime_state_books_an_energy_jump_without_consuming_events() -> None:
    dynamics, thermodynamic, state = _mtk_npt_runtime(
        TRICLINIC, AtomisticCellChart.ISOTROPIC
    )
    for _ in range(2):
        state = step_detailed(dynamics, state, thermodynamic).accepted_state
    displaced = eqx.tree_at(
        lambda value: value.kinematics.positions,
        state,
        state.kinematics.positions.at[0].add(jnp.asarray([0.02, -0.03, 0.025])),
    )
    primed = dynamics.prime_state(displaced, thermodynamic)
    assert bool(primed.successful)
    after = primed.accepted_state
    for read in (
        lambda value: value.time,
        lambda value: value.step_index,
        lambda value: value.random_key,
        lambda value: value.configuration_revision,
        lambda value: value.energy.accepted_steps,
    ):
        np.testing.assert_array_equal(read(after), read(displaced))
    assert_tree_equal(after.controllers, displaced.controllers)
    jump = float(after.energy.extended_energy) - float(displaced.energy.extended_energy)
    assert abs(jump) > 1e-6
    # The whole jump is external work; one subtraction of O(10) energies.
    np.testing.assert_allclose(
        float(after.energy.external_work) - float(displaced.energy.external_work),
        jump,
        rtol=1e-12,
        atol=1e-14,
    )
    controls = thermodynamic.state_at_replica(after.thermodynamic_state_index).controls
    assert bool(dynamics.force_cache_valid(after, controls))
    reference = dynamics.evaluate_state(after, thermodynamic)
    np.testing.assert_allclose(
        after.force.forces, reference.forces, rtol=1e-13, atol=1e-14
    )
    np.testing.assert_allclose(after.force.potential_energy, reference.energy, rtol=1e-13)


def test_checkpoint_free_replay_is_bitwise_deterministic() -> None:
    dynamics, thermodynamic, _, state = _monte_carlo_runtime(seed=4)
    plan = IsotropicMonteCarloBarostatPlan(0.2)

    def advance(current: Any) -> Any:
        for _ in range(2):
            current = step_detailed(dynamics, current, thermodynamic).accepted_state
            current = apply_isotropic_monte_carlo_barostat(
                dynamics, current, thermodynamic, plan
            ).accepted_state
        return current

    leaves, structure = jax.tree.flatten(state)
    # Host round trip of every array leaf: a restart from plain saved values.
    restored = jax.tree.unflatten(
        structure,
        [
            jnp.asarray(np.array(leaf, copy=True))
            if isinstance(leaf, jax.Array)
            else leaf
            for leaf in leaves
        ],
    )
    first = advance(state)
    second = advance(restored)
    assert int(first.step_index) == 2
    assert int(first.controllers.move_counter) == 2
    assert_tree_equal(first, second)
