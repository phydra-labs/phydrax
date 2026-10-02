# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Physical Euler action, bounded transactions, and resource-only replay."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest
from jax.typing import ArrayLike

from phydrax._sampling._addressing import derive_key, SampleAddress
from phydrax.operators.quantum.lattice._address import (
    QuantumAddressCodec,
    QuantumConfigurationDomain,
)
from phydrax.operators.quantum.lattice._column import QuantumLatticeColumnOperator
from phydrax.operators.quantum.lattice._column_compile import (
    prepare_quantum_lattice_columns,
    QuantumColumnResourcePolicy,
)
from phydrax.operators.quantum.lattice._model import (
    LocalOperatorPlan,
    LocalSpacePlan,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
)
from phydrax.solver._projector_monte_carlo import (
    initialize_projector_monte_carlo,
    prepare_projector_monte_carlo,
    solve_projector_monte_carlo,
    step_projector_monte_carlo,
    validate_projector_state,
)
from phydrax.solver._projector_monte_carlo_contracts import (
    CompressionPolicy,
    ControllerPolicy,
    PreparedProjectorMonteCarlo,
    ProjectorMonteCarloPlan,
    ProjectorMonteCarloProblem,
    ProjectorMonteCarloState,
    ProjectorMonteCarloStatus,
    SpawnPolicy,
)
from phydrax.units import derived_unit, HARTREE, METER
from tests._support.projector_monte_carlo import (
    complex_flux_control,
    control_keys,
    control_operator,
    fermionic_exterior_control,
    ProjectorFiniteControl,
    two_boson_two_site_control,
)


def _plan(
    *,
    support: int = 3,
    groups: int = 8,
    events: int = 2,
    attempts: int = 8,
    history: int = 8,
    replicas: int = 2,
    spawn: SpawnPolicy = "exact",
    compression: CompressionPolicy = "none",
    controller: ControllerPolicy = "fixed-shift",
    boost: float = 1.0,
    relative: float = 1.0,
    absolute: float = float("inf"),
    theta: float = 1.0,
    shift: float = 0.0,
) -> ProjectorMonteCarloPlan:
    return ProjectorMonteCarloPlan(
        replicas=replicas,
        support_capacity=support,
        group_capacity=groups,
        event_capacity=events,
        attempt_capacity=attempts,
        source_capacity=1,
        history_capacity=history,
        maximum_retained_bytes=10_000_000,
        maximum_workspace_bytes=10_000_000,
        spawn_policy=spawn,
        compression=compression,
        controller=controller,
        boost=boost,
        relative_threshold=relative,
        absolute_threshold=absolute,
        theta=theta,
        initial_shift=shift,
    )


def _prepared(
    control: ProjectorFiniteControl,
    coefficients: npt.NDArray[np.complex128],
    plan: ProjectorMonteCarloPlan,
    *,
    dt: float = 0.05,
) -> PreparedProjectorMonteCarlo:
    operator = control_operator(control)
    problem = ProjectorMonteCarloProblem(
        operator,
        control_keys(operator, control),
        coefficients,
        dt=dt,
        energy_unit=HARTREE,
        inverse_energy_unit=derived_unit("inverse-hartree", ((HARTREE, -1),)),
        provenance_id=control.provenance,
    )
    return prepare_projector_monte_carlo(problem, plan)


def _physical_vector(
    prepared: PreparedProjectorMonteCarlo,
    state: ProjectorMonteCarloState,
    control: ProjectorFiniteControl,
    replica: int = 0,
) -> npt.NDArray[np.complex128]:
    addresses = np.asarray(control_keys(prepared.original_operator, control))
    keys, values, active = (
        np.asarray(state.support_keys[replica]),
        np.asarray(state.coefficients[replica]),
        np.asarray(state.active[replica]),
    )
    result = np.zeros((addresses.shape[0],), dtype=np.complex128)
    for key, value in zip(keys[active], values[active], strict=True):
        matches = np.all(addresses == key, axis=1)
        if np.count_nonzero(matches) != 1:
            raise ValueError(
                "Result has an address outside the independent physical control."
            )
        result[matches] = value
    return result


@pytest.mark.parametrize(
    "factory",
    (two_boson_two_site_control, fermionic_exterior_control, complex_flux_control),
    ids=("boson", "fermion-sign", "complex-flux"),
)
def test_exact_step_matches_independent_physical_euler_action(
    factory: Callable[[], ProjectorFiniteControl],
) -> None:
    control = factory()
    coefficients = np.asarray((0.4 + 0.2j, -0.3j, 0.7 - 0.1j), dtype=np.complex128)
    prepared = _prepared(control, coefficients, _plan(shift=0.2))
    state = initialize_projector_monte_carlo(prepared, jax.random.key(42))
    result = step_projector_monte_carlo(prepared, state)
    expected = coefficients + 0.05 * (0.2 * coefficients - control.matrix @ coefficients)
    assert int(result.status) == ProjectorMonteCarloStatus.SUCCESS
    np.testing.assert_allclose(
        _physical_vector(prepared, result.state, control),
        expected,
        rtol=2e-15,
        atol=2e-15,
    )
    np.testing.assert_array_equal(
        state.coefficients,
        np.broadcast_to(
            np.asarray(prepared.initial_coefficients), state.coefficients.shape
        ),
    )
    assert int(state.step) == 0
    assert int(result.state.step) == int(result.state.history.count) == 1
    np.testing.assert_array_equal(
        result.state.history.applied_shifts[:, 0], np.asarray((0.2, 0.2))
    )


def test_initial_duplicate_support_coalesces_without_normalization() -> None:
    control = two_boson_two_site_control()
    operator = control_operator(control)
    keys = control_keys(operator, control)
    problem = ProjectorMonteCarloProblem(
        operator,
        jnp.stack((keys[1], keys[0], keys[1])),
        np.asarray((2 + 1j, 3, -1j), dtype=np.complex128),
        dt=0.05,
        energy_unit=HARTREE,
        inverse_energy_unit=derived_unit("inverse-hartree", ((HARTREE, -1),)),
        provenance_id=control.provenance,
    )
    prepared = prepare_projector_monte_carlo(problem, _plan())
    state = initialize_projector_monte_carlo(prepared, jax.random.key(1))
    np.testing.assert_array_equal(
        _physical_vector(prepared, state, control),
        np.asarray((3, 2, 0), dtype=np.complex128),
    )
    np.testing.assert_array_equal(
        state.populations, np.asarray((5.0, 5.0), dtype=np.float64)
    )


@pytest.mark.parametrize("seed", range(8), ids=lambda seed: f"root-{seed}")
def test_sampled_step_is_one_genuine_inverse_probability_outcome(seed: int) -> None:
    control = two_boson_two_site_control()
    prepared = _prepared(
        control,
        np.asarray((0, 1, 0), dtype=np.complex128),
        _plan(spawn="sampled", boost=0.5),
    )
    state = initialize_projector_monte_carlo(prepared, jax.random.key(seed))
    result = step_projector_monte_carlo(prepared, state)
    assert int(result.status) == ProjectorMonteCarloStatus.SUCCESS
    vector = _physical_vector(prepared, result.state, control)
    expected_outcomes = np.asarray(
        ((0, 1, 0), (0.2 * np.sqrt(2), 1, 0), (0, 1, 0.2 * np.sqrt(2))),
        dtype=np.complex128,
    )
    assert np.any(
        np.all(
            np.isclose(expected_outcomes, vector[None, :], rtol=1e-15, atol=1e-15), axis=1
        )
    )
    np.testing.assert_array_equal(
        result.evidence.maximum_attempts, np.ones((2,), dtype=np.int64)
    )


def test_exhaustive_finite_raw_proposal_expectation_matches_euler() -> None:
    control = two_boson_two_site_control()
    prepared = _prepared(
        control,
        np.asarray((0, 1, 0), dtype=np.complex128),
        _plan(replicas=1, spawn="sampled", boost=0.5),
    )
    operator = prepared.original_operator
    source = operator.domain.address(control.coordinates[1])
    role = SampleAddress(
        "projector-monte-carlo",
        "physical-step",
        target=(operator.domain.domain_id, operator.operator_id),
        role="spawn",
    )
    outcomes: dict[tuple[int, ...] | None, npt.NDArray[np.complex128]] = {}
    # Fixed keys locate each finite proposal outcome; probabilities are analytic,
    # not empirical frequencies or a stochastic energy-tolerance oracle.
    for seed in range(128):
        root = jax.random.key(seed)
        proposal_key = derive_key(
            root,
            role,
            0,
            0,
            0,
            *(
                tuple(
                    source.key_words[index] for index in range(source.key_words.shape[0])
                )
            ),
            0,
            0,
        )
        route = operator.sample_raw_excitation(proposal_key, source)
        outcome = (
            tuple(int(word) for word in np.asarray(route.target_key))
            if bool(route.valid & route.off_diagonal)
            else None
        )
        if outcome in outcomes:
            continue
        result = step_projector_monte_carlo(
            prepared, initialize_projector_monte_carlo(prepared, root)
        )
        assert int(result.status) == ProjectorMonteCarloStatus.SUCCESS
        outcomes[outcome] = _physical_vector(prepared, result.state, control)
        if len(outcomes) == 3:
            break
    if len(outcomes) != 3 or None not in outcomes:
        raise ValueError(
            "Fixed root fixture must cover both hopping routes and the null route."
        )
    expectation = 0.5 * outcomes[None]
    for outcome, vector in outcomes.items():
        if outcome is not None:
            expectation += 0.25 * vector
    expected = np.asarray((0, 1, 0), dtype=np.complex128) - 0.05 * control.matrix[:, 1]
    np.testing.assert_allclose(expectation, expected, rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize(
    "relative,absolute,expected_exact",
    ((0.25, float("inf"), True), (0.2500001, 1.0, False), (0.2500001, 0.999999, True)),
    ids=("relative-inclusive", "absolute-exclusive", "absolute-above"),
)
def test_semistochastic_boundaries_use_physical_routes(
    relative: float, absolute: float, expected_exact: bool
) -> None:
    control = two_boson_two_site_control()
    prepared = _prepared(
        control,
        np.asarray((0, 1, 0), dtype=np.complex128),
        _plan(spawn="semistochastic", relative=relative, absolute=absolute),
    )
    result = step_projector_monte_carlo(
        prepared, initialize_projector_monte_carlo(prepared, jax.random.key(2))
    )
    assert prepared.original_operator.raw_route_bound == 4
    np.testing.assert_array_equal(
        result.evidence.exact_sources, np.full((2,), int(expected_exact), dtype=np.int64)
    )
    np.testing.assert_array_equal(
        result.evidence.sampled_sources,
        np.full((2,), int(not expected_exact), dtype=np.int64),
    )


@pytest.mark.parametrize(
    "events", (1, 3, 7), ids=("single-event", "partial-chunks", "wide-chunk")
)
def test_chunk_and_storage_changes_preserve_sampled_randomness(events: int) -> None:
    control = complex_flux_control()
    coefficients = np.asarray((0.6 + 0.1j, 0.2j, -0.7), dtype=np.complex128)
    first = _prepared(
        control,
        coefficients,
        _plan(spawn="sampled", compression="threshold", theta=0.3, events=2),
    )
    second = _prepared(
        control,
        coefficients,
        _plan(
            spawn="sampled",
            compression="threshold",
            theta=0.3,
            events=events,
            groups=12,
            support=5,
            attempts=12,
        ),
    )
    assert first.scientific_id == second.scientific_id
    a = step_projector_monte_carlo(
        first, initialize_projector_monte_carlo(first, jax.random.key(91))
    )
    b = step_projector_monte_carlo(
        second, initialize_projector_monte_carlo(second, jax.random.key(91))
    )
    assert int(a.status) == int(b.status) == ProjectorMonteCarloStatus.SUCCESS
    np.testing.assert_array_equal(
        _physical_vector(first, a.state, control),
        _physical_vector(second, b.state, control),
    )
    np.testing.assert_array_equal(a.state.shifts, b.state.shifts)


def _cancelling_operator() -> QuantumLatticeColumnOperator:
    space = LocalSpacePlan("site", ("zero", "one"), (), np.zeros((2, 0), dtype=np.int32))
    local = LocalOperatorPlan(
        space, "flip", np.asarray(((0, 1), (1, 0)), dtype=np.complex128), ()
    )
    specification = QuantumLatticeSpecification(
        (space,),
        (
            QuantumLatticeTerm((local,), coefficient=1, label="positive"),
            QuantumLatticeTerm((local,), coefficient=-1, label="negative"),
        ),
    )
    domain = QuantumConfigurationDomain(specification, species_ids=("particle",))
    return QuantumLatticeColumnOperator(
        prepare_quantum_lattice_columns(
            specification, QuantumColumnResourcePolicy(maximum_column_targets=2)
        ),
        domain,
    )


def test_all_chunks_annihilate_before_one_final_compression() -> None:
    operator = _cancelling_operator()
    key = operator.domain.address(np.asarray((0,), dtype=np.int32)).key_words
    problem = ProjectorMonteCarloProblem(
        operator,
        key[None, :],
        np.asarray((1,), dtype=np.complex128),
        dt=0.1,
        energy_unit=HARTREE,
        inverse_energy_unit=derived_unit("inverse-hartree", ((HARTREE, -1),)),
        provenance_id="signed-duplicate-path-control",
    )
    prepared = prepare_projector_monte_carlo(
        problem, _plan(support=1, groups=2, events=1, compression="threshold", theta=1.0)
    )
    state = initialize_projector_monte_carlo(prepared, jax.random.key(3))
    result = step_projector_monte_carlo(prepared, state)
    assert int(result.status) == ProjectorMonteCarloStatus.SUCCESS
    np.testing.assert_array_equal(result.state.coefficients, state.coefficients)
    np.testing.assert_array_equal(result.state.support_keys, state.support_keys)
    np.testing.assert_allclose(
        result.evidence.pre_annihilation_norm, np.asarray((1.2, 1.2)), rtol=1e-15
    )
    np.testing.assert_array_equal(
        result.evidence.post_annihilation_norm, np.asarray((1.0, 1.0))
    )


@pytest.mark.parametrize(
    "groups,support,status",
    (
        (1, 1, ProjectorMonteCarloStatus.INTERMEDIATE_GROUP_OVERFLOW),
        (3, 1, ProjectorMonteCarloStatus.FINAL_SUPPORT_OVERFLOW),
    ),
    ids=("intermediate-distinct-groups", "final-retained-support"),
)
def test_capacity_failure_rolls_back_whole_batch(
    groups: int, support: int, status: ProjectorMonteCarloStatus
) -> None:
    control = two_boson_two_site_control()
    prepared = _prepared(
        control,
        np.asarray((0, 1, 0), dtype=np.complex128),
        _plan(support=support, groups=groups, events=1),
    )
    state = initialize_projector_monte_carlo(prepared, jax.random.key(4))
    result = step_projector_monte_carlo(prepared, state)
    assert int(result.status) == status
    assert not bool(result.accepted)
    for old, new in zip(
        jax.tree.leaves(state), jax.tree.leaves(result.state), strict=True
    ):
        np.testing.assert_array_equal(
            jax.random.key_data(old)
            if jax.dtypes.issubdtype(old.dtype, jax.dtypes.prng_key)
            else old,
            jax.random.key_data(new)
            if jax.dtypes.issubdtype(new.dtype, jax.dtypes.prng_key)
            else new,
        )


def test_attempt_limit_of_one_replica_rolls_back_other_replica() -> None:
    control = two_boson_two_site_control()
    prepared = _prepared(
        control,
        np.asarray((0, 1, 0), dtype=np.complex128),
        _plan(spawn="sampled", attempts=2),
    )
    state = initialize_projector_monte_carlo(prepared, jax.random.key(5))
    coefficients = state.coefficients.at[1].multiply(jnp.complex128(3))
    state = eqx.tree_at(
        lambda item: (item.coefficients, item.populations),
        state,
        (coefficients, jnp.sum(jnp.abs(coefficients), axis=1, dtype=jnp.float64)),
    )
    result = step_projector_monte_carlo(prepared, state)
    assert int(result.status) == ProjectorMonteCarloStatus.ATTEMPT_LIMIT
    np.testing.assert_array_equal(
        result.evidence.replica_status,
        np.asarray((0, ProjectorMonteCarloStatus.ATTEMPT_LIMIT), dtype=np.int32),
    )
    np.testing.assert_array_equal(result.state.coefficients, state.coefficients)
    assert int(result.state.step) == int(result.state.history.count) == 0


def test_extinction_refuses_before_double_log_feedback() -> None:
    control = two_boson_two_site_control(hopping=0.0, interaction=2.0)
    prepared = _prepared(
        control,
        np.asarray((1, 0, 0), dtype=np.complex128),
        _plan(controller="double-log"),
        dt=0.5,
    )
    state = initialize_projector_monte_carlo(prepared, jax.random.key(6))
    result = step_projector_monte_carlo(prepared, state)
    assert int(result.status) == ProjectorMonteCarloStatus.EXTINCTION
    np.testing.assert_array_equal(result.state.shifts, state.shifts)
    assert np.all(np.isfinite(result.state.shifts))
    assert int(result.state.step) == 0


def test_continuation_preserves_every_raw_step_and_applied_shift() -> None:
    control = complex_flux_control()
    prepared = _prepared(
        control,
        np.asarray((1, 0.2j, -0.1), dtype=np.complex128),
        _plan(controller="double-log", spawn="sampled", attempts=8, history=6),
    )
    state = initialize_projector_monte_carlo(prepared, jax.random.key(7))
    uninterrupted = solve_projector_monte_carlo(prepared, state, steps=6)
    first = solve_projector_monte_carlo(prepared, state, steps=2)
    resumed = solve_projector_monte_carlo(prepared, first.state, steps=4)
    assert (
        int(uninterrupted.status)
        == int(resumed.status)
        == ProjectorMonteCarloStatus.SUCCESS
    )
    np.testing.assert_array_equal(
        uninterrupted.state.coefficients, resumed.state.coefficients
    )
    np.testing.assert_array_equal(
        uninterrupted.state.history.applied_shifts, resumed.state.history.applied_shifts
    )
    np.testing.assert_array_equal(
        uninterrupted.state.history.projected_numerator,
        resumed.state.history.projected_numerator,
    )
    np.testing.assert_array_equal(
        resumed.state.history.valid, np.ones((6,), dtype=np.bool_)
    )
    assert int(resumed.state.history.count) == int(resumed.state.step) == 6


def test_history_exhaustion_keeps_last_committed_boundary() -> None:
    control = two_boson_two_site_control()
    prepared = _prepared(
        control, np.asarray((1, 1, 1), dtype=np.complex128), _plan(history=1)
    )
    state = initialize_projector_monte_carlo(prepared, jax.random.key(8))
    result = solve_projector_monte_carlo(prepared, state, steps=2)
    assert int(result.status) == ProjectorMonteCarloStatus.HISTORY_EXHAUSTED
    assert int(result.accepted_steps) == int(result.state.step) == 1
    assert int(result.state.history.count) == 1
    validate_projector_state(prepared, result.state)


def test_zero_trial_denominator_is_retained_as_raw_measurement() -> None:
    control = two_boson_two_site_control(hopping=0.0, interaction=2.0)
    operator = control_operator(control)
    keys = control_keys(operator, control)
    problem = ProjectorMonteCarloProblem(
        operator,
        keys[1:2],
        np.asarray((1,), dtype=np.complex128),
        trial_keys=keys[:1],
        trial_coefficients=np.asarray((1,), dtype=np.complex128),
        dt=0.05,
        energy_unit=HARTREE,
        inverse_energy_unit=derived_unit("inverse-hartree", ((HARTREE, -1),)),
        provenance_id=control.provenance,
    )
    prepared = prepare_projector_monte_carlo(problem, _plan())
    result = step_projector_monte_carlo(
        prepared, initialize_projector_monte_carlo(prepared, jax.random.key(9))
    )
    assert int(result.status) == ProjectorMonteCarloStatus.SUCCESS
    np.testing.assert_array_equal(
        result.state.history.projected_denominator[:, 0],
        np.zeros((2,), dtype=np.complex128),
    )
    assert bool(result.state.history.valid[0])


def test_legacy_key_is_refused_at_initialization() -> None:
    control = two_boson_two_site_control()
    prepared = _prepared(control, np.asarray((1, 1, 1), dtype=np.complex128), _plan())
    with pytest.raises(TypeError):
        initialize_projector_monte_carlo(prepared, jax.random.PRNGKey(0))


def test_independent_double_log_controllers_use_each_replica_population() -> None:
    control = two_boson_two_site_control()
    prepared = _prepared(
        control,
        np.asarray((1, 1, 0), dtype=np.complex128),
        _plan(controller="double-log"),
    )
    state = initialize_projector_monte_carlo(prepared, jax.random.key(10))
    values = state.coefficients.at[1].multiply(jnp.complex128(2))
    population = jnp.sum(jnp.abs(values), axis=1, dtype=jnp.float64)
    state = eqx.tree_at(
        lambda item: (item.coefficients, item.populations), state, (values, population)
    )
    result = step_projector_monte_carlo(prepared, state)
    assert int(result.status) == ProjectorMonteCarloStatus.SUCCESS
    next_population = np.asarray(result.state.populations)
    expected = (
        np.asarray(state.shifts)
        - 0.05 / 0.05 * np.log(next_population / np.asarray(state.populations))
        - 0.001 / 0.05 * np.log(next_population)
    )
    np.testing.assert_allclose(result.state.shifts, expected, rtol=1e-14, atol=1e-15)
    assert result.state.shifts[0] != result.state.shifts[1]


def test_numerical_observation_failure_rolls_back_candidate_and_history() -> None:
    control = two_boson_two_site_control(hopping=0.0, interaction=2.0)
    operator = control_operator(control)
    keys = control_keys(operator, control)[:1]
    problem = ProjectorMonteCarloProblem(
        operator,
        keys,
        np.asarray((1,), dtype=np.complex128),
        trial_keys=keys,
        trial_coefficients=np.asarray((1e308,), dtype=np.complex128),
        dt=0.01,
        energy_unit=HARTREE,
        inverse_energy_unit=derived_unit("inverse-hartree", ((HARTREE, -1),)),
        provenance_id=control.provenance,
    )
    prepared = prepare_projector_monte_carlo(problem, _plan())
    state = initialize_projector_monte_carlo(prepared, jax.random.key(11))
    result = step_projector_monte_carlo(prepared, state)
    assert int(result.status) == ProjectorMonteCarloStatus.OBSERVATION_FAILURE
    np.testing.assert_array_equal(result.state.coefficients, state.coefficients)
    np.testing.assert_array_equal(result.state.history.valid, state.history.valid)
    assert int(result.state.step) == int(result.state.history.count) == 0


def test_uncertified_original_hamiltonian_is_refused() -> None:
    space = LocalSpacePlan.boson("site", 2)
    lowering = LocalOperatorPlan(
        space, "lowering", np.asarray(((0, 1), (0, 0)), dtype=np.complex128), (-1,)
    )
    specification = QuantumLatticeSpecification(
        (space,), (QuantumLatticeTerm((lowering,), label="non-Hermitian"),)
    )
    domain = QuantumConfigurationDomain(specification, species_ids=("particle",))
    operator = QuantumLatticeColumnOperator(
        prepare_quantum_lattice_columns(
            specification, QuantumColumnResourcePolicy(maximum_column_targets=2)
        ),
        domain,
    )
    key = domain.address(np.asarray((1,), dtype=np.int32)).key_words
    with pytest.raises(ValueError):
        ProjectorMonteCarloProblem(
            operator,
            key[None, :],
            np.asarray((1,), dtype=np.complex128),
            dt=0.01,
            energy_unit=HARTREE,
            inverse_energy_unit=derived_unit("inverse-hartree", ((HARTREE, -1),)),
            provenance_id="uncertified-original-control",
        )


def test_nonnative_hamiltonian_is_refused_without_generic_fallback() -> None:
    with pytest.raises(TypeError):
        ProjectorMonteCarloProblem(
            np.eye(2, dtype=np.complex128),  # ty: ignore[invalid-argument-type]
            np.asarray(((0,),), dtype=np.uint32),
            np.asarray((1,), dtype=np.complex128),
            dt=0.01,
            energy_unit=HARTREE,
            inverse_energy_unit=derived_unit("inverse-hartree", ((HARTREE, -1),)),
            provenance_id="nonnative-refusal",
        )


@pytest.mark.parametrize("seed", range(4), ids=lambda seed: f"compression-root-{seed}")
def test_final_compression_draw_has_exact_threshold_outcomes_without_range_overflow(
    seed: int,
) -> None:
    operator = _cancelling_operator()
    key = operator.domain.address(np.asarray((0,), dtype=np.int32)).key_words
    theta = 1e200
    coefficient = 0.5e200 * (1j)
    problem = ProjectorMonteCarloProblem(
        operator,
        key[None, :],
        np.asarray((coefficient,), dtype=np.complex128),
        trial_keys=key[None, :],
        trial_coefficients=np.asarray((1e-200,), dtype=np.complex128),
        dt=0.1,
        energy_unit=HARTREE,
        inverse_energy_unit=derived_unit("inverse-hartree", ((HARTREE, -1),)),
        provenance_id="large-threshold-complex-phase",
    )
    prepared = prepare_projector_monte_carlo(
        problem,
        _plan(
            replicas=1,
            support=1,
            groups=2,
            events=1,
            compression="threshold",
            theta=theta,
        ),
    )
    root = jax.random.key(seed)
    state = initialize_projector_monte_carlo(prepared, root)
    role = SampleAddress(
        "projector-monte-carlo",
        "physical-step",
        target=(operator.domain.domain_id, operator.operator_id),
        role="compression",
    )
    target_key = derive_key(
        root, role, 0, 0, 0, *(tuple(key[index] for index in range(key.shape[0]))), 0, 0
    )
    draw = float(jax.random.uniform(target_key, (), dtype=jnp.float64))
    result = step_projector_monte_carlo(prepared, state)
    if draw < 0.5:
        assert int(result.status) == ProjectorMonteCarloStatus.SUCCESS
        np.testing.assert_allclose(
            result.state.coefficients,
            np.asarray(((1j * theta,),), dtype=np.complex128),
            rtol=1e-15,
        )
    else:
        assert int(result.status) == ProjectorMonteCarloStatus.EXTINCTION
        np.testing.assert_array_equal(result.state.coefficients, state.coefficients)


@pytest.mark.strict_jax
def test_wrong_packed_word_dtype_is_refused_before_conversion() -> None:
    operator = _cancelling_operator()
    with pytest.raises(TypeError):
        ProjectorMonteCarloProblem(
            operator,
            np.asarray(((0,),), dtype=np.int32),
            np.asarray((1,), dtype=np.complex128),
            dt=0.01,
            energy_unit=HARTREE,
            inverse_energy_unit=derived_unit("inverse-hartree", ((HARTREE, -1),)),
            provenance_id="packed-dtype-refusal",
        )


def test_x64_disabled_is_explicit_admission_refusal() -> None:
    operator = _cancelling_operator()
    with jax.enable_x64(False), pytest.raises(ValueError):
        ProjectorMonteCarloProblem(
            operator,
            np.asarray(((0,),), dtype=np.uint32),
            np.asarray((1,), dtype=np.complex128),
            dt=0.01,
            energy_unit=HARTREE,
            inverse_energy_unit=derived_unit("inverse-hartree", ((HARTREE, -1),)),
            provenance_id="precision-refusal",
        )


def _wide_resource_problem() -> ProjectorMonteCarloProblem:
    modes, support = 193, 1025
    spaces = tuple(
        LocalSpacePlan(
            f"resource-site-{index}",
            ("zero", "one", "two"),
            (),
            np.zeros((3, 0), dtype=np.int32),
        )
        for index in range(modes)
    )
    identity = LocalOperatorPlan(
        spaces[0], "identity", np.eye(3, dtype=np.complex128), ()
    )
    specification = QuantumLatticeSpecification(
        spaces, (QuantumLatticeTerm((identity,), label="resource-identity"),)
    )
    domain = QuantumConfigurationDomain(specification, species_ids=("finite",) * modes)
    operator = QuantumLatticeColumnOperator(
        prepare_quantum_lattice_columns(
            specification, QuantumColumnResourcePolicy(maximum_column_targets=1)
        ),
        domain,
    )
    keys = np.zeros((support, domain.codec.word_count), dtype=np.uint32)
    # Base-four digits 0, 1, 2 are valid ternary coordinates; digit 3 is not.
    keys[:9, -1] = np.asarray((0, 1, 2, 4, 5, 6, 8, 9, 10), dtype=np.uint32)
    coefficients = np.zeros((support,), dtype=np.complex128)
    coefficients[:9] = np.complex128(1)
    active = np.zeros((support,), dtype=np.bool_)
    active[:9] = True
    return ProjectorMonteCarloProblem(
        operator,
        keys,
        coefficients,
        initial_active=active,
        trial_keys=keys[:1],
        trial_coefficients=coefficients[:1],
        dt=0.05,
        energy_unit=HARTREE,
        inverse_energy_unit=derived_unit("inverse-hartree", ((HARTREE, -1),)),
        provenance_id="wide-padded-resource-validation",
    )


def _wide_resource_plan(
    source_capacity: int, workspace_limit: int = 700_000
) -> ProjectorMonteCarloPlan:
    return ProjectorMonteCarloPlan(
        replicas=1,
        support_capacity=1025,
        group_capacity=1025,
        event_capacity=1,
        attempt_capacity=1,
        source_capacity=source_capacity,
        history_capacity=2,
        maximum_retained_bytes=500_000,
        maximum_workspace_bytes=workspace_limit,
    )


@pytest.mark.parametrize("source_capacity", (1, 7), ids=("single", "partial-tail"))
def test_wide_padded_support_validation_and_step_respect_admitted_decode_workspace(
    source_capacity: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem = _wide_resource_problem()
    plan = _wide_resource_plan(source_capacity)
    modes = len(problem.hamiltonian.domain.site_ids)
    decoded_budget = 4 * source_capacity * modes
    original_decode = QuantumAddressCodec.decode

    def bounded_decode(codec: QuantumAddressCodec, keys: ArrayLike, /) -> jax.Array:
        words = jnp.asarray(keys)
        decoded_count = words.size // codec.word_count
        assert decoded_count * modes * 4 <= decoded_budget
        return original_decode(codec, words)

    # Observe real codec requests, refusing any decoded workset larger than C.
    # This catches preparation as well as validation before a giant allocation.
    monkeypatch.setattr(QuantumAddressCodec, "decode", bounded_decode)
    prepared = prepare_projector_monte_carlo(problem, plan)
    assert prepared.workspace_bytes <= plan.maximum_workspace_bytes
    assert 4 * plan.support_capacity * modes > plan.maximum_workspace_bytes
    state = initialize_projector_monte_carlo(prepared, jax.random.key(29))
    validate_projector_state(prepared, state)
    result = step_projector_monte_carlo(prepared, state)
    assert int(result.status) == ProjectorMonteCarloStatus.SUCCESS
    validate_projector_state(prepared, result.state)
    expected = np.zeros((1, plan.support_capacity), dtype=np.complex128)
    expected[:, :9] = np.complex128(0.95)
    np.testing.assert_allclose(result.state.coefficients, expected, rtol=2e-15, atol=0)
    np.testing.assert_array_equal(result.state.support_keys, state.support_keys)
    assert int(result.state.step) == int(result.state.history.count) == 1

    with pytest.raises(ValueError):
        prepare_projector_monte_carlo(
            problem, _wide_resource_plan(source_capacity, prepared.workspace_bytes - 1)
        )

    # Only domain validity changes: support order, active prefix, coefficients,
    # populations, and all scientific/history bindings remain committed-valid.
    invalid_keys = state.support_keys.at[0, 8, -1].set(jnp.uint32(11))
    invalid_state = eqx.tree_at(lambda item: item.support_keys, state, invalid_keys)
    with pytest.raises(ValueError):
        validate_projector_state(prepared, invalid_state)

    invalid_padding = state.support_keys.at[0, -1, -1].set(jnp.uint32(1))
    padded_state = eqx.tree_at(lambda item: item.support_keys, state, invalid_padding)
    with pytest.raises(ValueError):
        validate_projector_state(prepared, padded_state)

    late_problem = eqx.tree_at(
        lambda item: (
            item.initial_keys,
            item.initial_coefficients,
            item.initial_active,
        ),
        problem,
        (
            problem.initial_keys.at[-1, -1].set(jnp.uint32(3)),
            problem.initial_coefficients.at[-1].set(jnp.complex128(1)),
            problem.initial_active.at[-1].set(jnp.bool_(True)),
        ),
    )
    with pytest.raises(ValueError):
        prepare_projector_monte_carlo(late_problem, plan)


def test_spawn_stream_folds_every_address_word_not_just_common_prefix() -> None:
    spaces = tuple(
        LocalSpacePlan(
            f"site-{index}", ("zero", "one"), (), np.zeros((2, 0), dtype=np.int32)
        )
        for index in range(33)
    )
    flip = np.asarray(((0, 1), (1, 0)), dtype=np.complex128)
    terms = tuple(
        QuantumLatticeTerm(
            (LocalOperatorPlan(spaces[index], "flip", flip, ()),), label=f"flip-{index}"
        )
        for index in (1, 2)
    )
    specification = QuantumLatticeSpecification(spaces, terms)
    domain = QuantumConfigurationDomain(specification, species_ids=("finite",) * 33)
    operator = QuantumLatticeColumnOperator(
        prepare_quantum_lattice_columns(
            specification, QuantumColumnResourcePolicy(maximum_column_targets=2)
        ),
        domain,
    )
    first_coordinates = np.zeros((33,), dtype=np.int32)
    second_coordinates = first_coordinates.copy()
    second_coordinates[-1] = 1
    sources = (domain.address(first_coordinates), domain.address(second_coordinates))
    keys = jnp.stack(tuple(source.key_words for source in sources))
    assert keys.shape == (2, 2)
    assert keys[0, 0] == keys[1, 0]
    assert keys[0, 1] != keys[1, 1]
    problem = ProjectorMonteCarloProblem(
        operator,
        keys,
        np.asarray((1, 1), dtype=np.complex128),
        dt=0.05,
        energy_unit=HARTREE,
        inverse_energy_unit=derived_unit("inverse-hartree", ((HARTREE, -1),)),
        provenance_id="two-word-semantic-spawn-control",
    )
    prepared = prepare_projector_monte_carlo(
        problem, _plan(replicas=1, support=4, groups=4, spawn="sampled", boost=0.5)
    )
    role = SampleAddress(
        "projector-monte-carlo",
        "physical-step",
        target=(domain.domain_id, operator.operator_id),
        role="spawn",
    )
    root = jax.random.key(0)
    targets: tuple[tuple[int, ...], ...] = ()
    for seed in range(128):
        root = jax.random.key(seed)
        routes = tuple(
            operator.sample_raw_excitation(
                derive_key(
                    root, role, 0, 0, 0, source.key_words[0], source.key_words[1], 0, 0
                ),
                source,
            )
            for source in sources
        )
        if int(routes[0].route_index) != int(routes[1].route_index):
            targets = tuple(
                tuple(int(word) for word in np.asarray(route.target_key))
                for route in routes
            )
            break
    if not targets:
        raise ValueError("Fixed roots must distinguish the two full-address streams.")
    expected = {
        tuple(int(word) for word in np.asarray(source.key_words)): complex(1)
        for source in sources
    }
    for target in targets:
        expected[target] = complex(-0.1)
    result = step_projector_monte_carlo(
        prepared, initialize_projector_monte_carlo(prepared, root)
    )
    assert int(result.status) == ProjectorMonteCarloStatus.SUCCESS
    observed = {
        tuple(int(word) for word in key): complex(value)
        for key, value in zip(
            np.asarray(result.state.support_keys[0])[np.asarray(result.state.active[0])],
            np.asarray(result.state.coefficients[0])[np.asarray(result.state.active[0])],
            strict=True,
        )
    }
    assert observed == expected


def test_reciprocal_length_units_are_not_hamiltonian_energy_units() -> None:
    control = two_boson_two_site_control()
    operator = control_operator(control)
    with pytest.raises(ValueError, match="energy or explicit dimensionless"):
        ProjectorMonteCarloProblem(
            operator,
            control_keys(operator, control),
            control.ground_vector,
            dt=0.02,
            energy_unit=METER,
            inverse_energy_unit=derived_unit("inverse-meter", ((METER, -1),)),
            provenance_id=control.provenance,
        )


def test_boolean_step_is_wrong_scalar_kind() -> None:
    control = two_boson_two_site_control()
    operator = control_operator(control)
    with pytest.raises(TypeError, match="finite real number"):
        ProjectorMonteCarloProblem(
            operator,
            control_keys(operator, control),
            control.ground_vector,
            dt=True,
            energy_unit=HARTREE,
            inverse_energy_unit=derived_unit("inverse-hartree", ((HARTREE, -1),)),
            provenance_id=control.provenance,
        )
