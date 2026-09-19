import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.linalg import (
    DenseLU,
    FailurePolicy,
    LinearSolvePolicy,
    LowRankSolvePolicy,
    MixedPrecisionPolicy,
    PfaffianPolicy,
)
from phydrax.nn.quantum import (
    pfaffian_jastrow_incremental_target,
    PfaffianJastrowAmplitude,
    PfaffianJastrowCache,
)
from phydrax.operators import LogAmplitude
from phydrax.sampling import (
    MetropolisHastings,
    SingleCoordinateGaussianProposal,
    SingleCoordinateProposalPayload,
)


def _policy(**updates):
    return PfaffianPolicy(**updates)


def _two_particle_pairing(coordinates):
    pair = coordinates[1, 0] - coordinates[0, 0]
    zero = jnp.zeros((), dtype=pair.dtype)
    return jnp.stack((jnp.stack((zero, pair)), jnp.stack((-pair, zero))))


def _equivariant_pairing(coordinates):
    coordinate = coordinates[:, 0]
    features = jnp.stack(
        (
            jnp.ones_like(coordinate),
            coordinate,
            coordinate**2,
            coordinate**3,
        ),
        axis=-1,
    )
    return (
        features[:, 0, None] * features[None, :, 1]
        - features[:, 1, None] * features[None, :, 0]
        + features[:, 2, None] * features[None, :, 3]
        - features[:, 3, None] * features[None, :, 2]
    )


def _amplitude(pairing=_two_particle_pairing, jastrow=lambda coordinates: 0.0):
    return PfaffianJastrowAmplitude(
        pairing,
        jastrow,
        particle_count=2,
        spatial_dimension=1,
        pairing_id="test-pairing",
        cusp_id="test-cusp",
        policy=_policy(),
    )


def _update_policy(*, condition_limit=1.0e12):
    status = FailurePolicy("status")
    return LowRankSolvePolicy(
        LinearSolvePolicy(DenseLU(), failure=status),
        condition_limit=condition_limit,
        base_nonsingularity="asserted",
        failure=status,
    )


def _local_four_particle_pairing(coordinates):
    coordinate = coordinates[:, 0]
    base = jnp.asarray(
        [
            [0.0, 2.0, 0.3, -0.2],
            [-2.0, 0.0, 0.4, 0.1],
            [-0.3, -0.4, 0.0, 1.5],
            [0.2, -0.1, -1.5, 0.0],
        ],
        dtype=coordinate.dtype,
    )
    local = 0.1 * (coordinate[:, None] - coordinate[None, :])
    return (base + local).astype(jnp.complex64) * jnp.exp(0.2j)


def _complex_four_particle_jastrow(coordinates):
    return 0.03 * jnp.sum(coordinates**2) + 0.07j * jnp.sum(coordinates)


def _incremental_amplitude(pairing=_local_four_particle_pairing):
    return PfaffianJastrowAmplitude(
        pairing,
        _complex_four_particle_jastrow,
        particle_count=4,
        spatial_dimension=1,
        pairing_id="local-four-particle-pairing",
        cusp_id="complex-quadratic-cusp",
        policy=_policy(),
    )


def _incremental_target(model, *, capacity=4, refresh_cadence=32, tolerance=1e-5):
    return pfaffian_jastrow_incremental_target(
        model,
        capacity=capacity,
        update_policy=_update_policy(),
        maximum_chains=8,
        refresh_cadence=refresh_cadence,
        locality_tolerance=tolerance,
    )


def _coordinates():
    return jnp.asarray([[0.0], [0.8], [1.7], [2.6]])


def _payload(index, displacement):
    return SingleCoordinateProposalPayload(
        index=jnp.asarray(index, dtype=jnp.int32),
        displacement=jnp.asarray(displacement),
    )


def test_equivariant_pairing_is_antisymmetric_under_particle_exchange():
    def symmetric_jastrow(coordinates):
        return 0.05 * jnp.sum(coordinates**2) + 0.2j * jnp.sum(coordinates)

    amplitude = PfaffianJastrowAmplitude(
        _equivariant_pairing,
        symmetric_jastrow,
        particle_count=4,
        spatial_dimension=1,
        pairing_id="equivariant-polynomial-pairing",
        cusp_id="symmetric-quadratic-cusp",
        policy=_policy(),
    )
    coordinates = jnp.asarray([[0.0], [1.0], [2.0], [3.0]])
    exchanged_coordinates = coordinates[jnp.asarray([1, 0, 2, 3])]

    value = amplitude(coordinates)
    exchanged = amplitude(exchanged_coordinates)

    assert value.valid
    assert exchanged.valid
    assert jnp.allclose(exchanged.log_abs, value.log_abs)
    assert jnp.allclose(exchanged.phase, -value.phase)


def test_complex_pfaffian_phase_and_jastrow_log_compose_without_phase_loss():
    pairing_phase = jnp.exp(0.35j)

    def complex_pairing(coordinates):
        return _two_particle_pairing(coordinates).astype(jnp.complex64) * pairing_phase

    def complex_jastrow(_coordinates):
        return jnp.asarray(0.3 + 0.2j)

    amplitude = _amplitude(complex_pairing, complex_jastrow)
    coordinates = jnp.asarray([[0.0], [2.0]])
    value = amplitude(coordinates)

    assert isinstance(value, LogAmplitude)
    assert value.valid
    assert value.nonzero
    assert jnp.allclose(value.log_abs, jnp.log(2.0) + 0.3)
    assert jnp.allclose(value.phase, jnp.exp(0.55j))


def test_constructor_and_callable_outputs_require_exact_finite_shapes():
    with pytest.raises(ValueError, match="positive and even"):
        PfaffianJastrowAmplitude(
            _two_particle_pairing,
            lambda coordinates: 0.0,
            particle_count=3,
            spatial_dimension=1,
            pairing_id="odd-pairing",
            cusp_id="odd-cusp",
            policy=_policy(),
        )

    amplitude = _amplitude()
    with pytest.raises(ValueError, match="configuration"):
        amplitude(jnp.ones((2,)))
    with pytest.raises(ValueError, match="configuration"):
        amplitude(jnp.ones((3, 1)))
    with pytest.raises(ValueError, match="configuration"):
        amplitude(jnp.ones((2, 0)))

    wrong_pairing = _amplitude(lambda coordinates: jnp.zeros((2, 3)))
    with pytest.raises(ValueError, match="pairing_evaluator"):
        wrong_pairing(jnp.ones((2, 1)))

    wrong_jastrow = _amplitude(_two_particle_pairing, lambda coordinates: jnp.zeros((1,)))
    with pytest.raises(ValueError, match="jastrow"):
        wrong_jastrow(jnp.asarray([[0.0], [1.0]]))


def test_incremental_target_rejects_unsupported_mixed_precision_lu():
    status = FailurePolicy("status")
    update_policy = LowRankSolvePolicy(
        LinearSolvePolicy(
            DenseLU(),
            failure=status,
            precision=MixedPrecisionPolicy(
                factorization_dtype=jnp.float32,
                maximum_refinement_steps=1,
            ),
        ),
        base_nonsingularity="asserted",
        failure=status,
    )

    with pytest.raises(ValueError, match="full-precision"):
        pfaffian_jastrow_incremental_target(
            _incremental_amplitude(),
            capacity=4,
            update_policy=update_policy,
            maximum_chains=1,
        )


def test_required_skew_policy_failure_propagates_to_amplitude_validity():
    def non_skew_pairing(coordinates):
        scale = coordinates[1, 0] - coordinates[0, 0]
        return jnp.asarray([[0.0, 1.0], [-0.9, 0.0]]) * scale

    amplitude = PfaffianJastrowAmplitude(
        non_skew_pairing,
        lambda coordinates: 0.0,
        particle_count=2,
        spatial_dimension=1,
        pairing_id="non-skew-pairing",
        cusp_id="zero-cusp",
        policy=_policy(skew_mode="require", antisymmetry_tolerance=1e-12),
    )
    value = amplitude(jnp.asarray([[0.0], [1.0]]))

    assert not value.valid
    assert not value.nonzero


def test_pfaffian_node_is_an_invalid_zero_amplitude():
    amplitude = _amplitude(lambda coordinates: jnp.zeros((2, 2)))
    value = amplitude(jnp.asarray([[0.0], [1.0]]))

    assert not value.valid
    assert not value.nonzero
    assert jnp.isneginf(value.log_abs)


def test_incremental_target_rejects_a_pfaffian_node_without_singular_solve_state():
    model = _amplitude(_two_particle_pairing)
    target = pfaffian_jastrow_incremental_target(
        model,
        capacity=2,
        update_policy=_update_policy(),
        maximum_chains=1,
    )
    current = jnp.asarray([[0.0], [1.0]])
    proposed = jnp.asarray([[0.0], [0.0]])
    state = target.initialize(current)

    proposal = target.propose(state, proposed, _payload(1, -1.0))

    assert not proposal.valid
    assert not proposal.proposed_cache.native_valid
    assert not proposal.proposed_cache.compact_eligible
    assert jnp.isneginf(proposal.proposed_cache.log_abs)


def test_external_vmap_and_jit_preserve_canonical_batch_shape():
    amplitude = _amplitude(
        _two_particle_pairing,
        lambda coordinates: 0.1 * jnp.sum(coordinates),
    )
    batch = jnp.asarray(
        [
            [[0.0], [1.0]],
            [[-1.0], [2.0]],
            [[0.5], [2.5]],
        ]
    )

    values = jax.jit(jax.vmap(amplitude))(batch)

    expected = jnp.log(jnp.asarray([1.0, 3.0, 2.0])) + 0.1 * jnp.sum(batch, axis=(1, 2))
    assert values.log_abs.shape == (3,)
    assert values.phase.shape == (3,)
    assert values.valid.shape == (3,)
    assert jnp.all(values.valid)
    assert jnp.allclose(values.log_abs, expected)
    assert jnp.allclose(values.phase, jnp.ones((3,), dtype=values.phase.dtype))


def test_coordinate_first_and_second_derivatives_match_away_from_nodes():
    quadratic_coefficient = 0.2

    def jastrow(coordinates):
        return quadratic_coefficient * jnp.sum(coordinates**2) + 0.17j * jnp.sum(
            coordinates
        )

    amplitude = _amplitude(_two_particle_pairing, jastrow)
    coordinates = jnp.asarray([[0.3], [1.7]])

    def log_abs(flat_coordinates):
        return amplitude(flat_coordinates.reshape(2, 1)).log_abs

    flat = coordinates.reshape(-1)
    gradient = jax.grad(log_abs)(flat)
    hessian = jax.hessian(log_abs)(flat)
    separation = flat[1] - flat[0]
    inverse = 1.0 / separation
    expected_gradient = jnp.asarray(
        [
            -inverse + 2.0 * quadratic_coefficient * flat[0],
            inverse + 2.0 * quadratic_coefficient * flat[1],
        ]
    )
    inverse_squared = inverse**2
    expected_hessian = jnp.asarray(
        [
            [-inverse_squared + 2.0 * quadratic_coefficient, inverse_squared],
            [inverse_squared, -inverse_squared + 2.0 * quadratic_coefficient],
        ]
    )

    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.all(jnp.isfinite(hessian))
    assert jnp.allclose(gradient, expected_gradient, rtol=2e-5, atol=2e-6)
    assert jnp.allclose(hessian, expected_hessian, rtol=2e-5, atol=2e-6)


def test_incremental_target_compact_ratio_phase_and_acceptance_selection_are_exact():
    model = _incremental_amplitude()
    target = _incremental_target(model)
    current = _coordinates()
    proposed = current.at[2, 0].add(0.15)
    state = target.initialize(current)
    proposal = target.propose(state, proposed, _payload(2, 0.15))
    exact_current, exact_proposed = model(current), model(proposed)

    assert isinstance(state.cache, PfaffianJastrowCache)
    assert proposal.valid
    assert proposal.proposed_cache.compact_update
    assert not proposal.proposed_cache.rebased
    assert proposal.proposed_cache.native_valid
    assert proposal.proposed_cache.sequence.active_rank == 2
    assert jnp.allclose(
        proposal.log_ratio,
        2.0 * (exact_proposed.log_abs - exact_current.log_abs),
        rtol=2e-5,
        atol=2e-6,
    )
    assert jnp.allclose(
        proposal.proposed_cache.phase,
        exact_proposed.phase,
        rtol=2e-5,
        atol=2e-6,
    )

    rejected = target.commit(state, proposed, proposal, jnp.asarray(False))
    accepted = target.commit(state, proposed, proposal, jnp.asarray(True))
    assert jnp.array_equal(rejected.position, current)
    assert jnp.array_equal(rejected.cache.pairing_matrix, state.cache.pairing_matrix)
    assert rejected.cache.sequence.active_rank == 0
    assert jnp.array_equal(accepted.position, proposed)
    assert accepted.cache.sequence.active_rank == 2
    assert jnp.allclose(accepted.log_target, 2.0 * exact_proposed.log_abs)


def test_incremental_target_nonlocal_pairing_change_takes_exact_full_rebase():
    global_skew = jnp.asarray(
        [
            [0.0, 0.2, -0.3, 0.4],
            [-0.2, 0.0, 0.5, -0.6],
            [0.3, -0.5, 0.0, 0.7],
            [-0.4, 0.6, -0.7, 0.0],
        ]
    )

    def nonlocal_pairing(coordinates):
        return _local_four_particle_pairing(coordinates) + (
            0.05 * jnp.sum(coordinates) * global_skew
        )

    model = _incremental_amplitude(nonlocal_pairing)
    target = _incremental_target(model, tolerance=1e-7)
    current = _coordinates()
    proposed = current.at[1, 0].add(-0.2)
    state = target.initialize(current)
    proposal = target.propose(state, proposed, _payload(1, -0.2))
    exact = model(proposed)

    assert proposal.valid
    assert proposal.proposed_cache.rebased
    assert not proposal.proposed_cache.compact_update
    assert proposal.proposed_cache.locality_residual > target.cache_tolerance
    assert proposal.proposed_cache.sequence.active_rank == 0
    assert jnp.allclose(
        state.log_target + proposal.log_ratio,
        2.0 * exact.log_abs,
        rtol=2e-5,
        atol=2e-6,
    )
    assert jnp.allclose(proposal.proposed_cache.phase, exact.phase)


def test_incremental_target_never_approximates_weak_nonlocal_pairing_changes():
    global_skew = jnp.asarray(
        [
            [0.0, 0.2, -0.3, 0.4],
            [-0.2, 0.0, 0.5, -0.6],
            [0.3, -0.5, 0.0, 0.7],
            [-0.4, 0.6, -0.7, 0.0],
        ]
    )

    def weakly_nonlocal_pairing(coordinates):
        return _local_four_particle_pairing(coordinates) + (
            1.0e-9 * jnp.sum(coordinates) * global_skew
        )

    model = _incremental_amplitude(weakly_nonlocal_pairing)
    target = _incremental_target(model, tolerance=1.0e-5)
    current = _coordinates()
    proposed = current.at[1, 0].add(-0.2)
    state = target.initialize(current)
    proposal = target.propose(state, proposed, _payload(1, -0.2))
    exact = model(proposed)

    assert proposal.valid
    assert 0.0 < proposal.proposed_cache.locality_residual < target.cache_tolerance
    assert proposal.proposed_cache.rebased
    assert jnp.allclose(
        state.log_target + proposal.log_ratio,
        2.0 * exact.log_abs,
        rtol=2e-5,
        atol=2e-6,
    )


def test_incremental_target_rebases_when_fixed_capacity_is_exhausted():
    model = _incremental_amplitude()
    target = _incremental_target(model, capacity=2)
    current = _coordinates()
    first_position = current.at[0, 0].add(0.1)
    initial = target.initialize(current)
    first = target.propose(initial, first_position, _payload(0, 0.1))
    accepted = target.commit(initial, first_position, first, jnp.asarray(True))
    second_position = first_position.at[3, 0].add(-0.12)
    second = target.propose(accepted, second_position, _payload(3, -0.12))
    exact = model(second_position)

    assert first.proposed_cache.compact_update
    assert accepted.cache.sequence.remaining_capacity == 0
    assert second.valid
    assert second.proposed_cache.rebased
    assert second.proposed_cache.sequence.active_rank == 0
    assert jnp.allclose(
        accepted.log_target + second.log_ratio,
        2.0 * exact.log_abs,
        rtol=2e-5,
        atol=2e-6,
    )
    assert jnp.allclose(second.proposed_cache.phase, exact.phase)


def test_incremental_target_jit_vmap_matches_full_evaluation_for_multiple_chains():
    model = _incremental_amplitude()
    target = _incremental_target(model)
    positions = jnp.stack((_coordinates(), _coordinates() + 0.25))
    indices = jnp.asarray([0, 3], dtype=jnp.int32)
    displacements = jnp.asarray([0.11, -0.09])
    proposed = positions.at[jnp.arange(2), indices, 0].add(displacements)

    def initialize_many(bound_target, values):
        return jax.vmap(bound_target.initialize)(values)

    def propose_many(bound_target, states_, positions_, indices_, displacements_):
        def propose_one(state, position, index, displacement):
            return bound_target.propose(
                state,
                position,
                _payload(index, displacement),
            )

        return jax.vmap(propose_one)(
            states_,
            positions_,
            indices_,
            displacements_,
        )

    states = eqx.filter_jit(initialize_many)(target, positions)
    proposals = eqx.filter_jit(propose_many)(
        target,
        states,
        proposed,
        indices,
        displacements,
    )
    exact_current = jax.vmap(model)(positions)
    exact_proposed = jax.vmap(model)(proposed)

    assert jnp.all(proposals.valid)
    assert jnp.allclose(
        proposals.log_ratio,
        2.0 * (exact_proposed.log_abs - exact_current.log_abs),
        rtol=2e-5,
        atol=2e-6,
    )
    assert jnp.allclose(
        proposals.proposed_cache.phase,
        exact_proposed.phase,
        rtol=2e-5,
        atol=2e-6,
    )


def test_incremental_target_scheduled_exact_refresh_rebases_compact_history():
    model = _incremental_amplitude()
    target = _incremental_target(model, capacity=4, refresh_cadence=2)
    current = _coordinates()
    proposed = current.at[1, 0].add(0.13)
    initial = target.initialize(current)
    proposal = target.propose(initial, proposed, _payload(1, 0.13))
    accepted = target.commit(initial, proposed, proposal, jnp.asarray(True))
    refreshed = eqx.filter_jit(lambda state: target.refresh(state))(accepted)

    assert target.refresh_cadence == 2
    assert accepted.cache.sequence.active_rank == 2
    assert refreshed.valid
    assert refreshed.cache.rebased
    assert refreshed.cache.sequence.active_rank == 0
    assert jnp.allclose(refreshed.log_target, 2.0 * model(proposed).log_abs)
    assert jnp.allclose(refreshed.cache.phase, model(proposed).phase)


class _ScaledPairing(eqx.Module):
    scale: jax.Array

    def __call__(self, coordinates):
        return self.scale * _local_four_particle_pairing(coordinates)


def test_model_bound_target_factory_rebinds_parameter_changes_with_stable_identity():
    policy = _update_policy()
    factory = eqx.Partial(
        pfaffian_jastrow_incremental_target,
        capacity=4,
        update_policy=policy,
        maximum_chains=4,
        refresh_cadence=3,
        locality_tolerance=1e-5,
    )
    model = PfaffianJastrowAmplitude(
        _ScaledPairing(jnp.asarray(1.0)),
        _complex_four_particle_jastrow,
        particle_count=4,
        spatial_dimension=1,
        pairing_id="scaled-local-four-particle-pairing",
        cusp_id="complex-quadratic-cusp",
        policy=_policy(),
    )
    changed_model = eqx.tree_at(
        lambda amplitude: amplitude.pairing_evaluator.scale,
        model,
        jnp.asarray(1.4),
    )
    original_target = factory(model)
    changed_target = factory(changed_model)
    kernel = MetropolisHastings(SingleCoordinateGaussianProposal(0.1))
    state = kernel.initialize(original_target, _coordinates()[None, ...])
    rebound = eqx.filter_jit(lambda value: kernel.rebind(changed_target, value))(state)
    exact = 2.0 * changed_model(_coordinates()).log_abs

    assert original_target.target_id == changed_target.target_id
    assert rebound.target_id == state.target_id
    assert rebound.step_index == state.step_index
    assert jnp.all(rebound.valid)
    assert jnp.allclose(rebound.log_target, exact[None])
    assert jnp.allclose(rebound.cache.phase, changed_model(_coordinates()).phase[None])
