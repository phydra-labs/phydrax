import jax.numpy as jnp
import pytest

import phydrax.tensor_decomposition as td


def _real_rank_two_tensor(order=3):
    weights, factors = td.normalize_waring_components(
        jnp.asarray([1.7, -0.65], dtype=jnp.float32),
        jnp.asarray([[1.0, 0.35], [0.55, 1.0]], dtype=jnp.float32),
        order,
    )
    return weights, factors, td.reconstruct_symmetric_tensor(weights, factors, order)


def test_real_cubic_recovers_normalized_components_and_reconstruction():
    _, _, tensor = _real_rank_two_tensor()
    result = td.solve_symmetric_waring(td.SymmetricWaringProblem(tensor, 2))

    assert result.successful
    assert result.evidence.path_accepted
    assert result.evidence.physical_accepted
    assert result.evidence.refinement_attempted
    assert result.weights.shape == (2,)
    assert result.factors.shape == (2, 2)
    assert jnp.allclose(jnp.linalg.norm(result.factors, axis=1), 1.0, atol=2e-5)
    assert jnp.allclose(result.reconstruction, tensor, rtol=2e-5, atol=2e-5)
    assert result.relative_residual < 2e-5


def test_complex_cubic_uses_algebraic_power_and_recovers_tensor():
    order = 3
    weights, factors = td.normalize_waring_components(
        jnp.asarray([1.2 + 0.3j, -0.4 + 0.8j], dtype=jnp.complex64),
        jnp.asarray(
            [[1.0, 0.2 + 0.4j], [0.6 - 0.3j, 1.0]],
            dtype=jnp.complex64,
        ),
        order,
    )
    tensor = td.reconstruct_symmetric_tensor(weights, factors, order)
    result = td.solve_symmetric_waring(
        td.SymmetricWaringProblem(tensor, 2),
        refinement=td.SymmetricWaringRefinement(enabled=False),
    )

    assert result.successful
    assert jnp.iscomplexobj(result.weights)
    assert jnp.iscomplexobj(result.factors)
    assert jnp.allclose(result.reconstruction, tensor, rtol=3e-5, atol=3e-5)
    assert jnp.allclose(jnp.linalg.norm(result.factors, axis=1), 1.0, atol=3e-5)
    pivots = jnp.argmax(jnp.abs(result.factors), axis=1)
    pivot_values = result.factors[jnp.arange(result.rank), pivots]
    assert jnp.allclose(jnp.imag(pivot_values), 0.0, atol=3e-5)
    assert jnp.all(jnp.real(pivot_values) >= 0.0)


def test_component_normalization_is_permutation_scale_and_phase_invariant():
    order = 4
    weights = jnp.asarray([1.1 - 0.2j, -0.7 + 0.5j], dtype=jnp.complex64)
    factors = jnp.asarray(
        [[1.0 + 0.3j, -0.4j], [0.25 - 0.5j, 1.0]],
        dtype=jnp.complex64,
    )
    canonical_weights, canonical_factors = td.normalize_waring_components(
        weights, factors, order
    )
    permutation = jnp.asarray([1, 0])
    scales = jnp.asarray([0.4 + 1.1j, -1.3 + 0.2j], dtype=jnp.complex64)
    equivalent_weights = weights[permutation] / scales**order
    equivalent_factors = factors[permutation] * scales[:, None]
    normalized_weights, normalized_factors = td.normalize_waring_components(
        equivalent_weights,
        equivalent_factors,
        order,
    )

    assert jnp.allclose(normalized_weights, canonical_weights, rtol=2e-5, atol=2e-5)
    assert jnp.allclose(normalized_factors, canonical_factors, rtol=2e-5, atol=2e-5)
    assert jnp.allclose(
        td.reconstruct_symmetric_tensor(equivalent_weights, equivalent_factors, order),
        td.reconstruct_symmetric_tensor(weights, factors, order),
        rtol=2e-5,
        atol=2e-5,
    )


def test_over_requested_rank_reports_hankel_rank_without_padding_components():
    _, _, tensor = _real_rank_two_tensor(order=5)
    result = td.solve_symmetric_waring(
        td.SymmetricWaringProblem(tensor, 3),
        refinement=td.SymmetricWaringRefinement(enabled=False),
    )

    assert not result.successful
    assert result.status == int(
        td.SymmetricWaringStatus.OVER_REQUESTED_RANK_OR_REPEATED_FACTOR
    )
    assert result.evidence.observed_hankel_rank == 2
    assert result.rank == 0


def test_repeated_factor_ambiguity_fails_as_rank_collapse():
    factor = jnp.asarray([1.0, 0.45], dtype=jnp.float32)
    tensor = td.reconstruct_symmetric_tensor(
        jnp.asarray([0.8, -0.3], dtype=jnp.float32),
        jnp.stack((factor, factor)),
        3,
    )
    result = td.solve_symmetric_waring(
        td.SymmetricWaringProblem(tensor, 2),
        refinement=td.SymmetricWaringRefinement(enabled=False),
    )

    assert not result.successful
    assert result.status == int(
        td.SymmetricWaringStatus.OVER_REQUESTED_RANK_OR_REPEATED_FACTOR
    )
    assert result.evidence.observed_hankel_rank == 1
    assert "coalesce" in result.evidence.detail


def test_positive_dimensional_request_fails_as_multiple_decomposition():
    _, _, tensor = _real_rank_two_tensor()
    result = td.solve_symmetric_waring(
        td.SymmetricWaringProblem(tensor, 3),
        refinement=td.SymmetricWaringRefinement(enabled=False),
    )

    assert not result.successful
    assert result.status == int(
        td.SymmetricWaringStatus.NONGENERIC_OR_MULTIPLE_DECOMPOSITION
    )
    assert "positive-dimensional" in result.evidence.detail


def test_resource_policy_rejects_before_hankel_or_provider_work():
    _, _, tensor = _real_rank_two_tensor()
    plan = td.prepare_symmetric_waring(
        td.SymmetricWaringProblem(tensor, 2),
        resources=td.SymmetricWaringResourcePolicy(maximum_tensor_entries=7),
        refinement=td.SymmetricWaringRefinement(enabled=False),
    )
    result = td.solve_symmetric_waring(plan)

    assert not plan.resource_admitted
    assert plan.resource_rejection == "tensor entries require 8, exceeding limit 7"
    assert not result.successful
    assert result.status == int(td.SymmetricWaringStatus.RESOURCE_REJECTED)
    assert not result.evidence.path_accepted


def test_invalid_nonsymmetric_tensor_and_zero_factor_are_rejected():
    nonsymmetric = jnp.zeros((2, 2, 2), dtype=jnp.float32).at[0, 1, 1].set(1.0)
    with pytest.raises(ValueError, match="not symmetric"):
        td.SymmetricWaringProblem(nonsymmetric, 1)
    with pytest.raises(ValueError, match="nonzero"):
        td.normalize_waring_components(
            jnp.asarray([1.0]),
            jnp.zeros((1, 2)),
            3,
        )
