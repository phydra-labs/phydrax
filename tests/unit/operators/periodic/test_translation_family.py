import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.operators.periodic import (
    coalesce_periodic_translation_family,
    differentiate_periodic_translation_family,
    evaluate_periodic_translation_family,
    PeriodicResourceError,
    PeriodicTranslationFamilyPlan,
    PeriodicTranslationFamilyState,
    realize_periodic_translation_family,
    refresh_periodic_translation_family,
)


def _chain(*, dense_limit=1024, finite_limit=1024):
    return coalesce_periodic_translation_family(
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [[1], [1], [-1], [-1]],
        [[[-0.5]], [[-0.5]], [[-0.5]], [[-0.5]]],
        1,
        maximum_dense_entries=dense_limit,
        maximum_finite_entries=finite_limit,
    )


def test_reverse_coalescing_sparse_action_dense_and_derivative_agree():
    prepared = _chain()
    q = jnp.asarray([[0.0], [0.25]])
    dense = evaluate_periodic_translation_family(prepared, q)
    action = prepared.apply(q, jnp.ones((2, 1)))
    derivative = differentiate_periodic_translation_family(prepared, q)

    assert prepared.plan.relation.capacity == 2
    np.testing.assert_array_equal(np.asarray(prepared.plan.reverse_indices), [1, 0])
    np.testing.assert_allclose(action, dense[..., 0], atol=1.0e-12)
    np.testing.assert_allclose(dense[:, 0, 0], [-2.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(derivative[1, 0, 0, 0], 4.0 * np.pi, atol=1.0e-12)

    compiled = jax.jit(lambda points, vectors: prepared.apply(points, vectors))
    np.testing.assert_allclose(compiled(q, jnp.ones((2, 1))), action)


def test_refresh_preserves_structure_and_finite_boundaries_are_exact():
    prepared = _chain()
    doubled_state = PeriodicTranslationFamilyState(
        prepared.plan, 2.0 * prepared.state.values
    )
    refreshed = refresh_periodic_translation_family(prepared, doubled_state)
    open_chain = realize_periodic_translation_family(refreshed, (3,))
    ring = realize_periodic_translation_family(
        refreshed, (3,), periodic_axes=(True,), twists=(0.0,)
    )

    np.testing.assert_allclose(
        open_chain.to_dense(),
        [[0.0, -2.0, 0.0], [-2.0, 0.0, -2.0], [0.0, -2.0, 0.0]],
    )
    np.testing.assert_allclose(
        ring.to_dense(),
        [[0.0, -2.0, -2.0], [-2.0, 0.0, -2.0], [-2.0, -2.0, 0.0]],
    )
    assert refreshed.plan.plan_id == prepared.plan.plan_id
    assert refreshed.prepared_id != prepared.prepared_id


def test_family_rejects_broken_adjoint_and_resources_before_materialization():
    prepared = _chain(dense_limit=1, finite_limit=2)
    with pytest.raises(ValueError, match="Hermitian adjoint"):
        PeriodicTranslationFamilyState(
            prepared.plan, prepared.state.values.at[0, 0, 0].set(2.0)
        )
    with pytest.raises(PeriodicResourceError, match="dense"):
        prepared.evaluate([[0.0], [0.25]])
    with pytest.raises(PeriodicResourceError, match="Finite realization"):
        realize_periodic_translation_family(prepared, (2,))


def test_plan_rejects_noninvolutive_reverse_map():
    relation = prepared_relation = _chain().plan.relation
    with pytest.raises(ValueError, match="opposite translations"):
        PeriodicTranslationFamilyPlan(
            relation,
            [[1], [1]],
            [1, 0],
        )
    assert prepared_relation.capacity == 2
