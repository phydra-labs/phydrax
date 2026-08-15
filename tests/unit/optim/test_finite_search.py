#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.optim._finite import _exhaustive_minimum


def test_finite_axis_validates_correlated_array_payloads():
    axis = phx.optim.FiniteAxis(
        {
            "scalar": jnp.asarray([1.0, 2.0, 3.0]),
            "vector": jnp.arange(6.0).reshape((3, 2)),
        }
    )

    assert axis.size == 3
    assert axis.payload_shapes == ((), (2,))
    assert axis.point_spec()["scalar"].shape == ()
    assert axis.point_spec()["vector"].shape == (2,)

    with pytest.raises(ValueError, match="at least one"):
        phx.optim.FiniteAxis(())
    with pytest.raises(ValueError, match="leading candidate"):
        phx.optim.FiniteAxis(jnp.asarray(1.0))
    with pytest.raises(ValueError, match="nonempty"):
        phx.optim.FiniteAxis(jnp.empty((0, 2)))
    with pytest.raises(ValueError, match="same leading"):
        phx.optim.FiniteAxis(
            {
                "left": jnp.ones((2,)),
                "right": jnp.ones((3,)),
            }
        )
    with pytest.raises(TypeError, match="numerical or boolean"):
        phx.optim.FiniteAxis("abc")


def test_finite_product_indexing_preserves_structure_and_never_clips():
    space = phx.optim.FiniteProductSpace(
        {
            "x": phx.optim.FiniteAxis(jnp.asarray([10.0, 20.0])),
            "pair": phx.optim.FiniteAxis(
                {"vector": jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])}
            ),
        }
    )

    assert space.product_shape == (3, 2)
    assert space.size == 6
    assert space.axis_paths == ("['pair']", "['x']")
    np.testing.assert_array_equal(
        space.ravel_index((jnp.asarray([0, 2]), jnp.asarray([0, 1]))),
        jnp.asarray([0, 5]),
    )
    product_index = space.unravel_index(jnp.asarray([0, 5]))
    np.testing.assert_array_equal(product_index[0], jnp.asarray([0, 2]))
    np.testing.assert_array_equal(product_index[1], jnp.asarray([0, 1]))

    selected = space.take(jnp.asarray([0, 5]))
    np.testing.assert_array_equal(
        selected["pair"]["vector"],
        jnp.asarray([[1.0, 2.0], [5.0, 6.0]]),
    )
    np.testing.assert_array_equal(selected["x"], jnp.asarray([10.0, 20.0]))

    with pytest.raises(IndexError, match="out of range"):
        space.take(-1)
    with pytest.raises(IndexError, match="out of range"):
        space.take(space.size)
    with pytest.raises(IndexError, match="axis 0"):
        space.ravel_index((3, 0))
    with pytest.raises(ValueError, match="length"):
        space.ravel_index((0,))

    compiled = jax.jit(lambda index: space.take(index))
    selected_last = compiled(jnp.asarray(5))
    np.testing.assert_array_equal(selected_last["x"], space.take(5)["x"])
    np.testing.assert_array_equal(
        selected_last["pair"]["vector"],
        space.take(5)["pair"]["vector"],
    )


def test_finite_product_signature_covers_content_layout_and_dtype():
    first = phx.optim.FiniteProductSpace(
        (
            phx.optim.FiniteAxis(jnp.asarray([0.0, 1.0])),
            phx.optim.FiniteAxis(jnp.asarray([2.0, 3.0])),
        )
    )
    replay = phx.optim.FiniteProductSpace(
        (
            phx.optim.FiniteAxis(jnp.asarray([0.0, 1.0])),
            phx.optim.FiniteAxis(jnp.asarray([2.0, 3.0])),
        )
    )
    changed = phx.optim.FiniteProductSpace(
        (
            phx.optim.FiniteAxis(jnp.asarray([0.0, 1.0])),
            phx.optim.FiniteAxis(jnp.asarray([2.0, 4.0])),
        )
    )
    regrouped = phx.optim.FiniteProductSpace(
        phx.optim.FiniteAxis(
            {
                "left": jnp.asarray([0.0, 1.0]),
                "right": jnp.asarray([2.0, 3.0]),
            }
        )
    )

    assert first.signature() == replay.signature()
    assert first.signature() != changed.signature()
    assert first.signature() != regrouped.signature()


def test_finite_exhaustive_search_matches_dense_oracle_for_all_batch_layouts():
    space = phx.optim.FiniteProductSpace(
        (
            phx.optim.FiniteAxis(jnp.asarray([-1.0, 1.0])),
            phx.optim.FiniteAxis(jnp.asarray([0.0, 2.0, 4.0])),
        )
    )

    def evaluator(point):
        score = (point[0] - 1.0) ** 2 + (point[1] - 2.0) ** 2
        return score, jnp.asarray(True)

    for batch_size in (None, 1, 2, 4, 6, 20):
        search = phx.optim.FiniteExhaustiveSearch(batch_size)
        result = _exhaustive_minimum(evaluator, space, search)
        assert result.minimum == pytest.approx(0.0)
        assert int(result.flat_index) == 4
        assert tuple(int(index) for index in result.product_index) == (1, 1)
        assert int(result.attempted_evaluations) == 6
        assert int(result.invalid_evaluations) == 0


def test_finite_minimum_has_stable_ties_and_elementwise_indices():
    scalar_space = phx.optim.FiniteProductSpace(
        phx.optim.FiniteAxis(jnp.asarray([-2.0, -1.0, 1.0, 2.0]))
    )
    tied = _exhaustive_minimum(
        lambda value: (jnp.abs(value), jnp.asarray(True)),
        scalar_space,
        phx.optim.FiniteExhaustiveSearch(3),
    )
    assert int(tied.flat_index) == 1

    elementwise_space = phx.optim.FiniteProductSpace(
        phx.optim.FiniteAxis(jnp.asarray([-1.0, 0.0, 2.0]))
    )
    elementwise = _exhaustive_minimum(
        lambda value: (
            jnp.stack((value**2, (value - 2.0) ** 2)),
            jnp.ones((2,), dtype=bool),
        ),
        elementwise_space,
        phx.optim.FiniteExhaustiveSearch(2),
    )
    np.testing.assert_array_equal(elementwise.minimum, jnp.zeros((2,)))
    np.testing.assert_array_equal(elementwise.flat_index, jnp.asarray([1, 2]))
    np.testing.assert_array_equal(
        elementwise.product_index[0],
        jnp.asarray([1, 2]),
    )
    np.testing.assert_array_equal(
        elementwise.invalid_evaluations,
        jnp.zeros((2,), dtype=jnp.int64),
    )


def test_finite_minimum_counts_declared_and_nonfinite_invalidity():
    space = phx.optim.FiniteProductSpace(phx.optim.FiniteAxis(jnp.arange(5.0)))

    def evaluator(value):
        score = jnp.asarray([2.0, jnp.nan, -jnp.inf, 0.0, 1.0])[value.astype(int)]
        declared_valid = value != 3.0
        return score, declared_valid

    result = _exhaustive_minimum(
        evaluator,
        space,
        phx.optim.FiniteExhaustiveSearch(3),
    )
    assert result.minimum == pytest.approx(1.0)
    assert int(result.flat_index) == 4
    assert int(result.attempted_evaluations) == 5
    assert int(result.invalid_evaluations) == 3

    invalid = _exhaustive_minimum(
        lambda value: (jnp.asarray(jnp.nan), jnp.asarray(value >= 0.0)),
        space,
        phx.optim.FiniteExhaustiveSearch(2),
    )
    assert not bool(invalid.valid)
    assert jnp.isnan(invalid.minimum)
    assert int(invalid.flat_index) == -1
    assert int(invalid.product_index[0]) == -1
    assert int(invalid.attempted_evaluations) == 5
    assert int(invalid.invalid_evaluations) == 5


def test_finite_search_configuration_and_evaluator_contract_are_strict():
    assert phx.optim.FiniteExhaustiveSearch().effective_batch_size(5) == 1
    assert phx.optim.FiniteExhaustiveSearch(batch_size=10).effective_batch_size(5) == 5
    with pytest.raises(TypeError, match="positive integer"):
        phx.optim.FiniteExhaustiveSearch(True)
    with pytest.raises(TypeError, match="positive integer"):
        phx.optim.FiniteExhaustiveSearch(1.5)
    with pytest.raises(ValueError, match="positive"):
        phx.optim.FiniteExhaustiveSearch(0)

    space = phx.optim.FiniteProductSpace(phx.optim.FiniteAxis(jnp.asarray([0.0, 1.0])))
    with pytest.raises(TypeError, match="floating-point"):
        _exhaustive_minimum(
            lambda value: (value.astype(int), jnp.asarray(True)),
            space,
            phx.optim.FiniteExhaustiveSearch(),
        )
    with pytest.raises(TypeError, match="boolean"):
        _exhaustive_minimum(
            lambda value: (value, jnp.asarray(1)),
            space,
            phx.optim.FiniteExhaustiveSearch(),
        )
    with pytest.raises(ValueError, match="shapes must match"):
        _exhaustive_minimum(
            lambda value: (jnp.stack((value, value)), jnp.asarray(True)),
            space,
            phx.optim.FiniteExhaustiveSearch(),
        )


def test_finite_space_cardinality_overflow_is_rejected():
    huge = object.__new__(phx.optim.FiniteAxis)
    object.__setattr__(huge, "size", 2**32)
    with pytest.raises(OverflowError, match="64-bit"):
        phx.optim.FiniteProductSpace((huge, huge))
