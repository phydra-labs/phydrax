#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.nn import ModelBinding


def test_flat_model_binding_packs_scalar_and_vector_points():
    binding = ModelBinding.pointwise("flat")
    packed = binding.pack_point((jnp.asarray(2.0), jnp.asarray([3.0, 4.0])))

    assert packed.shape == (3,)
    assert jnp.allclose(packed, jnp.asarray([2.0, 3.0, 4.0]))


def test_flat_model_binding_preserves_shared_batch_shape():
    binding = ModelBinding.pointwise("flat")
    scalar = jnp.asarray([1.0, 2.0, 3.0])
    vector = jnp.asarray([[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]])
    packed = binding.pack_point((scalar, vector))

    assert packed.shape == (3, 3)
    assert jnp.allclose(packed[:, 0], scalar)
    assert jnp.allclose(packed[:, 1:], vector)


def test_structured_model_binding_preserves_coordinate_parts():
    binding = ModelBinding.pointwise("structured")
    packed = binding.pack_point(
        ((jnp.asarray(1.0), jnp.asarray(2.0)), jnp.asarray([3.0, 4.0]))
    )

    assert isinstance(packed, tuple)
    assert len(packed) == 3
    assert jnp.allclose(packed[0], 1.0)
    assert jnp.allclose(packed[1], 2.0)
    assert jnp.allclose(packed[2], jnp.asarray([3.0, 4.0]))


def test_flat_model_binding_rejects_coord_separable_tuple():
    binding = ModelBinding.pointwise("flat")
    with pytest.raises(ValueError, match="cannot pack tuple inputs"):
        binding.pack_point(((jnp.asarray([1.0]), jnp.asarray([2.0])),))
