#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr

from phydrax._trainable import partition_trainable
from phydrax.ml.quantum import (
    data_reuploading_feature_map,
    projected_iqp_feature_map,
)
from phydrax.operators.quantum import HilbertRegisterLayout


def test_projected_iqp_feature_map_returns_ordered_real_features():
    layout = HilbertRegisterLayout(("a", "b"), (2, 2))
    model = projected_iqp_feature_map(
        layout,
        repetitions=1,
        entanglement_edges=(("a", "b"),),
    )
    point = jnp.asarray([0.2, -0.4], dtype=jnp.float64)
    values = jax.jit(model)(point)

    assert values.shape == (6,)
    assert jnp.issubdtype(values.dtype, jnp.floating)
    assert jnp.all(jnp.isfinite(values))


def test_parameter_shift_model_matches_autodiff_primal_and_input_jacobian():
    layout = HilbertRegisterLayout(("a", "b"), (2, 2))
    key = jr.key(7)
    kwargs = {
        "entanglement_edges": (("a", "b"),),
        "readout_wire_ids": ("a", "b"),
    }
    autodiff_model = data_reuploading_feature_map(
        2,
        layout,
        1,
        key,
        gradient_method="autodiff",
        **kwargs,
    )
    shift_model = data_reuploading_feature_map(
        2,
        layout,
        1,
        key,
        gradient_method="parameter-shift",
        **kwargs,
    )
    point = jnp.asarray([0.15, -0.31], dtype=jnp.float64)

    assert jnp.allclose(autodiff_model(point), shift_model(point))
    assert jnp.allclose(
        jax.jacrev(autodiff_model)(point),
        jax.jacrev(shift_model)(point),
        atol=1e-8,
    )


def test_prepared_circuit_execution_is_not_trainable():
    layout = HilbertRegisterLayout(("q",), (2,))
    model = data_reuploading_feature_map(1, layout, 1, jr.key(3))
    trainable, _fixed = partition_trainable(model)
    leaves = jax.tree.leaves(trainable)

    assert len(leaves) == 2
    assert leaves[0].shape == (3,)
    assert leaves[1].shape == (3,)
