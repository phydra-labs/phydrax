#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

from phydrax.kernels import ExactQuantumStateFidelityKernel
from phydrax.ml.quantum import iqp_state_feature_map
from phydrax.operators.quantum import HilbertRegisterLayout


def test_exact_quantum_fidelity_kernel_is_psd_unit_diagonal_and_phase_invariant():
    layout = HilbertRegisterLayout(("a", "b"), (2, 2))
    feature_map = iqp_state_feature_map(
        layout,
        repetitions=1,
        entanglement_edges=(("a", "b"),),
    )
    kernel = ExactQuantumStateFidelityKernel(
        feature_map,
        feature_map.model_id,
    )
    points = jnp.asarray(
        [[0.1, -0.2], [0.4, 0.3], [-0.5, 0.7]],
        dtype=jnp.float64,
    )
    gram = kernel.matrix(points, points)

    assert jnp.allclose(gram, jnp.swapaxes(gram, -1, -2))
    assert jnp.allclose(kernel.diagonal(points), jnp.ones((3,)))
    assert jnp.min(jnp.linalg.eigvalsh(gram)) >= -1e-10
    assert jnp.allclose(kernel.pairwise(points[0], points[1]), gram[0, 1])


def test_exact_quantum_fidelity_cross_gram_matches_pairwise_evaluation():
    layout = HilbertRegisterLayout(("q",), (2,))
    feature_map = iqp_state_feature_map(layout)
    kernel = ExactQuantumStateFidelityKernel(feature_map, feature_map.model_id)
    left = jnp.asarray([[0.0], [0.3]], dtype=jnp.float64)
    right = jnp.asarray([[-0.2], [0.8], [1.1]], dtype=jnp.float64)
    matrix = kernel.matrix(left, right)
    reference = jnp.stack(
        tuple(jnp.stack(tuple(kernel.pairwise(x, y) for y in right)) for x in left)
    )

    assert matrix.shape == (2, 3)
    assert jnp.allclose(matrix, reference)
