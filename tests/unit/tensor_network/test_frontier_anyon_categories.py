#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.tensor_network._anyon import (
    AnyonicTensor,
    AnyonicTensorBlock,
    contract_anyonic_tensors,
    fibonacci_fusion_category,
    prepare_string_net_hamiltonian,
    StringNetPlan,
    z2_fusion_category,
)


jax.config.update("jax_enable_x64", True)


def test_frontier_fusion_categories_satisfy_pentagon_and_hexagon():
    for category in (z2_fusion_category(), fibonacci_fusion_category()):
        assert bool(category.coherence.coherent)
        assert float(category.coherence.f_unitarity_residual) < 5e-9
        assert float(category.coherence.r_unitarity_residual) < 5e-9
        assert float(category.coherence.pentagon_residual) < 5e-9
        assert float(category.coherence.hexagon_residual) < 5e-9
        assert category.coherence.claim == "finite-explicit-f-r-data-only"


def test_frontier_string_net_plaquette_is_a_commuting_projector():
    category = z2_fusion_category()
    prepared = prepare_string_net_hamiltonian(
        StringNetPlan(
            category,
            6,
            ((0, 1, 2), (3, 4, 5)),
            (
                ((0, 1), (1, 1)),
                ((3, 1), (4, 1)),
            ),
            maximum_hilbert_dimension=128,
        )
    )
    identity = jnp.eye(prepared.plan.hilbert_dimension, dtype=jnp.complex128)
    all_projectors = tuple(prepared.vertex_projectors) + tuple(
        prepared.plaquette_projectors
    )
    for projector in all_projectors:
        np.testing.assert_allclose(projector @ projector, projector, atol=1e-12)
        np.testing.assert_allclose(projector.conj().T, projector, atol=1e-12)
    np.testing.assert_allclose(
        prepared.plaquette_projectors[0] @ prepared.plaquette_projectors[1],
        prepared.plaquette_projectors[1] @ prepared.plaquette_projectors[0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        prepared.hamiltonian,
        4.0 * identity
        - jnp.sum(prepared.vertex_projectors, axis=0)
        - jnp.sum(prepared.plaquette_projectors, axis=0),
        atol=1e-12,
    )
    assert bool(prepared.evidence.projector_identities)
    assert "reference-only" in prepared.evidence.claim


def test_frontier_anyonic_contraction_enforces_dual_charge_and_quantum_trace():
    category = z2_fusion_category()
    left = AnyonicTensor(
        category,
        (1,),
        (AnyonicTensorBlock(("s",), jnp.asarray((2.0, 3.0))),),
    )
    right = AnyonicTensor(
        category,
        (-1,),
        (AnyonicTensorBlock(("s",), jnp.asarray((5.0, 7.0))),),
    )
    scalar = jax.jit(lambda first, second: contract_anyonic_tensors(first, second, 0, 0))(
        left, right
    )
    assert scalar.orientations == ()
    assert scalar.blocks[0].charges == ()
    np.testing.assert_allclose(scalar.blocks[0].data, 31.0)
