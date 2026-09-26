#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp

from phydrax.linalg import DenseLinearOperator, LinearSystem, prepare
from phydrax.operators.integral._convolution_quadrature import (
    ConvolutionQuadratureContourPolicy,
)
from phydrax.solver._dynamic_vector_cq import (
    prepare_dynamic_elasticity_fem_bem_cq_3d,
    prepare_dynamic_maxwell_fem_bem_cq_3d,
    PreparedDynamicElasticityFEMBEM3D,
    PreparedDynamicMaxwellFEMBEM3D,
)


def _node_family(parameter: jax.Array, action: str):
    base = jnp.asarray(
        [[parameter + 0.7, 0.3 + 0.2j], [0.1 - 0.4j, parameter + 1.1]],
        dtype=jnp.complex128,
    )
    if action == "transpose":
        base = base.T
    elif action == "adjoint":
        base = jnp.conj(base.T)
    return prepare(LinearSystem(DenseLinearOperator(base)))


def test_vector_cq_transpose_and_adjoint_are_dual_to_forward_history_map():
    with jax.enable_x64():
        options = dict(
            node_family_id="test-vector-resolvent",
            fft_length=16,
            contour_policy=ConvolutionQuadratureContourPolicy(tolerance=1.0e-14),
        )
        elasticity = prepare_dynamic_elasticity_fem_bem_cq_3d(
            _node_family, 2, 0.1, 5, **options
        )
        maxwell = prepare_dynamic_maxwell_fem_bem_cq_3d(
            _node_family, 2, 0.1, 5, **options
        )
        source = jnp.asarray(
            [[1.0, 0.2j], [-0.3, 0.5], [0.4j, -0.1], [0.0, 0.7], [0.6, -0.2j]],
            dtype=jnp.complex128,
        )
        probe = jnp.asarray(
            [[0.3, -0.5j], [0.8, 0.1], [-0.2, 0.9j], [0.4j, 0.0], [1.0, 0.25]],
            dtype=jnp.complex128,
        )
        elasticity_forward = elasticity.apply(source)
        transposed = elasticity.transpose_apply(probe)
        maxwell_forward = maxwell.apply(source)
        adjoint = maxwell.adjoint_apply(probe)

    assert isinstance(elasticity, PreparedDynamicElasticityFEMBEM3D)
    assert isinstance(maxwell, PreparedDynamicMaxwellFEMBEM3D)
    assert transposed.action == "transpose"
    assert adjoint.action == "adjoint"
    assert bool(transposed.successful)
    assert bool(adjoint.successful)
    assert jnp.allclose(
        jnp.sum(elasticity_forward.value * probe),
        jnp.sum(source * transposed.value),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    assert jnp.allclose(
        jnp.vdot(probe, maxwell_forward.value),
        jnp.vdot(adjoint.value, source),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
