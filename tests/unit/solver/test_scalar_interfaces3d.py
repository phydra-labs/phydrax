import jax.numpy as jnp

import phydrax as phx
from phydrax.operators.integral.layer_potential._scalar_interfaces3d import (
    ScalarTransmissionData3D,
    ScalarTransmissionMaterial3D,
)


def test_matching_laplace_transmission_recovers_manufactured_cauchy_blocks():
    vertices = jnp.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = jnp.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=jnp.int32)
    region = phx.geometry.MeshRegion(vertices, faces)
    policy = phx.operators.LaplaceSingleLayerDP0GalerkinPolicy3D(
        singular_order=3,
        absolute_tolerance=1.0e-3,
        relative_tolerance=1.0e-3,
    )
    minus_calderon = phx.operators.prepare_scalar_calderon_dp0_3d(region, policy=policy)
    plus_calderon = phx.operators.prepare_scalar_calderon_dp0_3d(region, policy=policy)
    minus = ScalarTransmissionMaterial3D("inside", minus_calderon, flux_coefficient=1.0)
    plus = ScalarTransmissionMaterial3D("outside", plus_calderon, flux_coefficient=2.0)
    prepared = phx.solver.prepare_scalar_transmission_3d(minus, plus)
    expected = tuple(jnp.linspace(0.1, 0.4, 4) for _ in range(4))
    right_hand_side = prepared.formulation.operator.mv(expected)

    result = phx.solver.solve_scalar_transmission_3d(
        prepared, ScalarTransmissionData3D(*right_hand_side)
    )

    assert bool(result.valid)
    assert result.relative_block_residual < 1.0e-10
    assert jnp.linalg.norm(result.dirichlet_continuity_defect) < 1.0e-10
    assert jnp.linalg.norm(result.weighted_flux_continuity_defect) < 1.0e-10
    trial = tuple(jnp.linspace(-0.3, 0.2, 4) for _ in range(4))
    dual = tuple(jnp.linspace(0.5, -0.1, 4) for _ in range(4))
    forward = prepared.formulation.operator.mv(trial)
    transposed = prepared.formulation.operator.transpose_mv(dual)
    left = sum(jnp.vdot(y, value) for y, value in zip(dual, forward, strict=True))
    right = sum(jnp.vdot(value, x) for value, x in zip(transposed, trial, strict=True))
    assert jnp.allclose(left, right, rtol=1.0e-10, atol=1.0e-10)
