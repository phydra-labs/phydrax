#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _grid_2d(points: Any = 17) -> Any:
    return phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(points),
            phx.discretization.UniformAxisSpec(points),
        ),
        axis_names=("xi", "eta"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))


def _affine_map(reference: Any) -> Any:
    xi, eta = reference
    return jnp.asarray((2.0 * xi + 0.2 * eta, 0.1 * xi + 1.5 * eta))


def _warped_map(reference: Any) -> Any:
    xi, eta = reference
    warp = jnp.sin(jnp.pi * xi) * jnp.sin(jnp.pi * eta)
    return jnp.asarray((xi + 0.05 * warp, eta + 0.03 * warp))


def _warped_map_3d(reference: Any) -> Any:
    xi, eta, zeta = reference
    warp = jnp.sin(jnp.pi * xi) * jnp.sin(jnp.pi * eta) * jnp.sin(jnp.pi * zeta)
    return jnp.asarray((xi + 0.02 * warp, eta - 0.015 * warp, zeta + 0.01 * warp))


def test_mapped_tensor_fd_scenario_1() -> None:
    grid = _grid_2d()
    mapped = phx.discretization.MappedTensorGridPlan(
        grid,
        _affine_map,
        sbp_order=4,
    ).prepare()
    physical_x = mapped.physical_coordinates[..., 0]
    physical_y = mapped.physical_coordinates[..., 1]
    scalar = physical_x**2 + 3.0 * physical_y

    gradient = mapped.gradient(scalar)
    volume = mapped.integral(jnp.ones(mapped.shape))

    np.testing.assert_allclose(
        gradient[..., 0],
        2.0 * physical_x,
        rtol=2e-10,
        atol=2e-10,
    )
    np.testing.assert_allclose(gradient[..., 1], 3.0, rtol=0.0, atol=2e-10)
    np.testing.assert_allclose(volume, 2.98, rtol=2e-11, atol=2e-11)
    assert mapped.metric_report.passed
    assert mapped.dual_face_layouts[0].shape == (16, 17)
    assert mapped.dual_face_layouts[1].shape == (17, 16)
    assert mapped.face_normals[0].shape == (16, 17, 2)
    assert mapped.face_measures[1].shape == (17, 16)
    mapped = phx.discretization.MappedTensorGridPlan(
        _grid_2d(21),
        _warped_map,
        sbp_order=4,
    ).prepare()
    constant = jnp.ones(mapped.shape)

    gradient = mapped.gradient(constant)
    divergence = mapped.divergence(jnp.ones(mapped.shape + (2,)))

    assert mapped.metric_report.metric_identity_residual < 5e-10
    assert mapped.metric_report.free_stream_residual < 5e-10
    np.testing.assert_allclose(gradient, 0.0, rtol=0.0, atol=5e-11)
    np.testing.assert_allclose(divergence, 0.0, rtol=0.0, atol=5e-10)
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(13),
            phx.discretization.UniformAxisSpec(13),
            phx.discretization.UniformAxisSpec(13),
        ),
        axis_names=("xi", "eta", "zeta"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    mapped = phx.discretization.MappedTensorGridPlan(
        grid,
        _warped_map_3d,
        sbp_order=4,
    ).prepare()

    assert mapped.metric_report.passed
    assert mapped.metric_report.metric_identity_residual < 2e-9
    assert mapped.metric_report.minimum_jacobian > 0.8


def test_mapped_tensor_fd_scenario_2() -> None:
    grid = _grid_2d(17)
    mapped = phx.discretization.MappedTensorGridPlan(
        grid,
        lambda reference: reference,
        sbp_order=4,
    ).prepare()
    x = mapped.physical_coordinates[..., 0]
    y = mapped.physical_coordinates[..., 1]
    state = x**2 + y**2
    operator = mapped.diffusion(jnp.asarray([[2.0, 0.0], [0.0, 3.0]]))

    action = operator.mv(state)

    np.testing.assert_allclose(action, 10.0, rtol=2e-9, atol=2e-9)
    assert operator.conservation_report.constant_state_residual < 1e-10
    mapped = phx.discretization.MappedTensorGridPlan(
        _grid_2d(13),
        _warped_map,
        sbp_order=4,
    ).prepare()
    state = jnp.sin(mapped.physical_coordinates[..., 0])

    gradient = jax.grad(lambda value: mapped.integral(mapped.laplacian(value) ** 2))(
        state
    )

    assert gradient.shape == mapped.shape
    assert jnp.all(jnp.isfinite(gradient))
    with pytest.raises(eqx.EquinoxRuntimeError, match="Jacobian"):
        phx.discretization.MappedTensorGridPlan(
            _grid_2d(),
            lambda reference: jnp.asarray((-reference[0], reference[1])),
            sbp_order=4,
        ).prepare()
