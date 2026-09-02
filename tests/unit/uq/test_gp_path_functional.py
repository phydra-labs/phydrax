# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def test_path_value_and_coordinate_functionals_use_signature_kernel():
    paths = jnp.asarray(
        [
            [[0.0], [0.2], [0.6]],
            [[0.0], [0.4], [0.7]],
        ]
    )
    kernel = phx.kernels.SignaturePDEKernel(
        phx.kernels.SquaredExponentialKernel(length_scale=0.8),
        polynomial_order=2,
    )
    value = phx.uq.path_value_functional((3, 1))
    derivative = phx.uq.path_partial_derivative_functional((3, 1), 1, 0)
    value_design = phx.uq.FunctionalDesign(
        (phx.uq.FunctionalObservationBlock(paths, value, name="values"),)
    )
    derivative_design = phx.uq.FunctionalDesign(
        (
            phx.uq.FunctionalObservationBlock(
                paths,
                derivative,
                name="coordinate-derivatives",
                valid_knot_count=3,
            ),
        )
    )
    assert jnp.allclose(
        phx.uq.functional_kernel_matrix(kernel, value_design, value_design),
        kernel.matrix(paths, paths),
    )
    covariance = phx.uq.functional_kernel_matrix(kernel, derivative_design, value_design)
    oracle = jax.jacrev(lambda path: kernel.matrix(path[None], paths)[0])(paths[0])
    assert jnp.allclose(covariance[0], oracle[:, 1, 0], atol=2e-4)


def test_path_blocks_may_vary_knot_count_and_padding_derivatives_fail():
    kernel = phx.kernels.SignaturePDEKernel(phx.kernels.SquaredExponentialKernel())
    short = phx.uq.FunctionalObservationBlock(
        jnp.zeros((1, 2, 1)),
        phx.uq.path_value_functional((2, 1)),
        name="short",
    )
    long = phx.uq.FunctionalObservationBlock(
        jnp.zeros((1, 4, 1)),
        phx.uq.path_value_functional((4, 1)),
        name="long",
    )
    design = phx.uq.FunctionalDesign((short, long))
    assert phx.uq.functional_kernel_matrix(kernel, design, design).shape == (2, 2)
    with pytest.raises(ValueError, match="inactive repeat padding"):
        phx.uq.FunctionalObservationBlock(
            jnp.zeros((1, 4, 1)),
            phx.uq.path_partial_derivative_functional((4, 1), 3, 0),
            name="invalid-padding",
            valid_knot_count=3,
        )
