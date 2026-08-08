#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._interpolation import BSplineGrid
from phydrax.control import (
    BSplineControlParameterization,
    ControlTimeGrid,
    PiecewiseConstantControlParameterization,
    PiecewiseLinearControlParameterization,
)


def test_piecewise_parameterizations_use_physical_time_and_exact_shapes():
    time_grid = ControlTimeGrid(
        jnp.asarray([0.0, 0.5, 1.0]),
        time_id="physical-time",
    )
    constant = PiecewiseConstantControlParameterization(
        time_grid,
        (1,),
        parameterization_id="constant",
    )
    linear = PiecewiseLinearControlParameterization(
        time_grid,
        (1,),
        parameterization_id="linear",
    )
    query = jnp.asarray([0.25, 0.75, 1.0])

    constant_values = constant.evaluate(jnp.asarray([[1.0], [2.0]]), query)
    linear_values = linear.evaluate(jnp.asarray([[0.0], [1.0], [0.0]]), query)

    assert constant.parameter_shape == (2, 1)
    assert linear.parameter_shape == (3, 1)
    assert np.allclose(np.asarray(constant_values[:, 0]), [1.0, 2.0, 2.0])
    assert np.allclose(np.asarray(linear_values[:, 0]), [0.5, 0.5, 0.0])
    with pytest.raises(ValueError, match="coefficients must have shape"):
        constant.evaluate(jnp.ones((3, 1)), query)
    with pytest.raises(eqx.EquinoxRuntimeError, match="outside its physical grid"):
        value = linear.evaluate(jnp.ones((3, 1)), jnp.asarray(1.1))
        jax.block_until_ready(value)


def test_fixed_grid_bspline_is_differentiable_and_certifies_coefficient_bounds():
    grid = BSplineGrid.open_uniform(3, 4, interval=(0.0, 1.0))
    parameterization = BSplineControlParameterization(
        grid,
        (2,),
        parameterization_id="spline-control",
    )
    coefficients = jnp.stack(
        (
            jnp.linspace(-0.8, 0.8, grid.coefficient_count),
            jnp.linspace(0.2, 1.2, grid.coefficient_count),
        ),
        axis=-1,
    )
    query = jnp.linspace(0.0, 1.0, 17)

    values = parameterization.evaluate(coefficients, query)
    gradient = jax.grad(
        lambda control_coefficients: jnp.sum(
            parameterization.evaluate(control_coefficients, query) ** 2
        )
    )(coefficients)
    certificate = parameterization.bound_certificate(
        coefficients,
        jnp.asarray([-1.0, 0.0]),
        jnp.asarray([1.0, 1.5]),
    )
    failed_certificate = parameterization.bound_certificate(
        coefficients,
        jnp.asarray([-0.5, 0.0]),
        jnp.asarray([1.0, 1.5]),
    )

    assert values.shape == (17, 2)
    assert gradient.shape == coefficients.shape
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert bool(certificate.certified)
    assert certificate.continuous_domain
    assert certificate.certificate_id == "control-bound:bspline-convex-hull"
    assert not bool(failed_certificate.certified)


def test_bspline_refinement_uses_canonical_diagnosed_grid_transfer():
    old_grid = BSplineGrid.open_uniform(3, 3, interval=(0.0, 1.0))
    new_grid = BSplineGrid(
        jnp.sort(jnp.concatenate((old_grid.knots, jnp.asarray([0.2, 0.8])))),
        old_grid.degree,
    )
    parameterization = BSplineControlParameterization(
        old_grid,
        (1,),
        parameterization_id="coarse",
    )
    coefficients = jnp.sin(jnp.arange(old_grid.coefficient_count, dtype=float))[:, None]

    refinement = parameterization.refine(
        new_grid,
        coefficients,
        parameterization_id="refined",
        method="exact",
    )
    query = jnp.linspace(0.0, 1.0, 101)
    old_values = parameterization.evaluate(coefficients, query)
    new_values = refinement.parameterization.evaluate(refinement.coefficients, query)

    assert refinement.transfer.method == "exact"
    assert refinement.transfer.projection_error_bound == 0.0
    assert refinement.source_parameterization_id == "coarse"
    assert refinement.target_parameterization_id == "refined"
    assert refinement.coefficients.shape == (new_grid.coefficient_count, 1)
    assert np.allclose(np.asarray(old_values), np.asarray(new_values), atol=2e-12)
