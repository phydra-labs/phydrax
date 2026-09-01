#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.discretization import iga


def test_isogeometric_poisson_exact_quadratic_solution():
    degree = 2
    grid = iga.BSplineGrid.open_uniform(
        degree,
        1,
        interval=(0.0, 1.0),
    )
    coordinates = grid.greville_abscissae
    xx, yy = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    geometry = iga.NURBSGeometryState(
        jnp.stack((xx, yy), axis=-1),
        jnp.ones((grid.coefficient_count, grid.coefficient_count)),
    )
    prepared = iga.IsogeometricPlan.isoparametric(
        (grid, grid),
        geometry,
        field_name="u",
        axis_names=("xi", "eta"),
        quadrature_policy=iga.IsogeometricQuadraturePolicy(degree + 1),
    ).prepare(numeric_version="poisson-test")
    source = phx.equations.coefficient(
        lambda points, args: (
            2.0
            * (
                points[..., 0] * (1.0 - points[..., 0])
                + points[..., 1] * (1.0 - points[..., 1])
            )
        ),
        coefficient_id="iga-poisson-test-source",
    )
    form = phx.equations.FiniteElementForm(
        "iga-poisson-test",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction("u", source),
        ),
    )
    compiled = phx.equations.compile_finite_element_problem(
        form,
        prepared,
        constraint=prepared.homogeneous_trace_constraint("u"),
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free",
            local_kernel="sum_factorized",
        ),
    )

    operator, right_hand_side = compiled.linear_system()
    result = phx.linalg.solve(operator, right_hand_side)
    solution = compiled.expand(result.value)
    expected_line = jnp.asarray((0.0, 0.5, 0.0), dtype=solution.dtype)
    expected = jnp.outer(expected_line, expected_line)
    residual = compiled.residual(result.value)

    assert bool(jnp.all(result.successful))
    np.testing.assert_allclose(solution, expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(residual, 0.0, rtol=1e-11, atol=1e-12)


def test_isogeometric_natural_load_uses_physical_boundary_measure():
    grid = iga.BSplineGrid.open_uniform(2, 1, interval=(0.0, 1.0))
    coordinates = grid.greville_abscissae
    xx, yy = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    prepared = iga.IsogeometricPlan.isoparametric(
        (grid, grid),
        iga.NURBSGeometryState(
            jnp.stack((xx, yy), axis=-1),
            jnp.ones((grid.coefficient_count, grid.coefficient_count)),
        ),
        field_name="u",
        axis_names=("xi", "eta"),
        quadrature_policy=iga.IsogeometricQuadraturePolicy(3),
    ).prepare(numeric_version="boundary-load-test")
    compiled = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "iga-natural-load",
            "u",
            (phx.equations.BoundaryLoadAction("u", 1.0),),
        ),
        prepared,
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free",
            local_kernel="sum_factorized",
        ),
    )

    residual = compiled.residual(compiled.state_space.zeros())

    np.testing.assert_allclose(jnp.sum(residual), -4.0, rtol=1e-12, atol=1e-12)

    def boundary_density(fields, geometry, context):
        del context
        if geometry.normal is None:
            raise ValueError("Prepared boundary functional requires normals.")
        return -fields["u"].value + 0.0 * jnp.sum(
            geometry.normal**2,
            axis=-1,
        )

    portable = phx.variational.Functional(
        "iga-portable-boundary-work",
        (
            phx.variational.LocalIntegralTerm(
                "boundary",
                region="boundary",
                fields=(phx.variational.FieldJetSpec("u", value=True),),
                density=boundary_density,
                density_id="iga-portable-boundary-work",
                normal=True,
            ),
        ),
        variable_fields=("u",),
    )
    portable_compiled = phx.equations.compile_finite_element_functional(
        portable,
        prepared,
        fields={"u": "u"},
        regions={
            "boundary": prepared.integration_domain("exterior_facet"),
        },
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free",
            local_kernel="sum_factorized",
        ),
    )
    portable_residual = portable_compiled.residual(portable_compiled.state_space.zeros())

    np.testing.assert_allclose(
        jnp.sum(portable_residual),
        -4.0,
        rtol=1e-12,
        atol=1e-12,
    )


def test_isogeometric_functional_flattens_basis_coefficients():
    grid = iga.BSplineGrid.open_uniform(2, 1, interval=(0.0, 1.0))
    coordinates = grid.greville_abscissae
    xx, yy = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    prepared = iga.IsogeometricPlan.isoparametric(
        (grid, grid),
        iga.NURBSGeometryState(
            jnp.stack((xx, yy), axis=-1),
            jnp.ones((grid.coefficient_count, grid.coefficient_count)),
        ),
        field_name="u",
        axis_names=("xi", "eta"),
        quadrature_policy=iga.IsogeometricQuadraturePolicy(3),
    ).prepare(numeric_version="functional-test")
    functional = phx.equations.FiniteElementFunctional(
        "iga-l2-x",
        "u",
        lambda values, gradients, points, context: values**2,
    )

    value = functional.evaluate(prepared, xx)

    np.testing.assert_allclose(value, 1.0 / 3.0, rtol=1e-12, atol=1e-12)

    portable = phx.variational.Functional(
        "iga-portable-l2",
        (
            phx.variational.LocalIntegralTerm(
                "body",
                region="body",
                fields=(phx.variational.FieldJetSpec("u", value=True),),
                density=lambda fields, geometry, context: fields["u"].value ** 2,
                density_id="iga-portable-square",
            ),
        ),
        variable_fields=("u",),
    )
    compiled = phx.equations.compile_finite_element_functional(
        portable,
        prepared,
        fields={"u": "u"},
        regions={"body": None},
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free",
            local_kernel="sum_factorized",
        ),
    )
    portable_value, portable_residual = compiled.value_and_residual(xx)

    np.testing.assert_allclose(portable_value, value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        portable_residual,
        jax.grad(compiled.potential)(xx),
        rtol=1e-12,
        atol=1e-12,
    )
