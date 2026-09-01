#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _mesh():
    return phx.discretization.CellMesh.from_triangles(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        jnp.asarray(((0, 1, 2),), dtype=jnp.int32),
    )


def _discretization(*names):
    element = phx.discretization.lagrange_element("triangle", 1)
    return phx.discretization.FiniteElementPlan(
        _mesh(),
        tuple(phx.discretization.FiniteElementFieldSpec(name, element) for name in names),
    ).prepare()


def test_cell_functional_residual_is_discrete_first_variation():
    discretization = _discretization("u")

    def density(fields, geometry, context):
        del geometry, context
        return 0.5 * jnp.sum(fields["u"].gradient ** 2, axis=-1)

    functional = phx.variational.Functional(
        "dirichlet-energy",
        (
            phx.variational.LocalIntegralTerm(
                "body",
                region="body",
                fields=(phx.variational.FieldJetSpec("u", gradient=True),),
                density=density,
                density_id="gradient-energy",
            ),
        ),
        variable_fields=("u",),
    )
    compiled = phx.equations.compile_finite_element_functional(
        functional,
        discretization,
        fields={"u": "u"},
        regions={"body": None},
    )
    state = jnp.asarray((0.2, -0.1, 0.4))

    np.testing.assert_allclose(
        compiled.residual(state),
        jax.grad(compiled.potential)(state),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    value, residual = compiled.value_and_residual(state)
    np.testing.assert_allclose(value, compiled.potential(state), atol=2.0e-12)
    np.testing.assert_allclose(
        residual,
        compiled.residual(state),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    assert compiled.potential_evaluation(state).term_values[0].shape == ()


def test_mixed_functional_varies_all_fields_without_double_counting_value():
    discretization = _discretization("u", "v")

    def density(fields, geometry, context):
        del geometry, context
        difference = fields["u"].value - fields["v"].value
        return 0.5 * difference**2

    functional = phx.variational.Functional(
        "coupled",
        (
            phx.variational.LocalIntegralTerm(
                "coupling",
                region="body",
                fields=(
                    phx.variational.FieldJetSpec("u", value=True),
                    phx.variational.FieldJetSpec("v", value=True),
                ),
                density=density,
                density_id="quadratic-coupling",
            ),
        ),
        variable_fields=("u", "v"),
    )
    compiled = phx.equations.compile_finite_element_functional(
        functional,
        discretization,
        fields={"u": "u", "v": "v"},
        regions={"body": None},
    )
    state = (jnp.asarray((0.2, 0.5, -0.1)), jnp.asarray((-0.3, 0.1, 0.4)))
    reference = jax.grad(compiled.potential)(state)
    residual = compiled.residual(state)

    assert compiled.block_dependency_graph() == ((True, True), (True, True))
    np.testing.assert_allclose(residual[0], reference[0], rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(residual[1], reference[1], rtol=2.0e-12, atol=2.0e-12)
    evaluation = compiled.potential_evaluation(state)
    np.testing.assert_allclose(evaluation.value, evaluation.term_values[0])


def test_exterior_functional_value_and_residual_use_same_boundary_measure():
    discretization = _discretization("u")

    def density(fields, geometry, context):
        del context
        if geometry.normal is None:
            raise ValueError("Boundary normal was not provided.")
        normal_norm = jnp.sum(geometry.normal**2, axis=-1)
        return -fields["u"].value + 0.0 * normal_norm

    functional = phx.variational.Functional(
        "boundary-work",
        (
            phx.variational.LocalIntegralTerm(
                "boundary",
                region="boundary",
                fields=(phx.variational.FieldJetSpec("u", value=True),),
                density=density,
                density_id="negative-trace",
                normal=True,
            ),
        ),
        variable_fields=("u",),
    )
    compiled = phx.equations.compile_finite_element_functional(
        functional,
        discretization,
        fields={"u": "u"},
        regions={"boundary": discretization.exterior_facet_domain},
    )
    state = jnp.ones((3,))
    perimeter = 2.0 + jnp.sqrt(2.0)

    np.testing.assert_allclose(compiled.potential(state), -perimeter, atol=2.0e-12)
    np.testing.assert_allclose(
        compiled.residual(state),
        jax.grad(compiled.potential)(state),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
