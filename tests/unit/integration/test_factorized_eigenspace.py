#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
import pytest

import phydrax as phx


def _batch(*, mask=None):
    weights = jnp.asarray([0.5, 0.5])
    return phx.integration.SeparableIntegrationBatch(
        {},
        {
            "x": cx.Field(weights, dims=("x",)),
            "y": cx.Field(weights, dims=("y",)),
        },
        mask=mask,
    )


def _field(x_values, y_values):
    x = phx.integration.AxisFactor(
        "x",
        jnp.asarray(x_values).reshape((2, 1, 2)),
        ("x",),
    )
    y = phx.integration.AxisFactor(
        "y",
        jnp.asarray(y_values).reshape((2, 1, 2)),
        ("y",),
    )
    return phx.integration.AxisFactorizedField(
        (x, y),
        phx.integration.AxisContractionPlan(
            (phx.integration.AxisProductTerm(("x", "y")),)
        ),
    )


def test_factorized_form_matches_dense_contraction_without_full_grid():
    field = _field(
        [[0.0, 1.0], [1.0, 1.0]],
        [[1.0, 0.0], [1.0, 1.0]],
    )
    batch = _batch()

    evaluation = phx.integration.factorized_inner_product(field, field, batch)
    dense = field.contract().data
    expected = oe.contract(
        "xyi,xyj,x,y->ij",
        jnp.conj(dense),
        dense,
        batch.weights_by_axis["x"].data,
        batch.weights_by_axis["y"].data,
    )

    assert bool(evaluation.valid)
    assert evaluation.avoided_full_materialization
    assert evaluation.full_point_count == 4
    assert evaluation.maximum_local_point_count == 2
    assert jnp.allclose(evaluation.value, expected, atol=2e-12)


def test_factorized_variational_eigenspace_assembles_gradient_energy():
    field = _field(
        [[0.0, 1.0], [1.0, 1.0]],
        [[1.0, 0.0], [1.0, 1.0]],
    )
    derivative_x = _field(
        [[1.0, 0.0], [1.0, 0.0]],
        [[1.0, 1.0], [1.0, 1.0]],
    )
    derivative_y = _field(
        [[1.0, 1.0], [1.0, 1.0]],
        [[0.0, 1.0], [0.0, 1.0]],
    )
    batch = _batch()
    mass = phx.integration.FactorizedBilinearTerm(field, field)
    stiffness = (
        phx.integration.FactorizedBilinearTerm(derivative_x, derivative_x),
        phx.integration.FactorizedBilinearTerm(derivative_y, derivative_y),
    )

    result = phx.terms.factorized_variational_eigenspace(
        stiffness,
        (mass,),
        batch,
    )

    assert bool(result.successful)
    assert jnp.allclose(result.stiffness.value, jnp.eye(2), atol=2e-12)
    assert jnp.allclose(
        result.mass.value,
        jnp.asarray([[0.5, 0.25], [0.25, 0.5]]),
        atol=2e-12,
    )
    assert jnp.allclose(result.eigenvalues, jnp.asarray([4.0 / 3.0, 4.0]), atol=2e-12)


def test_separable_mlp_factor_and_partial_paths_match_dense_evaluation():
    model = phx.nn.models.SeparableMLP(
        in_size=2,
        out_size=2,
        latent_size=3,
        width_size=5,
        depth=1,
        key=jr.key(4),
    )
    x = jnp.linspace(-0.5, 0.75, 4)
    y = jnp.linspace(-1.0, 1.0, 5)

    factorized = model.factorize_axes((x, y), ("x", "y"))
    derivative = model.factorize_axes((x, y), ("x", "y"), partial=(0, 1))
    dense = model((x, y))
    dense_derivative = jax.vmap(
        lambda x_value: jax.vmap(
            lambda y_value: jax.jacfwd(
                lambda coordinate: model(jnp.asarray([coordinate, y_value]))
            )(x_value)
        )(y)
    )(x)

    assert jnp.allclose(factorized.contract().data, dense, atol=2e-12)
    assert jnp.allclose(derivative.contract().data, dense_derivative, atol=2e-11)


def test_factorized_assembly_rejects_coupled_masks():
    field = _field(
        [[0.0, 1.0], [1.0, 1.0]],
        [[1.0, 0.0], [1.0, 1.0]],
    )
    mask = cx.Field(jnp.ones((2, 2), dtype=bool), dims=("x", "y"))

    with pytest.raises(ValueError, match="separable weights"):
        phx.integration.factorized_inner_product(field, field, _batch(mask=mask))
