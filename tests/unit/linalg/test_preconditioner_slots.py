#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Learned preconditioners inside neutral composite accelerator slots."""

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx
from phydrax.linalg._preconditioners import PrecisionCastPreconditioner
from phydrax.nn.models import MLP


la = phx.linalg


class _LearnedScaling(la.AbstractPreconditioner):
    """Entrywise learned scaling; a nonlinear map that claims no linearity."""

    model: MLP

    def __init__(self, model, space, /):
        self.model = model
        self.space = space
        self.properties = la.PreconditionerProperties()
        self.preconditioner_id = "test:learned-scaling"

    def apply(self, residual, /, *, iteration=None):
        del iteration
        values = self.space.validate(residual)
        scale = jax.nn.softplus(jax.vmap(self.model)(values[:, None]))
        return values * scale.astype(values.dtype)


def _model(seed, /, *, dtype=jnp.float64):
    model = MLP(
        in_size=1,
        out_size="scalar",
        width_size=4,
        depth=1,
        key=jax.random.key(seed),
    )
    return jax.tree.map(
        lambda leaf: leaf.astype(dtype) if eqx.is_inexact_array(leaf) else leaf, model
    )


def _operator(matrix, space, /):
    return la.DenseLinearOperator(matrix, source=space, target=space)


def _leaf_ids(tree):
    return {id(leaf) for leaf in jax.tree.leaves(tree)}


def test_learned_smoother_inside_multigrid_is_the_only_parameter_lane():
    fine = la.ArraySpace((3,), dtype=jnp.float64)
    coarse = la.ArraySpace((1,), dtype=jnp.float64)
    fine_operator = _operator(
        jnp.asarray([[3.0, -0.2, 0.1], [-0.2, 2.5, -0.3], [0.1, -0.3, 2.0]]), fine
    )
    coarse_operator = _operator(jnp.asarray([[1.25]]), coarse)
    learned = _LearnedScaling(_model(0), fine)
    hierarchy = la.MultigridHierarchy(
        (
            la.MultigridLevel(
                fine_operator,
                learned,
                restriction=la.DenseLinearOperator(
                    jnp.asarray([[0.4, 0.8, 0.2]]), source=fine, target=coarse
                ),
                prolongation=la.DenseLinearOperator(
                    jnp.asarray([[0.6], [0.5], [0.1]]), source=coarse, target=fine
                ),
            ),
            la.MultigridLevel(
                coarse_operator,
                la.DiagonalPreconditioner(jnp.asarray([1.25]), space=coarse),
            ),
        )
    )
    preconditioner = la.MultigridPreconditioner(hierarchy)

    parameters, model_state, fixed = phx.partition_parameters(preconditioner)

    assert _leaf_ids(parameters) == _leaf_ids(eqx.filter(learned.model, eqx.is_array))
    assert not jax.tree.leaves(model_state)
    assert _leaf_ids(fixed).isdisjoint(_leaf_ids(parameters))
    assert id(coarse_operator.matrix) in _leaf_ids(fixed)
    assert not preconditioner.properties.certifies("linear")

    residual = jnp.asarray([1.0, -0.75, 0.4])

    def loss(parameters):
        combined = phx.combine_parameters(parameters, model_state, fixed)
        return jnp.sum(combined.apply(residual) ** 2)

    gradient = jax.grad(loss)(parameters)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient))
    assert any(jnp.any(leaf != 0.0) for leaf in jax.tree.leaves(gradient))


def test_learned_inner_inside_precision_cast_stays_parameter():
    space = la.ArraySpace((3,), dtype=jnp.float64)
    lowered = la.ArraySpace((3,), dtype=jnp.float32)
    learned = _LearnedScaling(_model(1, dtype=jnp.float32), lowered)
    cast = PrecisionCastPreconditioner(learned, space, jnp.float32)

    parameters, _, fixed = phx.partition_parameters(cast)

    assert _leaf_ids(parameters) == _leaf_ids(eqx.filter(learned.model, eqx.is_array))
    assert cast.apply(jnp.ones(3)).dtype == jnp.float64
    assert not cast.properties.certifies("linear")

    analytic = PrecisionCastPreconditioner(
        la.DiagonalPreconditioner(jnp.full(3, 2.0, dtype=jnp.float32), space=lowered),
        space,
        jnp.float32,
    )
    analytic_parameters, _, _ = phx.partition_parameters(analytic)
    assert not jax.tree.leaves(analytic_parameters)


def test_preconditioner_slot_binds_models_with_accelerator_authority():
    contract = phx.bind_component(_model(2), la.AbstractPreconditioner).contract()

    assert contract.authority is phx.ComponentAuthority.ACCELERATOR
    assert contract.slot_semantic_id == la.AbstractPreconditioner.slot_semantic_id
