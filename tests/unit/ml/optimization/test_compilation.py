#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.ml import FeatureSchema, TargetSchema
from phydrax.ml.linear import LinearRegressorModel
from phydrax.ml.tree import TreeEnsemble


def _stump(split_kind):
    return TreeEnsemble(
        feature_index=jnp.asarray([[0, -1, -1]]),
        threshold=jnp.asarray([[2.0, 0.0, 0.0]]),
        left_child=jnp.asarray([[1, -1, -1]]),
        right_child=jnp.asarray([[2, -1, -1]]),
        default_left=jnp.zeros((1, 3), dtype="bool"),
        split_kind=jnp.asarray([[split_kind, 0, 0]]),
        leaf_value=jnp.asarray([[[0.0], [10.0], [20.0]]]),
        node_mask=jnp.ones((1, 3), dtype="bool"),
        leaf_mask=jnp.asarray([[False, True, True]]),
        tree_mask=jnp.asarray([True]),
        base_score=jnp.asarray([0.0]),
        feature_schema=FeatureSchema(("x",), layout_id="tree-x"),
        target_schema=TargetSchema("continuous", names=("y",)),
        out_size="scalar",
        max_steps=2,
    )


def test_affine_predictor_constraint_changes_observable_milp_optimum():
    base = phx.optim.MixedIntegerProgram(
        phx.optim.LinearProgram(
            jnp.asarray([-1.0]),
            bounds=phx.optim.Bounds(0.0, 1.0),
            problem_id="affine-base",
        ),
        binary_indices=(0,),
        program_id="affine-base-mixed",
    )
    model = LinearRegressorModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([0.0]),
        case_shape=(),
        target_shape=(1,),
    )
    binding = phx.ml.optimization.bind_predictor_inputs(
        base,
        jnp.asarray([[1.0]]),
        jnp.asarray([0.0]),
        feature_layout_id="x",
    )
    compilation = phx.ml.optimization.compile_linear_predictor_constraint(
        model,
        binding,
        phx.ml.optimization.PredictorOutputConstraint(0.2),
    )

    result = phx.optim.solve_mixed_integer_program(
        phx.ml.optimization.augment_linear_program(base, compilation)
    )

    assert result.successful
    np.testing.assert_allclose(result.primal[:2], [0.0, 0.0], atol=1e-8)


@pytest.mark.parametrize(("split_kind", "expected"), [(0, 3.0), (2, 2.0)])
def test_tree_compilation_preserves_strict_and_nonstrict_integer_ties(
    split_kind,
    expected,
):
    model = _stump(split_kind)
    base = phx.optim.MixedIntegerProgram(
        phx.optim.LinearProgram(
            jnp.asarray([1.0]),
            bounds=phx.optim.Bounds(0.0, 3.0),
            problem_id="tree-base",
        ),
        integer_indices=(0,),
        program_id="tree-base-mixed",
    )
    binding = phx.ml.optimization.bind_predictor_inputs(
        base,
        jnp.asarray([[1.0]]),
        jnp.asarray([0.0]),
        feature_layout_id=model.feature_schema.layout_id,
    )
    compilation = phx.ml.optimization.compile_tree_predictor_constraint(
        model,
        binding,
        phx.ml.optimization.PredictorOutputConstraint(
            15.0,
            sense="lower",
        ),
    )

    result = phx.optim.solve_mixed_integer_program(
        phx.ml.optimization.augment_linear_program(base, compilation)
    )

    assert result.successful
    assert result.primal[0] == pytest.approx(expected, abs=1e-7)
    assert model(jnp.asarray([[result.primal[0]]]))[0] >= 15.0


def test_tree_compilation_refuses_continuous_strict_split():
    model = _stump(0)
    base = phx.optim.LinearProgram(
        jnp.asarray([0.0]),
        bounds=phx.optim.Bounds(0.0, 3.0),
    )
    binding = phx.ml.optimization.bind_predictor_inputs(
        base,
        jnp.asarray([[1.0]]),
        jnp.asarray([0.0]),
        feature_layout_id=model.feature_schema.layout_id,
    )

    with pytest.raises(ValueError, match="discrete"):
        phx.ml.optimization.compile_tree_predictor_constraint(
            model,
            binding,
            phx.ml.optimization.PredictorOutputConstraint(15.0),
        )


def test_convex_hull_support_returns_witness_and_blocks_extrapolation():
    base = phx.optim.LinearProgram(
        jnp.asarray([-1.0]),
        bounds=phx.optim.Bounds(0.0, 2.0),
        problem_id="support-base",
    )
    binding = phx.ml.optimization.bind_predictor_inputs(
        base,
        jnp.asarray([[1.0]]),
        jnp.asarray([0.0]),
        feature_layout_id="x",
    )
    support = phx.ml.optimization.compile_convex_hull_support(
        binding,
        jnp.asarray([[0.0], [1.0]]),
        support_id="observed-points",
    )

    result = phx.optim.solve_linear_program(
        phx.ml.optimization.augment_with_support(base, support)
    )

    assert result.successful
    assert result.primal[0] == pytest.approx(1.0, abs=1e-7)
    witness = result.primal[support.witness_start : support.witness_stop]
    np.testing.assert_allclose(jnp.sum(witness), 1.0, atol=1e-8)
