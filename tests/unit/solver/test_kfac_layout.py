#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.random as jr
import pytest

import phydrax as phx
from phydrax._trainable import partition_trainable
from phydrax.solver._kfac_layout import discover_parameter_layout


def _layout(functions, *, exact_block_max_size=64, uncovered="error"):
    parameters, _ = partition_trainable(functions)
    return discover_parameter_layout(
        functions,
        parameters,
        exact_block_max_size=exact_block_max_size,
        uncovered=uncovered,
    )


def test_parameter_layout_discovers_ordinary_affine_bias_blocks():
    domain = phx.domain.Interval1d(0.0, 1.0)
    model = phx.nn.MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(3,),
        rwf=False,
        key=jr.key(0),
    )

    layout = _layout({"u": domain.Model("x")(model)})

    assert len(layout.affine_blocks) == 2
    assert layout.affine_blocks[0].input_size == 2
    assert layout.uncovered_block is None
    assert sum(block.parameter_count for block in layout.affine_blocks) == (
        layout.parameter_count
    )


def test_parameter_layout_discovers_learned_skip_projection():
    domain = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    model = phx.nn.MLP(
        in_size=2,
        out_size="scalar",
        hidden_sizes=(4,),
        skip_connection=True,
        rwf=False,
        key=jr.key(1),
    )

    layout = _layout({"u": domain.Model("x")(model)})

    assert len(layout.affine_blocks) == 3
    assert layout.affine_blocks[-1].name.endswith("residual_projection")
    assert layout.uncovered_block is None


def test_parameter_layout_excludes_geometry_and_uses_exact_uncovered_block():
    domain = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    model = phx.nn.MLP(
        in_size=2,
        out_size="scalar",
        hidden_sizes=(),
        rwf=False,
        key=jr.key(31),
    )
    functions = {
        "u": domain.Model("x")(model),
        "coefficient": domain.Parameter(0.5),
    }

    layout = _layout(functions, exact_block_max_size=1)

    assert layout.uncovered_block is not None
    assert layout.uncovered_block.parameter_count == 1
    assert layout.uncovered_block.approximation == "exact"


def test_parameter_layout_uses_explicit_diagonal_fallback_above_threshold():
    domain = phx.domain.Interval1d(0.0, 1.0)
    model = phx.nn.MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(),
        rwf=False,
        key=jr.key(33),
    )
    functions = {
        "u": domain.Model("x")(model),
        "coefficient": domain.Parameter(0.5),
    }

    layout = _layout(functions, exact_block_max_size=0, uncovered="diagonal")

    assert layout.uncovered_block is not None
    assert layout.uncovered_block.approximation == "diagonal"


def test_parameter_layout_rejects_shared_affine_parameters():
    domain = phx.domain.Interval1d(0.0, 1.0)
    model = phx.nn.MLP(
        in_size=2,
        out_size=2,
        hidden_sizes=(2,),
        rwf=False,
        key=jr.key(34),
    )
    model = eqx.tree_at(
        lambda candidate: candidate.layers[1].weight,
        model,
        model.layers[0].weight,
    )

    with pytest.raises(ValueError, match="shared or reused affine parameters"):
        _layout({"u": domain.Model("x")(model)})


def test_parameter_layout_rejects_complex_uncovered_parameters():
    domain = phx.domain.Interval1d(0.0, 1.0)
    model = phx.nn.MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(),
        rwf=False,
        key=jr.key(32),
    )
    functions = {
        "u": domain.Model("x")(model),
        "coefficient": domain.Parameter(0.5 + 0.25j),
    }

    with pytest.raises(ValueError, match="real trainable parameters"):
        _layout(functions, exact_block_max_size=4)


def test_parameter_layout_rejects_random_weight_factorization():
    domain = phx.domain.Interval1d(0.0, 1.0)
    model = phx.nn.MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(3,),
        rwf=True,
        key=jr.key(2),
    )

    with pytest.raises(ValueError, match="disable rwf"):
        _layout({"u": domain.Model("x")(model)})
