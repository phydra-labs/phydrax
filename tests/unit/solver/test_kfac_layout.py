#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.solver._kfac_layout import discover_parameter_layout


def _layout(
    functions: Any, *, exact_block_max_size: Any = 64, uncovered: Any = "error"
) -> Any:
    parameters, _, _ = phx.partition_parameters(functions)
    return discover_parameter_layout(
        functions,
        parameters,
        exact_block_max_size=exact_block_max_size,
        uncovered=uncovered,
    )


def test_parameter_layout_contracts() -> None:
    domain = phx.domain.Interval1d(0.0, 1.0)
    model = phx.nn.models.MLP(
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
    domain = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    model = phx.nn.models.MLP(
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
    domain = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    model = phx.nn.models.MLP(
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
    domain = phx.domain.Interval1d(0.0, 1.0)
    model = phx.nn.models.MLP(
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
    domain = phx.domain.Interval1d(0.0, 1.0)
    model = phx.nn.models.MLP(
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

    with pytest.raises(ValueError, match="require one explicit sharing_group"):
        _layout({"u": domain.Model("x")(model)})
    domain = phx.domain.Interval1d(0.0, 1.0)
    model = phx.nn.models.MLP(
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
    domain = phx.domain.Interval1d(0.0, 1.0)
    model = phx.nn.models.MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(3,),
        rwf=True,
        key=jr.key(2),
    )

    with pytest.raises(ValueError, match="requires an exact coordinate_pullback"):
        _layout({"u": domain.Model("x")(model)})
    domain = phx.domain.Interval1d(0.0, 1.0)
    stateful = phx.domain.DomainFunction(
        domain=domain,
        deps=("x",),
        func=_StatefulScale(jnp.asarray(2.0), jnp.asarray(0)),
    )

    with pytest.raises(ValueError, match=r"KFAC field 'u'.*MODEL_STATE.*calls"):
        _layout({"u": stateful})


class _StatefulScale(phx.ParameterOwner, eqx.Module):
    scale: jax.Array
    calls: jax.Array = phx.model_state_field()

    def __call__(self, x: Any, **_kwargs: Any) -> Any:
        return self.scale * x[0]
