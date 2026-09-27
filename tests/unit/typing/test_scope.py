from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import pytest
from hypothesis import given, strategies as st

import phydrax.typing as pt


pytestmark = [pytest.mark.strict_jax, pytest.mark.filterwarnings("error")]


class NodeDim(pt.Dim):
    pass


class ComponentDim(pt.Dim):
    pass


@given(
    first_size=st.integers(min_value=1, max_value=8),
    second_size=st.integers(min_value=1, max_value=8),
)
def test_scopes_share_bindings_locally_and_remain_independent(
    first_size: int,
    second_size: int,
) -> None:
    first, second = pt.Scope(), pt.Scope()
    pt.parse(first_size, pt.Size[NodeDim], "count", scope=first)
    pt.parse(jnp.zeros((first_size,)), pt.Float64[NodeDim], "values", scope=first)
    pt.parse(jnp.zeros((second_size,)), pt.Float64[NodeDim], "values", scope=second)

    assert (first.size(NodeDim), second.size(NodeDim)) == (first_size, second_size)
    conflicting = first_size + 1
    with pytest.raises(ValueError, match="other"):
        pt.parse(jnp.zeros((conflicting,)), pt.Float64[NodeDim], "other", scope=first)
    assert first.size(NodeDim) == first_size


def test_failed_parse_contracts() -> None:
    scope = pt.Scope()
    union = pt.Float64[NodeDim, Literal[3]] | pt.Float64[ComponentDim, ComponentDim]
    pt.parse(jnp.zeros((2, 2)), union, "matrix", scope=scope)
    assert scope.size(ComponentDim) == 2
    with pytest.raises(ValueError):
        scope.size(NodeDim)

    empty = pt.Scope()
    with pytest.raises(TypeError):
        pt.parse(
            jnp.ones((3,), dtype=jnp.int32),
            pt.Float64[NodeDim],
            "wrong_dtype",
            scope=empty,
        )
    with pytest.raises(ValueError):
        empty.size(NodeDim)

    with pytest.raises(ValueError):
        pt.parse(
            (2, jnp.zeros((3,))),
            tuple[pt.Size[NodeDim], pt.Float64[NodeDim]],
            "late_failure",
            scope=empty,
        )
    with pytest.raises(ValueError):
        empty.size(NodeDim)
    scope = pt.Scope()
    pt.parse(2, pt.Size[NodeDim], "count", scope=scope)
    with pytest.raises(ValueError):
        pt.parse(
            (2, jnp.zeros((3,))),
            tuple[pt.Size[NodeDim], pt.Float64[NodeDim]],
            "late_failure",
            scope=scope,
        )
    assert scope.size(NodeDim) == 2

    with pytest.raises(ValueError):
        scope.size(ComponentDim)
    with pytest.raises(TypeError):
        # ty: ignore[invalid-argument-type]
        scope.size(int)
