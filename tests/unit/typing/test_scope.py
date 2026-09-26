from typing import Literal

import jax.numpy as jnp
import pytest

import phydrax.typing as pt


class NodeDim(pt.Dim):
    pass


class ComponentDim(pt.Dim):
    pass


def test_scope_bindings_are_shared_across_checks_and_report_conflicts():
    scope = pt.Scope()
    pt.parse(3, pt.Size[NodeDim], "count", scope=scope)
    pt.parse(jnp.zeros((3,)), pt.Float64[NodeDim], "values", scope=scope)
    with pytest.raises(ValueError, match="count"):
        pt.parse(jnp.zeros((4,)), pt.Float64[NodeDim], "other", scope=scope)


def test_separate_scopes_are_independent():
    first, second = pt.Scope(), pt.Scope()
    pt.parse(jnp.zeros((2,)), pt.Float64[NodeDim], "a", scope=first)
    pt.parse(jnp.zeros((5,)), pt.Float64[NodeDim], "b", scope=second)
    assert (first.size(NodeDim), second.size(NodeDim)) == (2, 5)


def test_failed_union_alternatives_roll_back_every_binding():
    scope = pt.Scope()
    union = pt.Float64[NodeDim, Literal[3]] | pt.Float64[ComponentDim, ComponentDim]
    pt.parse(jnp.zeros((2, 2)), union, "matrix", scope=scope)
    assert scope.size(ComponentDim) == 2
    with pytest.raises(ValueError):
        scope.size(NodeDim)


def test_unbound_and_wrong_kind_lookups_are_refused():
    scope = pt.Scope()
    with pytest.raises(ValueError):
        scope.size(NodeDim)
    with pytest.raises(TypeError):
        scope.size(int)
