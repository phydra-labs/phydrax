#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from .._doc import DOC_KEY0


class ModelObjectiveProvider(abc.ABC):
    """Model node that explicitly contributes local scalar objective terms."""

    def model_objective_identity(self) -> int:
        """Return the identity used to deduplicate shared model nodes."""
        return id(self)

    def model_objective_children_first(self) -> bool:
        """Return whether nested provider terms precede this node's local terms."""
        return False

    def local_model_objective_labels(self) -> tuple[str, ...]:
        """Return labels for objectives contributed directly by this node."""
        return ()

    def local_model_objective_values(
        self,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: Array | None = None,
    ) -> tuple[Array, ...]:
        """Evaluate objectives contributed directly by this node."""
        del key, iter_
        return ()


def _direct_model_objective_providers(tree: Any, /) -> tuple[ModelObjectiveProvider, ...]:
    leaves = jax.tree_util.tree_leaves(
        tree,
        is_leaf=lambda value: (
            value is not tree and isinstance(value, ModelObjectiveProvider)
        ),
    )
    return tuple(value for value in leaves if isinstance(value, ModelObjectiveProvider))


def iter_model_objective_providers(tree: Any, /) -> Iterator[ModelObjectiveProvider]:
    """Yield explicit objective providers in stable PyTree order without duplicates."""
    seen_nodes: set[int] = set()
    seen_objectives: set[int] = set()

    def visit(value: Any) -> Iterator[ModelObjectiveProvider]:
        node_id = id(value)
        if node_id in seen_nodes:
            return
        seen_nodes.add(node_id)

        provider = value if isinstance(value, ModelObjectiveProvider) else None
        children = _direct_model_objective_providers(value)
        children_first = (
            provider is not None and provider.model_objective_children_first()
        )
        if children_first:
            for child in children:
                yield from visit(child)
        if provider is not None:
            objective_id = provider.model_objective_identity()
            if objective_id not in seen_objectives:
                seen_objectives.add(objective_id)
                yield provider
        if not children_first:
            for child in children:
                yield from visit(child)

    yield from visit(tree)


def model_objective_labels(tree: Any, /) -> tuple[str, ...]:
    """Return labels for every explicit model objective in stable order."""
    return tuple(
        label
        for provider in iter_model_objective_providers(tree)
        for label in provider.local_model_objective_labels()
    )


def model_objective_values(
    tree: Any,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    iter_: Array | None = None,
) -> tuple[Array, ...]:
    """Evaluate every explicit model objective using deterministic site keys."""
    values: list[Array] = []
    site = 0
    for provider in iter_model_objective_providers(tree):
        labels = provider.local_model_objective_labels()
        if not labels:
            continue
        local_values = provider.local_model_objective_values(
            key=jr.fold_in(key, site),
            iter_=iter_,
        )
        if len(local_values) != len(labels):
            raise ValueError(
                f"{type(provider).__name__} declared {len(labels)} model objective "
                f"labels but returned {len(local_values)} values."
            )
        values.extend(
            jnp.asarray(value, dtype=float).reshape(()) for value in local_values
        )
        site += len(local_values)
    return tuple(values)


__all__ = [
    "ModelObjectiveProvider",
    "iter_model_objective_providers",
    "model_objective_labels",
    "model_objective_values",
]
