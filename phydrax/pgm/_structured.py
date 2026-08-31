#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import jax.numpy as jnp
from jaxtyping import ArrayLike

from ._model import (
    DiscreteFactorGraph,
    DiscreteVariableGroup,
    IsingFactorGroup,
    PottsFactorGroup,
    VariableSelection,
)


def _edge_array(edges: ArrayLike, variable_count: int, /):
    array = jnp.asarray(edges)
    if not jnp.issubdtype(array.dtype, jnp.integer):
        raise TypeError("edges must contain integer variable indices.")
    array = array.astype(jnp.int32)
    if array.ndim != 2 or int(array.shape[1]) != 2:
        raise ValueError("edges must have shape (edge, 2).")
    if array.size and bool(jnp.any((array < 0) | (array >= variable_count))):
        raise ValueError("edge index is outside the variable range.")
    if array.size and bool(jnp.any(array[:, 0] == array[:, 1])):
        raise ValueError("Self edges must be folded into unary potentials.")
    return array


def ising_factor_graph(
    fields: ArrayLike,
    edges: ArrayLike,
    couplings: ArrayLike,
    /,
    *,
    name: str = "spins",
    shape: tuple[int, ...] | None = None,
) -> DiscreteFactorGraph:
    """Construct a binary Ising graph with spin convention s = 2x - 1."""
    field_values = jnp.asarray(fields)
    resolved_shape = tuple(field_values.shape) if shape is None else tuple(shape)
    variable_count = prod(resolved_shape) if resolved_shape else 1
    if field_values.size != variable_count:
        raise ValueError("fields must contain one value per Ising variable.")
    variables = DiscreteVariableGroup(name, shape=resolved_shape, num_states=2)
    edge_values = _edge_array(edges, variable_count)
    coupling_values = jnp.asarray(couplings).reshape((-1,))
    if coupling_values.shape != (int(edge_values.shape[0]),):
        raise ValueError("couplings must contain one value per edge.")
    groups = [
        IsingFactorGroup(
            (VariableSelection.all(variables),),
            field_values.reshape((-1,)),
        )
    ]
    if int(edge_values.shape[0]):
        groups.append(
            IsingFactorGroup(
                (
                    VariableSelection(variables, edge_values[:, 0]),
                    VariableSelection(variables, edge_values[:, 1]),
                ),
                coupling_values,
            )
        )
    return DiscreteFactorGraph((variables,), tuple(groups))


def potts_factor_graph(
    unary_log_potentials: ArrayLike,
    edges: ArrayLike,
    pairwise_log_potentials: ArrayLike,
    /,
    *,
    name: str = "states",
    shape: tuple[int, ...] | None = None,
) -> DiscreteFactorGraph:
    """Construct a uniform-cardinality unary/pairwise Potts factor graph."""
    unary = jnp.asarray(unary_log_potentials)
    if unary.ndim < 1:
        raise ValueError("unary_log_potentials must have a final state axis.")
    cardinality = int(unary.shape[-1])
    if cardinality < 1:
        raise ValueError("Potts cardinality must be positive.")
    resolved_shape = tuple(unary.shape[:-1]) if shape is None else tuple(shape)
    variable_count = prod(resolved_shape) if resolved_shape else 1
    if unary.size != variable_count * cardinality:
        raise ValueError("unary_log_potentials must have one row per Potts variable.")
    variables = DiscreteVariableGroup(
        name,
        shape=resolved_shape,
        num_states=cardinality,
    )
    edge_values = _edge_array(edges, variable_count)
    pairwise = jnp.asarray(pairwise_log_potentials)
    expected_pairwise = (int(edge_values.shape[0]), cardinality, cardinality)
    if pairwise.shape != expected_pairwise:
        raise ValueError(
            f"pairwise_log_potentials must have shape {expected_pairwise}; got {pairwise.shape}."
        )
    groups = [
        PottsFactorGroup(
            (VariableSelection.all(variables),),
            unary.reshape((variable_count, cardinality)),
        )
    ]
    if int(edge_values.shape[0]):
        groups.append(
            PottsFactorGroup(
                (
                    VariableSelection(variables, edge_values[:, 0]),
                    VariableSelection(variables, edge_values[:, 1]),
                ),
                pairwise,
            )
        )
    return DiscreteFactorGraph((variables,), tuple(groups))


__all__ = ["ising_factor_graph", "potts_factor_graph"]
