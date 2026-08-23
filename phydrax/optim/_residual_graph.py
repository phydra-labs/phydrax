#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._bounds import Bounds
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._tree_math import validate_real_inexact_tree
from ..linalg import AbstractLinearOperator, PyTreeSpace
from ._iterative import NonlinearLeastSquaresProblem
from ._riemannian import ParameterGeometry
from ._robust_losses import AbstractRobustLoss, robustify_residual


class ParameterBlock(StrictModule):
    """One parameter view with explicit extraction, replacement, and retraction."""

    extract: Callable[[PyTree[Any]], PyTree[Any]]
    replace: Callable[[PyTree[Any], PyTree[Any]], PyTree[Any]]
    retract: Callable[[PyTree[Any], PyTree[Any]], PyTree[Any]]
    bounds: Bounds | None
    block_id: str = eqx.field(static=True)
    constant: bool = eqx.field(static=True)
    geometry: ParameterGeometry | None
    elimination_group: int = eqx.field(static=True)

    def __init__(
        self,
        extract: Callable[[PyTree[Any]], PyTree[Any]],
        replace: Callable[[PyTree[Any], PyTree[Any]], PyTree[Any]],
        /,
        *,
        retract: Callable[[PyTree[Any], PyTree[Any]], PyTree[Any]] | None = None,
        bounds: Bounds | None = None,
        geometry: ParameterGeometry | None = None,
        block_id: str,
        constant: bool = False,
        elimination_group: int = 0,
    ):
        if not callable(extract) or not callable(replace):
            raise TypeError("extract and replace must be callable.")
        if retract is not None and not callable(retract):
            raise TypeError("retract must be callable or None.")
        if bounds is not None and not isinstance(bounds, Bounds):
            raise TypeError("bounds must be Bounds or None.")
        identifier = str(block_id)
        group = int(elimination_group)
        if not identifier:
            raise ValueError("block_id must be non-empty.")
        if geometry is not None and not isinstance(geometry, ParameterGeometry):
            raise TypeError("geometry must be ParameterGeometry or None.")
        if group < 0:
            raise ValueError("elimination_group must be non-negative.")
        self.extract = extract
        self.replace = replace
        self.geometry = geometry
        self.retract = (
            (
                (lambda value, tangent: geometry.retract(value, tangent))
                if geometry is not None
                else (
                    lambda value, tangent: jax.tree.map(
                        lambda point, delta: point + delta,
                        value,
                        tangent,
                    )
                )
            )
            if retract is None
            else retract
        )
        self.bounds = bounds
        self.block_id = identifier
        self.constant = bool(constant)
        self.elimination_group = group


class ResidualBlock(StrictModule):
    """One residual factor over an explicit tuple of parameter blocks."""

    function: Callable[[tuple[PyTree[Any], ...], Any], PyTree[Any]]
    weight: Any
    loss: AbstractRobustLoss | None
    parameter_ids: tuple[str, ...] = eqx.field(static=True)
    block_id: str = eqx.field(static=True)

    def __init__(
        self,
        function: Callable[[tuple[PyTree[Any], ...], Any], PyTree[Any]],
        parameter_ids: tuple[str, ...],
        /,
        *,
        weight: Any = None,
        loss: AbstractRobustLoss | None = None,
        block_id: str,
    ):
        if not callable(function):
            raise TypeError("function must be callable.")
        parameter_ids_ = tuple(str(value) for value in parameter_ids)
        if not parameter_ids_ or any(not value for value in parameter_ids_):
            raise ValueError("parameter_ids must be nonempty identifiers.")
        if loss is not None and not isinstance(loss, AbstractRobustLoss):
            raise TypeError("loss must be AbstractRobustLoss or None.")
        if weight is not None and not (
            callable(weight)
            or isinstance(weight, AbstractLinearOperator)
            or jnp.asarray(weight).ndim <= 2
        ):
            raise TypeError("weight must be scalar, matrix, operator, callable, or None.")
        identifier = str(block_id)
        if not identifier:
            raise ValueError("block_id must be non-empty.")
        self.function = function
        self.weight = weight
        self.loss = loss
        self.parameter_ids = parameter_ids_
        self.block_id = identifier

    def weighted_residual(self, values, args, /):
        """Evaluate one block after measurement weighting but before robust loss."""
        residual = validate_real_inexact_tree(
            self.function(values, args),
            name=f"residual block {self.block_id}",
        )
        if self.weight is None:
            return residual
        if isinstance(self.weight, AbstractLinearOperator):
            return self.weight.mv(residual)
        if callable(self.weight):
            return self.weight(residual)
        weight = jnp.asarray(self.weight)
        if weight.ndim == 0:
            return jax.tree.map(
                lambda value: weight * value,
                residual,
            )
        space = PyTreeSpace(residual)
        return weight @ space.flatten(residual)

    def evaluate(self, values, args, /):
        weighted = self.weighted_residual(values, args)
        return (
            weighted if self.loss is None else robustify_residual(weighted, self.loss)[0]
        )


class ResidualGraphProblem(StrictModule):
    """Block residual graph lowering to one ordinary least-squares problem."""

    parameter_blocks: tuple[ParameterBlock, ...]
    residual_blocks: tuple[ResidualBlock, ...]
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameter_blocks: tuple[ParameterBlock, ...],
        residual_blocks: tuple[ResidualBlock, ...],
        /,
        *,
        problem_id: str = "residual-graph",
    ):
        parameters = tuple(parameter_blocks)
        residuals = tuple(residual_blocks)
        if not parameters or not all(
            isinstance(value, ParameterBlock) for value in parameters
        ):
            raise TypeError("parameter_blocks must be a nonempty ParameterBlock tuple.")
        if not residuals or not all(
            isinstance(value, ResidualBlock) for value in residuals
        ):
            raise TypeError("residual_blocks must be a nonempty ResidualBlock tuple.")
        parameter_ids = tuple(value.block_id for value in parameters)
        residual_ids = tuple(value.block_id for value in residuals)
        if len(set(parameter_ids)) != len(parameter_ids):
            raise ValueError("Parameter block IDs must be unique.")
        if len(set(residual_ids)) != len(residual_ids):
            raise ValueError("Residual block IDs must be unique.")
        known = set(parameter_ids)
        if any(
            reference not in known
            for residual in residuals
            for reference in residual.parameter_ids
        ):
            raise ValueError("Every residual parameter reference must exist.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.parameter_blocks = parameters
        self.residual_blocks = residuals
        self.problem_id = identifier

    def parameter_values(self, parameters, /):
        return {
            block.block_id: block.extract(parameters) for block in self.parameter_blocks
        }

    def residual(self, parameters, args=None, /):
        values = self.parameter_values(parameters)
        return tuple(
            block.evaluate(
                tuple(values[identifier] for identifier in block.parameter_ids),
                args,
            )
            for block in self.residual_blocks
        )

    def retract(self, parameters, tangent_steps: dict[str, PyTree[Any]], /):
        result = parameters
        for block in self.parameter_blocks:
            if block.constant or block.block_id not in tangent_steps:
                continue
            value = block.extract(result)
            retracted = block.retract(value, tangent_steps[block.block_id])
            if block.bounds is not None:
                retracted = block.bounds.project(retracted)
            result = block.replace(result, retracted)
        return result

    def manifold_valid(self, parameters, /):
        valid = jnp.asarray(True)
        for block in self.parameter_blocks:
            if block.geometry is not None:
                valid = valid & block.geometry.contains(block.extract(parameters))
        return valid

    def as_least_squares_problem(self) -> NonlinearLeastSquaresProblem:
        return NonlinearLeastSquaresProblem(
            lambda parameters, args: self.residual(parameters, args),
            problem_id=self.problem_id,
        )

    @classmethod
    def from_residual(
        cls,
        residual: Callable[[PyTree[Any], Any], PyTree[Any]],
        /,
        *,
        problem_id: str = "single-residual-graph",
    ) -> ResidualGraphProblem:
        parameter = ParameterBlock(
            lambda value: value,
            lambda value, replacement: replacement,
            block_id="parameters",
        )
        block = ResidualBlock(
            lambda values, args: residual(values[0], args),
            ("parameters",),
            block_id="residual",
        )
        return cls((parameter,), (block,), problem_id=problem_id)


class FactorGraphCertificate(StrictModule):
    objective: Array
    gradient_norm: Array
    manifold_valid: Array
    finite: Array
    certified: Array


def factor_graph_certificate(
    graph: ResidualGraphProblem,
    parameters: PyTree[Any],
    args: Any = None,
    /,
    *,
    tolerance: float = 1e-8,
) -> FactorGraphCertificate:
    objective, gradient_norm = residual_graph_certificate(
        graph,
        parameters,
        args,
    )
    manifold_valid = graph.manifold_valid(parameters)
    finite = jnp.isfinite(objective) & jnp.isfinite(gradient_norm)
    return FactorGraphCertificate(
        objective,
        gradient_norm,
        manifold_valid,
        finite,
        finite & manifold_valid & (gradient_norm <= tolerance),
    )


class PreparedResidualGraph(StrictModule):
    """Static graph topology, block shapes, adjacency, and numeric version."""

    graph: ResidualGraphProblem
    parameter_sizes: Array
    residual_sizes: Array
    adjacency: Array
    ordering: Array
    graph_id: str = eqx.field(static=True)
    numeric_version: Array

    def __init__(
        self,
        graph: ResidualGraphProblem,
        parameter_sizes: Any,
        residual_sizes: Any,
        adjacency: Any,
        ordering: Any,
        /,
        *,
        graph_id: str,
        numeric_version: Any,
    ):
        if not isinstance(graph, ResidualGraphProblem):
            raise TypeError("graph must be ResidualGraphProblem.")
        self.graph = graph
        self.parameter_sizes = jnp.asarray(parameter_sizes, dtype=jnp.int32)
        self.residual_sizes = jnp.asarray(residual_sizes, dtype=jnp.int32)
        self.adjacency = jnp.asarray(adjacency, dtype=jnp.bool_)
        self.ordering = jnp.asarray(ordering, dtype=jnp.int32)
        self.graph_id = str(graph_id)
        self.numeric_version = jnp.asarray(numeric_version, dtype=jnp.int32)


def prepare_residual_graph(
    graph: ResidualGraphProblem,
    parameters: PyTree[Any],
    /,
    *,
    args: Any = None,
) -> PreparedResidualGraph:
    if not isinstance(graph, ResidualGraphProblem):
        raise TypeError("graph must be ResidualGraphProblem.")
    values = graph.parameter_values(parameters)
    residuals = graph.residual(parameters, args)
    parameter_sizes = [
        PyTreeSpace(values[block.block_id]).size for block in graph.parameter_blocks
    ]
    residual_sizes = [PyTreeSpace(value).size for value in residuals]
    parameter_index = {
        block.block_id: index for index, block in enumerate(graph.parameter_blocks)
    }
    adjacency = jnp.asarray(
        [
            [identifier in block.parameter_ids for identifier in parameter_index]
            for block in graph.residual_blocks
        ],
        dtype=jnp.bool_,
    )
    ordering = jnp.asarray(
        sorted(
            range(len(graph.parameter_blocks)),
            key=lambda index: (
                graph.parameter_blocks[index].elimination_group,
                graph.parameter_blocks[index].block_id,
            ),
        ),
        dtype=jnp.int32,
    )
    graph_id = canonical_fingerprint(
        {
            "kind": "residual-graph",
            "problem": graph.problem_id,
            "parameters": [
                {
                    "id": block.block_id,
                    "constant": block.constant,
                    "group": block.elimination_group,
                    "size": parameter_sizes[index],
                }
                for index, block in enumerate(graph.parameter_blocks)
            ],
            "residuals": [
                {
                    "id": block.block_id,
                    "parameters": list(block.parameter_ids),
                    "size": residual_sizes[index],
                }
                for index, block in enumerate(graph.residual_blocks)
            ],
        }
    )
    return PreparedResidualGraph(
        graph,
        parameter_sizes,
        residual_sizes,
        adjacency,
        ordering,
        graph_id=graph_id,
        numeric_version=0,
    )


def refresh_residual_graph(
    prepared: PreparedResidualGraph,
    graph: ResidualGraphProblem,
    parameters: PyTree[Any],
    /,
    *,
    args: Any = None,
) -> PreparedResidualGraph:
    if not isinstance(prepared, PreparedResidualGraph):
        raise TypeError("prepared must be PreparedResidualGraph.")
    candidate = prepare_residual_graph(graph, parameters, args=args)
    if candidate.graph_id != prepared.graph_id:
        raise ValueError(
            "Residual graph refresh changed static topology or block shapes."
        )
    return PreparedResidualGraph(
        graph,
        prepared.parameter_sizes,
        prepared.residual_sizes,
        prepared.adjacency,
        prepared.ordering,
        graph_id=prepared.graph_id,
        numeric_version=prepared.numeric_version + 1,
    )


def residual_graph_certificate(graph, parameters, args=None, /):
    residuals = graph.residual(parameters, args)
    objective = 0.5 * sum(
        jnp.real(jnp.vdot(value, value)) for value in jax.tree.leaves(residuals)
    )
    gradient = jax.grad(
        lambda value: (
            0.5
            * sum(
                jnp.real(jnp.vdot(item, item))
                for item in jax.tree.leaves(graph.residual(value, args))
            )
        )
    )(parameters)
    return objective, jnp.sqrt(
        sum(jnp.real(jnp.vdot(value, value)) for value in jax.tree.leaves(gradient))
    )


__all__ = [
    "ParameterBlock",
    "PreparedResidualGraph",
    "ResidualBlock",
    "ResidualGraphProblem",
    "prepare_residual_graph",
    "refresh_residual_graph",
    "residual_graph_certificate",
    "FactorGraphCertificate",
    "factor_graph_certificate",
]
