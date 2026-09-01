#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntEnum
from math import prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import (
    ArraySpace,
    LinearSolveResult,
    LinearSolveStatus,
    LinearSystem,
    PreparedLinearSolve,
    RHSLayout,
    solve,
)
from ..operators.integral._convolution_quadrature import (
    causal_prefix_fft,
    ConvolutionQuadratureContour,
    ConvolutionQuadratureContourPolicy,
    ConvolutionQuadratureMethod,
    prepare_convolution_quadrature_contour,
    reconstruct_causal_history,
)


ConvolutionQuadratureAction: TypeAlias = Literal["forward", "transpose", "adjoint"]
NodeSolvePreparation: TypeAlias = Callable[
    [Array, ConvolutionQuadratureAction], PreparedLinearSolve
]


class ConvolutionQuadratureStatus(IntEnum):
    """Aggregate status for one complete fixed-history action."""

    SUCCESS = 0
    NONFINITE_INPUT = 1
    NODE_SOLVE_FAILED = 2
    NONFINITE_OUTPUT = 3


class ConvolutionQuadratureDeclaration(StrictModule):
    """Caller-owned scientific envelope for a dynamic transfer family.

    The controller is restricted to square coordinate systems of ``dimension``.
    PDE, geometry, formulation, and physics-kernel validity remain the caller's
    responsibility and are recorded rather than inferred.
    """

    dimension: int = eqx.field(static=True)
    family_id: str = eqx.field(static=True)
    pde: str = eqx.field(static=True)
    geometry: str = eqx.field(static=True)
    formulation: str = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    precision: str = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        /,
        *,
        family_id: str,
        pde: str,
        geometry: str,
        formulation: str,
        provider: str,
        precision: str,
        non_goals: Sequence[str] = (
            "continuum certification",
            "physics-kernel construction",
        ),
    ):
        if isinstance(dimension, bool):
            raise TypeError("dimension must be an integer.")
        dimension_ = int(dimension)
        if dimension_ < 1:
            raise ValueError("dimension must be positive.")
        strings = tuple(
            str(value)
            for value in (family_id, pde, geometry, formulation, provider, precision)
        )
        if any(not value for value in strings):
            raise ValueError(
                "Convolution-quadrature declaration strings must be non-empty."
            )
        non_goals_ = tuple(str(value) for value in non_goals)
        if not non_goals_ or any(not value for value in non_goals_):
            raise ValueError("non_goals must contain explicit non-empty statements.")
        if "continuum certification" not in non_goals_:
            non_goals_ = non_goals_ + ("continuum certification",)
        self.dimension = dimension_
        (
            self.family_id,
            self.pde,
            self.geometry,
            self.formulation,
            self.provider,
            self.precision,
        ) = strings
        self.non_goals = non_goals_


class ConvolutionQuadratureResourceEvidence(StrictModule):
    """Controller storage and logical-workspace evidence.

    The workspace count is an upper bound for the controller's logical complex
    arrays per external right-hand side. Provider-owned opaque allocations are
    excluded explicitly, while retained JAX array leaves are counted once by
    object identity.
    """

    contour_node_count: int = eqx.field(static=True)
    solved_node_count_per_action: int = eqx.field(static=True)
    retained_prepared_solve_count: int = eqx.field(static=True)
    retained_array_bytes: int = eqx.field(static=True)
    controller_workspace_upper_bound_bytes_per_rhs: int = eqx.field(static=True)
    node_right_hand_sides_per_external_rhs: int = eqx.field(static=True)
    history_truncated: bool = eqx.field(static=True)
    provider_opaque_allocations_included: bool = eqx.field(static=True)
    estimate_kind: str = eqx.field(static=True)
    reason: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        contour_node_count: int,
        solved_node_count_per_action: int,
        retained_prepared_solve_count: int,
        retained_array_bytes: int,
        controller_workspace_upper_bound_bytes_per_rhs: int,
        node_right_hand_sides_per_external_rhs: int,
    ):
        counts = tuple(
            int(value)
            for value in (
                contour_node_count,
                solved_node_count_per_action,
                retained_prepared_solve_count,
                retained_array_bytes,
                controller_workspace_upper_bound_bytes_per_rhs,
                node_right_hand_sides_per_external_rhs,
            )
        )
        if any(value < 0 for value in counts):
            raise ValueError(
                "Convolution-quadrature resource counts must be non-negative."
            )
        (
            self.contour_node_count,
            self.solved_node_count_per_action,
            self.retained_prepared_solve_count,
            self.retained_array_bytes,
            self.controller_workspace_upper_bound_bytes_per_rhs,
            self.node_right_hand_sides_per_external_rhs,
        ) = counts
        self.history_truncated = False
        self.provider_opaque_allocations_included = False
        self.estimate_kind = "controller-upper-bound-plus-observed-retained-arrays"
        self.reason = (
            "Logical controller buffers are bounded exactly by fixed dimensions; "
            "opaque provider workspaces are reported by retained LinearSolveResult "
            "provenance rather than guessed here."
        )


class ConvolutionQuadratureErrorEvidence(StrictModule):
    """Nodewise algebraic evidence without a continuum-error claim."""

    node_statuses: Array
    node_relative_residuals: Array
    input_finite: Array
    node_solves_successful: Array
    output_finite: Array
    contour_radius: Array
    contour_tolerance_target: Array
    history_truncated: bool = eqx.field(static=True)
    continuum_certified: bool = eqx.field(static=True)
    error_scope: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        node_statuses: ArrayLike,
        node_relative_residuals: ArrayLike,
        input_finite: ArrayLike,
        node_solves_successful: ArrayLike,
        output_finite: ArrayLike,
        contour_radius: ArrayLike,
        contour_tolerance_target: ArrayLike,
    ):
        self.node_statuses = jnp.asarray(node_statuses, dtype=jnp.int32)
        self.node_relative_residuals = jnp.asarray(node_relative_residuals)
        self.input_finite = jnp.asarray(input_finite, dtype=bool)
        self.node_solves_successful = jnp.asarray(node_solves_successful, dtype=bool)
        self.output_finite = jnp.asarray(output_finite, dtype=bool)
        self.contour_radius = jnp.asarray(contour_radius)
        self.contour_tolerance_target = jnp.asarray(contour_tolerance_target)
        self.history_truncated = False
        self.continuum_certified = False
        self.error_scope = (
            "checked discrete node solves, finite output, and contour policy target only"
        )


class PreparedConvolutionQuadrature(StrictModule):
    """Prepared square CQ controller over caller-supplied complex systems.

    Dimensionality and scientific scope are in ``declaration``. ``forward_nodes``
    prepares the transfer action, while ``transpose_nodes`` and ``adjoint_nodes``
    prepare exact algebraic-transpose and Hilbert-adjoint transfer actions. The
    conjugacy-reduced real envelope needs only forward and transpose systems.
    All histories have a fixed complete length; no memory horizon is imposed.
    """

    contour: ConvolutionQuadratureContour
    declaration: ConvolutionQuadratureDeclaration
    forward_nodes: tuple[PreparedLinearSolve, ...]
    transpose_nodes: tuple[PreparedLinearSolve, ...]
    adjoint_nodes: tuple[PreparedLinearSolve, ...] | None
    resource_evidence: ConvolutionQuadratureResourceEvidence
    continuum_certified: bool = eqx.field(static=True)
    error_scope: str = eqx.field(static=True)
    node_indices: tuple[int, ...] = eqx.field(static=True)
    node_providers: tuple[str, ...] = eqx.field(static=True)
    node_dtypes: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        contour: ConvolutionQuadratureContour,
        declaration: ConvolutionQuadratureDeclaration,
        forward_nodes: tuple[PreparedLinearSolve, ...],
        transpose_nodes: tuple[PreparedLinearSolve, ...],
        adjoint_nodes: tuple[PreparedLinearSolve, ...] | None,
        resource_evidence: ConvolutionQuadratureResourceEvidence,
        node_indices: tuple[int, ...],
    ):
        self.contour = contour
        self.declaration = declaration
        self.forward_nodes = forward_nodes
        self.transpose_nodes = transpose_nodes
        self.adjoint_nodes = adjoint_nodes
        self.resource_evidence = resource_evidence
        self.continuum_certified = False
        self.error_scope = (
            "contour policy target before execution; no a posteriori or continuum bound"
        )
        self.node_indices = node_indices
        all_nodes = (
            forward_nodes
            + transpose_nodes
            + (() if adjoint_nodes is None else adjoint_nodes)
        )
        self.node_providers = tuple(node.plan.backend for node in all_nodes)
        self.node_dtypes = tuple(
            np.dtype(node.problem.operator.target.dtype).name for node in all_nodes
        )

    def apply(self, history: ArrayLike, /) -> "ConvolutionQuadratureResult":
        return apply_convolution_quadrature(self, history, action="forward")

    def transpose(self, history: ArrayLike, /) -> "ConvolutionQuadratureResult":
        return apply_convolution_quadrature(self, history, action="transpose")

    def adjoint(self, history: ArrayLike, /) -> "ConvolutionQuadratureResult":
        return apply_convolution_quadrature(self, history, action="adjoint")


class ConvolutionQuadratureResult(StrictModule):
    """Checked causal history plus every actually executed node solve.

    The exact discrete action is valid only when ``successful`` is true. On any
    node failure ``value`` is identically zero, ``candidate`` remains available
    for diagnosis, and the failing per-node ``LinearSolveResult`` is retained.
    This result certifies neither the caller's PDE model nor continuum error.
    """

    value: Array
    candidate: Array
    status: Array
    node_results: tuple[LinearSolveResult, ...]
    error_evidence: ConvolutionQuadratureErrorEvidence
    resource_evidence: ConvolutionQuadratureResourceEvidence
    contour: ConvolutionQuadratureContour
    declaration: ConvolutionQuadratureDeclaration
    action: ConvolutionQuadratureAction = eqx.field(static=True)
    frequency_indices: tuple[int, ...] = eqx.field(static=True)
    parameter_indices: tuple[int, ...] = eqx.field(static=True)
    right_hand_side_count: int = eqx.field(static=True)
    node_providers: tuple[str, ...] = eqx.field(static=True)
    node_precision_dtypes: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        value: Array,
        candidate: Array,
        status: ArrayLike,
        node_results: tuple[LinearSolveResult, ...],
        error_evidence: ConvolutionQuadratureErrorEvidence,
        resource_evidence: ConvolutionQuadratureResourceEvidence,
        contour: ConvolutionQuadratureContour,
        declaration: ConvolutionQuadratureDeclaration,
        action: ConvolutionQuadratureAction,
        frequency_indices: tuple[int, ...],
        parameter_indices: tuple[int, ...],
        right_hand_side_count: int,
    ):
        self.value = value
        self.candidate = candidate
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.node_results = node_results
        self.error_evidence = error_evidence
        self.resource_evidence = resource_evidence
        self.contour = contour
        self.declaration = declaration
        self.action = action
        self.frequency_indices = frequency_indices
        self.parameter_indices = parameter_indices
        self.right_hand_side_count = int(right_hand_side_count)
        self.node_providers = tuple(result.provenance.backend for result in node_results)
        self.node_precision_dtypes = tuple(
            np.dtype(jnp.asarray(result.value).dtype).name for result in node_results
        )

    @property
    def successful(self) -> Array:
        return self.status == int(ConvolutionQuadratureStatus.SUCCESS)


def _array_storage_bytes(value: object, /) -> int:
    arrays = {id(leaf): leaf for leaf in jax.tree.leaves(value) if eqx.is_array(leaf)}
    return sum(int(array.size * array.dtype.itemsize) for array in arrays.values())


def _validate_prepared_node(
    node: PreparedLinearSolve,
    declaration: ConvolutionQuadratureDeclaration,
    history_length: int,
    action: ConvolutionQuadratureAction,
    /,
) -> None:
    if not isinstance(node, PreparedLinearSolve):
        raise TypeError(
            f"Node preparation for {action!r} must return PreparedLinearSolve."
        )
    if not isinstance(node.problem, LinearSystem):
        raise ValueError(
            "Convolution quadrature supports square LinearSystem nodes only."
        )
    if node.plan.policy.failure.mode != "status":
        raise ValueError(
            "Node solves must use status failure mode for checked aggregation."
        )
    if node.rhs_layout is not None:
        raise ValueError(
            "Node solves must be prepared without a fixed RHSLayout; the controller "
            "adds the causal-prefix and caller batch axes at execution."
        )
    operator = node.problem.operator
    if operator.batch_shape:
        raise ValueError("Convolution-quadrature node operators must be unbatched.")
    if not isinstance(operator.source, ArraySpace) or not isinstance(
        operator.target, ArraySpace
    ):
        raise TypeError("Node systems require one-dimensional ArraySpace coordinates.")
    expected = (declaration.dimension,)
    if operator.source.shape != expected or operator.target.shape != expected:
        raise ValueError(
            f"Node system for {action!r} must have coordinate shape {expected}."
        )
    if not jnp.issubdtype(
        operator.source.dtype, jnp.complexfloating
    ) or not jnp.issubdtype(operator.target.dtype, jnp.complexfloating):
        raise TypeError(
            "Dynamic complex-parameter node spaces must use complex coordinates."
        )
    if history_length < 1:
        raise ValueError("history_length must be positive.")


def _workspace_upper_bound(
    contour: ConvolutionQuadratureContour,
    declaration: ConvolutionQuadratureDeclaration,
    node: PreparedLinearSolve,
    /,
) -> int:
    itemsize = max(
        np.dtype(node.problem.operator.source.dtype).itemsize,
        np.dtype(node.problem.operator.target.dtype).itemsize,
    )
    transform_values = (
        4 * contour.fft_length * contour.history_length * declaration.dimension
    )
    output_values = contour.history_length * declaration.dimension
    return int(itemsize * (transform_values + output_values))


def prepare_convolution_quadrature(
    prepare_node: NodeSolvePreparation,
    step_size: ArrayLike,
    history_length: int,
    declaration: ConvolutionQuadratureDeclaration,
    /,
    *,
    method: ConvolutionQuadratureMethod = "bdf2",
    fft_length: int | None = None,
    contour_policy: ConvolutionQuadratureContourPolicy | None = None,
    conjugate_symmetric: bool = False,
) -> PreparedConvolutionQuadrature:
    """Prepare every required complex-frequency solve in a bounded CQ envelope.

    ``prepare_node(parameter, action)`` must return a checked phydrax.linalg
    ``PreparedLinearSolve`` implementing the transfer action named by ``action``.
    Physics kernels and their parameter dependence remain entirely outside this
    controller. Conjugacy reduction is a caller assertion and restricts numeric
    execution to real histories.
    """
    if not callable(prepare_node):
        raise TypeError("prepare_node must be callable.")
    if not isinstance(declaration, ConvolutionQuadratureDeclaration):
        raise TypeError("declaration must be ConvolutionQuadratureDeclaration.")
    contour = prepare_convolution_quadrature_contour(
        step_size,
        history_length,
        method=method,
        fft_length=fft_length,
        policy=contour_policy,
        conjugate_symmetric=conjugate_symmetric,
    )
    indices = contour.solved_node_indices

    def prepare_action(action: ConvolutionQuadratureAction):
        nodes = tuple(
            prepare_node(contour.parameters[index], action) for index in indices
        )
        for node in nodes:
            _validate_prepared_node(
                node,
                declaration,
                contour.history_length,
                action,
            )
        return nodes

    forward = prepare_action("forward")
    transposed = prepare_action("transpose")
    adjointed = None if conjugate_symmetric else prepare_action("adjoint")
    all_nodes = forward + transposed + (() if adjointed is None else adjointed)
    retained_bytes = _array_storage_bytes((contour, all_nodes))
    resources = ConvolutionQuadratureResourceEvidence(
        contour_node_count=contour.fft_length,
        solved_node_count_per_action=len(indices),
        retained_prepared_solve_count=len(all_nodes),
        retained_array_bytes=retained_bytes,
        controller_workspace_upper_bound_bytes_per_rhs=_workspace_upper_bound(
            contour,
            declaration,
            forward[0],
        ),
        node_right_hand_sides_per_external_rhs=contour.history_length,
    )
    return PreparedConvolutionQuadrature(
        contour=contour,
        declaration=declaration,
        forward_nodes=forward,
        transpose_nodes=transposed,
        adjoint_nodes=adjointed,
        resource_evidence=resources,
        node_indices=indices,
    )


def _action_nodes(
    prepared: PreparedConvolutionQuadrature,
    action: ConvolutionQuadratureAction,
    /,
) -> tuple[PreparedLinearSolve, ...]:
    if action == "forward":
        return prepared.forward_nodes
    if action == "transpose":
        return prepared.transpose_nodes
    if action == "adjoint":
        if prepared.contour.conjugate_symmetric:
            return prepared.transpose_nodes
        if prepared.adjoint_nodes is None:
            raise ValueError("Full complex adjoint nodes were not prepared.")
        return prepared.adjoint_nodes
    raise ValueError("action must be 'forward', 'transpose', or 'adjoint'.")


def _solve_node_frequencies(
    prepared: PreparedConvolutionQuadrature,
    transformed_history: Array,
    action: ConvolutionQuadratureAction,
    /,
) -> tuple[
    Array,
    tuple[LinearSolveResult, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    contour = prepared.contour
    frequency_indices = contour.solved_node_indices
    nodes = _action_nodes(prepared, action)
    positions = {index: position for position, index in enumerate(prepared.node_indices)}
    parameter_indices = tuple(
        (
            (-frequency_index) % contour.fft_length
            if action == "adjoint" and not contour.conjugate_symmetric
            else frequency_index
        )
        for frequency_index in frequency_indices
    )

    results: list[LinearSolveResult] = []
    solved_values: dict[int, Array] = {}
    for frequency_index, parameter_index in zip(
        frequency_indices, parameter_indices, strict=True
    ):
        rhs = jnp.moveaxis(transformed_history[frequency_index], 1, 0)
        layout = RHSLayout(rhs.shape[1:])
        result = solve(nodes[positions[parameter_index]], rhs, rhs_layout=layout)
        results.append(result)
        solved_values[frequency_index] = jnp.asarray(result.value)

    if contour.conjugate_symmetric:
        complete = tuple(
            solved_values[index]
            if index in solved_values
            else jnp.conj(solved_values[(-index) % contour.fft_length])
            for index in range(contour.fft_length)
        )
    else:
        complete = tuple(solved_values[index] for index in range(contour.fft_length))
    return jnp.stack(complete), tuple(results), frequency_indices, parameter_indices


def apply_convolution_quadrature(
    prepared: PreparedConvolutionQuadrature,
    history: ArrayLike,
    /,
    *,
    action: ConvolutionQuadratureAction = "forward",
) -> ConvolutionQuadratureResult:
    """Apply a forward, total-history transpose, or total-history adjoint CQ map.

    History axes are ``(time, coordinate, batch...)``. The transpose and adjoint
    reverse time around a causal spatial transpose/adjoint action, yielding the
    exact transpose/adjoint of the finite lower-triangular CQ history map. In the
    full complex case, adjoint node parameters are frequency-reversed before
    applying the caller-prepared Hilbert adjoints.
    """
    if not isinstance(prepared, PreparedConvolutionQuadrature):
        raise TypeError("prepared must be PreparedConvolutionQuadrature.")
    if action not in ("forward", "transpose", "adjoint"):
        raise ValueError("action must be 'forward', 'transpose', or 'adjoint'.")
    values = jnp.asarray(history)
    if values.ndim < 2:
        raise ValueError("history must have time and coordinate axes.")
    expected = (prepared.contour.history_length, prepared.declaration.dimension)
    if values.shape[:2] != expected:
        raise ValueError(f"history must begin with shape {expected}; got {values.shape}.")
    if not jnp.issubdtype(values.dtype, jnp.inexact):
        values = values.astype(float)
    if prepared.contour.conjugate_symmetric and jnp.issubdtype(
        values.dtype, jnp.complexfloating
    ):
        raise TypeError("Conjugacy-reduced convolution quadrature requires real history.")

    causal_input = values if action == "forward" else values[::-1]
    transformed = causal_prefix_fft(causal_input, prepared.contour)
    node_values, node_results, frequency_indices, parameter_indices = (
        _solve_node_frequencies(prepared, transformed, action)
    )
    causal_candidate = reconstruct_causal_history(node_values, prepared.contour)
    candidate = causal_candidate if action == "forward" else causal_candidate[::-1]
    if prepared.contour.conjugate_symmetric:
        candidate = jnp.real(candidate)

    node_statuses = jnp.stack(tuple(result.status for result in node_results))
    node_residuals = jnp.stack(
        tuple(result.diagnostics.relative_residual for result in node_results)
    )
    input_finite = jnp.all(jnp.isfinite(values))
    node_success = jnp.all(node_statuses == int(LinearSolveStatus.SUCCESS))
    output_finite = jnp.all(jnp.isfinite(candidate))
    status = jnp.where(
        ~input_finite,
        int(ConvolutionQuadratureStatus.NONFINITE_INPUT),
        jnp.where(
            ~node_success,
            int(ConvolutionQuadratureStatus.NODE_SOLVE_FAILED),
            jnp.where(
                ~output_finite,
                int(ConvolutionQuadratureStatus.NONFINITE_OUTPUT),
                int(ConvolutionQuadratureStatus.SUCCESS),
            ),
        ),
    )
    successful = status == int(ConvolutionQuadratureStatus.SUCCESS)
    value = jnp.where(successful, candidate, jnp.zeros_like(candidate))
    evidence = ConvolutionQuadratureErrorEvidence(
        node_statuses=node_statuses,
        node_relative_residuals=node_residuals,
        input_finite=input_finite,
        node_solves_successful=node_success,
        output_finite=output_finite,
        contour_radius=prepared.contour.radius,
        contour_tolerance_target=prepared.contour.tolerance,
    )
    rhs_count = prod(values.shape[2:]) if values.ndim > 2 else 1
    return ConvolutionQuadratureResult(
        value=value,
        candidate=candidate,
        status=status,
        node_results=node_results,
        error_evidence=evidence,
        resource_evidence=prepared.resource_evidence,
        contour=prepared.contour,
        declaration=prepared.declaration,
        action=action,
        frequency_indices=frequency_indices,
        parameter_indices=parameter_indices,
        right_hand_side_count=rhs_count,
    )


__all__ = [
    "ConvolutionQuadratureDeclaration",
    "ConvolutionQuadratureErrorEvidence",
    "ConvolutionQuadratureResourceEvidence",
    "ConvolutionQuadratureResult",
    "ConvolutionQuadratureStatus",
    "PreparedConvolutionQuadrature",
    "apply_convolution_quadrature",
    "prepare_convolution_quadrature",
]
