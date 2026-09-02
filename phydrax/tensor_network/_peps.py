#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._contraction import (
    ContractionExecutionEvidence,
    ContractionResourcePolicy,
    execute_contraction,
    plan_contraction,
    prepare_contraction,
)
from ._precision import TensorNetworkPrecisionPolicy
from ._topology import ContractionLeg, ContractionOperand, ContractionStructure


class PEPS(StrictModule):
    """Finite row-major rectangular PEPS with explicit OBC unit boundary legs."""

    tensors: tuple[Array, ...]
    rows: int = eqx.field(static=True)
    columns: int = eqx.field(static=True)
    physical_dimensions: tuple[int, ...] = eqx.field(static=True)
    precision: TensorNetworkPrecisionPolicy
    state_id: str = eqx.field(static=True)
    numeric_version: Array

    def __init__(
        self,
        tensors: Sequence[ArrayLike],
        rows: int,
        columns: int,
        /,
        *,
        precision: TensorNetworkPrecisionPolicy | None = None,
        numeric_version: ArrayLike = 0,
    ):
        arrays = tuple(jnp.asarray(tensor) for tensor in tensors)
        rows_ = int(rows)
        columns_ = int(columns)
        if rows_ < 1 or columns_ < 1 or len(arrays) != rows_ * columns_:
            raise ValueError("PEPS tensors must fill one positive rectangular grid.")
        if any(array.ndim != 5 for array in arrays):
            raise ValueError(
                "PEPS tensors require axes (up, right, down, left, physical)."
            )
        if any(any(dimension < 1 for dimension in array.shape) for array in arrays):
            raise ValueError("PEPS tensor dimensions must be positive.")
        precision_ = TensorNetworkPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, TensorNetworkPrecisionPolicy):
            raise TypeError("precision must be TensorNetworkPrecisionPolicy or None.")
        precision_.validate_storage(arrays)
        for row in range(rows_):
            for column in range(columns_):
                tensor = arrays[row * columns_ + column]
                if row == 0 and tensor.shape[0] != 1:
                    raise ValueError("Top PEPS boundary bonds must have dimension one.")
                if row + 1 == rows_ and tensor.shape[2] != 1:
                    raise ValueError(
                        "Bottom PEPS boundary bonds must have dimension one."
                    )
                if column == 0 and tensor.shape[3] != 1:
                    raise ValueError("Left PEPS boundary bonds must have dimension one.")
                if column + 1 == columns_ and tensor.shape[1] != 1:
                    raise ValueError("Right PEPS boundary bonds must have dimension one.")
                if (
                    column + 1 < columns_
                    and tensor.shape[1] != arrays[row * columns_ + column + 1].shape[3]
                ):
                    raise ValueError("Neighboring horizontal PEPS bonds must match.")
                if (
                    row + 1 < rows_
                    and tensor.shape[2] != arrays[(row + 1) * columns_ + column].shape[0]
                ):
                    raise ValueError("Neighboring vertical PEPS bonds must match.")
        self.tensors = arrays
        self.rows = rows_
        self.columns = columns_
        self.physical_dimensions = tuple(int(array.shape[4]) for array in arrays)
        self.precision = precision_
        self.state_id = canonical_fingerprint(
            {
                "kind": "finite-obc-peps",
                "shape": (rows_, columns_),
                "tensor_shapes": tuple(array.shape for array in arrays),
                "dtype": str(arrays[0].dtype),
                "precision": precision_.policy_id,
            }
        )
        version = jnp.asarray(numeric_version, dtype=jnp.int32)
        if version.ndim != 0:
            raise ValueError("numeric_version must be scalar.")
        self.numeric_version = version


class PEPO(StrictModule):
    """Finite row-major rectangular PEPO with OBC and output/input physical axes."""

    tensors: tuple[Array, ...]
    rows: int = eqx.field(static=True)
    columns: int = eqx.field(static=True)
    output_dimensions: tuple[int, ...] = eqx.field(static=True)
    input_dimensions: tuple[int, ...] = eqx.field(static=True)
    precision: TensorNetworkPrecisionPolicy
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        tensors: Sequence[ArrayLike],
        rows: int,
        columns: int,
        /,
        *,
        precision: TensorNetworkPrecisionPolicy | None = None,
    ):
        arrays = tuple(jnp.asarray(tensor) for tensor in tensors)
        rows_ = int(rows)
        columns_ = int(columns)
        if rows_ < 1 or columns_ < 1 or len(arrays) != rows_ * columns_:
            raise ValueError("PEPO tensors must fill one positive rectangular grid.")
        if any(array.ndim != 6 for array in arrays):
            raise ValueError(
                "PEPO tensors require (up, right, down, left, output, input)."
            )
        if any(any(dimension < 1 for dimension in array.shape) for array in arrays):
            raise ValueError("PEPO tensor dimensions must be positive.")
        precision_ = TensorNetworkPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, TensorNetworkPrecisionPolicy):
            raise TypeError("precision must be TensorNetworkPrecisionPolicy or None.")
        precision_.validate_storage(arrays)
        for row in range(rows_):
            for column in range(columns_):
                tensor = arrays[row * columns_ + column]
                if tensor.shape[4] < 1 or tensor.shape[5] < 1:
                    raise ValueError("PEPO physical dimensions must be positive.")
                if (
                    row == 0
                    and tensor.shape[0] != 1
                    or row + 1 == rows_
                    and tensor.shape[2] != 1
                ):
                    raise ValueError("PEPO vertical OBC bonds must have dimension one.")
                if (
                    column == 0
                    and tensor.shape[3] != 1
                    or column + 1 == columns_
                    and tensor.shape[1] != 1
                ):
                    raise ValueError("PEPO horizontal OBC bonds must have dimension one.")
                if (
                    column + 1 < columns_
                    and tensor.shape[1] != arrays[row * columns_ + column + 1].shape[3]
                ):
                    raise ValueError("Neighboring horizontal PEPO bonds must match.")
                if (
                    row + 1 < rows_
                    and tensor.shape[2] != arrays[(row + 1) * columns_ + column].shape[0]
                ):
                    raise ValueError("Neighboring vertical PEPO bonds must match.")
        self.tensors = arrays
        self.rows = rows_
        self.columns = columns_
        self.output_dimensions = tuple(int(array.shape[4]) for array in arrays)
        self.input_dimensions = tuple(int(array.shape[5]) for array in arrays)
        self.precision = precision_
        self.operator_id = canonical_fingerprint(
            {
                "kind": "finite-obc-pepo",
                "shape": (rows_, columns_),
                "tensor_shapes": tuple(array.shape for array in arrays),
                "dtype": str(arrays[0].dtype),
                "precision": precision_.policy_id,
            }
        )


class PEPSContractionEvidence(StrictModule):
    network_id: str = eqx.field(static=True)
    replay_id: str = eqx.field(static=True)
    method: str = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    claim: str = eqx.field(static=True)
    global_error_bound_claimed: bool = eqx.field(static=True)
    local_transfer_elements: int = eqx.field(static=True)
    contraction: ContractionExecutionEvidence


class PEPSContractionResult(StrictModule):
    value: Array
    evidence: PEPSContractionEvidence


def _edge_label(row: int, column: int, direction: str, /) -> str:
    if direction == "up":
        return f"vertical:{row - 1}:{column}" if row else f"boundary:top:{column}"
    if direction == "right":
        return f"horizontal:{row}:{column}" if column >= 0 else ""
    if direction == "down":
        return f"vertical:{row}:{column}"
    return f"horizontal:{row}:{column - 1}" if column else f"boundary:left:{row}"


def _grid_structure(
    shapes: Sequence[tuple[int, int, int, int]], rows: int, columns: int, /
) -> ContractionStructure:
    nodes = []
    for row in range(rows):
        for column in range(columns):
            up, right, down, left = shapes[row * columns + column]
            labels = (
                _edge_label(row, column, "up"),
                f"horizontal:{row}:{column}"
                if column + 1 < columns
                else f"boundary:right:{row}",
                f"vertical:{row}:{column}"
                if row + 1 < rows
                else f"boundary:bottom:{column}",
                _edge_label(row, column, "left"),
            )
            nodes.append(
                ContractionOperand(
                    f"site:{row}:{column}",
                    tuple(
                        ContractionLeg(label, dimension)
                        for label, dimension in zip(
                            labels, (up, right, down, left), strict=True
                        )
                    ),
                )
            )
    return ContractionStructure(tuple(nodes), ())


def _double_layer_specification(left: PEPS, right: PEPS, /):
    if (
        left.rows != right.rows
        or left.columns != right.columns
        or left.physical_dimensions != right.physical_dimensions
    ):
        raise ValueError(
            "PEPS overlap requires identical lattice and physical dimensions."
        )
    shapes = tuple(
        tuple(int(a * b) for a, b in zip(first.shape[:4], second.shape[:4], strict=True))
        for first, second in zip(left.tensors, right.tensors, strict=True)
    )
    return _grid_structure(shapes, left.rows, left.columns), shapes


def contract_peps_exact(
    state: PEPS,
    other: PEPS | None = None,
    /,
    *,
    resources: ContractionResourcePolicy | None = None,
    optimizer: str = "greedy",
) -> PEPSContractionResult:
    """Exactly contract a small finite PEPS overlap after resource admission."""

    if not isinstance(state, PEPS) or (other is not None and not isinstance(other, PEPS)):
        raise TypeError("state and other must be PEPS values.")
    right = state if other is None else other
    if state.precision.policy_id != right.precision.policy_id:
        raise ValueError("PEPS overlap precision policies must match.")
    structure, shapes = _double_layer_specification(state, right)
    plan = plan_contraction(
        structure,
        precision=state.precision,
        resources=resources,
        optimizer=optimizer,
        dtype=str(state.tensors[0].dtype),
    )
    transfers = tuple(
        oe.contract(
            "urdlp,URDLp->uUrRdDlL", jnp.conj(first), second, optimize="greedy"
        ).reshape(shape)
        for first, second, shape in zip(state.tensors, right.tensors, shapes, strict=True)
    )
    contracted = execute_contraction(prepare_contraction(plan, transfers))
    replay_id = canonical_fingerprint(
        {
            "kind": "exact-peps-overlap",
            "left": state.state_id,
            "right": right.state_id,
            "plan": plan.plan_id,
        }
    )
    evidence = PEPSContractionEvidence(
        state.state_id,
        replay_id,
        "exact-full-network",
        True,
        "exact finite OBC PEPS overlap",
        False,
        sum(prod(shape) for shape in shapes),
        contracted.evidence,
    )
    return PEPSContractionResult(contracted.value, evidence)


def peps_amplitude(
    state: PEPS,
    configuration: Sequence[int],
    /,
    *,
    resources: ContractionResourcePolicy | None = None,
) -> PEPSContractionResult:
    """Exactly contract one computational-basis amplitude."""

    values = tuple(int(value) for value in configuration)
    if len(values) != len(state.tensors) or any(
        value < 0 or value >= dimension
        for value, dimension in zip(values, state.physical_dimensions, strict=True)
    ):
        raise ValueError("PEPS configuration lies outside the physical dimensions.")
    shapes = tuple(
        tuple(int(value) for value in tensor.shape[:4]) for tensor in state.tensors
    )
    structure = _grid_structure(shapes, state.rows, state.columns)
    plan = plan_contraction(
        structure,
        precision=state.precision,
        resources=resources,
        dtype=str(state.tensors[0].dtype),
    )
    arrays = tuple(
        tensor[..., value] for tensor, value in zip(state.tensors, values, strict=True)
    )
    contracted = execute_contraction(prepare_contraction(plan, arrays))
    evidence = PEPSContractionEvidence(
        state.state_id,
        canonical_fingerprint(
            {
                "kind": "exact-peps-amplitude",
                "state": state.state_id,
                "configuration": values,
                "plan": plan.plan_id,
            }
        ),
        "exact-basis-amplitude",
        True,
        "exact finite OBC PEPS basis amplitude",
        False,
        sum(prod(shape) for shape in shapes),
        contracted.evidence,
    )
    return PEPSContractionResult(contracted.value, evidence)


def contract_pepo_trace_exact(
    operator: PEPO,
    /,
    *,
    resources: ContractionResourcePolicy | None = None,
) -> PEPSContractionResult:
    """Exactly contract the trace of a finite OBC PEPO."""

    if any(
        output != input_
        for output, input_ in zip(
            operator.output_dimensions, operator.input_dimensions, strict=True
        )
    ):
        raise ValueError(
            "PEPO trace requires matching local output and input dimensions."
        )
    shapes = tuple(
        tuple(int(value) for value in tensor.shape[:4]) for tensor in operator.tensors
    )
    structure = _grid_structure(shapes, operator.rows, operator.columns)
    plan = plan_contraction(
        structure,
        precision=operator.precision,
        resources=resources,
        dtype=str(operator.tensors[0].dtype),
    )
    arrays = tuple(
        oe.contract("urdlpp->urdl", tensor, optimize=False) for tensor in operator.tensors
    )
    contracted = execute_contraction(prepare_contraction(plan, arrays))
    evidence = PEPSContractionEvidence(
        operator.operator_id,
        canonical_fingerprint(
            {
                "kind": "exact-pepo-trace",
                "operator": operator.operator_id,
                "plan": plan.plan_id,
            }
        ),
        "exact-pepo-trace",
        True,
        "exact finite OBC PEPO trace",
        False,
        sum(prod(shape) for shape in shapes),
        contracted.evidence,
    )
    return PEPSContractionResult(contracted.value, evidence)


__all__ = [
    "PEPO",
    "PEPS",
    "PEPSContractionEvidence",
    "PEPSContractionResult",
    "contract_pepo_trace_exact",
    "contract_peps_exact",
    "peps_amplitude",
]
