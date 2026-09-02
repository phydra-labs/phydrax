#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._transform_line import PreparedTransformLineSolve, TransformLineSolveResult


StructuredDistribution: TypeAlias = Literal["transverse-batch", "split-line"]
SplitLineAlgorithm: TypeAlias = Literal["partitioned-thomas", "spike", "pcr"]
StructuredAlgorithm: TypeAlias = Literal["local", "partitioned-thomas", "spike", "pcr"]


def _partition_bounds(size: int, count: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    base, remainder = divmod(size, count)
    sizes = tuple(base + (index < remainder) for index in range(count))
    starts: list[int] = []
    offset = 0
    for part_size in sizes:
        starts.append(offset)
        offset += part_size
    return tuple(starts), sizes


def _deterministic_sum(value: Array, axis: int = -1) -> Array:
    """Fixed binary-tree sum independent of backend reduction scheduling."""
    moved = jnp.moveaxis(value, axis, -1)
    size = int(moved.shape[-1])
    padded_size = 1 << max(0, (size - 1).bit_length())
    if padded_size != size:
        moved = jnp.pad(moved, [(0, 0)] * (moved.ndim - 1) + [(0, padded_size - size)])
    while padded_size > 1:
        moved = moved[..., 0::2] + moved[..., 1::2]
        padded_size //= 2
    return moved[..., 0]


def _deterministic_norm(value: Array) -> Array:
    flat = value.reshape((-1,))
    return jnp.sqrt(jnp.real(_deterministic_sum(jnp.conj(flat) * flat)))


def _tridiagonal_action(
    lower: Array, diagonal: Array, upper: Array, value: Array
) -> Array:
    result = diagonal * value
    if int(value.shape[-1]) > 1:
        result = result.at[..., 1:].add(lower * value[..., :-1])
        result = result.at[..., :-1].add(upper * value[..., 1:])
    return result


def _factor_tridiagonal(
    lower: Array, diagonal: Array, upper: Array
) -> tuple[Array, Array]:
    size = int(diagonal.size)
    pivots = jnp.zeros_like(diagonal).at[0].set(diagonal[0])
    multipliers = jnp.zeros((max(size - 1, 0),), dtype=diagonal.dtype)
    tiny = jnp.finfo(jnp.real(diagonal).dtype).tiny
    for index in range(1, size):
        previous = pivots[index - 1]
        safe = jnp.where(jnp.abs(previous) > tiny, previous, jnp.ones_like(previous))
        multiplier = lower[index - 1] / safe
        multipliers = multipliers.at[index - 1].set(multiplier)
        pivots = pivots.at[index].set(diagonal[index] - multiplier * upper[index - 1])
    return pivots, multipliers


def _solve_factored(pivots: Array, multipliers: Array, upper: Array, rhs: Array) -> Array:
    """Solve one factored line; the final axis is the physical line."""
    size = int(pivots.size)
    result = rhs
    for index in range(1, size):
        result = result.at[..., index].add(
            -multipliers[index - 1] * result[..., index - 1]
        )
    tiny = jnp.finfo(jnp.real(pivots).dtype).tiny
    safe_last = jnp.where(
        jnp.abs(pivots[-1]) > tiny, pivots[-1], jnp.ones_like(pivots[-1])
    )
    result = result.at[..., -1].set(result[..., -1] / safe_last)
    for index in reversed(range(size - 1)):
        safe = jnp.where(
            jnp.abs(pivots[index]) > tiny, pivots[index], jnp.ones_like(pivots[index])
        )
        result = result.at[..., index].set(
            (result[..., index] - upper[index] * result[..., index + 1]) / safe
        )
    return result


def _inverse_2x2(matrix: Array, tolerance: float) -> tuple[Array, Array]:
    a, b = matrix[..., 0, 0], matrix[..., 0, 1]
    c, d = matrix[..., 1, 0], matrix[..., 1, 1]
    determinant = a * d - b * c
    safe = jnp.where(
        jnp.abs(determinant) > tolerance, determinant, jnp.ones_like(determinant)
    )
    inverse = (
        jnp.stack((d, -b, -c, a), axis=-1).reshape(matrix.shape) / safe[..., None, None]
    )
    return inverse, determinant


class LinePartitionMetadata(StrictModule, NonTrainableState):
    """Stable ownership of contiguous line intervals, including uneven tails."""

    starts: tuple[int, ...] = eqx.field(static=True)
    sizes: tuple[int, ...] = eqx.field(static=True)
    stops: tuple[int, ...] = eqx.field(static=True)
    partition_count: int = eqx.field(static=True)
    global_size: int = eqx.field(static=True)
    minimum_size: int = eqx.field(static=True)
    maximum_size: int = eqx.field(static=True)
    uneven: bool = eqx.field(static=True)
    metadata_id: str = eqx.field(static=True)

    def __init__(self, global_size: int, partition_count: int, /):
        size = int(global_size)
        count = int(partition_count)
        if size < 1 or count < 1 or count > size:
            raise ValueError(
                "Contiguous partitioning requires 1 <= partition_count <= global_size."
            )
        starts, sizes = _partition_bounds(size, count)
        stops = tuple(
            start + part_size for start, part_size in zip(starts, sizes, strict=True)
        )
        self.starts = starts
        self.sizes = sizes
        self.stops = stops
        self.partition_count = count
        self.global_size = size
        self.minimum_size = min(sizes)
        self.maximum_size = max(sizes)
        self.uneven = self.minimum_size != self.maximum_size
        self.metadata_id = canonical_fingerprint(
            {
                "kind": "contiguous-line-partition",
                "size": size,
                "starts": starts,
                "sizes": sizes,
            }
        )


class StructuredSolveResourceEstimate(StrictModule, NonTrainableState):
    factor_bytes: int = eqx.field(static=True)
    spike_bytes: int = eqx.field(static=True)
    reduced_interface_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    total_bytes: int = eqx.field(static=True)
    maximum_bytes: int = eqx.field(static=True)
    within_budget: bool = eqx.field(static=True)


class StructuredSolveCommunicationEvidence(StrictModule, NonTrainableState):
    neighbor_rounds: int = eqx.field(static=True)
    global_rounds: int = eqx.field(static=True)
    scalar_values_per_line: int = eqx.field(static=True)
    deterministic_reduction: bool = eqx.field(static=True)
    host_gather: bool = eqx.field(static=True)
    communication_id: str = eqx.field(static=True)


class StructuredSolvePreparationEvidence(StrictModule, NonTrainableState):
    minimum_pivot: Array
    minimum_reduced_determinant: Array
    factor_residual: Array
    schur_residual: Array
    finite: Array
    resources: StructuredSolveResourceEstimate
    communication: StructuredSolveCommunicationEvidence
    algorithm: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class StructuredLineNullspacePolicy(StrictModule, NonTrainableState):
    """Exact declared right/left null vectors, compatibility, and mean-zero gauge."""

    right_null: Array
    left_null: Array
    pin_row: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        line_weights: ArrayLike,
        /,
        *,
        right_null: ArrayLike | None = None,
        pin_row: int = 0,
        policy_id: str | None = None,
    ):
        weights = jnp.asarray(line_weights)
        if (
            weights.ndim != 1
            or weights.size < 2
            or not jnp.issubdtype(weights.dtype, jnp.floating)
        ):
            raise ValueError(
                "line_weights must be a floating rank-one line with at least two entries."
            )
        weights_host = np.asarray(weights)
        if not np.all(np.isfinite(weights_host)) or np.any(weights_host <= 0.0):
            raise ValueError("line_weights must be finite and positive.")
        right = (
            jnp.ones_like(weights)
            if right_null is None
            else jnp.asarray(right_null, dtype=weights.dtype)
        )
        if right.shape != weights.shape or not np.all(np.isfinite(np.asarray(right))):
            raise ValueError("right_null must be finite and match line_weights.")
        normalization = _deterministic_sum(weights * right)
        if (
            not bool(np.isfinite(np.asarray(normalization)))
            or float(np.abs(np.asarray(normalization))) == 0.0
        ):
            raise ValueError(
                "line_weights and right_null must have a finite nonzero pairing."
            )
        left = weights / normalization
        pin = int(pin_row)
        if (
            pin < 0
            or pin >= int(weights.size)
            or float(np.abs(np.asarray(right[pin]))) == 0.0
        ):
            raise ValueError("pin_row must select a nonzero right-null entry.")
        identifier = policy_id or canonical_fingerprint(
            {
                "kind": "distributed-line-nullspace",
                "weights": array_tree_fingerprint(weights),
                "right": array_tree_fingerprint(right),
                "pin_row": pin,
            }
        )
        if not identifier:
            raise ValueError("policy_id must be non-empty.")
        self.right_null = right
        self.left_null = left
        self.pin_row = pin
        self.policy_id = str(identifier)


class StructuredSolveTopologyPlan(StrictModule, NonTrainableState):
    """Fail-closed topology and algorithm choice for structured line solves."""

    partitions: LinePartitionMetadata
    distribution: StructuredDistribution = eqx.field(static=True)
    algorithm: StructuredAlgorithm = eqx.field(static=True)
    line_size: int = eqx.field(static=True)
    transverse_line_count: int = eqx.field(static=True)
    maximum_reduced_interface_size: int = eqx.field(static=True)
    maximum_resource_bytes: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        line_size: int,
        partition_count: int,
        /,
        *,
        distribution: StructuredDistribution = "split-line",
        algorithm: StructuredAlgorithm | None = None,
        transverse_line_count: int = 1,
        maximum_reduced_interface_size: int = 64,
        maximum_resource_bytes: int = 512 * 1024**2,
        tolerance: float = 1.0e-10,
        plan_id: str | None = None,
    ):
        line_size_ = int(line_size)
        count = int(partition_count)
        transverse_count = int(transverse_line_count)
        if distribution not in ("transverse-batch", "split-line"):
            raise ValueError("distribution must be 'transverse-batch' or 'split-line'.")
        algorithm_ = (
            "local"
            if algorithm is None and distribution == "transverse-batch"
            else "partitioned-thomas"
            if algorithm is None
            else algorithm
        )
        if algorithm_ not in ("local", "partitioned-thomas", "spike", "pcr"):
            raise ValueError("Unknown structured line algorithm.")
        if line_size_ < 2 or transverse_count < 1:
            raise ValueError(
                "line_size must be at least two and transverse_line_count positive."
            )
        tolerance_ = float(tolerance)
        budget = int(maximum_resource_bytes)
        interface_bound = int(maximum_reduced_interface_size)
        if (
            not math.isfinite(tolerance_)
            or tolerance_ <= 0.0
            or budget <= 0
            or interface_bound < 2
        ):
            raise ValueError("Tolerance and resource/interface bounds must be positive.")
        partitioned_size = (
            transverse_count if distribution == "transverse-batch" else line_size_
        )
        partitions = LinePartitionMetadata(partitioned_size, count)
        if distribution == "transverse-batch" and algorithm_ != "local":
            raise ValueError(
                "Transverse-batch distribution keeps complete lines on local factors."
            )
        if distribution == "split-line" and algorithm_ == "local":
            raise ValueError("Split-line distribution requires a split-line algorithm.")
        if distribution == "split-line" and partitions.minimum_size < 2:
            raise ValueError(
                "Split-line partitions require at least two line entries each."
            )
        if algorithm_ == "spike" and 2 * count > interface_bound:
            raise ValueError(
                "SPIKE reduced interface exceeds maximum_reduced_interface_size."
            )
        if algorithm_ == "pcr":
            power_of_two = count > 0 and count & (count - 1) == 0
            if not power_of_two or partitions.uneven:
                raise ValueError("PCR requires balanced power-of-two partitions.")
        identifier = plan_id or canonical_fingerprint(
            {
                "kind": "structured-solve-topology",
                "line_size": line_size_,
                "transverse_lines": transverse_count,
                "partitions": partitions.metadata_id,
                "distribution": distribution,
                "algorithm": algorithm_,
                "interface_bound": interface_bound,
                "resource_bound": budget,
                "tolerance": tolerance_,
            }
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.partitions = partitions
        self.distribution = distribution
        self.algorithm = algorithm_
        self.line_size = line_size_
        self.transverse_line_count = transverse_count
        self.maximum_reduced_interface_size = interface_bound
        self.maximum_resource_bytes = budget
        self.tolerance = tolerance_
        self.plan_id = str(identifier)


class DistributedLineSolvePlan(StrictModule, NonTrainableState):
    topology: StructuredSolveTopologyPlan
    lower: Array
    diagonal: Array
    upper: Array
    nullspace: StructuredLineNullspacePolicy | None
    line_axis: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: StructuredSolveTopologyPlan,
        lower: ArrayLike,
        diagonal: ArrayLike,
        upper: ArrayLike,
        /,
        *,
        line_axis: int = -1,
        nullspace: StructuredLineNullspacePolicy | None = None,
        plan_id: str | None = None,
    ):
        if not isinstance(topology, StructuredSolveTopologyPlan):
            raise TypeError("topology must be StructuredSolveTopologyPlan.")
        if topology.distribution != "split-line":
            raise ValueError("DistributedLineSolvePlan requires split-line topology.")
        diagonal_ = jnp.asarray(diagonal)
        if not (
            jnp.issubdtype(diagonal_.dtype, jnp.floating)
            or jnp.issubdtype(diagonal_.dtype, jnp.complexfloating)
        ):
            raise TypeError("Distributed line coefficients require an inexact dtype.")
        lower_ = jnp.asarray(lower, dtype=diagonal_.dtype)
        upper_ = jnp.asarray(upper, dtype=diagonal_.dtype)
        size = topology.line_size
        if (
            diagonal_.shape != (size,)
            or lower_.shape != (size - 1,)
            or upper_.shape != (size - 1,)
        ):
            raise ValueError(
                "Expected diagonal (n,) and lower/upper (n-1,) for topology line_size n."
            )
        if (
            not np.all(np.isfinite(np.asarray(diagonal_)))
            or not np.all(np.isfinite(np.asarray(lower_)))
            or not np.all(np.isfinite(np.asarray(upper_)))
        ):
            raise ValueError("Line coefficients must be finite.")
        if nullspace is not None:
            if not isinstance(nullspace, StructuredLineNullspacePolicy):
                raise TypeError(
                    "nullspace must be StructuredLineNullspacePolicy or None."
                )
            if nullspace.right_null.shape != diagonal_.shape:
                raise ValueError("Nullspace vectors must match topology line_size.")
        identifier = plan_id or canonical_fingerprint(
            {
                "kind": "distributed-line-solve",
                "topology": topology.plan_id,
                "lower": array_tree_fingerprint(lower_),
                "diagonal": array_tree_fingerprint(diagonal_),
                "upper": array_tree_fingerprint(upper_),
                "nullspace": None if nullspace is None else nullspace.policy_id,
                "line_axis": int(line_axis),
            }
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.topology = topology
        self.lower = lower_
        self.diagonal = diagonal_
        self.upper = upper_
        self.nullspace = nullspace
        self.line_axis = int(line_axis)
        self.plan_id = str(identifier)

    def prepare(self, /) -> "PreparedDistributedLineSolve":
        return PreparedDistributedLineSolve(self)


class DistributedLineFactors(StrictModule, NonTrainableState):
    pivots: Array
    multipliers: Array
    local_upper: Array
    left_spikes: Array
    right_spikes: Array
    reduced_diagonal: Array
    reduced_lower: Array
    reduced_upper: Array
    modified_lower: Array
    modified_diagonal: Array
    modified_upper: Array
    factor_id: str = eqx.field(static=True)


class DistributedLineSolveResult(StrictModule):
    value: Array
    candidate: Array
    compatible_rhs: Array
    residual: Array
    residual_norm: Array
    relative_residual: Array
    compatibility_defect: Array
    compatibility_correction_norm: Array
    gauge_defect: Array
    converged: Array
    preparation: StructuredSolvePreparationEvidence
    plan_id: str = eqx.field(static=True)
    factor_id: str = eqx.field(static=True)
    nullspace_policy_id: str | None = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.converged


class PreparedDistributedLineSolve(StrictModule, NonTrainableState):
    plan: DistributedLineSolvePlan
    factors: DistributedLineFactors
    evidence: StructuredSolvePreparationEvidence
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: DistributedLineSolvePlan, /):
        if not isinstance(plan, DistributedLineSolvePlan):
            raise TypeError("plan must be DistributedLineSolvePlan.")
        topology = plan.topology
        count = topology.partitions.partition_count
        maximum_size = topology.partitions.maximum_size
        dtype = plan.diagonal.dtype
        itemsize = np.dtype(dtype).itemsize
        factor_scalars = count * (2 * maximum_size - 1)
        spike_scalars = 2 * count * maximum_size
        reduced_scalars = 12 * count
        if topology.algorithm == "spike":
            reduced_scalars += (2 * count) ** 2
        workspace_scalars = 8 * topology.line_size + 8 * count
        resources = StructuredSolveResourceEstimate(
            factor_bytes=factor_scalars * itemsize,
            spike_bytes=spike_scalars * itemsize,
            reduced_interface_bytes=reduced_scalars * itemsize,
            workspace_bytes=workspace_scalars * itemsize,
            total_bytes=(
                factor_scalars + spike_scalars + reduced_scalars + workspace_scalars
            )
            * itemsize,
            maximum_bytes=topology.maximum_resource_bytes,
            within_budget=(
                factor_scalars + spike_scalars + reduced_scalars + workspace_scalars
            )
            * itemsize
            <= topology.maximum_resource_bytes,
        )
        if not resources.within_budget:
            raise ValueError(
                "Distributed line factors and workspace exceed maximum_resource_bytes."
            )

        lower = plan.lower
        diagonal = plan.diagonal
        upper = plan.upper
        if plan.nullspace is not None:
            policy = plan.nullspace
            right_residual = _tridiagonal_action(
                lower, diagonal, upper, policy.right_null
            )
            left_residual = _tridiagonal_action(upper, diagonal, lower, policy.left_null)
            scale = jnp.maximum(
                1.0,
                jnp.max(jnp.abs(diagonal))
                + jnp.max(jnp.abs(lower))
                + jnp.max(jnp.abs(upper)),
            )
            null_valid = (
                jnp.max(jnp.abs(right_residual)) <= topology.tolerance * scale
            ) & (jnp.max(jnp.abs(left_residual)) <= topology.tolerance * scale)
            if not bool(np.asarray(null_valid)):
                raise ValueError(
                    "Declared right/left null vectors failed exact action evidence."
                )
            pin = policy.pin_row
            diagonal = diagonal.at[pin].set(1.0)
            if pin > 0:
                lower = lower.at[pin - 1].set(0.0)
            if pin < topology.line_size - 1:
                upper = upper.at[pin].set(0.0)

        pivots = jnp.ones((count, maximum_size), dtype=dtype)
        multipliers = jnp.zeros((count, max(maximum_size - 1, 1)), dtype=dtype)
        local_upper = jnp.zeros((count, max(maximum_size - 1, 1)), dtype=dtype)
        left_spikes = jnp.zeros((count, maximum_size), dtype=dtype)
        right_spikes = jnp.zeros((count, maximum_size), dtype=dtype)
        reduced_diagonal = jnp.broadcast_to(jnp.eye(2, dtype=dtype), (count, 2, 2))
        reduced_lower = jnp.zeros((count, 2, 2), dtype=dtype)
        reduced_upper = jnp.zeros((count, 2, 2), dtype=dtype)
        minimum_pivot = jnp.asarray(jnp.inf, dtype=jnp.real(diagonal).dtype)
        factor_residual = jnp.asarray(0.0, dtype=jnp.real(diagonal).dtype)
        for part, (start, size) in enumerate(
            zip(topology.partitions.starts, topology.partitions.sizes, strict=True)
        ):
            stop = start + size
            local_lower = lower[start : stop - 1]
            local_diagonal = diagonal[start:stop]
            local_upper_part = upper[start : stop - 1]
            local_pivots, local_multipliers = _factor_tridiagonal(
                local_lower, local_diagonal, local_upper_part
            )
            pivots = pivots.at[part, :size].set(local_pivots)
            multipliers = multipliers.at[part, : size - 1].set(local_multipliers)
            local_upper = local_upper.at[part, : size - 1].set(local_upper_part)
            reconstructed = local_pivots
            reconstructed = reconstructed.at[1:].add(local_multipliers * local_upper_part)
            diagonal_residual = jnp.max(jnp.abs(reconstructed - local_diagonal))
            lower_residual = jnp.max(
                jnp.abs(local_multipliers * local_pivots[:-1] - local_lower)
            )
            factor_residual = jnp.maximum(
                factor_residual, jnp.maximum(diagonal_residual, lower_residual)
            )
            minimum_pivot = jnp.minimum(minimum_pivot, jnp.min(jnp.abs(local_pivots)))
            left_rhs = jnp.zeros((size,), dtype=dtype)
            right_rhs = jnp.zeros((size,), dtype=dtype)
            if part > 0:
                left_rhs = left_rhs.at[0].set(lower[start - 1])
            if part < count - 1:
                right_rhs = right_rhs.at[-1].set(upper[stop - 1])
            left_response = _solve_factored(
                local_pivots, local_multipliers, local_upper_part, left_rhs
            )
            right_response = _solve_factored(
                local_pivots, local_multipliers, local_upper_part, right_rhs
            )
            left_spikes = left_spikes.at[part, :size].set(left_response)
            right_spikes = right_spikes.at[part, :size].set(right_response)
            if part > 0:
                reduced_lower = reduced_lower.at[part, :, 1].set(
                    left_response[jnp.asarray([0, size - 1])]
                )
            if part < count - 1:
                reduced_upper = reduced_upper.at[part, :, 0].set(
                    right_response[jnp.asarray([0, size - 1])]
                )

        minimum_reduced_determinant = _minimum_reduced_determinant(
            reduced_lower,
            reduced_diagonal,
            reduced_upper,
            topology.algorithm,
            topology.tolerance,
        )
        schur_residual = jnp.asarray(0.0, dtype=jnp.real(diagonal).dtype)
        finite = (
            jnp.all(jnp.isfinite(pivots))
            & jnp.all(jnp.isfinite(left_spikes))
            & jnp.all(jnp.isfinite(right_spikes))
            & jnp.isfinite(minimum_reduced_determinant)
            & (minimum_pivot > topology.tolerance)
            & (minimum_reduced_determinant > topology.tolerance)
        )
        if not bool(np.asarray(finite)):
            raise ValueError(
                "A local pivot or reduced Schur determinant is below tolerance."
            )
        communication = _communication_evidence(topology)
        evidence_id = canonical_fingerprint(
            {
                "kind": "structured-solve-preparation",
                "plan": plan.plan_id,
                "algorithm": topology.algorithm,
                "resources": resources.total_bytes,
                "communication": communication.communication_id,
            }
        )
        factor_id = canonical_fingerprint(
            {
                "kind": "distributed-line-factors",
                "plan": plan.plan_id,
                "evidence": evidence_id,
            }
        )
        self.plan = plan
        self.factors = DistributedLineFactors(
            pivots=pivots,
            multipliers=multipliers,
            local_upper=local_upper,
            left_spikes=left_spikes,
            right_spikes=right_spikes,
            reduced_diagonal=reduced_diagonal,
            reduced_lower=reduced_lower,
            reduced_upper=reduced_upper,
            modified_lower=lower,
            modified_diagonal=diagonal,
            modified_upper=upper,
            factor_id=factor_id,
        )
        self.evidence = StructuredSolvePreparationEvidence(
            minimum_pivot=minimum_pivot,
            minimum_reduced_determinant=minimum_reduced_determinant,
            factor_residual=factor_residual,
            schur_residual=schur_residual,
            finite=finite,
            resources=resources,
            communication=communication,
            algorithm=topology.algorithm,
            evidence_id=evidence_id,
        )
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-distributed-line", "factors": factor_id}
        )

    def solve(self, right_hand_side: ArrayLike, /) -> DistributedLineSolveResult:
        rhs = jnp.asarray(right_hand_side)
        axis = self.plan.line_axis % rhs.ndim
        if rhs.shape[axis] != self.plan.topology.line_size:
            raise ValueError("RHS line axis does not match topology line_size.")
        moved = jnp.moveaxis(rhs, axis, -1)
        batch_shape = moved.shape[:-1]
        batch_rhs = moved.reshape((-1, moved.shape[-1])).astype(
            self.factors.modified_diagonal.dtype
        )
        compatible = batch_rhs
        compatibility_defect = jnp.asarray(0.0, dtype=jnp.real(batch_rhs).dtype)
        compatibility_correction_norm = jnp.asarray(0.0, dtype=jnp.real(batch_rhs).dtype)
        policy = self.plan.nullspace
        if policy is not None:
            means = _deterministic_sum(jnp.conj(policy.left_null) * batch_rhs, axis=-1)
            compatibility_correction_norm = jnp.max(jnp.abs(means))
            compatible = batch_rhs - means[:, None] * policy.right_null
            compatible = compatible.at[:, policy.pin_row].set(0.0)
        candidate_batch = self._solve_compatible(compatible)
        gauge_defect = jnp.asarray(0.0, dtype=jnp.real(batch_rhs).dtype)
        if policy is not None:
            gauge_means = _deterministic_sum(
                jnp.conj(policy.left_null) * candidate_batch, axis=-1
            )
            candidate_batch = candidate_batch - gauge_means[:, None] * policy.right_null
            gauge_defect = jnp.max(
                jnp.abs(
                    _deterministic_sum(
                        jnp.conj(policy.left_null) * candidate_batch, axis=-1
                    )
                )
            )
        physical_compatible = compatible
        if policy is not None:
            physical_compatible = (
                batch_rhs
                - _deterministic_sum(jnp.conj(policy.left_null) * batch_rhs, axis=-1)[
                    :, None
                ]
                * policy.right_null
            )
            compatibility_defect = jnp.max(
                jnp.abs(
                    _deterministic_sum(
                        jnp.conj(policy.left_null) * physical_compatible, axis=-1
                    )
                )
            )
        residual_batch = (
            _tridiagonal_action(
                self.plan.lower, self.plan.diagonal, self.plan.upper, candidate_batch
            )
            - physical_compatible
        )
        residual_norm = _deterministic_norm(residual_batch)
        rhs_norm = _deterministic_norm(physical_compatible)
        relative = residual_norm / jnp.maximum(1.0, rhs_norm)
        converged = (
            self.evidence.finite
            & jnp.all(jnp.isfinite(candidate_batch))
            & (relative <= self.plan.topology.tolerance)
            & (gauge_defect <= self.plan.topology.tolerance)
            & (compatibility_defect <= self.plan.topology.tolerance)
        )
        candidate = jnp.moveaxis(
            candidate_batch.reshape(batch_shape + (moved.shape[-1],)), -1, axis
        )
        residual = jnp.moveaxis(
            residual_batch.reshape(batch_shape + (moved.shape[-1],)), -1, axis
        )
        compatible_rhs = jnp.moveaxis(
            physical_compatible.reshape(batch_shape + (moved.shape[-1],)), -1, axis
        )
        value = jnp.where(converged, candidate, jnp.zeros_like(candidate))
        return DistributedLineSolveResult(
            value=value,
            candidate=candidate,
            compatible_rhs=compatible_rhs,
            residual=residual,
            residual_norm=residual_norm,
            relative_residual=relative,
            compatibility_defect=compatibility_defect,
            compatibility_correction_norm=compatibility_correction_norm,
            gauge_defect=gauge_defect,
            converged=converged,
            preparation=self.evidence,
            plan_id=self.plan.plan_id,
            factor_id=self.factors.factor_id,
            nullspace_policy_id=None if policy is None else policy.policy_id,
        )

    def _solve_compatible(self, rhs: Array) -> Array:
        topology = self.plan.topology
        count = topology.partitions.partition_count
        local_solutions: list[Array] = []
        reduced_rhs = jnp.zeros((rhs.shape[0], count, 2), dtype=rhs.dtype)
        for part, (start, size) in enumerate(
            zip(topology.partitions.starts, topology.partitions.sizes, strict=True)
        ):
            local = _solve_factored(
                self.factors.pivots[part, :size],
                self.factors.multipliers[part, : size - 1],
                self.factors.local_upper[part, : size - 1],
                rhs[:, start : start + size],
            )
            local_solutions.append(local)
            reduced_rhs = reduced_rhs.at[:, part, 0].set(local[:, 0])
            reduced_rhs = reduced_rhs.at[:, part, 1].set(local[:, -1])
        if topology.algorithm == "partitioned-thomas":
            interface = _block_thomas(
                self.factors.reduced_lower,
                self.factors.reduced_diagonal,
                self.factors.reduced_upper,
                reduced_rhs,
                topology.tolerance,
            )
        elif topology.algorithm == "spike":
            interface = _spike_reduced_solve(
                self.factors.reduced_lower,
                self.factors.reduced_diagonal,
                self.factors.reduced_upper,
                reduced_rhs,
                topology.tolerance,
            )
        else:
            interface = _block_pcr(
                self.factors.reduced_lower,
                self.factors.reduced_diagonal,
                self.factors.reduced_upper,
                reduced_rhs,
                topology.tolerance,
            )
        result = jnp.zeros_like(rhs)
        for part, (start, size) in enumerate(
            zip(topology.partitions.starts, topology.partitions.sizes, strict=True)
        ):
            local = local_solutions[part]
            if part > 0:
                local = (
                    local
                    - self.factors.left_spikes[part, :size]
                    * interface[:, part - 1, 1, None]
                )
            if part < count - 1:
                local = (
                    local
                    - self.factors.right_spikes[part, :size]
                    * interface[:, part + 1, 0, None]
                )
            result = result.at[:, start : start + size].set(local)
        return result


def _minimum_reduced_determinant(
    lower: Array,
    diagonal: Array,
    upper: Array,
    algorithm: SplitLineAlgorithm,
    tolerance: float,
) -> Array:
    """Minimum elimination determinant for the selected reduced-interface path."""
    count = int(diagonal.shape[0])
    minimum = jnp.asarray(jnp.inf, dtype=jnp.real(diagonal).dtype)
    if algorithm == "partitioned-thomas":
        modified = diagonal
        for index in range(count):
            inverse, determinant = _inverse_2x2(modified[index], tolerance)
            minimum = jnp.minimum(minimum, jnp.abs(determinant))
            if index + 1 < count:
                multiplier = contract("ij,jk->ik", lower[index + 1], inverse)
                modified = modified.at[index + 1].add(
                    -contract("ij,jk->ik", multiplier, upper[index])
                )
        return minimum
    if algorithm == "spike":
        matrix = jnp.zeros((2 * count, 2 * count), dtype=diagonal.dtype)
        for part in range(count):
            location = slice(2 * part, 2 * part + 2)
            matrix = matrix.at[location, location].set(diagonal[part])
            if part > 0:
                matrix = matrix.at[location, slice(2 * part - 2, 2 * part)].set(
                    lower[part]
                )
            if part < count - 1:
                matrix = matrix.at[location, slice(2 * part + 2, 2 * part + 4)].set(
                    upper[part]
                )
        for pivot in range(2 * count):
            value = matrix[pivot, pivot]
            minimum = jnp.minimum(minimum, jnp.abs(value))
            safe = jnp.where(jnp.abs(value) > tolerance, value, jnp.ones_like(value))
            for row in range(pivot + 1, 2 * count):
                multiplier = matrix[row, pivot] / safe
                matrix = matrix.at[row, pivot:].add(-multiplier * matrix[pivot, pivot:])
        return minimum
    stride = 1
    current_lower, current_diagonal, current_upper = lower, diagonal, upper
    while stride < count:
        next_lower = jnp.zeros_like(current_lower)
        next_diagonal = current_diagonal
        next_upper = jnp.zeros_like(current_upper)
        for index in range(count):
            if index - stride >= 0:
                inverse, determinant = _inverse_2x2(
                    current_diagonal[index - stride], tolerance
                )
                minimum = jnp.minimum(minimum, jnp.abs(determinant))
                alpha = -contract("ij,jk->ik", current_lower[index], inverse)
                next_diagonal = next_diagonal.at[index].add(
                    contract("ij,jk->ik", alpha, current_upper[index - stride])
                )
                next_lower = next_lower.at[index].set(
                    contract("ij,jk->ik", alpha, current_lower[index - stride])
                )
            if index + stride < count:
                inverse, determinant = _inverse_2x2(
                    current_diagonal[index + stride], tolerance
                )
                minimum = jnp.minimum(minimum, jnp.abs(determinant))
                beta = -contract("ij,jk->ik", current_upper[index], inverse)
                next_diagonal = next_diagonal.at[index].add(
                    contract("ij,jk->ik", beta, current_lower[index + stride])
                )
                next_upper = next_upper.at[index].set(
                    contract("ij,jk->ik", beta, current_upper[index + stride])
                )
        current_lower = next_lower
        current_diagonal = next_diagonal
        current_upper = next_upper
        stride *= 2
    _, determinant = _inverse_2x2(current_diagonal, tolerance)
    return jnp.minimum(minimum, jnp.min(jnp.abs(determinant)))


def _block_thomas(
    lower: Array, diagonal: Array, upper: Array, rhs: Array, tolerance: float
) -> Array:
    count = int(diagonal.shape[0])
    modified_diagonal = diagonal
    modified_rhs = rhs
    for index in range(1, count):
        inverse, _ = _inverse_2x2(modified_diagonal[index - 1], tolerance)
        multiplier = contract("ij,jk->ik", lower[index], inverse)
        modified_diagonal = modified_diagonal.at[index].add(
            -contract("ij,jk->ik", multiplier, upper[index - 1])
        )
        modified_rhs = modified_rhs.at[:, index, :].add(
            -contract("ij,bj->bi", multiplier, modified_rhs[:, index - 1, :])
        )
    result = jnp.zeros_like(rhs)
    inverse, _ = _inverse_2x2(modified_diagonal[-1], tolerance)
    result = result.at[:, -1, :].set(
        contract("ij,bj->bi", inverse, modified_rhs[:, -1, :])
    )
    for index in reversed(range(count - 1)):
        inverse, _ = _inverse_2x2(modified_diagonal[index], tolerance)
        corrected = modified_rhs[:, index, :] - contract(
            "ij,bj->bi", upper[index], result[:, index + 1, :]
        )
        result = result.at[:, index, :].set(contract("ij,bj->bi", inverse, corrected))
    return result


def _dense_elimination(matrix: Array, rhs: Array, tolerance: float) -> Array:
    size = int(matrix.shape[0])
    transformed = matrix
    value = rhs
    for pivot in range(size):
        safe = jnp.where(
            jnp.abs(transformed[pivot, pivot]) > tolerance,
            transformed[pivot, pivot],
            jnp.ones_like(transformed[pivot, pivot]),
        )
        transformed = transformed.at[pivot, pivot:].set(transformed[pivot, pivot:] / safe)
        value = value.at[:, pivot].set(value[:, pivot] / safe)
        for row in range(pivot + 1, size):
            multiplier = transformed[row, pivot]
            transformed = transformed.at[row, pivot:].add(
                -multiplier * transformed[pivot, pivot:]
            )
            value = value.at[:, row].add(-multiplier * value[:, pivot])
    result = jnp.zeros_like(value)
    for row in reversed(range(size)):
        correction = (
            _deterministic_sum(
                transformed[row, row + 1 :] * result[:, row + 1 :], axis=-1
            )
            if row + 1 < size
            else jnp.zeros((value.shape[0],), dtype=value.dtype)
        )
        result = result.at[:, row].set(value[:, row] - correction)
    return result


def _spike_reduced_solve(
    lower: Array, diagonal: Array, upper: Array, rhs: Array, tolerance: float
) -> Array:
    count = int(diagonal.shape[0])
    matrix = jnp.zeros((2 * count, 2 * count), dtype=diagonal.dtype)
    for part in range(count):
        location = slice(2 * part, 2 * part + 2)
        matrix = matrix.at[location, location].set(diagonal[part])
        if part > 0:
            matrix = matrix.at[location, slice(2 * part - 2, 2 * part)].set(lower[part])
        if part < count - 1:
            matrix = matrix.at[location, slice(2 * part + 2, 2 * part + 4)].set(
                upper[part]
            )
    solved = _dense_elimination(matrix, rhs.reshape((rhs.shape[0], 2 * count)), tolerance)
    return solved.reshape(rhs.shape)


def _block_pcr(
    lower: Array, diagonal: Array, upper: Array, rhs: Array, tolerance: float
) -> Array:
    count = int(diagonal.shape[0])
    stride = 1
    current_lower, current_diagonal, current_upper, current_rhs = (
        lower,
        diagonal,
        upper,
        rhs,
    )
    while stride < count:
        next_lower = jnp.zeros_like(current_lower)
        next_diagonal = current_diagonal
        next_upper = jnp.zeros_like(current_upper)
        next_rhs = current_rhs
        for index in range(count):
            if index - stride >= 0:
                inverse, _ = _inverse_2x2(current_diagonal[index - stride], tolerance)
                alpha = -contract("ij,jk->ik", current_lower[index], inverse)
                next_diagonal = next_diagonal.at[index].add(
                    contract("ij,jk->ik", alpha, current_upper[index - stride])
                )
                next_rhs = next_rhs.at[:, index, :].add(
                    contract("ij,bj->bi", alpha, current_rhs[:, index - stride, :])
                )
                next_lower = next_lower.at[index].set(
                    contract("ij,jk->ik", alpha, current_lower[index - stride])
                )
            if index + stride < count:
                inverse, _ = _inverse_2x2(current_diagonal[index + stride], tolerance)
                beta = -contract("ij,jk->ik", current_upper[index], inverse)
                next_diagonal = next_diagonal.at[index].add(
                    contract("ij,jk->ik", beta, current_lower[index + stride])
                )
                next_rhs = next_rhs.at[:, index, :].add(
                    contract("ij,bj->bi", beta, current_rhs[:, index + stride, :])
                )
                next_upper = next_upper.at[index].set(
                    contract("ij,jk->ik", beta, current_upper[index + stride])
                )
        current_lower, current_diagonal, current_upper, current_rhs = (
            next_lower,
            next_diagonal,
            next_upper,
            next_rhs,
        )
        stride *= 2
    inverse, _ = _inverse_2x2(current_diagonal, tolerance)
    return contract("pij,bpj->bpi", inverse, current_rhs)


def _communication_evidence(
    topology: StructuredSolveTopologyPlan,
) -> StructuredSolveCommunicationEvidence:
    count = topology.partitions.partition_count
    if topology.distribution == "transverse-batch":
        neighbor_rounds, global_rounds, scalars = 0, 0, 0
    elif topology.algorithm == "partitioned-thomas":
        neighbor_rounds, global_rounds, scalars = (
            2 * max(count - 1, 0),
            0,
            8 * max(count - 1, 0),
        )
    elif topology.algorithm == "spike":
        neighbor_rounds, global_rounds, scalars = 2, 1, 4 * count * count
    else:
        stages = int(math.log2(count)) if count > 1 else 0
        neighbor_rounds, global_rounds, scalars = stages, 0, 16 * count * stages
    identifier = canonical_fingerprint(
        {
            "kind": "structured-line-communication",
            "plan": topology.plan_id,
            "neighbor": neighbor_rounds,
            "global": global_rounds,
            "scalars": scalars,
        }
    )
    return StructuredSolveCommunicationEvidence(
        neighbor_rounds=neighbor_rounds,
        global_rounds=global_rounds,
        scalar_values_per_line=scalars,
        deterministic_reduction=True,
        host_gather=False,
        communication_id=identifier,
    )


class PreparedTransverseBatchLineSolve(StrictModule, NonTrainableState):
    """Line-contiguous transverse sharding over an existing local transform-line solve."""

    topology: StructuredSolveTopologyPlan
    local: PreparedTransformLineSolve
    communication: StructuredSolveCommunicationEvidence
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self, topology: StructuredSolveTopologyPlan, local: PreparedTransformLineSolve, /
    ):
        if (
            not isinstance(topology, StructuredSolveTopologyPlan)
            or topology.distribution != "transverse-batch"
        ):
            raise ValueError(
                "Transverse adapter requires transverse-batch StructuredSolveTopologyPlan."
            )
        if not isinstance(local, PreparedTransformLineSolve):
            raise TypeError("local must be PreparedTransformLineSolve.")
        if (
            local.resources.line_size != topology.line_size
            or local.resources.line_count != topology.transverse_line_count
        ):
            raise ValueError(
                "Topology line size/count must exactly match existing transform-line factors."
            )
        self.topology = topology
        self.local = local
        self.communication = _communication_evidence(topology)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-transverse-batch-line",
                "topology": topology.plan_id,
                "local": local.prepared_id,
            }
        )

    def solve(self, right_hand_side: ArrayLike, /) -> TransformLineSolveResult:
        return self.local.solve(right_hand_side)


class ExtrudedAxisInvarianceCertificate(StrictModule, NonTrainableState):
    """Fail-closed proof that an extruded-axis transform commutes with every coupling."""

    geometry_defect: Array
    metric_defect: Array
    coefficient_defect: Array
    interface_defect: Array
    certified: Array
    tolerance: float = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry_defect: ArrayLike,
        metric_defect: ArrayLike,
        coefficient_defect: ArrayLike,
        interface_defect: ArrayLike,
        /,
        *,
        tolerance: float = 1.0e-12,
        certificate_id: str | None = None,
    ):
        tolerance_ = float(tolerance)
        if not math.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        defects = tuple(
            jnp.asarray(value)
            for value in (
                geometry_defect,
                metric_defect,
                coefficient_defect,
                interface_defect,
            )
        )
        if any(value.shape != () for value in defects):
            raise ValueError("Invariance defects must be scalar evidence.")
        certified = jnp.asarray(True)
        for value in defects:
            certified = certified & jnp.isfinite(value) & (jnp.abs(value) <= tolerance_)
        identifier = certificate_id or canonical_fingerprint(
            {
                "kind": "extruded-axis-invariance",
                "defects": tuple(repr(np.asarray(value).item()) for value in defects),
                "tolerance": tolerance_,
            }
        )
        if not identifier:
            raise ValueError("certificate_id must be non-empty.")
        (
            self.geometry_defect,
            self.metric_defect,
            self.coefficient_defect,
            self.interface_defect,
        ) = defects
        self.certified = certified
        self.tolerance = tolerance_
        self.certificate_id = str(identifier)


class MultiblockExtrudedReductionPlan(StrictModule, NonTrainableState):
    """Certified global multiblock operator with block-direct preconditioning only."""

    local_lower: Array
    local_diagonal: Array
    local_upper: Array
    interface_operator: Array
    mortar_matrix: Array
    certificate: ExtrudedAxisInvarianceCertificate
    tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    maximum_resource_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        local_lower: ArrayLike,
        local_diagonal: ArrayLike,
        local_upper: ArrayLike,
        interface_operator: ArrayLike,
        mortar_matrix: ArrayLike,
        certificate: ExtrudedAxisInvarianceCertificate,
        /,
        *,
        tolerance: float = 1.0e-10,
        maximum_iterations: int = 100,
        maximum_resource_bytes: int = 512 * 1024**2,
        plan_id: str | None = None,
    ):
        if not isinstance(certificate, ExtrudedAxisInvarianceCertificate):
            raise TypeError("certificate must be ExtrudedAxisInvarianceCertificate.")
        if not bool(np.asarray(certificate.certified)):
            raise ValueError(
                "Axis transform requires certified geometry/metric/coefficient/interface invariance."
            )
        diagonal = jnp.asarray(local_diagonal)
        if not (
            jnp.issubdtype(diagonal.dtype, jnp.floating)
            or jnp.issubdtype(diagonal.dtype, jnp.complexfloating)
        ):
            raise TypeError("Multiblock line coefficients require an inexact dtype.")
        lower = jnp.asarray(local_lower, dtype=diagonal.dtype)
        upper = jnp.asarray(local_upper, dtype=diagonal.dtype)
        interface = jnp.asarray(interface_operator, dtype=diagonal.dtype)
        mortar = jnp.asarray(mortar_matrix, dtype=diagonal.dtype)
        if (
            diagonal.ndim != 2
            or lower.shape != (diagonal.shape[0], diagonal.shape[1] - 1)
            or upper.shape != lower.shape
        ):
            raise ValueError(
                "Local block tridiagonal data must have shapes (b,n), (b,n-1), (b,n-1)."
            )
        if interface.ndim != 3 or interface.shape[1:] != diagonal.shape:
            raise ValueError(
                "interface_operator must have shape (interfaces, blocks, line_size)."
            )
        if mortar.shape != (interface.shape[0], interface.shape[0]):
            raise ValueError("mortar_matrix must be square on interface coordinates.")
        arrays = (lower, diagonal, upper, interface, mortar)
        if any(not np.all(np.isfinite(np.asarray(value))) for value in arrays):
            raise ValueError("Multiblock coefficients and interfaces must be finite.")
        tolerance_ = float(tolerance)
        iterations = int(maximum_iterations)
        budget = int(maximum_resource_bytes)
        if (
            not math.isfinite(tolerance_)
            or tolerance_ <= 0.0
            or iterations < 1
            or budget <= 0
        ):
            raise ValueError(
                "Tolerance, maximum_iterations, and resource budget must be positive."
            )
        identifier = plan_id or canonical_fingerprint(
            {
                "kind": "multiblock-extruded-reduction",
                "lower": array_tree_fingerprint(lower),
                "diagonal": array_tree_fingerprint(diagonal),
                "upper": array_tree_fingerprint(upper),
                "interface": array_tree_fingerprint(interface),
                "mortar": array_tree_fingerprint(mortar),
                "certificate": certificate.certificate_id,
                "tolerance": tolerance_,
                "iterations": iterations,
            }
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.local_lower = lower
        self.local_diagonal = diagonal
        self.local_upper = upper
        self.interface_operator = interface
        self.mortar_matrix = mortar
        self.certificate = certificate
        self.tolerance = tolerance_
        self.maximum_iterations = iterations
        self.maximum_resource_bytes = budget
        self.plan_id = str(identifier)

    def prepare(self, /) -> "PreparedMultiblockExtrudedReduction":
        return PreparedMultiblockExtrudedReduction(self)


class MultiblockExtrudedReductionResult(StrictModule):
    value: Array
    candidate: Array
    mortar_value: Array
    residual: Array
    mortar_residual: Array
    residual_norm: Array
    relative_residual: Array
    iterations: Array
    converged: Array
    certificate_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    local_direct_role: str = eqx.field(static=True)


class PreparedMultiblockExtrudedReduction(StrictModule, NonTrainableState):
    plan: MultiblockExtrudedReductionPlan
    pivots: Array
    multipliers: Array
    resources: StructuredSolveResourceEstimate
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: MultiblockExtrudedReductionPlan, /):
        if not isinstance(plan, MultiblockExtrudedReductionPlan):
            raise TypeError("plan must be MultiblockExtrudedReductionPlan.")
        blocks, size = plan.local_diagonal.shape
        itemsize = np.dtype(plan.local_diagonal.dtype).itemsize
        factor_scalars = int(blocks * (2 * size - 1))
        workspace_scalars = int(12 * blocks * size + 4 * plan.interface_operator.shape[0])
        total = (factor_scalars + workspace_scalars) * itemsize
        resources = StructuredSolveResourceEstimate(
            factor_bytes=factor_scalars * itemsize,
            spike_bytes=0,
            reduced_interface_bytes=0,
            workspace_bytes=workspace_scalars * itemsize,
            total_bytes=total,
            maximum_bytes=plan.maximum_resource_bytes,
            within_budget=total <= plan.maximum_resource_bytes,
        )
        if not resources.within_budget:
            raise ValueError(
                "Multiblock factors and iterative workspace exceed maximum_resource_bytes."
            )
        mortar_diagonal = jnp.diag(plan.mortar_matrix)
        if not bool(
            np.asarray(
                jnp.all(jnp.isfinite(mortar_diagonal))
                & jnp.all(jnp.abs(mortar_diagonal) > plan.tolerance)
            )
        ):
            raise ValueError(
                "The iterative mortar preconditioner requires finite nonzero diagonal entries."
            )
        pivots = jnp.zeros_like(plan.local_diagonal)
        multipliers = jnp.zeros_like(plan.local_lower)
        minimum = jnp.asarray(jnp.inf, dtype=jnp.real(plan.local_diagonal).dtype)
        for block in range(blocks):
            block_pivots, block_multipliers = _factor_tridiagonal(
                plan.local_lower[block],
                plan.local_diagonal[block],
                plan.local_upper[block],
            )
            pivots = pivots.at[block].set(block_pivots)
            multipliers = multipliers.at[block].set(block_multipliers)
            minimum = jnp.minimum(minimum, jnp.min(jnp.abs(block_pivots)))
        if not bool(
            np.asarray(jnp.all(jnp.isfinite(pivots)) & (minimum > plan.tolerance))
        ):
            raise ValueError(
                "A multiblock preconditioner factor is singular or nonfinite."
            )
        self.plan = plan
        self.pivots = pivots
        self.multipliers = multipliers
        self.resources = resources
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-multiblock-extruded",
                "plan": plan.plan_id,
                "resources": total,
            }
        )

    def solve(
        self,
        right_hand_side: ArrayLike,
        /,
        *,
        mortar_right_hand_side: ArrayLike | None = None,
    ) -> MultiblockExtrudedReductionResult:
        rhs = jnp.asarray(right_hand_side, dtype=self.plan.local_diagonal.dtype)
        if rhs.shape != self.plan.local_diagonal.shape:
            raise ValueError("right_hand_side must match (blocks, line_size).")
        interface_count = int(self.plan.interface_operator.shape[0])
        mortar_rhs = (
            jnp.zeros((interface_count,), dtype=rhs.dtype)
            if mortar_right_hand_side is None
            else jnp.asarray(mortar_right_hand_side, dtype=rhs.dtype)
        )
        if mortar_rhs.shape != (interface_count,):
            raise ValueError(
                "mortar_right_hand_side must match the global interface coordinate count."
            )
        combined_rhs = jnp.concatenate((rhs.reshape((-1,)), mortar_rhs))
        combined_candidate, iterations = _preconditioned_conjugate_gradient(
            self._apply_global,
            self._apply_block_preconditioner,
            combined_rhs,
            self.plan.maximum_iterations,
            self.plan.tolerance,
        )
        combined_residual = self._apply_global(combined_candidate) - combined_rhs
        primal_size = int(rhs.size)
        candidate = combined_candidate[:primal_size].reshape(rhs.shape)
        mortar_value = combined_candidate[primal_size:]
        residual = combined_residual[:primal_size].reshape(rhs.shape)
        mortar_residual = combined_residual[primal_size:]
        residual_norm = _deterministic_norm(combined_residual)
        relative = residual_norm / jnp.maximum(1.0, _deterministic_norm(combined_rhs))
        converged = jnp.all(jnp.isfinite(combined_candidate)) & (
            relative <= self.plan.tolerance
        )
        value = jnp.where(converged, candidate, jnp.zeros_like(candidate))
        return MultiblockExtrudedReductionResult(
            value=value,
            candidate=candidate,
            mortar_value=mortar_value,
            residual=residual,
            mortar_residual=mortar_residual,
            residual_norm=residual_norm,
            relative_residual=relative,
            iterations=iterations,
            converged=converged,
            certificate_id=self.plan.certificate.certificate_id,
            plan_id=self.plan.plan_id,
            local_direct_role="preconditioner-only",
        )

    def _apply_global(self, value: Array) -> Array:
        primal_size = int(self.plan.local_diagonal.size)
        primal = value[:primal_size].reshape(self.plan.local_diagonal.shape)
        mortar = value[primal_size:]
        local = jax.vmap(_tridiagonal_action)(
            self.plan.local_lower,
            self.plan.local_diagonal,
            self.plan.local_upper,
            primal,
        )
        primal_action = local + contract(
            "ibn,i->bn", self.plan.interface_operator, mortar
        )
        mortar_action = contract(
            "ibn,bn->i", self.plan.interface_operator, primal
        ) + contract("ij,j->i", self.plan.mortar_matrix, mortar)
        return jnp.concatenate((primal_action.reshape((-1,)), mortar_action))

    def _apply_block_preconditioner(self, value: Array) -> Array:
        primal_size = int(self.plan.local_diagonal.size)
        primal = value[:primal_size].reshape(self.plan.local_diagonal.shape)
        solved = []
        for block in range(int(primal.shape[0])):
            solved.append(
                _solve_factored(
                    self.pivots[block],
                    self.multipliers[block],
                    self.plan.local_upper[block],
                    primal[block],
                )
            )
        mortar = value[primal_size:] / jnp.diag(self.plan.mortar_matrix)
        return jnp.concatenate((jnp.stack(solved, axis=0).reshape((-1,)), mortar))


def _tree_inner(left: Array, right: Array) -> Array:
    return jnp.real(_deterministic_sum((jnp.conj(left) * right).reshape((-1,))))


def _preconditioned_conjugate_gradient(
    apply, precondition, rhs: Array, iterations: int, tolerance: float
) -> tuple[Array, Array]:
    value = jnp.zeros_like(rhs)
    residual = rhs - apply(value)
    preconditioned = precondition(residual)
    direction = preconditioned
    rz = _tree_inner(residual, preconditioned)
    active = _deterministic_norm(residual) > tolerance * jnp.maximum(
        1.0, _deterministic_norm(rhs)
    )
    used = jnp.asarray(0, dtype=jnp.int32)
    for _ in range(iterations):
        action = apply(direction)
        denominator = _tree_inner(direction, action)
        safe = jnp.where(
            jnp.abs(denominator) > 0.0, denominator, jnp.ones_like(denominator)
        )
        alpha = jnp.where(active, rz / safe, jnp.zeros_like(rz))
        value = value + alpha * direction
        residual = residual - alpha * action
        preconditioned = precondition(residual)
        rz_next = _tree_inner(residual, preconditioned)
        safe_rz = jnp.where(jnp.abs(rz) > 0.0, rz, jnp.ones_like(rz))
        beta = jnp.where(active, rz_next / safe_rz, jnp.zeros_like(rz))
        direction = preconditioned + beta * direction
        rz = rz_next
        just_active = _deterministic_norm(residual) > tolerance * jnp.maximum(
            1.0, _deterministic_norm(rhs)
        )
        used = used + active.astype(jnp.int32)
        active = active & just_active
    return value, used


__all__ = [
    "DistributedLineFactors",
    "DistributedLineSolvePlan",
    "DistributedLineSolveResult",
    "ExtrudedAxisInvarianceCertificate",
    "LinePartitionMetadata",
    "MultiblockExtrudedReductionPlan",
    "MultiblockExtrudedReductionResult",
    "PreparedDistributedLineSolve",
    "PreparedMultiblockExtrudedReduction",
    "PreparedTransverseBatchLineSolve",
    "SplitLineAlgorithm",
    "StructuredAlgorithm",
    "StructuredDistribution",
    "StructuredLineNullspacePolicy",
    "StructuredSolveCommunicationEvidence",
    "StructuredSolvePreparationEvidence",
    "StructuredSolveResourceEstimate",
    "StructuredSolveTopologyPlan",
]
