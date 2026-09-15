#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite, prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .. import ein
from .._fingerprint import canonical_fingerprint
from .._precision import precision_itemsize, PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ..linalg import HermitianPrecisionPolicy, HermitianSpectrum
from ._contraction import (
    ContractionPlan,
    ContractionPlanCache,
    ContractionResourcePolicy,
    execute_schedule,
    plan_contraction,
    prepare_contraction,
)
from ._precision import TensorNetworkPrecisionPolicy
from ._split import TensorTruncationEvidence, truncated_svd
from ._topology import ContractionLeg, ContractionOperand, ContractionStructure
from ._uniform_square import UniformSquareTensor


class TRGMethod(StrictModule):
    method_id: str = eqx.field(static=True)

    def __init__(self):
        self.method_id = "trg"


class HOTRGMethod(StrictModule):
    first_direction: Literal["vertical", "horizontal"] = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        first_direction: Literal["vertical", "horizontal"] = "vertical",
    ):
        if first_direction not in ("vertical", "horizontal"):
            raise ValueError("first_direction must be 'vertical' or 'horizontal'.")
        self.first_direction = first_direction
        self.method_id = "hotrg"


TensorRenormalizationMethod: TypeAlias = TRGMethod | HOTRGMethod


class TensorRenormalizationStatus(IntEnum):
    SUCCESS = 0
    NONFINITE_INPUT = 1
    ZERO_NORMALIZATION = 2
    FACTORIZATION_FAILED = 3
    NONFINITE_INTERMEDIATE = 4
    TERMINAL_PARTITION_INVALID = 5


class TensorRenormalizationProblem(StrictModule):
    tensor: UniformSquareTensor
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        tensor: UniformSquareTensor,
        /,
        *,
        problem_id: str = "uniform-square-partition-function",
    ):
        if not isinstance(tensor, UniformSquareTensor):
            raise TypeError("tensor must be a UniformSquareTensor.")
        if jnp.issubdtype(tensor.value.dtype, jnp.complexfloating):
            raise TypeError(
                "Tensor renormalization currently supports real partition tensors."
            )
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be nonempty.")
        self.tensor = tensor
        self.problem_id = identifier


class TensorRenormalizationResourcePolicy(StrictModule):
    maximum_tensor_elements: int = eqx.field(static=True)
    maximum_factorization_elements: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    maximum_history_elements: int = eqx.field(static=True)
    contractions: ContractionResourcePolicy
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_tensor_elements: int = 100_000_000,
        maximum_factorization_elements: int = 100_000_000,
        maximum_workspace_bytes: int = 2**31,
        maximum_history_elements: int = 1_000_000,
        contractions: ContractionResourcePolicy | None = None,
    ):
        limits = tuple(
            int(value)
            for value in (
                maximum_tensor_elements,
                maximum_factorization_elements,
                maximum_workspace_bytes,
                maximum_history_elements,
            )
        )
        if any(value < 1 for value in limits):
            raise ValueError("Tensor-renormalization resource limits must be positive.")
        contractions_ = (
            ContractionResourcePolicy() if contractions is None else contractions
        )
        if not isinstance(contractions_, ContractionResourcePolicy):
            raise TypeError("contractions must be ContractionResourcePolicy or None.")
        (
            self.maximum_tensor_elements,
            self.maximum_factorization_elements,
            self.maximum_workspace_bytes,
            self.maximum_history_elements,
        ) = limits
        self.contractions = contractions_
        self.policy_id = canonical_fingerprint(
            {
                "kind": "tensor-renormalization-resources",
                "limits": limits,
                "contractions": contractions_.policy_id,
            }
        )


class TensorRenormalizationPolicy(StrictModule):
    method: TensorRenormalizationMethod
    maximum_bond_dimension: int = eqx.field(static=True)
    steps: int = eqx.field(static=True)
    terminal_imaginary_tolerance: float = eqx.field(static=True)
    terminal_positivity_tolerance: float = eqx.field(static=True)
    resources: TensorRenormalizationResourcePolicy
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: TensorRenormalizationMethod,
        /,
        *,
        maximum_bond_dimension: int,
        steps: int,
        terminal_imaginary_tolerance: float = 1e-10,
        terminal_positivity_tolerance: float = 0.0,
        resources: TensorRenormalizationResourcePolicy | None = None,
    ):
        if not isinstance(method, (TRGMethod, HOTRGMethod)):
            raise TypeError("method must be TRGMethod or HOTRGMethod.")
        capacity = int(maximum_bond_dimension)
        count = int(steps)
        tolerances = (
            float(terminal_imaginary_tolerance),
            float(terminal_positivity_tolerance),
        )
        if capacity < 1 or count < 1:
            raise ValueError("Bond dimension and renormalization steps must be positive.")
        if any(not isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("Terminal tolerances must be finite and nonnegative.")
        resources_ = (
            TensorRenormalizationResourcePolicy() if resources is None else resources
        )
        if not isinstance(resources_, TensorRenormalizationResourcePolicy):
            raise TypeError(
                "resources must be TensorRenormalizationResourcePolicy or None."
            )
        self.method = method
        self.maximum_bond_dimension = capacity
        self.steps = count
        self.terminal_imaginary_tolerance = tolerances[0]
        self.terminal_positivity_tolerance = tolerances[1]
        self.resources = resources_
        self.policy_id = canonical_fingerprint(
            {
                "kind": "tensor-renormalization-policy",
                "method": method.method_id,
                "first_direction": (
                    method.first_direction if isinstance(method, HOTRGMethod) else None
                ),
                "maximum_bond_dimension": capacity,
                "steps": count,
                "terminal_tolerances": tolerances,
                "resources": resources_.policy_id,
            }
        )


class TensorRenormalizationStagePlan(StrictModule):
    kind: str = eqx.field(static=True)
    input_shape: tuple[int, int, int, int] = eqx.field(static=True)
    output_shape: tuple[int, int, int, int] = eqx.field(static=True)
    first_retained_rank: int = eqx.field(static=True)
    second_retained_rank: int = eqx.field(static=True)
    maximum_factorization_elements: int = eqx.field(static=True)
    merge: ContractionPlan | None
    coarse: ContractionPlan
    stage_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: str,
        input_shape: tuple[int, int, int, int],
        output_shape: tuple[int, int, int, int],
        first_retained_rank: int,
        second_retained_rank: int,
        maximum_factorization_elements: int,
        merge: ContractionPlan | None,
        coarse: ContractionPlan,
        /,
    ):
        self.kind = str(kind)
        self.input_shape = tuple(int(value) for value in input_shape)
        self.output_shape = tuple(int(value) for value in output_shape)
        self.first_retained_rank = int(first_retained_rank)
        self.second_retained_rank = int(second_retained_rank)
        self.maximum_factorization_elements = int(maximum_factorization_elements)
        self.merge = merge
        self.coarse = coarse
        self.stage_id = canonical_fingerprint(
            {
                "kind": self.kind,
                "input_shape": self.input_shape,
                "output_shape": self.output_shape,
                "ranks": (self.first_retained_rank, self.second_retained_rank),
                "merge": None if merge is None else merge.plan_id,
                "coarse": coarse.plan_id,
            }
        )


class TensorRenormalizationCostEstimate(StrictModule):
    maximum_tensor_elements: int = eqx.field(static=True)
    maximum_factorization_elements: int = eqx.field(static=True)
    history_elements: int = eqx.field(static=True)
    peak_workspace_bytes: int = eqx.field(static=True)
    estimated_flops: int = eqx.field(static=True)


class TensorRenormalizationPlan(StrictModule):
    policy: TensorRenormalizationPolicy
    stages: tuple[TensorRenormalizationStagePlan, ...]
    cost: TensorRenormalizationCostEstimate
    tensor_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedTensorRenormalization(StrictModule):
    problem: TensorRenormalizationProblem
    plan: TensorRenormalizationPlan
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)


class TensorRenormalizationDiagnostics(StrictModule):
    normalization_scales: Array
    first_discarded_weight_history: Array
    second_discarded_weight_history: Array
    retained_rank_history: Array
    finite_history: Array
    terminal_imaginary_residual: Array
    terminal_positive: Array
    status: Array
    exact: Array
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    claim: str = eqx.field(static=True)
    global_error_bound_claimed: bool = eqx.field(static=True)
    admitted_peak_bytes: int = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(TensorRenormalizationStatus.SUCCESS)


class TensorRenormalizationProvenance(StrictModule):
    method: str = eqx.field(static=True)
    tensor_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: Array


class TensorRenormalizationResult(StrictModule):
    final_tensor: UniformSquareTensor
    log_partition_density: Array
    normalization_contribution: Array
    terminal_correction: Array
    terminal_value: Array
    diagnostics: TensorRenormalizationDiagnostics
    provenance: TensorRenormalizationProvenance

    @property
    def successful(self) -> Array:
        return self.diagnostics.successful


def _operand(identifier: str, labels: tuple[str, ...], dimensions, /):
    return ContractionOperand(
        identifier,
        tuple(
            ContractionLeg(label, int(dimension))
            for label, dimension in zip(labels, dimensions, strict=True)
        ),
    )


def _contraction_plan(
    operands: tuple[ContractionOperand, ...],
    outputs: tuple[str, ...],
    *,
    precision: TensorNetworkPrecisionPolicy,
    resources: TensorRenormalizationResourcePolicy,
    dtype: str,
    cache: ContractionPlanCache,
) -> ContractionPlan:
    return plan_contraction(
        ContractionStructure(operands, outputs),
        precision=precision,
        resources=resources.contractions,
        cache=cache,
        optimizer="greedy",
        dtype=dtype,
    )


def _trg_stage_plan(
    shape: tuple[int, int, int, int],
    capacity: int,
    *,
    precision: TensorNetworkPrecisionPolicy,
    resources: TensorRenormalizationResourcePolicy,
    dtype: str,
    cache: ContractionPlanCache,
) -> TensorRenormalizationStagePlan:
    vertical, horizontal, _, _ = shape
    rank = min(capacity, vertical * horizontal)
    operands = (
        _operand("upper-right", ("u", "r", "U"), (vertical, horizontal, rank)),
        _operand("right-lower", ("r", "d", "R"), (horizontal, vertical, rank)),
        _operand("lower-left", ("D", "d", "l"), (rank, vertical, horizontal)),
        _operand("left-upper", ("L", "l", "u"), (rank, horizontal, vertical)),
    )
    coarse = _contraction_plan(
        operands,
        ("U", "R", "D", "L"),
        precision=precision,
        resources=resources,
        dtype=dtype,
        cache=cache,
    )
    output_shape = (rank, rank, rank, rank)
    matrix_elements = (vertical * horizontal) ** 2
    return TensorRenormalizationStagePlan(
        "trg",
        shape,
        output_shape,
        rank,
        rank,
        matrix_elements,
        None,
        coarse,
    )


def _hotrg_vertical_stage_plan(
    shape: tuple[int, int, int, int],
    capacity: int,
    *,
    precision: TensorNetworkPrecisionPolicy,
    resources: TensorRenormalizationResourcePolicy,
    dtype: str,
    cache: ContractionPlanCache,
) -> TensorRenormalizationStagePlan:
    vertical, horizontal, _, _ = shape
    retained = min(capacity, horizontal * horizontal)
    merge = _contraction_plan(
        (
            _operand("upper", ("u", "rt", "m", "lt"), shape),
            _operand("lower", ("m", "rb", "d", "lb"), shape),
        ),
        ("u", "rt", "rb", "d", "lt", "lb"),
        precision=precision,
        resources=resources,
        dtype=dtype,
        cache=cache,
    )
    merged_shape = (
        vertical,
        horizontal,
        horizontal,
        vertical,
        horizontal,
        horizontal,
    )
    coarse = _contraction_plan(
        (
            _operand(
                "merged",
                ("u", "rt", "rb", "d", "lt", "lb"),
                merged_shape,
            ),
            _operand(
                "right-isometry",
                ("rt", "rb", "r"),
                (horizontal, horizontal, retained),
            ),
            _operand(
                "left-isometry",
                ("lt", "lb", "l"),
                (horizontal, horizontal, retained),
            ),
        ),
        ("u", "r", "d", "l"),
        precision=precision,
        resources=resources,
        dtype=dtype,
        cache=cache,
    )
    return TensorRenormalizationStagePlan(
        "hotrg-vertical",
        shape,
        (vertical, retained, vertical, retained),
        retained,
        retained,
        horizontal**4,
        merge,
        coarse,
    )


def _hotrg_horizontal_stage_plan(
    shape: tuple[int, int, int, int],
    capacity: int,
    *,
    precision: TensorNetworkPrecisionPolicy,
    resources: TensorRenormalizationResourcePolicy,
    dtype: str,
    cache: ContractionPlanCache,
) -> TensorRenormalizationStagePlan:
    vertical, horizontal, _, _ = shape
    retained = min(capacity, vertical * vertical)
    merge = _contraction_plan(
        (
            _operand("left", ("ul", "m", "dl", "l"), shape),
            _operand("right", ("ur", "r", "dr", "m"), shape),
        ),
        ("ul", "ur", "r", "dl", "dr", "l"),
        precision=precision,
        resources=resources,
        dtype=dtype,
        cache=cache,
    )
    merged_shape = (
        vertical,
        vertical,
        horizontal,
        vertical,
        vertical,
        horizontal,
    )
    coarse = _contraction_plan(
        (
            _operand(
                "merged",
                ("ul", "ur", "r", "dl", "dr", "l"),
                merged_shape,
            ),
            _operand(
                "up-isometry",
                ("ul", "ur", "u"),
                (vertical, vertical, retained),
            ),
            _operand(
                "down-isometry",
                ("dl", "dr", "d"),
                (vertical, vertical, retained),
            ),
        ),
        ("u", "r", "d", "l"),
        precision=precision,
        resources=resources,
        dtype=dtype,
        cache=cache,
    )
    return TensorRenormalizationStagePlan(
        "hotrg-horizontal",
        shape,
        (retained, horizontal, retained, horizontal),
        retained,
        retained,
        vertical**4,
        merge,
        coarse,
    )


def _factorization_flops(kind: str, shape: tuple[int, int, int, int], /) -> int:
    vertical, horizontal, _, _ = shape
    if kind == "trg":
        dimension = vertical * horizontal
        return 16 * dimension**3
    pair_dimension = (
        horizontal * horizontal if kind == "hotrg-vertical" else vertical * vertical
    )
    other_dimension = vertical * vertical * horizontal * horizontal
    gram_flops = 4 * pair_dimension * pair_dimension * other_dimension
    return gram_flops + 8 * pair_dimension**3


def plan_tensor_renormalization(
    problem: TensorRenormalizationProblem,
    policy: TensorRenormalizationPolicy,
    /,
) -> TensorRenormalizationPlan:
    """Plan every static coarse-graining shape and refuse excess resources."""

    if not isinstance(problem, TensorRenormalizationProblem):
        raise TypeError("problem must be a TensorRenormalizationProblem.")
    if not isinstance(policy, TensorRenormalizationPolicy):
        raise TypeError("policy must be a TensorRenormalizationPolicy.")
    tensor = problem.tensor
    precision = tensor.precision
    dtype = str(tensor.value.dtype)
    cache = ContractionPlanCache(max(4, 2 * policy.steps))
    shape = tuple(int(dimension) for dimension in tensor.value.shape)
    stages: list[TensorRenormalizationStagePlan] = []
    for index in range(policy.steps):
        if isinstance(policy.method, TRGMethod):
            stage = _trg_stage_plan(
                shape,
                policy.maximum_bond_dimension,
                precision=precision,
                resources=policy.resources,
                dtype=dtype,
                cache=cache,
            )
        else:
            vertical_first = policy.method.first_direction == "vertical"
            vertical = vertical_first if index % 2 == 0 else not vertical_first
            builder = (
                _hotrg_vertical_stage_plan if vertical else _hotrg_horizontal_stage_plan
            )
            stage = builder(
                shape,
                policy.maximum_bond_dimension,
                precision=precision,
                resources=policy.resources,
                dtype=dtype,
                cache=cache,
            )
        stages.append(stage)
        shape = stage.output_shape

    maximum_tensor = max(
        [prod(problem.tensor.value.shape)]
        + [
            max(
                prod(stage.output_shape),
                0 if stage.merge is None else stage.merge.cost.output_elements,
            )
            for stage in stages
        ]
    )
    maximum_factorization = max(stage.maximum_factorization_elements for stage in stages)
    history = 6 * policy.steps + 5
    precision_probe = jnp.empty((), dtype=tensor.value.dtype)
    itemsize = max(
        precision_itemsize(str(role(precision_probe).dtype))
        for role in (
            precision.storage,
            precision.contraction,
            precision.factorization,
            precision.accumulation,
        )
    )
    peak_workspace = max(
        max(
            stage.coarse.cost.peak_live_bytes,
            0 if stage.merge is None else stage.merge.cost.peak_live_bytes,
            (
                8 * stage.maximum_factorization_elements
                + (0 if stage.merge is None else 3 * stage.merge.cost.output_elements)
            )
            * itemsize,
        )
        for stage in stages
    )
    flops = sum(
        stage.coarse.cost.estimated_flops
        + (0 if stage.merge is None else stage.merge.cost.estimated_flops)
        + _factorization_flops(stage.kind, stage.input_shape)
        for stage in stages
    )
    resources = policy.resources
    if maximum_tensor > resources.maximum_tensor_elements:
        raise MemoryError(
            "Tensor-renormalization tensors exceed maximum_tensor_elements."
        )
    if maximum_factorization > resources.maximum_factorization_elements:
        raise MemoryError(
            "Tensor-renormalization factorization exceeds maximum_factorization_elements."
        )
    if peak_workspace > resources.maximum_workspace_bytes:
        raise MemoryError("Tensor renormalization exceeds maximum_workspace_bytes.")
    if history > resources.maximum_history_elements:
        raise MemoryError("Tensor-renormalization histories exceed their capacity.")
    cost = TensorRenormalizationCostEstimate(
        maximum_tensor,
        maximum_factorization,
        history,
        peak_workspace,
        flops,
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "tensor-renormalization-plan",
            "problem": problem.problem_id,
            "tensor": tensor.tensor_id,
            "policy": policy.policy_id,
            "stages": tuple(stage.stage_id for stage in stages),
        }
    )
    return TensorRenormalizationPlan(
        policy,
        tuple(stages),
        cost,
        tensor.tensor_id,
        problem.problem_id,
        plan_id,
    )


def _validate_structure(
    problem: TensorRenormalizationProblem,
    plan: TensorRenormalizationPlan,
    /,
) -> None:
    if problem.problem_id != plan.problem_id:
        raise ValueError("Tensor-renormalization problem identity changed; replan.")
    if problem.tensor.tensor_id != plan.tensor_id:
        raise ValueError("Uniform square tensor structure changed; replan.")


def prepare_tensor_renormalization(
    problem: TensorRenormalizationProblem,
    plan_or_policy: TensorRenormalizationPlan | TensorRenormalizationPolicy,
    /,
) -> PreparedTensorRenormalization:
    if not isinstance(problem, TensorRenormalizationProblem):
        raise TypeError("problem must be a TensorRenormalizationProblem.")
    plan = (
        plan_or_policy
        if isinstance(plan_or_policy, TensorRenormalizationPlan)
        else plan_tensor_renormalization(problem, plan_or_policy)
    )
    _validate_structure(problem, plan)
    prepared_id = canonical_fingerprint(
        {"kind": "prepared-tensor-renormalization", "plan": plan.plan_id}
    )
    return PreparedTensorRenormalization(
        problem,
        plan,
        jnp.asarray(problem.tensor.numeric_version, dtype=jnp.int32),
        prepared_id,
    )


def refresh_tensor_renormalization(
    prepared: PreparedTensorRenormalization,
    problem: TensorRenormalizationProblem,
    /,
) -> PreparedTensorRenormalization:
    if not isinstance(prepared, PreparedTensorRenormalization):
        raise TypeError("prepared must be PreparedTensorRenormalization.")
    if not isinstance(problem, TensorRenormalizationProblem):
        raise TypeError("problem must be TensorRenormalizationProblem.")
    _validate_structure(problem, prepared.plan)
    return PreparedTensorRenormalization(
        problem,
        prepared.plan,
        prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
        prepared.prepared_id,
    )


def _execute_contraction(
    plan: ContractionPlan,
    operands: tuple[Array, ...],
    precision: TensorNetworkPrecisionPolicy,
    /,
) -> Array:
    prepared = prepare_contraction(plan, operands)
    values = precision.contraction(prepared.operands)
    return precision.storage(execute_schedule(plan.schedule, values))


def _hotrg_isometry(
    merged: Array,
    pair_dimension: int,
    retained_rank: int,
    first_permutation: tuple[int, ...],
    second_permutation: tuple[int, ...],
    precision: TensorNetworkPrecisionPolicy,
    /,
) -> tuple[
    Array,
    TensorTruncationEvidence,
    TensorTruncationEvidence,
    Array,
]:
    first = jnp.transpose(merged, first_permutation).reshape((pair_dimension, -1))
    second = jnp.transpose(merged, second_permutation).reshape((pair_dimension, -1))
    first = precision.factorization(first)
    second = precision.factorization(second)
    gram = ein.contract(
        "ik,jk->ij",
        first,
        jnp.conj(first),
        backend="jax",
    ) + ein.contract(
        "ik,jk->ij",
        second,
        jnp.conj(second),
        backend="jax",
    )
    gram = 0.5 * (gram + jnp.conj(gram.T))
    real_dtype = jnp.real(gram).dtype
    tolerance = float(128 * pair_dimension * jnp.finfo(real_dtype).eps)
    hermitian_precision = HermitianPrecisionPolicy(
        compute_dtype=precision.contraction_dtype,
        factorization_dtype=precision.factorization_dtype,
        accumulation_dtype=precision.accumulation_dtype,
        decision_dtype=precision.decision_dtype,
        output_dtype=precision.storage_dtype,
    )
    spectrum = HermitianSpectrum(
        gram,
        tolerance=tolerance,
        precision=hermitian_precision,
    )
    retained = min(int(retained_rank), pair_dimension)
    eigenvalues = jnp.maximum(jnp.real(spectrum.eigenvalues), 0.0)
    isometry = precision.storage(spectrum.eigenvectors[:, -retained:])
    discarded = precision.decision(
        precision.sum(eigenvalues[: pair_dimension - retained])
    )
    precision_evidence = precision.evidence_for(
        merged,
        children={"hotrg_gram": spectrum.precision_evidence},
        output_value=isometry,
    )
    primary = TensorTruncationEvidence(
        retained,
        pair_dimension,
        discarded,
        precision_evidence,
        precision.policy_id,
    )
    paired_basis = TensorTruncationEvidence(
        retained,
        pair_dimension,
        precision.decision(0.0),
        precision_evidence,
        precision.policy_id,
    )
    valid = (
        spectrum.valid
        & (spectrum.minimum_eigenvalue >= -tolerance)
        & jnp.all(jnp.isfinite(isometry))
        & primary.valid
    )
    return isometry, primary, paired_basis, valid


def _trg_step(
    value: Array,
    stage: TensorRenormalizationStagePlan,
    precision: TensorNetworkPrecisionPolicy,
    /,
) -> tuple[Array, TensorTruncationEvidence, TensorTruncationEvidence, Array]:
    vertical, horizontal, _, _ = stage.input_shape
    first_matrix = value.reshape((vertical * horizontal, vertical * horizontal))
    upper_right, lower_left, first = truncated_svd(
        first_matrix,
        maximum_rank=stage.first_retained_rank,
        absorb="split",
        precision=precision,
        evidence_source=value,
    )
    upper_right = upper_right.reshape((vertical, horizontal, stage.first_retained_rank))
    lower_left = lower_left.reshape((stage.first_retained_rank, vertical, horizontal))

    second_matrix = jnp.transpose(value, (1, 2, 3, 0)).reshape(
        (horizontal * vertical, horizontal * vertical)
    )
    right_lower, left_upper, second = truncated_svd(
        second_matrix,
        maximum_rank=stage.second_retained_rank,
        absorb="split",
        precision=precision,
        evidence_source=value,
    )
    right_lower = right_lower.reshape((horizontal, vertical, stage.second_retained_rank))
    left_upper = left_upper.reshape((stage.second_retained_rank, horizontal, vertical))
    output = _execute_contraction(
        stage.coarse,
        (upper_right, right_lower, lower_left, left_upper),
        precision,
    )
    factors_finite = (
        jnp.all(jnp.isfinite(upper_right))
        & jnp.all(jnp.isfinite(lower_left))
        & jnp.all(jnp.isfinite(right_lower))
        & jnp.all(jnp.isfinite(left_upper))
    )
    return output, first, second, factors_finite


def _hotrg_vertical_step(
    value: Array,
    stage: TensorRenormalizationStagePlan,
    precision: TensorNetworkPrecisionPolicy,
    /,
) -> tuple[Array, TensorTruncationEvidence, TensorTruncationEvidence, Array]:
    if stage.merge is None:
        raise RuntimeError("Vertical HOTRG stage lacks its merge contraction.")
    _, horizontal, _, _ = stage.input_shape
    merged = _execute_contraction(stage.merge, (value, value), precision)
    isometry, first, second, factorization_valid = _hotrg_isometry(
        merged,
        horizontal * horizontal,
        stage.first_retained_rank,
        (1, 2, 0, 3, 4, 5),
        (4, 5, 0, 3, 1, 2),
        precision,
    )
    isometry = isometry.reshape((horizontal, horizontal, stage.first_retained_rank))
    output = _execute_contraction(
        stage.coarse,
        (merged, jnp.conj(isometry), jnp.conj(isometry)),
        precision,
    )
    factors_finite = (
        factorization_valid
        & jnp.all(jnp.isfinite(merged))
        & jnp.all(jnp.isfinite(isometry))
    )
    return output, first, second, factors_finite


def _hotrg_horizontal_step(
    value: Array,
    stage: TensorRenormalizationStagePlan,
    precision: TensorNetworkPrecisionPolicy,
    /,
) -> tuple[Array, TensorTruncationEvidence, TensorTruncationEvidence, Array]:
    if stage.merge is None:
        raise RuntimeError("Horizontal HOTRG stage lacks its merge contraction.")
    vertical, _, _, _ = stage.input_shape
    merged = _execute_contraction(stage.merge, (value, value), precision)
    isometry, first, second, factorization_valid = _hotrg_isometry(
        merged,
        vertical * vertical,
        stage.first_retained_rank,
        (0, 1, 2, 3, 4, 5),
        (3, 4, 2, 5, 0, 1),
        precision,
    )
    isometry = isometry.reshape((vertical, vertical, stage.first_retained_rank))
    output = _execute_contraction(
        stage.coarse,
        (merged, jnp.conj(isometry), jnp.conj(isometry)),
        precision,
    )
    factors_finite = (
        factorization_valid
        & jnp.all(jnp.isfinite(merged))
        & jnp.all(jnp.isfinite(isometry))
    )
    return output, first, second, factors_finite


def _normalized(
    value: Array,
    precision: TensorNetworkPrecisionPolicy,
    /,
) -> tuple[Array, Array, Array, Array]:
    accumulated = precision.accumulation(value)
    finite = jnp.all(jnp.isfinite(accumulated))
    scale = precision.decision(jnp.max(jnp.abs(accumulated)))
    positive = finite & jnp.isfinite(scale) & (scale > 0.0)
    safe_scale = jnp.where(positive, scale, precision.decision(1.0))
    normalized = jnp.where(
        positive, accumulated / safe_scale, jnp.zeros_like(accumulated)
    )
    return precision.storage(normalized), scale, finite, positive


def _failure(status: Array, failed: Array, code: TensorRenormalizationStatus, /) -> Array:
    return jnp.where(
        (status == int(TensorRenormalizationStatus.SUCCESS)) & failed,
        int(code),
        status,
    ).astype(jnp.int32)


def run_tensor_renormalization(
    problem_or_prepared: TensorRenormalizationProblem | PreparedTensorRenormalization,
    policy: TensorRenormalizationPolicy | None = None,
    /,
) -> TensorRenormalizationResult:
    """Execute one admitted TRG or HOTRG plan with finite-step evidence."""

    if isinstance(problem_or_prepared, PreparedTensorRenormalization):
        if policy is not None:
            raise ValueError(
                "policy must be omitted for prepared tensor renormalization."
            )
        prepared = problem_or_prepared
    elif isinstance(problem_or_prepared, TensorRenormalizationProblem):
        if not isinstance(policy, TensorRenormalizationPolicy):
            raise TypeError("policy must be supplied for an unprepared problem.")
        prepared = prepare_tensor_renormalization(problem_or_prepared, policy)
    else:
        raise TypeError(
            "Expected TensorRenormalizationProblem or PreparedTensorRenormalization."
        )

    plan = prepared.plan
    precision = prepared.problem.tensor.precision
    value, initial_scale, initial_finite, initial_positive = _normalized(
        prepared.problem.tensor.value,
        precision,
    )
    status = jnp.asarray(int(TensorRenormalizationStatus.SUCCESS), dtype=jnp.int32)
    status = _failure(
        status,
        ~initial_finite,
        TensorRenormalizationStatus.NONFINITE_INPUT,
    )
    status = _failure(
        status,
        initial_finite & ~initial_positive,
        TensorRenormalizationStatus.ZERO_NORMALIZATION,
    )
    normalization_scales = [initial_scale]
    finite_history = [initial_finite]
    first_discarded = []
    second_discarded = []
    retained_ranks = []
    normalization = precision.decision(
        jnp.log(jnp.where(initial_positive, initial_scale, 1.0))
    )

    for index, stage in enumerate(plan.stages):
        if stage.kind == "trg":
            raw, first, second, factors_finite = _trg_step(value, stage, precision)
        elif stage.kind == "hotrg-vertical":
            raw, first, second, factors_finite = _hotrg_vertical_step(
                value, stage, precision
            )
        else:
            raw, first, second, factors_finite = _hotrg_horizontal_step(
                value, stage, precision
            )
        factorization_valid = first.valid & second.valid & factors_finite
        status = _failure(
            status,
            ~factorization_valid,
            TensorRenormalizationStatus.FACTORIZATION_FAILED,
        )
        value, scale, finite, positive = _normalized(raw, precision)
        status = _failure(
            status,
            ~finite,
            TensorRenormalizationStatus.NONFINITE_INTERMEDIATE,
        )
        status = _failure(
            status,
            finite & ~positive,
            TensorRenormalizationStatus.ZERO_NORMALIZATION,
        )
        site_weight = precision.decision(2.0 ** (index + 1))
        normalization = normalization + precision.decision(
            jnp.log(jnp.where(positive, scale, 1.0)) / site_weight
        )
        normalization_scales.append(scale)
        finite_history.append(finite & factorization_valid)
        first_discarded.append(first.discarded_weight)
        second_discarded.append(second.discarded_weight)
        retained_ranks.append(
            jnp.asarray(
                (first.retained_rank, second.retained_rank),
                dtype=jnp.int32,
            )
        )

    terminal = ein.contract(
        "abab->",
        precision.contraction(value),
        backend="jax",
    )
    terminal_finite = jnp.isfinite(terminal)
    terminal_real = precision.decision(jnp.real(terminal))
    terminal_imaginary = precision.decision(jnp.abs(jnp.imag(terminal)))
    terminal_scale = precision.decision(jnp.maximum(jnp.abs(terminal), 1.0))
    terminal_positive = terminal_real > plan.policy.terminal_positivity_tolerance
    terminal_valid = (
        terminal_finite
        & terminal_positive
        & (
            terminal_imaginary
            <= plan.policy.terminal_imaginary_tolerance * terminal_scale
        )
    )
    status = _failure(
        status,
        ~terminal_valid,
        TensorRenormalizationStatus.TERMINAL_PARTITION_INVALID,
    )
    terminal_correction = precision.decision(
        jnp.log(jnp.where(terminal_valid, terminal_real, 1.0))
        / precision.decision(2.0**plan.policy.steps)
    )
    log_partition_density = precision.output(normalization + terminal_correction)
    output_tensor = UniformSquareTensor(
        precision.storage(value),
        precision=precision,
        numeric_version=prepared.numeric_version,
    )
    accepted = status == int(TensorRenormalizationStatus.SUCCESS)
    exact = accepted & jnp.asarray(
        prepared.problem.tensor.vertical_bond_dimension == 1
        and prepared.problem.tensor.horizontal_bond_dimension == 1
    )
    diagnostics = TensorRenormalizationDiagnostics(
        jnp.stack(tuple(normalization_scales)),
        jnp.stack(tuple(first_discarded)),
        jnp.stack(tuple(second_discarded)),
        jnp.stack(tuple(retained_ranks)),
        jnp.stack(tuple(finite_history)),
        terminal_imaginary,
        terminal_positive,
        status,
        exact,
        precision.evidence_for(
            prepared.problem.tensor.value,
            output_value=output_tensor.value,
        ),
        "finite-step fixed-rank tensor renormalization; exact only for unit bond dimension; no global error bound",
        False,
        plan.cost.peak_workspace_bytes,
    )
    provenance = TensorRenormalizationProvenance(
        plan.policy.method.method_id,
        prepared.problem.tensor.tensor_id,
        prepared.problem.problem_id,
        plan.policy.policy_id,
        plan.plan_id,
        prepared.prepared_id,
        prepared.numeric_version,
    )
    return TensorRenormalizationResult(
        output_tensor,
        log_partition_density,
        precision.output(normalization),
        precision.output(terminal_correction),
        precision.output(terminal),
        diagnostics,
        provenance,
    )


__all__ = [
    "HOTRGMethod",
    "PreparedTensorRenormalization",
    "TRGMethod",
    "TensorRenormalizationCostEstimate",
    "TensorRenormalizationDiagnostics",
    "TensorRenormalizationMethod",
    "TensorRenormalizationPlan",
    "TensorRenormalizationPolicy",
    "TensorRenormalizationProblem",
    "TensorRenormalizationProvenance",
    "TensorRenormalizationResourcePolicy",
    "TensorRenormalizationResult",
    "TensorRenormalizationStagePlan",
    "TensorRenormalizationStatus",
    "plan_tensor_renormalization",
    "prepare_tensor_renormalization",
    "refresh_tensor_renormalization",
    "run_tensor_renormalization",
]
