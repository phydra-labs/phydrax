#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._precision import precision_itemsize, PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ._core import MatrixProductOperator, MatrixProductState
from ._precision import TensorNetworkPrecisionPolicy


class ContractionLeg(StrictModule):
    label: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)

    def __init__(self, label: str, dimension: int, /):
        label_ = str(label)
        dimension_ = int(dimension)
        if not label_ or dimension_ < 1:
            raise ValueError("Contraction legs require a label and positive dimension.")
        self.label = label_
        self.dimension = dimension_


class ContractionOperand(StrictModule):
    operand_id: str = eqx.field(static=True)
    legs: tuple[ContractionLeg, ...] = eqx.field(static=True)

    def __init__(
        self,
        operand_id: str,
        legs: Sequence[ContractionLeg],
        /,
    ):
        identifier = str(operand_id)
        values = tuple(legs)
        if not identifier or not values:
            raise ValueError("Contraction operands require an ID and at least one leg.")
        if any(not isinstance(leg, ContractionLeg) for leg in values):
            raise TypeError("legs must contain ContractionLeg values.")
        labels = tuple(leg.label for leg in values)
        if len(set(labels)) != len(labels):
            raise ValueError("A contraction operand cannot repeat one leg label.")
        self.operand_id = identifier
        self.legs = values


class ContractionStructure(StrictModule):
    operands: tuple[ContractionOperand, ...] = eqx.field(static=True)
    outputs: tuple[str, ...] = eqx.field(static=True)
    arithmetic_domain: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        operands: Sequence[ContractionOperand],
        outputs: Sequence[str],
        /,
        *,
        arithmetic_domain: str = "ordinary",
    ):
        values = tuple(operands)
        output_labels = tuple(str(label) for label in outputs)
        if not values or any(
            not isinstance(value, ContractionOperand) for value in values
        ):
            raise TypeError(
                "operands must be a nonempty sequence of ContractionOperand values."
            )
        operand_ids = tuple(value.operand_id for value in values)
        if len(set(operand_ids)) != len(operand_ids):
            raise ValueError("Contraction operand IDs must be unique.")
        if len(set(output_labels)) != len(output_labels):
            raise ValueError("Contraction output labels must be unique.")
        if arithmetic_domain != "ordinary":
            raise ValueError("Only ordinary product-sum contraction is supported.")
        dimensions: dict[str, int] = {}
        occurrences: Counter[str] = Counter()
        for operand in values:
            for leg in operand.legs:
                previous = dimensions.setdefault(leg.label, leg.dimension)
                if previous != leg.dimension:
                    raise ValueError("One contraction label has inconsistent dimensions.")
                occurrences[leg.label] += 1
        for label, count in occurrences.items():
            if label in output_labels:
                if count != 1:
                    raise ValueError("Every output label must occur exactly once.")
            elif count != 2:
                raise ValueError(
                    "Every contracted label must occur exactly twice; hyperedges are unsupported."
                )
        if any(label not in occurrences for label in output_labels):
            raise ValueError("Every output label must occur on one operand.")
        self.operands = values
        self.outputs = output_labels
        self.arithmetic_domain = arithmetic_domain
        self.structure_id = canonical_fingerprint(
            {
                "kind": "ordinary-labelled-contraction",
                "operands": tuple(
                    (
                        operand.operand_id,
                        tuple((leg.label, leg.dimension) for leg in operand.legs),
                    )
                    for operand in values
                ),
                "outputs": output_labels,
            }
        )


class ContractionResourcePolicy(StrictModule):
    maximum_operand_elements: int = eqx.field(static=True)
    maximum_intermediate_elements: int = eqx.field(static=True)
    maximum_output_elements: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    maximum_flops: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_operand_elements: int = 100_000_000,
        maximum_intermediate_elements: int = 100_000_000,
        maximum_output_elements: int = 100_000_000,
        maximum_workspace_bytes: int = 2**31,
        maximum_flops: int = 10**15,
    ):
        values = tuple(
            int(value)
            for value in (
                maximum_operand_elements,
                maximum_intermediate_elements,
                maximum_output_elements,
                maximum_workspace_bytes,
                maximum_flops,
            )
        )
        if any(value < 1 for value in values):
            raise ValueError("Contraction resource limits must be positive.")
        self.maximum_operand_elements = values[0]
        self.maximum_intermediate_elements = values[1]
        self.maximum_output_elements = values[2]
        self.maximum_workspace_bytes = values[3]
        self.maximum_flops = values[4]
        self.policy_id = canonical_fingerprint(
            {"kind": "contraction-resource-policy", "limits": values}
        )


class ContractionCostEstimate(StrictModule):
    operand_elements: int = eqx.field(static=True)
    output_elements: int = eqx.field(static=True)
    largest_intermediate_elements: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    estimated_flops: int = eqx.field(static=True)


class ContractionPlan(StrictModule):
    structure: ContractionStructure
    resources: ContractionResourcePolicy
    precision: TensorNetworkPrecisionPolicy
    cost: ContractionCostEstimate
    equation: str = eqx.field(static=True)
    path: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    optimizer: str = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedContraction(StrictModule):
    plan: ContractionPlan
    operands: tuple[Array, ...]
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)


class ContractionExecutionEvidence(StrictModule):
    finite: Array
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: Array


class ContractionResult(StrictModule):
    value: Array
    evidence: ContractionExecutionEvidence


def _equation(structure: ContractionStructure, /) -> str:
    labels = []
    for operand in structure.operands:
        for leg in operand.legs:
            if leg.label not in labels:
                labels.append(leg.label)
    symbols = {label: oe.get_symbol(index) for index, label in enumerate(labels)}
    inputs = ",".join(
        "".join(symbols[leg.label] for leg in operand.legs)
        for operand in structure.operands
    )
    output = "".join(symbols[label] for label in structure.outputs)
    return f"{inputs}->{output}"


def plan_contraction(
    structure: ContractionStructure,
    /,
    *,
    precision: TensorNetworkPrecisionPolicy | None = None,
    resources: ContractionResourcePolicy | None = None,
    optimizer: str = "greedy",
    dtype: str = "complex128",
) -> ContractionPlan:
    if not isinstance(structure, ContractionStructure):
        raise TypeError("structure must be ContractionStructure.")
    precision_ = TensorNetworkPrecisionPolicy() if precision is None else precision
    resources_ = ContractionResourcePolicy() if resources is None else resources
    if not isinstance(precision_, TensorNetworkPrecisionPolicy):
        raise TypeError("precision must be TensorNetworkPrecisionPolicy or None.")
    if not isinstance(resources_, ContractionResourcePolicy):
        raise TypeError("resources must be ContractionResourcePolicy or None.")
    if optimizer not in ("greedy", "optimal"):
        raise ValueError("optimizer must be 'greedy' or 'optimal'.")
    dtype_ = jnp.dtype(dtype).name
    shapes = tuple(
        tuple(leg.dimension for leg in operand.legs) for operand in structure.operands
    )
    equation = _equation(structure)
    path, information = oe.contract_path(
        equation, *shapes, shapes=True, optimize=optimizer
    )
    operand_elements = sum(prod(shape) for shape in shapes)
    output_dimensions = {
        leg.label: leg.dimension for operand in structure.operands for leg in operand.legs
    }
    output_elements = prod(output_dimensions[label] for label in structure.outputs)
    largest = int(information.largest_intermediate)
    flops = int(information.opt_cost)
    workspace_bytes = largest * precision_itemsize(dtype_)
    if operand_elements > resources_.maximum_operand_elements:
        raise MemoryError("Contraction operands exceed maximum_operand_elements.")
    if largest > resources_.maximum_intermediate_elements:
        raise MemoryError("Contraction path exceeds maximum_intermediate_elements.")
    if output_elements > resources_.maximum_output_elements:
        raise MemoryError("Contraction output exceeds maximum_output_elements.")
    if workspace_bytes > resources_.maximum_workspace_bytes:
        raise MemoryError("Contraction path exceeds maximum_workspace_bytes.")
    if flops > resources_.maximum_flops:
        raise MemoryError("Contraction path exceeds maximum_flops.")
    cost = ContractionCostEstimate(
        operand_elements,
        output_elements,
        largest,
        workspace_bytes,
        flops,
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "contraction-plan",
            "structure": structure.structure_id,
            "resources": resources_.policy_id,
            "precision": precision_.policy_id,
            "optimizer": optimizer,
            "dtype": dtype_,
            "equation": equation,
            "path": tuple(tuple(int(index) for index in step) for step in path),
        }
    )
    return ContractionPlan(
        structure,
        resources_,
        precision_,
        cost,
        equation,
        tuple(tuple(int(index) for index in step) for step in path),
        optimizer,
        dtype_,
        plan_id,
    )


def _validate_operands(plan: ContractionPlan, operands: Sequence[ArrayLike], /):
    arrays = tuple(jnp.asarray(value) for value in operands)
    if len(arrays) != len(plan.structure.operands):
        raise ValueError("Operand count differs from the contraction plan.")
    for array, specification in zip(arrays, plan.structure.operands, strict=True):
        expected = tuple(leg.dimension for leg in specification.legs)
        if array.shape != expected:
            raise ValueError(
                f"Operand {specification.operand_id!r} expected shape {expected}; "
                f"got {array.shape}."
            )
        if str(array.dtype) != plan.dtype:
            raise TypeError("Operand dtype differs from the contraction plan.")
    plan.precision.validate_storage(arrays)
    return arrays


def prepare_contraction(
    plan: ContractionPlan,
    operands: Sequence[ArrayLike],
    /,
) -> PreparedContraction:
    if not isinstance(plan, ContractionPlan):
        raise TypeError("plan must be ContractionPlan.")
    arrays = _validate_operands(plan, operands)
    prepared_id = canonical_fingerprint(
        {"kind": "prepared-contraction", "plan": plan.plan_id}
    )
    return PreparedContraction(
        plan,
        arrays,
        jnp.asarray(0, dtype=jnp.int32),
        prepared_id,
    )


def refresh_contraction(
    prepared: PreparedContraction,
    operands: Sequence[ArrayLike],
    /,
) -> PreparedContraction:
    if not isinstance(prepared, PreparedContraction):
        raise TypeError("prepared must be PreparedContraction.")
    arrays = _validate_operands(prepared.plan, operands)
    return PreparedContraction(
        prepared.plan,
        arrays,
        prepared.numeric_version + 1,
        prepared.prepared_id,
    )


def execute_contraction(prepared: PreparedContraction, /) -> ContractionResult:
    if not isinstance(prepared, PreparedContraction):
        raise TypeError("prepared must be PreparedContraction.")
    plan = prepared.plan
    operands = plan.precision.contraction(prepared.operands)
    value = oe.contract(plan.equation, *operands, optimize=plan.path)
    output = plan.precision.output(value)
    evidence = ContractionExecutionEvidence(
        jnp.all(jnp.isfinite(output)),
        plan.precision.evidence_for(prepared.operands, output_value=output),
        plan.structure.structure_id,
        plan.plan_id,
        prepared.prepared_id,
        prepared.numeric_version,
    )
    return ContractionResult(output, evidence)


def prepare_mps_inner_contraction(
    left: MatrixProductState,
    right: MatrixProductState,
    /,
    *,
    resources: ContractionResourcePolicy | None = None,
) -> PreparedContraction:
    if left.physical_dimensions != right.physical_dimensions:
        raise ValueError("MPS physical dimensions must match.")
    if left.precision.policy_id != right.precision.policy_id:
        raise ValueError("MPS precision policies must match.")
    operands = []
    arrays = []
    for site, (first, second) in enumerate(zip(left.tensors, right.tensors, strict=True)):
        operands.extend(
            (
                ContractionOperand(
                    f"bra-{site}",
                    (
                        ContractionLeg(f"bra-bond-{site}", first.shape[0]),
                        ContractionLeg(f"physical-{site}", first.shape[1]),
                        ContractionLeg(f"bra-bond-{site + 1}", first.shape[2]),
                    ),
                ),
                ContractionOperand(
                    f"ket-{site}",
                    (
                        ContractionLeg(f"ket-bond-{site}", second.shape[0]),
                        ContractionLeg(f"physical-{site}", second.shape[1]),
                        ContractionLeg(f"ket-bond-{site + 1}", second.shape[2]),
                    ),
                ),
            )
        )
        arrays.extend((jnp.conj(first), second))
    for name in ("bra", "ket"):
        operands.extend(
            (
                ContractionOperand(
                    f"{name}-left-boundary",
                    (ContractionLeg(f"{name}-bond-0", 1),),
                ),
                ContractionOperand(
                    f"{name}-right-boundary",
                    (ContractionLeg(f"{name}-bond-{left.site_count}", 1),),
                ),
            )
        )
        arrays.extend((jnp.ones((1,), dtype=left.tensors[0].dtype),) * 2)
    structure = ContractionStructure(tuple(operands), ())
    plan = plan_contraction(
        structure,
        precision=left.precision,
        resources=resources,
        dtype=str(left.tensors[0].dtype),
    )
    return prepare_contraction(plan, tuple(arrays))


def prepare_mpo_inner_contraction(
    left: MatrixProductOperator,
    right: MatrixProductOperator,
    /,
    *,
    resources: ContractionResourcePolicy | None = None,
) -> PreparedContraction:
    if (
        left.output_dimensions != right.output_dimensions
        or left.input_dimensions != right.input_dimensions
    ):
        raise ValueError("MPO dimensions must match.")
    if left.precision.policy_id != right.precision.policy_id:
        raise ValueError("MPO precision policies must match.")
    operands = []
    arrays = []
    for site, (first, second) in enumerate(zip(left.tensors, right.tensors, strict=True)):
        operands.extend(
            (
                ContractionOperand(
                    f"left-{site}",
                    (
                        ContractionLeg(f"left-bond-{site}", first.shape[0]),
                        ContractionLeg(f"output-{site}", first.shape[1]),
                        ContractionLeg(f"input-{site}", first.shape[2]),
                        ContractionLeg(f"left-bond-{site + 1}", first.shape[3]),
                    ),
                ),
                ContractionOperand(
                    f"right-{site}",
                    (
                        ContractionLeg(f"right-bond-{site}", second.shape[0]),
                        ContractionLeg(f"output-{site}", second.shape[1]),
                        ContractionLeg(f"input-{site}", second.shape[2]),
                        ContractionLeg(f"right-bond-{site + 1}", second.shape[3]),
                    ),
                ),
            )
        )
        arrays.extend((jnp.conj(first), second))
    for name in ("left", "right"):
        operands.extend(
            (
                ContractionOperand(
                    f"{name}-left-boundary",
                    (ContractionLeg(f"{name}-bond-0", 1),),
                ),
                ContractionOperand(
                    f"{name}-right-boundary",
                    (ContractionLeg(f"{name}-bond-{left.site_count}", 1),),
                ),
            )
        )
        arrays.extend((jnp.ones((1,), dtype=left.tensors[0].dtype),) * 2)
    structure = ContractionStructure(tuple(operands), ())
    plan = plan_contraction(
        structure,
        precision=left.precision,
        resources=resources,
        dtype=str(left.tensors[0].dtype),
    )
    return prepare_contraction(plan, tuple(arrays))


__all__ = [
    "ContractionCostEstimate",
    "ContractionExecutionEvidence",
    "ContractionLeg",
    "ContractionOperand",
    "ContractionPlan",
    "ContractionResourcePolicy",
    "ContractionResult",
    "ContractionStructure",
    "PreparedContraction",
    "execute_contraction",
    "plan_contraction",
    "prepare_contraction",
    "prepare_mpo_inner_contraction",
    "prepare_mps_inner_contraction",
    "refresh_contraction",
]
