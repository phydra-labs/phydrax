#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

import equinox as eqx
import numpy as np

from ..._strict import StrictModule
from ._validation import content_id, nonempty_string, string_tuple


class MethodKind(StrEnum):
    """Scientific status of the objective or model implemented by a method."""

    EXACT_MODEL = "exact_model"
    APPROXIMATE_MODEL = "approximate_model"
    RELAXED_OBJECTIVE = "relaxed_objective"
    HEURISTIC = "heuristic"
    LEARNED = "learned"


class ExecutionKind(StrEnum):
    """Numerical execution claim made independently of scientific semantics."""

    EXACT_DISCRETE = "exact_discrete"
    FLOATING_POINT_DIRECT = "floating_point_direct"
    ITERATIVE_TOLERANCE = "iterative_tolerance"
    STOCHASTIC_ESTIMATE = "stochastic_estimate"


class DifferentiationKind(StrEnum):
    """Derivative semantics exposed by a numerical implementation."""

    EXACT_AD = "exact_ad"
    ALMOST_EVERYWHERE = "almost_everywhere"
    IMPLICIT = "implicit"
    UNROLLED = "unrolled"
    STOCHASTIC_ESTIMATOR = "stochastic_estimator"
    SURROGATE = "surrogate"
    NONE = "none"


class OutputKind(StrEnum):
    """Primary semantic form of a method's output."""

    SCALAR = "scalar"
    ARRAY = "array"
    DISCRETE = "discrete"
    PROBABILISTIC = "probabilistic"
    RANKING = "ranking"
    PARTITION = "partition"
    GRAPH = "graph"
    SEQUENCE = "sequence"
    SET = "set"
    STRUCTURED = "structured"


def _dtype_name(name: str, value: str | None, /) -> str | None:
    if value is None:
        return None
    return nonempty_string(name, value)


class BioinformaticsMethodContract(StrictModule):
    """Canonical scientific, numerical, and differentiation contract for a method."""

    method_name: str = eqx.field(static=True)
    method_kind: MethodKind = eqx.field(static=True)
    execution_kind: ExecutionKind = eqx.field(static=True)
    differentiation_kind: DifferentiationKind = eqx.field(static=True)
    output_kind: OutputKind = eqx.field(static=True)
    conditioning_statement: str = eqx.field(static=True)
    truncation_statement: str = eqx.field(static=True)
    capacity_semantics: str = eqx.field(static=True)
    assumptions: tuple[str, ...] = eqx.field(static=True)
    nondifferentiable_outputs: tuple[str, ...] = eqx.field(static=True)
    input_dtype: str | None = eqx.field(static=True)
    compute_dtype: str | None = eqx.field(static=True)
    output_dtype: str | None = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        method_name: str,
        method_kind: MethodKind,
        execution_kind: ExecutionKind,
        differentiation_kind: DifferentiationKind,
        output_kind: OutputKind,
        /,
        *,
        conditioning_statement: str,
        truncation_statement: str,
        capacity_semantics: str,
        assumptions: Sequence[str] = (),
        nondifferentiable_outputs: Sequence[str] = (),
        input_dtype: str | None = None,
        compute_dtype: str | None = None,
        output_dtype: str | None = None,
        absolute_tolerance: float = 0.0,
        relative_tolerance: float = 0.0,
    ):
        method_name_ = nonempty_string("method_name", method_name)
        method_kind_ = MethodKind(method_kind)
        execution_kind_ = ExecutionKind(execution_kind)
        differentiation_kind_ = DifferentiationKind(differentiation_kind)
        output_kind_ = OutputKind(output_kind)
        conditioning_ = nonempty_string("conditioning_statement", conditioning_statement)
        truncation_ = nonempty_string("truncation_statement", truncation_statement)
        capacity_ = nonempty_string("capacity_semantics", capacity_semantics)
        assumptions_ = string_tuple("assumptions", assumptions)
        nondifferentiable_ = string_tuple(
            "nondifferentiable_outputs", nondifferentiable_outputs
        )
        input_dtype_ = _dtype_name("input_dtype", input_dtype)
        compute_dtype_ = _dtype_name("compute_dtype", compute_dtype)
        output_dtype_ = _dtype_name("output_dtype", output_dtype)
        absolute_ = float(absolute_tolerance)
        relative_ = float(relative_tolerance)
        if not np.isfinite(absolute_) or absolute_ < 0.0:
            raise ValueError("absolute_tolerance must be finite and non-negative.")
        if not np.isfinite(relative_) or relative_ < 0.0:
            raise ValueError("relative_tolerance must be finite and non-negative.")

        payload = {
            "absolute_tolerance": absolute_,
            "assumptions": assumptions_,
            "capacity_semantics": capacity_,
            "compute_dtype": compute_dtype_,
            "conditioning_statement": conditioning_,
            "differentiation_kind": differentiation_kind_.value,
            "execution_kind": execution_kind_.value,
            "input_dtype": input_dtype_,
            "method_kind": method_kind_.value,
            "method_name": method_name_,
            "nondifferentiable_outputs": nondifferentiable_,
            "output_dtype": output_dtype_,
            "output_kind": output_kind_.value,
            "relative_tolerance": relative_,
            "truncation_statement": truncation_,
        }
        self.method_name = method_name_
        self.method_kind = method_kind_
        self.execution_kind = execution_kind_
        self.differentiation_kind = differentiation_kind_
        self.output_kind = output_kind_
        self.conditioning_statement = conditioning_
        self.truncation_statement = truncation_
        self.capacity_semantics = capacity_
        self.assumptions = assumptions_
        self.nondifferentiable_outputs = nondifferentiable_
        self.input_dtype = input_dtype_
        self.compute_dtype = compute_dtype_
        self.output_dtype = output_dtype_
        self.absolute_tolerance = absolute_
        self.relative_tolerance = relative_
        self.contract_id = content_id("bioinformatics_method_contract", payload, ())
