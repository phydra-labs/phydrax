#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._precision import (
    PrecisionEvidenceEnvelope,
    PrecisionRequest,
    PrecisionResolution,
    real_precision_dtype_name,
)
from .._strict import AbstractAttribute, StrictModule


class FactorKernelCapabilities(StrictModule):
    """Static execution guarantees of one finite-discrete factor kernel."""

    sum_product: bool = eqx.field(static=True)
    max_product: bool = eqx.field(static=True)
    factor_beliefs: bool = eqx.field(static=True)
    scalar_conditional: bool = eqx.field(static=True)
    joint_conditional: bool = eqx.field(static=True)
    sparse_support: bool = eqx.field(static=True)
    hard_constraints: bool = eqx.field(static=True)
    smooth_parameters: bool = eqx.field(static=True)
    prepared_refresh: bool = eqx.field(static=True)
    batched: bool = eqx.field(static=True)
    shardable: bool = eqx.field(static=True)
    capability_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        sum_product: bool = False,
        max_product: bool = False,
        factor_beliefs: bool = False,
        scalar_conditional: bool = True,
        joint_conditional: bool = False,
        sparse_support: bool = False,
        hard_constraints: bool = True,
        smooth_parameters: bool = True,
        prepared_refresh: bool = True,
        batched: bool = True,
        shardable: bool = True,
    ):
        self.sum_product = bool(sum_product)
        self.max_product = bool(max_product)
        self.factor_beliefs = bool(factor_beliefs)
        self.scalar_conditional = bool(scalar_conditional)
        self.joint_conditional = bool(joint_conditional)
        self.sparse_support = bool(sparse_support)
        self.hard_constraints = bool(hard_constraints)
        self.smooth_parameters = bool(smooth_parameters)
        self.prepared_refresh = bool(prepared_refresh)
        self.batched = bool(batched)
        self.shardable = bool(shardable)
        self.capability_id = canonical_fingerprint(
            {
                "kind": "factor-kernel-capabilities",
                "sum_product": self.sum_product,
                "max_product": self.max_product,
                "factor_beliefs": self.factor_beliefs,
                "scalar_conditional": self.scalar_conditional,
                "joint_conditional": self.joint_conditional,
                "sparse_support": self.sparse_support,
                "hard_constraints": self.hard_constraints,
                "smooth_parameters": self.smooth_parameters,
                "prepared_refresh": self.prepared_refresh,
                "batched": self.batched,
                "shardable": self.shardable,
            }
        )


class AbstractDiscreteFactorKernel(StrictModule):
    """Open factor contract defined by batched local log-score evaluation."""

    kernel_id: AbstractAttribute[str]
    capabilities: AbstractAttribute[FactorKernelCapabilities]

    @abstractmethod
    def log_scores(self, parameters: Any, states: Array, /) -> Array:
        """Return one log score for every leading factor/batch entry."""
        raise NotImplementedError


class CallableFactorKernel(AbstractDiscreteFactorKernel):
    """Pure callable finite factor with an explicit stable identity and capabilities."""

    function: Callable[[Any, Array], Array] = eqx.field(static=True)
    capabilities: FactorKernelCapabilities
    kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        function: Callable[[Any, Array], Array],
        /,
        *,
        kernel_id: str,
        capabilities: FactorKernelCapabilities | None = None,
    ):
        if not callable(function):
            raise TypeError("function must be callable.")
        if not isinstance(kernel_id, str) or not kernel_id:
            raise ValueError("kernel_id must be non-empty.")
        selected = FactorKernelCapabilities() if capabilities is None else capabilities
        if not isinstance(selected, FactorKernelCapabilities):
            raise TypeError("capabilities must be FactorKernelCapabilities or None.")
        self.function = function
        self.capabilities = selected
        self.kernel_id = kernel_id

    def log_scores(self, parameters: Any, states: Array, /) -> Array:
        values = jnp.asarray(self.function(parameters, states))
        if values.shape != states.shape[:-1]:
            raise ValueError(
                "Callable factor kernel must return states.shape[:-1]; "
                f"got {values.shape} for states {states.shape}."
            )
        if jnp.iscomplexobj(values):
            raise TypeError("Callable factor scores must be real-valued.")
        return values


class FactorGraphPrecisionPolicy(StrictModule):
    """Evaluation, accumulation, decision, and output dtypes for graph inference."""

    evaluation_dtype: str = eqx.field(static=True)
    accumulation_dtype: str = eqx.field(static=True)
    decision_dtype: str = eqx.field(static=True)
    output_dtype: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        evaluation_dtype: Literal["float32", "float64"] = "float64",
        accumulation_dtype: Literal["float32", "float64"] | None = None,
        decision_dtype: Literal["float32", "float64"] | None = None,
        output_dtype: Literal["float32", "float64"] | None = None,
    ):
        evaluation = real_precision_dtype_name(evaluation_dtype)
        if evaluation not in ("float32", "float64"):
            raise ValueError("Factor-graph evaluation supports float32 or float64.")
        accumulation = real_precision_dtype_name(
            evaluation if accumulation_dtype is None else accumulation_dtype
        )
        decision = real_precision_dtype_name(
            accumulation if decision_dtype is None else decision_dtype
        )
        output = real_precision_dtype_name(
            evaluation if output_dtype is None else output_dtype
        )
        if accumulation not in ("float32", "float64") or decision not in (
            "float32",
            "float64",
        ):
            raise ValueError("Factor-graph accumulation and decision require float32/64.")
        self.evaluation_dtype = evaluation
        self.accumulation_dtype = accumulation
        self.decision_dtype = decision
        self.output_dtype = output
        self.policy_id = canonical_fingerprint(
            {
                "kind": "factor-graph-precision",
                "evaluation": evaluation,
                "accumulation": accumulation,
                "decision": decision,
                "output": output,
            }
        )

    def evaluation(self, value: Any, /) -> Array:
        return jnp.asarray(value, dtype=self.evaluation_dtype)

    def accumulation(self, value: Any, /) -> Array:
        return jnp.asarray(value, dtype=self.accumulation_dtype)

    def decision(self, value: Any, /) -> Array:
        return jnp.asarray(value, dtype=self.decision_dtype)

    def output(self, value: Any, /) -> Array:
        return jnp.asarray(value, dtype=self.output_dtype)

    def evidence(self) -> PrecisionEvidenceEnvelope:
        effective = {
            "compute": self.evaluation_dtype,
            "accumulation": self.accumulation_dtype,
            "certification": self.decision_dtype,
            "output": self.output_dtype,
        }
        request = PrecisionRequest("factor-graph", effective)
        resolution = PrecisionResolution(request, "jax", effective)
        return PrecisionEvidenceEnvelope(resolution, effective)


class FactorGraphResourcePolicy(StrictModule):
    """Caller-controlled static resource limits for graph planning."""

    maximum_configurations: int = eqx.field(static=True)
    maximum_dense_elements: int = eqx.field(static=True)
    maximum_message_entries: int = eqx.field(static=True)
    maximum_elimination_elements: int = eqx.field(static=True)
    maximum_treewidth: int = eqx.field(static=True)
    maximum_colors: int = eqx.field(static=True)
    maximum_retained_elements: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_configurations: int = 65_536,
        maximum_dense_elements: int = 1_000_000,
        maximum_message_entries: int = 10_000_000,
        maximum_elimination_elements: int = 10_000_000,
        maximum_treewidth: int = 24,
        maximum_colors: int = 1024,
        maximum_retained_elements: int = 100_000_000,
    ):
        values = {
            "maximum_configurations": int(maximum_configurations),
            "maximum_dense_elements": int(maximum_dense_elements),
            "maximum_message_entries": int(maximum_message_entries),
            "maximum_elimination_elements": int(maximum_elimination_elements),
            "maximum_treewidth": int(maximum_treewidth),
            "maximum_colors": int(maximum_colors),
            "maximum_retained_elements": int(maximum_retained_elements),
        }
        if any(value < 1 for value in values.values()):
            raise ValueError("Every factor-graph resource limit must be positive.")
        self.maximum_configurations = values["maximum_configurations"]
        self.maximum_dense_elements = values["maximum_dense_elements"]
        self.maximum_message_entries = values["maximum_message_entries"]
        self.maximum_elimination_elements = values["maximum_elimination_elements"]
        self.maximum_treewidth = values["maximum_treewidth"]
        self.maximum_colors = values["maximum_colors"]
        self.maximum_retained_elements = values["maximum_retained_elements"]
        self.policy_id = canonical_fingerprint(
            {"kind": "factor-graph-resources", **values}
        )


class FactorExecutionEvidence(StrictModule):
    """Static resource and capability evidence for one prepared factor group."""

    capabilities: FactorKernelCapabilities
    represented_configurations: int = eqx.field(static=True)
    dense_elements: int = eqx.field(static=True)
    message_entries: int = eqx.field(static=True)
    work_estimate: int = eqx.field(static=True)
    workspace_elements: int = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        capabilities: FactorKernelCapabilities,
        /,
        *,
        represented_configurations: int,
        dense_elements: int,
        message_entries: int,
        work_estimate: int,
        workspace_elements: int,
        kernel_id: str,
    ):
        if not isinstance(capabilities, FactorKernelCapabilities):
            raise TypeError("capabilities must be FactorKernelCapabilities.")
        counts = tuple(
            int(value)
            for value in (
                represented_configurations,
                dense_elements,
                message_entries,
                work_estimate,
                workspace_elements,
            )
        )
        if any(value < 0 for value in counts):
            raise ValueError("Factor execution counts must be non-negative.")
        if not isinstance(kernel_id, str) or not kernel_id:
            raise ValueError("kernel_id must be non-empty.")
        self.capabilities = capabilities
        (
            self.represented_configurations,
            self.dense_elements,
            self.message_entries,
            self.work_estimate,
            self.workspace_elements,
        ) = counts
        self.kernel_id = kernel_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "factor-execution-evidence",
                "kernel": kernel_id,
                "capabilities": capabilities.capability_id,
                "represented_configurations": counts[0],
                "dense_elements": counts[1],
                "message_entries": counts[2],
                "work_estimate": counts[3],
                "workspace_elements": counts[4],
            }
        )


__all__ = [
    "AbstractDiscreteFactorKernel",
    "CallableFactorKernel",
    "FactorExecutionEvidence",
    "FactorGraphPrecisionPolicy",
    "FactorGraphResourcePolicy",
    "FactorKernelCapabilities",
]
