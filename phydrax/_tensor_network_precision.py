#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from ._precision import (
    complex_precision_dtype,
    precision_dtype_name,
    precision_itemsize,
    PrecisionEvidenceEnvelope,
    PrecisionRequest,
    PrecisionResolution,
    real_precision_dtype_name,
)
from ._strict import StrictModule
from ._trainable import NonTrainableState


_SUPPORTED_DTYPES = frozenset(("float32", "float64", "complex64", "complex128"))


def _real_component(value: Any, /) -> str:
    name = precision_dtype_name(value)
    if name == "complex64":
        return "float32"
    if name == "complex128":
        return "float64"
    return real_precision_dtype_name(name)


def _effective_dtype(requested: str | None, observed: str, /) -> str:
    if requested is None:
        return observed
    if observed in ("complex64", "complex128") and requested in (
        "float32",
        "float64",
    ):
        return complex_precision_dtype(requested)
    return precision_dtype_name(requested)


def _tree_dtype(value: Any, owner: str, /) -> str:
    dtypes = {
        precision_dtype_name(leaf.dtype)
        for leaf in jax.tree.leaves(value)
        if eqx.is_inexact_array(leaf)
    }
    if len(dtypes) != 1:
        raise ValueError(
            f"{owner} must use one uniform inexact dtype; got {sorted(dtypes)}."
        )
    return next(iter(dtypes))


class TensorNetworkPrecisionPolicy(StrictModule, NonTrainableState):
    """Storage, contraction, factorization, reduction, and certification precision."""

    storage_dtype: str | None = eqx.field(static=True)
    contraction_dtype: str | None = eqx.field(static=True)
    factorization_dtype: str | None = eqx.field(static=True)
    accumulation_dtype: str | None = eqx.field(static=True)
    decision_dtype: str | None = eqx.field(static=True)
    output_dtype: str | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        storage_dtype: Any | None = None,
        contraction_dtype: Any | None = None,
        factorization_dtype: Any | None = None,
        accumulation_dtype: Any | None = None,
        decision_dtype: Any | None = None,
        output_dtype: Any | None = None,
    ):
        storage = None if storage_dtype is None else precision_dtype_name(storage_dtype)
        contraction = (
            None if contraction_dtype is None else precision_dtype_name(contraction_dtype)
        )
        factorization = (
            None
            if factorization_dtype is None
            else precision_dtype_name(factorization_dtype)
        )
        accumulation = (
            None
            if accumulation_dtype is None
            else precision_dtype_name(accumulation_dtype)
        )
        decision = (
            None if decision_dtype is None else real_precision_dtype_name(decision_dtype)
        )
        output = None if output_dtype is None else precision_dtype_name(output_dtype)
        requested = tuple(
            value
            for value in (
                storage,
                contraction,
                factorization,
                accumulation,
                decision,
                output,
            )
            if value is not None
        )
        if any(value not in _SUPPORTED_DTYPES for value in requested):
            raise ValueError(
                "Tensor-network precision supports float32/64 and complex64/128."
            )
        reduction_inputs = tuple(
            value for value in (contraction, factorization) if value is not None
        )
        if accumulation is not None and any(
            precision_itemsize(_real_component(accumulation))
            < precision_itemsize(_real_component(value))
            for value in reduction_inputs
        ):
            raise ValueError(
                "Tensor-network accumulation cannot be narrower than contraction "
                "or factorization precision."
            )
        if (
            decision is not None
            and accumulation is not None
            and precision_itemsize(decision)
            < precision_itemsize(_real_component(accumulation))
        ):
            raise ValueError(
                "Tensor-network decision precision cannot be narrower than accumulation."
            )
        request = PrecisionRequest(
            "tensor-network",
            {
                "storage": storage,
                "compute": contraction,
                "factorization": factorization,
                "accumulation": accumulation,
                "certification": decision,
                "output": output,
            },
        )
        self.storage_dtype = storage
        self.contraction_dtype = contraction
        self.factorization_dtype = factorization
        self.accumulation_dtype = accumulation
        self.decision_dtype = decision
        self.output_dtype = output
        self.policy_id = request.request_id

    @property
    def request(self) -> PrecisionRequest:
        return PrecisionRequest(
            "tensor-network",
            {
                "storage": self.storage_dtype,
                "compute": self.contraction_dtype,
                "factorization": self.factorization_dtype,
                "accumulation": self.accumulation_dtype,
                "certification": self.decision_dtype,
                "output": self.output_dtype,
            },
        )

    def validate_storage(self, value: Any, /) -> str:
        observed = _tree_dtype(value, "Tensor-network storage")
        if self.storage_dtype is not None and observed != self.storage_dtype:
            raise TypeError(
                f"Tensor-network storage dtype {observed} does not match "
                f"{self.storage_dtype}."
            )
        return observed

    def _cast(self, value: Any, requested: str | None, /):
        if requested is None:
            return value
        return jax.tree.map(
            lambda leaf: (
                leaf.astype(_effective_dtype(requested, precision_dtype_name(leaf.dtype)))
                if eqx.is_inexact_array(leaf)
                else leaf
            ),
            value,
        )

    def storage(self, value: Any, /):
        return self._cast(value, self.storage_dtype)

    def contraction(self, value: Any, /):
        return self._cast(value, self.contraction_dtype)

    def factorization(self, value: Any, /):
        return self._cast(value, self.factorization_dtype)

    def accumulation(self, value: Any, /):
        return self._cast(value, self.accumulation_dtype)

    def decision(self, value: Any, /):
        array = jnp.asarray(value)
        return array if self.decision_dtype is None else array.astype(self.decision_dtype)

    def output(self, value: Any, /):
        return self._cast(value, self.output_dtype)

    def sum(self, value: Any, /, *, axis: Any = None, keepdims: bool = False):
        return jnp.sum(
            self.accumulation(value),
            axis=axis,
            keepdims=keepdims,
        )

    def norm(self, value: Any, /, *, axis: Any = None, keepdims: bool = False):
        accumulated = self.accumulation(value)
        squared = jnp.sum(
            jnp.real(jnp.conj(accumulated) * accumulated),
            axis=axis,
            keepdims=keepdims,
        )
        return self.decision(jnp.sqrt(jnp.maximum(squared, 0.0)))

    def evidence_for(
        self,
        value: Any,
        /,
        *,
        children: dict[str, PrecisionEvidenceEnvelope] | None = None,
        output_value: Any | None = None,
    ) -> PrecisionEvidenceEnvelope:
        observed = self.validate_storage(value)
        contraction = _effective_dtype(self.contraction_dtype, observed)
        factorization = _effective_dtype(self.factorization_dtype, contraction)
        accumulation = _effective_dtype(self.accumulation_dtype, contraction)
        decision = (
            _real_component(accumulation)
            if self.decision_dtype is None
            else self.decision_dtype
        )
        output_observed = (
            observed
            if output_value is None
            else _tree_dtype(output_value, "Tensor-network output")
        )
        output = _effective_dtype(self.output_dtype, output_observed)
        resolution = PrecisionResolution(
            self.request,
            "phydrax-tensor-network",
            {
                "storage": observed,
                "compute": contraction,
                "factorization": factorization,
                "accumulation": accumulation,
                "certification": decision,
                "output": output,
            },
        )
        return PrecisionEvidenceEnvelope(
            resolution,
            dict(resolution.effective),
            children={} if children is None else children,
        )


__all__ = ["TensorNetworkPrecisionPolicy"]
