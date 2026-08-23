#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from ._fingerprint import canonical_fingerprint
from ._precision import (
    complex_precision_dtype,
    precision_dtype_name,
    precision_itemsize,
    PrecisionEvidenceEnvelope,
    PrecisionRequest,
    PrecisionResolution,
    real_precision_dtype_name,
    ScalarPrecisionDType,
)
from ._strict import StrictModule
from ._trainable import NonTrainableState
from .linalg import (
    ArraySpace,
    DiagonalPairing,
    EuclideanPairing,
    LinearSolvePolicy,
    MixedPrecisionPolicy,
    PyTreeSpace,
)


_SUPPORTED_NONLINEAR_DTYPES = frozenset(("float32", "float64", "complex64", "complex128"))


def _real_component(value: Any, /) -> str:
    name = precision_dtype_name(value)
    if name == "complex64":
        return "float32"
    if name == "complex128":
        return "float64"
    return real_precision_dtype_name(name)


def _tree_dtype(tree: Any, owner: str, /) -> ScalarPrecisionDType:
    dtypes = {
        precision_dtype_name(leaf.dtype)
        for leaf in jax.tree.leaves(tree)
        if eqx.is_inexact_array(leaf)
    }
    if len(dtypes) != 1:
        raise ValueError(
            f"{owner} must use one uniform inexact dtype; got {sorted(dtypes)}."
        )
    return next(iter(dtypes))


def _effective_dtype(
    requested: str | None,
    observed: ScalarPrecisionDType,
    /,
) -> ScalarPrecisionDType:
    if requested is None:
        return observed
    if observed in ("complex64", "complex128") and requested in (
        "float32",
        "float64",
    ):
        return complex_precision_dtype(requested)
    return precision_dtype_name(requested)


class NonlinearPrecisionPolicy(StrictModule, NonTrainableState):
    """State, residual, reduction, decision, and output precision for nonlinear work."""

    state_dtype: str | None = eqx.field(static=True)
    residual_dtype: str | None = eqx.field(static=True)
    model_dtype: str | None = eqx.field(static=True)
    direction_dtype: str | None = eqx.field(static=True)
    accumulation_dtype: str | None = eqx.field(static=True)
    decision_dtype: str | None = eqx.field(static=True)
    certificate_dtype: str | None = eqx.field(static=True)
    output_dtype: str | None = eqx.field(static=True)
    linear: MixedPrecisionPolicy | None
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        state_dtype: Any | None = None,
        residual_dtype: Any | None = None,
        model_dtype: Any | None = None,
        direction_dtype: Any | None = None,
        accumulation_dtype: Any | None = None,
        decision_dtype: Any | None = None,
        certificate_dtype: Any | None = None,
        output_dtype: Any | None = None,
        linear: MixedPrecisionPolicy | None = None,
    ):
        model = None if model_dtype is None else precision_dtype_name(model_dtype)
        state = (
            model
            if state_dtype is None and model is not None
            else (None if state_dtype is None else precision_dtype_name(state_dtype))
        )
        residual = (
            model
            if residual_dtype is None and model is not None
            else (
                None if residual_dtype is None else precision_dtype_name(residual_dtype)
            )
        )
        if model is not None and (
            (state is not None and state != model)
            or (residual is not None and residual != model)
        ):
            raise ValueError(
                "model_dtype must match explicit state_dtype and residual_dtype."
            )
        direction = (
            state if direction_dtype is None else precision_dtype_name(direction_dtype)
        )
        accumulation = (
            None
            if accumulation_dtype is None
            else precision_dtype_name(accumulation_dtype)
        )
        certificate = (
            None
            if certificate_dtype is None
            else real_precision_dtype_name(certificate_dtype)
        )
        decision = (
            certificate
            if decision_dtype is None and certificate is not None
            else (
                None
                if decision_dtype is None
                else real_precision_dtype_name(decision_dtype)
            )
        )
        if certificate is not None and decision != certificate:
            raise ValueError(
                "certificate_dtype and decision_dtype must identify one precision."
            )
        output = None if output_dtype is None else precision_dtype_name(output_dtype)
        if linear is not None and not isinstance(linear, MixedPrecisionPolicy):
            raise TypeError("linear must be a MixedPrecisionPolicy or None.")
        requested = tuple(
            value
            for value in (
                model,
                state,
                residual,
                direction,
                accumulation,
                decision,
                output,
            )
            if value is not None
        )
        if any(value not in _SUPPORTED_NONLINEAR_DTYPES for value in requested):
            raise ValueError("Nonlinear precision supports float32/64 and complex64/128.")
        reduction_inputs = tuple(
            value for value in (residual, direction) if value is not None
        )
        if accumulation is not None and any(
            precision_itemsize(_real_component(accumulation))
            < precision_itemsize(_real_component(value))
            for value in reduction_inputs
        ):
            raise ValueError(
                "Nonlinear accumulation precision cannot be narrower than "
                "residual or direction precision."
            )
        effective_accumulation = (
            accumulation
            if accumulation is not None
            else (residual if residual is not None else direction)
        )
        if (
            decision is not None
            and effective_accumulation is not None
            and precision_itemsize(decision)
            < precision_itemsize(_real_component(effective_accumulation))
        ):
            raise ValueError(
                "Nonlinear decision precision cannot be narrower than accumulation."
            )
        request = PrecisionRequest(
            "nonlinear",
            {
                "storage": state,
                "compute": residual,
                "basis": direction,
                "residual": residual,
                "accumulation": accumulation,
                "certification": decision,
                "output": output,
            },
        )
        self.model_dtype = model
        self.state_dtype = state
        self.residual_dtype = residual
        self.direction_dtype = direction
        self.accumulation_dtype = accumulation
        self.certificate_dtype = decision
        self.decision_dtype = decision
        self.output_dtype = output
        self.linear = linear
        self.policy_id = canonical_fingerprint(
            {
                "request": request.request_id,
                "linear": None if linear is None else repr(linear),
            }
        )

    @property
    def request(self) -> PrecisionRequest:
        return PrecisionRequest(
            "nonlinear",
            {
                "storage": self.state_dtype,
                "compute": self.residual_dtype,
                "basis": self.direction_dtype,
                "residual": self.residual_dtype,
                "accumulation": self.accumulation_dtype,
                "certification": self.decision_dtype,
                "output": self.output_dtype,
            },
        )

    def validate_trees(
        self,
        state: Any,
        residual: Any,
        /,
    ) -> tuple[ScalarPrecisionDType, ScalarPrecisionDType]:
        state_dtype = _tree_dtype(state, "Nonlinear state")
        residual_dtype = _tree_dtype(residual, "Nonlinear residual")
        if self.state_dtype is not None and state_dtype != self.state_dtype:
            raise TypeError(
                f"Nonlinear state dtype {state_dtype} does not match {self.state_dtype}."
            )
        if self.residual_dtype is not None and residual_dtype != self.residual_dtype:
            raise TypeError(
                "Nonlinear residual dtype "
                f"{residual_dtype} does not match {self.residual_dtype}."
            )
        return state_dtype, residual_dtype

    def state(self, value: Any, /):
        return self._cast_tree(value, self.state_dtype)

    def residual(self, value: Any, /):
        return self._cast_tree(value, self.residual_dtype)

    def _cast_tree(self, value: Any, dtype: str | None, /):
        if dtype is None:
            return value
        return jax.tree.map(
            lambda leaf: (
                leaf.astype(_effective_dtype(dtype, precision_dtype_name(leaf.dtype)))
                if eqx.is_inexact_array(leaf)
                else leaf
            ),
            value,
        )

    def model(self, value: Any, /):
        return self._cast_tree(value, self.model_dtype)

    def direction(self, value: Any, /):
        return self._cast_tree(value, self.direction_dtype)

    def certificate(self, value: Any, /):
        return self._cast_tree(value, self.certificate_dtype)

    def bind_linear(self, policy: LinearSolvePolicy, /) -> LinearSolvePolicy:
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("policy must be a LinearSolvePolicy.")
        if self.linear is None:
            return policy
        return eqx.tree_at(lambda value: value.precision, policy, self.linear)

    def validate_tolerance(self, tolerance: Any, /) -> None:
        value = float(tolerance)
        if not jnp.isfinite(value) or value < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")
        if self.certificate_dtype is None:
            return
        epsilon = float(jnp.finfo(jnp.dtype(self.certificate_dtype)).eps)
        if 0.0 < value < epsilon:
            raise ValueError(
                "Requested tolerance is below certificate precision epsilon."
            )

    def accumulation(self, value: Any, /):
        array = jnp.asarray(value)
        if self.accumulation_dtype is None or not jnp.issubdtype(
            array.dtype, jnp.inexact
        ):
            return array
        observed = precision_dtype_name(array.dtype)
        return array.astype(_effective_dtype(self.accumulation_dtype, observed))

    def decision(self, value: Any, /):
        array = jnp.asarray(value)
        return array if self.decision_dtype is None else array.astype(self.decision_dtype)

    def output(self, value: Any, /):
        array = jnp.asarray(value)
        if self.output_dtype is None or not jnp.issubdtype(array.dtype, jnp.inexact):
            return array
        observed = precision_dtype_name(array.dtype)
        return array.astype(_effective_dtype(self.output_dtype, observed))

    def validate_accumulation_space(self, space: Any, /) -> None:
        if self.accumulation_dtype is None:
            return
        supported = (
            isinstance(space, ArraySpace)
            and isinstance(space.pairing, (EuclideanPairing, DiagonalPairing))
        ) or (
            isinstance(space, PyTreeSpace) and isinstance(space.pairing, EuclideanPairing)
        )
        if not supported:
            raise TypeError(
                "Widened nonlinear accumulation requires an ArraySpace with "
                "Euclidean or diagonal pairing, or a Euclidean PyTreeSpace."
            )

    def inner(self, space: Any, left: Any, right: Any, /):
        if self.accumulation_dtype is None:
            return space.inner(left, right)
        self.validate_accumulation_space(space)
        if isinstance(space, PyTreeSpace):
            left_tree = space.validate(left)
            right_tree = space.validate(right)
            left_leaves = jax.tree.leaves(left_tree)
            right_leaves = jax.tree.leaves(right_tree)
            target = _effective_dtype(
                self.accumulation_dtype,
                precision_dtype_name(left_leaves[0].dtype),
            )
            terms = tuple(
                jnp.sum(jnp.conj(left_leaf.astype(target)) * right_leaf.astype(target))
                for left_leaf, right_leaf in zip(
                    left_leaves,
                    right_leaves,
                    strict=True,
                )
            )
            result = terms[0]
            for term in terms[1:]:
                result = result + term
            return result
        assert isinstance(space, ArraySpace)
        left_array = space.validate(left)
        right_array = space.validate(right)
        target = _effective_dtype(
            self.accumulation_dtype,
            precision_dtype_name(left_array.dtype),
        )
        left_array = left_array.astype(target)
        right_array = right_array.astype(target)
        if isinstance(space.pairing, DiagonalPairing):
            weights = space.pairing.weights.astype(_real_component(target))
            return jnp.sum(jnp.conj(left_array) * weights * right_array)
        return jnp.sum(jnp.conj(left_array) * right_array)

    def norm(self, space: Any, value: Any, /):
        squared = jnp.real(self.inner(space, value, value))
        return self.decision(jnp.sqrt(jnp.maximum(squared, 0.0)))

    def evidence_for(
        self,
        state: Any,
        residual: Any,
        /,
        *,
        children: dict[str, PrecisionEvidenceEnvelope] | None = None,
        output_value: Any | None = None,
    ) -> PrecisionEvidenceEnvelope:
        state_observed, residual_observed = self.validate_trees(state, residual)
        state_effective = _effective_dtype(self.state_dtype, state_observed)
        residual_effective = _effective_dtype(self.residual_dtype, residual_observed)
        accumulation_effective = _effective_dtype(
            self.accumulation_dtype,
            residual_effective,
        )
        decision_effective = (
            _real_component(accumulation_effective)
            if self.decision_dtype is None
            else self.decision_dtype
        )
        output_observed = (
            state_observed
            if output_value is None
            else _tree_dtype(output_value, "Nonlinear output")
        )
        output_effective = _effective_dtype(self.output_dtype, output_observed)
        request = self.request
        resolution = PrecisionResolution(
            request,
            "phydrax-nonlinear",
            {
                "storage": state_effective,
                "compute": residual_effective,
                "residual": residual_effective,
                "accumulation": accumulation_effective,
                "basis": _effective_dtype(
                    self.direction_dtype,
                    state_effective,
                ),
                "certification": decision_effective,
                "output": output_effective,
            },
        )
        return PrecisionEvidenceEnvelope(
            resolution,
            dict(resolution.effective),
            children={} if children is None else children,
        )


__all__ = ["NonlinearPrecisionPolicy"]
