#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class VortexSaltationEvidence(StrictModule):
    denominator: Array
    transversality_margin: Array
    reset_shape_valid: Array
    finite: Array
    differentiable: Array
    saltation_id: str = eqx.field(static=True)


class VortexSaltationMap(StrictModule, NonTrainableState):
    matrix: Array
    evidence: VortexSaltationEvidence
    saltation_id: str = eqx.field(static=True)

    @classmethod
    def build(
        cls,
        reset_jacobian: ArrayLike,
        pre_event_field: ArrayLike,
        post_event_field: ArrayLike,
        surface_normal: ArrayLike,
        surface_time_derivative: ArrayLike = 0.0,
        /,
        *,
        transversality_tolerance: float = 1.0e-10,
    ):
        reset, before, after, normal = (
            jnp.asarray(value)
            for value in (
                reset_jacobian,
                pre_event_field,
                post_event_field,
                surface_normal,
            )
        )
        time_derivative = jnp.asarray(surface_time_derivative, dtype=before.dtype)
        if (
            reset.ndim != 2
            or reset.shape[1] != before.size
            or reset.shape[0] != after.size
            or normal.shape != before.shape
            or time_derivative.shape != ()
        ):
            raise ValueError("Saltation reset/vector/surface shapes are incompatible.")
        mapped_before = reset @ before
        denominator = jnp.sum(normal * before) + time_derivative
        differentiable = jnp.abs(denominator) > transversality_tolerance
        safe = jnp.where(differentiable, denominator, 1.0)
        matrix = reset + (after - mapped_before)[:, None] * normal[None, :] / safe
        matrix = jnp.where(differentiable, matrix, reset)
        finite = jnp.all(jnp.isfinite(matrix)) & jnp.isfinite(denominator)
        identifier = canonical_fingerprint(
            {
                "kind": "vortex-saltation-map",
                "source_size": int(before.size),
                "target_size": int(after.size),
                "transversality_tolerance": float(transversality_tolerance),
            }
        )
        evidence = VortexSaltationEvidence(
            denominator,
            jnp.abs(denominator),
            jnp.asarray(True),
            finite,
            differentiable,
            identifier,
        )
        return cls(matrix, evidence, identifier)

    def jvp(self, tangent: ArrayLike, /) -> Array:
        value = jnp.asarray(tangent)
        if value.shape != (self.matrix.shape[1],):
            raise ValueError("Saltation tangent shape is incompatible.")
        return contract("ij,j->i", self.matrix, value)

    def vjp(self, cotangent: ArrayLike, /) -> Array:
        value = jnp.asarray(cotangent)
        if value.shape != (self.matrix.shape[0],):
            raise ValueError("Saltation cotangent shape is incompatible.")
        return contract("ij,i->j", self.matrix, value)


class UndefinedTopologyDerivative(StrictModule, NonTrainableState):
    event_name: str = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    derivative_id: str = eqx.field(static=True)

    def __init__(self, event_name: str, reason: str, /):
        if not str(event_name) or not str(reason):
            raise ValueError("Undefined derivative requires event name and reason.")
        self.event_name, self.reason = str(event_name), str(reason)
        self.derivative_id = canonical_fingerprint(
            {
                "kind": "undefined-vortex-topology-derivative",
                "event_name": self.event_name,
                "reason": self.reason,
            }
        )

    def apply(self, value: ArrayLike, /) -> Array:
        return eqx.error_if(
            jnp.asarray(value),
            jnp.asarray(True),
            f"Topology derivative is undefined for {self.event_name}: {self.reason}",
        )


__all__ = ["UndefinedTopologyDerivative", "VortexSaltationEvidence", "VortexSaltationMap"]
