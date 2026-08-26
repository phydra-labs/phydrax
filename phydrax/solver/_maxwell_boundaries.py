#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import NormCompatibleInterpolationPlan, StructuredCochainBridge


MaxwellBoundaryKind: TypeAlias = Literal["pec", "pmc", "impedance"]


class MaxwellBoundaryPlan(StrictModule):
    """Structured compatible trace condition with explicit power semantics."""

    kind: MaxwellBoundaryKind = eqx.field(static=True)
    admittance: Array | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: MaxwellBoundaryKind,
        /,
        *,
        admittance: ArrayLike | None = None,
    ):
        if kind not in ("pec", "pmc", "impedance"):
            raise ValueError("Unknown Maxwell boundary kind.")
        if kind == "impedance":
            if admittance is None:
                raise ValueError("Impedance boundaries require admittance.")
            value = jnp.asarray(admittance)
            if not jnp.issubdtype(value.dtype, jnp.inexact):
                value = value.astype(float)
            value = eqx.error_if(
                value,
                jnp.any(~jnp.isfinite(value)) | jnp.any(jnp.real(value) < 0.0),
                "Passive boundary admittance must be finite with nonnegative real part.",
            )
        else:
            if admittance is not None:
                raise ValueError("Only impedance boundaries accept admittance.")
            value = None
        self.kind = kind
        self.admittance = value
        self.plan_id = canonical_fingerprint(
            {
                "kind": "maxwell-boundary-plan",
                "boundary_kind": kind,
                "admittance": (None if value is None else array_tree_fingerprint(value)),
            }
        )

    def prepare(self, bridge: StructuredCochainBridge, /) -> PreparedMaxwellBoundary:
        return PreparedMaxwellBoundary(self, bridge)


class PreparedMaxwellBoundary(StrictModule):
    """Boundary masks and passive surface action on compatible cochains."""

    kind: MaxwellBoundaryKind = eqx.field(static=True)
    electric_boundary: Array
    magnetic_boundary: Array
    admittance: Array | None
    electric_measure: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: MaxwellBoundaryPlan,
        bridge: StructuredCochainBridge,
        /,
    ):
        if not isinstance(bridge, StructuredCochainBridge):
            raise TypeError("bridge must be a StructuredCochainBridge.")
        electric_boundary = jnp.asarray(bridge.cochain.boundary_masks[1], dtype=bool)
        magnetic_boundary = jnp.asarray(bridge.cochain.boundary_masks[2], dtype=bool)
        admittance = plan.admittance
        if admittance is not None:
            if admittance.shape not in ((), (1,), electric_boundary.shape):
                raise ValueError(
                    "Boundary admittance must be scalar or align with electric cochains."
                )
            admittance = jnp.broadcast_to(admittance, electric_boundary.shape)
        self.kind = plan.kind
        self.electric_boundary = electric_boundary
        self.magnetic_boundary = magnetic_boundary
        self.admittance = admittance
        self.electric_measure = bridge.cochain.hodge_stars[1]
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-maxwell-boundary",
                "plan": plan.plan_id,
                "bridge": bridge.bridge_id,
            }
        )

    def constrain_primary(
        self,
        displacement: ArrayLike,
        magnetic_flux: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        displacement_ = jnp.asarray(displacement)
        magnetic_ = jnp.asarray(magnetic_flux)
        if self.kind == "pec":
            displacement_ = jnp.where(self.electric_boundary, 0, displacement_)
        elif self.kind == "pmc":
            magnetic_ = jnp.where(self.magnetic_boundary, 0, magnetic_)
        return displacement_, magnetic_

    def constrain_fields(
        self,
        electric: ArrayLike,
        magnetic: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        electric_ = jnp.asarray(electric)
        magnetic_ = jnp.asarray(magnetic)
        if self.kind == "pec":
            electric_ = jnp.where(self.electric_boundary, 0, electric_)
        elif self.kind == "pmc":
            magnetic_ = jnp.where(self.magnetic_boundary, 0, magnetic_)
        return electric_, magnetic_

    def impedance_current(self, electric: ArrayLike, /) -> Array:
        value = jnp.asarray(electric)
        if self.kind != "impedance" or self.admittance is None:
            return jnp.zeros_like(value)
        return jnp.where(self.electric_boundary, self.admittance * value, 0)

    def dissipated_power(self, electric: ArrayLike, /) -> Array:
        value = jnp.asarray(electric)
        if self.kind != "impedance" or self.admittance is None:
            return jnp.asarray(0.0, dtype=value.real.dtype)
        density = jnp.real(self.admittance) * jnp.real(value * jnp.conj(value))
        return jnp.sum(
            jnp.where(self.electric_boundary, self.electric_measure * density, 0)
        )


class BlochCochainCalculus(StrictModule, NonTrainableState):
    """Unitary gauge-twisted cochain calculus on a periodic structured quotient."""

    bridge: StructuredCochainBridge
    wavevector: Array
    gauges: tuple[Array, ...]
    phases: Array
    calculus_id: str = eqx.field(static=True)

    def __init__(
        self,
        bridge: StructuredCochainBridge,
        wavevector: ArrayLike,
        /,
    ):
        if not isinstance(bridge, StructuredCochainBridge):
            raise TypeError("bridge must be a StructuredCochainBridge.")
        if not all(axis.periodic for axis in bridge.grid.structured_axes):
            raise ValueError("Bloch calculus requires periodic quotient axes.")
        wavevector_ = jnp.asarray(wavevector, dtype=float)
        if wavevector_.shape != (bridge.dimension,):
            raise ValueError("wavevector must have one entry per structured axis.")
        if bool(jnp.any(~jnp.isfinite(wavevector_))):
            raise ValueError("wavevector must be finite.")
        gauges = tuple(
            jnp.exp(1j * (coordinates @ wavevector_))
            for coordinates in bridge.cochain.coordinates
            if coordinates is not None
        )
        if len(gauges) != bridge.dimension + 1:
            raise RuntimeError("Bloch calculus requires coordinates on every degree.")
        lengths = jnp.asarray(
            [jnp.sum(axis.interval_widths) for axis in bridge.grid.structured_axes]
        )
        self.bridge = bridge
        self.wavevector = wavevector_
        self.gauges = gauges
        self.phases = jnp.exp(1j * wavevector_ * lengths)
        self.calculus_id = canonical_fingerprint(
            {
                "kind": "bloch-cochain-calculus",
                "bridge": bridge.bridge_id,
                "wavevector": array_tree_fingerprint(wavevector_),
            }
        )

    def exterior_derivative(self, degree: int, values: ArrayLike, /) -> Array:
        degree_ = int(degree)
        value = jnp.asarray(values)
        return jnp.conj(self.gauges[degree_ + 1]) * self.bridge.exterior_derivative(
            degree_, self.gauges[degree_] * value
        )

    def codifferential(self, degree: int, values: ArrayLike, /) -> Array:
        degree_ = int(degree)
        value = jnp.asarray(values)
        return jnp.conj(self.gauges[degree_ - 1]) * self.bridge.codifferential(
            degree_, self.gauges[degree_] * value
        )

    def chain_residual(self, degree: int, values: ArrayLike, /) -> Array:
        first = self.exterior_derivative(degree, values)
        return self.exterior_derivative(degree + 1, first)


class MaxwellInterfaceJump(StrictModule):
    """Paired conforming trace jump with explicit orientation."""

    left_indices: Array
    right_indices: Array
    orientation: Array
    jump: Array
    interface_id: str = eqx.field(static=True)

    def __init__(
        self,
        left_indices: ArrayLike,
        right_indices: ArrayLike,
        /,
        *,
        orientation: ArrayLike = 1.0,
        jump: ArrayLike = 0.0,
    ):
        left = np.asarray(left_indices)
        right = np.asarray(right_indices)
        if left.ndim != 1 or right.shape != left.shape:
            raise ValueError("Interface trace indices must be paired rank-one arrays.")
        if not np.issubdtype(left.dtype, np.integer) or not np.issubdtype(
            right.dtype, np.integer
        ):
            raise TypeError("Interface trace indices must be integers.")
        orientation_ = jnp.broadcast_to(jnp.asarray(orientation), left.shape)
        jump_ = jnp.broadcast_to(jnp.asarray(jump), left.shape)
        if bool(jnp.any(jnp.abs(orientation_) != 1.0)):
            raise ValueError("Interface orientation coefficients must be ±1.")
        self.left_indices = jnp.asarray(left, dtype=jnp.int32)
        self.right_indices = jnp.asarray(right, dtype=jnp.int32)
        self.orientation = orientation_
        self.jump = jump_
        self.interface_id = canonical_fingerprint(
            {
                "kind": "maxwell-interface-jump",
                "left": array_tree_fingerprint(left),
                "right": array_tree_fingerprint(right),
                "orientation": array_tree_fingerprint(orientation_),
                "jump": array_tree_fingerprint(jump_),
            }
        )

    def residual(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_ = jnp.asarray(left)[self.left_indices]
        right_ = jnp.asarray(right)[self.right_indices]
        return right_ - self.orientation * left_ - self.jump

    def enforce(self, left: ArrayLike, right: ArrayLike, /) -> tuple[Array, Array]:
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        target = self.orientation * left_[self.left_indices] + self.jump
        return left_, right_.at[self.right_indices].set(target)


class MaxwellInterfaceMortar(StrictModule, NonTrainableState):
    """Norm-compatible nonconforming trace transfer."""

    interpolation: NormCompatibleInterpolationPlan
    mortar_id: str = eqx.field(static=True)

    def __init__(self, interpolation: NormCompatibleInterpolationPlan, /):
        if not isinstance(interpolation, NormCompatibleInterpolationPlan):
            raise TypeError("interpolation must be NormCompatibleInterpolationPlan.")
        self.interpolation = interpolation
        self.mortar_id = canonical_fingerprint(
            {"kind": "maxwell-interface-mortar", "plan": interpolation.plan_id}
        )

    def traces(self, left: ArrayLike, right: ArrayLike, /) -> tuple[Array, Array]:
        return (
            self.interpolation.left_to_mortar(jnp.asarray(left)),
            self.interpolation.right_to_mortar(jnp.asarray(right)),
        )

    def restrict(
        self,
        left_mortar: ArrayLike,
        right_mortar: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        return (
            self.interpolation.mortar_to_left(jnp.asarray(left_mortar)),
            self.interpolation.mortar_to_right(jnp.asarray(right_mortar)),
        )


__all__ = [
    "BlochCochainCalculus",
    "MaxwellBoundaryKind",
    "MaxwellBoundaryPlan",
    "MaxwellInterfaceJump",
    "MaxwellInterfaceMortar",
    "PreparedMaxwellBoundary",
]
