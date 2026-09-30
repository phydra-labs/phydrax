#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import NormCompatibleInterpolationPlan, StructuredCochainBridge
from ..discretization._cell_de_rham import AbstractCellDeRhamComplex
from ..linalg import AbstractVectorSpace, apply_real_map_componentwise, ArraySpace
from ..typing import parse


MaxwellBoundaryKind: TypeAlias = Literal["pec", "pmc", "impedance"]


class MaxwellBoundaryPlan(StrictModule):
    """Structured compatible trace condition with explicit power semantics.

    Without ``support`` the condition acts on the trace of the nonperiodic
    domain boundary. ``support`` is an explicit boolean mask over the
    constrained cochain (electric for ``"pec"``/``"impedance"``, magnetic for
    ``"pmc"``) that replaces that trace, so conductors, gratings, and screens
    inside the domain are staircased onto the entities they contain.
    """

    kind: MaxwellBoundaryKind = eqx.field(static=True)
    admittance: Array | None
    support: Array | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: MaxwellBoundaryKind,
        /,
        *,
        admittance: ArrayLike | None = None,
        support: ArrayLike | None = None,
    ) -> None:
        kind = parse(kind, MaxwellBoundaryKind, "kind")
        if kind == "impedance":
            if admittance is None:
                raise ValueError("Impedance boundaries require admittance.")
            value = jnp.asarray(admittance)
            if not jnp.issubdtype(value.dtype, jnp.inexact):
                value = value.astype("float64")
            value = eqx.error_if(
                value,
                jnp.any(~jnp.isfinite(value)) | jnp.any(jnp.real(value) < 0.0),
                "Passive boundary admittance must be finite with nonnegative real part.",
            )
        else:
            if admittance is not None:
                raise ValueError("Only impedance boundaries accept admittance.")
            value = None
        if support is None:
            mask = None
        else:
            host = np.asarray(support)
            if host.dtype != np.bool_ or host.ndim != 1:
                raise TypeError("Boundary support must be a boolean cochain mask.")
            mask = jnp.asarray(host)
        self.kind = kind
        self.admittance = value
        self.support = mask
        self.plan_id = canonical_fingerprint(
            {
                "kind": "maxwell-boundary-plan",
                "boundary_kind": kind,
                "admittance": (None if value is None else array_tree_fingerprint(value)),
                "support": None if mask is None else array_tree_fingerprint(mask),
            }
        )

    def prepare(
        self,
        bridge: AbstractCellDeRhamComplex,
        layout: Any,
        /,
    ) -> PreparedMaxwellBoundary:
        return PreparedMaxwellBoundary(self, bridge, layout)


class PreparedMaxwellBoundary(StrictModule):
    """Boundary masks and passive surface action on compatible cochains."""

    kind: MaxwellBoundaryKind = eqx.field(static=True)
    electric_boundary: Array
    magnetic_boundary: Array
    admittance: Array | None
    electric_measure: AbstractVectorSpace
    magnetic_closedness_preserving: bool = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: MaxwellBoundaryPlan,
        bridge: AbstractCellDeRhamComplex,
        layout: Any,
        /,
    ) -> None:
        if not isinstance(bridge, AbstractCellDeRhamComplex):
            raise TypeError("bridge must be an AbstractCellDeRhamComplex.")
        electric_boundary = jnp.asarray(
            bridge.boundary_masks[layout.electric_degree],
            dtype=jnp.bool_,
        )
        magnetic_boundary = jnp.asarray(
            bridge.boundary_masks[layout.magnetic_degree],
            dtype=jnp.bool_,
        )
        if layout.polarization == "tez" and isinstance(bridge, StructuredCochainBridge):
            shape = bridge.orientation_shapes[layout.magnetic_degree][0]
            adjacent = np.zeros(shape, dtype=np.bool_)
            for axis, structured_axis in enumerate(bridge.grid.structured_axes):
                if structured_axis.periodic:
                    continue
                lower: list[slice | int] = [slice(None)] * bridge.dimension
                upper: list[slice | int] = [slice(None)] * bridge.dimension
                lower[axis], upper[axis] = 0, shape[axis] - 1
                adjacent[tuple(lower)] = True
                adjacent[tuple(upper)] = True
            magnetic_boundary = jnp.asarray(adjacent.reshape((-1,)))
        if plan.support is not None:
            constrained = magnetic_boundary if plan.kind == "pmc" else electric_boundary
            if plan.support.shape != constrained.shape:
                raise ValueError(
                    f"{plan.kind} boundary support must have shape {constrained.shape}."
                )
            if plan.kind == "pmc":
                magnetic_boundary = jnp.asarray(plan.support)
            else:
                electric_boundary = jnp.asarray(plan.support)
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
        self.electric_measure = bridge.hilbert_complex(boundary="absolute").space(
            layout.electric_degree
        )
        self.magnetic_closedness_preserving = plan.kind != "pmc"
        self.layout_id = layout.layout_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-maxwell-boundary",
                "plan": plan.plan_id,
                "realization": bridge.realization_id,
                "layout": layout.layout_id,
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
        current = self.impedance_current(value)
        metric = self.electric_measure
        paired = (
            apply_real_map_componentwise(metric.riesz, current)
            if isinstance(metric, ArraySpace)
            and not jnp.issubdtype(metric.dtype, jnp.complexfloating)
            else metric.riesz(current)
        )
        return jnp.real(jnp.vdot(value, paired))


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
    ) -> None:
        left = np.asarray(left_indices)
        right = np.asarray(right_indices)
        if (
            left.ndim != 1
            or right.shape != left.shape
            or not np.issubdtype(left.dtype, np.signedinteger)
            or not np.issubdtype(right.dtype, np.signedinteger)
            or np.any(left < 0)
            or np.any(right < 0)
            or np.unique(left).size != left.size
            or np.unique(right).size != right.size
        ):
            raise ValueError(
                "Interface trace indices must be unique paired nonnegative signed integers."
            )
        orientation_ = jnp.broadcast_to(jnp.asarray(orientation), left.shape)
        jump_ = jnp.broadcast_to(jnp.asarray(jump), left.shape)
        if bool(
            jnp.any(~jnp.isfinite(orientation_))
            | jnp.any(jnp.abs(orientation_) != 1.0)
            | jnp.any(~jnp.isfinite(jump_))
        ):
            raise ValueError(
                "Interface orientations must be finite ±1 and jumps must be finite."
            )
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
        left_value = jnp.asarray(left)
        right_value = jnp.asarray(right)
        if (
            left_value.ndim != 1
            or right_value.ndim != 1
            or bool(jnp.any(self.left_indices >= left_value.size))
            or bool(jnp.any(self.right_indices >= right_value.size))
        ):
            raise ValueError("Interface trace indices exceed the supplied traces.")
        selected_left = left_value[self.left_indices]
        selected_right = right_value[self.right_indices]
        return selected_right - self.orientation * selected_left - self.jump

    def enforce(self, left: ArrayLike, right: ArrayLike, /) -> tuple[Array, Array]:
        left_value = jnp.asarray(left)
        right_value = jnp.asarray(right)
        if (
            left_value.ndim != 1
            or right_value.ndim != 1
            or bool(jnp.any(self.left_indices >= left_value.size))
            or bool(jnp.any(self.right_indices >= right_value.size))
        ):
            raise ValueError("Interface trace indices exceed the supplied traces.")
        target = self.orientation * left_value[self.left_indices] + self.jump
        return left_value, right_value.at[self.right_indices].set(target)


class MaxwellInterfaceMortar(StrictModule, NonTrainableState):
    """Norm-compatible nonconforming trace transfer."""

    interpolation: NormCompatibleInterpolationPlan
    mortar_id: str = eqx.field(static=True)

    def __init__(self, interpolation: NormCompatibleInterpolationPlan, /) -> None:
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
    "MaxwellBoundaryKind",
    "MaxwellBoundaryPlan",
    "MaxwellInterfaceJump",
    "MaxwellInterfaceMortar",
    "PreparedMaxwellBoundary",
]
