#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..metrix._adm_exchange import ADMGridGeometry


GRMHDBoundaryKind: TypeAlias = Literal[
    "outflow", "horizon_outflow", "reflective", "conducting", "prescribed"
]
GRMHDBoundarySide: TypeAlias = Literal["lower", "upper"]


class GRMHDBoundaryTrace(StrictModule):
    exterior_primitive: Array
    boundary_electromotive: Array
    incoming_characteristic_suppressed: Array
    finite: Array
    physically_valid: Array
    boundary_id: str = eqx.field(static=True)


class GRMHDBoundaryCondition(StrictModule, NonTrainableState):
    """Metric-aware primitive exterior state with authoritative normal magnetic flux."""

    kind: GRMHDBoundaryKind = eqx.field(static=True)
    prescribed_primitive: Array
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: GRMHDBoundaryKind,
        /,
        *,
        prescribed_primitive: ArrayLike | None = None,
    ) -> None:
        if kind not in (
            "outflow",
            "horizon_outflow",
            "reflective",
            "conducting",
            "prescribed",
        ):
            raise ValueError("Unknown GRMHD boundary kind.")
        if kind == "prescribed":
            if prescribed_primitive is None:
                raise ValueError("A prescribed GRMHD boundary requires primitives.")
            primitive = np.asarray(prescribed_primitive, dtype=float)
            if primitive.shape != (8,) or np.any(~np.isfinite(primitive)):
                raise ValueError("Prescribed GRMHD primitive must be one finite state.")
        else:
            if prescribed_primitive is not None:
                raise ValueError("Only a prescribed GRMHD boundary accepts primitives.")
            primitive = np.zeros((8,), dtype=float)
        self.kind = kind
        self.prescribed_primitive = jnp.asarray(primitive)
        self.boundary_id = canonical_fingerprint(
            {
                "kind": f"grmhd-boundary:{kind}",
                "primitive": (
                    None if kind != "prescribed" else array_tree_fingerprint(primitive)
                ),
            }
        )

    def trace(
        self,
        interior_primitive: ArrayLike,
        normal_magnetic: ArrayLike,
        geometry: ADMGridGeometry,
        axis: int,
        side: GRMHDBoundarySide,
        /,
    ) -> GRMHDBoundaryTrace:
        primitive = jnp.asarray(interior_primitive)
        normal = jnp.asarray(normal_magnetic, dtype=primitive.dtype)
        axis_ = int(axis)
        if primitive.shape[:-1] != geometry.leading_shape or primitive.shape[-1] != 8:
            raise ValueError("GRMHD boundary primitive must match face geometry.")
        if normal.shape != primitive.shape[:-1]:
            raise ValueError("GRMHD boundary normal magnetic field has invalid shape.")
        if self.kind == "prescribed":
            exterior = jnp.broadcast_to(
                self.prescribed_primitive.astype(primitive.dtype), primitive.shape
            )
            suppressed = jnp.zeros(primitive.shape[:-1], dtype=bool)
        else:
            exterior = primitive
            transport = (
                geometry.alpha * primitive[..., 1 + axis_]
                - geometry.beta_contravariant[..., axis_]
            )
            outward = transport if side == "upper" else -transport
            incoming = outward < 0.0
            if self.kind in ("outflow", "horizon_outflow"):
                corrected_transport = jnp.where(incoming, 0.0, transport)
                corrected_velocity = (
                    corrected_transport + geometry.beta_contravariant[..., axis_]
                ) / geometry.alpha
                exterior = exterior.at[..., 1 + axis_].set(corrected_velocity)
                suppressed = incoming
            else:
                exterior = exterior.at[..., 1 + axis_].multiply(-1.0)
                suppressed = jnp.zeros_like(incoming)
        exterior = exterior.at[..., 5 + axis_].set(normal)
        velocity = exterior[..., 1:4]
        magnetic = exterior[..., 5:8]
        electric = -jnp.cross(velocity, magnetic)
        if self.kind == "conducting":
            for tangential in range(3):
                if tangential != axis_:
                    electric = electric.at[..., tangential].set(0.0)
        metric_velocity = ein.contract(
            "...ij,...j->...i", geometry.spatial_metric, velocity
        )
        speed_squared = ein.contract("...i,...i->...", metric_velocity, velocity)
        finite = jnp.all(jnp.isfinite(exterior), axis=-1) & jnp.all(
            jnp.isfinite(electric), axis=-1
        )
        physically_valid = (
            finite
            & geometry.physically_valid
            & (exterior[..., 0] > 0.0)
            & (exterior[..., 4] > 0.0)
            & (speed_squared < 1.0)
        )
        return GRMHDBoundaryTrace(
            exterior,
            electric,
            suppressed,
            finite,
            physically_valid,
            self.boundary_id,
        )


class GRMHDBoundaryPair(StrictModule, NonTrainableState):
    lower: GRMHDBoundaryCondition
    upper: GRMHDBoundaryCondition
    pair_id: str = eqx.field(static=True)

    def __init__(
        self,
        lower: GRMHDBoundaryCondition,
        upper: GRMHDBoundaryCondition,
        /,
    ) -> None:
        if not isinstance(lower, GRMHDBoundaryCondition) or not isinstance(
            upper, GRMHDBoundaryCondition
        ):
            raise TypeError("GRMHD boundary pair values are invalid.")
        self.lower = lower
        self.upper = upper
        self.pair_id = canonical_fingerprint(
            {
                "kind": "grmhd-boundary-pair",
                "lower": lower.boundary_id,
                "upper": upper.boundary_id,
            }
        )


__all__ = [
    "GRMHDBoundaryCondition",
    "GRMHDBoundaryKind",
    "GRMHDBoundaryPair",
    "GRMHDBoundarySide",
    "GRMHDBoundaryTrace",
]
