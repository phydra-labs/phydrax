#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from enum import IntFlag
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._admissibility import AdmissibilityHeader, AdmissibilityReason
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._contracts import CompiledGeometry


class FiniteRadiusErosionReason(IntFlag):
    INEXACT_PHYSICAL_GEOMETRY = 1 << 8
    NONUNIQUE_CLOSEST_POINT = 1 << 9
    IRREGULAR_CLOSEST_POINT = 1 << 10
    CENTER_OUTSIDE_ERODED_DOMAIN = 1 << 11


class FiniteRadiusErosionEvaluation(StrictModule):
    clearance: Array
    closest_point: Array
    inward_normal: Array
    source_entity_id: Array
    feature_margin: Array
    header: AdmissibilityHeader
    plan_id: str = eqx.field(static=True)


class AbstractFiniteRadiusWallPlan(StrictModule):
    """JAX-native exact center-clearance and inward-normal contract."""

    plan_id: str = eqx.field(static=True)

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self, points: ArrayLike, radii: ArrayLike, /
    ) -> FiniteRadiusErosionEvaluation:
        raise NotImplementedError


class FiniteRadiusErosionPlan(AbstractFiniteRadiusWallPlan, NonTrainableState):
    """Exact center-domain erosion from a physical closest-point contract."""

    geometry: CompiledGeometry
    geometry_id: str = eqx.field(static=True)
    fluid_side: Literal["inside", "outside"] = eqx.field(static=True)
    require_exact_physical: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: CompiledGeometry,
        fluid_side: Literal["inside", "outside"],
        /,
        *,
        geometry_id: str,
        require_exact_physical: bool = True,
    ) -> None:
        if not isinstance(geometry, CompiledGeometry):
            raise TypeError("Finite-radius erosion requires CompiledGeometry.")
        identity = str(geometry_id)
        if fluid_side not in ("inside", "outside") or not identity:
            raise ValueError("Finite-radius fluid side and geometry_id are required.")
        geometry.require_valid()
        self.geometry = geometry
        self.geometry_id = identity
        self.fluid_side = fluid_side
        self.require_exact_physical = bool(require_exact_physical)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-radius-eroded-domain",
                "geometry": identity,
                "fluid_side": fluid_side,
                "require_exact_physical": bool(require_exact_physical),
            }
        )

    @property
    def dimension(self) -> int:
        return self.geometry.ambient_dimension

    def evaluate(
        self, points: ArrayLike, radii: ArrayLike, /
    ) -> FiniteRadiusErosionEvaluation:
        point = jnp.asarray(points)
        radius = jnp.asarray(radii, dtype=point.dtype)
        if point.ndim < 2 or point.shape[-1] != self.dimension:
            raise ValueError("Finite-radius points must end in the geometry dimension.")
        if radius.shape != point.shape[:-1]:
            raise ValueError("Finite-radius values must match the point leading shape.")
        query = self.geometry.closest_point(point)
        distance_into_fluid = (
            -query.normal_coordinate
            if self.fluid_side == "inside"
            else query.normal_coordinate
        )
        inward_normal = (
            -query.oriented_normal
            if self.fluid_side == "inside"
            else query.oriented_normal
        )
        clearance = distance_into_fluid - radius
        exact = query.exact_to_physical or not self.require_exact_physical
        finite = (
            jnp.all(jnp.isfinite(point), axis=-1)
            & jnp.isfinite(radius)
            & jnp.isfinite(clearance)
            & jnp.all(jnp.isfinite(inward_normal), axis=-1)
        )
        supported = (
            finite
            & (radius >= 0.0)
            & query.normal_coordinate_valid
            & query.unique
            & query.regular
            & exact
            & (clearance >= 0.0)
        )
        reasons = jnp.zeros(radius.shape, dtype=jnp.uint32)
        reasons = jnp.where(
            finite,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), jnp.uint32),
        )
        reasons = jnp.where(
            exact,
            reasons,
            reasons
            | jnp.asarray(
                int(FiniteRadiusErosionReason.INEXACT_PHYSICAL_GEOMETRY), jnp.uint32
            ),
        )
        reasons = jnp.where(
            query.unique,
            reasons,
            reasons
            | jnp.asarray(
                int(FiniteRadiusErosionReason.NONUNIQUE_CLOSEST_POINT), jnp.uint32
            ),
        )
        reasons = jnp.where(
            query.regular & query.normal_coordinate_valid,
            reasons,
            reasons
            | jnp.asarray(
                int(FiniteRadiusErosionReason.IRREGULAR_CLOSEST_POINT), jnp.uint32
            ),
        )
        reasons = jnp.where(
            (radius >= 0.0) & (clearance >= 0.0),
            reasons,
            reasons
            | jnp.asarray(
                int(FiniteRadiusErosionReason.CENTER_OUTSIDE_ERODED_DOMAIN), jnp.uint32
            ),
        )
        margin = jnp.minimum(clearance, query.margin - radius)
        header = AdmissibilityHeader(
            jnp.where(supported, margin, jnp.minimum(margin, -1.0)),
            reasons,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "finite-radius-erosion-evidence", "plan": self.plan_id}
            ),
        )
        return FiniteRadiusErosionEvaluation(
            clearance,
            query.closest_point,
            inward_normal,
            query.source_entity_id,
            query.margin,
            header,
            self.plan_id,
        )


__all__ = [
    "AbstractFiniteRadiusWallPlan",
    "FiniteRadiusErosionEvaluation",
    "FiniteRadiusErosionPlan",
    "FiniteRadiusErosionReason",
]
