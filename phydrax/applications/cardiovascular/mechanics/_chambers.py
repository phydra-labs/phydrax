#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ...solid_mechanics import ClosedSurfacePressure
from ..anatomy._surfaces import OrientedChamberSurface


class ChamberVolumeResponse(StrictModule):
    """Mechanics view of anatomy-owned signed current-volume evidence."""

    volume: Array
    volume_gradient: Array
    oriented_area_vectors: Array
    finite: Array
    positive: Array
    valid: Array
    surface_id: str = eqx.field(static=True)
    orientation_id: str = eqx.field(static=True)
    coordinate_configuration: str = eqx.field(static=True)


class ChamberVolumePlan(StrictModule, NonTrainableState):
    """Fixed-topology mechanics adapter for an anatomy-owned chamber surface.

    ``OrientedChamberSurface`` remains the single owner of topology, reference
    coordinates, stable surface identity, closure certification, orientation,
    volume, and its coordinate derivative. This plan only supplies mechanics
    load semantics and an optional stricter positive-volume threshold.
    """

    surface: OrientedChamberSurface
    minimum_volume: float = eqx.field(static=True)
    orientation_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface: OrientedChamberSurface,
        /,
        *,
        minimum_volume: float | None = None,
        plan_id: str | None = None,
    ):
        if not isinstance(surface, OrientedChamberSurface):
            raise TypeError("surface must be anatomy.OrientedChamberSurface.")
        threshold = (
            surface.geometric_tolerance
            if minimum_volume is None
            else float(minimum_volume)
        )
        if not isfinite(threshold) or threshold < surface.geometric_tolerance:
            raise ValueError(
                "minimum_volume must be finite and no smaller than the anatomical "
                "geometric tolerance."
            )
        orientation_id = canonical_fingerprint(
            {
                "kind": "cardiac-chamber-outward-orientation",
                "surface_id": surface.surface_id,
                "normal": "outward-from-cavity-fluid",
            }
        )
        generated = canonical_fingerprint(
            {
                "kind": "cardiac-chamber-volume-mechanics-plan",
                "surface_id": surface.surface_id,
                "orientation_id": orientation_id,
                "minimum_volume": threshold.hex(),
            }
        )
        selected = generated if plan_id is None else str(plan_id)
        if not selected:
            raise ValueError("plan_id must be non-empty or None.")
        self.surface = surface
        self.minimum_volume = threshold
        self.orientation_id = orientation_id
        self.plan_id = selected

    @property
    def surface_id(self) -> str:
        return self.surface.surface_id

    def _coordinates(self, value: ArrayLike, /) -> Array:
        coordinates = jnp.asarray(value, dtype=self.surface.reference_coordinates.dtype)
        if coordinates.shape != self.surface.reference_coordinates.shape:
            raise ValueError(
                "Chamber coordinates must preserve the anatomy surface's fixed shape."
            )
        if jnp.issubdtype(coordinates.dtype, jnp.complexfloating):
            raise TypeError("Chamber coordinates must be real.")
        return coordinates

    def volume(self, current_coordinates: ArrayLike, /) -> Array:
        """Return anatomy's signed cavity volume in the current configuration."""
        coordinates = self._coordinates(current_coordinates)
        return self.surface.evaluate(coordinates).volume

    def volume_gradient(self, current_coordinates: ArrayLike, /) -> Array:
        """Return anatomy's exact fixed-topology nodal derivative ``dV/dx``."""
        coordinates = self._coordinates(current_coordinates)
        return self.surface.evaluate(coordinates).coordinate_derivative

    def evaluate(self, current_coordinates: ArrayLike, /) -> ChamberVolumeResponse:
        coordinates = self._coordinates(current_coordinates)
        candidate = self.surface.evaluate(coordinates)
        triangle = coordinates[self.surface.triangles]
        oriented_area = 0.5 * jnp.cross(
            triangle[:, 1] - triangle[:, 0],
            triangle[:, 2] - triangle[:, 0],
        )
        finite = candidate.evidence.finite & jnp.all(jnp.isfinite(oriented_area))
        positive = candidate.volume > self.minimum_volume
        valid = finite & positive & candidate.evidence.successful
        return ChamberVolumeResponse(
            candidate.volume,
            candidate.coordinate_derivative,
            oriented_area,
            finite,
            positive,
            valid,
            self.surface.surface_id,
            self.orientation_id,
            "current",
        )


class FollowerPressureResponse(StrictModule):
    """Conservative constant-pressure follower load and exact tangent."""

    pressure: Array
    volume: Array
    nodal_force: Array
    force_tangent: Array
    pressure_potential: Array
    finite: Array
    valid: Array
    load_id: str = eqx.field(static=True)
    surface_id: str = eqx.field(static=True)
    normal_orientation: str = eqx.field(static=True)
    force_configuration: str = eqx.field(static=True)


class FollowerPressurePlan(StrictModule, NonTrainableState):
    """Constant cavity pressure conjugate to the oriented anatomy volume.

    Anatomy certifies triangle normals outward from the cavity fluid. Positive
    pressure therefore has external virtual work ``p dV`` and conservative load
    potential ``-p V``. Nodal forces and their follower tangent are exact
    derivatives at the anatomy surface's fixed topology.
    """

    chamber: ChamberVolumePlan
    load_id: str = eqx.field(static=True)

    def __init__(
        self,
        chamber: ChamberVolumePlan,
        /,
        *,
        load_id: str | None = None,
    ):
        if not isinstance(chamber, ChamberVolumePlan):
            raise TypeError("chamber must be ChamberVolumePlan.")
        generated = canonical_fingerprint(
            {
                "kind": "cardiac-follower-pressure",
                "chamber_plan_id": chamber.plan_id,
                "sign": "positive-pressure-times-positive-volume-gradient",
            }
        )
        identifier = generated if load_id is None else str(load_id)
        if not identifier:
            raise ValueError("load_id must be non-empty or None.")
        self.chamber = chamber
        self.load_id = identifier

    @staticmethod
    def _pressure(value: ArrayLike, /) -> Array:
        pressure = jnp.asarray(value)
        if pressure.shape != () or jnp.issubdtype(pressure.dtype, jnp.complexfloating):
            raise ValueError("Cavity pressure must be one real scalar.")
        return pressure

    def nodal_force(
        self,
        current_coordinates: ArrayLike,
        pressure: ArrayLike,
        /,
    ) -> Array:
        pressure_ = self._pressure(pressure)
        return pressure_ * self.chamber.volume_gradient(current_coordinates)

    def evaluate(
        self,
        current_coordinates: ArrayLike,
        pressure: ArrayLike,
        /,
    ) -> FollowerPressureResponse:
        coordinates = self.chamber._coordinates(current_coordinates)
        pressure_ = self._pressure(pressure).astype(coordinates.dtype)
        chamber = self.chamber.evaluate(coordinates)
        force = self.nodal_force(coordinates, pressure_)
        tangent = jax.jacfwd(lambda value: self.nodal_force(value, pressure_))(
            coordinates
        )
        potential = -pressure_ * chamber.volume
        finite = (
            chamber.finite
            & jnp.isfinite(pressure_)
            & jnp.all(jnp.isfinite(force))
            & jnp.all(jnp.isfinite(tangent))
            & jnp.isfinite(potential)
        )
        return FollowerPressureResponse(
            pressure_,
            chamber.volume,
            force,
            tangent,
            potential,
            finite,
            finite & chamber.positive & chamber.valid,
            self.load_id,
            self.chamber.surface.surface_id,
            "outward-from-cavity-fluid",
            "current",
        )

    def virtual_work(
        self,
        current_coordinates: ArrayLike,
        pressure: ArrayLike,
        virtual_displacement: ArrayLike,
        /,
    ) -> Array:
        coordinates = self.chamber._coordinates(current_coordinates)
        virtual = jnp.asarray(virtual_displacement, dtype=coordinates.dtype)
        if virtual.shape != coordinates.shape:
            raise ValueError("virtual_displacement must match chamber coordinates.")
        return contract("ni,ni->", self.nodal_force(coordinates, pressure), virtual)

    def work_between(
        self,
        initial_coordinates: ArrayLike,
        final_coordinates: ArrayLike,
        pressure: ArrayLike,
        /,
    ) -> Array:
        """Return exact external work for one constant-pressure volume change."""
        pressure_ = self._pressure(pressure)
        return pressure_ * (
            self.chamber.volume(final_coordinates)
            - self.chamber.volume(initial_coordinates)
        )

    def solid_mechanics_load(
        self,
        pressure: ArrayLike,
        /,
    ) -> ClosedSurfacePressure:
        """Build the generic surface-quadrature load with identical sign semantics."""
        return ClosedSurfacePressure(
            pressure,
            closure_id=self.chamber.surface.surface_id,
            orientation_id=self.chamber.orientation_id,
            load_id=self.load_id,
        )


class MechanicsChamber(StrictModule, NonTrainableState):
    """Executor-free chamber geometry/load adapter for circulation coupling.

    This type owns no pressure-flow DAE storage. Circulation constructs its
    ``MechanicsChamberCoupling`` from ``chamber_id`` and a ``volume_rate``
    callback, preserving exclusive mechanics storage ownership.
    """

    volume_plan: ChamberVolumePlan
    pressure_plan: FollowerPressurePlan
    chamber_id: str = eqx.field(static=True)

    def __init__(
        self,
        chamber_id: str,
        volume_plan: ChamberVolumePlan,
        /,
        *,
        pressure_load_id: str | None = None,
    ):
        identifier = str(chamber_id)
        if not identifier:
            raise ValueError("chamber_id must be non-empty.")
        if not isinstance(volume_plan, ChamberVolumePlan):
            raise TypeError("volume_plan must be ChamberVolumePlan.")
        self.volume_plan = volume_plan
        self.pressure_plan = FollowerPressurePlan(
            volume_plan,
            load_id=pressure_load_id,
        )
        self.chamber_id = identifier

    def volume(self, current_coordinates: ArrayLike, /) -> Array:
        return self.volume_plan.volume(current_coordinates)

    def volume_rate(
        self,
        current_coordinates: ArrayLike,
        coordinate_velocity: ArrayLike,
        /,
    ) -> Array:
        coordinates = self.volume_plan._coordinates(current_coordinates)
        velocity = jnp.asarray(coordinate_velocity, dtype=coordinates.dtype)
        if velocity.shape != coordinates.shape:
            raise ValueError("coordinate_velocity must match chamber coordinates.")
        return contract(
            "ni,ni->",
            self.volume_plan.volume_gradient(coordinates),
            velocity,
        )

    def pressure_response(
        self,
        current_coordinates: ArrayLike,
        pressure: ArrayLike,
        /,
    ) -> FollowerPressureResponse:
        return self.pressure_plan.evaluate(current_coordinates, pressure)


__all__ = [
    "ChamberVolumePlan",
    "ChamberVolumeResponse",
    "FollowerPressurePlan",
    "FollowerPressureResponse",
    "MechanicsChamber",
]
