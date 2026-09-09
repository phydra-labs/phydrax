#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._conservation_boundary import (
    AbstractConservationBoundary,
    ALEBoundaryContext,
)
from ...discretization.finite_volume._physical_boundaries import (
    NoSlipAdiabaticWallBoundary,
    NoSlipIsothermalWallBoundary,
)
from ...equations._spalart_allmaras import (
    SpalartAllmarasArguments,
    SpalartAllmarasCompressibleSystem,
)


class PreparedWallDistanceField(StrictModule, NonTrainableState):
    distance: Array
    geometry_id: str = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def __init__(self, distance: ArrayLike, /, *, geometry_id: str):
        value = jnp.asarray(distance)
        geometry = str(geometry_id)
        if (
            value.ndim == 0
            or not geometry
            or np.any(~np.isfinite(np.asarray(value)))
            or np.any(np.asarray(value) <= 0.0)
        ):
            raise ValueError("Wall-distance field must be finite and positive.")
        self.distance = value
        self.geometry_id = geometry
        self.field_id = canonical_fingerprint(
            {
                "kind": "prepared-wall-distance",
                "geometry": geometry,
                "shape": tuple(value.shape),
                "minimum": float(np.min(np.asarray(value))),
                "maximum": float(np.max(np.asarray(value))),
            }
        )


class FlatWallDistancePlan(StrictModule, NonTrainableState):
    axis: int = eqx.field(static=True)
    coordinate: float = eqx.field(static=True)
    side: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, axis: int, coordinate: float, side: str, /):
        axis_ = int(axis)
        coordinate_ = float(coordinate)
        if axis_ < 0 or not np.isfinite(coordinate_) or side not in ("lower", "upper"):
            raise ValueError("Flat wall axis, coordinate, or side is invalid.")
        self.axis = axis_
        self.coordinate = coordinate_
        self.side = side
        self.plan_id = canonical_fingerprint(
            {
                "kind": "flat-wall-distance",
                "axis": axis_,
                "coordinate": coordinate_,
                "side": side,
            }
        )

    def prepare(
        self, cell_centers: ArrayLike, /, *, geometry_id: str
    ) -> PreparedWallDistanceField:
        centers = jnp.asarray(cell_centers)
        if centers.ndim < 2 or self.axis >= centers.shape[-1]:
            raise ValueError("Cell centers do not contain the wall-distance axis.")
        signed = (
            centers[..., self.axis] - self.coordinate
            if self.side == "lower"
            else self.coordinate - centers[..., self.axis]
        )
        return PreparedWallDistanceField(signed, geometry_id=geometry_id)


class SpalartAllmarasFreestreamPlan(StrictModule, NonTrainableState):
    working_to_molecular_ratio: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, working_to_molecular_ratio: float = 3.0, /):
        ratio = float(working_to_molecular_ratio)
        if not np.isfinite(ratio) or ratio <= 0.0:
            raise ValueError("SA-neg freestream working-variable ratio must be positive.")
        self.working_to_molecular_ratio = ratio
        self.plan_id = canonical_fingerprint(
            {"kind": "sa-negative-freestream", "ratio": ratio}
        )

    def primitive(
        self,
        system: SpalartAllmarasCompressibleSystem,
        base_primitive: ArrayLike,
        transport_args: Any = None,
        /,
    ) -> Array:
        if not isinstance(system, SpalartAllmarasCompressibleSystem):
            raise TypeError("SA-neg freestream requires its complete system.")
        base = jnp.asarray(base_primitive)
        gas = system.base.primitive_to_conserved(base)
        density = system.base.density(gas)
        viscosity = system.base.transport_properties(
            gas, transport_args
        ).dynamic_viscosity
        working = self.working_to_molecular_ratio * viscosity / density
        return jnp.concatenate((base, working[..., None]), axis=-1)


class SpalartAllmarasWallBoundary(AbstractConservationBoundary):
    """Gas wall plus odd SA working variable, yielding zero face value."""

    gas_boundary: NoSlipAdiabaticWallBoundary | NoSlipIsothermalWallBoundary

    def __init__(
        self,
        gas_boundary: NoSlipAdiabaticWallBoundary | NoSlipIsothermalWallBoundary,
        /,
    ):
        if not isinstance(
            gas_boundary,
            (NoSlipAdiabaticWallBoundary, NoSlipIsothermalWallBoundary),
        ):
            raise TypeError("SA-neg wall requires an adiabatic or isothermal gas wall.")
        self.gas_boundary = gas_boundary
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "sa-negative-wall",
                "gas_boundary": gas_boundary.boundary_id,
                "working_face_value": 0.0,
            }
        )

    def exterior_state(
        self,
        system: Any,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        if not isinstance(system, SpalartAllmarasCompressibleSystem):
            raise TypeError("SA-neg wall requires SpalartAllmarasCompressibleSystem.")
        gas_exterior = self.gas_boundary.exterior_state(
            system.base,
            time,
            system.gas_state(interior),
            coordinates,
            outward_normal,
            axis,
            args,
        )
        gas_primitive = system.base.conserved_to_primitive(gas_exterior)
        working = -system.working_variable(interior)
        return system.primitive_to_conserved(
            jnp.concatenate((gas_primitive, working[..., None]), axis=-1)
        )

    def ale_exterior_state(
        self,
        system: Any,
        interior: Array,
        context: ALEBoundaryContext,
        axis: int,
        /,
    ) -> Array:
        del system, interior, context, axis
        raise ValueError("ALE SA-neg wall semantics are unsupported.")


class SpalartAllmarasManufacturedEvidence(StrictModule):
    state: Array
    conserved_gradient: Array
    inviscid_divergence: Array
    diffusive_divergence: Array
    local_source: Array
    exact_rate: Array
    finite: Array
    successful: Array
    case_id: str = eqx.field(static=True)


class SpalartAllmarasManufacturedPlan(StrictModule, NonTrainableState):
    """Automatic strong-form SA-neg manufactured source on physical points."""

    system: SpalartAllmarasCompressibleSystem
    exact_primitive: Callable[[Array, Any], ArrayLike] = eqx.field(static=True)
    wall_distance: Callable[[Array, Any], ArrayLike] = eqx.field(static=True)
    case_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: SpalartAllmarasCompressibleSystem,
        exact_primitive: Callable[[Array, Any], ArrayLike],
        wall_distance: Callable[[Array, Any], ArrayLike],
        /,
        *,
        case_id: str,
    ):
        if (
            not isinstance(system, SpalartAllmarasCompressibleSystem)
            or not callable(exact_primitive)
            or not callable(wall_distance)
            or not str(case_id)
        ):
            raise ValueError("SA-neg manufactured case inputs are invalid.")
        self.system = system
        self.exact_primitive = exact_primitive
        self.wall_distance = wall_distance
        self.case_id = str(case_id)

    def evaluate(
        self,
        coordinates: ArrayLike,
        args: Any = None,
        /,
    ) -> SpalartAllmarasManufacturedEvidence:
        points = jnp.asarray(coordinates)
        if points.ndim < 1 or points.shape[-1] != self.system.dimension:
            raise ValueError("Manufactured coordinates have the wrong dimension.")

        def at_point(point):
            def state_at(location):
                return self.system.primitive_to_conserved(
                    jnp.asarray(self.exact_primitive(location, args))
                )

            state = state_at(point)
            gradient = jax.jacfwd(state_at)(point)
            sa_args = SpalartAllmarasArguments(
                jnp.asarray(self.wall_distance(point, args))
            )
            diffusion = self.system.diffusion_evaluation(state, gradient, sa_args)
            inviscid_divergence = jnp.zeros_like(state)
            for axis in range(self.system.dimension):
                jacobian = jax.jacfwd(
                    lambda location, axis_=axis: self.system.physical_flux(
                        state_at(location), axis_, args
                    )
                )(point)
                inviscid_divergence = inviscid_divergence + jacobian[..., axis]

            def diffusive_flux(location):
                local_state = state_at(location)
                local_gradient = jax.jacfwd(state_at)(location)
                local_arguments = SpalartAllmarasArguments(
                    jnp.asarray(self.wall_distance(location, args))
                )
                return self.system.diffusion_evaluation(
                    local_state, local_gradient, local_arguments
                ).flux

            diffusive_jacobian = jax.jacfwd(diffusive_flux)(point)
            diffusive_divergence = jnp.trace(diffusive_jacobian, axis1=-2, axis2=-1)
            exact_rate = -inviscid_divergence + diffusive_divergence + diffusion.source
            finite = (
                jnp.all(jnp.isfinite(state))
                & jnp.all(jnp.isfinite(gradient))
                & jnp.all(jnp.isfinite(exact_rate))
            )
            return (
                state,
                gradient,
                inviscid_divergence,
                diffusive_divergence,
                diffusion.source,
                exact_rate,
                finite,
                diffusion.successful & finite,
            )

        flat = points.reshape((-1, self.system.dimension))
        values = jax.vmap(at_point)(flat)
        cell_shape = points.shape[:-1]
        reshaped = tuple(value.reshape(cell_shape + value.shape[1:]) for value in values)
        return SpalartAllmarasManufacturedEvidence(*reshaped, self.case_id)


__all__ = [
    "FlatWallDistancePlan",
    "SpalartAllmarasManufacturedEvidence",
    "SpalartAllmarasManufacturedPlan",
    "PreparedWallDistanceField",
    "SpalartAllmarasFreestreamPlan",
    "SpalartAllmarasWallBoundary",
]
