#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._physical import RelativityScaleContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..metrix._adm_exchange import ADMGridGeometry
from ..metrix._spacetime_conventions import RelativityConvention
from ._hyperbolic_systems import AbstractAdmissibleSystem


_LEVI_CIVITA = (
    ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
    ((0.0, 0.0, -1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
)


def _metric_inner(left: Array, metric: Array, right: Array, /) -> Array:
    return contract("...i,...ij,...j->...", left, metric, right, backend="jax")


def _lower(vector: Array, metric: Array, /) -> Array:
    return contract("...ij,...j->...i", metric, vector, backend="jax")


class ForceFreeConstraintEvaluation(StrictModule):
    electric_squared: Array
    magnetic_squared: Array
    degeneracy_residual: Array
    magnetic_dominance: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    system_id: str = eqx.field(static=True)


class ForceFreeCurrentEvaluation(StrictModule):
    current: Array
    charge_density: Array
    drift_current: Array
    parallel_current: Array
    electric_curl: Array
    magnetic_curl: Array
    constraints: ForceFreeConstraintEvaluation
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    system_id: str = eqx.field(static=True)


class ForceFreeProjectionResult(StrictModule):
    electric_field: Array
    parallel_correction: Array
    dominance_scale: Array
    constraints: ForceFreeConstraintEvaluation
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    system_id: str = eqx.field(static=True)


class GRForceFreeSystem(AbstractAdmissibleSystem, NonTrainableState):
    """Eulerian 3+1 force-free Maxwell system with GLM constraint cleaning.

    The eight conserved components are contravariant ``E``, contravariant ``B``
    and electric/magnetic GLM scalars.  ``coordinate_flux`` applies lapse and
    shift to the local Eulerian flux.  The current closure consumes covariant
    spatial derivatives, so connection and curvilinear-coordinate treatment stay
    with the metric/discretization owner rather than being silently omitted.
    """

    scale: RelativityScaleContract
    convention: RelativityConvention
    electric_cleaning_rate: float = eqx.field(static=True)
    magnetic_cleaning_rate: float = eqx.field(static=True)
    degeneracy_tolerance: float = eqx.field(static=True)
    dominance_margin: float = eqx.field(static=True)

    def __init__(
        self,
        scale: RelativityScaleContract,
        convention: RelativityConvention,
        /,
        *,
        electric_cleaning_rate: float = 0.0,
        magnetic_cleaning_rate: float = 0.0,
        degeneracy_tolerance: float = 1.0e-10,
        dominance_margin: float = 1.0e-8,
    ) -> None:
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be RelativityScaleContract.")
        if not isinstance(convention, RelativityConvention):
            raise TypeError("convention must be RelativityConvention.")
        if convention.metric_signature != "mostly_plus":
            raise ValueError("GR force-free evolution requires mostly-plus signature.")
        light_speed = float(scale.speed_of_light)
        electric_rate = float(electric_cleaning_rate)
        magnetic_rate = float(magnetic_cleaning_rate)
        tolerance = float(degeneracy_tolerance)
        margin = float(dominance_margin)
        if (
            not np.isfinite(light_speed)
            or light_speed <= 0.0
            or not np.isfinite(electric_rate)
            or electric_rate < 0.0
            or not np.isfinite(magnetic_rate)
            or magnetic_rate < 0.0
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
            or not np.isfinite(margin)
            or margin <= 0.0
            or margin >= 1.0
        ):
            raise ValueError("GR force-free system parameters are invalid.")
        self.dimension = 3
        self.component_names = (
            "electric_x",
            "electric_y",
            "electric_z",
            "magnetic_x",
            "magnetic_y",
            "magnetic_z",
            "electric_cleaning",
            "magnetic_cleaning",
        )
        self.scale = scale
        self.convention = convention
        self.electric_cleaning_rate = electric_rate
        self.magnetic_cleaning_rate = magnetic_rate
        self.degeneracy_tolerance = tolerance
        self.dominance_margin = margin
        self.system_id = canonical_fingerprint(
            {
                "kind": "gr-force-free-glm-system",
                "scale": scale.scale_id,
                "convention": convention.convention_id,
                "electric_cleaning_rate": electric_rate,
                "magnetic_cleaning_rate": magnetic_rate,
                "degeneracy_tolerance": tolerance,
                "dominance_margin": margin,
            }
        )

    @property
    def speed_of_light(self) -> float:
        return float(self.scale.speed_of_light)

    @staticmethod
    def _state(state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape[-1:] != (8,):
            raise ValueError("Force-free state must have eight trailing components.")
        return value

    def conserved_to_primitive(self, state: Array, /) -> Array:
        return self._state(state)

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        return self._state(primitive)

    def local_physical_flux(self, state: ArrayLike, axis: int, /) -> Array:
        value = self._state(state)
        axis_ = int(axis)
        if axis_ not in (0, 1, 2):
            raise ValueError("Force-free flux axis must be zero, one, or two.")
        electric = value[..., :3]
        magnetic = value[..., 3:6]
        electric_cleaning = value[..., 6]
        magnetic_cleaning = value[..., 7]
        levi = (
            self.convention.spacetime_orientation
            * self.convention.future_time_orientation
            * jnp.asarray(_LEVI_CIVITA, dtype=value.dtype)
        )
        electric_flux = -self.speed_of_light * contract(
            "ik,...k->...i", levi[:, axis_, :], magnetic, backend="jax"
        )
        magnetic_flux = self.speed_of_light * contract(
            "ik,...k->...i", levi[:, axis_, :], electric, backend="jax"
        )
        electric_flux = electric_flux.at[..., axis_].add(
            self.speed_of_light * electric_cleaning
        )
        magnetic_flux = magnetic_flux.at[..., axis_].add(
            self.speed_of_light * magnetic_cleaning
        )
        return jnp.concatenate(
            (
                electric_flux,
                magnetic_flux,
                (self.speed_of_light * electric[..., axis_])[..., None],
                (self.speed_of_light * magnetic[..., axis_])[..., None],
            ),
            axis=-1,
        )

    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        del args
        return self.local_physical_flux(state, axis)

    def coordinate_flux(
        self,
        state: ArrayLike,
        axis: int,
        geometry: ADMGridGeometry,
        /,
    ) -> Array:
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be ADMGridGeometry.")
        if (
            geometry.scale_id != self.scale.scale_id
            or geometry.convention_id != self.convention.convention_id
        ):
            raise ValueError("ADM geometry and force-free system contracts differ.")
        value = self._state(state)
        if value.shape[:-1] != geometry.leading_shape:
            raise ValueError("Force-free state must match ADM geometry.")
        axis_ = int(axis)
        if axis_ not in (0, 1, 2):
            raise ValueError("Force-free flux axis must be zero, one, or two.")
        lapse = geometry.alpha.astype(value.dtype)
        shift_axis = geometry.beta_contravariant[..., axis_].astype(value.dtype)
        metric = geometry.spatial_metric.astype(value.dtype)
        inverse = geometry.inverse_spatial_metric.astype(value.dtype)
        volume = geometry.sqrt_det_spatial_metric.astype(value.dtype)
        electric = value[..., :3]
        magnetic = value[..., 3:6]
        electric_covector = _lower(electric, metric)
        magnetic_covector = _lower(magnetic, metric)
        levi = (
            self.convention.spacetime_orientation
            * self.convention.future_time_orientation
            * jnp.asarray(_LEVI_CIVITA, dtype=value.dtype)
        )
        electric_flux = (
            -lapse[..., None]
            * self.speed_of_light
            * contract(
                "ik,...k->...i",
                levi[:, axis_, :],
                magnetic_covector,
                backend="jax",
            )
            / volume[..., None]
            - shift_axis[..., None] * electric
            + lapse[..., None]
            * self.speed_of_light
            * inverse[..., :, axis_]
            * value[..., 6, None]
        )
        magnetic_flux = (
            lapse[..., None]
            * self.speed_of_light
            * contract(
                "ik,...k->...i",
                levi[:, axis_, :],
                electric_covector,
                backend="jax",
            )
            / volume[..., None]
            - shift_axis[..., None] * magnetic
            + lapse[..., None]
            * self.speed_of_light
            * inverse[..., :, axis_]
            * value[..., 7, None]
        )
        electric_cleaning_flux = (
            lapse * self.speed_of_light * electric[..., axis_]
            - shift_axis * value[..., 6]
        )
        magnetic_cleaning_flux = (
            lapse * self.speed_of_light * magnetic[..., axis_]
            - shift_axis * value[..., 7]
        )
        return jnp.concatenate(
            (
                electric_flux,
                magnetic_flux,
                electric_cleaning_flux[..., None],
                magnetic_cleaning_flux[..., None],
            ),
            axis=-1,
        )

    def coordinate_characteristic_bounds(
        self,
        unit_covector: ArrayLike,
        geometry: ADMGridGeometry,
        /,
    ) -> tuple[Array, Array]:
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be ADMGridGeometry.")
        if (
            geometry.scale_id != self.scale.scale_id
            or geometry.convention_id != self.convention.convention_id
        ):
            raise ValueError("ADM geometry and force-free system contracts differ.")
        normal = jnp.asarray(unit_covector)
        lapse_ = geometry.alpha.astype(normal.dtype)
        shift_ = geometry.beta_contravariant.astype(normal.dtype)
        inverse = geometry.inverse_spatial_metric.astype(normal.dtype)
        if normal.shape != geometry.leading_shape + (3,):
            raise ValueError("Characteristic covector must match ADM geometry.")
        normal_squared = contract(
            "...i,...ij,...j->...", normal, inverse, normal, backend="jax"
        )
        normal_speed = (
            lapse_ * self.speed_of_light * jnp.sqrt(jnp.maximum(normal_squared, 0.0))
        )
        transport = -contract("...i,...i->...", shift_, normal, backend="jax")
        return transport - normal_speed, transport + normal_speed

    def max_wave_speed(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> Array:
        del right, axis, args
        return jnp.full(left.shape[:-1], self.speed_of_light, dtype=left.dtype)

    def signal_bounds(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        speed = self.max_wave_speed(left, right, axis, args)
        return -speed, speed

    def normal_signal_bounds(
        self,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        del right, normal, args
        speed = jnp.full(left.shape[:-1], self.speed_of_light, dtype=left.dtype)
        return -speed, speed

    def constraint_evaluation(
        self,
        electric_field: ArrayLike,
        magnetic_field: ArrayLike,
        geometry: ADMGridGeometry,
        /,
    ) -> ForceFreeConstraintEvaluation:
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be ADMGridGeometry.")
        if (
            geometry.scale_id != self.scale.scale_id
            or geometry.convention_id != self.convention.convention_id
        ):
            raise ValueError("ADM geometry and force-free system contracts differ.")
        electric = jnp.asarray(electric_field)
        magnetic = jnp.asarray(magnetic_field, dtype=electric.dtype)
        metric = geometry.spatial_metric.astype(electric.dtype)
        if (
            electric.shape != geometry.leading_shape + (3,)
            or magnetic.shape != electric.shape
        ):
            raise ValueError("Force-free fields must match ADM geometry.")
        electric_squared = _metric_inner(electric, metric, electric)
        magnetic_squared = _metric_inner(magnetic, metric, magnetic)
        degeneracy = _metric_inner(electric, metric, magnetic)
        dominance = magnetic_squared - electric_squared
        finite = (
            geometry.finite
            & jnp.all(jnp.isfinite(electric), axis=-1)
            & jnp.all(jnp.isfinite(magnetic), axis=-1)
            & jnp.isfinite(electric_squared)
            & jnp.isfinite(magnetic_squared)
            & jnp.isfinite(degeneracy)
        )
        scale = jnp.maximum(magnetic_squared, jnp.asarray(1.0, dtype=electric.dtype))
        degeneracy_valid = jnp.abs(degeneracy) <= self.degeneracy_tolerance * scale
        dominance_valid = dominance >= self.dominance_margin * magnetic_squared
        physically_valid = (
            finite
            & geometry.physically_valid
            & (electric_squared >= 0.0)
            & (magnetic_squared > 0.0)
            & degeneracy_valid
            & dominance_valid
        )
        qualified = physically_valid
        derivative_valid = (
            qualified
            & (jnp.abs(degeneracy) < self.degeneracy_tolerance * scale)
            & (dominance > self.dominance_margin * magnetic_squared)
        )
        return ForceFreeConstraintEvaluation(
            electric_squared,
            magnetic_squared,
            degeneracy,
            dominance,
            finite,
            physically_valid,
            qualified,
            derivative_valid,
            self.system_id,
        )

    def constraint_current(
        self,
        electric_field: ArrayLike,
        magnetic_field: ArrayLike,
        electric_covariant_gradient: ArrayLike,
        magnetic_covariant_gradient: ArrayLike,
        geometry: ADMGridGeometry,
        /,
    ) -> ForceFreeCurrentEvaluation:
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be ADMGridGeometry.")
        if (
            geometry.scale_id != self.scale.scale_id
            or geometry.convention_id != self.convention.convention_id
        ):
            raise ValueError("ADM geometry and force-free system contracts differ.")
        electric = jnp.asarray(electric_field)
        magnetic = jnp.asarray(magnetic_field, dtype=electric.dtype)
        electric_gradient = jnp.asarray(electric_covariant_gradient, dtype=electric.dtype)
        magnetic_gradient = jnp.asarray(magnetic_covariant_gradient, dtype=electric.dtype)
        metric = geometry.spatial_metric.astype(electric.dtype)
        inverse = geometry.inverse_spatial_metric.astype(electric.dtype)
        volume = geometry.sqrt_det_spatial_metric.astype(electric.dtype)
        cell_shape = geometry.leading_shape
        tensor_shape = cell_shape + (3, 3)
        if (
            electric.shape != cell_shape + (3,)
            or magnetic.shape != cell_shape + (3,)
            or electric_gradient.shape != tensor_shape
            or magnetic_gradient.shape != tensor_shape
        ):
            raise ValueError("Force-free current fields must match ADM geometry.")
        constraints = self.constraint_evaluation(electric, magnetic, geometry)
        levi = (
            self.convention.spacetime_orientation
            * self.convention.future_time_orientation
            * jnp.asarray(_LEVI_CIVITA, dtype=electric.dtype)
        )
        safe_volume = jnp.where(volume > 0.0, volume, 1.0)
        electric_curl = (
            contract("ijk,...jk->...i", levi, electric_gradient, backend="jax")
            / safe_volume[..., None]
        )
        magnetic_curl = (
            contract("ijk,...jk->...i", levi, magnetic_gradient, backend="jax")
            / safe_volume[..., None]
        )
        charge = contract("...ij,...ij->...", inverse, electric_gradient, backend="jax")
        electric_covector = _lower(electric, metric)
        magnetic_covector = _lower(magnetic, metric)
        electric_cross_magnetic = (
            contract(
                "ijk,...j,...k->...i",
                levi,
                electric_covector,
                magnetic_covector,
                backend="jax",
            )
            / safe_volume[..., None]
        )
        safe_magnetic_squared = jnp.where(
            constraints.magnetic_squared > 0.0, constraints.magnetic_squared, 1.0
        )
        drift = (
            self.speed_of_light
            * charge[..., None]
            * electric_cross_magnetic
            / safe_magnetic_squared[..., None]
        )
        parallel_coefficient = (
            self.speed_of_light
            * (
                _metric_inner(magnetic, metric, magnetic_curl)
                - _metric_inner(electric, metric, electric_curl)
            )
            / safe_magnetic_squared
        )
        parallel = parallel_coefficient[..., None] * magnetic
        current = drift + parallel
        finite = (
            geometry.finite
            & constraints.finite
            & jnp.all(jnp.isfinite(electric_gradient), axis=(-2, -1))
            & jnp.all(jnp.isfinite(magnetic_gradient), axis=(-2, -1))
            & jnp.isfinite(charge)
            & jnp.all(jnp.isfinite(current), axis=-1)
        )
        physically_valid = (
            finite & geometry.physically_valid & constraints.physically_valid
        )
        qualified = physically_valid & (
            geometry.inverse_defect <= self.degeneracy_tolerance
        )
        derivative_valid = qualified & constraints.derivative_valid
        return ForceFreeCurrentEvaluation(
            current,
            charge,
            drift,
            parallel,
            electric_curl,
            magnetic_curl,
            constraints,
            finite,
            physically_valid,
            qualified,
            derivative_valid,
            self.system_id,
        )

    def coordinate_source(
        self,
        state: ArrayLike,
        current: ForceFreeCurrentEvaluation,
        geometry: ADMGridGeometry,
        /,
    ) -> Array:
        if not isinstance(current, ForceFreeCurrentEvaluation):
            raise TypeError("current must be ForceFreeCurrentEvaluation.")
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be ADMGridGeometry.")
        if (
            current.system_id != self.system_id
            or geometry.scale_id != self.scale.scale_id
            or geometry.convention_id != self.convention.convention_id
        ):
            raise ValueError("Force-free source contracts do not match.")
        value = self._state(state)
        if value.shape[:-1] != geometry.leading_shape:
            raise ValueError("Force-free state must match ADM geometry.")
        lapse = geometry.alpha.astype(value.dtype)
        source = jnp.zeros_like(value)
        source = source.at[..., :3].set(-lapse[..., None] * current.current)
        source = source.at[..., 6].set(
            lapse
            * (
                self.speed_of_light * current.charge_density
                - self.electric_cleaning_rate * value[..., 6]
            )
        )
        return source.at[..., 7].set(-lapse * self.magnetic_cleaning_rate * value[..., 7])

    def project_constraints(
        self,
        electric_field: ArrayLike,
        magnetic_field: ArrayLike,
        geometry: ADMGridGeometry,
        /,
    ) -> ForceFreeProjectionResult:
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be ADMGridGeometry.")
        electric = jnp.asarray(electric_field)
        magnetic = jnp.asarray(magnetic_field, dtype=electric.dtype)
        metric = geometry.spatial_metric.astype(electric.dtype)
        before = self.constraint_evaluation(electric, magnetic, geometry)
        safe_magnetic_squared = jnp.where(
            before.magnetic_squared > 0.0, before.magnetic_squared, 1.0
        )
        parallel_correction = (before.degeneracy_residual / safe_magnetic_squared)[
            ..., None
        ] * magnetic
        perpendicular = electric - parallel_correction
        perpendicular_squared = jnp.maximum(
            _metric_inner(perpendicular, metric, perpendicular), 0.0
        )
        maximum_squared = (1.0 - self.dominance_margin) ** 2 * before.magnetic_squared
        dominance_scale = jnp.minimum(
            1.0,
            jnp.sqrt(
                jnp.maximum(maximum_squared, 0.0)
                / jnp.maximum(perpendicular_squared, jnp.finfo(electric.dtype).tiny)
            ),
        )
        projected = perpendicular * dominance_scale[..., None]
        constraints = self.constraint_evaluation(projected, magnetic, geometry)
        finite = (
            before.finite & constraints.finite & jnp.all(jnp.isfinite(projected), axis=-1)
        )
        physically_valid = finite & constraints.physically_valid
        qualified = physically_valid
        projection_active = (jnp.abs(before.degeneracy_residual) > 0.0) | (
            dominance_scale < 1.0
        )
        derivative_valid = qualified & ~projection_active
        return ForceFreeProjectionResult(
            projected,
            parallel_correction,
            dominance_scale,
            constraints,
            finite,
            physically_valid,
            qualified,
            derivative_valid,
            self.system_id,
        )

    def damping_source(self, state: ArrayLike, /) -> Array:
        value = self._state(state)
        source = jnp.zeros_like(value)
        source = source.at[..., 6].set(-self.electric_cleaning_rate * value[..., 6])
        return source.at[..., 7].set(-self.magnetic_cleaning_rate * value[..., 7])

    def admissible(self, state: Array, /) -> Array:
        value = self._state(state)
        electric = value[..., :3]
        magnetic = value[..., 3:6]
        electric_squared = contract("...i,...i->...", electric, electric, backend="jax")
        magnetic_squared = contract("...i,...i->...", magnetic, magnetic, backend="jax")
        degeneracy = contract("...i,...i->...", electric, magnetic, backend="jax")
        scale = jnp.maximum(magnetic_squared, 1.0)
        return (
            jnp.all(jnp.isfinite(value), axis=-1)
            & (magnetic_squared > 0.0)
            & (jnp.abs(degeneracy) <= self.degeneracy_tolerance * scale)
            & (
                magnetic_squared - electric_squared
                >= self.dominance_margin * magnetic_squared
            )
        )

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        value = self._state(state)
        axis_ = int(axis)
        if axis_ not in (0, 1, 2):
            raise ValueError("Force-free reflection axis must be zero, one, or two.")
        reflected = value.at[..., axis_].multiply(-1.0)
        for tangential in range(3):
            if tangential != axis_:
                reflected = reflected.at[..., 3 + tangential].multiply(-1.0)
        return reflected
