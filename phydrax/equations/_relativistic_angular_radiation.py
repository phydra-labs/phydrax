#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import RelativityScaleContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..metrix._adm_exchange import ADMGridGeometry, StressEnergyProjection
from ..metrix._spacetime_conventions import RelativityConvention


class GRRadiationAngularClosureEvaluation(StrictModule):
    energy_density: Array
    flux_covector: Array
    flux_vector: Array
    pressure_tensor: Array

    eddington_tensor: Array
    reduced_flux: Array
    trace_residual: Array
    symmetry_residual: Array
    minimum_principal_minor: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    closure_id: str = eqx.field(static=True)


def _require_geometry(
    scale: RelativityScaleContract,
    convention: RelativityConvention,
    geometry: ADMGridGeometry,
    /,
) -> ADMGridGeometry:
    if not isinstance(geometry, ADMGridGeometry):
        raise TypeError("geometry must be ADMGridGeometry.")
    if (
        geometry.scale_id != scale.scale_id
        or geometry.convention_id != convention.convention_id
    ):
        raise ValueError("Angular radiation and geometry contracts differ.")
    return geometry


def _projection(
    evaluation: GRRadiationAngularClosureEvaluation,
    geometry: ADMGridGeometry,
    light_speed: float,
    owner_id: str,
    /,
) -> StressEnergyProjection:
    if not isinstance(evaluation, GRRadiationAngularClosureEvaluation):
        raise TypeError("evaluation must be GRRadiationAngularClosureEvaluation.")
    if evaluation.closure_id != owner_id:
        raise ValueError("Angular closure evaluation belongs to another plan.")
    metric = geometry.spatial_metric.astype(evaluation.energy_density.dtype)
    stress_covariant = contract(
        "...ik,...jl,...kl->...ij",
        metric,
        metric,
        evaluation.pressure_tensor,
        backend="jax",
    )
    projection_defect = jnp.max(
        jnp.abs(stress_covariant - jnp.swapaxes(stress_covariant, -1, -2)),
        axis=(-2, -1),
    )
    projection_id = canonical_fingerprint(
        {
            "kind": "gr-angular-radiation-stress-projection",
            "closure": owner_id,
            "geometry_lineage": geometry.geometry_lineage_id,
        }
    )
    return StressEnergyProjection(
        evaluation.energy_density,
        evaluation.flux_covector
        / jnp.asarray(light_speed, dtype=evaluation.energy_density.dtype),
        stress_covariant,
        geometry.active,
        evaluation.qualified,
        projection_defect,
        jnp.zeros_like(evaluation.energy_density),
        snapshot_token=geometry.snapshot_token,
        geometry_lineage_id=geometry.geometry_lineage_id,
        convention_id=geometry.convention_id,
        scale_id=geometry.scale_id,
        topology_id=geometry.topology_id,
        projection_id=projection_id,
    )


def _angular_evaluation(
    energy: Array,
    flux_vector: Array,
    pressure: Array,
    geometry: ADMGridGeometry,
    light_speed: float,
    tolerance: float,
    closure_id: str,
    /,
) -> GRRadiationAngularClosureEvaluation:
    metric = geometry.spatial_metric.astype(energy.dtype)
    flux_covector = contract("...ij,...j->...i", metric, flux_vector, backend="jax")
    safe_energy = jnp.where(energy > 0.0, energy, 1.0)
    eddington = pressure / safe_energy[..., None, None]
    trace = contract("...ij,...ij->...", metric, eddington, backend="jax")
    trace_residual = jnp.abs(trace - 1.0)
    symmetry_residual = jnp.max(
        jnp.abs(pressure - jnp.swapaxes(pressure, -1, -2)), axis=(-2, -1)
    )
    covariant_eddington = contract(
        "...ik,...jl,...kl->...ij",
        metric,
        metric,
        eddington,
        backend="jax",
    )
    diagonal = jnp.diagonal(covariant_eddington, axis1=-2, axis2=-1)
    minor_01 = (
        covariant_eddington[..., 0, 0] * covariant_eddington[..., 1, 1]
        - covariant_eddington[..., 0, 1] ** 2
    )
    minor_02 = (
        covariant_eddington[..., 0, 0] * covariant_eddington[..., 2, 2]
        - covariant_eddington[..., 0, 2] ** 2
    )
    minor_12 = (
        covariant_eddington[..., 1, 1] * covariant_eddington[..., 2, 2]
        - covariant_eddington[..., 1, 2] ** 2
    )
    determinant = (
        covariant_eddington[..., 0, 0]
        * (
            covariant_eddington[..., 1, 1] * covariant_eddington[..., 2, 2]
            - covariant_eddington[..., 1, 2] * covariant_eddington[..., 2, 1]
        )
        - covariant_eddington[..., 0, 1]
        * (
            covariant_eddington[..., 1, 0] * covariant_eddington[..., 2, 2]
            - covariant_eddington[..., 1, 2] * covariant_eddington[..., 2, 0]
        )
        + covariant_eddington[..., 0, 2]
        * (
            covariant_eddington[..., 1, 0] * covariant_eddington[..., 2, 1]
            - covariant_eddington[..., 1, 1] * covariant_eddington[..., 2, 0]
        )
    )
    minimum_minor = jnp.min(
        jnp.concatenate(
            (
                diagonal,
                minor_01[..., None],
                minor_02[..., None],
                minor_12[..., None],
                determinant[..., None],
            ),
            axis=-1,
        ),
        axis=-1,
    )
    flux_squared = contract("...i,...i->...", flux_covector, flux_vector, backend="jax")
    flux_norm = jnp.sqrt(jnp.maximum(flux_squared, 0.0))
    reduced_flux = flux_norm / (
        jnp.asarray(light_speed, dtype=energy.dtype) * safe_energy
    )
    finite = (
        geometry.finite
        & jnp.isfinite(energy)
        & jnp.all(jnp.isfinite(flux_vector), axis=-1)
        & jnp.all(jnp.isfinite(pressure), axis=(-2, -1))
        & jnp.isfinite(trace_residual)
        & jnp.isfinite(minimum_minor)
    )
    physical = (
        finite
        & geometry.physically_valid
        & (energy > 0.0)
        & (flux_squared >= 0.0)
        & (reduced_flux <= 1.0 + tolerance)
        & (minimum_minor >= -tolerance)
    )
    qualified = (
        physical & (trace_residual <= tolerance) & (symmetry_residual <= tolerance)
    )
    derivative = (
        qualified
        & (energy > tolerance)
        & (reduced_flux < 1.0 - tolerance)
        & (minimum_minor > tolerance)
    )
    return GRRadiationAngularClosureEvaluation(
        energy,
        flux_covector,
        flux_vector,
        pressure,
        eddington,
        reduced_flux,
        trace_residual,
        symmetry_residual,
        minimum_minor,
        finite,
        physical,
        qualified,
        derivative,
        closure_id,
    )


class VariableEddingtonTensorClosurePlan(StrictModule, NonTrainableState):
    scale: RelativityScaleContract
    convention: RelativityConvention
    tolerance: float = eqx.field(static=True)
    closure_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: RelativityScaleContract,
        convention: RelativityConvention,
        /,
        *,
        tolerance: float = 1.0e-9,
    ) -> None:
        if not isinstance(scale, RelativityScaleContract) or not isinstance(
            convention, RelativityConvention
        ):
            raise TypeError("VET closure requires relativity scale and convention.")
        if convention.metric_signature != "mostly_plus":
            raise ValueError("GR angular radiation requires mostly-plus signature.")
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("VET tolerance must be finite and positive.")
        self.scale = scale
        self.convention = convention
        self.tolerance = tolerance_
        self.closure_id = canonical_fingerprint(
            {
                "kind": "variable-eddington-tensor-closure",
                "scale": scale.scale_id,
                "convention": convention.convention_id,
                "tolerance": tolerance_,
            }
        )

    def evaluate(
        self,
        energy_density: ArrayLike,
        flux_covector: ArrayLike,
        eddington_tensor: ArrayLike,
        geometry: ADMGridGeometry,
        /,
    ) -> GRRadiationAngularClosureEvaluation:
        geometry = _require_geometry(self.scale, self.convention, geometry)
        energy = jnp.asarray(energy_density)
        flux = jnp.asarray(flux_covector, dtype=energy.dtype)
        eddington = jnp.asarray(eddington_tensor, dtype=energy.dtype)
        if (
            energy.shape != geometry.leading_shape
            or flux.shape != geometry.leading_shape + (3,)
            or eddington.shape != geometry.leading_shape + (3, 3)
        ):
            raise ValueError("VET fields must match ADM geometry.")
        flux_vector = contract(
            "...ij,...j->...i",
            geometry.inverse_spatial_metric,
            flux,
            backend="jax",
        )
        return _angular_evaluation(
            energy,
            flux_vector,
            energy[..., None, None] * eddington,
            geometry,
            float(self.scale.speed_of_light),
            self.tolerance,
            self.closure_id,
        )

    def coordinate_flux(
        self,
        evaluation: GRRadiationAngularClosureEvaluation,
        axis: int,
        geometry: ADMGridGeometry,
        /,
    ) -> Array:
        geometry = _require_geometry(self.scale, self.convention, geometry)
        if evaluation.closure_id != self.closure_id:
            raise ValueError("VET evaluation belongs to another closure plan.")
        axis_ = int(axis)
        if axis_ not in (0, 1, 2):
            raise ValueError("VET flux axis must be zero, one, or two.")
        lapse = geometry.alpha.astype(evaluation.energy_density.dtype)
        shift = geometry.beta_contravariant[..., axis_].astype(
            evaluation.energy_density.dtype
        )
        metric = geometry.spatial_metric.astype(evaluation.energy_density.dtype)
        pressure_mixed = contract(
            "...ij,...jk->...ik",
            metric,
            evaluation.pressure_tensor,
            backend="jax",
        )
        energy_flux = (
            lapse * evaluation.flux_vector[..., axis_] - shift * evaluation.energy_density
        )
        momentum_flux = (
            lapse[..., None]
            * float(self.scale.speed_of_light) ** 2
            * pressure_mixed[..., :, axis_]
            - shift[..., None] * evaluation.flux_covector
        )
        return jnp.concatenate((energy_flux[..., None], momentum_flux), axis=-1)

    def stress_energy_projection(
        self,
        evaluation: GRRadiationAngularClosureEvaluation,
        geometry: ADMGridGeometry,
        /,
    ) -> StressEnergyProjection:
        geometry = _require_geometry(self.scale, self.convention, geometry)
        return _projection(
            evaluation,
            geometry,
            float(self.scale.speed_of_light),
            self.closure_id,
        )


class DiscreteOrdinatesRadiationPlan(StrictModule, NonTrainableState):
    scale: RelativityScaleContract
    convention: RelativityConvention
    directions: Array
    weights: Array
    ordinate_count: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: RelativityScaleContract,
        convention: RelativityConvention,
        directions: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        tolerance: float = 1.0e-9,
    ) -> None:
        if not isinstance(scale, RelativityScaleContract) or not isinstance(
            convention, RelativityConvention
        ):
            raise TypeError("Discrete ordinates require relativity scale and convention.")
        if convention.metric_signature != "mostly_plus":
            raise ValueError("GR angular radiation requires mostly-plus signature.")
        direction_values = np.asarray(directions, dtype=np.float64)
        weight_values = np.asarray(weights, dtype=np.float64)
        tolerance_ = float(tolerance)
        if (
            direction_values.ndim != 2
            or direction_values.shape[1:] != (3,)
            or weight_values.shape != (direction_values.shape[0],)
            or direction_values.shape[0] < 2
            or np.any(~np.isfinite(direction_values))
            or np.any(np.linalg.norm(direction_values, axis=-1) <= 0.0)
            or np.any(~np.isfinite(weight_values))
            or np.any(weight_values <= 0.0)
            or not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
        ):
            raise ValueError("Discrete-ordinates quadrature is invalid.")
        normalized_weights = weight_values / np.sum(weight_values)
        self.scale = scale
        self.convention = convention
        self.directions = jnp.asarray(direction_values)
        self.weights = jnp.asarray(normalized_weights)
        self.ordinate_count = direction_values.shape[0]
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gr-discrete-ordinates-radiation",
                "scale": scale.scale_id,
                "convention": convention.convention_id,
                "directions": array_tree_fingerprint(direction_values),
                "weights": array_tree_fingerprint(normalized_weights),
                "tolerance": tolerance_,
            }
        )

    def normalized_directions(self, geometry: ADMGridGeometry, /) -> Array:
        geometry = _require_geometry(self.scale, self.convention, geometry)
        directions = jnp.broadcast_to(
            self.directions,
            geometry.leading_shape + self.directions.shape,
        )
        norm_squared = contract(
            "...ni,...ij,...nj->...n",
            directions,
            geometry.spatial_metric,
            directions,
            backend="jax",
        )
        return (
            directions
            / jnp.sqrt(jnp.maximum(norm_squared, jnp.finfo(directions.dtype).tiny))[
                ..., None
            ]
        )

    def evaluate(
        self, intensities: ArrayLike, geometry: ADMGridGeometry, /
    ) -> GRRadiationAngularClosureEvaluation:
        geometry = _require_geometry(self.scale, self.convention, geometry)
        intensity = jnp.asarray(intensities)
        if intensity.shape != geometry.leading_shape + (self.ordinate_count,):
            raise ValueError("Discrete-ordinates intensities must match ADM geometry.")
        directions = self.normalized_directions(geometry).astype(intensity.dtype)
        weights = self.weights.astype(intensity.dtype)
        weighted = intensity * weights
        energy = jnp.sum(weighted, axis=-1)
        flux_vector = float(self.scale.speed_of_light) * contract(
            "...n,...ni->...i", weighted, directions, backend="jax"
        )
        pressure = contract(
            "...n,...ni,...nj->...ij",
            weighted,
            directions,
            directions,
            backend="jax",
        )
        evaluation = _angular_evaluation(
            energy,
            flux_vector,
            pressure,
            geometry,
            float(self.scale.speed_of_light),
            self.tolerance,
            self.plan_id,
        )
        nonnegative = jnp.all(intensity >= 0.0, axis=-1)
        finite = evaluation.finite & jnp.all(jnp.isfinite(intensity), axis=-1)
        physical = evaluation.physically_valid & nonnegative
        qualified = evaluation.qualified & nonnegative
        derivative = evaluation.derivative_valid & jnp.all(intensity > 0.0, axis=-1)
        return eqx.tree_at(
            lambda value: (
                value.finite,
                value.physically_valid,
                value.qualified,
                value.derivative_valid,
            ),
            evaluation,
            (finite, physical, qualified, derivative),
        )

    def coordinate_flux(
        self, intensities: ArrayLike, axis: int, geometry: ADMGridGeometry, /
    ) -> Array:
        geometry = _require_geometry(self.scale, self.convention, geometry)
        axis_ = int(axis)
        if axis_ not in (0, 1, 2):
            raise ValueError("Discrete-ordinates flux axis must be zero, one, or two.")
        intensity = jnp.asarray(intensities)
        if intensity.shape != geometry.leading_shape + (self.ordinate_count,):
            raise ValueError("Discrete-ordinates intensities must match ADM geometry.")
        directions = self.normalized_directions(geometry).astype(intensity.dtype)
        speed = (
            geometry.alpha[..., None]
            * float(self.scale.speed_of_light)
            * directions[..., axis_]
            - geometry.beta_contravariant[..., axis_, None]
        )
        return speed * intensity

    def positivity_limited_step(
        self, intensities: ArrayLike, increment: ArrayLike, /
    ) -> tuple[Array, Array]:
        intensity = jnp.asarray(intensities)
        change = jnp.asarray(increment, dtype=intensity.dtype)
        if intensity.shape != change.shape or intensity.shape[-1:] != (
            self.ordinate_count,
        ):
            raise ValueError("Discrete-ordinates update shapes differ.")
        negative = change < 0.0
        safe_change = jnp.where(negative, -change, 1.0)
        factor = jnp.min(jnp.where(negative, intensity / safe_change, jnp.inf), axis=-1)
        factor = jnp.clip(factor, 0.0, 1.0)
        return intensity + factor[..., None] * change, factor

    def stress_energy_projection(
        self,
        evaluation: GRRadiationAngularClosureEvaluation,
        geometry: ADMGridGeometry,
        /,
    ) -> StressEnergyProjection:
        geometry = _require_geometry(self.scale, self.convention, geometry)
        return _projection(
            evaluation, geometry, float(self.scale.speed_of_light), self.plan_id
        )


class MonteCarloRadiationClosureEvaluation(StrictModule):
    angular: GRRadiationAngularClosureEvaluation
    packet_count: Array
    effective_packet_count: Array
    relative_sampling_error: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    plan_id: str = eqx.field(static=True)


class MonteCarloRadiationClosurePlan(StrictModule, NonTrainableState):
    scale: RelativityScaleContract
    convention: RelativityConvention
    minimum_effective_packets: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: RelativityScaleContract,
        convention: RelativityConvention,
        /,
        *,
        minimum_effective_packets: float = 4.0,
        tolerance: float = 1.0e-8,
    ) -> None:
        if not isinstance(scale, RelativityScaleContract) or not isinstance(
            convention, RelativityConvention
        ):
            raise TypeError(
                "Monte Carlo closure requires relativity scale and convention."
            )
        if convention.metric_signature != "mostly_plus":
            raise ValueError("GR angular radiation requires mostly-plus signature.")
        minimum = float(minimum_effective_packets)
        tolerance_ = float(tolerance)
        if (
            not np.isfinite(minimum)
            or minimum <= 0.0
            or not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
        ):
            raise ValueError("Monte Carlo closure controls are invalid.")
        self.scale = scale
        self.convention = convention
        self.minimum_effective_packets = minimum
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gr-monte-carlo-radiation-closure",
                "scale": scale.scale_id,
                "convention": convention.convention_id,
                "minimum_effective_packets": minimum,
                "tolerance": tolerance_,
            }
        )

    def evaluate(
        self,
        packet_energy_weights: ArrayLike,
        packet_directions: ArrayLike,
        cell_proper_volume: ArrayLike,
        geometry: ADMGridGeometry,
        /,
    ) -> MonteCarloRadiationClosureEvaluation:
        geometry = _require_geometry(self.scale, self.convention, geometry)
        weights = jnp.asarray(packet_energy_weights)
        directions = jnp.asarray(packet_directions, dtype=weights.dtype)
        proper_volume = jnp.asarray(cell_proper_volume, dtype=weights.dtype)
        if (
            weights.ndim != len(geometry.leading_shape) + 1
            or weights.shape[:-1] != geometry.leading_shape
            or directions.shape != weights.shape + (3,)
            or proper_volume.shape != geometry.leading_shape
        ):
            raise ValueError("Monte Carlo packet fields must match ADM geometry.")
        norm_squared = contract(
            "...ni,...ij,...nj->...n",
            directions,
            geometry.spatial_metric,
            directions,
            backend="jax",
        )
        normalized = (
            directions
            / jnp.sqrt(jnp.maximum(norm_squared, jnp.finfo(weights.dtype).tiny))[
                ..., None
            ]
        )
        safe_volume = jnp.where(proper_volume > 0.0, proper_volume, 1.0)
        energy = jnp.sum(weights, axis=-1) / safe_volume
        flux_vector = (
            float(self.scale.speed_of_light)
            * contract("...n,...ni->...i", weights, normalized, backend="jax")
            / safe_volume[..., None]
        )
        pressure = (
            contract(
                "...n,...ni,...nj->...ij",
                weights,
                normalized,
                normalized,
                backend="jax",
            )
            / safe_volume[..., None, None]
        )
        angular = _angular_evaluation(
            energy,
            flux_vector,
            pressure,
            geometry,
            float(self.scale.speed_of_light),
            self.tolerance,
            self.plan_id,
        )
        weight_sum = jnp.sum(weights, axis=-1)
        weight_squared_sum = jnp.sum(weights**2, axis=-1)
        effective = weight_sum**2 / jnp.where(
            weight_squared_sum > 0.0, weight_squared_sum, 1.0
        )
        packet_count = jnp.sum(weights > 0.0, axis=-1)
        relative_error = 1.0 / jnp.sqrt(jnp.maximum(effective, 1.0))
        finite = (
            angular.finite
            & jnp.all(jnp.isfinite(weights), axis=-1)
            & jnp.all(jnp.isfinite(directions), axis=(-2, -1))
            & jnp.isfinite(proper_volume)
        )
        physical = (
            angular.physically_valid
            & jnp.all(weights >= 0.0, axis=-1)
            & (proper_volume > 0.0)
            & (packet_count > 0)
        )
        qualified = (
            angular.qualified & physical & (effective >= self.minimum_effective_packets)
        )
        derivative = (
            qualified & angular.derivative_valid & jnp.all(weights > 0.0, axis=-1)
        )
        return MonteCarloRadiationClosureEvaluation(
            angular,
            packet_count,
            effective,
            relative_error,
            finite,
            physical,
            qualified,
            derivative,
            self.plan_id,
        )

    def stress_energy_projection(
        self,
        evaluation: MonteCarloRadiationClosureEvaluation,
        geometry: ADMGridGeometry,
        /,
    ) -> StressEnergyProjection:
        geometry = _require_geometry(self.scale, self.convention, geometry)
        if not isinstance(evaluation, MonteCarloRadiationClosureEvaluation):
            raise TypeError("evaluation must be MonteCarloRadiationClosureEvaluation.")
        if evaluation.plan_id != self.plan_id:
            raise ValueError("Monte Carlo evaluation belongs to another closure plan.")
        return _projection(
            evaluation.angular,
            geometry,
            float(self.scale.speed_of_light),
            self.plan_id,
        )


__all__ = [
    "DiscreteOrdinatesRadiationPlan",
    "GRRadiationAngularClosureEvaluation",
    "MonteCarloRadiationClosureEvaluation",
    "MonteCarloRadiationClosurePlan",
    "VariableEddingtonTensorClosurePlan",
]
