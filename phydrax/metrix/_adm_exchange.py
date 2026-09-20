#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


def _identity(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty stripped string.")
    return value


def _real_array(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must have a real floating-point dtype.")
    return array


def _snapshot_token(value: ArrayLike, /) -> Array:
    token = jnp.asarray(value)
    if token.shape != ():
        raise ValueError(f"snapshot_token must be scalar; got shape {token.shape}.")
    if not jnp.issubdtype(token.dtype, jnp.integer):
        raise TypeError("snapshot_token must have an integer dtype.")
    return token.astype(jnp.int32)


def _require_shape(value: Array, expected: tuple[int, ...], name: str, /) -> None:
    if value.shape != expected:
        raise ValueError(f"{name} must have shape {expected}; got {value.shape}.")


def _finite_tensor(value: Array, tensor_rank: int, /) -> Array:
    finite = jnp.isfinite(value)
    if tensor_rank == 0:
        return finite
    axes = tuple(range(value.ndim - tensor_rank, value.ndim))
    return jnp.all(finite, axis=axes)


class ADMGridGeometry(StrictModule, NonTrainableState):
    """Immutable fixed-shape geometry-to-matter ADM exchange snapshot.

    All numeric arrays have a shared leading lane shape. The vector and tensor
    fields end in ``(3,)`` and ``(3, 3)`` respectively. Static identities bind
    the snapshot to one chart, convention, scale, topology, and geometry
    lineage. ``snapshot_token`` identifies the exact dynamic stage realization.
    """

    alpha: Array
    beta_contravariant: Array
    spatial_metric: Array
    inverse_spatial_metric: Array
    sqrt_det_spatial_metric: Array
    extrinsic_curvature: Array
    active: Array
    valid: Array
    snapshot_token: Array
    chart_id: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_lineage_id: str = eqx.field(static=True)

    def __init__(
        self,
        alpha: ArrayLike,
        beta_contravariant: ArrayLike,
        spatial_metric: ArrayLike,
        inverse_spatial_metric: ArrayLike,
        sqrt_det_spatial_metric: ArrayLike,
        extrinsic_curvature: ArrayLike,
        active: ArrayLike,
        valid: ArrayLike,
        /,
        *,
        snapshot_token: ArrayLike,
        chart_id: str,
        convention_id: str,
        scale_id: str,
        topology_id: str,
        geometry_lineage_id: str,
    ):
        alpha_array = _real_array(alpha, "alpha")
        beta_array = _real_array(beta_contravariant, "beta_contravariant")
        spatial_array = _real_array(spatial_metric, "spatial_metric")
        inverse_array = _real_array(
            inverse_spatial_metric,
            "inverse_spatial_metric",
        )
        determinant_array = _real_array(
            sqrt_det_spatial_metric,
            "sqrt_det_spatial_metric",
        )
        extrinsic_array = _real_array(
            extrinsic_curvature,
            "extrinsic_curvature",
        )
        active_array = jnp.asarray(active, dtype=jnp.bool_)
        valid_array = jnp.asarray(valid, dtype=jnp.bool_)
        snapshot_token_array = _snapshot_token(snapshot_token)
        leading_shape = alpha_array.shape
        _require_shape(beta_array, leading_shape + (3,), "beta_contravariant")
        _require_shape(spatial_array, leading_shape + (3, 3), "spatial_metric")
        _require_shape(
            inverse_array,
            leading_shape + (3, 3),
            "inverse_spatial_metric",
        )
        _require_shape(
            determinant_array,
            leading_shape,
            "sqrt_det_spatial_metric",
        )
        _require_shape(
            extrinsic_array,
            leading_shape + (3, 3),
            "extrinsic_curvature",
        )
        _require_shape(active_array, leading_shape, "active")
        _require_shape(valid_array, leading_shape, "valid")
        numeric_arrays = (
            beta_array,
            spatial_array,
            inverse_array,
            determinant_array,
            extrinsic_array,
        )
        if any(value.dtype != alpha_array.dtype for value in numeric_arrays):
            raise TypeError("ADM grid numeric fields must share one dtype.")

        self.alpha = alpha_array
        self.beta_contravariant = beta_array
        self.spatial_metric = spatial_array
        self.inverse_spatial_metric = inverse_array
        self.sqrt_det_spatial_metric = determinant_array
        self.extrinsic_curvature = extrinsic_array
        self.active = active_array
        self.valid = valid_array
        self.snapshot_token = snapshot_token_array
        self.chart_id = _identity(chart_id, "chart_id")
        self.convention_id = _identity(convention_id, "convention_id")
        self.scale_id = _identity(scale_id, "scale_id")
        self.topology_id = _identity(topology_id, "topology_id")
        self.geometry_lineage_id = _identity(
            geometry_lineage_id,
            "geometry_lineage_id",
        )

    @property
    def leading_shape(self) -> tuple[int, ...]:
        return self.alpha.shape

    @property
    def finite(self) -> Array:
        return (
            _finite_tensor(self.alpha, 0)
            & _finite_tensor(self.beta_contravariant, 1)
            & _finite_tensor(self.spatial_metric, 2)
            & _finite_tensor(self.inverse_spatial_metric, 2)
            & _finite_tensor(self.sqrt_det_spatial_metric, 0)
            & _finite_tensor(self.extrinsic_curvature, 2)
        )

    @property
    def lapse_positive(self) -> Array:
        return self.alpha > 0.0

    @property
    def spatial_symmetric(self) -> Array:
        scale = jnp.maximum(
            jnp.max(jnp.abs(self.spatial_metric), axis=(-2, -1)),
            1.0,
        )
        tolerance = 128.0 * jnp.finfo(self.spatial_metric.dtype).eps * scale
        return self.spatial_symmetry_defect <= tolerance

    @property
    def spatial_symmetry_defect(self) -> Array:
        return jnp.max(
            jnp.abs(self.spatial_metric - jnp.swapaxes(self.spatial_metric, -1, -2)),
            axis=(-2, -1),
        )

    @property
    def extrinsic_symmetry_defect(self) -> Array:
        return jnp.max(
            jnp.abs(
                self.extrinsic_curvature - jnp.swapaxes(self.extrinsic_curvature, -1, -2)
            ),
            axis=(-2, -1),
        )

    @property
    def extrinsic_symmetric(self) -> Array:
        scale = jnp.maximum(
            jnp.max(jnp.abs(self.extrinsic_curvature), axis=(-2, -1)),
            1.0,
        )
        tolerance = 128.0 * jnp.finfo(self.extrinsic_curvature.dtype).eps * scale
        return self.extrinsic_symmetry_defect <= tolerance

    @property
    def minimum_spatial_eigenvalue(self) -> Array:
        symmetric = 0.5 * (
            self.spatial_metric + jnp.swapaxes(self.spatial_metric, -1, -2)
        )
        return jnp.min(jnp.linalg.eigvalsh(symmetric), axis=-1)

    @property
    def spatial_positive_definite(self) -> Array:
        return self.minimum_spatial_eigenvalue > 0.0

    @property
    def inverse_defect(self) -> Array:
        product = ein.contract(
            "...ik,...kj->...ij",
            self.spatial_metric,
            self.inverse_spatial_metric,
        )
        identity = jnp.eye(3, dtype=product.dtype)
        return jnp.max(jnp.abs(product - identity), axis=(-2, -1))

    @property
    def inverse_consistent(self) -> Array:
        tolerance = 128.0 * jnp.finfo(self.spatial_metric.dtype).eps
        return self.inverse_defect <= tolerance

    @property
    def determinant_defect(self) -> Array:
        determinant = jnp.linalg.det(self.spatial_metric)
        return jnp.abs(determinant - self.sqrt_det_spatial_metric**2)

    @property
    def determinant_consistent(self) -> Array:
        determinant = jnp.linalg.det(self.spatial_metric)
        scale = jnp.maximum(
            jnp.maximum(
                jnp.abs(determinant),
                self.sqrt_det_spatial_metric**2,
            ),
            1.0,
        )
        tolerance = 128.0 * jnp.finfo(self.spatial_metric.dtype).eps * scale
        return self.determinant_defect <= tolerance

    @property
    def physically_valid(self) -> Array:
        return (
            self.active
            & self.valid
            & self.finite
            & self.lapse_positive
            & self.spatial_symmetric
            & self.extrinsic_symmetric
            & self.spatial_positive_definite
            & self.inverse_consistent
            & (self.sqrt_det_spatial_metric > 0.0)
            & self.determinant_consistent
        )

    @property
    def all_active_valid(self) -> Array:
        return jnp.all(~self.active | self.physically_valid)


class StressEnergyProjection(StrictModule, NonTrainableState):
    """Immutable matter-to-spacetime Eulerian stress-energy projection.

    ``energy_density`` is ``E``, ``momentum_covector`` is ``S_i``, and
    ``stress_covariant`` is ``S_ij``. Projection and conservation defects are
    explicit per-lane magnitudes; they are never folded into or clipped out of
    the physical fields. ``geometry_lineage_id`` binds the static provider/grid
    family while ``snapshot_token`` binds the exact dynamic geometry stage.
    """

    energy_density: Array
    momentum_covector: Array
    stress_covariant: Array
    active: Array
    valid: Array
    projection_defect: Array
    conservation_defect: Array
    snapshot_token: Array
    geometry_lineage_id: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)

    def __init__(
        self,
        energy_density: ArrayLike,
        momentum_covector: ArrayLike,
        stress_covariant: ArrayLike,
        active: ArrayLike,
        valid: ArrayLike,
        projection_defect: ArrayLike,
        conservation_defect: ArrayLike,
        /,
        *,
        snapshot_token: ArrayLike,
        geometry_lineage_id: str,
        convention_id: str,
        scale_id: str,
        topology_id: str,
        projection_id: str,
    ):
        energy_array = _real_array(energy_density, "energy_density")
        momentum_array = _real_array(momentum_covector, "momentum_covector")
        stress_array = _real_array(stress_covariant, "stress_covariant")
        projection_array = _real_array(projection_defect, "projection_defect")
        conservation_array = _real_array(
            conservation_defect,
            "conservation_defect",
        )
        active_array = jnp.asarray(active, dtype=jnp.bool_)
        valid_array = jnp.asarray(valid, dtype=jnp.bool_)
        snapshot_token_array = _snapshot_token(snapshot_token)
        leading_shape = energy_array.shape
        _require_shape(momentum_array, leading_shape + (3,), "momentum_covector")
        _require_shape(stress_array, leading_shape + (3, 3), "stress_covariant")
        _require_shape(active_array, leading_shape, "active")
        _require_shape(valid_array, leading_shape, "valid")
        _require_shape(projection_array, leading_shape, "projection_defect")
        _require_shape(conservation_array, leading_shape, "conservation_defect")
        numeric_arrays = (
            momentum_array,
            stress_array,
            projection_array,
            conservation_array,
        )
        if any(value.dtype != energy_array.dtype for value in numeric_arrays):
            raise TypeError("Stress-energy numeric fields must share one dtype.")

        self.energy_density = energy_array
        self.momentum_covector = momentum_array
        self.stress_covariant = stress_array
        self.active = active_array
        self.valid = valid_array
        self.projection_defect = projection_array
        self.conservation_defect = conservation_array
        self.snapshot_token = snapshot_token_array
        self.geometry_lineage_id = _identity(
            geometry_lineage_id,
            "geometry_lineage_id",
        )
        self.convention_id = _identity(convention_id, "convention_id")
        self.scale_id = _identity(scale_id, "scale_id")
        self.topology_id = _identity(topology_id, "topology_id")
        self.projection_id = _identity(projection_id, "projection_id")

    @property
    def leading_shape(self) -> tuple[int, ...]:
        return self.energy_density.shape

    @property
    def finite(self) -> Array:
        return (
            _finite_tensor(self.energy_density, 0)
            & _finite_tensor(self.momentum_covector, 1)
            & _finite_tensor(self.stress_covariant, 2)
            & _finite_tensor(self.projection_defect, 0)
            & _finite_tensor(self.conservation_defect, 0)
        )

    @property
    def stress_symmetry_defect(self) -> Array:
        return jnp.max(
            jnp.abs(self.stress_covariant - jnp.swapaxes(self.stress_covariant, -1, -2)),
            axis=(-2, -1),
        )

    @property
    def stress_symmetric(self) -> Array:
        scale = jnp.maximum(
            jnp.max(jnp.abs(self.stress_covariant), axis=(-2, -1)),
            1.0,
        )
        tolerance = 128.0 * jnp.finfo(self.stress_covariant.dtype).eps * scale
        return self.stress_symmetry_defect <= tolerance

    @property
    def physically_valid(self) -> Array:
        return (
            self.active
            & self.valid
            & self.finite
            & self.stress_symmetric
            & (self.projection_defect >= 0.0)
            & (self.conservation_defect >= 0.0)
        )

    @property
    def all_active_valid(self) -> Array:
        return jnp.all(~self.active | self.physically_valid)

    def compatible_with(self, geometry: ADMGridGeometry, /) -> Array:
        """Return exact static-lineage and dynamic-snapshot compatibility."""
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be an ADMGridGeometry.")
        static_compatible = (
            self.geometry_lineage_id == geometry.geometry_lineage_id
            and self.convention_id == geometry.convention_id
            and self.scale_id == geometry.scale_id
            and self.topology_id == geometry.topology_id
            and self.leading_shape == geometry.leading_shape
        )
        return jnp.asarray(static_compatible) & (
            self.snapshot_token == geometry.snapshot_token
        )


def combine_stress_energy_projections(
    projections: tuple[StressEnergyProjection, ...],
    /,
) -> StressEnergyProjection:
    """Sum compatible stress-energy projections at one exact ADM snapshot."""

    values = tuple(projections)
    if not values or any(
        not isinstance(value, StressEnergyProjection) for value in values
    ):
        raise TypeError(
            "projections must be a non-empty tuple of StressEnergyProjection values."
        )
    first = values[0]
    static_identity = (
        first.geometry_lineage_id,
        first.convention_id,
        first.scale_id,
        first.topology_id,
        first.leading_shape,
    )
    if any(
        (
            value.geometry_lineage_id,
            value.convention_id,
            value.scale_id,
            value.topology_id,
            value.leading_shape,
        )
        != static_identity
        for value in values[1:]
    ):
        raise ValueError("Stress-energy projections have incompatible static identities.")
    energy = first.energy_density
    momentum = first.momentum_covector
    stress = first.stress_covariant
    projection_defect = first.projection_defect
    conservation_defect = first.conservation_defect
    valid = first.valid
    for value in values[1:]:
        energy = eqx.error_if(
            energy,
            (value.snapshot_token != first.snapshot_token)
            | jnp.any(value.active != first.active),
            "Stress-energy projections must share one snapshot and active mask.",
        )
        energy = energy + value.energy_density
        momentum = momentum + value.momentum_covector
        stress = stress + value.stress_covariant
        projection_defect = projection_defect + jnp.abs(value.projection_defect)
        conservation_defect = conservation_defect + jnp.abs(value.conservation_defect)
        valid = valid & value.valid
    return StressEnergyProjection(
        energy,
        momentum,
        stress,
        first.active,
        valid,
        projection_defect,
        conservation_defect,
        snapshot_token=first.snapshot_token,
        geometry_lineage_id=first.geometry_lineage_id,
        convention_id=first.convention_id,
        scale_id=first.scale_id,
        topology_id=first.topology_id,
        projection_id=canonical_fingerprint(
            {
                "kind": "combined-stress-energy-projection",
                "contributors": [value.projection_id for value in values],
            }
        ),
    )


__all__ = [
    "ADMGridGeometry",
    "StressEnergyProjection",
    "combine_stress_energy_projections",
]
