#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import finite_real_scalar
from ..discretization import CellMesh
from ..geometry.brep._patches import AbstractCurve, AbstractSurfacePatch
from ..typing import checked, ConvertibleToArray
from . import _organization
from ._metric import MeshMetricField
from ._scope import MeshingEntityKind, MeshingScope


class FeatureKind(StrEnum):
    CORNER = "corner"
    CURVE = "curve"
    SURFACE = "surface"
    MATERIAL_INTERFACE = "material_interface"


class BoundaryLayerRoute(StrEnum):
    """Realization route of one boundary-layer control.

    ``EXACT_SWEEP`` meshes a pre-partitioned straight slab cap to cap;
    ``CAD_EXTRUSION`` partitions the slab by exact CAD extrusion of planar walls
    and then sweeps it; ``ADVANCING`` grows native layers from a closed wall
    surface mesh; ``PROVIDER`` lowers the schedule to the provider's own
    boundary-layer machinery and audits the measured result.
    """

    EXACT_SWEEP = "exact_sweep"
    CAD_EXTRUSION = "cad_extrusion"
    ADVANCING = "advancing"
    PROVIDER = "provider"


class BoundaryLayerCollisionPolicy(StrEnum):
    """Resolution of predicted or certified front collisions.

    ``FAIL`` rejects with evidence; ``TERMINATE_LOCALLY`` drops the colliding
    columns' outer layers (pyramid/tetrahedron terminations);
    ``REDUCE_THICKNESS`` scales colliding columns down to at least the
    control's minimum thickness fraction; ``MERGE`` joins vertex-paired
    opposing fronts at their common midsurface and rejects unpaired collisions.
    """

    FAIL = "fail"
    TERMINATE_LOCALLY = "terminate_locally"
    REDUCE_THICKNESS = "reduce_thickness"
    MERGE = "merge"


class BoundaryLayerCornerPolicy(StrEnum):
    """Treatment of wall edges whose dihedral deviation exceeds the feature angle.

    ``FAN`` splits columns across convex ridges and fills the wedge with fan
    templates (corner patches where three or more ridges meet); ``SMOOTH``
    grows one visibility-optimal column per vertex; ``REJECT`` refuses every
    feature edge. Concave corners without an admissible visible direction are
    rejected under every policy.
    """

    FAN = "fan"
    SMOOTH = "smooth"
    REJECT = "reject"


class ProtectedFeature(StrictModule, NonTrainableState):
    scope: MeshingScope
    feature_kind: FeatureKind = eqx.field(static=True)
    maximum_deviation: float = eqx.field(static=True)
    hard: bool = eqx.field(static=True)
    feature_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        scope: MeshingScope,
        feature_kind: FeatureKind,
        /,
        *,
        maximum_deviation: float = 0.0,
        hard: bool = True,
    ) -> None:
        if not isinstance(feature_kind, FeatureKind):
            raise TypeError("feature_kind must be FeatureKind.")
        deviation = float(maximum_deviation)
        if not np.isfinite(deviation) or deviation < 0.0:
            raise ValueError("maximum_deviation must be finite and non-negative.")
        self.scope = scope
        self.feature_kind = feature_kind
        self.maximum_deviation = deviation
        self.hard = bool(hard)
        self.feature_id = canonical_fingerprint(
            {
                "kind": "protected-feature",
                "scope": scope.scope_id,
                "feature_kind": feature_kind.value,
                "maximum_deviation": deviation,
                "hard": bool(hard),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[ProtectedFeature], value: ProtectedFeature, /
    ) -> ProtectedFeature:
        if type(value) is not cls:
            raise TypeError("value must be ProtectedFeature.")
        if type(value.maximum_deviation) is not float:
            raise TypeError("Stored protected-feature maximum_deviation must be float.")
        if type(value.hard) is not bool:
            raise TypeError("Stored protected-feature hard must be bool.")
        rebuilt = cls(
            MeshingScope.from_stored_fields(value.scope),
            value.feature_kind,
            maximum_deviation=value.maximum_deviation,
            hard=value.hard,
        )
        if (
            rebuilt.feature_id != value.feature_id
            or rebuilt.feature_kind is not value.feature_kind
            or rebuilt.maximum_deviation != value.maximum_deviation
            or rebuilt.hard != value.hard
        ):
            raise ValueError("Stored protected feature differs from its fields.")
        return rebuilt


class RegionSeed(StrictModule, NonTrainableState):
    point: Array
    region_name: str = eqx.field(static=True)
    material_id: str = eqx.field(static=True)
    role: _organization.RegionRole = eqx.field(static=True)
    seed_id: str = eqx.field(static=True)

    def __init__(
        self,
        point: ArrayLike,
        region_name: str,
        material_id: str,
        role: _organization.RegionRole,
        /,
    ) -> None:
        coordinates = np.asarray(point, dtype=np.float64)
        region = str(region_name).strip()
        material = str(material_id).strip()
        if (
            coordinates.ndim != 1
            or coordinates.size == 0
            or not np.all(np.isfinite(coordinates))
        ):
            raise ValueError("Region seed point must be one finite coordinate vector.")
        if not region or not material:
            raise ValueError("Region seed identities must be non-empty.")
        if not isinstance(role, _organization.RegionRole):
            raise TypeError("role must be RegionRole.")
        self.point = jnp.asarray(coordinates)
        self.region_name = region
        self.material_id = material
        self.role = role
        self.seed_id = canonical_fingerprint(
            {
                "kind": "region-seed",
                "point": array_tree_fingerprint(coordinates),
                "region_name": region,
                "material_id": material,
                "role": role.value,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(cls: type[RegionSeed], value: RegionSeed, /) -> RegionSeed:
        if type(value) is not cls:
            raise TypeError("value must be RegionSeed.")
        rebuilt = cls(value.point, value.region_name, value.material_id, value.role)
        if (
            rebuilt.seed_id != value.seed_id
            or rebuilt.region_name != value.region_name
            or rebuilt.material_id != value.material_id
            or rebuilt.role is not value.role
            or not np.array_equal(np.asarray(rebuilt.point), np.asarray(value.point))
        ):
            raise ValueError("Stored region seed differs from its exact point.")
        return rebuilt


class HoleSeed(StrictModule, NonTrainableState):
    point: Array
    scope: MeshingScope
    seed_id: str = eqx.field(static=True)

    @checked
    def __init__(self, point: ArrayLike, scope: MeshingScope, /) -> None:
        coordinates = np.asarray(point, dtype=np.float64)
        if (
            coordinates.ndim != 1
            or coordinates.size == 0
            or not np.all(np.isfinite(coordinates))
        ):
            raise ValueError("Hole seed point must be one finite coordinate vector.")
        self.point = jnp.asarray(coordinates)
        self.scope = scope
        self.seed_id = canonical_fingerprint(
            {
                "kind": "hole-seed",
                "point": array_tree_fingerprint(coordinates),
                "scope": scope.scope_id,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(cls: type[HoleSeed], value: HoleSeed, /) -> HoleSeed:
        if type(value) is not cls:
            raise TypeError("value must be HoleSeed.")
        rebuilt = cls(
            value.point,
            MeshingScope.from_stored_fields(value.scope),
        )
        if (
            rebuilt.seed_id != value.seed_id
            or rebuilt.scope.scope_id != value.scope.scope_id
            or not np.array_equal(np.asarray(rebuilt.point), np.asarray(value.point))
        ):
            raise ValueError("Stored hole seed differs from its exact point and scope.")
        return rebuilt


class RegionControl(StrictModule, NonTrainableState):
    scope: MeshingScope
    region_name: str = eqx.field(static=True)
    material_id: str = eqx.field(static=True)
    role: _organization.RegionRole = eqx.field(static=True)
    meshing_enabled: bool = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        scope: MeshingScope,
        region_name: str,
        material_id: str,
        role: _organization.RegionRole,
        /,
        *,
        meshing_enabled: bool = True,
    ) -> None:
        region = str(region_name).strip()
        material = str(material_id).strip()
        if not region or not material:
            raise ValueError("Region identities must be non-empty.")
        if not isinstance(role, _organization.RegionRole):
            raise TypeError("role must be RegionRole.")
        self.scope = scope
        self.region_name = region
        self.material_id = material
        self.role = role
        self.meshing_enabled = bool(meshing_enabled)
        self.control_id = canonical_fingerprint(
            {
                "kind": "region-control",
                "scope": scope.scope_id,
                "region_name": region,
                "material_id": material,
                "role": role.value,
                "meshing_enabled": bool(meshing_enabled),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[RegionControl], value: RegionControl, /
    ) -> RegionControl:
        if type(value) is not cls:
            raise TypeError("value must be RegionControl.")
        if type(value.meshing_enabled) is not bool:
            raise TypeError("Stored region-control meshing_enabled must be bool.")
        rebuilt = cls(
            MeshingScope.from_stored_fields(value.scope),
            value.region_name,
            value.material_id,
            value.role,
            meshing_enabled=value.meshing_enabled,
        )
        if (
            rebuilt.control_id != value.control_id
            or rebuilt.region_name != value.region_name
            or rebuilt.material_id != value.material_id
            or rebuilt.role is not value.role
            or rebuilt.meshing_enabled != value.meshing_enabled
        ):
            raise ValueError("Stored region control differs from its fields.")
        return rebuilt


class PatchControl(StrictModule, NonTrainableState):
    """Explicit codimension-one boundary or interface request."""

    name: str = eqx.field(static=True)
    scope: MeshingScope
    adjacent_region_names: tuple[str, ...] = eqx.field(static=True)
    required: bool = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        name: str,
        scope: MeshingScope,
        adjacent_region_names: tuple[str, ...],
        /,
        *,
        required: bool = True,
    ) -> None:
        value = str(name).strip()
        if not value:
            raise ValueError("Patch control name must be non-empty.")
        if isinstance(adjacent_region_names, str):
            raise TypeError("adjacent_region_names must be an iterable of region names.")
        regions = tuple(sorted(str(region).strip() for region in adjacent_region_names))
        if len(regions) not in (1, 2):
            raise ValueError("Patch controls require one or two adjacent region names.")
        if any(not region for region in regions) or len(set(regions)) != len(regions):
            raise ValueError(
                "Patch controls require distinct non-empty adjacent region names."
            )
        self.name = value
        self.scope = scope
        self.adjacent_region_names = regions
        self.required = bool(required)
        self.control_id = canonical_fingerprint(
            {
                "kind": "patch-control",
                "name": value,
                "scope": scope.scope_id,
                "adjacent_region_names": regions,
                "required": bool(required),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[PatchControl], value: PatchControl, /
    ) -> PatchControl:
        if type(value) is not cls:
            raise TypeError("value must be PatchControl.")
        if type(value.required) is not bool:
            raise TypeError("Stored patch-control required must be bool.")
        rebuilt = cls(
            value.name,
            MeshingScope.from_stored_fields(value.scope),
            value.adjacent_region_names,
            required=value.required,
        )
        if (
            rebuilt.control_id != value.control_id
            or rebuilt.name != value.name
            or rebuilt.adjacent_region_names != value.adjacent_region_names
            or rebuilt.required != value.required
        ):
            raise ValueError("Stored patch control differs from its fields.")
        return rebuilt


class LayerSchedule(StrictModule, NonTrainableState):
    """Exact layer thicknesses, ordered from the wall outward."""

    thicknesses: tuple[float, ...] = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(self, thicknesses: ConvertibleToArray, /) -> None:
        values = np.asarray(thicknesses)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("thicknesses must be one non-empty vector.")
        if not (
            np.issubdtype(values.dtype, np.integer)
            or np.issubdtype(values.dtype, np.floating)
        ):
            raise TypeError("thicknesses must contain real numeric values.")
        normalized = values.astype("float64", copy=False)
        if not np.all(np.isfinite(normalized)) or np.any(normalized <= 0.0):
            raise ValueError("thicknesses must be finite and strictly positive.")
        explicit = tuple(float(value) for value in normalized)
        self.thicknesses = explicit
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "layer-schedule",
                "thicknesses": explicit,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[LayerSchedule], value: LayerSchedule, /
    ) -> LayerSchedule:
        if type(value) is not cls:
            raise TypeError("value must be LayerSchedule.")
        if type(value.thicknesses) is not tuple or any(
            type(thickness) is not float for thickness in value.thicknesses
        ):
            raise TypeError("Stored layer schedule thicknesses must be float values.")
        rebuilt = cls(value.thicknesses)
        if (
            rebuilt.schedule_id != value.schedule_id
            or rebuilt.thicknesses != value.thicknesses
        ):
            raise ValueError("Stored layer schedule differs from its exact thicknesses.")
        return rebuilt

    @classmethod
    def geometric(
        cls,
        layer_count: int,
        first_layer_thickness: float,
        /,
        *,
        growth_rate: float = 1.0,
    ) -> LayerSchedule:
        count_value = np.asarray(layer_count)
        if (
            count_value.ndim != 0
            or np.issubdtype(count_value.dtype, np.bool_)
            or not np.issubdtype(count_value.dtype, np.integer)
        ):
            raise TypeError("layer_count must be an integer.")
        count = int(count_value)
        if count <= 0:
            raise ValueError("layer_count must be positive.")
        first_value = np.asarray(first_layer_thickness)
        growth_value = np.asarray(growth_rate)
        if any(
            value.ndim != 0
            or np.issubdtype(value.dtype, np.bool_)
            or not (
                np.issubdtype(value.dtype, np.integer)
                or np.issubdtype(value.dtype, np.floating)
            )
            for value in (first_value, growth_value)
        ):
            raise TypeError(
                "first_layer_thickness and growth_rate must be real numeric scalars."
            )
        first = float(first_value)
        growth = float(growth_value)
        if not np.isfinite(first) or first <= 0.0:
            raise ValueError("first_layer_thickness must be positive and finite.")
        if not np.isfinite(growth) or growth <= 0.0:
            raise ValueError("growth_rate must be positive and finite.")
        return cls(tuple(first * growth**index for index in range(count)))

    @property
    def layer_count(self) -> int:
        return len(self.thicknesses)

    @property
    def total_thickness(self) -> float:
        return float(sum(self.thicknesses))

    @property
    def growth_rates(self) -> tuple[float, ...]:
        """Consecutive outward thickness ratios ``t[k + 1] / t[k]``."""
        return tuple(
            outer / inner
            for inner, outer in zip(
                self.thicknesses[:-1], self.thicknesses[1:], strict=True
            )
        )


def _layer_scope_dimensions(
    route: BoundaryLayerRoute,
    wall_scope: MeshingScope,
    volume_scope: MeshingScope | None,
    cap_scope: MeshingScope | None,
    /,
) -> None:
    """Enforce the scope signature each route consumes."""
    wall = wall_scope.entity_dimension
    volume = None if volume_scope is None else volume_scope.entity_dimension
    cap = None if cap_scope is None else cap_scope.entity_dimension
    geometry = wall_scope.entity_kind is MeshingEntityKind.GEOMETRY
    match route:
        case BoundaryLayerRoute.EXACT_SWEEP:
            valid = (wall, cap, volume) == (2, 2, 3)
            message = "EXACT_SWEEP requires wall and cap face scopes and a volume scope."
        case BoundaryLayerRoute.CAD_EXTRUSION:
            valid = (wall, cap, volume) == (2, None, 3)
            message = (
                "CAD_EXTRUSION requires a wall face scope, a volume scope, and no cap."
            )
        case BoundaryLayerRoute.ADVANCING:
            valid = (wall, cap, volume) == (2, None, 3 if geometry else None)
            message = (
                "ADVANCING requires a wall face scope, no cap, and a volume scope "
                "exactly for geometry-bound walls."
            )
        case BoundaryLayerRoute.PROVIDER:
            valid = (wall, cap, volume) in ((1, None, None), (2, None, 3))
            message = (
                "PROVIDER requires wall curves without a volume, or wall faces with "
                "a volume scope; it takes no cap."
            )
        case _:
            raise TypeError("route must be BoundaryLayerRoute.")
    if not valid:
        raise ValueError(message)
    if cap_scope is not None:
        if cap_scope.entity_set_id != wall_scope.entity_set_id:
            raise ValueError("Wall and cap face scopes must share one entity set.")
        if np.intersect1d(
            np.asarray(wall_scope.entity_ids), np.asarray(cap_scope.entity_ids)
        ).size:
            raise ValueError("Wall and cap face scopes must be disjoint.")


class BoundaryLayerControl(StrictModule, NonTrainableState):
    """One boundary-layer request grown from ``wall_scope`` by ``schedule``.

    ``volume_scope`` names the solids receiving the layers and ``cap_scope`` the
    exact target caps of an ``EXACT_SWEEP``. ``feature_angle`` (radians)
    classifies wall edges and bounds each fan sector; ``maximum_corner_stretch``
    bounds the column height amplification ``1 / min(direction . face normal)``
    that keeps a concave corner admissible. ``growth_rate_bounds`` optionally
    bounds the requested and the realized consecutive-layer ratios, and
    ``smoothing_iterations`` bounds the weighted Laplacian smoothing of column
    directions and heights. The core provider always keeps the cap and the
    remaining boundary fixed; ``core_maximum_size`` optionally caps its cell size.
    """

    wall_scope: MeshingScope
    schedule: LayerSchedule
    volume_scope: MeshingScope | None
    cap_scope: MeshingScope | None
    route: BoundaryLayerRoute = eqx.field(static=True)
    collision: BoundaryLayerCollisionPolicy = eqx.field(static=True)
    corner: BoundaryLayerCornerPolicy = eqx.field(static=True)
    feature_angle: float = eqx.field(static=True)
    minimum_thickness_fraction: float = eqx.field(static=True)
    growth_rate_bounds: tuple[float, float] | None = eqx.field(static=True)
    maximum_corner_stretch: float = eqx.field(static=True)
    smoothing_iterations: int = eqx.field(static=True)
    core_maximum_size: float | None = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        wall_scope: MeshingScope,
        schedule: LayerSchedule,
        /,
        *,
        route: BoundaryLayerRoute,
        volume_scope: MeshingScope | None = None,
        cap_scope: MeshingScope | None = None,
        collision: BoundaryLayerCollisionPolicy = BoundaryLayerCollisionPolicy.FAIL,
        corner: BoundaryLayerCornerPolicy = BoundaryLayerCornerPolicy.FAN,
        feature_angle: float = np.pi / 6.0,
        minimum_thickness_fraction: float = 0.5,
        growth_rate_bounds: tuple[float, float] | None = None,
        maximum_corner_stretch: float = 2.0,
        smoothing_iterations: int = 8,
        core_maximum_size: float | None = None,
    ) -> None:
        scopes = tuple(
            scope for scope in (wall_scope, volume_scope, cap_scope) if scope is not None
        )
        if not all(isinstance(scope, MeshingScope) for scope in scopes):
            raise TypeError("Boundary-layer scopes must be MeshingScope values or None.")
        if not isinstance(route, BoundaryLayerRoute):
            raise TypeError("route must be BoundaryLayerRoute.")
        if not isinstance(collision, BoundaryLayerCollisionPolicy):
            raise TypeError("collision must be BoundaryLayerCollisionPolicy.")
        if not isinstance(corner, BoundaryLayerCornerPolicy):
            raise TypeError("corner must be BoundaryLayerCornerPolicy.")
        binding = (
            wall_scope.source_id,
            wall_scope.source_revision,
            wall_scope.entity_kind,
        )
        if any(
            (scope.source_id, scope.source_revision, scope.entity_kind) != binding
            for scope in scopes[1:]
        ):
            raise ValueError("Boundary-layer scopes must share one source binding.")
        _layer_scope_dimensions(route, wall_scope, volume_scope, cap_scope)
        if (
            route is not BoundaryLayerRoute.ADVANCING
            and collision is not BoundaryLayerCollisionPolicy.FAIL
        ):
            raise ValueError(
                "Only the ADVANCING route resolves collisions; other routes FAIL."
            )
        angle = finite_real_scalar(feature_angle, "feature_angle")
        if not 0.0 < angle < np.pi:
            raise ValueError("feature_angle must lie strictly between zero and pi.")
        fraction = finite_real_scalar(
            minimum_thickness_fraction, "minimum_thickness_fraction"
        )
        if not 0.0 < fraction <= 1.0:
            raise ValueError("minimum_thickness_fraction must lie in (0, 1].")
        stretch = finite_real_scalar(maximum_corner_stretch, "maximum_corner_stretch")
        if stretch < 1.0:
            raise ValueError("maximum_corner_stretch must be at least one.")
        if isinstance(smoothing_iterations, bool) or not isinstance(
            smoothing_iterations, int
        ):
            raise TypeError("smoothing_iterations must be an integer.")
        if smoothing_iterations < 0:
            raise ValueError("smoothing_iterations must be non-negative.")
        bounds = None
        if growth_rate_bounds is not None:
            if len(growth_rate_bounds) != 2:
                raise ValueError("growth_rate_bounds must be (lower, upper).")
            lower = finite_real_scalar(growth_rate_bounds[0], "growth_rate_bounds")
            upper = finite_real_scalar(growth_rate_bounds[1], "growth_rate_bounds")
            if not 0.0 < lower <= upper:
                raise ValueError("growth_rate_bounds must satisfy 0 < lower <= upper.")
            if any(not lower <= rate <= upper for rate in schedule.growth_rates):
                raise ValueError("The schedule growth rates violate growth_rate_bounds.")
            bounds = (lower, upper)
        core_size = None
        if core_maximum_size is not None:
            core_size = finite_real_scalar(core_maximum_size, "core_maximum_size")
            if core_size <= 0.0:
                raise ValueError("core_maximum_size must be positive.")
        self.wall_scope = wall_scope
        self.schedule = schedule
        self.volume_scope = volume_scope
        self.cap_scope = cap_scope
        self.route = route
        self.collision = collision
        self.corner = corner
        self.feature_angle = angle
        self.minimum_thickness_fraction = fraction
        self.growth_rate_bounds = bounds
        self.maximum_corner_stretch = stretch
        self.smoothing_iterations = smoothing_iterations
        self.core_maximum_size = core_size
        self.control_id = canonical_fingerprint(
            {
                "kind": "boundary-layer-control",
                "wall_scope": wall_scope.scope_id,
                "volume_scope": None if volume_scope is None else volume_scope.scope_id,
                "cap_scope": None if cap_scope is None else cap_scope.scope_id,
                "schedule": schedule.schedule_id,
                "route": route.value,
                "collision": collision.value,
                "corner": corner.value,
                "feature_angle": angle,
                "minimum_thickness_fraction": fraction,
                "growth_rate_bounds": bounds,
                "maximum_corner_stretch": stretch,
                "smoothing_iterations": smoothing_iterations,
                "core_maximum_size": core_size,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[BoundaryLayerControl], value: BoundaryLayerControl, /
    ) -> BoundaryLayerControl:
        if type(value) is not cls:
            raise TypeError("value must be BoundaryLayerControl.")
        if (
            type(value.feature_angle) is not float
            or type(value.minimum_thickness_fraction) is not float
            or type(value.maximum_corner_stretch) is not float
            or type(value.smoothing_iterations) is not int
            or (
                value.core_maximum_size is not None
                and type(value.core_maximum_size) is not float
            )
            or (
                value.growth_rate_bounds is not None
                and (
                    type(value.growth_rate_bounds) is not tuple
                    or any(type(rate) is not float for rate in value.growth_rate_bounds)
                )
            )
        ):
            raise TypeError("Stored boundary-layer control fields are not canonical.")
        rebuilt = cls(
            MeshingScope.from_stored_fields(value.wall_scope),
            LayerSchedule.from_stored_fields(value.schedule),
            route=value.route,
            volume_scope=None
            if value.volume_scope is None
            else MeshingScope.from_stored_fields(value.volume_scope),
            cap_scope=None
            if value.cap_scope is None
            else MeshingScope.from_stored_fields(value.cap_scope),
            collision=value.collision,
            corner=value.corner,
            feature_angle=value.feature_angle,
            minimum_thickness_fraction=value.minimum_thickness_fraction,
            growth_rate_bounds=value.growth_rate_bounds,
            maximum_corner_stretch=value.maximum_corner_stretch,
            smoothing_iterations=value.smoothing_iterations,
            core_maximum_size=value.core_maximum_size,
        )
        if (
            rebuilt.control_id != value.control_id
            or rebuilt.route is not value.route
            or rebuilt.collision is not value.collision
            or rebuilt.corner is not value.corner
            or rebuilt.feature_angle != value.feature_angle
            or rebuilt.minimum_thickness_fraction != value.minimum_thickness_fraction
            or rebuilt.growth_rate_bounds != value.growth_rate_bounds
            or rebuilt.maximum_corner_stretch != value.maximum_corner_stretch
            or rebuilt.smoothing_iterations != value.smoothing_iterations
            or rebuilt.core_maximum_size != value.core_maximum_size
        ):
            raise ValueError("Stored boundary-layer control differs from its fields.")
        return rebuilt


class PeriodicConstraint(StrictModule, NonTrainableState):
    source_scope: MeshingScope
    target_scope: MeshingScope
    transform: Array
    tolerance: float = eqx.field(static=True)
    conforming_required: bool = eqx.field(static=True)
    constraint_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_scope: MeshingScope,
        target_scope: MeshingScope,
        transform: ArrayLike,
        /,
        *,
        tolerance: float = 1.0e-10,
        conforming_required: bool = True,
    ) -> None:
        if not isinstance(source_scope, MeshingScope) or not isinstance(
            target_scope, MeshingScope
        ):
            raise TypeError("Periodic scopes must be MeshingScope values.")
        if source_scope.source_revision != target_scope.source_revision:
            raise ValueError("Periodic scopes must share one source revision.")
        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
            raise ValueError("Periodic transform must be one square homogeneous matrix.")
        if not np.all(np.isfinite(matrix)) or not np.allclose(
            matrix[-1],
            np.eye(matrix.shape[0])[-1],
        ):
            raise ValueError("Periodic transform must be finite and homogeneous.")
        if abs(np.linalg.det(matrix[:-1, :-1])) <= np.finfo(np.float64).eps:
            raise ValueError("Periodic transform must be invertible.")
        threshold = float(tolerance)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("Periodic tolerance must be finite and non-negative.")
        self.source_scope = source_scope
        self.target_scope = target_scope
        self.transform = jnp.asarray(matrix)
        self.tolerance = threshold
        self.conforming_required = bool(conforming_required)
        self.constraint_id = canonical_fingerprint(
            {
                "kind": "periodic-meshing-constraint",
                "source_scope": source_scope.scope_id,
                "target_scope": target_scope.scope_id,
                "transform": array_tree_fingerprint(matrix),
                "tolerance": threshold,
                "conforming_required": bool(conforming_required),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[PeriodicConstraint], value: PeriodicConstraint, /
    ) -> PeriodicConstraint:
        if type(value) is not cls:
            raise TypeError("value must be PeriodicConstraint.")
        if type(value.conforming_required) is not bool:
            raise TypeError(
                "Stored periodic-constraint conforming_required must be bool."
            )
        stored_transform = np.asarray(value.transform)
        if type(value.tolerance) is not float or stored_transform.dtype != np.dtype(
            np.float64
        ):
            raise TypeError("Stored periodic-constraint fields are not canonical.")
        rebuilt = cls(
            MeshingScope.from_stored_fields(value.source_scope),
            MeshingScope.from_stored_fields(value.target_scope),
            value.transform,
            tolerance=value.tolerance,
            conforming_required=value.conforming_required,
        )
        if (
            rebuilt.constraint_id != value.constraint_id
            or rebuilt.tolerance != value.tolerance
            or rebuilt.conforming_required != value.conforming_required
            or not np.array_equal(
                np.asarray(rebuilt.transform), np.asarray(value.transform)
            )
        ):
            raise ValueError(
                "Stored periodic constraint differs from its exact transform."
            )
        return rebuilt


class BackgroundMetricMode(StrEnum):
    """How a Riemannian metric drives provider background sizing."""

    ISOTROPIC = "isotropic"
    ANISOTROPIC = "anisotropic"


class BackgroundMetricControl(StrictModule, NonTrainableState):
    """Metric sizing sampled on one explicit affine simplex background mesh.

    `ISOTROPIC` lowers each vertex metric to its smallest directional size
    `1 / sqrt(lambda_max)`, so no metric direction is under-resolved. `ANISOTROPIC`
    lowers the complete tensor. The `MeshMetricField` certifies its declared size
    and anisotropy bounds at construction; this control never repairs a metric.
    """

    mesh: CellMesh
    metric: MeshMetricField
    coordinate_contract: SpatialCoordinateContract
    mode: BackgroundMetricMode = eqx.field(static=True)
    maximum_metric_edge_length: float | None = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        mesh: CellMesh,
        metric: MeshMetricField,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        mode: BackgroundMetricMode = BackgroundMetricMode.ANISOTROPIC,
        maximum_metric_edge_length: float | None = None,
    ) -> None:
        if not isinstance(mode, BackgroundMetricMode):
            raise TypeError("mode must be BackgroundMetricMode.")
        kinds = {block.cell_kind for block in mesh.blocks}
        simplex = {2: "triangle", 3: "tetrahedron"}.get(mesh.topological_dimension)
        if kinds != {simplex}:
            raise ValueError(
                "Background metric meshes must contain one affine triangle or tetrahedron family."
            )
        scope = metric.scope
        vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        if (
            scope.source_id != mesh.mesh_id
            or scope.source_revision != mesh.numeric_version
            or scope.entity_kind is not MeshingEntityKind.MESH
            or scope.entity_dimension != 0
            or scope.entity_set_id != mesh.entity_set(0).entity_set_id
            or not np.array_equal(np.asarray(scope.entity_ids), np.sort(vertex_ids))
        ):
            raise ValueError(
                "Background metric must bind every vertex of its exact mesh revision."
            )
        if metric.values.shape[1:] != (mesh.ambient_dimension, mesh.ambient_dimension):
            raise ValueError(
                "Background metric tensors must match the mesh ambient dimension."
            )
        maximum_length = (
            None
            if maximum_metric_edge_length is None
            else float(maximum_metric_edge_length)
        )
        if maximum_length is not None and (
            not np.isfinite(maximum_length) or maximum_length <= 0.0
        ):
            raise ValueError("maximum_metric_edge_length must be positive and finite.")
        self.mesh = mesh
        self.metric = metric
        self.coordinate_contract = coordinate_contract
        self.mode = mode
        self.maximum_metric_edge_length = maximum_length
        self.control_id = canonical_fingerprint(
            {
                "kind": "background-metric-control",
                "mesh": mesh.mesh_id,
                "metric": metric.metric_id,
                "coordinates": coordinate_contract.spatial_id,
                "mode": mode.value,
                "maximum_metric_edge_length": maximum_length,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[BackgroundMetricControl], value: BackgroundMetricControl, /
    ) -> BackgroundMetricControl:
        if type(value) is not cls:
            raise TypeError("value must be BackgroundMetricControl.")
        if type(value.coordinate_contract) is not SpatialCoordinateContract:
            raise TypeError(
                "Stored background metric lost its exact coordinate contract."
            )
        if (
            value.maximum_metric_edge_length is not None
            and type(value.maximum_metric_edge_length) is not float
        ):
            raise TypeError("Stored background metric edge length must be float or None.")
        mesh = CellMesh.from_stored_fields(value.mesh)
        metric = MeshMetricField.from_stored_fields(value.metric)
        coordinate_contract = SpatialCoordinateContract.from_dict(
            value.coordinate_contract.to_dict()
        )
        rebuilt = cls(
            mesh,
            metric,
            coordinate_contract,
            mode=value.mode,
            maximum_metric_edge_length=value.maximum_metric_edge_length,
        )
        if (
            rebuilt.control_id != value.control_id
            or rebuilt.mode is not value.mode
            or rebuilt.maximum_metric_edge_length != value.maximum_metric_edge_length
        ):
            raise ValueError("Stored background-metric control differs from its inputs.")
        return rebuilt

    @property
    def vertex_metrics(self) -> np.ndarray:
        """Metric tensors in CellMesh vertex-row order."""
        # MeshingScope sorts identifiers; metric rows follow that order.
        order = np.searchsorted(
            np.asarray(self.metric.scope.entity_ids, dtype=np.int64),
            np.asarray(self.mesh.vertex_global_ids, dtype=np.int64),
        )
        return np.asarray(self.metric.values, dtype=np.float64)[order]


class SurfaceReconstructionControl(StrictModule, NonTrainableState):
    """Feature classification and fidelity bound for remeshing a discrete surface.

    Source-mesh edges whose dihedral angle exceeds `feature_angle` become sharp
    feature curves; boundary polylines split where their turning angle exceeds
    `curve_angle`. `maximum_deviation` bounds the two-sided vertex distance
    between the source and remeshed surfaces in source coordinates.
    """

    feature_angle: float = eqx.field(static=True)
    curve_angle: float = eqx.field(static=True)
    maximum_deviation: float = eqx.field(static=True)
    force_parametrizable_patches: bool = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    def __init__(
        self,
        feature_angle: float,
        maximum_deviation: float,
        /,
        *,
        curve_angle: float = np.pi,
        force_parametrizable_patches: bool = True,
    ) -> None:
        feature = float(feature_angle)
        curve = float(curve_angle)
        deviation = float(maximum_deviation)
        if not np.isfinite(feature) or not 0.0 < feature < np.pi:
            raise ValueError("feature_angle must lie strictly between zero and pi.")
        if not np.isfinite(curve) or not 0.0 < curve <= np.pi:
            raise ValueError("curve_angle must lie in (0, pi].")
        if not np.isfinite(deviation) or deviation <= 0.0:
            raise ValueError("maximum_deviation must be positive and finite.")
        self.feature_angle = feature
        self.curve_angle = curve
        self.maximum_deviation = deviation
        self.force_parametrizable_patches = bool(force_parametrizable_patches)
        self.control_id = canonical_fingerprint(
            {
                "kind": "surface-reconstruction-control",
                "feature_angle": feature,
                "curve_angle": curve,
                "maximum_deviation": deviation,
                "force_parametrizable_patches": bool(force_parametrizable_patches),
            }
        )


class TransfiniteCurveControl(StrictModule, NonTrainableState):
    """Bind an oriented source-curve interval to a fixed interval count."""

    curve: AbstractCurve
    source_id: str = eqx.field(static=True)
    parameter_range: tuple[float, float] = eqx.field(static=True)
    intervals: int = eqx.field(static=True)
    reversed: bool = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        curve: AbstractCurve,
        source_id: str,
        parameter_range: tuple[float, float],
        intervals: int,
        /,
        *,
        reversed: bool = False,
    ) -> None:
        if not isinstance(source_id, str) or not isinstance(reversed, bool):
            raise TypeError("source_id must be a string and reversed must be bool.")
        if not source_id.strip():
            raise ValueError("source_id must be non-empty.")
        if isinstance(intervals, bool) or not isinstance(intervals, int) or intervals < 1:
            raise ValueError("intervals must be a positive integer.")
        first, last = parameter_range
        lower = finite_real_scalar(first, "parameter_range[0]")
        upper = finite_real_scalar(last, "parameter_range[1]")
        if lower >= upper:
            raise ValueError("Transfinite curve ranges must be strictly increasing.")
        domain = curve.parameter_domain
        if domain is not None and (lower < domain[0] or upper > domain[1]):
            raise ValueError("Transfinite curve ranges must lie in the source domain.")
        bounds = (lower, upper)
        self.curve = curve
        self.source_id = source_id
        self.parameter_range = bounds
        self.intervals = intervals
        self.reversed = reversed
        self.control_id = canonical_fingerprint(
            {
                "kind": "transfinite-curve-control",
                "source": source_id,
                "curve": array_tree_fingerprint(curve),
                "range": bounds,
                "intervals": intervals,
                "reversed": reversed,
            }
        )


@final
class TransfiniteSurfaceControl(StrictModule, NonTrainableState):
    """Bind a source face chart to the two increasing logical face axes."""

    surface: AbstractSurfacePatch
    parameter_box: Array
    source_id: str = eqx.field(static=True)
    intervals: tuple[int, int] = eqx.field(static=True)
    permutation: tuple[int, int] = eqx.field(static=True)
    flips: tuple[bool, bool] = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        surface: AbstractSurfacePatch,
        source_id: str,
        parameter_box: ArrayLike,
        intervals: tuple[int, int],
        /,
        *,
        permutation: tuple[int, int] = (0, 1),
        flips: tuple[bool, bool] = (False, False),
    ) -> None:
        if not isinstance(source_id, str):
            raise TypeError("source_id must be a string.")
        if any(
            isinstance(axis, bool) or not isinstance(axis, int) for axis in permutation
        ) or any(not isinstance(flip, bool) for flip in flips):
            raise TypeError(
                "permutation axes must be integers and flips must be booleans."
            )
        if not source_id.strip():
            raise ValueError("source_id must be non-empty.")
        if len(intervals) != 2 or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 1
            for count in intervals
        ):
            raise ValueError("intervals must contain two positive integers.")
        if sorted(permutation) != [0, 1] or len(flips) != 2:
            raise ValueError(
                "permutation must contain each axis once and flips need two entries."
            )
        box = surface.validate_parameter_box(parameter_box)
        self.surface = surface
        self.parameter_box = jax.device_put(np.asarray(box, dtype=np.float64))
        self.source_id = source_id
        self.intervals = intervals
        self.permutation = permutation
        self.flips = flips
        self.control_id = canonical_fingerprint(
            {
                "kind": "transfinite-surface-control",
                "source": source_id,
                "surface": array_tree_fingerprint(surface),
                "box": box,
                "intervals": intervals,
                "permutation": permutation,
                "flips": flips,
            }
        )


@final
class BlockInterfaceControl(StrictModule, NonTrainableState):
    """Declare oriented logical-face gluing between two source blocks."""

    first_block: str = eqx.field(static=True)
    second_block: str = eqx.field(static=True)
    first_face: int = eqx.field(static=True)
    second_face: int = eqx.field(static=True)
    permutation: tuple[int, ...] = eqx.field(static=True)
    flips: tuple[bool, ...] = eqx.field(static=True)
    conforming: bool = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    def __init__(
        self,
        first_block: str,
        first_face: int,
        second_block: str,
        second_face: int,
        permutation: tuple[int, ...],
        flips: tuple[bool, ...],
        /,
        *,
        conforming: bool = True,
        tolerance: float = 1e-12,
    ) -> None:
        dimension = len(permutation)
        if not isinstance(first_block, str) or not isinstance(second_block, str):
            raise TypeError("block IDs must be strings.")
        if not isinstance(conforming, bool) or any(
            not isinstance(flip, bool) for flip in flips
        ):
            raise TypeError("conforming and face flips must be booleans.")
        if any(
            isinstance(axis, bool) or not isinstance(axis, int) for axis in permutation
        ):
            raise TypeError("permutation axes must be integers.")
        if dimension not in (1, 2) or sorted(permutation) != list(range(dimension)):
            raise ValueError("permutation must contain each face axis exactly once.")
        if len(flips) != dimension:
            raise ValueError("flips must match the face dimension.")
        if not first_block.strip() or not second_block.strip():
            raise ValueError("block IDs must be non-empty.")
        if any(
            isinstance(face, bool)
            or not isinstance(face, int)
            or not 0 <= face < 2 * (dimension + 1)
            for face in (first_face, second_face)
        ):
            raise ValueError("face indices must identify valid logical block faces.")
        tolerance_ = finite_real_scalar(tolerance, "tolerance")
        if tolerance_ < 0.0:
            raise ValueError("tolerance must be nonnegative.")
        self.first_block = first_block
        self.second_block = second_block
        self.first_face = first_face
        self.second_face = second_face
        self.permutation = permutation
        self.flips = tuple(flips)
        self.conforming = conforming
        self.tolerance = tolerance_
        self.control_id = canonical_fingerprint(
            {
                "kind": "block-interface-control",
                "first": (first_block, first_face),
                "second": (second_block, second_face),
                "permutation": permutation,
                "flips": self.flips,
                "conforming": conforming,
                "tolerance": tolerance_,
            }
        )


__all__ = [
    "BackgroundMetricControl",
    "BackgroundMetricMode",
    "BoundaryLayerCollisionPolicy",
    "BoundaryLayerControl",
    "BoundaryLayerCornerPolicy",
    "BoundaryLayerRoute",
    "FeatureKind",
    "HoleSeed",
    "LayerSchedule",
    "PatchControl",
    "PeriodicConstraint",
    "ProtectedFeature",
    "RegionControl",
    "RegionSeed",
    "SurfaceReconstructionControl",
    "TransfiniteCurveControl",
    "TransfiniteSurfaceControl",
    "BlockInterfaceControl",
]
