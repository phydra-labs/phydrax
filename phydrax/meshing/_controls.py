#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum
from typing import final

import equinox as eqx
import jax
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.discretization._cell_mesh as cell_mesh_owner

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import finite_real_scalar
from ..geometry._meshing_domain import MeshingDomain
from ..geometry.brep._patches import AbstractCurve, AbstractSurfacePatch
from ..geometry.brep._selection import RecipeSelectionBinding
from ..geometry.design._naming import RecipeEntityNames
from ..typing import checked, parse
from . import _organization_records
from ._metric import MeshMetricField
from ._scope import MeshingEntityKind, MeshingScope


class FeatureKind(StrEnum):
    CORNER = "corner"
    CURVE = "curve"
    SURFACE = "surface"
    MATERIAL_INTERFACE = "material_interface"


class BoundaryLayerRoute(StrEnum):
    """Select the realization route for a boundary-layer control.

    ``EXACT_SWEEP`` meshes a pre-partitioned straight slab cap to cap;
    ``CAD_EXTRUSION`` partitions the slab by CAD extrusion of planar walls
    and then sweeps it; ``ADVANCING`` grows native layers from a closed wall
    surface mesh; ``PROVIDER`` lowers the schedule to the provider's own
    boundary-layer machinery and audits the measured result.
    """

    EXACT_SWEEP = "exact_sweep"
    CAD_EXTRUSION = "cad_extrusion"
    ADVANCING = "advancing"
    PROVIDER = "provider"


class BoundaryLayerCollisionPolicy(StrEnum):
    """Select how predicted or certified front collisions are resolved.

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
    """Select treatment of wall edges exceeding the feature angle.

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
        feature_kind = parse(feature_kind, FeatureKind, "ProtectedFeature.feature_kind")
        deviation = float(maximum_deviation)
        if not np.isfinite(deviation) or deviation < 0.0:
            raise ValueError(
                f"ProtectedFeature.maximum_deviation must be finite and non-negative; got {deviation!r}."
            )
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


class RegionSeed(StrictModule, NonTrainableState):
    point: Array
    region_name: str = eqx.field(static=True)
    material_id: str = eqx.field(static=True)
    role: _organization_records.RegionRole = eqx.field(static=True)
    seed_id: str = eqx.field(static=True)

    def __init__(
        self,
        point: ArrayLike,
        region_name: str,
        material_id: str,
        role: _organization_records.RegionRole,
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
            raise ValueError(
                f"RegionSeed.point must be one finite coordinate vector; got shape={coordinates.shape}, dtype={coordinates.dtype}."
            )
        if not region or not material:
            raise ValueError(
                f"RegionSeed.identities must be non-empty; got region_name={region!r}, material_id={material!r}."
            )
        role = parse(role, _organization_records.RegionRole, "RegionSeed.role")
        self.point = jax.device_put(np.asarray(coordinates))
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
            raise ValueError(
                f"HoleSeed.point must be one finite coordinate vector; got shape={coordinates.shape}, dtype={coordinates.dtype}."
            )
        self.point = jax.device_put(np.asarray(coordinates))
        self.scope = scope
        self.seed_id = canonical_fingerprint(
            {
                "kind": "hole-seed",
                "point": array_tree_fingerprint(coordinates),
                "scope": scope.scope_id,
            }
        )


class RegionControl(StrictModule, NonTrainableState):
    scope: MeshingScope
    region_name: str = eqx.field(static=True)
    material_id: str = eqx.field(static=True)
    role: _organization_records.RegionRole = eqx.field(static=True)
    meshing_enabled: bool = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        scope: MeshingScope,
        region_name: str,
        material_id: str,
        role: _organization_records.RegionRole,
        /,
        *,
        meshing_enabled: bool = True,
    ) -> None:
        region = str(region_name).strip()
        material = str(material_id).strip()
        if not region or not material:
            raise ValueError(
                f"RegionControl.identities must be non-empty; got region_name={region!r}, material_id={material!r}."
            )
        role = parse(role, _organization_records.RegionRole, "RegionControl.role")
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


class PatchControl(StrictModule, NonTrainableState):
    """Declare a codimension-one boundary or interface request."""

    name: str = eqx.field(static=True)
    scope: MeshingScope
    adjacent_region_names: tuple[str, ...] = eqx.field(static=True)
    required: bool = eqx.field(static=True)
    control_id: str = eqx.field(static=True)
    recipe_binding: RecipeSelectionBinding | None

    @checked
    def __init__(
        self,
        name: str,
        scope: MeshingScope,
        adjacent_region_names: tuple[str, ...],
        /,
        *,
        required: bool = True,
        recipe_binding: RecipeSelectionBinding | None = None,
    ) -> None:
        value = str(name).strip()
        if not value:
            raise ValueError(f"PatchControl.name must be non-empty; got {value!r}.")
        if isinstance(adjacent_region_names, str):
            raise TypeError(
                f"PatchControl.adjacent_region_names must be an iterable of region names; got {type(adjacent_region_names).__name__}."
            )
        regions = tuple(sorted(str(region).strip() for region in adjacent_region_names))
        if len(regions) not in (1, 2):
            raise ValueError(
                f"PatchControl.adjacent_region_names must contain one or two adjacent region names; got {regions!r}."
            )
        if any(not region for region in regions) or len(set(regions)) != len(regions):
            raise ValueError(
                f"PatchControl.adjacent_region_names must contain distinct non-empty adjacent region names; got {regions!r}."
            )
        self.name = value
        self.scope = scope
        self.adjacent_region_names = regions
        self.required = bool(required)
        if recipe_binding is not None and scope.entity_dimension != (
            2 if recipe_binding.kind == "face" else 1
        ):
            raise ValueError(
                "PatchControl.recipe_binding must match the scope dimension; got a different entity family"
            )
        self.recipe_binding = recipe_binding
        self.control_id = canonical_fingerprint(
            {
                "kind": "patch-control",
                "name": value,
                "scope": scope.scope_id,
                "adjacent_region_names": regions,
                "required": bool(required),
            }
        )
        if recipe_binding is not None:
            self.control_id = canonical_fingerprint(
                {
                    "control": self.control_id,
                    "recipe_membership": tuple(
                        sorted(name.key for name in recipe_binding.names)
                    ),
                    "recipe_inventory": tuple(
                        sorted(name.key for name in recipe_binding.inventory)
                    ),
                }
            )

    @checked
    def rebind(self, names: RecipeEntityNames, domain: MeshingDomain, /) -> PatchControl:
        """Resolve stable recipe membership on a topology-preserving realization."""
        if self.recipe_binding is None:
            raise ValueError(
                "PatchControl.rebind must have a recipe-bound selection; got absent recipe binding"
            )
        return PatchControl(
            self.name,
            self.recipe_binding.scope(names, domain),
            self.adjacent_region_names,
            required=self.required,
            recipe_binding=self.recipe_binding,
        )


class LayerSchedule(StrictModule, NonTrainableState):
    """Store layer thicknesses in outward order from the wall."""

    thicknesses: tuple[float, ...] = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(self, thicknesses: ArrayLike, /) -> None:
        values = np.asarray(thicknesses)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(
                f"LayerSchedule.thicknesses must be one non-empty vector; got shape={values.shape}."
            )
        if not (
            np.issubdtype(values.dtype, np.integer)
            or np.issubdtype(values.dtype, np.floating)
        ):
            raise TypeError(
                f"LayerSchedule.thicknesses must contain real numeric values; got dtype={values.dtype}."
            )
        normalized = values.astype("float64", copy=False)
        if not np.all(np.isfinite(normalized)) or np.any(normalized <= 0.0):
            raise ValueError(
                f"LayerSchedule.thicknesses must be finite and strictly positive; got {normalized!r}."
            )
        explicit = tuple(float(value) for value in normalized)
        self.thicknesses = explicit
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "layer-schedule",
                "thicknesses": explicit,
            }
        )

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
            raise TypeError(
                f"LayerSchedule.layer_count must be an integer; got shape={count_value.shape}, dtype={count_value.dtype}."
            )
        count = int(count_value)
        if count <= 0:
            raise ValueError(
                f"LayerSchedule.layer_count must be positive; got {count!r}."
            )
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
                f"LayerSchedule.first_layer_thickness must be a real numeric scalar together with growth_rate; got first_shape={first_value.shape}, first_dtype={first_value.dtype}, growth_shape={growth_value.shape}, growth_dtype={growth_value.dtype}."
            )
        first = float(first_value)
        growth = float(growth_value)
        if not np.isfinite(first) or first <= 0.0:
            raise ValueError(
                f"LayerSchedule.first_layer_thickness must be positive and finite; got {first!r}."
            )
        if not np.isfinite(growth) or growth <= 0.0:
            raise ValueError(
                f"LayerSchedule.growth_rate must be positive and finite; got {growth!r}."
            )
        # ty: ignore[invalid-argument-type]
        return cls(tuple(first * growth**index for index in range(count)))

    @property
    def layer_count(self) -> int:
        return len(self.thicknesses)

    @property
    def total_thickness(self) -> float:
        return float(sum(self.thicknesses))

    @property
    def growth_rates(self) -> tuple[float, ...]:
        """Return consecutive outward thickness ratios ``t[k + 1] / t[k]``."""
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
            message = "use wall and cap face scopes and a volume scope for EXACT_SWEEP"
        case BoundaryLayerRoute.CAD_EXTRUSION:
            valid = (wall, cap, volume) == (2, None, 3)
            message = (
                "use a wall face scope, a volume scope, and no cap for CAD_EXTRUSION"
            )
        case BoundaryLayerRoute.ADVANCING:
            valid = (wall, cap, volume) == (2, None, 3 if geometry else None)
            message = "use a wall face scope and no cap for ADVANCING, with a volume scope for geometry-bound walls"
        case BoundaryLayerRoute.PROVIDER:
            valid = (wall, cap, volume) in ((1, None, None), (2, None, 3))
            message = "use wall curves without a volume or wall faces with a volume scope for PROVIDER, with no cap"
        case _:
            raise TypeError(
                f"BoundaryLayerControl.route must be BoundaryLayerRoute; got {type(route).__name__}."
            )
    if not valid:
        raise ValueError(
            f"BoundaryLayerControl.scopes must {message}; got wall_dimension={wall}, "
            f"cap_dimension={cap}, volume_dimension={volume}, geometry_bound={geometry}."
        )
    if cap_scope is not None:
        if cap_scope.entity_set_id != wall_scope.entity_set_id:
            raise ValueError(
                f"BoundaryLayerControl.wall_scope must share one entity set with cap_scope; got wall={wall_scope.entity_set_id!r}, cap={cap_scope.entity_set_id!r}."
            )
        if np.intersect1d(
            np.asarray(wall_scope.entity_ids), np.asarray(cap_scope.entity_ids)
        ).size:
            raise ValueError(
                "BoundaryLayerControl.wall_scope must be disjoint from cap_scope; got overlapping entity IDs."
            )


class BoundaryLayerControl(StrictModule, NonTrainableState):
    """Declare a boundary-layer request grown from ``wall_scope`` by ``schedule``.

    ``volume_scope`` names the solids receiving the layers and ``cap_scope`` the
    target caps of an ``EXACT_SWEEP``. ``feature_angle`` (radians)
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
        route = parse(route, BoundaryLayerRoute, "BoundaryLayerControl.route")
        collision = parse(
            collision, BoundaryLayerCollisionPolicy, "BoundaryLayerControl.collision"
        )
        corner = parse(corner, BoundaryLayerCornerPolicy, "BoundaryLayerControl.corner")
        binding = (
            wall_scope.source_id,
            wall_scope.source_revision,
            wall_scope.entity_kind,
        )
        if any(
            (scope.source_id, scope.source_revision, scope.entity_kind) != binding
            for scope in scopes[1:]
        ):
            raise ValueError(
                "BoundaryLayerControl.scopes must share one source binding; got differing source identities or revisions."
            )
        _layer_scope_dimensions(route, wall_scope, volume_scope, cap_scope)
        if (
            route is not BoundaryLayerRoute.ADVANCING
            and collision is not BoundaryLayerCollisionPolicy.FAIL
        ):
            raise ValueError(
                f"BoundaryLayerControl.collision must be FAIL outside the ADVANCING route; got route={route.value!r}, collision={collision.value!r}."
            )
        angle = finite_real_scalar(feature_angle, "feature_angle")
        if not 0.0 < angle < np.pi:
            raise ValueError(
                f"BoundaryLayerControl.feature_angle must lie strictly between zero and pi; got {feature_angle!r}."
            )
        fraction = finite_real_scalar(
            minimum_thickness_fraction, "minimum_thickness_fraction"
        )
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                f"BoundaryLayerControl.minimum_thickness_fraction must lie in (0, 1]; got {fraction!r}."
            )
        stretch = finite_real_scalar(maximum_corner_stretch, "maximum_corner_stretch")
        if stretch < 1.0:
            raise ValueError(
                f"BoundaryLayerControl.maximum_corner_stretch must be at least one; got {stretch!r}."
            )
        if isinstance(smoothing_iterations, bool) or not isinstance(
            smoothing_iterations, int
        ):
            raise TypeError(
                f"BoundaryLayerControl.smoothing_iterations must be an integer; got {type(smoothing_iterations).__name__}."
            )
        if smoothing_iterations < 0:
            raise ValueError(
                f"BoundaryLayerControl.smoothing_iterations must be non-negative; got {smoothing_iterations!r}."
            )
        bounds = None
        if growth_rate_bounds is not None:
            if len(growth_rate_bounds) != 2:
                raise ValueError(
                    f"BoundaryLayerControl.growth_rate_bounds must be (lower, upper); got {growth_rate_bounds!r}."
                )
            lower = finite_real_scalar(growth_rate_bounds[0], "growth_rate_bounds")
            upper = finite_real_scalar(growth_rate_bounds[1], "growth_rate_bounds")
            if not 0.0 < lower <= upper:
                raise ValueError(
                    f"BoundaryLayerControl.growth_rate_bounds must satisfy 0 < lower <= upper; got {(lower, upper)!r}."
                )
            if any(not lower <= rate <= upper for rate in schedule.growth_rates):
                raise ValueError(
                    f"BoundaryLayerControl.schedule must satisfy growth_rate_bounds; got rates={schedule.growth_rates!r}, bounds={(lower, upper)!r}."
                )
            bounds = (lower, upper)
        core_size = None
        if core_maximum_size is not None:
            core_size = finite_real_scalar(core_maximum_size, "core_maximum_size")
            if core_size <= 0.0:
                raise ValueError(
                    f"BoundaryLayerControl.core_maximum_size must be positive; got {core_size!r}."
                )
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


class BackgroundMetricMode(StrEnum):
    """Select how a Riemannian metric drives provider background sizing."""

    ISOTROPIC = "isotropic"
    ANISOTROPIC = "anisotropic"


class BackgroundMetricControl(StrictModule, NonTrainableState):
    """Bind metric sizing to an affine simplex background mesh.

    `ISOTROPIC` lowers each vertex metric to its smallest directional size
    `1 / sqrt(lambda_max)`, so no metric direction is under-resolved. `ANISOTROPIC`
    lowers the full tensor. The `MeshMetricField` certifies its declared size
    and anisotropy bounds at construction; this control never repairs a metric.
    """

    mesh: cell_mesh_owner.CellMesh
    metric: MeshMetricField
    coordinate_contract: SpatialCoordinateContract
    mode: BackgroundMetricMode = eqx.field(static=True)
    maximum_metric_edge_length: float | None = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        mesh: cell_mesh_owner.CellMesh,
        metric: MeshMetricField,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        mode: BackgroundMetricMode = BackgroundMetricMode.ANISOTROPIC,
        maximum_metric_edge_length: float | None = None,
    ) -> None:
        mode = parse(mode, BackgroundMetricMode, "BackgroundMetricControl.mode")
        kinds = {block.cell_kind for block in mesh.blocks}
        simplex = {2: "triangle", 3: "tetrahedron"}.get(mesh.topological_dimension)
        if kinds != {simplex}:
            raise ValueError(
                f"BackgroundMetricControl.mesh must contain one affine triangle or tetrahedron family; got families={sorted(kinds)!r}, dimension={mesh.topological_dimension}."
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
                "BackgroundMetricControl.metric must bind every vertex of its mesh revision; got a different source, revision, entity inventory, or selection."
            )
        if metric.values.shape[1:] != (mesh.ambient_dimension, mesh.ambient_dimension):
            raise ValueError(
                f"BackgroundMetricControl.metric.values must match the mesh ambient dimension; got tensor_shape={metric.values.shape[1:]}, ambient_dimension={mesh.ambient_dimension}."
            )
        maximum_length = (
            None
            if maximum_metric_edge_length is None
            else float(maximum_metric_edge_length)
        )
        if maximum_length is not None and (
            not np.isfinite(maximum_length) or maximum_length <= 0.0
        ):
            raise ValueError(
                f"BackgroundMetricControl.maximum_metric_edge_length must be positive and finite; got {maximum_length!r}."
            )
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

    @property
    def vertex_metrics(self) -> np.ndarray:
        """Return metric tensors in CellMesh vertex-row order."""
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
            raise ValueError(
                f"SurfaceReconstructionControl.feature_angle must lie strictly between zero and pi; got {feature_angle!r}."
            )
        if not np.isfinite(curve) or not 0.0 < curve <= np.pi:
            raise ValueError(
                f"SurfaceReconstructionControl.curve_angle must lie in (0, pi]; got {curve!r}."
            )
        if not np.isfinite(deviation) or deviation <= 0.0:
            raise ValueError(
                f"SurfaceReconstructionControl.maximum_deviation must be positive and finite; got {deviation!r}."
            )
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


@final
class TransfiniteCurveControl(StrictModule, NonTrainableState):
    """Store an oriented, trimmed source curve and its interval count."""

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
            raise TypeError(
                f"TransfiniteCurveControl.source_id must be a string and reversed must be bool; got source_type={type(source_id).__name__}, reversed_type={type(reversed).__name__}."
            )
        if not source_id.strip():
            raise ValueError(
                f"TransfiniteCurveControl.source_id must be non-empty; got {source_id!r}."
            )
        if isinstance(intervals, bool) or not isinstance(intervals, int) or intervals < 1:
            raise ValueError(
                f"TransfiniteCurveControl.intervals must be a positive integer; got {intervals!r}."
            )
        bounds = curve.validate_range(*parameter_range)
        self.curve = curve
        self.source_id = source_id
        self.parameter_range = bounds
        self.intervals = intervals
        self.reversed = bool(reversed)
        self.control_id = canonical_fingerprint(
            {
                "kind": "transfinite-curve-control",
                "source": source_id,
                "curve": array_tree_fingerprint(curve),
                "range": bounds,
                "intervals": intervals,
                "reversed": self.reversed,
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
            raise TypeError(
                f"TransfiniteSurfaceControl.source_id must be a string; got {type(source_id).__name__}."
            )
        if any(
            isinstance(axis, bool) or not isinstance(axis, int) for axis in permutation
        ) or any(not isinstance(flip, bool) for flip in flips):
            raise TypeError(
                f"TransfiniteSurfaceControl.permutation must contain integer axes and flips must contain boolean flips; got permutation={permutation!r}, flips={flips!r}."
            )
        if not source_id.strip():
            raise ValueError(
                f"TransfiniteSurfaceControl.source_id must be non-empty; got {source_id!r}."
            )
        if len(intervals) != 2 or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in intervals
        ):
            raise ValueError(
                f"TransfiniteSurfaceControl.intervals must be two positive integers; got {intervals!r}."
            )
        if sorted(permutation) != [0, 1] or len(flips) != 2:
            raise ValueError(
                f"TransfiniteSurfaceControl.permutation must contain an axis permutation and flips must contain two flips; got permutation={permutation!r}, flips={flips!r}."
            )
        box = surface.validate_parameter_box(parameter_box)
        self.surface = surface
        self.parameter_box = jax.device_put(np.asarray(box, dtype=np.float64))
        self.source_id = source_id
        self.intervals = intervals
        self.permutation = permutation
        self.flips = (flips[0], flips[1])
        self.control_id = canonical_fingerprint(
            {
                "kind": "transfinite-surface-control",
                "source": source_id,
                "surface": array_tree_fingerprint(surface),
                "box": box,
                "intervals": intervals,
                "permutation": permutation,
                "flips": self.flips,
            }
        )


@final
class BlockInterfaceControl(StrictModule, NonTrainableState):
    """Declare logical face gluing between two blocks.

    Faces are ``2 * axis + side``. The permutation transposes the second face
    into first-face axes; flips then reverse those axes. ``conforming=False``
    leaves both carriers independent and requires a declared MeshCoupling.
    """

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
            raise TypeError(
                f"BlockInterfaceControl.first_block must be a string together with second_block; got first_type={type(first_block).__name__}, second_type={type(second_block).__name__}."
            )
        if not isinstance(conforming, bool) or any(
            not isinstance(flip, bool) for flip in flips
        ):
            raise TypeError(
                f"BlockInterfaceControl.conforming must be boolean together with each flip; got conforming={conforming!r}, flips={flips!r}."
            )
        if any(
            isinstance(axis, bool) or not isinstance(axis, int) for axis in permutation
        ):
            raise TypeError(
                f"BlockInterfaceControl.permutation must contain integer axes; got {permutation!r}."
            )
        if dimension not in (1, 2) or sorted(permutation) != list(range(dimension)):
            raise ValueError(
                f"BlockInterfaceControl.permutation must contain each face axis once; got {permutation!r}."
            )
        if len(flips) != dimension:
            raise ValueError(
                f"BlockInterfaceControl.flips must match the face dimension; got flips={flips!r}, dimension={dimension}."
            )
        if not first_block.strip() or not second_block.strip():
            raise ValueError(
                f"BlockInterfaceControl.first_block must be non-empty together with second_block; got first={first_block!r}, second={second_block!r}."
            )
        if any(
            isinstance(face, bool)
            or not isinstance(face, int)
            or not 0 <= face < 2 * (dimension + 1)
            for face in (first_face, second_face)
        ):
            raise ValueError(
                f"BlockInterfaceControl.first_face must be inside the logical block together with second_face; got first={first_face!r}, second={second_face!r}, dimension={dimension}."
            )
        tolerance_ = finite_real_scalar(tolerance, "tolerance")
        if tolerance_ < 0.0:
            raise ValueError(
                f"BlockInterfaceControl.tolerance must be nonnegative; got {tolerance_!r}."
            )
        self.first_block = first_block
        self.second_block = second_block
        self.first_face = first_face
        self.second_face = second_face
        self.permutation = permutation
        self.flips = tuple(bool(value) for value in flips)
        self.conforming = bool(conforming)
        self.tolerance = tolerance_
        self.control_id = canonical_fingerprint(
            {
                "kind": "block-interface-control",
                "first": (first_block, first_face),
                "second": (second_block, second_face),
                "permutation": permutation,
                "flips": self.flips,
                "conforming": self.conforming,
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
    "ProtectedFeature",
    "RegionControl",
    "RegionSeed",
    "SurfaceReconstructionControl",
    "TransfiniteCurveControl",
    "TransfiniteSurfaceControl",
    "BlockInterfaceControl",
]
