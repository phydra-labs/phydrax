#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from . import _organization
from ._scope import MeshingScope


class FeatureKind(StrEnum):
    CORNER = "corner"
    CURVE = "curve"
    SURFACE = "surface"
    MATERIAL_INTERFACE = "material_interface"


class LayerTerminationPolicy(StrEnum):
    TRUNCATE = "truncate"
    COLLAPSE = "collapse"
    REJECT = "reject"


class ProtectedFeature(StrictModule, NonTrainableState):
    scope: MeshingScope
    feature_kind: FeatureKind = eqx.field(static=True)
    maximum_deviation: float = eqx.field(static=True)
    hard: bool = eqx.field(static=True)
    feature_id: str = eqx.field(static=True)

    def __init__(
        self,
        scope: MeshingScope,
        feature_kind: FeatureKind,
        /,
        *,
        maximum_deviation: float = 0.0,
        hard: bool = True,
    ):
        if not isinstance(scope, MeshingScope):
            raise TypeError("scope must be MeshingScope.")
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
    ):
        coordinates = np.asarray(point, dtype=float)
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


class HoleSeed(StrictModule, NonTrainableState):
    point: Array
    scope: MeshingScope
    seed_id: str = eqx.field(static=True)

    def __init__(self, point: ArrayLike, scope: MeshingScope, /):
        coordinates = np.asarray(point, dtype=float)
        if (
            coordinates.ndim != 1
            or coordinates.size == 0
            or not np.all(np.isfinite(coordinates))
        ):
            raise ValueError("Hole seed point must be one finite coordinate vector.")
        if not isinstance(scope, MeshingScope):
            raise TypeError("scope must be MeshingScope.")
        self.point = jnp.asarray(coordinates)
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
    role: _organization.RegionRole = eqx.field(static=True)
    meshing_enabled: bool = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    def __init__(
        self,
        scope: MeshingScope,
        region_name: str,
        material_id: str,
        role: _organization.RegionRole,
        /,
        *,
        meshing_enabled: bool = True,
    ):
        if not isinstance(scope, MeshingScope):
            raise TypeError("scope must be MeshingScope.")
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


class PatchControl(StrictModule, NonTrainableState):
    """Explicit codimension-one boundary or interface request."""

    name: str = eqx.field(static=True)
    scope: MeshingScope
    adjacent_region_names: tuple[str, ...] = eqx.field(static=True)
    required: bool = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        scope: MeshingScope,
        adjacent_region_names: tuple[str, ...],
        /,
        *,
        required: bool = True,
    ):
        value = str(name).strip()
        if not value:
            raise ValueError("Patch control name must be non-empty.")
        if not isinstance(scope, MeshingScope):
            raise TypeError("scope must be MeshingScope.")
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


class LayerSchedule(StrictModule, NonTrainableState):
    """Exact layer thicknesses, ordered from source face to target face."""

    thicknesses: tuple[float, ...] = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(self, thicknesses: ArrayLike, /):
        values = np.asarray(thicknesses)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("thicknesses must be one non-empty vector.")
        if not (
            np.issubdtype(values.dtype, np.integer)
            or np.issubdtype(values.dtype, np.floating)
        ):
            raise TypeError("thicknesses must contain real numeric values.")
        normalized = values.astype(float, copy=False)
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


class SweptLayerControl(StrictModule, NonTrainableState):
    """One complete source-to-target sweep with no implicit layer collapse."""

    source_scope: MeshingScope
    target_scope: MeshingScope
    volume_scope: MeshingScope
    schedule: LayerSchedule
    termination: LayerTerminationPolicy = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_scope: MeshingScope,
        target_scope: MeshingScope,
        volume_scope: MeshingScope,
        schedule: LayerSchedule,
        /,
        *,
        termination: LayerTerminationPolicy = LayerTerminationPolicy.REJECT,
    ):
        scopes = (source_scope, target_scope, volume_scope)
        if not all(isinstance(scope, MeshingScope) for scope in scopes):
            raise TypeError("Swept-layer scopes must be MeshingScope values.")
        if (
            source_scope.entity_dimension != 2
            or target_scope.entity_dimension != 2
            or volume_scope.entity_dimension != 3
        ):
            raise ValueError(
                "Swept layers require source/target face scopes and a volume scope."
            )
        binding = (
            source_scope.source_id,
            source_scope.source_revision,
            source_scope.entity_kind,
        )
        if any(
            (scope.source_id, scope.source_revision, scope.entity_kind) != binding
            for scope in scopes[1:]
        ):
            raise ValueError("Swept-layer scopes must share one source binding.")
        if source_scope.entity_set_id != target_scope.entity_set_id:
            raise ValueError("Swept-layer face scopes must share one entity set.")
        if np.intersect1d(
            np.asarray(source_scope.entity_ids),
            np.asarray(target_scope.entity_ids),
        ).size:
            raise ValueError(
                "Swept-layer source and target face scopes must be disjoint."
            )
        if not isinstance(schedule, LayerSchedule):
            raise TypeError("schedule must be LayerSchedule.")
        if not isinstance(termination, LayerTerminationPolicy):
            raise TypeError("termination must be LayerTerminationPolicy.")
        if termination is not LayerTerminationPolicy.REJECT:
            raise ValueError("Swept layers support only REJECT termination.")
        self.source_scope = source_scope
        self.target_scope = target_scope
        self.volume_scope = volume_scope
        self.schedule = schedule
        self.termination = termination
        self.control_id = canonical_fingerprint(
            {
                "kind": "swept-layer-control",
                "source_scope": source_scope.scope_id,
                "target_scope": target_scope.scope_id,
                "volume_scope": volume_scope.scope_id,
                "schedule": schedule.schedule_id,
                "termination": termination.value,
            }
        )


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
    ):
        if not isinstance(source_scope, MeshingScope) or not isinstance(
            target_scope, MeshingScope
        ):
            raise TypeError("Periodic scopes must be MeshingScope values.")
        if source_scope.source_revision != target_scope.source_revision:
            raise ValueError("Periodic scopes must share one source revision.")
        matrix = np.asarray(transform, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
            raise ValueError("Periodic transform must be one square homogeneous matrix.")
        if not np.all(np.isfinite(matrix)) or not np.allclose(
            matrix[-1],
            np.eye(matrix.shape[0])[-1],
        ):
            raise ValueError("Periodic transform must be finite and homogeneous.")
        if abs(np.linalg.det(matrix[:-1, :-1])) <= np.finfo(float).eps:
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


__all__ = [
    "FeatureKind",
    "HoleSeed",
    "LayerSchedule",
    "LayerTerminationPolicy",
    "PatchControl",
    "PeriodicConstraint",
    "ProtectedFeature",
    "RegionControl",
    "RegionSeed",
    "SweptLayerControl",
]
