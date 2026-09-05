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
from ._scope import MeshingScope


class FeatureKind(StrEnum):
    CORNER = "corner"
    CURVE = "curve"
    SURFACE = "surface"
    MATERIAL_INTERFACE = "material_interface"


class RegionRole(StrEnum):
    FLUID = "fluid"
    SOLID = "solid"
    VOID = "void"
    POROUS = "porous"
    USER = "user"


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
    role: RegionRole = eqx.field(static=True)
    seed_id: str = eqx.field(static=True)

    def __init__(
        self,
        point: ArrayLike,
        region_name: str,
        material_id: str,
        role: RegionRole,
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
        if not isinstance(role, RegionRole):
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


class VolumeRegionControl(StrictModule, NonTrainableState):
    scope: MeshingScope
    region_name: str = eqx.field(static=True)
    material_id: str = eqx.field(static=True)
    role: RegionRole = eqx.field(static=True)
    meshing_enabled: bool = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    def __init__(
        self,
        scope: MeshingScope,
        region_name: str,
        material_id: str,
        role: RegionRole,
        /,
        *,
        meshing_enabled: bool = True,
    ):
        if not isinstance(scope, MeshingScope):
            raise TypeError("scope must be MeshingScope.")
        region = str(region_name).strip()
        material = str(material_id).strip()
        if not region or not material:
            raise ValueError("Volume region identities must be non-empty.")
        if not isinstance(role, RegionRole):
            raise TypeError("role must be RegionRole.")
        self.scope = scope
        self.region_name = region
        self.material_id = material
        self.role = role
        self.meshing_enabled = bool(meshing_enabled)
        self.control_id = canonical_fingerprint(
            {
                "kind": "volume-region-control",
                "scope": scope.scope_id,
                "region_name": region,
                "material_id": material,
                "role": role.value,
                "meshing_enabled": bool(meshing_enabled),
            }
        )


def _layers(
    layer_count: int,
    first_layer_thickness: float,
    growth_rate: float,
    /,
) -> tuple[int, float, float]:
    count = int(layer_count)
    first = float(first_layer_thickness)
    growth = float(growth_rate)
    if count <= 0:
        raise ValueError("layer_count must be positive.")
    if not np.isfinite(first) or first <= 0.0:
        raise ValueError("first_layer_thickness must be positive and finite.")
    if not np.isfinite(growth) or growth < 1.0:
        raise ValueError("growth_rate must be finite and at least one.")
    return count, first, growth


class PrismLayerControl(StrictModule, NonTrainableState):
    surface_scope: MeshingScope
    volume_scope: MeshingScope
    layer_count: int = eqx.field(static=True)
    first_layer_thickness: float = eqx.field(static=True)
    growth_rate: float = eqx.field(static=True)
    termination: LayerTerminationPolicy = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface_scope: MeshingScope,
        volume_scope: MeshingScope,
        layer_count: int,
        first_layer_thickness: float,
        /,
        *,
        growth_rate: float = 1.2,
        termination: LayerTerminationPolicy = LayerTerminationPolicy.REJECT,
    ):
        if not isinstance(surface_scope, MeshingScope) or not isinstance(
            volume_scope, MeshingScope
        ):
            raise TypeError("Prism layer scopes must be MeshingScope values.")
        if surface_scope.source_revision != volume_scope.source_revision:
            raise ValueError("Prism layer scopes must share one source revision.")
        if not isinstance(termination, LayerTerminationPolicy):
            raise TypeError("termination must be LayerTerminationPolicy.")
        count, first, growth = _layers(layer_count, first_layer_thickness, growth_rate)
        self.surface_scope = surface_scope
        self.volume_scope = volume_scope
        self.layer_count = count
        self.first_layer_thickness = first
        self.growth_rate = growth
        self.termination = termination
        self.control_id = canonical_fingerprint(
            {
                "kind": "prism-layer-control",
                "surface_scope": surface_scope.scope_id,
                "volume_scope": volume_scope.scope_id,
                "layers": [count, first, growth],
                "termination": termination.value,
            }
        )


class ShellLayerControl(StrictModule, NonTrainableState):
    edge_scope: MeshingScope
    surface_scope: MeshingScope
    layer_count: int = eqx.field(static=True)
    first_layer_thickness: float = eqx.field(static=True)
    growth_rate: float = eqx.field(static=True)
    require_quadrilaterals: bool = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    def __init__(
        self,
        edge_scope: MeshingScope,
        surface_scope: MeshingScope,
        layer_count: int,
        first_layer_thickness: float,
        /,
        *,
        growth_rate: float = 1.2,
        require_quadrilaterals: bool = True,
    ):
        if not isinstance(edge_scope, MeshingScope) or not isinstance(
            surface_scope, MeshingScope
        ):
            raise TypeError("Shell layer scopes must be MeshingScope values.")
        if edge_scope.source_revision != surface_scope.source_revision:
            raise ValueError("Shell layer scopes must share one source revision.")
        count, first, growth = _layers(layer_count, first_layer_thickness, growth_rate)
        self.edge_scope = edge_scope
        self.surface_scope = surface_scope
        self.layer_count = count
        self.first_layer_thickness = first
        self.growth_rate = growth
        self.require_quadrilaterals = bool(require_quadrilaterals)
        self.control_id = canonical_fingerprint(
            {
                "kind": "shell-layer-control",
                "edge_scope": edge_scope.scope_id,
                "surface_scope": surface_scope.scope_id,
                "layers": [count, first, growth],
                "require_quadrilaterals": bool(require_quadrilaterals),
            }
        )


class ThinRegionLayerControl(StrictModule, NonTrainableState):
    source_scope: MeshingScope
    target_scope: MeshingScope
    volume_scope: MeshingScope
    layer_count: int = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_scope: MeshingScope,
        target_scope: MeshingScope,
        volume_scope: MeshingScope,
        layer_count: int,
        /,
    ):
        if not all(
            isinstance(scope, MeshingScope)
            for scope in (source_scope, target_scope, volume_scope)
        ):
            raise TypeError("Thin-region scopes must be MeshingScope values.")
        if (
            len(
                {
                    scope.source_revision
                    for scope in (source_scope, target_scope, volume_scope)
                }
            )
            != 1
        ):
            raise ValueError("Thin-region scopes must share one source revision.")
        count = int(layer_count)
        if count <= 0:
            raise ValueError("layer_count must be positive.")
        self.source_scope = source_scope
        self.target_scope = target_scope
        self.volume_scope = volume_scope
        self.layer_count = count
        self.control_id = canonical_fingerprint(
            {
                "kind": "thin-region-layer-control",
                "source_scope": source_scope.scope_id,
                "target_scope": target_scope.scope_id,
                "volume_scope": volume_scope.scope_id,
                "layer_count": count,
            }
        )


LayerControl = PrismLayerControl | ShellLayerControl | ThinRegionLayerControl


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
    "LayerControl",
    "LayerTerminationPolicy",
    "PeriodicConstraint",
    "PrismLayerControl",
    "ProtectedFeature",
    "RegionRole",
    "RegionSeed",
    "ShellLayerControl",
    "ThinRegionLayerControl",
    "VolumeRegionControl",
]
