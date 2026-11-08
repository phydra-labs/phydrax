#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Source-chart geometry for surface finite elements.

The carrier coordinate map remains the mesh's independently certified affine or
curved realization.  This object binds its cells to source charts and evaluates
the source immersion and its Jacobian, rather than projecting the carrier metric.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry._source_array_identity import source_arrays_equal
from ...geometry.brep._carrier_pieces import carrier_content_id
from ...geometry.brep._patches import (
    AbstractSurfacePatch,
    PlanePatch,
    surface_differential,
)
from .._cell_geometry import CellGeometrySpec
from .._cell_mesh import CellMesh


if TYPE_CHECKING:
    from ...geometry._brep_capped_material import BRepCappedMaterialMap
    from ...geometry._brep_sphere_material import BRepSphereMaterialMap
    from ...geometry._brep_torus_material import BRepTorusMaterialMap
    from ...geometry._source_boundary_records import SourceBoundaryQuery
    from ...geometry._surface_affine_cell_proof import ExactTriangle
    from ...geometry._surface_source_support import (
        PreparedSurfaceSourceSupport,
        SurfaceRestrictionRows,
    )
    from ...meshing._result import CellMeshingResult


def _require_matching_source_edges(
    mesh: CellMesh,
    patches: tuple[AbstractSurfacePatch, ...],
    rows: SurfaceRestrictionRows,
    /,
) -> None:
    """Prove each admitted interior edge has the same entire source trace.

    Equal affine UV endpoints on one patch give the same UV function on the
    whole edge. Between affine planes, equality of exact endpoint images proves
    equality of the complete affine physical traces. Periodic deck shifts and
    pole charts require their own symbolic/material-map proof; rounded period
    comparisons or coincident vertices are not such a proof.
    """
    cell_rows = {int(cell): row for row, cell in enumerate(rows.cell_ids)}
    incidence: dict[tuple[int, int], list[tuple[int, tuple, tuple]]] = {}
    for block in mesh.blocks:
        for cell_id, vertices in zip(
            np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
        ):
            row = cell_rows[int(cell_id)]
            corners = rows.source_corners[row]
            patch_index = int(rows.patches[row])
            for a, b in ((0, 1), (1, 2), (2, 0)):
                first, last = int(vertices[a]), int(vertices[b])
                uv_first, uv_last = corners[a], corners[b]
                if first > last:
                    first, last = last, first
                    uv_first, uv_last = uv_last, uv_first
                incidence.setdefault((first, last), []).append(
                    (patch_index, uv_first, uv_last)
                )

    def plane_point(patch: PlanePatch, uv: tuple) -> tuple[Fraction, ...]:
        origin = np.asarray(patch.origin, dtype=np.float64)
        first = np.asarray(patch.first_axis, dtype=np.float64)
        second = np.asarray(patch.second_axis, dtype=np.float64)
        return tuple(
            Fraction.from_float(float(origin[i]))
            + uv[0] * Fraction.from_float(float(first[i]))
            + uv[1] * Fraction.from_float(float(second[i]))
            for i in range(3)
        )

    for uses in incidence.values():
        if len(uses) == 1:
            continue
        if len(uses) != 2:
            raise ValueError("Source surface FEM requires manifold edge incidence.")
        (first_patch, first_a, first_b), (second_patch, second_a, second_b) = uses
        if first_patch == second_patch and first_a == second_a and first_b == second_b:
            continue
        left, right = patches[first_patch], patches[second_patch]
        if (
            isinstance(left, PlanePatch)
            and isinstance(right, PlanePatch)
            and plane_point(left, first_a) == plane_point(right, second_a)
            and plane_point(left, first_b) == plane_point(right, second_b)
        ):
            continue
        raise ValueError(
            "Source surface FEM requires a proved pointwise matching source chart "
            "on every interior edge; periodic, pole, and curved cross-patch "
            "edges require a conforming material chart."
        )


class SurfaceFiniteElementGeometry(StrictModule, NonTrainableState):
    """Fixed source charts for the cells of an independently retained carrier."""

    patches: tuple[AbstractSurfacePatch, ...]
    material_map: (
        BRepSphereMaterialMap | BRepTorusMaterialMap | BRepCappedMaterialMap | None
    )
    parameters: tuple[Array, ...]
    exact_parameters: tuple[tuple[ExactTriangle, ...], ...] = eqx.field(static=True)
    parameter_rounding_bounds: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    groups: tuple[tuple[tuple[int, Array], ...], ...]
    source_reference_arrays: tuple[Array, ...]
    source_static_values: tuple[Any, ...] = eqx.field(static=True)
    source_tree: Any = eqx.field(static=True)
    carrier_coordinates: Array
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    cell_global_ids: tuple[int, ...] = eqx.field(static=True)
    binding_identity: tuple[Any, ...] = eqx.field(static=True)
    source_root_cell_ids: tuple[int, ...] = eqx.field(static=True)

    @classmethod
    def from_result(
        cls: type[SurfaceFiniteElementGeometry],
        result: CellMeshingResult,
        source_query: SourceBoundaryQuery,
        /,
    ) -> SurfaceFiniteElementGeometry:
        """Bind the certified result and its source query to exact target charts."""
        from ...geometry._brep_capped_material import BRepCappedMaterialMap
        from ...geometry._brep_sphere_material import BRepSphereMaterialMap
        from ...geometry._brep_torus_material import BRepTorusMaterialMap
        from ...geometry._domain_boundary import (
            exact_plane_chart,
            MeshingDomainBoundarySource,
        )
        from ...geometry._meshing_domain import PatchPoleUse
        from ...geometry._surface_source_support import (
            PreparedSurfaceSourceSupport,
            SurfaceNativeRestrictionBoundarySource,
        )
        from ...geometry.brep._patches import RevolutionSurface
        from ...geometry.brep._source_embedding import _family
        from ...meshing._surface_boundary_sources import (
            SurfaceChartBoundarySource,
            SurfaceSubdivisionBoundarySource,
        )

        charts = result.surface_source
        if charts is None:
            raise ValueError("Source surface FEM requires retained source charts.")
        charts.require_bound(result.mesh, result.geometry, source_query)
        if isinstance(source_query, MeshingDomainBoundarySource):
            charts.require_root(result.mesh, result.geometry)
            support = PreparedSurfaceSourceSupport(
                charts.domain, result, charts.root_parameters, charts.boundary_source
            )
        elif isinstance(
            source_query,
            (
                SurfaceNativeRestrictionBoundarySource,
                SurfaceSubdivisionBoundarySource,
                SurfaceChartBoundarySource,
            ),
        ):
            support = source_query.support
        else:
            raise TypeError(
                "Source surface FEM requires a supported source boundary query."
            )
        material_map = None
        model = charts.domain.brep_authority
        if model is not None and len(model.patches) == 1:
            surface = model.patches[0]
            if (
                isinstance(surface, RevolutionSurface)
                and surface.exact_sphere_revolution_witness() is not None
            ):
                material_map = BRepSphereMaterialMap(result, source_query)
            elif (
                isinstance(surface, RevolutionSurface)
                and _family(surface) == "ring-torus"
            ):
                material_map = BRepTorusMaterialMap(result, source_query)
        elif model is not None and len(charts.domain.patches) == 3:
            planes = tuple(
                exact_plane_chart(charts.domain, index)
                for index in range(len(charts.domain.patches))
            )
            plane_rows = tuple(
                index for index, plane in enumerate(planes) if plane is not None
            )
            side_rows = tuple(
                index
                for index, patch in enumerate(charts.domain.patches)
                if _family(patch.surface) == "revolution-cylinder"
            )
            if (
                len(plane_rows) == 2
                and len(side_rows) == 1
                and all(
                    plane is not None
                    and len(plane.loops) == 1
                    and len(plane.loops[0]) == 1
                    and sum(
                        isinstance(use, PatchPoleUse)
                        for loop in charts.domain.patches[index].loops
                        for use in loop
                    )
                    == 1
                    for index in plane_rows
                    for plane in (planes[index],)
                )
            ):
                material_map = BRepCappedMaterialMap(result, source_query)
        return cls(result.mesh, result.geometry, support, material_map=material_map)

    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        support: PreparedSurfaceSourceSupport,
        /,
        *,
        material_map: BRepSphereMaterialMap
        | BRepTorusMaterialMap
        | BRepCappedMaterialMap
        | None = None,
    ) -> None:
        from ...geometry._brep_capped_material import BRepCappedMaterialMap
        from ...geometry._brep_sphere_material import BRepSphereMaterialMap
        from ...geometry._brep_torus_material import BRepTorusMaterialMap
        from ...geometry._rational import fraction_up
        from ...geometry._surface_affine_cell_proof import certify_affine_surface_cell
        from ...geometry._surface_source_support import PreparedSurfaceSourceSupport
        from ...meshing._result import CellMeshingResult

        if not isinstance(support, PreparedSurfaceSourceSupport):
            raise TypeError("Source surface FEM requires proved source restrictions.")
        rows = support.prove_native_restrictions(mesh, geometry)
        patches = tuple(patch.surface for patch in support.domain.patches)

        if mesh.topological_dimension != 2 or mesh.ambient_dimension != 3:
            raise ValueError("Source surface FEM requires a surface in three-space.")
        if any(block.cell_kind != "triangle" for block in mesh.blocks):
            raise ValueError("Source surface FEM requires triangular source charts.")
        if material_map is None:
            for patch_index, corners in zip(
                rows.patches, rows.source_corners, strict=True
            ):
                if (
                    len(corners) != 3
                    or certify_affine_surface_cell(
                        patches[int(patch_index)], (corners[0], corners[1], corners[2])
                    )
                    is None
                ):
                    raise ValueError(
                        "Source surface FEM requires a proved regular embedded source chart "
                        "for every cell."
                    )
            _require_matching_source_edges(mesh, patches, rows)
        elif (
            not isinstance(
                material_map,
                (BRepSphereMaterialMap, BRepTorusMaterialMap, BRepCappedMaterialMap),
            )
            or material_map.topology_id != mesh.topology_id
            or material_map.geometry_layout_id != geometry.geometry_layout_id
            or material_map.source_id != support.domain.source_id
            or material_map.source_revision != support.domain.source_revision
            or not isinstance(support.original, CellMeshingResult)
            or support.original.surface_source is None
            or material_map.source_chart_id
            != support.original.surface_source.source_chart_id
        ):
            raise ValueError("Material map must bind the same source and carrier.")
        lookup = {int(cell): row for row, cell in enumerate(rows.cell_ids)}
        if len(lookup) != len(rows.cell_ids):
            raise ValueError("Source surface chart cell identities must be unique.")
        parameters = []
        exact_parameters: list[tuple[ExactTriangle, ...]] = []
        rounding_bounds = []
        groups = []
        for block in mesh.blocks:
            indices = [lookup[int(cell)] for cell in np.asarray(block.global_ids)]
            raw = tuple(rows.source_corners[i] for i in indices)
            if any(len(cell) != 3 for cell in raw):
                raise ValueError(
                    "Source surface charts require finite triangle UV corners."
                )
            exact = tuple((cell[0], cell[1], cell[2]) for cell in raw)
            corners = np.asarray(exact, dtype=np.float64)
            if corners.shape != (block.cell_count, 3, 2) or not np.all(
                np.isfinite(corners)
            ):
                raise ValueError(
                    "Source surface charts require finite triangle UV corners."
                )
            patch_indices = np.asarray(rows.patches, dtype=np.int64)[indices]
            if np.any(patch_indices < 0) or np.any(patch_indices >= len(patches)):
                raise ValueError("Source surface charts reference an unknown patch.")
            exact_parameters.append(exact)
            rounding_bounds.append(
                tuple(
                    fraction_up(
                        max(
                            abs(value - Fraction.from_float(float(value)))
                            for point in cell
                            for value in point
                        )
                    )
                    for cell in exact
                )
            )
            parameters.append(jnp.asarray(corners))
            groups.append(
                tuple(
                    (
                        int(p),
                        jnp.asarray(np.flatnonzero(patch_indices == p), dtype=jnp.int32),
                    )
                    for p in np.unique(patch_indices)
                )
            )
        self.patches = tuple(patches)
        self.material_map = material_map
        self.parameters = tuple(parameters)
        self.exact_parameters = tuple(exact_parameters)
        self.parameter_rounding_bounds = tuple(rounding_bounds)
        self.groups = tuple(groups)
        self.carrier_coordinates = jnp.asarray(geometry.coordinates)
        leaves, self.source_tree = jax.tree.flatten(self._guard_view())
        self.source_reference_arrays = tuple(
            jnp.asarray(leaf) for leaf in leaves if eqx.is_array(leaf)
        )
        self.source_static_values = tuple(
            leaf for leaf in leaves if not eqx.is_array(leaf)
        )
        self.topology_id = mesh.topology_id
        self.geometry_layout_id = geometry.geometry_layout_id
        self.source_id = support.domain.source_id
        self.source_revision = support.domain.source_revision
        self.cell_global_ids = tuple(int(value) for value in rows.cell_ids)
        self.source_root_cell_ids = tuple(int(value) for value in rows.root_cell_ids)
        self.source_geometry_id = canonical_fingerprint(
            {
                "kind": "source-surface-finite-element-geometry",
                "support": support.support_id,
                "topology": mesh.topology_id,
                "carrier": geometry.geometry_layout_id,
                "cells": array_tree_fingerprint(np.asarray(rows.cell_ids)),
                "parameters": [
                    array_tree_fingerprint(np.asarray(value)) for value in parameters
                ],
                "exact_parameters": [
                    [
                        [[str(value) for value in point] for point in cell]
                        for cell in block
                    ]
                    for block in exact_parameters
                ],
                "patch_content": [carrier_content_id(patch) for patch in patches],
                "material_map": None
                if material_map is None
                else material_map.source_geometry_id,
                "patches": array_tree_fingerprint(np.asarray(rows.patches)),
            }
        )
        self.binding_identity = self._binding_identity()

    def evaluate_points(self, block_index: int, reference_points: ArrayLike, /) -> Array:
        """Evaluate source physical positions without forming a Jacobian."""
        return self._evaluate(block_index, reference_points, derivative=False)[0]

    def evaluate(
        self, block_index: int, reference_points: ArrayLike, /
    ) -> tuple[Array, Array]:
        physical, jacobian = self._evaluate(
            block_index, reference_points, derivative=True
        )
        if jacobian is None:
            raise RuntimeError(
                "Source surface geometry did not produce its requested Jacobian."
            )
        return physical, jacobian

    def _binding_identity(self, /) -> tuple[Any, ...]:
        exact_id = canonical_fingerprint(
            {
                "parameters": [
                    [
                        [[str(value) for value in point] for point in cell]
                        for cell in block
                    ]
                    for block in self.exact_parameters
                ],
                "rounding_bounds": self.parameter_rounding_bounds,
            }
        )
        return (
            self.topology_id,
            self.geometry_layout_id,
            self.source_geometry_id,
            self.source_id,
            self.source_revision,
            self.cell_global_ids,
            self.source_root_cell_ids,
            exact_id,
        )

    def _guard_view(self, /) -> tuple[Any, ...]:
        from ...geometry._brep_capped_material import BRepCappedMaterialMap

        material: Any = self.material_map
        if isinstance(material, BRepCappedMaterialMap):
            material = (
                "capped-source-guard",
                jax.tree.structure(material),
                material.topology_id,
                material.geometry_layout_id,
                material.source_geometry_id,
                material.source_id,
                material.source_revision,
                material.source_chart_id,
                material.guard_baselines,
            )
        return (
            self.patches,
            self.parameters,
            self.groups,
            self.carrier_coordinates,
            material,
        )

    def source_changed(self, /) -> Array:
        """Return the dynamic currentness predicate for the bound source proof."""
        from ...geometry._brep_capped_material import BRepCappedMaterialMap

        if self.binding_identity != self._binding_identity():
            raise ValueError("Source surface geometry binding changed after proof.")

        leaves, source_tree = jax.tree.flatten(self._guard_view())
        if source_tree != self.source_tree:
            raise ValueError("Source surface geometry structure changed after proof.")
        current_arrays = tuple(leaf for leaf in leaves if eqx.is_array(leaf))
        static_values = tuple(leaf for leaf in leaves if not eqx.is_array(leaf))
        if static_values != self.source_static_values or len(current_arrays) != len(
            self.source_reference_arrays
        ):
            raise ValueError("Source surface geometry metadata changed after proof.")
        changed = jnp.asarray(False)
        for current, original in zip(
            current_arrays, self.source_reference_arrays, strict=True
        ):
            current = jnp.asarray(current)
            if current.shape != original.shape or current.dtype != original.dtype:
                raise ValueError(
                    "Source surface geometry array layout changed after proof."
                )
            changed = changed | ~source_arrays_equal(current, original)
        if isinstance(self.material_map, BRepCappedMaterialMap):
            changed = changed | self.material_map.source_changed()
        return changed

    def _evaluate(
        self, block_index: int, reference_points: ArrayLike, /, *, derivative: bool
    ) -> tuple[Array, Array | None]:
        """Evaluate source physical positions and derivatives in reference axes."""
        points = jnp.asarray(reference_points)
        changed = self.source_changed()
        if self.material_map is not None:
            if derivative:
                physical, jacobian = self.material_map.evaluate(block_index, points)
                jacobian = eqx.error_if(
                    jacobian, changed, "Source surface geometry changed after proof."
                )
            else:
                physical = self.material_map.evaluate_points(block_index, points)
                jacobian = None
            physical = eqx.error_if(
                physical, changed, "Source surface geometry changed after proof."
            )
            return physical, jacobian
        corners = self.parameters[block_index]
        uv = corners[:, None, 0, :] + ein.contract(
            "qa,cad->cqd", points, corners[:, 1:, :] - corners[:, :1, :]
        )
        chart_jacobian = (
            jnp.swapaxes(corners[:, 1:, :] - corners[:, :1, :], 1, 2)
            if derivative
            else None
        )
        physical = jnp.zeros((corners.shape[0], points.shape[0], 3), dtype=uv.dtype)
        jacobian = (
            jnp.zeros((corners.shape[0], points.shape[0], 3, 2), dtype=uv.dtype)
            if derivative
            else None
        )
        for patch_index, index in self.groups[block_index]:
            patch = self.patches[patch_index]
            local_uv = uv[index]
            physical = physical.at[index].set(patch.evaluate(local_uv))
            if jacobian is not None and chart_jacobian is not None:
                differential = surface_differential(patch, local_uv)
                jacobian = jacobian.at[index].set(
                    ein.contract("cqdi,cij->cqdj", differential, chart_jacobian[index])
                )
        physical = eqx.error_if(
            physical, changed, "Source surface geometry changed after proof."
        )
        if jacobian is not None:
            jacobian = eqx.error_if(
                jacobian, changed, "Source surface geometry changed after proof."
            )
        return physical, jacobian

    def require_carrier(self, coordinates: ArrayLike, /) -> Array:
        points = jnp.asarray(coordinates)
        if points.shape != self.carrier_coordinates.shape:
            raise ValueError(
                "Source surface geometry requires its bound carrier coordinates."
            )
        if points.dtype != self.carrier_coordinates.dtype:
            raise ValueError("Source surface geometry requires the bound carrier dtype.")
        return eqx.error_if(
            points,
            ~source_arrays_equal(points, self.carrier_coordinates),
            "Source surface geometry cannot be reused after carrier coordinates change.",
        )
