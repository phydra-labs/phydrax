#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Source-bound radial material maps for an exact spherical B-Rep shell.

The exact source topology and native pole/seam roots prove that the retained
trim is a full sphere; the projective atlas separately proves the target map.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_complex import PolygonalConnectivity
from ..linalg._exact import determinant_exact
from ._sphere_material_atlas import _require_center_winding_one
from ._volume_material_map import require_polyhedral_carrier_geometry
from .brep._patches import _validate_surface_source_shadows, RevolutionSurface
from .brep._source_shell_admissibility import _single_shell_topology


if TYPE_CHECKING:
    from ..meshing._result import CellMeshingResult
    from ._source_boundary_records import SourceBoundaryQuery

type _Ray = tuple[Fraction, Fraction, Fraction]
type _RayTriangle = tuple[_Ray, _Ray, _Ray]


def _require_full_sphere_source(charts: object, /) -> None:
    """Prove the one-face full revolution from its own exact trim occurrences."""
    from ._meshing_domain import _PlacedSurfacePatch
    from ._surface_source_support import SurfaceSourceCharts
    from .brep._assembly import BRepAffinePlacement
    from .brep._boolean_core import AffinePCurve
    from .brep._carrier_pieces import carrier_content_id
    from .brep._correspondence import _periodic_pcurve_source
    from .brep._intersection import ConicParameterEndpoint, NativePeriodEndpoint
    from .brep._patches import ConicRootLinePCurve
    from .brep._source_embedding import _constant_chart_line, _sheet_sign
    from .brep._trim_helpers import PeriodOffset

    if not isinstance(charts, SurfaceSourceCharts):
        raise ValueError("A full sphere requires retained source chart authority.")
    charts.require_current()
    domain = charts.domain
    model = domain.brep_authority
    if (
        model is None
        or len(model.patches) != 1
        or len(domain.patches) != 1
        or tuple(domain.source_indices[2]) != (0,)
    ):
        raise ValueError("A full sphere requires exactly one retained B-Rep face.")
    model.require_current()
    surface = model.patches[0]
    declared = domain.patches[0].surface
    if isinstance(declared, _PlacedSurfacePatch):
        if (
            declared.placement.identity_payload()
            != BRepAffinePlacement.identity().identity_payload()
        ):
            raise ValueError("A placed sphere needs a proved radial placement transport.")
        declared = declared.source
    if not isinstance(surface, RevolutionSurface) or carrier_content_id(
        declared
    ) != carrier_content_id(surface):
        raise ValueError("The face changes its source revolution occurrence.")
    _validate_surface_source_shadows(surface)
    witness = surface.exact_sphere_revolution_witness()
    if witness is None or not _single_shell_topology(model, domain):
        raise ValueError("The exact spherical source shell is not proved.")
    center = witness[0]
    geometry = model.geometry
    loops = geometry.face_loops[0]
    if len(loops) != 1 or len(loops[0]) != 4:
        raise ValueError(
            "The spherical source trim must have two poles and two seam uses."
        )
    loop = loops[0]
    poles = [
        coedge
        for coedge in loop
        if geometry.edge_curves[geometry.coedge_edges[coedge]] < 0
    ]
    seams = [coedge for coedge in loop if coedge not in poles]
    if len(poles) != 2 or len(seams) != 2:
        raise ValueError("The spherical source trim needs exactly two pole uses.")
    edge = geometry.coedge_edges[seams[0]]
    if (
        geometry.coedge_edges[seams[1]] != edge
        or geometry.coedge_senses[seams[0]] != -geometry.coedge_senses[seams[1]]
    ):
        raise ValueError("The spherical seam needs opposed uses of one edge.")
    curve = geometry.curves[geometry.edge_curves[edge]]
    if carrier_content_id(curve) != carrier_content_id(surface.curve):
        raise ValueError("The spherical seam changes the source meridian.")
    transports = tuple(
        _periodic_pcurve_source(geometry.pcurves[c], surface) for c in seams
    )
    first, second = transports
    if first is None or second is None:
        raise ValueError("The spherical seam lacks native period authority.")
    if (
        {first[1], second[1]} != {(0, 0), (1, 0)}
        or _constant_chart_line(first[0], 0) != (Fraction(0), Fraction(0), Fraction(1))
        or _constant_chart_line(second[0], 0) != (Fraction(0), Fraction(0), Fraction(1))
    ):
        raise ValueError("The spherical seam does not span one exact native turn.")
    endpoints = geometry.edge_endpoint_roots[edge]
    if len(endpoints) != 2:
        raise ValueError("The spherical meridian needs exact conic endpoint roots.")
    start, stop = endpoints
    if not isinstance(start, ConicParameterEndpoint) or not isinstance(
        stop, ConicParameterEndpoint
    ):
        raise ValueError("The spherical meridian needs exact conic endpoint roots.")
    if carrier_content_id(start.root.conic) != carrier_content_id(
        surface.curve
    ) or carrier_content_id(stop.root.conic) != carrier_content_id(surface.curve):
        raise ValueError("The spherical pole roots change their meridian source.")
    a, b = start.root.exact_point(), stop.root.exact_point()
    if any(a[i] + b[i] != 2 * center[i] for i in range(3)):
        raise ValueError("The spherical pole roots are not antipodal.")
    a_low, a_high = start.parameter_enclosure()
    b_low, b_high = stop.parameter_enclosure()
    width = Fraction.from_float(float(b_high)) - Fraction.from_float(float(a_low))
    if (
        not a_high < b_low
        or _sheet_sign(width - PeriodOffset(Fraction(0), Fraction(1))) != -1
    ):
        raise ValueError("The spherical meridian is not a single half-circle.")
    root_ids = set()
    for coedge in poles:
        pcurve = geometry.pcurves[coedge]
        if (
            not isinstance(pcurve, AffinePCurve)
            or pcurve.matrix != ((Fraction(0), Fraction(1)), (Fraction(1), Fraction(0)))
            or pcurve.offset != (Fraction(0), Fraction(0))
            or not isinstance(pcurve.curve, ConicRootLinePCurve)
        ):
            raise ValueError("The pole use lacks its exact rooted native chart.")
        root_line = pcurve.curve
        if (
            root_line.phase_shift_exact != 0
            or root_line.height_origin_exact != 0
            or root_line.height_direction_exact != 1
        ):
            raise ValueError("The pole chart is not its exact native turn.")
        root_ids.add(root_line.root.root_id)
        period_roots = geometry.coedge_endpoint_roots[coedge]
        if len(period_roots) != 2:
            raise ValueError("The pole has no exact complete native period.")
        first_root, last_root = period_roots
        if (
            not isinstance(first_root, NativePeriodEndpoint)
            or not isinstance(last_root, NativePeriodEndpoint)
            or (first_root.rational, first_root.turns, first_root.axis)
            != (Fraction(0), 0, 0)
            or (last_root.rational, last_root.turns, last_root.axis)
            != (Fraction(0), 1, 0)
        ):
            raise ValueError("The pole has no exact complete native period.")
    if root_ids != {start.root.root_id, stop.root.root_id}:
        raise ValueError("The pole occurrences do not bind the meridian endpoints.")


class BRepSphereMaterialMap(StrictModule, NonTrainableState):
    """Exact-projective spherical cells over a current closed B-Rep sphere."""

    center: Array
    radius: Array
    rays: tuple[Array, ...]
    carrier_coordinates: Array
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_chart_id: str = eqx.field(static=True)

    def __init__(
        self, result: CellMeshingResult, source_query: SourceBoundaryQuery, /
    ) -> None:
        charts = result.surface_source
        if charts is None:
            raise ValueError("Spherical material maps need retained source charts.")
        charts.require_bound(result.mesh, result.geometry, source_query)
        _require_full_sphere_source(charts)
        domain = charts.domain
        model = domain.brep_authority
        if model is None or len(model.patches) != 1 or len(domain.patches) != 1:
            raise ValueError(
                "Spherical material maps require one complete source sphere."
            )
        surface = model.patches[0]
        if not isinstance(surface, RevolutionSurface):
            raise ValueError(
                "The source must have an exact spherical revolution witness."
            )
        _validate_surface_source_shadows(surface)
        witness = surface.exact_sphere_revolution_witness()
        if witness is None:
            raise ValueError("The source spherical revolution is not proved.")
        center, radius, _, _ = witness
        if (
            len(center) != 3
            or not isinstance(center[0], Fraction)
            or not isinstance(center[1], Fraction)
            or not isinstance(center[2], Fraction)
        ):
            raise ValueError(
                "Exact rational sphere center is required by the radial atlas."
            )
        center_exact: _Ray = (center[0], center[1], center[2])
        mesh, geometry = result.mesh, result.geometry
        if mesh.topological_dimension != 2 or mesh.ambient_dimension != 3:
            raise ValueError("Spherical material maps require a surface in three-space.")
        if any(block.cell_kind != "triangle" for block in mesh.blocks):
            raise ValueError("Spherical material maps require triangular cells.")
        connectivity = mesh.connectivity
        if not isinstance(connectivity, PolygonalConnectivity):
            raise ValueError("Spherical material maps require triangle incidence.")
        cell_count = sum(block.cell_count for block in mesh.blocks)
        if (
            cell_count == 0
            or int(connectivity.triangle_count) != cell_count
            or np.any(np.asarray(connectivity.boundary_edges, dtype=np.bool_))
            or np.any(np.asarray(connectivity.edge_cell_counts, dtype=np.int32) != 2)
        ):
            raise ValueError("Spherical material maps require a closed two-manifold.")
        vertices = np.concatenate(
            [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
        )
        if np.unique(vertices).size - len(connectivity.edges) + cell_count != 2:
            raise ValueError("Spherical material maps require Euler characteristic two.")
        edge_cells: dict[int, list[tuple[int, int]]] = {}
        cell_edges = np.asarray(connectivity.cell_edges, dtype=np.int64)
        edge_signs = np.asarray(connectivity.cell_edge_signs, dtype=np.int8)
        edge_valid = np.asarray(connectivity.cell_edge_valid, dtype=np.bool_)
        for row in range(cell_count):
            for edge, sign in zip(
                cell_edges[row, edge_valid[row]],
                edge_signs[row, edge_valid[row]],
                strict=True,
            ):
                if int(sign) not in (-1, 1):
                    raise ValueError("Sphere incidence has an invalid orientation.")
                edge_cells.setdefault(int(edge), []).append((row, int(sign)))
        adjacency = [set() for _ in range(cell_count)]
        for uses in edge_cells.values():
            if len(uses) != 2 or uses[0][1] + uses[1][1] != 0:
                raise ValueError("Sphere edges require two oppositely oriented uses.")
            a, b = uses[0][0], uses[1][0]
            adjacency[a].add(b)
            adjacency[b].add(a)
        seen, todo = {0}, [0]
        while todo:
            for neighbor in adjacency[todo.pop()] - seen:
                seen.add(neighbor)
                todo.append(neighbor)
        if len(seen) != cell_count:
            raise ValueError("The spherical material mesh must be connected.")
        require_polyhedral_carrier_geometry(mesh, geometry)
        certification = result.certification
        if certification is None or certification.embedding is None:
            raise ValueError("Spherical material maps need a certified embedded carrier.")
        embedding = certification.embedding
        embedding.binding.require(mesh, geometry)
        if embedding.status != "certified":
            raise ValueError("The spherical carrier embedding is not certified.")
        embedding.validity.require_bound(geometry, mesh=mesh)
        if not embedding.validity.all_certified:
            raise ValueError("The spherical carrier cells are not certified.")
        coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
        vertex_rays: tuple[_Ray, ...] = tuple(
            (
                Fraction.from_float(float(p[0])) - center_exact[0],
                Fraction.from_float(float(p[1])) - center_exact[1],
                Fraction.from_float(float(p[2])) - center_exact[2],
            )
            for p in coordinates
        )
        triangles: tuple[_RayTriangle, ...] = tuple(
            (vertex_rays[int(v[0])], vertex_rays[int(v[1])], vertex_rays[int(v[2])])
            for v in vertices
        )
        signs: list[bool] = []
        for rays in triangles:
            determinant = determinant_exact(
                tuple(tuple(rays[j][i] for j in range(3)) for i in range(3))
            )
            if not determinant:
                raise ValueError("A radial source cell passes through the sphere center.")
            signs.append(determinant > 0)
        if len(set(signs)) != 1:
            raise ValueError("Radial source cells have inconsistent orientations.")
        _require_center_winding_one(triangles)
        ray_arrays: list[Array] = []
        offset = 0
        for block in mesh.blocks:
            local = triangles[offset : offset + block.cell_count]
            ray_arrays.append(
                jnp.asarray([[[float(x) for x in ray] for ray in cell] for cell in local])
            )
            offset += block.cell_count
        self.center = jnp.asarray([float(value) for value in center_exact])
        self.radius = jnp.asarray(float(radius))
        self.rays = tuple(ray_arrays)
        self.carrier_coordinates = jnp.asarray(geometry.coordinates)
        self.topology_id = mesh.topology_id
        self.geometry_layout_id = geometry.geometry_layout_id
        self.source_id = domain.source_id
        self.source_revision = domain.source_revision
        self.source_chart_id = charts.source_chart_id
        self.source_geometry_id = canonical_fingerprint(
            {
                "kind": "brep-sphere-material-map",
                "source": charts.source_chart_id,
                "topology": mesh.topology_id,
                "geometry": geometry.geometry_layout_id,
                "coordinates": array_tree_fingerprint(coordinates),
            }
        )

    def evaluate_points(self, block_index: int, reference_points: ArrayLike, /) -> Array:
        return self._evaluate(block_index, reference_points, derivative=False)[0]

    def evaluate(
        self, block_index: int, reference_points: ArrayLike, /
    ) -> tuple[Array, Array]:
        points, jacobian = self._evaluate(block_index, reference_points, derivative=True)
        if jacobian is None:
            raise RuntimeError(
                "Sphere material derivative evaluation omitted its Jacobian."
            )
        return points, jacobian

    def _evaluate(
        self, block_index: int, reference_points: ArrayLike, /, *, derivative: bool
    ) -> tuple[Array, Array | None]:
        points = jnp.asarray(reference_points)
        rays = self.rays[block_index]
        basis = jnp.concatenate(
            (1 - jnp.sum(points, axis=-1, keepdims=True), points), axis=-1
        )
        vector = ein.contract("qk,ckd->cqd", basis, rays)
        norm = jnp.linalg.norm(vector, axis=-1, keepdims=True)
        direction = vector / norm
        physical = self.center[None, None, :] + self.radius * direction
        if not derivative:
            return physical, None
        gradient = jnp.swapaxes(rays[:, 1:] - rays[:, :1], 1, 2)
        radial_part = ein.contract("cqd,cdi->cqi", direction, gradient)
        jacobian = (
            self.radius
            * (
                gradient[:, None, :, :]
                - direction[..., :, None] * radial_part[..., None, :]
            )
            / norm[..., None]
        )
        return physical, jacobian
