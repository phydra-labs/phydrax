#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact-source torus material maps by certified normal-fiber projection."""

from __future__ import annotations

from fractions import Fraction
from itertools import count
from math import gcd, isqrt
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import phydrax.ein as ein
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_complex import PolygonalConnectivity
from ..linalg._exact import solve_exact
from ._interval_enclosure import (
    interval_add,
    interval_divide,
    interval_multiply,
    interval_sqrt,
    interval_square,
    interval_subtract,
)
from ._rational import fraction_down, fraction_up, rational_vector
from ._volume_material_map import require_polyhedral_carrier_geometry
from .brep._patches import (
    CircleCurve,
    RevolutionSurface,
    _validate_surface_source_shadows,
)
from .brep._source_embedding import _family
from .brep._source_shell_admissibility import _single_shell_topology

if TYPE_CHECKING:
    from ..meshing._result import CellMeshingResult
    from ._source_boundary_records import SourceBoundaryQuery


type _Vector = tuple[Fraction, Fraction, Fraction]
type _Triangle = tuple[_Vector, _Vector, _Vector]


def _dot(a: _Vector, b: _Vector, /) -> Fraction:
    return sum((x * y for x, y in zip(a, b, strict=True)), Fraction(0))


def _cross(a: _Vector, b: _Vector, /) -> _Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _sub(a: _Vector, b: _Vector, /) -> _Vector:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _fraction_interval(lo: Fraction, hi: Fraction, /) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(fraction_down(lo)), np.asarray(fraction_up(hi))


def _certify_full_torus(charts: object, /) -> RevolutionSurface:
    """Prove both native source seams and the complete retained torus trim."""
    from ._meshing_domain import _PlacedSurfacePatch
    from ._surface_source_support import SurfaceSourceCharts
    from .brep._assembly import BRepAffinePlacement
    from .brep._boolean_core import AffinePCurve
    from .brep._carrier_pieces import carrier_content_id
    from .brep._correspondence import _periodic_pcurve_source
    from .brep._intersection import ConicParameterEndpoint, NativePeriodEndpoint
    from .brep._patches import ConicRootLinePCurve
    from .brep._source_embedding import _constant_chart_line

    if not isinstance(charts, SurfaceSourceCharts):
        raise ValueError("A torus map requires current source charts.")
    charts.require_current()
    domain = charts.domain
    model = domain.brep_authority
    if (
        model is None
        or len(model.patches) != 1
        or len(domain.patches) != 1
        or tuple(domain.source_indices[2]) != (0,)
    ):
        raise ValueError("A full torus requires one retained B-Rep face.")
    model.require_current()
    surface = model.patches[0]
    declared = domain.patches[0].surface
    if isinstance(declared, _PlacedSurfacePatch):
        if (
            declared.placement.identity_payload()
            != BRepAffinePlacement.identity().identity_payload()
        ):
            raise ValueError("Placed torus charts need a certified placement transport.")
        declared = declared.source
    if (
        not isinstance(surface, RevolutionSurface)
        or not isinstance(surface.curve, CircleCurve)
        or carrier_content_id(declared) != carrier_content_id(surface)
    ):
        raise ValueError("The torus face changes its circular revolution source.")
    _validate_surface_source_shadows(surface)
    if _family(surface) != "ring-torus" or not _single_shell_topology(model, domain):
        raise ValueError("The source is not a proved closed ring-torus shell.")
    geometry = model.geometry
    loops = geometry.face_loops[0]
    if len(loops) != 1 or len(loops[0]) != 4:
        raise ValueError("A full torus requires the two native seam pairs.")
    by_edge: dict[int, list[int]] = {}
    for coedge in loops[0]:
        edge = geometry.coedge_edges[coedge]
        if geometry.edge_curves[edge] < 0:
            raise ValueError("A regular torus has no collapsed source pole.")
        by_edge.setdefault(edge, []).append(coedge)
    if len(by_edge) != 2 or any(len(uses) != 2 for uses in by_edge.values()):
        raise ValueError("Torus seam incidence is not the native two-deck quotient.")
    found: set[int] = set()
    for edge, uses in by_edge.items():
        if geometry.coedge_senses[uses[0]] != -geometry.coedge_senses[uses[1]]:
            raise ValueError(
                "A torus seam must use its source edge in opposite directions."
            )
        transports = tuple(
            _periodic_pcurve_source(geometry.pcurves[use], surface) for use in uses
        )
        first, second = transports
        if first is None or second is None:
            raise ValueError("A torus seam lacks current native-period authority.")
        if carrier_content_id(first[0]) != carrier_content_id(second[0]):
            raise ValueError("A torus seam changes its source pcurve.")
        shifts = {first[1], second[1]}
        if shifts == {(0, 0), (1, 0)}:
            axis = 0
            if _constant_chart_line(first[0], 0) != (
                Fraction(0),
                Fraction(0),
                Fraction(1),
            ) or carrier_content_id(
                geometry.curves[geometry.edge_curves[edge]]
            ) != carrier_content_id(surface.curve):
                raise ValueError(
                    "The torus meridian seam changes its source parameterization."
                )
            roots = geometry.edge_endpoint_roots[edge]
            if len(roots) != 2:
                raise ValueError("The torus meridian requires exact conic roots.")
            start, stop = roots
            if not isinstance(start, ConicParameterEndpoint) or not isinstance(
                stop, ConicParameterEndpoint
            ):
                raise ValueError("The torus meridian requires exact conic roots.")
            if (
                start.root.cosine_exact != stop.root.cosine_exact
                or start.root.sine_exact != stop.root.sine_exact
                or stop.root.turns - start.root.turns != 1
                or carrier_content_id(start.root.conic)
                != carrier_content_id(surface.curve)
                or carrier_content_id(stop.root.conic)
                != carrier_content_id(surface.curve)
            ):
                raise ValueError("The torus meridian is not one exact profile turn.")
        elif shifts == {(0, 0), (0, 1)}:
            axis = 1
            pcurve = first[0]
            if (
                not isinstance(pcurve, AffinePCurve)
                or pcurve.matrix
                != ((Fraction(0), Fraction(1)), (Fraction(1), Fraction(0)))
                or pcurve.offset != (Fraction(0), Fraction(0))
                or not isinstance(pcurve.curve, ConicRootLinePCurve)
                or pcurve.curve.phase_shift_exact != 0
                or pcurve.curve.height_origin_exact != 0
                or pcurve.curve.height_direction_exact != 1
                or carrier_content_id(pcurve.curve.root.conic)
                != carrier_content_id(surface.curve)
            ):
                raise ValueError(
                    "The torus ring seam lacks its rooted source profile chart."
                )
            for coedge in uses:
                roots = geometry.coedge_endpoint_roots[coedge]
                if len(roots) != 2:
                    raise ValueError("The torus ring seam lacks its exact native turn.")
                first_root, last_root = roots
                if (
                    not isinstance(first_root, NativePeriodEndpoint)
                    or not isinstance(last_root, NativePeriodEndpoint)
                    or (first_root.rational, first_root.turns, first_root.axis)
                    != (Fraction(0), 0, 0)
                    or (last_root.rational, last_root.turns, last_root.axis)
                    != (Fraction(0), 1, 0)
                ):
                    raise ValueError("The torus ring seam lacks its exact native turn.")
        else:
            raise ValueError("A torus seam has an unauthorized native shift.")
        if axis in found:
            raise ValueError("The torus source repeats one quotient axis.")
        found.add(axis)
    if found != {0, 1}:
        raise ValueError("The source torus lacks a native quotient axis.")
    return surface


def _source_frame(
    surface: RevolutionSurface, /
) -> tuple[_Vector, _Vector, Fraction, Fraction, _Vector]:
    profile = surface.curve
    if not isinstance(profile, CircleCurve):
        raise ValueError("The torus normal atlas requires a circular source profile.")
    origin = rational_vector(np.asarray(surface.axis_origin, dtype=np.float64))
    axis_values = rational_vector(np.asarray(surface.axis_direction, dtype=np.float64))
    center = profile.center_exact
    radius = profile.radius_exact
    if (
        len(origin) != 3
        or len(axis_values) != 3
        or len(center) != 3
        or not isinstance(center[0], Fraction)
        or not isinstance(center[1], Fraction)
        or not isinstance(center[2], Fraction)
        or not isinstance(radius, Fraction)
    ):
        raise ValueError(
            "The torus normal atlas requires rational exact frame coefficients."
        )
    o: _Vector = (origin[0], origin[1], origin[2])
    axis: _Vector = (axis_values[0], axis_values[1], axis_values[2])
    c: _Vector = (center[0], center[1], center[2])
    relative = _sub(c, o)
    height = _dot(relative, axis)
    base: _Vector = (
        o[0] + height * axis[0],
        o[1] + height * axis[1],
        o[2] + height * axis[2],
    )
    radial = _sub(c, base)
    squared = _dot(radial, radial)
    numerator, denominator = isqrt(squared.numerator), isqrt(squared.denominator)
    if (
        numerator * numerator != squared.numerator
        or denominator * denominator != squared.denominator
    ):
        raise ValueError("The torus major radius has no rational exact radial frame.")
    major = Fraction(numerator, denominator)
    if major <= radius or radius <= 0:
        raise ValueError("The torus must have a strict source-proved ring radius.")
    radial_unit: _Vector = (radial[0] / major, radial[1] / major, radial[2] / major)
    return base, axis, major, radius, radial_unit


def _normal_fiber_box_sign(
    triangle: _Triangle, base: _Vector, axis: _Vector, major: Fraction, /
) -> int | None:
    """Bound the torus normal dot the carrier normal on one exact subtriangle."""
    normal = _cross(_sub(triangle[1], triangle[0]), _sub(triangle[2], triangle[0]))
    radial: list[_Vector] = []
    for point in triangle:
        relative = _sub(point, base)
        z = _dot(relative, axis)
        radial.append(
            (
                relative[0] - z * axis[0],
                relative[1] - z * axis[1],
                relative[2] - z * axis[2],
            )
        )
    squared = _fraction_interval(Fraction(0), Fraction(0))
    for component in range(3):
        bounds = _fraction_interval(
            min(p[component] for p in radial), max(p[component] for p in radial)
        )
        squared = interval_add(squared, interval_square(bounds))
    rho = interval_sqrt(squared)
    if float(rho[0]) <= 0:
        return None
    products = tuple(_dot(normal, p) for p in radial)
    projected = interval_divide(_fraction_interval(min(products), max(products)), rho)
    constant = _dot(normal, _sub(triangle[0], base))
    result = interval_subtract(
        _fraction_interval(constant, constant),
        interval_multiply(_fraction_interval(major, major), projected),
    )
    if float(result[0]) > 0:
        return 1
    if float(result[1]) < 0:
        return -1
    return None


def _normal_fiber_sign(
    triangle: _Triangle, base: _Vector, axis: _Vector, major: Fraction, /
) -> int:
    """Prove strict normal transversality on a whole affine triangle."""
    pending: list[tuple[_Triangle, int]] = [(triangle, 0)]
    signs: set[int] = set()

    def midpoint(a: _Vector, b: _Vector, /) -> _Vector:
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)

    while pending:
        cell, depth = pending.pop()
        sign = _normal_fiber_box_sign(cell, base, axis, major)
        if sign is not None:
            signs.add(sign)
            if len(signs) > 1:
                raise ValueError("The torus normal fiber reverses cell orientation.")
            continue
        if depth == 8:
            raise ValueError("The torus normal fiber may be tangent to a carrier cell.")
        a, b, c = cell
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        pending.extend(
            (
                (part, depth + 1)
                for part in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))
            )
        )
    return next(iter(signs))


def _require_degree_one(
    triangles: tuple[_Triangle, ...],
    base: _Vector,
    axis: _Vector,
    major: Fraction,
    radial: _Vector,
    /,
) -> None:
    """Count preimages of one generic exact rational torus normal fiber."""
    tangent = _cross(axis, radial)
    for denominator in count(1):
        for i in range(denominator + 1):
            if gcd(i, denominator) != 1:
                continue
            k = Fraction(i, denominator)
            ck, sk = (1 - k * k) / (1 + k * k), 2 * k / (1 + k * k)
            for j in range(denominator + 1):
                if gcd(j, denominator) != 1:
                    continue
                l = Fraction(j, denominator)
                cl, sl = (1 - l * l) / (1 + l * l), 2 * l / (1 + l * l)
                direction: _Vector = (
                    cl * radial[0] + sl * tangent[0],
                    cl * radial[1] + sl * tangent[1],
                    cl * radial[2] + sl * tangent[2],
                )
                normal: _Vector = (
                    ck * direction[0] + sk * axis[0],
                    ck * direction[1] + sk * axis[1],
                    ck * direction[2] + sk * axis[2],
                )
                anchor: _Vector = (
                    base[0] + major * direction[0],
                    base[1] + major * direction[1],
                    base[2] + major * direction[2],
                )
                intersections, ambiguous = 0, False
                for triangle in triangles:
                    a, b = _sub(triangle[1], triangle[0]), _sub(triangle[2], triangle[0])
                    matrix = tuple((a[n], b[n], -normal[n]) for n in range(3))
                    rhs = tuple((anchor[n] - triangle[0][n],) for n in range(3))
                    try:
                        solution = solve_exact(matrix, rhs)
                    except ValueError:
                        ambiguous = True
                        break
                    u, v, t = solution[0][0], solution[1][0], solution[2][0]
                    if t <= 0 or major + t * ck <= 0 or u < 0 or v < 0 or u + v > 1:
                        continue
                    if u == 0 or v == 0 or u + v == 1:
                        ambiguous = True
                        break
                    intersections += 1
                if not ambiguous:
                    if intersections != 1:
                        raise ValueError(
                            "The torus carrier does not have normal-fiber degree one."
                        )
                    return


class BRepTorusMaterialMap(StrictModule, NonTrainableState):
    """Source torus normal projection of a proved embedded triangular carrier."""

    base: Array
    axis: Array
    major: Array
    minor: Array
    corners: tuple[Array, ...]
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
            raise ValueError("Torus material maps require retained source charts.")
        mesh, geometry = result.mesh, result.geometry
        charts.require_bound(mesh, geometry, source_query)
        surface = _certify_full_torus(charts)
        base, axis, major, minor, radial = _source_frame(surface)
        if (
            mesh.topological_dimension != 2
            or mesh.ambient_dimension != 3
            or any(block.cell_kind != "triangle" for block in mesh.blocks)
        ):
            raise ValueError(
                "Torus material maps require triangular cells in three-space."
            )
        connectivity = mesh.connectivity
        if not isinstance(connectivity, PolygonalConnectivity):
            raise ValueError("Torus material maps require triangle incidence.")
        cell_count = sum(block.cell_count for block in mesh.blocks)
        if (
            cell_count == 0
            or int(connectivity.triangle_count) != cell_count
            or np.any(np.asarray(connectivity.boundary_edges, dtype=np.bool_))
            or np.any(np.asarray(connectivity.edge_cell_counts, dtype=np.int32) != 2)
        ):
            raise ValueError("Torus material maps require a closed two-manifold.")
        vertices = np.concatenate(
            [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
        )
        if np.unique(vertices).size - len(connectivity.edges) + cell_count != 0:
            raise ValueError(
                "The torus material mesh must have Euler characteristic zero."
            )
        cell_edges = np.asarray(connectivity.cell_edges, dtype=np.int64)
        edge_signs = np.asarray(connectivity.cell_edge_signs, dtype=np.int8)
        edge_valid = np.asarray(connectivity.cell_edge_valid, dtype=np.bool_)
        uses: dict[int, list[tuple[int, int]]] = {}
        for row in range(cell_count):
            for edge, sign in zip(
                cell_edges[row, edge_valid[row]],
                edge_signs[row, edge_valid[row]],
                strict=True,
            ):
                if int(sign) not in (-1, 1):
                    raise ValueError(
                        "The torus carrier incidence has an invalid orientation."
                    )
                uses.setdefault(int(edge), []).append((row, int(sign)))
        adjacency = [set() for _ in range(cell_count)]
        for pair in uses.values():
            if len(pair) != 2 or pair[0][1] + pair[1][1] != 0:
                raise ValueError(
                    "Torus carrier edges require opposed two-cell incidence."
                )
            a, b = pair[0][0], pair[1][0]
            adjacency[a].add(b)
            adjacency[b].add(a)
        seen, pending = {0}, [0]
        while pending:
            for neighbor in adjacency[pending.pop()] - seen:
                seen.add(neighbor)
                pending.append(neighbor)
        if len(seen) != cell_count:
            raise ValueError("The torus material mesh must be connected.")
        require_polyhedral_carrier_geometry(mesh, geometry)
        certification = result.certification
        if certification is None or certification.embedding is None:
            raise ValueError("Torus material maps require a certified embedded carrier.")
        embedding = certification.embedding
        embedding.binding.require(mesh, geometry)
        if embedding.status != "certified":
            raise ValueError("The torus carrier embedding is not certified.")
        embedding.validity.require_bound(geometry, mesh=mesh)
        if not embedding.validity.all_certified:
            raise ValueError("The torus carrier cells are not certified.")
        coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
        exact_vertices: tuple[_Vector, ...] = tuple(
            (
                Fraction.from_float(float(p[0])),
                Fraction.from_float(float(p[1])),
                Fraction.from_float(float(p[2])),
            )
            for p in coordinates
        )
        triangles: tuple[_Triangle, ...] = tuple(
            (
                exact_vertices[int(v[0])],
                exact_vertices[int(v[1])],
                exact_vertices[int(v[2])],
            )
            for v in vertices
        )
        signs = {
            _normal_fiber_sign(triangle, base, axis, major) for triangle in triangles
        }
        if len(signs) != 1:
            raise ValueError("Torus carrier normal fibers reverse orientation.")
        _require_degree_one(triangles, base, axis, major, radial)
        offsets = np.cumsum([0, *(block.cell_count for block in mesh.blocks)])
        self.base = jnp.asarray(tuple(float(value) for value in base))
        self.axis = jnp.asarray(tuple(float(value) for value in axis))
        self.major = jnp.asarray(float(major))
        self.minor = jnp.asarray(float(minor))
        self.corners = tuple(
            jnp.asarray(coordinates[vertices[offsets[index] : offsets[index + 1]]])
            for index in range(len(mesh.blocks))
        )
        self.carrier_coordinates = jnp.asarray(geometry.coordinates)
        self.topology_id = mesh.topology_id
        self.geometry_layout_id = geometry.geometry_layout_id
        self.source_id = charts.domain.source_id
        self.source_revision = charts.domain.source_revision
        self.source_chart_id = charts.source_chart_id
        self.source_geometry_id = canonical_fingerprint(
            {
                "kind": "brep-torus-normal-material-map",
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
        point, jacobian = self._evaluate(block_index, reference_points, derivative=True)
        if jacobian is None:
            raise RuntimeError(
                "Torus material derivative evaluation omitted its Jacobian."
            )
        return point, jacobian

    def _evaluate(
        self, block_index: int, reference_points: ArrayLike, /, *, derivative: bool
    ) -> tuple[Array, Array | None]:
        points = jnp.asarray(reference_points)
        corners = self.corners[block_index]
        gradient = jnp.swapaxes(corners[:, 1:] - corners[:, :1], 1, 2)
        x = corners[:, None, 0, :] + ein.contract("qa,cda->cqd", points, gradient)
        relative = x - self.base[None, None, :]
        z = ein.contract("cqd,d->cq", relative, self.axis)
        radial = relative - z[..., None] * self.axis[None, None, :]
        rho = jnp.linalg.norm(radial, axis=-1, keepdims=True)
        unit = radial / rho
        normal_vector = relative - self.major * unit
        distance = jnp.linalg.norm(normal_vector, axis=-1, keepdims=True)
        normal = normal_vector / distance
        physical = self.base[None, None, :] + self.major * unit + self.minor * normal
        if not derivative:
            return physical, None
        dz = ein.contract("cdi,d->ci", gradient, self.axis)
        dr = gradient - self.axis[None, :, None] * dz[:, None, :]
        drho = ein.contract("cqd,cdi->cqi", unit, dr)
        du = (dr[:, None, :, :] - unit[..., :, None] * drho[..., None, :]) / rho[
            ..., None
        ]
        dq = gradient[:, None, :, :] - self.major * du
        ddistance = ein.contract("cqd,cqdi->cqi", normal, dq)
        dn = (dq - normal[..., :, None] * ddistance[..., None, :]) / distance[..., None]
        return physical, self.major * du + self.minor * dn
