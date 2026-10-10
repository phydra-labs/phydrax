#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact coincident support correspondence and source-cylinder arrangements.

Coincidence is a whole-carrier coefficient decision, never a tolerance test.
The axial arrangement retains a lateral fragment for each source-membership
cell; only caps separating two selected cells are canceled. Authoritative
curves, p-curves and face supports are retained independently of tessellation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from math import pi
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...typing import parse
from .._atlas import TrimDomain
from .._cad_revision import (
    AssociationCoverageEvidence,
    AssociationGraph,
    OccurrenceCorrespondence,
    OccurrenceCorrespondenceTransaction,
)
from ._boolean import (
    _selected,
    _source_revision,
    BRepBooleanFailure,
    BRepBooleanOperation,
    BRepBooleanPolicy,
    BRepBooleanResult,
)
from ._constructors import _Builder, assemble_brep_model, brep_trim_domain
from ._correspondence import _circle_correspondence
from ._intersection import (
    _exact_source_lift,
    NativePeriodEndpoint,
    RootEndpoint,
)
from ._intersection_curve import (
    _native_curve_period_symbol,
    _PeriodOffset,
    AffinePCurve,
    pcurve_periodic_source,
    PeriodicPCurve,
    SurfaceRegion,
)
from ._model import (
    _carrier_payload,
    BRepCurve,
    BRepEntityId,
    BRepGeometry,
    BRepModel,
    BRepPCurve,
)
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    CircleCurve,
    CylinderPatch,
    LineCurve,
    PlanePatch,
    SpherePatch,
)
from ._placed import PlacedSurface, same_source_pose
from ._root_bindings import BRepCurveSurfaceLift, BRepVertexRoot
from ._sewing import sew_brep


type _Vector = tuple[Fraction, ...]
type _UVVector = tuple[Fraction, Fraction]
type _Matrix = tuple[_UVVector, _UVVector]
type _Parent = tuple[int, int]

_IDENTITY: _Matrix = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
_ZERO: _UVVector = (Fraction(0), Fraction(0))
_PERIOD = 2.0 * pi


def _fractions(values: np.ndarray, /) -> _Vector:
    return tuple(Fraction(float(value)) for value in values.reshape(-1))


def _exact_float(value: Fraction, /) -> float:
    result = float(value)
    if not np.isfinite(result) or Fraction(result) != value:
        raise BRepBooleanFailure(
            "coincident chart coefficient needs an exact rational parameter carrier"
        )
    return result


def _parallel_offset(
    origin: _Vector, other: _Vector, axis: _Vector, /
) -> Fraction | None:
    pivot = next((index for index, value in enumerate(axis) if value != 0), None)
    if pivot is None:
        return None
    offset = (other[pivot] - origin[pivot]) / axis[pivot]
    return (
        offset
        if all(b - a == offset * k for a, b, k in zip(origin, other, axis, strict=True))
        else None
    )


def _plane_coordinates(
    first: _Vector, second: _Vector, value: _Vector, /
) -> _UVVector | None:
    # Host-only exact two-column coefficient identity, checked in all three rows.
    for row, column in ((0, 1), (0, 2), (1, 2)):
        determinant = first[row] * second[column] - first[column] * second[row]
        if determinant == 0:
            continue
        u = (value[row] * second[column] - value[column] * second[row]) / determinant
        v = (first[row] * value[column] - first[column] * value[row]) / determinant
        if all(a * u + b * v == c for a, b, c in zip(first, second, value, strict=True)):
            return u, v
        return None
    return None


@dataclass(frozen=True, slots=True)
class _AnalyticSupport:
    family: Literal["plane", "cylinder", "sphere"]
    origin: _Vector
    axes: tuple[_Vector, ...]


def _analytic_support(patch: AbstractSurfacePatch, /) -> _AnalyticSupport | None:
    """Exact expression coefficients, never a replacement physical carrier.

    Authored binary coefficients are rational, but their products and sums
    need not be binary-representable. Keep those products in Q throughout the
    proof. Radius-folded axes describe the actual trigonometric expression;
    orthogonality or normalization of a moved analytic copy is not assumed.
    """
    if isinstance(patch, PlacedSurface):
        source = _analytic_support(patch.definition)
        if source is None:
            return None
        matrix = tuple(_fractions(row) for row in np.asarray(patch.rotation))
        shift = _fractions(np.asarray(patch.translation))

        def mapped(value: _Vector) -> _Vector:
            return tuple(
                sum((a * b for a, b in zip(row, value, strict=True)), Fraction(0))
                for row in matrix
            )

        origin = tuple(a + b for a, b in zip(mapped(source.origin), shift, strict=True))
        return _AnalyticSupport(
            source.family, origin, tuple(mapped(axis) for axis in source.axes)
        )
    if isinstance(patch, PlanePatch):
        return _AnalyticSupport(
            "plane",
            _fractions(np.asarray(patch.origin)),
            (
                _fractions(np.asarray(patch.first_axis)),
                _fractions(np.asarray(patch.second_axis)),
            ),
        )
    if isinstance(patch, (CylinderPatch, SpherePatch)):
        radius = Fraction(float(patch.radius))
        radial = tuple(
            tuple(radius * value for value in _fractions(np.asarray(axis)))
            for axis in (patch.first_axis, patch.second_axis)
        )
        axis = _fractions(np.asarray(patch.axis))
        if isinstance(patch, CylinderPatch):
            return _AnalyticSupport(
                "cylinder", _fractions(np.asarray(patch.origin)), (*radial, axis)
            )
        return _AnalyticSupport(
            "sphere",
            _fractions(np.asarray(patch.center)),
            (*radial, tuple(radius * value for value in axis)),
        )
    return None


class CoincidentSurfaceCorrespondence(StrictModule):
    """Proof of ``first(q) == second(matrix q + offset)`` for every q.

    The rational coefficients are authoritative host preparation metadata.
    Transport refuses unrepresentable coefficients rather than rounding them.
    A proof does not identify source entities: callers publish both parents.
    """

    first: AbstractSurfacePatch
    second: AbstractSurfacePatch
    matrix: _Matrix = eqx.field(static=True)
    offset: _UVVector = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        first: AbstractSurfacePatch,
        second: AbstractSurfacePatch,
        matrix: _Matrix,
        offset: _UVVector,
    ) -> None:
        proved = _surface_map(first, second)
        if proved is None or proved != (matrix, offset):
            raise ValueError(
                "Coincident correspondence needs a complete exact carrier identity."
            )
        self.first, self.second = first, second
        self.matrix, self.offset = matrix, offset
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "native-coincident-surface-correspondence",
                "first": _carrier_payload(first),
                "second": _carrier_payload(second),
                "matrix": [
                    [(value.numerator, value.denominator) for value in row]
                    for row in matrix
                ],
                "offset": [(value.numerator, value.denominator) for value in offset],
            }
        )

    @property
    def orientation(self) -> int:
        determinant = (
            self.matrix[0][0] * self.matrix[1][1] - self.matrix[0][1] * self.matrix[1][0]
        )
        return 1 if determinant > 0 else -1


def _surface_map(
    first: AbstractSurfacePatch,
    second: AbstractSurfacePatch,
    /,
) -> tuple[_Matrix, _UVVector] | None:
    if canonical_fingerprint(_carrier_payload(first)) == canonical_fingerprint(
        _carrier_payload(second)
    ):
        return _IDENTITY, _ZERO
    if (
        isinstance(first, PlacedSurface)
        and isinstance(second, PlacedSurface)
        and same_source_pose(first, second)
    ):
        return _surface_map(first.definition, second.definition)
    a, b = _analytic_support(first), _analytic_support(second)
    if a is None or b is None or a.family != b.family:
        return None
    if a.family == "plane":
        first_axis, second_axis = b.axes[0], b.axes[1]
        columns = tuple(
            _plane_coordinates(first_axis, second_axis, axis) for axis in a.axes
        )
        delta = tuple(x - y for x, y in zip(a.origin, b.origin, strict=True))
        offset = _plane_coordinates(first_axis, second_axis, delta)
        if columns[0] is None or columns[1] is None or offset is None:
            return None
        matrix: _Matrix = ((columns[0][0], columns[1][0]), (columns[0][1], columns[1][1]))
        if matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0] == 0:
            return None
        return matrix, offset
    if a.family == "cylinder" and a.axes == b.axes:
        shift = _parallel_offset(b.origin, a.origin, b.axes[2])
        if shift is not None:
            return _IDENTITY, (Fraction(0), shift)
    if a.family == "sphere" and a.origin == b.origin and a.axes == b.axes:
        return _IDENTITY, _ZERO
    return None


def prove_surface_correspondence(
    first: AbstractSurfacePatch,
    second: AbstractSurfacePatch,
    /,
) -> CoincidentSurfaceCorrespondence | None:
    """Prove arbitrary identical carriers or exact posed analytic expressions.

    None means no identity has been proved, not that the supports are disjoint.
    Rational/procedural carriers with unequal parameterizations require their
    owning exact reparameterization certificate before a general DCEL overlay.
    """
    if not isinstance(first, AbstractSurfacePatch) or not isinstance(
        second, AbstractSurfacePatch
    ):
        raise TypeError("Coincidence operands must be native surface patches.")
    result = _surface_map(first, second)
    return (
        None
        if result is None
        else CoincidentSurfaceCorrespondence(first, second, *result)
    )


def transport_coincident_pcurve(
    proof: CoincidentSurfaceCorrespondence,
    pcurve: BRepPCurve,
    /,
) -> BRepPCurve:
    """Transport exact source p-curves without changing their parameter or roots.

    A general overlay must carry this proof alongside the original endpoint
    roots: a transformed numerical coordinate is not a new endpoint identity.
    """
    if not isinstance(proof, CoincidentSurfaceCorrespondence):
        raise TypeError("proof must be a CoincidentSurfaceCorrespondence.")
    if pcurve.ambient_dimension != 2:
        raise ValueError("A coincidence p-curve must be two-dimensional.")
    if proof.matrix == _IDENTITY and proof.offset == _ZERO:
        return pcurve
    return AffinePCurve(
        pcurve,
        np.asarray(proof.matrix, dtype=np.object_),
        np.asarray(proof.offset, dtype=np.object_),
    )


class CoincidentFaceArc(StrictModule):
    """Complete authored coedge in a proved common chart, never a UV alias.

    ``sense`` orients its DCEL loop; ``source_sense`` retains 3D traversal.
    Original endpoint carriers, roots and vertex identities remain authoritative.
    """

    pcurve: BRepPCurve
    source_pcurve: BRepPCurve
    curve: BRepCurve | None
    transport: CoincidentSurfaceCorrespondence
    edge_roots: tuple[RootEndpoint | None, RootEndpoint | None]
    coedge_roots: tuple[RootEndpoint | None, RootEndpoint | None]
    vertex_roots: tuple[BRepVertexRoot | None, BRepVertexRoot | None]
    face_id: BRepEntityId = eqx.field(static=True)
    edge_id: BRepEntityId = eqx.field(static=True)
    vertex_ids: tuple[BRepEntityId, BRepEntityId] = eqx.field(static=True)
    coedge: int = eqx.field(static=True)
    source_sense: int = eqx.field(static=True)
    sense: int = eqx.field(static=True)
    parameter_range: tuple[float, float] = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: BRepModel,
        face: int,
        coedge: int,
        transport: CoincidentSurfaceCorrespondence,
    ) -> None:
        geometry = model.geometry
        if geometry is None or not 0 <= face < len(geometry.face_loops):
            raise ValueError("Coincident arcs require an authoritative source face.")
        if not any(coedge in loop for loop in geometry.face_loops[face]):
            raise ValueError("The source coedge must belong to the declared face.")
        if canonical_fingerprint(
            _carrier_payload(model.patches[face])
        ) != canonical_fingerprint(_carrier_payload(transport.first)):
            raise ValueError(
                "Coincident arc transport must start on its actual source support."
            )
        edge = geometry.coedge_edges[coedge]
        curve_index = geometry.edge_curves[edge]
        start, end = geometry.edge_vertices[edge]
        source_pcurve = geometry.pcurves[coedge]
        self.pcurve = transport_coincident_pcurve(transport, source_pcurve)
        self.source_pcurve = source_pcurve
        self.curve = None if curve_index < 0 else geometry.curves[curve_index]
        self.transport = transport
        self.edge_roots = geometry.edge_endpoint_roots[edge]
        self.coedge_roots = geometry.coedge_endpoint_roots[coedge]
        self.vertex_roots = geometry.vertex_roots[start], geometry.vertex_roots[end]
        self.face_id = model.face_ids[face]
        self.edge_id = model.edge_ids[edge]
        self.vertex_ids = model.vertex_ids[start], model.vertex_ids[end]
        self.coedge = coedge
        self.source_sense = geometry.coedge_senses[coedge]
        self.sense = self.source_sense * transport.orientation
        first, last = np.asarray(geometry.edge_ranges[edge])
        self.parameter_range = float(first), float(last)
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "native-coincident-face-arc",
                "source": model.source_revision,
                "face": face,
                "coedge": coedge,
                "edge": edge,
                "transport": transport.certificate_id,
                "source_pcurve": _carrier_payload(source_pcurve),
            }
        )

    def parameter_enclosure(self) -> tuple[float, float]:
        first, last = self.parameter_range
        first_root, last_root = self.coedge_roots
        if first_root is not None:
            first = first_root.parameter_enclosure()[0]
        if last_root is not None:
            last = last_root.parameter_enclosure()[1]
        return first, last

    def enclosure(self) -> np.ndarray:
        return self.pcurve.bounding_box(*self.parameter_enclosure())


def _face_coorientations(
    model: BRepModel,
    face: int,
    chart_orientation: int,
) -> tuple[tuple[BRepEntityId, int], ...]:
    result = []
    for solid in model.topology.face_solids[face]:
        faces = model.topology.solid_faces[solid]
        incidence = model.topology.solid_face_orientations[solid][faces.index(face)]
        orientation = incidence * int(float(model.orientation[face])) * chart_orientation
        result.append((model.solid_ids[solid], orientation))
    return tuple(result)


class CoincidentFaceOverlay(StrictModule):
    """Complete trim workset and material-side facts on a proved common support.

    All outer/hole coedges and original roots are retained. Bounds cover complete
    source/target trim carriers, not sampled overlap. Parent DCEL owns certified
    open-cell trim membership; coorientation facts apply only to those cells.
    Domains use each original raw UV chart; only arc worksets/bounds are in the
    common first chart. Background membership includes any non-face-owning
    solids and is never inferred from an absent trim arc.
    """

    correspondence: CoincidentSurfaceCorrespondence
    first_loops: tuple[tuple[CoincidentFaceArc, ...], ...]
    second_loops: tuple[tuple[CoincidentFaceArc, ...], ...]
    first_domain: TrimDomain
    second_domain: TrimDomain
    parameter_bounds: Array
    first_face_id: BRepEntityId = eqx.field(static=True)
    second_face_id: BRepEntityId = eqx.field(static=True)
    first_coorientations: tuple[tuple[BRepEntityId, int], ...] = eqx.field(static=True)
    second_coorientations: tuple[tuple[BRepEntityId, int], ...] = eqx.field(static=True)
    first_background_required: bool = eqx.field(static=True)
    second_background_required: bool = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        first: BRepModel,
        first_face: int,
        second: BRepModel,
        second_face: int,
        correspondence: CoincidentSurfaceCorrespondence,
        policy: BRepBooleanPolicy,
    ) -> None:
        if not isinstance(policy, BRepBooleanPolicy):
            raise TypeError("Coincident overlay requires a BRepBooleanPolicy.")
        first_geometry, second_geometry = first.geometry, second.geometry
        if first_geometry is None or second_geometry is None:
            raise ValueError("Coincident overlay requires authoritative source geometry.")
        if not 0 <= first_face < len(first.patches) or not 0 <= second_face < len(
            second.patches
        ):
            raise ValueError(
                "Coincident overlay faces must exist in their source models."
            )
        expected = prove_surface_correspondence(
            second.patches[second_face], first.patches[first_face]
        )
        if expected is None or expected.certificate_id != correspondence.certificate_id:
            raise ValueError(
                "Coincident overlay needs the exact source-to-target chart proof."
            )
        identity = CoincidentSurfaceCorrespondence(
            first.patches[first_face],
            first.patches[first_face],
            _IDENTITY,
            _ZERO,
        )
        first_loops = tuple(
            tuple(
                CoincidentFaceArc(first, first_face, coedge, identity) for coedge in loop
            )
            for loop in first_geometry.face_loops[first_face]
        )
        second_loops = tuple(
            tuple(
                CoincidentFaceArc(second, second_face, coedge, correspondence)
                for coedge in (
                    loop if correspondence.orientation > 0 else tuple(reversed(loop))
                )
            )
            for loop in second_geometry.face_loops[second_face]
        )
        boxes = np.stack(
            tuple(
                arc.enclosure()
                for loops in (first_loops, second_loops)
                for loop in loops
                for arc in loop
            )
        )
        self.correspondence = correspondence
        self.first_loops, self.second_loops = first_loops, second_loops
        self.first_domain = brep_trim_domain(
            first_geometry,
            first_face,
            first.patches[first_face],
            tolerance=policy.trim_resolution,
            maximum_arcs=policy.maximum_cells,
        )
        self.second_domain = brep_trim_domain(
            second_geometry,
            second_face,
            second.patches[second_face],
            tolerance=policy.trim_resolution,
            maximum_arcs=policy.maximum_cells,
        )
        self.parameter_bounds = jnp.asarray(
            np.stack((np.min(boxes[:, 0], axis=0), np.max(boxes[:, 1], axis=0))),
            dtype=jnp.float64,
        )
        self.first_face_id, self.second_face_id = (
            first.face_ids[first_face],
            second.face_ids[second_face],
        )
        self.first_coorientations = _face_coorientations(first, first_face, 1)
        self.second_coorientations = _face_coorientations(
            second, second_face, correspondence.orientation
        )
        self.first_background_required = (
            len(self.first_coorientations) != first.topology.num_solids
        )
        self.second_background_required = (
            len(self.second_coorientations) != second.topology.num_solids
        )
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "native-complete-coincident-face-overlay",
                "first": (first.source_revision, first_face),
                "second": (second.source_revision, second_face),
                "correspondence": correspondence.certificate_id,
                "first_loops": tuple(
                    tuple(arc.certificate_id for arc in loop) for loop in first_loops
                ),
                "second_loops": tuple(
                    tuple(arc.certificate_id for arc in loop) for loop in second_loops
                ),
                "first_coorientations": tuple(
                    (entity.index, orientation)
                    for entity, orientation in self.first_coorientations
                ),
                "second_coorientations": tuple(
                    (entity.index, orientation)
                    for entity, orientation in self.second_coorientations
                ),
                "background_required": (
                    self.first_background_required,
                    self.second_background_required,
                ),
            }
        )

    @property
    def coorientation_pairs(self) -> tuple[tuple[BRepEntityId, BRepEntityId, int], ...]:
        """Same (+1) or opposed (-1) material side for each actual solid owner."""
        return tuple(
            (first, second, a * b)
            for first, a in self.first_coorientations
            for second, b in self.second_coorientations
        )


def prepare_coincident_face_overlay(
    first: BRepModel,
    first_face: int,
    second: BRepModel,
    second_face: int,
    /,
    *,
    policy: BRepBooleanPolicy,
) -> CoincidentFaceOverlay | None:
    """Prepare all exact trim arcs/material coorientations for parent DCEL."""
    if not isinstance(first, BRepModel) or not isinstance(second, BRepModel):
        raise TypeError("Coincident overlay operands must be BRepModel values.")
    if not isinstance(policy, BRepBooleanPolicy):
        raise TypeError("policy must be a BRepBooleanPolicy.")
    if first.coordinate_contract.spatial_id != second.coordinate_contract.spatial_id:
        raise ValueError(
            "Coincident overlay operands must share the coordinate contract."
        )
    if (
        isinstance(first_face, bool)
        or not isinstance(first_face, int)
        or isinstance(second_face, bool)
        or not isinstance(second_face, int)
    ):
        raise TypeError("Coincident overlay face indices must be integers.")
    if not 0 <= first_face < len(first.patches) or not 0 <= second_face < len(
        second.patches
    ):
        raise ValueError("Coincident overlay face indices must exist.")
    proof = prove_surface_correspondence(
        second.patches[second_face], first.patches[first_face]
    )
    if proof is None:
        return None
    if first.geometry is None or second.geometry is None:
        raise BRepBooleanFailure("Coincident overlay lost authoritative source trims")
    count = sum(
        len(loop)
        for geometry, face in (
            (first.geometry, first_face),
            (second.geometry, second_face),
        )
        for loop in geometry.face_loops[face]
    )
    if count > policy.sewing.maximum_coedges:
        raise BRepBooleanFailure("Coincident overlay coedge work budget exhausted")
    return CoincidentFaceOverlay(first, first_face, second, second_face, proof, policy)


@dataclass(frozen=True, slots=True)
class SourceEdgeFacePairCoverage:
    """Complete plane/cylinder support intersection carried by an authored edge."""

    faces: tuple[BRepEntityId, BRepEntityId]
    source_edge: _Parent
    edge_id: BRepEntityId
    lifts: tuple[BRepCurveSurfaceLift, BRepCurveSurfaceLift]
    parameter_roots: tuple[NativePeriodEndpoint, NativePeriodEndpoint]
    axial_parameter: Fraction
    plane_coefficients: tuple[Fraction, Fraction, Fraction]
    certificate_id: str


def _native_cylinder_ring_chart(model: BRepModel, face: int) -> bool:
    """Ground a 0/one-turn chart in authored scalar roots, not float extent."""
    from ._intersection_curve import _affine_curve_coefficients

    geometry = model.geometry
    if geometry is None:
        return False
    bounds = np.asarray(model.parameter_bounds)[face]
    if bounds[0, 0] != 0.0 or bounds[1, 0] != _PERIOD:
        return False
    for loop in geometry.face_loops[face]:
        for coedge in loop:
            roots = geometry.coedge_endpoint_roots[coedge]
            if not all(isinstance(root, NativePeriodEndpoint) for root in roots):
                continue
            first, last = roots
            if not isinstance(first, NativePeriodEndpoint) or not isinstance(
                last, NativePeriodEndpoint
            ):
                continue
            if first.exact_parameter != Fraction(
                0
            ) or last.exact_parameter != _PeriodOffset(Fraction(0), Fraction(1)):
                continue
            pcurve = geometry.pcurves[coedge]
            if not isinstance(pcurve, AbstractCurve):
                continue
            coefficients = _affine_curve_coefficients(
                pcurve, np.zeros(2, dtype=np.float64)
            )
            if (
                coefficients is not None
                and coefficients[0][0] == 0
                and coefficients[1] == (Fraction(1), Fraction(0))
            ):
                return True
    return False


def prove_source_ring_pair_coverage(
    models: tuple[BRepModel, ...],
    pair: tuple[_Parent, _Parent],
    source_edge: _Parent,
    pcurves: tuple[AbstractCurve, AbstractCurve],
) -> SourceEdgeFacePairCoverage | None:
    """Prove the whole support intersection is one represented native ring.

    The plane's exact radial coefficients vanish and its axial coefficient is
    nonzero, so every solution has one fixed cylinder height. A complete
    authored native-period edge with exact lifts covers every remaining angle.
    Parent graph coverage must also retain this edge on both source faces.
    """
    supports = tuple(
        _analytic_support(models[operand].patches[face]) for operand, face in pair
    )
    cylinder = next(
        (
            index
            for index, support in enumerate(supports)
            if support is not None and support.family == "cylinder"
        ),
        None,
    )
    plane = next(
        (
            index
            for index, support in enumerate(supports)
            if support is not None and support.family == "plane"
        ),
        None,
    )
    if cylinder is None or plane is None:
        return None
    cylindrical, planar = supports[cylinder], supports[plane]
    if cylindrical is None or planar is None:
        return None
    coefficients = tuple(
        _determinant(planar.axes[0], planar.axes[1], axis) for axis in cylindrical.axes
    )
    if coefficients[0] != 0 or coefficients[1] != 0 or coefficients[2] == 0:
        return None
    delta = tuple(a - b for a, b in zip(cylindrical.origin, planar.origin, strict=True))
    height = -_determinant(planar.axes[0], planar.axes[1], delta) / coefficients[2]
    cylinder_owner = pair[cylinder]
    if not _native_cylinder_ring_chart(models[cylinder_owner[0]], cylinder_owner[1]):
        return None
    geometry = models[source_edge[0]].geometry
    if geometry is None or geometry.edge_curves[source_edge[1]] < 0:
        return None
    curve = geometry.curves[geometry.edge_curves[source_edge[1]]]
    if (
        not isinstance(curve, AbstractCurve)
        or _native_curve_period_symbol(curve) != "two_pi"
    ):
        return None
    first_root, last_root = geometry.edge_endpoint_roots[source_edge[1]]
    if not isinstance(first_root, NativePeriodEndpoint) or not isinstance(
        last_root, NativePeriodEndpoint
    ):
        return None
    if first_root.exact_parameter != Fraction(
        0
    ) or last_root.exact_parameter != _PeriodOffset(Fraction(0), Fraction(1)):
        return None
    source_id = canonical_fingerprint(_carrier_payload(curve))
    if any(
        canonical_fingerprint(_carrier_payload(root.carrier)) != source_id
        for root in (first_root, last_root)
    ):
        return None
    first, last = (
        float(value) for value in np.asarray(geometry.edge_ranges)[source_edge[1]]
    )
    if first != first_root.parameter or last != last_root.parameter:
        return None
    regions = tuple(
        SurfaceRegion(
            models[operand].patches[face],
            np.asarray(models[operand].parameter_bounds)[face],
        )
        for operand, face in pair
    )
    if not all(
        _exact_source_lift(curve, pcurve, region, first, last)
        for pcurve, region in zip(pcurves, regions, strict=True)
    ):
        return None
    lifts = (
        BRepCurveSurfaceLift(curve, pcurves[0], regions[0], first, last),
        BRepCurveSurfaceLift(curve, pcurves[1], regions[1], first, last),
    )
    faces = (
        models[pair[0][0]].face_ids[pair[0][1]],
        models[pair[1][0]].face_ids[pair[1][1]],
    )
    edge_id = models[source_edge[0]].edge_ids[source_edge[1]]
    certificate = canonical_fingerprint(
        {
            "kind": "native-source-ring-complete-face-pair",
            "faces": tuple(asdict(face) for face in faces),
            "edge": asdict(edge_id),
            "supports": tuple(
                _carrier_payload(models[operand].patches[face]) for operand, face in pair
            ),
            "plane_coefficients": tuple(
                (value.numerator, value.denominator) for value in coefficients
            ),
            "height": (height.numerator, height.denominator),
            "lifts": tuple(lift.lift_id for lift in lifts),
            "native_period_roots": (first_root.root_id, last_root.root_id),
        }
    )
    return SourceEdgeFacePairCoverage(
        faces,
        source_edge,
        edge_id,
        lifts,
        (first_root, last_root),
        height,
        (coefficients[0], coefficients[1], coefficients[2]),
        certificate,
    )


class CoincidentOpenCell(StrictModule):
    """Oriented retention from parent-certified open-cell trim membership."""

    first_material_sides: tuple[bool, bool] = eqx.field(static=True)
    second_material_sides: tuple[bool, bool] = eqx.field(static=True)
    orientation: int = eqx.field(static=True)
    parents: tuple[BRepEntityId, ...] = eqx.field(static=True)
    classification_id: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        overlay: CoincidentFaceOverlay,
        operation: BRepBooleanOperation,
        first_present: bool,
        second_present: bool,
        first_background_inside: bool | None,
        second_background_inside: bool | None,
        classification_id: str,
    ) -> None:
        operation_ = parse(operation, BRepBooleanOperation, "operation")
        if not isinstance(overlay, CoincidentFaceOverlay):
            raise TypeError("Open-cell retention requires a CoincidentFaceOverlay.")
        if not isinstance(classification_id, str) or not classification_id:
            raise ValueError(
                "Open-cell membership needs its parent classification evidence ID."
            )
        for present, background, required in (
            (first_present, first_background_inside, overlay.first_background_required),
            (
                second_present,
                second_background_inside,
                overlay.second_background_required,
            ),
        ):
            if not isinstance(present, bool) or (
                background is not None and not isinstance(background, bool)
            ):
                raise TypeError(
                    "Certified open-cell membership decisions must be booleans."
                )
            if (not present or required) and background is None:
                raise ValueError(
                    "Absent trim faces or unowned solids require certified background membership."
                )
        sides = []
        for present, background, owners in (
            (first_present, first_background_inside, overlay.first_coorientations),
            (second_present, second_background_inside, overlay.second_coorientations),
        ):
            if present:
                if not owners:
                    raise BRepBooleanFailure(
                        "Coincident Boolean open cell has no declared solid owner"
                    )
                sides.append(
                    (
                        background is True or any(sign > 0 for _, sign in owners),
                        background is True or any(sign < 0 for _, sign in owners),
                    )
                )
            else:
                sides.append((background is True, background is True))
        negative = _selected(
            operation_,
            tuple((operand, 0) for operand, side in enumerate(sides) if side[0]),
        )
        positive = _selected(
            operation_,
            tuple((operand, 0) for operand, side in enumerate(sides) if side[1]),
        )
        orientation = (1 if negative else -1) if negative != positive else 0
        self.first_material_sides, self.second_material_sides = sides[0], sides[1]
        self.orientation = orientation
        self.parents = (
            ()
            if orientation == 0
            else tuple(
                entity
                for entity, present in (
                    (overlay.first_face_id, first_present),
                    (overlay.second_face_id, second_present),
                )
                if present
            )
        )
        self.classification_id = classification_id
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "native-coincident-open-cell-retention",
                "overlay": overlay.certificate_id,
                "classification": classification_id,
                "operation": operation_,
                "present": (first_present, second_present),
                "material_sides": tuple(sides),
                "orientation": orientation,
            }
        )


@dataclass(frozen=True, slots=True)
class _CylinderRegion:
    model: int
    lateral: int
    patch: CylinderPatch
    lower: Fraction
    upper: Fraction
    caps: tuple[int, int]
    rings: tuple[int, int]
    seam: LineCurve
    seam_pcurve: LineCurve


def _determinant(first: _Vector, second: _Vector, third: _Vector, /) -> Fraction:
    return (
        first[0] * (second[1] * third[2] - second[2] * third[1])
        - first[1] * (second[0] * third[2] - second[2] * third[0])
        + first[2] * (second[0] * third[1] - second[1] * third[0])
    )


def _seam_is_exact(curve: LineCurve, pcurve: LineCurve, patch: CylinderPatch, /) -> bool:
    uv = _fractions(np.asarray(pcurve.origin)), _fractions(np.asarray(pcurve.direction))
    if uv[0][0] not in (Fraction(0), Fraction(_PERIOD)) or uv[1][0] != 0:
        return False
    radius = Fraction(float(patch.radius))
    expected_origin = tuple(
        a + radius * b + uv[0][1] * k
        for a, b, k in zip(
            _fractions(np.asarray(patch.origin)),
            _fractions(np.asarray(patch.first_axis)),
            _fractions(np.asarray(patch.axis)),
            strict=True,
        )
    )
    expected_direction = tuple(uv[1][1] * k for k in _fractions(np.asarray(patch.axis)))
    return (
        _fractions(np.asarray(curve.origin)) == expected_origin
        and _fractions(np.asarray(curve.direction)) == expected_direction
    )


def _circle_vertex_is_exact(curve: CircleCurve, point: np.ndarray, /) -> bool:
    radius = Fraction(float(curve.radius))
    expected = tuple(
        center + radius * axis
        for center, axis in zip(
            _fractions(np.asarray(curve.center)),
            _fractions(np.asarray(curve.first_axis)),
            strict=True,
        )
    )
    return _fractions(point) == expected


def _cap_for_ring(model: BRepModel, lateral: int, edge: int, /) -> int | None:
    geometry = model.geometry
    if geometry is None:
        return None
    candidates = []
    for face, loops in enumerate(geometry.face_loops):
        if face == lateral or len(loops) != 1 or len(loops[0]) != 1:
            continue
        coedge = loops[0][0]
        patch = model.patches[face]
        pcurve = geometry.pcurves[coedge]
        curve_index = geometry.edge_curves[edge]
        if curve_index < 0:
            continue
        curve = geometry.curves[curve_index]
        if (
            geometry.coedge_edges[coedge] == edge
            and geometry.coedge_senses[coedge] == 1
            and isinstance(patch, PlanePatch)
            and isinstance(pcurve, CircleCurve)
            and isinstance(curve, CircleCurve)
            and _circle_correspondence(curve, pcurve, patch) == 0.0
        ):
            candidates.append(face)
    return candidates[0] if len(candidates) == 1 else None


def _cylinder_region(model: BRepModel, operand: int, /) -> _CylinderRegion | None:
    geometry = model.geometry
    if (
        geometry is None
        or model.topology.num_solids != 1
        or len(model.patches) != 3
        or geometry.solid_shells != ((0,),)
        or len(geometry.shell_faces) != 1
        or set(geometry.shell_faces[0]) != {0, 1, 2}
        or geometry.shell_orientations[0] != (1, 1, 1)
        or len(geometry.occurrences) != 1
        or geometry.occurrences[0].rotation
        != ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        or geometry.occurrences[0].translation != (0.0, 0.0, 0.0)
        or any(root is not None for root in geometry.vertex_roots)
        or any(root is not None for pair in geometry.edge_endpoint_roots for root in pair)
    ):
        return None
    laterals = [
        index
        for index, patch in enumerate(model.patches)
        if isinstance(patch, CylinderPatch)
    ]
    if len(laterals) != 1:
        return None
    lateral = laterals[0]
    patch = model.patches[lateral]
    if not isinstance(patch, CylinderPatch):
        raise TypeError("Cylinder admission lost its native surface family.")
    bounds = np.asarray(model.parameter_bounds)[lateral]
    if bounds[0, 0] != 0.0 or bounds[1, 0] != _PERIOD or bounds[0, 1] >= bounds[1, 1]:
        return None
    lower, upper = Fraction(float(bounds[0, 1])), Fraction(float(bounds[1, 1]))
    loops = geometry.face_loops[lateral]
    if len(loops) != 1 or len(loops[0]) != 4:
        return None
    coefficients = []
    turn = _PeriodOffset(Fraction(0), Fraction(1))
    for coedge in loops[0]:
        pcurve = geometry.pcurves[coedge]
        source = pcurve_periodic_source(pcurve, patch)
        if source is None:
            return None
        base, shifts = source
        if not isinstance(base, LineCurve) or shifts[1] != 0:
            return None
        edge = geometry.coedge_edges[coedge]
        curve_index = geometry.edge_curves[edge]
        if curve_index < 0:
            return None
        curve = geometry.curves[curve_index]
        first, last = map(float, np.asarray(geometry.edge_ranges)[edge])
        origin, direction = (
            _fractions(np.asarray(base.origin)),
            _fractions(np.asarray(base.direction)),
        )
        start: tuple[Fraction | _PeriodOffset, Fraction]
        end: tuple[Fraction | _PeriodOffset, Fraction]
        if isinstance(curve, CircleCurve):
            if (
                shifts != (0, 0)
                or (first, last) != (0.0, _PERIOD)
                or origin[0] != 0
                or direction != (Fraction(1), Fraction(0))
            ):
                return None
            start = Fraction(0), origin[1]
            end = turn, origin[1]
        elif isinstance(curve, LineCurve) and isinstance(pcurve, PeriodicPCurve):
            if shifts[0] not in (0, 1) or origin[0] != 0 or direction[0] != 0:
                return None
            u = Fraction(0) if shifts[0] == 0 else turn
            start = u, origin[1] + Fraction(first) * direction[1]
            end = u, origin[1] + Fraction(last) * direction[1]
        else:
            return None
        coefficients.append(
            (start, end) if geometry.coedge_senses[coedge] > 0 else (end, start)
        )
    corners = ((Fraction(0), lower), (turn, lower), (turn, upper), (Fraction(0), upper))
    expected = tuple((corners[index], corners[(index + 1) % 4]) for index in range(4))
    if tuple(coefficients) not in tuple(
        expected[index:] + expected[:index] for index in range(4)
    ):
        return None
    rings: dict[Fraction, int] = {}
    seam: tuple[LineCurve, LineCurve] | None = None
    seam_edges = set()
    for coedge in loops[0]:
        edge = geometry.coedge_edges[coedge]
        curve_index = geometry.edge_curves[edge]
        if curve_index < 0:
            return None
        curve, pcurve = geometry.curves[curve_index], geometry.pcurves[coedge]
        source = pcurve_periodic_source(pcurve, patch)
        if source is None:
            return None
        base, shifts = source
        if not isinstance(base, LineCurve):
            return None
        if isinstance(curve, CircleCurve):
            if (
                _circle_correspondence(curve, base, patch) != 0.0
                or shifts != (0, 0)
                or tuple(np.asarray(geometry.edge_ranges)[edge]) != (0.0, _PERIOD)
                or geometry.edge_vertices[edge][0] != geometry.edge_vertices[edge][1]
            ):
                return None
            if not _circle_vertex_is_exact(
                curve, np.asarray(geometry.vertex_points[geometry.edge_vertices[edge][0]])
            ):
                return None
            rings[Fraction(float(base.origin[1]))] = edge
        elif (
            isinstance(curve, LineCurve)
            and isinstance(pcurve, PeriodicPCurve)
            and _seam_is_exact(curve, base, patch)
        ):
            seam_edges.add(edge)
            if shifts == (0, 0):
                seam = curve, base
        else:
            return None
    if set(rings) != {lower, upper} or seam is None or len(seam_edges) != 1:
        return None
    first_cap = _cap_for_ring(model, lateral, rings[lower])
    last_cap = _cap_for_ring(model, lateral, rings[upper])
    if first_cap is None or last_cap is None:
        return None
    handedness = _determinant(
        _fractions(np.asarray(patch.first_axis)),
        _fractions(np.asarray(patch.second_axis)),
        _fractions(np.asarray(patch.axis)),
    )
    if handedness == 0 or float(model.orientation[lateral]) != (
        1.0 if handedness > 0 else -1.0
    ):
        return None
    for face, side in ((first_cap, -1), (last_cap, 1)):
        cap = model.patches[face]
        if not isinstance(cap, PlanePatch):
            return None
        normal = _determinant(
            _fractions(np.asarray(cap.first_axis)),
            _fractions(np.asarray(cap.second_axis)),
            _fractions(np.asarray(patch.axis)),
        )
        if normal == 0 or (normal * Fraction(float(model.orientation[face])) > 0) != (
            side > 0
        ):
            return None
    return _CylinderRegion(
        operand,
        lateral,
        patch,
        lower,
        upper,
        (first_cap, last_cap),
        (rings[lower], rings[upper]),
        *seam,
    )


@dataclass(frozen=True, slots=True)
class _AxialCell:
    lower: Fraction
    upper: Fraction
    members: tuple[_Parent, ...]


def _axial_components(
    regions: tuple[_CylinderRegion, _CylinderRegion],
    shift: Fraction,
    operation: BRepBooleanOperation,
    policy: BRepBooleanPolicy,
) -> tuple[tuple[_AxialCell, ...], ...]:
    intervals = (
        (regions[0].lower, regions[0].upper),
        (regions[1].lower + shift, regions[1].upper + shift),
    )
    coordinates = sorted({value for interval in intervals for value in interval})
    if len(coordinates) - 1 > policy.maximum_cells:
        raise BRepBooleanFailure("coincident axial arrangement cell budget exhausted")
    components: list[list[_AxialCell]] = []
    for lower, upper in zip(coordinates[:-1], coordinates[1:], strict=True):
        members = tuple(
            (operand, 0)
            for operand, (first, last) in enumerate(intervals)
            if first <= lower and upper <= last
        )
        if not _selected(operation, members):
            continue
        cell = _AxialCell(lower, upper, members)
        if components and components[-1][-1].upper == lower:
            components[-1].append(cell)
        else:
            components.append([cell])
    return tuple(tuple(component) for component in components)


def _source_rings(
    regions: tuple[_CylinderRegion, _CylinderRegion],
    shift: Fraction,
) -> dict[Fraction, tuple[tuple[_CylinderRegion, int, int], ...]]:
    result: dict[Fraction, list[tuple[_CylinderRegion, int, int]]] = {}
    for region, offset in zip(regions, (Fraction(0), shift), strict=True):
        for value, cap, ring in zip(
            (region.lower, region.upper), region.caps, region.rings, strict=True
        ):
            result.setdefault(value + offset, []).append((region, cap, ring))
    return {value: tuple(entries) for value, entries in result.items()}


def _stage_component(
    builder: _Builder,
    models: tuple[BRepModel, BRepModel],
    regions: tuple[_CylinderRegion, _CylinderRegion],
    cells: tuple[_AxialCell, ...],
    rings: dict[Fraction, tuple[tuple[_CylinderRegion, int, int], ...]],
    parents: list[tuple[_Parent, ...]],
) -> tuple[int, ...]:
    region = regions[0]
    faces = []
    ring_edges: dict[Fraction, int] = {}
    ring_vertices: dict[Fraction, int] = {}
    for value in (cells[0].lower, *(cell.upper for cell in cells)):
        source, _, edge = rings[value][0]
        geometry = models[source.model].geometry
        if geometry is None:
            raise BRepBooleanFailure("coincident source lost its authoritative geometry")
        curve = geometry.curves[geometry.edge_curves[edge]]
        vertex = builder.vertex(
            np.asarray(geometry.vertex_points[geometry.edge_vertices[edge][0]])
        )
        ring_vertices[value] = vertex
        ring_edges[value] = builder.edge(curve, 0.0, _PERIOD, vertex, vertex)
    lateral_sign = int(float(models[region.model].orientation[region.lateral]))
    seam_origin = _fractions(np.asarray(region.seam_pcurve.origin))[1]
    seam_slope = _fractions(np.asarray(region.seam_pcurve.direction))[1]
    for cell in cells:
        lower, upper = _exact_float(cell.lower), _exact_float(cell.upper)
        first, last = (
            (cell.lower - seam_origin) / seam_slope,
            (cell.upper - seam_origin) / seam_slope,
        )
        start, end = ring_vertices[cell.lower], ring_vertices[cell.upper]
        if seam_slope < 0:
            first, last, start, end = last, first, end, start
        edge = builder.edge(
            region.seam, _exact_float(first), _exact_float(last), start, end
        )
        sense = 1 if seam_slope > 0 else -1
        loop = [
            builder.coedge(
                ring_edges[cell.lower], 1, LineCurve((0.0, lower), (1.0, 0.0))
            ),
            builder.coedge(
                edge,
                sense,
                PeriodicPCurve(region.seam_pcurve, region.patch, period_shifts=(1, 0)),
            ),
            builder.coedge(
                ring_edges[cell.upper], -1, LineCurve((0.0, upper), (1.0, 0.0))
            ),
            builder.coedge(
                edge,
                -sense,
                PeriodicPCurve(region.seam_pcurve, region.patch, period_shifts=(0, 0)),
            ),
        ]
        faces.append(
            builder.face(
                region.patch,
                np.asarray(((0.0, lower), (_PERIOD, upper)), dtype=np.float64),
                [loop],
                models[region.model].physical_tags[region.lateral],
                None,
                lateral_sign,
            )
        )
        parents.append(
            tuple((operand, regions[operand].lateral) for operand, _ in cell.members)
        )
    for value, side in ((cells[0].lower, -1), (cells[-1].upper, 1)):
        source, cap, _ = rings[value][0]
        model = models[source.model]
        geometry = model.geometry
        if geometry is None:
            raise BRepBooleanFailure("coincident source lost its cap incidence")
        original = geometry.face_loops[cap][0][0]
        patch = model.patches[cap]
        if not isinstance(patch, PlanePatch):
            raise BRepBooleanFailure("coincident cap lost its exact plane carrier")
        normal = _determinant(
            _fractions(np.asarray(patch.first_axis)),
            _fractions(np.asarray(patch.second_axis)),
            _fractions(np.asarray(region.patch.axis)),
        )
        sign = side * (1 if normal > 0 else -1)
        coedge = builder.coedge(ring_edges[value], 1, geometry.pcurves[original])
        faces.append(
            builder.face(
                patch,
                np.asarray(model.parameter_bounds)[cap],
                [[coedge]],
                model.physical_tags[cap],
                None,
                sign,
            )
        )
        parents.append(
            tuple((entry.model, parent_cap) for entry, parent_cap, _ in rings[value])
        )
    return tuple(faces)


def _coincidence_graph(
    models: tuple[BRepModel, BRepModel],
    model: BRepModel,
    parents: tuple[tuple[_Parent, ...], ...],
    components: tuple[tuple[_AxialCell, ...], ...],
    certificate: str,
) -> AssociationGraph:
    from ._partition import cad_revision_from_brep_model

    source = _source_revision(models, certificate)
    target = cad_revision_from_brep_model(model)
    pairs = set()
    for solid, (faces, cells) in enumerate(
        zip(model.topology.solid_faces, components, strict=True)
    ):
        for operand, parent_solid in {
            member for cell in cells for member in cell.members
        }:
            pairs.add((f"operand:{operand}/solid:{parent_solid}", f"solid:{solid}"))
        for face in faces:
            for operand, parent_face in parents[face]:
                root = f"operand:{operand}/solid:0"
                pairs.add((root, f"solid:{solid}"))
                pairs.add((f"{root}/face:{parent_face}", f"solid:{solid}/face:{face}"))
    transaction = OccurrenceCorrespondenceTransaction(
        certificate,
        source.revision_id,
        target.revision_id,
        tuple(
            OccurrenceCorrespondence(first, second, certificate)
            for first, second in sorted(pairs)
        ),
        frozenset(),
        frozenset(),
        AssociationCoverageEvidence(
            True, True, certificate, "native-exact-coincident-cad-arrangement"
        ),
    )
    return AssociationGraph(source, target, transaction)


def coincident_boolean_brep(
    models: tuple[BRepModel, BRepModel],
    operation: BRepBooleanOperation,
    policy: BRepBooleanPolicy,
) -> BRepBooleanResult | None:
    """Regularized exact Boolean for proved same-frame full native cylinders.

    None is exclusively an admission outcome. Once the complete source carrier,
    cap, periodic rectangle and outward-region proof succeeds, any resource or
    representability failure raises and must not trigger an alternate route.
    Different radii/frames and non-analytic trims remain general-overlay inputs;
    they are not declared coincident by this helper.
    """
    if len(models) != 2 or not all(isinstance(model, BRepModel) for model in models):
        raise TypeError("Coincident Boolean requires exactly two BRepModel operands.")
    if not isinstance(policy, BRepBooleanPolicy):
        raise TypeError("policy must be a BRepBooleanPolicy.")
    operation_ = parse(operation, BRepBooleanOperation, "operation")
    if (
        models[0].coordinate_contract.spatial_id
        != models[1].coordinate_contract.spatial_id
    ):
        raise ValueError(
            "Coincident Boolean operands must share the coordinate contract."
        )
    first, second = _cylinder_region(models[0], 0), _cylinder_region(models[1], 1)
    if first is None or second is None:
        return None
    proof = prove_surface_correspondence(second.patch, first.patch)
    if proof is None or proof.matrix != _IDENTITY:
        return None
    shift = proof.offset[1]
    regions = first, second
    components = _axial_components(regions, shift, operation_, policy)
    face_count = sum(len(component) + 2 for component in components)
    coedge_count = sum(4 * len(component) + 2 for component in components)
    if face_count > policy.maximum_faces or coedge_count > policy.sewing.maximum_coedges:
        raise BRepBooleanFailure("coincident boundary allocation budget exhausted")
    builder = _Builder()
    parents: list[tuple[_Parent, ...]] = []
    rings = _source_rings(regions, shift)
    solid_faces = tuple(
        _stage_component(builder, models, regions, component, rings, parents)
        for component in components
    )
    geometry = BRepGeometry(
        vertex_points=np.asarray(builder.vertices, dtype=np.float64).reshape((-1, 3)),
        curves=tuple(builder.curves),
        edge_curves=tuple(builder.edge_curves),
        edge_ranges=np.asarray(builder.edge_ranges, dtype=np.float64).reshape((-1, 2)),
        edge_vertices=tuple(builder.edge_vertices),
        pcurves=tuple(builder.pcurves),
        coedge_edges=tuple(builder.coedge_edges),
        coedge_senses=tuple(builder.coedge_senses),
        face_loops=tuple(
            tuple(tuple(loop) for loop in face.loops) for face in builder.faces
        ),
        shell_faces=(),
        shell_orientations=(),
        solid_shells=(),
    )
    signs = np.asarray([face.orientation for face in builder.faces], dtype=np.float64)
    sewn = sew_brep(geometry, signs, solid_faces, policy=policy.sewing)
    certificate = canonical_fingerprint(
        {
            "kind": "native-exact-coincident-cad-boolean",
            "operation": operation_,
            "sources": tuple(model.model_id for model in models),
            "support_proof": proof.certificate_id,
            "geometry": sewn.geometry.geometry_id,
            "parents": tuple(parents),
            "sewing": sewn.certificate_id,
            "repair": "none",
        }
    )
    model = assemble_brep_model(
        sewn.geometry,
        tuple(face.patch for face in builder.faces),
        np.asarray([face.box for face in builder.faces], dtype=np.float64).reshape(
            (-1, 2, 2)
        ),
        signs,
        tuple(face.tag for face in builder.faces),
        coordinate_contract=models[0].coordinate_contract,
        source_id=f"native-boolean:{certificate}",
        source_format="native-boolean",
        source_digest=certificate,
        import_policy_id=certificate,
        tessellation=policy.tessellation,
        curve_surface_tolerance=policy.construction_tolerance,
    )
    graph = _coincidence_graph(models, model, tuple(parents), components, certificate)
    mapped = {pair.source_occurrence_id for pair in graph.transaction.correspondences}
    deleted = tuple(
        occurrence.occurrence_id
        for occurrence in graph.source_revision.occurrences
        if occurrence.occurrence_id not in mapped
    )
    return BRepBooleanResult(model, graph, operation_, certificate, deleted)
