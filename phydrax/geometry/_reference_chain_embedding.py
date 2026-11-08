#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact forward-reference partition proofs for restricted cell maps."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import cast, Literal, TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_enclosure import (
    add,
    axes,
    bernstein_coefficients,
    BernsteinDomain,
    compose_expression,
    constant,
    derivative,
    ExactScalar,
    Expression,
    expression_equal,
    linear_combinations,
    multiply,
    normalize_pyramid_reference_derivative,
    RationalPolynomial,
    scale,
    subtract,
)
from ..discretization._nested_reference_types import (
    exact_reference_cell_vertices,
    reference_chart_arguments_identity,
    ReferenceCellMapWitnesses,
    validate_reference_cell_map_witnesses,
)
from ..discretization._reference_cell import reference_cell_topology
from ._embedding_cells import prepare_embedding_cells
from ._restricted_embedding import reference_chart_composition_equal


if TYPE_CHECKING:
    from ._global_embedding import GlobalEmbeddingCertificate, ValidatedGlobalEmbedding


ReferenceChainStatus = Literal["certified", "violated", "unresolved"]
_ReferenceChainFindingStatus = Literal["violated", "unresolved"]
_Point3 = tuple[Fraction, Fraction, Fraction]


@dataclass(frozen=True, slots=True)
class ReferenceChainEmbeddingFinding:
    check: str
    status: _ReferenceChainFindingStatus
    parent_cell_ids: tuple[int, ...]
    target_cell_ids: tuple[int, ...]
    detail: str


@dataclass(frozen=True, slots=True)
class ReferenceChainChartPremise:
    parent_cell_id: int
    target_block_name: str
    target_cell_id: int
    source_cell_kind: str
    target_cell_kind: str
    target_to_parent_arguments: tuple[Expression, ...]
    parent_reference_vertices: tuple[tuple[Fraction, ...], ...]
    jacobian_sign: int
    jacobian_bernstein_controls: tuple[str, ...]
    composition_id: str


@dataclass(frozen=True, slots=True)
class ReferenceChainInternalFacePremise:
    parent_cell_id: int
    target_face_vertex_ids: tuple[int, ...]
    target_cell_ids: tuple[int, int]
    reference_trace: tuple[Expression, ...]
    physical_trace: tuple[Expression, ...]
    orientation_signs: tuple[int, int]
    trace_id: str


@dataclass(frozen=True, slots=True)
class ReferenceChainBoundaryFacePremise:
    parent_cell_id: int
    parent_facet_index: int
    target_face_vertex_ids: tuple[int, ...]
    target_cell_id: int
    parent_reference_vertices: tuple[tuple[Fraction, ...], ...]
    target_to_parent_face_trace: tuple[Expression, ...]
    face_jacobian_bernstein_controls: tuple[str, ...]
    projected_area: Fraction
    trace_id: str


@dataclass(frozen=True, slots=True)
class ReferenceChainBoundaryPremise:
    parent_cell_id: int
    parent_facet_index: int
    target_face_vertex_ids: tuple[tuple[int, ...], ...]
    oriented_boundary_edges: tuple[tuple[_Point3, _Point3], ...]
    mapped_projected_area: Fraction
    parent_projected_area: Fraction
    degree: int
    boundary_chain_id: str


@dataclass(frozen=True, slots=True)
class ReferenceChainParentPremise:
    parent_cell_id: int
    parent_cell_kind: str
    target_cell_ids: tuple[int, ...]
    charts: tuple[ReferenceChainChartPremise, ...]
    internal_faces: tuple[ReferenceChainInternalFacePremise, ...]
    boundary_faces: tuple[ReferenceChainBoundaryFacePremise, ...]
    boundary_facets: tuple[ReferenceChainBoundaryPremise, ...]
    oriented_degree: int
    premise_id: str


@dataclass(frozen=True, slots=True)
class ReferenceChainSharedStratumPremise:
    source_dimension: int
    target_entity_dimension: int
    source_vertex_ids: tuple[int, ...]
    incident_parent_cell_ids: tuple[int, ...]
    target_entity_ids_by_parent: tuple[tuple[int, tuple[int, ...]], ...]
    root_trace_ids_by_parent: tuple[tuple[int, str], ...]
    physical_trace_id: str
    orientation_signs: tuple[tuple[int, int, int], ...]
    premise_id: str


@dataclass(frozen=True, slots=True)
class ReferenceChainEmbeddingCertificate:
    """Cold-recomputable exact proof of a target complex partitioning parent cells."""

    status: ReferenceChainStatus
    findings: tuple[ReferenceChainEmbeddingFinding, ...]
    parent_mesh_id: str
    parent_topology_id: str
    parent_geometry_id: str
    target_mesh_id: str
    target_topology_id: str
    target_geometry_id: str
    reference_map_bank_id: str
    parent_embedding_id: str
    premises: tuple[ReferenceChainParentPremise, ...]
    cross_parent_premises: tuple[ReferenceChainSharedStratumPremise, ...]
    cross_parent_coverage_complete: bool
    certificate_id: str

    def revalidate(
        self,
        parent_mesh: CellMesh,
        parent_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        target_geometry: CellGeometrySpec,
        reference_maps: Mapping[str, ReferenceCellMapWitnesses | None],
        parent_embedding: GlobalEmbeddingCertificate,
        /,
    ) -> ReferenceChainEmbeddingCertificate:
        rebuilt = certify_reference_chain_embedding(
            parent_mesh,
            parent_geometry,
            target_mesh,
            target_geometry,
            reference_maps,
            parent_embedding,
        )
        if rebuilt.certificate_id != self.certificate_id:
            raise ValueError(
                "Stored reference-chain embedding does not reproduce from its exact operands."
            )
        return rebuilt


@dataclass(frozen=True, slots=True)
class _ReferenceFacet:
    index: int
    vertex_indices: tuple[int, ...]
    vertices: tuple[_Point3, ...]
    outward_normal: _Point3
    offset: Fraction
    oriented_vertices: tuple[_Point3, ...]
    projected_area: Fraction


@dataclass(frozen=True, slots=True)
class _TargetEntityOwner:
    block_name: str
    cell_row: int
    parent_cell_id: int
    local_vertex_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _TargetFaceOwner:
    block_name: str
    cell_row: int
    cell_id: int
    parent_cell_id: int
    local_face_index: int
    local_vertex_indices: tuple[int, ...]
    loop_vertex_rows: tuple[int, ...]


def _scalar_key(value: ExactScalar, /) -> tuple[object, ...]:
    if isinstance(value, Fraction):
        return "fraction", value.numerator, value.denominator
    return "exact", type(value).__qualname__, str(value)


def _point_key(point: _Point3, /) -> tuple[tuple[object, ...], ...]:
    return tuple(_scalar_key(value) for value in point)


def _expression_key(expression: Expression, /) -> tuple[object, ...]:
    if isinstance(expression, RationalPolynomial):
        return (
            "rational",
            tuple(
                (tuple(exponent), _scalar_key(value))
                for exponent, value in sorted(expression.numerator.items())
            ),
            tuple(
                (tuple(exponent), _scalar_key(value))
                for exponent, value in sorted(expression.denominator.items())
            ),
        )
    return (
        "polynomial",
        tuple(
            (tuple(exponent), _scalar_key(value))
            for exponent, value in sorted(expression.items())
        ),
    )


def _as_expression(value: object, /) -> Expression:
    if isinstance(value, RationalPolynomial):
        return value
    if isinstance(value, Mapping):
        return cast(Expression, dict(value))
    raise TypeError("Reference-chart arguments must be exact polynomial expressions.")


def _reference_map_key(witness: ReferenceCellMapWitnesses | None, /) -> object:
    if witness is None:
        return None
    exact_vertices = exact_reference_cell_vertices(witness)
    return {
        "target_cell_ids": tuple(
            int(value) for value in np.asarray(witness.target_cell_ids).tolist()
        ),
        "source_cell_ids": tuple(
            int(value) for value in np.asarray(witness.source_cell_ids).tolist()
        ),
        "target_cell_kind": witness.target_cell_kind,
        "source_cell_kind": witness.source_cell_kind,
        "source_reference_vertices": tuple(
            tuple(tuple(_scalar_key(value) for value in point) for point in cell)
            for cell in exact_vertices
        ),
        "chart_arguments": reference_chart_arguments_identity(
            witness.source_reference_chart_arguments
        ),
    }


def _map_bank_id(
    reference_maps: Mapping[str, ReferenceCellMapWitnesses | None], /
) -> str:
    return canonical_fingerprint(
        {
            "kind": "reference-cell-map-bank",
            "blocks": tuple(
                (name, _reference_map_key(witness))
                for name, witness in sorted(reference_maps.items())
            ),
        }
    )


def _result_status(
    findings: tuple[ReferenceChainEmbeddingFinding, ...], /
) -> ReferenceChainStatus:
    if any(finding.status == "violated" for finding in findings):
        return "violated"
    if findings:
        return "unresolved"
    return "certified"


def _finding(
    check: str,
    status: _ReferenceChainFindingStatus,
    detail: str,
    /,
    *,
    parent_cell_ids: tuple[int, ...] = (),
    target_cell_ids: tuple[int, ...] = (),
) -> ReferenceChainEmbeddingFinding:
    return ReferenceChainEmbeddingFinding(
        check, status, parent_cell_ids, target_cell_ids, detail
    )


def _certify_result(
    parent_mesh: CellMesh,
    parent_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    reference_map_bank_id: str,
    parent_embedding_id: str,
    findings: list[ReferenceChainEmbeddingFinding],
    premises: list[ReferenceChainParentPremise],
    cross_parent_premises: list[ReferenceChainSharedStratumPremise],
    cross_parent_coverage_complete: bool,
    /,
) -> ReferenceChainEmbeddingCertificate:
    findings_tuple = tuple(findings)
    premises_tuple = tuple(premises)
    cross_parent_premises_tuple = tuple(cross_parent_premises)
    status = _result_status(findings_tuple)
    if status == "certified" and not cross_parent_coverage_complete:
        status = "unresolved"
    parent_id = cell_geometry_id(parent_geometry)
    target_id = cell_geometry_id(target_geometry)
    certificate_id = canonical_fingerprint(
        {
            "kind": "reference-chain-embedding-certificate",
            "status": status,
            "parent_mesh": parent_mesh.mesh_id,
            "parent_topology": parent_mesh.topology_id,
            "parent_geometry": parent_id,
            "target_mesh": target_mesh.mesh_id,
            "target_topology": target_mesh.topology_id,
            "target_geometry": target_id,
            "reference_map_bank": reference_map_bank_id,
            "parent_embedding": parent_embedding_id,
            "findings": tuple(
                (
                    finding.check,
                    finding.status,
                    finding.parent_cell_ids,
                    finding.target_cell_ids,
                    finding.detail,
                )
                for finding in findings_tuple
            ),
            "premises": tuple(premise.premise_id for premise in premises_tuple),
            "cross_parent_coverage_complete": cross_parent_coverage_complete,
            "cross_parent_premises": tuple(
                premise.premise_id for premise in cross_parent_premises_tuple
            ),
        }
    )
    return ReferenceChainEmbeddingCertificate(
        status,
        findings_tuple,
        parent_mesh.mesh_id,
        parent_mesh.topology_id,
        parent_id,
        target_mesh.mesh_id,
        target_mesh.topology_id,
        target_id,
        reference_map_bank_id,
        parent_embedding_id,
        premises_tuple,
        cross_parent_premises_tuple,
        cross_parent_coverage_complete,
        certificate_id,
    )


def _sign(value: ExactScalar, /) -> int:
    if isinstance(value, Fraction):
        return (value > 0) - (value < 0)
    sign = getattr(value, "sign", None)
    if callable(sign):
        return int(sign())
    return (value > 0) - (value < 0)


def _domain_for_target(kind: str, /) -> tuple[BernsteinDomain, int] | None:
    topology = reference_cell_topology(kind)
    dimension = topology.dimension
    if kind in ("interval", "triangle", "tetrahedron") or kind.startswith("simplex:"):
        return "simplex", dimension
    if kind == "prism":
        return "prism", dimension
    if kind in ("quadrilateral", "hexahedron") or kind.startswith("tensor:"):
        return "box", dimension
    return None


def _determinant(matrix: tuple[tuple[Expression, ...], ...], /) -> Expression:
    size = len(matrix)
    if size == 1:
        return matrix[0][0]
    if size == 2:
        return subtract(
            multiply(matrix[0][0], matrix[1][1]),
            multiply(matrix[0][1], matrix[1][0]),
        )
    if size == 3:
        a, b, c = matrix[0]
        d, e, f = matrix[1]
        g, h, i = matrix[2]
        return add(
            subtract(
                multiply(a, subtract(multiply(e, i), multiply(f, h))),
                multiply(b, subtract(multiply(d, i), multiply(f, g))),
            ),
            multiply(c, subtract(multiply(d, h), multiply(e, g))),
        )
    raise ValueError("Reference-chain Jacobians support dimensions one through three.")


def _polynomial_control_sign(
    polynomial: Mapping[tuple[int, ...], ExactScalar],
    domain: BernsteinDomain,
    dimension: int,
    /,
) -> tuple[int, tuple[str, ...]] | None:
    controls = bernstein_coefficients(polynomial, domain, dimension)
    if not controls:
        return None
    signs = tuple(_sign(value) for value in controls)
    if all(value > 0 for value in signs):
        return 1, tuple(str(value) for value in controls)
    if all(value < 0 for value in signs):
        return -1, tuple(str(value) for value in controls)
    return None


def _strict_control_sign(
    expression: Expression,
    domain: BernsteinDomain,
    dimension: int,
    /,
) -> tuple[int, tuple[str, ...]] | None:
    if isinstance(expression, RationalPolynomial):
        numerator = _polynomial_control_sign(expression.numerator, domain, dimension)
        denominator = _polynomial_control_sign(expression.denominator, domain, dimension)
        if numerator is None or denominator is None:
            return None
        sign = numerator[0] * denominator[0]
        controls = tuple(f"numerator:{value}" for value in numerator[1]) + tuple(
            f"denominator:{value}" for value in denominator[1]
        )
        return sign, controls
    return _polynomial_control_sign(expression, domain, dimension)


def _chart_jacobian(
    arguments: tuple[Expression, ...], target_kind: str, /
) -> tuple[int, tuple[str, ...]] | None:
    topology = reference_cell_topology(target_kind)
    dimension = topology.dimension
    if dimension != 3 or len(arguments) != 3:
        return None
    if target_kind == "pyramid":
        # Pyramid charts are proved on the collapsed cube: pulled-back
        # pyramid-coordinate derivatives have exactly cancelled apex poles.
        jacobian = tuple(
            tuple(
                _reduced_expression(
                    normalize_pyramid_reference_derivative(expression, axis).expression
                )
                for axis in range(dimension)
            )
            for expression in arguments
        )
        if any(
            isinstance(value, RationalPolynomial) for row in jacobian for value in row
        ):
            return None
        return _strict_control_sign(_determinant(jacobian), "box", dimension)
    domain = _domain_for_target(target_kind)
    if domain is None:
        return None
    bernstein_domain, _ = domain
    jacobian = tuple(
        tuple(derivative(expression, axis) for axis in range(dimension))
        for expression in arguments
    )
    determinant = _determinant(jacobian)
    return _strict_control_sign(determinant, bernstein_domain, dimension)


def _reduced_expression(expression: Expression, /) -> Expression:
    """Exactly divide a rational expression whose denominator divides its numerator.

    Pyramid charts restricted to faces through the apex carry removable
    ``1-w`` denominators; single-divisor lex division leaves a zero remainder
    exactly when the quotient is polynomial. Other expressions are unchanged.
    """
    if not isinstance(expression, RationalPolynomial):
        return expression
    numerator = {key: value for key, value in expression.numerator.items() if value}
    denominator = {key: value for key, value in expression.denominator.items() if value}
    if not denominator or not all(
        isinstance(value, Fraction)
        for value in (*numerator.values(), *denominator.values())
    ):
        return expression
    lead_exponent = max(denominator)
    lead = cast(Fraction, denominator[lead_exponent])
    remainder: dict[tuple[int, ...], Fraction] = {
        key: cast(Fraction, value) for key, value in numerator.items()
    }
    quotient: dict[tuple[int, ...], Fraction] = {}
    while remainder:
        exponent = max(remainder)
        if any(
            power < lead_power
            for power, lead_power in zip(exponent, lead_exponent, strict=True)
        ):
            return expression
        shift = tuple(
            power - lead_power
            for power, lead_power in zip(exponent, lead_exponent, strict=True)
        )
        factor = remainder[exponent] / lead
        quotient[shift] = quotient.get(shift, Fraction(0)) + factor
        for key, value in denominator.items():
            product = tuple(
                first + second for first, second in zip(shift, key, strict=True)
            )
            updated = remainder.get(product, Fraction(0)) - factor * cast(Fraction, value)
            if updated:
                remainder[product] = updated
            else:
                remainder.pop(product, None)
    return {key: value for key, value in quotient.items() if value}


def _trace(expression: Expression, arguments: tuple[Expression, ...], /) -> Expression:
    return _reduced_expression(compose_expression(expression, arguments))


def _point3(values: ArrayLike, /) -> _Point3:
    point = np.asarray(values, dtype=np.float64)
    return (
        Fraction(float(point[0])),
        Fraction(float(point[1])),
        Fraction(float(point[2])),
    )


def _exact_point3(values: tuple[Fraction, ...], /) -> _Point3:
    if len(values) != 3:
        raise ValueError("Reference-chain points must have three coordinates.")
    return values[0], values[1], values[2]


def _sub(first: _Point3, second: _Point3, /) -> _Point3:
    return (
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    )


def _cross(first: _Point3, second: _Point3, /) -> _Point3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _dot(first: _Point3, second: _Point3, /) -> Fraction:
    return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))


def _area_vector(points: tuple[_Point3, ...], /) -> _Point3:
    x = y = z = Fraction(0)
    for index, point in enumerate(points):
        cross = _cross(point, points[(index + 1) % len(points)])
        x += cross[0]
        y += cross[1]
        z += cross[2]
    return x, y, z


def _reference_facets(kind: str, /) -> tuple[_ReferenceFacet, ...] | None:
    topology = reference_cell_topology(kind)
    if topology.dimension != 3:
        return None
    vertices = tuple(
        _point3(point) for point in np.asarray(topology.vertices, dtype=np.float64)
    )
    center: _Point3 = (
        sum((point[0] for point in vertices), Fraction(0)) / len(vertices),
        sum((point[1] for point in vertices), Fraction(0)) / len(vertices),
        sum((point[2] for point in vertices), Fraction(0)) / len(vertices),
    )
    result: list[_ReferenceFacet] = []
    for face_index, raw_face in enumerate(topology.entities[2]):
        face_indices = tuple(int(value) for value in raw_face)
        face_points = tuple(vertices[index] for index in face_indices)
        candidate_normal: _Point3 | None = None
        for first, second, third in combinations(face_points, 3):
            candidate = _cross(_sub(second, first), _sub(third, first))
            if any(candidate):
                candidate_normal = candidate
                break
        if candidate_normal is None:
            return None
        outward_normal = candidate_normal
        offset = _dot(outward_normal, face_points[0])
        interior = _dot(outward_normal, center) - offset
        if not interior:
            return None
        if interior > 0:
            outward_normal = (
                -outward_normal[0],
                -outward_normal[1],
                -outward_normal[2],
            )
            offset = -offset
        oriented = face_points
        if _dot(_area_vector(oriented), outward_normal) < 0:
            oriented = tuple(reversed(oriented))
            face_indices = tuple(reversed(face_indices))
        projected_area = _dot(_area_vector(oriented), outward_normal) / 2
        if projected_area <= 0:
            return None
        result.append(
            _ReferenceFacet(
                face_index,
                face_indices,
                face_points,
                outward_normal,
                offset,
                oriented,
                projected_area,
            )
        )
    return tuple(result)


def _reference_vertices(kind: str, /) -> tuple[_Point3, ...]:
    topology = reference_cell_topology(kind)
    return tuple(
        _point3(point) for point in np.asarray(topology.vertices, dtype=np.float64)
    )


def _face_arguments(
    target_kind: str,
    cell_vertex_rows: tuple[int, ...],
    face_vertex_rows: tuple[int, ...],
    reference_vertices: tuple[_Point3, ...],
    /,
) -> tuple[Expression, ...] | None:
    if reference_cell_topology(target_kind).dimension != 3 or len(
        face_vertex_rows
    ) not in (3, 4):
        return None
    local_by_row = {
        int(vertex_row): local for local, vertex_row in enumerate(cell_vertex_rows)
    }
    try:
        points = tuple(
            reference_vertices[local_by_row[int(vertex_row)]]
            for vertex_row in face_vertex_rows
        )
    except KeyError:
        return None
    first, second, third = points[:3]
    tangent_s = _sub(second, first)
    tangent_t = _sub(third, first)
    if len(points) == 4:
        fourth = points[3]
        if third != tuple(second[axis] + fourth[axis] - first[axis] for axis in range(3)):
            return None
        tangent_t = _sub(fourth, first)
    s, t = axes(2)
    return tuple(
        add(
            constant(first[axis], 2),
            add(scale(s, tangent_s[axis]), scale(t, tangent_t[axis])),
        )
        for axis in range(3)
    )


def _edge_arguments(
    target_kind: str,
    cell_vertex_rows: tuple[int, ...],
    edge_vertex_rows: tuple[int, int],
    reference_vertices: tuple[_Point3, ...],
    /,
) -> tuple[Expression, ...] | None:
    if reference_cell_topology(target_kind).dimension != 3:
        return None
    local_by_row = {
        int(vertex_row): local for local, vertex_row in enumerate(cell_vertex_rows)
    }
    try:
        first = reference_vertices[local_by_row[edge_vertex_rows[0]]]
        last = reference_vertices[local_by_row[edge_vertex_rows[1]]]
    except KeyError:
        return None
    parameter = axes(1)[0]
    tangent = _sub(last, first)
    return tuple(
        add(
            constant(first[axis], 1),
            scale(parameter, tangent[axis]),
        )
        for axis in range(3)
    )


def _bilinear(expression: Expression, /) -> bool:
    if isinstance(expression, RationalPolynomial):
        return False
    return all(
        len(exponent) == 2 and exponent[0] <= 1 and exponent[1] <= 1
        for exponent in expression
    )


def _affine_triangle(expression: Expression, /) -> bool:
    if isinstance(expression, RationalPolynomial):
        return False
    return all(len(exponent) == 2 and sum(exponent) <= 1 for exponent in expression)


def _plane_residual(
    trace: tuple[Expression, ...], normal: _Point3, offset: Fraction, /
) -> Expression:
    linear = linear_combinations(trace, (normal,))[0]
    return add(linear, constant(-offset, 2))


def _canonical_cycle(vertices: tuple[int, ...], /) -> tuple[int, ...]:
    if not vertices:
        return ()
    candidates = []
    for sequence in (vertices, tuple(reversed(vertices))):
        candidates.extend(
            sequence[index:] + sequence[:index] for index in range(len(sequence))
        )
    return min(candidates)


def _cycle_orientation(
    candidate: tuple[int, ...], canonical: tuple[int, ...], /
) -> int | None:
    if len(candidate) != len(canonical):
        return None
    size = len(candidate)
    for shift in range(size):
        if all(
            candidate[(index + shift) % size] == canonical[index] for index in range(size)
        ):
            return 1
    reversed_candidate = tuple(reversed(candidate))
    for shift in range(size):
        if all(
            reversed_candidate[(index + shift) % size] == canonical[index]
            for index in range(size)
        ):
            return -1
    return None


def _point_on_segment(
    point: _Point3, first: _Point3, last: _Point3, /
) -> tuple[bool, Fraction | None]:
    delta = _sub(last, first)
    axis = next((index for index, value in enumerate(delta) if value), None)
    if axis is None:
        return False, None
    parameter = (point[axis] - first[axis]) / delta[axis]
    if not 0 <= parameter <= 1:
        return False, parameter
    if any(point[index] != first[index] + parameter * delta[index] for index in range(3)):
        return False, parameter
    return True, parameter


def _oriented_edge_key(
    first: _Point3, last: _Point3, /
) -> tuple[tuple[_Point3, _Point3], int]:
    edge = (first, last) if first <= last else (last, first)
    return edge, 1 if (first, last) == edge else -1


def _boundary_chain(
    root_loop: tuple[_Point3, ...],
    face_loops: tuple[tuple[_Point3, ...], ...],
    /,
) -> tuple[bool, tuple[tuple[_Point3, _Point3], ...]]:
    actual: dict[tuple[_Point3, _Point3], int] = defaultdict(int)
    all_points: set[_Point3] = set()
    for loop in face_loops:
        all_points.update(loop)
        for index, first in enumerate(loop):
            last = loop[(index + 1) % len(loop)]
            edge, orientation = _oriented_edge_key(first, last)
            actual[edge] += orientation
    actual = {edge: sign for edge, sign in actual.items() if sign}

    expected: dict[tuple[_Point3, _Point3], int] = defaultdict(int)
    for index, first in enumerate(root_loop):
        last = root_loop[(index + 1) % len(root_loop)]
        parameters: dict[Fraction, _Point3] = {Fraction(0): first, Fraction(1): last}
        for point in all_points:
            on_edge, parameter = _point_on_segment(point, first, last)
            if on_edge and parameter is not None:
                parameters[parameter] = point
        ordered = tuple(parameters[key] for key in sorted(parameters))
        for start, end in zip(ordered, ordered[1:]):
            edge, orientation = _oriented_edge_key(start, end)
            expected[edge] += orientation
    expected = {edge: sign for edge, sign in expected.items() if sign}
    chain = tuple(
        (edge[0], edge[1]) if sign > 0 else (edge[1], edge[0])
        for edge, sign in sorted(actual.items())
    )
    return actual == expected, chain


def _reference_map_identity_matches(
    supplied: Mapping[str, ReferenceCellMapWitnesses | None],
    retained: Mapping[str, ReferenceCellMapWitnesses | None],
    /,
) -> bool:
    if set(supplied) != set(retained):
        return False
    return all(
        _reference_map_key(supplied[name]) == _reference_map_key(retained[name])
        for name in supplied
    )


def _mesh_entity_bank(
    mesh: CellMesh, dimension: int, /
) -> tuple[tuple[int, tuple[int, ...]], ...] | None:
    entity_ids = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
    if dimension == 0:
        if entity_ids.shape != mesh.vertex_global_ids.shape:
            return None
        return tuple(
            (int(identifier), (row,))
            for row, identifier in enumerate(entity_ids.tolist())
        )
    connectivity = mesh.connectivity
    if dimension == 1:
        raw_rows = getattr(connectivity, "edges", None)
        if raw_rows is None:
            return None
        rows = tuple(
            tuple(int(value) for value in row)
            for row in np.asarray(raw_rows, dtype=np.int64).tolist()
        )
    elif dimension == 2:
        raw_rows = getattr(connectivity, "faces", None)
        if raw_rows is not None:
            rows = tuple(
                tuple(int(value) for value in row)
                for row in np.asarray(raw_rows, dtype=np.int64).tolist()
            )
        else:
            offsets = getattr(connectivity, "face_vertex_offsets", None)
            values = getattr(connectivity, "face_vertex_values", None)
            if offsets is None or values is None:
                return None
            face_offsets = np.asarray(offsets, dtype=np.int64)
            face_values = np.asarray(values, dtype=np.int64)
            rows = tuple(
                tuple(
                    int(value)
                    for value in face_values[
                        int(face_offsets[row]) : int(face_offsets[row + 1])
                    ].tolist()
                )
                for row in range(face_offsets.size - 1)
            )
    else:
        return None
    if len(rows) != entity_ids.size:
        return None
    return tuple((int(entity_ids[row]), vertices) for row, vertices in enumerate(rows))


def _parent_reference_strata(
    parent_mesh: CellMesh, /
) -> tuple[
    dict[tuple[int, tuple[int, ...]], set[int]],
    dict[tuple[int, int, tuple[int, ...]], tuple[_Point3, ...]],
    dict[tuple[int, int, tuple[int, ...]], int],
    dict[tuple[int, int, tuple[int, ...]], tuple[int, ...]],
]:
    strata: dict[tuple[int, tuple[int, ...]], set[int]] = defaultdict(set)
    points_by_parent_stratum: dict[
        tuple[int, int, tuple[int, ...]], tuple[_Point3, ...]
    ] = {}
    face_index_by_parent_stratum: dict[tuple[int, int, tuple[int, ...]], int] = {}
    face_cycle_by_parent_stratum: dict[
        tuple[int, int, tuple[int, ...]], tuple[int, ...]
    ] = {}
    vertex_ids = np.asarray(parent_mesh.vertex_global_ids, dtype=np.int64)
    for block in parent_mesh.blocks:
        topology = reference_cell_topology(block.cell_kind)
        if topology.dimension != 3:
            continue
        reference_vertices = _reference_vertices(block.cell_kind)
        cells = np.asarray(block.vertices, dtype=np.int64)
        cell_ids = np.asarray(block.global_ids, dtype=np.int64)
        for cell_row in range(len(cells)):
            parent_cell_id = int(cell_ids[cell_row])
            source_vertex_ids = tuple(
                int(vertex_ids[int(cells[cell_row, local_vertex])])
                for local_vertex in range(len(reference_vertices))
            )
            reference_by_source_vertex = {
                source_id: reference_vertices[local_vertex]
                for local_vertex, source_id in enumerate(source_vertex_ids)
            }
            for dimension in range(3):
                for entity_row, local_entity in enumerate(topology.entities[dimension]):
                    entity_source_ids = tuple(
                        sorted(
                            source_vertex_ids[int(local_vertex)]
                            for local_vertex in local_entity
                        )
                    )
                    key = (dimension, entity_source_ids)
                    strata[key].add(parent_cell_id)
                    stratum_key = (parent_cell_id, dimension, entity_source_ids)
                    points_by_parent_stratum[stratum_key] = tuple(
                        reference_by_source_vertex[source_id]
                        for source_id in entity_source_ids
                    )
                    if dimension == 2:
                        face_index_by_parent_stratum[stratum_key] = entity_row
                        face_cycle_by_parent_stratum[stratum_key] = _canonical_cycle(
                            tuple(
                                source_vertex_ids[int(local_vertex)]
                                for local_vertex in local_entity
                            )
                        )
    return (
        dict(strata),
        points_by_parent_stratum,
        face_index_by_parent_stratum,
        face_cycle_by_parent_stratum,
    )


def _face_basis(
    source_vertex_ids: tuple[int, ...],
    source_face_cycle: tuple[int, ...],
    points_by_source_id: Mapping[int, _Point3],
    /,
) -> tuple[_Point3, _Point3, _Point3] | None:
    if len(source_vertex_ids) != len(source_face_cycle):
        return None
    try:
        points = tuple(
            points_by_source_id[identifier] for identifier in source_face_cycle
        )
    except KeyError:
        return None
    if len(points) == 3:
        origin, first, second = points
        return origin, _sub(first, origin), _sub(second, origin)
    if len(points) == 4:
        origin, first, opposite, fourth = points
        tangent_s = _sub(first, origin)
        tangent_t = _sub(fourth, origin)
        if opposite != tuple(
            origin[axis] + tangent_s[axis] + tangent_t[axis] for axis in range(3)
        ):
            return None
        return origin, tangent_s, tangent_t
    return None


def _face_coordinates(
    point: _Point3,
    basis: tuple[_Point3, _Point3, _Point3],
    /,
) -> tuple[Fraction, Fraction] | None:
    origin, tangent_s, tangent_t = basis
    axis_pair = next(
        (
            (first, second)
            for first, second in combinations(range(3), 2)
            if tangent_s[first] * tangent_t[second] - tangent_s[second] * tangent_t[first]
        ),
        None,
    )
    if axis_pair is None:
        return None
    first, second = axis_pair
    determinant = (
        tangent_s[first] * tangent_t[second] - tangent_s[second] * tangent_t[first]
    )
    delta_first = point[first] - origin[first]
    delta_second = point[second] - origin[second]
    parameter_s = (
        delta_first * tangent_t[second] - delta_second * tangent_t[first]
    ) / determinant
    parameter_t = (
        tangent_s[first] * delta_second - tangent_s[second] * delta_first
    ) / determinant
    if any(
        point[axis]
        != origin[axis] + parameter_s * tangent_s[axis] + parameter_t * tangent_t[axis]
        for axis in range(3)
    ):
        return None
    return parameter_s, parameter_t


def _point_in_source_stratum(
    point: _Point3,
    source_dimension: int,
    source_vertex_ids: tuple[int, ...],
    points: tuple[_Point3, ...],
    source_face_cycle: tuple[int, ...],
    /,
) -> tuple[Fraction, ...] | None:
    if source_dimension == 0:
        return () if len(points) == 1 and point == points[0] else None
    if source_dimension == 1:
        if len(points) != 2:
            return None
        on_segment, parameter = _point_on_segment(point, points[0], points[1])
        return (parameter,) if on_segment and parameter is not None else None
    if source_dimension != 2:
        return None
    basis = _face_basis(
        source_vertex_ids,
        source_face_cycle,
        dict(zip(source_vertex_ids, points, strict=True)),
    )
    if basis is None:
        return None
    coordinates = _face_coordinates(point, basis)
    if coordinates is None:
        return None
    first, second = coordinates
    if len(source_vertex_ids) == 3:
        return coordinates if first >= 0 and second >= 0 and first + second <= 1 else None
    return coordinates if 0 <= first <= 1 and 0 <= second <= 1 else None


def _trace_coordinates_on_source_stratum(
    trace: tuple[Expression, ...],
    parameter_dimension: int,
    source_dimension: int,
    source_vertex_ids: tuple[int, ...],
    points: tuple[_Point3, ...],
    source_face_cycle: tuple[int, ...],
    /,
) -> tuple[Expression, ...] | None:
    if len(trace) != 3 or source_dimension not in (1, 2):
        return None
    if source_dimension == 1:
        if len(points) != 2:
            return None
        origin, endpoint = points
        tangent = _sub(endpoint, origin)
        axis = next((index for index, value in enumerate(tangent) if value), None)
        if axis is None:
            return None
        parameter = scale(
            subtract(trace[axis], constant(origin[axis], parameter_dimension)),
            Fraction(1, 1) / tangent[axis],
        )
        if not all(
            expression_equal(
                trace[index],
                add(
                    constant(origin[index], parameter_dimension),
                    scale(parameter, tangent[index]),
                ),
            )
            for index in range(3)
        ):
            return None
        return (parameter,)
    basis = _face_basis(
        source_vertex_ids,
        source_face_cycle,
        dict(zip(source_vertex_ids, points, strict=True)),
    )
    if basis is None:
        return None
    origin, tangent_s, tangent_t = basis
    axis_pair = next(
        (
            (first, second)
            for first, second in combinations(range(3), 2)
            if tangent_s[first] * tangent_t[second] - tangent_s[second] * tangent_t[first]
        ),
        None,
    )
    if axis_pair is None:
        return None
    first, second = axis_pair
    determinant = (
        tangent_s[first] * tangent_t[second] - tangent_s[second] * tangent_t[first]
    )
    delta_first = subtract(trace[first], constant(origin[first], parameter_dimension))
    delta_second = subtract(trace[second], constant(origin[second], parameter_dimension))
    parameter_s = scale(
        subtract(
            scale(delta_first, tangent_t[second]),
            scale(delta_second, tangent_t[first]),
        ),
        Fraction(1, 1) / determinant,
    )
    parameter_t = scale(
        subtract(
            scale(delta_second, tangent_s[first]),
            scale(delta_first, tangent_s[second]),
        ),
        Fraction(1, 1) / determinant,
    )
    if not all(
        expression_equal(
            trace[axis],
            add(
                constant(origin[axis], parameter_dimension),
                add(
                    scale(parameter_s, tangent_s[axis]),
                    scale(parameter_t, tangent_t[axis]),
                ),
            ),
        )
        for axis in range(3)
    ):
        return None
    return parameter_s, parameter_t


def certify_reference_chain_embedding(
    parent_mesh: CellMesh,
    parent_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    reference_maps: Mapping[str, ReferenceCellMapWitnesses | None],
    parent_embedding: GlobalEmbeddingCertificate,
    /,
    *,
    _validated_parent_embedding: ValidatedGlobalEmbedding | None = None,
) -> ReferenceChainEmbeddingCertificate:
    """Prove a hexahedral reference-cell chain is a degree-one parent partition.

    The helper revalidates all operands and chart maps on every call. It never
    consumes a saved verdict or an artifact ID as proof. The exact root-chart
    Jacobians, internal traces, planar exterior patches, oriented boundary
    chains, and area supplements remain in the returned premise payload.
    """
    if not isinstance(parent_mesh, CellMesh) or not isinstance(target_mesh, CellMesh):
        raise TypeError(
            "Reference-chain proof requires parent and target CellMesh values."
        )
    if not isinstance(parent_geometry, CellGeometrySpec) or not isinstance(
        target_geometry, CellGeometrySpec
    ):
        raise TypeError(
            "Reference-chain proof requires parent and target CellGeometrySpec values."
        )
    if not isinstance(reference_maps, Mapping):
        raise TypeError("reference_maps must be a block-name mapping.")
    map_bank_id = _map_bank_id(reference_maps)
    findings: list[ReferenceChainEmbeddingFinding] = []
    premises: list[ReferenceChainParentPremise] = []
    cross_parent_premises: list[ReferenceChainSharedStratumPremise] = []
    cross_parent_coverage_complete = True
    parent_geometry_id = cell_geometry_id(parent_geometry)
    parent_embedding_id = ""

    def finish() -> ReferenceChainEmbeddingCertificate:
        return _certify_result(
            parent_mesh,
            parent_geometry,
            target_mesh,
            target_geometry,
            map_bank_id,
            parent_embedding_id,
            findings,
            premises,
            cross_parent_premises,
            cross_parent_coverage_complete,
        )

    try:
        (
            parent_source_strata,
            parent_stratum_points,
            parent_face_indices,
            parent_face_cycles,
        ) = _parent_reference_strata(parent_mesh)
    except (TypeError, ValueError, IndexError, KeyError) as error:
        findings.append(
            _finding(
                "parent_reference_strata",
                "unresolved",
                f"Exact parent reference strata are unavailable: {error}",
            )
        )
        cross_parent_coverage_complete = False
        return finish()
    shared_parent_strata = tuple(
        (dimension, source_ids, tuple(sorted(parent_ids)))
        for (dimension, source_ids), parent_ids in sorted(parent_source_strata.items())
        if len(parent_ids) > 1
    )
    cross_parent_coverage_complete = not shared_parent_strata

    from ._global_embedding import (
        GlobalEmbeddingCertificate as CanonicalGlobalEmbeddingCertificate,
        ValidatedGlobalEmbedding,
    )

    if not isinstance(parent_embedding, CanonicalGlobalEmbeddingCertificate):
        findings.append(
            _finding(
                "parent_global_embedding_type",
                "violated",
                "Parent injection must be a canonical GlobalEmbeddingCertificate bound to the supplied operands.",
            )
        )
        return finish()
    if _validated_parent_embedding is not None:
        if (
            type(_validated_parent_embedding) is not ValidatedGlobalEmbedding
            or _validated_parent_embedding.certificate is not parent_embedding
        ):
            raise ValueError(
                "Reference-chain proof requires its actual validated embedding."
            )
        _validated_parent_embedding.require(
            parent_mesh, parent_geometry, parent_embedding.validity
        )
    try:
        parent_embedding.require_current(parent_mesh, parent_geometry)
        rebuilt_parent_embedding = (
            CanonicalGlobalEmbeddingCertificate.from_stored_fields(parent_embedding)
            if _validated_parent_embedding is None
            else parent_embedding
        )
    except (TypeError, ValueError, IndexError) as error:
        findings.append(
            _finding(
                "parent_global_embedding_currentness",
                "violated",
                f"Parent global proof does not revalidate from its actual operands: {error}",
            )
        )
        return finish()
    if rebuilt_parent_embedding.certificate_id != parent_embedding.certificate_id:
        findings.append(
            _finding(
                "parent_global_embedding_replay",
                "violated",
                "Canonical parent global proof changed on full cold replay.",
            )
        )
        return finish()
    parent_embedding = rebuilt_parent_embedding
    parent_embedding_id = parent_embedding.certificate_id
    if parent_embedding.status != "certified":
        finding_status: _ReferenceChainFindingStatus = (
            "violated" if parent_embedding.status == "violated" else "unresolved"
        )
        findings.append(
            _finding(
                "parent_global_embedding",
                finding_status,
                f"Parent global-embedding status is {parent_embedding.status!r}.",
            )
        )
        return finish()

    restriction = target_geometry.restriction_source
    if restriction is None:
        findings.append(
            _finding(
                "reference_map_binding",
                "unresolved",
                "Target geometry has no retained reference-cell restriction source.",
            )
        )
        return finish()
    if (
        restriction.source_geometry_id != parent_geometry_id
        or restriction.source_topology_id != parent_mesh.topology_id
    ):
        findings.append(
            _finding(
                "reference_map_binding",
                "violated",
                "Restriction source geometry/topology does not bind the actual parent operands.",
            )
        )
        return finish()
    retained_maps = restriction.block_parent_reference_maps
    if not _reference_map_identity_matches(reference_maps, retained_maps):
        findings.append(
            _finding(
                "reference_map_binding",
                "violated",
                "Supplied reference maps differ from the target geometry's retained map bank.",
            )
        )
        return finish()

    try:
        parent_embeddings = prepare_embedding_cells(parent_mesh, parent_geometry)
        target_embeddings = prepare_embedding_cells(target_mesh, target_geometry)
    except (TypeError, ValueError, IndexError, ZeroDivisionError) as error:
        findings.append(
            _finding(
                "reference_map_recomposition",
                "unresolved",
                f"Exact parent/target coordinate expressions are unavailable: {error}",
            )
        )
        return finish()

    parent_embedding_by_id = {cell.global_id: cell for cell in parent_embeddings}
    target_embedding_by_row = {
        (cell.block_name, cell.local_index): cell for cell in target_embeddings
    }
    parent_cell_location: dict[int, tuple[str, int, int]] = {}
    for block_index, block in enumerate(parent_mesh.blocks):
        for row, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64)):
            key = int(identifier)
            if key in parent_cell_location:
                findings.append(
                    _finding(
                        "parent_cell_identity",
                        "violated",
                        f"Parent cell ID {key} occurs in multiple blocks.",
                        parent_cell_ids=(key,),
                    )
                )
            parent_cell_location[key] = (block.name, block_index, row)

    if set(reference_maps) != {block.name for block in target_mesh.blocks}:
        findings.append(
            _finding(
                "reference_map_coverage",
                "violated",
                "Reference-map blocks do not exactly cover target mesh blocks.",
            )
        )
        return finish()

    try:
        target_topologies = {
            block.name: reference_cell_topology(block.cell_kind)
            for block in target_mesh.blocks
        }
    except (TypeError, ValueError) as error:
        findings.append(
            _finding(
                "target_reference_family",
                "unresolved",
                f"Target cells require supported fixed reference families: {error}",
            )
        )
        return finish()
    unsupported_target_families = tuple(
        block.cell_kind
        for block in target_mesh.blocks
        if target_topologies[block.name].dimension != 3
        or any(
            len(face) not in (3, 4) for face in target_topologies[block.name].entities[2]
        )
    )
    if unsupported_target_families:
        findings.append(
            _finding(
                "target_reference_family",
                "unresolved",
                "Forward reference-chain proof requires fixed 3D target families with triangular or quadrilateral facets; unsupported "
                f"{tuple(sorted(set(unsupported_target_families)))}.",
            )
        )
        return finish()
    face_owners: dict[tuple[int, ...], list[_TargetFaceOwner]] = defaultdict(list)
    chart_arguments_by_row: dict[tuple[str, int], tuple[Expression, ...]] = {}
    exact_corners_by_row: dict[tuple[str, int], tuple[_Point3, ...]] = {}
    chart_sign_by_row: dict[tuple[str, int], int] = {}
    parent_ids_by_row: dict[tuple[str, int], int] = {}
    parent_reference_by_target_vertex: dict[tuple[int, int], _Point3] = {}
    target_physical_by_vertex: dict[int, tuple[ExactScalar, ...]] = {}
    chart_premise_by_parent: dict[int, list[ReferenceChainChartPremise]] = defaultdict(
        list
    )

    for block in target_mesh.blocks:
        restriction_block_index = restriction.block_names.index(block.name)
        witness = reference_maps.get(block.name)
        if witness is None:
            findings.append(
                _finding(
                    "reference_map_coverage",
                    "unresolved",
                    f"Target block {block.name!r} has no exact reference map.",
                    target_cell_ids=tuple(
                        int(value) for value in np.asarray(block.global_ids).tolist()
                    ),
                )
            )
            continue
        try:
            validate_reference_cell_map_witnesses(witness)
            exact_reference_vertices = tuple(
                tuple(_exact_point3(point) for point in cell)
                for cell in exact_reference_cell_vertices(witness)
            )
        except (TypeError, ValueError, ZeroDivisionError) as error:
            findings.append(
                _finding(
                    "reference_map_validity",
                    "violated",
                    f"Reference map for target block {block.name!r} is invalid: {error}",
                    target_cell_ids=tuple(
                        int(value) for value in np.asarray(block.global_ids).tolist()
                    ),
                )
            )
            continue
        if (
            witness.target_cell_kind != block.cell_kind
            or np.asarray(witness.target_cell_ids).shape != (block.cell_count,)
            or not np.array_equal(
                np.asarray(witness.target_cell_ids, dtype=np.int64),
                np.arange(block.cell_count, dtype=np.int64),
            )
            or np.asarray(witness.source_cell_ids).shape != (block.cell_count,)
            or len(exact_reference_vertices) != block.cell_count
        ):
            findings.append(
                _finding(
                    "reference_map_rows",
                    "violated",
                    f"Reference map rows do not align with target block {block.name!r}.",
                    target_cell_ids=tuple(
                        int(value) for value in np.asarray(block.global_ids).tolist()
                    ),
                )
            )
            continue
        chart_rows = witness.source_reference_chart_arguments
        if chart_rows is None:
            findings.append(
                _finding(
                    "reference_map_arguments",
                    "unresolved",
                    f"Target block {block.name!r} lacks exact reference-chart arguments.",
                    target_cell_ids=tuple(
                        int(value) for value in np.asarray(block.global_ids).tolist()
                    ),
                )
            )
            continue
        if len(chart_rows) != block.cell_count:
            findings.append(
                _finding(
                    "reference_map_arguments",
                    "violated",
                    f"Target block {block.name!r} chart rows do not align with cells.",
                )
            )
            continue
        parent_ids = np.asarray(witness.source_cell_ids, dtype=np.int64)
        restriction_parent_ids = np.asarray(
            restriction.parent_cell_ids[restriction_block_index], dtype=np.int64
        )
        if not np.array_equal(parent_ids, restriction_parent_ids):
            findings.append(
                _finding(
                    "reference_map_parent_ids",
                    "violated",
                    f"Reference-map parent IDs disagree for target block {block.name!r}.",
                    parent_cell_ids=tuple(int(value) for value in parent_ids.tolist()),
                )
            )
            continue
        target_rows = np.asarray(block.vertices, dtype=np.int64)
        for row, target_id_value in enumerate(
            np.asarray(block.global_ids, dtype=np.int64)
        ):
            target_id = int(target_id_value)
            parent_id = int(parent_ids[row])
            parent_location = parent_cell_location.get(parent_id)
            parent_cell = parent_embedding_by_id.get(parent_id)
            target_cell = target_embedding_by_row.get((block.name, row))
            if parent_location is None or parent_cell is None:
                findings.append(
                    _finding(
                        "reference_map_parent_cell",
                        "violated",
                        f"Reference map names absent parent cell ID {parent_id}.",
                        parent_cell_ids=(parent_id,),
                        target_cell_ids=(target_id,),
                    )
                )
                continue
            _, parent_block_index, _ = parent_location
            parent_block = parent_mesh.blocks[parent_block_index]
            if witness.source_cell_kind != parent_block.cell_kind:
                findings.append(
                    _finding(
                        "reference_map_parent_kind",
                        "violated",
                        f"Reference map source kind {witness.source_cell_kind!r} disagrees with parent cell kind {parent_block.cell_kind!r}.",
                        parent_cell_ids=(parent_id,),
                        target_cell_ids=(target_id,),
                    )
                )
                continue
            if target_cell is None:
                findings.append(
                    _finding(
                        "target_cell_map",
                        "unresolved",
                        f"Exact target-cell map is absent for cell ID {target_id}.",
                        parent_cell_ids=(parent_id,),
                        target_cell_ids=(target_id,),
                    )
                )
                continue
            arguments = tuple(_as_expression(value) for value in chart_rows[row])
            if len(arguments) != 3:
                findings.append(
                    _finding(
                        "reference_map_dimension",
                        "violated",
                        f"Chart for target cell ID {target_id} does not have three source coordinates.",
                        parent_cell_ids=(parent_id,),
                        target_cell_ids=(target_id,),
                    )
                )
                continue
            parent_result = reference_chart_composition_equal(
                parent_cell, target_cell, arguments
            )
            if parent_result is False:
                findings.append(
                    _finding(
                        "target_parent_map_identity",
                        "violated",
                        f"Target cell ID {target_id} is not the exact parent-map composition through its retained chart.",
                        parent_cell_ids=(parent_id,),
                        target_cell_ids=(target_id,),
                    )
                )
                continue
            if parent_result is None:
                findings.append(
                    _finding(
                        "target_parent_map_identity",
                        "unresolved",
                        f"Exact target-parent composition is unavailable for target cell ID {target_id}.",
                        parent_cell_ids=(parent_id,),
                        target_cell_ids=(target_id,),
                    )
                )
                continue
            jacobian = _chart_jacobian(arguments, block.cell_kind)
            if jacobian is None:
                findings.append(
                    _finding(
                        "reference_chart_jacobian",
                        "unresolved",
                        f"Whole-cell Jacobian sign is not certified for target cell ID {target_id}.",
                        parent_cell_ids=(parent_id,),
                        target_cell_ids=(target_id,),
                    )
                )
                continue
            jacobian_sign, jacobian_controls = jacobian
            if jacobian_sign <= 0:
                findings.append(
                    _finding(
                        "reference_chart_jacobian",
                        "violated",
                        f"Target-to-parent chart for target cell ID {target_id} is not positively oriented.",
                        parent_cell_ids=(parent_id,),
                        target_cell_ids=(target_id,),
                    )
                )
                continue
            chart_arguments_by_row[(block.name, row)] = arguments
            exact_corners_by_row[(block.name, row)] = exact_reference_vertices[row]
            chart_sign_by_row[(block.name, row)] = jacobian_sign
            parent_ids_by_row[(block.name, row)] = parent_id
            for local_vertex, target_vertex_row in enumerate(target_rows[row].tolist()):
                target_vertex_row = int(target_vertex_row)
                root_point = exact_reference_vertices[row][local_vertex]
                root_key = (parent_id, target_vertex_row)
                previous_root_point = parent_reference_by_target_vertex.get(root_key)
                if previous_root_point is not None and previous_root_point != root_point:
                    findings.append(
                        _finding(
                            "same_parent_vertex_root_trace",
                            "violated",
                            f"Target vertex row {target_vertex_row} has conflicting root-reference images in parent {parent_id}.",
                            parent_cell_ids=(parent_id,),
                            target_cell_ids=(target_id,),
                        )
                    )
                else:
                    parent_reference_by_target_vertex[root_key] = root_point
                physical_point = tuple(target_cell.corner_images[local_vertex])
                previous_physical_point = target_physical_by_vertex.get(target_vertex_row)
                if (
                    previous_physical_point is not None
                    and previous_physical_point != physical_point
                ):
                    findings.append(
                        _finding(
                            "shared_target_vertex_physical_trace",
                            "violated",
                            f"Target vertex row {target_vertex_row} has conflicting exact physical images.",
                            parent_cell_ids=(parent_id,),
                            target_cell_ids=(target_id,),
                        )
                    )
                else:
                    target_physical_by_vertex[target_vertex_row] = physical_point
            chart_key = canonical_fingerprint(
                {
                    "target_block": block.name,
                    "target_cell_id": target_id,
                    "parent_cell_id": parent_id,
                    "arguments": tuple(_expression_key(value) for value in arguments),
                }
            )
            chart_premise_by_parent[parent_id].append(
                ReferenceChainChartPremise(
                    parent_id,
                    block.name,
                    target_id,
                    witness.source_cell_kind,
                    witness.target_cell_kind,
                    arguments,
                    exact_reference_vertices[row],
                    jacobian_sign,
                    jacobian_controls,
                    chart_key,
                )
            )
            cell_vertices = tuple(int(value) for value in target_rows[row].tolist())
            for local_face_index, raw_face in enumerate(
                target_topologies[block.name].entities[2]
            ):
                local_vertices = tuple(int(value) for value in raw_face)
                loop_rows = tuple(cell_vertices[index] for index in local_vertices)
                face_owners[tuple(sorted(loop_rows))].append(
                    _TargetFaceOwner(
                        block.name,
                        row,
                        target_id,
                        parent_id,
                        local_face_index,
                        local_vertices,
                        loop_rows,
                    )
                )

    if findings:
        return finish()

    target_vertex_ids = np.asarray(target_mesh.vertex_global_ids, dtype=np.int64)
    target_reference_vertices_by_kind = {
        block.cell_kind: _reference_vertices(block.cell_kind)
        for block in target_mesh.blocks
    }
    internal_premises_by_parent: dict[int, list[ReferenceChainInternalFacePremise]] = (
        defaultdict(list)
    )
    boundary_faces_by_parent: dict[int, list[ReferenceChainBoundaryFacePremise]] = (
        defaultdict(list)
    )
    boundary_loops_by_parent_facet: dict[tuple[int, int], list[tuple[_Point3, ...]]] = (
        defaultdict(list)
    )
    boundary_target_faces_by_parent_facet: dict[
        tuple[int, int], list[tuple[int, ...]]
    ] = defaultdict(list)
    connected_cells_by_parent: dict[int, dict[int, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )

    for face_key, owners in sorted(face_owners.items()):
        if len(owners) > 2:
            findings.append(
                _finding(
                    "target_face_incidence",
                    "violated",
                    "Target mesh has a face with more than two incident cells.",
                    parent_cell_ids=tuple(
                        sorted({owner.parent_cell_id for owner in owners})
                    ),
                    target_cell_ids=tuple(sorted(owner.cell_id for owner in owners)),
                )
            )
            continue
        if len(owners) == 1:
            selected_owners = tuple(owners)
        else:
            selected_owners = tuple(owners)
        canonical_loop_rows = owners[0].loop_vertex_rows
        for owner in selected_owners:
            if (
                len(owners) == 2
                and owners[0].parent_cell_id == owners[1].parent_cell_id
                and owner is not owners[0]
            ):
                continue
            if len(owners) == 2 and owners[0].parent_cell_id != owners[1].parent_cell_id:
                # A source-cell interface is an exterior facet of both reference
                # parents; exact physical traces must still agree on the target.
                other = owners[1] if owner is owners[0] else owners[0]
                owner_embedding = target_embedding_by_row[
                    (owner.block_name, owner.cell_row)
                ]
                other_embedding = target_embedding_by_row[
                    (other.block_name, other.cell_row)
                ]
                owner_reference = target_reference_vertices_by_kind[owner_embedding.kind]
                other_reference = target_reference_vertices_by_kind[other_embedding.kind]
                owner_face_arguments = _face_arguments(
                    owner_embedding.kind,
                    tuple(
                        int(value)
                        for value in np.asarray(
                            target_mesh.blocks[
                                next(
                                    index
                                    for index, block in enumerate(target_mesh.blocks)
                                    if block.name == owner.block_name
                                )
                            ].vertices[owner.cell_row],
                            dtype=np.int64,
                        ).tolist()
                    ),
                    canonical_loop_rows,
                    owner_reference,
                )
                other_face_arguments = _face_arguments(
                    other_embedding.kind,
                    tuple(
                        int(value)
                        for value in np.asarray(
                            target_mesh.blocks[
                                next(
                                    index
                                    for index, block in enumerate(target_mesh.blocks)
                                    if block.name == other.block_name
                                )
                            ].vertices[other.cell_row],
                            dtype=np.int64,
                        ).tolist()
                    ),
                    canonical_loop_rows,
                    other_reference,
                )
                if owner_face_arguments is None or other_face_arguments is None:
                    findings.append(
                        _finding(
                            "parent_interface_trace",
                            "unresolved",
                            "Exact target reference-face charts are unavailable on a parent interface.",
                            parent_cell_ids=tuple(
                                sorted((owner.parent_cell_id, other.parent_cell_id))
                            ),
                            target_cell_ids=tuple(sorted((owner.cell_id, other.cell_id))),
                        )
                    )
                    continue
                owner_trace = tuple(
                    _trace(expression, owner_face_arguments)
                    for expression in owner_embedding.coordinates
                )
                other_trace = tuple(
                    _trace(expression, other_face_arguments)
                    for expression in other_embedding.coordinates
                )
                if not all(
                    expression_equal(first, second)
                    for first, second in zip(owner_trace, other_trace, strict=True)
                ):
                    findings.append(
                        _finding(
                            "parent_interface_trace",
                            "violated",
                            "Exact physical traces disagree across a target parent interface.",
                            parent_cell_ids=tuple(
                                sorted((owner.parent_cell_id, other.parent_cell_id))
                            ),
                            target_cell_ids=tuple(sorted((owner.cell_id, other.cell_id))),
                        )
                    )
                    continue
            block_index = next(
                index
                for index, block in enumerate(target_mesh.blocks)
                if block.name == owner.block_name
            )
            target_block = target_mesh.blocks[block_index]
            cell_vertex_rows = tuple(
                int(value)
                for value in np.asarray(
                    target_block.vertices[owner.cell_row], dtype=np.int64
                ).tolist()
            )
            target_reference_vertices = target_reference_vertices_by_kind[
                target_block.cell_kind
            ]
            face_arguments = _face_arguments(
                target_block.cell_kind,
                cell_vertex_rows,
                canonical_loop_rows,
                target_reference_vertices,
            )
            chart_arguments = chart_arguments_by_row.get(
                (owner.block_name, owner.cell_row)
            )
            if face_arguments is None or chart_arguments is None:
                findings.append(
                    _finding(
                        "target_face_chart",
                        "unresolved",
                        "Exact hexahedral face-chart restriction is unavailable.",
                        parent_cell_ids=(owner.parent_cell_id,),
                        target_cell_ids=(owner.cell_id,),
                    )
                )
                continue
            reference_trace = tuple(
                _trace(expression, face_arguments) for expression in chart_arguments
            )
            face_orientation = _cycle_orientation(
                owner.loop_vertex_rows, canonical_loop_rows
            )
            if face_orientation is None:
                findings.append(
                    _finding(
                        "target_face_orientation",
                        "violated",
                        "Incident target face loops are not cyclically equivalent.",
                        parent_cell_ids=(owner.parent_cell_id,),
                        target_cell_ids=(owner.cell_id,),
                    )
                )
                continue
            side_orientation = (
                chart_sign_by_row[(owner.block_name, owner.cell_row)] * face_orientation
            )
            if len(owners) == 2 and owners[0].parent_cell_id == owners[1].parent_cell_id:
                other = owners[1] if owner is owners[0] else owners[0]
                other_block_index = next(
                    index
                    for index, block in enumerate(target_mesh.blocks)
                    if block.name == other.block_name
                )
                other_block = target_mesh.blocks[other_block_index]
                other_cell_rows = tuple(
                    int(value)
                    for value in np.asarray(
                        other_block.vertices[other.cell_row], dtype=np.int64
                    ).tolist()
                )
                other_face_arguments = _face_arguments(
                    other_block.cell_kind,
                    other_cell_rows,
                    canonical_loop_rows,
                    target_reference_vertices_by_kind[other_block.cell_kind],
                )
                other_chart = chart_arguments_by_row.get(
                    (other.block_name, other.cell_row)
                )
                if other_face_arguments is None or other_chart is None:
                    findings.append(
                        _finding(
                            "internal_reference_trace",
                            "unresolved",
                            "Exact source-reference trace is unavailable on an internal target face.",
                            parent_cell_ids=(owner.parent_cell_id,),
                            target_cell_ids=tuple(sorted((owner.cell_id, other.cell_id))),
                        )
                    )
                    continue
                other_trace = tuple(
                    _trace(expression, other_face_arguments) for expression in other_chart
                )
                if not all(
                    expression_equal(first, second)
                    for first, second in zip(reference_trace, other_trace, strict=True)
                ):
                    findings.append(
                        _finding(
                            "internal_reference_trace",
                            "violated",
                            "Exact target-to-parent traces disagree across an internal face.",
                            parent_cell_ids=(owner.parent_cell_id,),
                            target_cell_ids=tuple(sorted((owner.cell_id, other.cell_id))),
                        )
                    )
                    continue
                other_orientation = _cycle_orientation(
                    other.loop_vertex_rows, canonical_loop_rows
                )
                if other_orientation is None or (
                    side_orientation
                    + chart_sign_by_row[(other.block_name, other.cell_row)]
                    * other_orientation
                    != 0
                ):
                    findings.append(
                        _finding(
                            "internal_reference_orientation",
                            "violated",
                            "Internal source-reference traces do not have opposite orientations.",
                            parent_cell_ids=(owner.parent_cell_id,),
                            target_cell_ids=tuple(sorted((owner.cell_id, other.cell_id))),
                        )
                    )
                    continue
                physical = tuple(
                    _trace(expression, face_arguments)
                    for expression in target_embedding_by_row[
                        (owner.block_name, owner.cell_row)
                    ].coordinates
                )
                trace_id = canonical_fingerprint(
                    {
                        "reference_trace": tuple(
                            _expression_key(value) for value in reference_trace
                        ),
                        "physical_trace": tuple(
                            _expression_key(value) for value in physical
                        ),
                    }
                )
                internal_premises_by_parent[owner.parent_cell_id].append(
                    ReferenceChainInternalFacePremise(
                        owner.parent_cell_id,
                        tuple(
                            int(target_vertex_ids[value]) for value in canonical_loop_rows
                        ),
                        (
                            min(owner.cell_id, other.cell_id),
                            max(owner.cell_id, other.cell_id),
                        ),
                        reference_trace,
                        physical,
                        (
                            side_orientation,
                            -side_orientation,
                        ),
                        trace_id,
                    )
                )
                connected_cells_by_parent[owner.parent_cell_id][owner.cell_id].add(
                    other.cell_id
                )
                connected_cells_by_parent[owner.parent_cell_id][other.cell_id].add(
                    owner.cell_id
                )
                continue

            parent_location = parent_cell_location.get(owner.parent_cell_id)
            if parent_location is None:
                findings.append(
                    _finding(
                        "parent_reference_cell",
                        "violated",
                        "Parent cell is absent while constructing its exterior chain.",
                        parent_cell_ids=(owner.parent_cell_id,),
                        target_cell_ids=(owner.cell_id,),
                    )
                )
                continue
            parent_kind = parent_mesh.blocks[parent_location[1]].cell_kind
            parent_facet_rows = _reference_facets(parent_kind)
            if parent_facet_rows is None:
                findings.append(
                    _finding(
                        "parent_reference_facets",
                        "unresolved",
                        f"Parent reference cell {parent_kind!r} has no supported planar facets.",
                        parent_cell_ids=(owner.parent_cell_id,),
                        target_cell_ids=(owner.cell_id,),
                    )
                )
                continue
            matching_facets = tuple(
                facet
                for facet in parent_facet_rows
                if expression_equal(
                    _plane_residual(reference_trace, facet.outward_normal, facet.offset),
                    constant(0, 2),
                )
            )
            if len(matching_facets) != 1:
                findings.append(
                    _finding(
                        "reference_boundary_facet",
                        "violated",
                        "Exterior target face does not map identically to exactly one parent reference facet.",
                        parent_cell_ids=(owner.parent_cell_id,),
                        target_cell_ids=(owner.cell_id,),
                    )
                )
                continue
            facet = matching_facets[0]
            face_vertex_count = len(owner.loop_vertex_rows)
            if face_vertex_count == 3:
                face_chart_valid = all(
                    _affine_triangle(expression) for expression in reference_trace
                )
            elif face_vertex_count == 4:
                face_chart_valid = all(
                    _bilinear(expression) for expression in reference_trace
                )
            else:
                face_chart_valid = False
            if not face_chart_valid:
                findings.append(
                    _finding(
                        "reference_boundary_face_map",
                        "unresolved",
                        "Parent-boundary target faces require affine triangle or bilinear quadrilateral charts.",
                        parent_cell_ids=(owner.parent_cell_id,),
                        target_cell_ids=(owner.cell_id,),
                    )
                )
                continue
            exact_cell_corners = exact_corners_by_row[(owner.block_name, owner.cell_row)]
            local_index_by_row = {
                vertex_row: local_index
                for local_index, vertex_row in enumerate(cell_vertex_rows)
            }
            try:
                face_loop = tuple(
                    exact_cell_corners[local_index_by_row[vertex_row]]
                    for vertex_row in owner.loop_vertex_rows
                )
            except (IndexError, KeyError):
                findings.append(
                    _finding(
                        "reference_boundary_corners",
                        "violated",
                        "Parent-reference face corners do not align with target incidence.",
                        parent_cell_ids=(owner.parent_cell_id,),
                        target_cell_ids=(owner.cell_id,),
                    )
                )
                continue
            normal = facet.outward_normal
            area_vector = _area_vector(face_loop)
            signed_area_twice = _dot(area_vector, normal)
            if signed_area_twice <= 0:
                findings.append(
                    _finding(
                        "reference_boundary_orientation",
                        "violated",
                        "Mapped exterior face does not have positive parent-outward orientation.",
                        parent_cell_ids=(owner.parent_cell_id,),
                        target_cell_ids=(owner.cell_id,),
                    )
                )
                continue
            if len(face_loop) == 3:
                turns: tuple[Fraction, ...] = ()
                jacobian_controls = (signed_area_twice,)
            else:
                turns = tuple(
                    _dot(
                        _cross(
                            _sub(face_loop[(index + 1) % 4], face_loop[index]),
                            _sub(
                                face_loop[(index + 2) % 4],
                                face_loop[(index + 1) % 4],
                            ),
                        ),
                        normal,
                    )
                    for index in range(4)
                )
                jacobian_controls = (
                    _dot(
                        _cross(
                            _sub(face_loop[1], face_loop[0]),
                            _sub(face_loop[3], face_loop[0]),
                        ),
                        normal,
                    ),
                    _dot(
                        _cross(
                            _sub(face_loop[1], face_loop[0]),
                            _sub(face_loop[2], face_loop[1]),
                        ),
                        normal,
                    ),
                    _dot(
                        _cross(
                            _sub(face_loop[2], face_loop[3]),
                            _sub(face_loop[3], face_loop[0]),
                        ),
                        normal,
                    ),
                    _dot(
                        _cross(
                            _sub(face_loop[2], face_loop[3]),
                            _sub(face_loop[2], face_loop[1]),
                        ),
                        normal,
                    ),
                )
            if any(value <= 0 for value in turns) or any(
                value <= 0 for value in jacobian_controls
            ):
                findings.append(
                    _finding(
                        "reference_boundary_face_degree",
                        "unresolved",
                        "Exterior target face chart lacks strictly positive convexity/Jacobian controls.",
                        parent_cell_ids=(owner.parent_cell_id,),
                        target_cell_ids=(owner.cell_id,),
                    )
                )
                continue
            face_area = signed_area_twice / 2
            boundary_loops_by_parent_facet[(owner.parent_cell_id, facet.index)].append(
                face_loop
            )
            boundary_target_faces_by_parent_facet[
                (owner.parent_cell_id, facet.index)
            ].append(
                tuple(int(target_vertex_ids[value]) for value in owner.loop_vertex_rows)
            )
            trace_id = canonical_fingerprint(
                {
                    "reference_face_trace": tuple(
                        _expression_key(value) for value in reference_trace
                    ),
                    "corners": tuple(_point_key(point) for point in face_loop),
                    "jacobian_controls": tuple(
                        _scalar_key(value) for value in jacobian_controls
                    ),
                }
            )
            boundary_faces_by_parent[owner.parent_cell_id].append(
                ReferenceChainBoundaryFacePremise(
                    owner.parent_cell_id,
                    facet.index,
                    tuple(
                        int(target_vertex_ids[value]) for value in owner.loop_vertex_rows
                    ),
                    owner.cell_id,
                    face_loop,
                    reference_trace,
                    tuple(str(value) for value in jacobian_controls),
                    face_area,
                    trace_id,
                )
            )

    parent_ids = tuple(
        sorted(
            {
                chart.parent_cell_id
                for values in chart_premise_by_parent.values()
                for chart in values
            }
        )
    )
    parent_facets_by_kind: dict[str, tuple[_ReferenceFacet, ...] | None] = {}
    for parent_id in parent_ids:
        parent_location = parent_cell_location.get(parent_id)
        if parent_location is None:
            continue
        _, parent_block_index, _ = parent_location
        parent_kind = parent_mesh.blocks[parent_block_index].cell_kind
        if parent_kind not in parent_facets_by_kind:
            parent_facets_by_kind[parent_kind] = _reference_facets(parent_kind)
        facets = parent_facets_by_kind[parent_kind]
        if facets is None:
            findings.append(
                _finding(
                    "parent_reference_facets",
                    "unresolved",
                    f"Parent reference cell {parent_kind!r} has no supported planar facet decomposition.",
                    parent_cell_ids=(parent_id,),
                )
            )
            continue
        parent_boundary_premises: list[ReferenceChainBoundaryPremise] = []
        for facet in facets:
            key = (parent_id, facet.index)
            loops = tuple(boundary_loops_by_parent_facet.get(key, ()))
            target_face_ids = tuple(boundary_target_faces_by_parent_facet.get(key, ()))
            if not loops:
                findings.append(
                    _finding(
                        "reference_boundary_coverage",
                        "violated",
                        f"No exterior target faces cover parent reference facet {facet.index}.",
                        parent_cell_ids=(parent_id,),
                    )
                )
                continue
            chain_ok, oriented_edges = _boundary_chain(facet.oriented_vertices, loops)
            mapped_area = sum(
                (_dot(_area_vector(loop), facet.outward_normal) / 2 for loop in loops),
                Fraction(0),
            )
            if not chain_ok:
                findings.append(
                    _finding(
                        "reference_boundary_chain",
                        "violated",
                        f"Oriented exterior edge chain does not equal parent facet {facet.index} boundary.",
                        parent_cell_ids=(parent_id,),
                        target_cell_ids=tuple(
                            sorted(
                                target_id
                                for face in target_face_ids
                                for target_id in face
                            )
                        ),
                    )
                )
            if mapped_area != facet.projected_area:
                findings.append(
                    _finding(
                        "reference_boundary_area",
                        "violated",
                        f"Exact exterior area on parent facet {facet.index} is not its reference area.",
                        parent_cell_ids=(parent_id,),
                    )
                )
            chain_id = canonical_fingerprint(
                {
                    "parent_cell_id": parent_id,
                    "parent_facet_index": facet.index,
                    "oriented_edges": tuple(
                        (_point_key(first), _point_key(last))
                        for first, last in oriented_edges
                    ),
                    "mapped_area": _scalar_key(mapped_area),
                    "parent_area": _scalar_key(facet.projected_area),
                }
            )
            parent_boundary_premises.append(
                ReferenceChainBoundaryPremise(
                    parent_id,
                    facet.index,
                    target_face_ids,
                    oriented_edges,
                    mapped_area,
                    facet.projected_area,
                    1 if chain_ok and mapped_area == facet.projected_area else 0,
                    chain_id,
                )
            )
        target_cell_ids = tuple(
            sorted(chart.target_cell_id for chart in chart_premise_by_parent[parent_id])
        )
        chart_rows = tuple(chart_premise_by_parent[parent_id])
        if len({chart.jacobian_sign for chart in chart_rows}) != 1:
            findings.append(
                _finding(
                    "reference_chart_orientation",
                    "violated",
                    "Target charts for one parent cell do not have one consistent orientation.",
                    parent_cell_ids=(parent_id,),
                    target_cell_ids=target_cell_ids,
                )
            )
        neighbors = connected_cells_by_parent[parent_id]
        if len(target_cell_ids) > 1:
            reached = {target_cell_ids[0]}
            frontier = [target_cell_ids[0]]
            while frontier:
                current = frontier.pop()
                for neighbor in neighbors.get(current, ()):
                    if neighbor not in reached:
                        reached.add(neighbor)
                        frontier.append(neighbor)
            if reached != set(target_cell_ids):
                findings.append(
                    _finding(
                        "reference_target_connectivity",
                        "unresolved",
                        "Target chart cells for one parent are not connected through verified internal faces.",
                        parent_cell_ids=(parent_id,),
                        target_cell_ids=target_cell_ids,
                    )
                )
        boundary_premises = tuple(parent_boundary_premises)
        parent_premise_id = canonical_fingerprint(
            {
                "parent_cell_id": parent_id,
                "parent_cell_kind": parent_kind,
                "target_cell_ids": target_cell_ids,
                "charts": tuple(chart.composition_id for chart in chart_rows),
                "internal_faces": tuple(
                    premise.trace_id for premise in internal_premises_by_parent[parent_id]
                ),
                "boundary_facets": tuple(
                    premise.boundary_chain_id for premise in boundary_premises
                ),
            }
        )
        premises.append(
            ReferenceChainParentPremise(
                parent_id,
                parent_kind,
                target_cell_ids,
                chart_rows,
                tuple(internal_premises_by_parent[parent_id]),
                tuple(boundary_faces_by_parent[parent_id]),
                boundary_premises,
                1
                if boundary_premises
                and all(premise.degree == 1 for premise in boundary_premises)
                else 0,
                parent_premise_id,
            )
        )

    def prove_cross_parent_shared_strata() -> None:
        nonlocal cross_parent_coverage_complete
        if not shared_parent_strata:
            cross_parent_coverage_complete = True
            return
        cross_parent_coverage_complete = False
        failed = False

        def report(
            check: str,
            status: _ReferenceChainFindingStatus,
            detail: str,
            parent_ids: tuple[int, ...],
        ) -> None:
            nonlocal failed
            failed = True
            findings.append(
                _finding(
                    check,
                    status,
                    detail,
                    parent_cell_ids=parent_ids,
                )
            )

        target_vertex_rows = {
            int(identifier): row
            for row, identifier in enumerate(target_vertex_ids.tolist())
        }
        entity_vertices: dict[int, dict[int, tuple[int, ...]]] = {}
        entity_id_by_vertices: dict[int, dict[frozenset[int], int]] = {}
        for dimension in range(3):
            bank = _mesh_entity_bank(target_mesh, dimension)
            if bank is None:
                report(
                    "cross_parent_target_entities",
                    "unresolved",
                    f"Target dimension-{dimension} entity connectivity is unavailable.",
                    (),
                )
                return
            vertices_by_id: dict[int, tuple[int, ...]] = {}
            ids_by_vertices: dict[frozenset[int], int] = {}
            for entity_id, vertex_rows in bank:
                try:
                    vertex_ids_for_entity = tuple(
                        int(target_vertex_ids[vertex_row]) for vertex_row in vertex_rows
                    )
                except (IndexError, TypeError):
                    report(
                        "cross_parent_target_entities",
                        "violated",
                        f"Target dimension-{dimension} entity {entity_id} has an absent vertex row.",
                        (),
                    )
                    continue
                key = frozenset(vertex_ids_for_entity)
                previous = ids_by_vertices.get(key)
                if previous is not None and previous != entity_id:
                    report(
                        "cross_parent_target_entities",
                        "violated",
                        f"Target dimension-{dimension} connectivity has duplicate identities {previous} and {entity_id}.",
                        (),
                    )
                    continue
                ids_by_vertices[key] = entity_id
                vertices_by_id[entity_id] = vertex_ids_for_entity
            entity_vertices[dimension] = vertices_by_id
            entity_id_by_vertices[dimension] = ids_by_vertices
        if failed:
            return

        owners_by_dimension: dict[int, dict[int, list[_TargetEntityOwner]]] = {
            dimension: defaultdict(list) for dimension in range(3)
        }
        target_blocks_by_name = {block.name: block for block in target_mesh.blocks}
        for block in target_mesh.blocks:
            topology = reference_cell_topology(block.cell_kind)
            cell_rows = np.asarray(block.vertices, dtype=np.int64)
            for cell_row in range(len(cell_rows)):
                parent_id = parent_ids_by_row.get((block.name, cell_row))
                if parent_id is None:
                    report(
                        "cross_parent_target_cell_binding",
                        "unresolved",
                        f"Target cell row {cell_row} in block {block.name!r} has no parent chart.",
                        (),
                    )
                    continue
                local_vertex_rows = tuple(int(value) for value in cell_rows[cell_row])
                cell_vertex_ids = tuple(
                    int(target_vertex_ids[vertex_row]) for vertex_row in local_vertex_rows
                )
                for dimension in range(3):
                    for local_vertices in topology.entities[dimension]:
                        local_indices = tuple(int(value) for value in local_vertices)
                        entity_key = frozenset(
                            cell_vertex_ids[local_index] for local_index in local_indices
                        )
                        entity_id = entity_id_by_vertices[dimension].get(entity_key)
                        if entity_id is None:
                            report(
                                "cross_parent_target_entity_owner",
                                "violated",
                                f"Target cell {int(np.asarray(block.global_ids, dtype=np.int64)[cell_row])} refers to an absent dimension-{dimension} entity.",
                                (parent_id,),
                            )
                            continue
                        owners_by_dimension[dimension][entity_id].append(
                            _TargetEntityOwner(
                                block.name,
                                cell_row,
                                parent_id,
                                local_indices,
                            )
                        )
        if failed:
            return

        def parent_facets_for_stratum(
            parent_id: int,
            source_dimension: int,
            source_vertex_ids: tuple[int, ...],
        ) -> tuple[int, ...] | None:
            location = parent_cell_location.get(parent_id)
            if location is None:
                return None
            parent_block = parent_mesh.blocks[location[1]]
            parent_cell_rows = tuple(
                int(value)
                for value in np.asarray(
                    parent_block.vertices[location[2]], dtype=np.int64
                ).tolist()
            )
            parent_vertex_ids = np.asarray(parent_mesh.vertex_global_ids, dtype=np.int64)
            topology = reference_cell_topology(parent_block.cell_kind)
            source_id_set = set(source_vertex_ids)
            result = tuple(
                face_index
                for face_index, local_face in enumerate(topology.entities[2])
                if source_id_set.issubset(
                    {
                        int(parent_vertex_ids[parent_cell_rows[int(local_vertex)]])
                        for local_vertex in local_face
                    }
                )
            )
            if source_dimension == 2:
                exact_face_index = parent_face_indices.get(
                    (parent_id, 2, source_vertex_ids)
                )
                if exact_face_index is None or result != (exact_face_index,):
                    return None
                return (exact_face_index,)
            return result

        def topological_candidates(
            parent_id: int,
            source_dimension: int,
            source_vertex_ids: tuple[int, ...],
            target_dimension: int,
            /,
        ) -> set[int] | None:
            containing_facets = parent_facets_for_stratum(
                parent_id, source_dimension, source_vertex_ids
            )
            if not containing_facets:
                return None
            candidates: set[int] = set()
            for boundary_face in boundary_faces_by_parent.get(parent_id, ()):
                if boundary_face.parent_facet_index not in containing_facets:
                    continue
                face_vertices = boundary_face.target_face_vertex_ids
                if target_dimension == 2:
                    if source_dimension != 2:
                        continue
                    key = frozenset(face_vertices)
                    entity_id = entity_id_by_vertices[2].get(key)
                    if entity_id is None:
                        return None
                    candidates.add(entity_id)
                elif target_dimension == 1:
                    for index, first in enumerate(face_vertices):
                        last = face_vertices[(index + 1) % len(face_vertices)]
                        entity_id = entity_id_by_vertices[1].get(frozenset((first, last)))
                        if entity_id is None:
                            return None
                        candidates.add(entity_id)
                elif target_dimension == 0:
                    for vertex_id in face_vertices:
                        entity_id = entity_id_by_vertices[0].get(frozenset((vertex_id,)))
                        if entity_id is None:
                            return None
                        candidates.add(entity_id)
            return candidates

        def trace_equal(first: tuple[object, ...], second: tuple[object, ...]) -> bool:
            if len(first) != len(second):
                return False
            for left, right in zip(first, second, strict=True):
                left_expression = isinstance(left, (RationalPolynomial, Mapping))
                right_expression = isinstance(right, (RationalPolynomial, Mapping))
                if left_expression or right_expression:
                    if not (
                        left_expression
                        and right_expression
                        and expression_equal(
                            cast(Expression, left), cast(Expression, right)
                        )
                    ):
                        return False
                elif left != right:
                    return False
            return True

        def trace_key(trace: tuple[object, ...]) -> tuple[object, ...]:
            return tuple(
                _expression_key(cast(Expression, value))
                if isinstance(value, (RationalPolynomial, Mapping))
                else _scalar_key(cast(ExactScalar, value))
                for value in trace
            )

        def trace_for_entity(
            parent_id: int,
            source_dimension: int,
            source_vertex_ids: tuple[int, ...],
            source_points: tuple[_Point3, ...],
            source_face_cycle: tuple[int, ...],
            target_dimension: int,
            entity_id: int,
            /,
        ) -> tuple[
            tuple[object, ...] | None,
            tuple[object, ...] | None,
            int | None,
            bool,
            bool,
        ]:
            entity_vertex_ids = entity_vertices[target_dimension][entity_id]
            endpoint_coordinates: list[tuple[Fraction, ...] | None] = []
            endpoint_rows: list[int] = []
            for target_vertex_id in entity_vertex_ids:
                target_row = target_vertex_rows.get(target_vertex_id)
                if target_row is None:
                    return None, None, None, False, True
                endpoint_rows.append(target_row)
                root_point = parent_reference_by_target_vertex.get(
                    (parent_id, target_row)
                )
                if root_point is None:
                    return None, None, None, False, True
                endpoint_coordinates.append(
                    _point_in_source_stratum(
                        root_point,
                        source_dimension,
                        source_vertex_ids,
                        source_points,
                        source_face_cycle,
                    )
                )
            if any(value is None for value in endpoint_coordinates):
                if source_dimension == 2 and target_dimension >= 1:
                    return None, None, None, False, True
                return None, None, None, True, False
            exact_endpoint_coordinates = tuple(
                cast(tuple[Fraction, ...], value) for value in endpoint_coordinates
            )
            endpoint_coordinates_by_id = dict(
                zip(entity_vertex_ids, exact_endpoint_coordinates, strict=True)
            )
            if target_dimension == 0:
                if len(endpoint_rows) != 1:
                    return None, None, None, False, True
                physical_point = target_physical_by_vertex.get(endpoint_rows[0])
                if physical_point is None:
                    return None, None, None, False, True
                return (
                    exact_endpoint_coordinates[0],
                    tuple(physical_point),
                    1,
                    True,
                    True,
                )

            selected_owners = tuple(
                owner
                for owner in owners_by_dimension[target_dimension].get(entity_id, ())
                if owner.parent_cell_id == parent_id
            )
            if not selected_owners:
                return None, None, None, False, True
            owner_records: list[tuple[tuple[object, ...], tuple[object, ...], int]] = []
            for owner in selected_owners:
                block = target_blocks_by_name[owner.block_name]
                cell_vertex_rows = tuple(
                    int(value)
                    for value in np.asarray(
                        block.vertices[owner.cell_row], dtype=np.int64
                    ).tolist()
                )
                reference_vertices = target_reference_vertices_by_kind[block.cell_kind]
                if target_dimension == 1:
                    ordered_vertex_ids = tuple(sorted(entity_vertex_ids))
                    ordered_vertex_rows = tuple(
                        target_vertex_rows[vertex_id] for vertex_id in ordered_vertex_ids
                    )
                    entity_arguments = _edge_arguments(
                        block.cell_kind,
                        cell_vertex_rows,
                        (ordered_vertex_rows[0], ordered_vertex_rows[1]),
                        reference_vertices,
                    )
                else:
                    raw_face_ids = tuple(
                        int(target_vertex_ids[cell_vertex_rows[local_index]])
                        for local_index in owner.local_vertex_indices
                    )
                    ordered_vertex_ids = _canonical_cycle(raw_face_ids)
                    ordered_vertex_rows = tuple(
                        target_vertex_rows[vertex_id] for vertex_id in ordered_vertex_ids
                    )
                    entity_arguments = _face_arguments(
                        block.cell_kind,
                        cell_vertex_rows,
                        ordered_vertex_rows,
                        reference_vertices,
                    )
                chart = chart_arguments_by_row.get((owner.block_name, owner.cell_row))
                target_cell = target_embedding_by_row.get(
                    (owner.block_name, owner.cell_row)
                )
                if entity_arguments is None or chart is None or target_cell is None:
                    return None, None, None, False, True
                root_reference_trace = tuple(
                    _trace(expression, entity_arguments) for expression in chart
                )
                root_trace = _trace_coordinates_on_source_stratum(
                    root_reference_trace,
                    target_dimension,
                    source_dimension,
                    source_vertex_ids,
                    source_points,
                    source_face_cycle,
                )
                if root_trace is None:
                    return None, None, None, False, True
                if target_dimension == 1:
                    parameter = axes(1)[0]
                    start_coordinates = endpoint_coordinates_by_id[ordered_vertex_ids[0]]
                    end_coordinates = endpoint_coordinates_by_id[ordered_vertex_ids[1]]
                    if len(start_coordinates) != len(root_trace) or len(
                        end_coordinates
                    ) != len(root_trace):
                        return None, None, None, False, True
                    if not all(
                        expression_equal(
                            expression,
                            add(
                                constant(start, 1),
                                scale(parameter, end - start),
                            ),
                        )
                        for expression, start, end in zip(
                            root_trace,
                            start_coordinates,
                            end_coordinates,
                            strict=True,
                        )
                    ):
                        return None, None, None, False, True
                elif target_dimension == 2:
                    if len(entity_vertex_ids) == 3:
                        face_chart_valid = all(
                            _affine_triangle(expression) for expression in root_trace
                        )
                    elif len(entity_vertex_ids) == 4:
                        face_chart_valid = all(
                            _bilinear(expression) for expression in root_trace
                        )
                    else:
                        face_chart_valid = False
                    if not face_chart_valid:
                        return None, None, None, False, True

                physical_trace = tuple(
                    _trace(expression, entity_arguments)
                    for expression in target_cell.coordinates
                )
                if target_dimension == 1:
                    differences = tuple(
                        endpoint_coordinates_by_id[ordered_vertex_ids[1]][index]
                        - endpoint_coordinates_by_id[ordered_vertex_ids[0]][index]
                        for index in range(
                            len(endpoint_coordinates_by_id[ordered_vertex_ids[0]])
                        )
                    )
                    first_difference = next(
                        (value for value in differences if value), None
                    )
                    if first_difference is None:
                        return None, None, None, False, True
                    orientation = 1 if first_difference > 0 else -1
                else:
                    face_vertex_rows = tuple(
                        target_vertex_rows[vertex_id] for vertex_id in ordered_vertex_ids
                    )
                    matching_owners = tuple(
                        face_owner
                        for face_owner in face_owners.get(
                            tuple(sorted(face_vertex_rows)), ()
                        )
                        if face_owner.parent_cell_id == parent_id
                        and face_owner.block_name == owner.block_name
                        and face_owner.cell_row == owner.cell_row
                    )
                    if len(matching_owners) != 1:
                        return None, None, None, False, True
                    face_orientation = _cycle_orientation(
                        matching_owners[0].loop_vertex_rows, face_vertex_rows
                    )
                    if face_orientation is None:
                        return None, None, None, False, True
                    orientation = (
                        chart_sign_by_row[(owner.block_name, owner.cell_row)]
                        * face_orientation
                    )
                owner_records.append(
                    (
                        tuple(root_trace),
                        tuple(physical_trace),
                        orientation,
                    )
                )
            first_root, first_physical, first_orientation = owner_records[0]
            if any(
                not trace_equal(first_root, root)
                or not trace_equal(first_physical, physical)
                or first_orientation != orientation
                for root, physical, orientation in owner_records[1:]
            ):
                return None, None, None, False, True
            return first_root, first_physical, first_orientation, True, True

        for (
            source_dimension,
            source_vertex_ids,
            incident_parent_ids,
        ) in shared_parent_strata:
            if source_dimension == 2 and len(incident_parent_ids) != 2:
                report(
                    "cross_parent_source_face_incidence",
                    "violated",
                    f"Shared source face {source_vertex_ids} has {len(incident_parent_ids)} incident parent cells, not two.",
                    incident_parent_ids,
                )
                continue
            stratum_failed = False
            for target_dimension in range(source_dimension + 1):
                records_by_parent: dict[
                    int,
                    dict[
                        int,
                        tuple[tuple[object, ...], tuple[object, ...], int],
                    ],
                ] = {}
                ids_by_parent: list[tuple[int, tuple[int, ...]]] = []
                for parent_id in incident_parent_ids:
                    stratum_key = (
                        parent_id,
                        source_dimension,
                        source_vertex_ids,
                    )
                    source_points = parent_stratum_points.get(stratum_key)
                    source_face_cycle = parent_face_cycles.get(stratum_key, ())
                    if source_points is None:
                        report(
                            "cross_parent_source_stratum",
                            "unresolved",
                            f"Parent cell {parent_id} lacks exact coordinates for source stratum {source_vertex_ids}.",
                            incident_parent_ids,
                        )
                        stratum_failed = True
                        records_by_parent[parent_id] = {}
                        ids_by_parent.append((parent_id, ()))
                        continue
                    candidates = topological_candidates(
                        parent_id,
                        source_dimension,
                        source_vertex_ids,
                        target_dimension,
                    )
                    if candidates is None:
                        report(
                            "cross_parent_source_boundary",
                            "unresolved",
                            f"Parent cell {parent_id} lacks the verified target boundary workset for source stratum {source_vertex_ids}.",
                            incident_parent_ids,
                        )
                        stratum_failed = True
                        records_by_parent[parent_id] = {}
                        ids_by_parent.append((parent_id, ()))
                        continue
                    records: dict[
                        int,
                        tuple[tuple[object, ...], tuple[object, ...], int],
                    ] = {}
                    for entity_id in sorted(candidates):
                        root_trace, physical_trace, orientation, valid, on_stratum = (
                            trace_for_entity(
                                parent_id,
                                source_dimension,
                                source_vertex_ids,
                                source_points,
                                source_face_cycle,
                                target_dimension,
                                entity_id,
                            )
                        )
                        if not on_stratum:
                            if source_dimension == 2:
                                report(
                                    "cross_parent_source_face_trace",
                                    "violated",
                                    f"Target entity {entity_id} on a verified parent-face boundary does not map into source face {source_vertex_ids}.",
                                    incident_parent_ids,
                                )
                                stratum_failed = True
                            continue
                        if (
                            not valid
                            or root_trace is None
                            or physical_trace is None
                            or orientation is None
                        ):
                            report(
                                "cross_parent_entity_trace",
                                "violated",
                                f"Target entity {entity_id} has no consistent exact root/physical trace in parent cell {parent_id}.",
                                incident_parent_ids,
                            )
                            stratum_failed = True
                            continue
                        records[entity_id] = (
                            root_trace,
                            physical_trace,
                            orientation,
                        )
                    records_by_parent[parent_id] = records
                    ids_by_parent.append((parent_id, tuple(sorted(records))))
                entity_sets = tuple(set(ids) for _, ids in ids_by_parent)
                if (
                    not entity_sets
                    or not entity_sets[0]
                    or any(entity_set != entity_sets[0] for entity_set in entity_sets[1:])
                ):
                    report(
                        "cross_parent_target_strata",
                        "violated",
                        f"Shared source stratum {source_vertex_ids} has different target dimension-{target_dimension} entity identities across parent cells: {ids_by_parent}.",
                        incident_parent_ids,
                    )
                    stratum_failed = True
                common_entity_ids = (
                    tuple(sorted(set.intersection(*entity_sets))) if entity_sets else ()
                )
                if common_entity_ids:
                    reference_parent_id = incident_parent_ids[0]
                    for entity_id in common_entity_ids:
                        reference_trace = records_by_parent[reference_parent_id].get(
                            entity_id
                        )
                        if reference_trace is None:
                            stratum_failed = True
                            continue
                        for parent_id in incident_parent_ids[1:]:
                            candidate_trace = records_by_parent[parent_id].get(entity_id)
                            if candidate_trace is None:
                                stratum_failed = True
                                continue
                            if not trace_equal(reference_trace[0], candidate_trace[0]):
                                report(
                                    "cross_parent_root_trace",
                                    "violated",
                                    f"Target entity {entity_id} has different exact source-chart traces across parent cells.",
                                    incident_parent_ids,
                                )
                                stratum_failed = True
                            if not trace_equal(reference_trace[1], candidate_trace[1]):
                                report(
                                    "cross_parent_physical_trace",
                                    "violated",
                                    f"Target entity {entity_id} has different exact physical traces across parent cells.",
                                    incident_parent_ids,
                                )
                                stratum_failed = True
                            if target_dimension == 1 and (
                                reference_trace[2] != candidate_trace[2]
                            ):
                                report(
                                    "cross_parent_orientation",
                                    "violated",
                                    f"Target edge {entity_id} reverses its exact source-stratum orientation across parent cells.",
                                    incident_parent_ids,
                                )
                                stratum_failed = True
                    if target_dimension == 2:
                        for entity_id in common_entity_ids:
                            signs = tuple(
                                records_by_parent[parent_id][entity_id][2]
                                for parent_id in incident_parent_ids
                            )
                            if len(signs) != 2 or signs[0] + signs[1] != 0:
                                report(
                                    "cross_parent_orientation",
                                    "violated",
                                    f"Target face {entity_id} does not have opposite oriented parent-boundary traces: {signs}.",
                                    incident_parent_ids,
                                )
                                stratum_failed = True

                if stratum_failed:
                    continue
                root_trace_ids = tuple(
                    (
                        parent_id,
                        canonical_fingerprint(
                            {
                                "source_dimension": source_dimension,
                                "source_vertex_ids": source_vertex_ids,
                                "target_entity_dimension": target_dimension,
                                "traces": tuple(
                                    (
                                        entity_id,
                                        trace_key(
                                            records_by_parent[parent_id][entity_id][0]
                                        ),
                                    )
                                    for entity_id in sorted(records_by_parent[parent_id])
                                ),
                            }
                        ),
                    )
                    for parent_id in incident_parent_ids
                )
                first_parent_records = records_by_parent[incident_parent_ids[0]]
                physical_trace_id = canonical_fingerprint(
                    {
                        "source_dimension": source_dimension,
                        "source_vertex_ids": source_vertex_ids,
                        "target_entity_dimension": target_dimension,
                        "traces": tuple(
                            (
                                entity_id,
                                trace_key(first_parent_records[entity_id][1]),
                            )
                            for entity_id in sorted(first_parent_records)
                        ),
                    }
                )
                orientation_signs = tuple(
                    (parent_id, entity_id, records_by_parent[parent_id][entity_id][2])
                    for parent_id in incident_parent_ids
                    for entity_id in sorted(records_by_parent[parent_id])
                )
                premise_id = canonical_fingerprint(
                    {
                        "source_dimension": source_dimension,
                        "target_entity_dimension": target_dimension,
                        "source_vertex_ids": source_vertex_ids,
                        "incident_parent_cell_ids": incident_parent_ids,
                        "target_entity_ids_by_parent": tuple(ids_by_parent),
                        "root_trace_ids_by_parent": root_trace_ids,
                        "physical_trace_id": physical_trace_id,
                        "orientation_signs": orientation_signs,
                    }
                )
                cross_parent_premises.append(
                    ReferenceChainSharedStratumPremise(
                        source_dimension,
                        target_dimension,
                        source_vertex_ids,
                        incident_parent_ids,
                        tuple(ids_by_parent),
                        root_trace_ids,
                        physical_trace_id,
                        orientation_signs,
                        premise_id,
                    )
                )
            if stratum_failed:
                failed = True
        cross_parent_coverage_complete = not failed

    prove_cross_parent_shared_strata()
    return finish()


__all__ = [
    "ReferenceChainEmbeddingCertificate",
    "ReferenceChainSharedStratumPremise",
    "ReferenceChainEmbeddingFinding",
    "ReferenceChainParentPremise",
    "ReferenceChainChartPremise",
    "ReferenceChainInternalFacePremise",
    "ReferenceChainBoundaryFacePremise",
    "ReferenceChainBoundaryPremise",
    "certify_reference_chain_embedding",
]
