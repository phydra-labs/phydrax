#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Scientific evidence and occurrence-qualified lineage for native B-Rep offsets."""

from __future__ import annotations

from fractions import Fraction
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from ...discretization._exact_algebraic_geometry import ExactScalar

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...typing import checked, parse
from .._cad_revision import (
    AssociationCoverageEvidence,
    AssociationGraph,
    CADOccurrence,
    OccurrenceCorrespondence,
    OccurrenceCorrespondenceTransaction,
)
from ._model import BRepEntityId, BRepModel
from ._parameter_range import CertifiedParameterRange, unknown_parameter_range
from ._patches import BSplineCurve, OffsetSurface, RevolutionSurface
from ..._interpolation._bspline_extraction import RationalBSplineCurveSpan


BRepOffsetVolumeStatus: TypeAlias = Literal["certified", "unresolved"]


@final
class BRepOffsetEvidence(StrictModule):
    """Bounds against the intended offset in model face and vertex order."""

    face_rounding_bounds: Array | None
    vertex_rounding_bounds: Array | None
    volume_enclosure: Array | None
    volume_status: BRepOffsetVolumeStatus = eqx.field(static=True)
    volume_reason: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        face_rounding_bounds: ArrayLike | None,
        vertex_rounding_bounds: ArrayLike | None,
        volume_enclosure: ArrayLike | None,
        volume_reason: str = "",
    ) -> None:
        faces = (
            None
            if face_rounding_bounds is None
            else jnp.asarray(face_rounding_bounds, dtype=jnp.float64)
        )
        vertices = (
            None
            if vertex_rounding_bounds is None
            else jnp.asarray(vertex_rounding_bounds, dtype=jnp.float64)
        )
        enclosure = (
            None
            if volume_enclosure is None
            else jnp.asarray(volume_enclosure, dtype=jnp.float64)
        )
        for name, values in (("face", faces), ("vertex", vertices)):
            if values is None:
                continue
            host = np.asarray(values)
            if host.ndim != 1 or np.any(~np.isfinite(host)) or np.any(host < 0.0):
                raise ValueError(
                    f"BRepOffsetEvidence.rounding_bounds must be finite nonnegative rows; got invalid {name} values"
                )
        if enclosure is not None:
            host = np.asarray(enclosure)
            if host.shape != (2,) or np.any(~np.isfinite(host)) or host[0] > host[1]:
                raise ValueError(
                    "BRepOffsetEvidence.volume_enclosure must have finite ordered endpoints; got an invalid interval"
                )
        if not isinstance(volume_reason, str):
            raise TypeError("volume_reason must be a string.")
        self.face_rounding_bounds = faces
        self.vertex_rounding_bounds = vertices
        self.volume_enclosure = enclosure
        self.volume_status = parse(
            "unresolved" if enclosure is None else "certified",
            BRepOffsetVolumeStatus,
            "offset.volume_status",
        )
        self.volume_reason = volume_reason


def _qualified_entity_ids(model: BRepModel, kind: str, /) -> tuple[BRepEntityId, ...]:
    geometry = model.geometry
    return tuple(
        sorted(
            {
                record.member
                for record in geometry.qualified_entity_incidence(
                    model.source_id, model.source_revision
                )
                if record.member.kind == kind
            }
        )
    )


def _entity_key(entity: BRepEntityId, /) -> tuple[str, int, tuple[str, ...]]:
    return entity.kind, entity.index, entity.occurrence_path


def _entity_images(
    source: BRepModel,
    target: BRepModel,
    kind: str,
    index_images: dict[int, int],
    /,
) -> tuple[tuple[BRepEntityId, BRepEntityId], ...]:
    source_ids = getattr(source, f"{kind}_ids")
    target_ids = getattr(target, f"{kind}_ids")
    images = []
    for source_id in source_ids:
        target_index = index_images.get(source_id.index)
        if target_index is None:
            continue
        if target_index < 0 or target_index >= len(target_ids):
            raise ValueError(f"Native {kind} image references an absent target entity.")
        images.append((source_id, target_ids[target_index]))
    return tuple(images)


def _images(
    source: BRepModel, target: BRepModel, kind: str, /
) -> tuple[tuple[BRepEntityId, BRepEntityId], ...]:
    source_ids = getattr(source, f"{kind}_ids")
    target_ids = getattr(target, f"{kind}_ids")
    if len(source_ids) != len(target_ids):
        raise ValueError(
            f"BRepOffsetResult.{kind}_images requires topology-preserving source and target banks."
        )
    return tuple(zip(source_ids, target_ids, strict=True))


def _namespace_paths(model: BRepModel, /) -> tuple[tuple[str, ...], ...]:
    geometry = model.geometry
    entity_paths = {
        entity.occurrence_path
        for kind in ("solid", "face", "edge", "vertex")
        for entity in _qualified_entity_ids(model, kind)
        if entity.occurrence_path
    }
    container_paths = {
        container.path for container in geometry.iter_assembly_containers()
    }
    occurrence_paths = {occurrence.path for occurrence in geometry.iter_occurrences()}
    return tuple(sorted(entity_paths | container_paths | occurrence_paths))


def _namespace_id(source_id: str, revision_id: str, path: tuple[str, ...], /) -> str:
    return canonical_fingerprint(
        {
            "kind": "cad-instance-namespace",
            "source_id": source_id,
            "source_revision": revision_id,
            "path": path,
        }
    )


def _association_graph_for_images(
    source: BRepModel,
    target: BRepModel,
    images: dict[str, dict[int, int]],
    operation: str,
    /,
) -> AssociationGraph:
    from ._partition import _entity_text, cad_revision_from_brep_model

    source_revision = cad_revision_from_brep_model(source)
    target_revision = cad_revision_from_brep_model(target)
    source_occurrences = {
        value.occurrence_id: value for value in source_revision.occurrences
    }
    target_occurrences = {
        value.occurrence_id: value for value in target_revision.occurrences
    }
    target_entities = {
        _entity_key(entity): entity
        for kind in ("solid", "face", "edge", "vertex")
        for entity in _qualified_entity_ids(target, kind)
    }
    source_entities = {
        _entity_text(entity): entity
        for kind in ("solid", "face", "edge", "vertex")
        for entity in _qualified_entity_ids(source, kind)
    }
    target_entity_occurrences = {
        (value.kind, value.entity_id): value
        for value in target_revision.occurrences
        if value.kind != "assembly"
    }
    source_entity_occurrences = {
        value.entity_id: value
        for value in source_revision.occurrences
        if value.kind != "assembly"
    }
    source_namespaces = _namespace_paths(source)
    target_namespaces = set(_namespace_paths(target))
    source_namespace_ids = {
        path: _namespace_id(source.source_id, source_revision.revision_id, path)
        for path in source_namespaces
    }
    target_namespace_ids = {
        path: _namespace_id(target.source_id, target_revision.revision_id, path)
        for path in target_namespaces
    }
    certificate_id = canonical_fingerprint(
        {
            "kind": "native-brep-operation-correspondence",
            "operation": operation,
            "source_model": source.model_id,
            "target_model": target.model_id,
            "images": {
                kind: tuple(sorted(rows.items())) for kind, rows in sorted(images.items())
            },
            "qualified_source": tuple(sorted(source_entities)),
            "qualified_target": tuple(
                sorted(_entity_text(value) for value in target_entities.values())
            ),
        }
    )
    correspondences = []
    mapped_source: set[str] = set()
    mapped_target: set[str] = set()

    def add(
        source_occurrence: CADOccurrence, target_occurrence: CADOccurrence, /
    ) -> None:
        correspondences.append(
            OccurrenceCorrespondence(
                source_occurrence.occurrence_id,
                target_occurrence.occurrence_id,
                canonical_fingerprint(
                    {
                        "kind": "native-brep-operation-occurrence-image",
                        "certificate": certificate_id,
                        "source_entity": source_occurrence.entity_id,
                        "target_entity": target_occurrence.entity_id,
                    }
                ),
            )
        )
        mapped_source.add(source_occurrence.occurrence_id)
        mapped_target.add(target_occurrence.occurrence_id)

    if set(source_namespace_ids) != set(target_namespace_ids):
        raise ValueError(
            "Native operation must preserve the exact assembly namespace bank."
        )
    for path, source_namespace_id in source_namespace_ids.items():
        source_occurrence = source_occurrences.get(source_namespace_id)
        target_namespace_id = target_namespace_ids.get(path)
        target_occurrence = (
            None
            if target_namespace_id is None
            else target_occurrences.get(target_namespace_id)
        )
        if source_occurrence is None or target_occurrence is None:
            raise ValueError(
                "Native operation must preserve each assembly namespace with exact path correspondence."
            )
        add(source_occurrence, target_occurrence)

    for source_text, source_entity in source_entities.items():
        target_index = images[source_entity.kind].get(source_entity.index)
        if target_index is None:
            continue
        target_entity = target_entities.get(
            (source_entity.kind, target_index, source_entity.occurrence_path)
        )
        if target_entity is None:
            raise ValueError(
                "Native source correspondence lost an occurrence-qualified target entity."
            )
        source_occurrence = source_entity_occurrences.get(source_text)
        target_occurrence = target_entity_occurrences.get(
            (target_entity.kind, _entity_text(target_entity))
        )
        if source_occurrence is None or target_occurrence is None:
            raise ValueError(
                "Native source correspondence has no occurrence inventory entry."
            )
        add(source_occurrence, target_occurrence)

    unresolved_source = frozenset()
    unresolved_target = frozenset()
    transaction = OccurrenceCorrespondenceTransaction(
        certificate_id,
        source_revision.revision_id,
        target_revision.revision_id,
        tuple(correspondences),
        frozenset(unresolved_source),
        frozenset(unresolved_target),
        AssociationCoverageEvidence(True, True, certificate_id, f"native-{operation}"),
    )
    return AssociationGraph(source_revision, target_revision, transaction)


def _association_graph(source: BRepModel, target: BRepModel, /) -> AssociationGraph:
    images = {
        kind: {
            entity.index: entity.index for entity in _qualified_entity_ids(source, kind)
        }
        for kind in ("solid", "face", "edge", "vertex")
    }
    return _association_graph_for_images(source, target, images, "offset")


def _topology_signature(model: BRepModel, /) -> tuple[object, ...]:
    geometry = model.geometry
    topology = geometry.topology
    qualified = tuple(
        sorted(
            (record.member.kind, record.member.index, record.member.occurrence_path)
            for record in geometry.qualified_entity_incidence(
                model.source_id, model.source_revision
            )
            if record.member.kind in ("solid", "face", "edge", "vertex")
        )
    )
    return (
        topology.face_wires,
        topology.face_edges,
        topology.edge_faces,
        topology.solid_faces,
        topology.solid_face_orientations,
        topology.face_solids,
        geometry.edge_curves,
        geometry.edge_vertices,
        geometry.coedge_edges,
        geometry.coedge_senses,
        geometry.face_loops,
        geometry.shell_faces,
        geometry.shell_orientations,
        geometry.solid_shells,
        geometry.assembly_graph.identity_payload(),
        qualified,
        tuple(int(value) for value in np.asarray(model.orientation)),
    )


@final
class BRepOffsetResult(StrictModule):
    """Complete offset model, mathematical evidence, and exact source lineage."""

    model: BRepModel
    source_model_id: str = eqx.field(static=True)
    face_images: tuple[tuple[BRepEntityId, BRepEntityId], ...] = eqx.field(static=True)
    edge_images: tuple[tuple[BRepEntityId, BRepEntityId], ...] = eqx.field(static=True)
    vertex_images: tuple[tuple[BRepEntityId, BRepEntityId], ...] = eqx.field(static=True)
    solid_images: tuple[tuple[BRepEntityId, BRepEntityId], ...] = eqx.field(static=True)
    association_graph: AssociationGraph
    evidence: BRepOffsetEvidence
    parameter_range: CertifiedParameterRange | None

    @checked
    def __init__(
        self,
        source_model: BRepModel,
        model: BRepModel,
        evidence: BRepOffsetEvidence,
        /,
        *,
        parameter_range: CertifiedParameterRange | None = None,
    ) -> None:
        if not isinstance(source_model, BRepModel) or not isinstance(model, BRepModel):
            raise TypeError(
                "BRepOffsetResult requires source and target BRepModel values."
            )
        if source_model.model_id == model.model_id:
            raise ValueError(
                "BRepOffsetResult requires distinct source and target revisions."
            )
        if _topology_signature(source_model) != _topology_signature(model):
            raise ValueError(
                "BRepOffsetResult requires exact source correspondence for every retained entity."
            )
        geometry = model.geometry
        face_bounds = evidence.face_rounding_bounds
        vertex_bounds = evidence.vertex_rounding_bounds
        if (face_bounds is not None and face_bounds.shape != (len(model.patches),)) or (
            vertex_bounds is not None
            and vertex_bounds.shape != (geometry.vertex_points.shape[0],)
        ):
            raise ValueError(
                "BRepOffsetResult.evidence must align with model face and vertex rows; got different counts"
            )
        self.model = model
        self.source_model_id = source_model.source_revision
        self.face_images = _images(source_model, model, "face")
        self.edge_images = _images(source_model, model, "edge")
        self.vertex_images = _images(source_model, model, "vertex")
        self.solid_images = _images(source_model, model, "solid")
        association_graph = _association_graph(source_model, model)
        if (
            association_graph.transaction.unresolved_source_ids
            or association_graph.transaction.unresolved_target_ids
        ):
            raise ValueError(
                "BRepOffsetResult requires exhaustive source/target entity correspondence."
            )
        self.association_graph = association_graph
        self.evidence = evidence
        self.parameter_range = parameter_range


def exact_operation_offset_result(
    source_model: BRepModel,
    model: BRepModel,
    /,
    *,
    include_parameter_range: bool = False,
) -> BRepOffsetResult:
    """Bind a complete analytic offset to its exact source entity IDs."""
    return BRepOffsetResult(
        source_model,
        model,
        BRepOffsetEvidence(
            face_rounding_bounds=None,
            vertex_rounding_bounds=None,
            volume_enclosure=None,
            volume_reason=(
                "smooth native offset intent has no rational-plane rounding proof; "
                "volume requires the certified mass-property query"
            ),
        ),
        parameter_range=(
            unknown_parameter_range(
                "distance", "smooth offset family has no certified parameter interval"
            )
            if include_parameter_range
            else None
        ),
    )


BRepOffsetEmbeddingStatus: TypeAlias = Literal["certified", "unresolved", "unsupported"]


@final
class BRepOffsetEmbeddingCertificate(StrictModule):
    """Whole-chart normal-isotopy finding for a source-defined offset."""

    status: BRepOffsetEmbeddingStatus = eqx.field(static=True)
    reason: str = eqx.field(static=True)

    @checked
    def __init__(self, status: BRepOffsetEmbeddingStatus, reason: str, /) -> None:
        self.status = parse(status, BRepOffsetEmbeddingStatus, "offset_embedding.status")
        if not isinstance(reason, str) or not reason:
            raise ValueError("offset embedding certificates require a reason.")
        self.reason = reason


type _RationalInterval = tuple[Fraction, Fraction]
type _RationalVector = tuple[_RationalInterval, ...]
type _ExactVector3 = tuple[Fraction, Fraction, Fraction]


def _fraction_vector3(values: tuple[ExactScalar, ...], /) -> _ExactVector3 | None:
    if len(values) != 3:
        return None
    first, second, third = values
    if not (
        isinstance(first, Fraction)
        and isinstance(second, Fraction)
        and isinstance(third, Fraction)
    ):
        return None
    return first, second, third


def _ri_add(first: _RationalInterval, second: _RationalInterval, /) -> _RationalInterval:
    return first[0] + second[0], first[1] + second[1]


def _ri_neg(value: _RationalInterval, /) -> _RationalInterval:
    return -value[1], -value[0]


def _ri_sub(first: _RationalInterval, second: _RationalInterval, /) -> _RationalInterval:
    return _ri_add(first, _ri_neg(second))


def _ri_mul(first: _RationalInterval, second: _RationalInterval, /) -> _RationalInterval:
    products = tuple(a * b for a in first for b in second)
    return min(products), max(products)


def _ri_scale(coefficient: Fraction, value: _RationalInterval, /) -> _RationalInterval:
    return (
        (coefficient * value[0], coefficient * value[1])
        if coefficient >= 0
        else (coefficient * value[1], coefficient * value[0])
    )


def _ri_square(value: _RationalInterval, /) -> _RationalInterval:
    lower = (
        Fraction(0) if value[0] <= 0 <= value[1] else min(value[0] ** 2, value[1] ** 2)
    )
    return lower, max(value[0] ** 2, value[1] ** 2)


def _ri_div_positive(
    numerator: _RationalInterval, denominator: _RationalInterval, /
) -> _RationalInterval | None:
    if denominator[0] <= 0 or denominator[1] < denominator[0]:
        return None
    return _ri_mul(
        numerator,
        (Fraction(1, 1) / denominator[1], Fraction(1, 1) / denominator[0]),
    )


def _rv_add(first: _RationalVector, second: _RationalVector, /) -> _RationalVector:
    return tuple(_ri_add(a, b) for a, b in zip(first, second, strict=True))


def _rv_sub(first: _RationalVector, second: _RationalVector, /) -> _RationalVector:
    return tuple(_ri_sub(a, b) for a, b in zip(first, second, strict=True))


def _rv_scale(coefficient: Fraction, value: _RationalVector, /) -> _RationalVector:
    return tuple(_ri_scale(coefficient, component) for component in value)


def _rv_dot(
    vector: tuple[Fraction, ...], interval: _RationalVector, /
) -> _RationalInterval:
    result = (Fraction(0), Fraction(0))
    for coefficient, component in zip(vector, interval, strict=True):
        result = _ri_add(result, _ri_scale(coefficient, component))
    return result


def _rv_cross(first: _RationalVector, second: _RationalVector, /) -> _RationalVector:
    return tuple(
        _ri_sub(_ri_mul(first[left], second[right]), _ri_mul(first[right], second[left]))
        for left, right in ((1, 2), (2, 0), (0, 1))
    )


def _rv_norm_squared(value: _RationalVector, /) -> _RationalInterval:
    components = tuple(_ri_square(component) for component in value)
    return (
        sum((component[0] for component in components), Fraction(0)),
        sum((component[1] for component in components), Fraction(0)),
    )


def _square_root_fraction_interval(
    value: _RationalInterval, /
) -> _RationalInterval | None:
    from ._exact import square_root_interval

    lower, upper = square_root_interval(value[0])[0], square_root_interval(value[1])[1]
    if not (np.isfinite(lower) and np.isfinite(upper)):
        return None
    return Fraction.from_float(lower), Fraction.from_float(upper)


def _bspline_jet_bounds(
    spans: tuple[RationalBSplineCurveSpan, ...], lower: Fraction, upper: Fraction, /
) -> tuple[_RationalVector, _RationalVector, _RationalVector] | None:
    value_rows: list[tuple[tuple[Fraction, Fraction], ...]] = []
    first_rows: list[tuple[tuple[Fraction, Fraction], ...]] = []
    second_rows: list[tuple[tuple[Fraction, Fraction], ...]] = []
    for span in spans:
        span_lower, span_upper = span.parameter_bounds
        if not isinstance(span_lower, Fraction) or not isinstance(span_upper, Fraction):
            return None
        first = max(lower, span_lower)
        last = min(upper, span_upper)
        if first > last:
            continue
        enclosure = span.enclose((first, last))
        value_rows.append(enclosure.value_bounds)
        first_rows.append(enclosure.first_derivative_bounds)
        second_rows.append(enclosure.second_derivative_bounds)
    if not value_rows:
        return None

    def hull(rows: list[tuple[tuple[Fraction, Fraction], ...]]) -> _RationalVector:
        dimensions = len(rows[0])
        return tuple(
            (
                min(row[axis][0] for row in rows),
                max(row[axis][1] for row in rows),
            )
            for axis in range(dimensions)
        )

    return hull(value_rows), hull(first_rows), hull(second_rows)


def _endpoint_spline_jet(
    span: RationalBSplineCurveSpan, parameter: Fraction, /
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]] | None:
    enclosure = span.enclose((parameter, parameter))
    values, derivatives = [], []
    for rows, output in (
        (enclosure.value_bounds, values),
        (enclosure.first_derivative_bounds, derivatives),
    ):
        for lower, upper in rows:
            if lower != upper:
                return None
            output.append(lower)
    return tuple(values), tuple(derivatives)


def _axis_meridian_frame(
    patch: RevolutionSurface, curve: BSplineCurve, /
) -> (
    tuple[
        _ExactVector3,
        _ExactVector3,
        tuple[_ExactVector3, _ExactVector3],
    ]
    | None
):
    from .._rational import rational_vector

    axis = _fraction_vector3(rational_vector(np.asarray(patch.axis_direction)))
    origin = _fraction_vector3(rational_vector(np.asarray(patch.axis_origin)))
    if axis is None or origin is None:
        return None
    if sum((component * component for component in axis), Fraction(0)) != 1:
        return None
    controls = []
    for row in curve.exact_control_points:
        control = _fraction_vector3(row)
        if control is None:
            return None
        controls.append(control)
    relative: list[_ExactVector3] = []
    for point in controls:
        relative.append(
            (
                point[0] - origin[0],
                point[1] - origin[1],
                point[2] - origin[2],
            )
        )

    def cross(first: _ExactVector3, second: _ExactVector3, /) -> _ExactVector3:
        return (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )

    normal = next(
        (cross(axis, point) for point in relative if any(cross(axis, point))), None
    )
    if normal is None or any(
        sum(
            (component * value for component, value in zip(normal, point, strict=True)),
            Fraction(0),
        )
        for point in relative
    ):
        return None
    radial = cross(axis, normal)
    return axis, origin, (normal, radial)


def _offset_certificate(
    status: BRepOffsetEmbeddingStatus, reason: str, /
) -> BRepOffsetEmbeddingCertificate:
    return BRepOffsetEmbeddingCertificate(status, reason)


@checked
def certify_revolution_offset_embedding(
    model: BRepModel, offset: OffsetSurface, /
) -> BRepOffsetEmbeddingCertificate:
    """Prove exact whole-chart normal isotopy for a closed B-spline meridian.

    Admission is one unplaced solid with one self-glued, pole-free RevolutionSurface
    chart over a positive-weight BSplineCurve. The axis is exactly unit, the
    meridian controls lie exactly in one plane containing that axis, and its
    closed endpoint value and tangent agree exactly. Adaptive rational Bernstein
    enclosures prove regularity and an axis-separated isotopy. Local tangent
    monotonicity excludes adjacent-cell contacts; disjoint interval images exclude
    every nonadjacent meridian contact. Subdivision stops only when binary64 has
    no interior parameter value, which returns unresolved rather than relying on
    samples or work quotas.
    """
    from ._assembly import BRepAffinePlacement
    from ._carrier_pieces import carrier_payload
    from ._intersection import NativePeriodEndpoint
    from ._patches import BSplineCurve, CircleCurve, LineCurve, RevolutionSurface
    from ..._interpolation import rational_bspline_curve_spans
    from .._rational import rational_vector

    if not isinstance(model, BRepModel):
        raise TypeError("offset embedding model must be a BRepModel.")
    if not isinstance(offset, OffsetSurface):
        return _offset_certificate(
            "unsupported", "offset carrier is not an exact OffsetSurface."
        )
    geometry = model.geometry
    if len(model.patches) != 1:
        return _offset_certificate(
            "unsupported",
            "the helper admits only one RevolutionSurface with a B-spline meridian.",
        )
    patch = model.patches[0]
    if not isinstance(patch, RevolutionSurface):
        return _offset_certificate(
            "unsupported", "the source face is not a RevolutionSurface."
        )
    curve = patch.curve
    if not isinstance(curve, BSplineCurve):
        return _offset_certificate(
            "unsupported", "the source revolution has no B-spline meridian."
        )
    if canonical_fingerprint(carrier_payload(offset.base)) != canonical_fingerprint(
        carrier_payload(patch)
    ):
        return _offset_certificate(
            "unsupported", "offset base is not the exact source face carrier."
        )
    if offset.frame_sign not in (-1, 1):
        return _offset_certificate(
            "unresolved", "the source normal orientation is not proved."
        )
    occurrences = tuple(geometry.assembly_graph.iter_occurrences())
    identity = BRepAffinePlacement.identity().identity_payload()
    if (
        len(occurrences) != 1
        or occurrences[0].solid_index != 0
        or occurrences[0].placement.identity_payload() != identity
        or geometry.topology.num_solids != 1
        or geometry.topology.num_faces != 1
        or geometry.topology.num_edges != 2
        or geometry.topology.num_vertices != 1
        or geometry.solid_shells != ((0,),)
        or geometry.shell_faces != ((0,),)
        or len(geometry.face_loops) != 1
        or len(geometry.face_loops[0]) != 1
        or len(geometry.face_loops[0][0]) != 4
        or any(vertices != (0, 0) for vertices in geometry.edge_vertices)
        or any(root is not None for root in geometry.vertex_roots)
    ):
        return _offset_certificate(
            "unsupported",
            "source topology is not one unplaced, self-glued, pole-free chart.",
        )
    if patch.native_period_symbols() != ("two_pi", None):
        return _offset_certificate(
            "unsupported", "the revolution chart lacks its native full turn."
        )
    if curve.parameter_domain is None or not curve.is_c1_on(*curve.parameter_domain):
        return _offset_certificate(
            "unresolved", "the B-spline meridian is not C1 over its source domain."
        )
    bounds = np.asarray(model.parameter_bounds, dtype=np.float64)
    if bounds.shape != (1, 2, 2):
        return _offset_certificate(
            "unsupported", "the source has no single rectangular chart bound."
        )
    domain = np.asarray(patch.parameter_domain, dtype=np.float64)
    if not np.array_equal(bounds[0], domain):
        return _offset_certificate(
            "unsupported", "the source face does not cover the full revolution chart."
        )
    u_lower, v_lower = (Fraction.from_float(float(value)) for value in bounds[0, 0])
    u_upper, v_upper = (Fraction.from_float(float(value)) for value in bounds[0, 1])
    period = patch.periods[0]
    if (
        period is None
        or u_lower != 0
        or u_upper != Fraction.from_float(float(period))
        or v_lower >= v_upper
    ):
        return _offset_certificate(
            "unsupported", "the source chart does not span one canonical revolution turn."
        )

    if len(geometry.edge_curves) != 2 or len(geometry.coedge_edges) != 4:
        return _offset_certificate(
            "unsupported", "the chart boundary has unsupported edge incidence."
        )
    curve_rows = tuple(
        None if index < 0 else geometry.curves[index] for index in geometry.edge_curves
    )
    profile_edges = tuple(
        index for index, item in enumerate(curve_rows) if isinstance(item, BSplineCurve)
    )
    circle_edges = tuple(
        index for index, item in enumerate(curve_rows) if isinstance(item, CircleCurve)
    )
    if len(profile_edges) != 1 or len(circle_edges) != 1:
        return _offset_certificate(
            "unsupported",
            "the chart requires one spline seam and one circular turn edge.",
        )
    profile_edge, circle_edge = profile_edges[0], circle_edges[0]
    profile_curve = curve_rows[profile_edge]
    turn_curve = curve_rows[circle_edge]
    if not isinstance(profile_curve, BSplineCurve) or not isinstance(
        turn_curve, CircleCurve
    ):
        return _offset_certificate(
            "unsupported", "source edge carriers do not match the admitted chart."
        )
    uses: dict[int, list[int]] = {profile_edge: [], circle_edge: []}
    sides: dict[tuple[int, int], list[int]] = {}
    for coedge, edge in enumerate(geometry.coedge_edges):
        if edge not in uses:
            return _offset_certificate(
                "unsupported", "a boundary coedge references a foreign edge."
            )
        uses[edge].append(coedge)
        pcurve = geometry.pcurves[coedge]
        if not isinstance(pcurve, LineCurve):
            return _offset_certificate(
                "unsupported", "chart boundary p-curves must be exact affine lines."
            )
        origin = rational_vector(np.asarray(pcurve.origin))
        direction = rational_vector(np.asarray(pcurve.direction))
        variable = tuple(
            axis for axis in (0, 1) if direction[axis] == 1 and direction[1 - axis] == 0
        )
        if len(variable) != 1:
            return _offset_certificate(
                "unsupported", "a chart boundary is not an identity isoline."
            )
        axis = variable[0]
        fixed = 1 - axis
        value = origin[fixed]
        if value == Fraction.from_float(float(bounds[0, 0, fixed])):
            side = 0
        elif value == Fraction.from_float(float(bounds[0, 1, fixed])):
            side = 1
        else:
            return _offset_certificate(
                "unsupported", "a chart boundary lies away from its source rectangle."
            )
        edge_range = tuple(
            Fraction.from_float(float(value)) for value in geometry.edge_ranges[edge]
        )
        if axis == 0:
            if (
                edge != circle_edge
                or origin[0] != u_lower
                or edge_range != (u_lower, u_upper)
                or origin[1] not in (v_lower, v_upper)
            ):
                return _offset_certificate(
                    "unsupported",
                    "a circular turn edge is not the exact full-period chart isoline.",
                )
        elif (
            edge != profile_edge
            or origin[0] != u_lower
            or origin[1] != v_lower
            or edge_range != (v_lower, v_upper)
        ):
            return _offset_certificate(
                "unsupported", "the spline seam is not the identity meridian isoline."
            )
        sides.setdefault((fixed, side), []).append(edge)
        roots = geometry.coedge_endpoint_roots[coedge]
        if axis == 0:
            first_root, last_root = roots
            if not isinstance(first_root, NativePeriodEndpoint) or not isinstance(
                last_root, NativePeriodEndpoint
            ):
                return _offset_certificate(
                    "unsupported", "a turn-edge endpoint lacks native period roots."
                )
            if (
                first_root.patch is None
                or last_root.patch is None
                or carrier_payload(first_root.patch) != carrier_payload(patch)
                or carrier_payload(last_root.patch) != carrier_payload(patch)
                or carrier_payload(first_root.curve) != carrier_payload(pcurve)
                or carrier_payload(last_root.curve) != carrier_payload(pcurve)
                or first_root.axis != 0
                or last_root.axis != 0
                or first_root.rational != 0
                or last_root.rational != 0
                or (first_root.turns, last_root.turns) != (Fraction(0), Fraction(1))
            ):
                return _offset_certificate(
                    "unsupported",
                    "a turn-edge root is not bound to the source chart period.",
                )
        elif any(root is not None for root in roots):
            return _offset_certificate(
                "unsupported", "the spline seam has unsupported endpoint roots."
            )
    if any(
        len(coedges) != 2
        or geometry.coedge_senses[coedges[0]] == geometry.coedge_senses[coedges[1]]
        for coedges in uses.values()
    ):
        return _offset_certificate(
            "unsupported", "each self-glued edge needs opposite coedge uses."
        )
    if (
        sides.get((1, 0)) != [circle_edge]
        or sides.get((1, 1)) != [circle_edge]
        or sides.get((0, 0)) != [profile_edge, profile_edge]
        or (0, 1) in sides
    ):
        return _offset_certificate(
            "unsupported",
            "source sides do not form the full revolution and closed meridian seams.",
        )
    first_turn_root, last_turn_root = geometry.edge_endpoint_roots[circle_edge]
    if not isinstance(first_turn_root, NativePeriodEndpoint) or not isinstance(
        last_turn_root, NativePeriodEndpoint
    ):
        return _offset_certificate(
            "unsupported", "the circular edge lacks native period roots."
        )
    for root, turns in (
        (first_turn_root, Fraction(0)),
        (last_turn_root, Fraction(1)),
    ):
        if (
            carrier_payload(root.curve) != carrier_payload(turn_curve)
            or root.patch is None
            or carrier_payload(root.patch) != carrier_payload(patch)
            or root.axis != 0
            or root.rational != 0
            or root.turns != turns
        ):
            return _offset_certificate(
                "unsupported",
                "the circular edge roots do not bind its complete source turn.",
            )
    if any(root is not None for root in geometry.edge_endpoint_roots[profile_edge]):
        return _offset_certificate(
            "unsupported", "the spline seam has unsupported endpoint roots."
        )
    if carrier_payload(profile_curve) != carrier_payload(curve):
        return _offset_certificate(
            "unsupported", "the spline seam is not the exact meridian carrier."
        )

    frame = _axis_meridian_frame(patch, curve)
    if frame is None:
        return _offset_certificate(
            "unsupported", "the meridian is not exactly planar about a unit source axis."
        )
    axis, origin, (plane_normal, radial_axis) = frame
    try:
        spans = rational_bspline_curve_spans(
            curve.exact_control_points,
            curve.exact_weights,
            curve.exact_knots,
            curve.degree,
        )
    except (TypeError, ValueError, ZeroDivisionError) as error:
        return _offset_certificate(
            "unresolved", f"rational meridian spans are unresolved: {error}"
        )
    if not spans:
        return _offset_certificate(
            "unresolved", "the B-spline meridian has no active rational spans."
        )
    endpoint_first = _endpoint_spline_jet(spans[0], v_lower)
    endpoint_last = _endpoint_spline_jet(spans[-1], v_upper)
    if endpoint_first is None or endpoint_last is None:
        return _offset_certificate(
            "unresolved", "the closed meridian endpoints have no exact rational jets."
        )
    if endpoint_first != endpoint_last:
        return _offset_certificate(
            "unresolved",
            "the closed meridian endpoints do not have an exact common point and tangent.",
        )
    circle = turn_curve
    profile_point = endpoint_first[0]
    relative_point = tuple(
        value - base for value, base in zip(profile_point, origin, strict=True)
    )
    axial = sum(
        (
            component * value
            for component, value in zip(axis, relative_point, strict=True)
        ),
        Fraction(0),
    )
    expected_center = tuple(
        base + axial * component for base, component in zip(origin, axis, strict=True)
    )
    radial_point = tuple(
        value - axial * component
        for value, component in zip(relative_point, axis, strict=True)
    )
    radial_squared = sum((value * value for value in radial_point), Fraction(0))
    first_axis, second_axis = circle.first_axis_exact, circle.second_axis_exact
    turn_frame = (
        first_axis[1] * second_axis[2] - first_axis[2] * second_axis[1],
        first_axis[2] * second_axis[0] - first_axis[0] * second_axis[2],
        first_axis[0] * second_axis[1] - first_axis[1] * second_axis[0],
    )
    circle_start = tuple(
        center + circle.radius_exact * direction
        for center, direction in zip(
            circle.center_exact, circle.first_axis_exact, strict=True
        )
    )
    if (
        circle.center_exact != expected_center
        or radial_squared != circle.radius_exact**2
        or turn_frame != axis
        or circle_start != profile_point
    ):
        return _offset_certificate(
            "unsupported",
            "the turn edge is not the exact revolution of the spline seam endpoint.",
        )
    delta = offset.distance_exact * offset.frame_sign
    if any(not isinstance(knot, Fraction) for knot in curve.exact_knots):
        return _offset_certificate(
            "unresolved", "the exact B-spline knots are not rational source parameters."
        )
    cells: list[tuple[Fraction, Fraction, RationalBSplineCurveSpan]] = []
    for span in spans:
        span_lower, span_upper = span.parameter_bounds
        if not isinstance(span_lower, Fraction) or not isinstance(span_upper, Fraction):
            return _offset_certificate(
                "unresolved", "a spline knot interval is not a rational source interval."
            )
        first = max(v_lower, span_lower)
        last = min(v_upper, span_upper)
        if first < last:
            cells.append((first, last, span))
    if not cells:
        return _offset_certificate(
            "unresolved", "the meridian has no positive-width parameter spans."
        )

    def split_cells(
        source_cells: list[tuple[Fraction, Fraction, RationalBSplineCurveSpan]],
        indices: set[int],
        /,
    ) -> list[tuple[Fraction, Fraction, RationalBSplineCurveSpan]] | None:
        refined: list[tuple[Fraction, Fraction, RationalBSplineCurveSpan]] = []
        for index, (first, last, span) in enumerate(source_cells):
            if index not in indices:
                refined.append((first, last, span))
                continue
            first_float, last_float = float(first), float(last)
            middle_float = first_float + 0.5 * (last_float - first_float)
            if not first_float < middle_float < last_float:
                return None
            middle = Fraction.from_float(float(middle_float))
            refined.extend(((first, middle, span), (middle, last, span)))
        return refined

    def interval_dot(
        first: _RationalVector, second: _RationalVector, /
    ) -> _RationalInterval:
        result = (Fraction(0), Fraction(0))
        for left, right in zip(first, second, strict=True):
            result = _ri_add(result, _ri_mul(left, right))
        return result

    def box_separated(first: _RationalVector, second: _RationalVector, /) -> bool:
        return any(
            first[axis][1] < second[axis][0] or second[axis][1] < first[axis][0]
            for axis in range(3)
        )

    def cell_certificate(
        first: Fraction, last: Fraction, span: RationalBSplineCurveSpan, /
    ) -> tuple[_RationalVector, _RationalVector, int] | None:
        jets = _bspline_jet_bounds((span,), first, last)
        if jets is None:
            return None
        position, tangent, bending = jets
        relative = _rv_sub(position, tuple((value, value) for value in origin))
        speed_squared = _rv_norm_squared(tangent)
        if speed_squared[0] <= 0:
            return None
        azimuth_tangent = _rv_cross(tuple((value, value) for value in axis), relative)
        normal_raw = _rv_cross(azimuth_tangent, tangent)
        normal_derivative_raw = _rv_add(
            _rv_cross(
                _rv_cross(tuple((value, value) for value in axis), tangent), tangent
            ),
            _rv_cross(azimuth_tangent, bending),
        )
        normal_squared = _rv_norm_squared(normal_raw)
        norm_interval = _square_root_fraction_interval(normal_squared)
        if norm_interval is None or norm_interval[0] <= 0:
            return None
        norm_lower, norm_upper = norm_interval
        normalized = tuple(
            _ri_div_positive(component, (norm_lower, norm_upper))
            for component in normal_raw
        )
        if any(component is None for component in normalized):
            return None
        normal = tuple(component for component in normalized if component is not None)
        dot_normal_derivative = (Fraction(0), Fraction(0))
        for normal_component, derivative_component in zip(
            normal, normal_derivative_raw, strict=True
        ):
            dot_normal_derivative = _ri_add(
                dot_normal_derivative,
                _ri_mul(normal_component, derivative_component),
            )
        differentiated = tuple(
            _ri_div_positive(
                _ri_sub(
                    derivative_component,
                    _ri_mul(normal_component, dot_normal_derivative),
                ),
                (norm_lower, norm_upper),
            )
            for normal_component, derivative_component in zip(
                normal, normal_derivative_raw, strict=True
            )
        )
        if any(component is None for component in differentiated):
            return None
        normal_derivative = tuple(
            component for component in differentiated if component is not None
        )
        target_tangent = _rv_add(tangent, _rv_scale(delta, normal_derivative))
        forward_dot = interval_dot(target_tangent, tangent)
        if forward_dot[0] <= 0:
            return None
        source_radial = _rv_dot(radial_axis, relative)
        target_position = _rv_add(relative, _rv_scale(delta, normal))
        target_radial = _rv_dot(radial_axis, target_position)
        side = (
            1
            if source_radial[0] > 0 and target_radial[0] > 0
            else -1
            if source_radial[1] < 0 and target_radial[1] < 0
            else 0
        )
        if side == 0:
            return None
        image = tuple(
            (
                min(relative[axis_index][0], target_position[axis_index][0]),
                max(relative[axis_index][1], target_position[axis_index][1]),
            )
            for axis_index in range(3)
        )
        direction = tuple((component[0] + component[1]) / 2 for component in tangent)
        if not any(direction) or _rv_dot(direction, tangent)[0] <= 0:
            return None
        return image, tangent, side

    while True:
        split: set[int] = set()
        facts: list[tuple[_RationalVector, _RationalVector, int]] = []
        for index, (first, last, span) in enumerate(cells):
            fact = cell_certificate(first, last, span)
            if fact is None:
                split.add(index)
            else:
                facts.append(fact)
        if not split:
            side = facts[0][2]
            if any(fact[2] != side for fact in facts[1:]):
                return _offset_certificate(
                    "unresolved",
                    "the meridian or its parallel offset reaches the revolution axis.",
                )
            count = len(cells)
            adjacent_pairs = tuple((index, index + 1) for index in range(count - 1))
            if count > 1:
                adjacent_pairs = (*adjacent_pairs, (count - 1, 0))
            for first_index, last_index in adjacent_pairs:
                first_fact, last_fact = facts[first_index], facts[last_index]
                first_tangent, last_tangent = first_fact[1], last_fact[1]
                tangent_hull = tuple(
                    (
                        min(first_tangent[axis_index][0], last_tangent[axis_index][0]),
                        max(first_tangent[axis_index][1], last_tangent[axis_index][1]),
                    )
                    for axis_index in range(3)
                )
                direction = tuple(
                    (component[0] + component[1]) / 2 for component in tangent_hull
                )
                if not any(direction) or _rv_dot(direction, tangent_hull)[0] <= 0:
                    split.update((first_index, last_index))
            for first_index in range(count):
                for last_index in range(first_index + 1, count):
                    if last_index == first_index + 1 or (
                        first_index == 0 and last_index == count - 1
                    ):
                        continue
                    first_fact, last_fact = facts[first_index], facts[last_index]
                    if box_separated(first_fact[0], last_fact[0]):
                        continue
                    first_width = cells[first_index][1] - cells[first_index][0]
                    last_width = cells[last_index][1] - cells[last_index][0]
                    if first_width >= last_width:
                        split.add(first_index)
                    if last_width >= first_width:
                        split.add(last_index)
        if not split:
            return _offset_certificate(
                "certified",
                "the whole meridian normal isotopy is locally injective and globally separated from the axis.",
            )
        refined = split_cells(cells, split)
        if refined is None:
            return _offset_certificate(
                "unresolved",
                "normal-isotopy separation remains undecided at parameter resolution.",
            )
        cells = refined
