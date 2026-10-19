#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepare embedded planar CAD boundary queries and native CAD source regions."""

from __future__ import annotations

from fractions import Fraction
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.geometry._domain_boundary as domain_boundary_owner
import phydrax.geometry._meshing_domain as domain_owner
import phydrax.geometry._source_fidelity as source_fidelity_owner
import phydrax.geometry.brep._planar_arrangement as planar_arrangement_owner

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import unique_identifiers
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..typing import checked
from ._interval_enclosure import interval_subtract
from ._planar_embedding import PlanarEmbedding
from ._source_boundary_records import SourceBoundaryDistance, SourceBoundarySamples
from .brep._intersection import NativePeriodEndpoint, RootEndpoint
from .brep._intersection_curve import IntersectionCurve
from .brep._model import BRepModel
from .brep._patches import (
    AbstractCurve,
    BernsteinRestrictionBank,
    CircleCurve,
    LineCurve,
    MeridianOverlaps,
)
from .brep._placed import source_transform_bounds
from .brep._request_preparations import TrimRequestPreparations


def _exact_planar_coefficients(
    value: np.ndarray, embedding: PlanarEmbedding, /, *, point: bool
) -> tuple[Fraction, Fraction]:
    """Project binary source coefficients without rounding the affine operation."""
    coordinates = tuple(Fraction(float(coordinate)) for coordinate in value)
    if point:
        coordinates = tuple(
            coordinate - Fraction(origin)
            for coordinate, origin in zip(coordinates, embedding.origin, strict=True)
        )
    projected = tuple(
        sum(
            (
                coordinate * Fraction(axis_coordinate)
                for coordinate, axis_coordinate in zip(coordinates, axis, strict=True)
            ),
            Fraction(),
        )
        for axis in (embedding.x_axis, embedding.y_axis)
    )
    return projected[0], projected[1]


def _planar_coefficient_error(
    exact: tuple[Fraction, Fraction], rounded: Array, /
) -> Fraction:
    return sum(
        (
            abs(coefficient - Fraction(float(value)))
            for coefficient, value in zip(exact, np.asarray(rounded), strict=True)
        ),
        Fraction(),
    )


def _outward_planar_error(error: Fraction, /) -> float:
    rounded = float(error)
    return float(np.nextafter(rounded, np.inf)) if Fraction(rounded) < error else rounded


@final
class _PlanarCurveProxy(StrictModule):
    """A narrowed native line/circle carrier and its projected coefficient bounds."""

    source: LineCurve | CircleCurve
    curve: AbstractCurve
    error: float = eqx.field(static=True)


def _planar_curve_proxy(
    curve: AbstractCurve | IntersectionCurve,
    embedding: PlanarEmbedding,
    first: float,
    last: float,
    /,
) -> _PlanarCurveProxy:
    """Return a native planar proxy and a whole-range coefficient error bound."""
    basis = np.stack((embedding.x_axis, embedding.y_axis), axis=1)
    match curve:
        case LineCurve():
            origin = _exact_planar_coefficients(
                np.asarray(curve.origin), embedding, point=True
            )
            direction = _exact_planar_coefficients(
                np.asarray(curve.direction), embedding, point=False
            )
            proxy = LineCurve(
                embedding.to_planar(np.asarray(curve.origin)),
                np.asarray(curve.direction) @ basis,
            )
            error = _planar_coefficient_error(origin, proxy.origin) + max(
                abs(Fraction(first)), abs(Fraction(last))
            ) * _planar_coefficient_error(direction, proxy.direction)
        case CircleCurve():
            center = _exact_planar_coefficients(
                np.asarray(curve.center), embedding, point=True
            )
            first_axis = _exact_planar_coefficients(
                np.asarray(curve.first_axis), embedding, point=False
            )
            second_axis = _exact_planar_coefficients(
                np.asarray(curve.second_axis), embedding, point=False
            )
            proxy = CircleCurve(
                embedding.to_planar(np.asarray(curve.center)),
                np.asarray(curve.first_axis) @ basis,
                np.asarray(curve.second_axis) @ basis,
                curve.radius,
            )
            error = _planar_coefficient_error(center, proxy.center) + Fraction(
                float(curve.radius)
            ) * (
                _planar_coefficient_error(first_axis, proxy.first_axis)
                + _planar_coefficient_error(second_axis, proxy.second_axis)
            )
        case _:
            raise ValueError(
                "EmbeddedPlanarBoundarySource.curves must use line or circle "
                f"carriers; got {type(curve).__name__}"
            )
    return _PlanarCurveProxy(curve, proxy, _outward_planar_error(error))


def _planar_query_range(
    curve: AbstractCurve,
    endpoints: tuple[RootEndpoint | None, RootEndpoint | None],
    coedge_endpoints: tuple[RootEndpoint | None, RootEndpoint | None],
    first: float,
    last: float,
    /,
) -> tuple[float, float]:
    """Retain binary trims or enclose an authored mathematical circle turn."""
    match endpoints, coedge_endpoints:
        case (None, None), (None, None):
            return curve.validate_range(first, last)
        case (
            (NativePeriodEndpoint() as start, NativePeriodEndpoint() as end),
            (NativePeriodEndpoint() as use_start, NativePeriodEndpoint() as use_end),
        ):
            match curve:
                case CircleCurve():
                    if (
                        start.rational == end.rational
                        and end.turns - start.turns == 1
                        and (use_start.rational, use_start.turns)
                        == (start.rational, start.turns)
                        and (use_end.rational, use_end.turns) == (end.rational, end.turns)
                    ):
                        return curve.validate_range(
                            start.parameter_enclosure()[0],
                            end.parameter_enclosure()[1],
                        )
                case _:
                    pass
        case _:
            pass
    raise ValueError(
        "EmbeddedPlanarBoundarySource.model must use binary trims or full "
        "native circle turns; got unsupported endpoint definitions"
    )


@final
class _EmbeddedPlanarCurve(AbstractCurve):
    """Project a source carrier's evaluation and interval boxes into one frame."""

    definition: AbstractCurve
    origin: Array
    frame: Array

    def __init__(self, definition: AbstractCurve, embedding: PlanarEmbedding, /) -> None:
        self.definition = definition
        self.origin = jnp.asarray(embedding.origin, dtype=jnp.float64)
        self.frame = jnp.asarray(
            (embedding.x_axis, embedding.y_axis, embedding.normal), dtype=jnp.float64
        )

    @property
    def ambient_dimension(self) -> int:
        return 2

    @property
    def period(self) -> float | None:
        return self.definition.period

    @property
    def parameter_domain(self) -> tuple[float, float] | None:
        return self.definition.parameter_domain

    def evaluate(self, parameters: Array, /) -> Array:
        return (self.definition.evaluate(parameters) - self.origin) @ self.frame[:2].T

    def bounding_box(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        box = self.definition.bounding_box(
            first, last, preparations=preparations, budget=budget
        )
        origin = np.asarray(self.origin)
        low, high = interval_subtract((box[0], box[1]), (origin, origin))
        low, high = source_transform_bounds(
            np.asarray(self.frame), np.asarray(low), np.asarray(high)
        )
        return np.stack((low[:2], high[:2]))

    def derivative_bounds(
        self,
        first: float,
        last: float,
        /,
        *,
        order: int = 1,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
        restriction_bank: BernsteinRestrictionBank | None = None,
        meridian_jets: list[MeridianOverlaps] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        low, high = self.definition.derivative_bounds(
            first,
            last,
            order=order,
            preparations=preparations,
            budget=budget,
            restriction_bank=restriction_bank,
            meridian_jets=meridian_jets,
        )
        low, high = source_transform_bounds(np.asarray(self.frame), low, high)
        return low[:2], high[:2]


@final
class EmbeddedPlanarBoundarySource(StrictModule, NonTrainableState):
    """Continuous projected boundary of one native planar face.

    ``curves`` are native planar coefficient proxies. ``projection_errors``
    bound their pointwise deviation from the projected source carriers over
    the whole trims. ``parameter_ranges`` retain oriented traversal, while
    interval queries use ascending ranges of the source operation trees.
    ``loop_edges`` indexes the flattened carriers in oriented face-loop order.
    """

    model: BRepModel
    embedding: PlanarEmbedding = eqx.field(static=True)
    curves: tuple[AbstractCurve, ...]
    parameter_ranges: tuple[tuple[float, float], ...] = eqx.field(static=True)
    loop_edges: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    projection_errors: tuple[float, ...] = eqx.field(static=True)
    projection_ids: tuple[str, ...] = eqx.field(static=True)
    _query_curves: tuple[AbstractCurve, ...]
    _query_ranges: tuple[tuple[float, float], ...] = eqx.field(static=True)

    @checked
    def __init__(self, model: BRepModel, embedding: PlanarEmbedding, /) -> None:

        geometry = model.geometry
        if geometry is None:
            raise ValueError(
                "EmbeddedPlanarBoundarySource.model.geometry must contain native "
                "curves and oriented loops; got None"
            )
        if len(model.patches) != 1 or geometry.solid_shells or geometry.occurrences:
            raise ValueError(
                "EmbeddedPlanarBoundarySource.model must contain one unplaced "
                f"planar face and no solids; got {len(model.patches)} faces, "
                f"{len(geometry.solid_shells)} solids, "
                f"{len(geometry.occurrences)} occurrences"
            )
        planar_arrangement_owner.require_coplanar_face(model, 0, embedding)
        if any(root is not None for root in geometry.vertex_roots) or any(
            root is not None and not isinstance(root, NativePeriodEndpoint)
            for endpoints in (
                *geometry.edge_endpoint_roots,
                *geometry.coedge_endpoint_roots,
            )
            for root in endpoints
        ):
            raise ValueError(
                "EmbeddedPlanarBoundarySource.model must use binary or native "
                "period endpoints; got implicit trim or vertex roots"
            )
        used_edges = {
            geometry.coedge_edges[coedge]
            for loop in geometry.face_loops[0]
            for coedge in loop
        }
        if used_edges != set(range(len(geometry.edge_curves))):
            raise ValueError(
                "EmbeddedPlanarBoundarySource.model must contain only its face "
                "boundary edges; got unreferenced edges"
            )
        reversed_face = float(np.asarray(model.orientation)[0]) < 0.0
        ranges = np.asarray(geometry.edge_ranges, dtype=np.float64)
        curves: list[AbstractCurve] = []
        query_curves: list[AbstractCurve] = []
        parameter_ranges: list[tuple[float, float]] = []
        query_ranges: list[tuple[float, float]] = []
        loop_edges: list[tuple[int, ...]] = []
        errors: list[float] = []
        identities: list[str] = []
        for loop in geometry.face_loops[0]:
            indices = []
            for coedge in reversed(loop) if reversed_face else loop:
                edge = geometry.coedge_edges[coedge]
                carrier = geometry.edge_curves[edge]
                if carrier < 0:
                    raise ValueError(
                        "EmbeddedPlanarBoundarySource.model must use "
                        f"nondegenerate boundary edges; got edge {edge}"
                    )
                curve = geometry.curves[carrier]
                first, last = (float(value) for value in ranges[edge])
                prepared_curve = _planar_curve_proxy(curve, embedding, first, last)
                proxy, error = prepared_curve.curve, prepared_curve.error
                query_range = _planar_query_range(
                    prepared_curve.source,
                    geometry.edge_endpoint_roots[edge],
                    geometry.coedge_endpoint_roots[coedge],
                    first,
                    last,
                )
                query_curve = _EmbeddedPlanarCurve(prepared_curve.source, embedding)
                sense = geometry.coedge_senses[coedge] * (-1 if reversed_face else 1)
                interval = (first, last) if sense > 0 else (last, first)
                indices.append(len(curves))
                curves.append(proxy)
                query_curves.append(query_curve)
                parameter_ranges.append(interval)
                query_ranges.append(query_range)
                errors.append(error)
                identities.append(
                    canonical_fingerprint(
                        {
                            "kind": "embedded-planar-source-curve",
                            "model": model.model_id,
                            "source": model.source_id,
                            "revision": model.source_revision,
                            "embedding": embedding.embedding_id,
                            "edge": edge,
                            "coedge": coedge,
                            "range": interval,
                            "proxy": array_tree_fingerprint(proxy),
                            "projection_error": error,
                        }
                    )
                )
            loop_edges.append(tuple(indices))
        self.model = model
        self.embedding = embedding
        self.curves = tuple(curves)
        self.parameter_ranges = tuple(parameter_ranges)
        self.loop_edges = tuple(loop_edges)
        self.projection_errors = tuple(errors)
        self.projection_ids = tuple(identities)
        self._query_curves = tuple(query_curves)
        self._query_ranges = tuple(query_ranges)

    @property
    def source_id(self) -> str:
        return self.model.source_id

    @property
    def source_revision(self) -> str:
        return self.model.source_revision

    @property
    def ambient_dimension(self) -> int:
        return 2

    @property
    def source_scope_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "embedded-planar-boundary-source-scope",
                "model": self.model.model_id,
                "embedding": self.embedding.embedding_id,
                "curves": self.projection_ids,
                "loops": self.loop_edges,
            }
        )

    def prepare_boundary_queries(
        self,
    ) -> source_fidelity_owner.ParametricCurveBoundarySource:
        """Prepare interval-enclosed queries of every projected source trim."""

        return source_fidelity_owner.ParametricCurveBoundarySource(
            self._query_curves,
            self._query_ranges,
            source_id=self.source_id,
            source_revision=self.source_revision,
        )

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance:
        """Return whole-carrier distance bounds in embedded planar coordinates."""
        return self.prepare_boundary_queries().boundary_distance(points)

    def boundary_samples(self, maximum_samples: int, /) -> SourceBoundarySamples:
        """Return interval-enclosed anchors and whole -trim covering radii."""
        return self.prepare_boundary_queries().boundary_samples(maximum_samples)


@final
class CadSourceDomain(StrictModule, NonTrainableState):
    """Bind one native CAD region and its boundary to a source revision.

    A solid has ambient dimension three. An explicit planar embedding selects
    a single trimmed face as a two-dimensional region. Region identities are
    independent declared metadata, not inferred from CAD display tags.
    """

    model: BRepModel
    embedding: PlanarEmbedding | None = eqx.field(static=True)
    region_ids: tuple[str, ...] = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        model: BRepModel,
        *,
        region_ids: tuple[str, ...],
        embedding: PlanarEmbedding | None = None,
    ) -> None:
        if not isinstance(region_ids, tuple):
            raise TypeError(
                "CadSourceDomain.region_ids must be an explicit tuple; "
                f"got {type(region_ids).__name__}"
            )
        regions = unique_identifiers(region_ids, "CadSourceDomain.region_ids")
        if len(regions) != 1:
            raise ValueError(
                f"CadSourceDomain.region_ids must declare one region; got {len(regions)}"
            )
        geometry = model.geometry
        if geometry is None:
            raise ValueError(
                "CadSourceDomain.model.geometry must contain native carriers "
                "and topology; got None"
            )
        if embedding is None:
            if len(geometry.solid_shells) != 1 or len(geometry.occurrences) > 1:
                raise ValueError(
                    "CadSourceDomain.model must contain one solid with at most "
                    f"one occurrence; got {len(geometry.solid_shells)} solids, "
                    f"{len(geometry.occurrences)} occurrences"
                )
            boundary_faces = {
                face
                for shell in geometry.solid_shells[0]
                for face in geometry.shell_faces[shell]
            }
            if boundary_faces != set(range(len(model.patches))):
                raise ValueError(
                    "CadSourceDomain.model must contain only its solid boundary "
                    "faces; got unreferenced faces"
                )
        else:
            EmbeddedPlanarBoundarySource(model, embedding)
        self.model = model
        self.embedding = embedding
        self.region_ids = regions
        self.domain_id = canonical_fingerprint(
            {
                "kind": "cad-source-domain",
                "model": model.model_id,
                "source": model.source_id,
                "revision": model.source_revision,
                "embedding": None if embedding is None else embedding.embedding_id,
                "regions": regions,
            }
        )

    @property
    def source_id(self) -> str:
        return self.model.source_id

    @property
    def source_revision(self) -> str:
        return self.model.source_revision

    @property
    def ambient_dimension(self) -> int:
        return 3 if self.embedding is None else 2

    def boundary_source(
        self,
    ) -> domain_boundary_owner.MeshingDomainBoundarySource | EmbeddedPlanarBoundarySource:
        """Return the whole native boundary of the declared CAD region."""
        if self.embedding is not None:
            return EmbeddedPlanarBoundarySource(self.model, self.embedding)
        domain = domain_owner.MeshingDomain.from_brep(self.model)
        return domain_boundary_owner.MeshingDomainBoundarySource(
            domain, tuple(range(len(domain.patches)))
        )
