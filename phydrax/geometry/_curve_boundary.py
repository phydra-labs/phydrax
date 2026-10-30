#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Continuous source-boundary queries for explicitly identified curves."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import final, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier, finite_real_scalar
from ..linalg._hermitian_spectral import fraction_sqrt_interval
from ._interval_enclosure import prepare_interval_function, PreparedIntervalFunction
from ._source_boundary_records import SourceBoundaryDistance, SourceBoundarySamples
from .brep._patches import AbstractCurve, SurfaceMappedCurve


if TYPE_CHECKING:
    from ..meshing._scope import _SourceEntityScopes

type _SourcePointKey = tuple[str, tuple[str, ...]]
type _EndpointSourcePointKeys = tuple[
    tuple[_SourcePointKey | None, _SourcePointKey | None], ...
]


def _parameter_ranges(
    curves: tuple[AbstractCurve, ...],
    values: tuple[tuple[float, float], ...],
    /,
) -> tuple[tuple[float, float], ...]:
    if len(values) != len(curves):
        raise ValueError("parameter_ranges must align with the source curve charts.")
    result: list[tuple[float, float]] = []
    for curve, interval in zip(curves, values, strict=True):
        if len(interval) != 2:
            raise ValueError(
                "Every source curve parameter range must have two endpoints."
            )
        first = finite_real_scalar(interval[0], "first_curve_parameter")
        last = finite_real_scalar(interval[1], "last_curve_parameter")
        if first == last:
            raise ValueError("Source curve parameter ranges must be finite and nonempty.")
        domain = curve.parameter_domain
        if domain is not None and (
            min(first, last) < domain[0] or max(first, last) > domain[1]
        ):
            raise ValueError("A source curve range must lie inside its parameter domain.")
        result.append((first, last))
    return tuple(result)


def _outward_float(value: Fraction, /) -> float:
    try:
        result = float(value)
    except OverflowError:
        return math.inf
    if not math.isfinite(result):
        return math.inf
    if Fraction.from_float(result) < value:
        return float(np.nextafter(result, math.inf))
    return result


def _inward_float(value: Fraction, /) -> float:
    try:
        result = float(value)
    except OverflowError:
        return 0.0
    if not math.isfinite(result):
        return 0.0
    if Fraction.from_float(result) > value:
        return float(np.nextafter(result, -math.inf))
    return max(result, 0.0)


def _sqrt_upper(value: Fraction, /) -> float:
    return _outward_float(fraction_sqrt_interval(value)[1])


def _sqrt_lower(value: Fraction, /) -> float:
    return _inward_float(fraction_sqrt_interval(value)[0])


def _box_radius(lower: np.ndarray, upper: np.ndarray, point: np.ndarray, /) -> float:
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        return math.inf
    squared = sum(
        (
            max(
                abs(Fraction.from_float(float(low)) - Fraction.from_float(float(value))),
                abs(Fraction.from_float(float(high)) - Fraction.from_float(float(value))),
            )
            ** 2
            for low, high, value in zip(lower, upper, point, strict=True)
        ),
        Fraction(0),
    )
    return _sqrt_upper(squared)


def _box_distance_upper(
    lower: np.ndarray,
    upper: np.ndarray,
    point_lower: np.ndarray,
    point_upper: np.ndarray,
    /,
) -> float:
    if not all(
        np.all(np.isfinite(values)) for values in (lower, upper, point_lower, point_upper)
    ):
        return math.inf
    squared = sum(
        (
            max(
                abs(
                    Fraction.from_float(float(low))
                    - Fraction.from_float(float(point_high))
                ),
                abs(
                    Fraction.from_float(float(high))
                    - Fraction.from_float(float(point_low))
                ),
            )
            ** 2
            for low, high, point_low, point_high in zip(
                lower, upper, point_lower, point_upper, strict=True
            )
        ),
        Fraction(0),
    )
    return _sqrt_upper(squared)


def _box_distance_lower(
    point: np.ndarray, lower: np.ndarray, upper: np.ndarray, /
) -> float:
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        return 0.0
    squared = sum(
        (
            max(
                Fraction.from_float(float(low)) - Fraction.from_float(float(value)),
                Fraction.from_float(float(value)) - Fraction.from_float(float(high)),
                Fraction(0),
            )
            ** 2
            for value, low, high in zip(point, lower, upper, strict=True)
        ),
        Fraction(0),
    )
    return _sqrt_lower(squared)


def _point_distance_upper(
    first: np.ndarray, second: np.ndarray, error: float, /
) -> float:
    squared = sum(
        (
            (Fraction.from_float(float(a)) - Fraction.from_float(float(b))) ** 2
            for a, b in zip(first, second, strict=True)
        ),
        Fraction(0),
    )
    distance_bound = _sqrt_upper(squared)
    if not math.isfinite(distance_bound) or not math.isfinite(error):
        return math.inf
    distance = Fraction.from_float(distance_bound)
    return _outward_float(distance + Fraction.from_float(error))


def _parameter_midpoint(first: float, last: float, /) -> float:
    lower, upper = sorted((first, last))
    midpoint = float((Fraction.from_float(lower) + Fraction.from_float(upper)) / 2)
    if lower < midpoint < upper:
        return midpoint
    return lower


def _prepare_curve(
    curve: AbstractCurve, first: float, last: float, /
) -> PreparedIntervalFunction | None:
    if not callable(getattr(curve, "bounding_box", None)):
        return prepare_interval_function(
            lambda parameters: curve.evaluate(parameters[..., 0]),
            1,
            batch_capacity=1,
        )
    if isinstance(curve, SurfaceMappedCurve):
        lower, upper = sorted((first, last))
        chart_box = np.asarray(curve.pcurve.bounding_box(lower, upper), dtype=np.float64)
        if (
            chart_box.shape != (2, 2)
            or not np.all(np.isfinite(chart_box))
            or np.any(chart_box[0] > chart_box[1])
        ):
            raise ValueError(
                "A mapped source curve needs a finite ordered p-curve interval."
            )
    return None


def _interval_bounds(
    curve: AbstractCurve,
    evaluator: PreparedIntervalFunction | None,
    first: float,
    last: float,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = sorted((first, last))
    bounding_box = getattr(curve, "bounding_box", None)
    if callable(bounding_box) and evaluator is None:
        bounds = np.asarray(bounding_box(lower, upper), dtype=np.float64)
        if bounds.shape != (2, curve.ambient_dimension):
            raise ValueError(
                "Curve bounding_box must return lower and upper coordinate vectors."
            )
        lo, hi = bounds
    else:
        if evaluator is None:
            raise TypeError(
                "A source curve needs a continuous bounding_box or interval evaluator."
            )
        low, high = evaluator.evaluate(
            np.asarray(((lower,),), dtype=np.float64),
            np.asarray(((upper,),), dtype=np.float64),
        )
        lo = np.asarray(low[0], dtype=np.float64)
        hi = np.asarray(high[0], dtype=np.float64)
    if (
        lo.shape != (curve.ambient_dimension,)
        or hi.shape != lo.shape
        or np.any(np.isnan(lo))
        or np.any(np.isnan(hi))
        or np.any(lo > hi)
    ):
        raise ValueError(
            "Curve interval evaluation did not produce ordered coordinate bounds."
        )
    return lo, hi


@final
class ParametricCurveBoundarySource(StrictModule, NonTrainableState):
    """Continuously enclose and query an explicitly identified curve chain.

    Each source chart is enclosed over its complete authored parameter interval.
    Distance lower bounds use those interval boxes; upper bounds use evaluated
    representatives with interval-bounded evaluation error. Sampling is only a
    consumer-facing view of the same continuous cover and never establishes
    interval authority on its own.
    """

    curves: tuple[AbstractCurve, ...]
    interval_evaluators: tuple[PreparedIntervalFunction | None, ...]
    parameter_ranges: tuple[tuple[float, float], ...] = eqx.field(static=True)
    curve_entity_ids: tuple[str, ...] = eqx.field(static=True)
    occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    chart_ids: tuple[str, ...] = eqx.field(static=True)
    endpoint_source_keys: tuple[
        tuple[tuple[str, tuple[str, ...]] | None, tuple[str, tuple[str, ...]] | None], ...
    ] = eqx.field(static=True)
    point_entity_ids: tuple[str, ...] = eqx.field(static=True)
    point_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    endpoint_point_bounds: tuple[
        tuple[tuple[tuple[float, ...], tuple[float, ...]] | None, ...], ...
    ] = eqx.field(static=True)
    point_lower: np.ndarray
    point_upper: np.ndarray
    source_scope_id: str = eqx.field(static=True)
    boxes: np.ndarray
    representatives: np.ndarray
    representative_covering_radii: np.ndarray
    representative_errors: np.ndarray
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    query_id: str = eqx.field(static=True)

    def __init__(
        self,
        curves: tuple[AbstractCurve, ...],
        parameter_ranges: tuple[tuple[float, float], ...],
        /,
        *,
        source_id: str,
        source_revision: str,
        curve_entity_ids: tuple[str, ...],
        occurrence_paths: tuple[tuple[str, ...], ...],
        chart_ids: tuple[str, ...],
        endpoint_source_keys: tuple[
            tuple[tuple[str, tuple[str, ...]] | None, tuple[str, tuple[str, ...]] | None],
            ...,
        ]
        | None = None,
        endpoint_point_bounds: tuple[
            tuple[tuple[tuple[float, ...], tuple[float, ...]] | None, ...], ...
        ]
        | None = None,
    ) -> None:
        if (
            not isinstance(curves, tuple)
            or not curves
            or any(not isinstance(curve, AbstractCurve) for curve in curves)
        ):
            raise TypeError("curves must be a nonempty tuple of AbstractCurve values.")
        dimensions = tuple(curve.ambient_dimension for curve in curves)
        if len(set(dimensions)) != 1 or dimensions[0] < 1:
            raise ValueError(
                "Source curve charts must share one positive ambient dimension."
            )
        ranges = _parameter_ranges(curves, parameter_ranges)
        identifiers = tuple(
            canonical_identifier(value, "curve_entity_id") for value in curve_entity_ids
        )
        occurrences = tuple(
            tuple(canonical_identifier(part, "occurrence_path") for part in path)
            for path in occurrence_paths
        )
        charts = tuple(canonical_identifier(value, "chart_id") for value in chart_ids)
        if any(
            len(values) != len(curves) for values in (identifiers, occurrences, charts)
        ):
            raise ValueError(
                "Curve entity IDs, occurrences, and chart IDs must align with curves."
            )
        if len(set(charts)) != len(charts):
            raise ValueError("chart_ids must uniquely identify every source curve chart.")
        raw_point_keys: _EndpointSourcePointKeys = (
            tuple((None, None) for _ in curves)
            if endpoint_source_keys is None
            else endpoint_source_keys
        )
        if len(raw_point_keys) != len(curves) or any(
            len(pair) != 2 for pair in raw_point_keys
        ):
            raise ValueError(
                "Endpoint source point keys must align with both ends of every curve."
            )
        point_keys: _EndpointSourcePointKeys = tuple(
            (
                None
                if first_key is None
                else (
                    canonical_identifier(first_key[0], "point_entity_id"),
                    tuple(
                        canonical_identifier(part, "occurrence_path")
                        for part in first_key[1]
                    ),
                ),
                None
                if second_key is None
                else (
                    canonical_identifier(second_key[0], "point_entity_id"),
                    tuple(
                        canonical_identifier(part, "occurrence_path")
                        for part in second_key[1]
                    ),
                ),
            )
            for first_key, second_key in raw_point_keys
        )
        raw_endpoint_bounds = (
            tuple((None, None) for _ in curves)
            if endpoint_point_bounds is None
            else endpoint_point_bounds
        )
        if len(raw_endpoint_bounds) != len(curves) or any(
            len(pair) != 2 for pair in raw_endpoint_bounds
        ):
            raise ValueError(
                "Endpoint point bounds must align with both ends of every curve."
            )
        normalized_endpoint_bounds = []
        for key_pair, bound_pair in zip(point_keys, raw_endpoint_bounds, strict=True):
            normalized_pair = []
            for key, bounds in zip(key_pair, bound_pair, strict=True):
                if bounds is None:
                    normalized_pair.append(None)
                    continue
                if key is None or len(bounds) != 2:
                    raise ValueError(
                        "A point enclosure requires an identified endpoint source."
                    )
                lower = tuple(float(value) for value in bounds[0])
                upper = tuple(float(value) for value in bounds[1])
                if (
                    len(lower) != dimensions[0]
                    or len(upper) != dimensions[0]
                    or not np.all(np.isfinite(lower))
                    or not np.all(np.isfinite(upper))
                    or any(first > last for first, last in zip(lower, upper, strict=True))
                ):
                    raise ValueError(
                        "Endpoint source point bounds must be finite ordered coordinate vectors."
                    )
                normalized_pair.append((lower, upper))
            normalized_endpoint_bounds.append(tuple(normalized_pair))
        normalized_endpoint_bounds = tuple(normalized_endpoint_bounds)

        programs = tuple(
            _prepare_curve(curve, first, last)
            for curve, (first, last) in zip(curves, ranges, strict=True)
        )
        boxes = np.empty((len(curves), 2, dimensions[0]), dtype=np.float64)
        representatives = np.empty((len(curves), dimensions[0]), dtype=np.float64)
        cover_radii = np.empty((len(curves),), dtype=np.float64)
        point_errors = np.empty((len(curves),), dtype=np.float64)
        for row, (curve, evaluator, (first, last)) in enumerate(
            zip(curves, programs, ranges, strict=True)
        ):
            boxes[row, 0], boxes[row, 1] = _interval_bounds(curve, evaluator, first, last)
            midpoint = _parameter_midpoint(first, last)
            sample = np.asarray(
                curve.evaluate(jnp.asarray(midpoint, dtype=jnp.float64)),
                dtype=np.float64,
            )
            if sample.shape != (dimensions[0],) or not np.all(np.isfinite(sample)):
                raise ValueError(
                    "Curve source representatives must evaluate to finite points."
                )
            if np.any(boxes[row, 0] > sample) or np.any(sample > boxes[row, 1]):
                raise ValueError(
                    "Curve interval bounds do not enclose their evaluated representative."
                )
            point_lower, point_upper = _interval_bounds(
                curve, evaluator, midpoint, midpoint
            )
            if np.any(point_lower > sample) or np.any(sample > point_upper):
                raise ValueError(
                    "Curve point error bounds do not enclose their evaluated representative."
                )
            representatives[row] = sample
            cover_radii[row] = _box_radius(boxes[row, 0], boxes[row, 1], sample)
            point_errors[row] = _box_radius(point_lower, point_upper, sample)

        point_bounds_by_key: dict[
            tuple[str, tuple[str, ...]], tuple[np.ndarray, np.ndarray]
        ] = {}
        for row, (curve, evaluator, (first, last), key_pair, bound_pair) in enumerate(
            zip(
                curves,
                programs,
                ranges,
                point_keys,
                normalized_endpoint_bounds,
                strict=True,
            )
        ):
            for endpoint, (parameter, key, supplied_bounds) in enumerate(
                zip((first, last), key_pair, bound_pair, strict=True)
            ):
                if key is None:
                    continue
                lower, upper = _interval_bounds(curve, evaluator, parameter, parameter)
                if supplied_bounds is not None:
                    lower = np.maximum(
                        lower, np.asarray(supplied_bounds[0], dtype=np.float64)
                    )
                    upper = np.minimum(
                        upper, np.asarray(supplied_bounds[1], dtype=np.float64)
                    )
                sample = np.asarray(
                    curve.evaluate(jnp.asarray(parameter, dtype=jnp.float64)),
                    dtype=np.float64,
                )
                if (
                    np.any(lower > upper)
                    or sample.shape != (dimensions[0],)
                    or not np.all(np.isfinite(sample))
                    or np.any(lower > sample)
                    or np.any(sample > upper)
                ):
                    raise ValueError(
                        "Source endpoint bounds disagree with the continuous curve chart."
                    )
                previous = point_bounds_by_key.get(key)
                if previous is not None:
                    lower = np.maximum(lower, previous[0])
                    upper = np.minimum(upper, previous[1])
                    if np.any(lower > upper):
                        raise ValueError(
                            "Repeated source point identities have disjoint enclosures."
                        )
                point_bounds_by_key[key] = (
                    np.array(lower, copy=True),
                    np.array(upper, copy=True),
                )
        point_keys_sorted = tuple(sorted(point_bounds_by_key))
        point_entity_ids = tuple(key[0] for key in point_keys_sorted)
        point_occurrences = tuple(key[1] for key in point_keys_sorted)
        point_lower = (
            np.stack([point_bounds_by_key[key][0] for key in point_keys_sorted])
            if point_keys_sorted
            else np.empty((0, dimensions[0]), dtype=np.float64)
        )
        point_upper = (
            np.stack([point_bounds_by_key[key][1] for key in point_keys_sorted])
            if point_keys_sorted
            else np.empty((0, dimensions[0]), dtype=np.float64)
        )
        for array in (
            boxes,
            representatives,
            cover_radii,
            point_errors,
            point_lower,
            point_upper,
        ):
            array.setflags(write=False)
        source = canonical_identifier(source_id, "source_id")
        revision = canonical_identifier(source_revision, "source_revision")
        self.curves = curves
        self.parameter_ranges = ranges
        self.curve_entity_ids = identifiers
        self.occurrence_paths = occurrences
        self.chart_ids = charts
        self.interval_evaluators = programs
        self.endpoint_source_keys = point_keys
        self.point_entity_ids = point_entity_ids
        self.point_occurrence_paths = point_occurrences
        self.endpoint_point_bounds = normalized_endpoint_bounds
        self.point_lower = point_lower
        self.point_upper = point_upper
        self.boxes = boxes
        self.representatives = representatives
        self.representative_covering_radii = cover_radii
        self.representative_errors = point_errors
        self.source_id = source
        self.source_revision = revision
        self.ambient_dimension = dimensions[0]
        self.source_scope_id = canonical_fingerprint(
            {
                "kind": "parametric-curve-source-scope",
                "source": source,
                "revision": revision,
                "charts": tuple(
                    (
                        identifiers[row],
                        occurrences[row],
                        charts[row],
                        ranges[row],
                    )
                    for row in range(len(curves))
                ),
                "point_endpoints": tuple(
                    (key_pair, bound_pair)
                    for key_pair, bound_pair in zip(
                        point_keys, normalized_endpoint_bounds, strict=True
                    )
                ),
            }
        )
        self.query_id = canonical_fingerprint(
            {
                "kind": "parametric-curve-boundary-query",
                "source": source,
                "revision": revision,
                "curves": array_tree_fingerprint(curves),
                "parameter_ranges": ranges,
                "curve_entity_ids": identifiers,
                "occurrence_paths": occurrences,
                "chart_ids": charts,
                "source_scope_id": self.source_scope_id,
                "boxes": array_tree_fingerprint(boxes),
                "representatives": array_tree_fingerprint(
                    (representatives, cover_radii, point_errors)
                ),
                "point_entity_ids": point_entity_ids,
                "point_occurrence_paths": point_occurrences,
                "point_bounds": array_tree_fingerprint((point_lower, point_upper)),
            }
        )

    def validate_scope(
        self,
        feature_source: _SourceEntityScopes,
        requested_entity_ids: ArrayLike,
        requested_occurrences: tuple[tuple[str, ...], ...],
        /,
    ) -> None:
        """Bind requested feature IDs to complete authored curve chart ranges."""
        from ..meshing._scope import _SourceEntityScopes

        if not isinstance(feature_source, _SourceEntityScopes):
            raise TypeError(
                "feature_source must implement canonical source entity scopes."
            )
        raw_ids = np.asarray(requested_entity_ids)
        if raw_ids.ndim != 1 or raw_ids.dtype.kind not in "iu":
            raise TypeError("requested_entity_ids must be an integer vector.")
        if (
            raw_ids.dtype.kind == "u"
            and raw_ids.size
            and np.any(raw_ids > np.iinfo(np.int64).max)
        ):
            raise ValueError("requested_entity_ids must fit int64.")
        identifiers = np.asarray(raw_ids, dtype=np.int64)
        occurrences = tuple(
            tuple(canonical_identifier(part, "occurrence_path") for part in path)
            for path in requested_occurrences
        )
        if (
            identifiers.size == 0
            or np.unique(identifiers).size != identifiers.size
            or len(occurrences) != identifiers.size
            or getattr(feature_source, "source_id", None) != self.source_id
            or getattr(feature_source, "source_revision", None) != self.source_revision
        ):
            raise ValueError(
                "Requested curve feature scope must bind unique IDs and this source revision."
            )
        keys = tuple(
            feature_source.source_entity_key_for_scope(1, int(identifier))
            for identifier in identifiers.tolist()
        )
        if any(
            path != occurrence
            for (_, path), occurrence in zip(keys, occurrences, strict=True)
        ):
            raise ValueError(
                "Requested curve feature occurrences differ from the source owner."
            )
        query_keys = set(zip(self.curve_entity_ids, self.occurrence_paths, strict=True))
        if any(key not in query_keys for key in keys):
            raise ValueError(
                "Curve source query omits a requested authored feature entity."
            )
        validator = getattr(feature_source, "validate_curve_boundary_query", None)
        if not callable(validator):
            raise TypeError("feature_source cannot validate authored curve chart ranges.")
        validator(self, identifiers)

    def boundary_distance(self, points: ArrayLike, /) -> SourceBoundaryDistance:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.ambient_dimension:
            raise ValueError(
                f"points must have shape (N, {self.ambient_dimension}); got {values.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("points must contain only finite coordinates.")
        lower = np.empty((values.shape[0],), dtype=np.float64)
        upper = np.empty((values.shape[0],), dtype=np.float64)
        for row in range(values.shape[0]):
            point = np.asarray(values[row], dtype=np.float64)
            lower[row] = min(
                _box_distance_lower(point, box[0], box[1]) for box in self.boxes
            )
            upper[row] = min(
                _point_distance_upper(
                    point,
                    np.asarray(self.representatives[sample_row], dtype=np.float64),
                    float(self.representative_errors[sample_row]),
                )
                for sample_row in range(self.representatives.shape[0])
            )
        return SourceBoundaryDistance(lower, upper, "certified")

    def chart_row(self, chart_id: str, /) -> int:
        identifier = canonical_identifier(chart_id, "chart_id")
        try:
            return self.chart_ids.index(identifier)
        except ValueError as error:
            raise ValueError(
                "chart_id is absent from this curve boundary query."
            ) from error

    def source_point_bounds(
        self,
        entity_id: str,
        occurrence_path: tuple[str, ...],
        /,
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (
            canonical_identifier(entity_id, "point_entity_id"),
            tuple(
                canonical_identifier(part, "occurrence_path") for part in occurrence_path
            ),
        )
        keys = tuple(zip(self.point_entity_ids, self.point_occurrence_paths, strict=True))
        try:
            row = keys.index(key)
        except ValueError as error:
            raise ValueError(
                "Source point key is absent from this curve boundary query."
            ) from error
        lower = np.array(self.point_lower[row], dtype=np.float64, copy=True)
        upper = np.array(self.point_upper[row], dtype=np.float64, copy=True)
        lower.setflags(write=False)
        upper.setflags(write=False)
        return lower, upper

    def chart_interval_bounds(
        self, chart_id: str, first: float, last: float, /
    ) -> tuple[np.ndarray, np.ndarray]:
        row = self.chart_row(chart_id)
        interval = _parameter_ranges(
            (self.curves[row],),
            ((first, last),),
        )[0]
        lower, upper = _interval_bounds(
            self.curves[row], self.interval_evaluators[row], interval[0], interval[1]
        )
        lower = np.array(lower, dtype=np.float64, copy=True)
        upper = np.array(upper, dtype=np.float64, copy=True)
        lower.setflags(write=False)
        upper.setflags(write=False)
        return lower, upper

    def chart_point_bounds(
        self, chart_id: str, parameter: float, /
    ) -> tuple[np.ndarray, np.ndarray]:
        row = self.chart_row(chart_id)
        value = finite_real_scalar(parameter, "curve_parameter")
        first, last = self.parameter_ranges[row]
        if value < min(first, last) or value > max(first, last):
            raise ValueError(
                "Curve point parameter lies outside its authored chart range."
            )
        lower, upper = _interval_bounds(
            self.curves[row], self.interval_evaluators[row], value, value
        )
        lower = np.array(lower, dtype=np.float64, copy=True)
        upper = np.array(upper, dtype=np.float64, copy=True)
        lower.setflags(write=False)
        upper.setflags(write=False)
        return lower, upper

    def require_current(self, /) -> None:
        rebuilt = ParametricCurveBoundarySource(
            self.curves,
            self.parameter_ranges,
            source_id=self.source_id,
            source_revision=self.source_revision,
            curve_entity_ids=self.curve_entity_ids,
            occurrence_paths=self.occurrence_paths,
            chart_ids=self.chart_ids,
            endpoint_source_keys=self.endpoint_source_keys,
            endpoint_point_bounds=self.endpoint_point_bounds,
        )
        if rebuilt.query_id != self.query_id:
            raise ValueError("Parametric curve boundary query content changed.")

    def boundary_cover(self, maximum_radius: float, /) -> SourceBoundarySamples:
        """Return a complete interval-certified cover at the requested physical radius."""
        if isinstance(maximum_radius, (bool, np.bool_)):
            raise TypeError(
                "maximum_radius must be a nonnegative finite physical distance."
            )
        radius_limit = finite_real_scalar(maximum_radius, "maximum_radius")
        if radius_limit < 0.0:
            raise ValueError(
                "maximum_radius must be a nonnegative finite physical distance."
            )
        points: list[np.ndarray] = []
        covering_radii: list[float] = []
        point_errors: list[float] = []
        for curve, evaluator, (first, last) in zip(
            self.curves, self.interval_evaluators, self.parameter_ranges, strict=True
        ):
            lower, upper = sorted((first, last))
            intervals = [(lower, upper)]
            while intervals:
                interval_first, interval_last = intervals.pop()
                bounds_lower, bounds_upper = _interval_bounds(
                    curve, evaluator, interval_first, interval_last
                )
                parameter = _parameter_midpoint(interval_first, interval_last)
                sample = np.asarray(
                    curve.evaluate(jnp.asarray(parameter, dtype=jnp.float64)),
                    dtype=np.float64,
                )
                point_lower, point_upper = _interval_bounds(
                    curve, evaluator, parameter, parameter
                )
                if (
                    sample.shape != (self.ambient_dimension,)
                    or not np.all(np.isfinite(sample))
                    or np.any(point_lower > sample)
                    or np.any(sample > point_upper)
                ):
                    raise ValueError(
                        "Curve boundary cover has no finite representative enclosure."
                    )
                cover_radius = _box_distance_upper(
                    bounds_lower, bounds_upper, point_lower, point_upper
                )
                point_error = _box_radius(point_lower, point_upper, sample)
                if cover_radius <= radius_limit:
                    points.append(sample)
                    covering_radii.append(cover_radius)
                    point_errors.append(point_error)
                    continue
                if not interval_first < parameter < interval_last:
                    raise ArithmeticError(
                        "Curve boundary cover cannot subdivide further at the requested physical radius."
                    )
                intervals.append((parameter, interval_last))
                intervals.append((interval_first, parameter))
        return SourceBoundarySamples(
            np.asarray(points, dtype=np.float64).reshape((-1, self.ambient_dimension)),
            np.asarray(covering_radii, dtype=np.float64),
            np.asarray(point_errors, dtype=np.float64),
            "certified",
            complete=True,
        )


__all__ = ["ParametricCurveBoundarySource"]
