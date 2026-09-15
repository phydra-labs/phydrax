#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._connection import LeviCivitaConnection
from ._metric import LorentzianMetric
from ._metric_domain import MetricDomainEvidence
from ._spacetime_conventions import RelativityConvention
from ._utils import _coordinates, _pointwise_array


class MetricInnerProductEvidence(StrictModule, NonTrainableState):
    """Evaluated frame Gram matrix and evidence for one declared target Gram matrix."""

    values: Array
    expected: Array
    residuals: Array
    maximum_absolute_residual: Array
    finite: Array
    domain_valid: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    tolerance: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        expected: ArrayLike,
        /,
        *,
        domain_valid: ArrayLike,
        derivative_valid: ArrayLike,
        tolerance: float,
        evidence_id: str,
    ):
        values_ = jnp.asarray(values)
        expected_ = jnp.asarray(expected, dtype=values_.dtype)
        if values_.ndim < 2 or values_.shape[-2] != values_.shape[-1]:
            raise ValueError("Inner-product values must have trailing square axes.")
        if expected_.shape[-2:] != values_.shape[-2:]:
            raise ValueError("Expected inner products must have the same trailing shape.")
        expected_ = jnp.broadcast_to(expected_, values_.shape)
        tolerance_ = float(tolerance)
        if tolerance_ < 0.0:
            raise ValueError("Inner-product tolerance must be non-negative.")
        if not str(evidence_id):
            raise ValueError("evidence_id must be nonempty.")
        leading_shape = values_.shape[:-2]
        domain = jnp.broadcast_to(jnp.asarray(domain_valid, dtype=bool), leading_shape)
        derivative = jnp.broadcast_to(
            jnp.asarray(derivative_valid, dtype=bool), leading_shape
        )
        residuals = values_ - expected_
        maximum = jnp.max(jnp.abs(residuals), axis=(-2, -1))
        finite = jnp.all(jnp.isfinite(values_), axis=(-2, -1)) & jnp.all(
            jnp.isfinite(expected_), axis=(-2, -1)
        )
        physical = finite & domain
        self.values = values_
        self.expected = expected_
        self.residuals = residuals
        self.maximum_absolute_residual = maximum
        self.finite = finite
        self.domain_valid = domain
        self.physically_valid = physical
        self.qualified = physical & (maximum <= tolerance_)
        self.derivative_valid = derivative & physical
        self.tolerance = tolerance_
        self.evidence_id = str(evidence_id)


class KillingInnerProductEvidence(StrictModule, NonTrainableState):
    """Causal and orbit-plane evidence for stationary and axial generators."""

    stationary_norm: Array
    stationary_axial_inner_product: Array
    axial_norm: Array
    orbit_gram_determinant: Array
    finite: Array
    stationary_timelike: Array
    axial_spacelike: Array
    orbit_plane_lorentzian: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        stationary_norm: ArrayLike,
        stationary_axial_inner_product: ArrayLike,
        axial_norm: ArrayLike,
        /,
        *,
        timelike_sign: int,
        evidence_id: str,
    ):
        stationary_norm_ = jnp.asarray(stationary_norm)
        cross_ = jnp.asarray(stationary_axial_inner_product, dtype=stationary_norm_.dtype)
        axial_norm_ = jnp.asarray(axial_norm, dtype=stationary_norm_.dtype)
        leading_shape = jnp.broadcast_shapes(
            stationary_norm_.shape, cross_.shape, axial_norm_.shape
        )
        stationary_norm_ = jnp.broadcast_to(stationary_norm_, leading_shape)
        cross_ = jnp.broadcast_to(cross_, leading_shape)
        axial_norm_ = jnp.broadcast_to(axial_norm_, leading_shape)
        if timelike_sign not in (-1, 1):
            raise ValueError("timelike_sign must be -1 or +1.")
        if not str(evidence_id):
            raise ValueError("evidence_id must be nonempty.")
        spatial_sign = -timelike_sign
        determinant = stationary_norm_ * axial_norm_ - cross_**2
        finite = (
            jnp.isfinite(stationary_norm_)
            & jnp.isfinite(cross_)
            & jnp.isfinite(axial_norm_)
        )
        stationary_timelike = timelike_sign * stationary_norm_ > 0.0
        axial_spacelike = spatial_sign * axial_norm_ > 0.0
        orbit_plane_lorentzian = determinant < 0.0
        physical = finite & axial_spacelike & orbit_plane_lorentzian
        scale = 1.0 + jnp.maximum(jnp.abs(axial_norm_), jnp.abs(determinant))
        margin = jnp.sqrt(jnp.finfo(stationary_norm_.dtype).eps) * scale
        self.stationary_norm = stationary_norm_
        self.stationary_axial_inner_product = cross_
        self.axial_norm = axial_norm_
        self.orbit_gram_determinant = determinant
        self.finite = finite
        self.stationary_timelike = stationary_timelike
        self.axial_spacelike = axial_spacelike
        self.orbit_plane_lorentzian = orbit_plane_lorentzian
        self.physically_valid = physical
        self.qualified = physical
        self.derivative_valid = (
            physical & (-determinant > margin) & (spatial_sign * axial_norm_ > margin)
        )
        self.evidence_id = str(evidence_id)


class OrthonormalTetrad(StrictModule, NonTrainableState):
    """Four oriented frame vectors, their dual coframe, and local evidence."""

    vectors: Array
    dual_covectors: Array
    inner_products: MetricInnerProductEvidence
    convention: RelativityConvention
    domain: MetricDomainEvidence
    orientation: Array
    orientation_residual: Array
    time_direction_residual: Array
    finite: Array
    domain_valid: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    tetrad_id: str = eqx.field(static=True)

    def __init__(
        self,
        vectors: ArrayLike,
        dual_covectors: ArrayLike,
        inner_products: MetricInnerProductEvidence,
        orientation: ArrayLike,
        orientation_residual: ArrayLike,
        time_direction_residual: ArrayLike,
        finite: ArrayLike,
        domain_valid: ArrayLike,
        physically_valid: ArrayLike,
        qualified: ArrayLike,
        derivative_valid: ArrayLike,
        /,
        *,
        convention: RelativityConvention,
        domain: MetricDomainEvidence,
        tetrad_id: str,
    ):
        vectors_ = jnp.asarray(vectors)
        dual_ = jnp.asarray(dual_covectors, dtype=vectors_.dtype)
        if vectors_.shape[-2:] != (4, 4):
            raise ValueError(
                "Orthonormal tetrad vectors must have trailing shape (4, 4)."
            )
        if dual_.shape != vectors_.shape:
            raise ValueError("Tetrad dual covectors must have the same shape as vectors.")
        if not isinstance(inner_products, MetricInnerProductEvidence):
            raise TypeError("inner_products must be MetricInnerProductEvidence.")
        if not isinstance(convention, RelativityConvention):
            raise TypeError("convention must be RelativityConvention.")
        if not isinstance(domain, MetricDomainEvidence):
            raise TypeError("domain must be MetricDomainEvidence.")
        if not str(tetrad_id):
            raise ValueError("tetrad_id must be nonempty.")
        self.vectors = vectors_
        self.dual_covectors = dual_
        self.inner_products = inner_products
        self.convention = convention
        self.domain = domain
        self.orientation = jnp.asarray(orientation)
        self.orientation_residual = jnp.asarray(orientation_residual)
        self.time_direction_residual = jnp.asarray(time_direction_residual)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.domain_valid = jnp.asarray(domain_valid, dtype=bool)
        self.physically_valid = jnp.asarray(physically_valid, dtype=bool)
        self.qualified = jnp.asarray(qualified, dtype=bool)
        self.derivative_valid = jnp.asarray(derivative_valid, dtype=bool)
        self.tetrad_id = str(tetrad_id)

    @property
    def time_vector(self) -> Array:
        return self.vectors[..., 0, :]

    @property
    def spatial_vectors(self) -> Array:
        return self.vectors[..., 1:, :]


class PrincipalNullTetrad(StrictModule, NonTrainableState):
    """Kinnersley-normalized Kerr principal null frame and its dual coframe."""

    vectors: Array
    dual_covectors: Array
    inner_products: MetricInnerProductEvidence
    convention: RelativityConvention
    domain: MetricDomainEvidence
    finite: Array
    domain_valid: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    tetrad_id: str = eqx.field(static=True)

    def __init__(
        self,
        vectors: ArrayLike,
        dual_covectors: ArrayLike,
        inner_products: MetricInnerProductEvidence,
        finite: ArrayLike,
        domain_valid: ArrayLike,
        physically_valid: ArrayLike,
        qualified: ArrayLike,
        derivative_valid: ArrayLike,
        /,
        *,
        convention: RelativityConvention,
        domain: MetricDomainEvidence,
        tetrad_id: str,
    ):
        vectors_ = jnp.asarray(vectors)
        dual_ = jnp.asarray(dual_covectors, dtype=vectors_.dtype)
        if vectors_.shape[-2:] != (4, 4):
            raise ValueError(
                "Principal null tetrad vectors must have trailing shape (4, 4)."
            )
        if dual_.shape != vectors_.shape:
            raise ValueError("Null-tetrad dual covectors must match the vector shape.")
        if not isinstance(inner_products, MetricInnerProductEvidence):
            raise TypeError("inner_products must be MetricInnerProductEvidence.")
        if not isinstance(convention, RelativityConvention):
            raise TypeError("convention must be RelativityConvention.")
        if not isinstance(domain, MetricDomainEvidence):
            raise TypeError("domain must be MetricDomainEvidence.")
        if not str(tetrad_id):
            raise ValueError("tetrad_id must be nonempty.")
        self.vectors = vectors_
        self.dual_covectors = dual_
        self.inner_products = inner_products
        self.convention = convention
        self.domain = domain
        self.finite = jnp.asarray(finite, dtype=bool)
        self.domain_valid = jnp.asarray(domain_valid, dtype=bool)
        self.physically_valid = jnp.asarray(physically_valid, dtype=bool)
        self.qualified = jnp.asarray(qualified, dtype=bool)
        self.derivative_valid = jnp.asarray(derivative_valid, dtype=bool)
        self.tetrad_id = str(tetrad_id)

    @property
    def outgoing(self) -> Array:
        return self.vectors[..., 0, :]

    @property
    def ingoing(self) -> Array:
        return self.vectors[..., 1, :]

    @property
    def polarization(self) -> Array:
        return self.vectors[..., 2, :]

    @property
    def polarization_conjugate(self) -> Array:
        return self.vectors[..., 3, :]


class TetradParallelTransportEvidence(StrictModule, NonTrainableState):
    """Directional covariant-derivative residuals for a tetrad field."""

    residuals: Array
    maximum_absolute_residual: Array
    finite: Array
    domain_valid: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    tolerance: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        residuals: ArrayLike,
        /,
        *,
        domain_valid: ArrayLike,
        derivative_valid: ArrayLike,
        tolerance: float,
        evidence_id: str,
    ):
        residuals_ = jnp.asarray(residuals)
        if residuals_.shape[-2:] != (4, 4):
            raise ValueError("Transport residuals must have trailing shape (4, 4).")
        tolerance_ = float(tolerance)
        if tolerance_ < 0.0:
            raise ValueError("Transport tolerance must be non-negative.")
        if not str(evidence_id):
            raise ValueError("evidence_id must be nonempty.")
        leading_shape = residuals_.shape[:-2]
        domain = jnp.broadcast_to(jnp.asarray(domain_valid, dtype=bool), leading_shape)
        derivative = jnp.broadcast_to(
            jnp.asarray(derivative_valid, dtype=bool), leading_shape
        )
        finite = jnp.all(jnp.isfinite(residuals_), axis=(-2, -1))
        maximum = jnp.max(jnp.abs(residuals_), axis=(-2, -1))
        physical = finite & domain
        self.residuals = residuals_
        self.maximum_absolute_residual = maximum
        self.finite = finite
        self.domain_valid = domain
        self.physically_valid = physical
        self.qualified = physical & (maximum <= tolerance_)
        self.derivative_valid = derivative & physical
        self.tolerance = tolerance_
        self.evidence_id = str(evidence_id)


def _require_four_metric(metric: LorentzianMetric, /) -> None:
    if not isinstance(metric, LorentzianMetric):
        raise TypeError("metric must be a LorentzianMetric.")
    if metric.chart.dimension != 4:
        raise ValueError("Killing-tetrad calculus requires a four-dimensional metric.")


def _axis(axis: int, /, *, name: str) -> int:
    resolved = int(axis)
    if not 0 <= resolved < 4:
        raise ValueError(f"{name} must be an axis in [0, 4).")
    return resolved


def _resolve_convention(
    metric: LorentzianMetric,
    convention: RelativityConvention | None,
    /,
) -> RelativityConvention:
    if convention is None:
        return RelativityConvention(metric_signature=metric.convention)
    if not isinstance(convention, RelativityConvention):
        raise TypeError("convention must be RelativityConvention or None.")
    if convention.metric_signature != metric.convention:
        raise ValueError("Relativity convention and metric signature must match.")
    return convention


def _orientation_sign(value: int, /, *, name: str) -> int:
    resolved = int(value)
    if resolved not in (-1, 1):
        raise ValueError(f"{name} must be either -1 or +1.")
    return resolved


def _identifier(
    kind: str,
    metric: LorentzianMetric,
    axes: tuple[int, ...],
    source_id: str | None,
    /,
    *,
    convention: RelativityConvention | None = None,
) -> str:
    source = "unbound" if source_id is None else str(source_id)
    if not source:
        raise ValueError("source_id must be nonempty when supplied.")
    convention_ = _resolve_convention(metric, convention)
    return canonical_fingerprint(
        {
            "kind": kind,
            "chart": {
                "name": metric.chart.name,
                "coordinates": list(metric.chart.coordinates),
            },
            "convention": convention_.convention_id,
            "axes": list(axes),
            "source": source,
        }
    )


def _domain_evidence(
    metric: LorentzianMetric,
    margin: Array,
    /,
    *,
    extra_valid: ArrayLike,
    boundary_tolerance: ArrayLike,
    domain_id: str,
) -> MetricDomainEvidence:
    return MetricDomainEvidence.from_margin(
        margin,
        chart=metric.chart,
        domain_id=domain_id,
        boundary_tolerance=boundary_tolerance,
        extra_valid=extra_valid,
    )


def stationary_killing_vector(
    coordinates: ArrayLike,
    /,
    *,
    time_axis: int = 0,
    orientation: int = 1,
) -> Array:
    """Return the oriented coordinate-stationary generator with batch shape."""
    axis = _axis(time_axis, name="time_axis")
    sign = _orientation_sign(orientation, name="orientation")
    values = _coordinates(coordinates, 4)
    return jnp.zeros_like(values).at[..., axis].set(sign)


def axial_killing_vector(
    coordinates: ArrayLike,
    /,
    *,
    axial_axis: int = 3,
    orientation: int = 1,
) -> Array:
    """Return the oriented coordinate-axial generator with batch shape."""
    axis = _axis(axial_axis, name="axial_axis")
    sign = _orientation_sign(orientation, name="orientation")
    values = _coordinates(coordinates, 4)
    return jnp.zeros_like(values).at[..., axis].set(sign)


def stationary_axial_inner_product_evidence(
    metric: LorentzianMetric,
    coordinates: ArrayLike,
    /,
    *,
    time_axis: int = 0,
    axial_axis: int = 3,
    convention: RelativityConvention | None = None,
    source_id: str | None = None,
) -> KillingInnerProductEvidence:
    """Evaluate the stationary/axial Gram determinant and causal classifications."""
    _require_four_metric(metric)
    convention_ = _resolve_convention(metric, convention)
    time_axis_ = _axis(time_axis, name="time_axis")
    axial_axis_ = _axis(axial_axis, name="axial_axis")
    if time_axis_ == axial_axis_:
        raise ValueError("time_axis and axial_axis must be distinct.")
    values = _coordinates(coordinates, 4)
    matrix = metric(values)
    stationary = stationary_killing_vector(
        values,
        time_axis=time_axis_,
        orientation=convention_.future_time_orientation,
    )
    axial = axial_killing_vector(
        values,
        axial_axis=axial_axis_,
        orientation=convention_.azimuthal_orientation,
    )
    stationary_norm = ein.contract("...i,...ij,...j->...", stationary, matrix, stationary)
    cross = ein.contract("...i,...ij,...j->...", stationary, matrix, axial)
    axial_norm = ein.contract("...i,...ij,...j->...", axial, matrix, axial)
    evidence_id = _identifier(
        "stationary-axial-inner-products-v1",
        metric,
        (time_axis_, axial_axis_),
        source_id,
        convention=convention_,
    )
    return KillingInnerProductEvidence(
        stationary_norm,
        cross,
        axial_norm,
        timelike_sign=metric.timelike_sign,
        evidence_id=evidence_id,
    )


def metric_inner_product_evidence(
    metric: LorentzianMetric,
    vectors: ArrayLike,
    expected_inner_products: ArrayLike,
    coordinates: ArrayLike,
    /,
    *,
    domain_valid: ArrayLike = True,
    derivative_valid: ArrayLike = True,
    tolerance: float = 1e-8,
    source_id: str | None = None,
) -> MetricInnerProductEvidence:
    """Evaluate all metric pairings of row-stored vectors against a target Gram matrix."""
    _require_four_metric(metric)
    points = _coordinates(coordinates, 4)
    matrix = metric(points)
    vectors_ = jnp.asarray(vectors)
    if vectors_.shape[-1:] != (4,):
        raise ValueError("Frame vectors must have trailing coordinate dimension four.")
    leading_shape = jnp.broadcast_shapes(matrix.shape[:-2], vectors_.shape[:-2])
    matrix = jnp.broadcast_to(matrix, leading_shape + (4, 4))
    vectors_ = jnp.broadcast_to(
        vectors_, leading_shape + (vectors_.shape[-2], vectors_.shape[-1])
    )
    values = ein.contract("...ai,...ij,...bj->...ab", vectors_, matrix, vectors_)
    evidence_id = _identifier(
        "metric-inner-product-evidence-v1",
        metric,
        (vectors_.shape[-2],),
        source_id,
    )
    return MetricInnerProductEvidence(
        values,
        expected_inner_products,
        domain_valid=domain_valid,
        derivative_valid=derivative_valid,
        tolerance=tolerance,
        evidence_id=evidence_id,
    )


def killing_equation_residual(
    metric: LorentzianMetric,
    vector_field: Callable[[Array], Array],
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return the symmetric Killing-equation residual ``∇_(μ ξ_ν)``."""
    _require_four_metric(metric)
    if not callable(vector_field):
        raise TypeError("vector_field must be callable.")

    def point_residual(point: Array, /) -> Array:
        vector = jnp.asarray(vector_field(point))
        if vector.shape != (4,):
            raise ValueError("A Killing vector field must return shape (4,).")
        matrix = metric(point)
        metric_derivative = jax.jacfwd(lambda value: metric(value))(point)
        vector_derivative = jax.jacfwd(vector_field)(point)
        lie_derivative = ein.contract(
            "r,mnr->mn", vector, metric_derivative
        ) + ein.contract("rn,rm->mn", matrix, vector_derivative)
        lie_derivative = lie_derivative + ein.contract(
            "mr,rn->mn", matrix, vector_derivative
        )
        return 0.5 * lie_derivative

    return _pointwise_array(point_residual, coordinates, 4)


def maximum_killing_equation_residual(
    metric: LorentzianMetric,
    vector_field: Callable[[Array], Array],
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return the pointwise maximum absolute Killing-equation residual."""
    residual = killing_equation_residual(metric, vector_field, coordinates)
    return jnp.max(jnp.abs(residual), axis=(-2, -1))


def _orthonormal_target(metric: LorentzianMetric, dtype: jnp.dtype, /) -> Array:
    timelike_sign = metric.timelike_sign
    return jnp.diag(
        jnp.asarray(
            (timelike_sign, -timelike_sign, -timelike_sign, -timelike_sign),
            dtype=dtype,
        )
    )


def _orthonormal_tetrad_from_matrix(
    metric: LorentzianMetric,
    matrix: Array,
    vectors: Array,
    /,
    *,
    convention: RelativityConvention,
    domain: MetricDomainEvidence,
    derivative_valid: ArrayLike,
    tolerance: float,
    time_axis: int,
    tetrad_id: str,
) -> OrthonormalTetrad:
    leading_shape = jnp.broadcast_shapes(matrix.shape[:-2], vectors.shape[:-2])
    if domain.margin.shape != leading_shape:
        raise ValueError(
            "Tetrad domain evidence shape must match the coordinate batch shape."
        )
    if not domain.chart.compatible_with(metric.chart):
        raise ValueError("Tetrad domain evidence and metric charts must match.")
    matrix = jnp.broadcast_to(matrix, leading_shape + (4, 4))
    vectors = jnp.broadcast_to(vectors, leading_shape + (4, 4))
    target = _orthonormal_target(metric, vectors.dtype)
    gram = ein.contract("...ai,...ij,...bj->...ab", vectors, matrix, vectors)
    inner_products = MetricInnerProductEvidence(
        gram,
        target,
        domain_valid=domain.physically_valid,
        derivative_valid=domain.derivative_valid & derivative_valid,
        tolerance=tolerance,
        evidence_id=canonical_fingerprint(
            {"kind": "orthonormal-tetrad-inner-products-v1", "tetrad": tetrad_id}
        ),
    )
    dual = ein.contract("ab,...bi,...ij->...aj", target, vectors, matrix)
    determinant = jnp.linalg.det(vectors)
    orientation = jnp.sign(determinant)
    orientation_residual = jnp.abs(orientation - convention.spacetime_orientation)
    time_component = vectors[..., 0, time_axis]
    time_direction_residual = jnp.maximum(
        -convention.future_time_orientation * time_component, 0.0
    )
    finite = (
        inner_products.finite
        & jnp.all(jnp.isfinite(vectors), axis=(-2, -1))
        & jnp.all(jnp.isfinite(dual), axis=(-2, -1))
    )
    causal_diagonal = jnp.diagonal(gram, axis1=-2, axis2=-1)
    timelike_sign = metric.timelike_sign
    causal = (timelike_sign * causal_diagonal[..., 0] > 0.0) & jnp.all(
        -timelike_sign * causal_diagonal[..., 1:] > 0.0, axis=-1
    )
    oriented = orientation == convention.spacetime_orientation
    future_directed = convention.future_time_orientation * time_component > 0.0
    physical = finite & domain.physically_valid & causal & oriented & future_directed
    qualified = physical & inner_products.qualified
    derivative = jnp.broadcast_to(
        jnp.asarray(derivative_valid, dtype=bool), leading_shape
    )
    return OrthonormalTetrad(
        vectors,
        dual,
        inner_products,
        orientation,
        orientation_residual,
        time_direction_residual,
        finite,
        domain.physically_valid,
        physical,
        qualified,
        domain.derivative_valid & derivative & physical,
        convention=convention,
        domain=domain,
        tetrad_id=tetrad_id,
    )


def orthonormal_tetrad(
    metric: LorentzianMetric,
    vectors: ArrayLike,
    coordinates: ArrayLike,
    /,
    *,
    convention: RelativityConvention | None = None,
    domain: MetricDomainEvidence | None = None,
    tolerance: float = 1e-8,
    time_axis: int = 0,
    source_id: str | None = None,
) -> OrthonormalTetrad:
    """Construct an evaluated oriented orthonormal-tetrad evidence record."""
    _require_four_metric(metric)
    convention_ = _resolve_convention(metric, convention)
    time_axis_ = _axis(time_axis, name="time_axis")
    points = _coordinates(coordinates, 4)
    matrix = metric(points)
    vectors_ = jnp.asarray(vectors)
    if jnp.iscomplexobj(vectors_):
        raise ValueError("Orthonormal tetrad vectors must be real.")
    if vectors_.shape[-2:] != (4, 4):
        raise ValueError("Orthonormal tetrad vectors must have trailing shape (4, 4).")
    leading_shape = jnp.broadcast_shapes(matrix.shape[:-2], vectors_.shape[:-2])
    tetrad_id = _identifier(
        "orthonormal-tetrad-v1",
        metric,
        (time_axis_,),
        source_id,
        convention=convention_,
    )
    if domain is None:
        domain_ = _domain_evidence(
            metric,
            jnp.ones(leading_shape, dtype=matrix.dtype),
            extra_valid=True,
            boundary_tolerance=0.0,
            domain_id=canonical_fingerprint(
                {"kind": "orthonormal-tetrad-domain-v1", "tetrad": tetrad_id}
            ),
        )
    else:
        domain_ = domain
    return _orthonormal_tetrad_from_matrix(
        metric,
        matrix,
        vectors_,
        convention=convention_,
        domain=domain_,
        derivative_valid=True,
        tolerance=tolerance,
        time_axis=time_axis_,
        tetrad_id=tetrad_id,
    )


def zamo_observer_tetrad(
    metric: LorentzianMetric,
    coordinates: ArrayLike,
    /,
    *,
    time_axis: int = 0,
    radial_axis: int = 1,
    polar_axis: int = 2,
    axial_axis: int = 3,
    convention: RelativityConvention | None = None,
    tolerance: float = 1e-8,
    source_id: str | None = None,
) -> OrthonormalTetrad:
    """Construct the future ZAMO-like frame, masked outside its timelike orbit domain."""
    _require_four_metric(metric)
    convention_ = _resolve_convention(metric, convention)
    axes = tuple(
        _axis(value, name=name)
        for value, name in (
            (time_axis, "time_axis"),
            (radial_axis, "radial_axis"),
            (polar_axis, "polar_axis"),
            (axial_axis, "axial_axis"),
        )
    )
    if len(set(axes)) != 4:
        raise ValueError("ZAMO tetrad axes must be distinct.")
    time_axis_, radial_axis_, polar_axis_, axial_axis_ = axes
    points = _coordinates(coordinates, 4)
    matrix = metric(points)
    leading_shape = matrix.shape[:-2]
    dtype = matrix.dtype
    coordinate_basis = jnp.broadcast_to(jnp.eye(4, dtype=dtype), leading_shape + (4, 4))
    stationary = coordinate_basis[..., time_axis_, :]
    axial = coordinate_basis[..., axial_axis_, :]
    axial_norm = matrix[..., axial_axis_, axial_axis_]
    stationary_axial = matrix[..., time_axis_, axial_axis_]
    tiny = jnp.finfo(dtype).tiny
    axial_denominator_valid = jnp.isfinite(axial_norm) & (jnp.abs(axial_norm) > tiny)
    safe_axial_norm = jnp.where(axial_denominator_valid, axial_norm, 1.0)
    angular_velocity = -stationary_axial / safe_axial_norm
    time_candidate = stationary + angular_velocity[..., None] * axial
    time_candidate = convention_.future_time_orientation * time_candidate
    time_norm = ein.contract(
        "...i,...ij,...j->...", time_candidate, matrix, time_candidate
    )
    timelike_sign = metric.timelike_sign
    spatial_sign = -timelike_sign
    signed_time_norm = timelike_sign * time_norm
    time_valid = (
        axial_denominator_valid
        & jnp.isfinite(signed_time_norm)
        & (signed_time_norm > 0.0)
        & (spatial_sign * axial_norm > 0.0)
    )
    time_scale = jax.lax.rsqrt(jnp.where(time_valid, signed_time_norm, 1.0))
    time_vector = jnp.where(
        time_valid[..., None], time_candidate * time_scale[..., None], 0.0
    )
    frame_vectors = [time_vector]
    frame_signs = [timelike_sign]
    normalized_margins = [
        signed_time_norm / (1.0 + jnp.abs(time_norm)),
        spatial_sign * axial_norm / (1.0 + jnp.abs(axial_norm)),
    ]

    for seed_axis in (radial_axis_, polar_axis_, axial_axis_):
        candidate = coordinate_basis[..., seed_axis, :]
        if seed_axis == axial_axis_:
            candidate = convention_.azimuthal_orientation * candidate
        for basis, sign in zip(frame_vectors, frame_signs, strict=True):
            pairing = ein.contract("...i,...ij,...j->...", candidate, matrix, basis)
            candidate = candidate - (sign * pairing)[..., None] * basis
        norm = ein.contract("...i,...ij,...j->...", candidate, matrix, candidate)
        signed_norm = spatial_sign * norm
        vector_valid = jnp.isfinite(signed_norm) & (signed_norm > 0.0)
        vector_scale = jax.lax.rsqrt(jnp.where(vector_valid, signed_norm, 1.0))
        vector = jnp.where(
            vector_valid[..., None], candidate * vector_scale[..., None], 0.0
        )
        frame_vectors.append(vector)
        frame_signs.append(spatial_sign)
        normalized_margins.append(signed_norm / (1.0 + jnp.abs(norm)))

    vectors = jnp.stack(frame_vectors, axis=-2)
    determinant_sign = jnp.sign(jnp.linalg.det(vectors))
    orientation_correction = convention_.spacetime_orientation * determinant_sign
    vectors = vectors.at[..., 1, :].multiply(
        jnp.where(determinant_sign != 0.0, orientation_correction, 1.0)[..., None]
    )
    domain_margin = jnp.min(jnp.stack(normalized_margins, axis=-1), axis=-1)
    finite_input = (
        jnp.all(jnp.isfinite(matrix), axis=(-2, -1))
        & axial_denominator_valid
        & jnp.isfinite(domain_margin)
    )
    tetrad_id = _identifier(
        "zamo-observer-tetrad-v1",
        metric,
        axes,
        source_id,
        convention=convention_,
    )
    domain = _domain_evidence(
        metric,
        domain_margin,
        extra_valid=finite_input & (domain_margin > 0.0),
        boundary_tolerance=jnp.sqrt(jnp.finfo(dtype).eps),
        domain_id=canonical_fingerprint(
            {"kind": "zamo-observer-domain-v1", "tetrad": tetrad_id}
        ),
    )
    vectors = jnp.where(domain.physically_valid[..., None, None], vectors, 0.0)
    return _orthonormal_tetrad_from_matrix(
        metric,
        matrix,
        vectors,
        convention=convention_,
        domain=domain,
        derivative_valid=True,
        tolerance=tolerance,
        time_axis=time_axis_,
        tetrad_id=tetrad_id,
    )


def kerr_principal_null_tetrad(
    metric: LorentzianMetric,
    mass: ArrayLike,
    spin_parameter: ArrayLike,
    coordinates: ArrayLike,
    /,
    *,
    time_axis: int = 0,
    radial_axis: int = 1,
    polar_axis: int = 2,
    axial_axis: int = 3,
    convention: RelativityConvention | None = None,
    tolerance: float = 1e-8,
    source_id: str | None = None,
) -> PrincipalNullTetrad:
    """Evaluate the exterior Boyer-Lindquist Kinnersley principal null tetrad."""
    _require_four_metric(metric)
    convention_ = _resolve_convention(metric, convention)
    axes = tuple(
        _axis(value, name=name)
        for value, name in (
            (time_axis, "time_axis"),
            (radial_axis, "radial_axis"),
            (polar_axis, "polar_axis"),
            (axial_axis, "axial_axis"),
        )
    )
    if len(set(axes)) != 4:
        raise ValueError("Principal-null-tetrad axes must be distinct.")
    time_axis_, radial_axis_, polar_axis_, axial_axis_ = axes
    points = _coordinates(coordinates, 4)
    matrix = metric(points)
    mass_ = jnp.asarray(mass, dtype=matrix.dtype)
    spin_ = jnp.asarray(spin_parameter, dtype=matrix.dtype)
    if mass_.shape != () or spin_.shape != ():
        raise ValueError("Kerr mass and spin_parameter must be scalars.")
    radius = points[..., radial_axis_]
    polar = points[..., polar_axis_]
    sine = jnp.sin(polar)
    cosine = jnp.cos(polar)
    delta = radius**2 - 2.0 * mass_ * radius + spin_**2
    sigma = radius**2 + spin_**2 * cosine**2
    subextremal = jnp.abs(spin_) <= mass_
    horizon_discriminant = jnp.maximum((mass_ - spin_) * (mass_ + spin_), 0.0)
    outer_horizon = mass_ + jnp.sqrt(horizon_discriminant)
    tiny = jnp.finfo(matrix.dtype).tiny
    denominator_valid = (
        jnp.isfinite(delta)
        & jnp.isfinite(sigma)
        & jnp.isfinite(sine)
        & (jnp.abs(delta) > tiny)
        & (sigma > tiny)
        & (jnp.abs(sine) > tiny)
    )
    parameter_valid = jnp.isfinite(mass_) & jnp.isfinite(spin_) & (mass_ > 0.0)
    safe_delta = jnp.where(denominator_valid, delta, 1.0)
    safe_sigma = jnp.where(denominator_valid, sigma, 1.0)
    safe_sine = jnp.where(denominator_valid, sine, 1.0)
    complex_dtype = jnp.result_type(matrix.dtype, jnp.complex64)
    leading_shape = points.shape[:-1]

    outgoing = jnp.zeros(leading_shape + (4,), dtype=complex_dtype)
    outgoing = outgoing.at[..., time_axis_].set((radius**2 + spin_**2) / safe_delta)
    outgoing = outgoing.at[..., radial_axis_].set(1.0)
    outgoing = outgoing.at[..., axial_axis_].set(spin_ / safe_delta)

    ingoing = jnp.zeros(leading_shape + (4,), dtype=complex_dtype)
    ingoing = ingoing.at[..., time_axis_].set((radius**2 + spin_**2) / (2.0 * safe_sigma))
    ingoing = ingoing.at[..., radial_axis_].set(-safe_delta / (2.0 * safe_sigma))
    ingoing = ingoing.at[..., axial_axis_].set(spin_ / (2.0 * safe_sigma))
    outgoing = convention_.future_time_orientation * outgoing
    ingoing = convention_.future_time_orientation * ingoing

    polarization_denominator = jnp.sqrt(2.0) * (radius + 1j * spin_ * cosine)
    safe_polarization_denominator = jnp.where(
        denominator_valid, polarization_denominator, 1.0 + 0.0j
    )
    polarization = jnp.zeros(leading_shape + (4,), dtype=complex_dtype)
    polarization = polarization.at[..., time_axis_].set(
        1j * spin_ * sine / safe_polarization_denominator
    )
    polarization = polarization.at[..., polar_axis_].set(
        1.0 / safe_polarization_denominator
    )
    polarization = polarization.at[..., axial_axis_].set(
        1j / (safe_sine * safe_polarization_denominator)
    )
    vectors = jnp.stack(
        (outgoing, ingoing, polarization, jnp.conj(polarization)), axis=-2
    )
    black_hole_radial_margin = (radius - outer_horizon) / (
        1.0 + jnp.abs(radius) + jnp.abs(outer_horizon)
    )
    overextremal_radial_margin = radius / (1.0 + jnp.abs(radius))
    radial_margin = jnp.where(
        subextremal, black_hole_radial_margin, overextremal_radial_margin
    )
    sigma_margin = sigma / (1.0 + jnp.abs(sigma))
    axis_margin = jnp.abs(sine)
    domain_margin = jnp.minimum(radial_margin, jnp.minimum(sigma_margin, axis_margin))
    tetrad_id = _identifier(
        "kerr-principal-null-tetrad-v1",
        metric,
        axes,
        source_id,
        convention=convention_,
    )
    domain = _domain_evidence(
        metric,
        domain_margin,
        extra_valid=parameter_valid & denominator_valid & (domain_margin > 0.0),
        boundary_tolerance=jnp.sqrt(jnp.finfo(matrix.dtype).eps),
        domain_id=canonical_fingerprint(
            {"kind": "kerr-principal-null-domain-v1", "tetrad": tetrad_id}
        ),
    )
    vectors = jnp.where(domain.physically_valid[..., None, None], vectors, 0.0)
    matrix_complex = matrix.astype(complex_dtype)
    timelike_sign = metric.timelike_sign
    spatial_sign = -timelike_sign
    expected = jnp.asarray(
        (
            (0.0, timelike_sign, 0.0, 0.0),
            (timelike_sign, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, spatial_sign),
            (0.0, 0.0, spatial_sign, 0.0),
        ),
        dtype=complex_dtype,
    )
    gram = ein.contract("...ai,...ij,...bj->...ab", vectors, matrix_complex, vectors)
    inner_products = MetricInnerProductEvidence(
        gram,
        expected,
        domain_valid=domain.physically_valid,
        derivative_valid=domain.derivative_valid,
        tolerance=tolerance,
        evidence_id=canonical_fingerprint(
            {"kind": "principal-null-inner-products-v1", "tetrad": tetrad_id}
        ),
    )
    dual = ein.contract("ab,...bi,...ij->...aj", expected, vectors, matrix_complex)
    finite = (
        inner_products.finite
        & jnp.all(jnp.isfinite(vectors), axis=(-2, -1))
        & jnp.all(jnp.isfinite(dual), axis=(-2, -1))
    )
    physical = finite & domain.physically_valid
    qualified = physical & inner_products.qualified
    return PrincipalNullTetrad(
        vectors,
        dual,
        inner_products,
        finite,
        domain.physically_valid,
        physical,
        qualified,
        domain.derivative_valid & physical,
        convention=convention_,
        domain=domain,
        tetrad_id=tetrad_id,
    )


def tetrad_dual(
    tetrad: OrthonormalTetrad | PrincipalNullTetrad,
    /,
) -> Array:
    """Return dual covectors ``θᴬ_μ`` satisfying ``θᴬ_μ e_B^μ = δᴬ_B``."""
    if isinstance(tetrad, OrthonormalTetrad):
        return tetrad.dual_covectors
    if isinstance(tetrad, PrincipalNullTetrad):
        return tetrad.dual_covectors
    raise TypeError("tetrad must be OrthonormalTetrad or PrincipalNullTetrad.")


def _tetrad_vectors(
    tetrad: OrthonormalTetrad | PrincipalNullTetrad | ArrayLike,
    /,
) -> Array:
    if isinstance(tetrad, OrthonormalTetrad):
        return tetrad.vectors
    if isinstance(tetrad, PrincipalNullTetrad):
        return tetrad.vectors
    values = jnp.asarray(tetrad)
    if values.shape[-2:] != (4, 4):
        raise ValueError("Tetrad arrays must have trailing shape (4, 4).")
    return values


def tetrad_project_vector(
    tetrad: OrthonormalTetrad | PrincipalNullTetrad,
    vector: ArrayLike,
    /,
) -> Array:
    """Project a coordinate-basis vector to tetrad components."""
    values = jnp.asarray(vector)
    if values.shape[-1:] != (4,):
        raise ValueError("Projected vectors must have trailing dimension four.")
    return ein.contract("...ai,...i->...a", tetrad_dual(tetrad), values)


def tetrad_reconstruct_vector(
    tetrad: OrthonormalTetrad | PrincipalNullTetrad,
    components: ArrayLike,
    /,
) -> Array:
    """Reconstruct a coordinate-basis vector from tetrad components."""
    values = jnp.asarray(components)
    if values.shape[-1:] != (4,):
        raise ValueError("Tetrad vector components must have trailing dimension four.")
    return ein.contract("...a,...ai->...i", values, tetrad.vectors)


def tetrad_project_covector(
    tetrad: OrthonormalTetrad | PrincipalNullTetrad,
    covector: ArrayLike,
    /,
) -> Array:
    """Project a coordinate-basis covector to its values on the tetrad vectors."""
    values = jnp.asarray(covector)
    if values.shape[-1:] != (4,):
        raise ValueError("Projected covectors must have trailing dimension four.")
    return ein.contract("...ai,...i->...a", tetrad.vectors, values)


def tetrad_reconstruct_covector(
    tetrad: OrthonormalTetrad | PrincipalNullTetrad,
    components: ArrayLike,
    /,
) -> Array:
    """Reconstruct a coordinate-basis covector from tetrad components."""
    values = jnp.asarray(components)
    if values.shape[-1:] != (4,):
        raise ValueError("Tetrad covector components must have trailing dimension four.")
    return ein.contract("...a,...ai->...i", values, tetrad_dual(tetrad))


def tetrad_parallel_transport_residual(
    metric: LorentzianMetric,
    tetrad_field: Callable[[Array], OrthonormalTetrad | PrincipalNullTetrad | ArrayLike],
    coordinates: ArrayLike,
    direction: ArrayLike,
    /,
) -> Array:
    """Return ``u^ν ∇_ν e_A^μ`` for all vectors of a differentiable tetrad field."""
    _require_four_metric(metric)
    if not callable(tetrad_field):
        raise TypeError("tetrad_field must be callable.")
    points = _coordinates(coordinates, 4)
    directions = jnp.asarray(direction, dtype=points.dtype)
    if directions.shape[-1:] != (4,):
        raise ValueError(
            "Parallel-transport direction must have trailing dimension four."
        )
    leading_shape = jnp.broadcast_shapes(points.shape[:-1], directions.shape[:-1])
    points = jnp.broadcast_to(points, leading_shape + (4,))
    directions = jnp.broadcast_to(directions, leading_shape + (4,))
    connection = LeviCivitaConnection(metric)

    def point_residual(point: Array, tangent: Array, /) -> Array:
        frame = _tetrad_vectors(tetrad_field(point))
        if frame.shape != (4, 4):
            raise ValueError(
                "A pointwise tetrad field must return trailing shape (4, 4)."
            )
        derivative = jax.jacfwd(lambda value: _tetrad_vectors(tetrad_field(value)))(point)
        coefficients = connection.coefficients(point)
        covariant_derivative = derivative + ein.contract(
            "mnr,ar->amn", coefficients, frame
        )
        return ein.contract("n,amn->am", tangent, covariant_derivative)

    if not leading_shape:
        return point_residual(points, directions)
    flattened_points = points.reshape((-1, 4))
    flattened_directions = directions.reshape((-1, 4))
    residuals = jax.vmap(point_residual)(flattened_points, flattened_directions)
    return residuals.reshape(leading_shape + (4, 4))


def _tetrad_field_status(
    tetrad_field: Callable[[Array], OrthonormalTetrad | PrincipalNullTetrad | ArrayLike],
    coordinates: Array,
    /,
) -> tuple[Array, Array]:
    def point_status(point: Array, /) -> tuple[Array, Array]:
        evaluated = tetrad_field(point)
        if isinstance(evaluated, OrthonormalTetrad):
            return evaluated.domain_valid, evaluated.derivative_valid
        if isinstance(evaluated, PrincipalNullTetrad):
            return evaluated.domain_valid, evaluated.derivative_valid
        _tetrad_vectors(evaluated)
        return jnp.asarray(True), jnp.asarray(True)

    if coordinates.ndim == 1:
        return point_status(coordinates)
    leading_shape = coordinates.shape[:-1]
    flattened = coordinates.reshape((-1, 4))
    domain, derivative = jax.vmap(point_status)(flattened)
    return domain.reshape(leading_shape), derivative.reshape(leading_shape)


def tetrad_parallel_transport_evidence(
    metric: LorentzianMetric,
    tetrad_field: Callable[[Array], OrthonormalTetrad | PrincipalNullTetrad | ArrayLike],
    coordinates: ArrayLike,
    direction: ArrayLike,
    /,
    *,
    tolerance: float = 1e-8,
    source_id: str | None = None,
) -> TetradParallelTransportEvidence:
    """Evaluate and qualify pointwise tetrad parallel-transport residuals."""
    _require_four_metric(metric)
    points = _coordinates(coordinates, 4)
    residuals = tetrad_parallel_transport_residual(
        metric, tetrad_field, points, direction
    )
    domain, derivative = _tetrad_field_status(tetrad_field, points)
    evidence_id = _identifier(
        "tetrad-parallel-transport-evidence-v1", metric, (), source_id
    )
    return TetradParallelTransportEvidence(
        residuals,
        domain_valid=domain,
        derivative_valid=derivative,
        tolerance=tolerance,
        evidence_id=evidence_id,
    )


__all__ = [
    "KillingInnerProductEvidence",
    "MetricInnerProductEvidence",
    "OrthonormalTetrad",
    "PrincipalNullTetrad",
    "TetradParallelTransportEvidence",
    "axial_killing_vector",
    "kerr_principal_null_tetrad",
    "killing_equation_residual",
    "maximum_killing_equation_residual",
    "metric_inner_product_evidence",
    "orthonormal_tetrad",
    "stationary_axial_inner_product_evidence",
    "stationary_killing_vector",
    "tetrad_dual",
    "tetrad_parallel_transport_evidence",
    "tetrad_parallel_transport_residual",
    "tetrad_project_covector",
    "tetrad_project_vector",
    "tetrad_reconstruct_covector",
    "tetrad_reconstruct_vector",
    "zamo_observer_tetrad",
]
