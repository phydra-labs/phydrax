#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._chart import CoordinateChart
from ._lorentzian import _assemble_adm_matrix
from ._metric import _metric_inverse, LorentzianConvention, LorentzianMetric


class ADMDecomposition(StrictModule):
    """Lapse, contravariant shift, and spatial metric on one ADM foliation."""

    lapse: Array
    shift: Array
    spatial_metric: Array
    chart: CoordinateChart
    convention: LorentzianConvention = eqx.field(static=True)

    def __init__(
        self,
        lapse: ArrayLike,
        shift: ArrayLike,
        spatial_metric: ArrayLike,
        /,
        *,
        chart: CoordinateChart,
        convention: LorentzianConvention = "mostly_plus",
    ):
        if not isinstance(chart, CoordinateChart):
            raise TypeError("chart must be a CoordinateChart.")
        if chart.dimension < 2:
            raise ValueError("An ADM decomposition requires at least one spatial axis.")
        if convention not in ("mostly_plus", "mostly_minus"):
            raise ValueError("convention must be 'mostly_plus' or 'mostly_minus'.")
        lapse_array = jnp.asarray(lapse)
        shift_array = jnp.asarray(shift)
        spatial_array = jnp.asarray(spatial_metric)
        spatial_dimension = chart.dimension - 1
        leading_shape = lapse_array.shape
        expected_shift = leading_shape + (spatial_dimension,)
        expected_spatial = leading_shape + (spatial_dimension, spatial_dimension)
        if shift_array.shape != expected_shift:
            raise ValueError(
                f"ADM shift must have shape {expected_shift}; got {shift_array.shape}."
            )
        if spatial_array.shape != expected_spatial:
            raise ValueError(
                "ADM spatial metric must have shape "
                f"{expected_spatial}; got {spatial_array.shape}."
            )
        if not jnp.issubdtype(lapse_array.dtype, jnp.floating):
            raise TypeError("ADM lapse must have a real floating-point dtype.")
        if not jnp.issubdtype(shift_array.dtype, jnp.floating):
            raise TypeError("ADM shift must have a real floating-point dtype.")
        if not jnp.issubdtype(spatial_array.dtype, jnp.floating):
            raise TypeError("ADM spatial metric must have a real floating-point dtype.")
        self.lapse = lapse_array
        self.shift = shift_array
        self.spatial_metric = spatial_array
        self.chart = chart
        self.convention = convention

    @property
    def spatial_dimension(self) -> int:
        return self.chart.dimension - 1

    @property
    def shift_covector(self) -> Array:
        return oe.contract("...ij,...j->...i", self.spatial_metric, self.shift)

    @property
    def spatial_inverse(self) -> Array:
        return _metric_inverse(self.spatial_metric, positive_definite=True)

    def spacetime_metric(self) -> Array:
        """Reconstruct the spacetime metric in the declared sign convention."""
        return _assemble_adm_matrix(
            self.lapse,
            self.shift,
            self.spatial_metric,
            self.convention,
        )

    @property
    def spacetime_inverse(self) -> Array:
        """Return the analytic inverse spacetime metric of this decomposition."""
        inverse_lapse_squared = 1.0 / self.lapse**2
        time_time = -inverse_lapse_squared
        time_space = self.shift * inverse_lapse_squared[..., None]
        spatial = (
            self.spatial_inverse
            - oe.contract(
                "...i,...j->...ij",
                self.shift,
                self.shift,
            )
            * inverse_lapse_squared[..., None, None]
        )
        first_row = jnp.concatenate((time_time[..., None], time_space), axis=-1)
        remaining = jnp.concatenate((time_space[..., :, None], spatial), axis=-1)
        inverse = jnp.concatenate((first_row[..., None, :], remaining), axis=-2)
        return inverse if self.convention == "mostly_plus" else -inverse


def decompose_adm_metric(
    metric: LorentzianMetric,
    coordinates: ArrayLike,
    /,
) -> ADMDecomposition:
    """Recover ADM fields from a Lorentzian metric in a time-first chart."""
    if not isinstance(metric, LorentzianMetric):
        raise TypeError("decompose_adm_metric requires a LorentzianMetric.")
    matrix = metric(coordinates)
    normalized = matrix if metric.convention == "mostly_plus" else -matrix
    spatial = normalized[..., 1:, 1:]
    shift_covector = normalized[..., 0, 1:]
    shift = jnp.linalg.solve(spatial, shift_covector[..., None])[..., 0]
    lapse_squared = (
        oe.contract("...i,...i->...", shift, shift_covector) - normalized[..., 0, 0]
    )
    asymmetry = jnp.max(
        jnp.abs(spatial - jnp.swapaxes(spatial, -1, -2)),
        axis=(-2, -1),
    )
    scale = jnp.maximum(jnp.max(jnp.abs(spatial), axis=(-2, -1)), 1.0)
    tolerance = 64.0 * jnp.finfo(spatial.dtype).eps * scale
    minimum_eigenvalue = jnp.min(jnp.linalg.eigvalsh(spatial), axis=-1)
    spatial = eqx.error_if(
        spatial,
        jnp.any(~jnp.isfinite(spatial))
        | jnp.any(asymmetry > tolerance)
        | jnp.any(minimum_eigenvalue <= 0.0),
        "ADM decomposition requires a finite symmetric positive-definite spatial block.",
    )
    shift = eqx.error_if(
        shift,
        jnp.any(~jnp.isfinite(shift)),
        "ADM decomposition produced a nonfinite shift.",
    )
    lapse_squared = eqx.error_if(
        lapse_squared,
        jnp.any(~jnp.isfinite(lapse_squared)) | jnp.any(lapse_squared <= 0.0),
        "ADM decomposition requires a finite positive squared lapse.",
    )
    return ADMDecomposition(
        jnp.sqrt(lapse_squared),
        shift,
        spatial,
        chart=metric.chart,
        convention=metric.convention,
    )


class ADMValidationReport(StrictModule):
    """Aggregated algebraic diagnostics for an evaluated ADM decomposition."""

    valid: Array
    finite: Array
    lapse_positive: Array
    spatial_symmetric: Array
    signature_matches: Array
    minimum_lapse: Array
    minimum_spatial_eigenvalue: Array
    maximum_spatial_asymmetry: Array
    maximum_inverse_residual: Array
    maximum_reconstruction_residual: Array

    def __init__(
        self,
        *,
        valid: Array,
        finite: Array,
        lapse_positive: Array,
        spatial_symmetric: Array,
        signature_matches: Array,
        minimum_lapse: Array,
        minimum_spatial_eigenvalue: Array,
        maximum_spatial_asymmetry: Array,
        maximum_inverse_residual: Array,
        maximum_reconstruction_residual: Array,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.lapse_positive = jnp.asarray(lapse_positive, dtype=bool)
        self.spatial_symmetric = jnp.asarray(spatial_symmetric, dtype=bool)
        self.signature_matches = jnp.asarray(signature_matches, dtype=bool)
        self.minimum_lapse = jnp.asarray(minimum_lapse)
        self.minimum_spatial_eigenvalue = jnp.asarray(minimum_spatial_eigenvalue)
        self.maximum_spatial_asymmetry = jnp.asarray(maximum_spatial_asymmetry)
        self.maximum_inverse_residual = jnp.asarray(maximum_inverse_residual)
        self.maximum_reconstruction_residual = jnp.asarray(
            maximum_reconstruction_residual
        )


def validate_adm_decomposition(
    decomposition: ADMDecomposition,
    /,
    *,
    reference_metric: ArrayLike | None = None,
    symmetry_tolerance: float = 1e-10,
    eigenvalue_tolerance: float = 1e-10,
    inverse_tolerance: float = 1e-9,
    reconstruction_tolerance: float = 1e-9,
    raise_on_error: bool = False,
) -> ADMValidationReport:
    """Validate ADM positivity, signature, inversion, and optional reconstruction."""
    if not isinstance(decomposition, ADMDecomposition):
        raise TypeError("decomposition must be an ADMDecomposition.")
    tolerances = (
        float(symmetry_tolerance),
        float(eigenvalue_tolerance),
        float(inverse_tolerance),
        float(reconstruction_tolerance),
    )
    if any(not isfinite(value) or value < 0.0 for value in tolerances):
        raise ValueError("ADM validation tolerances must be finite and non-negative.")
    lapse = decomposition.lapse
    shift = decomposition.shift
    spatial = decomposition.spatial_metric
    matrix = decomposition.spacetime_metric()
    inverse = decomposition.spacetime_inverse
    finite = (
        jnp.all(jnp.isfinite(lapse))
        & jnp.all(jnp.isfinite(shift))
        & jnp.all(jnp.isfinite(spatial))
    )
    minimum_lapse = jnp.min(lapse)
    lapse_positive = minimum_lapse > 0.0
    maximum_asymmetry = jnp.max(jnp.abs(spatial - jnp.swapaxes(spatial, -1, -2)))
    spatial_symmetric = maximum_asymmetry <= tolerances[0]
    spatial_eigenvalues = jnp.linalg.eigvalsh(
        0.5 * (spatial + jnp.swapaxes(spatial, -1, -2))
    )
    minimum_spatial_eigenvalue = jnp.min(spatial_eigenvalues)
    spatial_positive = minimum_spatial_eigenvalue > tolerances[1]
    spacetime_eigenvalues = jnp.linalg.eigvalsh(
        0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))
    )
    eigenvalue_scale = jnp.maximum(
        jnp.max(jnp.abs(spacetime_eigenvalues), axis=-1, keepdims=True),
        1.0,
    )
    signature_threshold = tolerances[1] * eigenvalue_scale
    positive_count = jnp.sum(
        spacetime_eigenvalues > signature_threshold,
        axis=-1,
    )
    negative_count = jnp.sum(
        spacetime_eigenvalues < -signature_threshold,
        axis=-1,
    )
    if decomposition.convention == "mostly_plus":
        expected_positive = decomposition.spatial_dimension
        expected_negative = 1
    else:
        expected_positive = 1
        expected_negative = decomposition.spatial_dimension
    signature_matches = jnp.all(
        (positive_count == expected_positive) & (negative_count == expected_negative)
    )
    identity = jnp.broadcast_to(
        jnp.eye(decomposition.chart.dimension, dtype=matrix.dtype),
        matrix.shape,
    )
    maximum_inverse_residual = jnp.max(jnp.abs(matrix @ inverse - identity))
    inverse_valid = maximum_inverse_residual <= tolerances[2]
    if reference_metric is None:
        maximum_reconstruction_residual = jnp.asarray(0.0, dtype=matrix.dtype)
    else:
        reference = jnp.asarray(reference_metric)
        if reference.shape != matrix.shape:
            raise ValueError(
                f"reference_metric must have shape {matrix.shape}; got {reference.shape}."
            )
        maximum_reconstruction_residual = jnp.max(jnp.abs(matrix - reference))
    reconstruction_valid = maximum_reconstruction_residual <= tolerances[3]
    valid = (
        finite
        & lapse_positive
        & spatial_symmetric
        & spatial_positive
        & signature_matches
        & inverse_valid
        & reconstruction_valid
    )
    report = ADMValidationReport(
        valid=valid,
        finite=finite,
        lapse_positive=lapse_positive,
        spatial_symmetric=spatial_symmetric,
        signature_matches=signature_matches,
        minimum_lapse=minimum_lapse,
        minimum_spatial_eigenvalue=minimum_spatial_eigenvalue,
        maximum_spatial_asymmetry=maximum_asymmetry,
        maximum_inverse_residual=maximum_inverse_residual,
        maximum_reconstruction_residual=maximum_reconstruction_residual,
    )
    if raise_on_error and not bool(jax.device_get(valid)):
        raise ValueError(
            "ADM validation failed: "
            f"finite={bool(jax.device_get(finite))}, "
            f"lapse_positive={bool(jax.device_get(lapse_positive))}, "
            f"spatial_symmetric={bool(jax.device_get(spatial_symmetric))}, "
            f"signature_matches={bool(jax.device_get(signature_matches))}, "
            "minimum_spatial_eigenvalue="
            f"{float(jax.device_get(minimum_spatial_eigenvalue))}, "
            "maximum_inverse_residual="
            f"{float(jax.device_get(maximum_inverse_residual))}, "
            "maximum_reconstruction_residual="
            f"{float(jax.device_get(maximum_reconstruction_residual))}."
        )
    return report


class ADMParameterization(StrictModule):
    """Signature-safe trainable ADM fields from unconstrained raw callables."""

    raw_lapse: Callable[[Array], Array]
    shift_function: Callable[[Array], Array]
    raw_spatial_factor: Callable[[Array], Array]
    chart: CoordinateChart
    minimum_lapse: float = eqx.field(static=True)
    minimum_spatial_diagonal: float = eqx.field(static=True)
    convention: LorentzianConvention = eqx.field(static=True)

    def __init__(
        self,
        raw_lapse: Callable[[Array], Array],
        shift: Callable[[Array], Array],
        raw_spatial_factor: Callable[[Array], Array],
        /,
        *,
        chart: CoordinateChart,
        minimum_lapse: float = 1e-6,
        minimum_spatial_diagonal: float = 1e-6,
        convention: LorentzianConvention = "mostly_plus",
    ):
        if not callable(raw_lapse):
            raise TypeError("raw_lapse must be callable.")
        if not callable(shift):
            raise TypeError("shift must be callable.")
        if not callable(raw_spatial_factor):
            raise TypeError("raw_spatial_factor must be callable.")
        if not isinstance(chart, CoordinateChart):
            raise TypeError("chart must be a CoordinateChart.")
        if chart.dimension < 2:
            raise ValueError("ADMParameterization requires at least one spatial axis.")
        lapse_floor = float(minimum_lapse)
        diagonal_floor = float(minimum_spatial_diagonal)
        if not isfinite(lapse_floor) or lapse_floor <= 0.0:
            raise ValueError("minimum_lapse must be finite and positive.")
        if not isfinite(diagonal_floor) or diagonal_floor <= 0.0:
            raise ValueError("minimum_spatial_diagonal must be finite and positive.")
        if convention not in ("mostly_plus", "mostly_minus"):
            raise ValueError("convention must be 'mostly_plus' or 'mostly_minus'.")
        self.raw_lapse = raw_lapse
        self.shift_function = shift
        self.raw_spatial_factor = raw_spatial_factor
        self.chart = chart
        self.minimum_lapse = lapse_floor
        self.minimum_spatial_diagonal = diagonal_floor
        self.convention = convention

    def _fields_point(self, coordinates: Array, /) -> tuple[Array, Array, Array]:
        raw_lapse = jnp.asarray(self.raw_lapse(coordinates))
        shift = jnp.asarray(self.shift_function(coordinates))
        raw_factor = jnp.asarray(self.raw_spatial_factor(coordinates))
        spatial_dimension = self.chart.dimension - 1
        if raw_lapse.shape != ():
            raise ValueError("raw_lapse must return one scalar.")
        if shift.shape != (spatial_dimension,):
            raise ValueError(
                f"shift must return shape {(spatial_dimension,)}; got {shift.shape}."
            )
        expected_factor = (spatial_dimension, spatial_dimension)
        if raw_factor.shape != expected_factor:
            raise ValueError(
                "raw_spatial_factor must return shape "
                f"{expected_factor}; got {raw_factor.shape}."
            )
        if not jnp.issubdtype(raw_lapse.dtype, jnp.floating):
            raise TypeError("raw_lapse must return a real floating-point scalar.")
        if not jnp.issubdtype(shift.dtype, jnp.floating):
            raise TypeError("shift must return a real floating-point vector.")
        if not jnp.issubdtype(raw_factor.dtype, jnp.floating):
            raise TypeError(
                "raw_spatial_factor must return a real floating-point matrix."
            )
        raw_lapse = eqx.error_if(
            raw_lapse,
            ~jnp.isfinite(raw_lapse),
            "raw_lapse must be finite.",
        )
        shift = eqx.error_if(
            shift,
            jnp.any(~jnp.isfinite(shift)),
            "shift must be finite.",
        )
        raw_factor = eqx.error_if(
            raw_factor,
            jnp.any(~jnp.isfinite(raw_factor)),
            "raw_spatial_factor must be finite.",
        )
        lapse = jax.nn.softplus(raw_lapse) + jnp.asarray(
            self.minimum_lapse,
            dtype=raw_lapse.dtype,
        )
        diagonal = jax.nn.softplus(jnp.diagonal(raw_factor)) + jnp.asarray(
            self.minimum_spatial_diagonal,
            dtype=raw_factor.dtype,
        )
        factor = jnp.tril(raw_factor, k=-1) + jnp.diag(diagonal)
        spatial = factor @ jnp.swapaxes(factor, -1, -2)
        return lapse, shift, spatial

    def decompose(self, coordinates: ArrayLike, /) -> ADMDecomposition:
        """Evaluate safe ADM fields at one point or a leading batch of points."""
        points = jnp.asarray(coordinates)
        if points.ndim < 1 or points.shape[-1] != self.chart.dimension:
            raise ValueError(
                "ADM coordinates must have trailing dimension "
                f"{self.chart.dimension}; got {points.shape}."
            )
        leading_shape = points.shape[:-1]
        if not leading_shape:
            lapse, shift, spatial = self._fields_point(points)
        else:
            flattened = points.reshape((-1, self.chart.dimension))
            lapse, shift, spatial = jax.vmap(self._fields_point)(flattened)
            lapse = lapse.reshape(leading_shape)
            shift = shift.reshape(leading_shape + (self.chart.dimension - 1,))
            spatial = spatial.reshape(
                leading_shape + (self.chart.dimension - 1, self.chart.dimension - 1)
            )
        return ADMDecomposition(
            lapse,
            shift,
            spatial,
            chart=self.chart,
            convention=self.convention,
        )

    def __call__(self, coordinates: ArrayLike, /) -> ADMDecomposition:
        return self.decompose(coordinates)

    def metric(self) -> LorentzianMetric:
        """Construct the Lorentzian metric induced by these safe fields."""
        return LorentzianMetric(
            _ParameterizedADMMetricMap(self),
            chart=self.chart,
            convention=self.convention,
        )


class _ParameterizedADMMetricMap(StrictModule):
    parameterization: ADMParameterization

    def __init__(self, parameterization: ADMParameterization, /):
        self.parameterization = parameterization

    def __call__(self, coordinates: Array, /) -> Array:
        lapse, shift, spatial = self.parameterization._fields_point(coordinates)
        return _assemble_adm_matrix(
            lapse,
            shift,
            spatial,
            self.parameterization.convention,
        )


def parameterized_adm_metric(
    raw_lapse: Callable[[Array], Array],
    shift: Callable[[Array], Array],
    raw_spatial_factor: Callable[[Array], Array],
    /,
    *,
    chart: CoordinateChart,
    minimum_lapse: float = 1e-6,
    minimum_spatial_diagonal: float = 1e-6,
    convention: LorentzianConvention = "mostly_plus",
) -> LorentzianMetric:
    """Construct a Lorentzian ADM metric from unconstrained trainable fields."""
    return ADMParameterization(
        raw_lapse,
        shift,
        raw_spatial_factor,
        chart=chart,
        minimum_lapse=minimum_lapse,
        minimum_spatial_diagonal=minimum_spatial_diagonal,
        convention=convention,
    ).metric()


__all__ = [
    "ADMDecomposition",
    "ADMParameterization",
    "ADMValidationReport",
    "decompose_adm_metric",
    "parameterized_adm_metric",
    "validate_adm_decomposition",
]
