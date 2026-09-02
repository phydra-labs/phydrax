#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import pi

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._strict import StrictModule
from ._adm import decompose_adm_metric
from ._chart import CoordinateChart
from ._connection import LeviCivitaConnection
from ._curvature import scalar_curvature
from ._metric import LorentzianMetric, RiemannianMetric
from ._utils import _coordinates, _pointwise_array


class ADMConstraintResiduals(StrictModule):
    """Hamiltonian scalar and contravariant momentum constraint residuals."""

    hamiltonian: Array
    momentum: Array

    def __init__(self, hamiltonian: ArrayLike, momentum: ArrayLike, /):
        self.hamiltonian = jnp.asarray(hamiltonian)
        self.momentum = jnp.asarray(momentum)

    @property
    def maximum_absolute(self) -> Array:
        """Return the largest absolute residual over every point and component."""
        return jnp.maximum(
            jnp.max(jnp.abs(self.hamiltonian)),
            jnp.max(jnp.abs(self.momentum)),
        )


class _SpatialSliceMetricMap(StrictModule):
    spacetime_metric: LorentzianMetric
    time: Array

    def __init__(
        self,
        spacetime_metric: LorentzianMetric,
        time: Array,
        /,
    ):
        self.spacetime_metric = spacetime_metric
        self.time = jnp.asarray(time)

    def __call__(self, spatial_coordinates: Array, /) -> Array:
        coordinates = jnp.concatenate((self.time[None], spatial_coordinates))
        return decompose_adm_metric(
            self.spacetime_metric,
            coordinates,
        ).spatial_metric


def _require_adm_metric(metric: LorentzianMetric, /) -> int:
    if not isinstance(metric, LorentzianMetric):
        raise TypeError("ADM hypersurface geometry requires a LorentzianMetric.")
    spatial_dimension = metric.chart.dimension - 1
    if spatial_dimension < 1:
        raise ValueError("ADM hypersurface geometry requires spacetime dimension >= 2.")
    return spatial_dimension


def _spatial_slice_metric(
    metric: LorentzianMetric,
    coordinates: Array,
    /,
) -> RiemannianMetric:
    spatial_chart = CoordinateChart(
        f"{metric.chart.name}_spatial_slice",
        metric.chart.coordinates[1:],
    )
    return RiemannianMetric(
        _SpatialSliceMetricMap(metric, coordinates[0]),
        chart=spatial_chart,
    )


def _normal_vector_at(metric: LorentzianMetric, coordinates: Array, /) -> Array:
    decomposition = decompose_adm_metric(metric, coordinates)
    return jnp.concatenate(
        (
            (1.0 / decomposition.lapse)[None],
            -decomposition.shift / decomposition.lapse,
        )
    )


def _normal_covector_at(metric: LorentzianMetric, coordinates: Array, /) -> Array:
    decomposition = decompose_adm_metric(metric, coordinates)
    timelike_sign = -1.0 if metric.convention == "mostly_plus" else 1.0
    return jnp.concatenate(
        (
            (timelike_sign * decomposition.lapse)[None],
            jnp.zeros_like(decomposition.shift),
        )
    )


def adm_normal_vector(
    metric: LorentzianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return the future-directed contravariant ADM unit normal."""
    dimension = _require_adm_metric(metric) + 1
    return _pointwise_array(
        lambda point: _normal_vector_at(metric, point),
        coordinates,
        dimension,
    )


def adm_normal_covector(
    metric: LorentzianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return the metric dual of the future-directed ADM unit normal."""
    dimension = _require_adm_metric(metric) + 1
    return _pointwise_array(
        lambda point: _normal_covector_at(metric, point),
        coordinates,
        dimension,
    )


def adm_spacetime_projector(
    metric: LorentzianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return the mixed spatial projector ``h^mu_nu`` of the ADM foliation."""
    dimension = _require_adm_metric(metric) + 1
    points = _coordinates(coordinates, dimension)
    normal = adm_normal_vector(metric, points)
    conormal = adm_normal_covector(metric, points)
    timelike_sign = -1.0 if metric.convention == "mostly_plus" else 1.0
    identity = jnp.eye(dimension, dtype=normal.dtype)
    return identity - timelike_sign * ein.contract(
        "...i,...j->...ij",
        normal,
        conormal,
    )


def _extrinsic_curvature_at(
    metric: LorentzianMetric,
    coordinates: Array,
    /,
) -> Array:
    decomposition = decompose_adm_metric(metric, coordinates)
    spatial_metric = _spatial_slice_metric(metric, coordinates)
    spatial_coordinates = coordinates[1:]
    connection = LeviCivitaConnection(spatial_metric).coefficients(spatial_coordinates)

    def shift_covector(point: Array, /) -> Array:
        fields = decompose_adm_metric(metric, point)
        return ein.contract("ij,j->i", fields.spatial_metric, fields.shift)

    def spatial_matrix(point: Array, /) -> Array:
        return decompose_adm_metric(metric, point).spatial_metric

    shift_covector_value = shift_covector(coordinates)
    shift_derivative = jax.jacfwd(shift_covector)(coordinates)
    spatial_derivative = jax.jacfwd(spatial_matrix)(coordinates)
    partial_shift = jnp.swapaxes(shift_derivative[:, 1:], -2, -1)
    covariant_shift = partial_shift - ein.contract(
        "kij,k->ij",
        connection,
        shift_covector_value,
    )
    return (
        covariant_shift
        + jnp.swapaxes(covariant_shift, -2, -1)
        - spatial_derivative[..., 0]
    ) / (2.0 * decomposition.lapse)


def adm_extrinsic_curvature(
    metric: LorentzianMetric,
    coordinates: ArrayLike,
    /,
) -> Array:
    r"""Return ``K_ij = -1/2 L_n gamma_ij`` on each ADM spatial slice."""
    dimension = _require_adm_metric(metric) + 1
    return _pointwise_array(
        lambda point: _extrinsic_curvature_at(metric, point),
        coordinates,
        dimension,
    )


def _hamiltonian_geometry_at(
    metric: LorentzianMetric,
    coordinates: Array,
    /,
) -> Array:
    decomposition = decompose_adm_metric(metric, coordinates)
    spatial_metric = _spatial_slice_metric(metric, coordinates)
    spatial_coordinates = coordinates[1:]
    extrinsic = _extrinsic_curvature_at(metric, coordinates)
    spatial_inverse = decomposition.spatial_inverse
    trace = ein.contract("ij,ij->", spatial_inverse, extrinsic)
    raised = ein.contract(
        "ik,jl,kl->ij",
        spatial_inverse,
        spatial_inverse,
        extrinsic,
    )
    extrinsic_square = ein.contract("ij,ij->", extrinsic, raised)
    return (
        scalar_curvature(spatial_metric, spatial_coordinates)
        + trace**2
        - extrinsic_square
    )


def _momentum_geometry_at(
    metric: LorentzianMetric,
    coordinates: Array,
    /,
) -> Array:
    spatial_metric = _spatial_slice_metric(metric, coordinates)
    spatial_coordinates = coordinates[1:]
    connection = LeviCivitaConnection(spatial_metric).coefficients(spatial_coordinates)
    time = coordinates[0]

    def trace_reversed(spatial_point: Array, /) -> Array:
        point = jnp.concatenate((time[None], spatial_point))
        decomposition = decompose_adm_metric(metric, point)
        extrinsic = _extrinsic_curvature_at(metric, point)
        inverse = decomposition.spatial_inverse
        trace = ein.contract("ij,ij->", inverse, extrinsic)
        raised = ein.contract("ik,jl,kl->ij", inverse, inverse, extrinsic)
        return raised - trace * inverse

    tensor = trace_reversed(spatial_coordinates)
    derivative = jax.jacfwd(trace_reversed)(spatial_coordinates)
    partial = ein.contract("ijj->i", derivative)
    first_connection = ein.contract("ijk,kj->i", connection, tensor)
    second_connection = ein.contract("jjk,ik->i", connection, tensor)
    return partial + first_connection + second_connection


def _coupling(value: ArrayLike, /) -> Array:
    coupling = jnp.asarray(value)
    if coupling.shape != ():
        raise ValueError("einstein_coupling must be scalar.")
    return eqx.error_if(
        coupling,
        ~jnp.isfinite(coupling) | (coupling < 0.0),
        "einstein_coupling must be finite and non-negative.",
    )


def _scalar_source(value: ArrayLike, leading_shape: tuple[int, ...], /) -> Array:
    source = jnp.asarray(value)
    if source.shape == ():
        return jnp.broadcast_to(source, leading_shape)
    if source.shape != leading_shape:
        raise ValueError(
            f"energy_density must have shape {leading_shape}; got {source.shape}."
        )
    return source


def _momentum_source(
    value: ArrayLike | None,
    leading_shape: tuple[int, ...],
    spatial_dimension: int,
    dtype,
    /,
) -> Array:
    expected = leading_shape + (spatial_dimension,)
    if value is None:
        return jnp.zeros(expected, dtype=dtype)
    source = jnp.asarray(value)
    if source.shape == (spatial_dimension,):
        return jnp.broadcast_to(source, expected)
    if source.shape != expected:
        raise ValueError(
            f"momentum_density must have shape {expected}; got {source.shape}."
        )
    return source


def adm_hamiltonian_constraint(
    metric: LorentzianMetric,
    coordinates: ArrayLike,
    /,
    *,
    energy_density: ArrayLike = 0.0,
    einstein_coupling: ArrayLike = 8.0 * pi,
) -> Array:
    """Return ``R[gamma] + K² - K_ij K^ij - 2 kappa rho``."""
    dimension = _require_adm_metric(metric) + 1
    points = _coordinates(coordinates, dimension)
    geometry = _pointwise_array(
        lambda point: _hamiltonian_geometry_at(metric, point),
        points,
        dimension,
    )
    source = _scalar_source(energy_density, points.shape[:-1])
    return geometry - 2.0 * _coupling(einstein_coupling) * source


def adm_momentum_constraint(
    metric: LorentzianMetric,
    coordinates: ArrayLike,
    /,
    *,
    momentum_density: ArrayLike | None = None,
    einstein_coupling: ArrayLike = 8.0 * pi,
) -> Array:
    """Return ``D_j(K^ij - gamma^ij K) - kappa S^i``."""
    spatial_dimension = _require_adm_metric(metric)
    dimension = spatial_dimension + 1
    points = _coordinates(coordinates, dimension)
    geometry = _pointwise_array(
        lambda point: _momentum_geometry_at(metric, point),
        points,
        dimension,
    )
    source = _momentum_source(
        momentum_density,
        points.shape[:-1],
        spatial_dimension,
        geometry.dtype,
    )
    return geometry - _coupling(einstein_coupling) * source


def adm_constraint_residuals(
    metric: LorentzianMetric,
    coordinates: ArrayLike,
    /,
    *,
    energy_density: ArrayLike = 0.0,
    momentum_density: ArrayLike | None = None,
    einstein_coupling: ArrayLike = 8.0 * pi,
) -> ADMConstraintResiduals:
    """Evaluate Hamiltonian and momentum constraints with shared source conventions."""
    return ADMConstraintResiduals(
        adm_hamiltonian_constraint(
            metric,
            coordinates,
            energy_density=energy_density,
            einstein_coupling=einstein_coupling,
        ),
        adm_momentum_constraint(
            metric,
            coordinates,
            momentum_density=momentum_density,
            einstein_coupling=einstein_coupling,
        ),
    )


__all__ = [
    "ADMConstraintResiduals",
    "adm_constraint_residuals",
    "adm_extrinsic_curvature",
    "adm_hamiltonian_constraint",
    "adm_momentum_constraint",
    "adm_normal_covector",
    "adm_normal_vector",
    "adm_spacetime_projector",
]
