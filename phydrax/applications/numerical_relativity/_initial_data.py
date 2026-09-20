#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite, pi

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    determinant_small_linear,
    inverse_small_linear,
    SmallLinearSolvePlan,
)
from ...metrix._adm_exchange import ADMGridGeometry
from ._status import NumericalRelativityStatus, ScientificStatus


_SMALL_3 = SmallLinearSolvePlan(3, refinement_iterations=1)


class ADMInitialData(StrictModule):
    """Evaluated vacuum ADM fields with pointwise domain evidence.

    ``domain_valid`` is deliberately separate from finiteness.  Analytic puncture
    formulae use finite guarded values at excluded points so a caller can inspect
    the failure without allowing those values into an evolution.
    """

    coordinates: Array
    lapse: Array
    shift: Array
    spatial_metric: Array
    extrinsic_curvature: Array
    conformal_factor: Array
    spatial_determinant: Array
    domain_valid: Array
    status: ScientificStatus
    data_id: str = eqx.field(static=True)

    @property
    def valid(self) -> Array:
        return self.status.finite & self.status.physically_valid

    def as_grid_geometry(
        self,
        /,
        *,
        chart_id: str,
        convention_id: str,
        scale_id: str,
        topology_id: str,
    ) -> ADMGridGeometry:
        """Bind these fields to the shared immutable ADM exchange contract."""

        inverse = inverse_small_linear(_SMALL_3, self.spatial_metric)
        lane_finite = (
            jnp.isfinite(self.lapse)
            & jnp.all(jnp.isfinite(self.shift), axis=-1)
            & jnp.all(jnp.isfinite(self.spatial_metric), axis=(-2, -1))
            & jnp.all(jnp.isfinite(self.extrinsic_curvature), axis=(-2, -1))
        )
        lane_valid = (
            self.domain_valid
            & lane_finite
            & inverse.successful
            & (self.lapse > 0.0)
            & (self.spatial_determinant > 0.0)
        )
        return ADMGridGeometry(
            self.lapse,
            self.shift,
            self.spatial_metric,
            inverse.value,
            jnp.sqrt(self.spatial_determinant),
            self.extrinsic_curvature,
            self.domain_valid,
            lane_valid,
            snapshot_token=0,
            chart_id=chart_id,
            convention_id=convention_id,
            scale_id=scale_id,
            topology_id=topology_id,
            geometry_lineage_id=canonical_fingerprint(
                {
                    "kind": "initial-data-adm-grid-geometry-lineage",
                    "data": self.data_id,
                    "chart": chart_id,
                    "convention": convention_id,
                    "scale": scale_id,
                    "topology": topology_id,
                }
            ),
        )


class ADMConstraintDiagnostics(StrictModule):
    """Vacuum Hamiltonian and momentum residuals at requested points."""

    hamiltonian: Array
    momentum: Array
    hamiltonian_linf: Array
    momentum_linf: Array
    finite: Array
    domain_valid: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array

    @property
    def maximum_absolute(self) -> Array:
        return jnp.maximum(self.hamiltonian_linf, self.momentum_linf)


class ADMChargeDiagnostics(StrictModule):
    """Finite-surface ADM mass, momentum, and angular-momentum evidence."""

    mass: Array
    linear_momentum: Array
    angular_momentum: Array
    mass_integrand: Array
    momentum_integrand: Array
    normal_normalization_defect: Array
    surface_closure_defect: Array
    finite: Array
    domain_valid: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    surface_id: str = eqx.field(static=True)


class MinkowskiInitialData(StrictModule, NonTrainableState):
    """Cartesian Minkowski slice."""

    data_id: str = eqx.field(static=True)

    def __init__(self):
        self.data_id = canonical_fingerprint({"kind": "minkowski-adm-initial-data"})

    def __call__(self, coordinates: ArrayLike, /) -> ADMInitialData:
        points = _coordinates(coordinates)
        leading = points.shape[:-1]
        dtype = points.dtype
        lapse = jnp.ones(leading, dtype=dtype)
        shift = jnp.zeros(leading + (3,), dtype=dtype)
        metric = jnp.broadcast_to(jnp.eye(3, dtype=dtype), leading + (3, 3))
        extrinsic = jnp.zeros_like(metric)
        return _package_initial_data(
            points,
            lapse,
            shift,
            metric,
            extrinsic,
            jnp.ones(leading, dtype=dtype),
            jnp.ones(leading, dtype=jnp.bool_),
            self.data_id,
        )


class IsotropicSchwarzschildInitialData(StrictModule, NonTrainableState):
    """Exterior time-symmetric Schwarzschild slice in isotropic coordinates."""

    mass: Array
    center: Array
    excision_radius: float = eqx.field(static=True)
    data_id: str = eqx.field(static=True)

    def __init__(
        self,
        mass: ArrayLike,
        /,
        *,
        center: ArrayLike = (0.0, 0.0, 0.0),
        excision_radius: float = 0.0,
    ):
        mass_host = float(np.asarray(mass))
        center_host = np.asarray(center, dtype=np.float64)
        excision = float(excision_radius)
        if not isfinite(mass_host) or mass_host <= 0.0:
            raise ValueError("Schwarzschild mass must be finite and positive.")
        if center_host.shape != (3,) or not np.all(np.isfinite(center_host)):
            raise ValueError("Schwarzschild center must be one finite three-vector.")
        if not isfinite(excision) or excision < 0.0:
            raise ValueError("excision_radius must be finite and non-negative.")
        self.mass = jnp.asarray(mass_host)
        self.center = jnp.asarray(center_host)
        self.excision_radius = excision
        self.data_id = canonical_fingerprint(
            {
                "kind": "isotropic-schwarzschild-adm-initial-data",
                "mass": mass_host,
                "center": center_host.tolist(),
                "excision_radius": excision,
            }
        )

    def __call__(self, coordinates: ArrayLike, /) -> ADMInitialData:
        points = _coordinates(coordinates)
        offset = points - self.center.astype(points.dtype)
        radius = jnp.sqrt(ein.contract("...i,...i->...", offset, offset))
        horizon = 0.5 * self.mass.astype(points.dtype)
        lower = jnp.maximum(
            jnp.asarray(self.excision_radius, dtype=points.dtype), horizon
        )
        domain = radius > lower
        safe_radius = jnp.where(domain, radius, jnp.asarray(1.0, dtype=points.dtype))
        ratio = horizon / safe_radius
        conformal = 1.0 + ratio
        lapse = (1.0 - ratio) / conformal
        leading = points.shape[:-1]
        identity = jnp.broadcast_to(jnp.eye(3, dtype=points.dtype), leading + (3, 3))
        metric = conformal[..., None, None] ** 4 * identity
        return _package_initial_data(
            points,
            lapse,
            jnp.zeros(leading + (3,), dtype=points.dtype),
            metric,
            jnp.zeros_like(metric),
            conformal,
            domain,
            self.data_id,
        )


class KerrSchildInitialData(StrictModule, NonTrainableState):
    """Stationary Cartesian Kerr slice in horizon-penetrating Kerr--Schild form.

    ``spin`` is the specific angular-momentum vector ``a = J / M``.  A scalar is
    accepted as the z component.  The subextremal and extremal domain ``|a| <= M``
    is represented exactly; the ring singularity remains excluded pointwise.
    """

    mass: Array
    spin: Array
    center: Array
    excision_radius: float = eqx.field(static=True)
    data_id: str = eqx.field(static=True)

    def __init__(
        self,
        mass: ArrayLike,
        spin: ArrayLike = 0.0,
        /,
        *,
        center: ArrayLike = (0.0, 0.0, 0.0),
        excision_radius: float = 0.0,
    ):
        mass_host = float(np.asarray(mass))
        spin_host = np.asarray(spin, dtype=np.float64)
        if spin_host.shape == ():
            spin_host = np.asarray((0.0, 0.0, float(spin_host)))
        center_host = np.asarray(center, dtype=np.float64)
        excision = float(excision_radius)
        if not isfinite(mass_host) or mass_host <= 0.0:
            raise ValueError("Kerr mass must be finite and positive.")
        if (
            spin_host.shape != (3,)
            or not np.all(np.isfinite(spin_host))
            or np.linalg.norm(spin_host) > mass_host
        ):
            raise ValueError("Kerr specific spin must be finite with |a| <= mass.")
        if center_host.shape != (3,) or not np.all(np.isfinite(center_host)):
            raise ValueError("Kerr center must be one finite three-vector.")
        if not isfinite(excision) or excision < 0.0:
            raise ValueError("excision_radius must be finite and non-negative.")
        self.mass = jnp.asarray(mass_host)
        self.spin = jnp.asarray(spin_host)
        self.center = jnp.asarray(center_host)
        self.excision_radius = excision
        self.data_id = canonical_fingerprint(
            {
                "kind": "kerr-schild-adm-initial-data",
                "mass": mass_host,
                "specific_spin": spin_host.tolist(),
                "center": center_host.tolist(),
                "excision_radius": excision,
            }
        )

    def __call__(self, coordinates: ArrayLike, /) -> ADMInitialData:
        points = _coordinates(coordinates)
        leading = points.shape[:-1]
        flat = points.reshape((-1, 3))
        mass = self.mass.astype(points.dtype)
        spin = self.spin.astype(points.dtype)
        center = self.center.astype(points.dtype)

        def evaluate(point):
            lapse, shift, metric, extrinsic, radius = _kerr_schild_point(
                point, mass, spin, center
            )
            return lapse, shift, metric, extrinsic, radius

        lapse, shift, metric, extrinsic, radius = jax.vmap(evaluate)(flat)
        lapse = lapse.reshape(leading)
        shift = shift.reshape(leading + (3,))
        metric = metric.reshape(leading + (3, 3))
        extrinsic = extrinsic.reshape(leading + (3, 3))
        radius = radius.reshape(leading)
        domain = radius > jnp.asarray(self.excision_radius, dtype=points.dtype)
        return _package_initial_data(
            points,
            lapse,
            shift,
            metric,
            extrinsic,
            jnp.ones(leading, dtype=points.dtype),
            domain,
            self.data_id,
        )


def minkowski_initial_data(coordinates: ArrayLike, /) -> ADMInitialData:
    return MinkowskiInitialData()(coordinates)


def isotropic_schwarzschild_initial_data(
    coordinates: ArrayLike,
    mass: ArrayLike,
    /,
    *,
    center: ArrayLike = (0.0, 0.0, 0.0),
    excision_radius: float = 0.0,
) -> ADMInitialData:
    return IsotropicSchwarzschildInitialData(
        mass, center=center, excision_radius=excision_radius
    )(coordinates)


def kerr_schild_initial_data(
    coordinates: ArrayLike,
    mass: ArrayLike,
    spin: ArrayLike = 0.0,
    /,
    *,
    center: ArrayLike = (0.0, 0.0, 0.0),
    excision_radius: float = 0.0,
) -> ADMInitialData:
    return KerrSchildInitialData(
        mass, spin, center=center, excision_radius=excision_radius
    )(coordinates)


def adm_constraint_diagnostics(
    field: Callable[[Array], ADMInitialData],
    coordinates: ArrayLike,
    /,
    *,
    tolerance: float = 1.0e-8,
) -> ADMConstraintDiagnostics:
    """Evaluate the vacuum ADM constraints by pointwise automatic differentiation."""

    tolerance_ = float(tolerance)
    if not isfinite(tolerance_) or tolerance_ < 0.0:
        raise ValueError("constraint tolerance must be finite and non-negative.")
    points = _coordinates(coordinates)
    flat = points.reshape((-1, 3))

    def at_point(point):
        data = field(point)

        def metric_map(query):
            return field(query).spatial_metric

        def extrinsic_map(query):
            return field(query).extrinsic_curvature

        metric = data.spatial_metric
        inverse_result = inverse_small_linear(_SMALL_3, metric)
        inverse = inverse_result.value
        metric_derivative = jax.jacfwd(metric_map)(point)
        christoffel = _christoffel(inverse, metric_derivative)
        christoffel_derivative = jax.jacfwd(
            lambda query: _christoffel(
                inverse_small_linear(_SMALL_3, metric_map(query)).value,
                jax.jacfwd(metric_map)(query),
            )
        )(point)
        ricci = (
            ein.contract("kijk->ij", christoffel_derivative)
            - ein.contract("kikj->ij", christoffel_derivative)
            + ein.contract("kij,lkl->ij", christoffel, christoffel)
            - ein.contract("kil,ljk->ij", christoffel, christoffel)
        )
        scalar = ein.contract("ij,ij->", inverse, ricci)
        extrinsic = data.extrinsic_curvature
        trace = ein.contract("ij,ij->", inverse, extrinsic)
        raised = ein.contract("ik,jl,kl->ij", inverse, inverse, extrinsic)
        hamiltonian = scalar + trace**2 - ein.contract("ij,ij->", extrinsic, raised)

        def trace_reversed(query):
            metric_q = metric_map(query)
            inverse_q = inverse_small_linear(_SMALL_3, metric_q).value
            extrinsic_q = extrinsic_map(query)
            trace_q = ein.contract("ij,ij->", inverse_q, extrinsic_q)
            raised_q = ein.contract("ik,jl,kl->ij", inverse_q, inverse_q, extrinsic_q)
            return raised_q - trace_q * inverse_q

        tensor = trace_reversed(point)
        derivative = jax.jacfwd(trace_reversed)(point)
        momentum = (
            ein.contract("ijj->i", derivative)
            + ein.contract("ijk,kj->i", christoffel, tensor)
            + ein.contract("jjk,ik->i", christoffel, tensor)
        )
        valid = data.domain_valid & inverse_result.successful
        return hamiltonian, momentum, valid, data.status.derivative_valid

    hamiltonian, momentum, domain, derivative_valid = jax.vmap(at_point)(flat)
    hamiltonian = hamiltonian.reshape(points.shape[:-1])
    momentum = momentum.reshape(points.shape[:-1] + (3,))
    domain = domain.reshape(points.shape[:-1])
    finite = jnp.all(jnp.isfinite(hamiltonian)) & jnp.all(jnp.isfinite(momentum))
    h_linf = jnp.max(jnp.abs(hamiltonian))
    m_linf = jnp.max(jnp.abs(momentum))
    converged = finite & (jnp.maximum(h_linf, m_linf) <= tolerance_)
    valid = jnp.all(domain)
    derivative = jnp.all(derivative_valid) & valid & finite
    return ADMConstraintDiagnostics(
        hamiltonian,
        momentum,
        h_linf,
        m_linf,
        finite,
        valid,
        converged,
        finite & valid,
        converged & valid,
        derivative,
    )


def adm_charge_diagnostics(
    field: Callable[[Array], ADMInitialData],
    surface_coordinates: ArrayLike,
    outward_normals: ArrayLike,
    area_weights: ArrayLike,
    /,
    *,
    angular_origin: ArrayLike = (0.0, 0.0, 0.0),
    surface_id: str = "adm-charge-surface",
    quadrature_converged: bool = True,
    surface_tolerance: float = 1.0e-6,
    quadrature_qualified: bool = True,
) -> ADMChargeDiagnostics:
    """Integrate standard asymptotically Cartesian ADM surface charges."""

    points = _coordinates(surface_coordinates)
    if points.ndim != 2:
        raise ValueError("ADM charge quadrature coordinates must have shape (q, 3).")
    normals = jnp.asarray(outward_normals, dtype=points.dtype)
    weights = jnp.asarray(area_weights, dtype=points.dtype)
    origin = jnp.asarray(angular_origin, dtype=points.dtype)
    if normals.shape != points.shape or weights.shape != points.shape[:-1]:
        raise ValueError("ADM charge normals/weights must match quadrature points.")
    if origin.shape != (3,):
        raise ValueError("angular_origin must be one three-vector.")
    identifier = str(surface_id)
    if not identifier:
        raise ValueError("surface_id must be non-empty.")
    tolerance = float(surface_tolerance)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("surface_tolerance must be finite and non-negative.")

    def integrands(point, normal):
        data = field(point)

        def metric_map(query):
            return field(query).spatial_metric

        metric = data.spatial_metric
        derivative = jax.jacfwd(metric_map)(point)
        mass_vector = ein.contract("ijj->i", derivative) - ein.contract(
            "jji->i", derivative
        )
        mass_density = ein.contract("i,i->", mass_vector, normal)
        inverse_result = inverse_small_linear(_SMALL_3, metric)
        trace = ein.contract("ij,ij->", inverse_result.value, data.extrinsic_curvature)
        momentum_density = ein.contract(
            "ij,j->i", data.extrinsic_curvature - trace * metric, normal
        )
        angular_density = jnp.cross(point - origin, momentum_density)
        valid = data.domain_valid & inverse_result.successful
        return (
            mass_density,
            momentum_density,
            angular_density,
            valid,
            data.status.derivative_valid,
        )

    mass_density, momentum_density, angular_density, domain, derivative = jax.vmap(
        integrands
    )(points, normals)
    mass = ein.contract("q,q->", weights, mass_density) / (16.0 * pi)
    momentum = ein.contract("q,qi->i", weights, momentum_density) / (8.0 * pi)
    angular = ein.contract("q,qi->i", weights, angular_density) / (8.0 * pi)
    normal_norms = jnp.sqrt(ein.contract("qi,qi->q", normals, normals))
    normalization_defect = jnp.max(jnp.abs(normal_norms - 1.0))
    total_area = jnp.sum(weights)
    closure_defect = jnp.sqrt(
        ein.contract(
            "i,i->",
            ein.contract("q,qi->i", weights, normals),
            ein.contract("q,qi->i", weights, normals),
        )
    ) / jnp.where(total_area > 0.0, total_area, 1.0)
    finite = (
        jnp.all(jnp.isfinite(points))
        & jnp.all(jnp.isfinite(normals))
        & jnp.all(jnp.isfinite(weights))
        & jnp.isfinite(mass)
        & jnp.all(jnp.isfinite(momentum))
        & jnp.all(jnp.isfinite(angular))
    )
    domain_valid = jnp.all(domain) & jnp.all(weights >= 0.0) & (total_area > 0.0)
    physical = finite & domain_valid
    converged = finite & jnp.asarray(bool(quadrature_converged))
    qualified = (
        converged
        & physical
        & jnp.asarray(bool(quadrature_qualified))
        & (normalization_defect <= tolerance)
        & (closure_defect <= tolerance)
    )
    derivative_valid = physical & jnp.all(derivative)
    return ADMChargeDiagnostics(
        mass,
        momentum,
        angular,
        mass_density,
        momentum_density,
        normalization_defect,
        closure_defect,
        finite,
        domain_valid,
        converged,
        physical,
        qualified,
        derivative_valid,
        canonical_fingerprint(
            {
                "kind": "adm-charge-quadrature",
                "surface_id": identifier,
                "point_count": points.shape[0],
                "surface_tolerance": tolerance,
            }
        ),
    )


def _coordinates(value: ArrayLike, /) -> Array:
    points = jnp.asarray(value)
    if points.shape == () or points.shape[-1] != 3:
        raise ValueError("Initial-data coordinates must have trailing shape (3,).")
    if not jnp.issubdtype(points.dtype, jnp.floating):
        points = points.astype("float64")
    return points


def _package_initial_data(
    coordinates,
    lapse,
    shift,
    metric,
    extrinsic,
    conformal_factor,
    domain_valid,
    data_id,
):
    determinant = determinant_small_linear(_SMALL_3, metric)
    finite = (
        jnp.all(jnp.isfinite(coordinates))
        & jnp.all(jnp.isfinite(lapse))
        & jnp.all(jnp.isfinite(shift))
        & jnp.all(jnp.isfinite(metric))
        & jnp.all(jnp.isfinite(extrinsic))
        & jnp.all(jnp.isfinite(conformal_factor))
        & jnp.all(jnp.isfinite(determinant))
    )
    lapse_positive = jnp.all(lapse > 0.0)
    conformal_positive = jnp.all(conformal_factor > 0.0)
    spatial_positive = jnp.all(determinant > 0.0)
    domain = jnp.all(domain_valid)
    physical = finite & lapse_positive & conformal_positive & spatial_positive & domain
    status_value = (
        jnp.where(finite, 0, int(NumericalRelativityStatus.NONFINITE_STATE))
        | jnp.where(lapse_positive, 0, int(NumericalRelativityStatus.NONPOSITIVE_LAPSE))
        | jnp.where(
            conformal_positive,
            0,
            int(NumericalRelativityStatus.NONPOSITIVE_CONFORMAL_FACTOR),
        )
        | jnp.where(
            spatial_positive,
            0,
            int(NumericalRelativityStatus.SINGULAR_CONFORMAL_METRIC),
        )
        | jnp.where(domain, 0, int(NumericalRelativityStatus.SOURCE_INVALID))
    ).astype(jnp.int32)
    return ADMInitialData(
        coordinates,
        lapse,
        shift,
        metric,
        extrinsic,
        conformal_factor,
        determinant,
        jnp.asarray(domain_valid, dtype=jnp.bool_),
        ScientificStatus(
            status_value,
            finite,
            True,
            physical,
            True,
            physical,
        ),
        data_id,
    )


def _christoffel(inverse: Array, metric_derivative: Array, /) -> Array:
    lowered = (
        jnp.transpose(metric_derivative, (0, 2, 1))
        + metric_derivative
        - jnp.transpose(metric_derivative, (2, 0, 1))
    )
    return 0.5 * ein.contract("kl,lij->kij", inverse, lowered)


def _kerr_spatial_fields(point, mass, spin, center):
    position = point - center
    radius_squared_cartesian = ein.contract("i,i->", position, position)
    spin_squared = ein.contract("i,i->", spin, spin)
    spin_dot_position = ein.contract("i,i->", spin, position)
    discriminant = (
        radius_squared_cartesian - spin_squared
    ) ** 2 + 4.0 * spin_dot_position**2
    radius_squared = 0.5 * (
        radius_squared_cartesian - spin_squared + jnp.sqrt(jnp.maximum(discriminant, 0.0))
    )
    radius = jnp.sqrt(jnp.maximum(radius_squared, 0.0))
    safe_radius = jnp.where(radius > 0.0, radius, 1.0)
    denominator = radius_squared + spin_squared
    safe_denominator = jnp.where(denominator > 0.0, denominator, 1.0)
    null_spatial = (
        radius * position
        - jnp.cross(spin, position)
        + spin_dot_position * spin / safe_radius
    ) / safe_denominator
    h_denominator = radius_squared**2 + spin_dot_position**2
    h = mass * radius**3 / jnp.where(h_denominator > 0.0, h_denominator, 1.0)
    h = jnp.where(radius > 0.0, h, 0.0)
    metric = jnp.eye(3, dtype=point.dtype) + 2.0 * h * ein.contract(
        "i,j->ij", null_spatial, null_spatial
    )
    lapse = 1.0 / jnp.sqrt(1.0 + 2.0 * h)
    shift = 2.0 * h * null_spatial / (1.0 + 2.0 * h)
    shift_covector = 2.0 * h * null_spatial
    return lapse, shift, metric, shift_covector, radius


def _kerr_schild_point(point, mass, spin, center):
    lapse, shift, metric, shift_covector, radius = _kerr_spatial_fields(
        point, mass, spin, center
    )
    metric_derivative = jax.jacfwd(
        lambda query: _kerr_spatial_fields(query, mass, spin, center)[2]
    )(point)
    shift_derivative = jax.jacfwd(
        lambda query: _kerr_spatial_fields(query, mass, spin, center)[3]
    )(point)
    inverse = inverse_small_linear(_SMALL_3, metric).value
    christoffel = _christoffel(inverse, metric_derivative)
    covariant_shift = jnp.swapaxes(shift_derivative, -1, -2) - ein.contract(
        "kij,k->ij", christoffel, shift_covector
    )
    extrinsic = (covariant_shift + jnp.swapaxes(covariant_shift, -1, -2)) / (2.0 * lapse)
    return lapse, shift, metric, extrinsic, radius
