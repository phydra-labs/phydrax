#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from operator import index
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike
from s2fft.recursions.risbo_jax import compute_full as _wigner_small_d

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    ArraySpace,
    DenseLinearOperator,
    DenseSVD,
    LeastSquaresProblem,
    LinearSolvePolicy,
    prepare as prepare_linear_solve,
    PreparedLinearSolve,
    RHSLayout,
    solve as linear_solve,
)
from ._spherical import SphericalSpectralDiscretization


HealpixOrdering: TypeAlias = Literal["ring", "nested"]


def _positive_bytes(value: int, name: str, /) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    resolved = index(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive.")
    return resolved


class SphericalSamplePlan(StrictModule, NonTrainableState):
    """Fixed-capacity scattered samples on a sphere.

    Points use ambient Cartesian coordinates. Inactive rows may contain arbitrary
    values, including NaNs; they are sanitized before geometry or arithmetic.
    """

    points: Array
    weights: Array
    active_mask: Array
    ordering: HealpixOrdering | None = eqx.field(static=True)
    nside: int | None = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    tikhonov: float = eqx.field(static=True)
    maximum_design_bytes: int = eqx.field(static=True)
    maximum_factor_bytes: int = eqx.field(static=True)
    sample_id: str = eqx.field(static=True)

    def __init__(
        self,
        points: ArrayLike,
        /,
        *,
        weights: ArrayLike | None = None,
        active_mask: ArrayLike | None = None,
        rank_tolerance: float = 1e-12,
        condition_limit: float = 1e12,
        tikhonov: float = 0.0,
        maximum_design_bytes: int = 512 * 1024**2,
        maximum_factor_bytes: int = 1024 * 1024**2,
        _ordering: HealpixOrdering | None = None,
        _nside: int | None = None,
    ):
        points_ = jnp.asarray(points)
        if points_.ndim != 2 or points_.shape[1] != 3 or points_.shape[0] == 0:
            raise ValueError("Spherical sample points must have nonempty shape (n, 3).")
        if jnp.iscomplexobj(points_):
            raise TypeError("Spherical sample points must be real.")
        points_ = points_.astype(float)
        count = int(points_.shape[0])
        mask = (
            jnp.ones((count,), dtype=bool)
            if active_mask is None
            else jnp.asarray(active_mask, dtype=bool)
        )
        if mask.shape != (count,):
            raise ValueError("active_mask must have shape (sample_capacity,).")
        weights_ = (
            jnp.ones((count,), dtype=points_.dtype)
            if weights is None
            else jnp.asarray(weights)
        )
        if weights_.shape != (count,) or jnp.iscomplexobj(weights_):
            raise ValueError("weights must be one real value per sample row.")
        weights_ = weights_.astype(points_.dtype)
        sanitized_weights = jnp.where(mask, weights_, 1.0)
        if not bool(
            jnp.all(jnp.isfinite(sanitized_weights)) & jnp.all(sanitized_weights > 0.0)
        ):
            raise ValueError(
                "Active spherical sample weights must be finite and positive."
            )
        tolerance = float(rank_tolerance)
        condition = float(condition_limit)
        regularization = float(tikhonov)
        if (
            not math.isfinite(tolerance)
            or tolerance <= 0.0
            or not math.isfinite(condition)
            or condition <= 1.0
            or not math.isfinite(regularization)
            or regularization < 0.0
        ):
            raise ValueError("Spherical sample rank/condition/regularization is invalid.")
        if _ordering not in (None, "ring", "nested"):
            raise ValueError("HEALPix ordering must be 'ring' or 'nested'.")
        self.points = points_
        self.weights = weights_
        self.active_mask = mask
        self.ordering = _ordering
        self.nside = _nside
        self.rank_tolerance = tolerance
        self.condition_limit = condition
        self.tikhonov = regularization
        self.maximum_design_bytes = _positive_bytes(
            maximum_design_bytes, "maximum_design_bytes"
        )
        self.maximum_factor_bytes = _positive_bytes(
            maximum_factor_bytes, "maximum_factor_bytes"
        )
        self.sample_id = canonical_fingerprint(
            {
                "kind": "spherical-sample-plan",
                "points": array_tree_fingerprint(points_),
                "weights": array_tree_fingerprint(weights_),
                "active_mask": array_tree_fingerprint(mask),
                "ordering": _ordering,
                "nside": _nside,
                "rank_tolerance": tolerance,
                "condition_limit": condition,
                "tikhonov": regularization,
                "maximum_design_bytes": self.maximum_design_bytes,
                "maximum_factor_bytes": self.maximum_factor_bytes,
            }
        )

    @classmethod
    def healpix(
        cls,
        nside: int,
        /,
        *,
        ordering: HealpixOrdering = "ring",
        active_mask: ArrayLike | None = None,
        rank_tolerance: float = 1e-12,
        condition_limit: float = 1e12,
        tikhonov: float = 0.0,
        maximum_design_bytes: int = 512 * 1024**2,
        maximum_factor_bytes: int = 1024 * 1024**2,
    ) -> "SphericalSamplePlan":
        if isinstance(nside, bool):
            raise TypeError("nside must be an integer.")
        nside_ = index(nside)
        if nside_ <= 0:
            raise ValueError("nside must be positive.")
        if ordering not in ("ring", "nested"):
            raise ValueError("HEALPix ordering must be 'ring' or 'nested'.")
        if ordering == "nested" and (nside_ & (nside_ - 1)):
            raise ValueError("Nested HEALPix ordering requires power-of-two nside.")
        points = _healpix_points(nside_, ordering)
        weights = np.full((12 * nside_**2,), 4.0 * math.pi / (12 * nside_**2))
        return cls(
            points,
            weights=weights,
            active_mask=active_mask,
            rank_tolerance=rank_tolerance,
            condition_limit=condition_limit,
            tikhonov=tikhonov,
            maximum_design_bytes=maximum_design_bytes,
            maximum_factor_bytes=maximum_factor_bytes,
            _ordering=ordering,
            _nside=nside_,
        )

    @property
    def sample_capacity(self) -> int:
        return int(self.points.shape[0])

    def prepare(
        self, discretization: SphericalSpectralDiscretization, /
    ) -> "PreparedSphericalSampleOperator":
        return PreparedSphericalSampleOperator(self, discretization)


class SphericalSampleReport(StrictModule, NonTrainableState):
    sample_capacity: int = eqx.field(static=True)
    active_count: int = eqx.field(static=True)
    mode_count: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    design_bytes: int = eqx.field(static=True)
    factor_bytes: int = eqx.field(static=True)
    condition_number: float = eqx.field(static=True)
    regularized: bool = eqx.field(static=True)
    exact_recovery: bool = eqx.field(static=True)
    ordering: HealpixOrdering | None = eqx.field(static=True)
    nside: int | None = eqx.field(static=True)
    active_weight_sum: float = eqx.field(static=True)
    report_id: str = eqx.field(static=True)


class SphericalSampleFitResult(StrictModule):
    coefficients: Array
    residual_norm: Array
    solve_status: Array
    rank: int = eqx.field(static=True)
    condition_number: float = eqx.field(static=True)
    regularized: bool = eqx.field(static=True)
    exact_recovery: bool = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class PreparedSphericalSampleOperator(StrictModule, NonTrainableState):
    plan: SphericalSamplePlan
    discretization: SphericalSpectralDiscretization
    design: Array
    fit_solve: PreparedLinearSolve
    mode_indices: Array
    report: SphericalSampleReport
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: SphericalSamplePlan,
        discretization: SphericalSpectralDiscretization,
        /,
    ):
        if not isinstance(plan, SphericalSamplePlan):
            raise TypeError("plan must be a SphericalSamplePlan.")
        if not isinstance(discretization, SphericalSpectralDiscretization):
            raise TypeError("discretization must be spherical spectral.")
        mask = np.asarray(plan.active_mask, dtype=bool)
        active_count = int(np.count_nonzero(mask))
        if active_count == 0:
            raise ValueError("At least one spherical sample must be active.")
        radius = float(discretization.radius)
        declared_points = np.asarray(plan.points)
        if plan.nside is not None:
            declared_points = radius * declared_points
        safe_points = np.where(mask[:, None], declared_points, [0.0, 0.0, radius])
        norms = np.linalg.norm(safe_points, axis=1)
        radial_tolerance = max(plan.rank_tolerance * radius, 64.0 * np.finfo(float).eps)
        if not np.all(np.isfinite(safe_points[mask])) or np.any(
            np.abs(norms[mask] - radius) > radial_tolerance
        ):
            raise ValueError("Active sample points must be finite and lie on the sphere.")
        theta = np.arccos(np.clip(safe_points[:, 2] / radius, -1.0, 1.0))
        phi = np.mod(np.arctan2(safe_points[:, 1], safe_points[:, 0]), 2.0 * math.pi)
        full_design, indices = _spin_harmonic_design(
            jnp.asarray(theta),
            jnp.asarray(phi),
            discretization,
        )
        design = _independent_design(full_design, discretization)
        design = jnp.where(plan.active_mask[:, None], design, 0.0)
        mode_count = int(design.shape[1])
        if plan.tikhonov == 0.0 and active_count < mode_count:
            raise ValueError(
                "Exact spherical fitting requires at least as many active samples as modes."
            )
        design_bytes = int(design.nbytes)
        factor_bytes = int(
            design.dtype.itemsize
            * (design.size + min(design.shape) ** 2 + min(design.shape))
        )
        if design_bytes > plan.maximum_design_bytes:
            raise ValueError("Spherical sample design exceeds maximum_design_bytes.")
        if factor_bytes > plan.maximum_factor_bytes:
            raise ValueError("Spherical sample factor exceeds maximum_factor_bytes.")
        source = ArraySpace((mode_count,), dtype=design.dtype)
        target = ArraySpace((plan.sample_capacity,), dtype=design.dtype)
        operator = DenseLinearOperator(
            design,
            source=source,
            target=target,
            operator_id=canonical_fingerprint(
                {
                    "kind": "spherical-sample-design",
                    "samples": plan.sample_id,
                    "discretization": discretization.prepared_id,
                }
            ),
        )
        problem = LeastSquaresProblem(
            operator,
            weights=jnp.where(plan.active_mask, plan.weights, 0.0),
        )
        fit_solve = prepare_linear_solve(
            problem,
            LinearSolvePolicy(DenseSVD(damping=plan.tikhonov)),
        )
        singular_values = np.asarray(fit_solve.state.reported_singular_values)
        largest = float(singular_values[0]) if singular_values.size else 0.0
        threshold = plan.rank_tolerance * largest
        rank = int(np.count_nonzero(singular_values > threshold))
        smallest = float(singular_values[-1]) if singular_values.size else 0.0
        condition = math.inf if smallest <= 0.0 else largest / smallest
        exact = plan.tikhonov == 0.0 and rank == mode_count
        if plan.tikhonov == 0.0 and (
            rank != mode_count
            or not math.isfinite(condition)
            or condition > plan.condition_limit
        ):
            raise ValueError(
                "Spherical sample design is rank deficient or exceeds condition_limit."
            )
        report_id = canonical_fingerprint(
            {
                "kind": "spherical-sample-report",
                "sample": plan.sample_id,
                "discretization": discretization.prepared_id,
                "rank": rank,
                "condition_number": condition,
                "exact_recovery": exact,
            }
        )
        report = SphericalSampleReport(
            sample_capacity=plan.sample_capacity,
            active_count=active_count,
            mode_count=mode_count,
            rank=rank,
            design_bytes=design_bytes,
            factor_bytes=factor_bytes,
            condition_number=condition,
            regularized=plan.tikhonov > 0.0,
            exact_recovery=exact,
            ordering=plan.ordering,
            nside=plan.nside,
            active_weight_sum=float(
                np.sum(np.where(mask, np.asarray(plan.weights), 0.0))
            ),
            report_id=report_id,
        )
        self.plan = plan
        self.discretization = discretization
        self.design = design
        self.fit_solve = fit_solve
        self.mode_indices = indices
        self.report = report
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-spherical-sample-operator",
                "sample": plan.sample_id,
                "discretization": discretization.prepared_id,
                "report": report_id,
            }
        )

    def evaluate(self, coefficients: ArrayLike, /) -> Array:
        coordinates = _coefficients_to_coordinates(
            coefficients, self.discretization, self.mode_indices
        )
        payload_shape = coordinates.shape[1:]
        flattened = coordinates.reshape((coordinates.shape[0], -1))
        values = oe.contract("nm,mk->nk", self.design, flattened, backend="jax")
        values = values.reshape((self.plan.sample_capacity,) + payload_shape)
        mask = self.plan.active_mask.reshape(
            (self.plan.sample_capacity,) + (1,) * len(payload_shape)
        )
        return jnp.where(mask, values, jnp.zeros((), dtype=values.dtype))

    def fit(self, values: ArrayLike, /) -> SphericalSampleFitResult:
        array = jnp.asarray(values)
        if array.ndim < 1 or array.shape[0] != self.plan.sample_capacity:
            raise ValueError(
                "Spherical sample values must begin with the fixed sample capacity."
            )
        if self.discretization.layout.reality and jnp.iscomplexobj(array):
            raise TypeError(
                "Real spin-zero spherical fitting requires real sample values."
            )
        array = array.astype(self.design.dtype)
        payload_shape = tuple(array.shape[1:])
        mask = self.plan.active_mask.reshape(
            (self.plan.sample_capacity,) + (1,) * len(payload_shape)
        )
        sanitized = jnp.where(mask, array, jnp.zeros((), dtype=array.dtype))
        result = linear_solve(
            self.fit_solve,
            sanitized,
            rhs_layout=None if not payload_shape else RHSLayout(payload_shape),
        )
        coefficients = _coordinates_to_coefficients(
            result.value, self.discretization, self.mode_indices
        )
        fitted = self.evaluate(coefficients)
        weighted = jnp.sqrt(jnp.where(self.plan.active_mask, self.plan.weights, 0.0))
        weight_shape = (self.plan.sample_capacity,) + (1,) * len(payload_shape)
        residual = jnp.linalg.norm(weighted.reshape(weight_shape) * (fitted - sanitized))
        return SphericalSampleFitResult(
            coefficients=coefficients,
            residual_norm=residual,
            solve_status=result.status,
            rank=self.report.rank,
            condition_number=self.report.condition_number,
            regularized=self.report.regularized,
            exact_recovery=self.report.exact_recovery,
            prepared_id=self.prepared_id,
        )


def _spin_harmonic_design(
    theta: Array,
    phi: Array,
    discretization: SphericalSpectralDiscretization,
    /,
) -> tuple[Array, Array]:
    layout = discretization.layout
    limit = layout.bandlimit
    initial = jnp.zeros((2 * limit - 1, 2 * limit - 1), dtype=theta.dtype)
    planes = []
    current = jnp.broadcast_to(initial, (theta.shape[0],) + initial.shape)
    for degree in range(limit):
        current = jax.vmap(
            lambda plane, angle: _wigner_small_d(plane, angle, limit, degree)
        )(current, theta)
        planes.append(current)
    degree_planes = jnp.stack(tuple(planes), axis=1)
    flat_degrees = layout.degrees.reshape((-1,))[layout.valid_indices]
    flat_orders = layout.orders.reshape((-1,))[layout.valid_indices]
    offset = limit - 1
    values = degree_planes[
        :,
        flat_degrees,
        flat_orders + offset,
        jnp.full_like(flat_orders, -layout.spin + offset),
    ]
    scale = ((-1.0) ** layout.spin) * jnp.sqrt(
        (2.0 * flat_degrees + 1.0) / (4.0 * jnp.pi)
    )
    design = values * scale[None, :] * jnp.exp(1j * phi[:, None] * flat_orders[None, :])
    return design, layout.valid_indices


def _independent_design(
    full_design: Array,
    discretization: SphericalSpectralDiscretization,
    /,
) -> Array:
    layout = discretization.layout
    if not layout.reality:
        return full_design.astype(
            jnp.dtype(discretization.plan.precision.coefficient_dtype)
        )
    valid_orders = layout.orders.reshape((-1,))[layout.valid_indices]
    columns = []
    for column, order in enumerate(np.asarray(valid_orders)):
        if order == 0:
            columns.append(jnp.real(full_design[:, column]))
        elif order > 0:
            columns.append(jnp.sqrt(2.0) * jnp.real(full_design[:, column]))
            columns.append(-jnp.sqrt(2.0) * jnp.imag(full_design[:, column]))
    return jnp.stack(tuple(columns), axis=1).astype(
        jnp.dtype(discretization.plan.precision.physical_dtype)
    )


def _coefficients_to_coordinates(
    coefficients: ArrayLike,
    discretization: SphericalSpectralDiscretization,
    indices: Array,
    /,
) -> Array:
    layout = discretization.layout
    modal = layout.mask_invalid(coefficients)
    if modal.ndim < 2 or tuple(modal.shape[:2]) != layout.coefficient_shape:
        raise ValueError("Spherical coefficients must begin with the modal shape.")
    if layout.reality:
        defect = layout.conjugacy_defect(modal)
        modal = eqx.error_if(
            modal,
            defect > 64.0 * jnp.finfo(jnp.real(modal).dtype).eps,
            "Spherical coefficients violate the declared real-field conjugacy.",
        )
    payload_shape = modal.shape[2:]
    flattened = modal.reshape((math.prod(layout.coefficient_shape),) + payload_shape)
    valid = flattened[indices]
    if not layout.reality:
        return valid
    orders = np.asarray(layout.orders.reshape((-1,))[indices])
    columns = []
    for column, order in enumerate(orders):
        if order == 0:
            columns.append(jnp.real(valid[column]))
        elif order > 0:
            columns.extend(
                (
                    jnp.sqrt(2.0) * jnp.real(valid[column]),
                    jnp.sqrt(2.0) * jnp.imag(valid[column]),
                )
            )
    return jnp.stack(tuple(columns), axis=0)


def _coordinates_to_coefficients(
    coordinates: ArrayLike,
    discretization: SphericalSpectralDiscretization,
    indices: Array,
    /,
) -> Array:
    layout = discretization.layout
    values = jnp.asarray(coordinates)
    payload_shape = values.shape[1:]
    dtype = jnp.dtype(discretization.plan.precision.coefficient_dtype)
    flat = jnp.zeros((math.prod(layout.coefficient_shape),) + payload_shape, dtype=dtype)
    if not layout.reality:
        flat = flat.at[indices].set(values.astype(dtype))
        return layout.mask_invalid(flat.reshape(layout.coefficient_shape + payload_shape))
    orders = np.asarray(layout.orders.reshape((-1,))[indices])
    cursor = 0
    for mode_position, order in enumerate(orders):
        if order == 0:
            flat = flat.at[indices[mode_position]].set(values[cursor].astype(dtype))
            cursor += 1
        elif order > 0:
            coefficient = (
                values[cursor].astype(dtype) + 1j * values[cursor + 1].astype(dtype)
            ) / jnp.sqrt(2.0)
            flat = flat.at[indices[mode_position]].set(coefficient)
            cursor += 2
    reshaped = flat.reshape(layout.coefficient_shape + payload_shape)
    return layout.canonicalize_reality(reshaped)


def _healpix_points(nside: int, ordering: HealpixOrdering, /) -> np.ndarray:
    if ordering == "nested":
        return _healpix_nested_points(nside)
    points = []
    for ring in range(1, nside):
        count = 4 * ring
        z = 1.0 - ring * ring / (3.0 * nside * nside)
        for position in range(1, count + 1):
            phi = (position - 0.5) * math.pi / (2.0 * ring)
            points.append(_unit_point(z, phi))
    for ring in range(nside, 3 * nside + 1):
        z = (2 * nside - ring) * 2.0 / (3.0 * nside)
        offset = 0.5 * (1 + ((ring + nside) & 1))
        for position in range(1, 4 * nside + 1):
            phi = (position - offset) * math.pi / (2.0 * nside)
            points.append(_unit_point(z, phi))
    for ring in range(nside - 1, 0, -1):
        count = 4 * ring
        z = -1.0 + ring * ring / (3.0 * nside * nside)
        for position in range(1, count + 1):
            phi = (position - 0.5) * math.pi / (2.0 * ring)
            points.append(_unit_point(z, phi))
    return np.asarray(points, dtype=float)


def _healpix_nested_points(nside: int, /) -> np.ndarray:
    jrll = (2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4)
    jpll = (1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7)
    points = []
    face_size = nside * nside
    for pixel in range(12 * face_size):
        face = pixel // face_size
        within = pixel % face_size
        ix = _compact_bits(within)
        iy = _compact_bits(within >> 1)
        jr = jrll[face] * nside - ix - iy - 1
        if jr < nside:
            nr = jr
            z = 1.0 - nr * nr / (3.0 * nside * nside)
            shift = 0
        elif jr > 3 * nside:
            nr = 4 * nside - jr
            z = -1.0 + nr * nr / (3.0 * nside * nside)
            shift = 0
        else:
            nr = nside
            z = (2 * nside - jr) * 2.0 / (3.0 * nside)
            shift = (jr - nside) & 1
        jp = (jpll[face] * nr + ix - iy + 1 + shift) // 2
        if jp > 4 * nr:
            jp -= 4 * nr
        if jp < 1:
            jp += 4 * nr
        phi = (jp - 0.5 * (shift + 1)) * math.pi / (2.0 * nr)
        points.append(_unit_point(z, phi))
    return np.asarray(points, dtype=float)


def _compact_bits(value: int, /) -> int:
    result = 0
    bit = 0
    while value:
        result |= (value & 1) << bit
        value >>= 2
        bit += 1
    return result


def _unit_point(z: float, phi: float, /) -> tuple[float, float, float]:
    radius = math.sqrt(max(0.0, 1.0 - z * z))
    return radius * math.cos(phi), radius * math.sin(phi), z


__all__ = [
    "HealpixOrdering",
    "PreparedSphericalSampleOperator",
    "SphericalSampleFitResult",
    "SphericalSamplePlan",
    "SphericalSampleReport",
]
