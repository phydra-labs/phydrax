# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Compiled weighted-SVD GMLS and polynomially augmented PHS fits.

Host preparation validates the declaration, assembles every requested
functional once and admits rows at one synchronization boundary. The numerical
fit is one pure traceable kernel mapped over padded fixed-size row chunks; the
same kernel serves preparation, fixed-support refresh and coordinate
derivatives, so refreshed stencils are differentiable in their coordinates on
a frozen support relation.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from enum import IntEnum
from functools import cache
from math import factorial, prod
from typing import assert_never, final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._admissibility import guard_derivative_validity
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import FactorizationPolicy, FailurePolicy, pseudoinverse, RankPolicy
from ...typing import parse
from ._capacity import bucketed_storage_capacity
from ._neighbors import _integer, PreparedMeshfreeNeighborhood, SmoothSupportEnvelope
from ._precision import MeshfreePrecisionRole
from ._types import (
    MeshfreeApproximation,
    MeshfreeRowStatus,
    StencilAcceptance,
    StencilWeightKernel,
)


type StencilTerms = tuple[tuple[int, ...], ...]

# Native dense SVD pseudoinverse with the unchanged relative rank cutoff. Its
# closed-form fixed-rank Moore-Penrose tangent is exact to roundoff, including
# minimum-norm GMLS factors and ill-conditioned PHS saddles, where an implicit
# normal-equation root would be iteration- or conditioning-limited.
_LOCAL_PSEUDOINVERSE = FactorizationPolicy(
    "svd",
    rank=RankPolicy(relative_cutoff=1e-12, require_full_rank=True),
    failure=FailurePolicy("status"),
)
# A local solve at unit roundoff ``u`` carries a relative forward error up to
# ``condition * u``: a row whose ``condition`` exceeds this limit over ``u`` is
# ILL_CONDITIONED in its fit precision (float64 caps at 4.5e12, above the
# default condition_limit; float32 caps at 8.4e3).
_FIT_FORWARD_ERROR_LIMIT = 1e-3
# Relative moment residual admitted for float64 fits; a reduced-precision fit
# admits its rounding bound ``2 n u sum|w| / max|m|`` when larger (``n`` terms
# per moment, design entries bounded by one in scaled offsets).
_MOMENT_RESIDUAL_LIMIT = 1e-9
_FLOAT64_ROUNDOFF = float(np.finfo(np.float64).eps)


@final
class LocalStencilPolicy(StrictModule):
    """Method-specific local approximation policy.

    GMLS owns its moving-least-squares ``weight_kernel`` (default
    ``"inverse-square"``) and has no radial power. PHS-RBF-FD owns its odd
    ``phs_power`` (default 3) and has no weight kernel; supplying the other
    method's parameter is refused rather than silently ignored.

    A GMLS ``support`` (:class:`SmoothSupportEnvelope`) selects the smooth
    fixed-radius route: offsets are scaled by the fixed physical radius and
    the compact kernel (default ``"wendland-c4"``) vanishes at that radius with
    its coordinate derivatives through ``coordinate_order`` (default one), so
    stencil weights are ``C^coordinate_order`` in the coordinates while a
    neighbor enters or leaves the support inside the candidate envelope.
    ``"wendland-c2"`` vanishes through order three, ``"wendland-c4"`` through
    five, and ``"compact-polynomial"`` is ``(1 - r^2)^(coordinate_order + 1)``.
    Nearest-neighbor GMLS and PHS-RBF-FD have no such claim and refuse it.
    """

    approximation: MeshfreeApproximation = eqx.field(static=True)
    polynomial_degree: int = eqx.field(static=True)
    phs_power: int | None = eqx.field(static=True)
    weight_kernel: StencilWeightKernel | None = eqx.field(static=True)
    support: SmoothSupportEnvelope | None
    coordinate_order: int | None = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    amplification_limit: float = eqx.field(static=True)
    acceptance: StencilAcceptance = eqx.field(static=True)
    chunk_rows: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        approximation: MeshfreeApproximation = "gmls",
        polynomial_degree: int = 2,
        phs_power: int | None = None,
        weight_kernel: StencilWeightKernel | None = None,
        support: SmoothSupportEnvelope | None = None,
        coordinate_order: int | None = None,
        condition_limit: float = 1e8,
        amplification_limit: float = 1e8,
        acceptance: StencilAcceptance = "refuse",
        chunk_rows: int = 128,
    ) -> None:
        approximation_ = parse(approximation, MeshfreeApproximation, "approximation")
        degree = _integer(polynomial_degree, "polynomial_degree", 0)
        power: int | None
        kernel: StencilWeightKernel | None
        order: int | None
        match approximation_:
            case "gmls":
                if phs_power is not None:
                    raise ValueError(
                        "GMLS has no radial power; phs_power belongs to phs-rbf-fd."
                    )
                power = None
                kernel, order = _gmls_kernel(weight_kernel, support, coordinate_order)
            case "phs-rbf-fd":
                if weight_kernel is not None:
                    raise ValueError(
                        "PHS-RBF-FD has no weight kernel; weight_kernel belongs to gmls."
                    )
                if support is not None or coordinate_order is not None:
                    raise ValueError(
                        "PHS-RBF-FD has no smooth fixed-radius support; support "
                        "and coordinate_order belong to compact-kernel gmls."
                    )
                kernel, order = None, None
                power = _integer(3 if phs_power is None else phs_power, "phs_power", 3)
                if power % 2 == 0:
                    raise ValueError("phs_power must be odd.")
                if degree < (power - 1) // 2:
                    raise ValueError(
                        "PHS polynomial degree must cover its conditional-definiteness order."
                    )
            case _:
                assert_never(approximation_)
        if not np.isfinite(condition_limit) or condition_limit <= 1:
            raise ValueError("condition_limit must be finite and exceed one.")
        if not np.isfinite(amplification_limit) or amplification_limit <= 0:
            raise ValueError("amplification_limit must be finite and positive.")
        acceptance_ = parse(acceptance, StencilAcceptance, "acceptance")
        chunk = _integer(chunk_rows, "chunk_rows")
        self.approximation = approximation_
        self.polynomial_degree = degree
        self.phs_power = power
        self.weight_kernel = kernel
        self.support = support
        self.coordinate_order = order
        self.condition_limit = float(condition_limit)
        self.amplification_limit = float(amplification_limit)
        self.acceptance = acceptance_
        self.chunk_rows = chunk


def _gmls_kernel(
    weight_kernel: StencilWeightKernel | None,
    support: SmoothSupportEnvelope | None,
    coordinate_order: int | None,
) -> tuple[StencilWeightKernel, int | None]:
    """Validate a GMLS kernel against its nearest-neighbor or smooth support."""
    if support is None:
        if coordinate_order is not None:
            raise ValueError("coordinate_order belongs to a smooth fixed-radius support.")
        kernel = parse(
            "inverse-square" if weight_kernel is None else weight_kernel,
            StencilWeightKernel,
            "weight_kernel",
        )
        match kernel:
            case "wendland-c2" | "inverse-square":
                return kernel, None
            case "wendland-c4" | "compact-polynomial":
                raise ValueError(
                    f"{kernel!r} has a fixed physical cutoff and needs a smooth support."
                )
            case _:
                assert_never(kernel)
    if not isinstance(support, SmoothSupportEnvelope):
        raise TypeError("support must be a SmoothSupportEnvelope or None.")
    kernel = parse(
        "wendland-c4" if weight_kernel is None else weight_kernel,
        StencilWeightKernel,
        "weight_kernel",
    )
    order = _integer(
        1 if coordinate_order is None else coordinate_order, "coordinate_order"
    )
    # Highest coordinate-derivative order of the kernel vanishing at the cutoff.
    match kernel:
        case "wendland-c2":
            vanishing = 3
        case "wendland-c4":
            vanishing = 5
        case "compact-polynomial":
            vanishing = order
        case "inverse-square":
            raise ValueError(
                "inverse-square is not compactly supported; a smooth support needs "
                "a compact kernel."
            )
        case _:
            assert_never(kernel)
    if order > vanishing:
        raise ValueError(
            f"{kernel!r} vanishes at the cutoff only through coordinate order {vanishing}."
        )
    return kernel, order


@final
class MeshfreeFunctional(StrictModule):
    multi_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    coefficients: tuple[float, ...] = eqx.field(static=True)
    row_coefficients: Array | None
    name: str = eqx.field(static=True)

    def __init__(
        self,
        multi_indices: tuple[tuple[int, ...], ...],
        coefficients: Sequence[float] | ArrayLike,
        *,
        row_coefficients: ArrayLike | None = None,
        name: str = "functional",
    ) -> None:
        indices = tuple(
            tuple(_integer(v, "derivative order", 0) for v in index)
            for index in multi_indices
        )
        values = np.asarray(coefficients, dtype=np.float64)
        if (
            not indices
            or not indices[0]
            or any(len(index) != len(indices[0]) for index in indices)
        ):
            raise ValueError(
                "multi_indices must be nonempty and have a shared dimension."
            )
        if values.shape != (len(indices),) or not np.all(np.isfinite(values)):
            raise ValueError("coefficients must be finite with one value per derivative.")
        rows = (
            None
            if row_coefficients is None
            else np.asarray(row_coefficients, dtype=np.float64)
        )
        if rows is not None and (
            rows.ndim != 2
            or rows.shape[1] != len(indices)
            or not np.all(np.isfinite(rows))
        ):
            raise ValueError(
                "row_coefficients must have shape (rows, derivative terms) and be finite."
            )
        if not isinstance(name, str) or not name:
            raise ValueError("name must be nonempty.")
        self.multi_indices = indices
        self.coefficients = tuple(float(v) for v in values)
        self.row_coefficients = None if rows is None else jnp.asarray(rows)
        self.name = name


@final
class LocalStencilEvidence(StrictModule):
    status: Array
    condition: Array
    rank: Array
    moment_residual: Array
    amplification: Array
    minimum_singular_value: Array


@final
class LocalStencilReport(StrictModule):
    """Admission summary; ``precision`` holds the effective ``(role, dtype)``
    pairs of the fit (validated against the neighborhood's precision policy)."""

    maximum_condition_number: float = eqx.field(static=True)
    minimum_singular_value: float = eqx.field(static=True)
    minimum_rank: int = eqx.field(static=True)
    maximum_moment_residual: float = eqx.field(static=True)
    maximum_amplification: float = eqx.field(static=True)
    worst_row: int = eqx.field(static=True)
    refused_rows: int = eqx.field(static=True)
    precision: tuple[tuple[str, str], ...] = eqx.field(static=True)
    report_id: str = eqx.field(static=True)


@final
class PreparedLocalStencils(StrictModule):
    """Admitted stencils of one support epoch and their current numerical state.

    ``prepared_id`` and ``report`` identify and summarize the admitted epoch:
    anchored coordinates, frozen relation, functionals, basis and policy.
    ``offsets``, ``weights`` and ``evidence`` are the dynamic numerical state;
    fixed-support refresh replaces only these leaves. Point-prepared stencils
    retain their anchored sources and targets; chart-prepared stencils retain
    only their anchored chart offsets.
    """

    neighborhood: PreparedMeshfreeNeighborhood
    functionals: tuple[MeshfreeFunctional, ...]
    policy: LocalStencilPolicy
    coefficients: Array
    offsets: Array
    anchor_offsets: Array
    anchor_sources: Array | None
    anchor_targets: Array | None
    weights: tuple[Array, ...]
    evidence: LocalStencilEvidence
    report: LocalStencilReport
    terms: StencilTerms = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class LocalStencilRefreshStatus(IntEnum):
    ACCEPTED = 0
    INVALID_COORDINATES = 1
    SUPPORT_EXCEEDED = 2
    ROW_REFUSED = 3


@final
class LocalStencilRefresh(StrictModule):
    """Candidate fixed-support refresh; consumers must inspect ``accepted``.

    ``stencils`` holds the refitted weights and per-row evidence on the frozen
    relation; the weights are NaN (with NaN derivatives) unless ``accepted``.
    ``displacement`` is the (non-differentiated) anchored motion bound and
    ``support_margin`` the remaining per-row selection or envelope trust; the
    support is certified only while every margin is strictly positive.
    """

    stencils: PreparedLocalStencils
    status: Array
    accepted: Array
    displacement: Array
    support_margin: Array


def _safe_norm(values: Array) -> Array:
    """Euclidean norm with a finite zero-vector derivative."""
    squared = jnp.sum(values * values, axis=-1)
    positive = squared > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, squared, 1.0)), 0.0)


@cache
def _basis_exponents(dimension: int, degree: int) -> tuple[tuple[int, ...], ...]:
    """Constant followed by the total-degree monomial exponents (static).

    Host-only and trace-independent; the ordering is that of
    ``TotalDegreePolynomialFeatures`` (increasing degree, then combinations).
    """
    rows = [(0,) * dimension]
    for total in range(1, degree + 1):
        for axes in itertools.combinations_with_replacement(range(dimension), total):
            rows.append(tuple(axes.count(axis) for axis in range(dimension)))
    return tuple(rows)


def polynomial_design(coordinates: Array, degree: int) -> Array:
    """Traceable total-degree design, constant column first, in given offsets.

    Static integer powers keep every coordinate derivative finite at the
    stencil center, including the zero-exponent factors.
    """
    columns = []
    for exponent in _basis_exponents(coordinates.shape[-1], degree):
        column = jnp.ones(coordinates.shape[:-1], dtype=coordinates.dtype)
        for axis, power in enumerate(exponent):
            if power:
                column = column * coordinates[..., axis] ** power
        columns.append(column)
    return jnp.stack(columns, axis=-1)


def weighted_svd_factors(
    design: Array, weights: Array, valid: Array
) -> tuple[Array, Array, Array, Array]:
    """Traceable shared kernel; returns factors, rank, condition, minimum singular."""
    # Masked entries take the zero branch with a finite derivative.
    root = jnp.where(valid, jnp.sqrt(jnp.where(valid, weights, 1.0)), 0.0)
    return root_weighted_svd_factors(design, root)


def root_weighted_svd_factors(
    design: Array, root: Array
) -> tuple[Array, Array, Array, Array]:
    """``weighted_svd_factors`` from square-root weights (zero off the support).

    Compact kernels supply their square root directly, so a weight vanishing
    at its cutoff has finite coordinate derivatives of every admitted order.
    """
    weighted = root[..., None] * design
    squared = jnp.sum(weighted * weighted, axis=1)
    positive = squared > 0
    scale = jnp.where(positive, jnp.sqrt(jnp.where(positive, squared, 1.0)), 1.0)
    normalized = weighted / scale[:, None, :]
    # A^+ of the column-normalized weighted design is the requested factor;
    # (A^T)^+ transposed equals A^+, so the F x K transpose is pseudo-inverted.
    result = pseudoinverse(jnp.swapaxes(normalized, -1, -2), _LOCAL_PSEUDOINVERSE)
    rank = result.diagnostics.rank
    condition = jnp.where(
        rank == design.shape[2], result.diagnostics.condition_estimate, jnp.inf
    )
    singular = result.diagnostics.singular_values
    if singular is None:
        raise RuntimeError("Native SVD did not return singular-value evidence.")
    factors = jnp.swapaxes(result.value, -1, -2) * root[:, None, :] / scale[..., None]
    return factors, rank, condition, jnp.min(singular, axis=-1)


def _radial_derivative(offsets: Array, index: tuple[int, ...], power: int) -> Array:
    def radial(x: Array) -> Array:
        squared = jnp.sum((x - offsets) ** 2, axis=-1)
        safe = jnp.where(squared > 0, squared, 1.0)
        return jnp.where(squared > 0, safe ** (power / 2), 0.0)

    derivative = radial
    for axis, order in enumerate(index):
        direction = jnp.zeros((offsets.shape[-1],), dtype=offsets.dtype).at[axis].set(1.0)
        for _ in range(order):
            previous = derivative
            derivative = lambda x, previous=previous, direction=direction: jax.jvp(
                previous, (x,), (direction,)
            )[1]
    return derivative(jnp.zeros((offsets.shape[-1],), dtype=offsets.dtype))


def _gmls_weight(ratio: Array, kernel: StencilWeightKernel | None) -> Array:
    # The support radius exceeds the furthest neighbor, including that
    # neighbor with strictly positive weight in every admitted row.
    match kernel:
        case "wendland-c2":
            return jnp.maximum(1 - ratio / 1.1, 0) ** 4 * (4 * ratio / 1.1 + 1)
        case "inverse-square":
            return 1 / jnp.maximum(ratio, 0.25) ** 2
        case "wendland-c4" | "compact-polynomial" | None:
            raise RuntimeError(
                "A nearest-neighbor GMLS policy must own a relative weight kernel."
            )
        case _:
            assert_never(kernel)


def _compact_root(
    coordinates: Array, valid: Array, policy: LocalStencilPolicy
) -> tuple[Array, Array]:
    """Square-root compact weights at the fixed physical radius and their support.

    Each square root vanishes at the cutoff together with the kernel's
    admitted coordinate derivatives, and outside slots take the zero branch,
    so the fitted weights are continuous through ``coordinate_order`` while a
    candidate crosses the cutoff. The Wendland profiles depend on ``r``
    through a zero-safe norm (both are even through first order at ``r = 0``).
    """
    support = policy.support
    if support is None:
        raise RuntimeError("Compact weights require a smooth support envelope.")
    squared = jnp.sum(coordinates * coordinates, axis=-1) / support.radius**2
    inside = valid & (squared < 1.0)
    masked = jnp.where(inside, squared, 0.0)
    ratio = _safe_norm(jnp.where(inside[..., None], coordinates, 0.0)) / support.radius
    match policy.weight_kernel:
        case "wendland-c2":
            root = (1.0 - ratio) ** 2 * jnp.sqrt(4.0 * ratio + 1.0)
        case "wendland-c4":
            root = (1.0 - ratio) ** 3 * jnp.sqrt(35.0 * ratio**2 + 18.0 * ratio + 3.0)
        case "compact-polynomial":
            if policy.coordinate_order is None:
                raise RuntimeError(
                    "A smooth support policy must own its coordinate order."
                )
            root = (1.0 - masked) ** ((policy.coordinate_order + 1) / 2)
        case "inverse-square" | None:
            raise RuntimeError("A smooth support policy must own a compact kernel.")
        case _:
            assert_never(policy.weight_kernel)
    return jnp.where(inside, root, 0.0), inside


def _gmls_factors(
    coordinates: Array,
    valid: Array,
    radius: Array,
    scale: Array,
    design: Array,
    policy: LocalStencilPolicy,
    support_offsets: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """GMLS factors and evidence with the rows' positively weighted support.

    ``support_offsets`` own the compact weight distance: the coordinates whose
    distance also selected the candidate envelope (ambient on surfaces).
    """
    if policy.support is None:
        return (
            *weighted_svd_factors(
                design, _gmls_weight(radius / scale[:, None], policy.weight_kernel), valid
            ),
            valid,
        )
    root, inside = _compact_root(support_offsets, valid, policy)
    return (*root_weighted_svd_factors(design, root), inside)


def _phs_fit(
    x: Array,
    valid: Array,
    design: Array,
    moments: Array,
    radial_rhs: Array,
    power: int,
) -> tuple[Array, Array, Array, Array]:
    rows, capacity, _ = x.shape
    features = design.shape[2]
    radial = _safe_norm(x[:, :, None, :] - x[:, None, :, :]) ** power
    polynomial = jnp.where(valid[..., None], design, 0.0)
    radial = jnp.where(valid[:, :, None] & valid[:, None, :], radial, 0.0)
    radial = radial + jnp.eye(capacity, dtype=x.dtype)[None, :, :] * (~valid)[:, :, None]
    saddle = jnp.concatenate(
        (
            jnp.concatenate((radial, polynomial), axis=2),
            jnp.concatenate(
                (
                    jnp.swapaxes(polynomial, 1, 2),
                    jnp.zeros((rows, features, features), dtype=x.dtype),
                ),
                axis=2,
            ),
        ),
        axis=1,
    )
    rhs = jnp.swapaxes(
        jnp.concatenate((jnp.where(valid[:, None, :], radial_rhs, 0.0), moments), axis=2),
        1,
        2,
    )
    # The saddle pseudoinverse is materialized (size capacity + features per
    # row) for its exact fixed-rank derivative; it is applied once to the
    # functionals' right-hand sides.
    result = pseudoinverse(saddle, _LOCAL_PSEUDOINVERSE)
    singular = result.diagnostics.singular_values
    if singular is None:
        raise RuntimeError("Native saddle SVD omitted singular-value evidence.")
    return (
        jnp.swapaxes((result.value @ rhs)[:, :capacity, :], 1, 2),
        result.diagnostics.rank,
        result.diagnostics.condition_estimate,
        jnp.min(singular, axis=-1),
    )


def chart_stencil_kernel(
    coordinates: Array,
    valid: Array,
    multi_indices: StencilTerms,
    coefficients: Array,
    policy: LocalStencilPolicy,
    *,
    support_offsets: Array | None = None,
) -> tuple[Array, LocalStencilEvidence]:
    """Pure device fixed-support fit; coefficient shape (rows, functionals, terms).

    Returns masked weights of shape (rows, functionals, neighbors) and honest
    pre-masking evidence. Admission/refusal belongs to the host preparation.
    A smooth support measures its compact weight in ``support_offsets`` (rows,
    neighbors, any dimension) when given, otherwise in the chart coordinates:
    a surface chart passes its ambient offsets so the weight distance is the
    distance that selected the ambient candidate envelope (chart distance never
    exceeds it, so chart weights could admit a source outside the envelope).
    """
    rows, capacity, dimension = coordinates.shape
    if (
        valid.shape != (rows, capacity)
        or coefficients.ndim != 3
        or coefficients.shape[0] != rows
        or coefficients.shape[2] != len(multi_indices)
        or (
            support_offsets is not None
            and (
                support_offsets.ndim != 3 or support_offsets.shape[:2] != (rows, capacity)
            )
        )
    ):
        raise ValueError(
            "Chart kernel shapes must match rows, neighbors and derivative terms."
        )
    exponents = _basis_exponents(dimension, policy.polynomial_degree)
    features = len(exponents)
    if not multi_indices or any(
        len(index) != dimension
        or any(order < 0 for order in index)
        or sum(index) > policy.polynomial_degree
        for index in multi_indices
    ):
        raise ValueError("Chart derivative terms must belong to the polynomial basis.")
    # The fit runs in the dtype of its offsets (the fit precision role).
    coefficients = coefficients.astype(coordinates.dtype)
    roundoff = float(jnp.finfo(coordinates.dtype).eps)
    condition_limit = min(policy.condition_limit, _FIT_FORWARD_ERROR_LIMIT / roundoff)
    power = policy.phs_power
    if power is not None and any(sum(index) >= power for index in multi_indices):
        raise ValueError("PHS power must exceed derivative order.")
    radius = _safe_norm(coordinates)
    if policy.support is None:
        scale = jnp.max(jnp.where(valid, radius, 0.0), axis=1)
        scale = jnp.where(scale > 0, scale, 1.0)
    else:
        # The fixed physical radius: no data-dependent (max-distance) scale
        # whose argmax could switch along a trajectory.
        scale = jnp.full((rows,), policy.support.radius, dtype=coordinates.dtype)
    x = coordinates / scale[:, None, None]
    design = polynomial_design(x, policy.polynomial_degree)
    moments = jnp.zeros((rows, coefficients.shape[1], features), dtype=x.dtype)
    radial_rhs = jnp.zeros((rows, coefficients.shape[1], capacity), dtype=x.dtype)
    for term, index in enumerate(multi_indices):
        factor = coefficients[:, :, term] / scale[:, None] ** sum(index)
        moments = moments.at[:, :, exponents.index(index)].add(
            factor * prod(factorial(v) for v in index)
        )
        if power is not None:
            radial_rhs = (
                radial_rhs
                + factor[:, :, None]
                * jax.vmap(lambda row: _radial_derivative(row, index, power))(x)[
                    :, None, :
                ]
            )
    match policy.approximation:
        case "gmls":
            factors, rank, condition, minimum, populated = _gmls_factors(
                coordinates,
                valid,
                radius,
                scale,
                design,
                policy,
                coordinates
                if support_offsets is None
                else support_offsets.astype(coordinates.dtype),
            )
            weights = moments @ factors
            if roundoff > _FLOAT64_ROUNDOFF:
                # A reduced-precision local solve reproduces the moments only
                # to condition * u; one refinement step with the same factors
                # restores rounding-level reproduction (condition * u < 1e-3 is
                # the admission). Float64 fits keep their unrefined route.
                weights = weights + (moments - weights @ design) @ factors
            expected_rank = features
        case "phs-rbf-fd":
            if power is None:
                raise RuntimeError("A PHS policy must own its radial power.")
            weights, rank, condition, minimum = _phs_fit(
                x, valid, design, moments, radial_rhs, power
            )
            expected_rank = capacity + features
            populated = valid
        case _:
            assert_never(policy.approximation)
    condition = jnp.where(rank == expected_rank, condition, jnp.inf)
    weights = jnp.where(valid[:, None, :], weights, 0.0)
    moment_scale = jnp.maximum(jnp.max(jnp.abs(moments), axis=2), 1.0)
    residual = jnp.max(
        jnp.max(jnp.abs(weights @ design - moments), axis=2) / moment_scale,
        axis=1,
    )
    residual_limit = (
        jnp.full((rows,), _MOMENT_RESIDUAL_LIMIT, dtype=x.dtype)
        if roundoff <= _FLOAT64_ROUNDOFF
        else jnp.maximum(
            _MOMENT_RESIDUAL_LIMIT,
            2
            * (capacity + features)
            * roundoff
            * jnp.max(jnp.sum(jnp.abs(weights), axis=2) / moment_scale, axis=1),
        )
    )
    amplification = jnp.max(jnp.sum(jnp.abs(weights), axis=2), axis=1)
    status = jnp.where(
        jnp.sum(populated, axis=1) < features,
        int(MeshfreeRowStatus.UNDERSAMPLED),
        jnp.where(
            rank < expected_rank,
            int(MeshfreeRowStatus.RANK_DEFICIENT),
            jnp.where(
                ~jnp.isfinite(condition) | (condition > condition_limit),
                int(MeshfreeRowStatus.ILL_CONDITIONED),
                jnp.where(
                    ~jnp.isfinite(amplification)
                    | (amplification > policy.amplification_limit),
                    int(MeshfreeRowStatus.EXCESSIVE_AMPLIFICATION),
                    jnp.where(
                        ~jnp.isfinite(residual) | (residual > residual_limit),
                        int(MeshfreeRowStatus.MOMENT_FAILURE),
                        int(MeshfreeRowStatus.VALID),
                    ),
                ),
            ),
        ),
    )
    evidence = LocalStencilEvidence(
        status=status,
        condition=condition,
        rank=rank,
        moment_residual=residual,
        amplification=amplification,
        minimum_singular_value=minimum,
    )
    return jnp.where((status == 0)[:, None, None], weights, 0.0), evidence


def map_row_chunks[T](
    function: Callable[[tuple[Array, ...]], T],
    rows: tuple[Array, ...],
    chunk_rows: int,
) -> T:
    """Map ``function`` over padded fixed-size leading-axis chunks on device.

    Padding rows are zero (masks false) and are discarded; the per-chunk
    working set is bounded by ``chunk_rows`` independently of the row count.
    """
    count = rows[0].shape[0]
    chunk = min(chunk_rows, count)
    padding = (-count) % chunk

    def split(value: Array) -> Array:
        padded = jnp.pad(value, ((0, padding),) + ((0, 0),) * (value.ndim - 1))
        return padded.reshape((-1, chunk) + value.shape[1:])

    outputs = jax.lax.map(function, tuple(split(value) for value in rows))
    return jax.tree.map(
        lambda value: value.reshape((-1,) + value.shape[2:])[:count], outputs
    )


def fit_chart_stencils(
    offsets: Array,
    valid: Array,
    terms: StencilTerms,
    coefficients: Array,
    policy: LocalStencilPolicy,
    *,
    support_offsets: Array | None = None,
) -> tuple[Array, LocalStencilEvidence]:
    """Traceable chunked fit of (rows, neighbors, dimension) chart offsets.

    Returns weights of shape (rows, functionals, neighbors) and per-row
    evidence. Rows are fitted in padded chunks of ``policy.chunk_rows`` so the
    result is independent of the chunk capacity up to floating-point batching.
    ``support_offsets`` (see :func:`chart_stencil_kernel`) measure the smooth
    support's compact weight distance.
    """
    if support_offsets is None:

        def fit(batch: tuple[Array, ...]) -> tuple[Array, LocalStencilEvidence]:
            chunk_offsets, chunk_valid, chunk_coefficients = batch
            return chart_stencil_kernel(
                chunk_offsets, chunk_valid, terms, chunk_coefficients, policy
            )

        return map_row_chunks(fit, (offsets, valid, coefficients), policy.chunk_rows)

    def fit_supported(batch: tuple[Array, ...]) -> tuple[Array, LocalStencilEvidence]:
        chunk_offsets, chunk_valid, chunk_coefficients, chunk_support = batch
        return chart_stencil_kernel(
            chunk_offsets,
            chunk_valid,
            terms,
            chunk_coefficients,
            policy,
            support_offsets=chunk_support,
        )

    return map_row_chunks(
        fit_supported,
        (offsets, valid, coefficients, support_offsets),
        policy.chunk_rows,
    )


class _AdmissionSummary(NamedTuple):
    maximum_condition: Array
    minimum_singular: Array
    minimum_rank: Array
    maximum_residual: Array
    maximum_amplification: Array
    worst_row: Array
    refused_rows: Array
    first_refused_row: Array
    first_refused_status: Array
    first_refused_condition: Array
    first_refused_amplification: Array


def _admission_fit(
    offsets: Array,
    valid: Array,
    coefficients: Array,
    active: Array,
    terms: StencilTerms,
    policy: LocalStencilPolicy,
) -> tuple[Array, LocalStencilEvidence, _AdmissionSummary]:
    weights, evidence = fit_chart_stencils(offsets, valid, terms, coefficients, policy)
    # Storage padding rows are inactive: they never enter the admission report.
    refused = (evidence.status != 0) & active
    first = jnp.argmax(refused)
    condition = jnp.where(active, evidence.condition, -jnp.inf)
    return (
        weights,
        evidence,
        _AdmissionSummary(
            maximum_condition=jnp.max(condition),
            minimum_singular=jnp.min(
                jnp.where(active, evidence.minimum_singular_value, jnp.inf)
            ),
            minimum_rank=jnp.min(
                jnp.where(active, evidence.rank, jnp.iinfo(evidence.rank.dtype).max)
            ),
            maximum_residual=jnp.max(
                jnp.where(active, evidence.moment_residual, -jnp.inf)
            ),
            maximum_amplification=jnp.max(
                jnp.where(active, evidence.amplification, -jnp.inf)
            ),
            worst_row=jnp.argmax(condition),
            refused_rows=jnp.sum(refused, dtype=jnp.int32),
            first_refused_row=first,
            first_refused_status=evidence.status[first],
            first_refused_condition=evidence.condition[first],
            first_refused_amplification=evidence.amplification[first],
        ),
    )


# Stable module-level compiled entry point: bucketed storage shapes, terms and
# the static policy select the executable; coordinates and coefficients are
# dynamic arguments, so clouds and hierarchy levels reuse executables.
_compiled_admission_fit = eqx.filter_jit(_admission_fit)


def _functional_terms(
    functionals: tuple[MeshfreeFunctional, ...],
    rows: int,
    dimension: int,
    policy: LocalStencilPolicy,
) -> tuple[StencilTerms, np.ndarray]:
    """Validate functionals and assemble shared term coefficients once."""
    if not functionals or any(not isinstance(f, MeshfreeFunctional) for f in functionals):
        raise TypeError("functionals must be nonempty MeshfreeFunctional values.")
    for functional in functionals:
        if len(functional.multi_indices[0]) != dimension or any(
            sum(index) > policy.polynomial_degree for index in functional.multi_indices
        ):
            raise ValueError("Functional dimension/order must fit the polynomial basis.")
        if (
            functional.row_coefficients is not None
            and functional.row_coefficients.shape[0] != rows
        ):
            raise ValueError("Functional row coefficients must match target rows.")
        if policy.phs_power is not None and any(
            sum(index) >= policy.phs_power for index in functional.multi_indices
        ):
            raise ValueError(
                "PHS power must exceed derivative order for regularity at stencil centers."
            )
    terms = tuple(dict.fromkeys(index for f in functionals for index in f.multi_indices))
    coefficients = np.zeros((rows, len(functionals), len(terms)), dtype=np.float64)
    for f_index, functional in enumerate(functionals):
        row_values = (
            None
            if functional.row_coefficients is None
            else np.asarray(functional.row_coefficients)
        )
        for term, index in enumerate(functional.multi_indices):
            row = 1.0 if row_values is None else row_values[:, term]
            coefficients[:, f_index, terms.index(index)] += (
                functional.coefficients[term] * row
            )
    return terms, coefficients


def _prepare(
    neighborhood: PreparedMeshfreeNeighborhood,
    offsets: np.ndarray,
    functionals: tuple[MeshfreeFunctional, ...],
    policy: LocalStencilPolicy,
    anchor_sources: Array | None,
    anchor_targets: Array | None,
) -> PreparedLocalStencils:
    rows, _, dimension = offsets.shape
    support_id = None if policy.support is None else policy.support.envelope_id
    envelope_id = (
        None if neighborhood.envelope is None else neighborhood.envelope.envelope_id
    )
    if support_id != envelope_id:
        raise ValueError(
            "Smooth fixed-radius GMLS needs a candidate-envelope neighborhood of the "
            "same SmoothSupportEnvelope, and an envelope neighborhood needs that policy."
        )
    terms, coefficients = _functional_terms(functionals, rows, dimension, policy)
    precision = neighborhood.precision
    # Offsets and functional coefficients enter the fit in its precision role;
    # admitted weights are stored in the coefficient role.
    offsets = offsets.astype(precision.fit_dtype)
    coefficients = coefficients.astype(precision.fit_dtype)
    storage = bucketed_storage_capacity(rows)
    valid = np.asarray(jax.device_get(neighborhood.relation.valid))
    weights, evidence, summary = _compiled_admission_fit(
        jax.device_put(_pad_rows(offsets, storage)),
        jax.device_put(_pad_rows(valid, storage)),
        jax.device_put(_pad_rows(coefficients, storage)),
        jax.device_put(np.arange(storage) < rows),
        terms,
        policy,
    )
    # The single host synchronization: admission, report and identity. The
    # logical rows are cut from storage on the host.
    host_weights, host_evidence, host = jax.device_get((weights, evidence, summary))
    host_weights = host_weights[:rows].astype(precision.coefficient_dtype)
    host_evidence = jax.tree.map(lambda value: value[:rows], host_evidence)
    host_status = host_evidence.status
    refused = int(host.refused_rows)
    if refused and policy.acceptance == "refuse":
        raise ValueError(
            f"Meshfree stencil row {int(host.first_refused_row)} refused: {MeshfreeRowStatus(int(host.first_refused_status)).name} (condition={float(host.first_refused_condition):.6g}, amplification={float(host.first_refused_amplification):.6g})."
        )
    identifier = canonical_fingerprint(
        {
            "kind": "local-stencils",
            "neighborhood": neighborhood.neighborhood_id,
            "coordinates": array_tree_fingerprint(offsets),
            "weights": array_tree_fingerprint(
                tuple(
                    np.ascontiguousarray(host_weights[:, index, :])
                    for index in range(len(functionals))
                )
            ),
            "status": array_tree_fingerprint(host_status),
            "functionals": tuple(
                {
                    "name": f.name,
                    "indices": f.multi_indices,
                    "coefficients": f.coefficients,
                    "row_coefficients": None
                    if f.row_coefficients is None
                    else array_tree_fingerprint(np.asarray(f.row_coefficients)),
                }
                for f in functionals
            ),
            "policy": {
                "approximation": policy.approximation,
                "degree": policy.polynomial_degree,
                "power": policy.phs_power,
                "weight": policy.weight_kernel,
                "support": support_id,
                "coordinate_order": policy.coordinate_order,
                "condition_limit": policy.condition_limit,
                "amplification_limit": policy.amplification_limit,
                "acceptance": policy.acceptance,
            },
        }
    )
    observed: dict[MeshfreePrecisionRole, np.dtype] = {
        "fit": weights.dtype,
        "coefficient": host_weights.dtype,
        "certification": neighborhood.distances.dtype,
    }
    if anchor_sources is not None:
        observed["geometry"] = anchor_sources.dtype
    precision.evidence(observed)
    report = LocalStencilReport(
        maximum_condition_number=float(host.maximum_condition),
        minimum_singular_value=float(host.minimum_singular),
        minimum_rank=int(host.minimum_rank),
        maximum_moment_residual=float(host.maximum_residual),
        maximum_amplification=float(host.maximum_amplification),
        worst_row=int(host.worst_row),
        refused_rows=refused,
        precision=tuple(
            (role, np.dtype(observed[role]).name)
            for role, _ in precision.roles
            if role in observed
        ),
        report_id=identifier,
    )
    # ``device_put`` is a transfer; ``jnp.asarray`` of host data would compile
    # a staging program per shape.
    device_offsets = jax.device_put(offsets)
    return PreparedLocalStencils(
        neighborhood=neighborhood,
        functionals=tuple(functionals),
        policy=policy,
        coefficients=jax.device_put(coefficients),
        offsets=device_offsets,
        anchor_offsets=device_offsets,
        anchor_sources=anchor_sources,
        anchor_targets=anchor_targets,
        weights=tuple(
            jax.device_put(np.ascontiguousarray(host_weights[:, index, :]))
            for index in range(len(functionals))
        ),
        evidence=jax.device_put(host_evidence),
        report=report,
        terms=terms,
        prepared_id=identifier,
    )


def _pad_rows(values: np.ndarray, storage: int) -> np.ndarray:
    """Zero (invalid) storage rows after the logical rows."""
    padding = np.zeros((storage - values.shape[0],) + values.shape[1:], values.dtype)
    return np.concatenate((values, padding))


def prepare_local_stencils(
    neighborhood: PreparedMeshfreeNeighborhood,
    sources: ArrayLike,
    targets: ArrayLike,
    functionals: tuple[MeshfreeFunctional, ...],
    policy: LocalStencilPolicy,
) -> PreparedLocalStencils:
    """Admit point stencils; offsets use the neighborhood's minimum image."""
    if not isinstance(neighborhood, PreparedMeshfreeNeighborhood) or not isinstance(
        policy, LocalStencilPolicy
    ):
        raise TypeError("A prepared neighborhood and LocalStencilPolicy are required.")
    # Coordinates are held in the geometry role; offsets are formed in the
    # wider of geometry and fit before the fit narrows them.
    precision = neighborhood.precision
    source, target = (
        np.asarray(sources, dtype=precision.geometry_dtype),
        np.asarray(targets, dtype=precision.geometry_dtype),
    )
    wide = np.promote_types(precision.geometry_dtype, precision.fit_dtype)
    relation = neighborhood.relation
    if (
        source.ndim != 2
        or target.ndim != 2
        or source.shape[1] != target.shape[1]
        or source.shape[0] != relation.source_size
        or target.shape[0] != relation.targets_per_case
    ):
        raise ValueError("Sources/targets must match neighborhood dimensions and sizes.")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("Stencil coordinates must be finite.")
    anchor_sources, anchor_targets = jax.device_put((source, target))
    return _prepare(
        neighborhood,
        neighborhood.host_offsets(source.astype(wide), target.astype(wide)),
        functionals,
        policy,
        anchor_sources,
        anchor_targets,
    )


def prepare_chart_stencils(
    neighborhood: PreparedMeshfreeNeighborhood,
    coordinates: ArrayLike,
    functionals: tuple[MeshfreeFunctional, ...],
    policy: LocalStencilPolicy,
) -> PreparedLocalStencils:
    """Admit stencils from explicit (rows, neighbors, dimension) chart offsets."""
    if not isinstance(neighborhood, PreparedMeshfreeNeighborhood) or not isinstance(
        policy, LocalStencilPolicy
    ):
        raise TypeError("A prepared neighborhood and LocalStencilPolicy are required.")
    offsets = np.asarray(coordinates, dtype=np.float64)
    if (
        offsets.ndim != 3
        or offsets.shape[:2] != neighborhood.relation.valid.shape
        or not np.all(np.isfinite(offsets))
    ):
        raise ValueError(
            "coordinates must be finite (rows, neighbors, dimensions) offsets."
        )
    return _prepare(neighborhood, offsets, functionals, policy, None, None)


def _refit(
    stencils: PreparedLocalStencils,
    offsets: Array,
    displacement: Array,
    admissible: Array,
) -> LocalStencilRefresh:
    # Refits run in the fit role of the admitted anchor offsets and publish
    # weights in the admitted coefficient role.
    offsets = offsets.astype(stencils.anchor_offsets.dtype)
    weights, evidence = fit_chart_stencils(
        offsets,
        stencils.neighborhood.relation.valid,
        stencils.terms,
        stencils.coefficients,
        stencils.policy,
    )
    margin = stencils.neighborhood.trust_margin - displacement
    # Equality is not certified: the selection gap may close completely. The
    # unmoved anchor reproduces its admitted selection even at a tie.
    supported = jnp.all(
        (displacement < stencils.neighborhood.trust_margin) | (displacement == 0)
    )
    match stencils.policy.acceptance:
        case "refuse":
            rows_refused = jnp.any(evidence.status != 0)
        case "mask":
            rows_refused = jnp.asarray(False)
        case _:
            assert_never(stencils.policy.acceptance)
    status = jnp.where(
        ~admissible,
        int(LocalStencilRefreshStatus.INVALID_COORDINATES),
        jnp.where(
            ~supported,
            int(LocalStencilRefreshStatus.SUPPORT_EXCEEDED),
            jnp.where(
                rows_refused,
                int(LocalStencilRefreshStatus.ROW_REFUSED),
                int(LocalStencilRefreshStatus.ACCEPTED),
            ),
        ),
    ).astype(jnp.int32)
    accepted = status == int(LocalStencilRefreshStatus.ACCEPTED)
    # A refused candidate publishes NaN weights. The mask is multiplicative so
    # tangents and cotangents are NaN as well: a selected constant would carry
    # a plausible zero derivative through a support or rank failure.
    weights = weights.astype(stencils.weights[0].dtype)
    weights = weights * jnp.where(accepted, 1.0, jnp.nan).astype(weights.dtype)
    # The unmoved anchor of a zero-margin (tied) selection is admitted, but no
    # open coordinate neighborhood keeps its support: its derivative is NaN.
    weights = guard_derivative_validity(
        weights,
        jnp.all(displacement < stencils.neighborhood.trust_margin),
        dependencies=(offsets,),
    )
    refreshed = eqx.tree_at(
        lambda item: (item.offsets, item.weights, item.evidence),
        stencils,
        (
            offsets,
            tuple(weights[:, index, :] for index in range(len(stencils.functionals))),
            evidence,
        ),
    )
    return LocalStencilRefresh(
        stencils=refreshed,
        status=status,
        accepted=accepted,
        displacement=displacement,
        support_margin=margin,
    )


def refresh_local_stencils(
    stencils: PreparedLocalStencils, sources: ArrayLike, targets: ArrayLike
) -> LocalStencilRefresh:
    """Refit point stencils at moved coordinates on the frozen relation.

    Traceable and differentiable in ``sources``/``targets``: JVP and VJP are
    derivatives of the published fixed-support weight map, with per-row rank
    and conditioning evidence. The support is certified when the largest
    anchored point displacement is strictly below every row's trust margin
    (the nearest-neighbor selection gap, or the declared smooth-support
    envelope); otherwise the status is ``SUPPORT_EXCEEDED`` and a new epoch
    must be prepared. Every refused refresh publishes NaN weights, so its
    derivatives are NaN rather than a silent masked zero.
    """
    if not isinstance(stencils, PreparedLocalStencils):
        raise TypeError("stencils must be PreparedLocalStencils.")
    anchor_sources, anchor_targets = stencils.anchor_sources, stencils.anchor_targets
    if anchor_sources is None or anchor_targets is None:
        raise ValueError(
            "Chart-prepared stencils have no anchored points; use refresh_chart_stencils."
        )
    source = jnp.asarray(sources, dtype=anchor_sources.dtype)
    target = jnp.asarray(targets, dtype=anchor_targets.dtype)
    if source.shape != anchor_sources.shape or target.shape != anchor_targets.shape:
        raise ValueError("Refresh must preserve anchored source and target shapes.")
    neighborhood = stencils.neighborhood
    moved = jnp.concatenate(
        (
            neighborhood.minimum_image(source - anchor_sources),
            neighborhood.minimum_image(target - anchor_targets),
        )
    )
    # Support certification is a discrete decision, not differentiated data.
    displacement = jax.lax.stop_gradient(jnp.max(jnp.linalg.norm(moved, axis=-1)))
    admissible = jnp.all(jnp.isfinite(source)) & jnp.all(jnp.isfinite(target))
    safe_source = jnp.where(jnp.isfinite(source), source, anchor_sources)
    safe_target = jnp.where(jnp.isfinite(target), target, anchor_targets)
    return _refit(
        stencils,
        neighborhood.offsets(safe_source, safe_target),
        jnp.where(admissible, displacement, jnp.inf),
        admissible,
    )


def refresh_chart_stencils(
    stencils: PreparedLocalStencils, offsets: ArrayLike, *, displacement: ArrayLike
) -> LocalStencilRefresh:
    """Refit chart stencils at owner-supplied offsets on the frozen relation.

    ``displacement`` is the chart owner's bound on anchored point motion in the
    neighborhood's coordinates; it certifies support against the trust margin.
    """
    if not isinstance(stencils, PreparedLocalStencils):
        raise TypeError("stencils must be PreparedLocalStencils.")
    chart = jnp.asarray(offsets, dtype=stencils.anchor_offsets.dtype)
    bound = jnp.asarray(displacement, dtype=chart.dtype)
    if chart.shape != stencils.anchor_offsets.shape or bound.shape != ():
        raise ValueError(
            "Chart refresh preserves offset shape and takes a scalar displacement."
        )
    bound = jax.lax.stop_gradient(bound)
    admissible = jnp.all(jnp.isfinite(chart)) & jnp.isfinite(bound) & (bound >= 0)
    return _refit(
        stencils,
        jnp.where(jnp.isfinite(chart), chart, stencils.anchor_offsets),
        jnp.where(admissible, bound, jnp.inf),
        admissible,
    )
