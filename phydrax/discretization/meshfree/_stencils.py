# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Chunked native weighted-SVD GMLS and polynomially augmented PHS fits."""

from __future__ import annotations

from collections.abc import Sequence
from math import factorial
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._polynomial._total_degree import TotalDegreePolynomialFeatures
from ..._strict import StrictModule
from ...linalg import (
    DenseLinearOperator,
    DenseSVD,
    FailurePolicy,
    LeastSquaresProblem,
    LinearSolvePolicy,
    RankPolicy,
    RHSLayout,
    solve,
)
from ...typing import parse
from ._neighbors import _integer, PreparedMeshfreeNeighborhood
from ._types import (
    MeshfreeApproximation,
    MeshfreeRowStatus,
    StencilAcceptance,
    StencilWeightKernel,
)


@final
class LocalStencilPolicy(StrictModule):
    approximation: MeshfreeApproximation = eqx.field(static=True)
    polynomial_degree: int = eqx.field(static=True)
    phs_power: int = eqx.field(static=True)
    weight_kernel: StencilWeightKernel = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    amplification_limit: float = eqx.field(static=True)
    acceptance: StencilAcceptance = eqx.field(static=True)
    chunk_rows: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        approximation: MeshfreeApproximation = "gmls",
        polynomial_degree: int = 2,
        phs_power: int = 3,
        weight_kernel: StencilWeightKernel = "inverse-square",
        condition_limit: float = 1e8,
        amplification_limit: float = 1e8,
        acceptance: StencilAcceptance = "refuse",
        chunk_rows: int = 128,
    ) -> None:
        approximation_ = parse(approximation, MeshfreeApproximation, "approximation")
        degree = _integer(polynomial_degree, "polynomial_degree", 0)
        power = _integer(phs_power, "phs_power", 3)
        if power % 2 == 0:
            raise ValueError("phs_power must be odd.")
        if approximation_ == "phs-rbf-fd" and degree < (power - 1) // 2:
            raise ValueError(
                "PHS polynomial degree must cover its conditional-definiteness order."
            )
        kernel = parse(weight_kernel, StencilWeightKernel, "weight_kernel")
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
        self.condition_limit = float(condition_limit)
        self.amplification_limit = float(amplification_limit)
        self.acceptance = acceptance_
        self.chunk_rows = chunk


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
    maximum_condition_number: float = eqx.field(static=True)
    minimum_singular_value: float = eqx.field(static=True)
    minimum_rank: int = eqx.field(static=True)
    maximum_moment_residual: float = eqx.field(static=True)
    maximum_amplification: float = eqx.field(static=True)
    worst_row: int = eqx.field(static=True)
    refused_rows: int = eqx.field(static=True)
    report_id: str = eqx.field(static=True)


@final
class PreparedLocalStencils(StrictModule):
    neighborhood: PreparedMeshfreeNeighborhood
    functionals: tuple[MeshfreeFunctional, ...]
    weights: tuple[Array, ...]
    evidence: LocalStencilEvidence
    report: LocalStencilReport
    prepared_id: str = eqx.field(static=True)


def weighted_svd_factors(
    design: Array, weights: Array, valid: Array
) -> tuple[Array, Array, Array, Array]:
    """Traceable shared kernel; returns factors, rank, condition, minimum singular."""
    root = jnp.sqrt(jnp.where(valid, weights, 0.0))
    weighted = root[..., None] * design
    scale = jnp.sqrt(jnp.sum(weighted * weighted, axis=1))
    scale = jnp.where(scale > 0, scale, 1.0)
    normalized = weighted / scale[:, None, :]
    # (A^T)^+ transposed equals A^+. Solve only F identity columns instead of
    # K columns, using the native undamped SVD minimum-norm route.
    feature_count = design.shape[2]
    identity = jnp.broadcast_to(
        jnp.eye(feature_count, dtype=design.dtype),
        (design.shape[0], feature_count, feature_count),
    )
    result = solve(
        LeastSquaresProblem(DenseLinearOperator(jnp.swapaxes(normalized, -1, -2))),
        identity,
        policy=LinearSolvePolicy(
            DenseSVD(),
            rank=RankPolicy(relative_cutoff=1e-12, require_full_rank=True),
            failure=FailurePolicy("status"),
        ),
        rhs_layout=RHSLayout((feature_count,)),
    )
    rank = jnp.min(result.diagnostics.rank, axis=-1)
    condition = jnp.max(result.diagnostics.condition_estimate, axis=-1)
    condition = jnp.where(rank == design.shape[2], condition, jnp.inf)
    singular = result.diagnostics.singular_values
    if singular is None:
        raise RuntimeError("Native SVD did not return singular-value evidence.")
    factors = jnp.swapaxes(result.value, -1, -2) * root[:, None, :] / scale[..., None]
    return factors, rank, condition, jnp.min(singular, axis=-1)


def prepare_local_stencils(
    neighborhood: PreparedMeshfreeNeighborhood,
    sources: ArrayLike,
    targets: ArrayLike,
    functionals: tuple[MeshfreeFunctional, ...],
    policy: LocalStencilPolicy,
) -> PreparedLocalStencils:
    if not isinstance(neighborhood, PreparedMeshfreeNeighborhood):
        raise TypeError("neighborhood must be a PreparedMeshfreeNeighborhood.")
    source, target = (
        np.asarray(sources, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
    )
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
    return prepare_chart_stencils(
        neighborhood,
        source[np.asarray(relation.source_indices)] - target[:, None, :],
        functionals,
        policy,
    )


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


def chart_stencil_kernel(
    coordinates: Array,
    valid: Array,
    multi_indices: tuple[tuple[int, ...], ...],
    coefficients: Array,
    policy: LocalStencilPolicy,
) -> tuple[Array, LocalStencilEvidence]:
    """Pure device fixed-support fit; coefficient shape (rows, functionals, terms).

    Returns masked weights of shape (rows, functionals, neighbors) and honest
    pre-masking evidence. Admission/refusal belongs to the host preparation.
    """
    rows, capacity, dimension = coordinates.shape
    if (
        valid.shape != (rows, capacity)
        or coefficients.ndim != 3
        or coefficients.shape[0] != rows
        or coefficients.shape[2] != len(multi_indices)
    ):
        raise ValueError(
            "Chart kernel shapes must match rows, neighbors and derivative terms."
        )
    polynomial_features = TotalDegreePolynomialFeatures(
        dimension, policy.polynomial_degree
    )
    features = polynomial_features.feature_count + 1
    exponents = jnp.concatenate(
        (jnp.zeros((1, dimension), dtype=jnp.int32), polynomial_features.exponents),
        axis=0,
    )
    if not multi_indices or any(
        len(index) != dimension
        or any(order < 0 for order in index)
        or sum(index) > policy.polynomial_degree
        for index in multi_indices
    ):
        raise ValueError("Chart derivative terms must belong to the polynomial basis.")
    if policy.approximation == "phs-rbf-fd" and any(
        sum(index) >= policy.phs_power for index in multi_indices
    ):
        raise ValueError("PHS power must exceed derivative order.")
    radius = jnp.linalg.norm(coordinates, axis=-1)
    scale = jnp.max(jnp.where(valid, radius, 0.0), axis=1)
    scale = jnp.where(scale > 0, scale, 1.0)
    x = coordinates / scale[:, None, None]
    design = jnp.prod(x[:, :, None, :] ** exponents[None, None, :, :], axis=-1)
    moments = jnp.zeros((rows, coefficients.shape[1], features), dtype=x.dtype)
    radial_rhs = jnp.zeros((rows, coefficients.shape[1], capacity), dtype=x.dtype)
    for term, index in enumerate(multi_indices):
        factor = coefficients[:, :, term] / scale[:, None] ** sum(index)
        basis_moment = jnp.all(exponents == jnp.asarray(index)[None, :], axis=1).astype(
            x.dtype
        )
        basis_moment = basis_moment * np.prod([factorial(v) for v in index])
        moments = moments + factor[:, :, None] * basis_moment[None, None, :]
        if policy.approximation == "phs-rbf-fd":
            radial_rhs = (
                radial_rhs
                + factor[:, :, None]
                * jax.vmap(lambda row: _radial_derivative(row, index, policy.phs_power))(
                    x
                )[:, None, :]
            )
    if policy.approximation == "gmls":
        ratio = radius / scale[:, None]
        # The support radius exceeds the furthest neighbor, including that
        # neighbor with strictly positive weight in every admitted row.
        weight = (
            jnp.maximum(1 - ratio / 1.1, 0) ** 4 * (4 * ratio / 1.1 + 1)
            if policy.weight_kernel == "wendland-c2"
            else 1 / jnp.maximum(ratio, 0.25) ** 2
        )
        factors, rank, condition, minimum = weighted_svd_factors(design, weight, valid)
        weights = moments @ factors
        expected_rank = features
    else:
        radial = (
            jnp.linalg.norm(x[:, :, None, :] - x[:, None, :, :], axis=-1)
            ** policy.phs_power
        )
        polynomial = jnp.where(valid[..., None], design, 0.0)
        radial = jnp.where(valid[:, :, None] & valid[:, None, :], radial, 0.0)
        radial = radial + jnp.eye(capacity)[None, :, :] * (~valid)[:, :, None]
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
            jnp.concatenate(
                (jnp.where(valid[:, None, :], radial_rhs, 0.0), moments), axis=2
            ),
            1,
            2,
        )
        result = solve(
            LeastSquaresProblem(DenseLinearOperator(saddle)),
            rhs,
            policy=LinearSolvePolicy(
                DenseSVD(),
                rank=RankPolicy(relative_cutoff=1e-12, require_full_rank=True),
                failure=FailurePolicy("status"),
            ),
            rhs_layout=RHSLayout((coefficients.shape[1],)),
        )
        weights = jnp.swapaxes(result.value[:, :capacity, :], 1, 2)
        rank = jnp.min(result.diagnostics.rank, axis=-1)
        condition = jnp.max(result.diagnostics.condition_estimate, axis=-1)
        singular = result.diagnostics.singular_values
        if singular is None:
            raise RuntimeError("Native saddle SVD omitted singular-value evidence.")
        minimum = jnp.min(singular, axis=-1)
        expected_rank = capacity + features
    condition = jnp.where(rank == expected_rank, condition, jnp.inf)
    weights = jnp.where(valid[:, None, :], weights, 0.0)
    residual = jnp.max(
        jnp.max(jnp.abs(weights @ design - moments), axis=2)
        / jnp.maximum(jnp.max(jnp.abs(moments), axis=2), 1.0),
        axis=1,
    )
    amplification = jnp.max(jnp.sum(jnp.abs(weights), axis=2), axis=1)
    status = jnp.where(
        jnp.sum(valid, axis=1) < features,
        int(MeshfreeRowStatus.UNDERSAMPLED),
        jnp.where(
            rank < expected_rank,
            int(MeshfreeRowStatus.RANK_DEFICIENT),
            jnp.where(
                ~jnp.isfinite(condition) | (condition > policy.condition_limit),
                int(MeshfreeRowStatus.ILL_CONDITIONED),
                jnp.where(
                    ~jnp.isfinite(amplification)
                    | (amplification > policy.amplification_limit),
                    int(MeshfreeRowStatus.EXCESSIVE_AMPLIFICATION),
                    jnp.where(
                        ~jnp.isfinite(residual) | (residual > 1e-9),
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


def prepare_chart_stencils(
    neighborhood: PreparedMeshfreeNeighborhood,
    coordinates: ArrayLike,
    functionals: tuple[MeshfreeFunctional, ...],
    policy: LocalStencilPolicy,
) -> PreparedLocalStencils:
    if not isinstance(neighborhood, PreparedMeshfreeNeighborhood) or not isinstance(
        policy, LocalStencilPolicy
    ):
        raise TypeError("A prepared neighborhood and LocalStencilPolicy are required.")
    offsets = np.asarray(coordinates, dtype=np.float64)
    valid = np.asarray(neighborhood.relation.valid)
    if (
        offsets.ndim != 3
        or offsets.shape[:2] != valid.shape
        or not np.all(np.isfinite(offsets))
    ):
        raise ValueError(
            "coordinates must be finite (rows, neighbors, dimensions) offsets."
        )
    rows, capacity, dimension = offsets.shape
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
        if policy.approximation == "phs-rbf-fd" and any(
            sum(index) >= policy.phs_power for index in functional.multi_indices
        ):
            raise ValueError(
                "PHS power must exceed derivative order for regularity at stencil centers."
            )
    outputs = [[] for _ in functionals]
    statuses, conditions, ranks, residuals, amplifications, minima = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for start in range(0, rows, policy.chunk_rows):
        stop = min(start + policy.chunk_rows, rows)
        terms = tuple(
            dict.fromkeys(index for f in functionals for index in f.multi_indices)
        )
        coefficients = np.zeros((stop - start, len(functionals), len(terms)))
        for f_index, functional in enumerate(functionals):
            for term, index in enumerate(functional.multi_indices):
                value = functional.coefficients[term]
                row = (
                    1.0
                    if functional.row_coefficients is None
                    else np.asarray(functional.row_coefficients)[start:stop, term]
                )
                coefficients[:, f_index, terms.index(index)] += value * row
        weights, evidence = chart_stencil_kernel(
            jnp.asarray(offsets[start:stop]),
            jnp.asarray(valid[start:stop]),
            terms,
            jnp.asarray(coefficients),
            policy,
        )
        for f_index, output in enumerate(outputs):
            output.append(weights[:, f_index, :])
        for target, value in zip(
            (statuses, conditions, ranks, residuals, amplifications, minima),
            (
                evidence.status,
                evidence.condition,
                evidence.rank,
                evidence.moment_residual,
                evidence.amplification,
                evidence.minimum_singular_value,
            ),
            strict=True,
        ):
            target.append(np.asarray(value).reshape((stop - start,)))
    status, condition, rank, residual, amplification, minimum = (
        np.concatenate(values)
        for values in (statuses, conditions, ranks, residuals, amplifications, minima)
    )
    refused = int(np.count_nonzero(status))
    if refused and policy.acceptance == "refuse":
        row = int(np.flatnonzero(status)[0])
        raise ValueError(
            f"Meshfree stencil row {row} refused: {MeshfreeRowStatus(int(status[row])).name} (condition={condition[row]:.6g}, amplification={amplification[row]:.6g})."
        )
    prepared_weights = tuple(jnp.concatenate(output) for output in outputs)
    identifier = canonical_fingerprint(
        {
            "kind": "local-stencils",
            "neighborhood": neighborhood.neighborhood_id,
            "coordinates": array_tree_fingerprint(offsets),
            "weights": array_tree_fingerprint(
                tuple(np.asarray(weight) for weight in prepared_weights)
            ),
            "status": array_tree_fingerprint(status),
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
                "condition_limit": policy.condition_limit,
                "amplification_limit": policy.amplification_limit,
                "acceptance": policy.acceptance,
            },
        }
    )
    report = LocalStencilReport(
        maximum_condition_number=float(np.max(condition)),
        minimum_singular_value=float(np.min(minimum)),
        minimum_rank=int(np.min(rank)),
        maximum_moment_residual=float(np.max(residual)),
        maximum_amplification=float(np.max(amplification)),
        worst_row=int(np.argmax(condition)),
        refused_rows=refused,
        report_id=identifier,
    )
    evidence = LocalStencilEvidence(
        status=jnp.asarray(status),
        condition=jnp.asarray(condition),
        rank=jnp.asarray(rank),
        moment_residual=jnp.asarray(residual),
        amplification=jnp.asarray(amplification),
        minimum_singular_value=jnp.asarray(minimum),
    )
    return PreparedLocalStencils(
        neighborhood=neighborhood,
        functionals=tuple(functionals),
        weights=prepared_weights,
        evidence=evidence,
        report=report,
        prepared_id=identifier,
    )
