#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..._sampling._addressing import derive_key, SampleAddress
from ..._strict import StrictModule
from ...linalg._operators import DenseLinearOperator
from ...linalg._policies import FailurePolicy, RankPolicy
from ...linalg._spaces import ArraySpace
from ...linalg._svd import svd
from ...linalg._svd_contracts import (
    DenseSVD,
    RandomizedSVD,
    SVDApproximationPolicy,
    SVDDifferentiationMode,
    SVDLeadingEvidence,
    SVDProblem,
    SVDRangeEvidence,
    SVDRankEvidence,
    SVDResourcePolicy,
    SVDSolvePolicy,
    SVDSolveStatus,
    SVDTolerancePolicy,
)
from ...typing import parse, PRNGKey
from .._contracts import (
    ML_INFEASIBLE,
    ML_INSUFFICIENT_DATA,
    ML_NONCONVERGED,
    ML_NONFINITE,
    ML_SUCCESS,
)


SubspaceGradientTarget: TypeAlias = Literal["projector", "basis", "none"]
_CASE_ADDRESS = SampleAddress(
    "ml", "weighted-subspace", target="case", role="native-root"
)


def parse_subspace_differentiation(
    value: SubspaceGradientTarget, /
) -> SubspaceGradientTarget:
    """Admit the meaningful ML subset of the native derivative modes."""
    mode = parse(value, SVDDifferentiationMode, "differentiate")
    if mode == "singular-values":
        raise ValueError(
            "Subspace fits require none, projector, or basis differentiation."
        )
    return parse(mode, SubspaceGradientTarget, "differentiate")


@final
class SpectralFitResult(StrictModule):
    """Native weighted fit with original-action energy and compact responses."""

    offset: Array
    components: Array
    singular_values: Array
    explained_energy: Array
    retained_energy: Array
    residual_energy: Array
    rank_evidence: SVDRankEvidence
    range_evidence: SVDRangeEvidence
    leading_evidence: SVDLeadingEvidence
    orthogonality_error: Array
    minimum_retained_gap: Array
    cutoff_gap: Array
    internal_gap: Array
    pivot_magnitude: Array
    pivot_margin: Array
    frame_correction: Array
    covariance_correction: Array
    projector_gradient_supported: Array
    basis_gradient_supported: Array
    native_status: Array
    derivative_status: Array
    admission_status: Array
    leading_certified: Array
    valid: Array
    status: Array
    centered: bool = eqx.field(static=True)
    method: str = eqx.field(static=True)


def _projection_residual_energy(weighted: Array, frame: Array, /) -> Array:
    """Measure the fitted projector with scratch bounded by retained capacity."""
    rows = weighted.shape[0]
    tile = min(frame.shape[1], rows, 32)
    steps = (rows + tile - 1) // tile

    def accumulate(index: int, total: Array) -> Array:
        indices = index * tile + jnp.arange(tile)
        block = jnp.take(weighted, indices, axis=0, mode="clip")
        residual = block - (block @ frame) @ frame.conj().T
        residual = jnp.where((indices < rows)[:, None], residual, 0)
        return total + jnp.sum(jnp.real(residual) ** 2 + jnp.imag(residual) ** 2)

    return jax.lax.fori_loop(
        0, steps, accumulate, jnp.asarray(0, dtype=weighted.real.dtype)
    )


def _fit_one(
    values: Array,
    weights: Array,
    admission: Array,
    key: PRNGKey | None,
    /,
    *,
    policy: SVDSolvePolicy,
    centered: bool,
) -> SpectralFitResult:
    finite_weights = jnp.isfinite(weights)
    nonnegative = weights >= 0.0
    active = finite_weights & nonnegative & (weights > 0.0)
    safe_weights = jnp.where(finite_weights & nonnegative, weights, 0.0)
    safe_values = jnp.where(active[:, None], values, 0)
    total_weight = jnp.sum(safe_weights)
    denominator = jnp.maximum(total_weight, jnp.finfo(weights.dtype).tiny)
    offset = jnp.sum(safe_weights[:, None] * safe_values, axis=0) / denominator
    if not centered:
        offset = jnp.zeros_like(offset)
    centered_values = jnp.where(active[:, None], safe_values - offset, 0)
    weighted = jnp.sqrt(safe_weights / denominator)[:, None] * centered_values
    finite = jnp.all(finite_weights) & jnp.all(jnp.isfinite(safe_values))
    feasible = jnp.all(nonnegative)
    enough = total_weight > 0.0
    admitted = finite & feasible & enough & admission
    admission_status = jnp.where(
        ~finite,
        ML_NONFINITE,
        jnp.where(
            ~feasible, ML_INFEASIBLE, jnp.where(enough, ML_SUCCESS, ML_INSUFFICIENT_DATA)
        ),
    ).astype(jnp.int32)
    # Failed admission reaches native's fixed-shape unavailable preparation path,
    # never a sanitized successful decomposition; preserve the owning cause above.
    operator_data = jnp.where(admitted, weighted, jnp.full_like(weighted, jnp.nan))
    operator = DenseLinearOperator(
        operator_data,
        source=ArraySpace(
            (operator_data.shape[1],),
            dtype=operator_data.dtype,
            space_id="ml-subspace-anonymous-features",
        ),
        target=ArraySpace(
            (operator_data.shape[0],),
            dtype=operator_data.dtype,
            space_id="ml-subspace-anonymous-samples",
        ),
    )
    native = svd(SVDProblem(operator), policy=policy, key=key)
    rows = jnp.swapaxes(jnp.conj(native.right_coordinates), -1, -2)
    projection_energy = native.diagnostics.projection_energies
    total_energy = jnp.sum(jnp.abs(weighted) ** 2)
    explained = jnp.where(total_energy > 0.0, projection_energy / total_energy, 0.0)
    residual = _projection_residual_energy(weighted, native.right_coordinates)
    retained = jnp.sum(explained)
    cutoff = native.diagnostics.cutoff_gap
    isolation = jnp.min(native.diagnostics.isolation_gaps, initial=jnp.inf)
    selected_energy = jax.lax.stop_gradient(native.singular_values**2)
    internal_gap = jnp.min(selected_energy[:-1] - selected_energy[1:], initial=jnp.inf)
    pivots = jnp.min(native.diagnostics.pivot_magnitudes, initial=jnp.inf)
    pivot_gap = jnp.min(native.diagnostics.pivot_gaps, initial=jnp.inf)
    rank_resolved = native.rank_evidence.available & (
        native.rank_evidence.lower_bound >= policy.count
    )
    # Fit quality and the requested derivative are distinct numerical contracts.
    valid = admitted & (native.primal_status == SVDSolveStatus.SUCCESS) & rank_resolved
    native_failure = jnp.where(
        (native.primal_status == SVDSolveStatus.NONFINITE_OUTPUT)
        | (native.primal_status == SVDSolveStatus.PREPARATION_FAILED),
        ML_NONFINITE,
        jnp.where(
            (native.primal_status == SVDSolveStatus.RANK_DEFICIENT) | ~rank_resolved,
            ML_INSUFFICIENT_DATA,
            ML_NONCONVERGED,
        ),
    )
    status = jnp.where(
        ~admitted,
        jnp.where(admission, admission_status, ML_INFEASIBLE),
        jnp.where(valid, ML_SUCCESS, native_failure),
    ).astype(jnp.int32)
    offset = jnp.where(admitted, offset, jnp.full_like(offset, jnp.nan))
    explained = jnp.where(admitted, explained, jnp.full_like(explained, jnp.nan))
    retained = jnp.where(admitted, retained, jnp.full_like(retained, jnp.nan))
    residual = jnp.where(admitted, residual, jnp.full_like(residual, jnp.nan))
    mode = policy.differentiation
    if mode in ("none", "projector"):
        rows = jax.lax.stop_gradient(rows)
        values_out = jax.lax.stop_gradient(native.singular_values)
        explained, retained, residual = jax.tree.map(
            jax.lax.stop_gradient, (explained, retained, residual)
        )
    else:
        values_out = native.singular_values
    if mode == "none":
        offset = jax.lax.stop_gradient(offset)
    return SpectralFitResult(
        offset=offset,
        components=rows,
        singular_values=values_out,
        explained_energy=explained,
        retained_energy=retained,
        residual_energy=residual,
        rank_evidence=native.rank_evidence,
        range_evidence=native.range_evidence,
        leading_evidence=native.leading_evidence,
        orthogonality_error=jnp.max(
            jnp.abs(rows @ jnp.conj(rows).T - jnp.eye(rows.shape[0], dtype=rows.dtype)),
            initial=0.0,
        ),
        minimum_retained_gap=isolation,
        cutoff_gap=cutoff,
        internal_gap=internal_gap,
        pivot_magnitude=pivots,
        pivot_margin=pivot_gap,
        frame_correction=jnp.swapaxes(
            jnp.conj(native.right_response.frame_correction), -1, -2
        ),
        covariance_correction=native.right_response.covariance_correction,
        projector_gradient_supported=valid
        & native.derivative_valid
        & (mode == "projector"),
        basis_gradient_supported=valid & native.derivative_valid & (mode == "basis"),
        native_status=native.primal_status,
        derivative_status=native.derivative_status,
        admission_status=admission_status,
        leading_certified=native.leading_evidence.certified,
        valid=valid,
        status=status,
        centered=centered,
        method="weighted-randomized-svd"
        if isinstance(policy.method, RandomizedSVD)
        else "weighted-svd",
    )


def fit_weighted_subspace(
    values: ArrayLike,
    weights: ArrayLike,
    /,
    *,
    rank: int,
    centered: bool = True,
    rcond: float | None = None,
    differentiate: SubspaceGradientTarget = "projector",
    method: DenseSVD | RandomizedSVD | None = None,
    tolerance: SVDTolerancePolicy | None = None,
    resources: SVDResourcePolicy | None = None,
    failure: FailurePolicy | None = None,
    key: PRNGKey | None = None,
    input_valid: ArrayLike | None = None,
) -> SpectralFitResult:
    """Fit normalized weighted arrays with bounded, case-addressed native solves."""
    x, w = jnp.asarray(values), jnp.asarray(weights, dtype=jnp.float64)
    if x.ndim < 2 or w.shape != x.shape[:-1]:
        raise ValueError("values and weights must end in (sample, feature) and sample.")
    available = min(x.shape[-2:])
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 < rank <= available:
        raise ValueError(f"rank must be an integer in [1, {available}].")
    mode = parse_subspace_differentiation(differentiate)
    method_ = DenseSVD() if method is None else method
    policy = SVDSolvePolicy(
        method_,
        count=rank,
        which="largest",
        tolerance=tolerance,
        rank=RankPolicy(relative_cutoff=rcond),
        resources=resources,
        differentiation=mode,
        failure=FailurePolicy("status") if failure is None else failure,
        approximation=SVDApproximationPolicy(
            require_leading=isinstance(method_, RandomizedSVD)
        ),
    )
    case_shape = tuple(x.shape[:-2])
    cases = 1
    for size in case_shape:
        cases *= size
    admitted = (
        jnp.ones(case_shape, dtype=jnp.bool_)
        if input_valid is None
        else jnp.broadcast_to(jnp.asarray(input_valid, dtype=jnp.bool_), case_shape)
    )
    if key is not None:
        parse(key, PRNGKey, "key")
    if isinstance(method_, RandomizedSVD) and key is None:
        raise ValueError("Randomized subspace fitting requires an explicit typed key.")
    if isinstance(method_, DenseSVD) and key is not None:
        raise ValueError("Dense subspace fitting does not consume a random key.")

    def fit_case(data: tuple[Array, Array, Array, Array]) -> SpectralFitResult:
        values_, weights_, valid_, index = data
        case_key = None if key is None else derive_key(key, _CASE_ADDRESS, index)
        return _fit_one(
            values_, weights_, valid_, case_key, policy=policy, centered=bool(centered)
        )

    outputs = jax.lax.map(
        fit_case,
        (
            x.reshape((cases,) + x.shape[-2:]),
            w.reshape((cases, w.shape[-1])),
            admitted.reshape((cases,)),
            jnp.arange(cases, dtype=jnp.uint32),
        ),
    )
    return jax.tree.map(
        lambda value: value.reshape(case_shape + value.shape[1:]), outputs
    )


__all__ = ["SpectralFitResult", "fit_weighted_subspace", "parse_subspace_differentiation"]
