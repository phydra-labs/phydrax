#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    continuous_lyapunov_equation,
    HermitianSpectrum,
    MatrixEquationPlan,
    MatrixEquationPolicy,
    MatrixEquationResult,
    solve_matrix_equation,
)


def _adjoint(value: Array, /) -> Array:
    return jnp.swapaxes(jnp.conj(value), -1, -2)


def _matrix_defect(value: Array, /) -> Array:
    return jnp.max(jnp.abs(value - _adjoint(value)), initial=0.0)


class SecondCumulantLayout(StrictModule, NonTrainableState):
    """Ordered first/second-cumulant coordinates for one physical state.

    The first cumulant occupies ``mean_indices`` and the covariance is only over
    ``eddy_indices``.  This is a statistical-dynamics layout, deliberately
    independent of every UQ covariance representation.
    """

    mean_indices: Array
    eddy_indices: Array
    state_size: int = eqx.field(static=True)
    mean_dimension: int = eqx.field(static=True)
    eddy_dimension: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        state_size: int,
        mean_indices: Sequence[int] | ArrayLike,
        /,
        *,
        eddy_indices: Sequence[int] | ArrayLike | None = None,
        layout_id: str | None = None,
    ):
        size = int(state_size)
        if size < 1:
            raise ValueError("state_size must be positive.")
        mean = np.asarray(mean_indices, dtype=np.int64).reshape((-1,))
        if eddy_indices is None:
            selected = np.zeros((size,), dtype=bool)
            if np.any(mean < 0) or np.any(mean >= size):
                raise ValueError("mean_indices must address the physical state.")
            selected[mean] = True
            eddy = np.flatnonzero(~selected)
        else:
            eddy = np.asarray(eddy_indices, dtype=np.int64).reshape((-1,))
        if (
            mean.size < 1
            or eddy.size < 1
            or np.unique(mean).size != mean.size
            or np.unique(eddy).size != eddy.size
            or np.any(mean < 0)
            or np.any(mean >= size)
            or np.any(eddy < 0)
            or np.any(eddy >= size)
            or np.intersect1d(mean, eddy).size
            or not np.array_equal(np.sort(np.concatenate((mean, eddy))), np.arange(size))
        ):
            raise ValueError(
                "Mean and eddy indices must be non-empty, disjoint, unique, and cover state_size."
            )
        payload = {
            "kind": "second-cumulant-layout",
            "state_size": size,
            "mean_indices": mean.tolist(),
            "eddy_indices": eddy.tolist(),
            "coordinate_semantics": "physical-statistical-not-uq",
        }
        identifier = (
            canonical_fingerprint(payload) if layout_id is None else str(layout_id)
        )
        if not identifier:
            raise ValueError("layout_id must be non-empty.")
        self.mean_indices = jnp.asarray(mean, dtype=jnp.int32)
        self.eddy_indices = jnp.asarray(eddy, dtype=jnp.int32)
        self.state_size = size
        self.mean_dimension = int(mean.size)
        self.eddy_dimension = int(eddy.size)
        self.layout_id = identifier

    @classmethod
    def from_partition(
        cls,
        partition,
        /,
        *,
        layout_id: str | None = None,
    ) -> "SecondCumulantLayout":
        from ._interactions import InteractionPartition

        if not isinstance(partition, InteractionPartition):
            raise TypeError("partition must be an InteractionPartition.")
        low = np.asarray(partition.low_mask).reshape((-1,))
        high = np.asarray(partition.high_mask).reshape((-1,))
        admissible = low | high
        compressed_low = low[admissible]
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "partitioned-second-cumulant-layout",
                    "partition": partition.partition_id,
                }
            )
            if layout_id is None
            else layout_id
        )
        return cls(
            int(np.count_nonzero(admissible)),
            np.flatnonzero(compressed_low),
            eddy_indices=np.flatnonzero(~compressed_low),
            layout_id=identifier,
        )

    def validate_mean(self, mean: ArrayLike, /) -> Array:
        value = jnp.asarray(mean)
        if value.shape != (self.mean_dimension,):
            raise ValueError(
                f"First cumulant must have shape {(self.mean_dimension,)}; got {value.shape}."
            )
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            raise TypeError("First cumulant coordinates must be inexact.")
        return value

    def validate_eddy(self, eddy: ArrayLike, /) -> Array:
        value = jnp.asarray(eddy)
        if value.shape != (self.eddy_dimension,):
            raise ValueError(
                f"Eddy coordinates must have shape {(self.eddy_dimension,)}; got {value.shape}."
            )
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            raise TypeError("Eddy coordinates must be inexact.")
        return value

    def embed_mean(self, mean: ArrayLike, /) -> Array:
        value = self.validate_mean(mean)
        return (
            jnp.zeros((self.state_size,), dtype=value.dtype)
            .at[self.mean_indices]
            .set(value)
        )

    def embed_eddy(self, eddy: ArrayLike, /) -> Array:
        value = self.validate_eddy(eddy)
        return (
            jnp.zeros((self.state_size,), dtype=value.dtype)
            .at[self.eddy_indices]
            .set(value)
        )

    def restrict_mean(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != (self.state_size,):
            raise ValueError("Physical state has an incompatible coordinate shape.")
        return value[self.mean_indices]

    def restrict_eddy(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != (self.state_size,):
            raise ValueError("Physical state has an incompatible coordinate shape.")
        return value[self.eddy_indices]

    def embed_covariance(self, covariance: ArrayLike, /) -> Array:
        value = jnp.asarray(covariance)
        expected = (self.eddy_dimension, self.eddy_dimension)
        if value.shape != expected:
            raise ValueError(
                f"Second cumulant must have shape {expected}; got {value.shape}."
            )
        result = jnp.zeros((self.state_size, self.state_size), dtype=value.dtype)
        return result.at[jnp.ix_(self.eddy_indices, self.eddy_indices)].set(value)


class CumulantStateEvidence(StrictModule):
    hermitian_defect: Array
    minimum_eigenvalue: Array
    numerical_rank: Array
    finite: Array
    hermitian: Array
    positive_semidefinite: Array
    rank_allowed: Array
    successful: Array
    layout_id: str = eqx.field(static=True)


class DenseCumulantState(StrictModule):
    mean: Array
    covariance: Array
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        mean: ArrayLike,
        covariance: ArrayLike,
        /,
        *,
        layout_id: str,
    ):
        mean_ = jnp.asarray(mean)
        covariance_ = jnp.asarray(covariance)
        identifier = str(layout_id)
        if (
            mean_.ndim != 1
            or covariance_.ndim != 2
            or covariance_.shape[0] != covariance_.shape[1]
        ):
            raise ValueError(
                "Dense cumulants require a vector mean and square covariance."
            )
        if not jnp.issubdtype(mean_.dtype, jnp.inexact) or not jnp.issubdtype(
            covariance_.dtype, jnp.inexact
        ):
            raise TypeError("Cumulant arrays must use inexact dtypes.")
        if not identifier:
            raise ValueError("layout_id must be non-empty.")
        self.mean = mean_
        self.covariance = covariance_
        self.layout_id = identifier


class FactorCumulantState(StrictModule):
    mean: Array
    factor: Array
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        mean: ArrayLike,
        factor: ArrayLike,
        /,
        *,
        layout_id: str,
    ):
        mean_ = jnp.asarray(mean)
        factor_ = jnp.asarray(factor)
        identifier = str(layout_id)
        if mean_.ndim != 1 or factor_.ndim != 2:
            raise ValueError("Factor cumulants require a vector mean and matrix factor.")
        if not jnp.issubdtype(mean_.dtype, jnp.inexact) or not jnp.issubdtype(
            factor_.dtype, jnp.inexact
        ):
            raise TypeError("Cumulant arrays must use inexact dtypes.")
        if not identifier:
            raise ValueError("layout_id must be non-empty.")
        self.mean = mean_
        self.factor = factor_
        self.layout_id = identifier

    @property
    def rank(self) -> int:
        return int(self.factor.shape[1])

    @property
    def covariance(self) -> Array:
        return oe.contract("ir,jr->ij", self.factor, jnp.conj(self.factor))


CumulantState = DenseCumulantState | FactorCumulantState


def state_evidence(
    layout: SecondCumulantLayout,
    state: CumulantState,
    /,
    *,
    hermitian_tolerance: float = 1.0e-10,
    psd_tolerance: float = 1.0e-10,
    maximum_rank: int | None = None,
) -> CumulantStateEvidence:
    if not isinstance(layout, SecondCumulantLayout):
        raise TypeError("layout must be a SecondCumulantLayout.")
    if not isinstance(state, (DenseCumulantState, FactorCumulantState)):
        raise TypeError("state must be a dense or factor cumulant state.")
    if state.layout_id != layout.layout_id:
        raise ValueError("Cumulant state belongs to another layout.")
    layout.validate_mean(state.mean)
    covariance = state.covariance
    expected = (layout.eddy_dimension, layout.eddy_dimension)
    if covariance.shape != expected:
        raise ValueError(
            f"Second cumulant must have shape {expected}; got {covariance.shape}."
        )
    hermitian_tol = float(hermitian_tolerance)
    psd_tol = float(psd_tolerance)
    rank_limit = layout.eddy_dimension if maximum_rank is None else int(maximum_rank)
    if (
        hermitian_tol < 0.0
        or psd_tol < 0.0
        or rank_limit < 0
        or rank_limit > layout.eddy_dimension
    ):
        raise ValueError("Cumulant evidence tolerances or rank limit are invalid.")
    spectrum = HermitianSpectrum(covariance, tolerance=hermitian_tol)
    defect = _matrix_defect(covariance)
    scale = jnp.maximum(jnp.max(jnp.abs(spectrum.eigenvalues), initial=0.0), 1.0)
    threshold = psd_tol * scale
    finite = jnp.all(jnp.isfinite(state.mean)) & jnp.all(jnp.isfinite(covariance))
    hermitian = spectrum.valid & (defect <= hermitian_tol)
    psd = spectrum.minimum_eigenvalue >= -threshold
    rank = jnp.sum(spectrum.eigenvalues > threshold)
    represented_rank = state.rank if isinstance(state, FactorCumulantState) else rank
    rank_allowed = represented_rank <= rank_limit
    successful = finite & hermitian & psd & rank_allowed
    return CumulantStateEvidence(
        hermitian_defect=defect,
        minimum_eigenvalue=spectrum.minimum_eigenvalue,
        numerical_rank=rank,
        finite=finite,
        hermitian=hermitian,
        positive_semidefinite=psd,
        rank_allowed=jnp.asarray(rank_allowed),
        successful=successful,
        layout_id=layout.layout_id,
    )


def require_valid_state(
    layout: SecondCumulantLayout,
    state: CumulantState,
    /,
    *,
    hermitian_tolerance: float = 1.0e-10,
    psd_tolerance: float = 1.0e-10,
    maximum_rank: int | None = None,
) -> CumulantStateEvidence:
    evidence = state_evidence(
        layout,
        state,
        hermitian_tolerance=hermitian_tolerance,
        psd_tolerance=psd_tolerance,
        maximum_rank=maximum_rank,
    )
    if not bool(np.asarray(evidence.finite)):
        raise ValueError("Cumulant state must be finite.")
    if not bool(np.asarray(evidence.hermitian)):
        raise ValueError("Second cumulant violates the Hermitian gate.")
    if not bool(np.asarray(evidence.positive_semidefinite)):
        raise ValueError("Second cumulant violates the positive-semidefinite gate.")
    if not bool(np.asarray(evidence.rank_allowed)):
        raise ValueError("Second cumulant violates the configured rank gate.")
    return evidence


class ForcingCovariance(StrictModule, NonTrainableState):
    covariance: Array
    hermitian_defect: Array
    minimum_eigenvalue: Array
    dimension: int = eqx.field(static=True)
    covariance_id: str = eqx.field(static=True)

    def __init__(
        self,
        covariance: ArrayLike,
        /,
        *,
        covariance_id: str | None = None,
        hermitian_tolerance: float = 1.0e-10,
        psd_tolerance: float = 1.0e-10,
    ):
        value = jnp.asarray(covariance)
        if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] < 1:
            raise ValueError("Forcing covariance must be a non-empty square matrix.")
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            raise TypeError("Forcing covariance must use an inexact dtype.")
        hermitian_tol = float(hermitian_tolerance)
        psd_tol = float(psd_tolerance)
        if hermitian_tol < 0.0 or psd_tol < 0.0:
            raise ValueError("Forcing covariance tolerances must be non-negative.")
        defect = _matrix_defect(value)
        spectrum = HermitianSpectrum(value, tolerance=hermitian_tol)
        scale = jnp.maximum(jnp.max(jnp.abs(spectrum.eigenvalues), initial=0.0), 1.0)
        if not bool(np.asarray(jnp.all(jnp.isfinite(value)))):
            raise ValueError("Forcing covariance must be finite.")
        if not bool(np.asarray(spectrum.valid & (defect <= hermitian_tol))):
            raise ValueError("Forcing covariance violates the Hermitian gate.")
        if not bool(np.asarray(spectrum.minimum_eigenvalue >= -psd_tol * scale)):
            raise ValueError("Forcing covariance violates the PSD gate.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "statistical-dynamics-forcing-covariance",
                    "array": array_tree_fingerprint(value),
                }
            )
            if covariance_id is None
            else str(covariance_id)
        )
        if not identifier:
            raise ValueError("covariance_id must be non-empty.")
        self.covariance = value
        self.hermitian_defect = defect
        self.minimum_eigenvalue = spectrum.minimum_eigenvalue
        self.dimension = int(value.shape[0])
        self.covariance_id = identifier

    @classmethod
    def from_factor(
        cls,
        factor: ArrayLike,
        /,
        *,
        covariance_id: str | None = None,
    ) -> "ForcingCovariance":
        value = jnp.asarray(factor)
        if value.ndim != 2 or value.shape[0] < 1:
            raise ValueError("Forcing factor must be a non-empty matrix.")
        covariance = oe.contract("ir,jr->ij", value, jnp.conj(value))
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "statistical-dynamics-forcing-factor",
                    "array": array_tree_fingerprint(value),
                }
            )
            if covariance_id is None
            else covariance_id
        )
        return cls(covariance, covariance_id=identifier)


class RankAdaptationPolicy(StrictModule, NonTrainableState):
    minimum_rank: int = eqx.field(static=True)
    maximum_rank: int = eqx.field(static=True)
    relative_threshold: float = eqx.field(static=True)
    absolute_threshold: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        minimum_rank: int,
        maximum_rank: int,
        /,
        *,
        relative_threshold: float = 0.0,
        absolute_threshold: float = 0.0,
    ):
        minimum = int(minimum_rank)
        maximum = int(maximum_rank)
        relative = float(relative_threshold)
        absolute = float(absolute_threshold)
        if (
            minimum < 0
            or maximum < minimum
            or relative < 0.0
            or absolute < 0.0
            or not np.isfinite(relative)
            or not np.isfinite(absolute)
        ):
            raise ValueError("Rank-adaptation bounds and thresholds are invalid.")
        self.minimum_rank = minimum
        self.maximum_rank = maximum
        self.relative_threshold = relative
        self.absolute_threshold = absolute
        self.policy_id = canonical_fingerprint(
            {
                "kind": "second-cumulant-rank-adaptation",
                "minimum_rank": minimum,
                "maximum_rank": maximum,
                "relative_threshold": relative,
                "absolute_threshold": absolute,
                "repair": "none",
            }
        )


class RankAdaptationEvent(StrictModule):
    old_rank: Array
    pre_truncation_rank: Array
    new_rank: Array
    threshold: Array
    discarded_variance: Array
    pre_truncation_error: Array
    triggered: Array
    accepted: Array
    policy_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    event_id: str = eqx.field(static=True)


class RankAdaptationResult(StrictModule):
    state: FactorCumulantState
    pre_truncation_covariance: Array
    event: RankAdaptationEvent


def factorize_cumulant(
    layout: SecondCumulantLayout,
    state: DenseCumulantState,
    policy: RankAdaptationPolicy,
    /,
    *,
    previous_rank: int | None = None,
    hermitian_tolerance: float = 1.0e-10,
    psd_tolerance: float = 1.0e-10,
) -> RankAdaptationResult:
    if not isinstance(policy, RankAdaptationPolicy):
        raise TypeError("policy must be a RankAdaptationPolicy.")
    if policy.maximum_rank > layout.eddy_dimension:
        raise ValueError("Rank policy exceeds the eddy covariance dimension.")
    require_valid_state(
        layout,
        state,
        hermitian_tolerance=hermitian_tolerance,
        psd_tolerance=psd_tolerance,
    )
    spectrum = HermitianSpectrum(state.covariance, tolerance=hermitian_tolerance)
    eigenvalues = np.asarray(spectrum.eigenvalues)
    eigenvectors = np.asarray(spectrum.eigenvectors)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    scale = max(float(eigenvalues[0]), 0.0)
    threshold = max(policy.absolute_threshold, policy.relative_threshold * scale)
    positive_rank = int(np.count_nonzero(eigenvalues > threshold))
    if positive_rank < policy.minimum_rank:
        raise ValueError(
            "Rank adaptation cannot satisfy minimum_rank without inventing covariance."
        )
    retained = min(positive_rank, policy.maximum_rank)
    retained_values = eigenvalues[:retained]
    if np.any(retained_values <= 0.0):
        raise ValueError("A retained covariance eigenvalue is not strictly positive.")
    factor = eigenvectors[:, :retained] * np.sqrt(retained_values)[None, :]
    factor_state = FactorCumulantState(
        state.mean,
        jnp.asarray(factor, dtype=state.covariance.dtype),
        layout_id=layout.layout_id,
    )
    reconstructed = factor_state.covariance
    pre_error = jnp.max(jnp.abs(spectrum.reconstruct() - state.covariance), initial=0.0)
    discarded = jnp.asarray(np.sum(eigenvalues[retained:]), dtype=eigenvalues.dtype)
    old = positive_rank if previous_rank is None else int(previous_rank)
    event_id = canonical_fingerprint(
        {
            "kind": "second-cumulant-rank-event",
            "layout": layout.layout_id,
            "policy": policy.policy_id,
            "old_rank": old,
            "pre_truncation_rank": positive_rank,
            "new_rank": retained,
            "threshold": threshold,
        }
    )
    event = RankAdaptationEvent(
        old_rank=jnp.asarray(old, dtype=jnp.int32),
        pre_truncation_rank=jnp.asarray(positive_rank, dtype=jnp.int32),
        new_rank=jnp.asarray(retained, dtype=jnp.int32),
        threshold=jnp.asarray(threshold, dtype=state.covariance.real.dtype),
        discarded_variance=discarded,
        pre_truncation_error=pre_error,
        triggered=jnp.asarray(retained != old),
        accepted=jnp.asarray(
            bool(np.all(np.isfinite(factor))) and retained >= policy.minimum_rank
        ),
        policy_id=policy.policy_id,
        layout_id=layout.layout_id,
        event_id=event_id,
    )
    return RankAdaptationResult(
        state=factor_state,
        pre_truncation_covariance=state.covariance,
        event=event,
    )


def densify_cumulant(state: CumulantState, /) -> DenseCumulantState:
    if isinstance(state, DenseCumulantState):
        return state
    if not isinstance(state, FactorCumulantState):
        raise TypeError("state must be a dense or factor cumulant state.")
    return DenseCumulantState(state.mean, state.covariance, layout_id=state.layout_id)


def cumulants_from_ensemble(
    layout: SecondCumulantLayout,
    members: ArrayLike,
    /,
    *,
    weights: ArrayLike | None = None,
    mean_subspace_tolerance: float = 1.0e-10,
    eddy_mean_tolerance: float = 1.0e-10,
) -> DenseCumulantState:
    """Reduce a finite QL/GQL ensemble to its exact first two cumulants."""
    if not isinstance(layout, SecondCumulantLayout):
        raise TypeError("layout must be a SecondCumulantLayout.")
    values = jnp.asarray(members)
    if values.ndim != 2 or values.shape[1] != layout.state_size or values.shape[0] < 1:
        raise ValueError(
            "Ensemble members must have shape (member_count, layout.state_size)."
        )
    if not jnp.issubdtype(values.dtype, jnp.inexact):
        raise TypeError("Ensemble members must use an inexact dtype.")
    count = int(values.shape[0])
    weights_ = (
        jnp.full((count,), 1.0 / count, dtype=values.real.dtype)
        if weights is None
        else jnp.asarray(weights, dtype=values.real.dtype)
    )
    if weights_.shape != (count,):
        raise ValueError("Ensemble weights must have one value per member.")
    total = jnp.sum(weights_)
    if not bool(
        np.asarray(
            jnp.all(jnp.isfinite(values))
            & jnp.all(jnp.isfinite(weights_))
            & jnp.all(weights_ >= 0.0)
            & (total > 0.0)
        )
    ):
        raise ValueError("Ensemble members and non-negative weights must be finite.")
    weights_ = weights_ / total
    full_mean = oe.contract("b,bi->i", weights_, values)
    low_values = values[:, layout.mean_indices]
    low_mean = full_mean[layout.mean_indices]
    low_defect = jnp.max(jnp.abs(low_values - low_mean[None, :]), initial=0.0)
    eddy_values = values[:, layout.eddy_indices]
    eddy_mean = oe.contract("b,bi->i", weights_, eddy_values)
    eddy_mean_defect = jnp.max(jnp.abs(eddy_mean), initial=0.0)
    if not bool(np.asarray(low_defect <= float(mean_subspace_tolerance))):
        raise ValueError("QL/GQL ensemble members must share one low/mean state.")
    if not bool(np.asarray(eddy_mean_defect <= float(eddy_mean_tolerance))):
        raise ValueError("QL/GQL ensemble eddies must have zero ensemble mean.")
    covariance = oe.contract(
        "b,bi,bj->ij",
        weights_,
        eddy_values,
        jnp.conj(eddy_values),
    )
    state = DenseCumulantState(
        low_mean,
        covariance,
        layout_id=layout.layout_id,
    )
    require_valid_state(
        layout,
        state,
        hermitian_tolerance=max(
            float(mean_subspace_tolerance), float(eddy_mean_tolerance)
        ),
        psd_tolerance=max(float(mean_subspace_tolerance), float(eddy_mean_tolerance)),
    )
    return state


class StationaryCovarianceResult(StrictModule):
    covariance: Array
    residual: Array
    successful: Array
    solve: MatrixEquationResult
    forcing_covariance_id: str = eqx.field(static=True)


def solve_stationary_covariance(
    linear_operator: ArrayLike,
    forcing: ForcingCovariance,
    /,
    *,
    policy: MatrixEquationPolicy | MatrixEquationPlan | None = None,
) -> StationaryCovarianceResult:
    """Solve ``A C + C A* + Q = 0`` through Phydrax matrix equations."""
    if not isinstance(forcing, ForcingCovariance):
        raise TypeError("forcing must be a ForcingCovariance.")
    operator = jnp.asarray(linear_operator)
    if operator.shape != (forcing.dimension, forcing.dimension):
        raise ValueError("Linear operator and forcing covariance dimensions differ.")
    problem = continuous_lyapunov_equation(
        operator,
        forcing.covariance,
        problem_id=canonical_fingerprint(
            {
                "kind": "stationary-statistical-dynamics-covariance",
                "forcing": forcing.covariance_id,
                "dimension": forcing.dimension,
            }
        ),
    )
    result = solve_matrix_equation(problem, policy=policy)
    covariance = result.value
    residual = (
        oe.contract("ij,jk->ik", operator, covariance)
        + oe.contract("ij,kj->ik", covariance, jnp.conj(operator))
        + forcing.covariance
    )
    residual_norm = jnp.sqrt(
        jnp.real(oe.contract("ij,ij->", jnp.conj(residual), residual))
    )
    return StationaryCovarianceResult(
        covariance=covariance,
        residual=residual_norm,
        successful=result.successful & jnp.all(jnp.isfinite(covariance)),
        solve=result,
        forcing_covariance_id=forcing.covariance_id,
    )


__all__ = [
    "CumulantState",
    "CumulantStateEvidence",
    "DenseCumulantState",
    "FactorCumulantState",
    "ForcingCovariance",
    "RankAdaptationEvent",
    "RankAdaptationPolicy",
    "RankAdaptationResult",
    "SecondCumulantLayout",
    "StationaryCovarianceResult",
    "densify_cumulant",
    "cumulants_from_ensemble",
    "factorize_cumulant",
    "require_valid_state",
    "solve_stationary_covariance",
    "state_evidence",
]
