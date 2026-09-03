#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import ArraySpace, DenseLinearOperator, OperatorProperties
from ...linalg.eigen import (
    DenseEigh,
    Eigenproblem,
    eigensolve,
    EigenSolvePolicy,
    GeneralizedEigenproblem,
)
from ...linalg.svd import svd, SVDProblem, SVDSolvePolicy
from .._trajectory import TrajectoryData, TrajectoryTransitions
from ._features import AbstractFeatureLibrary, FeatureEvaluation
from ._status import (
    IDENTIFICATION_INFEASIBLE,
    IDENTIFICATION_INSUFFICIENT_SAMPLES,
    IDENTIFICATION_NONFINITE,
    IDENTIFICATION_RANK_DEFICIENT,
    IDENTIFICATION_SUCCESS,
)


class LaggedPairWeighting(StrEnum):
    """Statistical evidence assigned to one valid lagged pair."""

    UNIFORM = "uniform"
    SOURCE = "source"
    GEOMETRIC = "geometric"


class LaggedKineticEvidence(StrictModule):
    lag: int = eqx.field(static=True)
    physical_lag_minimum: Array
    physical_lag_maximum: Array
    physical_lag_mean: Array
    uniform_physical_lag: Array
    valid_pair_count: Array
    excluded_pair_count: Array
    effective_samples: Array
    stationarity_defect: Array
    weighting: LaggedPairWeighting = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class VAMPModel(StrictModule):
    source_mean: Array
    target_mean: Array
    source_rotations: Array
    target_rotations: Array
    singular_values: Array

    def transform(self, values: ArrayLike, /) -> Array:
        array = jnp.asarray(values)
        return contract("...i,ij->...j", array - self.source_mean, self.source_rotations)

    def transform_targets(self, values: ArrayLike, /) -> Array:
        array = jnp.asarray(values)
        return contract("...i,ij->...j", array - self.target_mean, self.target_rotations)


class VAMPDiagnostics(StrictModule):
    singular_values: Array
    score: Array
    numerical_rank: Array
    minimum_singular_gap: Array
    repeated_spectrum: Array
    effective_samples: Array
    covariance_condition: Array
    lag: LaggedKineticEvidence


class VAMPResult(StrictModule):
    """Centered nontrivial VAMP singular functions and complete fit evidence."""

    model: VAMPModel
    diagnostics: VAMPDiagnostics
    valid: Array
    status: Array
    library: AbstractFeatureLibrary
    source_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def transform_source(self, states: ArrayLike, /) -> Array:
        evaluation = self.library.evaluate(states)
        return self.model.transform(evaluation.values)

    def transform_target(self, states: ArrayLike, /) -> Array:
        evaluation = self.library.evaluate(states)
        return self.model.transform_targets(evaluation.values)


class VACDiagnostics(StrictModule):
    eigenvalues: Array
    residual_norms: Array
    numerical_rank: Array
    minimum_eigengap: Array
    repeated_spectrum: Array
    effective_samples: Array
    reversibility_defect: Array
    lag: LaggedKineticEvidence


class VACResult(StrictModule):
    """Reversible variational eigenfunctions, including linear TICA fits."""

    mean: Array
    components: Array
    eigenvalues: Array
    diagnostics: VACDiagnostics
    valid: Array
    status: Array
    library: AbstractFeatureLibrary | None
    state_shape: tuple[int, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def transform(self, states: ArrayLike, /) -> Array:
        values = jnp.asarray(states)
        if self.library is None:
            rank = len(self.state_shape)
            if rank and tuple(values.shape[-rank:]) != self.state_shape:
                raise ValueError(f"states must end in state shape {self.state_shape}.")
            leading = values.shape if rank == 0 else values.shape[:-rank]
            features = values.reshape(leading + (int(np.prod(self.state_shape)),))
        else:
            features = self.library.evaluate(values).values
        return contract("...i,ij->...j", features - self.mean, self.components)

    def implied_timescales(self, /) -> tuple[Array, Array]:
        lag = self.diagnostics.lag.physical_lag_mean
        absolute = jnp.abs(self.eigenvalues)
        admissible = (
            self.valid
            & self.diagnostics.lag.uniform_physical_lag
            & jnp.isfinite(absolute)
            & (absolute > 0.0)
            & (absolute < 1.0)
        )
        safe = jnp.where(admissible, absolute, 0.5)
        values = -lag / jnp.log(safe)
        return jnp.where(admissible, values, jnp.nan), admissible


def _event_mask(mask: Array, event_rank: int, /) -> Array:
    return mask.reshape(mask.shape + (1,) * event_rank)


def _lagged_pair_data(
    data: TrajectoryData,
    lag: int,
    weighting: LaggedPairWeighting,
    lag_tolerance: float,
    /,
) -> tuple[TrajectoryTransitions, Array, LaggedKineticEvidence]:
    if not isinstance(data, TrajectoryData):
        raise TypeError("data must be TrajectoryData.")
    if not isinstance(weighting, LaggedPairWeighting):
        raise TypeError("weighting must be LaggedPairWeighting.")
    tolerance = float(lag_tolerance)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("lag_tolerance must be finite and nonnegative.")
    transitions = data.transitions(int(lag))
    valid = transitions.valid
    pair_count = data.capacity - transitions.lag
    if weighting is LaggedPairWeighting.UNIFORM:
        weights = jnp.ones(valid.shape, dtype=data.weights.dtype)
    elif weighting is LaggedPairWeighting.SOURCE:
        weights = data.weights[..., :pair_count]
    else:
        weights = transitions.weights
    weights = jnp.where(valid, weights, 0.0)
    deltas = data.coordinates[..., transitions.lag :] - data.coordinates[..., :pair_count]
    finite_delta = jnp.isfinite(deltas)
    pair_valid = valid & finite_delta
    count = jnp.sum(pair_valid)
    safe_delta = jnp.where(pair_valid, deltas, 0.0)
    denominator = jnp.maximum(count, 1).astype(deltas.dtype)
    lag_mean = jnp.sum(safe_delta) / denominator
    lag_minimum = jnp.min(jnp.where(pair_valid, deltas, jnp.inf))
    lag_maximum = jnp.max(jnp.where(pair_valid, deltas, -jnp.inf))
    lag_minimum = jnp.where(count > 0, lag_minimum, jnp.nan)
    lag_maximum = jnp.where(count > 0, lag_maximum, jnp.nan)
    scale = jnp.maximum(jnp.abs(lag_mean), 1.0)
    uniform = (
        (count > 0)
        & jnp.isfinite(lag_mean)
        & (lag_minimum > 0.0)
        & ((lag_maximum - lag_minimum) <= tolerance * scale)
    )
    weight_sum = jnp.sum(weights)
    weight_square_sum = jnp.sum(weights * weights)
    effective = jnp.where(
        weight_square_sum > 0.0, weight_sum * weight_sum / weight_square_sum, 0.0
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "lagged-kinetic-evidence",
            "dataset": data.dataset_id,
            "lag": int(lag),
            "weighting": weighting.value,
            "lag_tolerance": tolerance.hex(),
        }
    )
    evidence = LaggedKineticEvidence(
        lag=int(lag),
        physical_lag_minimum=lag_minimum,
        physical_lag_maximum=lag_maximum,
        physical_lag_mean=lag_mean,
        uniform_physical_lag=uniform,
        valid_pair_count=count.astype(jnp.int32),
        excluded_pair_count=(valid.size - count).astype(jnp.int32),
        effective_samples=effective,
        stationarity_defect=jnp.asarray(jnp.nan, dtype=deltas.dtype),
        weighting=weighting,
        dataset_id=data.dataset_id,
        evidence_id=evidence_id,
    )
    return transitions, weights, evidence


def _feature_pairs(
    data: TrajectoryData,
    library: AbstractFeatureLibrary,
    lag: int,
    weighting: LaggedPairWeighting,
    lag_tolerance: float,
    /,
) -> tuple[FeatureEvaluation, FeatureEvaluation, Array, Array, LaggedKineticEvidence]:
    if not isinstance(library, AbstractFeatureLibrary):
        raise TypeError("library must implement AbstractFeatureLibrary.")
    if library.state_layout.layout_id != data.state_layout.layout_id:
        raise ValueError("library and trajectory must use the same state layout.")
    if library.input_layout is not None:
        raise ValueError("Variational kinetic features must be state-only.")
    transitions, weights, evidence = _lagged_pair_data(
        data, lag, weighting, lag_tolerance
    )
    event_rank = len(data.state_layout.shape)
    source_states = jnp.where(
        _event_mask(transitions.valid, event_rank), transitions.source_states, 0.0
    )
    target_states = jnp.where(
        _event_mask(transitions.valid, event_rank), transitions.target_states, 0.0
    )
    source = library.evaluate(source_states)
    target = library.evaluate(target_states)
    valid = transitions.valid & source.valid & target.valid
    weights = jnp.where(valid, weights, 0.0)
    source_values = source.values.reshape((-1, library.num_features))
    target_values = target.values.reshape((-1, library.num_features))
    weights_flat = weights.reshape((-1,))
    total = jnp.sum(weights_flat)
    source_mean = contract("n,ni->i", weights_flat, source_values) / jnp.maximum(
        total, 1.0
    )
    target_mean = contract("n,ni->i", weights_flat, target_values) / jnp.maximum(
        total, 1.0
    )
    variance_scale = jnp.maximum(
        jnp.sqrt(jnp.sum(source_mean * source_mean + target_mean * target_mean)), 1.0
    )
    stationarity = jnp.sqrt(jnp.sum((source_mean - target_mean) ** 2)) / variance_scale
    evidence = eqx.tree_at(
        lambda value: value.stationarity_defect, evidence, stationarity
    )
    return source, target, valid.reshape((-1,)), weights_flat, evidence


def _weighted_covariances(
    source: Array, target: Array, valid: Array, weights: Array, /
) -> tuple[Array, Array, Array, Array, Array]:
    mask = valid & jnp.isfinite(weights) & (weights >= 0.0)
    weight = jnp.where(mask, weights, 0.0)
    x = jnp.where(mask[:, None], source, 0.0)
    y = jnp.where(mask[:, None], target, 0.0)
    total = jnp.sum(weight)
    denominator = jnp.maximum(total, 1.0)
    x_mean = contract("n,ni->i", weight, x) / denominator
    y_mean = contract("n,ni->i", weight, y) / denominator
    xc = jnp.where(mask[:, None], x - x_mean, 0.0)
    yc = jnp.where(mask[:, None], y - y_mean, 0.0)
    cxx = contract("ni,n,nj->ij", xc, weight, xc) / denominator
    cyy = contract("ni,n,nj->ij", yc, weight, yc) / denominator
    cxy = contract("ni,n,nj->ij", xc, weight, yc) / denominator
    return x_mean, y_mean, cxx, cyy, cxy


def _covariance_inverse_root(
    covariance: Array, regularization: float, identity: str, /
) -> tuple[Array, Array, Array, Array]:
    size = int(covariance.shape[0])
    properties = OperatorProperties(
        self_adjoint=True,
        positive_semidefinite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_semidefinite": "construction",
        },
    )
    spectrum = eigensolve(
        Eigenproblem(
            DenseLinearOperator(covariance, properties=properties),
            problem_id=identity,
        ),
        policy=EigenSolvePolicy(DenseEigh(), count=size),
    )
    eigenvalues = spectrum.eigenvalues
    eigenvectors = spectrum.eigenvectors
    maximum = jnp.maximum(jnp.max(jnp.abs(eigenvalues)), 1.0)
    cutoff = size * jnp.finfo(eigenvalues.dtype).eps * maximum
    retained = eigenvalues > cutoff
    rank = jnp.sum(retained).astype(jnp.int32)
    shifted = jnp.maximum(eigenvalues, 0.0) + jnp.asarray(
        regularization, dtype=eigenvalues.dtype
    )
    inverse = jax.lax.rsqrt(shifted)
    root = (eigenvectors * inverse[None, :]) @ eigenvectors.T
    smallest = jnp.min(jnp.where(retained, shifted, jnp.inf))
    condition = jnp.max(shifted) / smallest
    valid = spectrum.successful & jnp.all(jnp.isfinite(root))
    return root, rank, condition, valid


def fit_vamp(
    data: TrajectoryData,
    library: AbstractFeatureLibrary,
    /,
    *,
    lag: int = 1,
    n_modes: int = 2,
    regularization: float = 1.0e-8,
    weighting: LaggedPairWeighting = LaggedPairWeighting.GEOMETRIC,
    lag_tolerance: float = 1.0e-8,
) -> VAMPResult:
    """Fit centered nontrivial VAMP singular functions on valid lagged pairs."""

    modes = int(n_modes)
    regularization_ = float(regularization)
    if modes <= 0 or modes > library.num_features:
        raise ValueError("n_modes must lie between one and the feature count.")
    if not isfinite(regularization_) or regularization_ <= 0.0:
        raise ValueError("regularization must be finite and positive.")
    source, target, valid, weights, lag_evidence = _feature_pairs(
        data, library, lag, weighting, lag_tolerance
    )
    source_values = source.values.reshape((-1, library.num_features))
    target_values = target.values.reshape((-1, library.num_features))
    x_mean, y_mean, cxx, cyy, cxy = _weighted_covariances(
        source_values, target_values, valid, weights
    )
    x_root, x_rank, x_condition, x_valid = _covariance_inverse_root(
        cxx, regularization_, f"vamp-source:{data.dataset_id}:{int(lag)}"
    )
    y_root, y_rank, y_condition, y_valid = _covariance_inverse_root(
        cyy, regularization_, f"vamp-target:{data.dataset_id}:{int(lag)}"
    )
    whitened = x_root @ cxy @ y_root
    requested = min(library.num_features, modes + 1)
    factorization = svd(
        SVDProblem(
            DenseLinearOperator(whitened),
            problem_id=f"vamp-svd:{data.dataset_id}:{int(lag)}",
        ),
        policy=SVDSolvePolicy(count=requested, which="largest"),
    )
    singular_values = factorization.singular_values[:modes]
    source_rotations = x_root @ factorization.left_vectors[:, :modes]
    target_rotations = y_root @ factorization.right_vectors[:, :modes]
    if requested > modes:
        gaps = singular_values - factorization.singular_values[1 : modes + 1]
    elif modes > 1:
        gaps = singular_values[:-1] - singular_values[1:]
    else:
        gaps = jnp.asarray([jnp.inf], dtype=singular_values.dtype)
    minimum_gap = jnp.min(gaps)
    repeated = minimum_gap <= (
        64.0
        * jnp.finfo(singular_values.dtype).eps
        * jnp.maximum(jnp.max(singular_values), 1.0)
    )
    rank = jnp.minimum(x_rank, y_rank)
    finite = (
        jnp.all(jnp.isfinite(singular_values))
        & jnp.all(jnp.isfinite(source_rotations))
        & jnp.all(jnp.isfinite(target_rotations))
    )
    enough = lag_evidence.valid_pair_count > modes
    rank_ok = rank >= modes
    valid_fit = (
        factorization.successful
        & x_valid
        & y_valid
        & finite
        & enough
        & rank_ok
        & lag_evidence.uniform_physical_lag
    )
    status = jnp.where(
        ~finite,
        IDENTIFICATION_NONFINITE,
        jnp.where(
            ~enough,
            IDENTIFICATION_INSUFFICIENT_SAMPLES,
            jnp.where(
                ~rank_ok,
                IDENTIFICATION_RANK_DEFICIENT,
                jnp.where(
                    factorization.successful & lag_evidence.uniform_physical_lag,
                    IDENTIFICATION_SUCCESS,
                    IDENTIFICATION_INFEASIBLE,
                ),
            ),
        ),
    ).astype(jnp.int32)
    model = VAMPModel(
        source_mean=x_mean,
        target_mean=y_mean,
        source_rotations=source_rotations,
        target_rotations=target_rotations,
        singular_values=singular_values,
    )
    weight_sum = jnp.sum(weights)
    weight_square_sum = jnp.sum(weights * weights)
    effective = jnp.where(
        weight_square_sum > 0.0, weight_sum * weight_sum / weight_square_sum, 0.0
    )
    diagnostics = VAMPDiagnostics(
        singular_values=singular_values,
        score=jnp.sum(singular_values * singular_values),
        numerical_rank=rank,
        minimum_singular_gap=minimum_gap,
        repeated_spectrum=repeated,
        effective_samples=effective,
        covariance_condition=jnp.maximum(x_condition, y_condition),
        lag=lag_evidence,
    )
    method_id = canonical_fingerprint(
        {
            "kind": "vamp",
            "dataset": data.dataset_id,
            "library": library.library_id,
            "lag": int(lag),
            "modes": modes,
            "regularization": regularization_.hex(),
            "weighting": weighting.value,
        }
    )
    return VAMPResult(
        model=model,
        diagnostics=diagnostics,
        valid=valid_fit,
        status=status,
        library=library,
        source_id=data.source_id,
        method_id=method_id,
    )


def _canonicalize_columns(vectors: Array, /) -> Array:
    index = jnp.argmax(jnp.abs(vectors), axis=0)
    selected = jnp.take_along_axis(vectors, index[None, :], axis=0).reshape((-1,))
    sign = jnp.where(selected < 0.0, -1.0, 1.0)
    return vectors * sign[None, :]


def _fit_vac_features(
    data: TrajectoryData,
    source_values: Array,
    target_values: Array,
    pair_valid: Array,
    pair_weights: Array,
    lag_evidence: LaggedKineticEvidence,
    /,
    *,
    n_modes: int,
    regularization: float,
    library: AbstractFeatureLibrary | None,
    state_shape: tuple[int, ...],
    method_name: str,
) -> VACResult:
    features = int(source_values.shape[-1])
    values0 = source_values.reshape((-1, features))
    values1 = target_values.reshape((-1, features))
    mask = pair_valid.reshape((-1,))
    weights = jnp.where(mask, pair_weights.reshape((-1,)), 0.0)
    total = jnp.sum(weights)
    mean = contract("n,ni->i", weights, values0 + values1) / jnp.maximum(2.0 * total, 1.0)
    x = jnp.where(mask[:, None], values0 - mean, 0.0)
    y = jnp.where(mask[:, None], values1 - mean, 0.0)
    denominator = jnp.maximum(total, 1.0)
    c00 = contract("ni,n,nj->ij", x, weights, x) / denominator
    c11 = contract("ni,n,nj->ij", y, weights, y) / denominator
    c01 = contract("ni,n,nj->ij", x, weights, y) / denominator
    c0 = 0.5 * (c00 + c11)
    ctau = 0.5 * (c01 + c01.T)
    identity = jnp.eye(features, dtype=c0.dtype)
    metric = c0 + jnp.asarray(regularization, dtype=c0.dtype) * identity
    covariance_spectrum = eigensolve(
        Eigenproblem(
            DenseLinearOperator(
                c0,
                properties=OperatorProperties(
                    self_adjoint=True,
                    positive_semidefinite=True,
                    evidence={
                        "self_adjoint": "construction",
                        "positive_semidefinite": "construction",
                    },
                ),
            ),
            problem_id=f"{method_name}-covariance:{data.dataset_id}",
        ),
        policy=EigenSolvePolicy(DenseEigh(), count=features),
    )
    maximum_covariance = jnp.maximum(
        jnp.max(jnp.abs(covariance_spectrum.eigenvalues)), 1.0
    )
    covariance_cutoff = features * jnp.finfo(c0.dtype).eps * maximum_covariance
    numerical_rank = jnp.sum(covariance_spectrum.eigenvalues > covariance_cutoff).astype(
        jnp.int32
    )
    properties = OperatorProperties(
        self_adjoint=True, evidence={"self_adjoint": "construction"}
    )
    metric_properties = OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        positive_semidefinite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
            "positive_semidefinite": "construction",
        },
    )
    space = ArraySpace((features,), dtype=c0.dtype)
    solve = eigensolve(
        GeneralizedEigenproblem(
            DenseLinearOperator(ctau, source=space, target=space, properties=properties),
            DenseLinearOperator(
                metric, source=space, target=space, properties=metric_properties
            ),
            problem_id=f"{method_name}:{data.dataset_id}",
        ),
        policy=EigenSolvePolicy(DenseEigh(), count=features),
    )
    order = jnp.argsort(jnp.abs(solve.eigenvalues))[::-1]
    selected = order[:n_modes]
    eigenvalues = solve.eigenvalues[selected]
    components = _canonicalize_columns(solve.eigenvectors[:, selected])
    residuals = solve.residual_norms[selected]
    gaps = (
        jnp.abs(eigenvalues[:-1] - eigenvalues[1:])
        if n_modes > 1
        else jnp.asarray([jnp.inf], dtype=eigenvalues.dtype)
    )
    minimum_gap = jnp.min(gaps)
    scale = jnp.maximum(jnp.max(jnp.abs(eigenvalues)), 1.0)
    repeated = minimum_gap <= 64.0 * jnp.finfo(eigenvalues.dtype).eps * scale
    reversibility = jnp.sqrt(jnp.sum((c01 - c01.T) ** 2)) / jnp.maximum(
        jnp.sqrt(jnp.sum(c01 * c01)), jnp.finfo(c01.dtype).tiny
    )
    finite = (
        jnp.all(jnp.isfinite(eigenvalues))
        & jnp.all(jnp.isfinite(components))
        & jnp.all(jnp.isfinite(residuals))
    )
    enough = lag_evidence.valid_pair_count > features
    rank_ok = numerical_rank >= n_modes
    valid = (
        solve.successful
        & covariance_spectrum.successful
        & finite
        & enough
        & rank_ok
        & lag_evidence.uniform_physical_lag
    )
    status = jnp.where(
        ~finite,
        IDENTIFICATION_NONFINITE,
        jnp.where(
            ~enough,
            IDENTIFICATION_INSUFFICIENT_SAMPLES,
            jnp.where(
                ~rank_ok,
                IDENTIFICATION_RANK_DEFICIENT,
                jnp.where(
                    solve.successful & lag_evidence.uniform_physical_lag,
                    IDENTIFICATION_SUCCESS,
                    IDENTIFICATION_INFEASIBLE,
                ),
            ),
        ),
    ).astype(jnp.int32)
    diagnostics = VACDiagnostics(
        eigenvalues=eigenvalues,
        residual_norms=residuals,
        numerical_rank=numerical_rank,
        minimum_eigengap=minimum_gap,
        repeated_spectrum=repeated,
        effective_samples=lag_evidence.effective_samples,
        reversibility_defect=reversibility,
        lag=lag_evidence,
    )
    method_id = canonical_fingerprint(
        {
            "kind": method_name,
            "dataset": data.dataset_id,
            "library": None if library is None else library.library_id,
            "lag": lag_evidence.lag,
            "modes": n_modes,
            "regularization": float(regularization).hex(),
            "weighting": lag_evidence.weighting.value,
        }
    )
    return VACResult(
        mean=mean,
        components=components,
        eigenvalues=eigenvalues,
        diagnostics=diagnostics,
        valid=valid,
        status=status,
        library=library,
        state_shape=state_shape,
        source_id=data.source_id,
        method_id=method_id,
    )


def fit_vac(
    data: TrajectoryData,
    library: AbstractFeatureLibrary,
    /,
    *,
    lag: int = 1,
    n_modes: int = 2,
    regularization: float = 1.0e-8,
    weighting: LaggedPairWeighting = LaggedPairWeighting.GEOMETRIC,
    lag_tolerance: float = 1.0e-8,
) -> VACResult:
    """Fit reversible variational eigenfunctions in a declared feature space."""

    regularization_ = float(regularization)
    if not isfinite(regularization_) or regularization_ <= 0.0:
        raise ValueError("regularization must be finite and positive for VAC.")
    modes = int(n_modes)
    if modes <= 0 or modes > library.num_features:
        raise ValueError("n_modes must lie between one and the feature count.")
    source, target, valid, weights, evidence = _feature_pairs(
        data, library, lag, weighting, lag_tolerance
    )
    return _fit_vac_features(
        data,
        source.values,
        target.values,
        valid,
        weights,
        evidence,
        n_modes=modes,
        regularization=regularization_,
        library=library,
        state_shape=data.state_layout.shape,
        method_name="vac",
    )


def fit_tica(
    data: TrajectoryData,
    /,
    *,
    lag: int = 1,
    n_modes: int = 2,
    regularization: float = 1.0e-8,
    weighting: LaggedPairWeighting = LaggedPairWeighting.GEOMETRIC,
    lag_tolerance: float = 1.0e-8,
) -> VACResult:
    """Fit linear time-lagged independent components of a Euclidean state."""

    if not isinstance(data, TrajectoryData):
        raise TypeError("data must be TrajectoryData.")
    regularization_ = float(regularization)
    if not isfinite(regularization_) or regularization_ <= 0.0:
        raise ValueError("regularization must be finite and positive for TICA.")
    feature_count = data.state_layout.size
    modes = int(n_modes)
    if modes <= 0 or modes > feature_count:
        raise ValueError("n_modes must lie between one and the flattened state size.")
    transitions, weights, evidence = _lagged_pair_data(
        data, lag, weighting, lag_tolerance
    )
    event_rank = len(data.state_layout.shape)
    source = jnp.where(
        _event_mask(transitions.valid, event_rank), transitions.source_states, 0.0
    ).reshape(transitions.valid.shape + (feature_count,))
    target = jnp.where(
        _event_mask(transitions.valid, event_rank), transitions.target_states, 0.0
    ).reshape(transitions.valid.shape + (feature_count,))
    finite = jnp.all(jnp.isfinite(source), axis=-1) & jnp.all(
        jnp.isfinite(target), axis=-1
    )
    valid = transitions.valid & finite
    return _fit_vac_features(
        data,
        source,
        target,
        valid,
        jnp.where(valid, weights, 0.0),
        evidence,
        n_modes=modes,
        regularization=regularization_,
        library=None,
        state_shape=data.state_layout.shape,
        method_name="tica",
    )


__all__ = [
    "LaggedKineticEvidence",
    "LaggedPairWeighting",
    "VACDiagnostics",
    "VACResult",
    "VAMPDiagnostics",
    "VAMPModel",
    "VAMPResult",
    "fit_tica",
    "fit_vac",
    "fit_vamp",
]
