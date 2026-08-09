#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._finite_feature import AbstractFiniteFeatureKernel


def _proposal_id(
    frequencies: ArrayLike,
    directions: ArrayLike,
    phases: ArrayLike,
    log_proposal_density: ArrayLike,
    geometry_id: str,
    proposal_scale: float,
    /,
) -> str:
    digest = hashlib.sha256()
    digest.update(geometry_id.encode("utf-8"))
    digest.update(np.asarray(proposal_scale, dtype=np.float64).tobytes())
    for value in (frequencies, directions, phases, log_proposal_density):
        array = np.asarray(value)
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.dtype.str.encode("utf-8"))
        digest.update(array.tobytes())
    return f"{geometry_id}:{digest.hexdigest()[:16]}"


class NoncompactFeatureProposal(StrictModule, NonTrainableState):
    """Fixed parameter-independent spectral noise and geometric directions."""

    frequencies: Array
    directions: Array
    phases: Array
    log_proposal_density: Array
    geometry_id: str = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)
    proposal_scale: float = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)
    spectral_rank: int = eqx.field(static=True)

    def __init__(
        self,
        frequencies: ArrayLike,
        directions: ArrayLike,
        phases: ArrayLike,
        log_proposal_density: ArrayLike,
        /,
        *,
        geometry_id: str,
        proposal_id: str,
        proposal_scale: float,
    ):
        frequency_array = jnp.asarray(frequencies, dtype=float)
        direction_array = jnp.asarray(directions, dtype=float)
        phase_array = jnp.asarray(phases, dtype=float)
        log_density = jnp.asarray(log_proposal_density, dtype=float)
        if frequency_array.ndim != 2 or int(frequency_array.shape[0]) == 0:
            raise ValueError("frequencies must have shape (sample, spectral_rank).")
        sample_count = int(frequency_array.shape[0])
        if int(direction_array.shape[0]) != sample_count:
            raise ValueError("directions must have one entry per spectral sample.")
        if phase_array.shape != (sample_count,) or log_density.shape != (sample_count,):
            raise ValueError("phases and proposal densities must align with samples.")
        if not isinstance(geometry_id, str) or not geometry_id:
            raise ValueError("geometry_id must be a nonempty string.")
        if not isinstance(proposal_id, str) or not proposal_id:
            raise ValueError("proposal_id must be a nonempty string.")
        if not np.isfinite(float(proposal_scale)) or float(proposal_scale) <= 0.0:
            raise ValueError("proposal_scale must be finite and positive.")
        invalid = (
            jnp.any(~jnp.isfinite(frequency_array))
            | jnp.any(~jnp.isfinite(direction_array))
            | jnp.any(~jnp.isfinite(phase_array))
            | jnp.any(~jnp.isfinite(log_density))
        )
        self.frequencies = eqx.error_if(
            frequency_array,
            invalid,
            "Noncompact proposal arrays must be finite.",
        )
        self.directions = direction_array
        self.phases = phase_array
        self.log_proposal_density = log_density
        self.geometry_id = geometry_id
        self.proposal_id = proposal_id
        self.proposal_scale = float(proposal_scale)
        self.sample_count = sample_count
        self.spectral_rank = int(frequency_array.shape[1])

    def prefix(self, sample_count: int, /) -> NoncompactFeatureProposal:
        """Return a nested fixed-noise prefix without drawing new randomness."""
        count = int(sample_count)
        if count <= 0 or count > self.sample_count:
            raise ValueError("prefix sample_count must lie within the proposal.")
        frequencies = self.frequencies[:count]
        directions = self.directions[:count]
        phases = self.phases[:count]
        log_density = self.log_proposal_density[:count]
        prefix_id = _proposal_id(
            frequencies,
            directions,
            phases,
            log_density,
            self.geometry_id,
            self.proposal_scale,
        )
        return NoncompactFeatureProposal(
            frequencies,
            directions,
            phases,
            log_density,
            geometry_id=self.geometry_id,
            proposal_id=f"{prefix_id}:prefix={count}",
            proposal_scale=self.proposal_scale,
        )


class ImportanceFeatureDiagnostics(StrictModule):
    """Importance-weight degeneracy and Monte Carlo normalizer diagnostics."""

    normalized_weights: Array
    effective_sample_size: Array
    normalizer_estimate: Array
    monte_carlo_standard_error: Array
    maximum_normalized_weight: Array
    finite_importance_variance: Array
    proposal_id: str = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)

    def __init__(
        self,
        log_importance_weights: Array,
        proposal_id: str,
        /,
        *,
        finite_importance_variance: ArrayLike = True,
    ):
        log_weights = jnp.asarray(log_importance_weights, dtype=float)
        if log_weights.ndim != 1 or int(log_weights.shape[0]) == 0:
            raise ValueError("log_importance_weights must be a nonempty vector.")
        finite_variance = jnp.asarray(finite_importance_variance, dtype=bool)
        if finite_variance.ndim != 0:
            raise ValueError("finite_importance_variance must be scalar.")
        log_weights = eqx.error_if(
            log_weights,
            jnp.any(jnp.isnan(log_weights))
            | jnp.any(log_weights == jnp.inf)
            | jnp.all(log_weights == -jnp.inf),
            "Importance weights must have finite, nonzero total mass.",
        )
        maximum = jnp.max(log_weights)
        shifted = jnp.exp(log_weights - maximum)
        shifted_total = jnp.sum(shifted)
        normalized = shifted / shifted_total
        sample_count = int(log_weights.shape[0])
        log_normalizer = maximum + jnp.log(shifted_total) - math.log(float(sample_count))
        if sample_count == 1:
            sample_standard_error = jnp.asarray(jnp.inf, dtype=log_weights.dtype)
        else:
            shifted_mean = shifted_total / float(sample_count)
            centered_sum_squares = jnp.sum((shifted - shifted_mean) ** 2)
            log_standard_error = maximum + 0.5 * (
                jnp.log(centered_sum_squares)
                - math.log(float(sample_count))
                - math.log(float(sample_count - 1))
            )
            sample_standard_error = jnp.exp(log_standard_error)
        self.normalized_weights = normalized
        self.effective_sample_size = 1.0 / jnp.sum(normalized * normalized)
        self.normalizer_estimate = jnp.exp(log_normalizer)
        self.monte_carlo_standard_error = jnp.where(
            finite_variance,
            sample_standard_error,
            jnp.asarray(jnp.inf, dtype=log_weights.dtype),
        )
        self.maximum_normalized_weight = jnp.max(normalized)
        self.finite_importance_variance = finite_variance
        self.proposal_id = proposal_id
        self.sample_count = sample_count


def _multivariate_cauchy_log_density(frequencies: Array, scale: float, /) -> Array:
    rank = int(frequencies.shape[1])
    squared_radius = jnp.sum(frequencies * frequencies, axis=-1)
    return (
        math.lgamma(0.5 * (rank + 1))
        - math.lgamma(0.5)
        - 0.5 * rank * math.log(math.pi)
        - rank * math.log(scale)
        - 0.5 * (rank + 1) * jnp.log1p(squared_radius / (scale * scale))
    )


def hyperbolic_feature_proposal(
    key: PRNGKeyArray,
    dimension: int,
    sample_count: int,
    /,
    *,
    proposal_scale: float = 1.0,
) -> NoncompactFeatureProposal:
    """Draw fixed Cauchy frequencies, ideal-boundary directions, and phases."""
    resolved_dimension = int(dimension)
    count = int(sample_count)
    scale = float(proposal_scale)
    if resolved_dimension < 2 or count <= 0 or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Hyperbolic proposal dimensions, count, and scale are invalid.")
    frequency_key, direction_key, phase_key = jax.random.split(key, 3)
    frequencies = scale * jax.random.t(frequency_key, 1.0, (count, 1))
    raw_directions = jax.random.normal(direction_key, (count, resolved_dimension))
    directions = raw_directions / jnp.linalg.norm(raw_directions, axis=-1, keepdims=True)
    phases = 2.0 * jnp.pi * jax.random.uniform(phase_key, (count,))
    geometry_id = f"hyperbolic-H{resolved_dimension}"
    log_density = _multivariate_cauchy_log_density(frequencies, scale)
    return NoncompactFeatureProposal(
        frequencies,
        directions,
        phases,
        log_density,
        geometry_id=geometry_id,
        proposal_id=_proposal_id(
            frequencies,
            directions,
            phases,
            log_density,
            geometry_id,
            scale,
        ),
        proposal_scale=scale,
    )


def _orthogonal_frames(key: PRNGKeyArray, sample_count: int, dimension: int, /) -> Array:
    raw = jax.random.normal(key, (sample_count, dimension, dimension))

    def orthogonal(matrix):
        frame, triangular = jnp.linalg.qr(matrix)
        signs = jnp.where(jnp.diag(triangular) < 0.0, -1.0, 1.0)
        return frame * signs[None, :]

    return jax.vmap(orthogonal)(raw)


def spd_feature_proposal(
    key: PRNGKeyArray,
    matrix_dimension: int,
    sample_count: int,
    /,
    *,
    proposal_scale: float = 1.0,
) -> NoncompactFeatureProposal:
    """Draw fixed SPD spectral vectors, orthogonal flags, and phases."""
    dimension = int(matrix_dimension)
    count = int(sample_count)
    scale = float(proposal_scale)
    if dimension < 2 or count <= 0 or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("SPD proposal dimensions, count, and scale are invalid.")
    normal_key, radial_key, frame_key, phase_key = jax.random.split(key, 4)
    radial_square = 2.0 * jax.random.gamma(radial_key, 0.5, shape=(count, 1))
    frequencies = (
        scale
        * jax.random.normal(normal_key, (count, dimension))
        / jnp.sqrt(radial_square)
    )
    frames = _orthogonal_frames(frame_key, count, dimension)
    phases = 2.0 * jnp.pi * jax.random.uniform(phase_key, (count,))
    geometry_id = f"spd-SPD{dimension}"
    log_density = _multivariate_cauchy_log_density(frequencies, scale)
    return NoncompactFeatureProposal(
        frequencies,
        frames,
        phases,
        log_density,
        geometry_id=geometry_id,
        proposal_id=_proposal_id(
            frequencies,
            frames,
            phases,
            log_density,
            geometry_id,
            scale,
        ),
        proposal_scale=scale,
    )


def _positive_parameter(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.ndim != 0:
        raise ValueError(f"{name} must be scalar.")
    return eqx.error_if(
        array,
        ~jnp.isfinite(array) | (array <= 0.0),
        f"{name} must be finite and strictly positive.",
    )


def _matern_log_spectral_weight(
    eigenvalues: Array,
    length_scale: Array,
    smoothness: Array,
    spectral_dimension: float,
    /,
) -> Array:
    exponent = smoothness + 0.5 * spectral_dimension
    safe_eigenvalues = jnp.where(eigenvalues > 0.0, eigenvalues, 1.0)
    log_ratio = (
        2.0 * jnp.log(length_scale)
        + jnp.log(safe_eigenvalues)
        - jnp.log(2.0 * smoothness)
    )
    log_ratio = jnp.where(eigenvalues > 0.0, log_ratio, -jnp.inf)
    return -exponent * jnp.logaddexp(0.0, log_ratio)


def _hyperbolic_log_plancherel_density(
    frequencies: Array,
    dimension: int,
    /,
) -> Array:
    magnitude = jnp.abs(frequencies[:, 0])
    rho = 0.5 * (dimension - 1)
    if dimension % 2:
        shifts = jnp.arange(int(rho), dtype=magnitude.dtype)
        return jnp.sum(
            jnp.log(magnitude[:, None] ** 2 + shifts[None, :] ** 2),
            axis=-1,
        )
    shifts = jnp.arange(int(rho - 0.5), dtype=magnitude.dtype) + 0.5
    return (
        jnp.log(magnitude)
        + jnp.log(jnp.tanh(jnp.pi * magnitude))
        + jnp.sum(
            jnp.log(magnitude[:, None] ** 2 + shifts[None, :] ** 2),
            axis=-1,
        )
    )


def _spd_log_plancherel_density(frequencies: Array, /) -> Array:
    rank = int(frequencies.shape[1])
    row, column = np.triu_indices(rank, k=1)
    differences = jnp.abs(frequencies[:, row] - frequencies[:, column])
    return jnp.sum(
        jnp.log(differences) + jnp.log(jnp.tanh(jnp.pi * differences)),
        axis=-1,
    )


def _hyperbolic_points(points: ArrayLike, dimension: int, /) -> Array:
    array = jnp.asarray(points, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or int(array.shape[1]) != dimension + 1:
        raise ValueError(f"Hyperboloid points must have trailing size {dimension + 1}.")
    lorentz_norm = -(array[:, 0] ** 2) + jnp.sum(array[:, 1:] ** 2, axis=-1)
    return eqx.error_if(
        array,
        jnp.any(~jnp.isfinite(array))
        | jnp.any(array[:, 0] <= 0.0)
        | jnp.any(jnp.abs(lorentz_norm + 1.0) > 1e-6),
        "Hyperboloid points must lie on the future unit sheet.",
    )


class HyperbolicRandomFeatureKernel(AbstractFiniteFeatureKernel):
    """Fixed-noise Helgason plane-wave approximation on hyperbolic space."""

    proposal: NoncompactFeatureProposal
    length_scale: Array
    smoothness: Array
    dimension: int = eqx.field(static=True)

    def __init__(
        self,
        proposal: NoncompactFeatureProposal,
        length_scale: ArrayLike,
        smoothness: ArrayLike,
        /,
    ):
        if not isinstance(
            proposal, NoncompactFeatureProposal
        ) or not proposal.geometry_id.startswith("hyperbolic-H"):
            raise TypeError("proposal must be a hyperbolic NoncompactFeatureProposal.")
        if proposal.directions.ndim != 2:
            raise ValueError("Hyperbolic proposal directions must be rank two.")
        dimension = int(proposal.directions.shape[1])
        if proposal.geometry_id != f"hyperbolic-H{dimension}":
            raise ValueError(
                "Hyperbolic proposal geometry ID disagrees with its dimension."
            )
        direction_norms = np.sum(np.asarray(proposal.directions) ** 2, axis=-1)
        if dimension < 2 or not np.allclose(direction_norms, 1.0, rtol=1e-6, atol=1e-6):
            raise ValueError("Hyperbolic proposal directions must be unit vectors.")
        if proposal.spectral_rank != 1:
            raise ValueError("Hyperbolic proposals require scalar spectral frequencies.")
        self.proposal = proposal
        self.length_scale = _positive_parameter(length_scale, "length_scale")
        self.smoothness = _positive_parameter(smoothness, "smoothness")
        self.dimension = dimension

    def _log_importance_weights(self) -> Array:
        rho = 0.5 * (self.dimension - 1)
        eigenvalues = self.proposal.frequencies[:, 0] ** 2 + rho * rho
        target = _matern_log_spectral_weight(
            eigenvalues,
            self.length_scale,
            self.smoothness,
            float(self.dimension),
        ) + _hyperbolic_log_plancherel_density(
            self.proposal.frequencies,
            self.dimension,
        )
        return target - self.proposal.log_proposal_density

    def importance_diagnostics(self) -> ImportanceFeatureDiagnostics:
        return ImportanceFeatureDiagnostics(
            self._log_importance_weights(),
            self.proposal.proposal_id,
            finite_importance_variance=self.smoothness > 0.25,
        )

    def features(self, points: ArrayLike, /) -> Array:
        point_design = _hyperbolic_points(points, self.dimension)
        boundary_argument = (
            point_design[:, :1] - point_design[:, 1:] @ self.proposal.directions.T
        )
        busemann = jnp.log(boundary_argument)
        rho = 0.5 * (self.dimension - 1)
        phase = (
            busemann * self.proposal.frequencies[:, 0][None, :]
            + self.proposal.phases[None, :]
        )
        plane_waves = jnp.sqrt(2.0) * jnp.exp(-rho * busemann) * jnp.cos(phase)
        log_weights = self._log_importance_weights()
        scales = jnp.exp(0.5 * (log_weights - math.log(self.feature_rank)))
        features = plane_waves * scales[None, :]
        return eqx.error_if(
            features,
            jnp.any(~jnp.isfinite(features)),
            "Hyperbolic random features became nonfinite.",
        )

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_features = self.features(left)
        right_features = self.features(right)
        if left_features.shape[0] != 1 or right_features.shape[0] != 1:
            raise ValueError("pairwise requires one hyperbolic point per argument.")
        return jnp.dot(left_features[0], right_features[0])

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return self.features(left) @ self.features(right).T

    def diagonal(self, points: ArrayLike, /) -> Array:
        features = self.features(points)
        return jnp.sum(features * features, axis=-1)

    def resample(self, key: PRNGKeyArray, /) -> HyperbolicRandomFeatureKernel:
        """Return the same kernel parameters with an explicitly new proposal."""
        proposal = hyperbolic_feature_proposal(
            key,
            self.dimension,
            self.feature_rank,
            proposal_scale=self.proposal.proposal_scale,
        )
        return HyperbolicRandomFeatureKernel(proposal, self.length_scale, self.smoothness)

    @property
    def feature_rank(self) -> int:
        return self.proposal.sample_count

    @property
    def max_derivative_order(self) -> None:
        return None

    @property
    def is_unit_diagonal(self) -> bool:
        return False

    @property
    def kernel_id(self) -> str:
        return f"HyperbolicRandomFeatureKernel[{self.proposal.proposal_id}]"


def _spd_points(points: ArrayLike, dimension: int, /) -> Array:
    array = jnp.asarray(points, dtype=float)
    if array.shape == (dimension, dimension):
        array = array[None, :, :]
    elif array.ndim == 1 and int(array.size) == dimension * dimension:
        array = array.reshape((1, dimension, dimension))
    elif array.ndim == 2 and int(array.shape[1]) == dimension * dimension:
        array = array.reshape((array.shape[0], dimension, dimension))
    if array.ndim != 3 or tuple(array.shape[1:]) != (dimension, dimension):
        raise ValueError("SPD points must be square matrices or flattened matrices.")
    eigenvalues = jnp.linalg.eigvalsh(array)
    return eqx.error_if(
        array,
        jnp.any(~jnp.isfinite(array))
        | jnp.any(jnp.abs(array - jnp.swapaxes(array, -1, -2)) > 1e-7)
        | jnp.any(eigenvalues <= 0.0),
        "SPD points must be finite symmetric positive-definite matrices.",
    )


class SPDRandomFeatureKernel(AbstractFiniteFeatureKernel):
    """Fixed-noise spherical-plane-wave approximation on affine-invariant SPD(n)."""

    proposal: NoncompactFeatureProposal
    length_scale: Array
    smoothness: Array
    matrix_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        proposal: NoncompactFeatureProposal,
        length_scale: ArrayLike,
        smoothness: ArrayLike,
        /,
    ):
        if not isinstance(
            proposal, NoncompactFeatureProposal
        ) or not proposal.geometry_id.startswith("spd-SPD"):
            raise TypeError("proposal must be an SPD NoncompactFeatureProposal.")
        dimension = int(proposal.frequencies.shape[1])
        if proposal.geometry_id != f"spd-SPD{dimension}":
            raise ValueError("SPD proposal geometry ID disagrees with its spectral rank.")
        if dimension < 2 or proposal.directions.shape[1:] != (dimension, dimension):
            raise ValueError(
                "SPD proposal flags must match a spectral rank of at least two."
            )
        frames = np.asarray(proposal.directions)
        frame_gram = np.swapaxes(frames, -1, -2) @ frames
        if not np.allclose(
            frame_gram,
            np.eye(dimension)[None, :, :],
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError("SPD proposal flags must be orthogonal matrices.")
        self.proposal = proposal
        self.length_scale = _positive_parameter(length_scale, "length_scale")
        self.smoothness = _positive_parameter(smoothness, "smoothness")
        self.matrix_dimension = dimension

    def _log_importance_weights(self) -> Array:
        rank_indices = jnp.arange(self.matrix_dimension, dtype=float)
        rho = 0.25 * (self.matrix_dimension - 1.0 - 2.0 * rank_indices)
        eigenvalues = jnp.sum(self.proposal.frequencies**2, axis=-1) + jnp.sum(rho * rho)
        spectral_dimension = 0.5 * self.matrix_dimension * (self.matrix_dimension + 1)
        target = _matern_log_spectral_weight(
            eigenvalues,
            self.length_scale,
            self.smoothness,
            spectral_dimension,
        ) + _spd_log_plancherel_density(self.proposal.frequencies)
        return target - self.proposal.log_proposal_density

    def importance_diagnostics(self) -> ImportanceFeatureDiagnostics:
        return ImportanceFeatureDiagnostics(
            self._log_importance_weights(),
            self.proposal.proposal_id,
            finite_importance_variance=self.smoothness > 0.25,
        )

    def _horospherical_coordinates(self, points: Array, /) -> Array:
        def point_coordinates(point):
            def flag_coordinates(frame):
                rotated = frame.T @ point @ frame
                cholesky = jnp.linalg.cholesky(rotated)
                return 2.0 * jnp.log(jnp.diag(cholesky))

            return jax.vmap(flag_coordinates)(self.proposal.directions)

        return jax.vmap(point_coordinates)(points)

    def features(self, points: ArrayLike, /) -> Array:
        point_design = _spd_points(points, self.matrix_dimension)
        coordinates = self._horospherical_coordinates(point_design)
        rank_indices = jnp.arange(self.matrix_dimension, dtype=float)
        rho = 0.25 * (self.matrix_dimension - 1.0 - 2.0 * rank_indices)
        envelope = jnp.exp(-jnp.einsum("pmr,r->pm", coordinates, rho))
        phase = (
            jnp.einsum("pmr,mr->pm", coordinates, self.proposal.frequencies)
            + self.proposal.phases[None, :]
        )
        plane_waves = jnp.sqrt(2.0) * envelope * jnp.cos(phase)
        log_weights = self._log_importance_weights()
        scales = jnp.exp(0.5 * (log_weights - math.log(self.feature_rank)))
        features = plane_waves * scales[None, :]
        return eqx.error_if(
            features,
            jnp.any(~jnp.isfinite(features)),
            "SPD random features became nonfinite.",
        )

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_features = self.features(left)
        right_features = self.features(right)
        if left_features.shape[0] != 1 or right_features.shape[0] != 1:
            raise ValueError("pairwise requires one SPD point per argument.")
        return jnp.dot(left_features[0], right_features[0])

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return self.features(left) @ self.features(right).T

    def diagonal(self, points: ArrayLike, /) -> Array:
        features = self.features(points)
        return jnp.sum(features * features, axis=-1)

    def resample(self, key: PRNGKeyArray, /) -> SPDRandomFeatureKernel:
        """Return the same kernel parameters with an explicitly new proposal."""
        proposal = spd_feature_proposal(
            key,
            self.matrix_dimension,
            self.feature_rank,
            proposal_scale=self.proposal.proposal_scale,
        )
        return SPDRandomFeatureKernel(proposal, self.length_scale, self.smoothness)

    @property
    def feature_rank(self) -> int:
        return self.proposal.sample_count

    @property
    def max_derivative_order(self) -> None:
        return None

    @property
    def is_unit_diagonal(self) -> bool:
        return False

    @property
    def kernel_id(self) -> str:
        return f"SPDRandomFeatureKernel[{self.proposal.proposal_id}]"


__all__ = [
    "HyperbolicRandomFeatureKernel",
    "ImportanceFeatureDiagnostics",
    "NoncompactFeatureProposal",
    "SPDRandomFeatureKernel",
    "hyperbolic_feature_proposal",
    "spd_feature_proposal",
]
