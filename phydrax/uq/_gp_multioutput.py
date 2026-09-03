#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.kernels import AbstractPositiveDefiniteKernel

from .._strict import StrictModule
from ._constraint_conditioning import (
    ConstraintLikelihoodTerm,
    LinearGaussianConstraintConditioner,
)
from ._gp_actions import AbstractGaussianProcessActionPolicy
from ._gp_computation_aware import GaussianProcessComputationPolicy
from ._gp_computation_structured import (
    StructuredComputationAwareGaussianProcessFactor,
)
from ._gp_condition import _sample_gaussian_psd
from ._predictive import PredictiveField, SampleAxis


class MultiOutputDesign(StrictModule):
    """Flat observations identified by physical point and output channel."""

    points: Array
    output_index: Array
    source_index: Array
    output_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        points: ArrayLike,
        output_index: ArrayLike,
        /,
        *,
        output_names: tuple[str, ...],
        source_index: ArrayLike | None = None,
    ):
        point_array = _as_points(points)
        output_array = jnp.asarray(output_index, dtype=jnp.int32)
        names = _output_names(output_names)
        count = int(point_array.shape[0])
        if count <= 0:
            raise ValueError("Multi-output designs must contain at least one row.")
        if output_array.shape != (count,):
            raise ValueError("output_index must contain one index per design row.")
        source_array = (
            jnp.arange(count, dtype=jnp.int32)
            if source_index is None
            else jnp.asarray(source_index, dtype=jnp.int32)
        )
        if source_array.shape != (count,):
            raise ValueError("source_index must contain one index per design row.")
        if not bool(jnp.all(jnp.isfinite(point_array))):
            raise ValueError("Multi-output design points must be finite.")
        if bool(jnp.any(output_array < 0)) or bool(jnp.any(output_array >= len(names))):
            raise ValueError("output_index contains an unknown output channel.")
        if bool(jnp.any(source_array < 0)):
            raise ValueError("source_index must be nonnegative.")
        self.points = point_array
        self.output_index = output_array
        self.source_index = source_array
        self.output_names = names

    @classmethod
    def from_dense(
        cls,
        points: ArrayLike,
        /,
        *,
        output_names: tuple[str, ...],
        mask: ArrayLike | None = None,
    ) -> MultiOutputDesign:
        """Flatten a point-by-output layout, optionally omitting missing channels."""
        point_array = _as_points(points)
        names = _output_names(output_names)
        point_count = int(point_array.shape[0])
        output_count = len(names)
        active = (
            jnp.ones((point_count, output_count), dtype=bool)
            if mask is None
            else jnp.asarray(mask, dtype=bool)
        )
        if active.shape != (point_count, output_count):
            raise ValueError("mask must have shape (point, output).")
        flat_active = active.reshape((-1,))
        if not bool(jnp.any(flat_active)):
            raise ValueError("A multi-output design must retain at least one row.")
        repeated_points = jnp.repeat(point_array, output_count, axis=0)
        output_index = jnp.tile(jnp.arange(output_count, dtype=jnp.int32), point_count)
        source_index = jnp.repeat(jnp.arange(point_count, dtype=jnp.int32), output_count)
        return cls(
            repeated_points[flat_active],
            output_index[flat_active],
            output_names=names,
            source_index=source_index[flat_active],
        )

    @property
    def num_observations(self) -> int:
        return int(self.points.shape[0])

    @property
    def num_outputs(self) -> int:
        return len(self.output_names)

    @property
    def num_sources(self) -> int:
        return int(jnp.max(self.source_index)) + 1

    def flatten(self, values: ArrayLike, /, *, name: str = "values") -> Array:
        """Align flat or dense point-by-output values to this design."""
        array = jnp.asarray(values, dtype=float)
        if array.ndim == 1:
            if array.shape != (self.num_observations,):
                raise ValueError(f"{name} must align with multi-output design rows.")
            return array
        if array.ndim != 2 or int(array.shape[1]) != self.num_outputs:
            raise ValueError(
                f"{name} must be flat or have one column per output channel."
            )
        if int(array.shape[0]) == self.num_observations:
            row_index = jnp.arange(self.num_observations, dtype=jnp.int32)
            return array[row_index, self.output_index]
        if int(array.shape[0]) == self.num_sources:
            return array[self.source_index, self.output_index]
        raise ValueError(f"{name} must align with design rows or source points.")

    def dense(self, values: ArrayLike, /, *, fill_value: float = jnp.nan) -> Array:
        """Scatter flat values into source-by-output form, filling absent channels."""
        flat = self.flatten(values)
        result = jnp.full(
            (self.num_sources, self.num_outputs),
            fill_value,
            dtype=flat.dtype,
        )
        return result.at[self.source_index, self.output_index].set(flat)


class Coregionalization(StrictModule):
    """Positive-semidefinite output covariance B = W W transpose + diag(d squared)."""

    weights: Array
    diagonal_scale: Array
    output_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        weights: ArrayLike,
        diagonal_scale: ArrayLike,
        /,
        *,
        output_names: tuple[str, ...],
    ):
        names = _output_names(output_names)
        weight_array = jnp.asarray(weights, dtype=float)
        diagonal_array = jnp.asarray(diagonal_scale, dtype=float)
        if weight_array.ndim != 2 or weight_array.shape[0] != len(names):
            raise ValueError("weights must have shape (output, latent_rank).")
        if weight_array.shape[1] <= 0:
            raise ValueError("Coregionalization latent rank must be positive.")
        if diagonal_array.shape != (len(names),):
            raise ValueError("diagonal_scale must contain one value per output.")
        self.weights = eqx.error_if(
            weight_array,
            jnp.any(~jnp.isfinite(weight_array)),
            "Coregionalization weights must be finite.",
        )
        self.diagonal_scale = eqx.error_if(
            diagonal_array,
            jnp.any(~jnp.isfinite(diagonal_array)) | jnp.any(diagonal_array < 0.0),
            "Coregionalization diagonal scales must be finite and nonnegative.",
        )
        self.output_names = names

    @property
    def covariance(self) -> Array:
        return self.weights @ self.weights.T + jnp.diag(
            self.diagonal_scale * self.diagonal_scale
        )

    @property
    def num_outputs(self) -> int:
        return len(self.output_names)

    @property
    def kernel_id(self) -> str:
        return f"Coregionalization[rank={self.weights.shape[1]}]"


class AbstractMultiOutputKernel(StrictModule):
    """Positive-definite covariance over flat point-and-output designs."""

    @abstractmethod
    def matrix(
        self,
        left: MultiOutputDesign,
        right: MultiOutputDesign,
        /,
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def diagonal(self, design: MultiOutputDesign, /) -> Array:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_names(self) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def max_derivative_order(self) -> int | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def kernel_id(self) -> str:
        raise NotImplementedError


class IntrinsicCoregionalizationKernel(AbstractMultiOutputKernel):
    """One spatial kernel multiplied by one output coregionalization matrix."""

    spatial_kernel: AbstractPositiveDefiniteKernel
    coregionalization: Coregionalization

    def __init__(
        self,
        spatial_kernel: AbstractPositiveDefiniteKernel,
        coregionalization: Coregionalization,
        /,
    ):
        if not isinstance(spatial_kernel, AbstractPositiveDefiniteKernel):
            raise TypeError("spatial_kernel must be a positive-definite kernel.")
        if not isinstance(coregionalization, Coregionalization):
            raise TypeError("coregionalization must be a Coregionalization.")
        self.spatial_kernel = spatial_kernel
        self.coregionalization = coregionalization

    def matrix(
        self,
        left: MultiOutputDesign,
        right: MultiOutputDesign,
        /,
    ) -> Array:
        _validate_design_pair(left, right, output_names=self.output_names)
        output = self.coregionalization.covariance
        output_block = output[left.output_index[:, None], right.output_index[None, :]]
        return self.spatial_kernel.matrix(left.points, right.points) * output_block

    def diagonal(self, design: MultiOutputDesign, /) -> Array:
        _validate_design(design, output_names=self.output_names)
        output_diagonal = jnp.diag(self.coregionalization.covariance)
        return (
            self.spatial_kernel.diagonal(design.points)
            * output_diagonal[design.output_index]
        )

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.coregionalization.output_names

    @property
    def max_derivative_order(self) -> int | None:
        return self.spatial_kernel.max_derivative_order

    @property
    def kernel_id(self) -> str:
        return (
            "IntrinsicCoregionalizationKernel["
            f"{self.spatial_kernel.kernel_id},{self.coregionalization.kernel_id}]"
        )


class LinearModelCoregionalizationKernel(AbstractMultiOutputKernel):
    """Finite sum of spatial kernels with independent coregionalizations."""

    spatial_kernels: tuple[AbstractPositiveDefiniteKernel, ...]
    coregionalizations: tuple[Coregionalization, ...]
    _output_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        components: tuple[tuple[AbstractPositiveDefiniteKernel, Coregionalization], ...],
        /,
    ):
        if not components:
            raise ValueError("LinearModelCoregionalizationKernel needs one component.")
        spatial_kernels: list[AbstractPositiveDefiniteKernel] = []
        coregionalizations: list[Coregionalization] = []
        output_names: tuple[str, ...] | None = None
        for spatial_kernel, coregionalization in components:
            if not isinstance(spatial_kernel, AbstractPositiveDefiniteKernel):
                raise TypeError("Each spatial component must be a kernel.")
            if not isinstance(coregionalization, Coregionalization):
                raise TypeError("Each output component must be a Coregionalization.")
            if output_names is None:
                output_names = coregionalization.output_names
            elif coregionalization.output_names != output_names:
                raise ValueError("LMC components must use identical output names.")
            spatial_kernels.append(spatial_kernel)
            coregionalizations.append(coregionalization)
        if output_names is None:
            raise RuntimeError("LMC construction produced no output vocabulary.")
        self.spatial_kernels = tuple(spatial_kernels)
        self.coregionalizations = tuple(coregionalizations)
        self._output_names = output_names

    def matrix(
        self,
        left: MultiOutputDesign,
        right: MultiOutputDesign,
        /,
    ) -> Array:
        _validate_design_pair(left, right, output_names=self.output_names)
        value = self._component_matrix(0, left, right)
        for index in range(1, len(self.spatial_kernels)):
            value = value + self._component_matrix(index, left, right)
        return value

    def diagonal(self, design: MultiOutputDesign, /) -> Array:
        _validate_design(design, output_names=self.output_names)
        value = self._component_diagonal(0, design)
        for index in range(1, len(self.spatial_kernels)):
            value = value + self._component_diagonal(index, design)
        return value

    def _component_matrix(
        self,
        index: int,
        left: MultiOutputDesign,
        right: MultiOutputDesign,
    ) -> Array:
        output = self.coregionalizations[index].covariance
        output_block = output[left.output_index[:, None], right.output_index[None, :]]
        return (
            self.spatial_kernels[index].matrix(left.points, right.points) * output_block
        )

    def _component_diagonal(self, index: int, design: MultiOutputDesign) -> Array:
        output_diagonal = jnp.diag(self.coregionalizations[index].covariance)
        return (
            self.spatial_kernels[index].diagonal(design.points)
            * output_diagonal[design.output_index]
        )

    @property
    def output_names(self) -> tuple[str, ...]:
        return self._output_names

    @property
    def max_derivative_order(self) -> int | None:
        finite = tuple(
            order
            for kernel in self.spatial_kernels
            if (order := kernel.max_derivative_order) is not None
        )
        return None if not finite else min(finite)

    @property
    def kernel_id(self) -> str:
        components = ",".join(
            f"({kernel.kernel_id},{coregionalization.kernel_id})"
            for kernel, coregionalization in zip(
                self.spatial_kernels, self.coregionalizations, strict=True
            )
        )
        return f"LinearModelCoregionalizationKernel[{components}]"


class MultiOutputGaussianProcessLikelihoodState(StrictModule):
    """Resolved multi-output kernel, noise layout, and numerical jitter."""

    kernel: AbstractMultiOutputKernel
    noise_scale: Array
    jitter: Array
    noise_layout: Literal["output", "observation"] = eqx.field(static=True)

    def __init__(
        self,
        *,
        kernel: AbstractMultiOutputKernel,
        noise_scale: ArrayLike,
        noise_layout: Literal["output", "observation"] = "output",
        jitter: ArrayLike = 1e-8,
    ):
        if not isinstance(kernel, AbstractMultiOutputKernel):
            raise TypeError("kernel must be an AbstractMultiOutputKernel.")
        noise = jnp.asarray(noise_scale, dtype=float)
        if noise.ndim > 1 or (noise.ndim == 1 and noise.shape[0] == 0):
            raise ValueError("noise_scale must be scalar or a nonempty vector.")
        if noise_layout not in ("output", "observation"):
            raise ValueError("noise_layout must be 'output' or 'observation'.")
        if (
            noise.ndim == 1
            and noise_layout == "output"
            and noise.shape[0] != len(kernel.output_names)
        ):
            raise ValueError("Output noise must contain one scale per output.")
        jitter_array = jnp.asarray(jitter, dtype=float)
        if jitter_array.ndim != 0:
            raise ValueError("jitter must be scalar.")
        self.kernel = kernel
        self.noise_scale = eqx.error_if(
            noise,
            jnp.any(~jnp.isfinite(noise)) | jnp.any(noise < 0.0),
            "noise_scale must be finite and nonnegative.",
        )
        self.jitter = eqx.error_if(
            jitter_array,
            ~jnp.isfinite(jitter_array) | (jitter_array <= 0.0),
            "jitter must be finite and strictly positive.",
        )
        self.noise_layout = noise_layout

    def observation_noise(self, design: MultiOutputDesign, /) -> Array:
        _validate_design(design, output_names=self.kernel.output_names)
        if self.noise_scale.ndim == 0:
            return jnp.broadcast_to(self.noise_scale, (design.num_observations,))
        if self.noise_layout == "output":
            return self.noise_scale[design.output_index]
        if self.noise_scale.shape != (design.num_observations,):
            raise ValueError("Observation noise must contain one scale per design row.")
        return self.noise_scale


class MultiOutputGaussianProcessCondition(StrictModule):
    """Conditioned latent discrepancy over a flat heterotopic query design."""

    design: MultiOutputDesign
    mean: Array
    covariance: Array
    variance: Array

    def __init__(
        self,
        *,
        design: MultiOutputDesign,
        mean: ArrayLike,
        covariance: ArrayLike,
        variance: ArrayLike,
    ):
        _validate_design(design, output_names=design.output_names)
        mean_array = design.flatten(mean, name="conditioned GP mean")
        variance_array = design.flatten(variance, name="conditioned GP variance")
        covariance_array = jnp.asarray(covariance, dtype=float)
        count = design.num_observations
        if covariance_array.shape != (count, count):
            raise ValueError("Conditioned covariance must be square over query rows.")
        self.design = design
        self.mean = mean_array
        self.covariance = covariance_array
        self.variance = variance_array

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.design.output_names

    def dense_mean(self, *, fill_value: float = jnp.nan) -> Array:
        return self.design.dense(self.mean, fill_value=fill_value)

    def dense_variance(self, *, fill_value: float = jnp.nan) -> Array:
        return self.design.dense(self.variance, fill_value=fill_value)

    def sample(self, key: Array, /, *, num_samples: int) -> Array:
        """Draw coherent functions in the exact query-row ordering."""
        count = int(num_samples)
        if count <= 0:
            raise ValueError("num_samples must be positive.")
        return _sample_gaussian_psd(
            self.mean,
            self.covariance,
            key,
            num_samples=count,
        )

    def predictive_field(
        self,
        base_mean: ArrayLike,
        key: Array,
        /,
        *,
        num_samples: int,
        observation_variance: ArrayLike | None = None,
        sample_dim: str = "__phydra_uq_discrepancy",
        observation_dim: str = "observation",
    ) -> PredictiveField:
        """Add flat discrepancy draws without inventing absent output channels."""
        base = self.design.flatten(base_mean, name="multi-output physical mean")
        data = base + self.sample(key, num_samples=num_samples)
        valid_data = jnp.all(jnp.isfinite(data), axis=1)
        conditional = None
        if observation_variance is not None:
            variance = self.design.flatten(
                observation_variance,
                name="observation variance",
            )
            if bool(jnp.any(variance < 0.0)):
                raise ValueError("observation_variance must be nonnegative.")
            conditional = cx.Field(variance, dims=(observation_dim,))
        return PredictiveField(
            cx.Field(data, dims=(sample_dim, observation_dim)),
            (SampleAxis(sample_dim, "epistemic"),),
            conditional_variance=conditional,
            valid=cx.Field(valid_data, dims=(sample_dim,)),
        )


class MultiOutputGaussianProcessDiscrepancy(StrictModule):
    """Exact GP discrepancy over flat isotopic or heterotopic observations."""

    design: MultiOutputDesign
    observations: Array

    def __init__(
        self,
        design: MultiOutputDesign,
        observations: ArrayLike,
        /,
    ):
        if not isinstance(design, MultiOutputDesign):
            raise TypeError("design must be a MultiOutputDesign.")
        values = design.flatten(observations, name="multi-output GP observations")
        if not bool(jnp.all(jnp.isfinite(values))):
            raise ValueError("Multi-output GP observations must be finite.")
        self.design = design
        self.observations = values

    @classmethod
    def from_dense(
        cls,
        points: ArrayLike,
        observations: ArrayLike,
        /,
        *,
        output_names: tuple[str, ...] | None = None,
        mask: ArrayLike | None = None,
    ) -> MultiOutputGaussianProcessDiscrepancy:
        """Flatten a dense observation table while omitting masked channels."""
        values = jnp.asarray(observations, dtype=float)
        if values.ndim != 2 or int(values.shape[1]) <= 0:
            raise ValueError("observations must have shape (point, output).")
        names = (
            tuple(f"output_{index}" for index in range(values.shape[1]))
            if output_names is None
            else _output_names(output_names)
        )
        if len(names) != int(values.shape[1]):
            raise ValueError("output_names must align with observation columns.")
        active = (
            jnp.ones(values.shape, dtype=bool)
            if mask is None
            else jnp.asarray(mask, dtype=bool)
        )
        if active.shape != values.shape:
            raise ValueError("mask must have the same shape as observations.")
        if not bool(jnp.all(jnp.isfinite(values[active]))):
            raise ValueError("Observed multi-output values must be finite.")
        design = MultiOutputDesign.from_dense(
            points,
            output_names=names,
            mask=active,
        )
        return cls(design, values.reshape((-1,))[active.reshape((-1,))])

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.design.output_names

    @property
    def num_outputs(self) -> int:
        return self.design.num_outputs

    def residual(self, physical_mean: ArrayLike, /) -> Array:
        mean = self.design.flatten(physical_mean, name="multi-output physical mean")
        return self.observations - mean

    def log_marginal_likelihood(
        self,
        physical_mean: ArrayLike,
        /,
        *,
        state: MultiOutputGaussianProcessLikelihoodState,
    ) -> Array:
        """Marginalize the correlated latent discrepancy."""
        _validate_state(state, self.design)
        noise = state.observation_noise(self.design)
        residual = self.residual(physical_mean)
        likelihood = ConstraintLikelihoodTerm(
            residual,
            noise_scale=noise,
            likelihood_id="multi-output-gp-constraint",
        )
        conditioner = LinearGaussianConstraintConditioner(
            numerical_jitter=state.jitter,
            rank_tolerance=state.jitter,
        )
        return conditioner.log_evidence_from_covariance(
            jnp.zeros_like(residual),
            state.kernel.matrix(self.design, self.design),
            likelihood,
        )

    def condition(
        self,
        physical_mean: ArrayLike,
        query_design: MultiOutputDesign,
        /,
        *,
        state: MultiOutputGaussianProcessLikelihoodState,
    ) -> MultiOutputGaussianProcessCondition:
        """Condition a heterotopic latent discrepancy at another flat design."""
        _validate_state(state, self.design)
        _validate_design(query_design, output_names=self.output_names)
        noise = state.observation_noise(self.design)
        residual = self.residual(physical_mean)
        likelihood = ConstraintLikelihoodTerm(
            residual,
            noise_scale=noise,
            likelihood_id="multi-output-gp-constraint",
        )
        conditioner = LinearGaussianConstraintConditioner(
            numerical_jitter=state.jitter,
            rank_tolerance=state.jitter,
        )
        conditioned = conditioner.condition_from_covariances(
            jnp.zeros((query_design.num_observations,), dtype=residual.dtype),
            state.kernel.matrix(query_design, query_design),
            jnp.zeros_like(residual),
            state.kernel.matrix(self.design, self.design),
            state.kernel.matrix(query_design, self.design),
            likelihood,
        )
        covariance = conditioned.posterior_covariance
        return MultiOutputGaussianProcessCondition(
            design=query_design,
            mean=conditioned.posterior_mean,
            covariance=covariance,
            variance=jnp.maximum(jnp.diag(covariance), 0.0),
        )

    def computation_factor(
        self,
        *,
        state: MultiOutputGaussianProcessLikelihoodState,
        actions: AbstractGaussianProcessActionPolicy,
        computation: GaussianProcessComputationPolicy | None = None,
        residual: ArrayLike | None = None,
    ) -> StructuredComputationAwareGaussianProcessFactor:
        """Prepare an action-projected heterotopic covariance factor."""
        _validate_state(state, self.design)
        policy = (
            GaussianProcessComputationPolicy() if computation is None else computation
        )
        if not isinstance(policy, GaussianProcessComputationPolicy):
            raise TypeError("computation must be a GaussianProcessComputationPolicy.")
        return StructuredComputationAwareGaussianProcessFactor(
            state.kernel.matrix(self.design, self.design),
            state.observation_noise(self.design),
            state.jitter,
            actions,
            residual=residual,
            max_factorization_bytes=policy.max_factor_storage_bytes,
        )

    def computation_condition(
        self,
        physical_mean: ArrayLike,
        query_design: MultiOutputDesign,
        /,
        *,
        state: MultiOutputGaussianProcessLikelihoodState,
        actions: AbstractGaussianProcessActionPolicy,
        computation: GaussianProcessComputationPolicy | None = None,
    ) -> MultiOutputGaussianProcessCondition:
        """Condition heterotopic queries through action-projected observations."""
        _validate_state(state, self.design)
        _validate_design(query_design, output_names=self.output_names)
        residual = self.residual(physical_mean)
        factor = self.computation_factor(
            state=state,
            actions=actions,
            computation=computation,
            residual=residual,
        )
        mean, covariance, variance = factor.condition(
            residual,
            state.kernel.matrix(query_design, self.design),
            state.kernel.matrix(query_design, query_design),
        )
        return MultiOutputGaussianProcessCondition(
            design=query_design,
            mean=mean,
            covariance=covariance,
            variance=variance,
        )


def _as_points(value: ArrayLike) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError("GP points must have shape (point, coordinate).")
    return array


def _output_names(value: tuple[str, ...], /) -> tuple[str, ...]:
    names = tuple(value)
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("output_names must be distinct nonempty strings.")
    if any(not isinstance(name, str) for name in names):
        raise TypeError("output_names must contain only strings.")
    return names


def _validate_design(
    design: MultiOutputDesign,
    /,
    *,
    output_names: tuple[str, ...],
) -> None:
    if not isinstance(design, MultiOutputDesign):
        raise TypeError("Expected a MultiOutputDesign.")
    if design.output_names != output_names:
        raise ValueError("Multi-output design channel names do not match the kernel.")


def _validate_design_pair(
    left: MultiOutputDesign,
    right: MultiOutputDesign,
    /,
    *,
    output_names: tuple[str, ...],
) -> None:
    _validate_design(left, output_names=output_names)
    _validate_design(right, output_names=output_names)
    if left.points.shape[1] != right.points.shape[1]:
        raise ValueError("Multi-output point designs need equal coordinate size.")


def _validate_state(
    state: MultiOutputGaussianProcessLikelihoodState,
    design: MultiOutputDesign,
    /,
) -> None:
    if not isinstance(state, MultiOutputGaussianProcessLikelihoodState):
        raise TypeError("state must be a MultiOutputGaussianProcessLikelihoodState.")
    _validate_design(design, output_names=state.kernel.output_names)


__all__ = [
    "AbstractMultiOutputKernel",
    "Coregionalization",
    "IntrinsicCoregionalizationKernel",
    "LinearModelCoregionalizationKernel",
    "MultiOutputDesign",
    "MultiOutputGaussianProcessCondition",
    "MultiOutputGaussianProcessDiscrepancy",
    "MultiOutputGaussianProcessLikelihoodState",
]
