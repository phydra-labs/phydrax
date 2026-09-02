#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import AbstractLinearOperator, LinearSolveStatus
from ._gp_actions import (
    _ResolvedGaussianProcessActions,
    AbstractGaussianProcessActionPolicy,
    GaussianProcessActionKind,
)
from ._gp_computation_backend import (
    _build_projected_state,
    _computation_aware_elbo,
    _posterior_mean,
    _posterior_mean_from_solved_geometry,
    _predictive_geometry,
    _ProjectedGaussianProcessState,
)
from ._gp_condition import GaussianProcessCondition
from ._gp_likelihood import GaussianProcessLikelihoodState
from ._gp_scalar import (
    _as_design,
    _as_vector,
    _field_data,
    _query_output_dims,
    _require_state,
    _validated_factor_points,
    _validated_observations,
)


ComputationAwareGaussianProcessStatus: TypeAlias = Literal[0, 1, 2, 3, 4]
COMPUTATION_AWARE_GP_SUCCESS: ComputationAwareGaussianProcessStatus = 0
COMPUTATION_AWARE_GP_NONFINITE: ComputationAwareGaussianProcessStatus = 1
COMPUTATION_AWARE_GP_PROJECTED_NOISE_FAILURE: ComputationAwareGaussianProcessStatus = 2
COMPUTATION_AWARE_GP_PROJECTED_COVARIANCE_FAILURE: ComputationAwareGaussianProcessStatus = 3
COMPUTATION_AWARE_GP_CONDITION_LIMIT: ComputationAwareGaussianProcessStatus = 4


class GaussianProcessComputationPolicy(StrictModule):
    """Resource and conditioning limits for computation-aware GP execution."""

    max_workspace_bytes: int = eqx.field(static=True)
    max_factor_storage_bytes: int = eqx.field(static=True)
    max_condition_covariance_bytes: int = eqx.field(static=True)
    checkpoint_kernel_blocks: bool = eqx.field(static=True)
    projected_condition_limit: float | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_workspace_bytes: int = 64 * 1024 * 1024,
        max_factor_storage_bytes: int = 512 * 1024 * 1024,
        max_condition_covariance_bytes: int = 64 * 1024 * 1024,
        checkpoint_kernel_blocks: bool = True,
        projected_condition_limit: float | None = None,
    ):
        limits = (
            int(max_workspace_bytes),
            int(max_factor_storage_bytes),
            int(max_condition_covariance_bytes),
        )
        if any(value < 1 for value in limits):
            raise ValueError("Computation-aware GP byte limits must be positive.")
        condition_limit = (
            None
            if projected_condition_limit is None
            else float(projected_condition_limit)
        )
        if condition_limit is not None and (
            not math.isfinite(condition_limit) or condition_limit <= 1.0
        ):
            raise ValueError("projected_condition_limit must exceed one.")
        (
            self.max_workspace_bytes,
            self.max_factor_storage_bytes,
            self.max_condition_covariance_bytes,
        ) = limits
        self.checkpoint_kernel_blocks = bool(checkpoint_kernel_blocks)
        self.projected_condition_limit = condition_limit


class ComputationAwareGaussianProcessDiagnostics(StrictModule):
    """Projected-solve, action, and resource evidence for one reusable factor."""

    valid: Array
    status: Array
    projected_noise_condition: Array
    projected_covariance_condition: Array
    action_active_mask: Array
    action_residual_history: Array
    action_breakdown_mask: Array
    action_convergence_mask: Array
    action_selected_indices: Array
    active_action_count: Array
    action_kind: GaussianProcessActionKind = eqx.field(static=True)
    action_id: str = eqx.field(static=True)
    num_observations: int = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)
    action_storage_elements: int = eqx.field(static=True)
    factor_storage_elements: int = eqx.field(static=True)
    factor_storage_bytes: int = eqx.field(static=True)
    kernel_workspace_bytes: int = eqx.field(static=True)
    kernel_entry_count: int = eqx.field(static=True)
    kernel_row_batch_size: int = eqx.field(static=True)
    structurally_sparse_actions: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        projected: _ProjectedGaussianProcessState,
        actions: _ResolvedGaussianProcessActions,
        factor_storage_elements: int,
        condition_limit: float | None,
    ):
        noise_condition = projected.noise_factor.diagnostics.condition_estimate
        covariance_condition = projected.covariance_factor.diagnostics.condition_estimate
        finite = (
            jnp.all(jnp.isfinite(projected.kernel_action))
            & jnp.all(jnp.isfinite(projected.projected_noise))
            & jnp.all(jnp.isfinite(projected.projected_covariance))
        )
        noise_success = projected.noise_factor.status == int(LinearSolveStatus.SUCCESS)
        covariance_success = projected.covariance_factor.status == int(
            LinearSolveStatus.SUCCESS
        )
        if condition_limit is None:
            condition_success = jnp.asarray(True)
        else:
            condition_success = (
                jnp.isfinite(noise_condition)
                & jnp.isfinite(covariance_condition)
                & (noise_condition <= condition_limit)
                & (covariance_condition <= condition_limit)
            )
        status = jnp.where(
            ~finite,
            COMPUTATION_AWARE_GP_NONFINITE,
            jnp.where(
                ~noise_success,
                COMPUTATION_AWARE_GP_PROJECTED_NOISE_FAILURE,
                jnp.where(
                    ~covariance_success,
                    COMPUTATION_AWARE_GP_PROJECTED_COVARIANCE_FAILURE,
                    jnp.where(
                        ~condition_success,
                        COMPUTATION_AWARE_GP_CONDITION_LIMIT,
                        COMPUTATION_AWARE_GP_SUCCESS,
                    ),
                ),
            ),
        ).astype(jnp.int32)
        itemsize = int(projected.kernel_action.dtype.itemsize)
        self.valid = status == COMPUTATION_AWARE_GP_SUCCESS
        self.status = status
        self.projected_noise_condition = noise_condition
        self.projected_covariance_condition = covariance_condition
        self.action_active_mask = actions.active_mask
        self.action_residual_history = actions.residual_history
        self.action_breakdown_mask = actions.breakdown_mask
        self.action_convergence_mask = actions.convergence_mask
        self.action_selected_indices = actions.selected_indices
        self.active_action_count = jnp.sum(actions.active_mask, dtype=jnp.int32)
        self.action_kind = actions.kind
        self.action_id = actions.action_id
        self.num_observations = actions.num_observations
        self.num_actions = actions.num_actions
        self.action_storage_elements = actions.storage_elements
        self.factor_storage_elements = int(factor_storage_elements)
        self.factor_storage_bytes = int(factor_storage_elements) * itemsize
        self.kernel_workspace_bytes = projected.workspace_bytes
        self.kernel_entry_count = projected.kernel_entry_count
        self.kernel_row_batch_size = projected.row_batch_size
        self.structurally_sparse_actions = actions.structurally_sparse


class ComputationAwareGaussianProcessFactor(StrictModule):
    """Reusable action-projected covariance geometry for one scalar GP state."""

    observation_points: Array
    actions: AbstractLinearOperator
    projected: _ProjectedGaussianProcessState
    state: GaussianProcessLikelihoodState
    computation: GaussianProcessComputationPolicy
    diagnostics: ComputationAwareGaussianProcessDiagnostics

    def __init__(
        self,
        observation_points: ArrayLike,
        /,
        *,
        state: GaussianProcessLikelihoodState,
        actions: AbstractGaussianProcessActionPolicy,
        computation: GaussianProcessComputationPolicy | None = None,
        residual: ArrayLike | None = None,
    ):
        points = _validated_factor_points(observation_points)
        _require_state(state)
        if not isinstance(actions, AbstractGaussianProcessActionPolicy):
            raise TypeError("actions must implement AbstractGaussianProcessActionPolicy.")
        policy = (
            GaussianProcessComputationPolicy() if computation is None else computation
        )
        if not isinstance(policy, GaussianProcessComputationPolicy):
            raise TypeError("computation must be a GaussianProcessComputationPolicy.")
        resolved = actions.resolve(
            points,
            state=state,
            residual=None
            if residual is None
            else _as_vector(residual, name="GP residual"),
        )
        if resolved.num_observations != int(points.shape[0]):
            raise ValueError("Resolved GP actions must align with observation points.")
        projected = _build_projected_state(
            points,
            kernel=state.kernel,
            noise_scale=state.noise_scale,
            jitter=state.jitter,
            actions=resolved.operator,
            active_mask=resolved.active_mask,
            max_workspace_bytes=policy.max_workspace_bytes,
            max_factorization_bytes=policy.max_factor_storage_bytes,
            checkpoint=policy.checkpoint_kernel_blocks,
        )
        observation_count = resolved.num_observations
        action_count = resolved.num_actions
        factor_storage_elements = (
            resolved.storage_elements
            + int(projected.kernel_action.size)
            + int(projected.prior_diagonal.size)
            + int(projected.effective_observation_variance.size)
            + int(projected.projected_noise.size)
            + int(projected.projected_covariance.size)
            + 4 * action_count * action_count
        )
        factor_storage_bytes = factor_storage_elements * int(points.dtype.itemsize)
        if factor_storage_bytes > policy.max_factor_storage_bytes:
            raise ValueError(
                "Computation-aware GP retained factor storage exceeds its policy limit."
            )
        diagnostics = ComputationAwareGaussianProcessDiagnostics(
            projected=projected,
            actions=resolved,
            factor_storage_elements=factor_storage_elements,
            condition_limit=policy.projected_condition_limit,
        )
        self.observation_points = points
        self.actions = resolved.operator
        self.projected = projected
        self.state = state
        self.computation = policy
        self.diagnostics = diagnostics

    @property
    def factor_storage_elements(self) -> int:
        return self.diagnostics.factor_storage_elements

    def elbo(self, residual: ArrayLike, /) -> Array:
        """Evaluate the full-data computation-aware evidence lower bound."""
        values = self._validated_residual(residual)
        values = eqx.error_if(
            values,
            ~self.diagnostics.valid,
            "Computation-aware GP factor is numerically invalid.",
        )
        return _computation_aware_elbo(
            values,
            actions=self.actions,
            projected=self.projected,
        )

    def latent_moments(
        self,
        residual: ArrayLike,
        query_points: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        """Return mean and latent variance without a dense query covariance."""
        values = self._validated_residual(residual)
        query = _validated_factor_points(query_points)
        geometry = _predictive_geometry(
            query,
            self.observation_points,
            kernel=self.state.kernel,
            actions=self.actions,
            covariance_factor=self.projected.covariance_factor,
            max_workspace_bytes=self.computation.max_workspace_bytes,
            checkpoint=self.computation.checkpoint_kernel_blocks,
            full_covariance=False,
        )
        mean = _posterior_mean(
            values,
            actions=self.actions,
            covariance_factor=self.projected.covariance_factor,
            query_action=geometry.query_action,
        )
        return mean, geometry.variance

    def conditioner(
        self,
        query_points: ArrayLike,
        /,
        *,
        output_dim: str | None = "point",
    ) -> "ComputationAwareGaussianProcessConditioner":
        """Precompute low-rank residual geometry and full query covariance."""
        original_query = query_points
        query = _validated_factor_points(_field_data(query_points))
        covariance_bytes = int(query.shape[0]) ** 2 * int(query.dtype.itemsize)
        if covariance_bytes > self.computation.max_condition_covariance_bytes:
            raise ValueError(
                "Computation-aware GP query covariance exceeds its policy limit; "
                "use latent_moments for diagonal prediction."
            )
        geometry = _predictive_geometry(
            query,
            self.observation_points,
            kernel=self.state.kernel,
            actions=self.actions,
            covariance_factor=self.projected.covariance_factor,
            max_workspace_bytes=self.computation.max_workspace_bytes,
            checkpoint=self.computation.checkpoint_kernel_blocks,
            full_covariance=True,
        )
        assert geometry.covariance is not None
        return ComputationAwareGaussianProcessConditioner(
            query_points=query,
            actions=self.actions,
            solved_query_action=geometry.solved_query_action,
            covariance=geometry.covariance,
            variance=geometry.variance,
            observation_count=int(self.observation_points.shape[0]),
            output_dims=_query_output_dims(original_query, output_dim=output_dim),
        )

    def condition(
        self,
        residual: ArrayLike,
        query_points: ArrayLike,
        /,
        *,
        output_dim: str | None = "point",
    ) -> GaussianProcessCondition:
        """Condition one residual at a bounded full-covariance query design."""
        return self.conditioner(query_points, output_dim=output_dim).condition(residual)

    def _validated_residual(self, residual: ArrayLike, /) -> Array:
        values = _as_vector(residual, name="GP residual")
        if values.shape != (self.observation_points.shape[0],):
            raise ValueError("GP residual must align with factor observations.")
        return values


class ComputationAwareGaussianProcessConditioner(StrictModule):
    """Reusable low-rank residual projection and full query covariance."""

    query_points: Array
    actions: AbstractLinearOperator
    solved_query_action: Array
    covariance: Array
    variance: Array
    observation_count: int = eqx.field(static=True)
    output_dims: tuple[str | None, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        query_points: ArrayLike,
        actions: AbstractLinearOperator,
        solved_query_action: ArrayLike,
        covariance: ArrayLike,
        variance: ArrayLike,
        observation_count: int,
        output_dims: tuple[str | None, ...],
    ):
        query = _as_design(query_points)
        solved = jnp.asarray(solved_query_action)
        covariance_array = jnp.asarray(covariance)
        variance_array = _as_vector(variance, name="conditioned GP variance")
        query_count = int(query.shape[0])
        if solved.shape != (actions.source.size, query_count):
            raise ValueError("Solved query-action geometry has incompatible shape.")
        if covariance_array.shape != (query_count, query_count):
            raise ValueError("Conditioned GP covariance must be square over queries.")
        if variance_array.shape != (query_count,):
            raise ValueError("Conditioned GP variance must align with queries.")
        if int(observation_count) != actions.target.size:
            raise ValueError("Conditioner observation count must match its actions.")
        if len(output_dims) != 1:
            raise ValueError("Scalar GP output requires one output dimension.")
        self.query_points = query
        self.actions = actions
        self.solved_query_action = solved
        self.covariance = covariance_array
        self.variance = variance_array
        self.observation_count = int(observation_count)
        self.output_dims = tuple(output_dims)

    @property
    def storage_elements(self) -> int:
        return int(
            self.solved_query_action.size + self.covariance.size + self.variance.size
        )

    def condition(self, residual: ArrayLike, /) -> GaussianProcessCondition:
        values = _as_vector(residual, name="GP residual")
        if values.shape != (self.observation_count,):
            raise ValueError("GP residual must align with conditioner observations.")
        mean = _posterior_mean_from_solved_geometry(
            values,
            actions=self.actions,
            solved_query_action=self.solved_query_action,
        )
        return GaussianProcessCondition(
            query_points=self.query_points,
            mean=mean,
            covariance=self.covariance,
            variance=self.variance,
            output_dims=self.output_dims,
        )


class ComputationAwareGaussianProcessDiscrepancy(StrictModule):
    """Scalar model discrepancy conditioned through budgeted linear observations."""

    observation_points: Array
    observations: Array

    def __init__(
        self,
        observation_points: ArrayLike,
        observations: ArrayLike,
        /,
    ):
        points, values = _validated_observations(
            observation_points,
            observations,
            name="computation-aware GP observations",
        )
        self.observation_points = points
        self.observations = values

    def residual(self, physical_mean: ArrayLike, /) -> Array:
        mean = _as_vector(physical_mean, name="physical observation mean")
        if mean.shape != self.observations.shape:
            raise ValueError("physical_mean must align with GP observations.")
        return self.observations - mean

    def factor(
        self,
        *,
        state: GaussianProcessLikelihoodState,
        actions: AbstractGaussianProcessActionPolicy,
        computation: GaussianProcessComputationPolicy | None = None,
    ) -> ComputationAwareGaussianProcessFactor:
        return ComputationAwareGaussianProcessFactor(
            self.observation_points,
            state=state,
            actions=actions,
            computation=computation,
        )

    def elbo(
        self,
        physical_mean: ArrayLike,
        /,
        *,
        state: GaussianProcessLikelihoodState,
        actions: AbstractGaussianProcessActionPolicy,
        computation: GaussianProcessComputationPolicy | None = None,
    ) -> Array:
        residual = self.residual(physical_mean)
        factor = ComputationAwareGaussianProcessFactor(
            self.observation_points,
            state=state,
            actions=actions,
            computation=computation,
            residual=residual,
        )
        return factor.elbo(residual)

    def condition(
        self,
        physical_mean: ArrayLike,
        query_points: ArrayLike,
        /,
        *,
        state: GaussianProcessLikelihoodState,
        actions: AbstractGaussianProcessActionPolicy,
        computation: GaussianProcessComputationPolicy | None = None,
        output_dim: str | None = "point",
    ) -> GaussianProcessCondition:
        residual = self.residual(physical_mean)
        factor = ComputationAwareGaussianProcessFactor(
            self.observation_points,
            state=state,
            actions=actions,
            computation=computation,
            residual=residual,
        )
        return factor.condition(
            residual,
            query_points,
            output_dim=output_dim,
        )


__all__ = [
    "COMPUTATION_AWARE_GP_CONDITION_LIMIT",
    "COMPUTATION_AWARE_GP_NONFINITE",
    "COMPUTATION_AWARE_GP_PROJECTED_COVARIANCE_FAILURE",
    "COMPUTATION_AWARE_GP_PROJECTED_NOISE_FAILURE",
    "COMPUTATION_AWARE_GP_SUCCESS",
    "ComputationAwareGaussianProcessConditioner",
    "ComputationAwareGaussianProcessDiagnostics",
    "ComputationAwareGaussianProcessDiscrepancy",
    "ComputationAwareGaussianProcessFactor",
    "ComputationAwareGaussianProcessStatus",
    "GaussianProcessComputationPolicy",
]
