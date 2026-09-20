#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import AbstractVectorSpace, DualSpace
from ..optim import (
    AbstractLeastSquaresMethod,
    BoundedGaussNewton,
    Bounds,
    least_squares,
    LeastSquaresResult,
    NonlinearLeastSquaresProblem,
    OptimizationTermination,
)
from ._basis import ReducedBasisArtifact
from ._empirical_interpolation import (
    EmpiricalInterpolationArtifact,
    EmpiricalInterpolationPlan,
    prepare_empirical_interpolation,
)
from ._reduction import TrialTestReduction


class AbstractResidualProvider(StrictModule, NonTrainableState):
    state_space: eqx.AbstractVar[AbstractVectorSpace]
    residual_space: eqx.AbstractVar[AbstractVectorSpace]
    residual_id: eqx.AbstractVar[str]
    support_id: eqx.AbstractVar[str]
    geometry_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def residual(
        self,
        coordinate: Array,
        state: PyTree[Array],
        state_rate: PyTree[Array] | None,
        inputs: Any,
        /,
    ) -> PyTree[Array]:
        raise NotImplementedError


class FullResidualGalerkin(StrictModule, NonTrainableState):
    """Full-order-assisted nonlinear Galerkin reference."""

    reduction: TrialTestReduction
    provider: AbstractResidualProvider
    lift: PyTree[Array]
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        reduction: TrialTestReduction,
        provider: AbstractResidualProvider,
        /,
        *,
        lift: PyTree[Array] | None = None,
    ):
        if not isinstance(reduction, TrialTestReduction):
            raise TypeError("reduction must be a TrialTestReduction.")
        if not isinstance(provider, AbstractResidualProvider):
            raise TypeError("provider must be an AbstractResidualProvider.")
        if not provider.state_space.compatible(reduction.trial.full_space):
            raise ValueError("Residual provider state space must match the trial space.")
        if not provider.residual_space.compatible(DualSpace(reduction.test.full_space)):
            raise ValueError(
                "Residual provider output must lie in the dual test full space."
            )
        if (
            provider.support_id != reduction.support_id
            or provider.geometry_id != reduction.geometry_id
        ):
            raise ValueError(
                "Residual provider support and geometry must match the reduction."
            )
        resolved_lift = (
            reduction.trial.full_space.zeros()
            if lift is None
            else reduction.trial.full_space.validate(lift)
        )
        self.reduction = reduction
        self.provider = provider
        self.lift = resolved_lift
        self.model_id = canonical_fingerprint(
            {
                "kind": "full-residual-galerkin",
                "reduction": reduction.reduction_id,
                "residual": provider.residual_id,
                "lift": array_tree_fingerprint(resolved_lift)["sha256"],
            }
        )

    def residual(
        self,
        coordinate: ArrayLike,
        reduced_state: ArrayLike,
        reduced_rate: ArrayLike | None = None,
        inputs: Any = None,
        /,
    ) -> Array:
        state = self.reduction.trial.expand(reduced_state, self.lift)
        state_rate = (
            None
            if reduced_rate is None
            else self.reduction.trial.homogeneous_correction(reduced_rate)
        )
        residual = self.provider.residual(
            jnp.asarray(coordinate),
            state,
            state_rate,
            inputs,
        )
        return DualSpace(self.reduction.test.reduced_space).flatten(
            self.reduction.test.pullback_dual(
                self.provider.residual_space.validate(residual)
            )
        )


class AbstractStageResidualProvider(StrictModule, NonTrainableState):
    state_space: eqx.AbstractVar[AbstractVectorSpace]
    residual_space: eqx.AbstractVar[AbstractVectorSpace]
    residual_id: eqx.AbstractVar[str]
    support_id: eqx.AbstractVar[str]
    geometry_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def residual(
        self,
        source_coordinate: Array,
        target_coordinate: Array,
        source_state: PyTree[Array],
        target_state: PyTree[Array],
        inputs: Any,
        /,
    ) -> PyTree[Array]:
        raise NotImplementedError


class LSPGStepContext(StrictModule):
    source_coordinate: Array
    target_coordinate: Array
    source_reduced_state: Array
    inputs: Any


class ReducedLSPGProblem(StrictModule, NonTrainableState):
    """Full time-discrete residual minimization over a fixed trial chart."""

    reduction: TrialTestReduction
    provider: AbstractStageResidualProvider
    lift: PyTree[Array]
    residual_whitener: Array
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        reduction: TrialTestReduction,
        provider: AbstractStageResidualProvider,
        /,
        *,
        lift: PyTree[Array] | None = None,
    ):
        if not isinstance(reduction, TrialTestReduction):
            raise TypeError("reduction must be a TrialTestReduction.")
        if not isinstance(provider, AbstractStageResidualProvider):
            raise TypeError("provider must be an AbstractStageResidualProvider.")
        if not provider.state_space.compatible(reduction.trial.full_space):
            raise ValueError("Stage provider state space must match the trial space.")
        if (
            provider.support_id != reduction.support_id
            or provider.geometry_id != reduction.geometry_id
        ):
            raise ValueError(
                "Stage provider support and geometry must match the reduction."
            )
        residual_space = provider.residual_space
        basis = jnp.eye(
            residual_space.size, dtype=reduction.trial_basis.basis_matrix.dtype
        )
        gram = jnp.stack(
            tuple(
                jnp.stack(
                    tuple(
                        residual_space.inner(
                            residual_space.unflatten(basis[:, left]),
                            residual_space.unflatten(basis[:, right]),
                        )
                        for right in range(residual_space.size)
                    )
                )
                for left in range(residual_space.size)
            )
        )
        gram = 0.5 * (gram + jnp.conj(gram.T))
        eigenvalues = np.linalg.eigvalsh(np.asarray(gram))
        if np.min(eigenvalues) <= np.finfo(eigenvalues.dtype).eps * max(
            float(np.max(eigenvalues)), 1.0
        ):
            raise ValueError("LSPG residual pairing must be positive definite.")
        cholesky = jnp.linalg.cholesky(gram)
        resolved_lift = (
            reduction.trial.full_space.zeros()
            if lift is None
            else reduction.trial.full_space.validate(lift)
        )
        self.reduction = reduction
        self.provider = provider
        self.lift = resolved_lift
        self.residual_whitener = jnp.conj(cholesky.T)
        self.problem_id = canonical_fingerprint(
            {
                "kind": "reduced-lspg-problem",
                "reduction": reduction.reduction_id,
                "residual": provider.residual_id,
                "lift": array_tree_fingerprint(resolved_lift)["sha256"],
                "residual_gram": array_tree_fingerprint(gram)["sha256"],
            }
        )

    def residual(self, target_reduced_state: Array, context: LSPGStepContext, /) -> Array:
        source = self.reduction.trial.expand(context.source_reduced_state, self.lift)
        target = self.reduction.trial.expand(target_reduced_state, self.lift)
        residual = self.provider.residual(
            context.source_coordinate,
            context.target_coordinate,
            source,
            target,
            context.inputs,
        )
        coordinates = self.provider.residual_space.flatten(
            self.provider.residual_space.validate(residual)
        )
        return self.residual_whitener @ coordinates

    def solve(
        self,
        initial_reduced_state: ArrayLike,
        context: LSPGStepContext,
        /,
        *,
        method: AbstractLeastSquaresMethod | None = None,
        termination: OptimizationTermination | None = None,
    ) -> LeastSquaresResult:
        if not isinstance(context, LSPGStepContext):
            raise TypeError("context must be an LSPGStepContext.")
        problem = NonlinearLeastSquaresProblem(
            lambda state, step: self.residual(state, step),
            problem_id=self.problem_id,
        )
        return least_squares(
            problem,
            jnp.asarray(initial_reduced_state),
            method=method,
            termination=termination,
            args=context,
        )


class AbstractSampledNonlinearProvider(StrictModule, NonTrainableState):
    provider_id: eqx.AbstractVar[str]
    support_id: eqx.AbstractVar[str]
    geometry_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def evaluate_selected(
        self,
        state: PyTree[Array],
        node_indices: Array,
        inputs: Any,
        /,
    ) -> Array:
        raise NotImplementedError


class DEIMArtifact(StrictModule, NonTrainableState):
    node_indices: Array
    reduced_reconstruction: Array
    interpolation: EmpiricalInterpolationArtifact
    reduction_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)

    def evaluate(
        self,
        provider: AbstractSampledNonlinearProvider,
        state: PyTree[Array],
        inputs: Any = None,
        /,
    ) -> Array:
        if not isinstance(provider, AbstractSampledNonlinearProvider):
            raise TypeError("provider must be an AbstractSampledNonlinearProvider.")
        if (
            provider.provider_id != self.provider_id
            or provider.support_id != self.support_id
            or provider.geometry_id != self.geometry_id
        ):
            raise ValueError("Sampled nonlinear provider identity mismatch.")
        values = jnp.asarray(provider.evaluate_selected(state, self.node_indices, inputs))
        if values.shape != (self.node_indices.size,):
            raise ValueError("Selected nonlinear values must match the DEIM node count.")
        return self.reduced_reconstruction @ values


def prepare_deim(
    reduction: TrialTestReduction,
    nonlinear_basis: ReducedBasisArtifact,
    provider: AbstractSampledNonlinearProvider,
    /,
    *,
    plan: EmpiricalInterpolationPlan | None = None,
) -> DEIMArtifact:
    if not isinstance(reduction, TrialTestReduction):
        raise TypeError("reduction must be a TrialTestReduction.")
    if (
        not isinstance(nonlinear_basis, ReducedBasisArtifact)
        or nonlinear_basis.role != "nonlinear-term"
    ):
        raise TypeError("nonlinear_basis must use role='nonlinear-term'.")
    if not nonlinear_basis.subspace.space.compatible(
        DualSpace(reduction.test.full_space)
    ):
        raise ValueError("Nonlinear basis must lie in the full residual dual space.")
    if (
        nonlinear_basis.support_id != reduction.support_id
        or nonlinear_basis.geometry_id != reduction.geometry_id
    ):
        raise ValueError("Nonlinear basis support and geometry must match the reduction.")
    if not isinstance(provider, AbstractSampledNonlinearProvider):
        raise TypeError("provider must be an AbstractSampledNonlinearProvider.")
    if (
        provider.support_id != reduction.support_id
        or provider.geometry_id != reduction.geometry_id
    ):
        raise ValueError(
            "Sampled provider support and geometry must match the reduction."
        )
    interpolation = prepare_empirical_interpolation(nonlinear_basis, plan)
    reduced_reconstruction = reduction.test.dual_pullback.mv_block(
        jnp.asarray(interpolation.reconstruction_matrix)
    )
    artifact_id = canonical_fingerprint(
        {
            "kind": "deim-artifact",
            "reduction": reduction.reduction_id,
            "basis": nonlinear_basis.artifact_id,
            "interpolation": interpolation.artifact_id,
            "provider": provider.provider_id,
        }
    )
    return DEIMArtifact(
        jnp.asarray(interpolation.node_indices, dtype=jnp.int32),
        reduced_reconstruction,
        interpolation,
        reduction.reduction_id,
        provider.provider_id,
        reduction.support_id,
        reduction.geometry_id,
        artifact_id,
    )


class AbstractSampledStageResidualProvider(StrictModule, NonTrainableState):
    provider_id: eqx.AbstractVar[str]
    support_id: eqx.AbstractVar[str]
    geometry_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def evaluate_selected(
        self,
        source_coordinate: Array,
        target_coordinate: Array,
        source_state: PyTree[Array],
        target_state: PyTree[Array],
        node_indices: Array,
        inputs: Any,
        /,
    ) -> Array:
        raise NotImplementedError


class GNATArtifact(StrictModule, NonTrainableState):
    node_indices: Array
    reconstruction_matrix: Array
    interpolation: EmpiricalInterpolationArtifact
    provider_id: str = eqx.field(static=True)
    residual_space_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)


def prepare_gnat(
    residual_basis: ReducedBasisArtifact,
    provider: AbstractSampledStageResidualProvider,
    /,
    *,
    plan: EmpiricalInterpolationPlan | None = None,
) -> GNATArtifact:
    if (
        not isinstance(residual_basis, ReducedBasisArtifact)
        or residual_basis.role != "residual"
    ):
        raise TypeError("residual_basis must use role='residual'.")
    if not isinstance(provider, AbstractSampledStageResidualProvider):
        raise TypeError("provider must be an AbstractSampledStageResidualProvider.")
    if (
        provider.support_id != residual_basis.support_id
        or provider.geometry_id != residual_basis.geometry_id
    ):
        raise ValueError(
            "GNAT provider support and geometry must match its residual basis."
        )
    interpolation = prepare_empirical_interpolation(residual_basis, plan)
    return GNATArtifact(
        jnp.asarray(interpolation.node_indices, dtype=jnp.int32),
        jnp.asarray(interpolation.reconstruction_matrix),
        interpolation,
        provider.provider_id,
        residual_basis.subspace.space.space_id,
        provider.support_id,
        provider.geometry_id,
        canonical_fingerprint(
            {
                "kind": "gnat-artifact",
                "basis": residual_basis.artifact_id,
                "interpolation": interpolation.artifact_id,
                "provider": provider.provider_id,
            }
        ),
    )


class GNATLSPGProblem(StrictModule, NonTrainableState):
    lspg: ReducedLSPGProblem
    provider: AbstractSampledStageResidualProvider
    gnat: GNATArtifact
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        lspg: ReducedLSPGProblem,
        provider: AbstractSampledStageResidualProvider,
        gnat: GNATArtifact,
        /,
    ):
        if not isinstance(lspg, ReducedLSPGProblem):
            raise TypeError("lspg must be a ReducedLSPGProblem.")
        if not isinstance(provider, AbstractSampledStageResidualProvider):
            raise TypeError("provider must be an AbstractSampledStageResidualProvider.")
        if not isinstance(gnat, GNATArtifact):
            raise TypeError("gnat must be a GNATArtifact.")
        if provider.provider_id != gnat.provider_id:
            raise ValueError("GNAT sampled provider identity mismatch.")
        if gnat.residual_space_id != lspg.provider.residual_space.space_id:
            raise ValueError("GNAT residual-space identity must match the LSPG residual.")
        if gnat.reconstruction_matrix.shape[0] != lspg.provider.residual_space.size:
            raise ValueError(
                "GNAT residual reconstruction must match the LSPG residual space."
            )
        self.lspg = lspg
        self.provider = provider
        self.gnat = gnat
        self.problem_id = canonical_fingerprint(
            {
                "kind": "gnat-lspg-problem",
                "lspg": lspg.problem_id,
                "gnat": gnat.artifact_id,
            }
        )

    def residual(self, target_reduced_state: Array, context: LSPGStepContext, /) -> Array:
        source = self.lspg.reduction.trial.expand(
            context.source_reduced_state,
            self.lspg.lift,
        )
        target = self.lspg.reduction.trial.expand(
            target_reduced_state,
            self.lspg.lift,
        )
        sampled = self.provider.evaluate_selected(
            context.source_coordinate,
            context.target_coordinate,
            source,
            target,
            self.gnat.node_indices,
            context.inputs,
        )
        values = jnp.asarray(sampled)
        if values.shape != (self.gnat.node_indices.size,):
            raise ValueError("Selected GNAT residual must match the sample count.")
        reconstructed = self.gnat.reconstruction_matrix @ values
        return self.lspg.residual_whitener @ reconstructed

    def solve(
        self,
        initial_reduced_state: ArrayLike,
        context: LSPGStepContext,
        /,
        *,
        method: AbstractLeastSquaresMethod | None = None,
        termination: OptimizationTermination | None = None,
    ) -> LeastSquaresResult:
        problem = NonlinearLeastSquaresProblem(
            lambda state, step: self.residual(state, step),
            problem_id=self.problem_id,
        )
        return least_squares(
            problem,
            jnp.asarray(initial_reduced_state),
            method=method,
            termination=termination,
            args=context,
        )


class ThinGNATArtifact(StrictModule, NonTrainableState):
    node_indices: Array
    residual_factor: Array
    provider_id: str = eqx.field(static=True)
    residual_space_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)

    def residual(self, sampled_residual: ArrayLike, /) -> Array:
        value = jnp.asarray(sampled_residual)
        if value.shape[-1:] != (self.node_indices.size,):
            raise ValueError("Sampled residual must end in the GNAT sample axis.")
        return contract("ks,...s->...k", self.residual_factor, value)


def prepare_thin_gnat(
    residual_basis: ReducedBasisArtifact,
    node_indices: ArrayLike,
    residual_metric: ArrayLike,
    /,
    *,
    provider_id: str,
) -> ThinGNATArtifact:
    if (
        not isinstance(residual_basis, ReducedBasisArtifact)
        or residual_basis.role != "residual"
    ):
        raise TypeError("residual_basis must use role='residual'.")
    basis = residual_basis.basis_matrix
    nodes = jnp.asarray(node_indices, dtype=jnp.int32)
    metric = jnp.asarray(residual_metric)
    if nodes.ndim != 1 or nodes.size < residual_basis.rank:
        raise ValueError("Thin GNAT requires at least residual-rank samples.")
    if metric.shape != (basis.shape[0], basis.shape[0]):
        raise ValueError("Residual metric must match the residual full dimension.")
    sampled_basis = basis[nodes, :]
    singular = np.linalg.svd(np.asarray(sampled_basis), compute_uv=False)
    if singular[-1] <= np.finfo(singular.dtype).eps * max(float(singular[0]), 1.0):
        raise ValueError("Sampled residual basis is rank deficient.")
    reconstruction = basis @ jnp.linalg.pinv(sampled_basis)
    thin_gram = jnp.conj(reconstruction.T) @ metric @ reconstruction
    thin_gram = 0.5 * (thin_gram + jnp.conj(thin_gram.T))
    eigenvalues, eigenvectors = jnp.linalg.eigh(thin_gram)
    if (
        float(np.min(np.asarray(eigenvalues)))
        < -64.0 * np.finfo(np.asarray(eigenvalues).dtype).eps
    ):
        raise ValueError("Thin GNAT residual metric is not positive semidefinite.")
    clipped = jnp.maximum(eigenvalues, 0.0)
    factor = jnp.sqrt(clipped)[:, None] * jnp.conj(eigenvectors.T)
    provider = str(provider_id)
    if not provider:
        raise ValueError("provider_id must be non-empty.")
    artifact_id = canonical_fingerprint(
        {
            "kind": "thin-gnat-artifact",
            "basis": residual_basis.artifact_id,
            "provider": provider,
            "nodes": array_tree_fingerprint(nodes)["sha256"],
            "metric": array_tree_fingerprint(metric)["sha256"],
        }
    )
    return ThinGNATArtifact(
        nodes,
        factor,
        provider,
        residual_basis.subspace.space.space_id,
        residual_basis.support_id,
        residual_basis.geometry_id,
        artifact_id,
    )


class ECSWPlan(StrictModule, NonTrainableState):
    maximum_elements: int = eqx.field(static=True)
    minimum_weight: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, maximum_elements: int, /, *, minimum_weight: float = 1.0e-12):
        maximum = int(maximum_elements)
        minimum = float(minimum_weight)
        if maximum <= 0:
            raise ValueError("maximum_elements must be positive.")
        if not np.isfinite(minimum) or minimum < 0.0:
            raise ValueError("minimum_weight must be finite and nonnegative.")
        self.maximum_elements = maximum
        self.minimum_weight = minimum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "ecsw-plan",
                "maximum_elements": maximum,
                "minimum_weight": minimum,
            }
        )


class ECSWArtifact(StrictModule, NonTrainableState):
    element_indices: Array
    weights: Array
    training_residual_norm: Array
    kkt_residual_norm: Array
    valid: Array
    reduced_rank: int = eqx.field(static=True)
    reduction_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)


class AbstractElementResidualProvider(StrictModule, NonTrainableState):
    provider_id: eqx.AbstractVar[str]
    reduction_id: eqx.AbstractVar[str]
    support_id: eqx.AbstractVar[str]
    geometry_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def evaluate_elements(
        self,
        state: PyTree[Array],
        element_indices: Array,
        inputs: Any,
        /,
    ) -> Array:
        """Return selected reduced residual contributions with shape (elements, rank)."""
        raise NotImplementedError


def prepare_ecsw(
    reduced_element_contributions: ArrayLike,
    target_reduced_residual: ArrayLike,
    reduction: TrialTestReduction,
    provider: AbstractElementResidualProvider,
    plan: ECSWPlan,
    /,
) -> ECSWArtifact:
    contributions = jnp.asarray(reduced_element_contributions)
    target = jnp.asarray(target_reduced_residual)
    if contributions.ndim != 3:
        raise ValueError(
            "Element contributions must have shape (samples, elements, rank)."
        )
    samples, elements, rank = contributions.shape
    if target.shape != (samples, rank) or rank != reduction.rank:
        raise ValueError(
            "Target residual and reduction rank must match element contributions."
        )
    if not isinstance(provider, AbstractElementResidualProvider):
        raise TypeError("provider must be an AbstractElementResidualProvider.")
    if provider.reduction_id != reduction.reduction_id:
        raise ValueError("ECSW provider reduction identity mismatch.")
    if (
        provider.support_id != reduction.support_id
        or provider.geometry_id != reduction.geometry_id
    ):
        raise ValueError("ECSW provider support and geometry must match the reduction.")
    if not isinstance(plan, ECSWPlan):
        raise TypeError("plan must be an ECSWPlan.")
    design = jnp.swapaxes(contributions, 1, 2).reshape((samples * rank, elements))
    response = target.reshape((samples * rank,))
    initial = jnp.ones((elements,), dtype=design.dtype)
    bounded = NonlinearLeastSquaresProblem(
        lambda weights, _: design @ weights - response,
        bounds=Bounds(jnp.zeros_like(initial), jnp.full_like(initial, jnp.inf)),
        problem_id=f"ecsw:{plan.plan_id}:all-elements",
    )
    first = least_squares(bounded, initial, method=BoundedGaussNewton())
    if not bool(np.asarray(first.successful)):
        raise ValueError("ECSW full NNLS preparation failed.")
    candidate = jnp.maximum(jnp.asarray(first.parameters), 0.0)
    order = jnp.argsort(candidate)[::-1]
    selected = order[: min(plan.maximum_elements, elements)]
    selected = selected[candidate[selected] > plan.minimum_weight]
    if selected.size == 0:
        raise ValueError("ECSW selection produced no positive empirical weights.")
    selected_design = design[:, selected]
    selected_initial = candidate[selected]
    refit_problem = NonlinearLeastSquaresProblem(
        lambda weights, _: selected_design @ weights - response,
        bounds=Bounds(
            jnp.zeros_like(selected_initial),
            jnp.full_like(selected_initial, jnp.inf),
        ),
        problem_id=f"ecsw:{plan.plan_id}:selected-elements",
    )
    refit = least_squares(
        refit_problem,
        selected_initial,
        method=BoundedGaussNewton(),
    )
    if not bool(np.asarray(refit.successful)):
        raise ValueError("ECSW selected NNLS refit failed.")
    weights = jnp.maximum(jnp.asarray(refit.parameters), 0.0)
    residual_norm = jnp.linalg.norm(selected_design @ weights - response)
    gradient = jnp.conj(selected_design.T) @ (selected_design @ weights - response)
    kkt = jnp.max(
        jnp.abs(
            jnp.where(
                weights > plan.minimum_weight,
                gradient,
                jnp.minimum(gradient, 0.0),
            )
        )
    )
    valid = (
        jnp.all(jnp.isfinite(weights))
        & jnp.all(weights >= 0.0)
        & jnp.isfinite(residual_norm)
        & jnp.isfinite(kkt)
    )
    artifact_id = canonical_fingerprint(
        {
            "kind": "ecsw-artifact",
            "reduction": reduction.reduction_id,
            "provider": provider.provider_id,
            "plan": plan.plan_id,
            "training": array_tree_fingerprint(
                {
                    "contributions": contributions,
                    "target": target,
                }
            )["sha256"],
            "selected": array_tree_fingerprint(selected)["sha256"],
            "weights": array_tree_fingerprint(weights)["sha256"],
        }
    )
    return ECSWArtifact(
        selected.astype(jnp.int32),
        weights,
        residual_norm,
        kkt,
        valid,
        reduction.rank,
        reduction.reduction_id,
        provider.provider_id,
        reduction.support_id,
        reduction.geometry_id,
        artifact_id,
    )


def evaluate_ecsw(
    artifact: ECSWArtifact,
    provider: AbstractElementResidualProvider,
    state: PyTree[Array],
    inputs: Any = None,
    /,
) -> Array:
    if not isinstance(artifact, ECSWArtifact):
        raise TypeError("artifact must be an ECSWArtifact.")
    if not isinstance(provider, AbstractElementResidualProvider):
        raise TypeError("provider must be an AbstractElementResidualProvider.")
    if (
        provider.provider_id != artifact.provider_id
        or provider.reduction_id != artifact.reduction_id
        or provider.support_id != artifact.support_id
        or provider.geometry_id != artifact.geometry_id
    ):
        raise ValueError("ECSW provider identity mismatch.")
    contributions = jnp.asarray(
        provider.evaluate_elements(state, artifact.element_indices, inputs)
    )
    if contributions.shape != (
        artifact.element_indices.size,
        artifact.reduced_rank,
    ):
        raise ValueError("Selected element contributions have an invalid shape.")
    return contract("e,er->r", artifact.weights, contributions)


__all__ = [
    "AbstractElementResidualProvider",
    "AbstractResidualProvider",
    "AbstractSampledNonlinearProvider",
    "AbstractSampledStageResidualProvider",
    "AbstractStageResidualProvider",
    "DEIMArtifact",
    "ECSWArtifact",
    "ECSWPlan",
    "FullResidualGalerkin",
    "GNATArtifact",
    "GNATLSPGProblem",
    "LSPGStepContext",
    "ReducedLSPGProblem",
    "ThinGNATArtifact",
    "evaluate_ecsw",
    "prepare_deim",
    "prepare_ecsw",
    "prepare_gnat",
    "prepare_thin_gnat",
]
