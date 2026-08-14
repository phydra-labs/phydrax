#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Mapping
from typing import Any, NamedTuple

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from phydrax.conditions import AbstractResidualCondition, Observation
from phydrax.domain import (
    BatchEvaluator,
    ComponentSum,
    DomainComponent,
    DomainFunction,
    GridBatch,
    GridSampling,
    PointBatch,
    PointSampling,
)

from .._doc import DOC_KEY0
from .._strict import StrictModule
from .._term import AbstractEvaluatedScalarTerm, TermEvaluation
from ..integration import (
    AdaptiveIntegration,
    CallerIntegration,
    ComponentTarget,
    DensityTarget,
    FixedIntegration,
    from_samples,
    IntegrationRealization,
    IntegrationSource,
    MappedIntegrationBatch,
    PerStepIntegration,
    PointIntegrationBatch,
    reduce,
    SeparableIntegrationBatch,
    WeightedSampleBatch,
)
from ..integration._fixed import _as_domain_function, _component_reduction_weights
from ..integration._lowering import sum_over
from ..sampling.collocation._adaptive import AbstractCollocationPolicy
from ._data_metrics import supervised_data_metrics
from ._integrated import (
    checked_estimate_field,
    prepare_term_realization,
    resolve_term_realization,
    validate_condition_source,
)


_SOURCE_TYPES = (
    PerStepIntegration,
    FixedIntegration,
    CallerIntegration,
    AdaptiveIntegration,
)


def _adaptive_component(source: AdaptiveIntegration, /) -> DomainComponent:
    target = source.target
    while isinstance(target, DensityTarget):
        target = target.base
    if not isinstance(target, ComponentTarget):
        raise TypeError(
            "AdaptiveIntegration for ResidualPenalty requires a component target."
        )
    if isinstance(target.component, ComponentSum):
        raise TypeError(
            "AdaptiveIntegration requires one DomainComponent, not a component union."
        )
    return target.component


def _validate_adaptive_source(source: AdaptiveIntegration, /) -> None:
    _adaptive_component(source)
    if not isinstance(source.initial_plan, (PointSampling, GridSampling)):
        raise TypeError(
            "AdaptiveIntegration initial_plan must be a PointSampling or GridSampling."
        )
    if not isinstance(source.policy, AbstractCollocationPolicy):
        raise TypeError(
            "AdaptiveIntegration policy must be an AbstractCollocationPolicy."
        )


def _apply_local_weight(
    realization: IntegrationRealization,
    local_weight: cx.Field | None,
    /,
) -> IntegrationRealization:
    if local_weight is None:
        return realization
    if not isinstance(local_weight, cx.Field):
        raise TypeError(
            "Adaptive collocation loss weight must be a coordax.Field or None."
        )
    if any(dim is None for dim in local_weight.dims):
        raise ValueError("Adaptive collocation loss weight must use named batch axes.")
    batch = realization.batch
    if isinstance(batch, PointIntegrationBatch):
        unexpected = tuple(
            dim for dim in local_weight.named_dims if dim not in batch.weights.named_dims
        )
        if unexpected:
            raise ValueError(
                f"Adaptive collocation loss weight has incompatible axes {unexpected!r}."
            )
        weighted_batch = PointIntegrationBatch(
            batch.points,
            batch.weights * local_weight,
            axes=batch.axes,
            mask=batch.mask,
            target_mass=batch.target_mass,
            stratum_indices=batch.stratum_indices,
            num_strata=batch.num_strata,
            provenance=batch.provenance,
        )
    elif isinstance(batch, SeparableIntegrationBatch):
        total_weight = batch.total_weight()
        unexpected = tuple(
            dim for dim in local_weight.named_dims if dim not in total_weight.named_dims
        )
        if unexpected:
            raise ValueError(
                f"Adaptive collocation loss weight has incompatible axes {unexpected!r}."
            )
        coupled_weight = (
            local_weight
            if batch.coupled_weight is None
            else batch.coupled_weight * local_weight
        )
        weighted_batch = SeparableIntegrationBatch(
            batch.points,
            batch.weights_by_axis,
            axes=batch.axes,
            coupled_weight=coupled_weight,
            mask=batch.mask,
            target_mass=batch.target_mass,
            provenance=batch.provenance,
        )
    else:
        raise TypeError(
            "Adaptive collocation requires a point or separable integration batch."
        )
    return IntegrationRealization(
        realization.target,
        realization.plan,
        weighted_batch,
        realization.key,
    )


def _squared_frobenius_field(value: cx.Field, /) -> cx.Field:
    data = jnp.asarray(value.data)
    dims = value.dims
    squared = jnp.real(jnp.conj(data) * data)
    event_axes = tuple(index for index, dim in enumerate(dims) if dim is None)
    for axis in reversed(event_axes):
        squared = jnp.sum(squared, axis=axis)
        dims = dims[:axis] + dims[axis + 1 :]
    return cx.Field(squared, dims=dims)


class _QuadraticResidualData(NamedTuple):
    residuals: tuple[cx.Field, ...]
    coefficients: tuple[cx.Field, ...]
    loss: Array


def _target_reduction_weights(
    target: ComponentTarget | DensityTarget,
    batch: PointIntegrationBatch
    | SeparableIntegrationBatch
    | tuple[PointIntegrationBatch | SeparableIntegrationBatch, ...],
    /,
    *,
    key: Key[Array, ""],
    kwargs: dict[str, Any],
) -> cx.Field | tuple[cx.Field, ...]:
    base = target.base if isinstance(target, DensityTarget) else target
    if not isinstance(base, ComponentTarget):
        raise TypeError(
            "Quadratic residual data requires a component integration target."
        )
    weights = _component_reduction_weights(
        base,
        batch,
        key=key,
        kwargs=kwargs,
    )
    if not isinstance(target, DensityTarget):
        return weights

    if isinstance(base.component, ComponentSum):
        if not isinstance(batch, tuple) or not isinstance(weights, tuple):
            raise RuntimeError("Component-sum integration data must be tuples.")
        components = base.component.terms
        batches = batch
        base_weights = weights
        keys = tuple(jr.split(key, len(components)))
    else:
        if isinstance(batch, tuple) or isinstance(weights, tuple):
            raise RuntimeError("Single-component integration data must not be tuples.")
        components = (base.component,)
        batches = (batch,)
        base_weights = (weights,)
        keys = (key,)

    density_weights: list[cx.Field] = []
    for component, term_batch, base_weight, term_key in zip(
        components, batches, base_weights, keys, strict=True
    ):
        log_density = _as_domain_function(target.log_density, component)(
            term_batch.points,
            key=term_key,
            **kwargs,
        )
        if not isinstance(log_density, cx.Field):
            raise TypeError("Integration log density must return a coordax.Field.")
        log_data = jnp.asarray(log_density.data)
        if jnp.iscomplexobj(log_data):
            raise TypeError("Integration log density must be real.")
        density_weights.append(
            base_weight * cx.Field(jnp.exp(log_data), dims=log_density.dims)
        )

    if target.normalized:
        denominators: list[cx.Field] = []
        for coefficient, term_batch in zip(density_weights, batches, strict=True):
            denominator = coefficient
            for axis in term_batch.axes:
                denominator = sum_over(denominator, axis)
            denominators.append(denominator)
        normalization = denominators[0]
        for denominator in denominators[1:]:
            normalization = normalization + denominator
        density_weights = [coefficient / normalization for coefficient in density_weights]

    if isinstance(base.component, ComponentSum):
        return tuple(density_weights)
    return density_weights[0]


def _checked_quadratic_coefficient(coefficient: cx.Field, /) -> cx.Field:
    if any(dim is None for dim in coefficient.dims):
        raise ValueError(
            "KFAC residual reduction coefficients may not contain unnamed event axes."
        )
    raw = jnp.asarray(coefficient.data)
    if jnp.iscomplexobj(raw):
        raise TypeError("KFAC residual reduction coefficients must be real.")
    data = jnp.asarray(raw, dtype=float)
    data = jnp.where(
        data < 0.0,
        jnp.where(data >= -1e-12, 0.0, data),
        data,
    )
    data = jnp.asarray(
        eqx.error_if(
            data,
            jnp.any(~jnp.isfinite(data)) | jnp.any(data < 0.0),
            "KFAC requires finite nonnegative residual reduction coefficients.",
        )
    )
    return cx.Field(data, dims=coefficient.dims)


class _SquaredFrobeniusResidual(StrictModule, BatchEvaluator):
    residual: DomainFunction

    def __init__(self, residual: DomainFunction, /):
        self.residual = residual

    def __call_batch__(
        self,
        batch: PointBatch | GridBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        value = self.residual(batch, key=key, **kwargs)
        if not isinstance(value, cx.Field):
            raise TypeError("Residual evaluation must return a coordax.Field.")
        return _squared_frobenius_field(value)

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        value = jnp.asarray(self.residual.func(*args, key=key, **kwargs))
        return jnp.sum(jnp.real(jnp.conj(value) * value))


class _DensityWeightedResidual(StrictModule, BatchEvaluator):
    score: DomainFunction
    density: DomainFunction

    def __init__(self, score: DomainFunction, density: DomainFunction, /):
        self.score = score
        self.density = density

    def __call_batch__(
        self,
        batch: PointBatch | GridBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        score = self.score(batch, key=key, **kwargs)
        density = self.density(batch, key=key, **kwargs)
        if not isinstance(score, cx.Field) or not isinstance(density, cx.Field):
            raise TypeError(
                "Residual score and density must return coordax.Field values."
            )
        density_data = jnp.asarray(density.data)
        if jnp.iscomplexobj(density_data):
            raise TypeError("Penalty density must be real.")
        density_data = eqx.error_if(
            density_data,
            jnp.any(~jnp.isfinite(density_data) | (density_data < 0.0)),
            "Penalty density must be finite and nonnegative.",
        )
        checked_density = cx.Field(density_data, dims=density.dims)
        return score * checked_density

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        density = jnp.asarray(self.density.func(*args, key=key, **kwargs))
        if jnp.iscomplexobj(density):
            raise TypeError("Penalty density must be real.")
        density = eqx.error_if(
            density,
            jnp.any(~jnp.isfinite(density) | (density < 0.0)),
            "Penalty density must be finite and nonnegative.",
        )
        return self.score.func(*args, key=key, **kwargs) * density


def _realization_points(realization: IntegrationRealization, /) -> Any:
    batch = realization.batch
    if isinstance(batch, tuple):
        raise TypeError("Data diagnostics require one integration batch.")
    if isinstance(
        batch,
        (PointIntegrationBatch, SeparableIntegrationBatch, MappedIntegrationBatch),
    ):
        return batch.points
    if isinstance(batch, WeightedSampleBatch):
        return batch.samples
    raise TypeError(f"Unsupported data-diagnostics batch {type(batch).__name__}.")


class ResidualPenalty(AbstractEvaluatedScalarTerm):
    """Nonnegative local residual score reduced by an integration realization."""

    condition: AbstractResidualCondition
    source: IntegrationSource
    fields: tuple[str, ...] = eqx.field(static=True)
    scale: Array
    weight: Array
    density: DomainFunction | None
    label: str | None = eqx.field(static=True)
    data_accuracy_eps: float = eqx.field(static=True)

    def __init__(
        self,
        condition: AbstractResidualCondition,
        source: IntegrationSource,
        /,
        *,
        scale: ArrayLike = 1.0,
        density: DomainFunction | None = None,
        label: str | None = None,
        data_accuracy_eps: float = 1e-12,
    ):
        if not isinstance(condition, AbstractResidualCondition):
            raise TypeError("ResidualPenalty requires an AbstractResidualCondition.")
        if not isinstance(source, _SOURCE_TYPES):
            raise TypeError("ResidualPenalty requires a typed IntegrationSource.")
        validate_condition_source(condition.on, source)
        if isinstance(source, AdaptiveIntegration):
            _validate_adaptive_source(source)
        coefficient = jnp.asarray(scale, dtype=float)
        if coefficient.shape != ():
            raise ValueError("Term scale must be a scalar.")
        if not bool(jnp.isfinite(coefficient)) or float(coefficient) < 0.0:
            raise ValueError("Term scale must be finite and nonnegative.")
        if density is not None:
            if not isinstance(density, DomainFunction):
                raise TypeError("Penalty density must be a DomainFunction or None.")
            if not density.domain.same_support(condition.on.domain):
                raise ValueError(
                    "Penalty density domain is incompatible with the condition."
                )
        accuracy_eps = float(data_accuracy_eps)
        if not bool(jnp.isfinite(accuracy_eps)) or accuracy_eps <= 0.0:
            raise ValueError("data_accuracy_eps must be finite and positive.")
        self.condition = condition
        self.source = source
        self.fields = condition.fields
        self.scale = coefficient.reshape(())
        self.weight = self.scale
        self.density = density
        self.label = condition.label if label is None else str(label)
        self.data_accuracy_eps = accuracy_eps

    @property
    def sampling(self) -> PointSampling | GridSampling:
        """Return the solver-managed adaptive sampling plan."""
        if not isinstance(self.source, AdaptiveIntegration):
            raise TypeError(
                "ResidualPenalty sampling is only defined for AdaptiveIntegration."
            )
        return self.source.initial_plan

    @property
    def component(self) -> DomainComponent:
        """Return the single component managed by adaptive collocation."""
        if not isinstance(self.source, AdaptiveIntegration):
            raise TypeError(
                "ResidualPenalty component is only defined for AdaptiveIntegration."
            )
        return _adaptive_component(self.source)

    @property
    def policy(self) -> AbstractCollocationPolicy:
        """Return the policy owned by the adaptive integration source."""
        if not isinstance(self.source, AdaptiveIntegration):
            raise TypeError("ResidualPenalty policy requires AdaptiveIntegration.")
        return self.source.policy

    def sample(
        self,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> PointBatch | GridBatch:
        """Sample one structured batch from the adaptive source."""
        batch = self.component.sample(self.sampling, key=key)
        if not isinstance(batch, (PointBatch, GridBatch)):
            raise TypeError(
                "AdaptiveIntegration must sample one PointBatch or GridBatch."
            )
        return batch

    def _adaptive_realization(
        self,
        batch: PointBatch | GridBatch,
        local_weight: cx.Field | None,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> IntegrationRealization:
        """Materialize one policy batch against the adaptive source target."""
        if not isinstance(self.source, AdaptiveIntegration):
            raise TypeError(
                "Adaptive realization requires an AdaptiveIntegration source."
            )
        if not isinstance(batch, (PointBatch, GridBatch)):
            raise TypeError("Adaptive collocation requires one PointBatch or GridBatch.")
        realization = from_samples(self.source.target, batch, key=key)
        return _apply_local_weight(realization, local_weight)

    def _score_function(
        self,
        functions: Mapping[str, DomainFunction],
        /,
    ) -> DomainFunction:
        residual = self.condition.residual(functions)
        score = DomainFunction(
            domain=residual.domain,
            deps=residual.deps,
            func=_SquaredFrobeniusResidual(residual),
            metadata=residual.metadata,
        )
        if self.density is None:
            return score
        density = self.density
        if density.domain.labels != score.domain.labels:
            density = density.promote(score.domain)
        return DomainFunction(
            domain=score.domain,
            deps=tuple(dict.fromkeys(score.deps + density.deps)),
            func=_DensityWeightedResidual(score, density),
            metadata=score.metadata,
        )

    def pointwise_score(
        self,
        functions: Mapping[str, DomainFunction],
        batch: PointBatch | GridBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        """Evaluate the unreduced scalar residual score on one structured batch."""
        score = self._score_function(functions)(batch, key=key, **kwargs)
        if not isinstance(score, cx.Field):
            raise TypeError("Pointwise residual score must return a coordax.Field.")
        if any(dim is None for dim in score.dims):
            raise ValueError("Pointwise residual scalarization left event dimensions.")
        return score

    def term_evaluation(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array | None = None,
        realization: IntegrationRealization | None = None,
        **kwargs: Any,
    ) -> TermEvaluation:
        resolved = resolve_term_realization(
            self.source,
            key=key,
            realization=realization,
        )
        runtime_kwargs = dict(kwargs)
        if iter_ is not None:
            runtime_kwargs["iter_"] = iter_
        estimate = reduce(self._score_function(functions), resolved, **runtime_kwargs)
        field = checked_estimate_field(estimate)
        if field.dims != ():
            raise ValueError(
                f"ResidualPenalty must reduce to a scalar Field, got dims={field.dims}."
            )
        value = self.scale * jnp.asarray(field.data, dtype=float).reshape(())
        return TermEvaluation(value, diagnostics=estimate)

    def _quadratic_residual_data(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array | None = None,
        realization: IntegrationRealization | None = None,
        **kwargs: Any,
    ) -> _QuadraticResidualData:
        """Return residual fields and their exact nonnegative reduction coefficients."""
        resolved = resolve_term_realization(
            self.source,
            key=key,
            realization=(
                None if realization is None else prepare_term_realization(realization)
            ),
        )
        target = resolved.target
        if not isinstance(target, (ComponentTarget, DensityTarget)):
            raise TypeError(
                "Quadratic residual data requires a component integration target."
            )
        batch = resolved.batch
        if isinstance(batch, tuple):
            if not batch or any(
                not isinstance(term, (PointIntegrationBatch, SeparableIntegrationBatch))
                for term in batch
            ):
                raise TypeError(
                    "Quadratic residual data requires fixed point or separable batches."
                )
        elif not isinstance(batch, (PointIntegrationBatch, SeparableIntegrationBatch)):
            raise TypeError(
                "Quadratic residual data requires fixed point or separable batches."
            )

        runtime_kwargs = dict(kwargs)
        if iter_ is not None:
            runtime_kwargs["iter_"] = iter_
        evaluation_key = DOC_KEY0 if resolved.key is None else resolved.key
        reduction_weights = _target_reduction_weights(
            target,
            batch,
            key=evaluation_key,
            kwargs=runtime_kwargs,
        )

        base = target.base if isinstance(target, DensityTarget) else target
        if not isinstance(base, ComponentTarget):
            raise TypeError(
                "Quadratic residual data requires a component integration target."
            )
        if isinstance(base.component, ComponentSum):
            if not isinstance(batch, tuple) or not isinstance(reduction_weights, tuple):
                raise RuntimeError("Component-sum integration data must be tuples.")
            integration_batches = batch
            coefficients = reduction_weights
            term_keys = tuple(jr.split(evaluation_key, len(base.component.terms)))
        else:
            if isinstance(batch, tuple) or isinstance(reduction_weights, tuple):
                raise RuntimeError(
                    "Single-component integration data must not be tuples."
                )
            integration_batches = (batch,)
            coefficients = (reduction_weights,)
            term_keys = (evaluation_key,)

        residual_fn = self.condition.residual(functions)
        density = self.density
        if density is not None and density.domain.labels != residual_fn.domain.labels:
            density = density.promote(residual_fn.domain)

        residuals: list[cx.Field] = []
        checked_coefficients: list[cx.Field] = []
        total = jnp.asarray(0.0, dtype=float)
        for integration_batch, coefficient, term_key in zip(
            integration_batches,
            coefficients,
            term_keys,
            strict=True,
        ):
            points = integration_batch.points
            residual = residual_fn(points, key=term_key, **runtime_kwargs)
            if not isinstance(residual, cx.Field):
                raise TypeError("Residual evaluation must return a coordax.Field.")
            if density is not None:
                evaluated_density = density(
                    points,
                    key=term_key,
                    **runtime_kwargs,
                )
                if not isinstance(evaluated_density, cx.Field):
                    raise TypeError("Penalty density must return a coordax.Field.")
                density_data = jnp.asarray(evaluated_density.data)
                if jnp.iscomplexobj(density_data):
                    raise TypeError("Penalty density must be real.")
                density_data = eqx.error_if(
                    density_data,
                    jnp.any(~jnp.isfinite(density_data)) | jnp.any(density_data < 0.0),
                    "Penalty density must be finite and nonnegative.",
                )
                coefficient = coefficient * cx.Field(
                    density_data,
                    dims=evaluated_density.dims,
                )
            coefficient = _checked_quadratic_coefficient(self.scale * coefficient)
            weighted = coefficient * _squared_frobenius_field(residual)
            total = total + jnp.sum(jnp.asarray(weighted.data, dtype=float))
            residuals.append(residual)
            checked_coefficients.append(coefficient)
        return _QuadraticResidualData(
            tuple(residuals),
            tuple(checked_coefficients),
            total,
        )

    def data_metrics(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        realization: IntegrationRealization | None = None,
        **kwargs: Any,
    ) -> dict[str, Array]:
        """Evaluate prediction diagnostics when this penalty wraps an observation."""
        condition = self.condition
        if not isinstance(condition, Observation):
            return {}
        resolved = resolve_term_realization(
            self.source,
            key=key,
            realization=realization,
        )
        points = _realization_points(resolved)
        prediction_fn = condition.operator(
            *(functions[field] for field in condition.fields)
        )
        if not isinstance(prediction_fn, DomainFunction):
            raise TypeError("Observation operators must return a DomainFunction.")
        prediction = prediction_fn(points, key=key, **kwargs)
        target = condition.target(points, key=key, **kwargs)
        prediction_data = (
            prediction.data if isinstance(prediction, cx.Field) else prediction
        )
        target_data = target.data if isinstance(target, cx.Field) else target
        return supervised_data_metrics(
            jnp.asarray(prediction_data, dtype=float),
            jnp.asarray(target_data, dtype=float),
            eps=self.data_accuracy_eps,
        )

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array | None = None,
        realization: IntegrationRealization | None = None,
        **kwargs: Any,
    ) -> Array:
        return self.term_evaluation(
            functions,
            key=key,
            iter_=iter_,
            realization=realization,
            **kwargs,
        ).value


__all__ = ["ResidualPenalty"]
