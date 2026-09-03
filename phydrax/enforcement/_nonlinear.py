#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Mapping, Sequence
from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from .._frozendict import frozendict
from .._strict import AbstractAttribute, StrictModule
from .._tree_math import tree_allfinite, tree_norm, validate_real_inexact_tree
from ..conditions._evidence import (
    ConditionEvidence,
    ConditionRealizationStamp,
    NonlinearRetractionCertificate,
)
from ..conditions._ir import ArrayCodomain, Condition, ConditionQuantifier
from ..conditions._lowering import BoundCondition
from ..conditions._relations import (
    Complementarity,
    ConeKind,
    ConeMembership,
    Equality,
    Inequality,
    NoisyObservation,
)
from ..nonlinear._newton import NewtonKrylov
from ..nonlinear._types import (
    AbstractNonlinearMethod,
    nonlinear_status_message,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ..optim._programming._cones import SecondOrderCone
from ._feasibility import PositiveSemidefiniteProjection, SimplexProjection
from ._lifecycle import (
    AbstractRealizationSource,
    commit_refresh,
    propose_refresh,
    RealizationFailure,
    RealizationLifecyclePhase,
    RealizationLifecycleState,
    record_realization_stamp,
    RefreshValidator,
    validate_refresh,
)
from ._realization import (
    AbstractFieldRealization,
    ConditionEvaluationContext,
    FieldMap,
    FieldRealizationResult,
    RealizationStatus,
)


RetractionObjective: TypeAlias = Literal["local-root", "minimum-distance"]


class AbstractCorrectionChart(StrictModule):
    """Local coordinates whose origin reconstructs the supplied field mapping."""

    chart_id: AbstractAttribute[str]

    @abc.abstractmethod
    def origin(self, fields: FieldMap, /) -> PyTree[Array]:
        raise NotImplementedError

    @abc.abstractmethod
    def retract(
        self, fields: FieldMap, coordinates: PyTree[Any], /
    ) -> frozendict[str, Any]:
        raise NotImplementedError


class AdditiveCorrectionChart(AbstractCorrectionChart):
    """Add array-PyTree corrections to an explicit subset of named fields."""

    field_names: tuple[str, ...] = eqx.field(static=True)
    chart_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_names: str | Sequence[str],
        /,
        *,
        chart_id: str | None = None,
    ):
        names = (
            (field_names,)
            if isinstance(field_names, str)
            else tuple(str(name) for name in field_names)
        )
        if not names or any(not name for name in names):
            raise ValueError("An additive correction chart requires named fields.")
        if len(set(names)) != len(names):
            raise ValueError("Correction-chart field names must be unique.")
        identifier = (
            canonical_fingerprint({"kind": "additive-correction-chart", "fields": names})
            if chart_id is None
            else str(chart_id)
        )
        if not identifier:
            raise ValueError("chart_id must be non-empty.")
        self.field_names = names
        self.chart_id = identifier

    def _selected(self, fields: FieldMap, /) -> dict[str, Any]:
        missing = tuple(name for name in self.field_names if name not in fields)
        if missing:
            raise KeyError(f"Correction-chart fields are missing: {missing!r}.")
        return {name: fields[name] for name in self.field_names}

    def origin(self, fields: FieldMap, /) -> PyTree[Array]:
        selected = validate_real_inexact_tree(
            self._selected(fields), name="additive correction-chart fields"
        )
        return jax.tree.map(jnp.zeros_like, selected)

    def retract(
        self, fields: FieldMap, coordinates: PyTree[Any], /
    ) -> frozendict[str, Any]:
        selected = validate_real_inexact_tree(
            self._selected(fields), name="additive correction-chart fields"
        )
        correction = validate_real_inexact_tree(
            coordinates, name="additive correction coordinates"
        )
        if jax.tree.structure(selected) != jax.tree.structure(correction):
            raise ValueError("Correction coordinates do not match the chart structure.")
        updated = dict(fields)
        values = jax.tree.map(lambda base, delta: base + delta, selected, correction)
        updated.update(values)
        return frozendict(updated)


class CallableCorrectionChart(AbstractCorrectionChart):
    """A caller-defined local chart with an explicitly declared origin and retraction."""

    origin_function: Callable[[FieldMap], PyTree[Any]] = eqx.field(static=True)
    retraction_function: Callable[[FieldMap, PyTree[Any]], Mapping[str, Any]] = eqx.field(
        static=True
    )
    chart_id: str = eqx.field(static=True)

    def __init__(
        self,
        origin: Callable[[FieldMap], PyTree[Any]],
        retraction: Callable[[FieldMap, PyTree[Any]], Mapping[str, Any]],
        /,
        *,
        chart_id: str,
    ):
        if not callable(origin) or not callable(retraction):
            raise TypeError("Correction-chart origin and retraction must be callable.")
        identifier = str(chart_id)
        if not identifier:
            raise ValueError("chart_id must be non-empty.")
        self.origin_function = origin
        self.retraction_function = retraction
        self.chart_id = identifier

    def origin(self, fields: FieldMap, /) -> PyTree[Array]:
        return validate_real_inexact_tree(
            self.origin_function(fields), name="correction-chart origin"
        )

    def retract(
        self, fields: FieldMap, coordinates: PyTree[Any], /
    ) -> frozendict[str, Any]:
        origin = self.origin(fields)
        correction = validate_real_inexact_tree(
            coordinates, name="correction-chart coordinates"
        )
        if jax.tree.structure(origin) != jax.tree.structure(correction):
            raise ValueError("Correction coordinates do not match the chart origin.")
        result = self.retraction_function(fields, correction)
        if not isinstance(result, Mapping):
            raise TypeError("A correction-chart retraction must return a field mapping.")
        if set(result) != set(fields):
            raise ValueError(
                "A correction chart must preserve the complete field key set."
            )
        return frozendict(result)


class ImplicitRetractionEvidence(StrictModule):
    """Independent physical and first-order evidence for an implicit retraction."""

    stamp: ConditionRealizationStamp
    condition: ConditionEvidence
    certificate: NonlinearRetractionCertificate
    nonlinear_result: NonlinearResult
    correction_norm: Array
    stationarity_residual_norm: Array
    finite: Array
    certified: Array
    objective: RetractionObjective = eqx.field(static=True)
    chart_id: str = eqx.field(static=True)
    optimality_claim: str = eqx.field(static=True)


class NonlinearFieldRetraction(AbstractFieldRealization):
    """Retract fields to a typed condition in caller-supplied local coordinates.

    ``local-root`` solves the represented relation residual from the chart origin.
    ``minimum-distance`` solves the equality-constrained first-order KKT system for
    the squared Euclidean chart correction. The latter certifies stationarity only;
    it deliberately makes no global nearest-point claim.
    """

    chart: AbstractCorrectionChart
    method: AbstractNonlinearMethod
    termination: NonlinearTermination
    sources: tuple[AbstractRealizationSource, ...]
    refresh_validator: RefreshValidator | None = eqx.field(static=True)
    objective: RetractionObjective = eqx.field(static=True)
    certification_tolerance: float = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(
        self,
        chart: AbstractCorrectionChart,
        /,
        *,
        objective: RetractionObjective = "local-root",
        method: AbstractNonlinearMethod | None = None,
        termination: NonlinearTermination | None = None,
        sources: Sequence[AbstractRealizationSource] = (),
        refresh_validator: RefreshValidator | None = None,
        certification_tolerance: float = 1e-8,
        provider_id: str | None = None,
    ):
        method_ = NewtonKrylov() if method is None else method
        termination_ = NonlinearTermination() if termination is None else termination
        sources_ = tuple(sources)
        tolerance = float(certification_tolerance)
        if not isinstance(chart, AbstractCorrectionChart):
            raise TypeError("chart must be an AbstractCorrectionChart.")
        if objective not in ("local-root", "minimum-distance"):
            raise ValueError("Unknown nonlinear retraction objective.")
        if not isinstance(method_, AbstractNonlinearMethod):
            raise TypeError("method must be an AbstractNonlinearMethod or None.")
        if not isinstance(termination_, NonlinearTermination):
            raise TypeError("termination must be NonlinearTermination or None.")
        if any(not isinstance(source, AbstractRealizationSource) for source in sources_):
            raise TypeError("sources must contain AbstractRealizationSource values.")
        names = tuple(source.name for source in sources_)
        if len(set(names)) != len(names):
            raise ValueError("Realization source names must be unique.")
        if refresh_validator is not None and not callable(refresh_validator):
            raise TypeError("refresh_validator must be callable or None.")
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("certification_tolerance must be finite and positive.")
        identifier = (
            f"phydrax.enforcement.{objective}/{chart.chart_id}/{method_.method_id}"
            if provider_id is None
            else str(provider_id)
        )
        if not identifier:
            raise ValueError("provider_id must be non-empty.")
        self.chart = chart
        self.method = method_
        self.termination = termination_
        self.sources = sources_
        self.refresh_validator = refresh_validator
        self.objective = objective
        self.certification_tolerance = tolerance
        self.provider_id = identifier

    def _failed_state(
        self,
        state: RealizationLifecycleState,
        status: RealizationStatus,
        message: str,
        context: ConditionEvaluationContext,
        evidence: Any = None,
    ) -> RealizationLifecycleState:
        return RealizationLifecycleState(
            phase=RealizationLifecyclePhase.FAILED,
            generation=state.generation,
            accepted_step=state.accepted_step,
            parameter_revision=state.parameter_revision,
            values=state.values,
            source_stamps=state.source_stamps,
            realization_stamp=state.realization_stamp,
            last_failure=RealizationFailure(
                status,
                message,
                accepted_step=context.accepted_step,
                attempt=context.attempt,
                evidence=evidence,
            ),
        )

    @staticmethod
    def _bound(condition: Condition, fields: FieldMap, /) -> BoundCondition:
        missing = tuple(
            source for source in condition.fields.sources if source not in fields
        )
        if missing:
            raise KeyError(f"Condition field sources are missing: {missing!r}.")
        return BoundCondition(
            condition,
            {source: fields[source] for source in condition.fields.sources},
        )

    def realize(
        self,
        fields: FieldMap,
        state: RealizationLifecycleState | None = None,
        *,
        context: ConditionEvaluationContext,
    ) -> FieldRealizationResult:
        if not isinstance(fields, Mapping):
            raise TypeError("fields must be a mapping.")
        if not isinstance(context, ConditionEvaluationContext):
            raise TypeError("context must be a ConditionEvaluationContext.")
        current = RealizationLifecycleState.initial() if state is None else state
        if not isinstance(current, RealizationLifecycleState):
            raise TypeError("state must be RealizationLifecycleState or None.")
        field_values = frozendict(fields)
        self._bound(context.condition, field_values)

        proposal = propose_refresh(self.sources, current, context=context)
        validation = validate_refresh(proposal, self.refresh_validator)
        ready = commit_refresh(current, proposal, validation)
        if not validation.accepted:
            return FieldRealizationResult.failure(
                validation.status,
                state=ready,
                message=validation.message,
                evidence=validation.evidence,
            )
        if context.quantifier not in (
            ConditionQuantifier.deterministic,
            ConditionQuantifier.samplewise,
        ):
            message = (
                "Nonlinear field retraction cannot certify expectation, almost-sure, "
                "or chance quantifiers from one deterministic evaluation."
            )
            failed = self._failed_state(
                ready, RealizationStatus.UNSUPPORTED, message, context
            )
            return FieldRealizationResult.failure(
                RealizationStatus.UNSUPPORTED, state=failed, message=message
            )
        if isinstance(context.condition.relation, NoisyObservation):
            message = "Noisy observations require probabilistic conditioning evidence."
            failed = self._failed_state(
                ready, RealizationStatus.UNSUPPORTED, message, context
            )
            return FieldRealizationResult.failure(
                RealizationStatus.UNSUPPORTED, state=failed, message=message
            )

        origin = self.chart.origin(field_values)
        source_arguments = dict(ready.values)

        def physical_residual(coordinates, _args):
            candidate = self.chart.retract(field_values, coordinates)
            bound = self._bound(context.condition, candidate)
            value = bound.apply(key=context.prng_key, **source_arguments)
            return _relation_residual(bound, value)

        initial_residual = physical_residual(origin, None)
        if self.objective == "local-root":
            problem = NonlinearSystemProblem(
                physical_residual,
                problem_id=f"{context.condition_id}/local-retraction/{self.chart.chart_id}",
            )
            nonlinear_result = self.method.solve(
                problem,
                origin,
                termination=self.termination,
            )
            coordinates = nonlinear_result.state
            multipliers = None
            stationarity = jnp.asarray(0.0, dtype=tree_norm(initial_residual).dtype)
        else:
            multiplier_origin = jax.tree.map(jnp.zeros_like, initial_residual)

            def kkt_residual(kkt_state, _args):
                coordinates_, multipliers_ = kkt_state
                residual_, pullback = jax.vjp(
                    lambda value: physical_residual(value, None), coordinates_
                )
                adjoint = pullback(multipliers_)[0]
                stationarity_ = jax.tree.map(
                    lambda value, base, dual: value - base + dual,
                    coordinates_,
                    origin,
                    adjoint,
                )
                return stationarity_, residual_

            problem = NonlinearSystemProblem(
                kkt_residual,
                problem_id=(
                    f"{context.condition_id}/minimum-distance-retraction/"
                    f"{self.chart.chart_id}"
                ),
            )
            nonlinear_result = self.method.solve(
                problem,
                (origin, multiplier_origin),
                termination=self.termination,
            )
            coordinates, multipliers = nonlinear_result.state
            residual_at_solution, pullback = jax.vjp(
                lambda value: physical_residual(value, None), coordinates
            )
            adjoint = pullback(multipliers)[0]
            stationarity_tree = jax.tree.map(
                lambda value, base, dual: value - base + dual,
                coordinates,
                origin,
                adjoint,
            )
            stationarity = tree_norm(stationarity_tree)
            initial_residual = residual_at_solution

        candidate = self.chart.retract(field_values, coordinates)
        residual = physical_residual(coordinates, None)
        residual_norm = tree_norm(residual)
        correction = jax.tree.map(lambda value, base: value - base, coordinates, origin)
        correction_norm = tree_norm(correction)
        finite = (
            tree_allfinite(coordinates)
            & tree_allfinite(residual)
            & jnp.isfinite(stationarity)
        )
        tolerance = jnp.asarray(
            0.0 if context.exact_required else self.certification_tolerance,
            dtype=residual_norm.dtype,
        )
        certified = (
            finite
            & nonlinear_result.successful
            & (residual_norm <= tolerance)
            & (stationarity <= tolerance)
        )
        exact = finite & (residual_norm == 0.0) & (stationarity == 0.0)
        stamp = ConditionRealizationStamp(
            context.condition_id,
            self._bound(context.condition, field_values).bound_id,
            canonical_fingerprint(
                {
                    "condition": context.condition_id,
                    "chart": self.chart.chart_id,
                    "objective": self.objective,
                    "accepted_step": context.accepted_step,
                    "parameter_revision": context.parameter_revision,
                    "generation": ready.generation,
                }
            ),
            self.provider_id,
            quantifier=context.quantifier,
            exact=bool(exact),
        )
        certificate = NonlinearRetractionCertificate(
            stamp,
            residual_norm,
            tolerance,
            certified,
            certificate_id=f"{self.provider_id}/physical-residual",
            iterations=int(nonlinear_result.diagnostics.iterations),
        )
        condition_evidence = ConditionEvidence(
            stamp,
            residual_norm,
            certified,
            evidence_id=f"{self.provider_id}/implicit-retraction",
        )
        evidence = ImplicitRetractionEvidence(
            stamp=stamp,
            condition=condition_evidence,
            certificate=certificate,
            nonlinear_result=nonlinear_result,
            correction_norm=correction_norm,
            stationarity_residual_norm=stationarity,
            finite=finite,
            certified=certified,
            objective=self.objective,
            chart_id=self.chart.chart_id,
            optimality_claim=(
                "constraint-root"
                if self.objective == "local-root"
                else "first-order-stationary-distance"
            ),
        )
        if not bool(certified):
            if not bool(finite):
                status = RealizationStatus.NONFINITE
                message = "Nonlinear retraction produced non-finite numerical evidence."
            elif not bool(nonlinear_result.successful):
                status = RealizationStatus.SOLVE_FAILED
                message = nonlinear_status_message(int(nonlinear_result.status))
            else:
                status = RealizationStatus.VALIDATION_FAILED
                message = (
                    "Nonlinear retraction failed independent physical certification."
                )
            failed = self._failed_state(ready, status, message, context, evidence)
            return FieldRealizationResult.failure(
                status, state=failed, message=message, evidence=evidence
            )

        committed = record_realization_stamp(ready, stamp)
        return FieldRealizationResult.success(
            candidate,
            state=committed,
            stamp=stamp,
            evidence=evidence,
            unchanged=bool(correction_norm == 0.0),
        )


class LocalNonlinearRetraction(AbstractFieldRealization):
    """Local root retraction from a correction-chart origin."""

    realization: NonlinearFieldRetraction

    def __init__(self, chart: AbstractCorrectionChart, /, **kwargs: Any):
        self.realization = NonlinearFieldRetraction(
            chart, objective="local-root", **kwargs
        )

    def realize(
        self,
        fields: FieldMap,
        state: RealizationLifecycleState | None = None,
        *,
        context: ConditionEvaluationContext,
    ) -> FieldRealizationResult:
        return self.realization.realize(fields, state, context=context)


class MinimumDistanceRetraction(AbstractFieldRealization):
    """First-order stationary minimum-distance retraction in chart coordinates."""

    realization: NonlinearFieldRetraction

    def __init__(self, chart: AbstractCorrectionChart, /, **kwargs: Any):
        self.realization = NonlinearFieldRetraction(
            chart, objective="minimum-distance", **kwargs
        )

    def realize(
        self,
        fields: FieldMap,
        state: RealizationLifecycleState | None = None,
        *,
        context: ConditionEvaluationContext,
    ) -> FieldRealizationResult:
        return self.realization.realize(fields, state, context=context)


def _relation_residual(bound: BoundCondition, value: Any, /) -> PyTree[Array]:
    relation = bound.relation
    if isinstance(relation, Equality):
        residual = (
            value
            if not relation.has_target
            else jax.tree.map(lambda left, right: left - right, value, relation.target)
        )
    elif isinstance(relation, Inequality):
        if relation.strict_lower or relation.strict_upper:
            raise TypeError(
                "Strict inequalities require an interior parameterization, not a retraction."
            )
        projected = value
        if relation.has_lower:
            projected = jax.tree.map(jnp.maximum, projected, relation.lower)
        if relation.has_upper:
            projected = jax.tree.map(jnp.minimum, projected, relation.upper)
        residual = jax.tree.map(lambda left, right: left - right, value, projected)
    elif isinstance(relation, ConeMembership):
        if not isinstance(bound.codomain, ArrayCodomain):
            raise TypeError("Nonlinear cone retraction currently requires ArrayCodomain.")
        array = jnp.asarray(value)
        if relation.cone is ConeKind.nonnegative:
            projected = jnp.maximum(array, 0.0)
        elif relation.cone is ConeKind.nonpositive:
            projected = jnp.minimum(array, 0.0)
        elif relation.cone is ConeKind.simplex:
            axis = relation.axis % array.ndim
            moved = jnp.moveaxis(array, axis, -1)
            projected = jnp.moveaxis(SimplexProjection().apply(moved), -1, axis)
        elif relation.cone is ConeKind.second_order:
            axis = relation.axis % array.ndim
            moved = jnp.moveaxis(array, axis, -1)
            projected = jnp.moveaxis(
                SecondOrderCone(int(moved.shape[-1])).project(moved), -1, axis
            )
        else:
            projected = PositiveSemidefiniteProjection(int(array.shape[-1])).apply(array)
        residual = array - projected
    elif isinstance(relation, Complementarity):
        left, right = value
        residual = jax.tree.map(
            lambda primal, dual: primal - jnp.maximum(primal - dual, 0.0),
            left,
            right,
        )
    elif isinstance(relation, NoisyObservation):
        raise TypeError("Noisy observations do not define deterministic residual roots.")
    else:
        raise TypeError("Unsupported typed condition relation.")
    return validate_real_inexact_tree(residual, name="condition relation residual")


__all__ = [
    "AbstractCorrectionChart",
    "AdditiveCorrectionChart",
    "CallableCorrectionChart",
    "ImplicitRetractionEvidence",
    "LocalNonlinearRetraction",
    "MinimumDistanceRetraction",
    "NonlinearFieldRetraction",
    "RetractionObjective",
]
