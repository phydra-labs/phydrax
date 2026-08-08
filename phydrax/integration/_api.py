#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, cast

import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, Key

from phydrax.domain import ComponentSum, DomainFunction

from .._doc import DOC_KEY0
from .._sampling import AntitheticDesign, design_capabilities
from .._strict import StrictModule
from ._adaptive import integrate_adaptive
from ._estimates import IntegrationEstimate
from ._external import (
    integrate_discrete_measure,
    integrate_weighted_samples,
    materialize_discrete_target,
    materialize_weighted_target,
)
from ._fixed import integrate_fixed_component, integrate_fixed_density
from ._lowering import materialize_fixed_component, materialize_sampled_component
from ._mapped import integrate_mapped, materialize_mapped
from ._monte_carlo import (
    integrate_monte_carlo,
    materialize_importance,
    materialize_monte_carlo,
    materialize_stratified,
)
from ._multilevel import integrate_multilevel, materialize_multilevel
from ._plans import (
    AdaptiveQuadraturePlan,
    CellQuadraturePlan,
    FixedQuadraturePlan,
    ImportanceSamplingPlan,
    MonteCarloPlan,
    MultilevelMonteCarloPlan,
    ProductIntegrationPlan,
    QuasiMonteCarloPlan,
    SelfNormalizedEstimator,
    SparseGridPlan,
    StratifiedMonteCarloPlan,
)
from ._probability import (
    integrate_fixed_probability,
    materialize_fixed_probability,
)
from ._product import integrate_product, materialize_product
from ._sparse_grid import integrate_sparse_grid, materialize_sparse_grid
from ._status import IntegrationStatus
from ._targets import (
    ComponentTarget,
    DensityTarget,
    DiscreteMeasureTarget,
    MappedTarget,
    MultilevelTarget,
    ProbabilityTarget,
    WeightedSampleTarget,
)


_KEY_UNSET = object()


class IntegrationRealization(StrictModule):
    """A target, plan, reusable batch, execution key, and optional compression."""

    target: Any
    plan: Any
    batch: Any
    key: Any
    compression: Any | None = None

def _attach_compression(
    estimate: IntegrationEstimate,
    realization: IntegrationRealization,
    /,
) -> IntegrationEstimate:
    if realization.compression is None:
        return estimate
    from ._compression import CompressedIntegrationDiagnostics
    from ._estimates import IntegrationProvenance

    return IntegrationEstimate(
        estimate.value,
        status=estimate.status,
        num_evaluations=estimate.num_evaluations,
        error_estimate=None,
        error_kind=None,
        diagnostics=CompressedIntegrationDiagnostics(
            realization.compression,
            estimate.diagnostics,
        ),
        provenance=IntegrationProvenance(
            "compressed",
            estimate.provenance.target,
            estimate.provenance.realization,
        ),
    )



def _base_target(target: Any, /) -> Any:
    return target.base if isinstance(target, DensityTarget) else target


def _requires_random_key(plan: Any, /) -> bool:
    if isinstance(plan, ProductIntegrationPlan):
        return any(_requires_random_key(factor) for factor in plan.plans.values())
    if isinstance(
        plan,
        (ImportanceSamplingPlan, StratifiedMonteCarloPlan, MultilevelMonteCarloPlan),
    ):
        return True
    if isinstance(plan, (MonteCarloPlan, QuasiMonteCarloPlan)):
        design = plan.design
        base = design.base if isinstance(design, AntitheticDesign) else design
        return design_capabilities(base).randomized
    return False


def _is_deterministic_plan(plan: Any, /) -> bool:
    if isinstance(plan, ProductIntegrationPlan):
        return all(_is_deterministic_plan(factor) for factor in plan.plans.values())
    if isinstance(
        plan,
        (
            FixedQuadraturePlan,
            AdaptiveQuadraturePlan,
            CellQuadraturePlan,
            SparseGridPlan,
        ),
    ):
        return True
    if isinstance(plan, (MonteCarloPlan, QuasiMonteCarloPlan)):
        design = plan.design
        base = design.base if isinstance(design, AntitheticDesign) else design
        return not design_capabilities(base).randomized
    return False


def materialize(
    target: Any,
    plan: Any = None,
    /,
    *,
    key: Key[Array, ""] | object = _KEY_UNSET,
) -> IntegrationRealization:
    """Materialize a target under a typed plan without evaluating an integrand."""
    if isinstance(target, (DiscreteMeasureTarget, WeightedSampleTarget)):
        if plan is not None:
            raise TypeError("External measures do not take an integration plan.")
        if key is not _KEY_UNSET:
            raise ValueError("External measures do not consume a random key.")
        batch = (
            materialize_discrete_target(target)
            if isinstance(target, DiscreteMeasureTarget)
            else materialize_weighted_target(target)
        )
        return IntegrationRealization(target, None, batch, None)
    if plan is None:
        raise TypeError("An integration plan is required for this target.")
    if _requires_random_key(plan) and key is _KEY_UNSET:
        raise ValueError(
            f"{type(plan).__name__} has randomized execution and requires key=."
        )
    if _is_deterministic_plan(plan) and key is not _KEY_UNSET:
        raise ValueError(
            f"{type(plan).__name__} is deterministic and does not consume key=."
        )
    if key is _KEY_UNSET:
        sampling_key: Any = DOC_KEY0
        evaluation_key: Any = None
    else:
        sampling_key, evaluation_key = jr.split(cast(Key[Array, ""], key))
    base = _base_target(target)
    if isinstance(plan, FixedQuadraturePlan):
        if isinstance(base, ComponentTarget):
            batch = materialize_fixed_component(base, plan)
        elif isinstance(base, ProbabilityTarget):
            batch = materialize_fixed_probability(base, plan)
        else:
            raise TypeError(
                "Fixed quadrature requires a component or probability target."
            )
    elif isinstance(plan, AdaptiveQuadraturePlan):
        if not isinstance(base, ComponentTarget):
            raise TypeError("Adaptive quadrature requires a component target.")
        batch = None
    elif isinstance(plan, (MonteCarloPlan, QuasiMonteCarloPlan)):
        batch = materialize_monte_carlo(target, plan, key=sampling_key)
    elif isinstance(plan, StratifiedMonteCarloPlan):
        batch = materialize_stratified(target, plan, key=sampling_key)
    elif isinstance(plan, ImportanceSamplingPlan):
        batch = materialize_importance(target, plan, key=sampling_key)
    elif isinstance(plan, MultilevelMonteCarloPlan):
        if not isinstance(target, MultilevelTarget):
            raise TypeError("MultilevelMonteCarloPlan requires a multilevel target.")
        batch = materialize_multilevel(target, plan, sampling_key)
    elif isinstance(plan, SparseGridPlan):
        batch = materialize_sparse_grid(target, plan)
    elif isinstance(plan, CellQuadraturePlan):
        if not isinstance(base, MappedTarget):
            raise TypeError("CellQuadraturePlan requires a mapped target.")
        batch = materialize_mapped(base, plan)
    elif isinstance(plan, ProductIntegrationPlan):
        batch = materialize_product(target, plan, key=sampling_key)
    else:
        raise TypeError(f"Unsupported integration plan {type(plan).__name__}.")
    return IntegrationRealization(target, plan, batch, evaluation_key)


def from_samples(
    target: ComponentTarget | DensityTarget,
    points: Any,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
) -> IntegrationRealization:
    """Attach authoritative target-measure weights to an existing point batch."""
    base = _base_target(target)
    if not isinstance(base, ComponentTarget):
        raise TypeError("from_samples requires a component-based target.")
    if isinstance(base.component, ComponentSum):
        if not isinstance(points, tuple) or len(points) != len(base.component.terms):
            raise ValueError("Component unions require one aligned point batch per term.")
        batches = tuple(
            materialize_sampled_component(
                ComponentTarget(
                    component,
                    axes=base.axes,
                    normalized=base.normalized,
                ),
                term_points,
            )
            for component, term_points in zip(base.component.terms, points, strict=True)
        )
    else:
        if isinstance(points, tuple):
            raise TypeError("A single component target requires one point batch.")
        batches = materialize_sampled_component(base, points)
    return IntegrationRealization(target, None, batches, key)


def _integrand_leaf(value: Any, /) -> bool:
    return isinstance(value, (DomainFunction, cx.Field)) or callable(value)


def _reduce_integrand_tree(
    integrand: Any,
    realization: IntegrationRealization,
    kwargs: dict[str, Any],
    /,
) -> IntegrationEstimate | None:
    leaves, structure = jtu.tree_flatten(integrand, is_leaf=_integrand_leaf)
    if structure.num_nodes == 1:
        return None
    if not leaves:
        raise ValueError("An integration output PyTree must contain at least one leaf.")
    estimates = tuple(reduce(leaf, realization, **kwargs) for leaf in leaves)
    status = estimates[0].status
    for estimate in estimates[1:]:
        status = jnp.where(
            status == int(IntegrationStatus.CONVERGED),
            estimate.status,
            status,
        )
    errors = tuple(estimate.error_estimate for estimate in estimates)
    if all(error is not None for error in errors):
        error_estimate = jnp.max(jnp.stack(tuple(jnp.asarray(error) for error in errors)))
    else:
        error_estimate = None
    error_kinds = {estimate.error_kind for estimate in estimates}
    error_kind = estimates[0].error_kind if len(error_kinds) == 1 else None
    return IntegrationEstimate(
        jtu.tree_unflatten(structure, [estimate.value for estimate in estimates]),
        status=status,
        num_evaluations=jnp.max(
            jnp.stack(tuple(estimate.num_evaluations for estimate in estimates))
        ),
        error_estimate=error_estimate,
        error_kind=error_kind,
        diagnostics=jtu.tree_unflatten(
            structure, [estimate.diagnostics for estimate in estimates]
        ),
        provenance=estimates[0].provenance,
    )


def reduce(
    integrand: Any,
    realization: IntegrationRealization,
    /,
    **kwargs: Any,
):
    """Reduce an integrand against a reusable typed realization."""
    if not isinstance(realization, IntegrationRealization):
        raise TypeError("reduce expects an IntegrationRealization from materialize().")
    tree_estimate = _reduce_integrand_tree(integrand, realization, kwargs)
    if tree_estimate is not None:
        return tree_estimate
    target = realization.target
    plan = realization.plan
    key = DOC_KEY0 if realization.key is None else realization.key
    if isinstance(target, WeightedSampleTarget):
        estimate = integrate_weighted_samples(
            integrand, target, realization.batch, key=key, kwargs=kwargs
        )
        return _attach_compression(estimate, realization)
    if isinstance(target, DiscreteMeasureTarget):
        return integrate_discrete_measure(
            integrand, target, realization.batch, key=key, kwargs=kwargs
        )
    if isinstance(plan, MultilevelMonteCarloPlan):
        if not isinstance(target, MultilevelTarget):
            raise TypeError("MultilevelMonteCarloPlan requires a multilevel target.")
        return integrate_multilevel(integrand, realization.batch, **kwargs)
    base = _base_target(target)
    if plan is None and isinstance(base, ComponentTarget):
        if isinstance(target, DensityTarget):
            return integrate_fixed_density(
                integrand, target, realization.batch, key=key, kwargs=kwargs
            )
        return integrate_fixed_component(
            integrand, target, realization.batch, key=key, kwargs=kwargs
        )
    if isinstance(plan, FixedQuadraturePlan):
        if isinstance(base, ComponentTarget):
            if isinstance(target, DensityTarget):
                return integrate_fixed_density(
                    integrand, target, realization.batch, key=key, kwargs=kwargs
                )
            return integrate_fixed_component(
                integrand, target, realization.batch, key=key, kwargs=kwargs
            )
        return integrate_fixed_probability(
            integrand, target, realization.batch, key=key, kwargs=kwargs
        )
    if isinstance(plan, AdaptiveQuadraturePlan):
        return integrate_adaptive(integrand, target, plan, key=key, kwargs=kwargs)
    if isinstance(
        plan,
        (MonteCarloPlan, QuasiMonteCarloPlan, StratifiedMonteCarloPlan),
    ):
        return integrate_monte_carlo(
            integrand,
            target,
            realization.batch,
            plan=plan,
            key=key,
            kwargs=kwargs,
        )
    if isinstance(plan, ImportanceSamplingPlan):
        normalized = isinstance(plan.estimator, SelfNormalizedEstimator) or (
            isinstance(target, DensityTarget) and target.normalized
        )
        weighted_target = WeightedSampleTarget(
            realization.batch.samples,
            realization.batch.log_weights,
            normalized=normalized,
            independent=True,
            sample_axes=0,
            provenance=realization.batch.provenance,
        )
        return integrate_weighted_samples(
            integrand,
            weighted_target,
            realization.batch,
            normalized=normalized,
            key=key,
            kwargs=kwargs,
        )
    if isinstance(plan, SparseGridPlan):
        return integrate_sparse_grid(
            integrand, target, realization.batch, key=key, kwargs=kwargs
        )
    if isinstance(plan, CellQuadraturePlan):
        return integrate_mapped(
            integrand, target, realization.batch, key=key, kwargs=kwargs
        )
    if isinstance(plan, ProductIntegrationPlan):
        return integrate_product(
            integrand, target, realization.batch, key=key, kwargs=kwargs
        )
    raise TypeError(f"Unsupported realization plan {type(plan).__name__}.")


def integrate(
    integrand: Any,
    target: Any,
    plan: Any = None,
    /,
    *,
    key: Key[Array, ""] | object = _KEY_UNSET,
    **kwargs: Any,
):
    """Materialize and reduce an integration target in one call."""
    realization = materialize(target, plan, key=key)
    return reduce(integrand, realization, **kwargs)


__all__ = [
    "IntegrationRealization",
    "from_samples",
    "integrate",
    "materialize",
    "reduce",
]
