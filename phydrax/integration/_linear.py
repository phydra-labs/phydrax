#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable coefficient actions for already-materialized integration measures."""

from __future__ import annotations

from typing import Any

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from phydrax.domain import ComponentSum, DomainFunction, PointBatch

from .._callable import _ensure_special_kwonly_args
from .._doc import DOC_KEY0
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._api import _requires_random_key, IntegrationRealization
from ._batches import (
    MappedIntegrationBatch,
    PointIntegrationBatch,
    SeparableIntegrationBatch,
)
from ._external import _as_weight_field
from ._fixed import _as_domain_function, _batch_weight, _target_reduction_weights
from ._lowering import sum_over
from ._mapped import _mapped_values
from ._plans import (
    AdaptiveCubaturePlan,
    AdaptiveQuadraturePlan,
    AdaptiveSparseGridPlan,
    AdaptiveTrianglePlan,
)
from ._precision import IntegrationPrecisionPolicy
from ._product import ProductIntegrationRealization
from ._sparse_grid import SparseGridRealization
from ._targets import (
    ComponentTarget,
    DensityTarget,
    DiscreteMeasureTarget,
    MappedTarget,
    ProbabilityTarget,
)


_FixedBatch = PointIntegrationBatch | SeparableIntegrationBatch | MappedIntegrationBatch


class LinearReductionSchema(StrictModule):
    """Static named-axis schema of one prepared coefficient action."""

    target_id: str = eqx.field(static=True)
    reduced_axes: tuple[str, ...] = eqx.field(static=True)
    retained_axes: tuple[str, ...] = eqx.field(static=True)
    retained_shape: tuple[int, ...] = eqx.field(static=True)
    coefficient_dtype: str = eqx.field(static=True)
    coefficient_layout: tuple[tuple[str | None, ...], ...] = eqx.field(static=True)
    event_shape: tuple[int, ...] | None = eqx.field(static=True)
    event_dtype: str | None = eqx.field(static=True)
    quantifier: str = eqx.field(static=True)
    exactness: str = eqx.field(static=True)
    source_provenance: str = eqx.field(static=True)
    output_suffix_policy: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        target_id: str,
        reduced_axes: tuple[str, ...],
        retained_axes: tuple[str, ...],
        retained_shape: tuple[int, ...],
        coefficient_dtype: str,
        coefficient_layout: tuple[tuple[str | None, ...], ...],
        quantifier: str,
        exactness: str,
        source_provenance: str,
        event_shape: tuple[int, ...] | None = None,
        event_dtype: str | None = None,
        output_suffix_policy: str = "dynamic-positional-event-suffix",
    ):
        target_id_ = str(target_id)
        reduced = tuple(reduced_axes)
        retained = tuple(retained_axes)
        shape = tuple(int(size) for size in retained_shape)
        if not target_id_:
            raise ValueError("target_id must be non-empty.")
        if not reduced or any(not axis for axis in reduced):
            raise ValueError("reduced_axes must contain at least one non-empty axis.")
        if len(frozenset(reduced + retained)) != len(reduced) + len(retained):
            raise ValueError("Reduced and retained axes must be unique and disjoint.")
        if len(shape) != len(retained) or any(size <= 0 for size in shape):
            raise ValueError("retained_shape must align with retained_axes.")
        coefficient_dtype_ = str(coefficient_dtype)
        coefficient_layout_ = tuple(tuple(dims) for dims in coefficient_layout)
        quantifier_ = str(quantifier)
        exactness_ = str(exactness)
        source_provenance_ = str(source_provenance)
        suffix_policy = str(output_suffix_policy)
        if (
            not coefficient_dtype_
            or not coefficient_layout_
            or not quantifier_
            or not exactness_
            or not source_provenance_
            or not suffix_policy
        ):
            raise ValueError("Reduction schema provenance and layout must be non-empty.")
        if event_shape is not None or event_dtype is not None:
            raise ValueError(
                "Prepared reductions do not invent integrand event metadata; "
                "event_shape and event_dtype must remain None until condition binding."
            )
        self.target_id = target_id_
        self.reduced_axes = reduced
        self.retained_axes = retained
        self.retained_shape = shape
        self.coefficient_dtype = coefficient_dtype_
        self.coefficient_layout = coefficient_layout_
        self.event_shape = None
        self.event_dtype = None
        self.quantifier = quantifier_
        self.exactness = exactness_
        self.source_provenance = source_provenance_
        self.output_suffix_policy = suffix_policy

    @property
    def dtype(self) -> str | None:
        """Return the unresolved integrand event dtype supplied at condition binding."""
        return self.event_dtype

    @property
    def retained_layout(self) -> tuple[tuple[str, int], ...]:
        return tuple(zip(self.retained_axes, self.retained_shape, strict=True))


class LinearReductionEvidence(StrictModule, NonTrainableState):
    """Content and exactness evidence for a frozen coefficient action."""

    numeric_version: Array
    finite: Array
    realization_id: str = eqx.field(static=True)
    exactness: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)
    reduced_axes: tuple[str, ...] = eqx.field(static=True)
    retained_axes: tuple[str, ...] = eqx.field(static=True)
    coefficient_dtype: str = eqx.field(static=True)
    coefficient_shape: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    coefficient_layout: tuple[tuple[str | None, ...], ...] = eqx.field(static=True)
    normalized: bool = eqx.field(static=True)
    source_provenance: str = eqx.field(static=True)
    transformation_ids: tuple[str, ...] = eqx.field(static=True)


class PreparedLinearReduction(StrictModule, NonTrainableState):
    """An immutable linear action over fixed integration coefficients."""

    target: Any
    batches: tuple[_FixedBatch, ...]
    coefficient_fields: tuple[cx.Field, ...]
    extra_weight_fields: tuple[cx.Field, ...]
    precision: IntegrationPrecisionPolicy
    numeric_version: Array
    evidence: LinearReductionEvidence
    schema: LinearReductionSchema = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)

    @property
    def batch(self) -> _FixedBatch | tuple[_FixedBatch, ...]:
        return self.batches[0] if len(self.batches) == 1 else self.batches

    @property
    def points(self) -> Any:
        values = tuple(batch.points for batch in self.batches)
        return values[0] if len(values) == 1 else values

    @property
    def coefficients(self) -> cx.Field | tuple[cx.Field, ...]:
        return (
            self.coefficient_fields[0]
            if len(self.coefficient_fields) == 1
            else self.coefficient_fields
        )

    def apply(
        self,
        function: Any,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> Any:
        """Evaluate fixed points with the caller key and contract stored coefficients."""
        outputs = tuple(
            _apply_batch(
                function,
                self.target,
                batch,
                coefficient,
                index=index,
                reduced_axes=self.schema.reduced_axes,
                retained_axes=self.schema.retained_axes,
                retained_shape=self.schema.retained_shape,
                key=key,
                kwargs=kwargs,
                precision=self.precision,
            )
            for index, (batch, coefficient) in enumerate(
                zip(self.batches, self.coefficient_fields, strict=True)
            )
        )
        result = outputs[0]
        for output in outputs[1:]:
            result = jax.tree_util.tree_map(
                lambda left, right: left + right,
                result,
                output,
                is_leaf=lambda value: isinstance(value, cx.Field),
            )
        return _output_precision(result, self.precision)

    def refresh(
        self,
        realization: IntegrationRealization,
        /,
        *,
        retained_axes: tuple[str, ...] | None = None,
        weight: Any | None = None,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> PreparedLinearReduction:
        return refresh_linear_reduction(
            self,
            realization,
            retained_axes=retained_axes,
            weight=weight,
            key=key,
            **kwargs,
        )


def _base_target(target: Any, /) -> Any:
    return target.base if isinstance(target, DensityTarget) else target


def _fixed_batches(realization: IntegrationRealization, /) -> tuple[_FixedBatch, ...]:
    batch = realization.batch
    if batch is None and isinstance(
        realization.plan,
        (AdaptiveQuadraturePlan, AdaptiveCubaturePlan, AdaptiveTrianglePlan),
    ):
        raise ValueError(
            "Adaptive integration has no fixed coefficients until its action is frozen."
        )
    if isinstance(batch, (PointIntegrationBatch, SeparableIntegrationBatch)):
        return (batch,)
    if isinstance(batch, MappedIntegrationBatch):
        return (batch,)
    if isinstance(batch, SparseGridRealization):
        return (batch.batch,)
    if isinstance(batch, ProductIntegrationRealization):
        return tuple(batch.batches)
    if (
        isinstance(batch, tuple)
        and batch
        and all(
            isinstance(term, (PointIntegrationBatch, SeparableIntegrationBatch))
            for term in batch
        )
    ):
        return tuple(batch)
    raise TypeError(
        "Linear reduction requires fixed point, separable, mapped, product, or "
        "frozen sparse-grid coefficients."
    )


def _batch_axes(batch: _FixedBatch, /) -> tuple[str, ...]:
    if isinstance(batch, MappedIntegrationBatch):
        return (batch.axis,)
    return batch.axes


def _active_axes(batches: tuple[_FixedBatch, ...], /) -> tuple[str, ...]:
    axes = _batch_axes(batches[0])
    if any(_batch_axes(batch) != axes for batch in batches[1:]):
        raise ValueError("Additive fixed batches must share the same reduction axes.")
    return axes


def _mapped_coefficients(
    target: MappedTarget | DensityTarget,
    batch: MappedIntegrationBatch,
    /,
    *,
    reduced_axes: tuple[str, ...],
    key: Key[Array, ""],
    kwargs: dict[str, Any],
) -> cx.Field:
    base = _base_target(target)
    if not isinstance(base, MappedTarget):
        raise TypeError("Mapped batches require a mapped integration target.")
    coefficient = cx.Field(jnp.where(batch.mask, batch.weights, 0.0), dims=(batch.axis,))
    if isinstance(target, DensityTarget):
        log_values, output_dims = _mapped_values(
            target.log_density, batch, key=key, kwargs=kwargs
        )
        if output_dims or log_values.ndim != 1:
            raise ValueError("Mapped log density must be scalar-valued per point.")
        if jnp.iscomplexobj(log_values):
            raise TypeError("Mapped log density must be real.")
        coefficient = coefficient * cx.Field(jnp.exp(log_values), dims=(batch.axis,))
        if target.normalized:
            denominator = coefficient
            for axis in reduced_axes:
                denominator = sum_over(denominator, axis)
            coefficient = coefficient / denominator
    return coefficient


def _normalize_discrete(
    target: DiscreteMeasureTarget,
    coefficient: cx.Field,
    /,
    *,
    reduced_axes: tuple[str, ...],
) -> cx.Field:
    if not target.normalized:
        return coefficient
    denominator = coefficient
    for axis in reduced_axes:
        denominator = sum_over(denominator, axis)
    return coefficient / denominator


def _base_coefficients(
    target: Any,
    batches: tuple[_FixedBatch, ...],
    /,
    *,
    reduced_axes: tuple[str, ...],
    key: Key[Array, ""],
    kwargs: dict[str, Any],
) -> tuple[cx.Field, ...]:
    if isinstance(_base_target(target), MappedTarget):
        if len(batches) != 1 or not isinstance(batches[0], MappedIntegrationBatch):
            raise TypeError("Mapped targets require exactly one mapped batch.")
        return (
            _mapped_coefficients(
                target,
                batches[0],
                reduced_axes=reduced_axes,
                key=key,
                kwargs=kwargs,
            ),
        )
    if isinstance(target, DiscreteMeasureTarget):
        if len(batches) != 1 or isinstance(batches[0], MappedIntegrationBatch):
            raise TypeError("Discrete targets require one point or separable batch.")
        return (
            _normalize_discrete(
                target, _batch_weight(batches[0]), reduced_axes=reduced_axes
            ),
        )
    if not isinstance(_base_target(target), (ComponentTarget, ProbabilityTarget)):
        raise TypeError(
            "Prepared linear reductions support component, probability, density, "
            "mapped, and declared discrete targets."
        )
    if any(isinstance(batch, MappedIntegrationBatch) for batch in batches):
        raise TypeError("Component and probability targets require named fixed batches.")
    batch_argument: Any = batches[0] if len(batches) == 1 else batches
    coefficients = _target_reduction_weights(
        target,
        batch_argument,
        key=key,
        kwargs=kwargs,
        reduction_axes=reduced_axes,
    )
    return coefficients if isinstance(coefficients, tuple) else (coefficients,)


def _coerce_extra_weight(value: Any, reference: cx.Field, /) -> cx.Field:
    if callable(value):
        raise TypeError(
            "weight must be fixed data; callable or integrand-dependent weights are "
            "not valid linear reductions."
        )
    if isinstance(value, cx.Field):
        field = value
        if any(dim is None for dim in field.dims):
            raise ValueError("Coefficient weights may contain only named dimensions.")
    else:
        data = jnp.asarray(value)
        if data.ndim == 0:
            field = cx.Field(data, dims=())
        elif data.shape == reference.shape:
            field = cx.Field(data, dims=reference.dims)
        else:
            raise ValueError(
                "Array weights must be scalar or have the complete coefficient shape; "
                "use coordax.Field for named broadcasting."
            )
    data = jnp.asarray(field.data)
    if jnp.iscomplexobj(data):
        raise TypeError("Linear reduction coefficients must be real.")
    if not bool(jnp.all(jnp.isfinite(data))):
        raise ValueError("Linear reduction weights must be finite.")
    return field


def _extra_weights(
    weight: Any | None,
    coefficients: tuple[cx.Field, ...],
    /,
) -> tuple[cx.Field, ...]:
    if weight is None:
        return tuple(cx.Field(jnp.asarray(1.0), dims=()) for _ in coefficients)
    if len(coefficients) == 1:
        return (_coerce_extra_weight(weight, coefficients[0]),)
    if not isinstance(weight, tuple) or len(weight) != len(coefficients):
        raise ValueError("Additive reductions require one fixed weight per batch.")
    return tuple(
        _coerce_extra_weight(term_weight, coefficient)
        for term_weight, coefficient in zip(weight, coefficients, strict=True)
    )


def _cast_coefficients(
    coefficients: tuple[cx.Field, ...],
    precision: IntegrationPrecisionPolicy,
    /,
) -> tuple[cx.Field, ...]:
    fields = tuple(
        cx.Field(precision.accumulation(coefficient.data), dims=coefficient.dims)
        for coefficient in coefficients
    )
    for field in fields:
        data = jnp.asarray(field.data)
        if jnp.iscomplexobj(data):
            raise TypeError("Linear reduction coefficients must be real.")
        if not bool(jnp.all(jnp.isfinite(data))):
            raise ValueError(
                "Prepared coefficients must be finite with nonzero normalization mass."
            )
    return fields


def _retained_layout(
    coefficients: tuple[cx.Field, ...],
    reduced_axes: tuple[str, ...],
    requested_retained: tuple[str, ...],
    /,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    requested = tuple(requested_retained)
    if len(frozenset(requested)) != len(requested):
        raise ValueError("retained_axes must be unique.")
    layouts: list[tuple[str, ...]] = []
    shapes: list[tuple[int, ...]] = []
    for coefficient in coefficients:
        if any(dim is None for dim in coefficient.dims):
            raise ValueError("Coefficient fields may contain only named dimensions.")
        missing = tuple(
            axis for axis in reduced_axes if axis not in coefficient.named_dims
        )
        if missing:
            raise ValueError(f"Coefficient fields are missing reduced axes {missing!r}.")
        retained = tuple(
            axis for axis in coefficient.named_dims if axis not in reduced_axes
        )
        layouts.append(retained)
        shapes.append(tuple(int(coefficient.named_shape[axis]) for axis in retained))
    if any(layout != layouts[0] for layout in layouts[1:]) or any(
        shape != shapes[0] for shape in shapes[1:]
    ):
        raise ValueError(
            "Additive coefficient fields must share one retained-axis layout and shape."
        )
    missing_requested = tuple(axis for axis in requested if axis not in layouts[0])
    if missing_requested:
        raise ValueError(
            f"Requested retained axes {missing_requested!r} are absent after reduction."
        )
    return layouts[0], shapes[0]


def _type_id(value: Any, /) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _target_id(target: Any, /) -> str:
    if isinstance(target, ProbabilityTarget):
        return target.target_id
    if isinstance(target, DensityTarget):
        return canonical_fingerprint(
            {
                "kind": "density-target",
                "base": _target_id(target.base),
                "normalized": target.normalized,
                "density_type": _type_id(target.log_density),
            }
        )
    if isinstance(target, ComponentTarget):
        component = target.component
        term_count = len(component.terms) if isinstance(component, ComponentSum) else 1
        return canonical_fingerprint(
            {
                "kind": "component-target",
                "labels": list(component.domain.labels),
                "axes": target.axes,
                "normalized": target.normalized,
                "component_type": _type_id(component),
                "term_count": term_count,
            }
        )
    if isinstance(target, DiscreteMeasureTarget):
        return canonical_fingerprint(
            {
                "kind": "discrete-target",
                "axes": list(target.axes),
                "normalized": target.normalized,
                "provenance": target.provenance,
            }
        )
    if isinstance(target, MappedTarget):
        return canonical_fingerprint(
            {
                "kind": "mapped-target",
                "rule_type": _type_id(target.reference_rule),
                "mapping_type": _type_id(target.mapping),
                "jacobian_type": _type_id(target.jacobian),
            }
        )
    raise TypeError(f"Unsupported linear-reduction target {type(target).__name__}.")


def _plan_id(plan: Any, /) -> str:
    if plan is None:
        return "none"
    return canonical_fingerprint(
        {
            "kind": "integration-plan",
            "type": f"{type(plan).__module__}.{type(plan).__qualname__}",
            "static": repr(plan),
            "arrays": array_tree_fingerprint(plan),
        }
    )


def _transformation_ids(realization: IntegrationRealization, /) -> tuple[str, ...]:
    return tuple(
        canonical_fingerprint(
            {
                "kind": record.kind,
                "source": record.source_provenance,
                "target": record.target_provenance,
                "diagnostics": repr(record.diagnostics),
                "arrays": array_tree_fingerprint(record.diagnostics),
            }
        )
        for record in realization.transformations
    )


def _source_provenance(batches: tuple[_FixedBatch, ...], /) -> str:
    return "|".join(batch.provenance for batch in batches)


def _exactness(realization: IntegrationRealization, target: Any, /) -> str:
    if realization.transformations:
        return "exact-linear-action-on-transformed-fixed-realization"
    if isinstance(target, DiscreteMeasureTarget):
        return "exact-linear-action-on-declared-discrete-measure"
    if isinstance(realization.plan, AdaptiveSparseGridPlan):
        return "exact-linear-action-on-integrand-selected-frozen-realization"
    if realization.plan is not None and _requires_random_key(realization.plan):
        return "exact-linear-action-on-frozen-randomized-realization"
    if realization.key is not None and realization.plan is None:
        return "exact-linear-action-on-frozen-sampled-realization"
    return "exact-linear-action-on-fixed-realization"


def _normalized(target: Any, /) -> bool:
    if isinstance(target, (ComponentTarget, ProbabilityTarget, DensityTarget)):
        return target.normalized
    if isinstance(target, DiscreteMeasureTarget):
        return target.normalized
    if isinstance(target, MappedTarget):
        return False
    raise TypeError(f"Unsupported linear-reduction target {type(target).__name__}.")


def _quantifier(
    target: Any,
    reduced_axes: tuple[str, ...],
    active_axes: tuple[str, ...],
    /,
) -> str:
    prefix = "fiber-" if reduced_axes != active_axes else ""
    return prefix + ("mean" if _normalized(target) else "integral")


def _batch_id(
    realization: IntegrationRealization,
    batches: tuple[_FixedBatch, ...],
    coefficients: tuple[cx.Field, ...],
    transformation_ids: tuple[str, ...],
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "linear-reduction-numerics",
            "arrays": array_tree_fingerprint(
                {
                    "batches": batches,
                    "coefficients": coefficients,
                    "target": realization.target,
                }
            ),
            "coefficient_layout": [list(field.dims) for field in coefficients],
            "coefficient_shape": [list(field.shape) for field in coefficients],
            "transformations": list(transformation_ids),
        }
    )


def _realization_id(
    *,
    target_id: str,
    plan_id: str,
    batch_id: str,
    reduced_axes: tuple[str, ...],
    retained_axes: tuple[str, ...],
    exactness: str,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "prepared-linear-reduction",
            "target": target_id,
            "plan": plan_id,
            "batch": batch_id,
            "reduced_axes": list(reduced_axes),
            "retained_axes": list(retained_axes),
            "exactness": exactness,
        }
    )


def _prepare_linear_reduction(
    realization: IntegrationRealization,
    /,
    *,
    retained_axes: tuple[str, ...],
    weight: Any | None,
    key: Key[Array, ""],
    kwargs: dict[str, Any],
    numeric_version: Any,
) -> PreparedLinearReduction:
    if not isinstance(realization, IntegrationRealization):
        raise TypeError("realization must be an IntegrationRealization.")
    batches = _fixed_batches(realization)
    active_axes = _active_axes(batches)
    requested_retained = tuple(retained_axes)
    retained_active = frozenset(requested_retained) & frozenset(active_axes)
    reduced_axes = tuple(axis for axis in active_axes if axis not in retained_active)
    if not reduced_axes:
        raise ValueError("A linear reduction must reduce at least one active axis.")
    base_coefficients = _base_coefficients(
        realization.target,
        batches,
        reduced_axes=reduced_axes,
        key=key,
        kwargs=kwargs,
    )
    weights = _extra_weights(weight, base_coefficients)
    coefficients = _cast_coefficients(
        tuple(
            coefficient * extra
            for coefficient, extra in zip(base_coefficients, weights, strict=True)
        ),
        realization.precision,
    )
    retained, retained_shape = _retained_layout(
        coefficients, reduced_axes, requested_retained
    )
    target_id = _target_id(realization.target)
    plan_id = _plan_id(realization.plan)
    transformation_ids = _transformation_ids(realization)
    exactness = _exactness(realization, realization.target)
    coefficient_dtype = str(
        jnp.result_type(*(jnp.asarray(field.data).dtype for field in coefficients))
    )
    batch_id = _batch_id(realization, batches, coefficients, transformation_ids)
    realization_id = _realization_id(
        target_id=target_id,
        plan_id=plan_id,
        batch_id=batch_id,
        reduced_axes=reduced_axes,
        retained_axes=retained,
        exactness=exactness,
    )
    version = jnp.asarray(numeric_version, dtype=jnp.int32)
    source_provenance = _source_provenance(batches)
    coefficient_layout = tuple(tuple(field.dims) for field in coefficients)
    schema = LinearReductionSchema(
        target_id=target_id,
        reduced_axes=reduced_axes,
        retained_axes=retained,
        retained_shape=retained_shape,
        coefficient_dtype=coefficient_dtype,
        coefficient_layout=coefficient_layout,
        quantifier=_quantifier(realization.target, reduced_axes, active_axes),
        exactness=exactness,
        source_provenance=source_provenance,
    )
    evidence = LinearReductionEvidence(
        numeric_version=version,
        finite=jnp.asarray(True),
        realization_id=realization_id,
        exactness=exactness,
        target_id=target_id,
        plan_id=plan_id,
        batch_id=batch_id,
        reduced_axes=reduced_axes,
        retained_axes=retained,
        coefficient_dtype=coefficient_dtype,
        coefficient_shape=tuple(
            tuple(int(size) for size in field.shape) for field in coefficients
        ),
        coefficient_layout=coefficient_layout,
        normalized=_normalized(realization.target),
        source_provenance=source_provenance,
        transformation_ids=transformation_ids,
    )
    return PreparedLinearReduction(
        target=realization.target,
        batches=batches,
        coefficient_fields=coefficients,
        extra_weight_fields=weights,
        precision=realization.precision,
        numeric_version=version,
        evidence=evidence,
        schema=schema,
        provenance=f"{exactness}:{source_provenance}",
        realization_id=realization_id,
    )


def prepare_linear_reduction(
    realization: IntegrationRealization,
    /,
    *,
    retained_axes: tuple[str, ...] = (),
    weight: Any | None = None,
    key: Key[Array, ""] = DOC_KEY0,
    **kwargs: Any,
) -> PreparedLinearReduction:
    """Freeze one materialized realization as an immutable coefficient action."""
    return _prepare_linear_reduction(
        realization,
        retained_axes=tuple(retained_axes),
        weight=weight,
        key=key,
        kwargs=kwargs,
        numeric_version=0,
    )


def _same_schema(
    previous: LinearReductionSchema,
    candidate: LinearReductionSchema,
    /,
) -> bool:
    return (
        previous.target_id == candidate.target_id
        and previous.reduced_axes == candidate.reduced_axes
        and previous.retained_axes == candidate.retained_axes
        and previous.retained_shape == candidate.retained_shape
        and previous.coefficient_dtype == candidate.coefficient_dtype
        and previous.coefficient_layout == candidate.coefficient_layout
        and previous.quantifier == candidate.quantifier
        and previous.output_suffix_policy == candidate.output_suffix_policy
    )


def refresh_linear_reduction(
    prepared: PreparedLinearReduction,
    realization: IntegrationRealization,
    /,
    *,
    retained_axes: tuple[str, ...] | None = None,
    weight: Any | None = None,
    key: Key[Array, ""] = DOC_KEY0,
    **kwargs: Any,
) -> PreparedLinearReduction:
    """Refresh a fixed action and increment its version iff its numerics changed."""
    if not isinstance(prepared, PreparedLinearReduction):
        raise TypeError("prepared must be a PreparedLinearReduction.")
    selected_axes = (
        prepared.schema.retained_axes if retained_axes is None else tuple(retained_axes)
    )
    selected_weight: Any = weight
    if weight is None and any(
        field.dims or bool(jnp.any(jnp.asarray(field.data) != 1.0))
        for field in prepared.extra_weight_fields
    ):
        selected_weight = prepared.extra_weight_fields
        if len(prepared.extra_weight_fields) == 1:
            selected_weight = prepared.extra_weight_fields[0]
    candidate = _prepare_linear_reduction(
        realization,
        retained_axes=selected_axes,
        weight=selected_weight,
        key=key,
        kwargs=kwargs,
        numeric_version=prepared.numeric_version,
    )
    if not _same_schema(prepared.schema, candidate.schema):
        raise ValueError(
            "Refreshed linear reduction changed its target, retained layout, or "
            "quantifier; prepare a new symbolic action instead."
        )
    changed = (
        candidate.evidence.batch_id != prepared.evidence.batch_id
        or candidate.evidence.transformation_ids != prepared.evidence.transformation_ids
        or candidate.evidence.coefficient_dtype != prepared.evidence.coefficient_dtype
    )
    version = prepared.numeric_version + jnp.asarray(int(changed), dtype=jnp.int32)
    return eqx.tree_at(
        lambda value: (value.numeric_version, value.evidence.numeric_version),
        candidate,
        (version, version),
    )


def _component_for_index(target: ComponentTarget, index: int, /) -> Any:
    component = target.component
    if isinstance(component, ComponentSum):
        return component.terms[index]
    if index != 0:
        raise IndexError("Single-component reductions contain exactly one batch.")
    return component


def _evaluate_named(
    function: Any,
    target: Any,
    batch: PointIntegrationBatch | SeparableIntegrationBatch,
    /,
    *,
    index: int,
    key: Key[Array, ""],
    kwargs: dict[str, Any],
) -> Any:
    base = _base_target(target)
    if isinstance(base, ComponentTarget):
        component = _component_for_index(base, index)
        return _as_domain_function(function, component)(batch.points, key=key, **kwargs)
    if isinstance(base, ProbabilityTarget):
        domain_function = (
            function
            if isinstance(function, DomainFunction)
            else DomainFunction(domain=base.probability, deps=(), func=function)
        )
        return domain_function(batch.points, key=key, **kwargs)
    if isinstance(target, DiscreteMeasureTarget):
        if isinstance(batch.points, PointBatch):
            if isinstance(function, DomainFunction):
                return function(batch.points, key=key, **kwargs)
            if isinstance(function, cx.Field) or not callable(function):
                return function
            raise TypeError(
                "External PointBatch callables must be DomainFunction instances."
            )
        if callable(function):
            return _ensure_special_kwonly_args(function)(
                batch.points,
                key=key,
                **kwargs,
            )
        return function
    raise TypeError(f"Unsupported named reduction target {type(target).__name__}.")


def _canonical_output(
    field: cx.Field,
    /,
    *,
    retained_axes: tuple[str, ...],
    retained_shape: tuple[int, ...],
) -> cx.Field:
    missing = tuple(axis for axis in retained_axes if axis not in field.named_dims)
    if missing:
        raise ValueError(f"Reduction output is missing retained axes {missing!r}.")
    unexpected_named = tuple(
        axis for axis in field.named_dims if axis not in retained_axes
    )
    if unexpected_named:
        raise ValueError(
            "Integrand event outputs must be a positional suffix; unexpected named "
            f"axes are {unexpected_named!r}."
        )
    retained_positions = tuple(field.dims.index(axis) for axis in retained_axes)
    suffix_positions = tuple(
        position for position, dim in enumerate(field.dims) if dim not in retained_axes
    )
    permutation = retained_positions + suffix_positions
    data = jnp.asarray(field.data)
    if permutation != tuple(range(data.ndim)):
        data = jnp.transpose(data, permutation)
    observed_shape = tuple(int(data.shape[index]) for index in range(len(retained_axes)))
    if observed_shape != retained_shape:
        raise ValueError(
            f"Reduction retained shape {observed_shape!r} does not match "
            f"prepared shape {retained_shape!r}."
        )
    return cx.Field(
        data,
        dims=retained_axes + (None,) * (data.ndim - len(retained_axes)),
    )


def _reduce_field(
    value: Any,
    coefficient: cx.Field,
    batch: PointIntegrationBatch | SeparableIntegrationBatch,
    /,
    *,
    reduced_axes: tuple[str, ...],
    retained_axes: tuple[str, ...],
    retained_shape: tuple[int, ...],
    precision: IntegrationPrecisionPolicy,
) -> cx.Field:
    reference = _batch_weight(batch)
    field = value if isinstance(value, cx.Field) else _as_weight_field(value, reference)
    field = cx.Field(precision.evaluation(field.data), dims=field.dims)
    weighted = coefficient * field
    for axis in reduced_axes:
        weighted = sum_over(
            weighted,
            axis,
            accumulation_dtype=precision.accumulation_dtype,
        )
    return _canonical_output(
        weighted,
        retained_axes=retained_axes,
        retained_shape=retained_shape,
    )


def _apply_batch(
    function: Any,
    target: Any,
    batch: _FixedBatch,
    coefficient: cx.Field,
    /,
    *,
    index: int,
    reduced_axes: tuple[str, ...],
    retained_axes: tuple[str, ...],
    retained_shape: tuple[int, ...],
    key: Key[Array, ""],
    kwargs: dict[str, Any],
    precision: IntegrationPrecisionPolicy,
) -> Any:
    if isinstance(batch, MappedIntegrationBatch):
        values, output_dims = _mapped_values(function, batch, key=key, kwargs=kwargs)
        weighted = coefficient * cx.Field(
            precision.evaluation(values), dims=(batch.axis,) + output_dims
        )
        for axis in reduced_axes:
            weighted = sum_over(
                weighted,
                axis,
                accumulation_dtype=precision.accumulation_dtype,
            )
        return _canonical_output(
            weighted,
            retained_axes=retained_axes,
            retained_shape=retained_shape,
        )
    evaluated = _evaluate_named(
        function,
        target,
        batch,
        index=index,
        key=key,
        kwargs=kwargs,
    )
    return jax.tree_util.tree_map(
        lambda value: _reduce_field(
            value,
            coefficient,
            batch,
            reduced_axes=reduced_axes,
            retained_axes=retained_axes,
            retained_shape=retained_shape,
            precision=precision,
        ),
        evaluated,
        is_leaf=lambda value: isinstance(value, cx.Field),
    )


def _output_precision(value: Any, precision: IntegrationPrecisionPolicy, /) -> Any:
    return jax.tree_util.tree_map(
        lambda field: cx.Field(precision.output(field.data), dims=field.dims),
        value,
        is_leaf=lambda item: isinstance(item, cx.Field),
    )


__all__ = [
    "LinearReductionEvidence",
    "LinearReductionSchema",
    "PreparedLinearReduction",
    "prepare_linear_reduction",
    "refresh_linear_reduction",
]
