#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import comb, isfinite
from typing import Any, Literal

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from phydrax.conditions._ir import (
    AbstractConditionOperator,
    OperatorCapabilities,
    OperatorLinearization,
)
from phydrax.domain import (
    BatchEvaluator,
    CallbackDerivativeRule,
    DomainFunction,
    GridBatch,
    PointBatch,
    TrajectoryDatasetDomain,
)

from .._doc import DOC_KEY0
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..domain._trajectory_interpolation import (
    _broadcast_like,
    _RaggedTimeSeriesTable,
)
from ..operators.differential._hooks import with_derivative_rule


RaggedTimeSeriesHardInterpolation = Literal["linear", "cubic_hermite"]
RaggedTimeSeriesHardGate = Literal["sin2", "sin4"]

class RaggedTimeSeriesObservationAction(AbstractConditionOperator):
    """Exact finite restriction to every valid row/time observation."""

    field: str = eqx.field(static=True)
    domain: TrajectoryDatasetDomain
    components: tuple[int, ...] | None = eqx.field(static=True)
    capabilities: OperatorCapabilities = eqx.field(static=True)
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        field: str,
        domain: TrajectoryDatasetDomain,
        /,
        *,
        components: Sequence[int] | None = None,
    ):
        field_ = str(field)
        if not field_:
            raise ValueError("Trajectory observation field name must be non-empty.")
        if not isinstance(domain, TrajectoryDatasetDomain):
            raise TypeError(
                "RaggedTimeSeriesObservationAction requires TrajectoryDatasetDomain."
            )
        components_ = _validate_components(components)
        action_id = canonical_fingerprint(
            {
                "kind": "ragged-time-series-observation-action-v1",
                "field": field_,
                "components": components_,
                "lengths": array_tree_fingerprint(domain.lengths),
                "case_indices": array_tree_fingerprint(domain.flat_case_indices),
                "time_indices": array_tree_fingerprint(domain.flat_time_indices),
                "start": array_tree_fingerprint(domain.start),
                "dt": array_tree_fingerprint(domain.dt),
            }
        )
        self.field = field_
        self.domain = domain
        self.components = components_
        self.capabilities = OperatorCapabilities(is_linear=True)
        self.action_id = action_id

    @property
    def observation_count(self) -> int:
        return self.domain.total_observations

    def observation_batch(self) -> PointBatch:
        cases = self.domain.flat_case_indices
        indices = self.domain.flat_time_indices
        times = self.domain.observation_times(cases, indices)
        return self.domain.points_from_case_time(
            cases,
            times,
            time_indices=indices,
        )

    def _apply(self, values: Mapping[str, Any], /, *, key=None, **kwargs: Any) -> Array:
        if self.field not in values:
            raise KeyError(f"Missing trajectory field {self.field!r}.")
        value = values[self.field]
        if not isinstance(value, DomainFunction):
            raise TypeError(
                "RaggedTimeSeriesObservationAction acts on DomainFunction values."
            )
        if not value.domain.same_support(self.domain):
            raise ValueError(
                "Observed trajectory field must share the declared trajectory support."
            )
        batch = self.observation_batch()
        evaluated = value(batch, key=key, **kwargs)
        if not isinstance(evaluated, cx.Field):
            raise TypeError("Trajectory observation evaluation must return a Field.")
        axes = batch.structure.axis_names
        if axes is None or len(axes) != 1:
            raise ValueError("Trajectory observations require one coupled sampling axis.")
        axis = axes[0]
        if axis not in evaluated.named_dims:
            raise ValueError(
                "Trajectory observation output is missing its sampling axis."
            )
        data = jnp.moveaxis(
            jnp.asarray(evaluated.data),
            evaluated.dims.index(axis),
            0,
        )
        if int(data.shape[0]) != self.observation_count:
            raise ValueError(
                f"Trajectory observation output has {data.shape[0]} rows; expected "
                f"{self.observation_count}."
            )
        if self.components is None:
            return data
        if data.ndim < 2:
            raise ValueError("Trajectory components require a trailing event axis.")
        width = int(data.shape[-1])
        if any(component >= width for component in self.components):
            raise ValueError(
                f"Trajectory component indices {self.components!r} exceed width {width}."
            )
        return jnp.take(data, jnp.asarray(self.components, dtype=jnp.int32), axis=-1)

    def apply(
        self,
        values: Mapping[str, Any],
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> Array:
        return self._apply(values, key=key, **kwargs)

    def linear_action(
        self,
        values: Mapping[str, Any],
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> Array:
        return self._apply(values, key=key, **kwargs)

    def adjoint_action(
        self,
        value: Any,
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        del value, key, kwargs
        raise TypeError(
            "Trajectory observation adjoints require a declared correction or "
            "coefficient representation."
        )

    def linearize(
        self,
        values: Mapping[str, Any],
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> OperatorLinearization:
        del values, key, kwargs
        raise TypeError("Globally linear trajectory restrictions do not linearize.")


class RaggedTimeSeriesCorrectionEvidence(StrictModule):
    """Exact finite restriction and truthful off-node interpolation evidence."""

    provider_id: str = eqx.field(static=True)
    action_id: str = eqx.field(static=True)
    observation_count: int = eqx.field(static=True)
    restriction_scope: str = eqx.field(static=True)
    interpolation_scope: str = eqx.field(static=True)
    interpolation: RaggedTimeSeriesHardInterpolation = eqx.field(static=True)
    gate: RaggedTimeSeriesHardGate = eqx.field(static=True)
    maximum_derivative_order: int = eqx.field(static=True)
    continuity_order: int = eqx.field(static=True)
    gate_zero_order: int = eqx.field(static=True)
    interpolation_exact_off_nodes: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        provider_id: str,
        action_id: str,
        observation_count: int,
        interpolation: RaggedTimeSeriesHardInterpolation,
        gate: RaggedTimeSeriesHardGate,
    ):
        self.provider_id = str(provider_id)
        self.action_id = str(action_id)
        self.observation_count = int(observation_count)
        self.restriction_scope = "exact_finite_trajectory_observations"
        self.interpolation_scope = "rowwise_piecewise_extension"
        self.interpolation = interpolation
        self.gate = gate
        self.maximum_derivative_order = 1 if interpolation == "linear" else 2
        self.continuity_order = 0 if interpolation == "linear" else 1
        self.gate_zero_order = 1 if gate == "sin2" else 3
        self.interpolation_exact_off_nodes = False


def _padded_residual_values(
    action: RaggedTimeSeriesObservationAction,
    residual: Array,
    /,
) -> Array:
    values = jnp.asarray(residual)
    count = action.observation_count
    if values.ndim == 0 or int(values.shape[0]) != count:
        raise ValueError(
            f"Trajectory residual must have leading size {count}, got {values.shape}."
        )
    padded = jnp.zeros(
        (action.domain.size, action.domain.max_length) + values.shape[1:],
        dtype=values.dtype,
    )
    return padded.at[
        action.domain.flat_case_indices,
        action.domain.flat_time_indices,
    ].set(values)


def _trajectory_field_dims(
    batch: PointBatch,
    value: Array,
    domain: TrajectoryDatasetDomain,
    /,
) -> tuple[Any, ...]:
    time_field = batch[domain.time_label]
    if not isinstance(time_field, cx.Field):
        raise TypeError("Trajectory time coordinates must be a coordax.Field.")
    trailing = value.ndim - jnp.asarray(time_field.data).ndim
    if trailing < 0:
        raise ValueError("Trajectory correction has fewer axes than its sample field.")
    return time_field.dims + (None,) * trailing


class _RaggedCardinalCorrectionEvaluator(StrictModule, BatchEvaluator):
    table: _RaggedTimeSeriesTable
    components: tuple[int, ...] | None = eqx.field(static=True)
    output_width: int | None = eqx.field(static=True)
    derivative_order: int = eqx.field(static=True)

    def __call_batch__(
        self,
        batch: PointBatch | GridBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        del key, kwargs
        if not isinstance(batch, PointBatch):
            raise TypeError(
                "Ragged cardinal corrections require PointBatch evaluation."
            )
        order = self.derivative_order
        targets, gates = self.table.evaluate(batch, max_order=order)
        correction = jnp.zeros_like(targets[order])
        for gate_order in range(order + 1):
            target_order = order - gate_order
            gate_factor = (
                1.0 - gates[0] if gate_order == 0 else -gates[gate_order]
            )
            gate_b = _broadcast_like(gate_factor, targets[target_order])
            correction = correction + float(comb(order, gate_order)) * (
                gate_b * targets[target_order]
            )
        if self.components is None:
            return cx.Field(
                correction,
                dims=_trajectory_field_dims(batch, correction, self.table.domain),
            )
        width = self.output_width
        if width is None:
            raise ValueError("Component corrections require output_width.")
        if correction.ndim == 1:
            correction = correction[:, None]
        out = jnp.zeros(correction.shape[:-1] + (width,), dtype=correction.dtype)
        out = out.at[..., jnp.asarray(self.components, dtype=jnp.int32)].set(correction)
        return cx.Field(
            out,
            dims=_trajectory_field_dims(batch, out, self.table.domain),
        )


class RaggedTimeSeriesCorrectionAction(StrictModule):
    """Cardinal lift from finite ragged observations to a trajectory field."""

    observation: RaggedTimeSeriesObservationAction
    interpolation: RaggedTimeSeriesHardInterpolation = eqx.field(static=True)
    gate: RaggedTimeSeriesHardGate = eqx.field(static=True)
    snap_tol: float = eqx.field(static=True)
    output_width: int | None = eqx.field(static=True)
    field_names: tuple[str, ...] = eqx.field(static=True)
    evidence: RaggedTimeSeriesCorrectionEvidence

    def __init__(
        self,
        observation: RaggedTimeSeriesObservationAction,
        interpolation: RaggedTimeSeriesHardInterpolation,
        gate: RaggedTimeSeriesHardGate,
        snap_tol: float,
        output_width: int | None,
        field_names: tuple[str, ...],
        evidence: RaggedTimeSeriesCorrectionEvidence,
        /,
    ):
        if not isinstance(observation, RaggedTimeSeriesObservationAction):
            raise TypeError(
                "RaggedTimeSeriesCorrectionAction requires a trajectory observation."
            )
        if interpolation not in ("linear", "cubic_hermite"):
            raise ValueError("interpolation must be 'linear' or 'cubic_hermite'.")
        if gate not in ("sin2", "sin4"):
            raise ValueError("gate must be 'sin2' or 'sin4'.")
        snap = float(snap_tol)
        if not isfinite(snap) or snap < 0.0:
            raise ValueError("snap_tol must be finite and non-negative.")
        width = None if output_width is None else int(output_width)
        if width is not None and width <= 0:
            raise ValueError("output_width must be positive when provided.")
        if observation.components is not None:
            if width is None:
                raise ValueError(
                    "Partial component correction requires output_width."
                )
            if any(component >= width for component in observation.components):
                raise ValueError(
                    f"Trajectory components {observation.components!r} exceed "
                    f"output width {width}."
                )
        names = tuple(str(name) for name in field_names)
        if names != (observation.field,):
            raise ValueError(
                "Trajectory correction field_names must match its observation."
            )
        if evidence.action_id != observation.action_id:
            raise ValueError("Trajectory correction evidence belongs to another action.")
        self.observation = observation
        self.interpolation = interpolation
        self.gate = gate
        self.snap_tol = snap
        self.output_width = width
        self.field_names = names
        self.evidence = evidence

    def _field(self, residual: Array, /, *, derivative_order: int) -> DomainFunction:
        padded = _padded_residual_values(self.observation, residual)
        table = _RaggedTimeSeriesTable(
            domain=self.observation.domain,
            values=padded,
            interpolation=self.interpolation,
            gate=self.gate,
            snap_tol=self.snap_tol,
        )
        evaluator = _RaggedCardinalCorrectionEvaluator(
            table,
            self.observation.components,
            self.output_width,
            derivative_order,
        )
        return DomainFunction(
            domain=self.observation.domain,
            deps=self.observation.domain.labels,
            func=evaluator,
            metadata={
                "ragged_cardinal_correction": True,
                "provider_id": self.evidence.provider_id,
                "exact_scope": self.evidence.restriction_scope,
                "interpolation_scope": self.evidence.interpolation_scope,
            },
        )

    def lift(self, product_residual: Any, /) -> tuple[DomainFunction, ...]:
        residual = product_residual
        if isinstance(residual, tuple):
            if len(residual) != 1:
                raise ValueError("Ragged trajectory lift expects one residual block.")
            residual = residual[0]
        residual_array = jnp.asarray(residual)
        if self.observation.components is not None and (
            residual_array.ndim < 2
            or int(residual_array.shape[-1]) != len(self.observation.components)
        ):
            raise ValueError(
                "Partial trajectory residuals must have a trailing axis with "
                f"{len(self.observation.components)} entries, got "
                f"{residual_array.shape}."
            )

        def _make(derivative_order: int, /) -> DomainFunction:
            base = self._field(
                residual_array,
                derivative_order=derivative_order,
            )

            def _hook(
                *,
                var: str,
                axis: int | None,
                order: int,
                mode,
                backend,
                basis,
                periodic: bool,
            ) -> DomainFunction | None:
                del mode, basis, periodic
                if backend not in ("ad", "jet"):
                    return None
                if var != self.observation.domain.time_label or axis is not None:
                    return None
                requested = derivative_order + int(order)
                if requested > self.evidence.maximum_derivative_order:
                    raise ValueError(
                        f"interpolation={self.interpolation!r} supports correction "
                        f"derivatives only up to order "
                        f"{self.evidence.maximum_derivative_order}."
                    )
                return _make(requested)

            return with_derivative_rule(base, CallbackDerivativeRule(_hook))

        return (_make(0),)

    __call__ = lift


class RaggedTimeSeriesCorrectionProvider(StrictModule):
    """Finite-cardinal row-wise trajectory correction provider."""

    action: RaggedTimeSeriesCorrectionAction

    def __init__(
        self,
        observation: RaggedTimeSeriesObservationAction,
        /,
        *,
        interpolation: RaggedTimeSeriesHardInterpolation = "linear",
        gate: RaggedTimeSeriesHardGate = "sin2",
        components_output_width: int | None = None,
        snap_tol: float = 1e-10,
    ):
        if not isinstance(observation, RaggedTimeSeriesObservationAction):
            raise TypeError(
                "RaggedTimeSeriesCorrectionProvider requires a trajectory observation."
            )
        if interpolation not in ("linear", "cubic_hermite"):
            raise ValueError("interpolation must be 'linear' or 'cubic_hermite'.")
        if gate not in ("sin2", "sin4"):
            raise ValueError("gate must be 'sin2' or 'sin4'.")
        snap = float(snap_tol)
        if not isfinite(snap) or snap < 0.0:
            raise ValueError("snap_tol must be finite and non-negative.")
        if observation.components is not None and components_output_width is None:
            raise ValueError(
                "Partial component correction requires components_output_width."
            )
        provider_id = canonical_fingerprint(
            {
                "kind": "ragged-time-series-correction-provider-v1",
                "action": observation.action_id,
                "interpolation": interpolation,
                "gate": gate,
                "snap_tol": snap,
                "output_width": components_output_width,
            }
        )
        evidence = RaggedTimeSeriesCorrectionEvidence(
            provider_id=provider_id,
            action_id=observation.action_id,
            observation_count=observation.observation_count,
            interpolation=interpolation,
            gate=gate,
        )
        self.action = RaggedTimeSeriesCorrectionAction(
            observation,
            interpolation,
            gate,
            snap,
            components_output_width,
            (observation.field,),
            evidence,
        )

    @property
    def provider_id(self) -> str:
        return self.action.evidence.provider_id

    @property
    def evidence(self) -> RaggedTimeSeriesCorrectionEvidence:
        return self.action.evidence

    def candidate_action(self) -> RaggedTimeSeriesCorrectionAction:
        return self.action


def _validate_components(components: Sequence[int] | None, /) -> tuple[int, ...] | None:
    if components is None:
        return None
    out = tuple(int(component) for component in components)
    if not out:
        raise ValueError("components must be non-empty when provided.")
    if any(component < 0 for component in out):
        raise ValueError("components must contain non-negative indices.")
    if len(set(out)) != len(out):
        raise ValueError("components must not contain duplicate indices.")
    return out


def _blend_components(
    free: Array,
    hard: Array,
    components: tuple[int, ...] | None,
    /,
) -> Array:
    free_arr = jnp.asarray(free, dtype=float)
    hard_arr = jnp.asarray(hard, dtype=float)
    if components is None:
        return hard_arr
    if free_arr.ndim < 2:
        raise ValueError("components require a vector-valued trailing output axis.")
    width = int(free_arr.shape[-1])
    for component in components:
        if component >= width:
            raise ValueError(
                f"component index {component} is out of bounds for output width {width}."
            )
    component_idx = jnp.asarray(components, dtype=jnp.int32)
    mask = jnp.zeros((width,), dtype=float).at[component_idx].set(1.0)
    mask = mask.reshape((1,) * (free_arr.ndim - 1) + (width,))
    return free_arr + mask * (hard_arr - free_arr)


class _RaggedTimeSeriesHardAnsatz(StrictModule, BatchEvaluator):
    u_free: DomainFunction
    table: _RaggedTimeSeriesTable
    components: tuple[int, ...] | None

    def __init__(
        self,
        *,
        u_free: DomainFunction,
        table: _RaggedTimeSeriesTable,
        components: tuple[int, ...] | None,
    ):
        self.u_free = u_free
        self.table = table
        self.components = components

    def __call__(self, *args: Any, key=None, **kwargs: Any) -> Array:
        del args, key, kwargs
        raise TypeError(
            "Ragged time-series hard enforcement requires PointBatch evaluation."
        )

    def __call_batch__(
        self,
        batch: PointBatch | GridBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        if not isinstance(batch, PointBatch):
            raise TypeError(
                "Ragged time-series hard enforcement requires PointBatch evaluation."
            )
        free = self.u_free(batch, key=key, **kwargs)
        targets, gates = self.table.evaluate(batch, max_order=0)
        free_arr = jnp.asarray(free.data, dtype=float)
        target = targets[0]
        gate_b = _broadcast_like(gates[0], free_arr)
        hard = target + gate_b * (free_arr - target)
        out = _blend_components(free_arr, hard, self.components)
        return cx.Field(out, dims=free.dims)


class _RaggedTimeSeriesHardAnsatzDerivative(StrictModule, BatchEvaluator):
    order: int
    u_free_derivatives: tuple[DomainFunction, ...]
    table: _RaggedTimeSeriesTable
    components: tuple[int, ...] | None

    def __init__(
        self,
        *,
        order: int,
        u_free_derivatives: tuple[DomainFunction, ...],
        table: _RaggedTimeSeriesTable,
        components: tuple[int, ...] | None,
    ):
        self.order = int(order)
        self.u_free_derivatives = tuple(u_free_derivatives)
        self.table = table
        self.components = components

    def __call__(self, *args: Any, key=None, **kwargs: Any) -> Array:
        del args, key, kwargs
        raise TypeError(
            "Ragged time-series hard derivative requires PointBatch evaluation."
        )

    def __call_batch__(
        self,
        batch: PointBatch | GridBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        if not isinstance(batch, PointBatch):
            raise TypeError(
                "Ragged time-series hard derivative requires PointBatch evaluation."
            )
        targets, gates = self.table.evaluate(batch, max_order=self.order)
        free_fields = tuple(
            fn(batch, key=key, **kwargs) for fn in self.u_free_derivatives
        )
        free_arrays = tuple(jnp.asarray(field.data, dtype=float) for field in free_fields)

        hard = targets[self.order]
        for gate_order in range(self.order + 1):
            free_order = self.order - gate_order
            delta = free_arrays[free_order] - targets[free_order]
            gate_b = _broadcast_like(gates[gate_order], delta)
            hard = hard + float(comb(self.order, gate_order)) * gate_b * delta

        out = _blend_components(free_arrays[self.order], hard, self.components)
        return cx.Field(out, dims=free_fields[self.order].dims)


def enforce_ragged_time_series(
    u_free: DomainFunction,
    domain: TrajectoryDatasetDomain,
    values: ArrayLike,
    /,
    *,
    interpolation: RaggedTimeSeriesHardInterpolation = "linear",
    gate: RaggedTimeSeriesHardGate = "sin2",
    time_var: str = "t",
    components: Sequence[int] | None = None,
    snap_tol: float = 1e-10,
) -> DomainFunction:
    """Return a hard ansatz that exactly matches row-wise ragged time-series data.

    The returned `DomainFunction` is evaluated on trajectory `PointBatch` objects
    that carry the internal row/time indices emitted by `TrajectoryDatasetDomain`.
    `interpolation="linear"` supports first-order time derivatives. Use
    `interpolation="cubic_hermite"` for second-order time derivatives, preferably
    with `gate="sin4"`.

    When `components` is provided, only those trailing output components are
    hard-enforced; the remaining components are passed through from `u_free`.
    """
    if not isinstance(domain, TrajectoryDatasetDomain):
        raise TypeError("enforce_ragged_time_series requires a TrajectoryDatasetDomain.")
    if time_var != domain.time_label:
        raise ValueError(
            f"time_var must match the trajectory time label {domain.time_label!r}."
        )
    if not u_free.domain.same_support(domain):
        raise ValueError("u_free must be defined on the provided trajectory domain.")

    components_ = _validate_components(components)
    table = _RaggedTimeSeriesTable(
        domain=domain,
        values=values,
        interpolation=interpolation,
        gate=gate,
        snap_tol=float(snap_tol),
    )

    base = DomainFunction(
        domain=u_free.domain,
        deps=u_free.deps,
        func=_RaggedTimeSeriesHardAnsatz(
            u_free=u_free,
            table=table,
            components=components_,
        ),
        metadata={},
    )

    def _make_hook(offset: int, /):
        def _hook(
            *,
            var: str,
            axis: int | None,
            order: int,
            mode,
            backend,
            basis,
            periodic: bool,
        ) -> DomainFunction | None:
            if backend not in ("ad", "jet"):
                return None
            if var != domain.time_label:
                return None
            if axis is not None:
                return None
            n = int(offset) + int(order)
            return _make_derivative(
                n,
                mode=mode,
                backend=backend,
                basis=basis,
                periodic=periodic,
            )

        return _hook

    def _make_derivative(
        order: int,
        /,
        *,
        mode,
        backend,
        basis,
        periodic: bool,
    ) -> DomainFunction:
        n = int(order)
        if n < 0:
            raise ValueError("order must be non-negative.")
        if n == 0:
            return with_derivative_rule(base, CallbackDerivativeRule(_make_hook(0)))
        limit = table.max_derivative_order()
        if n > limit:
            raise ValueError(
                f"interpolation={table.interpolation!r} supports hard time "
                f"derivatives only up to order {limit}."
            )

        from ..operators.differential._domain_ops import partial_n

        u_free_derivatives = tuple(
            partial_n(
                u_free,
                var=domain.time_label,
                axis=None,
                order=k,
                mode=mode,
                backend=backend,
                basis=basis,
                periodic=periodic,
            )
            for k in range(n + 1)
        )
        out = DomainFunction(
            domain=u_free.domain,
            deps=u_free.deps,
            func=_RaggedTimeSeriesHardAnsatzDerivative(
                order=n,
                u_free_derivatives=u_free_derivatives,
                table=table,
                components=components_,
            ),
            metadata={},
        )
        return with_derivative_rule(out, CallbackDerivativeRule(_make_hook(n)))

    return with_derivative_rule(base, CallbackDerivativeRule(_make_hook(0)))


__all__ = [
    "RaggedTimeSeriesCorrectionAction",
    "RaggedTimeSeriesCorrectionEvidence",
    "RaggedTimeSeriesCorrectionProvider",
    "RaggedTimeSeriesHardGate",
    "RaggedTimeSeriesHardInterpolation",
    "RaggedTimeSeriesObservationAction",
    "enforce_ragged_time_series",
]
