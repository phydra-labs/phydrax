"""Host-side geophysical bindings; execution remains in native operator owners."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._frozendict import frozendict
from ...nn.operator import (
    FunctionSamples,
    OperatorBatch,
    OperatorCaseProvenance,
    OperatorTargetBatch,
    OperatorTask,
)
from ...nn.operator.adapters import ExternalOperatorAdapter
from ...nn.operator.training import (
    autoregressive_operator_rollout_routes,
    fit_operator,
    operator_integral,
    OperatorDataset,
    OperatorDatasetSplit,
    OperatorFitResult,
    OperatorRolloutControlRoute,
    OperatorRolloutRoute,
    OperatorSplitPolicy,
    project_operator_conservation,
    split_operator_dataset,
    TrainedOperator,
)
from ._quantities import GeophysicalQuantity
from ._time import GeophysicalTimeSpec


def _interval_contract(task: OperatorTask, kind: str, seconds: float) -> None:
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("The trained interval must be finite and positive.")
    if task.metadata.get("geophysical_target_kind") != kind:
        raise ValueError(
            f"Task must explicitly declare geophysical_target_kind={kind!r}."
        )
    if task.metadata.get("geophysical_step_seconds") != seconds:
        raise ValueError("Task and deployment intervals must agree exactly.")


def _bounds(values: Any, count: int, seconds: float) -> np.ndarray:
    bounds = np.asarray(values, dtype=np.float64)
    if bounds.shape != (count, 2) or not np.all(np.isfinite(bounds)):
        raise ValueError("Interval bounds must be finite (case, 2) seconds.")
    if not np.all(bounds[:, 1] - bounds[:, 0] == seconds):
        raise ValueError("Every target interval must equal the trained interval exactly.")
    return bounds


@dataclass(frozen=True)
class ColumnClosureBinding:
    """Ordered finite-interval increments, never an instantaneous ODE RHS.

    Each variable is a separate scalar task field on the same ordered column.
    Values are expressed in the declared quantity units; the caller must prepare
    compatible task dimensions/scales and physical layer masses in kg/m².
    """

    task: OperatorTask
    quantities: tuple[GeophysicalQuantity, ...]
    state_fields: tuple[str, ...]
    target_fields: tuple[str, ...]
    level_ids: tuple[str, ...]
    vertical_id: str
    interval_seconds: float

    def __post_init__(self):
        object.__setattr__(self, "quantities", tuple(self.quantities))
        object.__setattr__(self, "state_fields", tuple(self.state_fields))
        object.__setattr__(self, "target_fields", tuple(self.target_fields))
        object.__setattr__(self, "level_ids", tuple(self.level_ids))
        _interval_contract(self.task, "interval_increment", self.interval_seconds)
        size = len(self.quantities)
        if not size or len(self.state_fields) != size or len(self.target_fields) != size:
            raise ValueError("Ordered quantities, source fields and targets must align.")
        if len({q.name for q in self.quantities}) != size:
            raise ValueError("Column quantity names must be unique.")
        if (
            not self.vertical_id
            or not self.level_ids
            or len(set(self.level_ids)) != len(self.level_ids)
        ):
            raise ValueError(
                "Column levels require a vertical identity and unique ordered IDs."
            )
        for quantity, source_name, target_name in zip(
            self.quantities, self.state_fields, self.target_fields, strict=True
        ):
            source = self.task.field_by_name[source_name]
            target = self.task.field_by_name[target_name]
            if not source.is_source or not target.is_target:
                raise ValueError("Column bindings require source and target task fields.")
            if source.channels != "scalar" or target.channels != "scalar":
                raise ValueError(
                    "Bind each physical column variable as one scalar field."
                )
            if (
                source.dimension != quantity.unit.dimension
                or target.dimension != quantity.unit.dimension
            ):
                raise ValueError("Column quantity and native task dimensions disagree.")
        if len(set(self.state_fields)) != size or len(set(self.target_fields)) != size:
            raise ValueError("Column field bindings must be one-to-one.")

    @property
    def binding_id(self) -> str:
        return canonical_fingerprint(
            {
                "task": self.task.fingerprint,
                "quantities": [q.quantity_id for q in self.quantities],
                "sources": self.state_fields,
                "targets": self.target_fields,
                "levels": self.level_ids,
                "vertical": self.vertical_id,
                "interval_seconds": self.interval_seconds,
            }
        )


def column_closure_dataset(
    binding: ColumnClosureBinding,
    before: Any,
    after: Any,
    resolved_increment: Any,
    /,
    *,
    layer_mass: Any,
    interval_bounds: Any,
    forcing: Mapping[str, FunctionSamples],
    forcing_bounds: Mapping[str, Any],
    provenance: Sequence[OperatorCaseProvenance],
) -> OperatorDataset:
    """Build closure labels ``after - before - resolved_increment``.

    State arrays have (case, level, variable) order. Bounds are physical seconds,
    and every forcing source must cover exactly the target's finite interval.
    The resolved increment must use that same interval, not an instantaneous rate.
    """
    initial, final, resolved = map(np.asarray, (before, after, resolved_increment))
    if initial.ndim != 3 or initial.shape[1:] != (
        len(binding.level_ids),
        len(binding.quantities),
    ):
        raise ValueError("Column arrays require (case, ordered level, ordered variable).")
    if final.shape != initial.shape or resolved.shape != initial.shape:
        raise ValueError(
            "Initial/final states and resolved increments must have identical shapes."
        )
    if not all(np.all(np.isfinite(value)) for value in (initial, final, resolved)):
        raise ValueError("Column training states and increments must be finite.")
    count, levels, _ = initial.shape
    bounds = _bounds(interval_bounds, count, binding.interval_seconds)
    mass = np.asarray(layer_mass)
    if mass.shape != (count, levels) or not np.all(np.isfinite(mass) & (mass > 0)):
        raise ValueError("Layer masses must be positive finite (case, level) kg/m².")
    records = tuple(provenance)
    if len(records) != count or any(
        not {"scenario", "model", "member"}.issubset(record.identities)
        for record in records
    ):
        raise ValueError("Each column case requires scenario/model/member provenance.")
    if set(forcing) != set(forcing_bounds):
        raise ValueError("Every forcing source requires exact interval bounds.")
    for name in forcing:
        aligned = _bounds(forcing_bounds[name], count, binding.interval_seconds)
        if not np.array_equal(aligned, bounds):
            raise ValueError(f"Forcing {name!r} does not align with the target interval.")
    coordinates = jnp.arange(levels, dtype=float)[:, None]
    support_id = canonical_fingerprint(
        {"vertical": binding.vertical_id, "levels": binding.level_ids}
    )
    query = FunctionSamples(
        values=None,
        coordinates=coordinates,
        quadrature_weights=jnp.asarray(mass),
        support_id=support_id,
    )
    inputs = dict(forcing)
    queries = {}
    targets = {}
    specs = {}
    query_names = {}
    for index, (source_name, target_name) in enumerate(
        zip(binding.state_fields, binding.target_fields, strict=True)
    ):
        source = binding.task.field_by_name[source_name]
        target = binding.task.field_by_name[target_name]
        if source.source_name in inputs:
            raise ValueError("Forcing may not overwrite the column state.")
        inputs[source.source_name] = FunctionSamples(
            values=jnp.asarray(initial[..., index]),
            coordinates=coordinates,
            quadrature_weights=jnp.asarray(mass),
            support_id=support_id,
        )
        queries[target.query_name] = query
        targets[target_name] = jnp.asarray(
            final[..., index] - initial[..., index] - resolved[..., index]
        )
        specs[target_name] = target.output_spec
        query_names[target_name] = target.query_name
    batch = OperatorBatch(
        inputs=inputs, queries=queries, case_axes=("case",), case_shape=(count,)
    )
    binding.task.validate_batch(batch)
    return OperatorDataset(
        batch,
        OperatorTargetBatch.from_arrays(
            targets, batch, specs=specs, query_names=query_names
        ),
        provenance=records,
    )


def _simulation_identity(record: OperatorCaseProvenance) -> str:
    keys = ("scenario", "model", "member")
    if any(key not in record.identities for key in keys):
        raise ValueError(
            "Geophysical experiments require scenario/model/member identities."
        )
    return canonical_fingerprint({key: record.identities[key] for key in keys})


@dataclass(frozen=True)
class GeophysicalLearningExperiment:
    """A native task and leakage-safe partitions, without a second training engine."""

    task: OperatorTask
    split: OperatorDatasetSplit
    temporal_bounds: tuple[str, str] | None = None

    def __post_init__(self):
        partitions = (self.split.train, self.split.validation, self.split.test)
        policy = self.split.policy
        records = tuple(record for part in partitions for record in part.provenance)
        selected_keys = (
            tuple(sorted({key for record in records for key in record.identities}))
            if policy.group_by == "all"
            else policy.group_by
        )
        if self.split.group_keys != selected_keys:
            raise ValueError(
                "Split group keys must match the declared native evaluation policy."
            )
        if not selected_keys and policy.order_by is None:
            raise ValueError(
                "Evaluation must declare a grouped or chronological split policy."
            )
        bounds = None if self.temporal_bounds is None else tuple(self.temporal_bounds)
        if bounds is not None and (
            len(bounds) != 2
            or not all(bounds)
            or bounds[0] == bounds[1]
            or policy.order_by is None
        ):
            raise ValueError(
                "Temporal bounds require two distinct order coordinates and a chronological policy."
            )
        object.__setattr__(self, "temporal_bounds", bounds)
        seen_cases = set()
        seen_groups = {key: set() for key in selected_keys}
        previous_order_end = None
        previous_interval_end = None
        for partition in partitions:
            self.task.validate_batch(partition.batch)
            cases = {record.case_id for record in partition.provenance}
            if cases & seen_cases:
                raise ValueError(
                    "Training and evaluation partitions share case identities."
                )
            seen_cases.update(cases)
            for record in partition.provenance:
                identity = _simulation_identity(record)
                if (
                    "geophysical_simulation" in record.identities
                    and record.identities["geophysical_simulation"] != identity
                ):
                    raise ValueError(
                        "Composite simulation identity disagrees with its scenario/model/member."
                    )
            for key, seen in seen_groups.items():
                if policy.group_by != "all" and any(
                    key not in record.identities for record in partition.provenance
                ):
                    raise ValueError(
                        f"Every partition requires the selected {key!r} group identity."
                    )
                groups = {
                    record.identities[key]
                    for record in partition.provenance
                    if key in record.identities
                }
                if groups & seen:
                    raise ValueError(
                        f"Training and evaluation partitions share selected {key!r} groups."
                    )
                seen.update(groups)
            if policy.order_by is not None:
                if any(
                    policy.order_by not in record.order for record in partition.provenance
                ):
                    raise ValueError(
                        "Every chronological case requires the selected order coordinate."
                    )
                positions = [
                    record.order[policy.order_by] for record in partition.provenance
                ]
                if previous_order_end is not None and min(positions) < previous_order_end:
                    raise ValueError("Chronological partition order coordinates overlap.")
                previous_order_end = max(positions)
            if bounds is not None:
                if any(
                    not set(bounds).issubset(record.order)
                    for record in partition.provenance
                ):
                    raise ValueError(
                        "Every chronological window requires its declared temporal bounds."
                    )
                starts = [record.order[bounds[0]] for record in partition.provenance]
                ends = [record.order[bounds[1]] for record in partition.provenance]
                if any(end < start for start, end in zip(starts, ends, strict=True)):
                    raise ValueError(
                        "Temporal window ends must not precede their starts."
                    )
                if (
                    previous_interval_end is not None
                    and min(starts) < previous_interval_end
                ):
                    raise ValueError("Training and evaluation temporal windows overlap.")
                previous_interval_end = max(ends)

    @classmethod
    def prepare(
        cls,
        task: OperatorTask,
        dataset: OperatorDataset,
        /,
        *,
        policy: OperatorSplitPolicy | None = None,
        temporal_bounds: tuple[str, str] | None = None,
        train_fraction: float = 0.6,
        validation_fraction: float = 0.2,
    ):
        """Default to joint-simulation holdout; explicit policies define other questions.

        A selected native key protects that key independently. The default is
        one composite key, not three independently held-out physical axes.
        Chronological windows can instead select their own group key and declare
        start/end order coordinates so overlapping target support is rejected.
        """
        records = []
        for record in dataset.provenance:
            simulation = _simulation_identity(record)
            if (
                "geophysical_simulation" in record.identities
                and record.identities["geophysical_simulation"] != simulation
            ):
                raise ValueError(
                    "Composite simulation identity disagrees with its scenario/model/member."
                )
            records.append(
                OperatorCaseProvenance(
                    record.case_id,
                    identities={
                        **record.identities,
                        "geophysical_simulation": simulation,
                    },
                    order=record.order,
                )
            )
        prepared = replace(dataset, provenance=tuple(records))
        resolved = (
            OperatorSplitPolicy(group_by=("geophysical_simulation",))
            if policy is None
            else policy
        )
        task.validate_batch(prepared.batch)
        return cls(
            task,
            split_operator_dataset(
                prepared,
                policy=resolved,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
            ),
            temporal_bounds=temporal_bounds,
        )

    def fit(
        self,
        model,
        /,
        *,
        steps: int,
        learning_rate: float = 1e-3,
        batch_size: int | None = None,
        seed: int = 0,
        output_field_map: Mapping[str, str] | None = None,
        artifact_id: str = "",
        jit: bool = True,
    ) -> OperatorFitResult:
        """Run the requested updates; fit normalization on train, never validation."""
        if steps <= 0:
            raise ValueError(
                "A learned geophysical operator requires positive training steps."
            )
        train = self.split.train
        if not np.all(np.asarray(train.case_mask)) or not np.all(
            np.asarray(train.case_log_weights) == 0
        ):
            raise ValueError(
                "This experiment requires active equally weighted cases; quadrature weights remain physical."
            )
        return fit_operator(
            model,
            train,
            validation=self.split.validation,
            task=self.task,
            epochs=steps,
            steps=steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            seed=seed,
            normalization="fit",
            normalization_weighting="quadrature",
            output_field_map=output_field_map,
            artifact_id=artifact_id,
            jit=jit,
            provenance={
                "training_case_ids": [p.case_id for p in train.provenance],
                "group_keys": self.split.group_keys,
                "order_by": self.split.policy.order_by,
                "temporal_bounds": self.temporal_bounds,
            },
        )


@dataclass(frozen=True)
class ColumnClosureAdmission:
    increment: Any
    proposed_state: Any
    raw_budget: Any
    corrected_budget: Any
    budget_correction: Any
    local_correction: Any
    admitted: bool
    quantity_ids: tuple[str, ...]
    budget_units: tuple[str, ...]

    def require_state(self):
        if not self.admitted:
            raise ValueError(
                "Learned closure failed physical domain admission; no state may be committed."
            )
        return self.proposed_state


def admit_column_closure(
    binding: ColumnClosureBinding,
    before: Any,
    increment: Any,
    /,
    *,
    layer_mass: Any,
    target_budget: Any,
    domain_admission: Callable[[Any], Any],
) -> ColumnClosureAdmission:
    """Project each variable's mass integral, then run the real domain's admission.

    Prescribed budgets have quantity-unit kg/m² (heat capacity etc. must be
    included explicitly by the caller when converting energy to a temperature
    integral). No positivity clipping or implicit conservation assumption occurs.
    """
    state, raw = jnp.asarray(before), jnp.asarray(increment)
    expected = (len(binding.level_ids), len(binding.quantities))
    if state.ndim < 2 or state.shape[-2:] != expected or raw.shape != state.shape:
        raise ValueError(
            "Closure state and increment must preserve level/variable order."
        )
    case_shape = state.shape[:-2]
    mass = jnp.asarray(layer_mass)
    budget = jnp.asarray(target_budget)
    if mass.shape != state.shape[:-1] or budget.shape != case_shape + (expected[1],):
        raise ValueError("Masses and prescribed budgets must align with column cases.")
    if not callable(domain_admission):
        raise TypeError("Deployment requires the native domain admission callable.")
    if not all(
        np.all(np.isfinite(np.asarray(x))) for x in (state, raw, mass, budget)
    ) or not np.all(np.asarray(mass) > 0):
        raise ValueError(
            "Closure data and budgets must be finite with positive layer masses."
        )
    query = FunctionSamples(
        values=None,
        coordinates=jnp.arange(expected[0], dtype=float)[:, None],
        quadrature_weights=mass,
    )
    corrected = project_operator_conservation(raw, query, budget, case_shape=case_shape)
    raw_budget = operator_integral(raw, query, case_shape=case_shape)
    corrected_budget = operator_integral(corrected, query, case_shape=case_shape)
    proposed = state + corrected
    admitted = bool(np.all(np.asarray(domain_admission(proposed)))) and bool(
        np.all(np.isfinite(np.asarray(proposed)))
    )
    budget_scale = jnp.sum(jnp.abs(corrected) * mass[..., None], axis=-2) + jnp.abs(
        budget
    )
    tolerance = 32 * jnp.finfo(corrected.dtype).eps * jnp.maximum(budget_scale, 1.0)
    admitted = admitted and bool(
        np.all(np.asarray(jnp.abs(corrected_budget - budget) <= tolerance))
    )
    return ColumnClosureAdmission(
        corrected,
        proposed,
        raw_budget,
        corrected_budget,
        corrected_budget - raw_budget,
        corrected - raw,
        admitted,
        tuple(q.quantity_id for q in binding.quantities),
        tuple(f"({q.unit.symbol}) kg m^-2" for q in binding.quantities),
    )


def deploy_column_closure(
    trained: TrainedOperator,
    binding: ColumnClosureBinding,
    batch: OperatorBatch,
    /,
    *,
    layer_mass: Any,
    target_budget: Any,
    resolved_increment: Any,
    domain_admission: Callable[[Any], Any],
    key=None,
) -> ColumnClosureAdmission:
    """Consume a trained native operator, convert its interval labels, and admit."""
    if (
        not isinstance(trained, TrainedOperator)
        or trained.task_fingerprint != binding.task.fingerprint
    ):
        raise ValueError("Closure deployment requires the trained binding's native task.")
    if (
        isinstance(trained.execution_model, ExternalOperatorAdapter)
        or "external_manifest" in trained.provenance
    ):
        raise TypeError(
            "External model execution is outside the column deployment schema."
        )
    prediction = trained.predict(batch, key=key)
    before = jnp.stack(
        [
            batch.input(binding.task.field_by_name[name].source_name).values
            for name in binding.state_fields
        ],
        axis=-1,
    )
    delta = jnp.stack(
        [prediction.field(name).values for name in binding.target_fields], axis=-1
    )
    resolved = jnp.asarray(resolved_increment)
    if resolved.shape != before.shape:
        raise ValueError(
            "Resolved deployment increment must match the complete column state."
        )
    return admit_column_closure(
        binding,
        before + resolved,
        delta,
        layer_mass=layer_mass,
        target_budget=target_budget,
        domain_admission=domain_admission,
    )


@dataclass(frozen=True)
class GeophysicalForcingSchedule:
    """Prepared interval controls in the forecast clock's numeric time units."""

    source_name: str
    time_id: str
    bounds: tuple[tuple[float, float], ...]
    samples: tuple[FunctionSamples, ...]

    def __post_init__(self):
        object.__setattr__(
            self, "bounds", tuple(tuple(float(v) for v in pair) for pair in self.bounds)
        )
        object.__setattr__(self, "samples", tuple(self.samples))
        if (
            not self.source_name
            or not self.time_id
            or not self.samples
            or len(self.bounds) != len(self.samples)
        ):
            raise ValueError(
                "Forcing schedules require named, time-bound interval samples."
            )
        bounds = np.asarray(self.bounds)
        if (
            bounds.shape != (len(self.samples), 2)
            or not np.all(np.isfinite(bounds))
            or not np.all(bounds[:, 1] > bounds[:, 0])
        ):
            raise ValueError("Forcing interval bounds must be finite and ordered.")
        if any(sample.values is None for sample in self.samples):
            raise ValueError(
                "Forcing schedule entries must contain physical source values."
            )

    @property
    def schedule_id(self):
        return canonical_fingerprint(
            {
                "source": self.source_name,
                "time": self.time_id,
                "bounds": self.bounds,
                "samples": array_tree_fingerprint(self.samples),
                "supports": [sample.support_id for sample in self.samples],
            }
        )

    def route(self) -> OperatorRolloutControlRoute:
        # Native deployment invokes policies on the host with a concrete step.
        def policy(batch, step, key):
            del batch, key
            return self.samples[int(step)]

        return OperatorRolloutControlRoute(self.source_name, policy, self.schedule_id)


@dataclass(frozen=True)
class GeophysicalForecastRequest:
    time: GeophysicalTimeSpec
    initialization: str
    steps: int
    scenario_id: str
    model_id: str
    member_ids: tuple[str, ...]
    experiment_id: str

    def __post_init__(self):
        object.__setattr__(self, "member_ids", tuple(self.member_ids))
        if not isinstance(self.steps, int) or self.steps <= 0:
            raise ValueError("Forecast requests require a positive integer horizon.")
        if (
            not all((self.scenario_id, self.model_id, self.experiment_id))
            or not self.member_ids
            or len(set(self.member_ids)) != len(self.member_ids)
            or not all(self.member_ids)
        ):
            raise ValueError(
                "Forecasts require scenario/model/experiment identities and unique members."
            )
        self.time.encode((self.initialization,))

    @property
    def request_id(self):
        return canonical_fingerprint(
            {
                "clock": self.time.time_id,
                "initialization": float(self.time.encode((self.initialization,))[0]),
                "steps": self.steps,
                "scenario": self.scenario_id,
                "model": self.model_id,
                "members": self.member_ids,
                "experiment": self.experiment_id,
            }
        )


@dataclass(frozen=True)
class GeophysicalForecastContinuation:
    request: GeophysicalForecastRequest
    batch: OperatorBatch
    next_step: int
    key: Any
    adapter_id: str


@dataclass(frozen=True)
class GeophysicalForecastProduct:
    request: GeophysicalForecastRequest
    predictions: tuple[Any, ...]
    lead_seconds: tuple[float, ...]
    valid_times: tuple[str, ...]
    continuation: GeophysicalForecastContinuation
    quantities: Mapping[str, GeophysicalQuantity]

    def field(self, name: str):
        """Return (lead, member, native spatial axes..., native channels...)."""
        return jnp.stack(
            [prediction.field(name).values for prediction in self.predictions]
        )


class NativeGeophysicalForecast:
    """Host request adapter for complete native task-state recurrence.

    All both-role fields must recur; every other source must have interval
    controls. Any recurrent history/memory belongs in those native task fields,
    not in hidden adapter state. External model wrappers are not accepted.
    """

    def __init__(
        self,
        trained: TrainedOperator,
        /,
        *,
        step_seconds: float,
        quantities: Mapping[str, GeophysicalQuantity],
        routes: Sequence[OperatorRolloutRoute],
        forcing: Sequence[GeophysicalForcingSchedule] = (),
    ):
        if not isinstance(trained, TrainedOperator):
            raise TypeError("Forecast execution accepts only a native TrainedOperator.")
        if (
            isinstance(trained.execution_model, ExternalOperatorAdapter)
            or "external_manifest" in trained.provenance
        ):
            raise TypeError(
                "External model execution is outside the native forecast schema."
            )
        _interval_contract(trained.task, "next_state", step_seconds)
        if not trained.contract.capabilities.autoregressive_rollout:
            raise ValueError(
                "This native architecture does not support autoregressive rollout."
            )
        self.trained, self.step_seconds = trained, float(step_seconds)
        self.quantities = frozendict(quantities)
        self.routes, self.forcing = tuple(routes), tuple(forcing)
        states = {
            field.name
            for field in trained.task.fields
            if field.is_source and field.is_target
        }
        if (
            not states
            or {route.task_field for route in self.routes} != states
            or len(self.routes) != len(states)
        ):
            raise ValueError(
                "Every native recurrent state field must be routed exactly once."
            )
        targets = {field.name for field in trained.task.fields if field.is_target}
        if set(self.quantities) != targets:
            raise ValueError("Every forecast output requires a physical quantity.")
        for name, quantity in self.quantities.items():
            if quantity.unit.dimension != trained.task.field_by_name[name].dimension:
                raise ValueError("Forecast quantity and native task dimensions disagree.")
        control_names = {
            field.source_name
            for field in trained.task.fields
            if field.is_source and not field.is_target
        }
        if {control.source_name for control in self.forcing} != control_names or len(
            self.forcing
        ) != len(control_names):
            raise ValueError(
                "Every nonrecurrent source requires an explicit aligned forcing schedule."
            )
        self.adapter_id = canonical_fingerprint(
            {
                "task": trained.task_fingerprint,
                "contract": trained.contract_fingerprint,
                "normalization": trained.normalization_fingerprint,
                "weights": array_tree_fingerprint(trained.execution_model),
                "output_map": dict(trained.output_field_map),
                "output_pipeline": None
                if trained.output_pipeline is None
                else trained.output_pipeline.fingerprint,
                "dtype": trained.dtype_policy.to_dict(),
                "fixed_queries": dict(trained.fixed_query_fingerprints),
                "step_seconds": self.step_seconds,
                "routes": [
                    (r.source_name, r.prediction_name, r.task_field, r.transfer_id)
                    for r in self.routes
                ],
                "forcing": [f.schedule_id for f in self.forcing],
                "quantities": {
                    name: q.quantity_id for name, q in self.quantities.items()
                },
            }
        )

    def start(
        self,
        request: GeophysicalForecastRequest,
        batch: OperatorBatch,
        /,
        *,
        steps: int | None = None,
        key=None,
    ) -> GeophysicalForecastProduct:
        if request.steps > self.trained.task.problem.rollout_steps:
            raise ValueError(
                "Forecast horizon exceeds the trained native rollout contract."
            )
        if batch.case_axes != ("member",) or batch.case_shape != (
            len(request.member_ids),
        ):
            raise ValueError(
                "Forecast batch must have the request's explicit member case axis."
            )
        initial = float(request.time.encode((request.initialization,))[0])
        dt = self.step_seconds / request.time.seconds_per_unit
        expected = initial + np.arange(request.steps + 1, dtype=np.float64) * dt
        for schedule in self.forcing:
            if (
                schedule.time_id != request.time.time_id
                or len(schedule.samples) != request.steps
            ):
                raise ValueError("Forcing clock/horizon must equal the forecast request.")
            if not np.array_equal(
                np.asarray(schedule.bounds),
                np.stack((expected[:-1], expected[1:]), axis=-1),
            ):
                raise ValueError(
                    "Forcing intervals do not align exactly with forecast valid times."
                )
        continuation = GeophysicalForecastContinuation(
            request, batch, 0, key, self.adapter_id
        )
        return self.resume(continuation, steps=request.steps if steps is None else steps)

    def resume(
        self, continuation: GeophysicalForecastContinuation, /, *, steps: int
    ) -> GeophysicalForecastProduct:
        if continuation.adapter_id != self.adapter_id:
            raise ValueError(
                "Continuation belongs to different weights, routes, normalization or forcing."
            )
        if continuation.request.steps > self.trained.task.problem.rollout_steps:
            raise ValueError(
                "Continuation horizon exceeds the trained native rollout contract."
            )
        if (
            not isinstance(steps, int)
            or steps <= 0
            or continuation.next_step < 0
            or continuation.next_step + steps > continuation.request.steps
        ):
            raise ValueError(
                "Continuation must stay within its original positive forecast horizon."
            )
        rollout = autoregressive_operator_rollout_routes(
            self.trained,
            continuation.batch,
            steps,
            self.routes,
            control_routes=tuple(schedule.route() for schedule in self.forcing),
            key=continuation.key,
            step_offset=continuation.next_step,
        )
        leads = tuple(
            self.step_seconds * index
            for index in range(continuation.next_step + 1, rollout.next_step + 1)
        )
        request = continuation.request
        initial = float(request.time.encode((request.initialization,))[0])
        valid = request.time.decode(
            initial + np.asarray(leads) / request.time.seconds_per_unit
        )
        next_state = GeophysicalForecastContinuation(
            request,
            rollout.final_batch,
            rollout.next_step,
            continuation.key,
            self.adapter_id,
        )
        return GeophysicalForecastProduct(
            request, rollout.predictions, leads, valid, next_state, self.quantities
        )
