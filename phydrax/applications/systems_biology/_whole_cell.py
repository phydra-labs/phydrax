#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity multirate whole-cell assembly with atomic state commits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._network import PreparedStoichiometricNetwork, StoichiometricRuntime


class WholeCellStatus(IntEnum):
    """Fail-closed whole-cell candidate and commit status."""

    SUCCESS = 0
    INVALID_STATE = 1
    PROCESS_FAILURE = 2
    CONSERVATION_FAILURE = 3
    APPROXIMATION_INVALID = 4


_WHOLE_CELL_SUCCESS = WholeCellStatus.SUCCESS
_WHOLE_CELL_INVALID_STATE = WholeCellStatus.INVALID_STATE
_WHOLE_CELL_PROCESS_FAILURE = WholeCellStatus.PROCESS_FAILURE
_WHOLE_CELL_CONSERVATION_FAILURE = WholeCellStatus.CONSERVATION_FAILURE
_WHOLE_CELL_APPROXIMATION_INVALID = WholeCellStatus.APPROXIMATION_INVALID


def _name(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{owner} must be a non-empty, trimmed string.")
    return value


def _path_name(value: str, owner: str, /) -> str:
    name = _name(value, owner)
    if "." in name:
        raise ValueError(f"{owner} must not contain the reserved '.' delimiter.")
    return name


class ExchangeFieldSpec(StrictModule, NonTrainableState):
    """Typed scalar amount exchanged by independently prepared cell processes."""

    name: str = eqx.field(static=True)
    quantity: str = eqx.field(static=True)
    unit: str = eqx.field(static=True)
    reservoir: bool = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        /,
        *,
        quantity: str = "count",
        unit: str = "molecule",
        reservoir: bool = False,
    ):
        if not isinstance(reservoir, bool):
            raise TypeError("reservoir must be bool.")
        self.name = _path_name(name, "Exchange field name")
        self.quantity = _name(quantity, "Exchange field quantity")
        self.unit = _name(unit, "Exchange field unit")
        self.reservoir = reservoir


class WholeCellProcessBinding(StrictModule, NonTrainableState):
    """Total, typed mapping from one process network's species to exchange fields."""

    name: str = eqx.field(static=True)
    network: PreparedStoichiometricNetwork
    species_to_fields: tuple[tuple[str, str], ...] = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        network: PreparedStoichiometricNetwork,
        species_to_fields: Mapping[str, str],
        /,
    ):
        if not isinstance(network, PreparedStoichiometricNetwork):
            raise TypeError("network must be PreparedStoichiometricNetwork.")
        if not isinstance(species_to_fields, Mapping):
            raise TypeError("species_to_fields must be a mapping.")
        name_value = _path_name(name, "Whole-cell process name")
        mapping = tuple(
            sorted(
                (
                    _name(species, "Bound species name"),
                    _name(field, "Bound exchange field name"),
                )
                for species, field in species_to_fields.items()
            )
        )
        species_names = tuple(item.name for item in network.plan.species)
        if {species for species, _ in mapping} != set(species_names):
            raise ValueError(
                "species_to_fields must map every network species exactly once."
            )
        if len(mapping) != len(species_names):
            raise ValueError("species_to_fields contains duplicate species.")
        field_names = tuple(field for _, field in mapping)
        if len(set(field_names)) != len(field_names):
            raise ValueError(
                "One process cannot map multiple species to one exchange field."
            )
        self.name = name_value
        self.network = network
        self.species_to_fields = mapping
        self.binding_id = canonical_fingerprint(
            {
                "kind": "whole-cell-process-binding",
                "name": name_value,
                "network": network.network_id,
                "mapping": mapping,
            }
        )


class MultirateScheduleEntry(StrictModule, NonTrainableState):
    """Static substep count for one process within every atomic macro step."""

    process_name: str = eqx.field(static=True)
    substeps: int = eqx.field(static=True)
    minimum_copy_number: float = eqx.field(static=True)
    require_regime_valid: bool = eqx.field(static=True)

    def __init__(
        self,
        process_name: str,
        substeps: int,
        /,
        *,
        minimum_copy_number: float = 20.0,
        require_regime_valid: bool = True,
    ):
        name = _path_name(process_name, "Scheduled process name")
        if isinstance(substeps, bool) or not isinstance(substeps, (int, np.integer)):
            raise ValueError("substeps must be a positive integer.")
        count = int(substeps)
        if count <= 0:
            raise ValueError("substeps must be a positive integer.")
        if isinstance(minimum_copy_number, bool) or not isinstance(
            minimum_copy_number, (int, float, np.integer, np.floating)
        ):
            raise TypeError("minimum_copy_number must be numeric.")
        minimum = float(minimum_copy_number)
        if not np.isfinite(minimum) or minimum < 0.0:
            raise ValueError("minimum_copy_number must be finite and nonnegative.")
        if not isinstance(require_regime_valid, bool):
            raise TypeError("require_regime_valid must be bool.")
        self.process_name = name
        self.substeps = count
        self.minimum_copy_number = minimum
        self.require_regime_valid = require_regime_valid


class WholeCellRuntime(StrictModule):
    """Per-process runtime parameters aligned with a prepared assembly."""

    process_runtimes: tuple[StoichiometricRuntime, ...]

    def __init__(self, process_runtimes: Sequence[StoichiometricRuntime], /):
        values = tuple(process_runtimes)
        if any(not isinstance(item, StoichiometricRuntime) for item in values):
            raise TypeError(
                "process_runtimes must contain StoichiometricRuntime objects."
            )
        self.process_runtimes = values


class WholeCellState(StrictModule):
    """Fixed-capacity coupled amounts, cumulative ledgers, and commit epoch."""

    values: Array
    source_ledger: Array
    sink_ledger: Array
    epoch: Array
    assembly_id: str = eqx.field(static=True)
    lineage_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        source_ledger: ArrayLike,
        sink_ledger: ArrayLike,
        epoch: ArrayLike,
        assembly_id: str,
        lineage_id: str,
        /,
    ):
        amounts = jnp.asarray(values, dtype=float)
        source = jnp.asarray(source_ledger, dtype=amounts.dtype)
        sink = jnp.asarray(sink_ledger, dtype=amounts.dtype)
        epoch_value = jnp.asarray(epoch, dtype=jnp.int32)
        if (
            amounts.ndim != 1
            or source.shape != amounts.shape
            or sink.shape != amounts.shape
        ):
            raise ValueError(
                "Whole-cell state arrays must be equal one-dimensional shapes."
            )
        if epoch_value.shape != ():
            raise ValueError("Whole-cell epoch must be scalar.")
        self.values = amounts
        self.source_ledger = source
        self.sink_ledger = sink
        self.epoch = epoch_value
        self.assembly_id = _name(assembly_id, "Whole-cell assembly ID")
        self.lineage_id = _name(lineage_id, "Whole-cell lineage ID")


class WholeCellCheckpoint(StrictModule, NonTrainableState):
    """Host snapshot whose identity covers every state and ledger value."""

    state: WholeCellState
    checkpoint_id: str = eqx.field(static=True)
    assembly_id: str = eqx.field(static=True)


class WholeCellStepEvaluation(StrictModule):
    """Uncommitted coupled candidate with process, ledger, and conservation evidence."""

    base_values: Array
    base_source_ledger: Array
    base_sink_ledger: Array
    candidate: Array
    source_delta: Array
    sink_delta: Array
    process_valid: Array
    process_regime_valid: Array
    conservation_residual: Array
    state_valid: Array
    finite: Array
    regime_valid: Array
    valid: Array
    status: Array
    base_lineage_id: str = eqx.field(static=True)
    base_epoch: Array
    assembly_id: str = eqx.field(static=True)

    def commit(self, state: WholeCellState, /) -> WholeCellCommitResult:
        if not isinstance(state, WholeCellState):
            raise TypeError("state must be WholeCellState.")
        if state.lineage_id != self.base_lineage_id:
            raise ValueError("State and evaluation lineage identities must match.")
        if state.assembly_id != self.assembly_id:
            raise ValueError("State and evaluation assembly identities must match.")
        if state.values.shape != self.candidate.shape:
            raise ValueError("State and evaluation capacities must match.")
        current_state = (
            (state.epoch == self.base_epoch)
            & jnp.array_equal(state.values, self.base_values)
            & jnp.array_equal(state.source_ledger, self.base_source_ledger)
            & jnp.array_equal(state.sink_ledger, self.base_sink_ledger)
        )
        accepted = self.valid & current_state
        committed = WholeCellState(
            jnp.where(accepted, self.candidate, state.values),
            jnp.where(
                accepted,
                state.source_ledger + self.source_delta,
                state.source_ledger,
            ),
            jnp.where(
                accepted,
                state.sink_ledger + self.sink_delta,
                state.sink_ledger,
            ),
            jnp.where(
                accepted,
                state.epoch + jnp.asarray(1, dtype=state.epoch.dtype),
                state.epoch,
            ),
            state.assembly_id,
            state.lineage_id,
        )
        status = jnp.where(
            current_state,
            self.status,
            jnp.asarray(_WHOLE_CELL_INVALID_STATE, dtype=jnp.int32),
        )
        return WholeCellCommitResult(committed, accepted, status, self.assembly_id)


class WholeCellCommitResult(StrictModule):
    """Atomic commit result; rejected candidates preserve all state components."""

    state: WholeCellState
    committed: Array
    status: Array
    assembly_id: str = eqx.field(static=True)


class WholeCellAssemblyPlan(StrictModule, NonTrainableState):
    """Fixed capacities, typed process bindings, and a complete multirate schedule."""

    name: str = eqx.field(static=True)
    fields: tuple[ExchangeFieldSpec, ...]
    processes: tuple[WholeCellProcessBinding, ...]
    schedule: tuple[MultirateScheduleEntry, ...]
    field_capacity: int = eqx.field(static=True)
    process_capacity: int = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        fields: Sequence[ExchangeFieldSpec],
        processes: Sequence[WholeCellProcessBinding],
        schedule: Sequence[MultirateScheduleEntry],
        /,
        *,
        field_capacity: int,
        process_capacity: int,
        conservation_tolerance: float = 1.0e-6,
    ):
        name_value = _name(name, "Whole-cell plan name")
        field_values = tuple(fields)
        process_values = tuple(processes)
        schedule_values = tuple(schedule)
        if not field_values or any(
            not isinstance(item, ExchangeFieldSpec) for item in field_values
        ):
            raise TypeError("fields must contain ExchangeFieldSpec objects.")
        if not process_values or any(
            not isinstance(item, WholeCellProcessBinding) for item in process_values
        ):
            raise TypeError("processes must contain WholeCellProcessBinding objects.")
        if not schedule_values or any(
            not isinstance(item, MultirateScheduleEntry) for item in schedule_values
        ):
            raise TypeError("schedule must contain MultirateScheduleEntry objects.")
        field_names = tuple(item.name for item in field_values)
        process_names = tuple(item.name for item in process_values)
        schedule_names = tuple(item.process_name for item in schedule_values)
        if len(set(field_names)) != len(field_names):
            raise ValueError("Exchange field names must be unique.")
        if len(set(process_names)) != len(process_names):
            raise ValueError("Whole-cell process names must be unique.")
        if len(set(schedule_names)) != len(schedule_names) or set(schedule_names) != set(
            process_names
        ):
            raise ValueError("Schedule must contain every process exactly once.")
        fields_by_name = {item.name: item for item in field_values}
        for binding in process_values:
            species_by_name = {item.name: item for item in binding.network.plan.species}
            for species_name, field_name in binding.species_to_fields:
                if field_name not in fields_by_name:
                    raise ValueError(
                        f"Process {binding.name!r} maps to unknown field {field_name!r}."
                    )
                species = species_by_name[species_name]
                field = fields_by_name[field_name]
                if (
                    species.quantity != field.quantity
                    or species.unit != field.unit
                    or species.reservoir != field.reservoir
                ):
                    raise ValueError(
                        f"Exchange field {field_name!r} type or reservoir "
                        f"semantics are incompatible with species {species_name!r}."
                    )
        if (
            isinstance(field_capacity, bool)
            or not isinstance(field_capacity, (int, np.integer))
            or isinstance(process_capacity, bool)
            or not isinstance(process_capacity, (int, np.integer))
        ):
            raise ValueError("Whole-cell capacities must be integers.")
        fields_capacity = int(field_capacity)
        processes_capacity = int(process_capacity)
        if fields_capacity < len(field_values) or fields_capacity <= 0:
            raise ValueError("field_capacity must cover every exchange field.")
        if processes_capacity < len(process_values) or processes_capacity <= 0:
            raise ValueError("process_capacity must cover every process.")
        if isinstance(conservation_tolerance, bool) or not isinstance(
            conservation_tolerance, (int, float, np.integer, np.floating)
        ):
            raise TypeError("conservation_tolerance must be numeric.")
        tolerance = float(conservation_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("conservation_tolerance must be finite and nonnegative.")
        self.name = name_value
        self.fields = field_values
        self.processes = process_values
        self.schedule = schedule_values
        self.field_capacity = fields_capacity
        self.process_capacity = processes_capacity
        self.conservation_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "whole-cell-assembly-plan",
                "name": name_value,
                "fields": [
                    (item.name, item.quantity, item.unit, item.reservoir)
                    for item in field_values
                ],
                "processes": [item.binding_id for item in process_values],
                "schedule": [
                    (
                        item.process_name,
                        item.substeps,
                        item.minimum_copy_number,
                        item.require_regime_valid,
                    )
                    for item in schedule_values
                ],
                "capacities": (fields_capacity, processes_capacity),
                "conservation_tolerance": tolerance,
            }
        )

    def prepare(self) -> PreparedWholeCellAssembly:
        return PreparedWholeCellAssembly(self)


class PreparedWholeCellAssembly(StrictModule, NonTrainableState):
    """Prepared mapping arrays and atomic compiled multirate execution runtime."""

    plan: WholeCellAssemblyPlan
    process_mappings: tuple[Array, ...]
    process_order: tuple[int, ...] = eqx.field(static=True)
    field_mask: Array
    conservation_basis: Array
    conservation_basis_units: tuple[str, ...] = eqx.field(static=True)
    assembly_id: str = eqx.field(static=True)

    def __init__(self, plan: WholeCellAssemblyPlan, /):
        if not isinstance(plan, WholeCellAssemblyPlan):
            raise TypeError("plan must be WholeCellAssemblyPlan.")
        field_index = {item.name: index for index, item in enumerate(plan.fields)}
        process_index = {item.name: index for index, item in enumerate(plan.processes)}
        mappings = []
        global_rows = []
        for binding in plan.processes:
            local_species = {
                item.name: index
                for index, item in enumerate(binding.network.plan.species)
            }
            mapping = np.empty(binding.network.species_count, dtype=np.int32)
            for species_name, field_name in binding.species_to_fields:
                mapping[local_species[species_name]] = field_index[field_name]
            mappings.append(jnp.asarray(mapping))
            for local_row in np.asarray(binding.network.stoichiometry):
                global_row = np.zeros(plan.field_capacity, dtype=float)
                global_row[mapping] = local_row
                global_rows.append(global_row)
        rows = np.asarray(global_rows)
        basis_rows = []
        basis_units = []
        field_types = tuple(
            dict.fromkeys((field.quantity, field.unit) for field in plan.fields)
        )
        for quantity, unit in field_types:
            indices = np.asarray(
                [
                    index
                    for index, field in enumerate(plan.fields)
                    if (field.quantity, field.unit) == (quantity, unit)
                ],
                dtype=np.int32,
            )
            block = rows[:, indices]
            singular_values = np.linalg.svd(block, compute_uv=False)
            threshold = (
                max(block.shape)
                * np.finfo(float).eps
                * max(float(np.max(singular_values, initial=0.0)), 1.0)
            )
            rank = int(np.sum(singular_values > threshold))
            _, _, right = np.linalg.svd(block, full_matrices=True)
            for local_basis in right[rank:]:
                embedded = np.zeros(plan.field_capacity, dtype=float)
                embedded[indices] = local_basis
                basis_rows.append(embedded)
                basis_units.append(f"{quantity}:{unit}")
        basis = (
            np.asarray(basis_rows)
            if basis_rows
            else np.zeros((0, plan.field_capacity), dtype=float)
        )
        order = tuple(process_index[item.process_name] for item in plan.schedule)
        field_mask = np.arange(plan.field_capacity) < len(plan.fields)
        assembly_id = canonical_fingerprint(
            {
                "kind": "prepared-whole-cell-assembly",
                "plan": plan.plan_id,
                "mappings": [
                    array_tree_fingerprint(np.asarray(item)) for item in mappings
                ],
            }
        )
        self.plan = plan
        self.process_mappings = tuple(mappings)
        self.process_order = order
        self.field_mask = jnp.asarray(field_mask)
        self.conservation_basis = jnp.asarray(basis)
        self.conservation_basis_units = tuple(basis_units)
        self.assembly_id = assembly_id

    def default_runtime(self) -> WholeCellRuntime:
        return WholeCellRuntime(
            tuple(item.network.default_runtime() for item in self.plan.processes)
        )

    def initial_state(self, values: ArrayLike, /) -> WholeCellState:
        raw = jnp.asarray(values)
        if raw.dtype == jnp.bool_:
            raise TypeError("Initial whole-cell values must not be boolean.")
        amounts = raw.astype(float)
        if amounts.shape == (len(self.plan.fields),):
            amounts = jnp.pad(
                amounts, (0, self.plan.field_capacity - len(self.plan.fields))
            )
        if amounts.shape != (self.plan.field_capacity,):
            raise ValueError("Initial values must match field count or field_capacity.")
        host = np.asarray(amounts)
        if (
            np.any(~np.isfinite(host))
            or np.any(host < 0.0)
            or np.any(host[~np.asarray(self.field_mask)] != 0.0)
        ):
            raise ValueError(
                "Initial values must be finite, nonnegative, and zero in unused capacity."
            )
        lineage_id = canonical_fingerprint(
            {
                "kind": "whole-cell-lineage",
                "assembly": self.assembly_id,
                "initial": array_tree_fingerprint(host),
            }
        )
        return WholeCellState(
            amounts,
            jnp.zeros_like(amounts),
            jnp.zeros_like(amounts),
            jnp.asarray(0, dtype=jnp.int32),
            self.assembly_id,
            lineage_id,
        )

    def checkpoint(self, state: WholeCellState, /) -> WholeCellCheckpoint:
        self._validate_state_identity(state)
        checkpoint_id = canonical_fingerprint(
            {
                "kind": "whole-cell-checkpoint",
                "assembly": self.assembly_id,
                "lineage": state.lineage_id,
                "epoch": int(state.epoch),
                "values": array_tree_fingerprint(np.asarray(state.values)),
                "source": array_tree_fingerprint(np.asarray(state.source_ledger)),
                "sink": array_tree_fingerprint(np.asarray(state.sink_ledger)),
            }
        )
        return WholeCellCheckpoint(state, checkpoint_id, self.assembly_id)

    def step(
        self,
        state: WholeCellState,
        duration: ArrayLike,
        runtime: WholeCellRuntime | None = None,
        /,
    ) -> WholeCellStepEvaluation:
        self._validate_state_identity(state)
        dt = jnp.asarray(duration, dtype=state.values.dtype)
        if dt.shape != ():
            raise ValueError("Whole-cell step duration must be scalar.")
        runtime_value = self.default_runtime() if runtime is None else runtime
        if not isinstance(runtime_value, WholeCellRuntime):
            raise TypeError("runtime must be WholeCellRuntime.")
        if len(runtime_value.process_runtimes) != len(self.plan.processes):
            raise ValueError("Whole-cell runtime must align with every process.")
        total_delta = jnp.zeros_like(state.values)
        total_source = jnp.zeros_like(state.values)
        total_sink = jnp.zeros_like(state.values)
        process_valid = jnp.ones((self.plan.process_capacity,), dtype=bool)
        process_regime_valid = jnp.ones((self.plan.process_capacity,), dtype=bool)
        required_regime_valid = jnp.asarray(True)
        for schedule_index, binding_index in enumerate(self.process_order):
            schedule = self.plan.schedule[schedule_index]
            binding = self.plan.processes[binding_index]
            mapping = self.process_mappings[binding_index]
            initial_local = state.values[mapping]
            local = initial_local
            local_minimum = jnp.min(
                jnp.where(
                    binding.network.copy_number_mask & ~binding.network.reservoir_mask,
                    local,
                    jnp.inf,
                ),
                initial=jnp.inf,
            )
            local_source = jnp.zeros_like(local)
            local_sink = jnp.zeros_like(local)
            local_valid = jnp.asarray(True)
            substep_duration = dt / float(schedule.substeps)
            for _ in range(schedule.substeps):
                evaluation = binding.network.evaluate(
                    local,
                    runtime_value.process_runtimes[binding_index],
                    mode="deterministic",
                )
                candidate = local + substep_duration * evaluation.drift
                step_valid = evaluation.successful & jnp.all(
                    jnp.isfinite(candidate) & (candidate >= 0.0)
                )
                local_minimum = jnp.minimum(
                    local_minimum,
                    jnp.min(
                        jnp.where(
                            binding.network.copy_number_mask
                            & ~binding.network.reservoir_mask,
                            candidate,
                            jnp.inf,
                        ),
                        initial=jnp.inf,
                    ),
                )
                local_valid = local_valid & step_valid
                local = jnp.where(step_valid, candidate, local)
                local_source = local_source + substep_duration * evaluation.source_rate
                local_sink = local_sink + substep_duration * evaluation.sink_rate
            total_delta = total_delta.at[mapping].add(local - initial_local)
            total_source = total_source.at[mapping].add(local_source)
            total_sink = total_sink.at[mapping].add(local_sink)
            process_valid = process_valid.at[binding_index].set(local_valid)
            local_regime_valid = local_minimum >= schedule.minimum_copy_number
            process_regime_valid = process_regime_valid.at[binding_index].set(
                local_regime_valid
            )
            required_regime_valid = required_regime_valid & (
                ~jnp.asarray(schedule.require_regime_valid) | local_regime_valid
            )
        candidate = state.values + total_delta
        for schedule_index, binding_index in enumerate(self.process_order):
            schedule = self.plan.schedule[schedule_index]
            binding = self.plan.processes[binding_index]
            mapping = self.process_mappings[binding_index]
            global_minimum = jnp.min(
                jnp.where(
                    binding.network.copy_number_mask & ~binding.network.reservoir_mask,
                    candidate[mapping],
                    jnp.inf,
                ),
                initial=jnp.inf,
            )
            global_regime_valid = global_minimum >= schedule.minimum_copy_number
            process_regime_valid = process_regime_valid.at[binding_index].set(
                process_regime_valid[binding_index] & global_regime_valid
            )
            required_regime_valid = required_regime_valid & (
                ~jnp.asarray(schedule.require_regime_valid) | global_regime_valid
            )
        state_valid = (
            jnp.all(jnp.isfinite(state.values) & (state.values >= 0.0))
            & jnp.all(jnp.isfinite(state.source_ledger) & (state.source_ledger >= 0.0))
            & jnp.all(jnp.isfinite(state.sink_ledger) & (state.sink_ledger >= 0.0))
            & jnp.all(jnp.where(self.field_mask, True, state.values == 0.0))
            & jnp.all(jnp.where(self.field_mask, True, state.source_ledger == 0.0))
            & jnp.all(jnp.where(self.field_mask, True, state.sink_ledger == 0.0))
            & (state.epoch >= 0)
            & jnp.isfinite(dt)
            & (dt >= 0.0)
        )
        finite = (
            jnp.all(jnp.isfinite(candidate))
            & jnp.all(jnp.isfinite(total_source))
            & jnp.all(jnp.isfinite(total_sink))
        )
        residual = contract(
            "ks,s->k",
            self.conservation_basis,
            total_delta + total_sink - total_source,
        )
        conservation_valid = jnp.all(
            jnp.abs(residual) <= self.plan.conservation_tolerance
        )
        process_success = jnp.all(process_valid)
        regime_valid = jnp.all(process_regime_valid)
        valid = (
            state_valid
            & finite
            & process_success
            & required_regime_valid
            & conservation_valid
            & jnp.all(candidate >= 0.0)
        )
        status = jnp.where(
            ~state_valid,
            _WHOLE_CELL_INVALID_STATE,
            jnp.where(
                ~process_success | ~finite | jnp.any(candidate < 0.0),
                _WHOLE_CELL_PROCESS_FAILURE,
                jnp.where(
                    ~required_regime_valid,
                    _WHOLE_CELL_APPROXIMATION_INVALID,
                    jnp.where(
                        ~conservation_valid,
                        _WHOLE_CELL_CONSERVATION_FAILURE,
                        _WHOLE_CELL_SUCCESS,
                    ),
                ),
            ),
        )
        return WholeCellStepEvaluation(
            state.values,
            state.source_ledger,
            state.sink_ledger,
            candidate,
            total_source,
            total_sink,
            process_valid,
            process_regime_valid,
            residual,
            state_valid,
            finite,
            regime_valid,
            valid,
            jnp.asarray(status, dtype=jnp.int32),
            state.lineage_id,
            state.epoch,
            self.assembly_id,
        )

    def _validate_state_identity(self, state: WholeCellState, /) -> None:
        if not isinstance(state, WholeCellState):
            raise TypeError("state must be WholeCellState.")
        if state.assembly_id != self.assembly_id:
            raise ValueError("Whole-cell state belongs to a different prepared assembly.")
        if state.values.shape != (self.plan.field_capacity,):
            raise ValueError("Whole-cell state capacity does not match the assembly.")

    def evidence_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {
            "whole_cell.plan_id": self.plan.plan_id,
            "whole_cell.prepared_id": self.assembly_id,
            "whole_cell.name": self.plan.name,
            "whole_cell.field_count": len(self.plan.fields),
            "whole_cell.process_count": len(self.plan.processes),
            "whole_cell.field_capacity": self.plan.field_capacity,
            "whole_cell.process_capacity": self.plan.process_capacity,
        }
        for field in self.plan.fields:
            prefix = f"whole_cell.field.{field.name}"
            fields[f"{prefix}.quantity"] = field.quantity
            fields[f"{prefix}.unit"] = field.unit
            fields[f"{prefix}.reservoir"] = field.reservoir
        for process in self.plan.processes:
            prefix = f"whole_cell.process.{process.name}"
            fields[f"{prefix}.binding_id"] = process.binding_id
            fields[f"{prefix}.network_id"] = process.network.network_id
        for entry in self.plan.schedule:
            fields[f"whole_cell.schedule.{entry.process_name}.substeps"] = entry.substeps
            fields[f"whole_cell.schedule.{entry.process_name}.minimum_copy_number"] = (
                entry.minimum_copy_number
            )
            fields[f"whole_cell.schedule.{entry.process_name}.require_regime_valid"] = (
                entry.require_regime_valid
            )
        return fields

    def evidence_units(self) -> dict[str, str]:
        """Return declared units aligned with whole-cell evidence fields."""
        units = {
            "whole_cell.plan_id": "identity",
            "whole_cell.prepared_id": "identity",
            "whole_cell.name": "label",
            "whole_cell.field_count": "count",
            "whole_cell.process_count": "count",
            "whole_cell.field_capacity": "count",
            "whole_cell.process_capacity": "count",
        }
        for field in self.plan.fields:
            prefix = f"whole_cell.field.{field.name}"
            units[f"{prefix}.quantity"] = "label"
            units[f"{prefix}.unit"] = "label"
            units[f"{prefix}.reservoir"] = "boolean"
        for process in self.plan.processes:
            prefix = f"whole_cell.process.{process.name}"
            units[f"{prefix}.binding_id"] = "identity"
            units[f"{prefix}.network_id"] = "identity"
        for entry in self.plan.schedule:
            units[f"whole_cell.schedule.{entry.process_name}.substeps"] = "count"
            units[f"whole_cell.schedule.{entry.process_name}.minimum_copy_number"] = (
                "count"
            )
            units[f"whole_cell.schedule.{entry.process_name}.require_regime_valid"] = (
                "boolean"
            )
        return units


__all__ = [
    "ExchangeFieldSpec",
    "MultirateScheduleEntry",
    "PreparedWholeCellAssembly",
    "WholeCellAssemblyPlan",
    "WholeCellCheckpoint",
    "WholeCellCommitResult",
    "WholeCellProcessBinding",
    "WholeCellRuntime",
    "WholeCellState",
    "WholeCellStepEvaluation",
    "WholeCellStatus",
]
