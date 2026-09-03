#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._training import TargetParameterState, TrainingProgress
from ..domain import DomainFunction
from ..enforcement import EnforcementState
from ..sampling.collocation import CausalTimeSlabSchedule
from ..terms import ResidualBlockLayout, ResidualBlockRef


PseudoTimeFreshness = Literal["every_update", "periodic", "experimental_fixed"]
BalanceMethod = Literal["gradient_norm", "ntk_trace"]
CausalGateSignal = Literal["physical", "surrogate"]


def _callable_identity(value: Callable[..., Any], /) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    qualname = getattr(value, "__qualname__", type(value).__qualname__)
    identity = f"{module}.{qualname}"
    code = getattr(value, "__code__", None)
    if code is None:
        return identity
    implementation = canonical_fingerprint(
        {
            "bytecode": code.co_code.hex(),
            "constants": repr(code.co_consts),
            "names": code.co_names,
        }
    )
    return f"{identity}:{code.co_firstlineno}:{implementation}"


class ResidualRelaxationMap(StrictModule, NonTrainableState):
    """Map named physical fields into one residual's pseudo-time codomain."""

    operator: Callable[..., DomainFunction] = eqx.field(static=True)
    fields: tuple[str, ...] = eqx.field(static=True)
    blocks: ResidualBlockLayout | None
    map_id: str = eqx.field(static=True)

    def __init__(
        self,
        fields: str | Sequence[str],
        operator: Callable[..., DomainFunction],
        /,
        *,
        blocks: ResidualBlockLayout | None = None,
        map_id: str | None = None,
    ):
        fields_ = (str(fields),) if isinstance(fields, str) else tuple(map(str, fields))
        if not fields_ or any(not field for field in fields_):
            raise ValueError("Relaxation fields must be non-empty names.")
        if len(set(fields_)) != len(fields_):
            raise ValueError("Relaxation fields must be unique.")
        if not callable(operator):
            raise TypeError("Relaxation operator must be callable.")
        if blocks is not None and not isinstance(blocks, ResidualBlockLayout):
            raise TypeError("blocks must be a ResidualBlockLayout or None.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "residual-relaxation-map",
                    "fields": fields_,
                    "operator": _callable_identity(operator),
                    "blocks": None if blocks is None else blocks.layout_id,
                }
            )
            if map_id is None
            else str(map_id)
        )
        if not identifier:
            raise ValueError("map_id must be non-empty.")
        self.operator = operator
        self.fields = fields_
        self.blocks = blocks
        self.map_id = identifier

    def field(self, functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        missing = tuple(name for name in self.fields if name not in functions)
        if missing:
            raise KeyError(f"Missing relaxation fields {missing!r}.")
        result = self.operator(*(functions[name] for name in self.fields))
        if not isinstance(result, DomainFunction):
            raise TypeError("Relaxation operators must return a DomainFunction.")
        return result


class PseudoTransientAdaptation(StrictModule, NonTrainableState):
    """Safeguarded directional inverse-step adaptation."""

    start: int = eqx.field(static=True)
    every: int = eqx.field(static=True)
    momentum: float = eqx.field(static=True)
    minimum_inverse_step: float = eqx.field(static=True)
    maximum_inverse_step: float = eqx.field(static=True)
    minimum_state_displacement: float = eqx.field(static=True)
    minimum_residual_displacement: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        start: int = 2,
        every: int = 1000,
        momentum: float = 0.9,
        minimum_inverse_step: float = 1e-2,
        maximum_inverse_step: float = 1e2,
        minimum_state_displacement: float = 1e-12,
        minimum_residual_displacement: float = 1e-12,
    ):
        start_ = int(start)
        every_ = int(every)
        scalars = tuple(
            float(value)
            for value in (
                momentum,
                minimum_inverse_step,
                maximum_inverse_step,
                minimum_state_displacement,
                minimum_residual_displacement,
            )
        )
        if start_ < 1 or every_ < 1:
            raise ValueError("Pseudo-time adaptation start and every must be positive.")
        if any(not isfinite(value) for value in scalars):
            raise ValueError("Pseudo-time adaptation values must be finite.")
        if not 0.0 <= scalars[0] < 1.0:
            raise ValueError("Pseudo-time momentum must lie in [0, 1).")
        if scalars[1] <= 0.0 or scalars[2] < scalars[1]:
            raise ValueError("Pseudo-time inverse-step bounds are invalid.")
        if scalars[3] < 0.0 or scalars[4] < 0.0:
            raise ValueError("Pseudo-time displacement thresholds must be non-negative.")
        self.start = start_
        self.every = every_
        (
            self.momentum,
            self.minimum_inverse_step,
            self.maximum_inverse_step,
            self.minimum_state_displacement,
            self.minimum_residual_displacement,
        ) = scalars

    def due(self, step: int, /) -> bool:
        value = int(step)
        return value >= self.start and (value - self.start) % self.every == 0


class PseudoTransientPolicy(StrictModule, NonTrainableState):
    """Pseudo-transient transform for one authored residual term."""

    term_index: int = eqx.field(static=True)
    relaxation: ResidualRelaxationMap
    initial_inverse_step: Array
    adaptation: PseudoTransientAdaptation | None
    freshness: PseudoTimeFreshness = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        term_index: int,
        relaxation: ResidualRelaxationMap,
        /,
        *,
        inverse_step: ArrayLike = 1.0,
        adaptation: PseudoTransientAdaptation | None = None,
        freshness: PseudoTimeFreshness = "every_update",
    ):
        index = int(term_index)
        if index < 0:
            raise ValueError("term_index must be non-negative.")
        if not isinstance(relaxation, ResidualRelaxationMap):
            raise TypeError("relaxation must be a ResidualRelaxationMap.")
        values = jnp.asarray(inverse_step, dtype=float)
        if values.ndim > 1 or values.size < 1:
            raise ValueError("inverse_step must be a scalar or one-dimensional array.")
        if not bool(jnp.all(jnp.isfinite(values) & (values > 0.0))):
            raise ValueError("inverse_step values must be finite and positive.")
        if adaptation is not None and not isinstance(
            adaptation, PseudoTransientAdaptation
        ):
            raise TypeError("adaptation must be PseudoTransientAdaptation or None.")
        if freshness not in ("every_update", "periodic", "experimental_fixed"):
            raise ValueError("Unknown pseudo-time freshness policy.")
        self.term_index = index
        self.relaxation = relaxation
        self.initial_inverse_step = values.reshape(()) if values.ndim == 0 else values
        self.adaptation = adaptation
        self.freshness = freshness
        self.policy_id = canonical_fingerprint(
            {
                "kind": "pseudo-transient-policy",
                "term_index": index,
                "relaxation": relaxation.map_id,
                "inverse_step": jax.device_get(values).tolist(),
                "adaptation": (
                    None
                    if adaptation is None
                    else {
                        "start": adaptation.start,
                        "every": adaptation.every,
                        "momentum": adaptation.momentum,
                        "minimum_inverse_step": adaptation.minimum_inverse_step,
                        "maximum_inverse_step": adaptation.maximum_inverse_step,
                        "minimum_state_displacement": (
                            adaptation.minimum_state_displacement
                        ),
                        "minimum_residual_displacement": (
                            adaptation.minimum_residual_displacement
                        ),
                    }
                ),
                "freshness": freshness,
            }
        )


class CausalResidualPolicy(StrictModule, NonTrainableState):
    """Detached physical-time slab gates for one residual term."""

    term_index: int = eqx.field(static=True)
    time_label: str = eqx.field(static=True)
    schedule: CausalTimeSlabSchedule
    gate_signal: CausalGateSignal = eqx.field(static=True)
    per_block: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        term_index: int,
        time_label: str,
        schedule: CausalTimeSlabSchedule,
        /,
        *,
        gate_signal: CausalGateSignal = "physical",
        per_block: bool = True,
    ):
        index = int(term_index)
        label = str(time_label)
        if index < 0 or not label:
            raise ValueError("Causal term index and time label are invalid.")
        if not isinstance(schedule, CausalTimeSlabSchedule):
            raise TypeError("schedule must be a CausalTimeSlabSchedule.")
        if schedule.overlap_fraction != 0.0:
            raise ValueError(
                "Causal residual loss initially requires non-overlapping slabs."
            )
        if gate_signal not in ("physical", "surrogate"):
            raise ValueError("Unknown causal gate signal.")
        self.term_index = index
        self.time_label = label
        self.schedule = schedule
        self.gate_signal = gate_signal
        self.per_block = bool(per_block)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "causal-residual-policy",
                "term_index": index,
                "time_label": label,
                "schedule": schedule.schedule_id,
                "gate_signal": gate_signal,
                "per_block": bool(per_block),
            }
        )


class FunctionalTermBalancePolicy(StrictModule, NonTrainableState):
    """EMA term or residual-block balancing on a frozen surrogate."""

    method: BalanceMethod = eqx.field(static=True)
    blocks: tuple[ResidualBlockRef, ...]
    start: int = eqx.field(static=True)
    every: int = eqx.field(static=True)
    momentum: float = eqx.field(static=True)
    minimum: float = eqx.field(static=True)
    maximum: float = eqx.field(static=True)
    ntk_probes: int = eqx.field(static=True)
    maximum_relative_standard_error: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        blocks: Sequence[ResidualBlockRef],
        /,
        *,
        method: BalanceMethod = "gradient_norm",
        start: int = 10,
        every: int = 1000,
        momentum: float = 0.9,
        minimum: float = 1e-3,
        maximum: float = 1e3,
        ntk_probes: int = 16,
        maximum_relative_standard_error: float = 0.25,
    ):
        blocks_ = tuple(blocks)
        if not blocks_ or any(
            not isinstance(block, ResidualBlockRef) for block in blocks_
        ):
            raise TypeError("blocks must contain ResidualBlockRef values.")
        keys = tuple((block.term_index, block.block_name) for block in blocks_)
        if len(set(keys)) != len(keys):
            raise ValueError("Balanced residual blocks must be unique.")
        if method not in ("gradient_norm", "ntk_trace"):
            raise ValueError("Unknown functional term balance method.")
        start_ = int(start)
        every_ = int(every)
        probes = int(ntk_probes)
        scalars = tuple(
            float(value)
            for value in (
                momentum,
                minimum,
                maximum,
                maximum_relative_standard_error,
            )
        )
        if start_ < 1 or every_ < 1 or probes < 1:
            raise ValueError("Balance cadence and probe count must be positive.")
        if any(not isfinite(value) for value in scalars):
            raise ValueError("Balance controls must be finite.")
        if not 0.0 <= scalars[0] < 1.0:
            raise ValueError("Balance momentum must lie in [0, 1).")
        if scalars[1] <= 0.0 or scalars[2] < scalars[1] or scalars[3] <= 0.0:
            raise ValueError("Balance bounds or error threshold are invalid.")
        self.method = method
        self.blocks = blocks_
        self.start = start_
        self.every = every_
        (
            self.momentum,
            self.minimum,
            self.maximum,
            self.maximum_relative_standard_error,
        ) = scalars
        self.ntk_probes = probes
        self.policy_id = canonical_fingerprint(
            {
                "kind": "functional-term-balance",
                "method": method,
                "blocks": keys,
                "start": start_,
                "every": every_,
                "momentum": self.momentum,
                "minimum": self.minimum,
                "maximum": self.maximum,
                "ntk_probes": probes,
                "maximum_relative_standard_error": (self.maximum_relative_standard_error),
            }
        )

    def due(self, step: int, /) -> bool:
        value = int(step)
        return value >= self.start and (value - self.start) % self.every == 0


class FunctionalDiagnosticsPolicy(StrictModule, NonTrainableState):
    """Bounded gradient-alignment and NTK monitoring cadence."""

    every: int = eqx.field(static=True)
    gradient_alignment: bool = eqx.field(static=True)
    ntk: bool = eqx.field(static=True)
    ntk_probes: int = eqx.field(static=True)
    ntk_eigenvalues: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        every: int = 1000,
        gradient_alignment: bool = True,
        ntk: bool = False,
        ntk_probes: int = 16,
        ntk_eigenvalues: int = 8,
    ):
        values = tuple(int(value) for value in (every, ntk_probes, ntk_eigenvalues))
        if any(value < 1 for value in values):
            raise ValueError("Diagnostic cadence and capacities must be positive.")
        self.every, self.ntk_probes, self.ntk_eigenvalues = values
        self.gradient_alignment = bool(gradient_alignment)
        self.ntk = bool(ntk)

    def due(self, step: int, /) -> bool:
        return int(step) % self.every == 0


class FunctionalSelectionPolicy(StrictModule, NonTrainableState):
    """Fixed-evaluation model selection and optional early stopping."""

    every: int = eqx.field(static=True)
    mode: Literal["min", "max"] = eqx.field(static=True)
    min_delta: float = eqx.field(static=True)
    patience: int | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        every: int = 100,
        mode: Literal["min", "max"] = "min",
        min_delta: float = 0.0,
        patience: int | None = None,
    ):
        every_ = int(every)
        delta = float(min_delta)
        patience_ = None if patience is None else int(patience)
        if every_ < 1 or not isfinite(delta) or delta < 0.0:
            raise ValueError("Selection cadence or min_delta is invalid.")
        if mode not in ("min", "max"):
            raise ValueError("Selection mode must be 'min' or 'max'.")
        if patience_ is not None and patience_ < 1:
            raise ValueError("Selection patience must be positive or None.")
        self.every = every_
        self.mode = mode
        self.min_delta = delta
        self.patience = patience_

    def due(self, step: int, /) -> bool:
        return int(step) % self.every == 0


class FunctionalCheckpointPolicy(StrictModule, NonTrainableState):
    """Atomic functional-training checkpoint publication policy."""

    path: str = eqx.field(static=True)
    every: int = eqx.field(static=True)
    save_final: bool = eqx.field(static=True)

    def __init__(
        self,
        path: str | Path,
        /,
        *,
        every: int = 1000,
        save_final: bool = True,
    ):
        path_ = str(Path(path))
        every_ = int(every)
        if not path_ or every_ < 1:
            raise ValueError("Checkpoint path and cadence are invalid.")
        self.path = path_
        self.every = every_
        self.save_final = bool(save_final)

    def due(self, step: int, /) -> bool:
        return int(step) % self.every == 0


class FunctionalTrainingPlan(StrictModule, NonTrainableState):
    """Backend-neutral controls for one stateful functional training run."""

    pseudo_transient: tuple[PseudoTransientPolicy, ...]
    causal: tuple[CausalResidualPolicy, ...]
    term_balance: FunctionalTermBalancePolicy | None
    diagnostics: FunctionalDiagnosticsPolicy | None
    selection: FunctionalSelectionPolicy | None
    checkpoint: FunctionalCheckpointPolicy | None
    sharding: Any
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        pseudo_transient: Sequence[PseudoTransientPolicy] = (),
        causal: Sequence[CausalResidualPolicy] = (),
        term_balance: FunctionalTermBalancePolicy | None = None,
        diagnostics: FunctionalDiagnosticsPolicy | None = None,
        selection: FunctionalSelectionPolicy | None = None,
        checkpoint: FunctionalCheckpointPolicy | None = None,
        sharding: Any = None,
    ):
        pseudo = tuple(pseudo_transient)
        causal_ = tuple(causal)
        if any(not isinstance(value, PseudoTransientPolicy) for value in pseudo):
            raise TypeError("pseudo_transient must contain PseudoTransientPolicy values.")
        if any(not isinstance(value, CausalResidualPolicy) for value in causal_):
            raise TypeError("causal must contain CausalResidualPolicy values.")
        if len({value.term_index for value in pseudo}) != len(pseudo):
            raise ValueError("At most one pseudo-transient policy may target each term.")
        if len({value.term_index for value in causal_}) != len(causal_):
            raise ValueError("At most one causal policy may target each term.")
        expected = (
            (term_balance, FunctionalTermBalancePolicy, "term_balance"),
            (diagnostics, FunctionalDiagnosticsPolicy, "diagnostics"),
            (selection, FunctionalSelectionPolicy, "selection"),
            (checkpoint, FunctionalCheckpointPolicy, "checkpoint"),
        )
        for value, value_type, name in expected:
            if value is not None and not isinstance(value, value_type):
                raise TypeError(f"{name} has an invalid policy type.")
        if sharding is not None:
            from ._functional_sharding import FunctionalShardingPolicy

            if not isinstance(sharding, FunctionalShardingPolicy):
                raise TypeError("sharding must be FunctionalShardingPolicy or None.")
        self.pseudo_transient = pseudo
        self.causal = causal_
        self.term_balance = term_balance
        self.diagnostics = diagnostics
        self.selection = selection
        self.checkpoint = checkpoint
        self.sharding = sharding
        self.plan_id = canonical_fingerprint(
            {
                "kind": "functional-training-plan",
                "pseudo_transient": [value.policy_id for value in pseudo],
                "causal": [value.policy_id for value in causal_],
                "term_balance": None if term_balance is None else term_balance.policy_id,
                "diagnostics": (
                    None
                    if diagnostics is None
                    else {
                        "every": diagnostics.every,
                        "gradient_alignment": diagnostics.gradient_alignment,
                        "ntk": diagnostics.ntk,
                        "ntk_probes": diagnostics.ntk_probes,
                        "ntk_eigenvalues": diagnostics.ntk_eigenvalues,
                    }
                ),
                "selection": (
                    None
                    if selection is None
                    else {
                        "every": selection.every,
                        "mode": selection.mode,
                        "min_delta": selection.min_delta,
                        "patience": selection.patience,
                    }
                ),
                "checkpoint": (
                    None
                    if checkpoint is None
                    else {
                        "path": checkpoint.path,
                        "every": checkpoint.every,
                        "save_final": checkpoint.save_final,
                    }
                ),
                "sharding": (
                    None
                    if sharding is None
                    else {
                        "policy_id": sharding.policy_id,
                        "axis_mapping": dict(sharding.axis_mapping),
                        "mesh_shape": dict(sharding.mesh.shape),
                    }
                ),
            }
        )

    @property
    def stateful(self) -> bool:
        return bool(self.pseudo_transient) or self.term_balance is not None


class FunctionalTrainingState(StrictModule):
    """Exact resumable state at an accepted functional update boundary."""

    current_functions: PyTree[Any]
    best_functions: PyTree[Any]
    previous_functions: PyTree[Any] | None
    optimizer_state: PyTree[Any]
    target_state: TargetParameterState | None
    enforcement_state: EnforcementState | None
    key: Key[Array, ""]
    pseudo_inverse_steps: tuple[Array, ...]
    term_multipliers: Array
    previous_gradient: PyTree[Any] | None
    progress: TrainingProgress = eqx.field(static=True)
    run_id: str = eqx.field(static=True)
    gradient_accumulation: int = eqx.field(static=True)
    training_seconds: float = eqx.field(static=True)
    resumed_from_step: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        current_functions: PyTree[Any],
        best_functions: PyTree[Any],
        optimizer_state: PyTree[Any],
        key: Key[Array, ""],
        progress: TrainingProgress,
        run_id: str,
        gradient_accumulation: int = 1,
        target_state: TargetParameterState | None = None,
        enforcement_state: EnforcementState | None = None,
        previous_functions: PyTree[Any] | None = None,
        pseudo_inverse_steps: Sequence[ArrayLike] = (),
        term_multipliers: ArrayLike = (),
        previous_gradient: PyTree[Any] | None = None,
        training_seconds: float = 0.0,
        resumed_from_step: int = 0,
    ):
        if not isinstance(progress, TrainingProgress):
            raise TypeError("progress must be a TrainingProgress.")
        if target_state is not None and not isinstance(
            target_state, TargetParameterState
        ):
            raise TypeError("target_state must be a TargetParameterState or None.")
        if enforcement_state is not None and not isinstance(
            enforcement_state, EnforcementState
        ):
            raise TypeError("enforcement_state must be EnforcementState or None.")
        identifier = str(run_id)
        seconds = float(training_seconds)
        resumed = int(resumed_from_step)
        accumulation = int(gradient_accumulation)
        if (
            not identifier
            or not isfinite(seconds)
            or seconds < 0.0
            or resumed < 0
            or accumulation <= 0
        ):
            raise ValueError("Functional training state metadata is invalid.")
        self.current_functions = current_functions
        self.best_functions = best_functions
        self.previous_functions = previous_functions
        self.optimizer_state = optimizer_state
        self.target_state = target_state
        self.enforcement_state = enforcement_state
        self.key = key
        self.pseudo_inverse_steps = tuple(
            jnp.asarray(value) for value in pseudo_inverse_steps
        )
        self.term_multipliers = jnp.asarray(term_multipliers, dtype=float).reshape((-1,))
        self.previous_gradient = previous_gradient
        self.progress = progress
        self.run_id = identifier
        self.gradient_accumulation = accumulation
        self.training_seconds = seconds
        self.resumed_from_step = resumed


__all__ = [
    "BalanceMethod",
    "CausalGateSignal",
    "CausalResidualPolicy",
    "FunctionalCheckpointPolicy",
    "FunctionalDiagnosticsPolicy",
    "FunctionalSelectionPolicy",
    "FunctionalTermBalancePolicy",
    "FunctionalTrainingPlan",
    "FunctionalTrainingState",
    "PseudoTimeFreshness",
    "PseudoTransientAdaptation",
    "PseudoTransientPolicy",
    "ResidualRelaxationMap",
]
