#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-shape contracts for cardiac cellular reaction models.

A reaction model owns one homogeneous, model-specific state layout.  The final
axis is the named channel axis; all preceding axes are independent cells.  The
layout is deliberately not a padded union of the states used by different cell
models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite
from typing import Mapping, Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array


ArrayLike = Array | np.ndarray | float


def _unique_nonempty(names: tuple[str, ...], label: str) -> tuple[str, ...]:
    if not names:
        raise ValueError(f"{label} must contain at least one name.")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{label} names must be non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} names must be unique.")
    return names


@dataclass(frozen=True)
class CardiacReactionStateLayout:
    """Named model-specific state channels stored on the final array axis."""

    state_names: tuple[str, ...]
    state_units: tuple[str, ...]
    gate_names: tuple[str, ...]
    concentration_names: tuple[str, ...]
    voltage_name: str = "voltage_mV"
    _indices: Mapping[str, int] = field(init=False, repr=False, compare=False)
    _gate_indices: tuple[int, ...] = field(init=False, repr=False)
    _concentration_indices: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        names = _unique_nonempty(tuple(self.state_names), "state_names")
        units = tuple(self.state_units)
        gates = tuple(self.gate_names)
        concentrations = tuple(self.concentration_names)
        if len(units) != len(names):
            raise ValueError("state_units must have one entry per state channel.")
        if any(not isinstance(unit, str) or not unit for unit in units):
            raise ValueError("state_units entries must be non-empty strings.")
        if self.voltage_name not in names:
            raise ValueError("voltage_name must identify a state channel.")
        if not set(gates).issubset(names):
            raise ValueError("Every gate name must identify a state channel.")
        if not set(concentrations).issubset(names):
            raise ValueError("Every concentration name must identify a state channel.")
        if set(gates) & set(concentrations):
            raise ValueError("Gate and concentration channel sets must be disjoint.")
        indices = {name: index for index, name in enumerate(names)}
        object.__setattr__(self, "state_names", names)
        object.__setattr__(self, "state_units", units)
        object.__setattr__(self, "gate_names", gates)
        object.__setattr__(self, "concentration_names", concentrations)
        object.__setattr__(self, "_indices", indices)
        object.__setattr__(self, "_gate_indices", tuple(indices[name] for name in gates))
        object.__setattr__(
            self,
            "_concentration_indices",
            tuple(indices[name] for name in concentrations),
        )

    @property
    def state_count(self) -> int:
        return len(self.state_names)

    @property
    def gate_count(self) -> int:
        return len(self.gate_names)

    @property
    def voltage_index(self) -> int:
        return self._indices[self.voltage_name]

    @property
    def gate_indices(self) -> tuple[int, ...]:
        return self._gate_indices

    @property
    def concentration_indices(self) -> tuple[int, ...]:
        return self._concentration_indices

    def index(self, name: str, /) -> int:
        """Return the pinned channel index for ``name``."""
        if name not in self._indices:
            raise KeyError(f"Unknown cardiac reaction state {name!r}.")
        return self._indices[name]

    def channel(self, state: Array, name: str, /) -> Array:
        """Select one named state channel without changing the cell axes."""
        self.require_shape(state)
        return state[..., self.index(name)]

    def require_shape(self, state: ArrayLike, /) -> Array:
        resolved = jnp.asarray(state)
        if resolved.ndim == 0 or resolved.shape[-1] != self.state_count:
            raise ValueError(
                "reaction state must have final axis size "
                f"{self.state_count}, received shape {resolved.shape}."
            )
        return resolved

    def pack(self, channels: Mapping[str, ArrayLike], /) -> Array:
        """Pack a complete named structure-of-arrays mapping."""
        supplied = tuple(channels)
        missing = tuple(name for name in self.state_names if name not in channels)
        extra = tuple(name for name in supplied if name not in self._indices)
        if missing or extra:
            raise ValueError(
                f"state channels do not match layout; missing={missing}, extra={extra}."
            )
        arrays = tuple(jnp.asarray(channels[name]) for name in self.state_names)
        return jnp.stack(jnp.broadcast_arrays(*arrays), axis=-1)

    def unpack(self, state: ArrayLike, /) -> dict[str, Array]:
        """Return an inspectable named structure-of-arrays view."""
        resolved = self.require_shape(state)
        return {name: resolved[..., index] for index, name in enumerate(self.state_names)}


@dataclass(frozen=True)
class CardiacReactionParameterLayout:
    """Pinned ordering and units for numeric model parameters."""

    parameter_names: tuple[str, ...]
    parameter_units: tuple[str, ...]
    _indices: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        names = _unique_nonempty(tuple(self.parameter_names), "parameter_names")
        units = tuple(self.parameter_units)
        if len(units) != len(names):
            raise ValueError("parameter_units must have one entry per parameter.")
        if any(not isinstance(unit, str) or not unit for unit in units):
            raise ValueError("parameter_units entries must be non-empty strings.")
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "parameter_units", units)
        object.__setattr__(self, "_indices", {name: i for i, name in enumerate(names)})

    @property
    def parameter_count(self) -> int:
        return len(self.parameter_names)

    def index(self, name: str, /) -> int:
        if name not in self._indices:
            raise KeyError(f"Unknown cardiac reaction parameter {name!r}.")
        return self._indices[name]

    def require_shape(self, parameters: ArrayLike, /) -> Array:
        resolved = jnp.asarray(parameters)
        if resolved.ndim == 0 or resolved.shape[-1] != self.parameter_count:
            raise ValueError(
                "reaction parameters must have final axis size "
                f"{self.parameter_count}, received shape {resolved.shape}."
            )
        return resolved

    def pack(self, values: Mapping[str, ArrayLike], /) -> Array:
        missing = tuple(name for name in self.parameter_names if name not in values)
        extra = tuple(name for name in values if name not in self._indices)
        if missing or extra:
            raise ValueError(
                f"parameters do not match layout; missing={missing}, extra={extra}."
            )
        arrays = tuple(jnp.asarray(values[name]) for name in self.parameter_names)
        return jnp.stack(jnp.broadcast_arrays(*arrays), axis=-1)


class CardiacReactionEvaluation(eqx.Module):
    """Pure candidate rates, decomposed currents, calcium, and validity evidence."""

    state_rate: Array
    gate_steady_state: Array
    gate_time_constant_ms: Array
    current_density_uA_per_mm2: Array
    total_outward_current_uA_per_mm2: Array
    calcium_cytosol_mM: Array
    calcium_cytosol_rate_mM_per_ms: Array
    calcium_sr_flux_mM_per_ms: Array
    calcium_membrane_current_uA_per_mm2: Array
    charge_balance_residual_uA_per_mm2: Array
    valid: Array
    current_names: tuple[str, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def current(self, name: str, /) -> Array:
        """Select one outward-positive current by its declared name."""
        if name not in self.current_names:
            raise KeyError(f"Unknown reaction current {name!r}.")
        return self.current_density_uA_per_mm2[..., self.current_names.index(name)]


@runtime_checkable
class CardiacReactionModel(Protocol):
    """Protocol for pure fixed-shape cardiac reaction models."""

    model_id: str
    state_layout: CardiacReactionStateLayout
    parameter_layout: CardiacReactionParameterLayout
    current_names: tuple[str, ...]
    default_parameters: Array
    membrane_capacitance_uF_per_mm2: float
    membrane_surface_to_volume_per_mm: float

    def initialize(
        self,
        batch_shape: tuple[int, ...] = (),
        *,
        dtype: object | None = None,
    ) -> Array: ...

    def evaluate(
        self,
        state: Array,
        parameters: Array | None = None,
        *,
        stimulus_current_uA_per_mm2: ArrayLike = 0.0,
    ) -> CardiacReactionEvaluation: ...

    def rates(
        self,
        state: Array,
        parameters: Array | None = None,
        *,
        stimulus_current_uA_per_mm2: ArrayLike = 0.0,
    ) -> Array: ...

    def exact_gate_update(
        self,
        state: Array,
        dt_ms: ArrayLike,
        parameters: Array | None = None,
    ) -> Array: ...

    def currents(self, state: Array, parameters: Array | None = None) -> Array: ...

    def admissible(self, state: Array, parameters: Array | None = None) -> Array: ...

    def validate_state(
        self, state: ArrayLike, parameters: ArrayLike | None = None
    ) -> None: ...


@dataclass(frozen=True)
class ReactionPlan:
    """Host-side plan pinning one homogeneous model, block size, and dtype."""

    model: CardiacReactionModel
    node_count: int
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.float64))
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.node_count, int) or isinstance(self.node_count, bool):
            raise TypeError("node_count must be an integer.")
        if self.node_count <= 0:
            raise ValueError("node_count must be positive.")
        if not isinstance(self.model, CardiacReactionModel):
            raise TypeError("model does not implement CardiacReactionModel.")
        dtype = np.dtype(self.dtype)
        if dtype.kind != "f":
            raise TypeError("reaction dtype must be floating point.")
        identity = (
            f"cardiac-reaction-plan-v1\0{self.model.model_id}\0{self.node_count}\0"
            f"{dtype.str}\0{self.model.state_layout.state_names!r}"
        )
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "plan_id", sha256(identity.encode("utf-8")).hexdigest())

    def prepare(self) -> PreparedReaction:
        return PreparedReaction(self)


@dataclass(frozen=True)
class PreparedReaction:
    """Concrete adapter consumed by fixed-topology tissue worksets.

    ``gates`` in this split interface contains every model-local state after
    voltage, including concentrations.  ``true_gate_count`` distinguishes the
    first-order gates that receive an exponential update.
    """

    plan: ReactionPlan
    model_id: str = field(init=False)
    node_count: int = field(init=False)
    state_count: int = field(init=False)
    gate_count: int = field(init=False)
    true_gate_count: int = field(init=False)
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        layout = self.plan.model.state_layout
        if layout.voltage_index != 0:
            raise ValueError(
                "PreparedReaction requires voltage to be state channel zero."
            )
        object.__setattr__(self, "model_id", self.plan.model.model_id)
        object.__setattr__(self, "node_count", self.plan.node_count)
        object.__setattr__(self, "state_count", layout.state_count)
        object.__setattr__(self, "gate_count", layout.state_count - 1)
        object.__setattr__(self, "true_gate_count", layout.gate_count)
        object.__setattr__(self, "plan_id", self.plan.plan_id)

    @property
    def model(self) -> CardiacReactionModel:
        return self.plan.model

    def initialize(
        self,
        node_count: int | None = None,
        dtype: object | None = None,
    ) -> tuple[Array, Array]:
        count = self.node_count if node_count is None else node_count
        if count != self.node_count:
            raise ValueError("node_count differs from the pinned reaction plan.")
        resolved_dtype = self.plan.dtype if dtype is None else np.dtype(dtype)
        if resolved_dtype != self.plan.dtype:
            raise ValueError("dtype differs from the pinned reaction plan.")
        state = self.model.initialize((count,), dtype=resolved_dtype)
        return state[..., 0], state[..., 1:]

    def _join(self, voltage_mV: Array, gates: Array) -> Array:
        voltage = jnp.asarray(voltage_mV)
        local = jnp.asarray(gates)
        if voltage.shape != (self.node_count,):
            raise ValueError(
                f"voltage_mV must have shape {(self.node_count,)}, got {voltage.shape}."
            )
        if local.shape != (self.node_count, self.gate_count):
            raise ValueError(
                "gates must have shape "
                f"{(self.node_count, self.gate_count)}, got {local.shape}."
            )
        return jnp.concatenate((voltage[..., None], local), axis=-1)

    def rates(
        self,
        voltage_mV: Array,
        gates: Array,
        stimulus_uA_per_mm3: ArrayLike = 0.0,
        parameters: Array | None = None,
    ) -> tuple[Array, Array]:
        state = self._join(voltage_mV, gates)
        evaluation = self.model.evaluate(state, parameters)
        stimulus = jnp.asarray(stimulus_uA_per_mm3, dtype=state.dtype)
        applied_rate = stimulus / (
            self.model.membrane_surface_to_volume_per_mm
            * self.model.membrane_capacitance_uF_per_mm2
        )
        return evaluation.state_rate[..., 0] + applied_rate, evaluation.state_rate[
            ..., 1:
        ]

    def exact_gate_update(
        self,
        voltage_mV: Array,
        gates: Array,
        dt_ms: ArrayLike,
        parameters: Array | None = None,
    ) -> Array:
        state = self._join(voltage_mV, gates)
        updated = self.model.exact_gate_update(state, dt_ms, parameters)
        return updated[..., 1:]

    def currents(
        self,
        voltage_mV: Array,
        gates: Array,
        parameters: Array | None = None,
    ) -> Array:
        state = self._join(voltage_mV, gates)
        surface = self.model.evaluate(state, parameters).total_outward_current_uA_per_mm2
        return self.model.membrane_surface_to_volume_per_mm * surface

    def current_components(
        self,
        voltage_mV: Array,
        gates: Array,
        parameters: Array | None = None,
    ) -> Array:
        return self.model.currents(self._join(voltage_mV, gates), parameters)

    def admissible(
        self,
        voltage_mV: Array,
        gates: Array,
        parameters: Array | None = None,
    ) -> Array:
        return self.model.admissible(self._join(voltage_mV, gates), parameters)


def plan_reaction(
    model: CardiacReactionModel,
    node_count: int,
    *,
    dtype: object = np.float64,
) -> ReactionPlan:
    """Plan one immutable homogeneous reaction block."""
    return ReactionPlan(model=model, node_count=node_count, dtype=np.dtype(dtype))


def prepare_reaction(plan: ReactionPlan, /) -> PreparedReaction:
    """Materialize a host-inspectable fixed-shape reaction adapter."""
    return plan.prepare()


def require_positive_finite(value: float, name: str, /) -> float:
    """Validate a positive scalar model construction parameter."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar, not bool.")
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return resolved


__all__ = [
    "CardiacReactionEvaluation",
    "CardiacReactionModel",
    "CardiacReactionParameterLayout",
    "CardiacReactionStateLayout",
    "PreparedReaction",
    "ReactionPlan",
    "plan_reaction",
    "prepare_reaction",
]
