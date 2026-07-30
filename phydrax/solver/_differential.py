#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import isfinite
from typing import Any, Literal, TypeAlias

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from .._frozendict import frozendict
from .._strict import StrictModule
from .._uncertainty import UncertaintySource, validate_uncertainty_source


DifferentialInterpretation: TypeAlias = Literal["ito", "stratonovich"]
LevyAreaKind: TypeAlias = Literal["brownian", "space_time", "space_time_time"]
DifferentialVectorField: TypeAlias = Callable[[Array, Array, Any], ArrayLike]


class DifferentialProblem(StrictModule):
    """Finite-dimensional initial-value problem with optional stochastic forcing."""

    drift: DifferentialVectorField
    initial_state: Array
    t0: Array
    t1: Array
    args: Any
    diffusion: DifferentialVectorField | None
    interpretation: DifferentialInterpretation

    def __init__(
        self,
        drift: DifferentialVectorField,
        initial_state: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        args: Any = None,
        diffusion: DifferentialVectorField | None = None,
        interpretation: DifferentialInterpretation = "ito",
    ):
        if not callable(drift):
            raise TypeError("DifferentialProblem drift must be callable.")
        if diffusion is not None and not callable(diffusion):
            raise TypeError("DifferentialProblem diffusion must be callable or None.")
        start = jnp.asarray(t0, dtype=float)
        end = jnp.asarray(t1, dtype=float)
        if start.shape != () or end.shape != ():
            raise ValueError("DifferentialProblem t0 and t1 must be scalar.")
        start = eqx.error_if(
            start,
            ~(jnp.isfinite(start) & jnp.isfinite(end)),
            "DifferentialProblem time bounds must be finite.",
        )
        end = eqx.error_if(
            end,
            ~(end > start),
            "DifferentialProblem requires t1 > t0.",
        )
        if interpretation not in ("ito", "stratonovich"):
            raise ValueError("interpretation must be 'ito' or 'stratonovich'.")
        self.drift = drift
        self.initial_state = jnp.asarray(initial_state)
        self.t0 = start
        self.t1 = end
        self.args = args
        self.diffusion = diffusion
        self.interpretation = interpretation

    @property
    def stochastic(self) -> bool:
        return self.diffusion is not None


class WienerDriver(StrictModule):
    """One reproducible finite-dimensional Wiener-process realization."""

    key: Array
    noise_shape: tuple[int, ...]
    tolerance: float
    levy_area: LevyAreaKind
    basis_id: str | None
    realization_id: str | int | None

    def __init__(
        self,
        key: Key[Array, ""],
        noise_shape: Sequence[int] = (),
        /,
        *,
        tolerance: float = 1e-3,
        levy_area: LevyAreaKind = "brownian",
        basis_id: str | None = None,
        realization_id: str | int | None = None,
    ):
        try:
            key_data = jr.key_data(key)
        except (TypeError, ValueError) as exc:
            raise TypeError("WienerDriver key must be a scalar JAX PRNG key.") from exc
        if key_data.shape != (2,):
            raise ValueError("WienerDriver key must be one scalar JAX PRNG key.")
        shape = tuple(int(size) for size in noise_shape)
        if any(size <= 0 for size in shape):
            raise ValueError("WienerDriver noise dimensions must be positive.")
        tolerance_value = float(tolerance)
        if not isfinite(tolerance_value) or tolerance_value <= 0.0:
            raise ValueError("WienerDriver tolerance must be finite and positive.")
        if levy_area not in ("brownian", "space_time", "space_time_time"):
            raise ValueError(
                "levy_area must be 'brownian', 'space_time', or 'space_time_time'."
            )
        if basis_id is not None and (not isinstance(basis_id, str) or not basis_id):
            raise ValueError("WienerDriver basis_id must be a non-empty string or None.")
        if realization_id is not None and not isinstance(realization_id, (str, int)):
            raise TypeError(
                "WienerDriver realization_id must be a string, integer, or None."
            )
        if isinstance(realization_id, str) and not realization_id:
            raise ValueError("WienerDriver realization_id must not be empty.")
        self.key = key
        self.noise_shape = shape
        self.tolerance = tolerance_value
        self.levy_area = levy_area
        self.basis_id = basis_id
        self.realization_id = realization_id


class DifferentialSolution(StrictModule):
    """Saved trajectory values plus solver and stochastic-driver provenance."""

    times: Array
    states: Array
    valid: Array
    sample_shape: tuple[int, ...]
    backend_result: Any
    stats: frozendict[str, Any]
    event_mask: Any
    driver: WienerDriver | None
    realization_keys: Array | None
    solver_name: str
    interpretation: DifferentialInterpretation

    def __init__(
        self,
        *,
        times: ArrayLike,
        states: ArrayLike,
        valid: ArrayLike,
        sample_shape: Sequence[int] = (),
        backend_result: Any,
        stats: dict[str, Any] | frozendict[str, Any],
        event_mask: Any = None,
        driver: WienerDriver | None = None,
        realization_keys: Array | None = None,
        solver_name: str,
        interpretation: DifferentialInterpretation,
    ):
        samples = tuple(int(size) for size in sample_shape)
        if any(size <= 0 for size in samples):
            raise ValueError("DifferentialSolution sample dimensions must be positive.")
        times_array = jnp.asarray(times)
        states_array = jnp.asarray(states)
        valid_array = jnp.asarray(valid, dtype=bool)
        if times_array.ndim != len(samples) + 1:
            raise ValueError(
                "DifferentialSolution times must have shape sample_shape + (num_times,)."
            )
        if times_array.shape[: len(samples)] != samples:
            raise ValueError("DifferentialSolution times do not match sample_shape.")
        trajectory_shape = samples + (int(times_array.shape[-1]),)
        if states_array.shape[: len(trajectory_shape)] != trajectory_shape:
            raise ValueError(
                "DifferentialSolution states must begin with sample_shape + (num_times,)."
            )
        if valid_array.shape != trajectory_shape:
            raise ValueError(
                f"DifferentialSolution valid must have shape {trajectory_shape}; "
                f"got {valid_array.shape}."
            )
        if not isinstance(solver_name, str) or not solver_name:
            raise ValueError("DifferentialSolution solver_name must be non-empty.")
        if interpretation not in ("ito", "stratonovich"):
            raise ValueError("interpretation must be 'ito' or 'stratonovich'.")
        if realization_keys is not None:
            key_data = jr.key_data(realization_keys)
            if key_data.shape[:-1] != samples or key_data.shape[-1:] != (2,):
                raise ValueError(
                    "DifferentialSolution realization keys must align with sample_shape."
                )
        self.times = times_array
        self.states = states_array
        self.valid = valid_array
        self.sample_shape = samples
        self.backend_result = backend_result
        self.stats = frozendict(dict(stats))
        self.event_mask = event_mask
        self.driver = driver
        self.realization_keys = realization_keys
        self.solver_name = solver_name
        self.interpretation = interpretation

    @property
    def num_times(self) -> int:
        return int(self.times.shape[-1])

    @property
    def successful(self) -> Array:
        """Whether every requested saved value is finite for each realization."""
        return jnp.all(self.valid, axis=-1)

    def to_predictive(
        self,
        /,
        *,
        sample_dim: str = "__phydra_uq_process",
        time_dim: str = "t",
        state_dims: Sequence[str | None] | None = None,
        source: UncertaintySource = "process",
    ) -> Any:
        """Convert an ensemble trajectory to a coordinate-aware predictive field."""
        if len(self.sample_shape) != 1:
            raise ValueError(
                "DifferentialSolution.to_predictive requires one ensemble sample axis."
            )
        if not isinstance(sample_dim, str) or not sample_dim:
            raise ValueError("sample_dim must be a non-empty string.")
        if not isinstance(time_dim, str) or not time_dim or time_dim == sample_dim:
            raise ValueError("time_dim must be non-empty and distinct from sample_dim.")
        source_value = validate_uncertainty_source(source)
        state_ndim = self.states.ndim - 2
        if state_dims is None:
            resolved_state_dims = (None,) * state_ndim
        else:
            resolved_state_dims = tuple(state_dims)
            if len(resolved_state_dims) != state_ndim:
                raise ValueError(
                    f"state_dims must contain {state_ndim} entries; "
                    f"got {len(resolved_state_dims)}."
                )
        from ..uq._predictive import PredictiveField, SampleAxis

        samples = cx.Field(
            self.states,
            dims=(sample_dim, time_dim) + resolved_state_dims,
        )
        sample_valid = cx.Field(self.successful, dims=(sample_dim,))
        return PredictiveField(
            samples,
            (SampleAxis(sample_dim, source_value),),
            valid=sample_valid,
        )


__all__ = [
    "DifferentialInterpretation",
    "DifferentialProblem",
    "DifferentialSolution",
    "DifferentialVectorField",
    "LevyAreaKind",
    "WienerDriver",
]
