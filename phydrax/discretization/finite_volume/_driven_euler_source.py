#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pure, cell-addressed body forcing and decaying heat for Euler dynamics."""

from __future__ import annotations

from typing import Any, final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from ..._sampling._addressing import derive_key, SampleAddress
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


_CELL_ADDRESS = SampleAddress("finite-volume", "driven-euler", role="cell")
_NEXT_ADDRESS = SampleAddress("finite-volume", "driven-euler", role="accepted-step")


class DrivenEulerSourceArgs(NamedTuple):
    """Driving banks in the native dynamics' full cell-row ordering.

    ``material`` has columns acceleration amplitude, specific heating rate,
    and heating depletion rate. ``history`` is elapsed driving age and
    ``clock`` is the time at which material and history were last committed.
    ``keys`` contains one typed JAX key per cell; ``active`` selects rows
    receiving forcing and accepted updates. Material parameters are assumed
    nonnegative. Construct keys from persistent IDs with :func:`driven_euler_cell_keys`.
    A staged source evaluation receives the rows selected by the native source
    cell indices; the accepted-step update always receives the full banks.
    """

    material: Array
    history: Array
    keys: Array
    clock: Array
    active: Array

    def select_rows(self, indices: Array, /) -> DrivenEulerSourceArgs:
        """Gather every bank in the native source-cell order."""
        return DrivenEulerSourceArgs(
            self.material[indices],
            self.history[indices],
            self.keys[indices],
            self.clock[indices],
            self.active[indices],
        )


def driven_euler_cell_keys(root_key: Array, id_hi: Array, id_lo: Array, /) -> Array:
    """Derive typed cell keys from high and low uint32 global-ID words.

    The two-word representation is exact even when JAX x64 is disabled and
    remains stable across partitioning and row permutations.
    """
    hi, lo = jnp.asarray(id_hi), jnp.asarray(id_lo)
    if hi.ndim != 1 or lo.shape != hi.shape:
        raise ValueError("Global ID words must be equally shaped rank-one arrays.")
    if hi.dtype != jnp.uint32 or lo.dtype != jnp.uint32:
        raise TypeError("Global ID words must have uint32 dtype.")
    if not jax.dtypes.issubdtype(root_key.dtype, jax.dtypes.prng_key) or root_key.shape:
        raise TypeError("root_key must be a scalar typed JAX key.")
    return jax.vmap(lambda h, l: derive_key(root_key, _CELL_ADDRESS, h, l))(hi, lo)


def _check_banks(args: DrivenEulerSourceArgs, rows: int, /) -> None:
    if not isinstance(args, DrivenEulerSourceArgs):
        raise TypeError("args must be DrivenEulerSourceArgs.")
    if args.material.shape != (rows, 3):
        raise ValueError("material must have shape (cell_count, 3).")
    if args.history.shape != (rows,) or args.clock.shape != (rows,):
        raise ValueError("history and clock must have shape (cell_count,).")
    if args.active.shape != (rows,) or args.active.dtype != jnp.bool_:
        raise ValueError("active must be a boolean array of shape (cell_count,).")
    if args.keys.shape != (rows,) or not jax.dtypes.issubdtype(
        args.keys.dtype, jax.dtypes.prng_key
    ):
        raise TypeError("keys must be a typed JAX key array of shape (cell_count,).")


@final
class DrivenEulerSource(StrictModule, NonTrainableState):
    """Cellwise stochastic-phase acceleration and exponentially decaying heat.

    For conservative Euler state ``(rho, rho*u..., E)``, acceleration acts
    along the first coordinate. The momentum rate is ``rho*a`` and the total
    energy rate is ``rho*u_x*a + rho*q``. The acceleration is nonnegative,
    with a random phase held fixed throughout each attempted time step. A
    successful step changes that phase; source evaluations never mutate banks.
    """

    angular_frequency: float = eqx.field(static=True)

    def __init__(self, angular_frequency: float) -> None:
        frequency = float(angular_frequency)
        if not np.isfinite(frequency) or frequency < 0:
            raise ValueError("angular_frequency must be finite and nonnegative.")
        self.angular_frequency = frequency

    def __call__(
        self,
        time: Array,
        state: Array,
        coordinates: Array,
        args: DrivenEulerSourceArgs,
        /,
    ) -> Array:
        if state.ndim != 2 or coordinates.ndim != 2:
            raise ValueError("state and coordinates must be rank-two arrays.")
        rows, dimension = coordinates.shape
        if dimension < 1 or state.shape != (rows, dimension + 2):
            raise ValueError(
                "Euler state must have density, momentum, and energy columns."
            )
        _check_banks(args, rows)
        phase = jax.vmap(lambda key: jr.uniform(key, (), dtype=state.dtype))(args.keys)
        dtype = state.dtype
        material = jnp.asarray(args.material, dtype=dtype)
        history = jnp.asarray(args.history, dtype=dtype)
        elapsed = jnp.asarray(time, dtype=dtype) - jnp.asarray(args.clock, dtype=dtype)
        angle = self.angular_frequency * (history + elapsed) + (2 * jnp.pi) * phase
        acceleration = material[:, 0] * (0.5 + 0.5 * jnp.sin(angle))
        heating = material[:, 1] * jnp.exp(-material[:, 2] * elapsed)
        force = state[:, 0] * acceleration
        energy = state[:, 1] * acceleration + state[:, 0] * heating
        value = jnp.zeros_like(state).at[:, 1].set(force).at[:, -1].set(energy)
        return jnp.where(args.active[:, None], value, jnp.zeros_like(value))


def advance_driven_euler_banks(
    prior_state: Any, result: Any, banks: DrivenEulerSourceArgs, /
) -> DrivenEulerSourceArgs:
    """Commit bank evolution after an accepted step via ``advance_with_auxiliary``.

    This callback is invoked only by the runtime's accepted branch. It keeps
    inactive rows bitwise unchanged, including their typed keys.
    """
    del prior_state
    rows = banks.material.shape[0]
    _check_banks(banks, rows)
    material_dt = jnp.asarray(result.accepted_step_size, dtype=banks.material.dtype)
    history_dt = jnp.asarray(result.accepted_step_size, dtype=banks.history.dtype)
    clock_dt = jnp.asarray(result.accepted_step_size, dtype=banks.clock.dtype)
    material = banks.material.at[:, 1].set(
        banks.material[:, 1] * jnp.exp(-banks.material[:, 2] * material_dt)
    )
    keys = jax.vmap(lambda key: derive_key(key, _NEXT_ADDRESS))(banks.keys)
    return DrivenEulerSourceArgs(
        jnp.where(banks.active[:, None], material, banks.material),
        jnp.where(banks.active, banks.history + history_dt, banks.history),
        jnp.where(banks.active, keys, banks.keys),
        jnp.where(banks.active, banks.clock + clock_dt, banks.clock),
        banks.active,
    )


__all__ = [
    "DrivenEulerSource",
    "DrivenEulerSourceArgs",
    "advance_driven_euler_banks",
    "driven_euler_cell_keys",
]
