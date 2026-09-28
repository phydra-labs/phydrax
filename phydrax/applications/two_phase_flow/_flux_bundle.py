#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Geometric face-flux bundle of one directionally split VOF transport step."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

from ..._strict import StrictModule
from ._vof import FaceTuple


def cell_net_flux(value: Array, axis: int, periodic: bool, /) -> Array:
    """Net outflow of every cell from face values along ``axis``.

    Periodic face ``i`` joins cells ``i - 1`` and ``i``; nonperiodic axes carry
    both physical boundary faces.
    """

    moved = jnp.moveaxis(value, axis, 0)
    difference = (
        jnp.roll(moved, -1, axis=0) - moved if periodic else moved[1:] - moved[:-1]
    )
    return jnp.moveaxis(difference, 0, axis)


def face_donor(value: Array, flux: Array, axis: int, periodic: bool, /) -> Array:
    """Upwind (donor) cell value of every face along ``axis``."""

    moved = jnp.moveaxis(value, axis, 0)
    moved_flux = jnp.moveaxis(flux, axis, 0)
    if periodic:
        lower = jnp.roll(moved, 1, axis=0)
        upper = moved
    else:
        lower = jnp.concatenate((moved[:1], moved), axis=0)
        upper = jnp.concatenate((moved, moved[-1:]), axis=0)
    return jnp.moveaxis(jnp.where(moved_flux >= 0.0, lower, upper), 0, axis)


class TwoPhaseFluxBundle(StrictModule):
    """Per-sweep face fluxes that produced one step's liquid content.

    Each axis is swept once per step, in the cyclic order that starts at
    ``sweep_offset``. For the sweep along ``axis``, the liquid content update
    is

    ```text
    content <- content - dt (net(liquid_rates[axis]) - dilation_rates[axis])
    ```

    ``liquid_rates``/``total_rates`` are integrated face volume rates, and
    ``dilation_rates`` holds the liquid-content Weymouth–Yue source, including
    any declared compartment dilatation. Consumers replay the same ordered
    face rates: gas markers also replay the liquid source, while conservative
    mixture material scalars use the total face rates.
    """

    previous_liquid_content: Array
    liquid_rates: FaceTuple
    total_rates: FaceTuple
    dilation_rates: tuple[Array, ...]
    sweep_offset: Array
    step_size: Array

    def sweep_contents(self, periodic: tuple[bool, ...], /) -> tuple[Array, ...]:
        """Liquid content before each sweep position (``D + 1`` entries).

        Entry ``k`` is the content before the ``k``-th sweep of the cyclic
        order; the last entry is the final content.
        """

        dimension = len(self.liquid_rates)
        dt = self.step_size

        def ordered(offset: int, /) -> Callable[[], tuple[Array, ...]]:
            def run() -> tuple[Array, ...]:
                contents = [self.previous_liquid_content]
                for position in range(dimension):
                    axis = (offset + position) % dimension
                    net = cell_net_flux(self.liquid_rates[axis], axis, periodic[axis])
                    contents.append(contents[-1] - dt * (net - self.dilation_rates[axis]))
                return tuple(contents)

            return run

        return jax.lax.switch(
            jnp.asarray(self.sweep_offset, dtype=jnp.int32) % dimension,
            [ordered(offset) for offset in range(dimension)],
        )


__all__ = ["TwoPhaseFluxBundle", "cell_net_flux", "face_donor"]
