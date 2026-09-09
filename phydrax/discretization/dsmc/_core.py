#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class DSMCSpeciesPlan(StrictModule, NonTrainableState):
    names: tuple[str, ...] = eqx.field(static=True)
    molecular_masses: Array
    reference_diameters: Array
    viscosity_exponents: Array
    rotational_degrees: Array
    vibrational_temperatures: Array
    charges: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        names: Sequence[str],
        molecular_masses: ArrayLike,
        reference_diameters: ArrayLike,
        viscosity_exponents: ArrayLike,
        rotational_degrees: ArrayLike,
        vibrational_temperatures: ArrayLike,
        charges: ArrayLike,
        /,
    ):
        names_ = tuple(str(value) for value in names)
        arrays = tuple(
            np.asarray(value, dtype=float)
            for value in (
                molecular_masses,
                reference_diameters,
                viscosity_exponents,
                rotational_degrees,
                vibrational_temperatures,
                charges,
            )
        )
        shape = (len(names_),)
        if (
            not names_
            or len(set(names_)) != len(names_)
            or any(not value for value in names_)
            or any(value.shape != shape for value in arrays)
            or any(np.any(~np.isfinite(value)) for value in arrays)
            or np.any(arrays[0] <= 0.0)
            or np.any(arrays[1] <= 0.0)
            or np.any((arrays[2] < 0.5) | (arrays[2] > 1.0))
            or np.any(arrays[3] < 0.0)
            or np.any(arrays[4] < 0.0)
        ):
            raise ValueError("DSMC species masses, collision data, or modes are invalid.")
        self.names = names_
        (
            self.molecular_masses,
            self.reference_diameters,
            self.viscosity_exponents,
            self.rotational_degrees,
            self.vibrational_temperatures,
            self.charges,
        ) = tuple(jnp.asarray(value) for value in arrays)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dsmc-species",
                "names": names_,
                "molecular_masses": array_tree_fingerprint(self.molecular_masses),
                "reference_diameters": array_tree_fingerprint(self.reference_diameters),
                "viscosity_exponents": array_tree_fingerprint(self.viscosity_exponents),
                "rotational_degrees": array_tree_fingerprint(self.rotational_degrees),
                "vibrational_temperatures": array_tree_fingerprint(
                    self.vibrational_temperatures
                ),
                "charges": array_tree_fingerprint(self.charges),
            }
        )

    @property
    def species_count(self) -> int:
        return len(self.names)


class DSMCParticleState(StrictModule):
    position: Array
    velocity: Array
    species_index: Array
    rotational_energy: Array
    vibrational_energy: Array
    statistical_weight: Array
    cell_id: Array
    active: Array
    incarnation: Array

    @property
    def capacity(self) -> int:
        return self.position.shape[0]


class DSMCStructuredCellPlan(StrictModule, NonTrainableState):
    lower: Array
    upper: Array
    cell_counts: tuple[int, ...] = eqx.field(static=True)
    cell_widths: Array
    cell_volume: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        lower: ArrayLike,
        upper: ArrayLike,
        cell_counts: Sequence[int],
        /,
    ):
        lower_ = np.asarray(lower, dtype=float)
        upper_ = np.asarray(upper, dtype=float)
        counts = tuple(int(value) for value in cell_counts)
        if (
            lower_.ndim != 1
            or upper_.shape != lower_.shape
            or len(counts) != lower_.size
            or lower_.size not in (1, 2, 3)
            or any(value <= 0 for value in counts)
            or np.any(~np.isfinite(lower_))
            or np.any(~np.isfinite(upper_))
            or np.any(upper_ <= lower_)
        ):
            raise ValueError("DSMC structured cell bounds or counts are invalid.")
        widths = (upper_ - lower_) / np.asarray(counts)
        self.lower = jnp.asarray(lower_)
        self.upper = jnp.asarray(upper_)
        self.cell_counts = counts
        self.cell_widths = jnp.asarray(widths)
        self.cell_volume = float(np.prod(widths))
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dsmc-structured-cells",
                "lower": tuple(float(value) for value in lower_),
                "upper": tuple(float(value) for value in upper_),
                "counts": counts,
            }
        )

    @property
    def dimension(self) -> int:
        return len(self.cell_counts)

    @property
    def cell_count(self) -> int:
        return int(np.prod(self.cell_counts))

    def locate(self, position: ArrayLike, /) -> tuple[Array, Array]:
        value = jnp.asarray(position)
        if value.shape[-1] != self.dimension:
            raise ValueError("DSMC particle positions have the wrong dimension.")
        inside = jnp.all((value >= self.lower) & (value < self.upper), axis=-1)
        index = jnp.floor((value - self.lower) / self.cell_widths).astype(jnp.int32)
        index = jnp.clip(index, 0, jnp.asarray(self.cell_counts) - 1)
        flat = jnp.ravel_multi_index(
            tuple(index[..., axis] for axis in range(self.dimension)),
            self.cell_counts,
            mode="clip",
        )
        return flat, inside


class DSMCStreamingResult(StrictModule):
    state: DSMCParticleState
    crossed_boundary: Array
    deactivated_count: Array
    maximum_displacement_cells: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class DSMCStreamingPlan(StrictModule, NonTrainableState):
    cells: DSMCStructuredCellPlan
    boundary_kinds: tuple[tuple[str, str], ...] = eqx.field(static=True)
    maximum_crossings: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cells: DSMCStructuredCellPlan,
        boundary_kinds: Sequence[
            tuple[
                Literal["periodic", "specular", "open"],
                Literal["periodic", "specular", "open"],
            ]
        ],
        /,
        *,
        maximum_crossings: int = 8,
    ):
        boundaries = tuple((str(pair[0]), str(pair[1])) for pair in boundary_kinds)
        capacity = int(maximum_crossings)
        allowed = {"periodic", "specular", "open"}
        if (
            not isinstance(cells, DSMCStructuredCellPlan)
            or len(boundaries) != cells.dimension
            or any(
                lower not in allowed or upper not in allowed
                for lower, upper in boundaries
            )
            or any(
                (lower == "periodic") != (upper == "periodic")
                for lower, upper in boundaries
            )
            or capacity <= 0
        ):
            raise ValueError(
                "DSMC streaming boundaries or crossing capacity are invalid."
            )
        self.cells = cells
        self.boundary_kinds = boundaries
        self.maximum_crossings = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dsmc-streaming",
                "cells": cells.plan_id,
                "boundaries": boundaries,
                "maximum_crossings": capacity,
            }
        )

    def advance(
        self, state: DSMCParticleState, step_size: ArrayLike, /
    ) -> DSMCStreamingResult:
        step = jnp.asarray(step_size, dtype=state.position.dtype)
        if (
            state.position.shape != state.velocity.shape
            or state.position.shape[-1] != self.cells.dimension
        ):
            raise ValueError("DSMC position and velocity arrays are incompatible.")
        displacement = step * state.velocity
        position = state.position + displacement
        velocity = state.velocity
        active = state.active
        crossed = jnp.zeros((state.capacity, self.cells.dimension), dtype=bool)
        domain_length = self.cells.upper - self.cells.lower
        maximum_cells = jnp.max(jnp.abs(displacement) / self.cells.cell_widths)
        for axis, (lower_kind, upper_kind) in enumerate(self.boundary_kinds):
            below = position[:, axis] < self.cells.lower[axis]
            above = position[:, axis] >= self.cells.upper[axis]
            crossed = crossed.at[:, axis].set(below | above)
            if lower_kind == "periodic":
                wrapped = self.cells.lower[axis] + jnp.mod(
                    position[:, axis] - self.cells.lower[axis], domain_length[axis]
                )
                position = position.at[:, axis].set(wrapped)
            else:
                if lower_kind == "specular":
                    position = position.at[:, axis].set(
                        jnp.where(
                            below,
                            2.0 * self.cells.lower[axis] - position[:, axis],
                            position[:, axis],
                        )
                    )
                    velocity = velocity.at[:, axis].set(
                        jnp.where(below, -velocity[:, axis], velocity[:, axis])
                    )
                else:
                    active = active & ~below
                if upper_kind == "specular":
                    position = position.at[:, axis].set(
                        jnp.where(
                            above,
                            2.0 * self.cells.upper[axis] - position[:, axis],
                            position[:, axis],
                        )
                    )
                    velocity = velocity.at[:, axis].set(
                        jnp.where(above, -velocity[:, axis], velocity[:, axis])
                    )
                else:
                    active = active & ~above
        cell_id, inside = self.cells.locate(position)
        active = active & inside
        updated = DSMCParticleState(
            position,
            velocity,
            state.species_index,
            state.rotational_energy,
            state.vibrational_energy,
            state.statistical_weight,
            jnp.where(active, cell_id, -1),
            active,
            state.incarnation,
        )
        finite = jnp.all(jnp.isfinite(position)) & jnp.all(jnp.isfinite(velocity))
        successful = finite & (maximum_cells <= self.maximum_crossings)
        return DSMCStreamingResult(
            updated,
            crossed,
            jnp.sum(state.active & ~active),
            maximum_cells,
            finite,
            successful,
            self.plan_id,
        )


__all__ = [
    "DSMCParticleState",
    "DSMCSpeciesPlan",
    "DSMCStreamingPlan",
    "DSMCStreamingResult",
    "DSMCStructuredCellPlan",
]
