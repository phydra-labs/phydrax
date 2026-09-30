#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core
from jax.typing import ArrayLike

from ._topology import CellComplexTopology, OrientedIncidence


def reorient_cochain(values: ArrayLike, signs: ArrayLike, /, *, cell_axis: int) -> Array:
    """Change the oriented cell basis along an explicitly declared value axis."""
    array = jnp.asarray(values)
    orientation = jnp.asarray(signs)
    if orientation.ndim != 1:
        raise ValueError("Orientation signs must be a rank-1 array.")
    axis = cell_axis + array.ndim if cell_axis < 0 else cell_axis
    if axis < 0 or axis >= array.ndim:
        raise ValueError("cell_axis is out of range.")
    if array.shape[axis] != orientation.size:
        raise ValueError("Orientation signs must match the cochain cell axis.")
    if not isinstance(orientation, jax_core.Tracer):
        if np.any(np.abs(np.asarray(orientation)) != 1):
            raise ValueError("Orientation signs must be ±1.")
    broadcast_shape = tuple(
        orientation.size if index == axis else 1 for index in range(array.ndim)
    )
    return array * orientation.reshape(broadcast_shape)


def reorient_cell_complex(
    topology: CellComplexTopology, signs: Sequence[ArrayLike], /
) -> CellComplexTopology:
    """Change positive-degree cell orientations without changing vertex identity."""
    if not isinstance(topology, CellComplexTopology):
        raise TypeError("Reorientation requires CellComplexTopology.")
    orientations = tuple(np.asarray(value) for value in signs)
    if len(orientations) != len(topology.entity_sets):
        raise ValueError("signs must supply one array per degree.")
    for degree, (orientation, entities) in enumerate(
        zip(orientations, topology.entity_sets, strict=True)
    ):
        if orientation.shape != (entities.count,) or np.any(np.abs(orientation) != 1):
            raise ValueError(f"signs[{degree}] must contain {entities.count} values ±1.")
    if np.any(orientations[0] != 1):
        raise ValueError("Vertices have canonical orientation and cannot be reoriented.")
    incidences: list[OrientedIncidence] = []
    for incidence in topology.incidences:
        valid = np.asarray(incidence.relation.valid, dtype=np.bool_)
        source = np.asarray(incidence.relation.source_indices)[valid]
        target = np.asarray(incidence.relation.target_indices)[valid]
        coefficients = np.asarray(incidence.signs).copy()
        coefficients[valid] *= (
            orientations[incidence.degree - 1][source]
            * orientations[incidence.degree][target]
        )
        incidences.append(
            OrientedIncidence(
                incidence.degree,
                topology.entity_sets[incidence.degree - 1],
                topology.entity_sets[incidence.degree],
                incidence.relation,
                coefficients,
            )
        )
    return CellComplexTopology(topology.entity_sets, incidences)


__all__ = ["reorient_cochain", "reorient_cell_complex"]
