#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Quantum-number-resolved projected-sphere observables."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...applications.fuzzy_space import evaluate_fuzzy_many_body_observables
from ._sphere import PreparedHaldaneSphereHamiltonian


class QuantumHallSphereObservables(StrictModule, NonTrainableState):
    orbital_occupations: Array
    one_body_density: Array
    entanglement_levels: Array
    entanglement_particle_numbers: Array
    entanglement_twice_projections: Array
    entanglement_entropy: Array
    total_twice_projection: Array
    norm_residual: Array
    one_body_trace_residual: Array
    probability_residual: Array
    result_id: str = eqx.field(static=True)


def evaluate_sphere_observables(
    prepared: PreparedHaldaneSphereHamiltonian,
    state: ArrayLike,
    /,
    *,
    orbital_cut: int,
) -> QuantumHallSphereObservables:
    if not isinstance(prepared, PreparedHaldaneSphereHamiltonian):
        raise TypeError("prepared must be PreparedHaldaneSphereHamiltonian.")
    low_level = prepared.many_body
    vector = np.asarray(state, dtype=np.complex128)
    if vector.shape != (low_level.dimension,):
        raise ValueError("Sphere state has the wrong sector dimension.")
    norm = float(np.vdot(vector, vector).real)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Sphere state must have finite positive norm.")
    vector = vector / np.sqrt(norm)
    base = evaluate_fuzzy_many_body_observables(
        low_level,
        vector,
        orbital_cut=orbital_cut,
    )
    cut = int(orbital_cut)
    occupations = np.asarray(low_level.occupations, dtype=np.int32)
    projections = np.arange(
        -prepared.sphere.twice_orbital_spin,
        prepared.sphere.twice_orbital_spin + 1,
        2,
        dtype=np.int32,
    )
    grouped: dict[
        tuple[int, int], list[tuple[tuple[int, ...], tuple[int, ...], complex]]
    ] = {}
    for occupation, coefficient in zip(occupations, vector, strict=True):
        left = tuple(int(value) for value in occupation[:cut])
        right = tuple(int(value) for value in occupation[cut:])
        block = (
            sum(left),
            sum(value * int(projections[index]) for index, value in enumerate(left)),
        )
        grouped.setdefault(block, []).append((left, right, coefficient))
    levels = []
    particle_labels = []
    projection_labels = []
    probabilities = []
    for block in sorted(grouped):
        entries = grouped[block]
        left_states = tuple(sorted({entry[0] for entry in entries}))
        right_states = tuple(sorted({entry[1] for entry in entries}))
        left_rank = {value: index for index, value in enumerate(left_states)}
        right_rank = {value: index for index, value in enumerate(right_states)}
        amplitude = np.zeros((len(left_states), len(right_states)), dtype=np.complex128)
        for left, right, coefficient in entries:
            amplitude[left_rank[left], right_rank[right]] = coefficient
        singular = np.linalg.svd(amplitude, compute_uv=False)
        block_probabilities = singular**2
        block_probabilities = block_probabilities[
            block_probabilities > low_level.plan.tolerance
        ]
        probabilities.extend(float(value) for value in block_probabilities)
        levels.extend(float(-np.log(value)) for value in block_probabilities)
        particle_labels.extend((block[0],) * block_probabilities.size)
        projection_labels.extend((block[1],) * block_probabilities.size)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    level_array = np.asarray(levels, dtype=np.float64)
    particle_array = np.asarray(particle_labels, dtype=np.int32)
    projection_array = np.asarray(projection_labels, dtype=np.int32)
    order = np.lexsort((level_array, projection_array, particle_array))
    probability_residual = abs(float(np.sum(probability_array)) - 1.0)
    trace_residual = abs(
        float(np.trace(np.asarray(base.one_body_density)).real)
        - prepared.sphere.particle_count
    )
    result_id = canonical_fingerprint(
        {
            "kind": "quantum-hall-sphere-observables",
            "prepared": prepared.prepared_id,
            "orbital_cut": cut,
            "state": array_tree_fingerprint(vector),
        }
    )
    return QuantumHallSphereObservables(
        base.occupations,
        base.one_body_density,
        jnp.asarray(level_array[order]),
        jnp.asarray(particle_array[order]),
        jnp.asarray(projection_array[order]),
        base.orbital_entanglement_entropy,
        base.total_twice_projection,
        base.norm_residual,
        jnp.asarray(trace_residual),
        jnp.asarray(probability_residual),
        result_id,
    )


__all__ = ["QuantumHallSphereObservables", "evaluate_sphere_observables"]
