#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._strict import StrictModule
from ..discretization import ParticleNeighborhoodState
from ._potential_program import PreparedAtomisticPotentialProgram


class AtomisticCellEvaluation(StrictModule):
    energy: Array
    cell_gradient: Array
    stress: Array
    successful: Array


def atomistic_cell_energy_and_stress(
    potential: PreparedAtomisticPotentialProgram,
    fractional_positions: ArrayLike,
    neighborhood: ParticleNeighborhoodState,
    /,
    *,
    image_counts: ArrayLike | None = None,
    species: ArrayLike | None = None,
    alchemical_lambda: ArrayLike = 1.0,
) -> AtomisticCellEvaluation:
    """Differentiate energy with respect to homogeneous strain at fixed fractions."""

    if not isinstance(potential, PreparedAtomisticPotentialProgram):
        raise TypeError("potential must be PreparedAtomisticPotentialProgram.")
    cell = potential.system.cell
    if cell is None:
        raise ValueError("Cell stress requires a periodic atomistic cell.")
    if potential.plan.requirements.directed_graph:
        raise ValueError(
            "Cell derivatives for directed learned graphs require separately qualified support."
        )
    if not potential.plan.capabilities.cell_derivative:
        raise ValueError(
            "Every potential term must support cell derivatives for stress evaluation."
        )
    fractional = jnp.asarray(
        fractional_positions, dtype=potential.system.plan.coordinate_dtype
    )
    expected = (potential.system.capacity, 3)
    if fractional.shape != expected:
        raise ValueError(f"fractional_positions must have shape {expected}.")
    images = (
        jnp.zeros(expected, dtype=jnp.int32)
        if image_counts is None
        else jnp.asarray(image_counts, dtype=jnp.int32)
    )
    if images.shape != expected:
        raise ValueError(f"image_counts must have shape {expected}.")
    reference_vectors = cell.vectors.astype(fractional.dtype)
    identity = jnp.eye(3, dtype=fractional.dtype)

    def strained_energy(strain):
        deformation = identity + strain
        vectors = contract("ij,kj->ki", deformation, reference_vectors)
        positions = cell.origin.astype(fractional.dtype) + contract(
            "ni,ij->nj", fractional, vectors
        )
        unwrapped = cell.origin.astype(fractional.dtype) + contract(
            "ni,ij->nj", fractional + images.astype(fractional.dtype), vectors
        )
        energy, auxiliary = potential.energy(
            positions,
            neighborhood,
            unwrapped_positions=unwrapped,
            species=species,
            alchemical_lambda=alchemical_lambda,
            cell=cell,
            fractional_positions=fractional,
            cell_vectors=vectors,
        )
        return energy, auxiliary[2]

    (energy, successful), gradient = jax.value_and_grad(strained_energy, has_aux=True)(
        jnp.zeros((3, 3), dtype=fractional.dtype)
    )
    stress = 0.5 * (gradient + gradient.T) / cell.volume
    finite = jnp.isfinite(energy) & jnp.all(jnp.isfinite(stress))
    return AtomisticCellEvaluation(energy, gradient, stress, successful & finite)


__all__ = ["AtomisticCellEvaluation", "atomistic_cell_energy_and_stress"]
