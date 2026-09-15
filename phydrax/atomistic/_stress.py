#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._strict import StrictModule
from ..discretization import ParticleNeighborhoodState
from ._alchemical import PreparedControlledHamiltonian
from ._potential_program import (
    AtomisticInteractionScaleState,
    PreparedAtomisticPotentialProgram,
)


class AtomisticCellEvaluation(StrictModule):
    energy: Array
    cell_gradient: Array
    stress: Array
    control_derivatives: Array
    successful: Array


def atomistic_cell_energy_and_stress(
    potential: PreparedAtomisticPotentialProgram | PreparedControlledHamiltonian,
    fractional_positions: ArrayLike,
    neighborhood: ParticleNeighborhoodState,
    /,
    *,
    image_counts: ArrayLike | None = None,
    species: ArrayLike | None = None,
    interaction_scales: AtomisticInteractionScaleState | None = None,
    state_index: ArrayLike | None = None,
    control_values: ArrayLike | None = None,
    context_kwargs: dict[str, Any] | None = None,
) -> AtomisticCellEvaluation:
    """Differentiate one scalar in cell strain and optional control coordinates."""

    if not isinstance(
        potential, (PreparedAtomisticPotentialProgram, PreparedControlledHamiltonian)
    ):
        raise TypeError("potential must be a prepared fixed or controlled Hamiltonian.")
    cell = potential.system.cell
    if cell is None:
        raise ValueError("Cell stress requires a periodic atomistic cell.")
    if potential.plan.requirements.directed_graph:
        raise ValueError(
            "Cell derivatives for directed learned graphs require separately qualified support."
        )
    capabilities = (
        potential.potential.plan.capabilities
        if isinstance(potential, PreparedControlledHamiltonian)
        else potential.plan.capabilities
    )
    if not capabilities.cell_derivative:
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
    extra = {} if context_kwargs is None else dict(context_kwargs)

    def geometry(strain):
        deformation = identity + strain
        vectors = contract("ij,kj->ki", deformation, reference_vectors)
        positions = cell.origin.astype(fractional.dtype) + contract(
            "ni,ij->nj", fractional, vectors
        )
        unwrapped = cell.origin.astype(fractional.dtype) + contract(
            "ni,ij->nj", fractional + images.astype(fractional.dtype), vectors
        )
        return vectors, positions, unwrapped

    if isinstance(potential, PreparedControlledHamiltonian):
        if interaction_scales is not None:
            raise ValueError(
                "interaction_scales cannot override a controlled Hamiltonian partition."
            )
        controls, _, state_valid = potential._resolve_controls(
            state_index, control_values, fractional.dtype
        )

        def strained_energy(strain, control):
            vectors, positions, unwrapped = geometry(strain)
            energy, auxiliary = potential.energy(
                positions,
                neighborhood,
                control_values=control,
                unwrapped_positions=unwrapped,
                species=species,
                cell=cell,
                fractional_positions=fractional,
                cell_vectors=vectors,
                **extra,
            )
            return energy, auxiliary[2]

        (energy, successful), (gradient, control_gradient) = jax.value_and_grad(
            strained_energy, argnums=(0, 1), has_aux=True
        )(jnp.zeros((3, 3), dtype=fractional.dtype), controls)
        controls_valid = jnp.all(
            jnp.isfinite(controls) & (controls >= 0.0) & (controls <= 1.0)
        )
        successful = successful & state_valid & controls_valid
    else:
        if state_index is not None or control_values is not None:
            raise ValueError(
                "state_index and control_values require a PreparedControlledHamiltonian."
            )

        def strained_energy(strain):
            vectors, positions, unwrapped = geometry(strain)
            energy, auxiliary = potential.energy(
                positions,
                neighborhood,
                unwrapped_positions=unwrapped,
                species=species,
                interaction_scales=interaction_scales,
                cell=cell,
                fractional_positions=fractional,
                cell_vectors=vectors,
                **extra,
            )
            return energy, auxiliary[2]

        (energy, successful), gradient = jax.value_and_grad(
            strained_energy, has_aux=True
        )(jnp.zeros((3, 3), dtype=fractional.dtype))
        control_gradient = jnp.zeros((0,), dtype=fractional.dtype)
    stress = 0.5 * (gradient + gradient.T) / cell.volume
    finite = (
        jnp.isfinite(energy)
        & jnp.all(jnp.isfinite(stress))
        & jnp.all(jnp.isfinite(control_gradient))
    )
    accepted = successful & finite
    nan = jnp.asarray(jnp.nan, dtype=energy.dtype)
    return AtomisticCellEvaluation(
        energy=jnp.where(accepted, energy, nan),
        cell_gradient=jnp.where(accepted, gradient, nan),
        stress=jnp.where(accepted, stress, nan),
        control_derivatives=jnp.where(accepted, control_gradient, nan),
        successful=accepted,
    )


__all__ = ["AtomisticCellEvaluation", "atomistic_cell_energy_and_stress"]
