#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._strict import StrictModule
from ..discretization import ParticleImageNeighborhoodState, ParticleNeighborhoodState
from ._alchemical import PreparedControlledHamiltonian
from ._potential import AtomisticStressConvention
from ._potential_program import (
    AtomisticInteractionScaleState,
    PreparedAtomisticPotentialProgram,
)


class AtomisticCellEvaluation(StrictModule):
    """Fixed-fractional energy, strain derivative, stress and control derivatives.

    ``cell_gradient`` is ``dE/dstrain`` for the column deformation
    ``F = I + strain`` (row vectors ``H @ F.T``) and ``stress`` follows
    ``stress_convention``; both come from one scalar derivative pass at fixed
    topology and integer images. A failed state is NaN with ``successful`` false.
    """

    energy: Array
    cell_gradient: Array
    stress: Array
    control_derivatives: Array
    successful: Array
    stress_convention: AtomisticStressConvention = eqx.field(static=True)


def atomistic_cell_energy_and_stress(
    potential: PreparedAtomisticPotentialProgram | PreparedControlledHamiltonian,
    fractional_positions: ArrayLike,
    neighborhood: ParticleNeighborhoodState | ParticleImageNeighborhoodState,
    /,
    *,
    image_counts: ArrayLike | None = None,
    cell_vectors: ArrayLike | None = None,
    species: ArrayLike | None = None,
    interaction_scales: AtomisticInteractionScaleState | None = None,
    state_index: ArrayLike | None = None,
    control_values: ArrayLike | None = None,
    context_kwargs: dict[str, Any] | None = None,
) -> AtomisticCellEvaluation:
    """Differentiate one scalar in cell strain and optional control coordinates.

    Positions are ``origin + s @ H`` for the fixed fractional coordinates ``s``
    and the runtime row lattice ``H`` (``cell_vectors``, default the prepared
    cell); ``image_counts`` fix the unwrapped representation. Every term must own
    a cell derivative and the cell must be a full 3x3 lattice; a partially
    periodic cell reports stress per declared embedding volume.
    """

    if not isinstance(
        potential, (PreparedAtomisticPotentialProgram, PreparedControlledHamiltonian)
    ):
        raise TypeError("potential must be a prepared fixed or controlled Hamiltonian.")
    cell = potential.system.cell
    if cell is None:
        raise ValueError("Cell stress requires a periodic atomistic cell.")
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
    vectors = (
        cell.vectors.astype(fractional.dtype)
        if cell_vectors is None
        else jnp.asarray(cell_vectors, dtype=fractional.dtype)
    )
    if vectors.shape != cell.vectors.shape:
        raise ValueError(f"cell_vectors must have shape {cell.vectors.shape}.")
    positions = cell.cartesian_with_vectors(fractional, vectors)
    kwargs: dict[str, Any] = {
        "unwrapped_positions": cell.cartesian_with_vectors(
            fractional + images.astype(fractional.dtype), vectors
        ),
        "species": species,
        "cell": cell,
        "fractional_positions": fractional,
        "cell_vectors": vectors,
    }
    extra = {} if context_kwargs is None else dict(context_kwargs)
    if set(extra) & (set(kwargs) | {"compute_stress", "interaction_scales"}):
        raise ValueError("context_kwargs cannot override the strained cell binding.")
    kwargs.update(extra)
    if isinstance(potential, PreparedControlledHamiltonian):
        if interaction_scales is not None:
            raise ValueError(
                "interaction_scales cannot override a controlled Hamiltonian partition."
            )
        controlled = potential.evaluate(
            positions,
            neighborhood,
            compute_stress=True,
            state_index=state_index,
            control_values=control_values,
            **kwargs,
        )
        evaluation_energy = controlled.energy
        strain_derivative = controlled.strain_derivative
        stress = controlled.stress
        control_derivatives = controlled.dU_dcontrols
        successful = controlled.successful
        convention = controlled.stress_convention
    else:
        if state_index is not None or control_values is not None:
            raise ValueError(
                "state_index and control_values require a PreparedControlledHamiltonian."
            )
        evaluation = potential.evaluate(
            positions,
            neighborhood,
            compute_stress=True,
            interaction_scales=interaction_scales,
            **kwargs,
        )
        evaluation_energy = evaluation.energy
        strain_derivative = evaluation.strain_derivative
        stress = evaluation.stress
        control_derivatives = jnp.zeros((0,), dtype=fractional.dtype)
        successful = evaluation.successful
        convention = evaluation.stress_convention
    if strain_derivative is None or stress is None or convention is None:
        raise RuntimeError("Requested stress evaluation returned no strain response.")
    return AtomisticCellEvaluation(
        energy=evaluation_energy,
        cell_gradient=strain_derivative,
        stress=stress,
        control_derivatives=control_derivatives,
        successful=successful,
        stress_convention=convention,
    )


__all__ = ["AtomisticCellEvaluation", "atomistic_cell_energy_and_stress"]
