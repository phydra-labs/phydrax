#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._thermodynamics import (
    AbstractKineticThermodynamicClosure,
    BinaryThermodynamicParameters,
    ThermodynamicForceRepresentation,
)
from ..._trainable import NonTrainableState
from ._interfacial import (
    isotropic_divergence,
    isotropic_gradient,
    isotropic_laplacian,
    natural_wetting_gradient,
)
from ._lattice import LatticeBoltzmannVelocitySet


class BinaryKineticThermodynamicFields(StrictModule):
    """Discrete phase thermodynamics and both force representations."""

    phase: Array
    gradient: Array
    laplacian: Array
    bulk_energy_density: Array
    gradient_energy_density: Array
    chemical_potential: Array
    symmetric_stress: Array
    chemical_force_density: Array
    stress_force_density: Array
    selected_force_density: Array
    force_representation_residual: Array


def isotropic_tensor_divergence(
    tensor: ArrayLike,
    velocity_set: LatticeBoltzmannVelocitySet,
    cell_size: ArrayLike = 1.0,
    /,
) -> Array:
    """Return the divergence over the final tensor index using one lattice stencil."""

    value = jnp.asarray(tensor)
    dimension = velocity_set.dimension
    if value.shape[-2:] != (dimension, dimension):
        raise ValueError("tensor must end in the lattice dimension twice.")
    components = []
    for row in range(dimension):
        divergence = jnp.zeros(value.shape[:-2], dtype=value.dtype)
        for column in range(dimension):
            derivative = isotropic_gradient(
                value[..., row, column], velocity_set, cell_size
            )[..., column]
            divergence = divergence + derivative
        components.append(divergence)
    return jnp.stack(tuple(components), axis=-1)


class PreparedBinaryKineticThermodynamics(StrictModule, NonTrainableState):
    """Binary closure bound to one lattice stencil and force representation."""

    closure: AbstractKineticThermodynamicClosure
    velocity_set: LatticeBoltzmannVelocitySet
    force_representation: ThermodynamicForceRepresentation = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        closure: AbstractKineticThermodynamicClosure,
        velocity_set: LatticeBoltzmannVelocitySet,
        force_representation: ThermodynamicForceRepresentation,
        /,
    ):
        if not isinstance(closure, AbstractKineticThermodynamicClosure):
            raise TypeError("closure must implement AbstractKineticThermodynamicClosure.")
        if not isinstance(velocity_set, LatticeBoltzmannVelocitySet):
            raise TypeError("velocity_set must be LatticeBoltzmannVelocitySet.")
        if not isinstance(force_representation, ThermodynamicForceRepresentation):
            raise TypeError(
                "force_representation must be ThermodynamicForceRepresentation."
            )
        velocity_set.require("fourth-order-isotropy")
        self.closure = closure
        self.velocity_set = velocity_set
        self.force_representation = force_representation
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-binary-kinetic-thermodynamics",
                "closure": closure.closure_id,
                "lattice": velocity_set.lattice_id,
                "force_representation": force_representation.value,
            }
        )

    def evaluate(
        self,
        phase: ArrayLike,
        parameters: BinaryThermodynamicParameters,
        /,
        *,
        wall_normal: ArrayLike | None = None,
        wetting_mask: ArrayLike | None = None,
        cell_size: ArrayLike = 1.0,
    ) -> BinaryKineticThermodynamicFields:
        phi = jnp.asarray(phase)
        if phi.ndim != self.velocity_set.dimension:
            raise ValueError("phase rank must equal the lattice dimension.")
        if (wall_normal is None) != (wetting_mask is None):
            raise ValueError("wall_normal and wetting_mask must be supplied together.")
        gradient = isotropic_gradient(phi, self.velocity_set, cell_size)
        if wall_normal is None:
            laplacian = isotropic_laplacian(phi, self.velocity_set, cell_size)
        elif wetting_mask is not None:
            gradient = natural_wetting_gradient(
                phi,
                gradient,
                wall_normal,
                parameters.wetting_strength,
                parameters.gradient_coefficient,
                wetting_mask,
            )
            laplacian = isotropic_divergence(gradient, self.velocity_set, cell_size)
        else:
            raise RuntimeError("Wetting boundary arguments violated their invariant.")
        local = self.closure.evaluate_local(phi, gradient, laplacian, parameters)
        chemical_force = -phi[..., None] * isotropic_gradient(
            local.chemical_potential,
            self.velocity_set,
            cell_size,
        )
        stress_force = -isotropic_tensor_divergence(
            local.symmetric_stress,
            self.velocity_set,
            cell_size,
        )
        selected = (
            chemical_force
            if self.force_representation
            is ThermodynamicForceRepresentation.CHEMICAL_POTENTIAL_GRADIENT
            else stress_force
        )
        return BinaryKineticThermodynamicFields(
            phi,
            gradient,
            laplacian,
            local.bulk_energy_density,
            local.gradient_energy_density,
            local.chemical_potential,
            local.symmetric_stress,
            chemical_force,
            stress_force,
            selected,
            stress_force - chemical_force,
        )


__all__ = [
    "BinaryKineticThermodynamicFields",
    "PreparedBinaryKineticThermodynamics",
    "isotropic_tensor_divergence",
]
