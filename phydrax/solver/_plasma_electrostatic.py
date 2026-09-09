#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..equations._ionized_gas import (
    IonizedMultitemperatureEulerSystem,
    IonizedMultitemperatureNavierStokesSystem,
)
from ._cochain_electrostatic import (
    CochainElectrostaticPlan,
    CochainElectrostaticResult,
)


class ElectrostaticPlasmaCouplingResult(StrictModule):
    electrostatic: CochainElectrostaticResult
    cell_electric_field: Array
    momentum_source: Array
    total_energy_source: Array
    electron_energy_source: Array
    charge_work: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ElectrostaticPlasmaCouplingPlan(StrictModule, NonTrainableState):
    """Project ionized cell charge to a cochain Poisson solve and back."""

    system: IonizedMultitemperatureEulerSystem | IonizedMultitemperatureNavierStokesSystem
    electrostatic: CochainElectrostaticPlan
    cell_to_node: Array
    edge_to_cell_vector: Array
    cell_shape: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: IonizedMultitemperatureEulerSystem
        | IonizedMultitemperatureNavierStokesSystem,
        electrostatic: CochainElectrostaticPlan,
        cell_to_node: ArrayLike,
        edge_to_cell_vector: ArrayLike,
        cell_shape: tuple[int, ...],
        /,
    ):
        if not isinstance(
            system,
            (
                IonizedMultitemperatureEulerSystem,
                IonizedMultitemperatureNavierStokesSystem,
            ),
        ) or not isinstance(electrostatic, CochainElectrostaticPlan):
            raise TypeError(
                "Electrostatic plasma coupling requires ionized gas and cochain plans."
            )
        shape = tuple(int(value) for value in cell_shape)
        cell_count = int(np.prod(shape))
        node_count = electrostatic.bridge.cochain.cell_counts[0]
        edge_count = electrostatic.bridge.cochain.cell_counts[1]
        to_node = np.asarray(cell_to_node, dtype=float)
        to_cell = np.asarray(edge_to_cell_vector, dtype=float)
        if (
            not shape
            or any(value <= 0 for value in shape)
            or to_node.shape != (node_count, cell_count)
            or to_cell.shape != (cell_count * system.dimension, edge_count)
            or np.any(~np.isfinite(to_node))
            or np.any(~np.isfinite(to_cell))
            or not np.allclose(np.sum(to_node, axis=0), 1.0, atol=1.0e-12)
        ):
            raise ValueError("Electrostatic projection matrices are invalid.")
        self.system = system
        self.electrostatic = electrostatic
        self.cell_to_node = jnp.asarray(to_node)
        self.edge_to_cell_vector = jnp.asarray(to_cell)
        self.cell_shape = shape
        self.plan_id = canonical_fingerprint(
            {
                "kind": "electrostatic-plasma-coupling",
                "system": system.system_id,
                "electrostatic": electrostatic.plan_id,
                "cell_to_node": array_tree_fingerprint(self.cell_to_node),
                "edge_to_cell_vector": array_tree_fingerprint(self.edge_to_cell_vector),
                "cell_shape": shape,
            }
        )

    def solve(
        self,
        state: ArrayLike,
        /,
        *,
        current_density: ArrayLike | None = None,
        initial_potential: ArrayLike | None = None,
    ) -> ElectrostaticPlasmaCouplingResult:
        value = jnp.asarray(state)
        expected = self.cell_shape + (self.system.component_count,)
        if value.shape != expected:
            raise ValueError("Plasma state does not match electrostatic cell topology.")
        cell_charge = self.system.charge_density(value).reshape((-1,))
        nodal_charge = contract("nc,c->n", self.cell_to_node, cell_charge, backend="jax")
        electrostatic = self.electrostatic.solve(
            nodal_charge, initial_potential=initial_potential
        )
        cell_electric = contract(
            "de,e->d",
            self.edge_to_cell_vector,
            electrostatic.electric,
            backend="jax",
        ).reshape(self.cell_shape + (self.system.dimension,))
        momentum_source = cell_charge.reshape(self.cell_shape)[..., None] * cell_electric
        if current_density is None:
            current = jnp.zeros_like(cell_electric)
        else:
            current = jnp.asarray(current_density, dtype=value.dtype)
            if current.shape != cell_electric.shape:
                raise ValueError("Current density must match cell electric field.")
        charge_work = contract("...i,...i->...", current, cell_electric, backend="jax")
        total_energy_source = charge_work
        electron_energy_source = charge_work
        finite = (
            electrostatic.finite
            & jnp.all(jnp.isfinite(cell_electric))
            & jnp.all(jnp.isfinite(momentum_source))
            & jnp.all(jnp.isfinite(charge_work))
        )
        return ElectrostaticPlasmaCouplingResult(
            electrostatic,
            cell_electric,
            momentum_source,
            total_energy_source,
            electron_energy_source,
            charge_work,
            finite,
            finite & electrostatic.successful,
            self.plan_id,
        )


__all__ = [
    "ElectrostaticPlasmaCouplingPlan",
    "ElectrostaticPlasmaCouplingResult",
]
