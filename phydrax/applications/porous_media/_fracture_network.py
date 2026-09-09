#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class FractureNetworkState(StrictModule):
    fracture_component_inventory_kg: Array
    intersection_component_inventory_kg: Array
    fracture_energy_J: Array
    intersection_energy_J: Array
    time_s: Array
    network_id: str = eqx.field(static=True)


class FractureNetworkStepResult(StrictModule):
    state: FractureNetworkState
    matrix_component_source_kg_s: Array
    matrix_energy_source_W: Array
    component_balance_kg: Array
    energy_balance_J: Array
    successful: Array


class MixedDimensionalFractureNetworkPlan(StrictModule, NonTrainableState):
    """Resolved 2D fracture cells, 1D/0D intersections, and 3D matrix exchange."""

    fracture_storage_m3: Array
    intersection_storage_m3: Array
    fracture_edges: Array
    fracture_conductance_m3_Pa_s: Array
    intersection_connections: Array
    intersection_conductance_m3_Pa_s: Array
    matrix_parent_cells: Array
    matrix_exchange_conductance_m3_Pa_s: Array
    matrix_cell_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    network_id: str = eqx.field(static=True)

    def __init__(
        self,
        fracture_storage_m3: ArrayLike,
        intersection_storage_m3: ArrayLike,
        fracture_edges: ArrayLike,
        fracture_conductance_m3_Pa_s: ArrayLike,
        intersection_connections: ArrayLike,
        intersection_conductance_m3_Pa_s: ArrayLike,
        matrix_parent_cells: ArrayLike,
        matrix_exchange_conductance_m3_Pa_s: ArrayLike,
        matrix_cell_count: int,
        component_count: int,
        /,
    ):
        fracture_storage = np.asarray(fracture_storage_m3, dtype=float)
        intersection_storage = np.asarray(intersection_storage_m3, dtype=float)
        edges, connections = (
            np.asarray(fracture_edges),
            np.asarray(intersection_connections),
        )
        fracture_conductance = np.asarray(fracture_conductance_m3_Pa_s, dtype=float)
        intersection_conductance = np.asarray(
            intersection_conductance_m3_Pa_s, dtype=float
        )
        parents = np.asarray(matrix_parent_cells)
        matrix_conductance = np.asarray(matrix_exchange_conductance_m3_Pa_s, dtype=float)
        matrix_count, components = int(matrix_cell_count), int(component_count)
        fractures, intersections = fracture_storage.size, intersection_storage.size
        if (
            fracture_storage.ndim != 1
            or fractures == 0
            or intersection_storage.ndim != 1
            or np.any(~np.isfinite(fracture_storage))
            or np.any(fracture_storage <= 0)
            or np.any(~np.isfinite(intersection_storage))
            or np.any(intersection_storage <= 0)
            or edges.ndim != 2
            or edges.shape[1] != 2
            or np.any(edges < 0)
            or np.any(edges >= fractures)
            or np.any(edges[:, 0] == edges[:, 1])
            or fracture_conductance.shape != (edges.shape[0],)
            or np.any(~np.isfinite(fracture_conductance))
            or np.any(fracture_conductance <= 0)
            or connections.ndim != 2
            or connections.shape[1] != 2
            or np.any(connections[:, 0] < 0)
            or np.any(connections[:, 0] >= fractures)
            or np.any(connections[:, 1] < 0)
            or np.any(connections[:, 1] >= intersections)
            or intersection_conductance.shape != (connections.shape[0],)
            or np.any(~np.isfinite(intersection_conductance))
            or np.any(intersection_conductance <= 0)
            or parents.shape != (fractures,)
            or np.any(parents < 0)
            or np.any(parents >= matrix_count)
            or matrix_conductance.shape != (fractures,)
            or np.any(~np.isfinite(matrix_conductance))
            or np.any(matrix_conductance < 0)
            or matrix_count <= 0
            or components <= 0
        ):
            raise ValueError(
                "Mixed-dimensional fracture topology/storage/conductance is invalid."
            )
        self.fracture_storage_m3, self.intersection_storage_m3 = (
            jnp.asarray(fracture_storage),
            jnp.asarray(intersection_storage),
        )
        self.fracture_edges = jnp.asarray(edges, dtype=jnp.int32)
        self.fracture_conductance_m3_Pa_s = jnp.asarray(fracture_conductance)
        self.intersection_connections = jnp.asarray(connections, dtype=jnp.int32)
        self.intersection_conductance_m3_Pa_s = jnp.asarray(intersection_conductance)
        self.matrix_parent_cells = jnp.asarray(parents, dtype=jnp.int32)
        self.matrix_exchange_conductance_m3_Pa_s = jnp.asarray(matrix_conductance)
        self.matrix_cell_count, self.component_count = matrix_count, components
        self.network_id = canonical_fingerprint(
            {
                "kind": "mixed-dimensional-fracture-network",
                "fracture_storage_m3": fracture_storage,
                "intersection_storage_m3": intersection_storage,
                "fracture_edges": edges,
                "fracture_conductance_m3_Pa_s": fracture_conductance,
                "intersection_connections": connections,
                "intersection_conductance_m3_Pa_s": intersection_conductance,
                "matrix_parents": parents,
                "matrix_exchange_conductance_m3_Pa_s": matrix_conductance,
                "matrix_cell_count": matrix_count,
                "component_count": components,
            }
        )

    def initial_state(
        self,
        fracture_component_inventory_kg: ArrayLike,
        intersection_component_inventory_kg: ArrayLike,
        fracture_energy_J: ArrayLike,
        intersection_energy_J: ArrayLike,
        /,
        *,
        time_s: ArrayLike = 0.0,
    ) -> FractureNetworkState:
        fracture_inventory = jnp.asarray(fracture_component_inventory_kg)
        intersection_inventory = jnp.asarray(intersection_component_inventory_kg)
        fracture_energy = jnp.asarray(fracture_energy_J)
        intersection_energy = jnp.asarray(intersection_energy_J)
        time = jnp.asarray(time_s)
        expected_fracture = (
            self.fracture_storage_m3.size,
            self.component_count,
        )
        expected_intersection = (
            self.intersection_storage_m3.size,
            self.component_count,
        )
        if (
            fracture_inventory.shape != expected_fracture
            or intersection_inventory.shape != expected_intersection
            or fracture_energy.shape != (expected_fracture[0],)
            or intersection_energy.shape != (expected_intersection[0],)
            or time.shape != ()
        ):
            raise ValueError("Fracture-network initial state shapes are invalid.")
        fracture_inventory = eqx.error_if(
            fracture_inventory,
            jnp.any(~jnp.isfinite(fracture_inventory))
            | jnp.any(fracture_inventory < 0)
            | jnp.any(~jnp.isfinite(intersection_inventory))
            | jnp.any(intersection_inventory < 0)
            | jnp.any(~jnp.isfinite(fracture_energy))
            | jnp.any(~jnp.isfinite(intersection_energy))
            | ~jnp.isfinite(time),
            "Fracture-network initial inventories, energy, and time must be finite and physical.",
        )
        return FractureNetworkState(
            fracture_inventory,
            intersection_inventory,
            fracture_energy,
            intersection_energy,
            time,
            self.network_id,
        )

    def step(
        self,
        state: FractureNetworkState,
        dt_s: ArrayLike,
        fracture_pressure_Pa: ArrayLike,
        intersection_pressure_Pa: ArrayLike,
        matrix_pressure_Pa: ArrayLike,
        fracture_concentration_kg_m3: ArrayLike,
        intersection_concentration_kg_m3: ArrayLike,
        matrix_concentration_kg_m3: ArrayLike,
        fracture_enthalpy_J_m3: ArrayLike,
        intersection_enthalpy_J_m3: ArrayLike,
        matrix_enthalpy_J_m3: ArrayLike,
        /,
    ) -> FractureNetworkStepResult:
        if (
            not isinstance(state, FractureNetworkState)
            or state.network_id != self.network_id
        ):
            raise ValueError("Fracture-network state belongs to a different plan.")
        dt = jnp.asarray(dt_s)
        fracture_pressure = jnp.asarray(fracture_pressure_Pa)
        intersection_pressure = jnp.asarray(intersection_pressure_Pa)
        matrix_pressure = jnp.asarray(matrix_pressure_Pa)
        fracture_concentration = jnp.asarray(fracture_concentration_kg_m3)
        intersection_concentration = jnp.asarray(intersection_concentration_kg_m3)
        matrix_concentration = jnp.asarray(matrix_concentration_kg_m3)
        fracture_enthalpy = jnp.asarray(fracture_enthalpy_J_m3)
        intersection_enthalpy = jnp.asarray(intersection_enthalpy_J_m3)
        matrix_enthalpy = jnp.asarray(matrix_enthalpy_J_m3)
        fractures, intersections = (
            self.fracture_storage_m3.size,
            self.intersection_storage_m3.size,
        )
        if (
            dt.shape != ()
            or fracture_pressure.shape != (fractures,)
            or intersection_pressure.shape != (intersections,)
            or matrix_pressure.shape != (self.matrix_cell_count,)
            or fracture_concentration.shape != (fractures, self.component_count)
            or intersection_concentration.shape != (intersections, self.component_count)
            or matrix_concentration.shape
            != (self.matrix_cell_count, self.component_count)
            or fracture_enthalpy.shape != (fractures,)
            or intersection_enthalpy.shape != (intersections,)
            or matrix_enthalpy.shape != (self.matrix_cell_count,)
            or state.fracture_component_inventory_kg.shape
            != (fractures, self.component_count)
            or state.intersection_component_inventory_kg.shape
            != (intersections, self.component_count)
            or state.fracture_energy_J.shape != (fractures,)
            or state.intersection_energy_J.shape != (intersections,)
            or state.time_s.shape != ()
        ):
            raise ValueError("Fracture-network state/property shapes are invalid.")
        all_values = (
            fracture_pressure,
            intersection_pressure,
            matrix_pressure,
            fracture_concentration,
            intersection_concentration,
            matrix_concentration,
            fracture_enthalpy,
            intersection_enthalpy,
            matrix_enthalpy,
        )
        dt = eqx.error_if(
            dt,
            ~jnp.isfinite(dt)
            | (dt <= 0)
            | any(jnp.any(~jnp.isfinite(value)) for value in all_values)
            | jnp.any(fracture_concentration < 0)
            | jnp.any(intersection_concentration < 0)
            | jnp.any(matrix_concentration < 0)
            | jnp.any(~jnp.isfinite(state.fracture_component_inventory_kg))
            | jnp.any(state.fracture_component_inventory_kg < 0)
            | jnp.any(~jnp.isfinite(state.intersection_component_inventory_kg))
            | jnp.any(state.intersection_component_inventory_kg < 0)
            | jnp.any(~jnp.isfinite(state.fracture_energy_J))
            | jnp.any(~jnp.isfinite(state.intersection_energy_J))
            | ~jnp.isfinite(state.time_s),
            "Fracture-network timestep/properties must be finite and concentrations nonnegative.",
        )
        left, right = self.fracture_edges[:, 0], self.fracture_edges[:, 1]
        fracture_rate = self.fracture_conductance_m3_Pa_s * (
            fracture_pressure[left] - fracture_pressure[right]
        )
        fracture_upstream = jnp.where(
            (fracture_rate >= 0)[:, None],
            fracture_concentration[left],
            fracture_concentration[right],
        )
        fracture_energy_upstream = jnp.where(
            fracture_rate >= 0, fracture_enthalpy[left], fracture_enthalpy[right]
        )
        fracture_component_flux = fracture_rate[:, None] * fracture_upstream
        fracture_energy_flux = fracture_rate * fracture_energy_upstream
        connection_fracture = self.intersection_connections[:, 0]
        connection_intersection = self.intersection_connections[:, 1]
        intersection_rate = self.intersection_conductance_m3_Pa_s * (
            fracture_pressure[connection_fracture]
            - intersection_pressure[connection_intersection]
        )
        connection_concentration = jnp.where(
            (intersection_rate >= 0)[:, None],
            fracture_concentration[connection_fracture],
            intersection_concentration[connection_intersection],
        )
        connection_enthalpy = jnp.where(
            intersection_rate >= 0,
            fracture_enthalpy[connection_fracture],
            intersection_enthalpy[connection_intersection],
        )
        connection_component_flux = intersection_rate[:, None] * connection_concentration
        connection_energy_flux = intersection_rate * connection_enthalpy
        matrix_rate = self.matrix_exchange_conductance_m3_Pa_s * (
            matrix_pressure[self.matrix_parent_cells] - fracture_pressure
        )
        matrix_upstream = jnp.where(
            (matrix_rate >= 0)[:, None],
            matrix_concentration[self.matrix_parent_cells],
            fracture_concentration,
        )
        matrix_energy_upstream = jnp.where(
            matrix_rate >= 0,
            matrix_enthalpy[self.matrix_parent_cells],
            fracture_enthalpy,
        )
        matrix_component_flux = matrix_rate[:, None] * matrix_upstream
        matrix_energy_flux = matrix_rate * matrix_energy_upstream
        fracture_component_rate = jnp.zeros_like(fracture_concentration)
        fracture_component_rate = fracture_component_rate.at[left].add(
            -fracture_component_flux
        )
        fracture_component_rate = fracture_component_rate.at[right].add(
            fracture_component_flux
        )
        fracture_component_rate = fracture_component_rate.at[connection_fracture].add(
            -connection_component_flux
        )
        fracture_component_rate = fracture_component_rate + matrix_component_flux
        intersection_component_rate = (
            jnp.zeros_like(intersection_concentration)
            .at[connection_intersection]
            .add(connection_component_flux)
        )
        fracture_energy_rate = jnp.zeros_like(fracture_enthalpy)
        fracture_energy_rate = fracture_energy_rate.at[left].add(-fracture_energy_flux)
        fracture_energy_rate = fracture_energy_rate.at[right].add(fracture_energy_flux)
        fracture_energy_rate = (
            fracture_energy_rate.at[connection_fracture].add(-connection_energy_flux)
            + matrix_energy_flux
        )
        intersection_energy_rate = (
            jnp.zeros_like(intersection_enthalpy)
            .at[connection_intersection]
            .add(connection_energy_flux)
        )
        candidate = FractureNetworkState(
            state.fracture_component_inventory_kg + dt * fracture_component_rate,
            state.intersection_component_inventory_kg + dt * intersection_component_rate,
            state.fracture_energy_J + dt * fracture_energy_rate,
            state.intersection_energy_J + dt * intersection_energy_rate,
            state.time_s + dt,
            self.network_id,
        )
        matrix_component_source = (
            jnp.zeros((self.matrix_cell_count, self.component_count))
            .at[self.matrix_parent_cells]
            .add(-matrix_component_flux)
        )
        matrix_energy_source = (
            jnp.zeros((self.matrix_cell_count,))
            .at[self.matrix_parent_cells]
            .add(-matrix_energy_flux)
        )
        component_balance = (
            jnp.sum(
                candidate.fracture_component_inventory_kg
                - state.fracture_component_inventory_kg,
                axis=0,
            )
            + jnp.sum(
                candidate.intersection_component_inventory_kg
                - state.intersection_component_inventory_kg,
                axis=0,
            )
            + dt * jnp.sum(matrix_component_source, axis=0)
        )
        energy_balance = (
            jnp.sum(candidate.fracture_energy_J - state.fracture_energy_J)
            + jnp.sum(candidate.intersection_energy_J - state.intersection_energy_J)
            + dt * jnp.sum(matrix_energy_source)
        )
        successful = (
            jnp.all(jnp.isfinite(candidate.fracture_component_inventory_kg))
            & jnp.all(candidate.fracture_component_inventory_kg >= 0)
            & jnp.all(candidate.intersection_component_inventory_kg >= 0)
            & jnp.all(jnp.isfinite(candidate.fracture_energy_J))
            & jnp.all(jnp.isfinite(candidate.intersection_energy_J))
        )
        committed = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old), candidate, state
        )
        return FractureNetworkStepResult(
            committed,
            matrix_component_source,
            matrix_energy_source,
            component_balance,
            energy_balance,
            successful,
        )


__all__ = [
    "FractureNetworkState",
    "FractureNetworkStepResult",
    "MixedDimensionalFractureNetworkPlan",
]
