#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Stationary periodic force/stress derivatives with complete named ledgers."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import PeriodicCell
from ...units import derived_unit, ENERGY, LENGTH, UnitDefinition


StationaryEnergyKind = Literal["total-energy", "free-energy"]
StationaryDerivativeRole = Literal[
    "hellmann-feynman", "pulay", "entropy", "nonlocal", "ionic"
]
PeriodicStationaryEnergyFunction = Callable[[Array, Array], Array]
_REQUIRED_ROLES = frozenset(("hellmann-feynman", "pulay", "entropy", "nonlocal", "ionic"))


class PeriodicStationaryEnergyComponent(StrictModule, NonTrainableState):
    """One explicit stationary functional term and its reproducible definition."""

    energy_function: PeriodicStationaryEnergyFunction = eqx.field(static=True)
    name: str = eqx.field(static=True)
    role: StationaryDerivativeRole = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)
    component_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        role: StationaryDerivativeRole,
        energy_function: PeriodicStationaryEnergyFunction,
        definition_id: str,
        /,
    ):
        name_ = str(name).strip()
        definition = str(definition_id).strip()
        if not name_ or role not in _REQUIRED_ROLES or not callable(energy_function):
            raise ValueError(
                "Stationary energy component name, role, or callable is invalid."
            )
        if not definition:
            raise ValueError(
                "Stationary energy component definition_id must be non-empty."
            )
        self.energy_function = energy_function
        self.name = name_
        self.role = role
        self.definition_id = definition
        self.component_id = canonical_fingerprint(
            {
                "kind": "periodic-stationary-energy-component",
                "name": name_,
                "role": role,
                "definition": definition,
            }
        )


class PeriodicDerivativeLedger(StrictModule, NonTrainableState):
    """Per-term energy, force, and cell-stress contributions with exact closure."""

    component_energies: Array
    component_forces: Array
    component_stresses: Array
    total_energy: Array
    total_forces: Array
    total_stress: Array
    energy_closure_residual: Array
    force_closure_residual: Array
    stress_closure_residual: Array
    component_names: tuple[str, ...] = eqx.field(static=True)
    component_roles: tuple[StationaryDerivativeRole, ...] = eqx.field(static=True)
    component_ids: tuple[str, ...] = eqx.field(static=True)
    energy_unit: UnitDefinition
    force_unit: UnitDefinition
    stress_unit: UnitDefinition
    ledger_id: str = eqx.field(static=True)

    def __init__(
        self,
        components: tuple[PeriodicStationaryEnergyComponent, ...],
        component_energies: ArrayLike,
        component_forces: ArrayLike,
        component_stresses: ArrayLike,
        total_energy: ArrayLike,
        total_forces: ArrayLike,
        total_stress: ArrayLike,
        energy_unit: UnitDefinition,
        length_unit: UnitDefinition,
        /,
    ):
        rows = tuple(components)
        names = tuple(value.name for value in rows)
        roles = tuple(value.role for value in rows)
        identifiers = tuple(value.component_id for value in rows)
        energies = jnp.asarray(component_energies)
        forces = jnp.asarray(component_forces, dtype=energies.dtype)
        stresses = jnp.asarray(component_stresses, dtype=energies.dtype)
        energy = jnp.asarray(total_energy, dtype=energies.dtype).reshape(())
        force = jnp.asarray(total_forces, dtype=energies.dtype)
        stress = jnp.asarray(total_stress, dtype=energies.dtype)
        count = len(rows)
        if (
            count == 0
            or any(
                not isinstance(value, PeriodicStationaryEnergyComponent) for value in rows
            )
            or len(set(names)) != count
            or len(set(roles)) != count
            or set(roles) != _REQUIRED_ROLES
            or energies.shape != (count,)
            or forces.ndim != 3
            or forces.shape[0] != count
            or forces.shape[2] != 3
            or stresses.shape != (count, 3, 3)
            or force.shape != forces.shape[1:]
            or stress.shape != (3, 3)
        ):
            raise ValueError(
                "Periodic derivative ledger components or tensors do not align."
            )
        if not isinstance(energy_unit, UnitDefinition) or energy_unit.dimension != ENERGY:
            raise TypeError("Derivative ledger energy_unit must have energy dimension.")
        if not isinstance(length_unit, UnitDefinition) or length_unit.dimension != LENGTH:
            raise TypeError("Derivative ledger length_unit must have length dimension.")
        if energy_unit.reference_system_id != length_unit.reference_system_id:
            raise ValueError(
                "Derivative energy and length units must share a reference system."
            )
        force_unit = derived_unit(
            f"{energy_unit.symbol}/{length_unit.symbol}",
            ((energy_unit, 1), (length_unit, -1)),
        )
        stress_unit = derived_unit(
            f"{energy_unit.symbol}/{length_unit.symbol}^3",
            ((energy_unit, 1), (length_unit, -3)),
        )
        energy_residual = jnp.abs(jnp.sum(energies) - energy)
        force_residual = jnp.max(jnp.abs(jnp.sum(forces, axis=0) - force), initial=0.0)
        stress_residual = jnp.max(
            jnp.abs(jnp.sum(stresses, axis=0) - stress), initial=0.0
        )
        self.component_energies = energies
        self.component_forces = forces
        self.component_stresses = stresses
        self.total_energy = energy
        self.total_forces = force
        self.total_stress = stress
        self.energy_closure_residual = energy_residual
        self.force_closure_residual = force_residual
        self.stress_closure_residual = stress_residual
        self.component_names = names
        self.component_roles = roles
        self.component_ids = identifiers
        self.energy_unit = energy_unit
        self.force_unit = force_unit
        self.stress_unit = stress_unit
        self.ledger_id = canonical_fingerprint(
            {
                "kind": "periodic-derivative-ledger",
                "components": list(identifiers),
                "names": list(names),
                "roles": list(roles),
                "energy_unit": energy_unit.unit_id,
                "force_unit": force_unit.unit_id,
                "stress_unit": stress_unit.unit_id,
                "arrays": array_tree_fingerprint(
                    {
                        "component_energies": np.asarray(energies),
                        "component_forces": np.asarray(forces),
                        "component_stresses": np.asarray(stresses),
                        "total_energy": np.asarray(energy),
                        "total_forces": np.asarray(force),
                        "total_stress": np.asarray(stress),
                        "closure": np.asarray(
                            (energy_residual, force_residual, stress_residual)
                        ),
                    }
                ),
            }
        )


class PeriodicStationaryDerivativeEvidence(StrictModule, NonTrainableState):
    stationarity_residual: Array
    force_directional_residual: Array
    stress_directional_residual: Array
    successful: Array
    stationarity_tolerance: float = eqx.field(static=True)
    directional_tolerance: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        stationarity_residual: ArrayLike,
        force_directional_residual: ArrayLike,
        stress_directional_residual: ArrayLike,
        successful: ArrayLike,
        stationarity_tolerance: float,
        directional_tolerance: float,
        /,
    ):
        values = jnp.asarray(
            (
                stationarity_residual,
                force_directional_residual,
                stress_directional_residual,
            )
        ).reshape((3,))
        stationarity = float(stationarity_tolerance)
        directional = float(directional_tolerance)
        if (
            not isfinite(stationarity)
            or stationarity < 0.0
            or not isfinite(directional)
            or directional <= 0.0
        ):
            raise ValueError("Stationary derivative evidence tolerances are invalid.")
        admitted = (
            jnp.asarray(successful, dtype=bool).reshape(())
            & jnp.all(jnp.isfinite(values))
            & jnp.all(values >= 0.0)
            & (values[0] <= stationarity)
            & (values[1] <= directional)
            & (values[2] <= directional)
        )
        self.stationarity_residual = values[0]
        self.force_directional_residual = values[1]
        self.stress_directional_residual = values[2]
        self.successful = admitted
        self.stationarity_tolerance = stationarity
        self.directional_tolerance = directional
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "periodic-stationary-derivative-evidence",
                "stationarity_tolerance": stationarity.hex(),
                "directional_tolerance": directional.hex(),
                "successful": bool(admitted),
                "residuals": array_tree_fingerprint(np.asarray(values)),
            }
        )


class PeriodicStationaryDerivativeResult(StrictModule, NonTrainableState):
    energy: Array
    forces: Array
    stress: Array
    ledger: PeriodicDerivativeLedger
    evidence: PeriodicStationaryDerivativeEvidence
    successful: Array
    energy_kind: StationaryEnergyKind = eqx.field(static=True)
    cell_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        energy: ArrayLike,
        forces: ArrayLike,
        stress: ArrayLike,
        ledger: PeriodicDerivativeLedger,
        evidence: PeriodicStationaryDerivativeEvidence,
        energy_kind: StationaryEnergyKind,
        cell_id: str,
        plan_id: str,
        /,
    ):
        if not isinstance(ledger, PeriodicDerivativeLedger):
            raise TypeError("ledger must be PeriodicDerivativeLedger.")
        if not isinstance(evidence, PeriodicStationaryDerivativeEvidence):
            raise TypeError("evidence must be PeriodicStationaryDerivativeEvidence.")
        if energy_kind not in ("total-energy", "free-energy"):
            raise ValueError("energy_kind must be total-energy or free-energy.")
        energy_ = jnp.asarray(energy).reshape(())
        force = jnp.asarray(forces, dtype=energy_.dtype)
        stress_ = jnp.asarray(stress, dtype=energy_.dtype)
        if force.ndim != 2 or force.shape[1] != 3 or stress_.shape != (3, 3):
            raise ValueError("Stationary derivative result tensor shapes are invalid.")
        identifiers = tuple(str(value).strip() for value in (cell_id, plan_id))
        if any(not value for value in identifiers):
            raise ValueError("Stationary derivative identities must be non-empty.")
        finite = (
            jnp.isfinite(energy_)
            & jnp.all(jnp.isfinite(force))
            & jnp.all(jnp.isfinite(stress_))
        )
        self.energy = energy_
        self.forces = force
        self.stress = stress_
        self.ledger = ledger
        self.evidence = evidence
        self.successful = evidence.successful & finite
        self.energy_kind = energy_kind
        self.cell_id, self.plan_id = identifiers
        self.result_id = canonical_fingerprint(
            {
                "kind": "periodic-stationary-derivative-result",
                "energy_kind": energy_kind,
                "cell": self.cell_id,
                "plan": self.plan_id,
                "ledger": ledger.ledger_id,
                "evidence": evidence.evidence_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "energy": np.asarray(energy_),
                        "forces": np.asarray(force),
                        "stress": np.asarray(stress_),
                    }
                ),
            }
        )


class PeriodicStationaryDerivativePlan(StrictModule, NonTrainableState):
    """Differentiate a complete stationary total/free-energy ledger and verify directions."""

    cell: PeriodicCell
    components: tuple[PeriodicStationaryEnergyComponent, ...]
    energy_unit: UnitDefinition
    length_unit: UnitDefinition
    energy_kind: StationaryEnergyKind = eqx.field(static=True)
    stationarity_tolerance: float = eqx.field(static=True)
    directional_step: float = eqx.field(static=True)
    directional_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell: PeriodicCell,
        components: tuple[PeriodicStationaryEnergyComponent, ...],
        energy_unit: UnitDefinition,
        length_unit: UnitDefinition,
        /,
        *,
        energy_kind: StationaryEnergyKind,
        stationarity_tolerance: float = 1.0e-9,
        directional_step: float = 1.0e-5,
        directional_tolerance: float = 1.0e-6,
    ):
        rows = tuple(components)
        if (
            not isinstance(cell, PeriodicCell)
            or cell.rank != 3
            or not cell.fully_periodic
        ):
            raise TypeError(
                "Stationary periodic derivatives require a fully periodic 3D cell."
            )
        if (
            len(rows) != len(_REQUIRED_ROLES)
            or any(
                not isinstance(value, PeriodicStationaryEnergyComponent) for value in rows
            )
            or {value.role for value in rows} != _REQUIRED_ROLES
            or len({value.name for value in rows}) != len(rows)
        ):
            raise ValueError(
                "Stationary derivatives require exactly one Hellmann--Feynman, Pulay, "
                "entropy, nonlocal, and ionic component."
            )
        if not isinstance(energy_unit, UnitDefinition) or energy_unit.dimension != ENERGY:
            raise TypeError(
                "Stationary derivative energy_unit must have energy dimension."
            )
        if not isinstance(length_unit, UnitDefinition) or length_unit.dimension != LENGTH:
            raise TypeError(
                "Stationary derivative length_unit must have length dimension."
            )
        if energy_unit.reference_system_id != length_unit.reference_system_id:
            raise ValueError("Stationary derivative units must share a reference system.")
        if energy_kind not in ("total-energy", "free-energy"):
            raise ValueError(
                "energy_kind must explicitly select total-energy or free-energy."
            )
        stationarity = float(stationarity_tolerance)
        step = float(directional_step)
        directional = float(directional_tolerance)
        if (
            not isfinite(stationarity)
            or stationarity < 0.0
            or not isfinite(step)
            or step <= 0.0
            or not isfinite(directional)
            or directional <= 0.0
        ):
            raise ValueError("Stationary derivative policy is invalid.")
        self.cell = cell
        self.components = rows
        self.energy_unit = energy_unit
        self.length_unit = length_unit
        self.energy_kind = energy_kind
        self.stationarity_tolerance = stationarity
        self.directional_step = step
        self.directional_tolerance = directional
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-stationary-derivative-plan",
                "cell": cell.cell_id,
                "components": [value.component_id for value in rows],
                "energy_unit": energy_unit.unit_id,
                "length_unit": length_unit.unit_id,
                "energy_kind": energy_kind,
                "stationarity_tolerance": stationarity.hex(),
                "directional_step": step.hex(),
                "directional_tolerance": directional.hex(),
            }
        )

    def evaluate(
        self,
        positions: ArrayLike,
        stationarity_residual: ArrayLike,
        position_direction: ArrayLike,
        strain_direction: ArrayLike,
        /,
        *,
        cell_vectors: ArrayLike | None = None,
    ) -> PeriodicStationaryDerivativeResult:
        coordinate = jnp.asarray(positions)
        cell = (
            jnp.asarray(self.cell.vectors, dtype=coordinate.dtype)
            if cell_vectors is None
            else jnp.asarray(cell_vectors, dtype=coordinate.dtype)
        )
        position_direction_ = jnp.asarray(position_direction, dtype=coordinate.dtype)
        strain_direction_ = jnp.asarray(strain_direction, dtype=coordinate.dtype)
        stationarity = jnp.asarray(stationarity_residual, dtype=coordinate.dtype).reshape(
            ()
        )
        if (
            coordinate.ndim != 2
            or coordinate.shape[1] != 3
            or cell.shape != (3, 3)
            or position_direction_.shape != coordinate.shape
            or strain_direction_.shape != (3, 3)
        ):
            raise ValueError(
                "Stationary derivative geometry or directions have invalid shapes."
            )
        arrays = (coordinate, cell, position_direction_, strain_direction_, stationarity)
        if any(not bool(jnp.all(jnp.isfinite(value))) for value in arrays):
            raise ValueError("Stationary derivative inputs must be finite.")
        if float(stationarity) < 0.0:
            raise ValueError("stationarity_residual must be non-negative.")
        position_norm = jnp.sqrt(jnp.sum(position_direction_**2))
        strain_symmetric = 0.5 * (strain_direction_ + strain_direction_.T)
        strain_norm = jnp.sqrt(jnp.sum(strain_symmetric**2))
        if float(position_norm) == 0.0 or float(strain_norm) == 0.0:
            raise ValueError("Directional verification vectors must be nonzero.")
        position_direction_ = position_direction_ / position_norm
        strain_symmetric = strain_symmetric / strain_norm

        def total_energy(current_positions, current_cell):
            value = jnp.asarray(0.0, dtype=current_positions.dtype)
            for component in self.components:
                value = value + jnp.asarray(
                    component.energy_function(current_positions, current_cell)
                ).reshape(())
            return value

        energy, position_gradient = jax.value_and_grad(total_energy, argnums=0)(
            coordinate, cell
        )

        def strained_total(strain):
            deformation = jnp.eye(3, dtype=cell.dtype) + strain
            return total_energy(
                coordinate @ deformation.T,
                cell @ deformation.T,
            )

        volume = jnp.abs(jnp.linalg.det(cell))
        strain_gradient = jax.grad(strained_total)(jnp.zeros_like(cell))
        forces = -position_gradient
        stress = 0.5 * (strain_gradient + strain_gradient.T) / volume
        component_energies = []
        component_forces = []
        component_stresses = []
        for component in self.components:
            component_energy, component_gradient = jax.value_and_grad(
                component.energy_function, argnums=0
            )(coordinate, cell)

            def strained_component(strain, component=component):
                deformation = jnp.eye(3, dtype=cell.dtype) + strain
                return component.energy_function(
                    coordinate @ deformation.T,
                    cell @ deformation.T,
                )

            component_strain_gradient = jax.grad(strained_component)(jnp.zeros_like(cell))
            component_energies.append(jnp.asarray(component_energy).reshape(()))
            component_forces.append(-component_gradient)
            component_stresses.append(
                0.5 * (component_strain_gradient + component_strain_gradient.T) / volume
            )
        ledger = PeriodicDerivativeLedger(
            self.components,
            jnp.stack(tuple(component_energies)),
            jnp.stack(tuple(component_forces)),
            jnp.stack(tuple(component_stresses)),
            energy,
            forces,
            stress,
            self.energy_unit,
            self.length_unit,
        )
        step = self.directional_step
        finite_position = (
            total_energy(coordinate + step * position_direction_, cell)
            - total_energy(coordinate - step * position_direction_, cell)
        ) / (2.0 * step)
        predicted_position = -jnp.sum(forces * position_direction_)
        force_directional_residual = jnp.abs(
            finite_position - predicted_position
        ) / jnp.maximum(
            jnp.maximum(jnp.abs(finite_position), jnp.abs(predicted_position)), 1.0
        )

        def deformed_energy(scale):
            deformation = jnp.eye(3, dtype=cell.dtype) + scale * strain_symmetric
            return total_energy(
                coordinate @ deformation.T,
                cell @ deformation.T,
            )

        finite_strain = (deformed_energy(step) - deformed_energy(-step)) / (2.0 * step)
        predicted_strain = volume * jnp.sum(stress * strain_symmetric)
        stress_directional_residual = jnp.abs(
            finite_strain - predicted_strain
        ) / jnp.maximum(
            jnp.maximum(jnp.abs(finite_strain), jnp.abs(predicted_strain)), 1.0
        )
        closure = jnp.maximum(
            ledger.energy_closure_residual,
            jnp.maximum(ledger.force_closure_residual, ledger.stress_closure_residual),
        )
        evidence = PeriodicStationaryDerivativeEvidence(
            stationarity,
            force_directional_residual,
            stress_directional_residual,
            closure <= self.directional_tolerance,
            self.stationarity_tolerance,
            self.directional_tolerance,
        )
        return PeriodicStationaryDerivativeResult(
            energy,
            forces,
            stress,
            ledger,
            evidence,
            self.energy_kind,
            self.cell.cell_id,
            self.plan_id,
        )


__all__ = [
    "PeriodicDerivativeLedger",
    "PeriodicStationaryDerivativeEvidence",
    "PeriodicStationaryDerivativePlan",
    "PeriodicStationaryDerivativeResult",
    "PeriodicStationaryEnergyComponent",
    "PeriodicStationaryEnergyFunction",
    "StationaryDerivativeRole",
    "StationaryEnergyKind",
]
