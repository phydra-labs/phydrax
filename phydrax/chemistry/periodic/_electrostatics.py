#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared three-dimensional Ewald sums and governed GTH components."""

from __future__ import annotations

from math import isfinite
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import AtomisticUnitSystem
from ...discretization import PeriodicCell
from ...ein import contract
from ...units import derived_unit, ENERGY, LENGTH, UnitDefinition
from .._result import ElectronicEnergyLedger
from ._source import PeriodicProvenanceManifest


EwaldNeutralityKind = Literal["require-neutral", "uniform-background"]


class PeriodicEwaldEvidence(StrictModule, NonTrainableState):
    """Neutrality, component closure, and conservative tensor evidence."""

    net_charge: Array
    charge_residual: Array
    minimum_pair_distance: Array
    energy_closure_residual: Array
    force_balance_residual: Array
    stress_symmetry_residual: Array
    background_applied: Array
    successful: Array
    neutrality: EwaldNeutralityKind = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        net_charge: ArrayLike,
        minimum_pair_distance: ArrayLike,
        energy_closure_residual: ArrayLike,
        force_balance_residual: ArrayLike,
        stress_symmetry_residual: ArrayLike,
        background_applied: ArrayLike,
        successful: ArrayLike,
        neutrality: EwaldNeutralityKind,
        residual_tolerance: float,
        /,
    ):
        if neutrality not in ("require-neutral", "uniform-background"):
            raise ValueError("Unknown Ewald neutrality policy.")
        tolerance = float(residual_tolerance)
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError(
                "Ewald evidence residual_tolerance must be finite and positive."
            )
        net = jnp.asarray(net_charge).reshape(())
        distance = jnp.asarray(minimum_pair_distance, dtype=net.dtype).reshape(())
        residuals = jnp.asarray(
            (energy_closure_residual, force_balance_residual, stress_symmetry_residual),
            dtype=net.dtype,
        ).reshape((3,))
        background = jnp.asarray(background_applied, dtype=jnp.bool_).reshape(())
        finite = (
            jnp.isfinite(net)
            & jnp.isfinite(distance)
            & jnp.all(jnp.isfinite(residuals))
            & (distance > 0.0)
            & jnp.all(residuals >= 0.0)
        )
        admitted = (
            jnp.asarray(successful, dtype=jnp.bool_).reshape(())
            & finite
            & jnp.all(residuals <= tolerance)
        )
        self.net_charge = net
        self.charge_residual = jnp.abs(net)
        self.minimum_pair_distance = distance
        self.energy_closure_residual = residuals[0]
        self.force_balance_residual = residuals[1]
        self.stress_symmetry_residual = residuals[2]
        self.background_applied = background
        self.successful = admitted
        self.neutrality = neutrality
        self.residual_tolerance = tolerance
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "periodic-ewald-evidence",
                "neutrality": neutrality,
                "background_applied": bool(background),
                "residual_tolerance": tolerance.hex(),
                "successful": bool(admitted),
                "arrays": array_tree_fingerprint(
                    {
                        "net_charge": np.asarray(net),
                        "minimum_pair_distance": np.asarray(distance),
                        "residuals": np.asarray(residuals),
                    }
                ),
            }
        )


class PeriodicEwaldResult(StrictModule, NonTrainableState):
    energy: Array
    forces: Array
    stress: Array
    energy_ledger: ElectronicEnergyLedger
    evidence: PeriodicEwaldEvidence
    successful: Array
    energy_unit: UnitDefinition
    force_unit: UnitDefinition
    stress_unit: UnitDefinition
    charge_unit: UnitDefinition
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    cell_id: str = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        energy: ArrayLike,
        forces: ArrayLike,
        stress: ArrayLike,
        energy_ledger: ElectronicEnergyLedger,
        evidence: PeriodicEwaldEvidence,
        units: AtomisticUnitSystem,
        /,
        *,
        plan_id: str,
        prepared_id: str,
        cell_id: str,
    ):
        energy_ = jnp.asarray(energy).reshape(())
        force = jnp.asarray(forces, dtype=energy_.dtype)
        stress_ = jnp.asarray(stress, dtype=energy_.dtype)
        if force.ndim != 2 or force.shape[1] != 3 or stress_.shape != (3, 3):
            raise ValueError("Ewald forces and stress have invalid shapes.")
        if not isinstance(energy_ledger, ElectronicEnergyLedger):
            raise TypeError("energy_ledger must be ElectronicEnergyLedger.")
        if not isinstance(evidence, PeriodicEwaldEvidence):
            raise TypeError("evidence must be PeriodicEwaldEvidence.")
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("units must be AtomisticUnitSystem.")
        if energy_ledger.energy_unit != units.scale.energy_unit:
            raise ValueError("Ewald ledger energy unit differs from the unit system.")
        identifiers = tuple(
            str(value).strip() for value in (plan_id, prepared_id, cell_id)
        )
        if any(not value for value in identifiers):
            raise ValueError("Ewald result identities must be non-empty.")
        finite = (
            jnp.isfinite(energy_)
            & jnp.all(jnp.isfinite(force))
            & jnp.all(jnp.isfinite(stress_))
        )
        self.energy = energy_
        self.forces = force
        self.stress = stress_
        self.energy_ledger = energy_ledger
        self.evidence = evidence
        self.successful = evidence.successful & finite
        self.energy_unit = units.scale.energy_unit
        self.force_unit = units.scale.force_unit
        self.stress_unit = units.pressure_unit
        self.charge_unit = units.charge_unit
        self.plan_id, self.prepared_id, self.cell_id = identifiers
        self.unit_system_id = units.unit_system_id
        self.result_id = canonical_fingerprint(
            {
                "kind": "periodic-ewald-result",
                "plan": self.plan_id,
                "prepared": self.prepared_id,
                "cell": self.cell_id,
                "unit_system": units.unit_system_id,
                "ledger": energy_ledger.ledger_id,
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


class PeriodicEwaldPlan(StrictModule, NonTrainableState):
    cell: PeriodicCell
    units: AtomisticUnitSystem
    alpha: float = eqx.field(static=True)
    real_shell: int = eqx.field(static=True)
    reciprocal_shell: int = eqx.field(static=True)
    neutrality: EwaldNeutralityKind = eqx.field(static=True)
    charge_tolerance: float = eqx.field(static=True)
    minimum_distance: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell: PeriodicCell,
        units: AtomisticUnitSystem,
        alpha: float,
        /,
        *,
        real_shell: int = 4,
        reciprocal_shell: int = 4,
        neutrality: EwaldNeutralityKind = "require-neutral",
        charge_tolerance: float = 1.0e-12,
        minimum_distance: float = 1.0e-10,
        residual_tolerance: float = 1.0e-9,
    ):
        if (
            not isinstance(cell, PeriodicCell)
            or not cell.fully_periodic
            or cell.rank != 3
        ):
            raise TypeError(
                "Periodic Ewald requires a fully periodic three-dimensional cell."
            )
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("Periodic Ewald requires AtomisticUnitSystem.")
        alpha_ = float(alpha)
        real = int(real_shell)
        reciprocal = int(reciprocal_shell)
        charge_tolerance_ = float(charge_tolerance)
        minimum = float(minimum_distance)
        residual_tolerance_ = float(residual_tolerance)
        if neutrality not in ("require-neutral", "uniform-background"):
            raise ValueError("neutrality must be require-neutral or uniform-background.")
        if (
            not isfinite(alpha_)
            or alpha_ <= 0.0
            or real < 0
            or reciprocal < 1
            or not isfinite(charge_tolerance_)
            or charge_tolerance_ <= 0.0
            or not isfinite(minimum)
            or minimum <= 0.0
            or not isfinite(residual_tolerance_)
            or residual_tolerance_ <= 0.0
        ):
            raise ValueError("Ewald splitting, shells, or tolerances are invalid.")
        self.cell = cell
        self.units = units
        self.alpha = alpha_
        self.real_shell = real
        self.reciprocal_shell = reciprocal
        self.neutrality = neutrality
        self.charge_tolerance = charge_tolerance_
        self.minimum_distance = minimum
        self.residual_tolerance = residual_tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-ewald-plan",
                "cell": cell.cell_id,
                "unit_system": units.unit_system_id,
                "alpha": alpha_.hex(),
                "real_shell": real,
                "reciprocal_shell": reciprocal,
                "neutrality": neutrality,
                "charge_tolerance": charge_tolerance_.hex(),
                "minimum_distance": minimum.hex(),
                "residual_tolerance": residual_tolerance_.hex(),
            }
        )

    def prepare(self, /) -> "PreparedPeriodicEwald":
        return PreparedPeriodicEwald(self)


class PreparedPeriodicEwald(StrictModule, NonTrainableState):
    plan: PeriodicEwaldPlan
    real_indices: Array
    reciprocal_indices: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: PeriodicEwaldPlan, /):
        if not isinstance(plan, PeriodicEwaldPlan):
            raise TypeError("plan must be PeriodicEwaldPlan.")
        real = np.stack(
            np.meshgrid(
                *(np.arange(-plan.real_shell, plan.real_shell + 1) for _ in range(3)),
                indexing="ij",
            ),
            axis=-1,
        ).reshape((-1, 3))
        reciprocal = np.stack(
            np.meshgrid(
                *(
                    np.arange(-plan.reciprocal_shell, plan.reciprocal_shell + 1)
                    for _ in range(3)
                ),
                indexing="ij",
            ),
            axis=-1,
        ).reshape((-1, 3))
        reciprocal = reciprocal[np.any(reciprocal != 0, axis=1)]
        self.plan = plan
        self.real_indices = jnp.asarray(real)
        self.reciprocal_indices = jnp.asarray(reciprocal)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-periodic-ewald",
                "plan": plan.plan_id,
                "arrays": array_tree_fingerprint(
                    {"real_indices": real, "reciprocal_indices": reciprocal}
                ),
            }
        )

    def _energy(self, positions: Array, charges: Array, cell: Array, /):
        determinant = jnp.dot(cell[0], jnp.cross(cell[1], cell[2]))
        volume = jnp.abs(determinant)
        inverse = (
            jnp.stack(
                (
                    jnp.cross(cell[1], cell[2]),
                    jnp.cross(cell[2], cell[0]),
                    jnp.cross(cell[0], cell[1]),
                ),
                axis=1,
            )
            / determinant
        )
        constant = self.plan.units.coulomb_constant
        translations = self.real_indices @ cell
        displacement = (
            positions[:, None, None, :]
            - positions[None, :, None, :]
            + translations[None, None, :, :]
        )
        distance_squared = jnp.sum(displacement**2, axis=-1)
        zero_translation = jnp.all(self.real_indices == 0, axis=1)
        diagonal = jnp.eye(positions.shape[0], dtype=jnp.bool_)[:, :, None]
        include = ~(diagonal & zero_translation[None, None, :])
        safe_distance = jnp.sqrt(
            jnp.where(
                include,
                jnp.maximum(distance_squared, self.plan.minimum_distance**2),
                1.0,
            )
        )
        pair_charge = charges[:, None, None] * charges[None, :, None]
        real_energy = (
            0.5
            * constant
            * jnp.sum(
                jnp.where(
                    include,
                    pair_charge
                    * jsp.special.erfc(self.plan.alpha * safe_distance)
                    / safe_distance,
                    0.0,
                )
            )
        )
        wavevectors = 2.0 * jnp.pi * self.reciprocal_indices @ inverse.T
        squared = jnp.sum(wavevectors**2, axis=1)
        phase = positions @ wavevectors.T
        structure = jnp.sum(charges[:, None] * jnp.exp(1.0j * phase), axis=0)
        reciprocal_kernel = jnp.exp(-squared / (4.0 * self.plan.alpha**2)) / squared
        reciprocal_energy = (
            2.0
            * jnp.pi
            * constant
            / volume
            * jnp.sum(reciprocal_kernel * jnp.abs(structure) ** 2)
        )
        self_energy = -constant * self.plan.alpha / jnp.sqrt(jnp.pi) * jnp.sum(charges**2)
        net_charge = jnp.sum(charges)
        background = (
            -jnp.pi * constant * net_charge**2 / (2.0 * self.plan.alpha**2 * volume)
            if self.plan.neutrality == "uniform-background"
            else jnp.asarray(0.0, dtype=positions.dtype)
        )
        components = (real_energy, reciprocal_energy, self_energy, background)
        return sum(components), components

    def evaluate(
        self,
        positions: ArrayLike,
        charges: ArrayLike,
        /,
        *,
        cell_vectors: ArrayLike | None = None,
    ) -> PeriodicEwaldResult:
        coordinate = jnp.asarray(positions)
        charge = jnp.asarray(charges, dtype=coordinate.dtype)
        cell = (
            jnp.asarray(self.plan.cell.vectors, dtype=coordinate.dtype)
            if cell_vectors is None
            else jnp.asarray(cell_vectors, dtype=coordinate.dtype)
        )
        if (
            coordinate.ndim != 2
            or coordinate.shape[0] == 0
            or coordinate.shape[1] != 3
            or charge.shape != (coordinate.shape[0],)
            or cell.shape != (3, 3)
        ):
            raise ValueError("Ewald positions, charges, and cell have invalid shapes.")
        coordinate_host = np.asarray(coordinate)
        charge_host = np.asarray(charge)
        cell_host = np.asarray(cell)
        determinant_host = float(np.linalg.det(cell_host))
        if (
            np.any(~np.isfinite(coordinate_host))
            or np.any(~np.isfinite(charge_host))
            or np.any(~np.isfinite(cell_host))
            or abs(determinant_host) <= np.finfo(cell_host.dtype).eps
        ):
            raise ValueError(
                "Ewald geometry, charges, and cell must be finite and nonsingular."
            )
        fractional = coordinate_host @ np.linalg.inv(cell_host)
        fractional_displacement = fractional[:, None, :] - fractional[None, :, :]
        fractional_displacement -= np.rint(fractional_displacement)
        pair_distance = np.linalg.norm(fractional_displacement @ cell_host, axis=-1)
        np.fill_diagonal(pair_distance, np.inf)
        minimum_pair_distance = float(np.min(pair_distance, initial=np.inf))
        if minimum_pair_distance <= self.plan.minimum_distance:
            raise ValueError("Distinct Ewald sites violate minimum_distance.")
        net_charge_host = float(np.sum(charge_host))
        if (
            self.plan.neutrality == "require-neutral"
            and abs(net_charge_host) > self.plan.charge_tolerance
        ):
            raise ValueError(
                "Charged Ewald cells require neutrality='uniform-background'."
            )
        (energy, components), position_gradient = jax.value_and_grad(
            self._energy, argnums=0, has_aux=True
        )(coordinate, charge, cell)
        determinant = jnp.dot(cell[0], jnp.cross(cell[1], cell[2]))
        volume = jnp.abs(determinant)

        def strained_energy(strain):
            deformation = jnp.eye(3, dtype=cell.dtype) + strain
            return self._energy(
                coordinate @ deformation.T,
                charge,
                cell @ deformation.T,
            )[0]

        strain_gradient = jax.grad(strained_energy)(jnp.zeros_like(cell))
        stress = 0.5 * (strain_gradient + jnp.conj(strain_gradient.T)) / volume
        forces = -position_gradient
        ledger = ElectronicEnergyLedger(
            ("real-space", "reciprocal-space", "self", "uniform-background"),
            jnp.asarray(components),
            energy,
            self.plan.units.scale.energy_unit,
        )
        force_balance = jnp.max(jnp.abs(jnp.sum(forces, axis=0)), initial=0.0)
        stress_symmetry = jnp.max(jnp.abs(stress - stress.T), initial=0.0)
        background_applied = (
            self.plan.neutrality == "uniform-background"
            and abs(net_charge_host) > self.plan.charge_tolerance
        )
        successful = (
            jnp.isfinite(energy)
            & jnp.all(jnp.isfinite(forces))
            & jnp.all(jnp.isfinite(stress))
            & (
                jnp.abs(jnp.sum(charge)) <= self.plan.charge_tolerance
                if self.plan.neutrality == "require-neutral"
                else jnp.asarray(True)
            )
        )
        evidence = PeriodicEwaldEvidence(
            jnp.sum(charge),
            minimum_pair_distance,
            ledger.closure_residual,
            force_balance,
            stress_symmetry,
            background_applied,
            successful,
            self.plan.neutrality,
            self.plan.residual_tolerance,
        )
        return PeriodicEwaldResult(
            energy,
            forces,
            stress,
            ledger,
            evidence,
            self.plan.units,
            plan_id=self.plan.plan_id,
            prepared_id=self.prepared_id,
            cell_id=self.plan.cell.cell_id,
        )


class GTHProjectorChannel(StrictModule, NonTrainableState):
    angular_momentum: int = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    coupling: Array
    channel_id: str = eqx.field(static=True)

    def __init__(self, angular_momentum: int, radius: float, coupling: ArrayLike, /):
        angular = int(angular_momentum)
        radius_ = float(radius)
        coupling_ = jnp.asarray(coupling)
        if (
            angular < 0
            or angular > 6
            or not isfinite(radius_)
            or radius_ <= 0.0
            or coupling_.ndim != 2
            or coupling_.shape[0] != coupling_.shape[1]
            or not bool(jnp.all(jnp.isfinite(coupling_)))
        ):
            raise ValueError(
                "GTH projector angular momentum, radius, or coupling is invalid."
            )
        if not np.allclose(
            np.asarray(coupling_),
            np.asarray(jnp.conj(coupling_.T)),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("GTH projector coupling must be Hermitian.")
        self.angular_momentum = angular
        self.radius = radius_
        self.coupling = coupling_
        self.channel_id = canonical_fingerprint(
            {
                "kind": "gth-projector-channel",
                "angular_momentum": angular,
                "radius": radius_.hex(),
                "coupling": array_tree_fingerprint(np.asarray(coupling_)),
            }
        )


class GTHComponentEvidence(StrictModule, NonTrainableState):
    component_norms: Array
    successful: Array
    source_manifest_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        component_norms: ArrayLike,
        successful: ArrayLike,
        source_manifest_id: str,
        plan_id: str,
        /,
    ):
        norms = jnp.asarray(component_norms).reshape((-1,))
        source = str(source_manifest_id).strip()
        plan = str(plan_id).strip()
        if norms.size == 0 or not source or not plan:
            raise ValueError("GTH component evidence is incomplete.")
        finite = jnp.all(jnp.isfinite(norms)) & jnp.all(norms >= 0.0)
        admitted = jnp.asarray(successful, dtype=jnp.bool_).reshape(()) & finite
        self.component_norms = norms
        self.successful = admitted
        self.source_manifest_id = source
        self.plan_id = plan
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "gth-component-evidence",
                "source_manifest": source,
                "plan": plan,
                "successful": bool(admitted),
                "component_norms": array_tree_fingerprint(np.asarray(norms)),
            }
        )


class GTHLocalEvaluation(StrictModule, NonTrainableState):
    values: Array
    squared_wavevectors: Array
    evidence: GTHComponentEvidence
    coefficient_unit: UnitDefinition
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        squared_wavevectors: ArrayLike,
        evidence: GTHComponentEvidence,
        coefficient_unit: UnitDefinition,
        /,
    ):
        values_ = jnp.asarray(values)
        squared = jnp.asarray(squared_wavevectors, dtype=values_.real.dtype)
        if values_.shape != squared.shape:
            raise ValueError("GTH local values must align with squared wavevectors.")
        if not isinstance(evidence, GTHComponentEvidence):
            raise TypeError("evidence must be GTHComponentEvidence.")
        if not isinstance(coefficient_unit, UnitDefinition):
            raise TypeError("coefficient_unit must be UnitDefinition.")
        self.values = values_
        self.squared_wavevectors = squared
        self.evidence = evidence
        self.coefficient_unit = coefficient_unit
        self.result_id = canonical_fingerprint(
            {
                "kind": "gth-local-evaluation",
                "evidence": evidence.evidence_id,
                "coefficient_unit": coefficient_unit.unit_id,
                "arrays": array_tree_fingerprint(
                    {"values": np.asarray(values_), "squared": np.asarray(squared)}
                ),
            }
        )

    @property
    def successful(self) -> Array:
        return self.evidence.successful


class GTHNonlocalEvaluation(StrictModule, NonTrainableState):
    energy: Array
    channel_energies: Array
    evidence: GTHComponentEvidence
    energy_unit: UnitDefinition
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        energy: ArrayLike,
        channel_energies: ArrayLike,
        evidence: GTHComponentEvidence,
        energy_unit: UnitDefinition,
        /,
    ):
        channel = jnp.asarray(channel_energies)
        energy_ = jnp.asarray(energy, dtype=channel.real.dtype).reshape(())
        if channel.ndim != 1 or channel.size == 0:
            raise ValueError("GTH nonlocal channel energies must be a non-empty vector.")
        if not isinstance(evidence, GTHComponentEvidence):
            raise TypeError("evidence must be GTHComponentEvidence.")
        if not isinstance(energy_unit, UnitDefinition):
            raise TypeError("energy_unit must be UnitDefinition.")
        self.energy = energy_
        self.channel_energies = channel
        self.evidence = evidence
        self.energy_unit = energy_unit
        self.result_id = canonical_fingerprint(
            {
                "kind": "gth-nonlocal-evaluation",
                "evidence": evidence.evidence_id,
                "energy_unit": energy_unit.unit_id,
                "arrays": array_tree_fingerprint(
                    {"energy": np.asarray(energy_), "channels": np.asarray(channel)}
                ),
            }
        )

    @property
    def successful(self) -> Array:
        return self.evidence.successful


class GTHPseudopotentialPlan(StrictModule, NonTrainableState):
    ionic_charge: float = eqx.field(static=True)
    local_radius: float = eqx.field(static=True)
    local_coefficients: Array
    channels: tuple[GTHProjectorChannel, ...]
    energy_unit: UnitDefinition
    length_unit: UnitDefinition
    source_manifest: PeriodicProvenanceManifest
    local_coefficient_unit: UnitDefinition
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        ionic_charge: float,
        local_radius: float,
        local_coefficients: ArrayLike,
        energy_unit: UnitDefinition,
        length_unit: UnitDefinition,
        source_manifest: PeriodicProvenanceManifest,
        /,
        *,
        channels: tuple[GTHProjectorChannel, ...] = (),
    ):
        charge = float(ionic_charge)
        radius = float(local_radius)
        coefficients = jnp.asarray(local_coefficients)
        channels_ = tuple(channels)
        if (
            not isfinite(charge)
            or charge <= 0.0
            or not isfinite(radius)
            or radius <= 0.0
            or coefficients.shape != (4,)
            or not bool(jnp.all(jnp.isfinite(coefficients)))
            or any(not isinstance(value, GTHProjectorChannel) for value in channels_)
        ):
            raise ValueError("GTH local parameters or projector channels are invalid.")
        if not isinstance(energy_unit, UnitDefinition) or energy_unit.dimension != ENERGY:
            raise TypeError("GTH energy_unit must have energy dimension.")
        if not isinstance(length_unit, UnitDefinition) or length_unit.dimension != LENGTH:
            raise TypeError("GTH length_unit must have length dimension.")
        if energy_unit.reference_system_id != length_unit.reference_system_id:
            raise ValueError("GTH energy and length units must share a reference system.")
        if not isinstance(source_manifest, PeriodicProvenanceManifest):
            raise TypeError("GTH source_manifest must be PeriodicProvenanceManifest.")
        angular = tuple(value.angular_momentum for value in channels_)
        if len(set(angular)) != len(angular):
            raise ValueError("GTH projector angular momenta must be unique.")
        coefficient_unit = derived_unit(
            f"{energy_unit.symbol}*{length_unit.symbol}^3",
            ((energy_unit, 1), (length_unit, 3)),
        )
        self.ionic_charge = charge
        self.local_radius = radius
        self.local_coefficients = coefficients
        self.channels = channels_
        self.energy_unit = energy_unit
        self.length_unit = length_unit
        self.source_manifest = source_manifest
        self.local_coefficient_unit = coefficient_unit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gth-pseudopotential-plan",
                "ionic_charge": charge.hex(),
                "local_radius": radius.hex(),
                "local_coefficients": array_tree_fingerprint(np.asarray(coefficients)),
                "channels": [value.channel_id for value in channels_],
                "energy_unit": energy_unit.unit_id,
                "length_unit": length_unit.unit_id,
                "source_manifest": source_manifest.manifest_id,
            }
        )

    def local_reciprocal(self, squared_wavevectors: ArrayLike, /) -> GTHLocalEvaluation:
        squared = jnp.asarray(squared_wavevectors)
        if (
            squared.size == 0
            or bool(jnp.any(~jnp.isfinite(squared)))
            or bool(jnp.any(squared < 0.0))
        ):
            raise ValueError(
                "Squared GTH wavevectors must be non-empty, finite, and non-negative."
            )
        scaled = self.local_radius**2 * squared
        gaussian = jnp.exp(-0.5 * scaled)
        polynomial = (
            self.local_coefficients[0]
            + self.local_coefficients[1] * (3.0 - scaled)
            + self.local_coefficients[2] * (15.0 - 10.0 * scaled + scaled**2)
            + self.local_coefficients[3]
            * (105.0 - 105.0 * scaled + 21.0 * scaled**2 - scaled**3)
        )
        coulomb = jnp.where(
            squared > 0.0,
            -4.0
            * jnp.pi
            * self.ionic_charge
            * gaussian
            / jnp.where(squared > 0.0, squared, 1.0),
            0.0,
        )
        local = (2.0 * jnp.pi) ** 1.5 * self.local_radius**3 * gaussian * polynomial
        values = coulomb + local
        evidence = GTHComponentEvidence(
            jnp.asarray((jnp.max(jnp.abs(coulomb)), jnp.max(jnp.abs(local)))),
            jnp.all(jnp.isfinite(values)),
            self.source_manifest.manifest_id,
            self.plan_id,
        )
        return GTHLocalEvaluation(
            values,
            squared,
            evidence,
            self.local_coefficient_unit,
        )

    def nonlocal_energy(
        self, projector_overlaps: tuple[ArrayLike, ...], /
    ) -> GTHNonlocalEvaluation:
        if not self.channels:
            raise ValueError("This GTH plan has no nonlocal projector channels.")
        if len(projector_overlaps) != len(self.channels):
            raise ValueError("GTH projector overlaps must align with channels.")
        energies = []
        for channel, overlaps in zip(self.channels, projector_overlaps, strict=True):
            values = jnp.asarray(overlaps)
            if values.shape[-1] != channel.coupling.shape[0] or not bool(
                jnp.all(jnp.isfinite(values))
            ):
                raise ValueError(
                    "GTH projector overlaps must be finite and match channel rank."
                )
            energies.append(
                jnp.real(
                    contract(
                        "...i,ij,...j->",
                        jnp.conj(values),
                        channel.coupling,
                        values,
                        backend="jax",
                    )
                )
            )
        channel_energies = jnp.stack(tuple(energies))
        total = jnp.sum(channel_energies)
        evidence = GTHComponentEvidence(
            jnp.abs(channel_energies),
            jnp.isfinite(total),
            self.source_manifest.manifest_id,
            self.plan_id,
        )
        return GTHNonlocalEvaluation(
            total,
            channel_energies,
            evidence,
            self.energy_unit,
        )


__all__ = [
    "EwaldNeutralityKind",
    "GTHComponentEvidence",
    "GTHLocalEvaluation",
    "GTHNonlocalEvaluation",
    "GTHProjectorChannel",
    "GTHPseudopotentialPlan",
    "PeriodicEwaldEvidence",
    "PeriodicEwaldPlan",
    "PeriodicEwaldResult",
    "PreparedPeriodicEwald",
]
