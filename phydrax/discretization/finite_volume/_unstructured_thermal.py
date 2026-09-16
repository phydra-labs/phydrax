#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._geometry_protocol import FiniteVolumeStageMetrics
from ._unstructured import UnstructuredFiniteVolumeDiscretization


UnstructuredThermalBoundaryKind: TypeAlias = Literal[
    "adiabatic", "temperature", "heat_flux"
]


class UnstructuredThermalBoundaryCondition(StrictModule, NonTrainableState):
    kind: UnstructuredThermalBoundaryKind = eqx.field(static=True)
    value: float = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: UnstructuredThermalBoundaryKind = "adiabatic",
        value=0.0,
        /,
    ):
        if kind not in ("adiabatic", "temperature", "heat_flux"):
            raise ValueError("Unknown unstructured thermal boundary kind.")
        value_ = float(value)
        if not np.isfinite(value_) or (kind == "adiabatic" and value_ != 0.0):
            raise ValueError(
                "Thermal boundary data must be finite and adiabatic data zero."
            )
        self.kind = kind
        self.value = value_
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "unstructured-thermal-boundary",
                "boundary_kind": kind,
                "value": value_,
                "heat_flux_sign": "outward-loss",
            }
        )


class UnstructuredThermalDiffusionEvaluation(StrictModule):
    temperature: Array
    conductivity: Array
    face_conductivity: Array
    face_normal_heat_flux: Array
    face_energy_rate: Array
    cell_energy_rate: Array
    explicit_step_restriction: Array
    boundary_energy_rate: Array
    conservation_defect: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)


class UnstructuredTwoMaterialThermalDiffusionPlan(StrictModule, NonTrainableState):
    discretization: UnstructuredFiniteVolumeDiscretization
    phase0_conductivity: float = eqx.field(static=True)
    phase1_conductivity: float = eqx.field(static=True)
    boundaries: tuple[UnstructuredThermalBoundaryCondition, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: UnstructuredFiniteVolumeDiscretization,
        phase0_conductivity,
        phase1_conductivity,
        /,
        *,
        boundaries: Mapping[
            str,
            UnstructuredThermalBoundaryCondition | UnstructuredThermalBoundaryKind,
        ]
        | None = None,
    ):
        if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            raise TypeError("Thermal diffusion requires unstructured FV geometry.")
        conductivity0 = float(phase0_conductivity)
        conductivity1 = float(phase1_conductivity)
        if (
            not np.isfinite(conductivity0)
            or not np.isfinite(conductivity1)
            or conductivity0 <= 0.0
            or conductivity1 <= 0.0
        ):
            raise ValueError("Phase conductivities must be positive and finite.")
        supplied = (
            {}
            if boundaries is None
            else {str(name): value for name, value in boundaries.items()}
        )
        unknown = set(supplied).difference(discretization.boundary_patch_names)
        if unknown:
            raise ValueError(
                f"Thermal boundaries reference unknown patches {sorted(unknown)!r}."
            )
        conditions = tuple(
            (
                supplied[name]
                if isinstance(
                    supplied.get(name, "adiabatic"),
                    UnstructuredThermalBoundaryCondition,
                )
                else UnstructuredThermalBoundaryCondition(supplied.get(name, "adiabatic"))
            )
            for name in discretization.boundary_patch_names
        )
        self.discretization = discretization
        self.phase0_conductivity = conductivity0
        self.phase1_conductivity = conductivity1
        self.boundaries = conditions
        self.plan_id = canonical_fingerprint(
            {
                "kind": "unstructured-two-material-thermal-diffusion",
                "geometry": discretization.prepared_id,
                "phase0_conductivity": conductivity0,
                "phase1_conductivity": conductivity1,
                "boundaries": [condition.boundary_id for condition in conditions],
            }
        )

    def evaluate(
        self,
        system: Any,
        state: ArrayLike,
        /,
        *,
        stage_metrics: FiniteVolumeStageMetrics | None = None,
    ) -> UnstructuredThermalDiffusionEvaluation:
        from ...equations._multiphase import TwoMaterialVOFSystem

        if not isinstance(system, TwoMaterialVOFSystem):
            raise TypeError("Thermal diffusion requires TwoMaterialVOFSystem.")
        value = system._state(state)
        if value.shape != (
            self.discretization.cell_count,
            system.component_count,
        ):
            raise ValueError("Thermal state must contain one value per prepared cell.")
        if stage_metrics is None:
            cell_centers = self.discretization.cell_centers
            cell_volumes = self.discretization.cell_volumes
            face_measures = self.discretization.face_measures
            face_centers = self.discretization.face_centers
            geometry_id = self.discretization.geometry_id
            stage_evidence = jnp.asarray(True)
            active_cells = jnp.ones((self.discretization.cell_count,), dtype=jnp.bool_)
            face_active = jnp.ones_like(
                self.discretization.face_measures, dtype=jnp.bool_
            )
        else:
            if not isinstance(stage_metrics, FiniteVolumeStageMetrics):
                raise TypeError("stage_metrics must be FiniteVolumeStageMetrics or None.")
            if stage_metrics.cell_count != self.discretization.cell_count:
                raise ValueError("Thermal stage and prepared cell counts differ.")
            physical_blocks = tuple(
                block
                for block in stage_metrics.face_blocks
                if block.layout.block_kind == "physical"
            )
            if len(physical_blocks) != 1:
                raise ValueError(
                    "Thermal moving geometry requires one physical face block."
                )
            block = physical_blocks[0]
            positions = jnp.argsort(block.layout.face_ids)
            expected_ids = jnp.arange(
                self.discretization.face_measures.size, dtype=jnp.int32
            )
            face_measures = eqx.error_if(
                block.face_measures[positions],
                jnp.any(block.layout.face_ids[positions] != expected_ids),
                "Thermal stage face IDs do not match prepared physical routes.",
            )
            face_centers = block.face_centers[positions]
            active_cells = stage_metrics.active_cell_mask
            face_active = block.layout.active_mask[positions]
            cell_centers = stage_metrics.cell_centers
            cell_volumes = stage_metrics.effective_cell_volumes
            geometry_id = stage_metrics.geometry_family_id
            stage_evidence = stage_metrics.evidence.passed
        temperature = system.eos.temperature(value)
        alpha0 = value[:, system.alpha_index]
        conductivity = (
            alpha0 * self.phase0_conductivity + (1.0 - alpha0) * self.phase1_conductivity
        )
        owner = self.discretization.owner_cells.astype(jnp.int32)
        neighbour = self.discretization.neighbour_cells.astype(jnp.int32)
        safe_neighbour = jnp.maximum(neighbour, 0)
        internal = (
            (neighbour >= 0)
            & face_active
            & active_cells[owner]
            & active_cells[safe_neighbour]
        )
        owner_conductivity = conductivity[owner]
        neighbour_conductivity = conductivity[safe_neighbour]
        face_conductivity = (
            2.0
            * owner_conductivity
            * neighbour_conductivity
            / jnp.maximum(
                owner_conductivity + neighbour_conductivity,
                jnp.finfo(conductivity.dtype).tiny,
            )
        )
        center_difference = cell_centers[safe_neighbour] - cell_centers[owner]
        distance = jnp.linalg.norm(center_difference, axis=-1)
        safe_distance = jnp.maximum(distance, 64.0 * jnp.finfo(distance.dtype).eps)
        temperature_jump = temperature[safe_neighbour] - temperature[owner]
        owner_inward_flux = face_conductivity * temperature_jump / safe_distance
        owner_inward_flux = jnp.where(internal, owner_inward_flux, 0.0)
        boundary_distance = jnp.linalg.norm(face_centers - cell_centers[owner], axis=-1)
        safe_boundary_distance = jnp.maximum(
            boundary_distance, 64.0 * jnp.finfo(boundary_distance.dtype).eps
        )
        boundary_conductance = jnp.zeros_like(owner_inward_flux)
        boundary_patch_ids = self.discretization.boundary_patch_ids
        for patch_id, condition in enumerate(self.boundaries):
            mask = (boundary_patch_ids == patch_id) & face_active & active_cells[owner]
            if condition.kind == "temperature":
                boundary_flux = (
                    owner_conductivity
                    * (condition.value - temperature[owner])
                    / safe_boundary_distance
                )
                boundary_conductance = jnp.where(
                    mask,
                    owner_conductivity * face_measures / safe_boundary_distance,
                    boundary_conductance,
                )
            elif condition.kind == "heat_flux":
                boundary_flux = jnp.full_like(owner_inward_flux, -condition.value)
            else:
                boundary_flux = jnp.zeros_like(owner_inward_flux)
            owner_inward_flux = jnp.where(mask, boundary_flux, owner_inward_flux)
        face_energy_rate = owner_inward_flux * face_measures
        cell_energy_rate = jnp.zeros((self.discretization.cell_count,), dtype=value.dtype)
        cell_energy_rate = cell_energy_rate.at[owner].add(face_energy_rate)
        cell_energy_rate = cell_energy_rate.at[safe_neighbour].add(
            jnp.where(internal, -face_energy_rate, 0.0)
        )
        cell_energy_rate = jnp.where(active_cells, cell_energy_rate, 0.0)
        pressure = system.pressure(value)
        density0, density1 = system.phase_densities(value)
        cp0 = system.eos.material_0.specific_heat_cp(density0, pressure)
        cp1 = system.eos.material_1.specific_heat_cp(density1, pressure)
        volumetric_capacity = value[:, 0] * cp0 + value[:, 1] * cp1
        inverse_rate = jnp.zeros_like(volumetric_capacity)
        conductance = jnp.where(
            internal,
            face_conductivity * face_measures / safe_distance,
            0.0,
        )
        conductance = conductance + boundary_conductance
        inverse_rate = inverse_rate.at[owner].add(conductance)
        inverse_rate = inverse_rate.at[safe_neighbour].add(
            jnp.where(internal, conductance, 0.0)
        )
        inverse_rate = inverse_rate / jnp.maximum(
            volumetric_capacity * cell_volumes,
            jnp.finfo(value.dtype).tiny,
        )
        maximum_rate = jnp.max(inverse_rate)
        restriction = jnp.where(maximum_rate > 0.0, 1.0 / maximum_rate, jnp.inf)
        boundary_rate = jnp.sum(jnp.where(~internal, face_energy_rate, 0.0))
        defect = jnp.sum(cell_energy_rate) - boundary_rate
        finite = (
            jnp.all(jnp.isfinite(temperature))
            & jnp.all(jnp.isfinite(conductivity))
            & jnp.all(jnp.isfinite(face_energy_rate))
            & jnp.all(jnp.isfinite(cell_energy_rate))
            & ~jnp.isnan(restriction)
            & jnp.isfinite(defect)
            & stage_evidence
        )
        scale = jnp.maximum(1.0, jnp.sum(jnp.abs(face_energy_rate)))
        tolerance = 256.0 * jnp.finfo(value.dtype).eps * scale
        return UnstructuredThermalDiffusionEvaluation(
            temperature,
            conductivity,
            face_conductivity,
            -owner_inward_flux,
            face_energy_rate,
            cell_energy_rate,
            restriction,
            boundary_rate,
            defect,
            finite,
            finite & (jnp.abs(defect) <= tolerance),
            self.plan_id,
            geometry_id,
        )


__all__ = [
    "UnstructuredThermalBoundaryCondition",
    "UnstructuredThermalBoundaryKind",
    "UnstructuredThermalDiffusionEvaluation",
    "UnstructuredTwoMaterialThermalDiffusionPlan",
]
