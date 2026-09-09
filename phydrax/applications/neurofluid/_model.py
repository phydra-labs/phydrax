#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Typed neurofluid cases, tracer calibration, and forward transport assembly."""

from __future__ import annotations

from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...discretization import (
    CircleAverageKernel,
    EmbeddedMeasureTransferPlan,
    EmbeddedSourceAssociation,
    PreparedMetricNetwork,
)
from ...equations import (
    BulkDGTransportPlan,
    MixedDimensionalTransportPlan,
    NetworkTransportPlan,
    PreparedMixedDimensionalTransport,
)
from ...geometry import CompartmentComplex
from ...imaging import DiffusionTensorImage, LabelVolume, MedicalImageAsset
from ...meshing import CompartmentMeshingResult
from ...units import conversion_factor, derived_unit, LENGTH, TIME, UnitDefinition


@dataclass(frozen=True, slots=True)
class NeurofluidCase:
    case_id: str
    segmentation: LabelVolume
    compartments: CompartmentComplex
    bulk_mesh: CompartmentMeshingResult
    network: PreparedMetricNetwork
    concentration_series: MedicalImageAsset | None = None
    diffusion_tensor: DiffusionTensorImage | None = None
    case_revision: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.case_id, str)
            or not self.case_id
            or self.case_id != self.case_id.strip()
        ):
            raise ValueError("case_id must be a canonical non-empty identifier.")
        if not isinstance(self.segmentation, LabelVolume):
            raise TypeError("segmentation must be LabelVolume.")
        if not isinstance(self.compartments, CompartmentComplex):
            raise TypeError("compartments must be CompartmentComplex.")
        if self.compartments.source_revision != self.segmentation.label_volume_id:
            raise ValueError("Compartment complex and segmentation revisions differ.")
        self.compartments.require_valid_adjacency()
        if not isinstance(self.bulk_mesh, CompartmentMeshingResult):
            raise TypeError("bulk_mesh must be CompartmentMeshingResult.")
        compartment_ids = {
            value.compartment_id for value in self.compartments.compartments
        }
        if (
            set(self.bulk_mesh.cell_compartment_ids) != compartment_ids
            or {zone.name for zone in self.bulk_mesh.zones} != compartment_ids
        ):
            raise ValueError(
                "Compartment mesh zones must exactly match the compartment complex."
            )
        if not isinstance(self.network, PreparedMetricNetwork):
            raise TypeError("network must be PreparedMetricNetwork.")
        spatial_ids = {
            self.segmentation.asset.spatial_affine.coordinate_contract.spatial_id,
            self.bulk_mesh.result.coordinate_contract.spatial_id,
            self.network.coordinate_contract.spatial_id,
        }
        if len(spatial_ids) != 1:
            raise ValueError(
                "Segmentation, bulk mesh, and network coordinate contracts differ."
            )
        if self.concentration_series is not None:
            if not isinstance(self.concentration_series, MedicalImageAsset):
                raise TypeError("concentration_series must be MedicalImageAsset or None.")
            if (
                self.concentration_series.spatial_affine.coordinate_contract.spatial_id
                not in spatial_ids
            ):
                raise ValueError("Concentration and case coordinate contracts differ.")
        if self.diffusion_tensor is not None:
            if not isinstance(self.diffusion_tensor, DiffusionTensorImage):
                raise TypeError("diffusion_tensor must be DiffusionTensorImage or None.")
            if (
                self.diffusion_tensor.asset.spatial_affine.coordinate_contract.spatial_id
                not in spatial_ids
            ):
                raise ValueError("Diffusion tensor and case coordinate contracts differ.")
        object.__setattr__(
            self,
            "case_revision",
            canonical_fingerprint(
                {
                    "kind": "neurofluid-case",
                    "case": self.case_id,
                    "segmentation": self.segmentation.label_volume_id,
                    "compartments": self.compartments.complex_id,
                    "bulk_mesh": self.bulk_mesh.result.mesh.mesh_id,
                    "network": self.network.network_id,
                    "concentration": None
                    if self.concentration_series is None
                    else self.concentration_series.content_id,
                    "diffusion": None
                    if self.diffusion_tensor is None
                    else self.diffusion_tensor.tensor_image_id,
                }
            ),
        )


class TracerConcentrationEvidence(StrictModule):
    valid: Array
    finite: Array
    nonnegative_state_candidate: Array
    successful: Array


class TracerConcentrationResult(StrictModule):
    values: Array
    valid_mask: Array
    evidence: TracerConcentrationEvidence
    baseline_asset_id: str = eqx.field(static=True)
    contrast_asset_id: str = eqx.field(static=True)
    calibration_id: str = eqx.field(static=True)


@dataclass(frozen=True, slots=True)
class TracerRelaxivityCalibration:
    relaxivity: float
    relaxivity_unit: UnitDefinition
    concentration_unit: UnitDefinition
    relaxation_time_unit: UnitDefinition
    calibration_id: str = field(init=False)

    def __post_init__(self) -> None:
        relaxivity = float(self.relaxivity)
        if not np.isfinite(relaxivity) or relaxivity <= 0.0:
            raise ValueError("relaxivity must be finite and positive.")
        if not isinstance(self.concentration_unit, UnitDefinition):
            raise TypeError("concentration_unit must be UnitDefinition.")
        if (
            not isinstance(self.relaxation_time_unit, UnitDefinition)
            or self.relaxation_time_unit.dimension != TIME
        ):
            raise ValueError("relaxation_time_unit must have time dimension.")
        if not isinstance(self.relaxivity_unit, UnitDefinition):
            raise TypeError("relaxivity_unit must be UnitDefinition.")
        expected_relaxivity_unit = derived_unit(
            f"1/({self.concentration_unit.symbol}·{self.relaxation_time_unit.symbol})",
            ((self.concentration_unit, -1), (self.relaxation_time_unit, -1)),
        )
        relaxivity = float(
            relaxivity
            * float(conversion_factor(self.relaxivity_unit, expected_relaxivity_unit))
        )
        object.__setattr__(self, "relaxivity_unit", expected_relaxivity_unit)
        object.__setattr__(self, "relaxivity", relaxivity)
        object.__setattr__(
            self,
            "calibration_id",
            canonical_fingerprint(
                {
                    "kind": "tracer-relaxivity-calibration",
                    "relaxivity": relaxivity.hex(),
                    "concentration_unit": self.concentration_unit.unit_id,
                    "time_unit": self.relaxation_time_unit.unit_id,
                    "relaxivity_unit": expected_relaxivity_unit.unit_id,
                }
            ),
        )

    def evaluate(
        self, baseline_t1: MedicalImageAsset, contrast_t1: MedicalImageAsset, /
    ) -> TracerConcentrationResult:
        if not isinstance(baseline_t1, MedicalImageAsset) or not isinstance(
            contrast_t1, MedicalImageAsset
        ):
            raise TypeError("Tracer calibration requires two MedicalImageAsset values.")
        if baseline_t1.spatial_affine.affine_id != contrast_t1.spatial_affine.affine_id:
            raise ValueError("Baseline and contrast T1 affines differ.")
        if baseline_t1.values.shape != contrast_t1.values.shape:
            raise ValueError("Baseline and contrast T1 shapes differ.")
        if (
            baseline_t1.quantity.unit.dimension != TIME
            or contrast_t1.quantity.unit.dimension != TIME
        ):
            raise ValueError(
                "Tracer calibration inputs must carry relaxation-time units."
            )
        baseline = jnp.asarray(baseline_t1.values) * float(
            conversion_factor(baseline_t1.quantity.unit, self.relaxation_time_unit)
        )
        contrast = jnp.asarray(contrast_t1.values) * float(
            conversion_factor(contrast_t1.quantity.unit, self.relaxation_time_unit)
        )
        valid = (
            jnp.asarray(baseline_t1.valid_mask)
            & jnp.asarray(contrast_t1.valid_mask)
            & (baseline > 0.0)
            & (contrast > 0.0)
        )
        safe_baseline = jnp.where(valid, baseline, 1.0)
        safe_contrast = jnp.where(valid, contrast, 1.0)
        values = (1.0 / safe_contrast - 1.0 / safe_baseline) / self.relaxivity
        values = jnp.where(valid, values, jnp.nan)
        finite = jnp.all(jnp.where(valid, jnp.isfinite(values), True))
        evidence = TracerConcentrationEvidence(
            valid,
            finite,
            jnp.all(jnp.where(valid, values >= 0.0, True)),
            finite & jnp.any(valid),
        )
        return TracerConcentrationResult(
            values,
            valid,
            evidence,
            baseline_t1.asset_id,
            contrast_t1.asset_id,
            self.calibration_id,
        )


@dataclass(frozen=True, slots=True)
class NeurofluidTransportUnits:
    length_unit: UnitDefinition
    time_unit: UnitDefinition
    concentration_unit: UnitDefinition
    diffusivity_unit: UnitDefinition = field(init=False)
    velocity_unit: UnitDefinition = field(init=False)
    volume_flow_unit: UnitDefinition = field(init=False)
    exchange_rate_unit: UnitDefinition = field(init=False)
    unit_id: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.length_unit, UnitDefinition)
            or self.length_unit.dimension != LENGTH
        ):
            raise ValueError("length_unit must have length dimension.")
        if (
            not isinstance(self.time_unit, UnitDefinition)
            or self.time_unit.dimension != TIME
        ):
            raise ValueError("time_unit must have time dimension.")
        if not isinstance(self.concentration_unit, UnitDefinition):
            raise TypeError("concentration_unit must be UnitDefinition.")
        diffusivity = derived_unit(
            f"{self.length_unit.symbol}²/{self.time_unit.symbol}",
            ((self.length_unit, 2), (self.time_unit, -1)),
        )
        velocity = derived_unit(
            f"{self.length_unit.symbol}/{self.time_unit.symbol}",
            ((self.length_unit, 1), (self.time_unit, -1)),
        )
        flow = derived_unit(
            f"{self.length_unit.symbol}³/{self.time_unit.symbol}",
            ((self.length_unit, 3), (self.time_unit, -1)),
        )
        exchange = derived_unit(f"1/{self.time_unit.symbol}", ((self.time_unit, -1),))
        object.__setattr__(self, "diffusivity_unit", diffusivity)
        object.__setattr__(self, "velocity_unit", velocity)
        object.__setattr__(self, "volume_flow_unit", flow)
        object.__setattr__(self, "exchange_rate_unit", exchange)
        object.__setattr__(
            self,
            "unit_id",
            canonical_fingerprint(
                {
                    "kind": "neurofluid-transport-units",
                    "length": self.length_unit.unit_id,
                    "time": self.time_unit.unit_id,
                    "concentration": self.concentration_unit.unit_id,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class NeurofluidTransportParameters:
    porosity: np.ndarray
    bulk_diffusivity: np.ndarray
    bulk_velocity: np.ndarray
    bulk_boundary_volume_flux: np.ndarray
    bulk_boundary_inflow_concentration: np.ndarray
    bulk_removal_rate: np.ndarray
    network_diffusivity: np.ndarray
    units: NeurofluidTransportUnits
    network_volume_flow: np.ndarray
    exchange_coefficients: np.ndarray
    averaging_radius: float
    reservoir_volumes: np.ndarray
    reservoir_coefficients: np.ndarray
    parameter_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.units, NeurofluidTransportUnits):
            raise TypeError("units must be NeurofluidTransportUnits.")
        arrays = {}
        raw_values = (
            ("porosity", self.porosity),
            ("bulk_diffusivity", self.bulk_diffusivity),
            ("bulk_velocity", self.bulk_velocity),
            ("bulk_boundary_volume_flux", self.bulk_boundary_volume_flux),
            (
                "bulk_boundary_inflow_concentration",
                self.bulk_boundary_inflow_concentration,
            ),
            ("bulk_removal_rate", self.bulk_removal_rate),
            ("network_diffusivity", self.network_diffusivity),
            ("network_volume_flow", self.network_volume_flow),
            ("exchange_coefficients", self.exchange_coefficients),
            ("reservoir_volumes", self.reservoir_volumes),
            ("reservoir_coefficients", self.reservoir_coefficients),
        )
        for name, raw in raw_values:
            value = np.asarray(raw, dtype=float)
            if np.any(~np.isfinite(value)):
                raise ValueError(f"{name} must be finite.")
            value = np.array(value, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
            arrays[name] = array_tree_fingerprint(value)
        radius = float(self.averaging_radius)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("averaging_radius must be finite and positive.")
        if np.any(self.porosity <= 0.0) or np.any(self.bulk_diffusivity < 0.0):
            raise ValueError("Porosity must be positive and diffusivity non-negative.")
        if np.any(self.bulk_boundary_inflow_concentration < 0.0) or np.any(
            self.bulk_removal_rate < 0.0
        ):
            raise ValueError(
                "Bulk inflow concentration and removal rate must be non-negative."
            )
        if np.any(self.network_diffusivity < 0.0) or np.any(
            self.exchange_coefficients < 0.0
        ):
            raise ValueError(
                "Network diffusivity and exchange coefficients must be non-negative."
            )
        if np.any(self.reservoir_volumes <= 0.0) or np.any(
            self.reservoir_coefficients < 0.0
        ):
            raise ValueError(
                "Reservoir volumes must be positive and coefficients non-negative."
            )
        object.__setattr__(self, "averaging_radius", radius)
        object.__setattr__(
            self,
            "parameter_id",
            canonical_fingerprint(
                {
                    "kind": "neurofluid-transport-parameters",
                    "arrays": arrays,
                    "units": self.units.unit_id,
                    "radius": radius.hex(),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class NeurofluidTransportPlan:
    case: NeurofluidCase
    parameters: NeurofluidTransportParameters
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.case, NeurofluidCase):
            raise TypeError("case must be NeurofluidCase.")
        if not isinstance(self.parameters, NeurofluidTransportParameters):
            raise TypeError("parameters must be NeurofluidTransportParameters.")
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "neurofluid-transport-plan",
                    "case": self.case.case_revision,
                    "parameters": self.parameters.parameter_id,
                }
            ),
        )

    def prepare(self) -> PreparedMixedDimensionalTransport:
        mesh = self.case.bulk_mesh.result.mesh
        bulk_plan = BulkDGTransportPlan(
            mesh,
            self.parameters.porosity,
            self.parameters.bulk_diffusivity,
            self.parameters.bulk_velocity,
            boundary_volume_flux=self.parameters.bulk_boundary_volume_flux,
            boundary_inflow_concentration=self.parameters.bulk_boundary_inflow_concentration,
            removal_rate=self.parameters.bulk_removal_rate,
        )
        bulk = bulk_plan.prepare()
        network_plan = NetworkTransportPlan(
            self.case.network,
            self.parameters.network_diffusivity,
            self.parameters.network_volume_flow,
        )
        network = self.case.network
        tangents = np.zeros_like(np.asarray(network.mesh.coordinates))
        np.add.at(tangents, np.asarray(network.senders), np.asarray(network.tangents))
        np.add.at(tangents, np.asarray(network.receivers), np.asarray(network.tangents))
        tangent_norm = np.linalg.norm(tangents, axis=1)
        if np.any(tangent_norm <= 0.0):
            raise ValueError("Every network node requires a nonzero averaging tangent.")
        tangents /= tangent_norm[:, None]
        transfer = EmbeddedMeasureTransferPlan(
            mesh,
            self.case.bulk_mesh.result.coordinate_contract,
            np.asarray(network.mesh.coordinates),
            np.asarray(network.node_measures),
            np.asarray(bulk.mass),
            network.network_id,
            CircleAverageKernel(self.parameters.averaging_radius),
            EmbeddedSourceAssociation.CELL,
            tangents,
        ).prepare()
        tips = np.flatnonzero(np.asarray(network.tip_mask))
        if len(tips) != len(self.parameters.reservoir_volumes):
            raise ValueError("Reservoir parameters must match network tip count.")
        mixed = MixedDimensionalTransportPlan(
            bulk_plan,
            network_plan,
            transfer,
            self.parameters.exchange_coefficients,
            tips,
            self.parameters.reservoir_volumes,
            self.parameters.reservoir_coefficients,
        )
        return mixed.prepare()


class NeurofluidDiagnosticReport(StrictModule):
    compartment_ids: tuple[str, ...] = eqx.field(static=True)
    compartment_mass: Array
    total_mass: Array
    external_loss_rate: Array
    exchange_defect: Array
    minimum_concentration: Array
    finite: Array
    successful: Array
    report_id: str = eqx.field(static=True)


def neurofluid_diagnostics(
    runtime: PreparedMixedDimensionalTransport,
    state,
    cell_compartment_indices: Array,
    compartment_ids: tuple[str, ...],
    /,
) -> NeurofluidDiagnosticReport:
    assignment = jnp.asarray(cell_compartment_indices)
    if assignment.shape != runtime.bulk.mass.shape or not jnp.issubdtype(
        assignment.dtype, jnp.integer
    ):
        raise ValueError("cell_compartment_indices must identify each bulk cell.")
    if (
        not compartment_ids
        or jnp.any(assignment < 0)
        or jnp.any(assignment >= len(compartment_ids))
    ):
        raise ValueError("Compartment assignments must lie in the declared range.")
    ledger = runtime.ledger(state)
    weighted = runtime.bulk.mass * state.bulk
    masses = (
        jnp.zeros((len(compartment_ids),), dtype=weighted.dtype)
        .at[assignment]
        .add(weighted)
    )
    finite = ledger.finite & jnp.all(jnp.isfinite(masses))
    report_id = canonical_fingerprint(
        {
            "kind": "neurofluid-diagnostic-report",
            "runtime": runtime.runtime_id,
            "compartments": list(compartment_ids),
        }
    )
    return NeurofluidDiagnosticReport(
        compartment_ids,
        masses,
        ledger.total_mass,
        ledger.external_loss_rate,
        ledger.exchange_defect,
        ledger.minimum_concentration,
        finite,
        finite & ledger.successful,
        report_id,
    )


__all__ = [
    "NeurofluidCase",
    "NeurofluidDiagnosticReport",
    "NeurofluidTransportParameters",
    "NeurofluidTransportPlan",
    "TracerConcentrationEvidence",
    "NeurofluidTransportUnits",
    "TracerConcentrationResult",
    "TracerRelaxivityCalibration",
    "neurofluid_diagnostics",
]
