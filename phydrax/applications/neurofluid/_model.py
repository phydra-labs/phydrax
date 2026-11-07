#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Typed neurofluid cases, tracer calibration, and forward transport assembly."""

from __future__ import annotations

from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

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
    MixedDimensionalTransportState,
    NetworkTransportPlan,
    PreparedMixedDimensionalTransport,
)
from ...geometry import CompartmentMeshingSource
from ...geometry._affine_plc_domain import AffinePlcDomain
from ...imaging import DiffusionTensorImage, MedicalImageAsset
from ...meshing import CellMeshingResult, GeometryAssociationKind
from ...meshing._association import GeometrySourceEntityRole
from ...meshing._organization import MeshPatch, MeshZoneRole, RegionRole
from ...meshing._scope import MeshingEntityKind
from ...units import conversion_factor, derived_unit, LENGTH, TIME, UnitDefinition


def _require_compartment_mesh_binding(
    source: CompartmentMeshingSource, result: CellMeshingResult, /
) -> None:
    source.validate_source_integrity()
    compartments = {
        value.compartment_id: value for value in source.compartments.compartments
    }
    compartment_ids = set(compartments)
    source_regions: dict[tuple[str, tuple[str, ...]], str] = {}
    source_regions_by_scope_id: dict[int, str] = {}
    for identifier in source.scope_entity_ids(3).tolist():
        key, occurrence = source.source_entity_key_for_scope(3, int(identifier))
        if key not in compartments:
            raise ValueError(
                "Compartment source region keys must identify declared compartments."
            )
        identity = (key, occurrence)
        if identity in source_regions:
            raise ValueError("Compartment source region identities must be unique.")
        source_regions[identity] = key
        source_regions_by_scope_id[int(identifier)] = key
    if set(source_regions.values()) != compartment_ids:
        raise ValueError(
            "Compartment source scopes must cover every declared compartment."
        )

    mesh = result.mesh
    if (
        mesh.topological_dimension != 3
        or mesh.ambient_dimension != 3
        or not mesh.blocks
        or any(
            block.cell_kind not in ("tetrahedron", "simplex:3") for block in mesh.blocks
        )
    ):
        raise ValueError(
            "Neurofluid bulk results must contain three-dimensional tetrahedra."
        )
    # The bulk boundary is the mesh exterior: a conforming refinement of the
    # outer surface (its size target refines the exterior), bound to it by
    # source identity and provenance.
    outer = source.outer_surface
    boundary = result.boundary
    if (
        boundary is None
        or boundary.metadata.source_id != outer.metadata.source_id
        or boundary.metadata.source_revision != outer.metadata.source_revision
        or outer.model_id not in boundary.metadata.provenance
    ):
        raise ValueError(
            "Bulk mesh boundary must refine the compartment source outer surface."
        )
    cell_set = mesh.entity_set(3)
    cell_ids = np.asarray(cell_set.entity_ids, dtype=np.int64)
    cell_associations = tuple(
        value
        for value in result.associations
        if value.target_entity_set_id == cell_set.entity_set_id
    )
    if len(cell_associations) != 1:
        raise ValueError("Bulk mesh must retain one source association for every cell.")
    association = cell_associations[0]
    targets = np.asarray(association.target_global_ids, dtype=np.int64)
    if (
        association.association_kind is not GeometryAssociationKind.PIECEWISE_LINEAR
        or association.source_id != source.source_id
        or association.source_revision != source.source_revision
        or not association.exact
        or not np.array_equal(np.sort(targets), np.sort(cell_ids))
        or not np.all(np.asarray(association.resolved, dtype=np.bool_))
        or np.any(np.asarray(association.ambiguous, dtype=np.bool_))
        or association.source_roles != (GeometrySourceEntityRole.REGION,) * targets.size
    ):
        raise ValueError(
            "Bulk cell associations must exactly identify regions in this compartment source revision."
        )
    cell_compartments: dict[int, str] = {}
    for target, source_id, occurrence in zip(
        targets.tolist(),
        association.source_entity_ids,
        association.source_occurrence_paths,
        strict=True,
    ):
        compartment_id = source_regions.get((source_id, occurrence))
        if compartment_id is None:
            raise ValueError(
                "Bulk cells must associate with declared source compartment regions."
            )
        cell_compartments[int(target)] = compartment_id
    if set(cell_compartments.values()) != compartment_ids:
        raise ValueError("Bulk source associations must represent every compartment.")

    certification = result.certification
    if certification is None:
        raise ValueError(
            "Neurofluid bulk results require computed source-region coverage certification."
        )
    certification.require_passed()
    inputs = certification.inputs
    domain = inputs.domain
    coverage = certification.coverage
    if (
        inputs.mesh_id != mesh.mesh_id
        or inputs.topology_id != mesh.topology_id
        or inputs.source_id != source.source_id
        or inputs.source_revision != source.source_revision
        or inputs.schedule.route != "mapped_volume"
        or not isinstance(domain, AffinePlcDomain)
        or domain.source_id != source.source_id
        or domain.source_revision != source.source_revision
        or domain.authority is None
        or domain.authority.source_id != source.source_id
        or domain.authority.source_revision != source.source_revision
        or inputs.cell_regions is None
        or coverage is None
    ):
        raise ValueError(
            "Neurofluid bulk certification must bind this source's volume regions."
        )
    domain.require_current()
    domain_region_ids = tuple(domain.region_ids)
    if (
        len(set(domain_region_ids)) != len(domain_region_ids)
        or set(domain_region_ids) != compartment_ids
        or coverage.status != "certified"
        or coverage.source_id != source.source_id
        or coverage.domain_id != domain.domain_id
        or coverage.region_ids != domain_region_ids
        or not np.array_equal(
            np.asarray(coverage.cell_regions, dtype=np.int64),
            np.asarray(inputs.cell_regions, dtype=np.int64),
        )
    ):
        raise ValueError(
            "Neurofluid bulk certification must preserve the source's certified region table."
        )
    certified_cell_ids = np.asarray(inputs.cell_global_ids, dtype=np.int64)
    certified_region_indices = np.asarray(inputs.cell_regions, dtype=np.int64)
    if (
        certified_cell_ids.shape != cell_ids.shape
        or certified_region_indices.shape != cell_ids.shape
        or not np.array_equal(np.sort(certified_cell_ids), np.sort(cell_ids))
    ):
        raise ValueError(
            "Neurofluid bulk certification must assign every target cell exactly once."
        )
    certified_compartments = {
        int(cell_id): domain_region_ids[int(region)]
        for cell_id, region in zip(
            certified_cell_ids.tolist(), certified_region_indices.tolist(), strict=True
        )
    }
    if certified_compartments != cell_compartments:
        raise ValueError(
            "Certified source-region assignments must match actual target-cell associations."
        )

    zone_cell_ids: set[int] = set()
    zone_material_by_id: dict[str, str] = {}
    zone_ids_by_material: dict[str, set[str]] = {}
    for zone in result.zones:
        scope = zone.scope
        material_id = zone.material_id
        if (
            zone.role is not MeshZoneRole.REGION
            or material_id is None
            or material_id not in compartment_ids
            or scope.source_id != mesh.mesh_id
            or scope.source_revision != mesh.numeric_version
            or scope.entity_kind is not MeshingEntityKind.MESH
            or scope.entity_dimension != 3
            or scope.entity_set_id != cell_set.entity_set_id
        ):
            raise ValueError(
                "Bulk material zones must bind declared source compartments and mesh cells."
            )
        identifiers = {int(value) for value in np.asarray(scope.entity_ids).tolist()}
        if zone_cell_ids & identifiers:
            raise ValueError("Bulk material zones must partition cells without overlap.")
        if any(
            cell_compartments.get(identifier) != material_id for identifier in identifiers
        ):
            raise ValueError(
                "Bulk material assignments must match source-region cell associations."
            )
        expected_role = None
        try:
            expected_role = RegionRole(compartments[material_id].material_role)
        except ValueError:
            pass
        if expected_role is not None and zone.region_role is not expected_role:
            raise ValueError(
                "Bulk material zone roles must match the source material role."
            )
        zone_cell_ids.update(identifiers)
        zone_material_by_id[zone.zone_id] = material_id
        zone_ids_by_material.setdefault(material_id, set()).add(zone.zone_id)
    if zone_cell_ids != set(cell_ids) or set(zone_ids_by_material) != compartment_ids:
        raise ValueError(
            "Bulk material zones must cover exactly the source-associated cells."
        )

    source_facets = {
        source.source_entity_key_for_scope(2, int(identifier))
        for identifier in source.scope_entity_ids(2).tolist()
    }
    face_set = mesh.entity_set(2)
    face_associations = tuple(
        value
        for value in result.associations
        if value.target_entity_set_id == face_set.entity_set_id
    )
    if len(face_associations) != 1:
        raise ValueError(
            "Bulk interfaces require one source association for every mesh face."
        )
    face_association = face_associations[0]
    face_targets = np.asarray(face_association.target_global_ids, dtype=np.int64)
    face_ids = np.asarray(face_set.entity_ids, dtype=np.int64)
    if (
        face_association.association_kind is not GeometryAssociationKind.PIECEWISE_LINEAR
        or face_association.source_id != source.source_id
        or face_association.source_revision != source.source_revision
        or not face_association.exact
        or not np.array_equal(np.sort(face_targets), np.sort(face_ids))
        or not np.all(np.asarray(face_association.resolved, dtype=np.bool_))
        or np.any(np.asarray(face_association.ambiguous, dtype=np.bool_))
    ):
        raise ValueError("Bulk face associations must exactly bind this source revision.")
    face_row_by_id: dict[int, int] = {}
    for row, (target, source_id, occurrence, role) in enumerate(
        zip(
            face_targets.tolist(),
            face_association.source_entity_ids,
            face_association.source_occurrence_paths,
            face_association.source_roles,
            strict=True,
        )
    ):
        identity = (source_id, occurrence)
        if (
            (role is GeometrySourceEntityRole.FACET and identity not in source_facets)
            or (
                role is GeometrySourceEntityRole.REGION and identity not in source_regions
            )
            or role
            not in (
                GeometrySourceEntityRole.FACET,
                GeometrySourceEntityRole.REGION,
            )
        ):
            raise ValueError(
                "Bulk faces must associate with actual source facets or regions."
            )
        face_row_by_id[int(target)] = row
    mesh_adjacencies: set[tuple[str, str]] = set()
    for first, second in face_association.source_region_sides:
        if first is None or second is None:
            continue
        if first not in compartment_ids or second not in compartment_ids:
            raise ValueError(
                "Bulk face sides must identify declared source compartments."
            )
        if first != second:
            mesh_adjacencies.add((first, second) if first < second else (second, first))
    if mesh_adjacencies != set(source.compartments.observed_adjacencies):
        raise ValueError(
            "Bulk source-face adjacency must match the observed label adjacency."
        )

    source_interfaces = {
        value.definition.interface_id: value.definition
        for value in source.interfaces.surfaces
    }
    patches_by_name: dict[str, list[MeshPatch]] = {}
    for patch in result.patches:
        patches_by_name.setdefault(patch.name, []).append(patch)
    for interface_id, definition in source_interfaces.items():
        matches = patches_by_name.get(interface_id)
        if matches is None or len(matches) != 1:
            raise ValueError(
                "Bulk mesh must retain each extracted source interface exactly once."
            )
        patch = matches[0]
        endpoints = (definition.first_compartment_id, definition.second_compartment_id)
        side_pair = patch.source_adjacent_region_ids
        if side_pair is None or set(side_pair) != set(endpoints):
            raise ValueError(
                "Bulk interface sides must match its declared source endpoints."
            )
        for identifier in np.asarray(patch.scope.entity_ids, dtype=np.int64).tolist():
            row = face_row_by_id.get(int(identifier))
            if row is None or (
                face_association.source_roles[row] is not GeometrySourceEntityRole.FACET
                or face_association.source_region_sides[row] != side_pair
                or int(np.asarray(face_association.orientations)[row]) not in (-1, 1)
            ):
                raise ValueError(
                    "Bulk interface orientation must match its actual source-facet sides."
                )
        adjacent_materials = {
            zone_material_by_id[identifier]
            for identifier in patch.adjacent_zone_ids
            if identifier in zone_material_by_id
        }
        if adjacent_materials != set(endpoints):
            raise ValueError(
                "Bulk interface adjacency must match its source compartment endpoints."
            )

    for evidence in result.region_boundary_evidence:
        scope = evidence.source_scope
        if (
            scope.source_id != source.source_id
            or scope.source_revision != source.source_revision
            or scope.entity_kind is not MeshingEntityKind.GEOMETRY
            or scope.entity_dimension != 3
            or scope.entity_set_id != source.scope_entity_set_id(3)
            or scope.entity_ids.shape != (1,)
        ):
            raise ValueError(
                "Bulk region-boundary evidence must bind this compartment source."
            )
        compartment_id = source_regions_by_scope_id.get(int(scope.entity_ids[0]))
        if (
            compartment_id is None
            or compartment_id != evidence.source_region_id
            or evidence.material_id != compartment_id
        ):
            raise ValueError(
                "Bulk region-boundary evidence must preserve source material identity."
            )
        try:
            expected_role = RegionRole(compartments[compartment_id].material_role)
        except ValueError:
            expected_role = None
        if expected_role is not None and evidence.role is not expected_role:
            raise ValueError(
                "Bulk region-boundary evidence must preserve the source material role."
            )


@dataclass(frozen=True, slots=True)
class NeurofluidCase:
    case_id: str
    compartment_source: CompartmentMeshingSource
    bulk_mesh: CellMeshingResult
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
        if not isinstance(self.compartment_source, CompartmentMeshingSource):
            raise TypeError("compartment_source must be CompartmentMeshingSource.")
        self.compartment_source.compartments.require_valid_adjacency()
        if not isinstance(self.bulk_mesh, CellMeshingResult):
            raise TypeError("bulk_mesh must be CellMeshingResult.")
        _require_compartment_mesh_binding(self.compartment_source, self.bulk_mesh)
        if not isinstance(self.network, PreparedMetricNetwork):
            raise TypeError("network must be PreparedMetricNetwork.")
        spatial_ids = {
            self.compartment_source.coordinate_contract.spatial_id,
            self.bulk_mesh.coordinate_contract.spatial_id,
            self.network.coordinate_contract.spatial_id,
        }
        if len(spatial_ids) != 1:
            raise ValueError(
                "Compartment source, bulk mesh, and network coordinate contracts differ."
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
                    "source": self.compartment_source.source_id,
                    "segmentation": self.compartment_source.source_revision,
                    "compartments": self.compartment_source.compartments.complex_id,
                    "bulk_mesh": self.bulk_mesh.result_id,
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
            value = np.asarray(raw, dtype=np.float64)
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
        mesh = self.case.bulk_mesh.mesh
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
            self.case.bulk_mesh.coordinate_contract,
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
    state: MixedDimensionalTransportState,
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
