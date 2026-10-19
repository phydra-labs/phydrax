#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.discretization._cell_geometry as cell_geometry_owner
import phydrax.discretization._cell_mesh as cell_mesh_owner

from .._fingerprint import canonical_fingerprint
from ..geometry._domain_coverage import DomainCoverageCertificate
from ..typing import checked
from . import (
    _collective_organization as collective_organization_owner,
    _initial_certification as initial_certification_owner,
    _organization_records as record_owner,
    _publication_lowering as publication_owner,
    _result as result_owner,
)

# Module references defer symbol resolution across mutually dependent execution owners.
from ._scope import (
    local_logical_lookup,
    MeshingEntityKind,
    MeshingScope,
    MeshScopeProjection,
    selected_ids,
)


@checked
def lower_region_meshing_evidence(
    original: result_owner.CellMeshingResult,
    target: cell_mesh_owner.CellMesh,
    geometry: cell_geometry_owner.CellGeometrySpec,
    zones: tuple[record_owner.MeshZone, ...],
    patches: tuple[record_owner.MeshPatch, ...],
    coverage: DomainCoverageCertificate,
    /,
) -> record_owner.RegionMeshingEvidence:
    """Bind source material definitions to accepted global banks and their local views."""
    evidence = original.region_evidence
    if evidence is None:
        raise ValueError(
            "lower_region_meshing_evidence.original.region_evidence must bind source regions; got missing region evidence"
        )
    cells, regions, facets, interfaces, orientations = (
        record_owner._logical_region_inventory(
            target, evidence.domain, evidence.interface_definitions
        )
    )
    local_cells = jnp.asarray(target.entity_set(3).entity_ids, dtype=jnp.int64)
    local_regions, valid = local_logical_lookup(cells, regions, local_cells)
    if not np.all(valid):
        raise ValueError(
            "lower_region_meshing_evidence.target must select cells from the global inventory; got an absent target cell"
        )
    assignments = tuple(evidence.domain.region_ids[index] for index in local_regions)
    facet_records = record_owner.local_interface_records(
        target, facets, interfaces, orientations, evidence.interface_definitions
    )
    if len(original.zones) != len(zones) or len(original.patches) != len(patches):
        raise ValueError(
            "lower_region_meshing_evidence.organization must retain all source definitions; got missing definitions"
        )
    zone_ids = {
        before.zone_id: after.zone_id
        for before, after in zip(original.zones, zones, strict=True)
    }
    patch_ids = {
        before.patch_id: after.patch_id
        for before, after in zip(original.patches, patches, strict=True)
    }
    renewed = record_owner.RegionMeshingEvidence(
        target,
        geometry,
        evidence.source_revision,
        evidence.source_complex_id,
        assignments,
        tuple(
            (region, zone_ids[identifier])
            for region, identifier in evidence.region_zone_ids
        ),
        tuple(
            (interface, patch_ids[identifier])
            for interface, identifier in evidence.interface_patch_ids
        ),
        evidence.interface_definitions,
        facet_records,
        evidence.domain,
        coverage,
        certificate_limits=evidence.certificate_limits,
    )
    renewed.require_current(target, zones, patches, geometry=geometry)
    return renewed


@checked
def retained_mesh_organization_projections(
    result: result_owner.CellMeshingResult,
    /,
) -> tuple[
    tuple[tuple[str, MeshScopeProjection], ...] | None,
    tuple[tuple[str, record_owner.MeshAttributeProjection], ...] | None,
]:
    """Recover retained scope and attribute projections without global recomputation."""
    storage = result.mesh.storage
    if storage is None:
        return None, None
    evidence = result.collective_evidence
    if evidence is None or evidence.evidence_id != storage.evidence_id:
        raise ValueError(
            "retained_mesh_organization_projections.collective_evidence must bind retained storage; got missing or different collective evidence"
        )
    scopes: list[tuple[str, MeshScopeProjection]] = []
    for family, records in (
        ("patch", result.patches),
        ("zone", result.zones),
        ("label", result.labels),
        ("attribute", result.attributes),
    ):
        for index, record in enumerate(records):
            receipt = record.scope.projection
            name = f"organization/{family}/{index}/membership"
            if receipt is None or (
                receipt.membership_name != name
                or receipt.evidence_id != evidence.evidence_id
                or receipt.source_id != result.mesh.mesh_id
                or receipt.source_revision != result.mesh.numeric_version
                or receipt.scope_id != record.scope.scope_id
                or receipt.members is not record.scope.global_entity_ids
            ):
                raise ValueError(
                    "retained_mesh_organization_projections.scopes must retain prepared scope banks; got missing or mismatched scope projections"
                )
            receipt.local_mask(result.mesh)
            scopes.append((name, receipt))
    attributes: list[tuple[str, record_owner.MeshAttributeProjection]] = []
    for index, record in enumerate(result.attributes):
        receipt = record._attribute_projection
        name = f"organization/attribute/{index}/values"
        if receipt is None or (
            receipt.value_name != name
            or receipt.global_values is not record.global_values
            or receipt.values_content_id != record.values_content_id
        ):
            raise ValueError(
                "retained_mesh_organization_projections.attributes must retain prepared value banks; got missing or mismatched value projections"
            )
        actual = receipt.local_values(record.scope)
        if (
            actual.shape != record.values.shape
            or actual.dtype != record.values.dtype
            or not np.array_equal(
                np.asarray(actual),
                np.asarray(record.values),
            )
        ):
            raise ValueError(
                "retained_mesh_organization_projections.attributes must match local value banks; got different local values"
            )
        identity = canonical_fingerprint(
            {
                "kind": "mesh-attribute",
                "name": record.name,
                "role": record.role.value,
                "scope": record.scope.scope_id,
                "unit": None if record.unit is None else record.unit.unit_id,
                "values": record.values_content_id,
            }
        )
        if identity != record.attribute_id:
            raise ValueError(
                "retained_mesh_organization_projections.attributes must retain value-content binding; got a different attribute identity"
            )
        attributes.append((name, receipt))
    return tuple(scopes), tuple(attributes)


@checked
def prepare_mesh_organization_scopes(
    original: result_owner.CellMeshingResult
    | initial_certification_owner.InitialCollectiveMeshEvidence,
    evidence: result_owner.CollectiveMeshEvidence
    | initial_certification_owner.InitialCollectiveMeshEvidence,
    publication: publication_owner.PublicationProjection,
    source_revision: str,
    /,
) -> tuple[tuple[str, MeshScopeProjection], ...]:
    """Prepare all global scientific scope identities before owner-local construction."""

    match original:
        case initial_certification_owner.InitialCollectiveMeshEvidence():
            groups = collective_organization_owner.initial_organization_definitions(
                original
            )
        case result_owner.CellMeshingResult():
            groups = (
                ("patch", original.patches),
                ("zone", original.zones),
                ("label", original.labels),
                ("attribute", original.attributes),
            )
        case _:
            raise TypeError(
                "prepare_mesh_organization_scopes.original must be an initial evidence or cell meshing result; "
                f"got {type(original).__name__}."
            )
    prepared = tuple(
        (
            name,
            MeshScopeProjection(
                evidence,
                publication,
                source_revision,
                record.scope.entity_dimension,
                name,
            ),
        )
        for family, records in groups
        for index, record in enumerate(records)
        for name in (f"organization/{family}/{index}/membership",)
    )
    result = []
    for name, receipt in prepared:
        if name.startswith("organization/zone/"):
            exclusive = tuple(
                other
                for other_name, other in prepared
                if other_name.startswith("organization/zone/")
                and other_name != name
                and other.dimension == receipt.dimension
            )
            receipt = MeshScopeProjection(
                evidence,
                publication,
                source_revision,
                receipt.dimension,
                name,
                exclusive_scopes=exclusive,
            )
        result.append((name, receipt))
    return tuple(result)


@checked
def lower_mesh_organization(
    original: result_owner.CellMeshingResult
    | initial_certification_owner.InitialCollectiveMeshEvidence,
    target: cell_mesh_owner.CellMesh,
    logical_arrays: Mapping[str, Array],
    /,
    *,
    scope_projections: tuple[tuple[str, MeshScopeProjection], ...] | None = None,
    attribute_projections: tuple[tuple[str, record_owner.MeshAttributeProjection], ...]
    | None = None,
) -> tuple[
    tuple[record_owner.MeshPatch, ...],
    tuple[record_owner.MeshZone, ...],
    tuple[record_owner.MeshLabel, ...],
    tuple[record_owner.MeshAttribute, ...],
]:
    """Lower scientific definitions using certified global membership banks."""

    match original:
        case initial_certification_owner.InitialCollectiveMeshEvidence():
            patch_group, zone_group, label_group, attribute_group = (
                collective_organization_owner.initial_organization_definitions(original)
            )
            patch_records, zone_records = patch_group[1], zone_group[1]
            label_records, attribute_records = label_group[1], attribute_group[1]
        case result_owner.CellMeshingResult():
            patch_records, zone_records = original.patches, original.zones
            label_records, attribute_records = original.labels, original.attributes
        case _:
            raise TypeError(
                "lower_mesh_organization.original must be an initial evidence or cell meshing result; "
                f"got {type(original).__name__}."
            )
    if target.storage is None:
        raise ValueError(
            "lower_mesh_organization.target.storage must contain certified logical storage; got missing storage"
        )
    storage = target.storage
    projections = {} if scope_projections is None else dict(scope_projections)
    local_bases: dict[int, MeshingScope] = {}

    def scope_for(collection: str, index: int, scope: MeshingScope) -> MeshingScope:
        membership_name = f"organization/{collection}/{index}/membership"
        if scope_projections is not None:
            receipt = projections[membership_name]
            source_banks = dict(receipt.publication.source_arrays)
            if logical_arrays[membership_name] is not source_banks[membership_name]:
                raise ValueError(
                    "lower_mesh_organization.scope_projections must retain accepted membership banks; got a different membership bank"
                )
            return MeshingScope(
                target.mesh_id,
                target.numeric_version,
                MeshingEntityKind.MESH,
                scope.entity_dimension,
                target.entity_set(scope.entity_dimension).entity_set_id,
                receipt.members,
                local_mesh=target,
                _projection=receipt,
            )
        degree = scope.entity_dimension
        count = storage.global_entity_counts[degree]
        name = (
            "vertex_global_ids"
            if degree == 0
            else "cell_global_ids"
            if degree == target.topological_dimension
            else f"entity_global_ids_{degree}"
        )
        ids = logical_arrays[name]
        membership = logical_arrays[f"organization/{collection}/{index}/membership"]
        if membership.shape != ids.shape or membership.dtype != jnp.bool_:
            raise ValueError(
                "lower_mesh_organization.membership must match its logical entity table; got incompatible membership axes"
            )
        if bool(jax.device_get(jnp.any(membership[count:]))):
            raise ValueError(
                "lower_mesh_organization.membership must select only logical entity rows; got selected padding"
            )
        members = selected_ids(ids[:count], membership[:count])
        basis = local_bases.get(degree)
        lowered = MeshingScope(
            target.mesh_id,
            target.numeric_version,
            MeshingEntityKind.MESH,
            degree,
            target.entity_set(degree).entity_set_id,
            members,
            local_mesh=target if basis is None else None,
            _local_basis=basis,
        )
        local_bases.setdefault(degree, lowered)
        return lowered

    zones = tuple(
        record_owner.MeshZone(
            zone.name,
            zone.role,
            scope_for("zone", index, zone.scope),
            material_id=zone.material_id,
            region_role=zone.region_role,
        )
        for index, zone in enumerate(zone_records)
    )
    record_owner.validate_mesh_zones(zones)
    zone_ids = {
        before.zone_id: after.zone_id
        for before, after in zip(zone_records, zones, strict=True)
    }
    patches = tuple(
        record_owner.MeshPatch(
            patch.name,
            scope_for("patch", index, patch.scope),
            connected=patch.connected,
            adjacent_zone_ids=tuple(
                zone_ids[identifier] for identifier in patch.adjacent_zone_ids
            ),
            source_adjacent_region_ids=patch.source_adjacent_region_ids,
        )
        for index, patch in enumerate(patch_records)
    )
    labels = tuple(
        record_owner.MeshLabel(label.name, scope_for("label", index, label.scope))
        for index, label in enumerate(label_records)
    )
    attributes = []
    attribute_receipts = (
        {} if attribute_projections is None else dict(attribute_projections)
    )
    for index, attribute in enumerate(attribute_records):
        scope = scope_for("attribute", index, attribute.scope)
        if attribute_projections is not None:
            receipt = attribute_receipts[f"organization/attribute/{index}/values"]
            if (
                logical_arrays[receipt.value_name]
                is not dict(receipt.scope_projection.publication.source_arrays)[
                    receipt.value_name
                ]
            ):
                raise ValueError(
                    "lower_mesh_organization.attribute_projections must retain accepted value banks; got a different value bank"
                )
            attributes.append(
                record_owner.MeshAttribute(
                    attribute.name,
                    attribute.role,
                    scope,
                    receipt.global_values,
                    unit=attribute.unit,
                    _projection=receipt,
                )
            )
            continue
        degree = scope.entity_dimension
        name = (
            "vertex_global_ids"
            if degree == 0
            else "cell_global_ids"
            if degree == target.topological_dimension
            else f"entity_global_ids_{degree}"
        )
        ids = logical_arrays[name][: storage.global_entity_counts[degree]]
        rows = jnp.argsort(ids, stable=True)
        ordered_ids = ids[rows]
        positions = jnp.searchsorted(ordered_ids, scope.global_entity_ids)
        values = logical_arrays[f"organization/attribute/{index}/values"]
        if (
            values.shape[0] != logical_arrays[name].shape[0]
            or values.shape[1:] != attribute.component_shape
        ):
            raise ValueError(
                "lower_mesh_organization.attribute_values must match logical entity and component axes; got incompatible value axes"
            )
        attributes.append(
            record_owner.MeshAttribute(
                attribute.name,
                attribute.role,
                scope,
                values[rows[positions]],
                unit=attribute.unit,
            )
        )
    return patches, zones, labels, tuple(attributes)


@checked
def validate_local_mesh_organization(
    original: result_owner.CellMeshingResult
    | initial_certification_owner.InitialCollectiveMeshEvidence,
    target: cell_mesh_owner.CellMesh,
    logical_arrays: Mapping[str, Array],
    patches: tuple[record_owner.MeshPatch, ...],
    zones: tuple[record_owner.MeshZone, ...],
    labels: tuple[record_owner.MeshLabel, ...],
    attributes: tuple[record_owner.MeshAttribute, ...],
    /,
    *,
    scope_projections: tuple[tuple[str, MeshScopeProjection], ...] | None = None,
    attribute_projections: tuple[tuple[str, record_owner.MeshAttributeProjection], ...]
    | None = None,
) -> None:
    """Validate definitions and lowered membership, including empty local views."""
    expected = lower_mesh_organization(
        original,
        target,
        logical_arrays,
        scope_projections=scope_projections,
        attribute_projections=attribute_projections,
    )
    actual = (patches, zones, labels, attributes)
    for provided, required in zip(actual, expected, strict=True):
        if len(provided) != len(required):
            raise ValueError(
                "validate_local_mesh_organization.records must contain every definition once; got missing or duplicate definitions"
            )
        for record, witness in zip(provided, required, strict=True):
            if type(record) is not type(
                witness
            ) or record_owner.organization_definition_id(
                record
            ) != record_owner.organization_definition_id(witness):
                raise ValueError(
                    "validate_local_mesh_organization.records must match scientific definitions; got an inconsistent definition"
                )
            record.scope.require_same_local_placement(
                witness.scope, preserve_global_identity=scope_projections is not None
            )
            if isinstance(record, record_owner.MeshAttribute) and isinstance(
                witness, record_owner.MeshAttribute
            ):
                for values, expected_values in (
                    (record.global_values, witness.global_values),
                    (record.values, witness.values),
                ):
                    if (
                        attribute_projections is not None
                        and values is record.global_values
                    ):
                        if values is not expected_values:
                            raise ValueError(
                                "validate_local_mesh_organization.attributes must retain accepted global value banks; got a different global bank object"
                            )
                        continue
                    if (
                        values.shape != expected_values.shape
                        or values.dtype != expected_values.dtype
                        or not bool(jax.device_get(jnp.all(values == expected_values)))
                    ):
                        raise ValueError(
                            "validate_local_mesh_organization.attributes must match accepted values; got inconsistent attribute values"
                        )


__all__: list[str] = []
