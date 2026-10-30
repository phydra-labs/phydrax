#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Gmsh CAD source adaptation, session import cache, and stable entity resolution."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import jax.numpy as jnp
import numpy as np

from ...geometry._meshing_domain import MeshingDomain
from ...geometry.brep import BRepModel, BRepPartitionResult, BRepSource
from ...geometry.brep._assembly import BRepAffinePlacement
from ...interchange import explore_unique, read_occt_shape
from .._contracts import MeshingFailure, MeshingFailureCategory
from .._planar_bands import PlanarBandResult
from .._scope import MeshingEntityKind, MeshingScope
from .._trace import MeshingStageKind


_BRepMeshingSource = BRepModel | BRepSource | BRepPartitionResult | PlanarBandResult


def _brep_model(source: _BRepMeshingSource, /) -> BRepModel:
    if isinstance(source, PlanarBandResult):
        return source.partition.model
    if isinstance(source, BRepPartitionResult):
        return source.model
    if isinstance(source, BRepSource):
        return source.model
    if isinstance(source, BRepModel):
        return source
    raise TypeError(
        "source must be a BRepModel, BRepSource, BRepPartitionResult, or PlanarBandResult."
    )


def _source_domain(
    source: BRepModel, /, domain: MeshingDomain | None = None
) -> MeshingDomain:
    if domain is None:
        return MeshingDomain.from_brep(source)
    if (
        domain.source_id != source.report.source_id
        or domain.source_revision != source.source_revision
    ):
        raise ValueError("MeshingDomain must bind the supplied BRep owner and revision.")
    return domain


def _scope_source_records(
    source: BRepModel,
    scope: MeshingScope,
    dimension: int,
    /,
    domain: MeshingDomain | None = None,
) -> tuple[tuple[str, tuple[str, ...], int], ...] | None:
    """Resolve numeric scope IDs through their canonical source key and occurrence."""
    if dimension not in range(4):
        return None
    if domain is not None and (
        domain.source_id != source.report.source_id
        or domain.source_revision != source.source_revision
    ):
        return None
    domain = _source_domain(source, domain)
    identifiers = np.asarray(scope.entity_ids, dtype=np.int64)
    if (
        scope.source_id != domain.source_id
        or scope.source_revision != domain.source_revision
        or scope.entity_kind is not MeshingEntityKind.GEOMETRY
        or scope.entity_dimension != dimension
        or scope.entity_set_id != domain.scope_entity_set_id(dimension)
        or not np.all(np.isin(identifiers, domain.scope_entity_ids(dimension)))
    ):
        return None
    if dimension == 3:
        source_rows = domain.region_source_indices
    else:
        source_rows = domain.source_indices[dimension]
    result = []
    for identifier in identifiers:
        source_key, occurrence = domain.source_entity_key_for_scope(
            dimension, int(identifier)
        )
        slot = domain.resolve_entity_id(dimension, source_key, occurrence_path=occurrence)
        result.append((source_key, occurrence, int(source_rows[slot])))
    return tuple(result)


def _scope_indices(
    source: BRepModel,
    scope: MeshingScope,
    dimension: int,
    /,
    domain: MeshingDomain | None = None,
) -> np.ndarray | None:
    domain = _source_domain(source, domain)
    records = _scope_source_records(source, scope, dimension, domain)
    if records is None:
        return None
    try:
        _source_entity_key_banks(source, domain)
    except MeshingFailure:
        return None
    return np.sort(np.asarray([row for _, _, row in records], dtype=np.int64))


def _source_entity_key_banks(
    source: BRepModel, /, domain: MeshingDomain | None = None
) -> tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]:
    """Build canonical source-key tables indexed only by validated definition rows."""
    domain = _source_domain(source, domain)
    counts = (
        source.report.num_vertices,
        source.report.num_edges,
        source.report.num_faces,
        source.topology.num_solids,
    )
    banks = []
    for dimension, count in enumerate(counts):
        if dimension == 3:
            source_rows = domain.region_source_indices
            occurrences = domain.region_source_occurrences
        else:
            source_rows = domain.source_indices[dimension]
            occurrences = domain.source_occurrences[dimension]
        if (
            len(source_rows) != count
            or len(occurrences) != count
            or set(source_rows) != set(range(count))
        ):
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "Gmsh requires one declared occurrence per BRep definition row.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        bank: list[tuple[str, tuple[str, ...]] | None] = [None] * count
        for slot, row in enumerate(source_rows):
            bank[int(row)] = (domain.entity_id(dimension, slot), occurrences[slot])
        if any(key is None for key in bank):
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "Gmsh source entity keys do not form a complete BRep definition-row bijection.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        banks.append(tuple(key for key in bank if key is not None))
    return tuple(banks)


def _source_occurrence_issue(
    source: BRepModel, /, domain: MeshingDomain | None = None
) -> str | None:
    try:
        _source_entity_key_banks(source, domain)
    except MeshingFailure as error:
        return str(error)
    return None


def _source_mapping_issue(
    source: BRepModel, /, domain: MeshingDomain | None = None
) -> str | None:
    graph = source.geometry.assembly_graph
    identity = BRepAffinePlacement.identity()
    if any(use.placement != identity for use in graph.roots) or any(
        use.placement != identity
        for definition in graph.definitions
        for use in definition.uses
    ):
        return "Gmsh cannot preserve nonidentity assembly placements in this import path."
    return _source_occurrence_issue(source, domain)


def _source_entity_ids(
    source: BRepModel, dimension: int, source_rows: Any, /
) -> tuple[str, ...]:
    identifiers = np.asarray(source_rows)
    if identifiers.ndim != 1 or not np.issubdtype(identifiers.dtype, np.integer):
        raise TypeError("Source entity rows must be a one-dimensional integer array.")
    bank = _source_entity_key_banks(source)[dimension]
    if identifiers.size and (np.any(identifiers < 0) or np.any(identifiers >= len(bank))):
        raise ValueError("A source entity row lies outside its BRep definition bank.")
    return tuple(bank[int(row)][0] for row in identifiers)


def _entity_scope(
    source: BRepModel,
    dimension: int,
    source_rows: Any,
    /,
    domain: MeshingDomain | None = None,
) -> MeshingScope:
    domain = _source_domain(source, domain)
    if dimension not in range(4):
        raise ValueError("BRep entity scope dimension must be between zero and three.")
    source_keys = _source_entity_key_banks(source, domain)[dimension]
    scope_ids = domain.scope_entity_ids(dimension)
    scope_keys = domain.scope_entity_key_banks[dimension]
    identifiers_by_key = {
        key: int(identifier)
        for identifier, key in zip(scope_ids, scope_keys, strict=True)
    }
    selected = []
    for value in source_rows:
        row = int(value)
        if row < 0 or row >= len(source_keys):
            raise ValueError("BRep definition row is outside its source entity bank.")
        selected.append(identifiers_by_key[source_keys[row]])
    return MeshingScope.from_geometry(
        domain, np.asarray(selected, dtype=np.int64), dimension=dimension
    )


def _scope_for_entity(source: BRepModel, dimension: int, entity: Any, /) -> MeshingScope:
    return MeshingScope.from_geometry(source, entity, dimension=dimension)


def _source_scale(source: BRepModel, /) -> float:
    return max(float(np.ptp(np.asarray(source.mesh_vertices), axis=0).max()), 1.0)


def _source_entity_reference_banks(
    source: BRepModel,
    provenance: tuple[tuple[str, str], ...],
    /,
) -> tuple[tuple[int, ...], ...]:
    """Resolve exporter declaration references by native source definition row."""
    names = ("vertex", "edge", "face", "solid")
    kinds = ("Ve", "Ed", "Fa", "So")
    counts = (
        source.report.num_vertices,
        source.report.num_edges,
        source.report.num_faces,
        source.topology.num_solids,
    )
    banks: list[list[int | None]] = [[None] * count for count in counts]
    for label, external in provenance:
        name, separator, row_text = label.partition(":")
        if not separator or name not in names:
            continue
        dimension = names.index(name)
        try:
            row = int(row_text)
            reference_text, external_kind = external.split(" ", 1)
            if not reference_text.startswith("TShapes#"):
                raise ValueError("Exporter reference is not a TShapes declaration.")
            reference = int(reference_text.removeprefix("TShapes#"))
        except (ValueError, TypeError):
            row = -1
            reference = -1
            external_kind = ""
        if (
            row < 0
            or row >= counts[dimension]
            or reference <= 0
            or external_kind.split(" ", 1)[0] != kinds[dimension]
            or banks[dimension][row] is not None
        ):
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "Native CAD exporter provenance is not a unique source-definition correspondence.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        banks[dimension][row] = reference
    if any(value is None for bank in banks for value in bank):
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Native CAD exporter provenance omits a source definition.",
            stage=MeshingStageKind.SCOPE_RESOLUTION.value,
        )
    resolved = tuple(
        tuple(value for value in bank if value is not None) for bank in banks
    )
    if any(len(set(bank)) != len(bank) for bank in resolved):
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Distinct source definitions share one native CAD declaration.",
            stage=MeshingStageKind.SCOPE_RESOLUTION.value,
        )
    return resolved


@dataclass(slots=True)
class _CadImport:
    """One digest-verified CAD artifact owned by a Gmsh session."""

    digest: str
    snapshot: Path
    shape: object
    source_entity_references: tuple[tuple[int, ...], ...]
    inventory: tuple[tuple[int, int], ...] | None
    entity_maps: dict[str, _CadEntityMap]
    source_tag_entities: tuple[tuple[int, int, int], ...] | None = None
    source_mapping_mode: bool | None = None


class _CadImportCache:
    """Session CAD imports keyed by source owner, revision, and coordinates.

    Each entry is a private native BRep-text export whose decoded source revision
    was matched exactly before Gmsh import.
    """

    def __init__(self, capacity: int, /) -> None:
        self._capacity = capacity
        self._entries: OrderedDict[tuple[str, str, str], _CadImport] = OrderedDict()
        self._workspace = TemporaryDirectory(prefix="phydrax-gmsh-cad-")
        self._created = 0
        self.hits = 0
        self.misses = 0

    @property
    def revisions(self) -> tuple[str, ...]:
        return tuple(sorted({revision for _, revision, _ in self._entries}))

    def _evict(self, key: tuple[str, str, str], /) -> None:
        entry = self._entries.pop(key)
        entry.snapshot.unlink(missing_ok=True)
        entry.snapshot.with_name(f"{entry.snapshot.stem}-entity-root.brep").unlink(
            missing_ok=True
        )

    def acquire(self, source: BRepModel, /) -> _CadImport:
        report = source.report
        key = (
            report.source_id,
            source.source_revision,
            source.coordinate_contract.spatial_id,
        )
        entry = self._entries.get(key)
        if entry is not None:
            if (
                not entry.snapshot.is_file()
                or hashlib.sha256(entry.snapshot.read_bytes()).hexdigest() != entry.digest
            ):
                self._evict(key)
                entry = None
        if entry is not None:
            self.hits += 1
            self._entries.move_to_end(key)
            return entry

        self.misses += 1
        self._created += 1
        snapshot = Path(self._workspace.name) / f"{self._created}.brep"
        from ...interchange._cad import CadImportPolicy, CadInterchangeError
        from ...interchange._cad_brep_text import read_brep_text, write_brep_text

        tolerance = float(report.curve_surface_tolerance)
        scale = float(report.curve_surface_scale)
        if (
            not np.isfinite(tolerance)
            or tolerance < 0.0
            or not np.isfinite(scale)
            or scale <= 0.0
        ):
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "Gmsh native export requires finite source accuracy and positive source scale.",
                stage=MeshingStageKind.SOURCE_INSPECTION.value,
            )
        policy = CadImportPolicy(
            source.coordinate_contract,
            relative_geometric_tolerance=tolerance / scale,
        )
        try:
            export = write_brep_text(source, snapshot)
            source_entity_references = _source_entity_reference_banks(
                source, export.provenance
            )
            digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            decoded = read_brep_text(
                snapshot,
                policy,
                trusted_root=self._workspace.name,
                source_length_unit=source.coordinate_contract.length_unit,
            )
        except CadInterchangeError as error:
            snapshot.unlink(missing_ok=True)
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                f"Gmsh could not export and re-read this native BRep source: {error}",
                stage=MeshingStageKind.SOURCE_INSPECTION.value,
            ) from error
        except MeshingFailure:
            snapshot.unlink(missing_ok=True)
            raise
        if decoded.source_digest != digest:
            snapshot.unlink(missing_ok=True)
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "The native BRep export changed during revision verification.",
                stage=MeshingStageKind.SOURCE_INSPECTION.value,
            )
        if decoded.model.source_revision != source.source_revision:
            snapshot.unlink(missing_ok=True)
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "The native BRep export does not reconstruct the supplied source revision.",
                stage=MeshingStageKind.SOURCE_INSPECTION.value,
            )
        shape, source_format, snapshot_digest = read_occt_shape(snapshot)
        if snapshot_digest != digest or source_format != "brep":
            snapshot.unlink(missing_ok=True)
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "The verified native BRep export changed before OCCT import.",
                stage=MeshingStageKind.SOURCE_INSPECTION.value,
            )
        entry = _CadImport(
            digest,
            snapshot,
            shape,
            source_entity_references,
            None,
            {},
        )
        self._entries[key] = entry
        while len(self._entries) > self._capacity:
            self._evict(next(iter(self._entries)))
        return entry

    def _entity_root_template(self, entry: _CadImport, /) -> tuple[Path, int, int]:
        """Create a private BRep template whose root can select one TShapes record."""
        references = tuple(
            reference for bank in entry.source_entity_references for reference in bank
        )
        if not references:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "The native CAD export contains no source entity declarations.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        data = entry.snapshot.read_bytes()
        lines = data.splitlines(keepends=True)
        root_index = next(
            (index for index in range(len(lines) - 1, -1, -1) if lines[index].strip()),
            None,
        )
        if root_index is None:
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "The native CAD export has no root declaration.",
                stage=MeshingStageKind.SOURCE_INSPECTION.value,
            )
        root_line = lines[root_index]
        newline = b"\r\n" if root_line.endswith(b"\r\n") else b"\n"
        width = max(
            len(root_line.rstrip(b"\r\n")),
            len(f"+{max(references)} 0"),
        )
        offset = sum(len(line) for line in lines[:root_index])
        replacement = f"+{max(references)} 0".ljust(width).encode("ascii") + newline
        path = entry.snapshot.with_name(f"{entry.snapshot.stem}-entity-root.brep")
        path.write_bytes(data[:offset] + replacement + data[offset + len(root_line) :])
        return path, offset, width

    @staticmethod
    def _select_entity_root(
        path: Path, offset: int, width: int, reference: int, /
    ) -> None:
        with path.open("r+b") as stream:
            stream.seek(offset)
            stream.write(f"+{reference} 0".ljust(width).encode("ascii"))

    def import_shapes(
        self,
        gmsh: Any,
        entry: _CadImport,
        source: BRepModel,
        source_mapping: bool,
        /,
    ) -> None:
        """Import native CAD and retain exact tags for scoped source definitions.

        Scoped plans import each source entity by its exporter declaration and
        accept only the explicit Gmsh OCC fragment correspondence.
        """
        if (
            entry.source_mapping_mode is not None
            and entry.source_mapping_mode != source_mapping
        ):
            entry.inventory = None
            entry.source_tag_entities = None
            entry.entity_maps.clear()
        full_imported: tuple[tuple[int, int], ...] | None = None
        if source_mapping:
            mapping_issue = _source_mapping_issue(source)
            if mapping_issue is not None:
                raise MeshingFailure(
                    MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                    mapping_issue,
                    stage=MeshingStageKind.SCOPE_RESOLUTION.value,
                )
            root_path, root_offset, root_width = self._entity_root_template(entry)
            inputs: list[tuple[int, int, int]] = []
            for dimension, references in enumerate(entry.source_entity_references):
                for row, reference in enumerate(references):
                    self._select_entity_root(
                        root_path, root_offset, root_width, reference
                    )
                    root_imported = tuple(
                        (int(entity_dimension), int(tag))
                        for entity_dimension, tag in gmsh.model.occ.importShapes(
                            str(root_path), highestDimOnly=False
                        )
                    )
                    roots = tuple(
                        tag
                        for entity_dimension, tag in root_imported
                        if entity_dimension == dimension
                    )
                    if len(roots) != 1:
                        raise MeshingFailure(
                            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                            "A source declaration did not import as one identified Gmsh root entity.",
                            stage=MeshingStageKind.SCOPE_RESOLUTION.value,
                        )
                    inputs.append((dimension, row, roots[0]))
            input_entities = tuple((dimension, tag) for dimension, _, tag in inputs)
            if not input_entities or len(set(input_entities)) != len(input_entities):
                raise MeshingFailure(
                    MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                    "Source-directed Gmsh inputs are empty or reuse one provider tag.",
                    stage=MeshingStageKind.SCOPE_RESOLUTION.value,
                )
            gmsh.model.occ.synchronize()
            source_scale = _source_scale(source)
            geometry_tolerance = max(
                float(source.report.curve_surface_tolerance),
                64.0 * np.finfo(np.float64).eps * source_scale,
            )
            input_signatures = tuple(
                _provider_entity_signature(gmsh, dimension, tag)
                for dimension, tag in input_entities
            )
            # Gmsh documents this map by input order; each input is a specific
            # exporter-provenance root, not an entity inferred by traversal.
            output_entities, source_maps = gmsh.model.occ.fragment(
                list(input_entities), [], removeObject=True, removeTool=True
            )
            gmsh.model.occ.synchronize()
            if len(source_maps) != len(inputs):
                raise MeshingFailure(
                    MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                    "Gmsh fragmentation omitted a source-directed input correspondence.",
                    stage=MeshingStageKind.SCOPE_RESOLUTION.value,
                )
            output_set = {
                (int(dimension), int(tag)) for dimension, tag in output_entities
            }
            banks: list[list[int | None]] = [
                [None] * len(references) for references in entry.source_entity_references
            ]
            for input_index, ((dimension, row, _), outputs) in enumerate(
                zip(inputs, source_maps, strict=True)
            ):
                mapped = tuple(
                    (int(output_dimension), int(tag)) for output_dimension, tag in outputs
                )
                same_dimension = tuple(
                    entity for entity in mapped if entity[0] == dimension
                )
                if len(same_dimension) != 1 or any(
                    entity not in output_set for entity in mapped
                ):
                    raise MeshingFailure(
                        MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                        "Gmsh fragmentation did not preserve one source entity as one provider entity.",
                        stage=MeshingStageKind.SCOPE_RESOLUTION.value,
                    )
                # The exporter-linked fragment map establishes identity; measures and
                # bounds only corroborate that its output geometry was preserved.
                output_signature = _provider_entity_signature(
                    gmsh, dimension, same_dimension[0][1]
                )
                if not _provider_geometry_matches(
                    dimension,
                    input_signatures[input_index],
                    output_signature,
                    source_scale,
                    geometry_tolerance,
                ):
                    raise MeshingFailure(
                        MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                        "Gmsh fragmentation changed source entity geometry.",
                        stage=MeshingStageKind.SCOPE_RESOLUTION.value,
                    )
                banks[dimension][row] = same_dimension[0][1]
            if any(value is None for bank in banks for value in bank):
                raise MeshingFailure(
                    MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                    "Gmsh fragmentation omitted a source definition from its provider map.",
                    stage=MeshingStageKind.SCOPE_RESOLUTION.value,
                )
            source_tag_entities = tuple(
                (dimension, row, int(tag))
                for dimension, bank in enumerate(banks)
                for row, tag in enumerate(bank)
                if tag is not None
            )
            mapped_tags = {(dimension, tag) for dimension, _, tag in source_tag_entities}
            if len(mapped_tags) != len(source_tag_entities):
                raise MeshingFailure(
                    MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                    "Distinct source definitions collapsed to one Gmsh provider entity.",
                    stage=MeshingStageKind.SCOPE_RESOLUTION.value,
                )
        else:
            full_imported = tuple(
                (int(dimension), int(tag))
                for dimension, tag in gmsh.model.occ.importShapes(
                    str(entry.snapshot), highestDimOnly=False
                )
            )
            gmsh.model.occ.synchronize()
            source_tag_entities = None
            if not full_imported:
                raise MeshingFailure(
                    MeshingFailureCategory.CONVERSION_FAILED,
                    "Gmsh imported no entities from the verified native BRep snapshot.",
                    stage=MeshingStageKind.SOURCE_INSPECTION.value,
                )
        inventory = tuple(
            (int(dimension), int(tag)) for dimension, tag in gmsh.model.getEntities()
        )
        if full_imported is not None and (
            len(set(full_imported)) != len(full_imported)
            or set(inventory) != set(full_imported)
        ):
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "Gmsh's CAD import result does not cover its full entity inventory.",
                stage=MeshingStageKind.SOURCE_INSPECTION.value,
            )
        if source_tag_entities is not None and set(inventory) != {
            (dimension, tag) for dimension, _, tag in source_tag_entities
        }:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "The source-directed Gmsh model contains entities outside its proven source map.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        if entry.inventory is None:
            entry.inventory = inventory
            entry.source_tag_entities = source_tag_entities
        elif (
            entry.inventory != inventory
            or entry.source_tag_entities != source_tag_entities
        ):
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "Re-importing the verified CAD revision changed its provider correspondence.",
                stage=MeshingStageKind.SOURCE_INSPECTION.value,
            )
        entry.source_mapping_mode = source_mapping

    def entity_map(
        self,
        entry: _CadImport,
        kind: str,
        resolve: Callable[[], _CadEntityMap],
        /,
    ) -> _CadEntityMap:
        resolved = entry.entity_maps.get(kind)
        if resolved is None:
            resolved = resolve()
            entry.entity_maps[kind] = resolved
        return resolved

    def close(self) -> None:
        self._entries.clear()
        self._workspace.cleanup()


def _scope_samples(
    source: BRepModel,
    scope: MeshingScope,
    /,
    domain: MeshingDomain | None = None,
) -> tuple[np.ndarray, ...]:
    """Sample native source definitions resolved by canonical owner/revision keys."""
    records = _scope_source_records(source, scope, scope.entity_dimension, domain)
    if records is None:
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Scope does not select an entity of the supplied BRep owner and revision.",
            stage=MeshingStageKind.SCOPE_RESOLUTION.value,
        )
    if any(occurrence for _, occurrence, _ in records):
        try:
            _source_entity_key_banks(source, _source_domain(source, domain))
        except MeshingFailure as error:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "Gmsh cannot preserve multiple occurrences of one BRep definition.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            ) from error
    rows = tuple(row for _, _, row in records)
    if scope.entity_dimension == 2:
        face_ids = np.asarray(source.triangle_face_ids)
        parameters = np.asarray(source.triangle_parameters)
        result = []
        for face in rows:
            triangles = np.flatnonzero(face_ids == face)
            if not triangles.size:
                raise MeshingFailure(
                    MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                    "Source face has no interior samples for periodic resolution.",
                )
            selected = triangles[
                np.linspace(0, len(triangles) - 1, min(3, len(triangles)), dtype=np.int64)
            ]
            uv = np.mean(parameters[selected], axis=1)
            result.append(np.asarray(source.patches[face].evaluate(jnp.asarray(uv))))
        return tuple(result)
    if scope.entity_dimension == 3:
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Solid scopes require a complete source-to-volume import correspondence.",
            stage=MeshingStageKind.SCOPE_RESOLUTION.value,
        )
    geometry = source.geometry
    if scope.entity_dimension == 0:
        vertices = np.asarray(geometry.vertex_points, dtype=np.float64)
        return tuple(vertices[row : row + 1] for row in rows)
    result = []
    for edge in rows:
        curve_index = geometry.edge_curves[edge]
        if curve_index < 0:
            vertex = geometry.edge_vertices[edge][0]
            point = np.asarray(geometry.vertex_points[vertex], dtype=np.float64)
            result.append(np.repeat(point[None, :], 3, axis=0))
            continue
        first, last = np.asarray(geometry.edge_ranges[edge], dtype=np.float64)
        parameters = np.linspace(float(first), float(last), 5)[1:-1]
        sampled = geometry.curves[curve_index].evaluate(jnp.asarray(parameters))
        result.append(np.asarray(sampled, dtype=np.float64))
    return tuple(result)


def _resolve_entities(
    source: BRepModel,
    scope: MeshingScope,
    /,
    *,
    cad_entities: _CadEntityMap | None,
    domain: MeshingDomain | None = None,
) -> tuple[int, ...]:
    if cad_entities is None:
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Gmsh has no checked source-to-import correspondence for this scope.",
            stage=MeshingStageKind.SCOPE_RESOLUTION.value,
        )
    return cad_entities.tags_for_scope(source, scope, domain=domain)


@dataclass(frozen=True, slots=True)
class _CadEntityMap:
    source_id: str
    source_revision: str
    source_entity_keys: tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]
    vertex_to_point: tuple[int, ...] = ()
    face_to_surface: tuple[int, ...] = ()
    solid_to_volume: tuple[int, ...] = ()
    edge_to_curve: tuple[int, ...] = ()

    def tags_for_scope(
        self,
        source: BRepModel,
        scope: MeshingScope,
        /,
        *,
        domain: MeshingDomain | None = None,
    ) -> tuple[int, ...]:
        if (
            self.source_id != source.report.source_id
            or self.source_revision != source.source_revision
        ):
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "The imported Gmsh entity map belongs to another BRep owner or revision.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        records = _scope_source_records(source, scope, scope.entity_dimension, domain)
        if records is None:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "Scope does not select an entity of the supplied BRep owner and revision.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        tag_banks = (
            self.vertex_to_point,
            self.edge_to_curve,
            self.face_to_surface,
            self.solid_to_volume,
        )
        entity_keys = self.source_entity_keys[scope.entity_dimension]
        tags = tag_banks[scope.entity_dimension]
        if len(tags) != len(entity_keys):
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "The imported Gmsh tag bank is incomplete for the supplied BRep source.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        result = []
        for source_key, occurrence, row in records:
            if (
                row >= len(entity_keys)
                or row >= len(tags)
                or entity_keys[row] != (source_key, occurrence)
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                    "Canonical BRep scope key is absent from the imported Gmsh correspondence.",
                    stage=MeshingStageKind.SCOPE_RESOLUTION.value,
                )
            result.append(tags[row])
        return tuple(result)


def _source_tag_banks(
    source_tag_entities: tuple[tuple[int, int, int], ...] | None,
    source: BRepModel,
    /,
) -> tuple[tuple[int, ...], ...]:
    """Build source-row tag banks from explicit provider input/output records."""
    if source_tag_entities is None:
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Gmsh did not retain the source-directed provider correspondence.",
            stage=MeshingStageKind.SCOPE_RESOLUTION.value,
        )
    counts = (
        source.report.num_vertices,
        source.report.num_edges,
        source.report.num_faces,
        source.topology.num_solids,
    )
    banks: list[list[int | None]] = [[None] * count for count in counts]
    for dimension, row, tag in source_tag_entities:
        if (
            dimension not in range(4)
            or row < 0
            or row >= counts[dimension]
            or tag <= 0
            or banks[dimension][row] is not None
        ):
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "Gmsh source correspondence contains an invalid or repeated definition row.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        banks[dimension][row] = tag
    if any(value is None for bank in banks for value in bank):
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Gmsh source correspondence omits a definition row.",
            stage=MeshingStageKind.SCOPE_RESOLUTION.value,
        )
    resolved = tuple(
        tuple(value for value in bank if value is not None) for bank in banks
    )
    if any(len(set(bank)) != len(bank) for bank in resolved):
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Distinct source definitions map to one Gmsh provider tag.",
            stage=MeshingStageKind.SCOPE_RESOLUTION.value,
        )
    return resolved


def _provider_entity_signature(
    gmsh: Any, dimension: int, tag: int, /
) -> tuple[float, np.ndarray, np.ndarray]:
    measure = 0.0 if dimension == 0 else float(gmsh.model.occ.getMass(dimension, tag))
    center = np.asarray(
        gmsh.model.getValue(0, tag, [])
        if dimension == 0
        else gmsh.model.occ.getCenterOfMass(dimension, tag),
        dtype=np.float64,
    ).reshape(3)
    bounds = np.asarray(
        gmsh.model.getBoundingBox(dimension, tag), dtype=np.float64
    ).reshape(6)
    return measure, center, bounds


def _provider_geometry_matches(
    dimension: int,
    source_signature: tuple[float, np.ndarray, np.ndarray],
    target_signature: tuple[float, np.ndarray, np.ndarray],
    scale: float,
    tolerance: float,
    /,
) -> bool:
    source_measure, source_center, source_bounds = source_signature
    target_measure, target_center, target_bounds = target_signature
    roundoff = 64.0 * np.finfo(np.float64).eps * scale
    coordinate_tolerance = max(tolerance, roundoff)
    if (
        not np.isfinite(source_measure)
        or not np.isfinite(target_measure)
        or not np.all(np.isfinite(source_center))
        or not np.all(np.isfinite(target_center))
        or not np.all(np.isfinite(source_bounds))
        or not np.all(np.isfinite(target_bounds))
        or np.max(np.abs(source_center - target_center), initial=0.0)
        > coordinate_tolerance
        or np.max(np.abs(source_bounds - target_bounds), initial=0.0)
        > coordinate_tolerance
    ):
        return False
    if dimension == 0:
        return True
    measure_tolerance = coordinate_tolerance * scale ** (dimension - 1)
    measure_tolerance += (
        64.0
        * np.finfo(np.float64).eps
        * max(scale**dimension, abs(source_measure), abs(target_measure))
    )
    return abs(source_measure - target_measure) <= measure_tolerance


def _match_entities(
    gmsh: Any, dimension: int, samples: Any, candidates: Any, tolerance: float, /
) -> tuple[int, ...]:
    result = []
    for points in samples:
        matches = []
        for tag in candidates:
            if tag in result:
                continue
            closest, _ = gmsh.model.getClosestPoint(
                dimension, tag, np.asarray(points).reshape(-1)
            )
            closest = np.asarray(closest).reshape((-1, 3))
            if (
                closest.shape == points.shape
                and np.max(np.linalg.norm(closest - points, axis=1)) <= tolerance
                and gmsh.model.isInside(dimension, tag, closest.reshape(-1))
                == len(points)
            ):
                matches.append(tag)
        if len(matches) != 1:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "BRep entity does not resolve uniquely to the imported Gmsh geometry.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        result.append(matches[0])
    return tuple(result)


def _validate_source_solids(source: BRepModel, shape: Any, /) -> tuple[Any, ...]:
    """Validate solids without attributing TopExp traversal order to source rows."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
    from OCP.TopoDS import TopoDS

    topology = source.topology
    if (
        any(not faces for faces in topology.solid_faces)
        or any(len(owners) not in (1, 2) for owners in topology.face_solids)
        or any(
            orientations[0] == orientations[1]
            for face_index, owners in enumerate(topology.face_solids)
            if len(owners) == 2
            for orientations in (
                tuple(
                    topology.solid_face_orientations[solid][
                        topology.solid_faces[solid].index(face_index)
                    ]
                    for solid in owners
                ),
            )
        )
    ):
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Source solids require nonempty manifold boundaries with opposite shared-face orientations.",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
        )
    solids = explore_unique(shape, TopAbs_SOLID, TopoDS.Solid)
    if len(solids) != topology.num_solids or any(
        not BRepCheck_Analyzer(solid).IsValid() for solid in solids
    ):
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Reopened BRep solids do not match the valid imported solid inventory.",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
        )
    scale = _source_scale(source)
    volume_tolerance = 1.0e-12 * scale**3
    contact_tolerance = 1.0e-9 * scale
    for left_index, left in enumerate(solids):
        # Shared faces are compared by TShape identity within this imported model.
        left_faces = explore_unique(left, TopAbs_FACE, TopoDS.Face)
        for right in solids[left_index + 1 :]:
            distance = BRepExtrema_DistShapeShape(left, right)
            distance.Perform()
            if not distance.IsDone():
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SOURCE,
                    "BRep solid contact validation did not complete.",
                    stage=MeshingStageKind.SOURCE_INSPECTION.value,
                )
            right_faces = explore_unique(right, TopAbs_FACE, TopoDS.Face)
            shared_face = any(
                first.IsSame(second) for first in left_faces for second in right_faces
            )
            if not shared_face and float(distance.Value()) <= contact_tolerance:
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SOURCE,
                    "BRep solids touch without a complete shared topological face.",
                    stage=MeshingStageKind.SOURCE_INSPECTION.value,
                )
            common = BRepAlgoAPI_Common(left, right)
            common.Build()
            if not common.IsDone():
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SOURCE,
                    "BRep solid overlap validation did not complete.",
                    stage=MeshingStageKind.SOURCE_INSPECTION.value,
                )
            properties = GProp_GProps()
            BRepGProp.VolumeProperties_s(common.Shape(), properties)
            if abs(float(properties.Mass())) > volume_tolerance:
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SOURCE,
                    "BRep solid interiors overlap; provider-side ownership fragmentation is prohibited.",
                    stage=MeshingStageKind.SOURCE_INSPECTION.value,
                )
    return tuple(solids)


def _validate_imported_entity_inventory(
    gmsh: Any, tag_banks: tuple[tuple[int, ...], ...], /
) -> None:
    for dimension, tags in enumerate(tag_banks):
        actual = {int(tag) for _, tag in gmsh.model.getEntities(dimension)}
        if len(tags) != len(set(tags)) or set(tags) != actual:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                f"Gmsh imported dimension-{dimension} tags are not a source-bank bijection.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )


def _validate_edge_vertex_incidence(
    gmsh: Any,
    source: BRepModel,
    vertex_tags: tuple[int, ...],
    edge_tags: tuple[int, ...],
    /,
) -> None:
    for edge_index, curve in enumerate(edge_tags):
        _, downward = gmsh.model.getAdjacencies(1, curve)
        actual = {int(value) for value in np.asarray(downward, dtype=np.int64)}
        expected = {
            vertex_tags[int(vertex)]
            for vertex in source.geometry.edge_vertices[edge_index]
        }
        if actual != expected:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "Gmsh curve endpoints differ from source BRep edge-vertex incidence.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )


def _validate_face_edge_incidence(
    gmsh: Any,
    source: BRepModel,
    edge_tags: tuple[int, ...],
    face_tags: tuple[int, ...],
    /,
) -> None:
    for face_index, surface in enumerate(face_tags):
        boundary = gmsh.model.getBoundary(
            [(2, surface)], combined=False, oriented=False, recursive=False
        )
        actual = {abs(int(tag)) for dimension, tag in boundary if dimension == 1}
        expected = {
            edge_tags[int(edge)] for edge in source.topology.face_edges[face_index]
        }
        if len(actual) != len(boundary) or actual != expected:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "Gmsh surfaces differ from source BRep face-edge incidence.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
    for edge_index, curve in enumerate(edge_tags):
        upward, _ = gmsh.model.getAdjacencies(1, curve)
        actual = {int(value) for value in np.asarray(upward, dtype=np.int64)}
        expected = {
            face_tags[int(face)] for face in source.topology.edge_faces[edge_index]
        }
        if actual != expected:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "Gmsh curves differ from source BRep edge-face incidence.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )


def _resolve_planar_cad_entity_map(
    gmsh: Any,
    source: BRepModel,
    source_tag_entities: tuple[tuple[int, int, int], ...] | None,
    /,
) -> _CadEntityMap:
    if source.topology.num_solids:
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Strict planar CAD meshing requires a zero-solid BRep.",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
        )
    domain = _source_domain(source)
    source_entity_keys = _source_entity_key_banks(source, domain)
    vertex_tags, edge_tags, face_tags, _ = _source_tag_banks(source_tag_entities, source)
    tag_banks = (vertex_tags, edge_tags, face_tags, ())
    _validate_imported_entity_inventory(gmsh, tag_banks)
    _validate_edge_vertex_incidence(gmsh, source, vertex_tags, edge_tags)
    _validate_face_edge_incidence(gmsh, source, edge_tags, face_tags)
    return _CadEntityMap(
        source.report.source_id,
        source.source_revision,
        source_entity_keys,
        vertex_to_point=vertex_tags,
        face_to_surface=face_tags,
        edge_to_curve=edge_tags,
    )


def _volume_boundaries(
    gmsh: Any, volume_tags: tuple[int, ...], /
) -> tuple[dict[int, frozenset[int]], dict[tuple[int, int], int]]:
    boundary_faces: dict[int, frozenset[int]] = {}
    boundary_orientations: dict[tuple[int, int], int] = {}
    for volume in volume_tags:
        occurrences = gmsh.model.getBoundary(
            [(3, volume)], combined=False, oriented=True, recursive=False
        )
        if any(dimension != 2 for dimension, _ in occurrences):
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "An imported Gmsh volume has a non-surface boundary occurrence.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        tags = tuple(abs(int(tag)) for _, tag in occurrences)
        if len(set(tags)) != len(tags):
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "An imported Gmsh volume repeats a boundary surface.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        boundary_faces[volume] = frozenset(tags)
        boundary_orientations.update(
            (
                ((volume, abs(int(tag))), 1 if int(tag) > 0 else -1)
                for _, tag in occurrences
            )
        )
    return boundary_faces, boundary_orientations


def _resolve_cad_entity_map(
    gmsh: Any,
    source: BRepModel,
    shape: Any,
    source_tag_entities: tuple[tuple[int, int, int], ...] | None,
    /,
) -> _CadEntityMap:
    _validate_source_solids(source, shape)
    domain = _source_domain(source)
    source_entity_keys = _source_entity_key_banks(source, domain)
    vertex_tags, edge_tags, face_tags, imported_volumes = _source_tag_banks(
        source_tag_entities, source
    )
    _validate_imported_entity_inventory(
        gmsh, (vertex_tags, edge_tags, face_tags, imported_volumes)
    )
    _validate_edge_vertex_incidence(gmsh, source, vertex_tags, edge_tags)
    _validate_face_edge_incidence(gmsh, source, edge_tags, face_tags)
    boundary_faces, boundary_orientations = _volume_boundaries(gmsh, imported_volumes)

    solid_to_volume = imported_volumes
    if set(solid_to_volume) != set(boundary_faces):
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "The source-directed solid map is not a bijection with imported Gmsh volumes.",
            stage=MeshingStageKind.SCOPE_RESOLUTION.value,
        )
    for solid, faces in enumerate(source.topology.solid_faces):
        expected = frozenset(face_tags[face] for face in faces)
        if boundary_faces[solid_to_volume[solid]] != expected:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "Gmsh volume boundaries differ from source-directed solid face incidence.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )

    for face_index, surface in enumerate(face_tags):
        source_owners = source.topology.face_solids[face_index]
        expected_volumes = {solid_to_volume[owner] for owner in source_owners}
        upward, _ = gmsh.model.getAdjacencies(2, surface)
        actual_volumes = {int(value) for value in np.asarray(upward, dtype=np.int64)}
        if actual_volumes != expected_volumes:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "Gmsh surface-to-volume adjacency differs from source BRep incidence.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        if len(source_owners) == 2:
            first, second = tuple(expected_volumes)
            if (
                boundary_orientations[first, surface]
                == boundary_orientations[second, surface]
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                    "A shared Gmsh surface has inconsistent volume orientations.",
                    stage=MeshingStageKind.SCOPE_RESOLUTION.value,
                )
    return _CadEntityMap(
        source.report.source_id,
        source.source_revision,
        source_entity_keys,
        vertex_to_point=vertex_tags,
        face_to_surface=face_tags,
        solid_to_volume=tuple(solid_to_volume),
        edge_to_curve=edge_tags,
    )
