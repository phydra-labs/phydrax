#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Request-owned bindings for identical cells in successive certified carriers."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction
from typing import assert_never, Literal, TYPE_CHECKING

import jax
import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._exact_algebraic_geometry import RadicalElement


if TYPE_CHECKING:
    from ..geometry._embedding_state import MeshCertificateLimits
    from ..geometry._global_embedding import GlobalEmbeddingCertificate
    from ._cell_geometry import CellGeometrySpec
    from ._cell_geometry_validity import CellValidityCertificate
    from ._cell_mesh import CellMesh
    from ._coordinate_budget import CoordinateEnclosureBudget
    from ._coordinate_enclosure import ExactScalar


def cell_certificate_source_content_id(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    validity: CellValidityCertificate,
    embedding: GlobalEmbeddingCertificate | None,
    /,
) -> str:
    """Bind certificate payloads and their numerical source, including bound arrays."""
    from ..lifecycle._meshing_source_records import stored_record_content_id
    from ._cell_geometry_validity import cell_geometry_id

    return canonical_fingerprint(
        {
            "kind": "cell-certificate-source-content",
            "mesh": (mesh.mesh_id, array_tree_fingerprint(mesh)),
            "geometry": (
                cell_geometry_id(geometry),
                stored_record_content_id(geometry, excluded=""),
            ),
            "validity": stored_record_content_id(validity, excluded=""),
            "embedding": None
            if embedding is None
            else stored_record_content_id(embedding, excluded=""),
        }
    )


@dataclass(frozen=True, slots=True)
class _CellGeometryPredecessorSource:
    """Inputs for atomic restriction preparation, not an earned proof or token.

    Each consumption prepares and authenticates a new predecessor on the
    consumer's ledger. No predecessor, admission or validity verdict is carried.
    """

    mesh: CellMesh
    geometry: CellGeometrySpec
    validity: CellValidityCertificate


@dataclass(frozen=True, slots=True)
class CellGeometryPredecessor:
    """Bind unchanged target rows to authenticated source-cell certificates.

    ``source_limits`` is the complete limits object that admitted the source
    embedding certificate; its ID must equal that certificate's binding.
    """

    source_mesh: CellMesh
    source_geometry: CellGeometrySpec
    source_validity: CellValidityCertificate
    source_embedding: GlobalEmbeddingCertificate | None
    source_limits: MeshCertificateLimits | None
    source_rows: tuple[int, ...]
    target_mesh: CellMesh
    target_geometry: CellGeometrySpec
    content_id: str

    def require_bound(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        """Authenticate content, reusing only this request's unchanged owner tree."""
        if budget is not None and _earned_predecessor_content(
            self, mesh, geometry, budget
        ):
            return
        if not _predecessor_content_is_bound(self, mesh, geometry, budget=budget):
            raise ValueError(
                "CellGeometryPredecessor.content_id must bind its source certificates and target rows; got changed content"
            )
        _require_predecessor_source_binding(self)
        if budget is not None:
            _earn_predecessor_content(self, mesh, geometry, budget)

    @classmethod
    def from_stored_fields(
        cls: type[CellGeometryPredecessor],
        value: CellGeometryPredecessor,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> CellGeometryPredecessor:
        """Revalidate stored bindings, never issue a new embedding theorem.

        An active native request supplies the actual target ticket for the
        complete content scan. Outside execution this is record-integrity
        validation only; downstream theorem owners must admit their own proof.
        """
        from .._meshcore import current_native_execution_scope
        from .._validation import canonical_identifier
        from ..geometry._embedding_state import MeshCertificateLimits
        from ..geometry._global_embedding import GlobalEmbeddingCertificate
        from ._cell_geometry import CellGeometrySpec
        from ._cell_geometry_validity import CellValidityCertificate
        from ._cell_mesh import CellMesh
        from ._coordinate_budget import CoordinateEnclosureBudget

        if type(value) is not cls:
            raise TypeError(
                "CellGeometryPredecessor.stored_fields must be the exact owning record type"
            )
        expected = (
            (value.source_mesh, CellMesh),
            (value.target_mesh, CellMesh),
            (value.source_geometry, CellGeometrySpec),
            (value.target_geometry, CellGeometrySpec),
            (value.source_validity, CellValidityCertificate),
        )
        if any(not isinstance(owner, kind) for owner, kind in expected) or (
            value.source_embedding is not None
            and not isinstance(value.source_embedding, GlobalEmbeddingCertificate)
        ) or (
            value.source_limits is not None
            and not isinstance(value.source_limits, MeshCertificateLimits)
        ):
            raise TypeError(
                "CellGeometryPredecessor.stored_fields must retain actual mesh, geometry and certificate owners"
            )
        source_count = sum(block.cell_count for block in value.source_mesh.blocks)
        target_count = sum(block.cell_count for block in value.target_mesh.blocks)
        if type(value.source_rows) is not tuple or len(value.source_rows) != target_count or any(
            type(row) is not int or row < -1 or row >= source_count
            for row in value.source_rows
        ):
            raise ValueError(
                "CellGeometryPredecessor.source_rows must bind each target to a source row or -1"
            )
        canonical_identifier(value.content_id, "CellGeometryPredecessor.content_id")
        rebuilt = cls(*(getattr(value, field.name) for field in fields(cls)))
        scope = current_native_execution_scope()
        if budget is not None or scope is None:
            rebuilt.require_bound(
                rebuilt.target_mesh, rebuilt.target_geometry, budget=budget
            )
        else:
            with scope.ticket_view() as ticket:
                view = CoordinateEnclosureBudget(
                    ticket,
                    maximum_work_units=scope.limits.maximum_work_units,
                    maximum_memory_bytes=scope.limits.maximum_managed_bytes,
                )
                with view.activate():
                    rebuilt.require_bound(
                        rebuilt.target_mesh, rebuilt.target_geometry, budget=view
                    )
        return rebuilt


def _require_predecessor_source_binding(predecessor: CellGeometryPredecessor) -> None:
    predecessor.source_validity.require_bound(
        predecessor.source_geometry, mesh=predecessor.source_mesh
    )
    if not predecessor.source_validity.all_certified:
        raise ValueError(
            "CellGeometryPredecessor.source_validity must certify every imported source cell; got undecided source cells"
        )
    _require_source_limits(predecessor.source_embedding, predecessor.source_limits)
    if predecessor.source_embedding is not None:
        predecessor.source_embedding.binding.require(
            predecessor.source_mesh, predecessor.source_geometry
        )
        if (
            predecessor.source_embedding.validity_certificate_id
            != predecessor.source_validity.certificate_id
        ):
            raise ValueError(
                "CellGeometryPredecessor.source_embedding must bind its source validity certificate; got another validity ID"
            )


@dataclass(frozen=True, slots=True)
class _PredecessorContentAdmission:
    """An issuer-paid digest and strong owners, never an ID-only authority.

    Tuple edges and builtin scalar leaves cannot change. Every dataclass field
    edge, including static fields, and every Fraction's integer pair is checked
    on reuse. Numerical leaves must be the actual immutable JAX owners whose
    bytes the issuer hashed; even readonly NumPy banks take the full-hash route.
    """

    predecessor: CellGeometryPredecessor
    mesh: CellMesh
    geometry: CellGeometrySpec
    owner_fields: tuple[tuple[object, type, tuple[str, ...], tuple[object, ...]], ...]


def _earned_predecessor_content(
    predecessor: CellGeometryPredecessor,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    budget: CoordinateEnclosureBudget,
    /,
) -> bool:
    admission = budget.predecessor_content_admission_cache.get(id(predecessor))
    budget.reserve(1)
    if (
        admission is None
        or admission.predecessor is not predecessor
        or admission.mesh is not mesh
        or admission.geometry is not geometry
    ):
        return False
    return _immutable_owner_fields_match(admission.owner_fields, budget)


def _immutable_owner_fields_match(
    owner_fields: tuple[tuple[object, type, tuple[str, ...], tuple[object, ...]], ...],
    budget: CoordinateEnclosureBudget,
) -> bool:
    for owner, owner_type, names, values in owner_fields:
        budget.reserve(1 + len(names))
        if (
            type(owner) is not owner_type
            or (
                not is_dataclass(owner)
                and type(owner) is not Fraction
                and tuple(vars(owner)) != names
            )
        ) or any(
            getattr(owner, name) is not value
            for name, value in zip(names, values, strict=True)
        ):
            return False
    return True


def _earn_predecessor_content(
    predecessor: CellGeometryPredecessor,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    budget: CoordinateEnclosureBudget,
    /,
) -> None:
    """Seal supported current fields only after their original content proof."""
    if _earned_predecessor_content(predecessor, mesh, geometry, budget):
        return
    owner_fields = _snapshot_immutable_owner_fields((predecessor, mesh, geometry), budget)
    if owner_fields is not None:
        admission = _PredecessorContentAdmission(predecessor, mesh, geometry, owner_fields)
        budget.retain_owner(admission)
        budget.predecessor_content_admission_cache[id(predecessor)] = admission


def _snapshot_immutable_owner_fields(
    owners: tuple[object, ...],
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[object, type, tuple[str, ...], tuple[object, ...]], ...] | None:
    """Snapshot supported immutable-array graphs; mutable arrays have no fast path."""
    from .._model._artifacts import artifact_value_codec_for
    from ..exterior._form_type import FormType, FormValueSpec

    with budget.temporary_scope():
        budget.reserve(0, 512)
        pending: list[object] = list(owners)
        seen: dict[int, object] = {}
        owner_fields: list[
            tuple[object, type, tuple[str, ...], tuple[object, ...]]
        ] = []
        while pending:
            owner = pending.pop()
            if seen.get(id(owner)) is owner:
                continue
            budget.reserve(1, 128)
            seen[id(owner)] = owner
            if owner is None or type(owner) in (bool, int, float, complex, str, bytes):
                continue
            if isinstance(owner, jax.Array):
                continue
            if type(owner) is tuple:
                budget.reserve(0, 64 + 16 * len(owner))
                pending.extend(owner)
                continue
            if type(owner) is Fraction:
                names = ("numerator", "denominator")
            elif is_dataclass(owner) and not isinstance(owner, type):
                names = tuple(member.name for member in fields(owner))
            elif (
                (artifact_value_codec_for(owner) is not None or type(owner) in (FormType, FormValueSpec))
                and hasattr(owner, "__dict__")
            ):
                # Closed form descriptors and registered array-free value codecs.
                # Snapshot actual attributes, not a claimed value/source ID.
                names = tuple(vars(owner))
            else:
                # Unsupported owners have no earned immutability contract.
                # In particular flags.writeable=False does not freeze NumPy.
                return None
            budget.reserve(len(names), 128 + 32 * len(names))
            values = tuple(getattr(owner, name) for name in names)
            owner_fields.append((owner, type(owner), names, values))
            pending.extend(values)
        result = tuple(owner_fields)
        budget.retain_owner(result)
        return result


def _reserve_predecessor_content(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    source_embedding: GlobalEmbeddingCertificate | None,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    budget: CoordinateEnclosureBudget,
    /,
) -> None:
    """Keep the original full source-authentication scan charge at its issuer."""
    meshes = source_mesh, target_mesh
    geometries = source_geometry, target_geometry
    budget.reserve(
        sum(mesh.coordinates.size for mesh in meshes)
        + sum(
            block.vertices.size + block.global_ids.size
            for mesh in meshes
            for block in mesh.blocks
        )
        + sum(geometry.coordinates.size for geometry in geometries)
        + sum(route.size for geometry in geometries for route in geometry.geometry_dofs)
        + (0 if source_embedding is None else
           len(source_embedding.evaluated_checks) + len(source_embedding.findings)),
        2048,
    )


def _predecessor_content_is_bound(
    predecessor: CellGeometryPredecessor,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> bool:
    """Use an earned actual-owner proof or preserve the original exact digest."""
    if budget is not None:
        if _earned_predecessor_content(predecessor, mesh, geometry, budget):
            return True
        _reserve_predecessor_content(
            predecessor.source_mesh, predecessor.source_geometry,
            predecessor.source_embedding, mesh, geometry, budget,
        )
    return predecessor.content_id == _predecessor_content_id(
        predecessor.source_mesh,
        predecessor.source_geometry,
        predecessor.source_validity,
        predecessor.source_embedding,
        predecessor.source_limits,
        mesh,
        geometry,
        predecessor.source_rows,
    )


def _require_source_limits(
    embedding: GlobalEmbeddingCertificate | None,
    limits: MeshCertificateLimits | None,
    /,
) -> None:
    """Refuse carried limits that did not admit the bound source embedding."""
    if limits is not None and (
        embedding is None or embedding.binding.limits_id != limits.limits_id
    ):
        raise ValueError(
            "CellGeometryPredecessor.source_limits must be the limits bound by its source embedding; "
            f"got {limits.limits_id!r}, embedding binds "
            f"{None if embedding is None else embedding.binding.limits_id!r}"
        )


def _predecessor_content_id(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    source_validity: CellValidityCertificate,
    source_embedding: GlobalEmbeddingCertificate | None,
    source_limits: MeshCertificateLimits | None,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    source_rows: tuple[int, ...],
    /,
) -> str:
    from ..lifecycle._meshing_source_records import stored_record_content_id
    from ._cell_geometry_validity import cell_geometry_id

    return canonical_fingerprint(
        {
            "kind": "cell-geometry-predecessor",
            "source": cell_certificate_source_content_id(
                source_mesh, source_geometry, source_validity, source_embedding
            ),
            "source_limits": None if source_limits is None else source_limits.limits_id,
            "target": (target_mesh.mesh_id, array_tree_fingerprint(target_mesh)),
            "target_geometry": (
                cell_geometry_id(target_geometry),
                stored_record_content_id(target_geometry, excluded=""),
            ),
            "source_rows": source_rows,
        }
    )


def _coefficient_content_key(
    value: ExactScalar, /
) -> tuple[int, int] | tuple[Literal["radical"], tuple[tuple[int, int, int], ...]]:
    """Preserve rational identity tokens and encode reduced radical terms."""
    match value:
        case Fraction():
            return value.numerator, value.denominator
        case RadicalElement():
            return (
                "radical",
                tuple(
                    (radicand, coefficient.numerator, coefficient.denominator)
                    for radicand, coefficient in value.terms
                ),
            )
        case _:
            assert_never(value)


def _cell_content_keys(
    mesh: CellMesh, geometry: CellGeometrySpec, budget: CoordinateEnclosureBudget, /,
    *, selected_ids: np.ndarray | None = None,
) -> tuple[tuple[int, str], ...]:
    """Bind each cell's source map, ordered incidence, and carrier bytes."""
    from ._coordinate_enclosure import (
        coordinate_source_signature,
        prepared_coordinate_source_bank,
    )

    bank = prepared_coordinate_source_bank(geometry, budget=budget)
    elements, routes, _ = geometry.resolve(
        mesh, exact_source_prepared=True, budget=budget
    )
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    binding = (
        None if geometry.exact_source is None else geometry.exact_source.source_id,
        None if geometry.periodic_source is None else geometry.periodic_source.source_id,
    )
    keys: list[tuple[int, str]] = []
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        selected = (
            np.arange(block.cell_count)
            if selected_ids is None else np.flatnonzero(np.isin(identifiers, selected_ids))
        )
        if not selected.size:
            continue
        definition = canonical_fingerprint(
            (coordinate_source_signature(element), array_tree_fingerprint(element))
        )
        vertex_rows = np.asarray(block.vertices, dtype=np.int64)
        dof_rows = np.asarray(route, dtype=np.int64)
        for row in selected.tolist():
            vertices = vertex_rows[row : row + 1].reshape(-1)
            dofs = dof_rows[row : row + 1].reshape(-1)
            local = tuple(bank[int(dofs[index])] for index in range(dofs.size))
            budget.reserve(vertices.size * (mesh.ambient_dimension + 1) + len(local))
            keys.append(
                (
                    int(identifiers[row]),
                    canonical_fingerprint(
                        {
                            "kind": block.cell_kind,
                            "vertices": vertex_ids[vertices],
                            "carrier": coordinates[vertices],
                            "element": definition,
                            "source_binding": binding,
                            "coefficients": tuple(
                                tuple(_coefficient_content_key(value) for value in row)
                                for row in local
                            ),
                        }
                    ),
                )
            )
    return tuple(keys)


def _matching_definition_ids(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    shared_ids: np.ndarray,
    budget: CoordinateEnclosureBudget,
    /,
) -> np.ndarray:
    """Exclude unequal block definitions before examining cell-local content.

    Kind, source binding and the full declared element definition are literal
    members of every existing content key. Unequal members cannot import a
    cell, even when a descendant reuses its parent's global ID.
    """
    from ._coordinate_enclosure import coordinate_source_signature

    def binding(geometry: CellGeometrySpec) -> tuple[str | None, str | None]:
        return (
            None if geometry.exact_source is None else geometry.exact_source.source_id,
            None
            if geometry.periodic_source is None
            else geometry.periodic_source.source_id,
        )

    if binding(source_geometry) != binding(target_geometry):
        return np.empty((0,), dtype=np.int64)
    source_elements, _, _ = source_geometry._resolve_layout(source_mesh)
    target_elements, _, _ = target_geometry._resolve_layout(target_mesh)
    count = sum(block.cell_count for block in (*source_mesh.blocks, *target_mesh.blocks))
    with budget.temporary_scope():
        budget.reserve(count, 1024 + 32 * count)
        groups: dict[tuple[str, str], list[np.ndarray]] = {}
        for block, element in zip(source_mesh.blocks, source_elements, strict=True):
            definition = canonical_fingerprint(
                (coordinate_source_signature(element), array_tree_fingerprint(element))
            )
            ids = np.asarray(block.global_ids, dtype=np.int64)
            groups.setdefault((block.cell_kind, definition), []).append(
                ids[np.isin(ids, shared_ids)]
            )
        source_ids = {
            definition: np.concatenate(parts) for definition, parts in groups.items()
        }
        candidates = []
        for block, element in zip(target_mesh.blocks, target_elements, strict=True):
            definition = canonical_fingerprint(
                (coordinate_source_signature(element), array_tree_fingerprint(element))
            )
            ids = source_ids.get((block.cell_kind, definition))
            if ids is not None:
                candidates.append(
                    np.intersect1d(ids, np.asarray(block.global_ids, dtype=np.int64))
                )
        return (
            np.unique(np.concatenate(candidates))
            if candidates
            else np.empty((0,), dtype=np.int64)
        )


def prepare_cell_geometry_predecessor(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    source_validity: CellValidityCertificate,
    source_embedding: GlobalEmbeddingCertificate | None,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    /,
    *,
    source_limits: MeshCertificateLimits | None = None,
    budget: CoordinateEnclosureBudget,
) -> CellGeometryPredecessor:
    """Authenticate content correspondence without rebuilding cell polynomials.

    ``source_limits`` carries the original limits of a source embedding so a
    restricted-root import can rebuild its exact binding.
    """
    source_validity.require_bound(source_geometry, mesh=source_mesh)
    if not source_validity.all_certified:
        raise ValueError(
            "CellGeometryPredecessor.source_validity must certify every imported source cell; got undecided source cells"
        )
    _require_source_limits(source_embedding, source_limits)
    from ._cell_geometry_validity import cell_geometry_id

    record = target_geometry.restriction_source
    source_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in source_mesh.blocks]
    )
    target_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in target_mesh.blocks]
    )
    budget.reserve(source_ids.size + target_ids.size, source_ids.nbytes + target_ids.nbytes)
    restricted = (
        record is not None
        and record.source_topology_id == source_mesh.topology_id
        and record.source_geometry_id == cell_geometry_id(source_geometry)
    )
    shared_ids = np.intersect1d(source_ids, target_ids) if restricted else None
    if shared_ids is not None and shared_ids.size:
        shared_ids = _matching_definition_ids(
            source_mesh,
            source_geometry,
            target_mesh,
            target_geometry,
            shared_ids,
            budget,
        )
    if shared_ids is not None and not shared_ids.size:
        # No ID with an equal block definition can have an identical content
        # key. The restricted-root theorem separately authenticates the source
        # and proves every child chart.
        source = {}
        rows = (-1,) * target_ids.size
    elif shared_ids is not None:
        # Equal IDs and equal block definitions are necessary, not sufficient.
        # Keep their full cell-local comparisons and original flattened rows.
        source_positions = {int(value): row for row, value in enumerate(source_ids)}
        source = {
            key: source_positions[key[0]]
            for key in _cell_content_keys(
                source_mesh, source_geometry, budget, selected_ids=shared_ids
            )
        }
        target = {
            key[0]: source.get(key, -1)
            for key in _cell_content_keys(
                target_mesh, target_geometry, budget, selected_ids=shared_ids
            )
        }
        rows = tuple(target.get(int(identifier), -1) for identifier in target_ids)
    else:
        source = {
            key: row
            for row, key in enumerate(
                _cell_content_keys(source_mesh, source_geometry, budget)
            )
        }
        rows = tuple(
            source.get(key, -1)
            for key in _cell_content_keys(target_mesh, target_geometry, budget)
        )
    _reserve_predecessor_content(
        source_mesh, source_geometry, source_embedding,
        target_mesh, target_geometry, budget,
    )
    predecessor = CellGeometryPredecessor(
        source_mesh,
        source_geometry,
        source_validity,
        source_embedding,
        source_limits,
        rows,
        target_mesh,
        target_geometry,
        _predecessor_content_id(
            source_mesh,
            source_geometry,
            source_validity,
            source_embedding,
            source_limits,
            target_mesh,
            target_geometry,
            rows,
        ),
    )
    _require_predecessor_source_binding(predecessor)
    _earn_predecessor_content(predecessor, target_mesh, target_geometry, budget)
    budget.retain_basis((tuple(source), rows))
    budget.retain_owner(predecessor)
    return predecessor
