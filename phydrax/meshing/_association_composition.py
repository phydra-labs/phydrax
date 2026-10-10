#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Composition of original PLC namespace banks through the canonical PLC engine."""

from __future__ import annotations

from typing import final, NamedTuple, TYPE_CHECKING

import equinox as eqx
import jax
import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import nonnegative_integer, positive_integer
from ..discretization import CellMesh
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..geometry._mapped_source_support import MappedSourceFacetAuthority
from ..typing import Dim, HostInt64, parse
from ._association import (
    _PlcBankContext,
    _target_dimension,
    AssociationPropagationError,
    GeometryAssociation,
    GeometryAssociationKind,
    PlcAssociationTransfer,
    PlcEntityClasses,
)
from ._plc_mapped_support import MappedPlcSupport, prepare_mapped_plc_support


if TYPE_CHECKING:
    from ..geometry._mesh_certificates import (
        DomainCoverageCertificate,
        GlobalEmbeddingCertificate,
    )
    from ._certification import MeshCertificationPreparedEvidence
    from ._certification_inputs import MeshCertificationInputs
    from ._lineage import MeshLineage
    from ._result import CellMeshingResult


class _BankFacetDim(Dim):
    """Original facet identifiers of one authoritative namespace."""


class _BankRegionDim(Dim, minimum=1):
    """Explicit source-region to shared material-region mapping."""


class _BankOffsetDim(Dim, minimum=3):
    """One boundary offset per explicitly declared source namespace."""


class ComposedEntityClasses(NamedTuple):
    """Every original namespace claim, plus collision-free joint class codes."""

    dimensions: np.ndarray
    namespace_dimensions: tuple[np.ndarray, ...]
    namespace_indices: tuple[np.ndarray, ...]
    resolved: np.ndarray
    codes: np.ndarray


@final
class ComposedAssociationTransfer(StrictModule, NonTrainableState):
    """Compose namespace-disjoint PLC plans without changing their source identities.

    One complete material plan owns whole-domain coverage. Each other plan owns
    its original primitive bank and an explicit material map. Lineage propagation
    and continuous support use ``PlcAssociationTransfer`` itself; this record only
    dispatches banks and composes their independently checked claims. Numerical
    maps remain leaves, and the common query/byte admission is never multiplied
    by the number of banks.
    """

    __strict_contract__ = True
    plans: tuple[PlcAssociationTransfer, ...]
    facet_indices: HostInt64[_BankFacetDim]
    facet_offsets: HostInt64[_BankOffsetDim]
    region_maps: HostInt64[_BankRegionDim]
    region_offsets: HostInt64[_BankOffsetDim]
    reference_meshes: tuple[CellMesh | None, ...]
    reference_geometries: tuple[CellGeometrySpec | None, ...]
    material_namespace: tuple[str, str] = eqx.field(static=True)
    maximum_support_queries: int = eqx.field(static=True)
    maximum_data_bytes: int = eqx.field(static=True)
    preparation_work_units: int = eqx.field(static=True)
    material_index: int = eqx.field(static=True)
    class_radices: tuple[int, ...] = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        plans: tuple[PlcAssociationTransfer, ...],
        facet_indices: np.ndarray,
        facet_offsets: np.ndarray,
        region_maps: np.ndarray,
        region_offsets: np.ndarray,
        reference_meshes: tuple[CellMesh | None, ...],
        reference_geometries: tuple[CellGeometrySpec | None, ...],
        /,
        *,
        material_namespace: tuple[str, str],
        maximum_support_queries: int,
        maximum_data_bytes: int,
        preparation_work_units: int = 0,
    ) -> None:
        if (
            type(plans) is not tuple
            or len(plans) < 2
            or any(type(plan) is not PlcAssociationTransfer for plan in plans)
        ):
            raise TypeError(
                "Composition requires an explicit tuple of original PLC namespace plans."
            )
        if any(plan.domain.ambient_dimension != 3 for plan in plans):
            raise ValueError(
                "Shared material coverage requires three-dimensional PLC banks."
            )
        if (
            type(reference_meshes) is not tuple
            or type(reference_geometries) is not tuple
            or len(reference_meshes) != len(plans)
            or len(reference_geometries) != len(plans)
        ):
            raise ValueError(
                "Every original namespace must explicitly declare its mapped source authority or absence."
            )
        for mesh, geometry, plan in zip(
            reference_meshes, reference_geometries, plans, strict=True
        ):
            if (mesh is None) != (geometry is None):
                raise ValueError(
                    "Mapped primitive authority requires its original mesh and coordinate expression together."
                )
            if mesh is not None and geometry is not None:
                if (
                    type(mesh) is not CellMesh
                    or type(geometry) is not CellGeometrySpec
                    or mesh.entity_set(2).count != plan.facet_regions.shape[0]
                ):
                    raise ValueError(
                        "Mapped facet banks must bind their exact original reference entities."
                    )
                geometry.resolve(mesh)
        raw = tuple(
            np.asarray(value)
            for value in (facet_indices, facet_offsets, region_maps, region_offsets)
        )
        if any(
            not np.issubdtype(value.dtype, np.integer) or value.ndim != 1 for value in raw
        ):
            raise TypeError(
                "Original namespace banks require one-dimensional integer leaves."
            )
        (
            original_facets,
            original_facet_offsets,
            original_regions,
            original_region_offsets,
        ) = raw
        for offsets, values in (
            (original_facet_offsets, original_facets),
            (original_region_offsets, original_regions),
        ):
            if (
                offsets.shape != (len(plans) + 1,)
                or offsets[0] != 0
                or offsets[-1] != values.size
                or np.any(offsets[1:] < offsets[:-1])
            ):
                raise ValueError(
                    "Every original namespace requires exact bounded numerical bank offsets."
                )
        keys = tuple((plan.domain.source_id, plan.source_revision) for plan in plans)
        if len(set(keys)) != len(keys) or material_namespace not in keys:
            raise ValueError(
                "Original namespaces must be disjoint and explicitly identify the material owner."
            )
        if len({plan.coordinate_contract.spatial_id for plan in plans}) != 1:
            raise ValueError(
                "Composed original banks must use the same physical coordinate frame."
            )
        queries = positive_integer(maximum_support_queries, "maximum_support_queries")
        byte_limit = positive_integer(maximum_data_bytes, "maximum_data_bytes")
        preparation = nonnegative_integer(
            preparation_work_units, "preparation_work_units"
        )
        if queries > min(plan.maximum_support_queries for plan in plans):
            raise ValueError(
                "Composition cannot widen any original source-support allowance."
            )
        order = sorted(range(len(plans)), key=keys.__getitem__)
        ordered = tuple(plans[index] for index in order)
        material_index = order.index(keys.index(material_namespace))
        material = ordered[material_index]
        facets: list[HostInt64[_BankFacetDim]] = []
        maps: list[HostInt64[_BankRegionDim]] = []
        radices: list[int] = []
        capacity = 1
        retained = 0
        for index, plan in zip(order, ordered, strict=True):
            raw_facets = original_facets[
                original_facet_offsets[index] : original_facet_offsets[index + 1]
            ]
            raw_map = original_regions[
                original_region_offsets[index] : original_region_offsets[index + 1]
            ]
            facet = parse(
                np.array(raw_facets, dtype=np.int64, copy=True),
                HostInt64[_BankFacetDim],
                "facet_indices",
            )
            regions = parse(
                np.array(raw_map, dtype=np.int64, copy=True),
                HostInt64[_BankRegionDim],
                "region_map",
            )
            if (
                facet.shape != (plan.facet_regions.shape[0],)
                or np.any(facet < 0)
                or np.unique(facet).size != facet.size
            ):
                raise ValueError(
                    "Every original facet must have one explicit distinct source identifier."
                )
            if (
                regions.shape != plan.region_indices.shape
                or np.unique(regions).size != regions.size
                or np.any(~np.isin(regions, material.region_indices))
            ):
                raise ValueError(
                    "Every original region must inject into the complete material namespace."
                )
            for local, label in enumerate(regions):
                position = np.flatnonzero(material.region_indices == label)
                if (
                    position.size != 1
                    or plan.domain.region_ids[local]
                    != material.domain.region_ids[int(position[0])]
                ):
                    raise ValueError(
                        "Composed material maps must preserve exact authoritative physical names."
                    )
            facet.setflags(write=False)
            regions.setflags(write=False)
            facets.append(facet)
            maps.append(regions)
            radix = 1 + sum(
                table.size
                for table in (
                    plan.vertex_indices,
                    plan.edge_indices,
                    facet,
                    plan.region_indices,
                )
            )
            capacity *= radix
            if capacity - 1 > np.iinfo(np.int64).max:
                raise ValueError(
                    "The exact joint namespace class registry exceeds int64 capacity."
                )
            radices.append(radix)
            retained += (
                facet.nbytes
                + regions.nbytes
                + sum(
                    table.nbytes
                    for table in (
                        plan.domain.vertices,
                        plan.domain.facets,
                        plan.domain.facet_regions,
                        plan.source_vertices,
                        plan.edge_vertices,
                        plan.vertex_indices,
                        plan.edge_indices,
                        plan.triangle_vertices,
                        plan.triangle_facets,
                        plan.facet_regions,
                        plan.region_indices,
                    )
                )
            )
        seen_buffers: set[int] = set()
        for value in jax.tree_util.tree_leaves((reference_meshes, reference_geometries)):
            if (
                isinstance(value, (np.ndarray, jax.Array))
                and id(value) not in seen_buffers
            ):
                retained += value.nbytes
                seen_buffers.add(id(value))
        retained += 16 * (len(plans) + 1)
        if not np.array_equal(maps[material_index], material.region_indices):
            raise ValueError(
                "The complete material owner requires the identity material map."
            )
        if retained > byte_limit:
            raise ValueError(
                "Original namespace plans exceed their declared retained-byte allowance."
            )
        self.plans = ordered
        combined_facets = parse(
            np.concatenate(facets), HostInt64[_BankFacetDim], "facet_indices"
        )
        combined_regions = parse(
            np.concatenate(maps), HostInt64[_BankRegionDim], "region_maps"
        )
        facet_bounds = parse(
            np.cumsum(np.asarray((0, *(value.size for value in facets)), dtype=np.int64)),
            HostInt64[_BankOffsetDim],
            "facet_offsets",
        )
        region_bounds = parse(
            np.cumsum(np.asarray((0, *(value.size for value in maps)), dtype=np.int64)),
            HostInt64[_BankOffsetDim],
            "region_offsets",
        )
        for value in (combined_facets, combined_regions, facet_bounds, region_bounds):
            value.setflags(write=False)
        self.facet_indices = combined_facets
        self.facet_offsets = facet_bounds
        self.region_maps = combined_regions
        self.region_offsets = region_bounds
        self.reference_meshes = tuple(reference_meshes[index] for index in order)
        self.reference_geometries = tuple(reference_geometries[index] for index in order)
        self.material_namespace = material_namespace
        self.maximum_support_queries = queries
        self.maximum_data_bytes = byte_limit
        self.preparation_work_units = preparation
        self.material_index = material_index
        self.class_radices = tuple(radices)
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "composed-association-transfer",
                "plans": tuple(plan.transfer_id for plan in ordered),
                "maps": array_tree_fingerprint(
                    (
                        self.facet_indices,
                        self.facet_offsets,
                        self.region_maps,
                        self.region_offsets,
                    )
                ),
                "mapped_sources": tuple(
                    None
                    if mesh is None or geometry is None
                    else (mesh.mesh_id, cell_geometry_id(geometry))
                    for mesh, geometry in zip(
                        self.reference_meshes, self.reference_geometries, strict=True
                    )
                ),
                "material_namespace": material_namespace,
                "maximum_support_queries": queries,
                "maximum_data_bytes": byte_limit,
                "preparation_work_units": preparation,
            }
        )

    def _contexts(self, source: CellMeshingResult, /) -> tuple[_PlcBankContext, ...]:
        material = self.plans[self.material_index]
        keys = {(plan.domain.source_id, plan.source_revision) for plan in self.plans}
        if any(
            value.association_kind is not GeometryAssociationKind.PIECEWISE_LINEAR
            or (value.source_id, value.source_revision) not in keys
            for value in source.associations
        ):
            raise ValueError(
                "Composition requires every association to bind one exact original namespace."
            )
        owners = tuple(
            value
            for value in source.associations
            if (value.source_id, value.source_revision) == self.material_namespace
            and _target_dimension(source.mesh, value) == 3
        )
        if len(owners) != 1:
            raise ValueError(
                "Composition requires one complete original material-cell association."
            )
        owner = owners[0]
        ids = np.asarray(source.mesh.entity_set(3).entity_ids, dtype=np.int64)
        if not np.array_equal(
            np.sort(np.asarray(owner.target_global_ids)), np.sort(ids)
        ) or np.any(np.asarray(owner.source_dimensions) != 3):
            raise ValueError(
                "Composition material ownership must cover every actual source cell."
            )
        cell_regions = np.asarray(owner.source_indices, dtype=np.int64)[
            owner.target_rows(ids)
        ]
        certificate = source.certification
        if certificate is None or not certificate.passed:
            raise ValueError(
                "Composition requires the actual accepted scientific source certificate."
            )
        allocation = sum(
            source.mesh.entity_set(dimension).count for dimension in range(4)
        ) * (41 * len(self.plans) + 17)
        if allocation > self.maximum_data_bytes:
            raise ValueError(
                "Composed source classes exceed their declared temporary-byte allowance."
            )
        proof = prepare_mapped_plc_support(
            source.mesh,
            source.geometry,
            material.domain,
            cell_regions,
            material.region_indices,
            maximum_support_queries=self.maximum_support_queries,
            embedding=certificate.embedding,
            validity=source.audit.validity,
            certificate_limits=certificate.request.limits,
            certification_request=certificate.request,
        )
        work = [self.maximum_support_queries]
        contexts: list[_PlcBankContext] = []
        for index in range(len(self.plans)):
            context = _PlcBankContext(
                proof,
                self.facet_indices[
                    self.facet_offsets[index] : self.facet_offsets[index + 1]
                ],
                self.region_maps[
                    self.region_offsets[index] : self.region_offsets[index + 1]
                ],
                work,
            )
            mesh, geometry = (
                self.reference_meshes[index],
                self.reference_geometries[index],
            )
            if mesh is not None and geometry is not None:
                plan = self.plans[index]
                # The material proof's ledger owns this exact root preparation too.
                context.mapped_facets = MappedSourceFacetAuthority.prepare(
                    mesh,
                    geometry,
                    plan.triangle_vertices,
                    plan.triangle_facets,
                    ledger=proof.ledger,
                )
            contexts.append(context)
        return tuple(contexts)

    def source_associations(
        self, source: CellMeshingResult, /
    ) -> tuple[GeometryAssociation, ...]:
        contexts = self._contexts(source)
        for plan, context in zip(self.plans, contexts, strict=True):
            plan.source_associations(source, _bank=context)
        return source.associations

    def _classes(
        self, source: CellMeshingResult, /
    ) -> tuple[tuple[_PlcBankContext, ...], tuple[tuple[PlcEntityClasses, ...], ...]]:
        contexts = self._contexts(source)
        levels = tuple(
            plan.classes(source, _bank=context)
            for plan, context in zip(self.plans, contexts, strict=True)
        )
        return contexts, levels

    def classes(self, source: CellMeshingResult, /) -> tuple[ComposedEntityClasses, ...]:
        contexts, levels = self._classes(source)
        result: list[ComposedEntityClasses] = []
        for dimension in range(4):
            count = source.mesh.entity_set(dimension).count
            # A joint code is a mixed-radix tuple of explicit original claims,
            # never a geometric hash or a nearest-source classification.
            codes = np.zeros(count, dtype=np.int64)
            dims = np.full(count, 4, dtype=np.int64)
            factor = 1
            namespace_indices: list[np.ndarray] = []
            namespace_dimensions: list[np.ndarray] = []
            for context, bank, radix in zip(
                contexts, levels, self.class_radices, strict=True
            ):
                level = bank[dimension]
                codes += np.where(level.resolved, level.codes + 1, 0) * factor
                factor *= radix
                dims = np.minimum(dims, np.where(level.resolved, level.dimensions, 4))
                indices = np.array(level.indices, copy=True)
                selected = level.dimensions == 2
                indices[selected] = context.facet_indices[indices[selected]]
                indices.setflags(write=False)
                namespace_indices.append(indices)
                namespace_dimensions.append(level.dimensions)
            resolved = dims < 4
            boundary = np.zeros((count,), dtype=np.bool_)
            if dimension < source.mesh.topological_dimension:
                boundary = contexts[0].mapped.boundary_entities[dimension]
                if boundary.shape != (count,):
                    raise ValueError(
                        "Composed boundary authority must align with the target entity set."
                    )
            missing = ~resolved | (boundary & (dims == 3))
            if np.any(missing):
                raise AssociationPropagationError(
                    "Composed boundary strata lack original source authority.",
                    np.asarray(source.mesh.entity_set(dimension).entity_ids)[missing],
                )
            for value in (dims, resolved, codes):
                value.setflags(write=False)
            result.append(
                ComposedEntityClasses(
                    dims,
                    tuple(namespace_dimensions),
                    tuple(namespace_indices),
                    resolved,
                    codes,
                )
            )
        return tuple(result)

    def protected_edges(
        self, source: CellMeshingResult, /, *, midpoint_required: bool = True
    ) -> np.ndarray:
        contexts, levels = self._classes(source)
        protected = np.zeros(source.mesh.entity_set(1).count, dtype=np.bool_)
        for plan, context, bank in zip(self.plans, contexts, levels, strict=True):
            protected |= plan.protected_edges(
                source, midpoint_required=midpoint_required, _bank=context, _classes=bank
            )
        return protected

    def propagate(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: CellMesh,
        /,
        *,
        geometry: CellGeometrySpec,
        embedding: GlobalEmbeddingCertificate | None = None,
        coverage: DomainCoverageCertificate | None = None,
        _certification_request: MeshCertificationInputs | None = None,
        _prepared_target: list[MeshCertificationPreparedEvidence] | None = None,
    ) -> tuple[GeometryAssociation, ...]:
        if embedding is not None or coverage is not None:
            raise ValueError(
                "Composed host PLC propagation requires its original whole-domain premises."
            )
        contexts = self._contexts(source)
        material = self.material_index
        target_proofs: list[MappedPlcSupport] = []
        output = self.plans[material].propagate(
            source,
            lineage,
            target,
            geometry=geometry,
            _bank=contexts[material],
            _certification_request=_certification_request,
            _prepared_target=_prepared_target,
            _mapped_target=target_proofs,
        )
        if len(target_proofs) != 1:
            raise ValueError(
                "The complete material owner must establish one exact target support proof."
            )
        for index, (plan, context) in enumerate(zip(self.plans, contexts, strict=True)):
            if index == material:
                continue
            context.target_mapped = target_proofs[0]
            output += plan.propagate(
                source, lineage, target, geometry=geometry, _bank=context
            )
        return output


__all__ = ["ComposedAssociationTransfer", "ComposedEntityClasses"]
