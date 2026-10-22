#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Compose PLC source namespace banks through the PLC support engine."""

from __future__ import annotations

from typing import final, NamedTuple, TYPE_CHECKING

import equinox as eqx
import jax
import numpy as np

import phydrax.discretization._cell_mesh as cell_mesh_owner
from phydrax.geometry._embedding_state import MeshCertificateLimits
from phydrax.meshing._association_state import (
    GeometryAssociation,
    GeometryAssociationKind,
    target_dimension,
)

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import nonnegative_integer, positive_integer
from ..discretization import _coordinate_enclosure as algebra
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_transition_types import CellGeometryTransition
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..geometry._mapped_source_support import MappedSourceFacetAuthority
from ..typing import Dim, HostInt64, parse
from . import (
    _association as association_owner,
    _plc_mapped_support as plc_mapped_support_owner,
)
from ._association_routes import AssociationPropagationError, PlcEntityClasses


if TYPE_CHECKING:
    from ..geometry._domain_coverage import DomainCoverageCertificate
    from ..geometry._global_embedding import GlobalEmbeddingCertificate
    from ._certification import MeshCertificationPreparedEvidence
    from ._certification_inputs import MeshCertificationInputs
    from ._lineage_records import MeshLineage
    from ._result import CellMeshingResult


class _BankFacetDim(Dim):
    """Source facet identifiers of one declared namespace."""


class _BankRegionDim(Dim, minimum=1):
    """Explicit source-region to shared material-region mapping."""


class _BankOffsetDim(Dim, minimum=3):
    """One boundary offset per explicitly declared source namespace."""


class ComposedEntityClasses(NamedTuple):
    """Return each source namespace claim and collision-free joint class codes."""

    dimensions: np.ndarray
    namespace_dimensions: tuple[np.ndarray, ...]
    namespace_indices: tuple[np.ndarray, ...]
    resolved: np.ndarray
    codes: np.ndarray


class _CompositionInputs(StrictModule):
    facets: np.ndarray
    facet_offsets: np.ndarray
    regions: np.ndarray
    region_offsets: np.ndarray
    keys: tuple[tuple[str, str], ...]
    queries: int
    byte_limit: int
    preparation: int


class _ComposedBanks(StrictModule):
    facets: list[np.ndarray]
    maps: list[np.ndarray]
    radices: list[int]
    retained: int


def _validate_composition_inputs(
    plans: tuple[association_owner.PlcAssociationTransfer, ...],
    facet_indices: np.ndarray,
    facet_offsets: np.ndarray,
    region_maps: np.ndarray,
    region_offsets: np.ndarray,
    reference_meshes: tuple[cell_mesh_owner.CellMesh | None, ...],
    reference_geometries: tuple[CellGeometrySpec | None, ...],
    /,
    *,
    material_namespace: tuple[str, str],
    maximum_support_queries: int,
    maximum_data_bytes: int,
    preparation_work_units: int = 0,
) -> _CompositionInputs:
    """Validate namespace identities, mapped-source bindings, and bank offsets in input order."""
    if (
        type(plans) is not tuple
        or len(plans) < 2
        or any(
            type(plan) is not association_owner.PlcAssociationTransfer for plan in plans
        )
    ):
        raise TypeError(
            f"ComposedAssociationTransfer.plans must be a tuple of at least two PLC plans; got {plans!r}"
        )
    if any(plan.domain.ambient_dimension != 3 for plan in plans):
        raise ValueError(
            f"ComposedAssociationTransfer.plans must use three-dimensional PLC domains; got dimensions {tuple(plan.domain.ambient_dimension for plan in plans)!r}"
        )
    if (
        type(reference_meshes) is not tuple
        or type(reference_geometries) is not tuple
        or len(reference_meshes) != len(plans)
        or len(reference_geometries) != len(plans)
    ):
        raise ValueError(
            f"ComposedAssociationTransfer.reference_sources must provide mesh and geometry tuples aligned with plans; got mesh tuple {reference_meshes!r}, geometry tuple {reference_geometries!r}, plans {len(plans)}"
        )
    for mesh, geometry, plan in zip(
        reference_meshes, reference_geometries, plans, strict=True
    ):
        if (mesh is None) != (geometry is None):
            raise ValueError(
                f"ComposedAssociationTransfer.reference_sources must pair a mesh with its coordinate expression; got mesh present {mesh is not None}, geometry present {geometry is not None}"
            )
        if mesh is not None and geometry is not None:
            if (
                type(mesh) is not cell_mesh_owner.CellMesh
                or type(geometry) is not CellGeometrySpec
                or mesh.entity_set(2).count != plan.facet_regions.shape[0]
            ):
                raise ValueError(
                    f"ComposedAssociationTransfer.reference_sources must bind CellMesh and CellGeometrySpec with one face per facet; got mesh {type(mesh).__name__}, geometry {type(geometry).__name__}, face count {mesh.entity_set(2).count if type(mesh) is cell_mesh_owner.CellMesh else None}, facet count {plan.facet_regions.shape[0]}"
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
            f"ComposedAssociationTransfer.banks must contain integer vectors; got shapes and dtypes {tuple((value.shape, str(value.dtype)) for value in raw)!r}"
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
                f"ComposedAssociationTransfer.bank_offsets must partition bank values monotonically from zero; got offsets {offsets!r}, values {values.size}, namespaces {len(plans)}"
            )
    keys = tuple((plan.domain.source_id, plan.source_revision) for plan in plans)
    if len(set(keys)) != len(keys) or material_namespace not in keys:
        raise ValueError(
            f"ComposedAssociationTransfer.material_namespace must select one of the distinct source namespaces; got keys {keys!r}, selection {material_namespace!r}"
        )
    if len({plan.coordinate_contract.spatial_id for plan in plans}) != 1:
        raise ValueError(
            f"ComposedAssociationTransfer.coordinate_contract must identify one physical frame; got spatial IDs {tuple(plan.coordinate_contract.spatial_id for plan in plans)!r}"
        )
    queries = positive_integer(maximum_support_queries, "maximum_support_queries")
    byte_limit = positive_integer(maximum_data_bytes, "maximum_data_bytes")
    preparation = nonnegative_integer(preparation_work_units, "preparation_work_units")
    if queries > min(plan.maximum_support_queries for plan in plans):
        raise ValueError(
            f"ComposedAssociationTransfer.maximum_support_queries must not exceed any plan allowance; got {queries}, plan allowances {tuple(plan.maximum_support_queries for plan in plans)!r}"
        )
    return _CompositionInputs(
        original_facets,
        original_facet_offsets,
        original_regions,
        original_region_offsets,
        keys,
        queries,
        byte_limit,
        preparation,
    )


def _prepare_composed_banks(
    order: list[int],
    ordered: tuple[association_owner.PlcAssociationTransfer, ...],
    material: association_owner.PlcAssociationTransfer,
    inputs: _CompositionInputs,
) -> _ComposedBanks:
    """Prepare ordered source banks and their collision-free mixed-radix class capacity."""
    original_facets, original_facet_offsets = inputs.facets, inputs.facet_offsets
    original_regions, original_region_offsets = inputs.regions, inputs.region_offsets
    facets: list[np.ndarray] = []
    maps: list[np.ndarray] = []
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
                f"ComposedAssociationTransfer.facet_indices must uniquely name each nonnegative source facet; got indices {facet!r}, facet count {plan.facet_regions.shape[0]}"
            )
        if (
            regions.shape != plan.region_indices.shape
            or np.unique(regions).size != regions.size
            or np.any(~np.isin(regions, material.region_indices))
        ):
            raise ValueError(
                f"ComposedAssociationTransfer.region_maps must inject declared source regions into material regions; got map {regions!r}, source regions {plan.region_indices!r}, material regions {material.region_indices!r}"
            )
        for local, label in enumerate(regions):
            position = np.flatnonzero(material.region_indices == label)
            if (
                position.size != 1
                or plan.domain.region_ids[local]
                != material.domain.region_ids[int(position[0])]
            ):
                raise ValueError(
                    f"ComposedAssociationTransfer.region_maps must preserve physical region names; got source name {plan.domain.region_ids[local]!r}, material positions {position!r}, material names {material.domain.region_ids!r}"
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
                f"ComposedAssociationTransfer.class_radices must fit joint class codes in int64; got maximum code {capacity - 1}, limit {np.iinfo(np.int64).max}"
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
    return _ComposedBanks(facets, maps, radices, retained)


@final
class ComposedAssociationTransfer(StrictModule, NonTrainableState):
    """Compose namespace-disjoint PLC plans while retaining their source identities.

    One material plan certifies whole-domain coverage. Other plans retain their
    primitive banks and explicit material maps. ``PlcAssociationTransfer``
    certifies lineage propagation and continuous support; this record combines
    the resulting namespace claims. Numerical maps remain dynamic leaves, and
    all banks share one admitted query and byte allowance.
    """

    __strict_contract__ = True
    plans: tuple[association_owner.PlcAssociationTransfer, ...]
    facet_indices: HostInt64[_BankFacetDim]
    facet_offsets: HostInt64[_BankOffsetDim]
    region_maps: HostInt64[_BankRegionDim]
    region_offsets: HostInt64[_BankOffsetDim]
    reference_meshes: tuple[cell_mesh_owner.CellMesh | None, ...]
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
        plans: tuple[association_owner.PlcAssociationTransfer, ...],
        facet_indices: np.ndarray,
        facet_offsets: np.ndarray,
        region_maps: np.ndarray,
        region_offsets: np.ndarray,
        reference_meshes: tuple[cell_mesh_owner.CellMesh | None, ...],
        reference_geometries: tuple[CellGeometrySpec | None, ...],
        /,
        *,
        material_namespace: tuple[str, str],
        maximum_support_queries: int,
        maximum_data_bytes: int,
        preparation_work_units: int = 0,
    ) -> None:
        inputs = _validate_composition_inputs(
            plans,
            facet_indices,
            facet_offsets,
            region_maps,
            region_offsets,
            reference_meshes,
            reference_geometries,
            material_namespace=material_namespace,
            maximum_support_queries=maximum_support_queries,
            maximum_data_bytes=maximum_data_bytes,
            preparation_work_units=preparation_work_units,
        )
        keys, queries, byte_limit, preparation = (
            inputs.keys,
            inputs.queries,
            inputs.byte_limit,
            inputs.preparation,
        )
        order = sorted(range(len(plans)), key=keys.__getitem__)
        ordered = tuple(plans[index] for index in order)
        material_index = order.index(keys.index(material_namespace))
        material = ordered[material_index]
        banks = _prepare_composed_banks(order, ordered, material, inputs)
        facets, maps, radices, retained = (
            banks.facets,
            banks.maps,
            banks.radices,
            banks.retained,
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
                f"ComposedAssociationTransfer.material_region_map must be the identity map; got {maps[material_index]!r}, expected {material.region_indices!r}"
            )
        if retained > byte_limit:
            raise ValueError(
                f"ComposedAssociationTransfer.maximum_data_bytes must admit retained source banks; got bytes {retained}, limit {byte_limit}"
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

    def _material_cell_regions(self, source: CellMeshingResult, /) -> np.ndarray:
        """Material-namespace region labels of every source cell in entity order."""
        keys = {(plan.domain.source_id, plan.source_revision) for plan in self.plans}
        if any(
            value.association_kind is not GeometryAssociationKind.PIECEWISE_LINEAR
            or (value.source_id, value.source_revision) not in keys
            for value in source.associations
        ):
            raise ValueError(
                f"ComposedAssociationTransfer.source_associations must bind PLC rows to declared namespaces; got bindings {tuple((value.association_kind, value.source_id, value.source_revision) for value in source.associations)!r}, namespaces {keys!r}"
            )
        owners = tuple(
            value
            for value in source.associations
            if (value.source_id, value.source_revision) == self.material_namespace
            and target_dimension(source.mesh, value) == 3
        )
        if len(owners) != 1:
            raise ValueError(
                f"ComposedAssociationTransfer.material_association must select one cell table; got count {len(owners)} for namespace {self.material_namespace!r}"
            )
        owner = owners[0]
        ids = np.asarray(source.mesh.entity_set(3).entity_ids, dtype=np.int64)
        if not np.array_equal(
            np.sort(np.asarray(owner.target_global_ids)), np.sort(ids)
        ) or np.any(np.asarray(owner.source_dimensions) != 3):
            raise ValueError(
                f"ComposedAssociationTransfer.material_association must bind all cells to region strata; got target IDs {np.asarray(owner.target_global_ids)!r}, cell IDs {ids!r}, dimensions {np.asarray(owner.source_dimensions)!r}"
            )
        return np.asarray(owner.source_indices, dtype=np.int64)[owner.target_rows(ids)]

    def bind_source_proof(
        self,
        source: CellMeshingResult,
        proof: plc_mapped_support_owner.MappedPlcSupportProof,
        /,
        *,
        budget: CoordinateEnclosureBudget,
        import_charges: bool = True,
    ) -> plc_mapped_support_owner.MappedPlcSupport:
        """Bind an ended material source-support proof to a request ledger.

        The source, material domain, allowance, cell regions and certified
        request are authenticated as for ``PlcAssociationTransfer``;
        ``import_charges=False`` is for the request whose own charged
        preparation receipt already contains the proof's work.
        """
        certificate = source.certification
        if certificate is None:
            raise ValueError(
                "ComposedAssociationTransfer.source_proof must bind a certified volume source; got no certification"
            )
        material = self.plans[self.material_index]
        return plc_mapped_support_owner.bind_mapped_plc_support(
            proof,
            source.mesh,
            source.geometry,
            material.domain,
            self._material_cell_regions(source),
            maximum_support_queries=self.maximum_support_queries,
            certification_request=certificate.request,
            budget=budget,
            import_charges=import_charges,
        )

    def _contexts(
        self,
        source: CellMeshingResult,
        /,
        *,
        budget: CoordinateEnclosureBudget,
        _prepared_source: list[plc_mapped_support_owner.MappedPlcSupport] | None = None,
    ) -> tuple[association_owner.PlcBankContext, ...]:
        material = self.plans[self.material_index]
        cell_regions = self._material_cell_regions(source)
        certificate = source.certification
        if certificate is None or not certificate.passed:
            raise ValueError(
                f"ComposedAssociationTransfer.source_certification must have passed; got certificate present {certificate is not None}, passed {None if certificate is None else certificate.passed}"
            )
        allocation = sum(
            source.mesh.entity_set(dimension).count for dimension in range(4)
        ) * (41 * len(self.plans) + 17)
        if allocation > self.maximum_data_bytes:
            raise ValueError(
                f"ComposedAssociationTransfer.maximum_data_bytes must admit temporary source classes; got bytes {allocation}, limit {self.maximum_data_bytes}"
            )
        prepared = [] if _prepared_source is None else _prepared_source
        if len(prepared) > 1:
            raise ValueError(
                f"ComposedAssociationTransfer.prepared_source must contain at most one material support proof; got count {len(prepared)}"
            )
        if not prepared:
            prepared.append(
                plc_mapped_support_owner.prepare_mapped_plc_support(
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
                    budget=budget,
                )
            )
        else:
            # A proof earned earlier in this request is admitted only on this
            # ledger, for this exact source carrier, domain, allowance and labels.
            earned = prepared[0]
            if (
                earned.ledger is not budget
                or earned.mesh is not source.mesh
                or earned.geometry is not source.geometry
                or earned.domain.domain_id != material.domain.domain_id
                or earned.limits.maximum_candidate_pairs != self.maximum_support_queries
                or not np.array_equal(earned.cell_regions, cell_regions)
                or not plc_mapped_support_owner._support_proof_is_issued(earned.proof)
            ):
                raise ValueError(
                    f"ComposedAssociationTransfer.prepared_source must be bound to this request ledger and match source, material domain, support allowance and cell regions; got same ledger {earned.ledger is budget}, domain {earned.domain.domain_id!r}, allowance {earned.limits.maximum_candidate_pairs}, expected {(material.domain.domain_id, self.maximum_support_queries)!r}"
                )
            earned.prepared.require(
                source.mesh, source.geometry, certificate.request, budget=budget
            )
        proof = prepared[0]
        work = [self.maximum_support_queries]
        contexts: list[association_owner.PlcBankContext] = []
        for index in range(len(self.plans)):
            context = association_owner.PlcBankContext(
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
                # The material proof ledger charges this root preparation too.
                context.mapped_facets = MappedSourceFacetAuthority.prepare(
                    mesh,
                    geometry,
                    plan.triangle_vertices,
                    plan.triangle_facets,
                    budget=proof.ledger,
                )
            contexts.append(context)
        return tuple(contexts)

    def source_associations(
        self,
        source: CellMeshingResult,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        _prepared_source: list[plc_mapped_support_owner.MappedPlcSupport] | None = None,
    ) -> tuple[GeometryAssociation, ...]:
        with association_owner.support_budget(
            MeshCertificateLimits().maximum_work_units,
            self.maximum_support_queries,
            budget,
            maximum_memory_bytes=(
                MeshCertificateLimits().maximum_scratch_bytes
                if budget is None
                else budget.maximum_memory_bytes
            ),
        ) as budget:
            contexts = self._contexts(
                source, budget=budget, _prepared_source=_prepared_source
            )
            for plan, context in zip(self.plans, contexts, strict=True):
                plan.source_associations(source, _bank=context, budget=budget)
            return source.associations

    def _classes(
        self,
        source: CellMeshingResult,
        /,
        *,
        budget: CoordinateEnclosureBudget,
        _prepared_source: list[plc_mapped_support_owner.MappedPlcSupport] | None,
    ) -> tuple[
        tuple[association_owner.PlcBankContext, ...],
        tuple[tuple[PlcEntityClasses, ...], ...],
    ]:
        contexts = self._contexts(
            source, budget=budget, _prepared_source=_prepared_source
        )
        levels = tuple(
            plan.classes(source, _bank=context, budget=budget)
            for plan, context in zip(self.plans, contexts, strict=True)
        )
        return contexts, levels

    def classes(
        self,
        source: CellMeshingResult,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        _prepared_contexts: list[association_owner.PlcBankContext] | None = None,
        _prepared_source: list[plc_mapped_support_owner.MappedPlcSupport] | None = None,
    ) -> tuple[ComposedEntityClasses, ...]:
        with association_owner.support_budget(
            MeshCertificateLimits().maximum_work_units,
            self.maximum_support_queries,
            budget,
            maximum_memory_bytes=(
                MeshCertificateLimits().maximum_scratch_bytes
                if budget is None
                else budget.maximum_memory_bytes
            ),
        ) as budget:
            contexts, levels = self._classes(
                source, budget=budget, _prepared_source=_prepared_source
            )
            if _prepared_contexts is not None:
                _prepared_contexts.extend(contexts)
            result: list[ComposedEntityClasses] = []
            for dimension in range(4):
                count = source.mesh.entity_set(dimension).count
                # A joint code is a mixed-radix tuple of explicit source claims,
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
                            f"ComposedAssociationTransfer.boundary_entities must align with target rows; got shape {boundary.shape!r}, rows {count}"
                        )
                missing = ~resolved | (boundary & (dims == 3))
                if np.any(missing):
                    raise AssociationPropagationError(
                        f"ComposedAssociationTransfer.source_strata must resolve boundary rows below region dimension; got unresolved target IDs {np.asarray(source.mesh.entity_set(dimension).entity_ids)[missing]!r}",
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

    def _exact_split_support(
        self,
        context: association_owner.PlcBankContext,
        plan: association_owner.PlcAssociationTransfer,
        row: int,
        dimension: int,
        index: int,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> bool:
        """A whole-edge source proof also proves its half-reference point."""
        if dimension not in (1, 2):
            return False
        authority = plan.edge_indices if dimension == 1 else context.facet_indices
        matches = np.flatnonzero(authority == index)
        if matches.size != 1:
            raise ValueError(
                f"ComposedAssociationTransfer.split_source must identify one declared source stratum; got {matches.size} matches"
            )
        local = int(matches[0])
        if dimension == 2 and context.mapped_facets is not None:
            supported, _, context.work[0] = context.mapped_facets.support(
                context.mapped, 1, row, local, context.work[0], budget=budget
            )
            return supported
        supported, _ = context.mapped.entity_support(
            1,
            row,
            dimension,
            local,
            edge_vertices=plan.edge_vertices,
            triangle_vertices=plan.triangle_vertices,
            triangle_facets=plan.triangle_facets,
            work=context.work,
            source_vertices=plan.source_vertices,
        )
        return supported

    def protected_edges(
        self,
        source: CellMeshingResult,
        /,
        *,
        midpoint_required: bool = True,
        _classes: tuple[ComposedEntityClasses, ...] | None = None,
        _prepared_contexts: list[association_owner.PlcBankContext] | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        with association_owner.support_budget(
            MeshCertificateLimits().maximum_work_units,
            self.maximum_support_queries,
            budget,
            maximum_memory_bytes=(
                MeshCertificateLimits().maximum_scratch_bytes
                if budget is None
                else budget.maximum_memory_bytes
            ),
        ) as budget:
            retained_contexts = [] if _prepared_contexts is None else _prepared_contexts
            levels = (
                self.classes(source, budget=budget, _prepared_contexts=retained_contexts)
                if _classes is None
                else _classes
            )
            vertices = plc_mapped_support_owner.plc_mapped_vertices(source.mesh, 1)
            endpoints = np.asarray(source.mesh.coordinates, dtype=np.float64)[vertices]
            midpoints = np.sum(endpoints * np.float64(0.5), axis=1)
            protected = np.all(midpoints == endpoints[:, 0], axis=1) | np.all(
                midpoints == endpoints[:, 1], axis=1
            )
            if midpoint_required:
                edge_classes = levels[1]
                contexts = retained_contexts or self._contexts(source, budget=budget)
                if len(contexts) != len(self.plans) or any(
                    context.mapped.mesh is not source.mesh
                    or context.mapped.geometry is not source.geometry
                    or not np.array_equal(
                        context.facet_indices,
                        self.facet_indices[
                            self.facet_offsets[index] : self.facet_offsets[index + 1]
                        ],
                    )
                    for index, context in enumerate(contexts)
                ):
                    raise ValueError(
                        "ComposedAssociationTransfer.prepared_contexts must bind the current immutable source and namespace banks; got a different source or bank"
                    )
                representatives: dict[int, np.ndarray] = {}
                for namespace, plan in enumerate(self.plans):
                    dimensions = edge_classes.namespace_dimensions[namespace]
                    indices = edge_classes.namespace_indices[namespace]
                    context = contexts[namespace]
                    for row in np.flatnonzero(
                        (dimensions >= 0) & (dimensions < 3) & ~protected
                    ).tolist():
                        numeric = representatives.get(row)
                        if numeric is None:
                            exact = context.mapped.edge_midpoint(row)
                            numeric = algebra.rounded_point(exact)
                            budget.retain_basis((numeric,))
                            representatives[row] = numeric
                        collapsed = np.array_equal(
                            numeric, endpoints[row, 0]
                        ) or np.array_equal(numeric, endpoints[row, 1])
                        protected[row] |= collapsed or not self._exact_split_support(
                            context,
                            plan,
                            row,
                            int(dimensions[row]),
                            int(indices[row]),
                            budget=budget,
                        )
            return protected

    def propagate(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: cell_mesh_owner.CellMesh,
        /,
        *,
        geometry: CellGeometrySpec,
        embedding: GlobalEmbeddingCertificate | None = None,
        coverage: DomainCoverageCertificate | None = None,
        _certification_request: MeshCertificationInputs | None = None,
        _geometry_transition: CellGeometryTransition | None = None,
        _prepared_target: list[MeshCertificationPreparedEvidence] | None = None,
        _prepared_source: list[plc_mapped_support_owner.MappedPlcSupport] | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[GeometryAssociation, ...]:
        with association_owner.support_budget(
            MeshCertificateLimits().maximum_work_units,
            self.maximum_support_queries,
            budget,
            maximum_memory_bytes=(
                MeshCertificateLimits().maximum_scratch_bytes
                if budget is None
                else budget.maximum_memory_bytes
            ),
        ) as budget:
            if embedding is not None or coverage is not None:
                raise ValueError(
                    f"ComposedAssociationTransfer.target_support must be prepared by its material plan; got supplied embedding {embedding is not None}, coverage {coverage is not None}"
                )
            contexts = self._contexts(
                source, budget=budget, _prepared_source=_prepared_source
            )
            material = self.material_index
            target_proofs: list[plc_mapped_support_owner.MappedPlcSupport] = []
            output = self.plans[material].propagate(
                source,
                lineage,
                target,
                geometry=geometry,
                _bank=contexts[material],
                _certification_request=_certification_request,
                _geometry_transition=_geometry_transition,
                _prepared_target=_prepared_target,
                _mapped_target=target_proofs,
                budget=budget,
            )
            if len(target_proofs) != 1:
                raise ValueError(
                    f"ComposedAssociationTransfer.target_support must contain one material proof; got count {len(target_proofs)}"
                )
            for index, (plan, context) in enumerate(
                zip(self.plans, contexts, strict=True)
            ):
                if index == material:
                    continue
                context.target_mapped = target_proofs[0]
                output += plan.propagate(
                    source,
                    lineage,
                    target,
                    geometry=geometry,
                    _bank=context,
                    budget=budget,
                )
            return output


__all__ = ["ComposedAssociationTransfer", "ComposedEntityClasses"]
