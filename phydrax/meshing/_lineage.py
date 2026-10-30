#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum, StrEnum
from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike as JaxArrayLike
from numpy.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry_transition_types import (
    cell_geometry_id,
    CellGeometryTransition,
)
from ..discretization.fem import (
    FiniteElementTopologyTransfer,
    vertex_interpolation_transfer,
)
from ..geometry._volume_material_deformation import MaterialReferenceDeclaration
from ..geometry._volume_material_map import VolumeMaterialChain
from ..sparse import RowRelation, SparseLinearMap
from ..typing import checked
from ._result import CellMeshingResult


class EntityLineageKind(IntEnum):
    PRESERVED = 0
    REFINED_FROM = 1
    COARSENED_INTO = 2
    SPLIT_FROM = 3
    MERGED_INTO = 4
    COLLAPSED_INTO = 5
    SWAPPED_FROM = 6
    RELOCATED = 7
    GENERATED_ON_GEOMETRY = 8
    UNKNOWN = 9


class MeshTransitionKind(StrEnum):
    REFINE = "refine"
    COARSEN = "coarsen"
    REMESH = "remesh"
    REPAIR = "repair"
    PARTITION = "partition"


def interpolation_rows_preserve_constants(
    weights: ArrayLike, valid: ArrayLike, /
) -> bool:
    """Whether every supported binary64 interpolation row sums to one exactly."""
    coefficients = np.asarray(weights, dtype=np.float64)
    support = np.asarray(valid, dtype=np.bool_)
    if (
        coefficients.ndim != 2
        or support.shape != coefficients.shape
        or not coefficients.shape[0]
        or not np.all(np.any(support, axis=1))
        or not np.all(np.isfinite(coefficients[support]))
    ):
        return False
    return all(
        sum(
            (
                Fraction.from_float(float(value))
                for value in coefficients[row, support[row]]
            ),
            Fraction(0),
        ) == 1
        for row in range(coefficients.shape[0])
    )




class EntityLineage(StrictModule, NonTrainableState):
    dimension: int = eqx.field(static=True)
    source_entity_set_id: str = eqx.field(static=True)
    target_entity_set_id: str = eqx.field(static=True)
    source_global_ids: Array
    target_global_ids: Array
    relation_kinds: Array
    created_target_ids: Array
    deleted_source_ids: Array
    lineage_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        source_entity_set_id: str,
        target_entity_set_id: str,
        source_global_ids: ArrayLike,
        target_global_ids: ArrayLike,
        relation_kinds: ArrayLike,
        /,
        *,
        created_target_ids: ArrayLike = (),
        deleted_source_ids: ArrayLike = (),
    ) -> None:
        source_set = str(source_entity_set_id).strip()
        target_set = str(target_entity_set_id).strip()
        if not source_set or not target_set:
            raise ValueError("Entity lineage set identities must be non-empty.")
        dimension_ = int(dimension)
        if dimension_ < 0:
            raise ValueError("Entity lineage dimension must be non-negative.")
        source = np.asarray(source_global_ids, dtype=np.int64)
        target = np.asarray(target_global_ids, dtype=np.int64)
        kinds = np.asarray(relation_kinds, dtype=np.int32)
        created = np.asarray(created_target_ids, dtype=np.int64)
        deleted = np.asarray(deleted_source_ids, dtype=np.int64)
        if (
            source.ndim != 1
            or target.shape != source.shape
            or kinds.shape != source.shape
        ):
            raise ValueError("Entity lineage relations must be aligned rank-one arrays.")
        if created.ndim != 1 or deleted.ndim != 1:
            raise ValueError("Created and deleted entity IDs must be rank-one arrays.")
        if (
            np.any(source < 0)
            or np.any(target < 0)
            or np.any(created < 0)
            or np.any(deleted < 0)
        ):
            raise ValueError("Entity lineage IDs must be non-negative.")
        supported = np.asarray(
            [int(value) for value in EntityLineageKind], dtype=np.int32
        )
        if np.any(~np.isin(kinds, supported)):
            raise ValueError("Entity lineage contains an unsupported relation kind.")
        if dimension_ == 0 and np.any(
            kinds == int(EntityLineageKind.SPLIT_FROM)
        ):
            raise ValueError(
                "Vertex split ancestry must be represented by interpolation stencils."
            )
        if np.intersect1d(target, created).size or np.intersect1d(source, deleted).size:
            raise ValueError(
                "Created/deleted IDs cannot also appear in lineage relations."
            )
        self.dimension = dimension_
        self.source_entity_set_id = source_set
        self.target_entity_set_id = target_set
        self.source_global_ids = jnp.asarray(source)
        self.target_global_ids = jnp.asarray(target)
        self.relation_kinds = jnp.asarray(kinds)
        self.created_target_ids = jnp.asarray(created)
        self.deleted_source_ids = jnp.asarray(deleted)
        self.lineage_id = canonical_fingerprint(
            {
                "kind": "entity-lineage",
                "dimension": dimension_,
                "source_entity_set_id": source_set,
                "target_entity_set_id": target_set,
                "source_global_ids": array_tree_fingerprint(source),
                "target_global_ids": array_tree_fingerprint(target),
                "relation_kinds": array_tree_fingerprint(kinds),
                "created_target_ids": array_tree_fingerprint(created),
                "deleted_source_ids": array_tree_fingerprint(deleted),
            }
        )



class MeshLineage(StrictModule, NonTrainableState):
    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    entities: tuple[EntityLineage, ...]
    lineage_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_topology_id: str,
        target_topology_id: str,
        entities: tuple[EntityLineage, ...],
        /,
    ) -> None:
        source = str(source_topology_id).strip()
        target = str(target_topology_id).strip()
        records = tuple(entities)
        if not source or not target:
            raise ValueError("Mesh lineage topology identities must be non-empty.")
        if not records or not all(isinstance(value, EntityLineage) for value in records):
            raise ValueError("Mesh lineage requires EntityLineage records.")
        dimensions = tuple(value.dimension for value in records)
        if dimensions != tuple(sorted(set(dimensions))):
            raise ValueError(
                "Mesh lineage must contain one ordered record per dimension."
            )
        self.source_topology_id = source
        self.target_topology_id = target
        self.entities = records
        self.lineage_id = canonical_fingerprint(
            {
                "kind": "mesh-lineage",
                "source_topology": source,
                "target_topology": target,
                "entities": [value.lineage_id for value in records],
            }
        )

    def entity_lineage(self, dimension: int, /) -> EntityLineage:
        target = int(dimension)
        for value in self.entities:
            if value.dimension == target:
                return value
        raise KeyError(f"No lineage record for dimension {target}.")


class VertexInterpolationStencil(StrictModule, NonTrainableState):
    """Exact source rows with binary64 action weights and rational coefficients.

    ``exact_weights`` is the authoritative source-map row; ``weights`` is its
    binary64 execution projection. Repeated source IDs remain additive, including
    multiple exact binary64 components for a dyadic coefficient.
    """

    source_entity_set_id: str = eqx.field(static=True)
    target_entity_set_id: str = eqx.field(static=True)
    target_global_ids: Array
    source_global_ids: Array
    weights: Array
    valid: Array
    exact_weights: tuple[tuple[Fraction, ...], ...] = eqx.field(static=True)
    preserves_constants: bool = eqx.field(static=True)
    stencil_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_entity_set_id: str,
        target_entity_set_id: str,
        target_global_ids: ArrayLike,
        source_global_ids: ArrayLike,
        weights: ArrayLike,
        valid: ArrayLike,
        /,
        *,
        preserves_constants: bool = True,
        exact_weights: tuple[tuple[Fraction, ...], ...] | None = None,
    ) -> None:
        source_set = str(source_entity_set_id).strip()
        target_set = str(target_entity_set_id).strip()
        targets = np.asarray(target_global_ids, dtype=np.int64)
        sources = np.asarray(source_global_ids, dtype=np.int64)
        coefficients = np.asarray(weights, dtype=np.float64)
        valid_ = np.asarray(valid, dtype=np.bool_)
        if not source_set or not target_set:
            raise ValueError("Interpolation entity set identities must be non-empty.")
        if targets.ndim != 1 or sources.ndim != 2:
            raise ValueError("Interpolation targets and sources must be rank one/two.")
        if sources.shape != coefficients.shape or valid_.shape != sources.shape:
            raise ValueError(
                "Interpolation source, weight, and validity shapes must match."
            )
        if sources.shape[0] != targets.shape[0] or targets.size == 0:
            raise ValueError("Every interpolation target requires one stencil row.")
        if np.any(targets < 0) or np.any(sources[valid_] < 0):
            raise ValueError("Interpolation entity IDs must be non-negative.")
        source_rows = tuple(tuple(int(value) for value in row) for row in sources.tolist())
        valid_rows = tuple(tuple(bool(value) for value in row) for row in valid_.tolist())
        coefficient_rows = tuple(
            tuple(float(value) for value in row) for row in coefficients.tolist()
        )
        for source_row, row_valid in zip(source_rows, valid_rows, strict=True):
            active_sources = tuple(
                source_id
                for source_id, is_valid in zip(source_row, row_valid, strict=True)
                if is_valid
            )
            if len(set(active_sources)) != len(active_sources):
                raise ValueError(
                    "Interpolation rows must have one exact coefficient per source ID."
                )
        if np.any(~np.isfinite(coefficients[valid_])):
            raise ValueError("Interpolation weights must be finite.")
        if exact_weights is None:
            exact = tuple(
                tuple(
                    Fraction.from_float(value) if is_valid else Fraction(0)
                    for value, is_valid in zip(row, row_valid, strict=True)
                )
                for row, row_valid in zip(
                    coefficient_rows, valid_rows, strict=True
                )
            )
        else:
            exact = tuple(tuple(row) for row in exact_weights)
            if len(exact) != sources.shape[0] or any(
                len(row) != sources.shape[1] for row in exact
            ):
                raise ValueError("Exact interpolation weights must align with rows.")
            if any(not isinstance(value, Fraction) for row in exact for value in row):
                raise TypeError("Exact interpolation weights must be Fraction values.")
            if any(
                not is_valid and value != 0
                for row, row_valid in zip(exact, valid_rows, strict=True)
                for value, is_valid in zip(row, row_valid, strict=True)
            ):
                raise ValueError("Invalid interpolation slots require exact zero weights.")
            if any(
                is_valid
                and (
                    float(value) != coefficient
                    or (value != 0 and coefficient == 0.0)
                )
                for row, row_valid, coefficient_row in zip(
                    exact, valid_rows, coefficient_rows, strict=True
                )
                for value, is_valid, coefficient in zip(
                    row, row_valid, coefficient_row, strict=True
                )
            ):
                raise ValueError(
                    "Binary64 interpolation weights must be the direct exact-weight projection."
                )
        if preserves_constants and any(
            sum(
                (value for value, is_valid in zip(row, row_valid, strict=True) if is_valid),
                Fraction(0),
            )
            != 1
            for row, row_valid in zip(exact, valid_rows, strict=True)
        ):
            raise ValueError(
                "Constant-preserving exact interpolation rows must sum to one exactly."
            )
        self.source_entity_set_id = source_set
        self.target_entity_set_id = target_set
        self.target_global_ids = jnp.asarray(targets)
        self.source_global_ids = jnp.asarray(sources)
        self.weights = jnp.asarray(coefficients)
        self.valid = jnp.asarray(valid_)
        self.exact_weights = exact
        self.preserves_constants = bool(preserves_constants)
        self.stencil_id = canonical_fingerprint(
            {
                "kind": "vertex-interpolation-stencil",
                "source_entity_set_id": source_set,
                "target_entity_set_id": target_set,
                "target_global_ids": array_tree_fingerprint(targets),
                "source_global_ids": array_tree_fingerprint(sources),
                "weights": array_tree_fingerprint(coefficients),
                "exact_weights": [
                    [(value.numerator, value.denominator) for value in row]
                    for row in exact
                ],
                "valid": array_tree_fingerprint(valid_),
                "preserves_constants": bool(preserves_constants),
            }
        )

    def _source_routes(self, source_global_ids: ArrayLike, /) -> np.ndarray:
        """Resolve stencil source IDs to positions in one caller-supplied ID order."""

        identifiers = np.asarray(source_global_ids, dtype=np.int64)
        if identifiers.ndim != 1 or identifiers.size == 0:
            raise ValueError("source_global_ids must be one non-empty rank-1 array.")
        order = np.argsort(identifiers, kind="stable")
        ordered = identifiers[order]
        if np.any(ordered[1:] == ordered[:-1]):
            raise ValueError("source_global_ids must be unique.")
        references = np.asarray(self.source_global_ids)
        valid = np.asarray(self.valid)
        position = np.minimum(np.searchsorted(ordered, references), identifiers.size - 1)
        if np.any(valid & (ordered[position] != references)):
            raise ValueError("Interpolation stencil references an unavailable source ID.")
        return np.where(valid, order[position], 0).astype(np.int32)

    def as_transfer(
        self,
        source_global_ids: ArrayLike,
        /,
        *,
        source_topology_id: str,
        target_topology_id: str,
        preserves_linear: bool = False,
        conservative: bool = False,
        source_coordinates: ArrayLike | None = None,
        target_coordinates: ArrayLike | None = None,
        source_measures: ArrayLike | None = None,
        target_measures: ArrayLike | None = None,
    ) -> FiniteElementTopologyTransfer:
        """Resolve this stencil into a sparse transfer over one source DOF order.

        Linear-preservation and conservation claims are certified by the transfer
        owner from the supplied row-aligned coordinates and P1 DOF measures.
        """

        valid = np.asarray(self.valid, dtype=np.bool_)
        if not np.all(np.any(valid, axis=1)):
            raise ValueError("Every interpolation target requires valid source ancestry.")
        identifiers = np.asarray(source_global_ids, dtype=np.int64)
        return vertex_interpolation_transfer(
            self._source_routes(identifiers),
            np.asarray(self.weights),
            np.asarray(self.valid),
            source_size=identifiers.size,
            source_topology_id=source_topology_id,
            target_topology_id=target_topology_id,
            preserves_linear=preserves_linear,
            conservative=conservative,
            # ty: ignore[invalid-argument-type]
            source_coordinates=source_coordinates,
            # ty: ignore[invalid-argument-type]
            target_coordinates=target_coordinates,
            # ty: ignore[invalid-argument-type]
            source_measures=source_measures,
            # ty: ignore[invalid-argument-type]
            target_measures=target_measures,
        )

    def apply(
        self,
        source_global_ids: ArrayLike,
        values: JaxArrayLike,
        /,
    ) -> Array:
        """Apply row routes; repeated IDs contribute additively to the same DOF."""

        valid = np.asarray(self.valid, dtype=np.bool_)
        if not np.all(np.any(valid, axis=1)):
            raise ValueError("Every interpolation target requires valid source ancestry.")
        identifiers = np.asarray(source_global_ids, dtype=np.int64)
        source_values = jnp.asarray(values)
        routes = self._source_routes(identifiers)
        if source_values.ndim == 0 or source_values.shape[0] != identifiers.size:
            raise ValueError("Source values must align with source_global_ids.")
        relation = RowRelation(routes, source_size=identifiers.size, valid=self.valid)
        return SparseLinearMap(
            relation,
            self.weights,
            operator_id=canonical_fingerprint(
                {
                    "kind": "vertex-interpolation-stencil-map",
                    "stencil": self.stencil_id,
                    "source_global_ids": array_tree_fingerprint(identifiers),
                }
            ),
        ).mv(source_values)


class CellMeshTransition(StrictModule, NonTrainableState):
    source_mesh_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    target: CellMeshingResult
    lineage: MeshLineage
    vertex_stencil: VertexInterpolationStencil | None
    geometry_transition: CellGeometryTransition | None
    volume_material_chain: VolumeMaterialChain | None
    material_reference: MaterialReferenceDeclaration | None
    transition_kind: MeshTransitionKind = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source_mesh_id: str,
        source_topology_id: str,
        target: CellMeshingResult,
        lineage: MeshLineage,
        transition_kind: MeshTransitionKind,
        /,
        *,
        vertex_stencil: VertexInterpolationStencil | None = None,
        geometry_transition: CellGeometryTransition | None = None,
        volume_material_chain: VolumeMaterialChain | None = None,
        material_reference: MaterialReferenceDeclaration | None = None,
    ) -> None:
        mesh_id = str(source_mesh_id).strip()
        topology_id = str(source_topology_id).strip()
        if not mesh_id or not topology_id:
            raise ValueError("Transition source identities must be non-empty.")
        if (
            lineage.source_topology_id != topology_id
            or lineage.target_topology_id != target.mesh.topology_id
        ):
            raise ValueError(
                "Transition lineage endpoints do not match source/target topology."
            )
        if not isinstance(transition_kind, MeshTransitionKind):
            raise TypeError("transition_kind must be MeshTransitionKind.")
        if vertex_stencil is not None and not isinstance(
            vertex_stencil, VertexInterpolationStencil
        ):
            raise TypeError("vertex_stencil must be VertexInterpolationStencil or None.")
        if geometry_transition is not None:
            if not isinstance(geometry_transition, CellGeometryTransition):
                raise TypeError("geometry_transition must be CellGeometryTransition or None.")
            if (
                geometry_transition.source_mesh.mesh_id != mesh_id
                or geometry_transition.source_topology_id != topology_id
                or geometry_transition.target_mesh.mesh_id != target.mesh.mesh_id
                or geometry_transition.target_topology_id != target.mesh.topology_id
            ):
                raise ValueError("Geometry transition endpoints do not match the mesh transition.")
            geometry_transition.require_bound(
                geometry_transition.source_mesh,
                geometry_transition.source_geometry,
                target.mesh,
                target.geometry,
            )
        if material_reference is not None and not isinstance(
            material_reference, MaterialReferenceDeclaration
        ):
            raise TypeError("material_reference must be MaterialReferenceDeclaration or None.")
        if volume_material_chain is not None:
            if not isinstance(volume_material_chain, VolumeMaterialChain):
                raise TypeError("volume_material_chain must be VolumeMaterialChain or None.")
            if (
                volume_material_chain.source_mesh_id != mesh_id
                or volume_material_chain.target_mesh_id != target.mesh.mesh_id
                or volume_material_chain.target_geometry_id != cell_geometry_id(target.geometry)
            ):
                raise ValueError("Volume material chain endpoints do not match the mesh transition.")
            chain_reference = volume_material_chain.reference
            if material_reference is not None and (
                material_reference.declaration_id != chain_reference.declaration_id
            ):
                raise ValueError("Transition material reference differs from its retained material chain.")
            material_reference = chain_reference
        if material_reference is not None:
            material_reference = MaterialReferenceDeclaration.revalidate(
                material_reference,
                material_reference.root_mesh,
                material_reference.root_geometry,
                material_reference.source_map,
                material_reference.root_coverage,
            )
            if volume_material_chain is None and (
                material_reference.root_mesh.mesh_id != target.mesh.mesh_id
                or material_reference.root_mesh.topology_id != target.mesh.topology_id
                or cell_geometry_id(material_reference.root_geometry)
                != cell_geometry_id(target.geometry)
                or (
                    geometry_transition is not None
                    and cell_geometry_id(material_reference.root_geometry)
                    != geometry_transition.target_geometry_id
                )
            ):
                raise ValueError(
                    "An unchained material reference must bind this transition's accepted target state."
                )
        self.source_mesh_id = mesh_id
        self.source_topology_id = topology_id
        self.target = target
        self.lineage = lineage
        self.vertex_stencil = vertex_stencil
        self.geometry_transition = geometry_transition
        self.volume_material_chain = volume_material_chain
        self.material_reference = material_reference
        self.transition_kind = transition_kind
        self.transition_id = canonical_fingerprint(
            {
                "kind": "cell-mesh-transition",
                "source_mesh": mesh_id,
                "source_topology": topology_id,
                "target": target.result_id,
                "lineage": lineage.lineage_id,
                "vertex_stencil": None
                if vertex_stencil is None
                else vertex_stencil.stencil_id,
                "geometry_transition": (
                    None
                    if geometry_transition is None
                    else geometry_transition.transition_id
                ),
                "volume_material_chain": (
                    None
                    if volume_material_chain is None
                    else volume_material_chain.chain_id
                ),
                "material_reference": (
                    None
                    if material_reference is None
                    else material_reference.declaration_id
                ),
                "transition_kind": transition_kind.value,
            }
        )


__all__ = [
    "CellMeshTransition",
    "EntityLineage",
    "EntityLineageKind",
    "MeshLineage",
    "MeshTransitionKind",
    "VertexInterpolationStencil",
    "interpolation_rows_preserve_constants",
]
