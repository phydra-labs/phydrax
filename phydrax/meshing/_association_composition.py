#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Compose independent source-association namespaces without identity loss."""

from __future__ import annotations

from typing import final

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh
from ..discretization._cell_geometry_transition_types import CellGeometryTransition
from ..geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryFaceImageWitness,
)
from ..geometry._source_fidelity import ValidatedSourceFidelityWitness
from ..typing import checked
from ._association import (
    AssociationEntityClasses,
    AssociationTransfer,
    GeometryAssociation,
)
from ._certification_inputs import ScopedFidelityInput
from ._lineage import MeshLineage, VertexInterpolationStencil
from ._result import CellMeshingResult


def _joint_codes(
    levels: tuple[tuple[AssociationEntityClasses, ...], ...],
    dimension: int,
    row_count: int,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dimensions = np.full(row_count, -1, dtype=np.int64)
    indices = np.full(row_count, -1, dtype=np.int64)
    resolved = np.zeros(row_count, dtype=np.bool_)
    source_codes: list[np.ndarray] = []
    for bank in levels:
        level = bank[dimension]
        mask = np.asarray(level.resolved, dtype=np.bool_)
        if mask.shape != (row_count,):
            raise ValueError(
                "Association namespace class rows must align with the carrier mesh."
            )
        dims = np.asarray(level.dimensions, dtype=np.int64)
        rows = mask & (dims >= 0)
        dimensions[rows & ((dimensions < 0) | (dims < dimensions))] = dims[
            rows & ((dimensions < 0) | (dims < dimensions))
        ]
        indices[(indices < 0) & rows] = np.asarray(level.indices, dtype=np.int64)[
            (indices < 0) & rows
        ]
        resolved |= rows
        source_codes.append(np.where(rows, np.asarray(level.codes, dtype=np.int64), -1))
    keys = tuple(
        sorted(
            {
                tuple(int(values[row]) for values in source_codes)
                for row in np.flatnonzero(resolved).tolist()
            }
        )
    )
    code_by_key = {key: index for index, key in enumerate(keys)}
    codes = np.full(row_count, -1, dtype=np.int64)
    for row in np.flatnonzero(resolved).tolist():
        codes[row] = code_by_key[tuple(int(values[row]) for values in source_codes)]
    return dimensions, indices, np.column_stack((resolved, codes)).astype(np.int64)


@final
class ComposedAssociationTransfer(StrictModule, NonTrainableState):
    """Combine transfer routes for disjoint source namespaces."""

    transfers: tuple[AssociationTransfer, ...]
    transfer_id: str = eqx.field(static=True)

    def __init__(self, transfers: tuple[AssociationTransfer, ...], /) -> None:
        from ._association import BRepAssociationTransfer
        from ._implicit_association_transfer import ImplicitAssociationTransfer
        from ._layer_core_association_transfer import NativeLayerCoreAssociationTransfer
        from ._mapped_reference_association import MappedReferenceAssociationTransfer
        from ._surface_association_transfer import SurfaceAssociationTransfer

        admitted = (
            BRepAssociationTransfer,
            ImplicitAssociationTransfer,
            MappedReferenceAssociationTransfer,
            NativeLayerCoreAssociationTransfer,
            SurfaceAssociationTransfer,
            ComposedAssociationTransfer,
        )
        if (
            type(transfers) is not tuple
            or len(transfers) < 2
            or any(not isinstance(value, admitted) for value in transfers)
        ):
            raise TypeError(
                "transfers must contain at least two source-association transfer routes."
            )
        namespaces = tuple(
            namespace for value in transfers for namespace in value.source_namespaces
        )
        if len(set(namespaces)) != len(namespaces):
            raise ValueError("Composed source-association namespaces must be disjoint.")
        ordered = tuple(
            sorted(
                transfers, key=lambda value: (value.source_namespaces, value.transfer_id)
            )
        )
        self.transfers = ordered
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "composed-association-transfer",
                "transfers": tuple(value.transfer_id for value in ordered),
                "namespaces": tuple(sorted(namespaces)),
            }
        )

    @property
    def source_namespaces(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                namespace
                for value in self.transfers
                for namespace in value.source_namespaces
            )
        )

    def validate_source(self, source: CellMeshingResult, /) -> None:
        if not isinstance(source, CellMeshingResult):
            raise TypeError("source must be CellMeshingResult.")
        for transfer in self.transfers:
            transfer.validate_source(source)

    def classes(
        self, source: CellMeshingResult, /
    ) -> tuple[AssociationEntityClasses, ...]:
        self.validate_source(source)
        banks = tuple(transfer.classes(source) for transfer in self.transfers)
        result = []
        for dimension in range(source.mesh.topological_dimension + 1):
            dims, indices, state = _joint_codes(
                banks,
                dimension,
                source.mesh.entity_set(dimension).count,
            )
            result.append(
                AssociationEntityClasses(
                    dims,
                    indices,
                    state[:, 0].astype(np.bool_),
                    state[:, 1],
                )
            )
        return tuple(result)

    @checked
    def propagate(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: CellMesh,
        /,
        *,
        vertex_stencil: VertexInterpolationStencil | None,
        geometry_transition: CellGeometryTransition | None,
        target_fidelity_witness: SourceBoundaryChartWitness
        | SourceBoundaryFaceImageWitness
        | None,
        target_scoped_fidelity: tuple[ScopedFidelityInput, ...],
        validated_fidelity: tuple[ValidatedSourceFidelityWitness, ...],
    ) -> tuple[GeometryAssociation, ...]:
        self.validate_source(source)
        if not isinstance(lineage, MeshLineage) or not isinstance(target, CellMesh):
            raise TypeError("lineage and target must be MeshLineage and CellMesh values.")
        records = tuple(
            association
            for transfer in self.transfers
            for association in transfer.propagate(
                source,
                lineage,
                target,
                vertex_stencil=vertex_stencil,
                geometry_transition=geometry_transition,
                target_fidelity_witness=target_fidelity_witness,
                target_scoped_fidelity=target_scoped_fidelity,
                validated_fidelity=validated_fidelity,
            )
        )
        return tuple(
            sorted(
                records,
                key=lambda value: (
                    value.source_id,
                    value.source_revision,
                    value.target_entity_set_id,
                    value.association_id,
                ),
            )
        )


__all__ = ["ComposedAssociationTransfer"]
