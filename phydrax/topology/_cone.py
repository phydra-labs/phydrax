#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._coefficients import PrimeField
from ._complex import compact_boundary
from ._integer import block_matrix, ExactChainComplex, ExactIntegerCOO
from ._maps import CellularChainMap
from ._reduction import field_rank
from ._resources import TopologyReductionEvidence, TopologyResourcePolicy


class MappingConeResult(StrictModule, NonTrainableState):
    """Field-qualified homology dimensions of one exact mapping cone."""

    cone: ExactChainComplex
    field: PrimeField
    dimensions: tuple[int, ...] = eqx.field(static=True)
    acyclic: bool = eqx.field(static=True)
    evidence: TopologyReductionEvidence
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        cone: ExactChainComplex,
        field: PrimeField,
        dimensions: Sequence[int],
        evidence: TopologyReductionEvidence,
        /,
    ):
        values = tuple(int(value) for value in dimensions)
        if len(values) != len(cone.counts) or any(value < 0 for value in values):
            raise ValueError("Mapping-cone dimensions do not match the chain complex.")
        self.cone = cone
        self.field = field
        self.dimensions = values
        self.acyclic = all(value == 0 for value in values)
        self.evidence = evidence
        self.result_id = canonical_fingerprint(
            {
                "kind": "mapping-cone-result",
                "cone": cone.complex_id,
                "field": field.field_id,
                "dimensions": list(values),
                "evidence": evidence.evidence_id,
            }
        )


def _boundary_matrix(complex, degree: int, /) -> ExactIntegerCOO:
    return ExactIntegerCOO.from_boundary(compact_boundary(complex, degree))


def mapping_cone(chain_map: CellularChainMap, /) -> ExactChainComplex:
    """Construct ``Cone(F)_n = C_n(Y) ⊕ C_{n-1}(X)`` exactly."""
    maximum = len(chain_map.degree_maps)
    source_counts = chain_map.source.layout.counts
    target_counts = chain_map.target.layout.counts
    boundaries = []
    cone_counts = tuple(
        (target_counts[degree] if degree < len(target_counts) else 0)
        + (source_counts[degree - 1] if 0 < degree <= len(source_counts) else 0)
        for degree in range(maximum + 1)
    )
    for degree, column_count in enumerate(cone_counts):
        row_count = 0 if degree == 0 else cone_counts[degree - 1]
        source_id = f"cone:{chain_map.map_id}:degree:{degree}"
        target_id = f"cone:{chain_map.map_id}:degree:{degree - 1}"
        if degree == 0:
            boundaries.append(
                ExactIntegerCOO.zero(
                    0,
                    column_count,
                    source_id=source_id,
                    target_id=target_id,
                )
            )
            continue
        y_degree_count = target_counts[degree] if degree < len(target_counts) else 0
        x_shifted_count = source_counts[degree - 1]
        y_lower_count = target_counts[degree - 1]
        x_lower_shifted_count = source_counts[degree - 2] if degree > 1 else 0
        blocks: dict[tuple[int, int], ExactIntegerCOO] = {}
        if degree < len(target_counts):
            blocks[(0, 0)] = _boundary_matrix(chain_map.target, degree)
        else:
            blocks[(0, 0)] = ExactIntegerCOO.zero(
                y_lower_count,
                y_degree_count,
                source_id=f"{chain_map.target.subcomplex_id}:degree:{degree}",
                target_id=f"{chain_map.target.subcomplex_id}:degree:{degree - 1}",
            )
        blocks[(0, 1)] = chain_map.degree_maps[degree - 1]
        if degree > 1:
            blocks[(1, 1)] = _boundary_matrix(
                chain_map.source,
                degree - 1,
            ).scale(-1)
        else:
            blocks[(1, 1)] = ExactIntegerCOO.zero(
                0,
                x_shifted_count,
                source_id=f"{chain_map.source.subcomplex_id}:degree:0",
                target_id=f"{chain_map.source.subcomplex_id}:degree:-1",
            )
        boundaries.append(
            block_matrix(
                blocks,
                (y_lower_count, x_lower_shifted_count),
                (y_degree_count, x_shifted_count),
                source_id=source_id,
                target_id=target_id,
            )
        )
        if boundaries[-1].row_count != row_count:
            raise RuntimeError("Mapping-cone boundary shape is inconsistent.")
    return ExactChainComplex(
        boundaries,
        complex_id=f"mapping-cone:{chain_map.map_id}",
    )


def _field_columns(matrix: ExactIntegerCOO, field: PrimeField, /):
    columns = []
    for column in matrix.columns():
        values = {
            row: field.normalize(value)
            for row, value in column.items()
            if field.normalize(value)
        }
        columns.append(values)
    return columns


def compute_mapping_cone_homology(
    chain_map: CellularChainMap,
    /,
    *,
    coefficients: PrimeField,
    resources: TopologyResourcePolicy | None = None,
) -> MappingConeResult:
    """Audit a chain map as a quasi-isomorphism over one explicit prime field."""
    policy = TopologyResourcePolicy() if resources is None else resources
    cone = mapping_cone(chain_map)
    ranks = []
    operations = 0
    peak_entries = 0
    for boundary in cone.boundaries:
        rank, stats = field_rank(
            _field_columns(boundary, coefficients),
            coefficients,
            policy,
        )
        ranks.append(rank)
        operations += stats.operations
        peak_entries = max(peak_entries, stats.peak_entries)
    dimensions = tuple(
        cone.counts[degree]
        - ranks[degree]
        - (ranks[degree + 1] if degree + 1 < len(ranks) else 0)
        for degree in range(len(cone.counts))
    )
    evidence = TopologyReductionEvidence(
        "exact-mapping-cone-homology",
        coefficients.field_id,
        {
            "cells": sum(cone.counts),
            "boundary_nonzeros": sum(value.nonzero_count for value in cone.boundaries),
            "operations": operations,
            "peak_reduction_entries": peak_entries,
        },
    )
    return MappingConeResult(cone, coefficients, dimensions, evidence)


__all__ = ["MappingConeResult", "compute_mapping_cone_homology", "mapping_cone"]
