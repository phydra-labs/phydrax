#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellComplexTopology
from ._coefficients import CoefficientDomain, PrimeField, RationalField
from ._complex import (
    CellComplexPair,
    CellSubcomplex,
    compact_boundary,
    CompactBoundary,
    CompactCellLayout,
)
from ._reduction import (
    field_columns,
    field_rank,
    FieldVector,
    homology_representatives,
    integer_columns,
    rational_rank,
    ReductionStats,
    transpose_columns,
    verify_boundary_composition,
    verify_integer_boundary_composition,
)
from ._resources import (
    TopologyReductionEvidence,
    TopologyResourceError,
    TopologyResourcePolicy,
)


RepresentativeKind: TypeAlias = Literal["none", "cycles", "cocycles", "both"]
BasisKind: TypeAlias = Literal["chain", "cochain"]


class FiniteFieldBasis(StrictModule, NonTrainableState):
    """Sparse-storage finite-field generators in ambient cell coordinates."""

    cell_indices: Array
    entity_ids: Array
    generator_indices: Array
    coefficients: Array
    degree: int = eqx.field(static=True)
    kind: BasisKind = eqx.field(static=True)
    generator_count: int = eqx.field(static=True)
    ambient_cell_count: int = eqx.field(static=True)
    field: PrimeField
    entity_set_id: str = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        degree: int,
        kind: BasisKind,
        vectors: Sequence[FieldVector],
        layout: CompactCellLayout,
        topology: CellComplexTopology,
        field: PrimeField,
        /,
    ):
        degree_ = int(degree)
        if degree_ < 0 or degree_ > layout.max_degree:
            raise ValueError("Finite-field basis degree is outside the compact layout.")
        if kind not in ("chain", "cochain"):
            raise ValueError("Finite-field basis kind must be 'chain' or 'cochain'.")
        if not isinstance(field, PrimeField):
            raise TypeError("Finite-field bases require a PrimeField.")
        cells = []
        entities = []
        generators = []
        coefficients = []
        compact_to_ambient = np.asarray(layout.compact_to_ambient[degree_])
        entity_values = np.asarray(topology.entity_sets[degree_].entity_ids)
        for generator, vector in enumerate(vectors):
            for compact_index, coefficient in sorted(vector.items()):
                if compact_index < 0 or compact_index >= compact_to_ambient.size:
                    raise ValueError(
                        "Basis coefficient addresses an invalid compact cell."
                    )
                normalized = field.normalize(coefficient)
                if normalized == 0:
                    continue
                ambient_index = int(compact_to_ambient[compact_index])
                cells.append(ambient_index)
                entities.append(int(entity_values[ambient_index]))
                generators.append(generator)
                coefficients.append(normalized)
        cell_array = np.asarray(cells, dtype=np.int32)
        entity_array = np.asarray(entities, dtype=np.int64)
        generator_array = np.asarray(generators, dtype=np.int32)
        coefficient_array = np.asarray(coefficients, dtype=np.int64)
        self.cell_indices = jnp.asarray(cell_array)
        self.entity_ids = jnp.asarray(entity_array)
        self.generator_indices = jnp.asarray(generator_array)
        self.coefficients = jnp.asarray(coefficient_array)
        self.degree = degree_
        self.kind = kind
        self.generator_count = len(tuple(vectors))
        self.ambient_cell_count = topology.entity_sets[degree_].count
        self.field = field
        self.entity_set_id = topology.entity_sets[degree_].entity_set_id
        self.basis_id = canonical_fingerprint(
            {
                "kind": "finite-field-basis",
                "basis_kind": kind,
                "degree": degree_,
                "layout": layout.layout_id,
                "field": field.field_id,
                "cells": array_tree_fingerprint(cell_array),
                "generators": array_tree_fingerprint(generator_array),
                "coefficients": array_tree_fingerprint(coefficient_array),
            }
        )

    @property
    def nonzero_count(self) -> int:
        return int(self.coefficients.shape[0])


class HomologyDegreeResult(StrictModule, NonTrainableState):
    """Exact homology dimensions and optional generators in one degree."""

    degree: int = eqx.field(static=True)
    chain_dimension: int = eqx.field(static=True)
    boundary_rank: int = eqx.field(static=True)
    incoming_boundary_rank: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    cycles: FiniteFieldBasis | None
    cocycles: FiniteFieldBasis | None
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        degree: int,
        chain_dimension: int,
        boundary_rank: int,
        incoming_boundary_rank: int,
        /,
        *,
        cycles: FiniteFieldBasis | None = None,
        cocycles: FiniteFieldBasis | None = None,
    ):
        degree_ = int(degree)
        chain_size = int(chain_dimension)
        outgoing = int(boundary_rank)
        incoming = int(incoming_boundary_rank)
        dimension = chain_size - outgoing - incoming
        if chain_size < 0 or outgoing < 0 or incoming < 0 or dimension < 0:
            raise ValueError("Homology ranks are inconsistent with the chain dimension.")
        if cycles is not None and cycles.generator_count != dimension:
            raise ValueError("Cycle representative count must equal homology dimension.")
        if cocycles is not None and cocycles.generator_count != dimension:
            raise ValueError(
                "Cocycle representative count must equal homology dimension."
            )
        self.degree = degree_
        self.chain_dimension = chain_size
        self.boundary_rank = outgoing
        self.incoming_boundary_rank = incoming
        self.dimension = dimension
        self.cycles = cycles
        self.cocycles = cocycles
        self.result_id = canonical_fingerprint(
            {
                "kind": "homology-degree-result",
                "degree": degree_,
                "chain_dimension": chain_size,
                "boundary_rank": outgoing,
                "incoming_boundary_rank": incoming,
                "dimension": dimension,
                "cycles": None if cycles is None else cycles.basis_id,
                "cocycles": None if cocycles is None else cocycles.basis_id,
            }
        )


class HomologyResult(StrictModule, NonTrainableState):
    """Exact prime-field homology of one compact complex or pair."""

    degrees: tuple[HomologyDegreeResult, ...]
    field: PrimeField
    evidence: TopologyReductionEvidence
    source_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    reduced: bool = eqx.field(static=True)
    euler_characteristic: int = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        degrees: Sequence[HomologyDegreeResult],
        field: PrimeField,
        evidence: TopologyReductionEvidence,
        /,
        *,
        source_id: str,
        topology_id: str,
        reduced: bool,
        euler_characteristic: int,
    ):
        values = tuple(degrees)
        indices = tuple(value.degree for value in values)
        if indices != tuple(sorted(set(indices))):
            raise ValueError("Homology degree results must be unique and ordered.")
        self.degrees = values
        self.field = field
        self.evidence = evidence
        self.source_id = str(source_id)
        self.topology_id = str(topology_id)
        self.reduced = bool(reduced)
        self.euler_characteristic = int(euler_characteristic)
        self.result_id = canonical_fingerprint(
            {
                "kind": "homology-result",
                "source": self.source_id,
                "topology": self.topology_id,
                "field": field.field_id,
                "reduced": bool(reduced),
                "euler_characteristic": int(euler_characteristic),
                "degrees": [value.result_id for value in values],
                "evidence": evidence.evidence_id,
            }
        )

    def degree(self, degree: int, /) -> HomologyDegreeResult:
        target = int(degree)
        for value in self.degrees:
            if value.degree == target:
                return value
        raise KeyError(f"No homology result was requested for degree {target}.")

    @property
    def dimensions(self) -> tuple[int, ...]:
        return tuple(value.dimension for value in self.degrees)


class BettiDimensionResult(StrictModule, NonTrainableState):
    """Exact field-qualified homology dimensions without representatives."""

    degrees: tuple[int, ...] = eqx.field(static=True)
    dimensions: tuple[int, ...] = eqx.field(static=True)
    coefficient_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    reduced: bool = eqx.field(static=True)
    evidence: TopologyReductionEvidence
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        degrees: Sequence[int],
        dimensions: Sequence[int],
        coefficient_id: str,
        source_id: str,
        topology_id: str,
        evidence: TopologyReductionEvidence,
        /,
        *,
        reduced: bool,
    ):
        degrees_ = tuple(int(value) for value in degrees)
        dimensions_ = tuple(int(value) for value in dimensions)
        if len(degrees_) != len(dimensions_) or any(value < 0 for value in dimensions_):
            raise ValueError("Betti degrees and dimensions are inconsistent.")
        self.degrees = degrees_
        self.dimensions = dimensions_
        self.coefficient_id = str(coefficient_id)
        self.source_id = str(source_id)
        self.topology_id = str(topology_id)
        self.reduced = bool(reduced)
        self.evidence = evidence
        self.result_id = canonical_fingerprint(
            {
                "kind": "betti-dimension-result",
                "degrees": list(degrees_),
                "dimensions": list(dimensions_),
                "coefficient": self.coefficient_id,
                "source": self.source_id,
                "topology": self.topology_id,
                "reduced": bool(reduced),
                "evidence": evidence.evidence_id,
            }
        )

    def dimension(self, degree: int, /) -> int:
        target = int(degree)
        for current, value in zip(self.degrees, self.dimensions, strict=True):
            if current == target:
                return value
        raise KeyError(f"No Betti dimension was requested for degree {target}.")


def _resolved_complex(
    value: CellComplexTopology | CellSubcomplex | CellComplexPair,
    /,
) -> CellSubcomplex | CellComplexPair:
    if isinstance(value, CellComplexTopology):
        return CellSubcomplex.full(value)
    if isinstance(value, (CellSubcomplex, CellComplexPair)):
        return value
    raise TypeError(
        "Topology analysis requires CellComplexTopology, CellSubcomplex, or "
        "CellComplexPair."
    )


def _layout(value: CellSubcomplex | CellComplexPair, /) -> CompactCellLayout:
    return value.layout if isinstance(value, CellSubcomplex) else value.quotient_layout


def _source_id(value: CellSubcomplex | CellComplexPair, /) -> str:
    return value.subcomplex_id if isinstance(value, CellSubcomplex) else value.pair_id


def _boundaries(
    value: CellSubcomplex | CellComplexPair,
    policy: TopologyResourcePolicy,
    /,
    *,
    reduced: bool,
) -> tuple[CompactBoundary, ...]:
    layout = _layout(value)
    if layout.cell_count > policy.max_cells:
        raise TopologyResourceError("Compact topology exceeds max_cells.")
    boundaries = [
        compact_boundary(value, degree) for degree in range(layout.max_degree + 1)
    ]
    if reduced:
        if isinstance(value, CellComplexPair):
            raise ValueError("Reduced relative homology is not supported.")
        count = layout.counts[0]
        boundaries[0] = CompactBoundary(
            0,
            1,
            count,
            np.zeros((count,), dtype=np.int32),
            np.arange(count, dtype=np.int32),
            np.ones((count,), dtype=np.int64),
            source_id=f"{value.subcomplex_id}:augmentation",
        )
    nonzeros = sum(boundary.nonzero_count for boundary in boundaries)
    if nonzeros > policy.max_boundary_nonzeros:
        raise TopologyResourceError("Compact topology exceeds max_boundary_nonzeros.")
    integer = tuple(integer_columns(boundary) for boundary in boundaries)
    for lower, upper in zip(integer[:-1], integer[1:], strict=True):
        verify_integer_boundary_composition(lower, upper, policy)
    return tuple(boundaries)


def _requested_degrees(
    layout: CompactCellLayout,
    degrees: Sequence[int] | None,
    /,
    *,
    reduced: bool,
) -> tuple[int, ...]:
    available = tuple(range(-1 if reduced else 0, layout.max_degree + 1))
    if degrees is None:
        return available
    requested = tuple(sorted(set(int(value) for value in degrees)))
    if any(value not in available for value in requested):
        raise ValueError(f"Requested homology degrees must lie in {available}.")
    return requested


def _verify_field_boundaries(
    columns: tuple[list[FieldVector], ...],
    field: PrimeField,
    policy: TopologyResourcePolicy,
    /,
) -> ReductionStats:
    aggregate = ReductionStats()
    for lower, upper in zip(columns[:-1], columns[1:], strict=True):
        current = verify_boundary_composition(lower, upper, field, policy)
        aggregate.operations += current.operations
    return aggregate


def _field_basis(
    degree: int,
    kind: BasisKind,
    vectors: Sequence[FieldVector],
    value: CellSubcomplex | CellComplexPair,
    field: PrimeField,
    /,
) -> FiniteFieldBasis:
    return FiniteFieldBasis(
        degree,
        kind,
        vectors,
        _layout(value),
        value.topology,
        field,
    )


def _verify_euler(
    layout: CompactCellLayout,
    results: Sequence[HomologyDegreeResult],
    /,
    *,
    reduced: bool,
) -> int:
    ordinary = sum(((-1) ** degree) * count for degree, count in enumerate(layout.counts))
    available = tuple(range(-1 if reduced else 0, layout.max_degree + 1))
    if tuple(value.degree for value in results) != available:
        return ordinary
    expected = ordinary - 1 if reduced else ordinary
    homological = sum(
        (-1 if value.degree == -1 else (-1) ** value.degree) * value.dimension
        for value in results
    )
    if homological != expected:
        raise RuntimeError("Exact homology violates the Euler–Poincaré identity.")
    return ordinary


def compute_homology(
    complex_or_pair: CellComplexTopology | CellSubcomplex | CellComplexPair,
    /,
    *,
    coefficients: PrimeField,
    degrees: Sequence[int] | None = None,
    reduced: bool = False,
    representatives: RepresentativeKind = "none",
    resources: TopologyResourcePolicy | None = None,
) -> HomologyResult:
    """Compute exact prime-field homology and optional generators on the host."""
    if not isinstance(coefficients, PrimeField):
        raise TypeError("compute_homology requires an explicit PrimeField.")
    if representatives not in ("none", "cycles", "cocycles", "both"):
        raise ValueError("Unknown homology representative policy.")
    policy = TopologyResourcePolicy() if resources is None else resources
    if not isinstance(policy, TopologyResourcePolicy):
        raise TypeError("resources must be a TopologyResourcePolicy.")
    value = _resolved_complex(complex_or_pair)
    layout = _layout(value)
    boundaries = _boundaries(value, policy, reduced=bool(reduced))
    columns = tuple(field_columns(boundary, coefficients) for boundary in boundaries)
    verification = _verify_field_boundaries(columns, coefficients, policy)
    requested = _requested_degrees(layout, degrees, reduced=bool(reduced))
    results = []
    operations = verification.operations
    peak_entries = 0
    representative_entries = 0
    for degree in requested:
        if degree == -1:
            incoming_rank = 0 if layout.counts[0] == 0 else 1
            results.append(HomologyDegreeResult(-1, 1, 0, incoming_rank))
            continue
        boundary = columns[degree]
        incoming = columns[degree + 1] if degree < layout.max_degree else []
        cycles = None
        cocycles = None
        if representatives in ("cycles", "both"):
            chain_vectors, outgoing_rank, incoming_rank, stats = homology_representatives(
                boundary,
                incoming,
                coefficients,
                policy,
            )
            cycles = _field_basis(degree, "chain", chain_vectors, value, coefficients)
        else:
            outgoing_rank, stats = field_rank(boundary, coefficients, policy)
            incoming_rank, stats = field_rank(
                incoming,
                coefficients,
                policy,
                stats=stats,
            )
        if representatives in ("cocycles", "both"):
            outgoing_cochain = transpose_columns(
                incoming,
                layout.counts[degree],
                coefficients,
            )
            incoming_cochain = (
                transpose_columns(
                    boundary,
                    boundaries[degree].row_count,
                    coefficients,
                )
                if degree > 0 or reduced
                else []
            )
            cochain_vectors, cochain_outgoing, cochain_incoming, stats = (
                homology_representatives(
                    outgoing_cochain,
                    incoming_cochain,
                    coefficients,
                    policy,
                    stats=stats,
                )
            )
            if cochain_outgoing != incoming_rank or cochain_incoming != outgoing_rank:
                raise RuntimeError("Chain and cochain exact ranks disagree.")
            cocycles = _field_basis(
                degree,
                "cochain",
                cochain_vectors,
                value,
                coefficients,
            )
        result = HomologyDegreeResult(
            degree,
            layout.counts[degree],
            outgoing_rank,
            incoming_rank,
            cycles=cycles,
            cocycles=cocycles,
        )
        results.append(result)
        operations += stats.operations
        peak_entries = max(peak_entries, stats.peak_entries)
        representative_entries = max(
            representative_entries,
            stats.representative_entries,
        )
    euler = _verify_euler(layout, results, reduced=bool(reduced))
    evidence = TopologyReductionEvidence(
        "exact-prime-field-homology",
        coefficients.field_id,
        {
            "cells": layout.cell_count,
            "boundary_nonzeros": sum(value.nonzero_count for value in boundaries),
            "operations": operations,
            "peak_reduction_entries": peak_entries,
            "representative_entries": representative_entries,
        },
    )
    return HomologyResult(
        results,
        coefficients,
        evidence,
        source_id=_source_id(value),
        topology_id=value.topology.topology_id,
        reduced=bool(reduced),
        euler_characteristic=euler,
    )


def compute_betti_dimensions(
    complex_or_pair: CellComplexTopology | CellSubcomplex | CellComplexPair,
    /,
    *,
    coefficients: CoefficientDomain,
    degrees: Sequence[int] | None = None,
    reduced: bool = False,
    resources: TopologyResourcePolicy | None = None,
) -> BettiDimensionResult:
    """Compute exact field-qualified homology dimensions without representatives."""
    policy = TopologyResourcePolicy() if resources is None else resources
    if not isinstance(policy, TopologyResourcePolicy):
        raise TypeError("resources must be a TopologyResourcePolicy.")
    value = _resolved_complex(complex_or_pair)
    layout = _layout(value)
    boundaries = _boundaries(value, policy, reduced=bool(reduced))
    requested = _requested_degrees(layout, degrees, reduced=bool(reduced))
    if isinstance(coefficients, PrimeField):
        result = compute_homology(
            value,
            coefficients=coefficients,
            degrees=requested,
            reduced=bool(reduced),
            resources=policy,
        )
        return BettiDimensionResult(
            tuple(value.degree for value in result.degrees),
            tuple(value.dimension for value in result.degrees),
            coefficients.field_id,
            result.source_id,
            result.topology_id,
            result.evidence,
            reduced=bool(reduced),
        )
    if not isinstance(coefficients, RationalField):
        raise TypeError("Unknown exact topology coefficient domain.")
    ranks = []
    operations = 0
    peak_entries = 0
    maximum_bits = 0
    for boundary in boundaries:
        rank, stats = rational_rank(boundary, policy)
        ranks.append(rank)
        operations += stats.operations
        peak_entries = max(peak_entries, stats.peak_entries)
        maximum_bits = max(maximum_bits, stats.maximum_bit_length)
    dimensions = []
    for degree in requested:
        if degree == -1:
            dimensions.append(1 if layout.counts[0] == 0 else 0)
            continue
        incoming = ranks[degree + 1] if degree < layout.max_degree else 0
        dimension = layout.counts[degree] - ranks[degree] - incoming
        if dimension < 0:
            raise RuntimeError("Exact rational ranks produce a negative Betti dimension.")
        dimensions.append(dimension)
    evidence = TopologyReductionEvidence(
        "exact-rational-rank-homology",
        coefficients.field_id,
        {
            "cells": layout.cell_count,
            "boundary_nonzeros": sum(value.nonzero_count for value in boundaries),
            "operations": operations,
            "peak_reduction_entries": peak_entries,
            "maximum_coefficient_bits": maximum_bits,
        },
    )
    return BettiDimensionResult(
        requested,
        dimensions,
        coefficients.field_id,
        _source_id(value),
        value.topology.topology_id,
        evidence,
        reduced=bool(reduced),
    )


__all__ = [
    "BettiDimensionResult",
    "FiniteFieldBasis",
    "HomologyDegreeResult",
    "HomologyResult",
    "RepresentativeKind",
    "compute_betti_dimensions",
    "compute_homology",
]
