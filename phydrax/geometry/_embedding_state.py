#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Compact mathematical findings and witnesses for embedding certificates."""

from __future__ import annotations

from fractions import Fraction
from typing import final, Literal, NamedTuple

import equinox as eqx
import numpy as np

from .._archive_columns import (
    integer_column,
    integer_column_values,
    TextColumnArchiveRecipe,
)
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier, nonnegative_integer
from ..typing import checked, parse


type MeshFindingStatus = Literal["violated", "unresolved"]
type MeshCertificateEntityKind = Literal[
    "mesh",
    "vertex",
    "cell",
    "facet",
    "source_entity",
    "source_cell",
    "source_facet",
    "source_region",
    "source_sample",
]
type CoordinateMapScope = Literal["affine", "mapped"]


class MeshCertificateFinding(StrictModule, NonTrainableState):
    """One mathematical violation or unresolved check and its source entities."""

    check: str
    status: MeshFindingStatus
    entity_kind: MeshCertificateEntityKind
    entity_ids: tuple[int, ...]
    detail: str
    finding_id: str

    def __init__(
        self,
        check: str,
        status: MeshFindingStatus,
        entity_kind: MeshCertificateEntityKind,
        entity_ids: tuple[int, ...] = (),
        /,
        *,
        detail: str = "",
    ) -> None:
        check_ = canonical_identifier(check, "finding_check")
        status_ = parse(status, MeshFindingStatus, "finding status")
        entity_kind_ = parse(
            entity_kind, MeshCertificateEntityKind, "certificate entity kind"
        )
        ids = tuple(entity_ids)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in ids
        ):
            raise ValueError("finding entity IDs must be nonnegative integers.")
        detail_ = str(detail)
        self.check = check_
        self.status = status_
        self.entity_kind = entity_kind_
        self.entity_ids = ids
        self.detail = detail_
        self.finding_id = canonical_fingerprint(
            {
                "kind": "mesh-certificate-finding",
                "check": check_,
                "status": status_,
                "entity_kind": entity_kind_,
                "entity_ids": ids,
                "detail": detail_,
            }
        )


class EmbeddingCounts(StrictModule, NonTrainableState):
    """Counts of bound cells and mathematical boundary entities."""

    cell_count: int
    source_bound_count: int
    boundary_facet_count: int
    shell_count: int

    def __init__(
        self,
        cell_count: int = 0,
        source_bound_count: int = 0,
        boundary_facet_count: int = 0,
        shell_count: int = 0,
    ) -> None:
        values = (cell_count, source_bound_count, boundary_facet_count, shell_count)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ValueError("Embedding entity counts must be nonnegative integers.")
        (
            self.cell_count,
            self.source_bound_count,
            self.boundary_facet_count,
            self.shell_count,
        ) = values


class MeshEmbeddingWitness(StrictModule, NonTrainableState):
    """Input-bound exact premises retained by one global embedding result."""

    theorem: str
    premise_ids: tuple[str, ...]
    entity_ids: tuple[int, ...]
    exact_values: tuple[str, ...]
    witness_id: str

    def __init__(
        self,
        theorem: str,
        premise_ids: tuple[str, ...],
        entity_ids: tuple[int, ...] = (),
        exact_values: tuple[str, ...] = (),
        /,
    ) -> None:
        theorem_ = canonical_identifier(theorem, "embedding_theorem")
        premises = tuple(
            canonical_identifier(value, "embedding_premise") for value in premise_ids
        )
        entities = tuple(entity_ids)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in entities
        ):
            raise ValueError("Embedding witness entity IDs must be nonnegative integers.")
        values = tuple(str(value) for value in exact_values)
        self.theorem = theorem_
        self.premise_ids = premises
        self.entity_ids = entities
        self.exact_values = values
        self.witness_id = canonical_fingerprint(
            {
                "kind": "mesh-embedding-witness",
                "theorem": theorem_,
                "premises": premises,
                "entities": entities,
                "exact_values": values,
            }
        )

    @classmethod
    def from_stored_fields(cls, witness: MeshEmbeddingWitness, /) -> MeshEmbeddingWitness:
        renewed = cls(
            witness.theorem,
            witness.premise_ids,
            witness.entity_ids,
            witness.exact_values,
        )
        if renewed.witness_id != witness.witness_id:
            raise ValueError("Stored mesh-embedding witness identity is malformed.")
        return renewed


class MeshEmbeddingWitnessArchiveRecipe(NamedTuple):
    """Archive form of an embedding witness with its per-entity rows as arrays.

    Entity IDs are one int64 array and exact values one UTF-8 text column, so a
    witness over every mesh entity does not put one manifest node per row.
    :meth:`restore` rebuilds through the constructor and requires the stored
    witness identity.
    """

    theorem: str
    premise_ids: tuple[str, ...]
    entity_ids: np.ndarray
    exact_values: TextColumnArchiveRecipe
    witness_id: str

    @classmethod
    def from_witness(
        cls, witness: MeshEmbeddingWitness, /
    ) -> MeshEmbeddingWitnessArchiveRecipe:
        if type(witness) is not MeshEmbeddingWitness:
            raise TypeError("witness must be MeshEmbeddingWitness.")
        return cls(
            witness.theorem,
            witness.premise_ids,
            integer_column(witness.entity_ids),
            TextColumnArchiveRecipe.encode(witness.exact_values),
            witness.witness_id,
        )

    def restore(self) -> MeshEmbeddingWitness:
        renewed = MeshEmbeddingWitness(
            self.theorem,
            self.premise_ids,
            integer_column_values(self.entity_ids),
            self.exact_values.decode(),
        )
        if renewed.witness_id != self.witness_id:
            raise ValueError("Stored mesh-embedding witness identity is malformed.")
        return renewed


class EmbeddingState(StrictModule):
    """Request-local ordered collection of mathematical embedding findings."""

    findings: list[MeshCertificateFinding]
    checks: list[str]

    def __init__(
        self,
        findings: tuple[MeshCertificateFinding, ...] = (),
        checks: tuple[str, ...] = (),
    ) -> None:
        if not all(isinstance(value, MeshCertificateFinding) for value in findings):
            raise TypeError("embedding findings must be MeshCertificateFinding values.")
        self.findings = list(findings)
        self.checks = [canonical_identifier(value, "embedding_check") for value in checks]

    def add_check(self, check: str, /) -> None:
        check_ = canonical_identifier(check, "embedding_check")
        if check_ not in self.checks:
            self.checks.append(check_)

    def add(
        self,
        check: str,
        status: MeshFindingStatus,
        entity_kind: MeshCertificateEntityKind,
        entity_ids: tuple[int, ...] = (),
        /,
        *,
        detail: str = "",
    ) -> None:
        self.findings.append(
            MeshCertificateFinding(check, status, entity_kind, entity_ids, detail=detail)
        )

    @property
    def status(self) -> Literal["certified", "violated", "unresolved"]:
        if any(value.status == "violated" for value in self.findings):
            return "violated"
        if self.findings:
            return "unresolved"
        return "certified"


def _positive_fraction(value: tuple[int, int], name: str, /) -> Fraction:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
        or value[0] <= 0
        or value[1] <= 0
    ):
        raise ValueError(f"{name} must be a positive exact integer ratio.")
    result = Fraction(value[0], value[1])
    if (result.numerator, result.denominator) != value:
        raise ValueError(f"{name} must be in reduced exact form.")
    return result


@final
class PeriodicAffineTorusDegreeEvidence(StrictModule, NonTrainableState):
    """Bind an exact degree-one proof to the current affine torus inputs."""

    mesh_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    periodic_topology_id: str = eqx.field(static=True)
    input_binding_id: str = eqx.field(static=True)
    affinity_certificate_id: str = eqx.field(static=True)
    manifold_certificate_id: str = eqx.field(static=True)
    trace_certificate_id: str = eqx.field(static=True)
    orientation_certificate_id: str = eqx.field(static=True)
    pullback_area: tuple[int, int] = eqx.field(static=True)
    lattice_area: tuple[int, int] = eqx.field(static=True)
    covering_degree: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh_id: str,
        geometry_id: str,
        periodic_topology_id: str,
        input_binding_id: str,
        affinity_certificate_id: str,
        manifold_certificate_id: str,
        trace_certificate_id: str,
        orientation_certificate_id: str,
        pullback_area: tuple[int, int],
        lattice_area: tuple[int, int],
    ) -> None:
        names = (
            "mesh_id",
            "geometry_id",
            "periodic_topology_id",
            "input_binding_id",
            "affinity_certificate_id",
            "manifold_certificate_id",
            "trace_certificate_id",
            "orientation_certificate_id",
        )
        identifiers = tuple(
            canonical_identifier(value, f"PeriodicAffineTorusDegreeEvidence.{name}")
            for name, value in zip(
                names,
                (
                    mesh_id,
                    geometry_id,
                    periodic_topology_id,
                    input_binding_id,
                    affinity_certificate_id,
                    manifold_certificate_id,
                    trace_certificate_id,
                    orientation_certificate_id,
                ),
                strict=True,
            )
        )
        pullback = _positive_fraction(
            pullback_area, "PeriodicAffineTorusDegreeEvidence.pullback_area"
        )
        lattice = _positive_fraction(
            lattice_area, "PeriodicAffineTorusDegreeEvidence.lattice_area"
        )
        degree = pullback / lattice
        if degree != 1:
            raise ValueError(
                "PeriodicAffineTorusDegreeEvidence must certify exact degree one."
            )
        (
            self.mesh_id,
            self.geometry_id,
            self.periodic_topology_id,
            self.input_binding_id,
            self.affinity_certificate_id,
            self.manifold_certificate_id,
            self.trace_certificate_id,
            self.orientation_certificate_id,
        ) = identifiers
        self.pullback_area = (pullback.numerator, pullback.denominator)
        self.lattice_area = (lattice.numerator, lattice.denominator)
        self.covering_degree = degree.numerator
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "periodic-affine-torus-degree-one",
                **dict(zip(names, identifiers, strict=True)),
                "pullback_area": self.pullback_area,
                "lattice_area": self.lattice_area,
                "covering_degree": self.covering_degree,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls, value: PeriodicAffineTorusDegreeEvidence, /
    ) -> PeriodicAffineTorusDegreeEvidence:
        if not isinstance(value, cls):
            raise TypeError("value must be PeriodicAffineTorusDegreeEvidence.")
        rebuilt = cls(
            value.mesh_id,
            value.geometry_id,
            value.periodic_topology_id,
            value.input_binding_id,
            value.affinity_certificate_id,
            value.manifold_certificate_id,
            value.trace_certificate_id,
            value.orientation_certificate_id,
            value.pullback_area,
            value.lattice_area,
        )
        if (
            rebuilt.mesh_id != value.mesh_id
            or rebuilt.geometry_id != value.geometry_id
            or rebuilt.periodic_topology_id != value.periodic_topology_id
            or rebuilt.input_binding_id != value.input_binding_id
            or rebuilt.affinity_certificate_id != value.affinity_certificate_id
            or rebuilt.manifold_certificate_id != value.manifold_certificate_id
            or rebuilt.trace_certificate_id != value.trace_certificate_id
            or rebuilt.orientation_certificate_id != value.orientation_certificate_id
            or rebuilt.pullback_area != value.pullback_area
            or rebuilt.lattice_area != value.lattice_area
            or rebuilt.covering_degree != value.covering_degree
            or rebuilt.evidence_id != value.evidence_id
        ):
            raise ValueError(
                "Stored PeriodicAffineTorusDegreeEvidence does not reproduce its exact theorem."
            )
        return rebuilt


def _fraction_pair(value: tuple[int, int], name: str, /) -> Fraction:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
        or value[1] <= 0
    ):
        raise ValueError(f"{name} must be an exact integer ratio.")
    result = Fraction(value[0], value[1])
    if (result.numerator, result.denominator) != value:
        raise ValueError(f"{name} must be in reduced exact form.")
    return result


type PeriodicPartitionDirection = Literal["refine", "coarsen"]


@final
class PeriodicRestrictionPartitionEvidence(StrictModule, NonTrainableState):
    """Bind the exact convex parent partition and its source proof."""

    mesh_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    periodic_topology_id: str = eqx.field(static=True)
    validity_certificate_id: str = eqx.field(static=True)
    facet_trace_certificate_id: str = eqx.field(static=True)
    boundary_subdivision_certificate_id: str = eqx.field(static=True)
    identification_id: str = eqx.field(static=True)
    source_mesh_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    source_validity_certificate_id: str = eqx.field(static=True)
    source_embedding_certificate_id: str = eqx.field(static=True)
    planes: tuple[tuple[tuple[int, int], ...], ...] = eqx.field(static=True)
    child_cell_ids: tuple[int, ...] = eqx.field(static=True)
    parent_cell_ids: tuple[int, ...] = eqx.field(static=True)
    sign_vectors: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    child_volumes: tuple[tuple[int, int], ...] = eqx.field(static=True)
    partition_cell_ids: tuple[int, ...] = eqx.field(static=True)
    partition_volumes: tuple[tuple[int, int], ...] = eqx.field(static=True)
    direction: PeriodicPartitionDirection = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh_id: str,
        geometry_id: str,
        periodic_topology_id: str,
        validity_certificate_id: str,
        facet_trace_certificate_id: str,
        boundary_subdivision_certificate_id: str,
        identification_id: str,
        source_mesh_id: str,
        source_geometry_id: str,
        source_validity_certificate_id: str,
        source_embedding_certificate_id: str,
        planes: tuple[tuple[tuple[int, int], ...], ...],
        child_cell_ids: tuple[int, ...],
        parent_cell_ids: tuple[int, ...],
        sign_vectors: tuple[tuple[int, ...], ...],
        child_volumes: tuple[tuple[int, int], ...],
        partition_cell_ids: tuple[int, ...],
        partition_volumes: tuple[tuple[int, int], ...],
        *,
        direction: PeriodicPartitionDirection,
    ) -> None:
        names = (
            "mesh_id",
            "geometry_id",
            "periodic_topology_id",
            "validity_certificate_id",
            "facet_trace_certificate_id",
            "boundary_subdivision_certificate_id",
            "identification_id",
            "source_mesh_id",
            "source_geometry_id",
            "source_validity_certificate_id",
            "source_embedding_certificate_id",
        )
        identifiers = tuple(
            canonical_identifier(value, f"PeriodicRestrictionPartitionEvidence.{name}")
            for name, value in zip(
                names,
                (
                    mesh_id,
                    geometry_id,
                    periodic_topology_id,
                    validity_certificate_id,
                    facet_trace_certificate_id,
                    boundary_subdivision_certificate_id,
                    identification_id,
                    source_mesh_id,
                    source_geometry_id,
                    source_validity_certificate_id,
                    source_embedding_certificate_id,
                ),
                strict=True,
            )
        )
        direction_ = parse(
            direction,
            PeriodicPartitionDirection,
            "PeriodicRestrictionPartitionEvidence.direction",
        )
        exact_planes = tuple(
            tuple(
                _fraction_pair(value, "PeriodicRestrictionPartitionEvidence.planes")
                for value in plane
            )
            for plane in planes
        )
        if (
            not exact_planes
            or any(len(plane) != 4 or not any(plane[:3]) for plane in exact_planes)
            or len(set(exact_planes)) != len(exact_planes)
        ):
            raise ValueError(
                "PeriodicRestrictionPartitionEvidence.planes must be distinct exact nonzero planes."
            )
        children = tuple(
            nonnegative_integer(
                value, "PeriodicRestrictionPartitionEvidence.child_cell_id"
            )
            for value in child_cell_ids
        )
        parents_by_child = tuple(
            nonnegative_integer(
                value, "PeriodicRestrictionPartitionEvidence.parent_cell_id"
            )
            for value in parent_cell_ids
        )
        partition_parents = tuple(
            nonnegative_integer(
                value, "PeriodicRestrictionPartitionEvidence.partition_cell_id"
            )
            for value in partition_cell_ids
        )
        count = len(children)
        if (
            count == 0
            or len(set(children)) != count
            or len(parents_by_child) != count
            or len(sign_vectors) != count
            or len(child_volumes) != count
        ):
            raise ValueError(
                "PeriodicRestrictionPartitionEvidence must list each partition child exactly once."
            )
        if (
            not partition_parents
            or len(set(partition_parents)) != len(partition_parents)
            or len(partition_volumes) != len(partition_parents)
        ):
            raise ValueError(
                "PeriodicRestrictionPartitionEvidence must list each partition parent exactly once."
            )
        volumes = {
            parent: _positive_fraction(
                ratio, "PeriodicRestrictionPartitionEvidence.partition_volume"
            )
            for parent, ratio in zip(partition_parents, partition_volumes, strict=True)
        }
        totals = dict.fromkeys(partition_parents, Fraction(0))
        seen_signs: dict[int, set[tuple[int, ...]]] = {
            parent: set() for parent in partition_parents
        }
        signs_by_parent: dict[int, list[tuple[int, ...]]] = {
            parent: [] for parent in partition_parents
        }
        signs = tuple(tuple(vector) for vector in sign_vectors)
        exact_child_volumes = tuple(
            _positive_fraction(ratio, "PeriodicRestrictionPartitionEvidence.child_volume")
            for ratio in child_volumes
        )
        for parent, sign, volume in zip(
            parents_by_child, signs, exact_child_volumes, strict=True
        ):
            if (
                parent not in volumes
                or len(sign) != len(exact_planes)
                or any(value not in (-1, 0, 1) for value in sign)
                or sign in seen_signs[parent]
            ):
                raise ValueError(
                    "PeriodicRestrictionPartitionEvidence children must retain unique exact separation vectors."
                )
            seen_signs[parent].add(sign)
            signs_by_parent[parent].append(sign)
            totals[parent] += volume
        for parent, parent_signs in signs_by_parent.items():
            for first_index, first in enumerate(parent_signs):
                for second in parent_signs[first_index + 1 :]:
                    if not any(
                        left != 0 and right == -left
                        for left, right in zip(first, second, strict=True)
                    ):
                        raise ValueError(
                            "PeriodicRestrictionPartitionEvidence child interiors must be separated by a retained exact plane."
                        )
        if any(totals[parent] != volume for parent, volume in volumes.items()):
            raise ValueError(
                "PeriodicRestrictionPartitionEvidence child volumes must exactly sum to each parent volume."
            )

        (
            self.mesh_id,
            self.geometry_id,
            self.periodic_topology_id,
            self.validity_certificate_id,
            self.facet_trace_certificate_id,
            self.boundary_subdivision_certificate_id,
            self.identification_id,
            self.source_mesh_id,
            self.source_geometry_id,
            self.source_validity_certificate_id,
            self.source_embedding_certificate_id,
        ) = identifiers
        self.planes = tuple(
            tuple((value.numerator, value.denominator) for value in plane)
            for plane in exact_planes
        )
        self.child_cell_ids = children
        self.parent_cell_ids = parents_by_child
        self.sign_vectors = signs
        self.child_volumes = tuple(
            (volume.numerator, volume.denominator) for volume in exact_child_volumes
        )
        self.partition_cell_ids = partition_parents
        self.partition_volumes = tuple(
            (volumes[parent].numerator, volumes[parent].denominator)
            for parent in partition_parents
        )
        self.direction = direction_
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "periodic-restriction-partition-evidence",
                **dict(zip(names, identifiers, strict=True)),
                "direction": direction_,
                "planes": self.planes,
                "children": tuple(
                    zip(
                        self.child_cell_ids,
                        self.parent_cell_ids,
                        self.sign_vectors,
                        self.child_volumes,
                        strict=True,
                    )
                ),
                "parents": tuple(
                    zip(
                        self.partition_cell_ids,
                        self.partition_volumes,
                        strict=True,
                    )
                ),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls, value: PeriodicRestrictionPartitionEvidence, /
    ) -> PeriodicRestrictionPartitionEvidence:
        if not isinstance(value, cls):
            raise TypeError("value must be PeriodicRestrictionPartitionEvidence.")
        rebuilt = cls(
            value.mesh_id,
            value.geometry_id,
            value.periodic_topology_id,
            value.validity_certificate_id,
            value.facet_trace_certificate_id,
            value.boundary_subdivision_certificate_id,
            value.identification_id,
            value.source_mesh_id,
            value.source_geometry_id,
            value.source_validity_certificate_id,
            value.source_embedding_certificate_id,
            value.planes,
            value.child_cell_ids,
            value.parent_cell_ids,
            value.sign_vectors,
            value.child_volumes,
            value.partition_cell_ids,
            value.partition_volumes,
            direction=value.direction,
        )
        if (
            rebuilt.mesh_id != value.mesh_id
            or rebuilt.geometry_id != value.geometry_id
            or rebuilt.periodic_topology_id != value.periodic_topology_id
            or rebuilt.validity_certificate_id != value.validity_certificate_id
            or rebuilt.facet_trace_certificate_id != value.facet_trace_certificate_id
            or rebuilt.boundary_subdivision_certificate_id
            != value.boundary_subdivision_certificate_id
            or rebuilt.identification_id != value.identification_id
            or rebuilt.source_mesh_id != value.source_mesh_id
            or rebuilt.source_geometry_id != value.source_geometry_id
            or rebuilt.source_validity_certificate_id
            != value.source_validity_certificate_id
            or rebuilt.source_embedding_certificate_id
            != value.source_embedding_certificate_id
            or rebuilt.planes != value.planes
            or rebuilt.child_cell_ids != value.child_cell_ids
            or rebuilt.parent_cell_ids != value.parent_cell_ids
            or rebuilt.sign_vectors != value.sign_vectors
            or rebuilt.child_volumes != value.child_volumes
            or rebuilt.partition_cell_ids != value.partition_cell_ids
            or rebuilt.partition_volumes != value.partition_volumes
            or rebuilt.direction != value.direction
            or rebuilt.evidence_id != value.evidence_id
        ):
            raise ValueError(
                "Stored PeriodicRestrictionPartitionEvidence does not reproduce its exact partition."
            )
        return rebuilt


@final
class PeriodicMappedTorusDegreeEvidence(StrictModule, NonTrainableState):
    """Bind a mapped local-homeomorphism degree proof to current periodic data."""

    mesh_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    periodic_topology_id: str = eqx.field(static=True)
    validity_certificate_id: str = eqx.field(static=True)
    input_binding_id: str = eqx.field(static=True)
    facet_trace_certificate_id: str = eqx.field(static=True)
    local_injectivity_certificate_id: str = eqx.field(static=True)
    manifold_certificate_id: str = eqx.field(static=True)
    pullback_volume: tuple[int, int] = eqx.field(static=True)
    lattice_volume: tuple[int, int] = eqx.field(static=True)
    covering_degree: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh_id: str,
        geometry_id: str,
        periodic_topology_id: str,
        validity_certificate_id: str,
        input_binding_id: str,
        facet_trace_certificate_id: str,
        local_injectivity_certificate_id: str,
        manifold_certificate_id: str,
        pullback_volume: tuple[int, int],
        lattice_volume: tuple[int, int],
    ) -> None:
        names = (
            "mesh_id",
            "geometry_id",
            "periodic_topology_id",
            "validity_certificate_id",
            "input_binding_id",
            "facet_trace_certificate_id",
            "local_injectivity_certificate_id",
            "manifold_certificate_id",
        )
        identifiers = tuple(
            canonical_identifier(value, f"PeriodicMappedTorusDegreeEvidence.{name}")
            for name, value in zip(
                names,
                (
                    mesh_id,
                    geometry_id,
                    periodic_topology_id,
                    validity_certificate_id,
                    input_binding_id,
                    facet_trace_certificate_id,
                    local_injectivity_certificate_id,
                    manifold_certificate_id,
                ),
                strict=True,
            )
        )
        pullback = _positive_fraction(
            pullback_volume, "PeriodicMappedTorusDegreeEvidence.pullback_volume"
        )
        lattice = _positive_fraction(
            lattice_volume, "PeriodicMappedTorusDegreeEvidence.lattice_volume"
        )
        degree = pullback / lattice
        if degree != 1:
            raise ValueError(
                "PeriodicMappedTorusDegreeEvidence must certify exact degree one."
            )
        (
            self.mesh_id,
            self.geometry_id,
            self.periodic_topology_id,
            self.validity_certificate_id,
            self.input_binding_id,
            self.facet_trace_certificate_id,
            self.local_injectivity_certificate_id,
            self.manifold_certificate_id,
        ) = identifiers
        self.pullback_volume = (pullback.numerator, pullback.denominator)
        self.lattice_volume = (lattice.numerator, lattice.denominator)
        self.covering_degree = degree.numerator
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "periodic-mapped-torus-degree-one",
                **dict(zip(names, identifiers, strict=True)),
                "pullback_volume": self.pullback_volume,
                "lattice_volume": self.lattice_volume,
                "covering_degree": self.covering_degree,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls, value: PeriodicMappedTorusDegreeEvidence, /
    ) -> PeriodicMappedTorusDegreeEvidence:
        if not isinstance(value, cls):
            raise TypeError("value must be PeriodicMappedTorusDegreeEvidence.")
        rebuilt = cls(
            value.mesh_id,
            value.geometry_id,
            value.periodic_topology_id,
            value.validity_certificate_id,
            value.input_binding_id,
            value.facet_trace_certificate_id,
            value.local_injectivity_certificate_id,
            value.manifold_certificate_id,
            value.pullback_volume,
            value.lattice_volume,
        )
        if (
            rebuilt.mesh_id != value.mesh_id
            or rebuilt.geometry_id != value.geometry_id
            or rebuilt.periodic_topology_id != value.periodic_topology_id
            or rebuilt.validity_certificate_id != value.validity_certificate_id
            or rebuilt.input_binding_id != value.input_binding_id
            or rebuilt.facet_trace_certificate_id != value.facet_trace_certificate_id
            or rebuilt.local_injectivity_certificate_id
            != value.local_injectivity_certificate_id
            or rebuilt.manifold_certificate_id != value.manifold_certificate_id
            or rebuilt.pullback_volume != value.pullback_volume
            or rebuilt.lattice_volume != value.lattice_volume
            or rebuilt.covering_degree != value.covering_degree
            or rebuilt.evidence_id != value.evidence_id
        ):
            raise ValueError(
                "Stored PeriodicMappedTorusDegreeEvidence does not reproduce its exact theorem."
            )
        return rebuilt


__all__ = [
    "CoordinateMapScope",
    "EmbeddingCounts",
    "EmbeddingState",
    "MeshCertificateEntityKind",
    "MeshCertificateFinding",
    "MeshEmbeddingWitness",
    "MeshEmbeddingWitnessArchiveRecipe",
    "MeshFindingStatus",
    "PeriodicAffineTorusDegreeEvidence",
    "PeriodicMappedTorusDegreeEvidence",
    "PeriodicPartitionDirection",
    "PeriodicRestrictionPartitionEvidence",
]
