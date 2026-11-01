from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_geometry_transition_types import (
    CellGeometryTransition,
    CellGeometryTransitionEvidence,
)
from phydrax.discretization._cell_geometry_validity import certify_cell_geometry_validity
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._nested_reference_types import NestedReferenceWitnesses
from phydrax.geometry._global_embedding import (
    certify_global_embedding,
    GlobalEmbeddingCertificate,
)
from phydrax.geometry._nested_restriction_embedding import (
    NESTED_RESTRICTION_ROUTE,
    NestedRestrictionPremise,
)


_HALF = Fraction(1, 2)
_TRIANGLE = (
    (Fraction(0), Fraction(0)),
    (Fraction(1), Fraction(0)),
    (Fraction(0), Fraction(1)),
)


def _midpoint(
    first: tuple[Fraction, ...], second: tuple[Fraction, ...]
) -> tuple[Fraction, ...]:
    return tuple((left + right) / 2 for left, right in zip(first, second, strict=True))


def _square_surface() -> tuple[CellMesh, CellGeometrySpec]:
    coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    block = CellBlock(
        "surface",
        "triangle",
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
        global_ids=np.asarray((10, 11), dtype=np.int64),
    )
    mesh = CellMesh(coordinates, (block,), vertex_global_ids=np.arange(4, dtype=np.int64))
    return mesh, CellGeometrySpec.affine(mesh)


def _refined_square(
    source: CellMesh, /, *, overlapping: bool = False
) -> tuple[CellMesh, CellGeometrySpec, NestedReferenceWitnesses]:
    """Split each parent triangle at its edge midpoints into four children.

    ``overlapping`` adds a fifth child of the first parent that covers part of
    its corner child.
    """
    points = [
        tuple(float(value) for value in row) for row in np.asarray(source.coordinates)
    ]
    midpoint_rows: dict[tuple[int, int], int] = {}

    def midpoint_row(first: int, second: int, /) -> int:
        key = (min(first, second), max(first, second))
        row = midpoint_rows.get(key)
        if row is None:
            row = len(points)
            points.append(
                tuple(
                    (points[first][axis] + points[second][axis]) / 2 for axis in range(3)
                )
            )
            midpoint_rows[key] = row
        return row

    rows: list[tuple[int, int, int]] = []
    fine_ids: list[int] = []
    coarse_ids: list[int] = []
    references: list[tuple[tuple[Fraction, ...], ...]] = []
    parents = np.asarray(source.blocks[0].vertices, dtype=np.int64).tolist()
    parent_ids = np.asarray(source.blocks[0].global_ids, dtype=np.int64).tolist()
    for parent, parent_id in zip(parents, parent_ids, strict=True):
        a, b, c = parent
        ab, bc, ca = midpoint_row(a, b), midpoint_row(b, c), midpoint_row(c, a)
        ra, rb, rc = _TRIANGLE
        rab, rbc, rca = _midpoint(ra, rb), _midpoint(rb, rc), _midpoint(rc, ra)
        children = [
            ((a, ab, ca), (ra, rab, rca)),
            ((ab, b, bc), (rab, rb, rbc)),
            ((ca, bc, c), (rca, rbc, rc)),
            ((ab, bc, ca), (rab, rbc, rca)),
        ]
        if overlapping and parent_id == parent_ids[0]:
            # A child on two new vertices a quarter along the parent's sides
            # from ``a``: it overlaps the corner child without sharing an edge.
            quarter = Fraction(1, 4)
            near_b, near_c = len(points), len(points) + 1
            for corner in (b, c):
                points.append(
                    tuple(
                        points[a][axis] + (points[corner][axis] - points[a][axis]) / 4
                        for axis in range(3)
                    )
                )
            children.append(
                (
                    (a, near_b, near_c),
                    (ra, (quarter, Fraction(0)), (Fraction(0), quarter)),
                )
            )
        for vertices, reference in children:
            rows.append(vertices)
            fine_ids.append(100 + len(rows))
            coarse_ids.append(parent_id)
            references.append(reference)
    block = CellBlock(
        "surface",
        "triangle",
        np.asarray(rows, dtype=np.int32),
        global_ids=np.asarray(fine_ids, dtype=np.int64),
    )
    target = CellMesh(
        np.asarray(points, dtype=np.float64),
        (block,),
        vertex_global_ids=np.arange(len(points), dtype=np.int64),
    )
    witness = NestedReferenceWitnesses(
        np.asarray(fine_ids, dtype=np.int64),
        np.asarray(coarse_ids, dtype=np.int64),
        np.asarray(
            [
                [[float(value) for value in point] for point in cell]
                for cell in references
            ],
            dtype=np.float64,
        ),
        fine_reference_exact_vertices=tuple(references),
    )
    return target, CellGeometrySpec.affine(target), witness


def _unit_tetrahedron() -> tuple[CellMesh, CellGeometrySpec]:
    coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    block = CellBlock(
        "volume",
        "tetrahedron",
        np.asarray(((0, 1, 2, 3),), dtype=np.int32),
        global_ids=np.asarray((7,), dtype=np.int64),
    )
    mesh = CellMesh(coordinates, (block,), vertex_global_ids=np.arange(4, dtype=np.int64))
    return mesh, CellGeometrySpec.affine(mesh)


def _centroid_split(
    source: CellMesh,
) -> tuple[CellMesh, CellGeometrySpec, NestedReferenceWitnesses]:
    corners = (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    centroid = (Fraction(1, 4),) * 3
    points = np.vstack((np.asarray(source.coordinates), ((0.25, 0.25, 0.25),)))
    rows = []
    references = []
    for omitted in range(4):
        # Replacing corner ``omitted`` by the centroid keeps the orientation.
        vertices = [4 if index == omitted else index for index in range(4)]
        reference = tuple(
            centroid if index == omitted else corners[index] for index in range(4)
        )
        rows.append(tuple(vertices))
        references.append(reference)
    fine_ids = np.arange(20, 24, dtype=np.int64)
    block = CellBlock(
        "volume", "tetrahedron", np.asarray(rows, dtype=np.int32), global_ids=fine_ids
    )
    target = CellMesh(points, (block,), vertex_global_ids=np.arange(5, dtype=np.int64))
    witness = NestedReferenceWitnesses(
        fine_ids,
        np.full((4,), 7, dtype=np.int64),
        np.asarray(
            [
                [[float(value) for value in point] for point in cell]
                for cell in references
            ],
            dtype=np.float64,
        ),
        fine_reference_exact_vertices=tuple(references),
    )
    return target, CellGeometrySpec.affine(target), witness


def _premise(
    source: CellMesh,
    source_geometry: CellGeometrySpec,
    target: CellMesh,
    target_geometry: CellGeometrySpec,
    witness: NestedReferenceWitnesses,
) -> tuple[NestedRestrictionPremise, GlobalEmbeddingCertificate]:
    source_validity = certify_cell_geometry_validity(source_geometry, mesh=source)
    source_embedding = certify_global_embedding(source, source_geometry, source_validity)
    assert source_embedding.status == "certified"
    transition = CellGeometryTransition(
        source,
        source_geometry,
        target,
        target_geometry,
        evidence=CellGeometryTransitionEvidence(
            "nested_restriction", refinement_map_count=len(witness.fine_cell_ids)
        ),
        refinement=witness,
    )
    return (
        NestedRestrictionPremise(source, source_geometry, source_embedding, transition),
        source_embedding,
    )


@pytest.mark.parametrize("fixture", ["square", "tetrahedron"])
def test_nested_restriction_embedding_matches_full_proof(fixture: str) -> None:
    if fixture == "square":
        source, source_geometry = _square_surface()
        target, target_geometry, witness = _refined_square(source)
    else:
        source, source_geometry = _unit_tetrahedron()
        target, target_geometry, witness = _centroid_split(source)
    premise, source_embedding = _premise(
        source, source_geometry, target, target_geometry, witness
    )
    validity = certify_cell_geometry_validity(target_geometry, mesh=target)
    full = certify_global_embedding(target, target_geometry, validity)
    nested = certify_global_embedding(
        target, target_geometry, validity, nested_restriction=premise
    )

    evidence = nested.nested_restriction
    assert evidence is not None
    assert evidence.route == NESTED_RESTRICTION_ROUTE
    assert evidence.source_embedding_certificate_id == source_embedding.certificate_id
    assert evidence.transition_id == premise.transition.transition_id
    assert evidence.child_count == len(witness.fine_cell_ids)
    assert nested.nested_restriction_premise is premise
    assert full.nested_restriction is None
    # Identical verdict, inventory and binding as the full proof.
    assert full.status == nested.status == "certified"
    assert full.findings == nested.findings == ()
    assert full.counts == nested.counts
    assert full.binding.binding_id == nested.binding.binding_id
    assert "nested_restriction_parent_tiling" in nested.evaluated_checks
    assert any(
        item.theorem == "nested_restriction_embedding"
        and source_embedding.certificate_id in item.premise_ids
        for item in nested.witnesses
    )
    # Cold restoration re-proves the same nested theorem from its premise.
    assert (
        GlobalEmbeddingCertificate.from_stored_fields(nested).certificate_id
        == nested.certificate_id
    )
    with pytest.raises(ValueError, match="nested-restriction"):
        nested.require_inputs(target, target_geometry, validity)


def test_overlapping_child_falls_back_to_the_full_violation() -> None:
    source, source_geometry = _square_surface()
    target, target_geometry, witness = _refined_square(source, overlapping=True)
    premise, _ = _premise(source, source_geometry, target, target_geometry, witness)
    validity = certify_cell_geometry_validity(target_geometry, mesh=target)
    full = certify_global_embedding(target, target_geometry, validity)
    nested = certify_global_embedding(
        target, target_geometry, validity, nested_restriction=premise
    )
    # The overlap breaks the parent volume identity, so the theorem does not
    # apply and the target is proved from scratch, with the same violation.
    assert nested.nested_restriction is None
    assert nested.nested_restriction_premise is None
    assert full.status == nested.status == "violated"
    assert nested.certificate_id == full.certificate_id
