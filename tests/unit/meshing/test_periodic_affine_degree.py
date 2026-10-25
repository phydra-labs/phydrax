# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Independent topology, cocycle and multiplicity guards for the affine torus theorem."""

from copy import copy
from fractions import Fraction

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry._embedding_state import (
    coordinate_certificate_request,
    EmbeddingState,
    MeshCertificateLimits,
    PeriodicAffineTorusDegreeEvidence,
)
from phydrax.geometry._periodic_affine_degree import (
    _one_cycle,
    certify_affine_torus_degree,
)


def _torus(degree: int = 1) -> phx.discretization.CellMesh:
    count = 4
    indices = np.stack(np.meshgrid(np.arange(count), np.arange(count), indexing="ij"), axis=-1).reshape((-1, 2))
    corners = np.asarray([
        triangle
        for i in range(count)
        for j in range(count)
        for triangle in (
            ((i, j), (i + 1, j), (i + 1, j + 1)),
            ((i, j), (i + 1, j + 1), (i, j + 1)),
        )
    ], dtype=np.int64)
    vectors = np.asarray(((1.0, 0.0), (0.25, 1.0)))
    source_vectors = vectors.copy()
    source_vectors[0] *= degree
    shifts = corners // count
    shifts[..., 0] *= degree
    return phx.meshing.publish_periodic_simplices(
        indices / count @ source_vectors,
        (corners[..., 0] % count) * count + corners[..., 1] % count,
        shifts,
        phx.discretization.PeriodicCell(vectors),
    )


def _prove(
    mesh: phx.discretization.CellMesh, geometry: phx.discretization.CellGeometrySpec | None = None
) -> tuple[PeriodicAffineTorusDegreeEvidence | bool, EmbeddingState, int]:
    geometry = phx.discretization.CellGeometrySpec.affine(mesh) if geometry is None else geometry
    state = EmbeddingState([], [])
    limits = MeshCertificateLimits(maximum_work_units=40_960_000, maximum_scratch_bytes=81_920_000)
    with coordinate_certificate_request(limits) as budget:
        proof = certify_affine_torus_degree(state, mesh, geometry, np.asarray(mesh.entity_set(2).entity_ids), budget=budget)
        work = budget.work_units
    return proof, state, work


def test_affine_torus_degree_proves_current_closed_source_without_pair_contacts() -> None:
    mesh = _torus()
    proof, state, work = _prove(mesh)
    assert isinstance(proof, PeriodicAffineTorusDegreeEvidence)
    assert proof.covering_degree == 1
    assert proof.pullback_area == proof.lattice_area == (1, 1)
    assert state.clean and state.candidate_pairs == 0
    assert "periodic_affine_torus_degree_one" in state.checks and work > 0
    renewed, _, _ = _prove(mesh)
    assert isinstance(renewed, PeriodicAffineTorusDegreeEvidence)
    assert renewed.evidence_id == proof.evidence_id


def test_positive_double_cover_is_not_degree_one() -> None:
    mesh = _torus(2)
    points = np.asarray(mesh.coordinates)
    rows = np.asarray(mesh.blocks[0].vertices)
    triangles = points[rows]
    first, second = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    determinants = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    assert np.all(determinants > 0) and np.sum(determinants) / 2 == 2
    proof, state, _ = _prove(mesh)
    assert proof is True and not state.clean
    assert [finding.check for finding in state.findings] == ["periodic_affine_torus_degree_one"]


def test_vertex_link_requires_one_cycle_not_just_two_sided_edges() -> None:
    a, b, c, d = ((index, 0, 0) for index in range(4))
    assert _one_cycle({a: b, b: c, c: d, d: a})
    assert not _one_cycle({a: b, b: a, c: d, d: c})
    assert not _one_cycle({a: b, b: c, c: b})


def test_positive_branched_star_retains_its_area_multiplicity() -> None:
    # The cyclic domain has eight rays; its PL z^2 image traverses the diamond twice.
    rays = ((1, 0), (0, 1), (-1, 0), (0, -1)) * 2
    determinants = [Fraction(a[0] * b[1] - a[1] * b[0]) for a, b in zip(rays, rays[1:] + rays[:1], strict=True)]
    assert all(value > 0 for value in determinants)
    diamond_area = Fraction(2)
    assert sum(determinants) / (2 * diamond_area) == 2
    # A vertex-link cycle alone does not remove this positive local branch.
    germs = tuple((index, 0, 0) for index in range(8))
    assert _one_cycle(dict(zip(germs, germs[1:] + germs[:1], strict=True)))
    proof, state, _ = _prove(_torus(2))
    assert proof is True and any(finding.check == "periodic_affine_torus_degree_one" for finding in state.findings)


def test_declared_corner_cocycle_is_reconstructed_not_trusted() -> None:
    mesh = _torus()
    topology = mesh.periodic_topology
    assert topology is not None
    changed = topology.corner_shifts.at[0, 0].add(1)
    broken = eqx.tree_at(lambda item: item.periodic_topology.corner_shifts, mesh, changed)
    proof, state, _ = _prove(broken)
    assert proof is True and not state.clean
    assert state.findings[0].check == "periodic_affine_torus_manifold"


def test_changed_current_coordinate_source_cannot_borrow_carrier_geometry() -> None:
    mesh = _torus()
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    changed = eqx.tree_at(lambda item: item.coordinates, geometry, geometry.coordinates.at[0, 0].add(0.125))
    proof, state, _ = _prove(mesh, changed)
    assert proof is True and not state.clean
    assert state.findings[0].check in ("periodic_affine_torus_trace", "periodic_affine_torus_orientation")


def test_partial_translation_identification_is_outside_degree_admission() -> None:
    mesh = _torus()
    topology = mesh.periodic_topology
    assert topology is not None
    assert isinstance(topology.cell, phx.discretization.PeriodicCell)
    partial = phx.discretization.PeriodicCell(topology.cell.vectors, periodic_axes=(True, False))
    changed = eqx.tree_at(lambda item: item.periodic_topology.cell, mesh, partial)
    proof, state, _ = _prove(changed)
    assert proof is False and state.clean and not state.checks


def test_degree_evidence_recomputes_integral_and_rejects_changed_stored_degree() -> None:
    proof, _, _ = _prove(_torus())
    assert isinstance(proof, PeriodicAffineTorusDegreeEvidence)
    restored = PeriodicAffineTorusDegreeEvidence.from_stored_fields(proof)
    assert restored.evidence_id == proof.evidence_id
    changed = copy(proof)
    object.__setattr__(changed, "covering_degree", 2)
    with pytest.raises(ValueError, match="current exact degree"):
        PeriodicAffineTorusDegreeEvidence.from_stored_fields(changed)
    with pytest.raises(ValueError, match="integer degree one"):
        PeriodicAffineTorusDegreeEvidence(
            proof.mesh_id, proof.geometry_id, proof.periodic_topology_id,
            proof.input_binding_id, proof.affinity_certificate_id,
            proof.manifold_certificate_id, proof.trace_certificate_id,
            proof.orientation_certificate_id, (2, 1), (1, 1),
        )


def test_negative_orientation_is_not_an_admitted_positive_cover() -> None:
    proof, state, _ = _prove(_torus(-1))
    assert proof is True and not state.clean
    assert state.findings[0].check == "periodic_affine_torus_orientation"
