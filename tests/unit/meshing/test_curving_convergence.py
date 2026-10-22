"""Positive curving convergence, periodic coupling and relaxation acceptance."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import meshcore_available
from phydrax.geometry._approximate_source_coverage import (
    ApproximateSourceCoverageCertificate,
)
from phydrax.geometry._domain_planar_boundary import CadSourceDomain
from phydrax.meshing._curving import HighOrderCurvingResult
from tests._support.cad_curving import (
    circular_p2_fidelity_bound,
    CURVING_ASSOCIATION_POLICY,
    CURVING_COORDINATE_CONTRACT,
    CURVING_STATUS,
    cylinder_mesh,
    icosphere,
    lateral_error,
    plate_mesh,
    plate_projection,
    sphere_error,
    sphere_fidelity_bound,
    sphere_projection,
)


# Unchanged empirical allowance from the pre-refactor straight-edge, seam, plane-distance and zero-mean reference assertions.
CURVING_COORDINATE_ROUNDOFF_ATOL = 1e-12
# Unchanged empirical allowance from the pre-refactor manufactured sphere harmonic convergence fixture.
SPHERE_PDE_ERROR_BOUND = 0.06


def test_native_embedded_sphere_laplace_beltrami_converges_with_zero_mean() -> None:
    from examples.native_surface_meshing import generate, solve

    if not meshcore_available():
        pytest.skip("Native sphere meshing requires compiled meshcore.")
    coarse = generate(0.8)
    fine = generate(0.4)
    coarse_error, coarse_mean = solve(coarse)
    fine_error, fine_mean = solve(fine)
    # Source is independently the unit sphere; this is a solved harmonic,
    # measured after radial lift, not a surface-area consistency check.
    assert fine_error < 0.5 * coarse_error
    assert fine_error < SPHERE_PDE_ERROR_BOUND
    assert abs(coarse_mean) < CURVING_COORDINATE_ROUNDOFF_ATOL
    assert abs(fine_mean) < CURVING_COORDINATE_ROUNDOFF_ATOL


def test_p2_sphere_curving_preserves_convergence_order() -> None:
    projection = sphere_projection()
    errors = []
    for level in (1, 2):
        mesh = icosphere(level)
        policy = phx.meshing.HighOrderCurvingPolicy(
            fidelity_tolerance=sphere_fidelity_bound(mesh),
            degree=2,
            relaxation_rounds=1,
        )
        association = phx.meshing.associate_mesh_vertices(
            mesh, projection, policy=CURVING_ASSOCIATION_POLICY
        )
        result = phx.meshing.curve_cell_mesh(mesh, association, projection, policy=policy)
        assert result.status is CURVING_STATUS.CURVED
        assert result.evidence.certificate.all_certified
        assert result.evidence.maximum_residual <= policy.residual_tolerance
        assert result.evidence.fidelity is not None
        assert result.evidence.fidelity.status == "certified"
        assert (
            max(
                result.evidence.fidelity.mesh_to_source_upper,
                result.evidence.fidelity.source_to_mesh_upper,
            )
            <= policy.fidelity_tolerance
        )
        errors.append(
            (sphere_error(result.straight, mesh), sphere_error(result.geometry, mesh))
        )
    (straight_coarse, curved_coarse), (straight_fine, curved_fine) = errors
    assert curved_coarse < 0.05 * straight_coarse
    # These are the pre-branch second-order facet and at-least-third-order P2 bounds.
    assert 3.0 < straight_coarse / straight_fine < 5.0
    assert curved_coarse / curved_fine > 7.0


def test_p2_periodic_cylinder_curving_preserves_coupled_nodes() -> None:

    projection = phx.geometry.prepare_brep_projection(
        phx.geometry.brep_cylinder(
            1.0, 2.0, coordinate_contract=CURVING_COORDINATE_CONTRACT
        )
    )
    mesh = cylinder_mesh(2, 2)
    association = phx.meshing.associate_mesh_vertices(
        mesh, projection, policy=CURVING_ASSOCIATION_POLICY
    )
    result = phx.meshing.certify_cell_mesh(
        mesh, CURVING_COORDINATE_CONTRACT, associations=(association,)
    )
    part = phx.meshing.MeshPart("cylinder", result)
    coordinates = np.asarray(mesh.coordinates)
    identifiers = np.asarray(mesh.vertex_global_ids)
    bottom = np.isclose(coordinates[:, 2], 0.0)
    top = np.isclose(coordinates[:, 2], 2.0)
    top_ids = np.sort(identifiers[top])
    below = {
        tuple(np.round(coordinates[row, :2], 12)): identifiers[row]
        for row in np.flatnonzero(bottom)
    }
    source_ids = np.asarray(
        [
            below[tuple(np.round(coordinates[identifiers == value][0, :2], 12))]
            for value in top_ids
        ],
        dtype=np.int64,
    )
    translation = np.asarray([0.0, 0.0, 2.0], dtype=np.float64)
    coupling = phx.meshing.PeriodicCoupling(
        part,
        part,
        part.scope(0, np.sort(identifiers[bottom])),
        part.scope(0, top_ids),
        np.eye(3, dtype=np.float64),
        translation,
        source_ids=source_ids,
    )
    curved = phx.meshing.curve_cell_mesh(
        mesh,
        association,
        projection,
        policy=phx.meshing.HighOrderCurvingPolicy(
            fidelity_tolerance=circular_p2_fidelity_bound(np.pi / 12),
            degree=2,
            relaxation_rounds=1,
        ),
        periodic=(coupling,),
        domain=CadSourceDomain(projection.query.model, region_ids=("cylinder",)),
        cell_regions=np.zeros(mesh.blocks[0].cell_count, dtype=np.int64),
    )
    nodes = np.asarray(curved.geometry.coordinates)
    straight = np.asarray(curved.straight.coordinates)
    top_nodes = np.flatnonzero(np.isclose(straight[:, 2], 2.0))
    bottom_nodes = np.flatnonzero(np.isclose(straight[:, 2], 0.0))
    source_rows, target_rows = coupling.match_points(
        straight[bottom_nodes], straight[top_nodes]
    )
    assert curved.status is CURVING_STATUS.CURVED
    assert curved.evidence.certificate.all_certified
    assert curved.evidence.fidelity is not None
    assert curved.evidence.fidelity.status == "certified"
    assert curved.evidence.coverage is not None
    assert curved.evidence.coverage.status == "certified"
    assert isinstance(curved.evidence.coverage, ApproximateSourceCoverageCertificate)
    assert curved.evidence.coverage.distance_upper <= curved.evidence.fidelity.tolerance
    assert (
        curved.evidence.coverage.boundary_degree_id
        == curved.evidence.fidelity.boundary_degree_id
    )
    assert lateral_error(curved.geometry, mesh) < 0.05 * lateral_error(
        curved.straight, mesh
    )
    np.testing.assert_array_equal(
        nodes[top_nodes[target_rows]], nodes[bottom_nodes[source_rows]] + translation
    )
    assert np.max(np.abs(nodes[top_nodes] - straight[top_nodes])) > 0.01


def test_p2_verification_distinguishes_curved_and_straight_geometry() -> None:
    projection = sphere_projection()
    mesh = icosphere(1)
    association = phx.meshing.associate_mesh_vertices(
        mesh, projection, policy=CURVING_ASSOCIATION_POLICY
    )
    policy = phx.meshing.HighOrderCurvingPolicy(
        fidelity_tolerance=sphere_fidelity_bound(mesh), degree=2, relaxation_rounds=1
    )
    curved = phx.meshing.curve_cell_mesh(mesh, association, projection, policy=policy)
    accepted = phx.meshing.verify_curved_geometry(
        curved.geometry, mesh, association, projection, policy=policy
    )
    straight = phx.meshing.verify_curved_geometry(
        curved.straight, mesh, association, projection, policy=policy
    )
    assert accepted.accepted
    assert accepted.certificate.all_certified
    assert not straight.accepted
    assert straight.maximum_residual > 0.01


def test_nonconverged_relaxation_replaces_a_valid_projection_only_when_permitted() -> (
    None
):
    projection = sphere_projection()
    mesh = icosphere(1)
    association = phx.meshing.associate_mesh_vertices(
        mesh, projection, policy=CURVING_ASSOCIATION_POLICY
    )

    def curve(rounds: int, accept: bool) -> HighOrderCurvingResult:
        policy = phx.meshing.HighOrderCurvingPolicy(
            fidelity_tolerance=sphere_fidelity_bound(mesh),
            degree=2,
            relaxation_rounds=rounds,
            termination=phx.optim.OptimizationTermination(maximum_steps=1),
            accept_valid_nonconverged_relaxation=accept,
        )
        return phx.meshing.curve_cell_mesh(mesh, association, projection, policy=policy)

    projected = curve(0, False)
    retained = curve(1, False)
    replaced = curve(1, True)
    budget = (phx.optim.OptimizationStatus.MAXIMUM_STEPS_REACHED,)
    assert projected.status is CURVING_STATUS.CURVED and projected.accepted_round == 0
    assert retained.status is CURVING_STATUS.CURVED and retained.accepted_round == 0
    assert retained.candidate is not None
    assert retained.candidate.accepted and retained.relaxation_statuses == budget
    assert retained.candidate_geometry is not None
    retained.candidate.certificate.require_bound(retained.candidate_geometry, mesh=mesh)
    assert not np.array_equal(
        retained.candidate_geometry.coordinates, retained.geometry.coordinates
    )
    np.testing.assert_array_equal(
        retained.geometry.coordinates, projected.geometry.coordinates
    )
    assert replaced.status is CURVING_STATUS.CURVED and replaced.accepted_round == 1
    assert replaced.evidence is not None
    assert replaced.relaxation_statuses == budget and replaced.evidence.accepted
    assert not np.array_equal(
        replaced.geometry.coordinates, projected.geometry.coordinates
    )


def test_inverting_curving_rolls_back_and_converged_relaxation_repairs_it() -> None:

    projection = plate_projection()
    mesh = plate_mesh((0.8, 0.8))
    association = phx.meshing.associate_mesh_vertices(
        mesh, projection, policy=CURVING_ASSOCIATION_POLICY
    )

    def curve(rounds: int, steps: int) -> HighOrderCurvingResult:
        policy = phx.meshing.HighOrderCurvingPolicy(
            fidelity_tolerance=circular_p2_fidelity_bound(np.pi / 4),
            degree=2,
            relaxation_rounds=rounds,
            termination=phx.optim.OptimizationTermination(maximum_steps=steps),
        )
        return phx.meshing.curve_cell_mesh(
            mesh,
            association,
            projection,
            policy=policy,
            domain=CadSourceDomain(
                projection.query.model,
                region_ids=("plate",),
                embedding=projection.embedding,
            ),
            cell_regions=np.zeros(mesh.blocks[0].cell_count, dtype=np.int64),
        )

    projected = curve(0, 64)
    capped = curve(2, 64)
    relaxed = curve(2, 512)
    assert projected.status is CURVING_STATUS.ROLLED_BACK_INVALID
    assert projected.candidate is not None
    assert projected.candidate.certificate.invalid_count > 0
    np.testing.assert_array_equal(
        projected.geometry.coordinates, projected.straight.coordinates
    )
    assert capped.status is CURVING_STATUS.ROLLED_BACK_NONCONVERGED
    assert capped.candidate is not None
    assert capped.accepted_round is None and capped.candidate.accepted
    assert phx.optim.OptimizationStatus.SUCCESS not in capped.relaxation_statuses
    np.testing.assert_array_equal(
        capped.geometry.coordinates, capped.straight.coordinates
    )
    assert relaxed.status is CURVING_STATUS.CURVED
    assert relaxed.accepted_round is not None
    assert relaxed.evidence is not None
    assert (
        relaxed.relaxation_statuses[relaxed.accepted_round - 1]
        is phx.optim.OptimizationStatus.SUCCESS
    )
    assert relaxed.evidence.certificate.all_certified
    assert relaxed.evidence.minimum_scaled_jacobian > 0.0
    assert relaxed.evidence.fidelity is not None
    assert relaxed.evidence.fidelity.status == "certified"
    assert relaxed.evidence.coverage is not None
    assert relaxed.evidence.coverage.status == "certified"
    assert isinstance(relaxed.evidence.coverage, ApproximateSourceCoverageCertificate)
    assert relaxed.evidence.coverage.distance_upper <= relaxed.evidence.fidelity.tolerance


def test_cad_region_coverage_requires_every_cell_assignment() -> None:

    projection = plate_projection()
    mesh = plate_mesh((1.2, 1.2))
    association = phx.meshing.associate_mesh_vertices(
        mesh, projection, policy=CURVING_ASSOCIATION_POLICY
    )
    regions = np.zeros(mesh.blocks[0].cell_count, dtype=np.int64)
    regions[0] = -1
    result = phx.meshing.curve_cell_mesh(
        mesh,
        association,
        projection,
        policy=phx.meshing.HighOrderCurvingPolicy(
            fidelity_tolerance=circular_p2_fidelity_bound(np.pi / 4),
            degree=2,
            relaxation_rounds=0,
        ),
        domain=CadSourceDomain(
            projection.query.model, region_ids=("plate",), embedding=projection.embedding
        ),
        cell_regions=regions,
    )
    assert result.status is CURVING_STATUS.ROLLED_BACK_CERTIFICATION
    assert result.candidate is not None
    assert result.candidate.certificate.all_certified
    assert result.candidate.fidelity is not None
    assert result.candidate.fidelity.status == "certified"
    assert isinstance(result.candidate.coverage, ApproximateSourceCoverageCertificate)
    assert result.candidate.coverage.status == "unresolved"
    assert any(
        finding.check == "approximate_source_region_assignment"
        for finding in result.candidate.coverage.findings
    )
    assert not result.candidate.accepted
    np.testing.assert_array_equal(
        result.geometry.coordinates, result.straight.coordinates
    )


def test_inadmissible_intermediate_rounds_defer_their_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.meshing._curving as curving_owner

    projection = sphere_projection()
    mesh = icosphere(1)
    association = phx.meshing.associate_mesh_vertices(
        mesh, projection, policy=CURVING_ASSOCIATION_POLICY
    )
    evaluated: list[np.ndarray] = []
    original = curving_owner._evidence

    def counted(
        mesh_: phx.meshing.CellMesh,
        geometry: phx.discretization.CellGeometrySpec,
        *args: Any,
        **kwargs: Any,
    ) -> curving_owner.CurvedGeometryEvidence:
        evaluated.append(np.asarray(geometry.coordinates).copy())
        return original(mesh_, geometry, *args, **kwargs)

    monkeypatch.setattr(curving_owner, "_evidence", counted)

    def curve(accept: bool) -> HighOrderCurvingResult:
        evaluated.clear()
        policy = phx.meshing.HighOrderCurvingPolicy(
            fidelity_tolerance=sphere_fidelity_bound(mesh),
            degree=2,
            relaxation_rounds=3,
            termination=phx.optim.OptimizationTermination(maximum_steps=1),
            accept_valid_nonconverged_relaxation=accept,
        )
        return phx.meshing.curve_cell_mesh(mesh, association, projection, policy=policy)

    replaced = curve(True)
    every_round = [value.copy() for value in evaluated]
    retained = curve(False)
    # Rounds 1 and 2 cannot be accepted; only the projection and round 3 are proved.
    assert len(every_round) == 4 and len(evaluated) == 2
    np.testing.assert_array_equal(evaluated[0], every_round[0])
    np.testing.assert_array_equal(evaluated[1], every_round[3])
    assert retained.status is CURVING_STATUS.CURVED and retained.accepted_round == 0
    assert replaced.accepted_round == 3
    assert retained.candidate is not None and retained.candidate_geometry is not None
    np.testing.assert_array_equal(retained.candidate_geometry.coordinates, every_round[3])
    retained.candidate.certificate.require_bound(retained.candidate_geometry, mesh=mesh)


def test_curving_node_jacobi_is_the_constructed_smoothing_normal_gram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jax
    import jax.numpy as jnp

    import phydrax.meshing._curving as owner

    mesh = phx.discretization.CellMesh.from_tetrahedra(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        np.asarray(((0, 1, 2, 3),), dtype=np.int32),
    )
    policy = phx.meshing.HighOrderCurvingPolicy(fidelity_tolerance=0.01)
    geometry = owner.straight_geometry(
        mesh, 2, maximum_nodes=policy.maximum_geometry_nodes,
        maximum_entries=policy.maximum_tabulation_entries,
        maximum_bernstein_nodes=policy.validity.maximum_bernstein_nodes,
    )
    coordinates = np.asarray(geometry.coordinates)
    blocks = owner._distortion_blocks(mesh, geometry, 2)
    rows = np.asarray((4,), dtype=np.int64)
    normal = np.asarray((((0.5, 0.5, 0.0), (0.5, 0.5, 0.0),
                          (0.0, 0.0, 0.0)),))
    energy = owner._curving_round_energy(blocks, coordinates, rows, normal, 1.0, policy)
    block = blocks[0]
    _, determinant, _ = block.ratios(jnp.asarray(coordinates))
    regularized = 0.5 * (
        determinant + jnp.sqrt(determinant**2 + 4 * energy.regularization**2)
    )
    scale = 1.0 / (3.0 * regularized ** (2.0 / 3.0))

    def residual(value: Any) -> Any:
        _, _, jacobian = block.ratios(value)
        normalized = jacobian @ block.target_inverses
        smoothing = normalized * jnp.sqrt(
            2.0 * energy.weights[0] * scale
            / (block.routes.shape[0] * block.gradients.shape[0])
        )[..., None, None]
        penalty = jnp.einsum("bij,bj->bi", energy.normal_projectors, value[rows])
        return jnp.concatenate((
            smoothing.ravel(),
            (jnp.sqrt(2.0 * energy.weights[1]) * penalty).ravel(),
            (jnp.sqrt(2.0 * energy.weights[2]) * value).ravel(),
        ))

    jacobian = np.asarray(jax.jacfwd(residual)(jnp.asarray(coordinates)))
    expected = np.einsum("mni,mnj->nij", jacobian, jacobian)
    actual = np.asarray(owner._curving_node_gram(energy, jnp.asarray(coordinates)))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    fixed = np.zeros((coordinates.shape[0],), dtype=np.bool_)
    fixed[:4] = True
    count = coordinates.shape[0]
    admitted_entries = (
        block.routes.shape[0] * block.gradients.shape[0]
        * (block.routes.shape[1] * 3 + 27 + 2 * block.routes.shape[1] + 8)
        + 8 * count * 9
        + block.gradients.size + block.target_inverses.size
    )
    tight = phx.meshing.HighOrderCurvingPolicy(
        fidelity_tolerance=0.01, maximum_tabulation_entries=admitted_entries
    )
    method = owner._curving_volume_method(energy, coordinates, fixed, tight)
    assert method.preconditioner is not None
    assert method.preconditioner.properties.certifies("positive_definite")

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Gram allocation must follow workspace admission.")

    monkeypatch.setattr(owner, "_curving_node_gram", forbidden)
    refused = phx.meshing.HighOrderCurvingPolicy(
        fidelity_tolerance=0.01, maximum_tabulation_entries=admitted_entries - 1
    )
    with pytest.raises(ValueError, match="node-Gram preconditioner workspace"):
        owner._curving_volume_method(energy, coordinates, fixed, refused)
