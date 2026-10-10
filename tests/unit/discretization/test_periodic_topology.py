from __future__ import annotations

from collections.abc import Callable
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest
from jax import Array, Device
from jax.typing import ArrayLike

import phydrax as phx
from phydrax.discretization import (
    CellBlock,
    CellMesh,
    FiniteElementSpec,
    periodic_orbit_measures,
    PeriodicCell,
    PeriodicMeshTopology,
)
from phydrax.discretization._periodic_topology import PeriodicIsometryGroup
from phydrax.meshing._periodic import (
    _require_periodic_topology,
    _simplex_entity_corners,
    certify_periodic_embedding,
    refine_periodic_mesh,
)
from phydrax.meshing.providers._native_periodic import (
    NativePeriodicSource,
    PeriodicAssociationTransfer,
)


def _periodic(
    points: np.ndarray,
    block: CellBlock,
    representatives: np.ndarray,
    shifts: np.ndarray,
    cell: PeriodicCell | PeriodicIsometryGroup,
) -> CellMesh:
    lifted = CellMesh(points, (block,))
    topology = PeriodicMeshTopology(lifted, cell, representatives, shifts)
    return CellMesh(points, (block,), periodic_topology=topology)


def _square_lift() -> tuple[np.ndarray, np.ndarray]:
    # Corners of the unit square are lattice images of one quotient vertex.
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    return points, points.astype(np.int64)


def _two_triangle_torus() -> CellMesh:
    points, shifts = _square_lift()
    # Cell IDs are deliberately reversed so publication must reorder cells.
    block = CellBlock(
        "torus",
        "triangle",
        np.asarray(((0, 2, 3), (0, 1, 2)), dtype=np.int32),
        global_ids=np.asarray((7, 3)),
    )
    return _periodic(
        points, block, np.zeros(4, dtype=np.int64), shifts, PeriodicCell(np.eye(2))
    )


def _kuhn_torus() -> CellMesh:
    # The eight cube corners are lattice images of one quotient vertex; the six
    # Kuhn simplices along the monotone corner paths fill the cube.
    shifts = np.asarray(
        [(bit & 1, (bit >> 1) & 1, (bit >> 2) & 1) for bit in range(8)], dtype=np.int64
    )
    tetrahedra = []
    for order in permutations(range(3)):
        corner = 0
        path = [corner]
        for axis in order:
            corner |= 1 << axis
            path.append(corner)
        frame = shifts[path[1:]] - shifts[path[0]]
        if np.linalg.det(frame.astype(np.float64)) < 0.0:
            path[1], path[2] = path[2], path[1]
        tetrahedra.append(path)
    block = CellBlock("torus", "tetrahedron", np.asarray(tetrahedra, dtype=np.int32))
    return _periodic(
        shifts.astype(np.float64),
        block,
        np.zeros(8, dtype=np.int64),
        shifts,
        PeriodicCell(np.eye(3)),
    )


def _cochain_map_defect(mesh: CellMesh, degree: int, seed: int) -> float:
    periodic = mesh.periodic_topology
    assert periodic is not None
    quotient = periodic.quotient.incidences[degree - 1]
    lifted = mesh.topology.incidences[degree - 1]
    values = np.random.default_rng(seed).normal(
        size=periodic.quotient.entities(degree - 1).count
    )
    lifted_then_derived = lifted.exterior_derivative().mv(
        periodic.identification(degree - 1).mv(values)
    )
    derived_then_lifted = periodic.identification(degree).mv(
        quotient.exterior_derivative().mv(values)
    )
    return float(np.max(np.abs(lifted_then_derived - derived_then_lifted)))


def test_two_triangle_torus_publishes_winding_quotient() -> None:
    result = phx.meshing.certify_cell_mesh(
        _two_triangle_torus(), phx.SpatialCoordinateContract.si()
    )
    mesh = result.mesh
    periodic = mesh.periodic_topology

    assert periodic is not None
    np.testing.assert_array_equal(mesh.blocks[0].global_ids, (3, 7))
    assert [entities.count for entities in periodic.quotient.entity_sets] == [1, 3, 2]
    assert periodic.euler_characteristic == 0
    # Three quotient edges join the single representative to itself and differ
    # only by their normalized winding shift.
    keys = periodic.entity_keys(1)
    assert len(set(keys)) == 3
    assert {key[:2] for key in keys} == {(0, 0)}
    first, second = (
        incidence.scipy_boundary().toarray() for incidence in periodic.quotient.incidences
    )
    np.testing.assert_array_equal(first, 0.0)
    np.testing.assert_array_equal(first @ second, 0.0)
    for degree in (1, 2):
        assert _cochain_map_defect(mesh, degree, degree) < 1.0e-14


def test_two_triangle_torus_integrates_each_orbit_once() -> None:
    report = periodic_orbit_measures(_two_triangle_torus())

    np.testing.assert_allclose(report.orbit_measures, (0.5, 0.5), rtol=1.0e-14)
    np.testing.assert_allclose(report.total_measure, 1.0, rtol=1.0e-14)
    assert np.all(np.asarray(report.valid))
    assert report.relative_coverage_defect is not None
    assert report.relative_coverage_defect < 1.0e-14


def test_kuhn_three_torus_quotient_complex() -> None:
    mesh = _kuhn_torus()
    periodic = mesh.periodic_topology

    assert periodic is not None
    counts = [entities.count for entities in periodic.quotient.entity_sets]
    assert counts == [1, 7, 12, 6]
    assert periodic.euler_characteristic == 0
    assert not np.any(np.asarray(periodic.quotient.entities(2).subset("boundary").mask))
    for degree in (1, 2, 3):
        assert _cochain_map_defect(mesh, degree, degree) < 1.0e-13
    report = periodic_orbit_measures(mesh)
    np.testing.assert_allclose(report.total_measure, 1.0, rtol=1.0e-13)


def test_periodic_topology_is_part_of_mesh_identity() -> None:
    mesh = _two_triangle_torus()
    lifted = CellMesh(mesh.coordinates, mesh.blocks)

    assert lifted.periodic_topology is None
    assert mesh.topology_id != lifted.topology_id
    periodic = mesh.periodic_topology
    assert periodic is not None
    assert periodic.lifted_topology_id == lifted.topology_id
    moved = mesh.with_coordinates(
        np.asarray(mesh.coordinates) + 0.25, numeric_version="1"
    )
    assert moved.topology_id == mesh.topology_id
    # Moving one lattice image alone breaks the lift of its representative.
    stretched = np.asarray(mesh.coordinates).copy()
    stretched[1, 0] += 0.1
    with pytest.raises(ValueError, match="inconsistent"):
        mesh.with_coordinates(stretched, numeric_version="2")


def test_inconsistent_lattice_shifts_are_refused() -> None:
    points, shifts = _square_lift()
    lifted = CellMesh.from_triangles(points, np.asarray(((0, 1, 2), (0, 2, 3))))
    wrong = shifts.copy()
    wrong[2] = (1, 0)

    with pytest.raises(ValueError, match="inconsistent"):
        PeriodicMeshTopology(
            lifted, PeriodicCell(np.eye(2)), np.zeros(4, dtype=np.int64), wrong
        )
    shifted_representative = shifts.copy()
    shifted_representative[0] = (1, 0)
    with pytest.raises(ValueError, match="zero shift"):
        PeriodicMeshTopology(
            lifted,
            PeriodicCell(np.eye(2)),
            np.zeros(4, dtype=np.int64),
            shifted_representative,
        )


def test_orientation_conflict_across_the_seam_is_refused() -> None:
    # Each triangle owns its lift; the second is reversed, so both induce the
    # same orientation on their shared quotient diagonal.
    points = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    )
    shifts = points.astype(np.int64)
    lifted = CellMesh.from_triangles(points, np.asarray(((0, 1, 2), (3, 5, 4))))

    with pytest.raises(ValueError, match="orientation conflict"):
        PeriodicMeshTopology(
            lifted, PeriodicCell(np.eye(2)), np.asarray((0, 0, 0, 0, 0, 0)), shifts
        )


def test_repeated_top_cell_orbit_is_refused() -> None:
    points = np.asarray(((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (1.0, 0.0), (1.5, 0.0)))
    points = np.concatenate((points, ((1.0, 0.5),)))
    shifts = np.asarray(((0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (1, 0)))
    lifted = CellMesh.from_triangles(points, np.asarray(((0, 1, 2), (3, 4, 5))))

    with pytest.raises(ValueError, match="exactly once"):
        PeriodicMeshTopology(
            lifted, PeriodicCell(np.eye(2)), np.asarray((0, 1, 2, 0, 1, 2)), shifts
        )


def test_descriptor_is_bound_to_its_lifted_topology() -> None:
    periodic = _two_triangle_torus().periodic_topology
    points, _ = _square_lift()

    other = CellMesh.from_triangles(points, np.asarray(((0, 1, 3), (1, 2, 3))))
    with pytest.raises(ValueError, match="not prepared"):
        CellMesh(points, other.blocks, periodic_topology=periodic)


def _rotational_wedge() -> CellMesh:
    rotation = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    points = np.asarray(((1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (0.0, 2.0)))
    block = CellBlock(
        "wedge", "triangle", np.asarray(((0, 1, 3), (0, 3, 2)), dtype=np.int32)
    )
    return _periodic(
        points,
        block,
        np.asarray((0, 1, 0, 1)),
        np.asarray(((0,), (0,), (1,), (1,))),
        PeriodicIsometryGroup(rotation[None]),
    )


def _half_turn_wedge() -> CellMesh:
    rotation = np.diag(np.asarray((-1.0, -1.0, 1.0), dtype=np.float64))
    points = np.asarray(
        ((1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (0.0, 2.0), (-1.0, 0.0), (-2.0, 0.0)),
        dtype=np.float64,
    )
    block = CellBlock(
        "half-turn",
        "triangle",
        np.asarray(
            ((0, 1, 3), (0, 3, 2), (2, 3, 5), (2, 5, 4)),
            dtype=np.int32,
        ),
    )
    return _periodic(
        points,
        block,
        np.asarray((0, 1, 2, 3, 0, 1), dtype=np.int64),
        np.asarray(((0,), (0,), (0,), (0,), (1,), (1,)), dtype=np.int64),
        PeriodicIsometryGroup(rotation[None]),
    )


def _projection_error(
    mesh: CellMesh, element: FiniteElementSpec, vector: np.ndarray
) -> float:
    D = phx.discretization
    prepared = D.FiniteElementPlan(mesh, D.FiniteElementFieldSpec("u", element)).prepare()
    rule = phx.integration.ReferenceTetrahedronRule(
        phx.integration.GaussLegendreRule(4)
    ).materialize()
    geometry = prepared.evaluate_block_geometry(
        "u", 0, prepared.default_runtime.coordinates, rule.points, rule.weights
    )
    dofs = prepared.dof_maps[0]
    routes = np.asarray(dofs.cell_dofs[0])
    basis = np.einsum(
        "cqav,cai->cqiv",
        np.asarray(geometry.basis_values),
        np.asarray(dofs.cell_transforms[0]),
    )
    weights = np.asarray(geometry.physical_weights)
    local_load = np.einsum("cq,cqiv,v->ci", weights, basis, vector)
    load = np.zeros(dofs.global_dof_count)
    np.add.at(load, routes, local_load)
    coefficients = np.linalg.solve(prepared.mass.to_scipy().toarray(), load)
    evaluated = np.einsum("cqiv,ci->cqv", basis, coefficients[routes])
    return float(np.sqrt(np.sum(weights * np.sum((evaluated - vector) ** 2, axis=-1))))


@pytest.mark.parametrize(
    "element",
    (
        phx.discretization.form_element(
            "tetrahedron", 2, 1, twist="twisted", proxy="flux"
        ),
        phx.discretization.form_element(
            "tetrahedron", 2, 1, family="full", twist="twisted", proxy="flux"
        ),
        phx.discretization.form_element(
            "tetrahedron", 2, 2, family="full", twist="twisted", proxy="flux"
        ),
        phx.discretization.form_element("tetrahedron", 1, 1, proxy="circulation"),
        phx.discretization.form_element("tetrahedron", 1, 2, proxy="circulation"),
    ),
    ids=("rt0", "bdm1", "bdm2", "nedelec0", "nedelec1"),
)
def test_quotient_compatible_moments_reproduce_periodic_vector_fields(
    element: FiniteElementSpec,
) -> None:
    assert (
        _projection_error(_kuhn_torus(), element, np.asarray((0.4, -0.7, 1.2))) < 2.0e-12
    )


@pytest.mark.parametrize("degree", (2, 3, 4, 5))
def test_quotient_pk_trace_nodes_match_every_seam_isometry(degree: int) -> None:
    D = phx.discretization
    mesh = _kuhn_torus()
    prepared = D.FiniteElementPlan(
        mesh, D.FiniteElementFieldSpec("u", D.lagrange_element("tetrahedron", degree))
    ).prepare()
    dofs = prepared.dof_maps[0]
    nodes = np.asarray(dofs.dof_coordinates)
    lifted = np.asarray(mesh.coordinates)[np.asarray(mesh.blocks[0].vertices)]
    weights = np.asarray(dofs.cell_coordinate_weights[0])
    local = np.einsum("ia,cad->cid", weights, lifted)
    periodic_values = lambda points: (
        np.cos(2.0 * np.pi * points[..., 0]) + np.sin(2.0 * np.pi * points[..., 1])
    )
    np.testing.assert_allclose(
        periodic_values(local),
        periodic_values(nodes)[np.asarray(dofs.cell_dofs[0])],
        atol=2.0e-12,
    )
    assert not np.any(np.asarray(dofs.boundary_dof_mask))


@pytest.mark.parametrize(
    "domain_factory,rotation,image_count",
    (
        (_rotational_wedge, np.asarray(((0.0, -1.0), (1.0, 0.0)), dtype=np.float64), 4),
        (_half_turn_wedge, -np.eye(2, dtype=np.float64), 2),
    ),
    ids=("quarter-turn", "half-turn"),
)
@pytest.mark.parametrize("family", ("Hdiv", "Hcurl"))
def test_rotational_wedge_uses_actual_vector_isometry(
    family: str,
    domain_factory: Callable[[], CellMesh],
    rotation: np.ndarray,
    image_count: int,
) -> None:
    D = phx.discretization
    mesh = domain_factory()
    element = (
        D.form_element("triangle", 1, 1, twist="twisted", proxy="flux")
        if family == "Hdiv"
        else D.form_element("triangle", 1, 1, proxy="circulation")
    )
    prepared = D.FiniteElementPlan(mesh, D.FiniteElementFieldSpec("v", element)).prepare()
    import jax.numpy as jnp

    def field(points: Array, args: object) -> Array:
        return (
            points
            if family == "Hdiv"
            else jnp.stack((-points[..., 1], points[..., 0]), axis=-1)
        )

    coefficients = np.asarray(prepared.project("v", field))
    geometry = prepared.evaluate_block_geometry(
        "v",
        0,
        prepared.default_runtime.coordinates,
        np.asarray(((0.5, 0.0), (0.5, 0.5))),
        np.ones(2),
    )
    dofs = prepared.dof_maps[0]
    basis = np.einsum(
        "cqav,cai->cqiv",
        np.asarray(geometry.basis_values),
        np.asarray(dofs.cell_transforms[0]),
    )
    vectors = np.einsum(
        "cqiv,ci->cqv", basis, coefficients[np.asarray(dofs.cell_dofs[0])]
    )
    master, target = vectors[0, 0], vectors[-1, 1]
    np.testing.assert_allclose(target, rotation @ master, atol=1.0e-13)
    assert not np.allclose(target, master)
    np.testing.assert_allclose(
        vectors, field(geometry.physical_points, None), atol=1.0e-13
    )
    report = certify_periodic_embedding(mesh)
    assert report.image_count == image_count
    refined = refine_periodic_mesh(mesh)
    np.testing.assert_allclose(
        refined.quotient.total_measure,
        periodic_orbit_measures(mesh).total_measure,
        atol=1.0e-13,
    )
    certify_periodic_embedding(refined.mesh)


@pytest.mark.parametrize(
    "domain_factory",
    (_rotational_wedge, _half_turn_wedge),
    ids=("quarter-turn", "half-turn"),
)
@pytest.mark.parametrize("degree", (2, 3, 4))
def test_rotational_h1_quotient_reproduces_and_transfers_invariant_scalar(
    domain_factory: Callable[[], CellMesh],
    degree: int,
) -> None:
    import jax.numpy as jnp

    D = phx.discretization
    mesh = domain_factory()
    refinement = refine_periodic_mesh(mesh)
    field = D.FiniteElementFieldSpec("u", D.lagrange_element("triangle", degree))
    source_geometry = D.CellGeometrySpec.affine(mesh)
    target_geometry = D.CellGeometrySpec.affine(refinement.mesh)
    source = D.FiniteElementPlan(mesh, field, coordinate_spec=source_geometry).prepare()
    target = D.FiniteElementPlan(
        refinement.mesh, field, coordinate_spec=target_geometry
    ).prepare()

    def invariant(points: Array, args: object) -> Array:
        return jnp.sum(points * points, axis=-1)

    coefficients = np.asarray(source.project("u", invariant))
    geometry = source.evaluate_block_geometry(
        "u",
        0,
        source.default_runtime.coordinates,
        np.asarray(((0.2, 0.3), (0.4, 0.1)), dtype=np.float64),
        np.ones(2, dtype=np.float64),
    )
    dofs = source.dof_maps[0]
    basis = np.asarray(geometry.basis_values)
    evaluated = np.einsum("qi,ci->cq", basis, coefficients[np.asarray(dofs.cell_dofs[0])])
    np.testing.assert_allclose(
        evaluated, invariant(geometry.physical_points, None), atol=1.0e-12
    )
    transfer = D.prepare_nested_field_transfer(
        source,
        target,
        refinement.parent_cells,
        field_name="u",
        parent_reference_vertices=refinement.parent_reference_vertices,
        source_geometry=source_geometry,
        target_geometry=target_geometry,
    )
    assert transfer.evidence.passed
    np.testing.assert_allclose(
        transfer.transfer.apply(coefficients),
        target.project("u", invariant),
        atol=1.0e-12,
    )
    rng = np.random.default_rng(29)
    primal = rng.normal(size=(source.dof_maps[0].global_dof_count, 2))
    dual = rng.normal(size=(target.dof_maps[0].global_dof_count, 2))
    np.testing.assert_allclose(
        np.vdot(transfer.transfer.apply(primal), dual),
        np.vdot(primal, transfer.transfer.pullback(dual)),
        atol=1.0e-11,
    )


def test_noncommuting_rotation_translation_cycles_are_refused() -> None:
    rotation = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    translation = np.asarray(((1.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    with pytest.raises(ValueError):
        PeriodicIsometryGroup(np.stack((rotation, translation)))


def test_periodic_embedding_checks_nontrivial_self_images() -> None:
    mesh = _two_triangle_torus()
    report = certify_periodic_embedding(mesh)
    assert report.image_count == 9
    with pytest.raises(ValueError, match="exceeding"):
        certify_periodic_embedding(mesh, maximum_images=8)


def _native_periodic_material_fixture(
    domain: CellMesh,
) -> tuple[
    NativePeriodicSource, phx.meshing.SurfaceMeshingSpec, phx.meshing.CellMeshingResult
]:
    M = phx.meshing
    source = NativePeriodicSource(
        domain, "periodic-materials", "r1", cell_regions=np.asarray((0, 1))
    )

    def scope(degree: int, ids: tuple[int, ...]) -> phx.meshing.MeshingScope:
        return M.MeshingScope(
            source.source_id,
            source.source_revision,
            M.MeshingEntityKind.GEOMETRY,
            degree,
            f"entities:{degree}",
            np.asarray(ids, dtype=np.int64),
        )

    regions = scope(2, (0, 1))
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 2, M.CellFamilyPolicy(required=("triangle",))),
        regions,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
        ),
        size_controls=(
            M.UniformSizeControl(
                regions, 0.5, maximum_size=0.75, strength=M.SizeControlStrength.SOFT
            ),
        ),
        protected_features=(M.ProtectedFeature(scope(1, (0,)), M.FeatureKind.CURVE),),
        region_controls=(
            M.RegionControl(scope(2, (0,)), "first", "fluid", M.RegionRole.FLUID),
            M.RegionControl(scope(2, (1,)), "second", "solid", M.RegionRole.SOLID),
        ),
    )
    result = (
        M.NativeMeshingProvider(M.NativeMeshingOptions("periodic_delaunay"))
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )
    return source, specification, result


@pytest.mark.parametrize(
    "domain_factory",
    (_two_triangle_torus, _rotational_wedge),
    ids=("torus", "rotational-wedge"),
)
def test_native_periodic_material_and_feature_orbits_survive_size_refinement(
    domain_factory: Callable[[], CellMesh],
) -> None:
    M = phx.meshing
    domain = domain_factory()
    source, specification, result = _native_periodic_material_fixture(domain)

    certification = result.certification
    if certification is None:
        raise AssertionError("Accepted periodic publication requires its certification.")
    assert result.audit.passed and certification.passed
    assert {zone.material_id for zone in result.zones} == {"fluid", "solid"}
    achieved, requested = (
        dict(result.compliance.achieved),
        dict(result.compliance.requested),
    )
    for region in (0, 1):
        key = f"periodic:region:{region}:measure"
        np.testing.assert_allclose(achieved[key], requested[key], atol=1.0e-12)
    assert len(result.labels) == 1
    label_ids = np.asarray(result.labels[0].scope.entity_ids)
    entities = result.mesh.topology.entities(1)
    mask = np.isin(np.asarray(entities.entity_ids), label_ids)
    orbit = np.asarray(_require_periodic_topology(result.mesh).orbits(1)[0])
    assert np.all(mask == np.isin(orbit, orbit[mask]))
    leaders = np.asarray(_require_periodic_topology(result.mesh).orbit_representatives(1))
    edges = np.asarray(result.mesh.coordinates)[_simplex_entity_corners(result.mesh, 1)]
    lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
    source_leader = int(
        np.asarray(_require_periodic_topology(domain).orbit_representatives(1))[0]
    )
    source_edge = np.asarray(domain.coordinates)[
        _simplex_entity_corners(domain, 1)[source_leader]
    ]
    np.testing.assert_allclose(
        np.sum(lengths[leaders[mask[leaders]]]),
        np.linalg.norm(source_edge[1] - source_edge[0]),
        atol=1.0e-13,
    )
    assert len(result.associations) == 3
    assert all(item.complete and item.exact for item in result.associations)
    transfer = PeriodicAssociationTransfer(source, specification)
    transfer.source_associations(result)
    refined = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            result,
            M.MarkedMeshAdaptation(
                np.asarray(
                    result.mesh.entity_set(result.mesh.topological_dimension).entity_ids
                )
            ),
            policy=M.MeshAdaptationPolicy(
                M.MeshAdaptationRoute.NATIVE_BISECTION,
                limits=specification.limits,
                association_transfer=transfer,
            ),
        )
    )
    assert refined.status is M.MeshAdaptationStatus.COMPLETE
    assert len(refined.target.associations) == 3
    transfer.source_associations(refined.target)
    before_group = _require_periodic_topology(domain).cell
    after_group = _require_periodic_topology(refined.target.mesh).cell
    assert type(after_group) is type(before_group)
    if isinstance(before_group, PeriodicIsometryGroup):
        if not isinstance(after_group, PeriodicIsometryGroup):
            raise AssertionError(
                "Finite rotational source identity must remain finite rotational."
            )
        assert after_group.group_id == before_group.group_id
        assert after_group.orders == before_group.orders

    # Exercise compatible transfer on the actual material/feature publication,
    # not on a separately refined unlabeled rotational fixture.
    import jax.numpy as jnp

    D = phx.discretization
    field = D.FiniteElementFieldSpec(
        "flux", D.form_element("triangle", 1, 2, twist="twisted", proxy="flux")
    )
    source_field = D.FiniteElementPlan(
        result.mesh,
        field,
        coordinate_spec=result.geometry,
    ).prepare()
    target_field = D.FiniteElementPlan(
        refined.target.mesh,
        field,
        coordinate_spec=refined.target.geometry,
    ).prepare()
    field_transfer = D.prepare_nested_field_transfer(
        source_field,
        target_field,
        refined.parent_cells,
        field_name="flux",
        parent_reference_vertices=refined.parent_reference_vertices,
        source_geometry=result.geometry,
        target_geometry=refined.target.geometry,
    )
    assert field_transfer.evidence.passed
    assert (
        field_transfer.evidence.defect("commuting") <= field_transfer.evidence.tolerance
    )

    def compatible_flux(points: jnp.ndarray, args: object) -> jnp.ndarray:
        if isinstance(before_group, PeriodicIsometryGroup):
            return points
        return jnp.broadcast_to(jnp.asarray((1.0, 0.0)), points.shape)

    np.testing.assert_allclose(
        field_transfer.transfer.apply(source_field.project("flux", compatible_flux)),
        target_field.project("flux", compatible_flux),
        atol=1.0e-12,
    )


def test_periodic_embedding_refuses_self_overlap_despite_positive_local_cell() -> None:
    points = np.asarray(((0.0, 0.0), (2.0, 0.0), (0.0, 2.0)))
    mesh = _periodic(
        points,
        CellBlock("wrapping", "triangle", np.asarray(((0, 1, 2),), dtype=np.int32)),
        np.zeros(3, dtype=np.int32),
        points.astype(np.int32),
        PeriodicCell(np.eye(2)),
    )
    assert np.linalg.det(points[1:] - points[:1]) > 0.0
    with pytest.raises(ValueError, match="overlap under image"):
        certify_periodic_embedding(mesh)


@pytest.mark.parametrize(
    "domain_factory",
    (_two_triangle_torus, _rotational_wedge),
    ids=("torus", "rotational-wedge"),
)
def test_periodic_adaptation_refine_coarsen_restores_scientific_quotient(
    domain_factory: Callable[[], CellMesh],
) -> None:
    M = phx.meshing
    source = M.certify_cell_mesh(domain_factory(), phx.SpatialCoordinateContract.si())
    policy = M.MeshAdaptationPolicy(M.MeshAdaptationRoute.NATIVE_BISECTION)
    refined = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            source,
            M.MarkedMeshAdaptation(
                np.asarray(
                    source.mesh.entity_set(source.mesh.topological_dimension).entity_ids
                )
            ),
            policy=policy,
        )
    )
    assert refined.status is M.MeshAdaptationStatus.COMPLETE
    assert refined.target.mesh.periodic_topology is not None
    np.testing.assert_allclose(
        periodic_orbit_measures(refined.target.mesh).total_measure,
        periodic_orbit_measures(source.mesh).total_measure,
        atol=1.0e-12,
    )
    coarsened = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            refined.target,
            M.MarkedMeshAdaptation(
                (),
                np.asarray(
                    refined.target.mesh.entity_set(
                        refined.target.mesh.topological_dimension
                    ).entity_ids
                ),
                hierarchy=refined.hierarchy,
            ),
            policy=policy,
        )
    )
    assert coarsened.status is M.MeshAdaptationStatus.COMPLETE
    assert coarsened.target.mesh.topology_id == source.mesh.topology_id
    np.testing.assert_array_equal(
        coarsened.target.mesh.vertex_global_ids, source.mesh.vertex_global_ids
    )
    for degree in range(source.mesh.topological_dimension + 1):
        original = _require_periodic_topology(source.mesh).quotient.entities(degree)
        restored = _require_periodic_topology(coarsened.target.mesh).quotient.entities(
            degree
        )
        np.testing.assert_array_equal(restored.entity_ids, original.entity_ids)


def test_periodic_orbit_bisection_refuses_label_conforming_closure() -> None:
    M = phx.meshing
    source = M.certify_cell_mesh(
        _two_triangle_torus(), phx.SpatialCoordinateContract.si()
    )
    policy = M.MeshAdaptationPolicy(
        M.MeshAdaptationRoute.NATIVE_BISECTION,
        compatibility=M.BisectionCompatibility.CONFORMING_CLOSURE,
    )
    with pytest.raises(ValueError, match="consumes no Maubach labels"):
        M.prepare_mesh_adaptation(
            source,
            M.MarkedMeshAdaptation(
                np.asarray(
                    source.mesh.entity_set(source.mesh.topological_dimension).entity_ids
                )
            ),
            policy=policy,
        )


def test_decimal_image_rounding_is_bound_without_false_quotient_overlap() -> None:
    points = np.asarray(((0.3, 0.6), (1.3, 0.6), (1.3, 1.6), (0.3, 1.6)))
    mesh = _periodic(
        points,
        CellBlock(
            "decimal", "triangle", np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
        ),
        np.zeros(4, dtype=np.int32),
        np.asarray(((0, 0), (1, 0), (1, 1), (0, 1))),
        PeriodicCell(np.eye(2)),
    )
    evidence = certify_periodic_embedding(mesh)
    assert evidence.coordinate_scope == "exact_dyadic_group_lift"
    assert (
        0.0 < evidence.maximum_coordinate_residual <= evidence.coordinate_residual_bound
    )
    np.testing.assert_allclose(
        periodic_orbit_measures(mesh).total_measure, 1.0, atol=1.0e-14
    )


@pytest.mark.parametrize(
    "domain_factory",
    (_rotational_wedge, _half_turn_wedge),
    ids=("quarter-turn", "half-turn"),
)
@pytest.mark.parametrize("family", ("Hdiv", "Hcurl"))
@pytest.mark.parametrize("order", (1, 2, 3))
def test_rotational_orbit_refinement_transfers_compatible_fields(
    family: str,
    order: int,
    domain_factory: Callable[[], CellMesh],
) -> None:
    import jax.numpy as jnp

    D = phx.discretization
    mesh = domain_factory()
    refinement = refine_periodic_mesh(mesh)
    element = (
        D.form_element("triangle", 1, order, twist="twisted", proxy="flux")
        if family == "Hdiv"
        else D.form_element("triangle", 1, order, proxy="circulation")
    )
    field = D.FiniteElementFieldSpec("v", element)
    source_geometry = D.CellGeometrySpec.affine(mesh)
    target_geometry = D.CellGeometrySpec.affine(refinement.mesh)
    source = D.FiniteElementPlan(mesh, field, coordinate_spec=source_geometry).prepare()
    target = D.FiniteElementPlan(
        refinement.mesh, field, coordinate_spec=target_geometry
    ).prepare()
    function = lambda points, args: (
        points
        if family == "Hdiv"
        else jnp.stack((-points[..., 1], points[..., 0]), axis=-1)
    )
    transfer = D.prepare_nested_field_transfer(
        source,
        target,
        refinement.parent_cells,
        field_name="v",
        parent_reference_vertices=refinement.parent_reference_vertices,
        source_geometry=source_geometry,
        target_geometry=target_geometry,
    )
    assert transfer.evidence.passed
    np.testing.assert_allclose(
        transfer.transfer.apply(source.project("v", function)),
        target.project("v", function),
        atol=1.0e-12,
    )
    assert transfer.evidence.defect("commuting") <= transfer.evidence.tolerance
    rng = np.random.default_rng(37)
    primal = rng.normal(size=(source.dof_maps[0].global_dof_count, 3))
    dual = rng.normal(size=(target.dof_maps[0].global_dof_count, 3))
    np.testing.assert_allclose(
        np.vdot(transfer.transfer.apply(primal), dual),
        np.vdot(primal, transfer.transfer.pullback(dual)),
        atol=1.0e-11,
    )


@pytest.mark.parametrize("family", ("Hdiv", "Hcurl"))
@pytest.mark.parametrize("order", (2, 3))
def test_high_order_rotational_moments_reproduce_equivariant_fields(
    family: str, order: int
) -> None:
    import jax.numpy as jnp

    D = phx.discretization
    mesh = _rotational_wedge()
    element = D.form_element(
        "triangle",
        1,
        order,
        twist="twisted" if family == "Hdiv" else "untwisted",
        proxy="flux" if family == "Hdiv" else "circulation",
    )
    prepared = D.FiniteElementPlan(mesh, D.FiniteElementFieldSpec("v", element)).prepare()

    def field(points: Array, args: object) -> Array:
        radial = (
            points
            if family == "Hdiv"
            else jnp.stack((-points[..., 1], points[..., 0]), axis=-1)
        )
        return (
            radial * (1.0 + jnp.sum(points**2, axis=-1, keepdims=True))
            if order == 3
            else radial
        )

    coefficients = np.asarray(prepared.project("v", field))
    geometry = prepared.evaluate_block_geometry(
        "v",
        0,
        prepared.default_runtime.coordinates,
        np.asarray(((0.2, 0.3), (0.6, 0.1), (0.1, 0.8))),
        np.ones(3),
    )
    dofs = prepared.dof_maps[0]
    basis = np.einsum(
        "cqav,cai->cqiv",
        np.asarray(geometry.basis_values),
        np.asarray(dofs.cell_transforms[0]),
    )
    evaluated = np.einsum(
        "cqiv,ci->cqv", basis, coefficients[np.asarray(dofs.cell_dofs[0])]
    )
    np.testing.assert_allclose(
        evaluated, field(geometry.physical_points, None), atol=2.0e-10
    )


@pytest.mark.parametrize("form_degree", (1, 2))
def test_tensor_quotient_face_moments_use_canonical_cube_charts(form_degree: int) -> None:
    import jax.numpy as jnp

    D = phx.discretization
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        )
    )
    # Row numbering changes each face's canonical chart but not the geometry.
    order = np.asarray((6, 2, 5, 0, 7, 1, 4, 3))
    points = points[order]
    mesh = _periodic(
        points,
        CellBlock("torus", "hexahedron", np.argsort(order).astype(np.int32)[None]),
        np.full(8, int(np.flatnonzero(order == 0)[0]), dtype=np.int64),
        points.astype(np.int64),
        PeriodicCell(np.eye(3)),
    )
    element = D.form_element(
        "hexahedron",
        form_degree,
        2,
        family="tensor-trimmed",
        twist="twisted" if form_degree == 2 else "untwisted",
        proxy="flux" if form_degree == 2 else "circulation",
    )
    prepared = D.FiniteElementPlan(mesh, D.FiniteElementFieldSpec("v", element)).prepare()
    vector = jnp.asarray((0.4, -0.7, 1.2))

    def field(points: Array, args: object) -> Array:
        polynomial = points[..., 0] * (1.0 - points[..., 0])
        direction = jnp.asarray((0.0, 1.0, 0.0) if form_degree == 1 else (1.0, 0.0, 0.0))
        return jnp.broadcast_to(vector, points.shape) + polynomial[..., None] * direction

    coefficients = np.asarray(prepared.project("v", field))
    geometry = prepared.evaluate_block_geometry(
        "v",
        0,
        prepared.default_runtime.coordinates,
        np.asarray(((0.2, 0.3, 0.4), (0.7, 0.8, 0.6))),
        np.ones(2),
    )
    dofs = prepared.dof_maps[0]
    basis = np.einsum(
        "cqav,cai->cqiv",
        np.asarray(geometry.basis_values),
        np.asarray(dofs.cell_transforms[0]),
    )
    evaluated = np.einsum(
        "cqiv,ci->cqv", basis, coefficients[np.asarray(dofs.cell_dofs[0])]
    )
    np.testing.assert_allclose(
        evaluated, field(geometry.physical_points, None), atol=2.0e-12
    )
    assert not np.any(np.asarray(dofs.boundary_dof_mask))


def test_periodic_lift_refusals_leave_the_caller_reusable() -> None:
    mesh = _two_triangle_torus()
    topology = _require_periodic_topology(mesh)
    original_id = topology.periodic_topology_id
    coordinates = np.asarray(mesh.coordinates).copy()
    coordinates[1, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        topology.require_lift(coordinates)
    coordinates[1] = np.asarray(mesh.coordinates)[1]
    topology.require_lift(coordinates)
    with pytest.raises(ValueError, match="exceeding"):
        certify_periodic_embedding(mesh, maximum_images=8)
    assert certify_periodic_embedding(mesh).image_count == 9
    assert topology.periodic_topology_id == original_id


def test_periodic_shift_storage_refuses_winding_truncation() -> None:
    points, shifts = _square_lift()
    block = CellBlock(
        "torus", "triangle", np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    )
    lifted = CellMesh(points, (block,))
    authored = shifts.copy()
    authored[1, 0] = 1 << 32
    with pytest.raises(ValueError, match="int32 image representation"):
        PeriodicMeshTopology(
            lifted, PeriodicCell(np.eye(2)), np.zeros(4, dtype=np.int64), authored
        )
    np.testing.assert_array_equal(authored[1], (1 << 32, 0))
    accepted = PeriodicMeshTopology(
        lifted, PeriodicCell(np.eye(2)), np.zeros(4, dtype=np.int64), shifts
    )
    assert accepted.quotient.entities(1).count == 3


def _rotation_translation_wedge() -> CellMesh:
    base = np.asarray(_rotational_wedge().coordinates)
    points = np.concatenate((np.c_[base, np.zeros(4)], np.c_[base, np.ones(4)]), axis=0)
    tetrahedra = []
    for a, b, c in ((0, 1, 3), (0, 2, 3)):
        for corners in ((a, b, c, c + 4), (a, b, b + 4, c + 4), (a, a + 4, b + 4, c + 4)):
            vertices = list(corners)
            if np.linalg.det((points[vertices[1:]] - points[vertices[0]]).T) < 0.0:
                vertices[1], vertices[2] = vertices[2], vertices[1]
            tetrahedra.append(vertices)
    rotation = np.eye(4)
    rotation[:2, :2] = np.asarray(((0.0, -1.0), (1.0, 0.0)))
    translation = np.eye(4)
    translation[2, 3] = 1.0
    shifts = np.asarray([(int(index % 4 >= 2), index // 4) for index in range(8)])
    return _periodic(
        points,
        CellBlock("wedge", "tetrahedron", np.asarray(tetrahedra, dtype=np.int32)),
        np.asarray((0, 1, 0, 1, 0, 1, 0, 1)),
        shifts,
        PeriodicIsometryGroup(np.stack((rotation, translation))),
    )


@pytest.mark.parametrize("family", ("Hdiv", "Hcurl", "H1"))
def test_high_order_rotation_translation_combination(family: str) -> None:
    import jax.numpy as jnp

    D = phx.discretization
    mesh = _rotation_translation_wedge()
    element = (
        D.lagrange_element("tetrahedron", 3)
        if family == "H1"
        else D.form_element(
            "tetrahedron",
            2 if family == "Hdiv" else 1,
            2,
            twist="twisted" if family == "Hdiv" else "untwisted",
            proxy="flux" if family == "Hdiv" else "circulation",
        )
    )
    prepared = D.FiniteElementPlan(mesh, D.FiniteElementFieldSpec("u", element)).prepare()

    def field(points: Array, args: object) -> Array:
        if family == "H1":
            return points[..., 0] ** 2 + points[..., 1] ** 2
        xy = (
            points[..., :2]
            if family == "Hdiv"
            else jnp.stack((-points[..., 1], points[..., 0]), axis=-1)
        )
        return jnp.concatenate((xy, jnp.ones_like(points[..., 2:])), axis=-1)

    coefficients = np.asarray(prepared.project("u", field))
    geometry = prepared.evaluate_block_geometry(
        "u",
        0,
        prepared.default_runtime.coordinates,
        np.asarray(((0.1, 0.2, 0.3), (0.4, 0.1, 0.2))),
        np.ones(2),
    )
    dofs = prepared.dof_maps[0]
    local = coefficients[np.asarray(dofs.cell_dofs[0])]
    if family == "H1":
        evaluated = np.einsum("qi,ci->cq", np.asarray(geometry.basis_values), local)
    else:
        basis = np.einsum(
            "cqav,cai->cqiv",
            np.asarray(geometry.basis_values),
            np.asarray(dofs.cell_transforms[0]),
        )
        evaluated = np.einsum("cqiv,ci->cqv", basis, local)
    np.testing.assert_allclose(
        evaluated, field(geometry.physical_points, None), atol=2.0e-11
    )
    cell = _require_periodic_topology(mesh).cell
    if not isinstance(cell, D.PeriodicIsometryGroup):
        raise TypeError("The wedge must retain its authored rotation/translation group.")
    assert cell.orders == (4, 0)


@pytest.mark.parametrize("domain_factory", (_two_triangle_torus, _kuhn_torus))
@pytest.mark.parametrize("degree", (3, 4))
def test_coordinate_lattice_node_order_lowers_to_consistent_quotient(
    domain_factory: Callable[[], CellMesh], degree: int
) -> None:
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element

    D = phx.discretization
    mesh = domain_factory()
    element = coordinate_lagrange_element(mesh.blocks[0].cell_kind, degree)
    dofs = D.FiniteElementDofMap(mesh, (element,))
    lifted = np.asarray(mesh.coordinates)[np.asarray(mesh.blocks[0].vertices)]
    local = np.einsum("ia,cad->cid", np.asarray(dofs.cell_coordinate_weights[0]), lifted)
    quotient = np.asarray(dofs.dof_coordinates)[np.asarray(dofs.cell_dofs[0])]
    np.testing.assert_allclose(
        np.cos(2.0 * np.pi * local), np.cos(2.0 * np.pi * quotient), atol=2.0e-12
    )
    assert not np.any(np.asarray(dofs.boundary_dof_mask))


@pytest.mark.parametrize("degree", (2, 3, 4))
@pytest.mark.parametrize("rotational", (False, True), ids=("translation", "quarter-turn"))
def test_gmsh_actual_node_orbits_match_native_reference_numbering(
    degree: int,
    rotational: bool,
) -> None:
    """Compare actual external node pairs, not a fitted coordinate pairing."""
    gmsh = pytest.importorskip(
        "gmsh", reason="optional Gmsh orbit reference is not installed"
    )
    D = phx.discretization
    mesh = _rotational_wedge() if rotational else _two_triangle_torus()
    element = D.lagrange_element("triangle", degree)
    dofs = D.FiniteElementDofMap(mesh, (element,))
    routes = np.asarray(dofs.cell_dofs[0])
    # Canonical entity ownership gives the authored edge charts directly:
    # source 0->1 and target 2->1. Floating reference coordinates are not
    # scientific identity and may retain roundoff at reference boundaries.
    vertex_rows = tuple(entity[0] for entity in element.entity_dofs[0])
    source_rows = np.asarray(
        (vertex_rows[0], *element.entity_dofs[1][0], vertex_rows[1]),
        dtype=np.int64,
    )
    target_rows = np.asarray(
        (vertex_rows[2], *reversed(element.entity_dofs[1][1]), vertex_rows[1]),
        dtype=np.int64,
    )
    native_source = routes[0 if rotational else 1, source_rows]
    native_target = routes[1 if rotational else 0, target_rows]
    np.testing.assert_array_equal(native_source, native_target)

    already_initialized = bool(gmsh.isInitialized())
    if not already_initialized:
        gmsh.initialize()
    previous_model = gmsh.model.getCurrent()
    gmsh.model.add("native-orbit-reference")
    try:
        points = [
            gmsh.model.geo.addPoint(float(x), float(y), 0.0)
            for x, y in np.asarray(mesh.coordinates)
        ]
        master = gmsh.model.geo.addLine(points[0], points[1])
        slave_corners = (2, 3) if rotational else (3, 2)
        slave = gmsh.model.geo.addLine(points[slave_corners[0]], points[slave_corners[1]])
        gmsh.model.geo.synchronize()
        for curve in (master, slave):
            gmsh.model.mesh.setTransfiniteCurve(curve, 2)
        action = np.eye(4)
        if rotational:
            action[:2, :2] = ((0.0, -1.0), (1.0, 0.0))
        else:
            action[1, 3] = 1.0
        gmsh.model.mesh.setPeriodic(1, [slave], [master], action.ravel().tolist())
        gmsh.model.mesh.generate(1)
        gmsh.model.mesh.setOrder(degree)
        master_tag, slave_tags, master_tags, affine = gmsh.model.mesh.getPeriodicNodes(
            1, slave, includeHighOrderNodes=True
        )
        assert master_tag == master
        np.testing.assert_allclose(np.asarray(affine).reshape(4, 4), action, atol=0.0)

        def reference_tags(curve: int) -> np.ndarray:
            types, _, nodes = gmsh.model.mesh.getElements(1, curve)
            assert len(types) == 1
            _, dimension, order, count, reference, _ = (
                gmsh.model.mesh.getElementProperties(int(types[0]))
            )
            assert dimension == 1 and order == degree
            connectivity = np.asarray(nodes[0]).reshape(-1, count)
            assert connectivity.shape[0] == 1
            # Gmsh's own reference element orders the tags; physical positions
            # are never rounded or used as scientific identity.
            return connectivity[0, np.argsort(np.asarray(reference))]

        source_tags = reference_tags(master)
        target_tags = reference_tags(slave)
        source_identity = dict(
            zip(source_tags.tolist(), native_source.tolist(), strict=True)
        )
        target_identity = dict(
            zip(target_tags.tolist(), native_target.tolist(), strict=True)
        )
        assert len(slave_tags) == degree + 1
        assert set(slave_tags.tolist()) == set(target_identity)
        assert set(master_tags.tolist()) == set(source_identity)
        for target_tag, source_tag in zip(slave_tags, master_tags, strict=True):
            assert target_identity[int(target_tag)] == source_identity[int(source_tag)]
            target_point = np.asarray(gmsh.model.mesh.getNode(int(target_tag))[0])
            source_point = np.asarray(gmsh.model.mesh.getNode(int(source_tag))[0])
            np.testing.assert_allclose(
                target_point, action[:3, :3] @ source_point + action[:3, 3], atol=2.0e-12
            )
    finally:
        gmsh.model.remove()
        if already_initialized:
            if previous_model:
                gmsh.model.setCurrent(previous_model)
        else:
            gmsh.finalize()


@pytest.mark.parametrize(
    ("family", "order"),
    (("H1", 4), ("Hdiv", 3), ("Hcurl", 3)),
)
def test_native_rotational_publication_transfers_high_order_fields(
    family: str,
    order: int,
) -> None:
    import jax.numpy as jnp

    D = phx.discretization
    source_owner, specification, publication, adaptation = (
        _native_rotational_refinement_fixture(volume=True)
    )
    if not isinstance(source_owner.domain, CellMesh):
        raise TypeError(
            "The rotational native source must retain its actual lifted carrier."
        )
    assert publication.audit.passed
    assert publication.certification is not None and publication.certification.passed
    mesh = publication.mesh
    original_coordinates = np.asarray(mesh.coordinates).copy()
    original_regions = source_owner.cell_regions.copy()
    source_binding = source_owner.binding_id
    topology_id = _require_periodic_topology(mesh).periodic_topology_id
    from phydrax.meshing._periodic import PeriodicRefinement

    refinement = adaptation.hierarchy
    if not isinstance(refinement, PeriodicRefinement):
        raise TypeError(
            "Native orbit adaptation must retain its actual periodic reference ancestry."
        )
    assert refinement.mesh.mesh_id == adaptation.target.mesh.mesh_id
    element = (
        D.lagrange_element(mesh.blocks[0].cell_kind, order)
        if family == "H1"
        else D.form_element(
            mesh.blocks[0].cell_kind,
            mesh.topological_dimension - 1 if family == "Hdiv" else 1,
            order,
            twist="twisted" if family == "Hdiv" else "untwisted",
            proxy="flux" if family == "Hdiv" else "circulation",
        )
    )
    field = D.FiniteElementFieldSpec("u", element)
    source, target = (
        D.FiniteElementPlan(
            carrier.mesh, field, coordinate_spec=carrier.geometry
        ).prepare()
        for carrier in (publication, adaptation.target)
    )

    def equivariant(points: Array, args: object) -> Array:
        if family == "H1":
            return jnp.sum(points * points, axis=-1)
        if family == "Hdiv":
            return points
        swirl = jnp.stack((-points[..., 1], points[..., 0]), axis=-1)
        return jnp.concatenate((swirl, jnp.ones_like(points[..., 2:])), axis=-1)

    transfer = D.prepare_nested_field_transfer(
        source,
        target,
        refinement.parent_cells,
        field_name="u",
        parent_reference_vertices=refinement.parent_reference_vertices,
        source_geometry=publication.geometry,
        target_geometry=adaptation.target.geometry,
    )
    assert transfer.evidence.passed
    np.testing.assert_allclose(
        transfer.transfer.apply(source.project("u", equivariant)),
        target.project("u", equivariant),
        atol=2.0e-11,
    )
    if family != "H1":
        assert transfer.evidence.defect("commuting") <= transfer.evidence.tolerance
    rng = np.random.default_rng(41)
    primal = rng.normal(size=(source.dof_maps[0].global_dof_count, 2))
    dual = rng.normal(size=(target.dof_maps[0].global_dof_count, 2))
    np.testing.assert_allclose(
        np.vdot(transfer.transfer.apply(primal), dual),
        np.vdot(primal, transfer.transfer.pullback(dual)),
        atol=2.0e-11,
    )
    np.testing.assert_array_equal(mesh.coordinates, original_coordinates)
    np.testing.assert_array_equal(source_owner.cell_regions, original_regions)
    assert source_owner.binding_id == source_binding
    assert _require_periodic_topology(mesh).periodic_topology_id == topology_id
    np.testing.assert_allclose(
        refinement.quotient.total_measure,
        periodic_orbit_measures(mesh).total_measure,
        atol=2.0e-12,
    )


def _exact_periodic_rounding_fixture() -> tuple[
    CellMesh,
    phx.discretization.CellGeometrySpec,
    PeriodicIsometryGroup,
    np.ndarray,
    np.ndarray,
]:
    from fractions import Fraction

    from phydrax.discretization._exact_power_geometry import (
        ExactPowerCellGeometryLinearActionSource,
        ExactPowerCellGeometrySource,
    )
    from phydrax.geometry._triangulation import PeriodicPowerPreparation

    sites = np.asarray(((1.0, 0.0, 0.0), (1.0, 3.0, 0.0)))
    weights = np.asarray((0.0, 7.0))
    carrier = np.asarray(
        ((1.0, 0.0, 0.0), (2.0, 1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    generator = np.eye(4)
    generator[0, 3] = -1.0
    group = PeriodicIsometryGroup(generator[None], tolerance=0.0)
    preparation = PeriodicPowerPreparation(
        sites,
        weights,
        carrier,
        group,
        maximum_images=64,
    )
    original_images = []
    for site in range(2):
        selected = np.flatnonzero(
            (np.asarray(preparation.image_sites) == site)
            & np.all(np.asarray(preparation.image_exponents) == 0, axis=1)
        )
        assert selected.size == 1
        original_images.append(int(selected[0]))
    origin_image = np.flatnonzero(
        (np.asarray(preparation.image_sites) == 0)
        & np.all(np.asarray(preparation.image_exponents) == 1, axis=1)
    )
    assert origin_image.size == 1
    parent = ExactPowerCellGeometrySource(
        sites,
        weights,
        carrier,
        np.asarray(((0, 1, 2, 3),), dtype=np.int64),
        np.asarray((0, 2, 3, 4), dtype=np.int64),
        np.asarray(
            (
                original_images[0],
                original_images[1],
                int(origin_image[0]),
                int(origin_image[0]),
            ),
            dtype=np.int64,
        ),
        np.asarray(((0, 1, -1, -1), (2, -1, -1, -1), (3, -1, -1, -1)), dtype=np.int64),
        periodic_preparation=preparation,
    )
    source = ExactPowerCellGeometryLinearActionSource(
        parent,
        np.asarray(((0,), (0,), (1,), (2,)), dtype=np.int64),
        ((Fraction(1),),) * 4,
        np.asarray(((0,), (1,), (0,), (0,)), dtype=np.int64),
        periodic_preparation=preparation,
    )
    exact = source.prepare()
    assert exact.vertices[0] == (Fraction(4, 3), Fraction(1, 3), Fraction(0))
    assert exact.vertices[1] == (Fraction(1, 3), Fraction(1, 3), Fraction(0))
    faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
    lifted = CellMesh.from_polyhedra(
        exact.rounded_vertices,
        (tuple(np.asarray(face, dtype=np.int64) for face in faces),),
    )
    geometry = phx.discretization.CellGeometrySpec.power(lifted, source)
    return (
        lifted,
        geometry,
        group,
        np.asarray((0, 0, 2, 3)),
        np.asarray(((0,), (1,), (0,), (0,))),
    )


def test_exact_periodic_source_admits_independently_rounded_group_images(
    tmp_path: Path,
) -> None:
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )

    lifted, geometry, group, representatives, shifts = _exact_periodic_rounding_fixture()
    original_bits = np.asarray(lifted.coordinates).view(np.uint64).copy()
    with pytest.raises(ValueError, match="inconsistent"):
        PeriodicMeshTopology(lifted, group, representatives, shifts)
    topology = PeriodicMeshTopology(
        lifted,
        group,
        representatives,
        shifts,
        actual_geometry=geometry,
    )
    topology.require_lift(lifted.coordinates)
    np.testing.assert_array_equal(topology.vertex_fixed_generators, False)
    connectivity = lifted.connectivity
    if not isinstance(connectivity, phx.discretization.PolyhedralConnectivity):
        raise TypeError(
            "The exact periodic source requires packed polyhedral connectivity."
        )
    mesh = CellMesh(
        lifted.coordinates,
        lifted.blocks,
        vertex_global_ids=lifted.vertex_global_ids,
        polyhedral_connectivity=connectivity,
        periodic_topology=topology,
    )
    canonical = phx.meshing.canonicalize_cell_mesh(mesh)
    authority = _require_periodic_topology(canonical).actual_geometry
    assert authority is not None and authority.exact_source is not None
    assert geometry.exact_source is not None
    assert authority.exact_source.source_id == geometry.exact_source.source_id
    np.testing.assert_array_equal(
        np.asarray(canonical.coordinates).view(np.uint64), original_bits
    )
    receipt = write_meshing_source_closure(
        tmp_path / "exact-periodic-authority", (lifted, topology)
    )
    restored_lifted, restored = read_meshing_source_closure(
        receipt.path,
        expected_content_id=receipt.content_id,
    )
    rebuilt = restored.rebuilt(restored_lifted)
    assert rebuilt.periodic_topology_id == topology.periodic_topology_id
    assert rebuilt.actual_geometry is not None
    assert rebuilt.actual_geometry.source_coordinates() == geometry.source_coordinates()
    rebuilt.require_lift(restored_lifted.coordinates)
    np.testing.assert_array_equal(
        np.asarray(restored_lifted.coordinates).view(np.uint64), original_bits
    )


def test_exact_periodic_source_refuses_changed_rne_and_false_orbit() -> None:
    import equinox as eqx

    lifted, geometry, group, representatives, shifts = _exact_periodic_rounding_fixture()
    altered = np.asarray(geometry.coordinates).copy()
    altered[1, 0] = np.nextafter(altered[1, 0], np.inf)
    wrong_geometry = eqx.tree_at(lambda value: value.coordinates, geometry, altered)
    with pytest.raises(ValueError, match="RNE carrier"):
        PeriodicMeshTopology(
            lifted,
            group,
            representatives,
            shifts,
            actual_geometry=wrong_geometry,
        )
    wrong_shifts = shifts.copy()
    wrong_shifts[1] = 2
    with pytest.raises(ValueError, match="authored periodic group lifts"):
        PeriodicMeshTopology(
            lifted,
            group,
            representatives,
            wrong_shifts,
            actual_geometry=geometry,
        )


@pytest.mark.parametrize("condition_control", [None, 4.0])
def test_periodic_cell_restoration_authenticates_original_controls_and_derived_fields(
    condition_control: float | None,
) -> None:
    vectors = np.asarray(((1.0, 0.25), (0.0, 1.0)))
    origin = np.asarray((-0.0, 0.0))
    cell = PeriodicCell(
        vectors,
        origin=origin,
        periodic_axes=(True, False),
        maximum_condition_number=condition_control,
        maximum_image_count=1024,
    )
    assert cell.maximum_condition_number == condition_control
    assert cell.maximum_image_count == 1024
    np.testing.assert_array_equal(
        np.asarray(cell.vectors).view(np.uint64), vectors.view(np.uint64)
    )
    np.testing.assert_array_equal(
        np.asarray(cell.origin).view(np.uint64), origin.view(np.uint64)
    )
    original_id = cell.cell_id
    cell.validate_restored()
    assert cell.cell_id == original_id
    replay = PeriodicCell(
        cell.vectors,
        origin=cell.origin,
        periodic_axes=cell.periodic_axes,
        maximum_condition_number=cell.maximum_condition_number,
        maximum_image_count=cell.maximum_image_count,
    )
    assert replay.cell_id == cell.cell_id
    np.testing.assert_array_equal(replay.inverse_vectors, cell.inverse_vectors)
    np.testing.assert_array_equal(replay.image_shifts, cell.image_shifts)


@pytest.mark.parametrize(
    "field",
    [
        "inverse_vectors",
        "reciprocal_vectors",
        "periodic_mask",
        "image_shifts",
        "unique_image_radius",
        "condition_number",
        "certified_condition_number",
        "image_extent",
        "volume",
        "cell_id",
        "maximum_image_count",
        "maximum_condition_number",
    ],
)
def test_periodic_cell_restoration_rejects_forged_derived_or_control_fields(
    field: str,
) -> None:
    from copy import copy

    cell = PeriodicCell(np.eye(2), maximum_condition_number=2.0, maximum_image_count=1024)
    malformed = copy(cell)
    value = getattr(cell, field)
    if field == "cell_id":
        altered = "foreign-cell"
    elif field == "periodic_mask":
        altered = ~value
    else:
        altered = value + 1
    object.__setattr__(malformed, field, altered)
    with pytest.raises(ValueError, match="Restored PeriodicCell"):
        malformed.validate_restored()
    cell.validate_restored()


def _native_rotational_refinement_fixture(
    *,
    distributed: bool = False,
    volume: bool = False,
    mixed: bool = False,
) -> tuple[
    NativePeriodicSource,
    phx.meshing.SurfaceMeshingSpec | phx.meshing.VolumeMeshingSpec,
    phx.meshing.CellMeshingResult,
    phx.meshing.MeshAdaptationResult,
]:
    M = phx.meshing
    source, specification, publication = (
        _native_periodic_volume_material_fixture(mixed=mixed)
        if volume
        else _native_periodic_material_fixture(_rotational_wedge())
    )
    partition_policy = (
        M.MeshPartitionPolicy(M.MeshPartitionKind.MORTON, 2, halo_width=1)
        if distributed
        else None
    )
    distribution = (
        M.prepare_mesh_distribution(
            M.MeshPart("rotational-source", publication),
            policy=partition_policy,
        )
        if partition_policy is not None
        else None
    )
    adaptation = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            publication,
            M.MarkedMeshAdaptation(
                np.asarray(
                    publication.mesh.entity_set(
                        publication.mesh.topological_dimension
                    ).entity_ids
                )
            ),
            policy=M.MeshAdaptationPolicy(
                M.MeshAdaptationRoute.NATIVE_BISECTION,
                limits=specification.limits,
                association_transfer=PeriodicAssociationTransfer(source, specification),
                distribution=distribution,
                partition_policy=partition_policy,
            ),
        )
    )
    assert adaptation.status is M.MeshAdaptationStatus.COMPLETE
    assert adaptation.target.audit.passed
    assert (
        adaptation.target.certification is not None
        and adaptation.target.certification.passed
    )
    if distributed:
        assert adaptation.distribution is not None
        original = adaptation.distribution.source.part.carrier
        successor = adaptation.distribution.target.part.carrier
        assert isinstance(original, M.CellMeshingResult) and isinstance(
            successor, M.CellMeshingResult
        )
        assert original.result_id == publication.result_id
        assert successor.result_id == adaptation.target.result_id
    return source, specification, publication, adaptation


def test_native_rotational_distribution_transfers_actual_source_frames() -> None:
    import jax

    from phydrax.meshing._periodic import PeriodicRefinement

    devices = tuple(jax.local_devices()[:2])
    if len(devices) != 2:
        pytest.skip(
            "Requires two actual JAX devices for quotient halo and paired-transfer execution."
        )
    _, specification, _, adaptation = _native_rotational_refinement_fixture(
        distributed=True
    )
    distribution = adaptation.distribution
    refinement = adaptation.hierarchy
    assert distribution is not None and isinstance(refinement, PeriodicRefinement)
    D = phx.discretization
    element = D.form_element(
        "triangle",
        1,
        2,
        twist="twisted",
        proxy="flux",
    )
    _check_native_distributed_periodic_transfer(
        distribution.source,
        distribution.target,
        element,
        refinement.parent_cells,
        refinement.parent_reference_vertices,
        specification.limits,
        devices,
    )


def _check_native_distributed_periodic_transfer(
    source_distribution: phx.meshing.MeshDistribution,
    target_distribution: phx.meshing.MeshDistribution,
    element: FiniteElementSpec,
    parent_cells: ArrayLike,
    reference_vertices: ArrayLike,
    limits: phx.meshing.MeshingLimits,
    devices: tuple[Device, ...],
) -> None:
    import jax.numpy as jnp

    D = phx.discretization
    original, successor = (
        source_distribution.part.carrier,
        target_distribution.part.carrier,
    )
    assert isinstance(original, phx.meshing.CellMeshingResult)
    assert isinstance(successor, phx.meshing.CellMeshingResult)
    field = D.FiniteElementFieldSpec("u", element)
    source_plan = D.FiniteElementPlan(
        original.mesh, field, coordinate_spec=original.geometry
    )
    target_plan = D.FiniteElementPlan(
        successor.mesh, field, coordinate_spec=successor.geometry
    )
    source, target = source_plan.prepare(), target_plan.prepare()
    transfer = D.prepare_nested_field_transfer(
        source,
        target,
        parent_cells,
        field_name="u",
        parent_reference_vertices=reference_vertices,
        source_geometry=original.geometry,
        target_geometry=successor.geometry,
    )
    assert transfer.evidence.passed
    target_frame = D.FiniteElementGlobalDofOwnership(
        target_plan,
        target_distribution.partition,
        source_discretization=target,
    )
    source_preparation = source_distribution.prepare_finite_element_closures(
        source_plan,
        limits=limits,
        cell_capacity=1024,
        vertex_capacity=1024,
        message_capacity=1024,
        axis_name="parts",
        source_discretization=source,
        transfer=transfer,
        target_authority=target_frame,
    )
    target_preparation = target_distribution.prepare_finite_element_closures(
        target_plan,
        limits=limits,
        cell_capacity=1024,
        vertex_capacity=1024,
        message_capacity=1024,
        axis_name="parts",
        source_discretization=target,
    )
    source_authority, source_programs = (
        source_preparation.authority,
        source_preparation.programs,
    )
    target_authority, target_programs = (
        target_preparation.authority,
        target_preparation.programs,
    )
    assert (
        source_authority.source_field.field_space_id
        == source.field_spaces[0].field_space_id
    )
    assert (
        target_authority.source_field.field_space_id
        == target.field_spaces[0].field_space_id
    )
    assert target_authority.scientific_keys == target_frame.scientific_keys
    for authority, programs in (
        (source_authority, source_programs),
        (target_authority, target_programs),
    ):
        owned = np.concatenate(
            [
                np.asarray(program.global_ids)[np.asarray(program.owned_mask)]
                for program in programs
            ]
        )
        np.testing.assert_array_equal(np.sort(owned), np.asarray(authority.global_ids))
    paired = D.prepare_owner_local_finite_element_transfer(
        transfer,
        source_authority,
        target_authority,
        source_programs,
        target_programs,
    )
    execution_limits = D.FiniteElementExecutionLimits(
        1024,
        1024,
        min(limits.maximum_connectivity_entries, 1024 * element.local_dof_count**2),
        1024
        * max(
            1,
            2
            * sum(
                len(pair.source_halo.permutations) + len(pair.target_halo.permutations)
                for pair in paired
            ),
        ),
        limits.maximum_scratch_bytes,
    )
    rng = np.random.default_rng(53)
    values = rng.normal(size=(source.dof_maps[0].global_dof_count, 2))
    duals = rng.normal(size=(target.dof_maps[0].global_dof_count, 2))
    source_rows = tuple(
        np.asarray(source_authority.global_to_source)[np.asarray(program.global_ids)]
        for program in source_programs
    )
    target_rows = tuple(
        np.asarray(target_authority.global_to_source)[np.asarray(program.global_ids)]
        for program in target_programs
    )
    forward = D.execute_owner_local_finite_element_transfer(
        paired,
        tuple(values[rows] for rows in source_rows),
        devices=devices,
        execution_limits=execution_limits,
    )
    transpose = D.execute_owner_local_finite_element_transfer(
        paired,
        tuple(duals[rows] for rows in target_rows),
        devices=devices,
        direction="transpose",
        execution_limits=execution_limits,
    )
    adjoint = D.execute_owner_local_finite_element_transfer(
        paired,
        tuple(duals[rows] for rows in target_rows),
        devices=devices,
        direction="adjoint",
        execution_limits=execution_limits,
    )
    expected = (
        np.asarray(transfer.transfer.apply(values)),
        np.asarray(transfer.transfer.pullback(duals)),
        np.asarray(transfer.transfer.primal.adjoint_mv_block(jnp.asarray(duals))),
    )
    for output, reference, programs, rows_by_rank in (
        (forward, expected[0], target_programs, target_rows),
        (transpose, expected[1], source_programs, source_rows),
        (adjoint, expected[2], source_programs, source_rows),
    ):
        for rank, (program, rows) in enumerate(zip(programs, rows_by_rank, strict=True)):
            actual = np.asarray(output)[rank]
            np.testing.assert_allclose(
                actual[: len(rows)],
                np.where(np.asarray(program.owned_mask)[:, None], reference[rows], 0),
                atol=2.0e-11,
            )
            np.testing.assert_array_equal(actual[len(rows) :], 0.0)
    forward_pairing = sum(
        np.vdot(np.asarray(forward)[rank, : len(rows)], duals[rows])
        for rank, rows in enumerate(target_rows)
    )
    reverse_pairing = sum(
        np.vdot(values[rows], np.asarray(transpose)[rank, : len(rows)])
        for rank, rows in enumerate(source_rows)
    )
    np.testing.assert_allclose(forward_pairing, reverse_pairing, atol=2.0e-11)


@pytest.mark.parametrize(
    "axis_vertex", (False, True), ids=("free-orbits", "composed-stabilizer")
)
def test_exact_periodic_fixed_flags_do_not_hide_composed_stabilizers(
    axis_vertex: bool,
) -> None:
    from phydrax.discretization._exact_power_geometry import ExactPowerCellGeometrySource
    from phydrax.discretization._periodic_topology import PeriodicIsometryIdentityError

    points = (
        np.asarray(((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0.0, 0.0, 2.0)))
        if axis_vertex
        else np.asarray(
            ((1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 1.0, 2.0))
        )
    )
    source = ExactPowerCellGeometrySource(
        np.zeros((1, 3)),
        np.zeros(1),
        points,
        np.asarray(((0, 1, 2, 3),), dtype=np.int64),
        np.arange(5, dtype=np.int64),
        np.zeros(4, dtype=np.int64),
        np.c_[np.arange(4, dtype=np.int64), np.full((4, 3), -1, dtype=np.int64)],
    )
    faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
    lifted = CellMesh.from_polyhedra(
        source.prepare().rounded_vertices,
        (tuple(np.asarray(face, dtype=np.int64) for face in faces),),
    )
    geometry = phx.discretization.CellGeometrySpec.power(lifted, source)
    group = PeriodicIsometryGroup(
        np.stack((np.diag((1.0, -1.0, -1.0, 1.0)), np.diag((-1.0, 1.0, -1.0, 1.0)))),
        tolerance=0.0,
    )
    representatives = np.arange(4, dtype=np.int64)
    shifts = np.zeros((4, 2), dtype=np.int64)
    if axis_vertex:
        with pytest.raises(
            PeriodicIsometryIdentityError, match="composed vertex stabilizer"
        ):
            PeriodicMeshTopology(
                lifted, group, representatives, shifts, actual_geometry=geometry
            )
    else:
        topology = PeriodicMeshTopology(
            lifted,
            group,
            representatives,
            shifts,
            actual_geometry=geometry,
        )
        topology.require_lift(lifted.coordinates)


def _native_periodic_volume_material_fixture(
    *, mixed: bool = False
) -> tuple[
    NativePeriodicSource,
    phx.meshing.VolumeMeshingSpec,
    phx.meshing.CellMeshingResult,
]:
    M = phx.meshing
    extruded = _rotation_translation_wedge()
    original_group = _require_periodic_topology(extruded).cell
    assert isinstance(original_group, PeriodicIsometryGroup)
    if mixed:
        topology = _require_periodic_topology(extruded)
        domain = extruded
    else:
        # An independent finite-group family has physical top/bottom caps;
        # the authored rotation+translation source is not modified.
        group = PeriodicIsometryGroup(np.asarray(original_group.generators)[:1])
        lifted = CellMesh(
            extruded.coordinates,
            extruded.blocks,
            vertex_global_ids=extruded.vertex_global_ids,
        )
        topology = PeriodicMeshTopology(
            lifted,
            group,
            np.asarray((0, 1, 0, 1, 4, 5, 4, 5)),
            np.asarray(((0,), (0,), (1,), (1,), (0,), (0,), (1,), (1,))),
        )
        domain = CellMesh(
            lifted.coordinates,
            lifted.blocks,
            vertex_global_ids=lifted.vertex_global_ids,
            periodic_topology=topology,
        )
    source = NativePeriodicSource(
        domain,
        "periodic-mixed-volume-materials" if mixed else "periodic-volume-materials",
        "r1",
        cell_regions=np.asarray((0, 0, 0, 1, 1, 1), dtype=np.int64),
    )

    def scope(degree: int, ids: np.ndarray) -> phx.meshing.MeshingScope:
        return M.MeshingScope(
            source.source_id,
            source.source_revision,
            M.MeshingEntityKind.GEOMETRY,
            degree,
            f"entities:{degree}",
            ids,
        )

    regions = scope(3, np.asarray((0, 1), dtype=np.int64))
    specification = M.VolumeMeshingSpec(
        M.CellMeshingTarget(3, 3, M.CellFamilyPolicy(required=("tetrahedron",))),
        scope(2, np.asarray(topology.quotient.entities(2).entity_ids)),
        M.VolumeFillStrategy.SIMPLEX,
        # The lifecycle fixture below owns the refinement transition. Keep this
        # publication on the original six-cell source instead of pre-refining it.
        size_controls=(
            M.UniformSizeControl(
                regions, 3.0, maximum_size=3.0, strength=M.SizeControlStrength.SOFT
            ),
        ),
        protected_features=(
            M.ProtectedFeature(
                scope(1, np.asarray((0,), dtype=np.int64)), M.FeatureKind.CURVE
            ),
        ),
        region_controls=(
            M.RegionControl(
                scope(3, np.asarray((0,), dtype=np.int64)),
                "first",
                "fluid",
                M.RegionRole.FLUID,
            ),
            M.RegionControl(
                scope(3, np.asarray((1,), dtype=np.int64)),
                "second",
                "solid",
                M.RegionRole.SOLID,
            ),
        ),
    )
    result = (
        M.NativeMeshingProvider(M.NativeMeshingOptions("periodic_delaunay"))
        .plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    assert isinstance(result, M.CellMeshingResult)
    assert {zone.material_id for zone in result.zones} == {"fluid", "solid"}
    return source, specification, result


def test_periodic_exact_material_binding_preserves_wide_scientific_region_ids() -> None:
    from phydrax.meshing.providers._native_periodic_constraints import (
        bind_periodic_source_strata,
        prepare_periodic_source_strata,
    )

    source = _two_triangle_torus()
    original_regions = np.asarray((3 << 35, 7 << 40), dtype=np.int64)
    snapshot = original_regions.copy()
    strata = prepare_periodic_source_strata(source, original_regions, ())
    regions, ancestry, work = bind_periodic_source_strata(
        strata.points,
        strata.cells,
        strata.shifts,
        strata,
        256,
    )
    np.testing.assert_array_equal(regions, original_regions)
    np.testing.assert_array_equal(strata.regions, snapshot)
    assert ancestry == ((0,), (1,))
    assert 0 < work <= 256


@pytest.mark.parametrize("tolerance", (0.0, 1.0e-3))
def test_nondiscrete_exact_proper_rotation_is_not_promoted_by_tolerance(
    tolerance: float,
) -> None:
    # Two exact dyadic reflections compose to a proper rotation with
    # cos(angle)=1/8. A rational cosine other than 0, +/-1/2, +/-1
    # cannot have finite rotational order; its zero-pitch powers are nondiscrete.
    direction = np.asarray((0.75, 0.25, 0.25, 0.25, 0.5))
    reflection = np.eye(5) - 2.0 * np.outer(direction, direction)
    coordinate_reflection = np.diag((-1.0, 1.0, 1.0, 1.0, 1.0))
    generator = np.eye(6)
    generator[:5, :5] = reflection @ coordinate_reflection
    source_bits = generator.view(np.uint64).copy()
    with pytest.raises(ValueError):
        PeriodicIsometryGroup(generator[None], tolerance=tolerance)
    np.testing.assert_array_equal(generator.view(np.uint64), source_bits)


@pytest.mark.parametrize(
    ("family", "order"),
    (("H1", 4), ("Hdiv", 3), ("Hcurl", 3)),
)
def test_native_mixed_isometry_publication_keeps_original_field_source(
    family: str,
    order: int,
) -> None:
    from phydrax.meshing._periodic import PeriodicRefinement

    source_owner, specification, publication, adaptation = (
        _native_rotational_refinement_fixture(
            volume=True,
            mixed=True,
        )
    )
    assert isinstance(source_owner.domain, CellMesh)
    original_topology = _require_periodic_topology(source_owner.domain)
    original_group = original_topology.cell
    assert isinstance(original_group, PeriodicIsometryGroup)
    assert original_group.orders == (4, 0)
    # Equality binds the original generators/exponent axes, not a substituted
    # independent finite rotation plus a different translation generator.
    target_group = _require_periodic_topology(adaptation.target.mesh).cell
    assert isinstance(target_group, PeriodicIsometryGroup)
    assert target_group.group_id == original_group.group_id
    np.testing.assert_array_equal(target_group.generators, original_group.generators)
    refinement = adaptation.hierarchy
    assert isinstance(refinement, PeriodicRefinement)
    D = phx.discretization
    element = (
        D.lagrange_element("tetrahedron", order)
        if family == "H1"
        else D.form_element(
            "tetrahedron",
            2 if family == "Hdiv" else 1,
            order,
            twist="twisted" if family == "Hdiv" else "untwisted",
            proxy="flux" if family == "Hdiv" else "circulation",
        )
    )
    field = D.FiniteElementFieldSpec("u", element)
    source, target = (
        D.FiniteElementPlan(
            carrier.mesh, field, coordinate_spec=carrier.geometry
        ).prepare()
        for carrier in (publication, adaptation.target)
    )
    transfer = D.prepare_nested_field_transfer(
        source,
        target,
        refinement.parent_cells,
        field_name="u",
        parent_reference_vertices=refinement.parent_reference_vertices,
        source_geometry=publication.geometry,
        target_geometry=adaptation.target.geometry,
    )
    assert transfer.evidence.passed
    if family != "H1":
        assert transfer.evidence.defect("commuting") <= transfer.evidence.tolerance
    rng = np.random.default_rng(67)
    primal = rng.normal(size=(source.dof_maps[0].global_dof_count, 2))
    dual = rng.normal(size=(target.dof_maps[0].global_dof_count, 2))
    forward = np.asarray(transfer.transfer.apply(primal))
    pulled = np.asarray(transfer.transfer.pullback(dual))
    np.testing.assert_allclose(
        np.vdot(forward, dual), np.vdot(primal, pulled), atol=2.0e-11
    )


def test_affine_carrier_admission_retains_complete_original_p1_action() -> None:
    from phydrax.discretization._cell_geometry import (
        BarycentricCellGeometryElement,
        CellGeometrySpec,
        coordinate_lagrange_element,
    )
    from phydrax.discretization._cell_geometry_transfer import is_affine_cell_geometry

    source_coordinates = np.asarray(((1.0, 1.0), (3.0, 1.0), (1.0, 3.0)))
    action = np.asarray(((2.0, -0.5, 0.25), (0.5, 0.75, -0.25), (-0.25, 0.25, 1.5)))
    source_bits, action_bits = (
        source_coordinates.view(np.uint64).copy(),
        action.view(np.uint64).copy(),
    )
    target = CellMesh.from_triangles(
        action @ source_coordinates, np.asarray(((0, 1, 2),))
    )
    root = coordinate_lagrange_element("triangle", 1)
    element = BarycentricCellGeometryElement(root, action)
    geometry = CellGeometrySpec(
        {target.blocks[0].name: element},
        {target.blocks[0].name: np.asarray(((0, 1, 2),), dtype=np.int64)},
        source_coordinates,
    )
    assert is_affine_cell_geometry(target, geometry)
    np.testing.assert_array_equal(source_coordinates.view(np.uint64), source_bits)
    np.testing.assert_array_equal(
        np.asarray(element.barycentric_weights).view(np.uint64), action_bits
    )
    different = CellMesh.from_triangles(
        np.asarray(target.coordinates) + np.asarray((0.125, 0.0)),
        np.asarray(((0, 1, 2),)),
    )
    assert not is_affine_cell_geometry(different, geometry)


@pytest.mark.parametrize(
    "warped", (False, True), ids=("affine-cube", "genuine-q1-cross-term")
)
def test_affinity_admission_proves_tensor_map_not_degree_or_layout(warped: bool) -> None:
    from phydrax.discretization._cell_geometry import CellGeometrySpec
    from phydrax.discretization._cell_geometry_transfer import is_affine_cell_geometry
    from phydrax.discretization._reference_cell import reference_cell_topology

    coordinates = np.asarray(
        reference_cell_topology("hexahedron").vertices, dtype=np.float64
    ).copy()
    if warped:
        coordinates[6, 0] += 0.25
    mesh = CellMesh(
        coordinates,
        (CellBlock("hex", "hexahedron", np.arange(8, dtype=np.int64)[None]),),
    )
    geometry = CellGeometrySpec.affine(mesh)
    assert is_affine_cell_geometry(mesh, geometry) is (not warped)


def test_original_periodic_source_membership_covers_every_scientific_cell_block() -> None:
    from phydrax.meshing.providers._native_periodic import _periodic_source_associations

    owner, specification, publication = _native_periodic_material_fixture(
        _rotational_wedge()
    )
    source = publication.mesh
    block = source.blocks[0]
    vertices, identifiers = np.asarray(block.vertices), np.asarray(block.global_ids)
    grouped_blocks = (
        CellBlock(
            "even-cells", block.cell_kind, vertices[::2], global_ids=identifiers[::2]
        ),
        CellBlock(
            "odd-cells", block.cell_kind, vertices[1::2], global_ids=identifiers[1::2]
        ),
    )
    lifted = CellMesh(
        source.coordinates,
        grouped_blocks,
        vertex_global_ids=source.vertex_global_ids,
    )
    original = _require_periodic_topology(source)
    topology = PeriodicMeshTopology(
        lifted,
        original.cell,
        original.vertex_representatives,
        original.vertex_shifts,
    )
    target = CellMesh(
        lifted.coordinates,
        grouped_blocks,
        vertex_global_ids=lifted.vertex_global_ids,
        periodic_topology=topology,
    )
    associations = _periodic_source_associations(owner, specification, target)
    cell_association = associations[-1]
    actual_ids = np.asarray(target.entity_set(target.topological_dimension).entity_ids)
    rows = cell_association.target_rows(actual_ids)
    assert cell_association.complete and cell_association.exact
    assert np.unique(rows).size == actual_ids.size
    original_association = publication.associations[-1]
    original_rows = original_association.target_rows(actual_ids)
    np.testing.assert_array_equal(
        np.asarray(cell_association.source_indices)[rows],
        np.asarray(original_association.source_indices)[original_rows],
    )
    np.testing.assert_array_equal(target.coordinates, source.coordinates)
    np.testing.assert_array_equal(
        np.sort(np.concatenate((identifiers[::2], identifiers[1::2]))),
        np.sort(identifiers),
    )


def test_periodic_history_authenticates_retained_reference_and_sibling_banks() -> None:
    from phydrax.meshing._periodic import PeriodicRefinement

    history = refine_periodic_mesh(_two_triangle_torus())

    def renew(
        source_geometry: phx.discretization.CellGeometrySpec = history.source_geometry,
        parent_reference_vertices: np.ndarray = history.parent_reference_vertices,
        parent_cells: np.ndarray = history.parent_cells,
    ) -> PeriodicRefinement:
        return PeriodicRefinement(
            mesh=history.mesh,
            source=history.source,
            source_geometry=source_geometry,
            parent_reference_vertices=parent_reference_vertices,
            parent_cells=parent_cells,
            cell_generations=history.cell_generations,
            bisected_edges=history.bisected_edges,
            quotient=history.quotient,
            previous=history.previous,
            retired=history.retired,
        )

    renewed = renew()
    assert renewed.refinement_id == history.refinement_id
    np.testing.assert_array_equal(
        renewed.parent_reference_vertices, history.parent_reference_vertices
    )
    geometry = history.source_geometry
    changed_geometry = phx.discretization.CellGeometrySpec(
        dict(zip(geometry.block_names, geometry.elements, strict=True)),
        dict(zip(geometry.block_names, geometry.geometry_dofs, strict=True)),
        geometry.coordinates + 0.125,
    )
    with pytest.raises(ValueError, match="source geometry differs"):
        renew(source_geometry=changed_geometry)
    parents = np.asarray(history.parent_cells)
    siblings = np.flatnonzero(parents == parents[0])
    assert siblings.size >= 2
    duplicated = np.asarray(history.parent_reference_vertices).copy()
    duplicated[siblings[1]] = duplicated[siblings[0]]
    with pytest.raises(ValueError, match="duplicates.*sibling"):
        renew(parent_reference_vertices=duplicated)
    outside = np.asarray(history.parent_reference_vertices).copy()
    outside[0, 0, 0] = 2.0
    with pytest.raises(ValueError, match="leave.*parent"):
        renew(parent_reference_vertices=outside)
    missing = np.asarray(history.parent_cells).copy()
    missing[0] = sum(block.cell_count for block in history.source.blocks)
    with pytest.raises(ValueError, match="invalid retained"):
        renew(parent_cells=missing)


def test_native_periodic_history_pairs_actual_grouped_carrier_rows() -> None:
    D = phx.discretization
    _, _, publication, adaptation = _native_rotational_refinement_fixture()
    history = adaptation.hierarchy
    assert isinstance(history, phx.meshing.PeriodicRefinement)
    source_ids = np.concatenate(
        tuple(np.asarray(block.global_ids) for block in publication.mesh.blocks)
    )
    target_ids = np.concatenate(
        tuple(np.asarray(block.global_ids) for block in adaptation.target.mesh.blocks)
    )
    origin = adaptation.target.geometry.restriction_source
    assert origin is not None
    retained = np.concatenate(
        tuple(
            np.asarray(origin.block_parent_cell_ids[block.name])
            for block in adaptation.target.mesh.blocks
        )
    )
    np.testing.assert_array_equal(source_ids[history.parent_cells], retained)
    field = D.FiniteElementFieldSpec("u", D.lagrange_element("triangle", 2))
    source, target = tuple(
        D.FiniteElementPlan(
            carrier.mesh, field, coordinate_spec=carrier.geometry
        ).prepare()
        for carrier in (publication, adaptation.target)
    )
    transfer = D.prepare_nested_field_transfer(
        source,
        target,
        history.parent_cells,
        field_name="u",
        parent_reference_vertices=history.parent_reference_vertices,
        source_geometry=publication.geometry,
        target_geometry=adaptation.target.geometry,
    )
    assert transfer.evidence.passed
    # A global-SCI sort is not the actual coefficient-action block row order.
    misaligned = np.argsort(target_ids, kind="stable")
    assert not np.array_equal(source_ids[history.parent_cells[misaligned]], retained)
    with pytest.raises(ValueError, match="scientific root identities"):
        D.prepare_nested_field_transfer(
            source,
            target,
            history.parent_cells[misaligned],
            field_name="u",
            parent_reference_vertices=history.parent_reference_vertices[misaligned],
            source_geometry=publication.geometry,
            target_geometry=adaptation.target.geometry,
        )


def _grouped_periodic_domain(domain: CellMesh) -> CellMesh:
    block = domain.blocks[0]
    blocks = tuple(
        CellBlock(
            name,
            block.cell_kind,
            np.asarray(block.vertices)[rows],
            global_ids=np.asarray(block.global_ids)[rows],
        )
        for name, rows in (
            ("first-source-cohort", np.arange(0, block.cell_count, 2)),
            ("last-source-cohort", np.arange(1, block.cell_count, 2)),
        )
    )
    lifted = CellMesh(
        domain.coordinates, blocks, vertex_global_ids=domain.vertex_global_ids
    )
    periodic = _require_periodic_topology(domain).rebuilt(lifted)
    return CellMesh(
        domain.coordinates,
        blocks,
        vertex_global_ids=domain.vertex_global_ids,
        periodic_topology=periodic,
    )


@pytest.mark.parametrize(
    "domain_factory",
    (_two_triangle_torus, _rotational_wedge),
    ids=("translation", "proper-rotation"),
)
def test_grouped_periodic_refinement_selects_late_source_cohort(
    domain_factory: Callable[[], CellMesh],
) -> None:
    from phydrax.meshing._periodic import _periodic_vertex_stencils

    source = _grouped_periodic_domain(domain_factory())
    late = source.blocks[0].cell_count
    refined = refine_periodic_mesh(source, cells=np.asarray((late,), dtype=np.int64))
    assert any(block.name == "last-source-cohort" for block in refined.mesh.blocks)
    assert np.any((refined.parent_cells >= late) & (refined.cell_generations > 0))
    supports, weights = _periodic_vertex_stencils(refined)
    assert supports.shape[0] == refined.mesh.coordinates.shape[0]
    np.testing.assert_array_equal(np.sum(weights, axis=1), np.ones(weights.shape[0]))
    assert np.all(np.isin(supports[weights != 0], np.asarray(source.vertex_global_ids)))
    source_rows = np.concatenate(
        tuple(np.asarray(block.vertices) for block in source.blocks)
    )
    target_rows = np.concatenate(
        tuple(np.asarray(block.vertices) for block in refined.mesh.blocks)
    )
    assert source_rows.shape[1] == target_rows.shape[1]
    assert refined.parent_cells.shape == (target_rows.shape[0],)


@pytest.mark.parametrize(
    "domain_factory",
    (_two_triangle_torus, _rotational_wedge),
    ids=("translation", "proper-rotation"),
)
def test_grouped_periodic_native_cycle_preserves_retired_scientific_orbits(
    domain_factory: Callable[[], CellMesh],
) -> None:
    M = phx.meshing
    source = M.certify_cell_mesh(
        _grouped_periodic_domain(domain_factory()), phx.SpatialCoordinateContract.si()
    )
    policy = M.MeshAdaptationPolicy(M.MeshAdaptationRoute.NATIVE_BISECTION)
    refined = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            source,
            M.MarkedMeshAdaptation(source.mesh.entity_set(2).entity_ids),
            policy=policy,
        )
    )
    assert refined.status is M.MeshAdaptationStatus.COMPLETE
    coarsened = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            refined.target,
            M.MarkedMeshAdaptation(
                (),
                refined.target.mesh.entity_set(2).entity_ids,
                hierarchy=refined.hierarchy,
            ),
            policy=policy,
        )
    )
    assert coarsened.status is M.MeshAdaptationStatus.COMPLETE
    assert coarsened.target.mesh.topology_id == source.mesh.topology_id
    repeated = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            coarsened.target,
            M.MarkedMeshAdaptation(
                coarsened.target.mesh.entity_set(2).entity_ids,
                hierarchy=coarsened.hierarchy,
            ),
            policy=policy,
        )
    )
    assert repeated.status is M.MeshAdaptationStatus.COMPLETE
    from phydrax.meshing._periodic import PeriodicRefinement

    assert isinstance(repeated.hierarchy, PeriodicRefinement)
    assert isinstance(coarsened.hierarchy, PeriodicRefinement)
    assert repeated.hierarchy.previous is coarsened.hierarchy
    assert coarsened.hierarchy.retired is refined.hierarchy
    for degree in range(3):
        np.testing.assert_array_equal(
            _require_periodic_topology(repeated.target.mesh)
            .quotient.entities(degree)
            .entity_ids,
            _require_periodic_topology(refined.target.mesh)
            .quotient.entities(degree)
            .entity_ids,
        )


def test_owner_local_periodic_mesh_default_source_archive_preserves_actual_projection(
    tmp_path: Path,
) -> None:
    from copy import copy

    import jax

    from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
    from phydrax.lifecycle._meshing_sources import (
        _native_authority_equal,
        read_meshing_source_closure,
        validate_meshing_source_closure,
        write_meshing_source_closure,
    )

    if len(jax.devices("cpu")) < 2:
        pytest.skip("Actual owner-local source preparation requires two CPU devices.")
    M, D = phx.meshing, phx.discretization
    source_owner, specification, source = _native_periodic_material_fixture(
        _rotational_wedge()
    )
    distribution = M.prepare_mesh_distribution(
        M.MeshPart("rotational-source", source),
        policy=M.MeshPartitionPolicy(M.MeshPartitionKind.MORTON, 2, halo_width=1),
    )
    plan = D.FiniteElementPlan(
        source.mesh,
        D.FiniteElementFieldSpec("u", D.lagrange_element("triangle", 2)),
        coordinate_spec=source.geometry,
    )
    prepared = plan.prepare()
    closure = distribution.prepare_finite_element_closures(
        plan,
        limits=specification.limits,
        cell_capacity=1024,
        vertex_capacity=1024,
        message_capacity=1024,
        axis_name="parts",
        source_discretization=prepared,
    )
    local = closure.programs[0].local_discretization.mesh
    assert local.storage is not None and local.periodic_topology is not None
    receipt = write_meshing_source_closure(
        tmp_path / "actual-local-periodic-source",
        (source_owner, specification, source, local),
    )
    restored_owner, restored_specification, restored_source, restored = (
        read_meshing_source_closure(
            receipt.path,
            expected_content_id=receipt.content_id,
        )
    )
    assert restored_owner.source_id == source_owner.source_id
    assert restored_specification.specification_id == specification.specification_id
    assert restored_source.result_id == source.result_id
    assert restored.storage.storage_id == local.storage.storage_id
    assert _native_authority_equal(local, restored, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS)
    assert _native_authority_equal(
        local.periodic_topology,
        restored.periodic_topology,
        limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
    )
    missing = copy(local)
    object.__setattr__(missing, "periodic_topology", None)
    missing_discretization = copy(closure.programs[0].local_discretization)
    object.__setattr__(missing_discretization, "mesh", missing)
    missing_program = copy(closure.programs[0])
    object.__setattr__(missing_program, "local_discretization", missing_discretization)
    missing_closure = copy(closure)
    object.__setattr__(
        missing_closure, "programs", (missing_program, *closure.programs[1:])
    )
    with pytest.raises(ValueError):
        validate_meshing_source_closure(
            (source_owner, specification, source, missing_closure)
        )
    corrupted = copy(local.periodic_topology)
    object.__setattr__(
        corrupted,
        "vertex_shifts",
        corrupted.vertex_shifts.at[0, 0].add(1),
    )
    bad = copy(local)
    object.__setattr__(bad, "periodic_topology", corrupted)
    with pytest.raises(ValueError):
        validate_meshing_source_closure(bad)
