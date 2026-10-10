#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any, Literal

import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_geometry import coordinate_lagrange_element
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization.fem import (
    discontinuous_element,
    form_element,
    lagrange_element,
    prepare_l2_projection_target,
    prepare_nested_field_transfer,
    prepare_projection_field_transfer,
)
from phydrax.geometry import CommonRefinementPolicy, prepare_common_refinement


D = phx.discretization
lc = phx.lifecycle

# Nested meshes are built by hand (red refinement of triangles, one edge
# bisection of two face-sharing tetrahedra) so the parent witness is an
# independent reference, not an output of the code under test.
_SOURCE_TRIANGLE_VERTICES = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.2, 0.9), (0.1, 1.0)))
_SOURCE_TRIANGLES = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)


def _red_refinement() -> tuple[Any, Any, np.ndarray]:
    vertices = [tuple(point) for point in _SOURCE_TRIANGLE_VERTICES]
    midpoints: dict[tuple[int, int], int] = {}

    def midpoint(first: int, second: int) -> int:
        key = (min(first, second), max(first, second))
        if key not in midpoints:
            midpoints[key] = len(vertices)
            vertices.append(
                tuple(
                    0.5
                    * (
                        _SOURCE_TRIANGLE_VERTICES[first]
                        + _SOURCE_TRIANGLE_VERTICES[second]
                    )
                )
            )
        return midpoints[key]

    children, parents = [], []
    for parent, (a, b, c) in enumerate(_SOURCE_TRIANGLES.tolist()):
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        children += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        parents += [parent] * 4
    source = D.CellMesh.from_triangles(_SOURCE_TRIANGLE_VERTICES, _SOURCE_TRIANGLES)
    target = D.CellMesh.from_triangles(
        np.asarray(vertices), np.asarray(children, dtype=np.int32)
    )
    return source, target, np.asarray(parents, dtype=np.int64)


def _positive(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    corners = points[cells]
    frames = corners[:, 1:] - corners[:, :1]
    negative = np.linalg.det(frames) < 0.0
    cells = cells.copy()
    cells[negative] = cells[negative][:, [1, 0, *range(2, cells.shape[1])]]
    return cells


def _tetrahedral_bisection() -> tuple[Any, Any, np.ndarray]:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.5, 0.5, 0.0),
        )
    )
    source = _positive(points[:5], np.asarray(((0, 1, 2, 3), (1, 2, 3, 4))))
    target = _positive(
        points, np.asarray(((0, 1, 5, 3), (0, 5, 2, 3), (1, 5, 3, 4), (5, 2, 3, 4)))
    )
    return (
        D.CellMesh.from_tetrahedra(points[:5], source.astype(np.int32)),
        D.CellMesh.from_tetrahedra(points, target.astype(np.int32)),
        np.asarray((0, 0, 1, 1), dtype=np.int64),
    )


def _pair(element: Any, source: Any, target: Any) -> tuple[Any, Any]:
    spec = D.FiniteElementFieldSpec("u", element)
    return (
        D.FiniteElementPlan(source, spec).prepare(),
        D.FiniteElementPlan(target, spec).prepare(),
    )


def _epochs(source: Any, target: Any) -> tuple[Any, Any]:
    return (
        D.TopologyEpoch(
            0,
            cell_geometry_id(D.CellGeometrySpec.affine(source)),
            source.topology_id,
            "serial",
        ),
        D.TopologyEpoch(
            1,
            cell_geometry_id(D.CellGeometrySpec.affine(target)),
            target.topology_id,
            "serial",
        ),
    )


def _quadratic(points: Any, args: object) -> Any:
    del args
    x, y = points[..., 0], points[..., 1]
    return 1.5 - 0.7 * x + 2.0 * y + 0.3 * x * x - 1.1 * x * y + 0.4 * y * y


def _affine(points: Any, args: object) -> Any:
    del args
    return 0.25 + 2.0 * points[..., 0] - 3.0 * points[..., 1]


@pytest.mark.parametrize(
    ("degree", "function"),
    ((1, _affine), (2, _quadratic)),
    ids=("p1-affine", "p2-quadratic"),
)
def test_nested_h1_transfer_reproduces_constants_and_polynomials(
    degree: int, function: Any
) -> None:
    source_mesh, target_mesh, parents = _red_refinement()
    source, target = _pair(lagrange_element("triangle", degree), source_mesh, target_mesh)

    transfer = prepare_nested_field_transfer(source, target, parents, field_name="u")
    polynomial = transfer.transfer.apply(source.project("u", function))
    constant = transfer.transfer.apply(jnp.ones((source.dof_maps[0].global_dof_count,)))

    assert transfer.evidence.passed
    assert transfer.semantics == "nested-interpolation"
    assert transfer.geometry.relation == "exact-restriction"
    assert transfer.transfer.preserves_constants and transfer.transfer.conservative
    np.testing.assert_allclose(polynomial, target.project("u", function), atol=1e-13)
    np.testing.assert_allclose(constant, 1.0, atol=1e-14)


@pytest.mark.parametrize("degree", (0, 1), ids=("dg0", "dg1"))
def test_nested_discontinuous_transfer_conserves_extensive_content(degree: int) -> None:
    source_mesh, target_mesh, parents = _red_refinement()
    source, target = _pair(
        discontinuous_element("triangle", degree), source_mesh, target_mesh
    )
    transfer = prepare_nested_field_transfer(source, target, parents, field_name="u")
    source_epoch, target_epoch = _epochs(source_mesh, target_mesh)
    transition = transfer.epoch_transition(source_epoch, target_epoch)
    assert isinstance(transition, D.TopologyEpochTransition)
    count = source.dof_maps[0].global_dof_count
    values = jnp.asarray(np.random.default_rng(3).uniform(0.5, 2.0, count))
    corners = _SOURCE_TRIANGLE_VERTICES[_SOURCE_TRIANGLES]
    edges = corners[:, 1:] - corners[:, :1]
    area = 0.5 * np.sum(
        np.abs(edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0])
    )

    result = transition.apply(values)
    unit = transition.apply(jnp.ones((count,)))

    assert bool(result.successful)
    np.testing.assert_allclose(result.target_content, result.source_content, rtol=1e-13)
    np.testing.assert_allclose(unit.source_content, area, rtol=1e-13)
    np.testing.assert_allclose(unit.target_content, area, rtol=1e-13)


def _divergent_flux(points: Any, args: object) -> Any:
    del args
    return jnp.stack((0.5 + 2.0 * points[..., 0], -1.0 + 2.0 * points[..., 1]), axis=-1)


def _solenoidal_flux(points: Any, args: object) -> Any:
    del args
    return jnp.broadcast_to(jnp.asarray((1.0, -2.0)), points.shape)


def _circulating_field(points: Any, args: object) -> Any:
    del args
    return jnp.stack((0.3 - 1.5 * points[..., 1], 0.8 + 1.5 * points[..., 0]), axis=-1)


def _gradient_field(points: Any, args: object) -> Any:
    del args
    return jnp.broadcast_to(jnp.asarray((2.0, -0.5)), points.shape)


@pytest.mark.parametrize(
    ("element", "function", "semantics"),
    (
        (
            form_element("triangle", 1, 1, twist="twisted", proxy="flux"),
            _divergent_flux,
            "contravariant-piola",
        ),
        (
            form_element("triangle", 1, 1, twist="twisted", proxy="flux"),
            _solenoidal_flux,
            "contravariant-piola",
        ),
        (
            form_element("triangle", 1, 1, proxy="circulation"),
            _circulating_field,
            "covariant-piola",
        ),
        (
            form_element("triangle", 1, 1, proxy="circulation"),
            _gradient_field,
            "covariant-piola",
        ),
    ),
    ids=("rt0-flux", "rt0-divergence-free", "nedelec-circulation", "nedelec-curl-free"),
)
def test_planar_compatible_transfer_preserves_moments_with_commuting_evidence(
    element: Any, function: Any, semantics: str
) -> None:
    source_mesh, target_mesh, parents = _red_refinement()
    source, target = _pair(element, source_mesh, target_mesh)

    transfer = prepare_nested_field_transfer(source, target, parents, field_name="u")
    moments = transfer.transfer.apply(source.project("u", function))

    assert transfer.semantics == semantics
    assert transfer.evidence.passed
    assert transfer.evidence.defect("commuting") <= transfer.evidence.tolerance
    # Midpoint edge moments of the (linear) field on the target are the exact
    # flux/circulation functionals of the target space.
    np.testing.assert_allclose(moments, target.project("u", function), atol=1e-13)


@pytest.mark.parametrize(
    "element",
    (
        form_element("tetrahedron", 2, 1, twist="twisted", proxy="flux"),
        form_element("tetrahedron", 2, 1, family="full", twist="twisted", proxy="flux"),
        form_element("tetrahedron", 2, 2, family="full", twist="twisted", proxy="flux"),
    ),
    ids=("rt0", "bdm1", "bdm2"),
)
def test_tetrahedral_hdiv_transfer_certifies_nesting_and_divergence_commutation(
    element: Any,
) -> None:
    source_mesh, target_mesh, parents = _tetrahedral_bisection()
    source, target = _pair(element, source_mesh, target_mesh)

    transfer = prepare_nested_field_transfer(source, target, parents, field_name="u")

    assert transfer.semantics == "contravariant-piola"
    assert transfer.evidence.passed
    for name in ("containment", "continuity", "reproduction", "commuting"):
        assert transfer.evidence.defect(name) <= transfer.evidence.tolerance


def _edge_integrals(mesh: Any, constant: np.ndarray, rotation: np.ndarray) -> Any:
    """Exact lower-to-higher edge circulations of ``constant + rotation x x``."""

    coordinates = np.asarray(mesh.coordinates)
    edges = np.asarray(mesh.connectivity.edges)
    start, end = coordinates[edges[:, 0]], coordinates[edges[:, 1]]
    values = constant + np.cross(rotation, 0.5 * (start + end))
    return np.sum(values * (end - start), axis=1)


def test_tetrahedral_nedelec_transfer_preserves_circulation_and_curl_free_fields() -> (
    None
):
    source_mesh, target_mesh, parents = _tetrahedral_bisection()
    source, target = _pair(
        form_element("tetrahedron", 1, 1, proxy="circulation"), source_mesh, target_mesh
    )
    constant, rotation = np.asarray((0.2, -1.0, 0.4)), np.asarray((1.1, -0.3, 0.6))
    potential = np.asarray(source_mesh.coordinates) @ np.asarray((1.0, 2.0, -1.0))
    edges = np.asarray(source_mesh.connectivity.edges)

    transfer = prepare_nested_field_transfer(source, target, parents, field_name="u")
    circulation = transfer.transfer.apply(
        jnp.asarray(_edge_integrals(source_mesh, constant, rotation))
    )
    gradient = transfer.transfer.apply(
        jnp.asarray(potential[edges[:, 1]] - potential[edges[:, 0]])
    )

    assert transfer.semantics == "covariant-piola" and transfer.evidence.passed
    for name in ("containment", "continuity", "reproduction", "commuting"):
        assert transfer.evidence.defect(name) <= transfer.evidence.tolerance
    np.testing.assert_allclose(
        circulation, _edge_integrals(target_mesh, constant, rotation), atol=1e-13
    )
    # The independent topological incidence of the edge space sees no curl.
    np.testing.assert_allclose(
        D.FiniteElementDeRhamComplex(
            target_mesh, family="trimmed", order=1
        ).exterior_derivative(1, gradient),
        0.0,
        atol=1e-13,
    )


def test_incompatible_field_family_is_refused() -> None:
    source_mesh, target_mesh, parents = _red_refinement()
    source = D.FiniteElementPlan(
        source_mesh, D.FiniteElementFieldSpec("u", lagrange_element("triangle", 1))
    ).prepare()
    target = D.FiniteElementPlan(
        target_mesh,
        D.FiniteElementFieldSpec(
            "u", form_element("triangle", 1, 1, twist="twisted", proxy="flux")
        ),
    ).prepare()

    with pytest.raises(ValueError):
        prepare_nested_field_transfer(source, target, parents, field_name="u")


def test_piola_semantics_refuse_mismatched_field_conformity() -> None:
    source_mesh, target_mesh, parents = _red_refinement()
    source, target = _pair(lagrange_element("triangle", 1), source_mesh, target_mesh)
    transfer = prepare_nested_field_transfer(source, target, parents, field_name="u")

    with pytest.raises(ValueError):
        D.FieldTransfer(
            source.field_spaces[0],
            target.field_spaces[0],
            transfer.transfer.primal,
            geometry=transfer.geometry,
            properties=D.TransferProperties(semantics="covariant-piola"),
        )


def test_rejected_transfer_leaves_the_accepted_composition_untouched() -> None:
    source_mesh, target_mesh, parents = _red_refinement()
    source, target = _pair(
        form_element("triangle", 1, 1, proxy="circulation"), source_mesh, target_mesh
    )
    # A wrong witness: every child claims the first parent.
    wrong = prepare_nested_field_transfer(
        source, target, np.zeros_like(parents), field_name="u"
    )
    source_epoch, target_epoch = _epochs(source_mesh, target_mesh)
    values = source.project("u", _circulating_field)
    transition = wrong.epoch_transition(source_epoch, target_epoch)

    def entry(value: Any, epoch: Any) -> Any:
        return lc.CompositionEntry(
            jnp.asarray(value),
            entry_id="field/u",
            role="physical-state",
            owner_id="test",
            structure_id=epoch.epoch_id,
            revision_id=f"field-u-{epoch.index}",
            semantics_id="edge-circulation",
        )

    composition = lc.Composition((entry(values, source_epoch),), boundary_id="accepted")
    receipt = lc.commit_composition_rebind(
        lc.CompositionRebind(
            composition,
            transports=(
                transition.composition_transport(
                    composition.entry("field/u"),
                    entry(transition.apply(values).values, target_epoch),
                ),
            ),
        ),
        accepted_boundary=True,
    )

    assert not wrong.evidence.passed
    assert wrong.evidence.defect("containment") > wrong.evidence.tolerance
    assert wrong.geometry.relation == "bounded-reconstruction"
    assert not receipt.published
    assert receipt.composition is composition


# Non-nested remeshing: jittered interior vertices make the target cells cross
# the source cells, while the grid plane x = 1/2 stays a union of faces of both
# meshes so plane fluxes are sums of oriented face moments.
def _jittered(points: np.ndarray, count: int, jitter: float, seed: int) -> np.ndarray:
    interior = np.all((points > 1e-12) & (points < 1.0 - 1e-12), axis=1)
    shift = np.random.default_rng(seed).uniform(-1.0, 1.0, points.shape)
    shift[np.isclose(points[:, 0], 0.5), 0] = 0.0
    return points + np.where(interior[:, None], jitter / count * shift, 0.0)


def _kuhn_cube(count: int, jitter: float = 0.0) -> Any:
    axis = np.linspace(0.0, 1.0, count + 1)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        (-1, 3)
    )
    cells = []
    for corner in np.ndindex(count, count, count):
        for order in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
            vertex = np.asarray(corner)
            path = [vertex.copy()]
            for axis_index in order:
                vertex[axis_index] += 1
                path.append(vertex.copy())
            cells.append([np.ravel_multi_index(tuple(v), (count + 1,) * 3) for v in path])
    moved = _jittered(points, count, jitter, 7)
    tetrahedra = _positive(moved, np.asarray(cells, dtype=np.int32))
    return D.CellMesh.from_tetrahedra(moved, tetrahedra.astype(np.int32))


def _square(count: int, jitter: float = 0.0) -> Any:
    axis = np.linspace(0.0, 1.0, count + 1)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))
    cells = []
    for i, j in np.ndindex(count, count):
        a, b = i * (count + 1) + j, (i + 1) * (count + 1) + j
        cells.extend(((a, b, b + 1), (a, b + 1, a + 1)))
    moved = _jittered(points, count, jitter, 11)
    triangles = _positive(moved, np.asarray(cells, dtype=np.int32))
    return D.CellMesh.from_triangles(moved, triangles.astype(np.int32))


def _projection(element: Any, source_mesh: Any, target_mesh: Any) -> tuple[Any, Any]:
    source, target = _pair(element, source_mesh, target_mesh)
    refinement = prepare_common_refinement(
        source_mesh,
        target_mesh,
        policy=CommonRefinementPolicy(overlap_simplices=True),
    )
    return target, prepare_projection_field_transfer(
        source,
        prepare_l2_projection_target(target, field_name="u"),
        refinement,
        field_name="u",
    )


def _face_fluxes(mesh: Any, field: Any) -> np.ndarray:
    """Face fluxes along the right-hand normal of the sorted face vertices.

    The edge-midpoint rule integrates quadratic fields exactly on triangles.
    """

    corners = np.asarray(mesh.coordinates)[np.asarray(mesh.connectivity.faces)]
    area_normals = 0.5 * np.cross(
        corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    )
    midpoints = 0.5 * (corners + np.roll(corners, -1, axis=1))
    return np.sum(np.mean(field(midpoints), axis=1) * area_normals, axis=-1)


def _plane_flux(mesh: Any, fluxes: Any) -> float:
    """Flux in +x through the faces of the plane x = 1/2."""

    corners = np.asarray(mesh.coordinates)[np.asarray(mesh.connectivity.faces)]
    plane = np.all(np.isclose(corners[..., 0], 0.5), axis=1)
    direction = np.sign(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])[:, 0]
    )
    return float(np.sum((direction * np.asarray(fluxes))[plane]))


def _rt0_field(points: np.ndarray) -> np.ndarray:
    return np.asarray((0.3, -1.1, 0.7)) + 1.6 * points


def _solenoidal_field(points: np.ndarray) -> np.ndarray:
    # Divergence-free with total flux 4/3 through every plane x = constant.
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return np.stack((1.0 + y * y, z * x, y), axis=-1)


def test_nonnested_hdiv_projection_reproduces_rt0_fields_and_plane_flux() -> None:
    source_mesh, target_mesh = _kuhn_cube(2), _kuhn_cube(4, 0.3)
    _, transfer = _projection(
        form_element("tetrahedron", 2, 1, twist="twisted", proxy="flux"),
        source_mesh,
        target_mesh,
    )
    evidence = transfer.evidence

    reproduced = transfer.transfer.apply(
        jnp.asarray(_face_fluxes(source_mesh, _rt0_field))
    )
    solenoidal = _face_fluxes(source_mesh, _solenoidal_field)
    projected = transfer.transfer.apply(jnp.asarray(solenoidal))

    assert transfer.semantics == "contravariant-piola" and evidence.passed
    assert transfer.geometry.relation == "bounded-reconstruction"
    for name in (
        "coverage",
        "reproduction",
        "content",
        "commuting",
        "divergence-content",
    ):
        assert evidence.defect(name) <= evidence.tolerance
    assert evidence.estimate("solve-residual") <= 1e-10
    np.testing.assert_allclose(
        reproduced, _face_fluxes(target_mesh, _rt0_field), atol=1e-12
    )
    np.testing.assert_allclose(
        _plane_flux(source_mesh, solenoidal), 4.0 / 3.0, rtol=1e-13
    )
    # The interior plane flux of the projected field matches the source flux to
    # the projection's own discretization accuracy.
    np.testing.assert_allclose(_plane_flux(target_mesh, projected), 4.0 / 3.0, rtol=2e-2)


def test_nonnested_hcurl_projection_reproduces_nedelec_fields() -> None:
    source_mesh, target_mesh = _kuhn_cube(2), _kuhn_cube(3, 0.3)
    _, transfer = _projection(
        form_element("tetrahedron", 1, 1, proxy="circulation"), source_mesh, target_mesh
    )
    constant, rotation = np.asarray((0.5, -0.2, 1.3)), np.asarray((-0.7, 0.9, 0.4))

    circulation = transfer.transfer.apply(
        jnp.asarray(_edge_integrals(source_mesh, constant, rotation))
    )

    assert transfer.evidence.passed
    assert transfer.evidence.defect("reproduction") <= transfer.evidence.tolerance
    assert transfer.evidence.defect("curl-content") <= transfer.evidence.tolerance
    assert transfer.evidence.defect("commuting") <= transfer.evidence.tolerance
    assert transfer.evidence.estimate("cellwise-curl-realizability") > 1e-3
    np.testing.assert_allclose(
        circulation, _edge_integrals(target_mesh, constant, rotation), atol=1e-12
    )


@pytest.mark.parametrize(
    ("element", "function"),
    (
        (form_element("triangle", 1, 1, twist="twisted", proxy="flux"), _divergent_flux),
        (form_element("triangle", 1, 1, proxy="circulation"), _circulating_field),
    ),
    ids=("rt0", "nedelec"),
)
def test_nonnested_planar_compatible_projection_reproduces_lowest_order_fields(
    element: Any, function: Any
) -> None:
    source_mesh, target_mesh = _square(2), _square(5, 0.3)
    source = D.FiniteElementPlan(source_mesh, D.FiniteElementFieldSpec("u", element))
    target, transfer = _projection(element, source_mesh, target_mesh)

    moments = transfer.transfer.apply(source.prepare().project("u", function))

    assert transfer.evidence.passed
    np.testing.assert_allclose(moments, target.project("u", function), atol=1e-12)


def test_nonnested_projection_refuses_mismatched_piola_maps() -> None:
    source_mesh, target_mesh = _square(2), _square(3, 0.3)
    source, _ = _pair(
        form_element("triangle", 1, 1, twist="twisted", proxy="flux"),
        source_mesh,
        target_mesh,
    )
    _, target = _pair(
        form_element("triangle", 1, 1, proxy="circulation"), source_mesh, target_mesh
    )

    with pytest.raises(ValueError):
        prepare_projection_field_transfer(
            source,
            prepare_l2_projection_target(target, field_name="u"),
            prepare_common_refinement(
                source_mesh,
                target_mesh,
                policy=CommonRefinementPolicy(overlap_simplices=True),
            ),
            field_name="u",
        )


def _bisect(
    source: Any, refine: Any = (), coarsen: Any = (), hierarchy: Any = None
) -> Any:
    return phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MarkedMeshAdaptation(
                np.asarray(refine, dtype=np.int64),
                np.asarray(coarsen, dtype=np.int64),
                hierarchy=hierarchy,
            ),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
            ),
        )
    )


def test_topology_transaction_projects_compatible_fields_on_nonnested_adaptation() -> (
    None
):
    mesh = D.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5))),
        np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=np.int32),
        vertex_global_ids=np.asarray((1, 2, 3, 4, 5)),
        cell_global_ids=np.asarray((10, 20, 30, 40)),
    )
    refined = _bisect(
        phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si()), (10,)
    )
    children = np.setdiff1d(
        np.concatenate(
            [np.asarray(block.global_ids) for block in refined.target.mesh.blocks]
        ),
        np.asarray(mesh.blocks[0].global_ids),
    )
    coarsened = _bisect(refined.target, coarsen=children, hierarchy=refined.hierarchy)
    fields = D.FiniteElementFieldSpec(
        "flux", form_element("triangle", 1, 1, twist="twisted", proxy="flux")
    )
    fine = D.FiniteElementPlan(refined.target.mesh, fields).prepare()
    accepted = phx.solver.FiniteElementAcceptedState(
        (fine.project("flux", _divergent_flux),),
        0.0,
        0,
        refined.target.mesh.topology_id,
        "prepared",
        "compiled",
    )
    transaction = phx.solver.FiniteElementTopologyTransaction(
        lambda candidate_mesh, values, materials, lineage, args: True, fields=fields
    )

    result = transaction.execute(accepted, refined.target.mesh, coarsened)
    coarse = D.FiniteElementPlan(result.mesh, fields).prepare()

    assert phx.solver.refinement_parent_cells(coarsened) is None
    assert bool(result.committed) and result.receipt is not None
    assert result.transfers[0].evidence.passed
    # RT0 contains the flux a + 2x, so coarsening by projection is lossless.
    np.testing.assert_allclose(
        result.state.fields[0], coarse.project("flux", _divergent_flux), atol=1e-12
    )


def _opposite_cube() -> Any:
    source = _kuhn_cube(1)
    coordinates = np.asarray(source.coordinates).copy()
    coordinates[:, 0] = 1.0 - coordinates[:, 0]
    cells = _positive(coordinates, np.asarray(source.blocks[0].vertices).copy())
    return D.CellMesh.from_tetrahedra(coordinates, cells)


def _vector_geometry(
    discretization: Any, reference: np.ndarray
) -> tuple[np.ndarray, ...]:
    rule = phx.integration.ReferenceTetrahedronRule(
        phx.integration.GaussLegendreRule(4)
    ).materialize()
    geometry = discretization.evaluate_block_geometry(
        "u",
        0,
        discretization.default_runtime.coordinates,
        reference,
        np.ones(reference.shape[0]),
    )
    dof_map = discretization.dof_maps[0]
    transforms = np.asarray(dof_map.cell_transforms[0])
    basis = np.einsum("cqav,cai->cqiv", np.asarray(geometry.basis_values), transforms)
    gradients = np.einsum(
        "cqavd,cai->cqivd", np.asarray(geometry.physical_gradients), transforms
    )
    element = discretization.elements[0][0]
    if element.mapping == "contravariant_piola":
        derivative = np.trace(gradients, axis1=-2, axis2=-1)[..., None]
    else:
        derivative = np.stack(
            (
                gradients[..., 2, 1] - gradients[..., 1, 2],
                gradients[..., 0, 2] - gradients[..., 2, 0],
                gradients[..., 1, 0] - gradients[..., 0, 1],
            ),
            axis=-1,
        )
    return (
        np.asarray(dof_map.cell_dofs[0]),
        basis,
        derivative,
        np.asarray(rule.points),
        np.asarray(rule.weights),
    )


def _overlap_values(
    source: Any,
    target: Any,
    refinement: Any,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rule = phx.integration.ReferenceTetrahedronRule(
        phx.integration.GaussLegendreRule(4)
    ).materialize()
    simplices = np.asarray(refinement.simplices)
    entries = np.repeat(
        np.arange(len(refinement.source_cells)), np.diff(refinement.simplex_offsets)
    )
    source_cells = np.asarray(refinement.source_cells)[entries]
    target_cells = np.asarray(refinement.target_cells)[entries]
    frames = np.swapaxes(simplices[:, 1:] - simplices[:, :1], -1, -2)
    physical = simplices[:, :1] + np.asarray(rule.points)[None] @ np.swapaxes(
        frames, -1, -2
    )
    weights = np.abs(np.linalg.det(frames))[:, None] * np.asarray(rule.weights)[None]
    source_corners = np.asarray(source.mesh.coordinates)[
        np.asarray(source.mesh.blocks[0].vertices)
    ]
    source_frames = np.swapaxes(source_corners[:, 1:] - source_corners[:, :1], -1, -2)
    reference = np.linalg.solve(
        source_frames[source_cells, None],
        (physical - source_corners[source_cells, None, 0])[..., None],
    )[..., 0]
    routes, _, derivatives, _, _ = _vector_geometry(source, reference.reshape((-1, 3)))
    points = np.arange(reference.shape[0] * reference.shape[1]).reshape(
        reference.shape[:2]
    )
    derivative = derivatives[source_cells[:, None], points]
    values = np.asarray(
        phx.ein.contract("sqnw,sn->sqw", derivative, coefficients[routes[source_cells]])
    )
    target_corners = np.asarray(target.mesh.coordinates)[
        np.asarray(target.mesh.blocks[0].vertices)
    ]
    target_frames = np.swapaxes(target_corners[:, 1:] - target_corners[:, :1], -1, -2)
    target_reference = np.linalg.solve(
        target_frames[target_cells, None],
        (physical - target_corners[target_cells, None, 0])[..., None],
    )[..., 0]
    return target_cells, target_reference, weights, values


@pytest.mark.parametrize(
    "element",
    (
        form_element("tetrahedron", 2, 1, twist="twisted", proxy="flux"),
        form_element("tetrahedron", 2, 1, family="full", twist="twisted", proxy="flux"),
        form_element("tetrahedron", 2, 2, family="full", twist="twisted", proxy="flux"),
    ),
    ids=("rt0", "bdm1", "bdm2"),
)
def test_nonnested_divergence_commutes_for_arbitrary_moments(element: Any) -> None:
    source, target = _pair(element, _kuhn_cube(1), _opposite_cube())
    refinement = prepare_common_refinement(
        source.mesh, target.mesh, policy=CommonRefinementPolicy(overlap_simplices=True)
    )
    transfer = prepare_projection_field_transfer(
        source,
        prepare_l2_projection_target(target, field_name="u"),
        refinement,
        field_name="u",
    )
    values = np.random.default_rng(51).normal(size=source.dof_maps[0].global_dof_count)
    transferred = np.asarray(transfer.transfer.apply(values))
    rule = phx.integration.ReferenceTetrahedronRule(
        phx.integration.GaussLegendreRule(4)
    ).materialize()
    routes, _, derivatives, reference, reference_weights = _vector_geometry(
        target, np.asarray(rule.points)
    )
    corners = np.asarray(target.mesh.coordinates)[
        np.asarray(target.mesh.blocks[0].vertices)
    ]
    measures = np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1]))
    degree = element.degree - 1
    tests = (
        np.ones((reference.shape[0], 1))
        if degree == 0
        else np.column_stack((np.ones(reference.shape[0]), reference))
    )
    divergence = np.asarray(
        phx.ein.contract("cqnw,cn->cqw", derivatives, transferred[routes])
    )
    actual = np.asarray(
        phx.ein.contract(
            "cq,qp,cqw->cpw",
            measures[:, None] * reference_weights[None],
            tests,
            divergence,
        )
    )
    cells, points, weights, source_divergence = _overlap_values(
        source, target, refinement, values
    )
    overlap_tests = (
        np.ones(points.shape[:2] + (1,))
        if degree == 0
        else np.concatenate((np.ones(points.shape[:2] + (1,)), points), axis=-1)
    )
    expected = np.zeros_like(actual)
    np.add.at(
        expected,
        cells,
        np.asarray(
            phx.ein.contract("sq,sqp,sqw->spw", weights, overlap_tests, source_divergence)
        ),
    )
    # Independently integrate every target polynomial divergence moment,
    # including the linear moments needed by BDM2, for an arbitrary source.
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("degree", (0, 1))
def test_nonnested_curl_commutes_with_its_realizable_companion(degree: int) -> None:
    source, target = _pair(
        form_element("tetrahedron", 1, degree + 1, proxy="circulation"),
        _kuhn_cube(1),
        _opposite_cube(),
    )
    refinement = prepare_common_refinement(
        source.mesh, target.mesh, policy=CommonRefinementPolicy(overlap_simplices=True)
    )
    prepared = prepare_l2_projection_target(target, field_name="u")
    transfer = prepare_projection_field_transfer(
        source, prepared, refinement, field_name="u"
    )
    values = np.random.default_rng(32).normal(size=source.dof_maps[0].global_dof_count)
    transferred = np.asarray(transfer.transfer.apply(values))
    rule = phx.integration.ReferenceTetrahedronRule(
        phx.integration.GaussLegendreRule(4)
    ).materialize()
    routes, _, curls, _, reference_weights = _vector_geometry(
        target, np.asarray(rule.points)
    )
    corners = np.asarray(target.mesh.coordinates)[
        np.asarray(target.mesh.blocks[0].vertices)
    ]
    measures = np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1]))
    weights = measures[:, None] * reference_weights[None]
    local = np.asarray(phx.ein.contract("cq,cqiw,cqjw->cij", weights, curls, curls))
    size = target.dof_maps[0].global_dof_count
    gram = np.zeros((size, size))
    np.add.at(gram, (routes[:, :, None], routes[:, None, :]), local)
    cells, points, overlap_weights, source_curl = _overlap_values(
        source, target, refinement, values
    )
    _, _, overlap_curls, _, _ = _vector_geometry(target, points.reshape((-1, 3)))
    point_indices = np.arange(points.shape[0] * points.shape[1]).reshape(points.shape[:2])
    overlap_curls = overlap_curls[cells[:, None], point_indices]
    load = np.zeros((size,))
    np.add.at(
        load,
        routes[cells],
        np.asarray(
            phx.ein.contract(
                "sq,sqnw,sqw->sn", overlap_weights, overlap_curls, source_curl
            )
        ),
    )
    # Independent curl-space Galerkin projection: nullspace coefficients do not
    # matter, while the resulting physical curl is unique.
    companion = np.linalg.lstsq(gram, load, rcond=1e-12)[0]
    actual = np.asarray(phx.ein.contract("cqiw,ci->cqw", curls, transferred[routes]))
    expected = np.asarray(phx.ein.contract("cqiw,ci->cqw", curls, companion[routes]))
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(
        np.asarray(phx.ein.contract("cq,cqw->w", weights, actual)),
        np.asarray(phx.ein.contract("sq,sqw->w", overlap_weights, source_curl)),
        atol=2e-11,
    )
    dual = np.random.default_rng(9).normal(size=size)
    np.testing.assert_allclose(
        dual @ transferred,
        values @ np.asarray(transfer.transfer.pullback(dual)),
        atol=1e-11,
    )


def test_projection_refuses_different_hdiv_families_with_the_same_mapping() -> None:
    source, _ = _pair(
        form_element("tetrahedron", 2, 1, twist="twisted", proxy="flux"),
        _kuhn_cube(1),
        _opposite_cube(),
    )
    _, target = _pair(
        form_element("tetrahedron", 2, 1, family="full", twist="twisted", proxy="flux"),
        _kuhn_cube(1),
        _opposite_cube(),
    )
    refinement = prepare_common_refinement(
        source.mesh, target.mesh, policy=CommonRefinementPolicy(overlap_simplices=True)
    )
    with pytest.raises(ValueError):
        prepare_projection_field_transfer(
            source,
            prepare_l2_projection_target(target, field_name="u"),
            refinement,
            field_name="u",
        )


def test_nonnested_piola_projection_refuses_curved_coordinate_maps() -> None:
    mesh = D.CellMesh.from_tetrahedra(
        np.concatenate((np.zeros((1, 3)), np.eye(3))),
        np.asarray(((0, 1, 2, 3),), dtype=np.int32),
    )
    coordinate_element = coordinate_lagrange_element("tetrahedron", 2)
    nodes = np.asarray(coordinate_element.reference_nodes).copy()
    nodes[:, 2] += 0.05 * nodes[:, 0] * (1.0 - nodes[:, 0])
    curved = D.FiniteElementPlan(
        mesh,
        D.FiniteElementFieldSpec(
            "u", form_element("tetrahedron", 1, 2, proxy="circulation")
        ),
        coordinate_spec=D.CellGeometrySpec(
            {"tetrahedra": coordinate_element},
            {"tetrahedra": np.arange(nodes.shape[0], dtype=np.int32)[None]},
            nodes,
        ),
    ).prepare()
    # A common refinement of affine corners cannot authorize dropping a
    # non-affine coordinate map in a compatible field remap.
    with pytest.raises(ValueError):
        prepare_l2_projection_target(curved, field_name="u")


@pytest.mark.parametrize("mapped", (False, True), ids=("affine", "curved"))
@pytest.mark.parametrize("compatible", (False, True), ids=("scalar", "hcurl"))
def test_epoch_admission_rejects_same_topology_wrong_geometry_and_precision(
    mapped: bool, compatible: bool
) -> None:
    mesh = D.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    if mapped:
        coordinate = coordinate_lagrange_element("triangle", 2)
        values = np.asarray(coordinate.reference_nodes).copy()
        values[:, 0] += 0.125 * values[:, 0] * values[:, 1]
        geometry = D.CellGeometrySpec(
            {"triangles": coordinate},
            {"triangles": np.arange(values.shape[0], dtype=np.int32)[None, :]},
            values,
        )
    else:
        geometry = D.CellGeometrySpec.affine(mesh)
    element = (
        form_element("triangle", 1, 1, proxy="circulation")
        if compatible
        else discontinuous_element("triangle", 0)
    )
    prepared = D.FiniteElementPlan(
        mesh, D.FiniteElementFieldSpec("u", element), coordinate_spec=geometry
    ).prepare()
    transfer = prepare_nested_field_transfer(
        prepared,
        prepared,
        np.asarray((0,), dtype=np.int32),
        field_name="u",
        parent_reference_vertices=np.asarray((((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),)),
        source_geometry=geometry,
        target_geometry=geometry,
    )
    source_epoch = D.TopologyEpoch(
        0, cell_geometry_id(geometry), mesh.topology_id, "serial"
    )
    target_epoch = D.TopologyEpoch(
        1, cell_geometry_id(geometry), mesh.topology_id, "serial"
    )
    accepted = np.linspace(0.2, 1.1, prepared.field_spaces[0].vector_space.size)
    untouched = accepted.copy()
    transition = transfer.epoch_transition(source_epoch, target_epoch)
    result = transition.apply(accepted)
    assert bool(result.successful)
    np.testing.assert_allclose(result.values, accepted, atol=1e-12)
    for wrong_source in (False, True):
        old = (
            D.TopologyEpoch(0, "stale-coordinate-map", mesh.topology_id, "serial")
            if wrong_source
            else source_epoch
        )
        new = (
            target_epoch
            if wrong_source
            else D.TopologyEpoch(1, "stale-coordinate-map", mesh.topology_id, "serial")
        )
        with pytest.raises(ValueError):
            transfer.epoch_transition(old, new)
        with pytest.raises(ValueError):
            if isinstance(transition, D.TopologyEpochTransition):
                D.TopologyEpochTransition(
                    old,
                    new,
                    transition.transfer,
                    transition.source_measures,
                    transition.target_measures,
                    measure_defect_bound=transition.measure_defect_bound,
                )
            else:
                D.FieldEpochTransition(
                    old,
                    new,
                    transition.transfer,
                    evidence_passed=True,
                    evidence_id=transfer.evidence.evidence_id,
                )
    with pytest.raises(TypeError):
        transition.apply(accepted.astype(np.float32))
    np.testing.assert_array_equal(accepted, untouched)


def test_embedded_twisted_flux_transfer_preserves_moments_across_reversed_charts() -> (
    None
):
    from phydrax.discretization.fem._surface_chart_compatible import (
        prepare_surface_chart_compatible_transfer,
    )
    from tests.unit.discretization.test_mapped_nested_fv import (
        _curved_surface_chart_fv_fixture,
    )

    _, _, chart, _ = _curved_surface_chart_fv_fixture()
    element = form_element("triangle", 1, 1, twist="twisted", proxy="flux")
    field = D.FiniteElementFieldSpec("flux", element)
    source = D.FiniteElementPlan(
        chart.source_mesh,
        field,
        coordinate_spec=chart.source_geometry,
    ).prepare()
    target = D.FiniteElementPlan(
        chart.target_mesh,
        field,
        coordinate_spec=chart.target_geometry,
    ).prepare()
    # Witness banks are sorted by scientific cell ID, not mesh/DOF row order.
    source_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in source.mesh.blocks]
    )
    target_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in target.mesh.blocks]
    )
    source_order = np.searchsorted(
        np.asarray(chart.source_witness.cell_global_ids), source_ids
    )
    target_order = np.searchsorted(
        np.asarray(chart.target_witness.cell_global_ids), target_ids
    )
    source_charts = np.asarray(chart.source_witness.charts)[source_order]
    target_charts = np.asarray(chart.target_witness.charts)[target_order]
    source_determinants = np.linalg.det(source_charts[:, 1:] - source_charts[:, :1])
    target_determinants = np.linalg.det(target_charts[:, 1:] - target_charts[:, :1])
    assert np.all(source_determinants < 0.0)
    assert np.all(target_determinants > 0.0)

    # Exact moments of the material RT0 vector a + b * UV. A twisted flux
    # pulls back with |det A|, not det A, even when the local chart reverses.
    constant, slope = np.asarray((0.3, -0.8)), 0.7
    vertices = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    edges = np.asarray(((0, 1), (0, 2), (1, 2)))
    tangents = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    normals = np.column_stack((tangents[:, 1], -tangents[:, 0]))
    midpoints = 0.5 * (vertices[edges[:, 0]] + vertices[edges[:, 1]])

    def moments(discretization: Any, charts: np.ndarray) -> np.ndarray:
        dofs = discretization.dof_maps[0]
        coefficients = np.full(dofs.global_dof_count, np.nan)
        for cell, corners in enumerate(charts):
            frame = (corners[1:] - corners[:1]).T
            material = constant + slope * (corners[0] + midpoints @ frame.T)
            reference = abs(np.linalg.det(frame)) * np.linalg.solve(frame, material.T).T
            local = np.sum(reference * normals, axis=-1)
            canonical = np.linalg.solve(np.asarray(dofs.cell_transforms[0])[cell], local)
            routes = np.asarray(dofs.cell_dofs[0])[cell]
            shared = np.isfinite(coefficients[routes])
            np.testing.assert_allclose(
                coefficients[routes][shared], canonical[shared], atol=1e-14
            )
            coefficients[routes] = canonical
        return coefficients

    prepared = prepare_surface_chart_compatible_transfer(
        source,
        target,
        chart,
        field_name="flux",
        source_geometry=chart.source_geometry,
        target_geometry=chart.target_geometry,
    )
    values = moments(source, source_charts)
    transferred = prepared.field_transfer.transfer.apply(jnp.asarray(values))
    assert prepared.field_transfer.evidence.passed
    np.testing.assert_allclose(transferred, moments(target, target_charts), atol=2e-14)
    source_content = prepared.source_differential.mv(values)
    target_content = prepared.target_differential.mv(transferred)
    # div(a + b * UV) = 2b and chart area = |det A| / 2: the
    # integrated material divergence remains positive under chart reversal.
    np.testing.assert_allclose(
        source_content, slope * np.abs(source_determinants), atol=2e-14
    )
    np.testing.assert_allclose(target_content, slope * target_determinants, atol=2e-14)
    np.testing.assert_allclose(
        target_content,
        prepared.differential_companion.mv(source_content),
        atol=2e-14,
    )
    target_load = jnp.arange(transferred.size, dtype=jnp.float64) + 0.25
    np.testing.assert_allclose(
        jnp.vdot(target_load, transferred),
        jnp.vdot(prepared.field_transfer.transfer.pullback(target_load), values),
        atol=2e-14,
    )


@pytest.mark.parametrize(
    "proxy,twist", [("circulation", "untwisted"), ("flux", "twisted")]
)
def test_order_three_material_transfer_preserves_independent_moments_and_companion(
    proxy: Literal["circulation", "flux"],
    twist: Literal["untwisted", "twisted"],
) -> None:
    from phydrax.discretization.fem._surface_chart_compatible import (
        prepare_surface_chart_compatible_transfer,
    )
    from tests.unit.discretization.test_mapped_nested_fv import (
        _curved_surface_chart_fv_fixture,
    )

    _, _, chart, _ = _curved_surface_chart_fv_fixture()
    element = form_element("triangle", 1, 3, family="trimmed", twist=twist, proxy=proxy)
    basis = element.form_basis
    if basis is None:
        raise TypeError("The actual order-three field must retain its owning form basis.")
    field = D.FiniteElementFieldSpec("material", element)
    source = D.FiniteElementPlan(
        chart.source_mesh, field, coordinate_spec=chart.source_geometry
    ).prepare()
    target = D.FiniteElementPlan(
        chart.target_mesh, field, coordinate_spec=chart.target_geometry
    ).prepare()

    def coefficients(discretization: Any, corners: np.ndarray) -> np.ndarray:
        points = np.asarray(basis.functional_points)
        dofs = discretization.dof_maps[0]
        values = np.full(dofs.global_dof_count, np.nan)
        for cell, triangle in enumerate(corners):
            frame = (triangle[1:] - triangle[:1]).T
            uv = triangle[0] + points @ frame.T
            u, v = uv[:, 0], uv[:, 1]
            material = np.column_stack(
                (0.3 + 0.4 * u * u - 0.2 * u * v, -0.8 + 0.7 * v * v + 0.1 * u * v)
            )
            if proxy == "flux":
                reference = (
                    abs(np.linalg.det(frame)) * np.linalg.solve(frame, material.T).T
                )
                components = np.column_stack((-reference[:, 1], reference[:, 0]))
            else:
                components = material @ frame
            local = np.asarray(basis.interpolate(jnp.asarray(components)))
            canonical = np.linalg.solve(np.asarray(dofs.cell_transforms[0])[cell], local)
            rows = np.asarray(dofs.cell_dofs[0])[cell]
            shared = np.isfinite(values[rows])
            np.testing.assert_allclose(
                values[rows][shared], canonical[shared], atol=2e-12
            )
            values[rows] = canonical
        return values

    def chart_rows(discretization: Any, witness: Any) -> np.ndarray:
        ids = np.concatenate(
            [np.asarray(block.global_ids) for block in discretization.mesh.blocks]
        )
        return np.asarray(witness.charts)[
            np.searchsorted(np.asarray(witness.cell_global_ids), ids)
        ]

    prepared = prepare_surface_chart_compatible_transfer(
        source,
        target,
        chart,
        field_name="material",
        source_geometry=chart.source_geometry,
        target_geometry=chart.target_geometry,
    )
    old = coefficients(source, chart_rows(source, chart.source_witness))
    expected = coefficients(target, chart_rows(target, chart.target_witness))
    actual = prepared.field_transfer.transfer.apply(jnp.asarray(old))
    assert prepared.field_transfer.evidence.passed
    np.testing.assert_allclose(actual, expected, atol=2e-12)
    np.testing.assert_allclose(
        prepared.target_differential.mv(actual),
        prepared.differential_companion.mv(prepared.source_differential.mv(old)),
        atol=2e-12,
    )
    load = jnp.linspace(-1.0, 2.0, actual.size, dtype=jnp.float64)
    np.testing.assert_allclose(
        jnp.vdot(load, actual),
        jnp.vdot(prepared.field_transfer.transfer.pullback(load), jnp.asarray(old)),
        atol=2e-12,
    )


@pytest.mark.parametrize(
    "proxy,twist", [("circulation", "untwisted"), ("flux", "twisted")]
)
def test_order_two_sphere_material_transfer_commutes_and_is_dual_on_native_remesh(
    proxy: Literal["circulation", "flux"],
    twist: Literal["untwisted", "twisted"],
) -> None:
    from phydrax.discretization import prepare_sphere_chart_deformation
    from phydrax.discretization.fem import prepare_sphere_chart_compatible_transfer
    from tests.unit.geometry.test_sphere_material_atlas import _accepted

    # Two independently generated native sphere meshes: genuine projective
    # material pieces, not a shared chart or an identity correspondence.
    _, _, source_atlas = _accepted(1.2)
    _, _, target_atlas = _accepted(1.0)
    chart = prepare_sphere_chart_deformation(
        source_atlas, target_atlas, maximum_fidelity=0.3, maximum_displacement=0.5
    )
    element = form_element("triangle", 1, 2, family="trimmed", twist=twist, proxy=proxy)
    field = D.FiniteElementFieldSpec("material", element)
    source = D.FiniteElementPlan(
        chart.source_mesh, field, coordinate_spec=chart.source_geometry
    ).prepare()
    target = D.FiniteElementPlan(
        chart.target_mesh, field, coordinate_spec=chart.target_geometry
    ).prepare()
    prepared = prepare_sphere_chart_compatible_transfer(
        source,
        target,
        chart,
        field_name="material",
        source_geometry=chart.source_geometry,
        target_geometry=chart.target_geometry,
    )
    assert prepared.field_transfer.evidence.passed
    old = jnp.sin(jnp.arange(source.dof_maps[0].global_dof_count, dtype=jnp.float64))
    new = prepared.field_transfer.transfer.apply(old)
    np.testing.assert_allclose(
        prepared.target_differential.mv(new),
        prepared.differential_companion.mv(prepared.source_differential.mv(old)),
        atol=1e-11,
    )
    load = jnp.cos(jnp.arange(new.size, dtype=jnp.float64))
    np.testing.assert_allclose(
        jnp.vdot(load, new),
        jnp.vdot(prepared.field_transfer.transfer.pullback(load), old),
        atol=1e-11,
    )


def test_nested_p1_transfer_on_embedded_surface_reports_physical_area_measures() -> None:
    # Three unit right triangles of a tetrahedron corner in R^3; the first is
    # bisected through its hypotenuse midpoint. Measures are surface areas.
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    source_mesh = D.CellMesh.from_triangles(
        points, np.asarray(((0, 1, 2), (0, 2, 3), (0, 3, 1)), dtype=np.int32)
    )
    refined = np.concatenate((points, ((0.5, 0.5, 0.0),)))
    target_mesh = D.CellMesh.from_triangles(
        refined, np.asarray(((0, 1, 4), (0, 4, 2), (0, 2, 3), (0, 3, 1)), dtype=np.int32)
    )
    field = D.FiniteElementFieldSpec("u", lagrange_element("triangle", 1))
    source = D.FiniteElementPlan(
        source_mesh, field, coordinate_spec=D.CellGeometrySpec.affine(source_mesh)
    ).prepare()
    target = D.FiniteElementPlan(
        target_mesh, field, coordinate_spec=D.CellGeometrySpec.affine(target_mesh)
    ).prepare()
    reference = np.asarray(
        (
            ((0.0, 0.0), (1.0, 0.0), (0.5, 0.5)),
            ((0.0, 0.0), (0.5, 0.5), (0.0, 1.0)),
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        )
    )
    transfer = prepare_nested_field_transfer(
        source,
        target,
        np.asarray((0, 0, 1, 2)),
        field_name="u",
        parent_reference_vertices=reference,
    )
    assert transfer.evidence.passed
    source_measures = np.asarray(transfer.source_measures)
    target_measures = np.asarray(transfer.target_measures)
    np.testing.assert_allclose(source_measures.sum(), 1.5, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(target_measures.sum(), 1.5, rtol=0.0, atol=1e-14)

    def linear(coordinates: np.ndarray) -> np.ndarray:
        return 2.0 + coordinates[:, 0] - 3.0 * coordinates[:, 2]

    values = linear(np.asarray(source.dof_maps[0].dof_coordinates))
    mapped = np.asarray(transfer.transfer.apply(jnp.asarray(values)))
    np.testing.assert_allclose(
        mapped, linear(np.asarray(target.dof_maps[0].dof_coordinates)), atol=1e-14
    )
    # Independent inventory: each unit right triangle has area 1/2 and a linear
    # integrand integrates to area times its vertex mean.
    exact = sum(
        0.5 * np.mean(linear(points[list(cell)]))
        for cell in ((0, 1, 2), (0, 2, 3), (0, 3, 1))
    )
    np.testing.assert_allclose(source_measures @ values, exact, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(target_measures @ mapped, exact, rtol=0.0, atol=1e-14)
