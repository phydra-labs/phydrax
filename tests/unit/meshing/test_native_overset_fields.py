# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.typing import ArrayLike, NDArray

from phydrax import SpatialCoordinateContract
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_complex import (
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from phydrax.discretization._cell_geometry import coordinate_lagrange_element
from phydrax.discretization._hexahedral import HexahedralConnectivity
from phydrax.discretization._polyhedral_locator import PreparedPolyhedralCellLocator
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem import (
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
)
from phydrax.discretization.finite_volume import (
    CellPolynomialReconstructionPlan,
    prepare_finite_volume_field_reconstruction,
    UnstructuredFiniteVolumePlan,
)
from phydrax.discretization.finite_volume._field_view import (
    UnstructuredFiniteVolumeFieldReconstructionKernel,
)
from phydrax.linalg import ArraySpace
from phydrax.meshing import CellMeshingResult, certify_cell_mesh, MeshingScope
from phydrax.meshing._assembly import MeshAssembly, MeshPart
from phydrax.meshing._coupling import OversetCoupling
from phydrax.meshing._overset import (
    OversetConnectivity,
    OversetPartSpec,
    OversetPolicy,
    prepare_overset_connectivity,
    prepare_overset_field_transfer,
    PreparedOversetFieldTransfer,
)


jax.config.update("jax_enable_x64", True)


def _carrier(part: MeshPart) -> CellMeshingResult:
    carrier = part.carrier
    if not isinstance(carrier, CellMeshingResult):
        raise TypeError(
            "The overset field fixture requires a certified cell-mesh producer."
        )
    return carrier


def _part(
    name: str, mesh: CellMesh, geometry: CellGeometrySpec | None = None
) -> MeshPart:
    return MeshPart(
        name, certify_cell_mesh(mesh, SpatialCoordinateContract.si(), geometry=geometry)
    )


def _target(points: ArrayLike) -> tuple[MeshPart, MeshingScope]:
    points = np.asarray(points)
    dimension = points.shape[1]
    kind = "triangle" if dimension == 2 else "tetrahedron"
    mesh = CellMesh(
        np.asarray(points, dtype=np.float64),
        (
            CellBlock(
                "receptors", kind, np.arange(len(points)).reshape((-1, dimension + 1))
            ),
        ),
    )
    part = _part("target", mesh)
    connectivity = _carrier(part).mesh.connectivity
    if isinstance(connectivity, PolygonalConnectivity):
        boundary = np.asarray(connectivity.boundary_edges)
    elif isinstance(
        connectivity,
        (TetrahedralConnectivity, HexahedralConnectivity, PolyhedralConnectivity),
    ):
        boundary = np.asarray(connectivity.boundary_faces)
    else:
        raise TypeError(
            "The receptor fixture requires two- or three-dimensional connectivity."
        )
    scope = part.scope(
        dimension - 1,
        np.asarray(_carrier(part).mesh.entity_set(dimension - 1).entity_ids)[boundary],
    )
    return part, scope


def _connect(source: MeshPart, points: ArrayLike) -> OversetConnectivity:
    target, boundary = _target(points)
    return prepare_overset_connectivity(
        MeshAssembly((source, target)),
        (OversetPartSpec(source.name), OversetPartSpec(target.name, boundary=boundary)),
        policy=OversetPolicy(fringe_layers=1),
    )


def _sites(offset: float = 0.0) -> NDArray[np.float64]:
    return np.asarray(
        [[0.15, 0.15, 0.2], [0.3, 0.15, 0.2], [0.15, 0.3, 0.2], [0.15, 0.15, 0.35]]
    ) + [offset, 0, 0]


def _mapped_source(kinds: Sequence[str]) -> MeshPart:
    blocks, corners, elements, dofs, coordinates = [], [], {}, {}, []
    for index, kind in enumerate(kinds):
        name = f"block{index}"
        element = coordinate_lagrange_element(kind, 2)
        reference = np.asarray(element.reference_nodes)
        physical = reference.copy()
        physical[:, 0] += 0.1 * reference[:, 1] * reference[:, 2] + 2 * index
        vertex_points = np.asarray(
            reference_cell_topology(kind).vertices, dtype=np.float64
        )
        vertex_points[:, 0] += 0.1 * vertex_points[:, 1] * vertex_points[:, 2] + 2 * index
        blocks.append(
            CellBlock(
                name,
                kind,
                np.arange(len(vertex_points))[None] + sum(len(p) for p in corners),
                global_ids=np.asarray([90 - 17 * index]),
            )
        )
        corners.append(vertex_points)
        elements[name] = element
        dofs[name] = np.arange(len(reference))[None] + sum(len(p) for p in coordinates)
        coordinates.append(physical)
    mesh = (
        CellMesh.from_mixed_3d(np.concatenate(corners), tuple(blocks), polyhedra={})
        if "prism" in kinds
        else CellMesh(np.concatenate(corners), tuple(blocks))
    )
    geometry = CellGeometrySpec(elements, dofs, np.concatenate(coordinates))
    return _part("source", mesh, geometry)


def _check_transpose(
    route: PreparedOversetFieldTransfer,
    coefficients: Mapping[str, Array],
    values: Mapping[str, Array],
) -> None:
    dual = {
        name: jnp.arange(value.size, dtype=jnp.float64).reshape(value.shape) + 0.7
        for name, value in values.items()
    }
    assert bool(route.duality_evidence(coefficients, dual).valid)
    _, pullback = jax.vjp(
        lambda c: route.apply({"source": c})["target"], coefficients["source"]
    )
    np.testing.assert_allclose(
        route.transpose(dual)["source"], pullback(dual["target"])[0], atol=2e-11
    )


@pytest.mark.parametrize("components", [False, True])
@pytest.mark.parametrize("kinds", [("hexahedron",), ("prism",), ("hexahedron", "prism")])
def test_mapped_quadratic_global_field_and_mixed_block_grouping(
    kinds: Sequence[str],
    components: bool,
) -> None:
    source = _mapped_source(kinds)
    registration = _connect(
        source, np.concatenate([_sites(2 * i) for i in range(len(kinds))])
    )
    source = registration.assembly.part("source")
    field = FiniteElementPlan(
        _carrier(source).mesh,
        FiniteElementFieldSpec(
            "u",
            {
                block.name: lagrange_element(block.cell_kind, 2)
                for block in _carrier(source).mesh.blocks
            },
            component_shape=(2,) if components else (),
        ),
        coordinate_spec=_carrier(source).geometry,
    ).prepare()
    route = prepare_overset_field_transfer(registration, {"source": field}, "u")
    nodes = np.asarray(field.dof_maps[0].dof_coordinates)

    def polynomial(p: NDArray[np.float64]) -> NDArray[np.float64]:
        scalar = 3 + p[:, 0] + p[:, 1] ** 2 - 2 * p[:, 2] + p[:, 1] * p[:, 2]
        return np.stack((scalar, -0.5 * scalar + 0.3), axis=-1) if components else scalar

    coefficients = {"source": jnp.asarray(polynomial(nodes))}
    values = route.apply(coefficients)
    rows = registration.receptors_of("target").receptor_rows
    sites = np.asarray(_carrier(registration.assembly.part("target")).mesh.coordinates)[
        rows
    ]
    np.testing.assert_allclose(values["target"], polynomial(sites), atol=2e-10)
    assert {packet.donor_block for packet in route.packets} == {
        f"block{i}" for i in range(len(kinds))
    }
    assert not route.conservative
    for overlay in route.assembly.couplings:
        if not isinstance(overlay, OversetCoupling):
            raise TypeError(
                "The field transfer must publish its concrete overset coupling."
            )
        assert isinstance(overlay, OversetCoupling)
        query = overlay.field_query
        if query is None:
            raise ValueError(
                "The field-bound overset coupling omitted its prepared query."
            )
        assert overlay.source_scope.entity_dimension == 3
        assert overlay.target_scope.entity_dimension == 0
        assert overlay.donor_weights is None
        np.testing.assert_allclose(
            overlay.transfer(coefficients["source"]),
            polynomial(np.asarray(query.points)),
            atol=2e-10,
        )
    _check_transpose(route, coefficients, values)
    with pytest.raises(ValueError, match="actual field binding"):
        prepare_overset_field_transfer(registration, {}, "u")
    with pytest.raises(KeyError):
        prepare_overset_field_transfer(registration, {"source": field}, "absent")


def _polyhedral_source(
    multiblock: bool = False, *, numeric_version: str = "0"
) -> MeshPart:
    topology = reference_cell_topology("hexahedron")
    keys = sorted({(x, y, z) for x in range(3) for y in range(3) for z in range(3)})
    index = {key: row for row, key in enumerate(keys)}
    cells = []
    for x in range(2):
        for y in range(2):
            for z in range(2):
                rows = [
                    index[tuple(np.asarray((x, y, z)) + vertex)]
                    for vertex in topology.vertices
                ]
                cells.append(
                    tuple(np.asarray(rows)[list(face)] for face in topology.entities[2])
                )
    ids = 17 * np.arange(len(cells)) + 100
    mesh = (
        CellMesh.from_mixed_3d(
            np.asarray(keys, dtype=np.float64),
            (),
            polyhedra={"left": cells[:4], "right": cells[4:]},
            polyhedral_cell_global_ids={"left": ids[:4], "right": ids[4:]},
            numeric_version=numeric_version,
        )
        if multiblock
        else CellMesh.from_polyhedra(
            np.asarray(keys, dtype=np.float64),
            cells,
            cell_global_ids=ids,
            numeric_version=numeric_version,
        )
    )
    return _part("source", mesh)


@pytest.mark.parametrize("multiblock", [False, True])
def test_polyhedral_kexact_owner_affine_and_constant_with_exact_transpose(
    multiblock: bool,
) -> None:
    points = np.concatenate((_sites(), _sites(1.0))) if multiblock else _sites()
    registration = _connect(_polyhedral_source(multiblock), points)
    source = registration.assembly.part("source")
    fv = UnstructuredFiniteVolumePlan.from_cell_mesh(_carrier(source).mesh).prepare()
    polynomial = CellPolynomialReconstructionPlan(1).prepare(fv)
    owner = prepare_finite_volume_field_reconstruction(
        fv,
        polynomial,
        locator=PreparedPolyhedralCellLocator(_carrier(source).mesh),
    )
    route = prepare_overset_field_transfer(registration, {"source": owner}, "u")
    coefficients = {
        "source": 2
        + fv.cell_centers[:, :1]
        - 3 * fv.cell_centers[:, 1:2]
        + 4 * fv.cell_centers[:, 2:3]
    }
    values = route.apply(coefficients)
    sites = np.asarray(_carrier(registration.assembly.part("target")).mesh.coordinates)
    rows = registration.receptors_of("target").receptor_rows
    expected = 2 + sites[rows, :1] - 3 * sites[rows, 1:2] + 4 * sites[rows, 2:3]
    np.testing.assert_allclose(values["target"], expected, atol=2e-11)
    np.testing.assert_allclose(
        route.apply({"source": jnp.ones(fv.state_shape)})["target"], 1, atol=2e-12
    )
    _check_transpose(route, coefficients, values)
    for coupling in route.assembly.couplings:
        if not isinstance(coupling, OversetCoupling):
            raise TypeError(
                "The FV field transfer must publish its concrete overset coupling."
            )
        sites = np.asarray(
            registration.assembly.part("target").point_coordinates(coupling.target_scope)
        )
        expected = 2 + sites[:, :1] - 3 * sites[:, 1:2] + 4 * sites[:, 2:3]
        actual = coupling.transfer(coefficients["source"])
        np.testing.assert_allclose(actual, expected, atol=2e-11)
        dual = jnp.arange(actual.size, dtype=jnp.float64).reshape(actual.shape) + 0.3
        np.testing.assert_allclose(
            jnp.vdot(actual, dual),
            jnp.vdot(coefficients["source"], coupling.transpose(dual)),
            atol=2e-11,
        )
    kernel = owner.kernel
    if not isinstance(kernel, UnstructuredFiniteVolumeFieldReconstructionKernel):
        raise TypeError(
            "The polyhedral reconstruction requires its unstructured FV owner."
        )
    packet_kernel = route.queries[0].reconstruction.kernel
    if not isinstance(packet_kernel, UnstructuredFiniteVolumeFieldReconstructionKernel):
        raise TypeError(
            "The packet query must retain the unstructured FV reconstruction owner."
        )
    with pytest.raises(ValueError, match="inactive hole/fringe"):
        kernel.prepare_packet_query(
            owner,
            packet_kernel.locator,
            route.packets[0].points,
            jnp.zeros(fv.cell_count, dtype=bool),
        )


@pytest.mark.parametrize(
    ("family", "isometry"),
    [
        ("nedelec1", "rotation"),
        ("rt0", None),
        ("bdm1", "reflection"),
    ],
)
def test_refined_compatible_field_true_global_coefficients(
    family: str,
    isometry: str | None,
) -> None:
    from phydrax.discretization._cell_geometry_transfer import (
        transition_nested_cell_geometry,
    )
    from phydrax.discretization.fem import form_element
    from phydrax.meshing._mixed_adaptation import adapt_mixed_mesh
    from phydrax.meshing._topology_edit import assemble_topology_edit

    root = _mapped_source(("tetrahedron",))
    outcome = adapt_mixed_mesh(
        _carrier(root).mesh, refine_cell_ids=_carrier(root).mesh.blocks[0].global_ids
    )
    refined, _, _ = assemble_topology_edit(
        _carrier(root).mesh, outcome.edit, numeric_version="refined"
    )
    transition = transition_nested_cell_geometry(
        _carrier(root).mesh,
        _carrier(root).geometry,
        refined,
        CellGeometrySpec.affine(refined),
        refinement=outcome.edit.refinement,
    )
    geometry = transition.geometry
    refined = refined.with_coordinates(
        transition.vertex_coordinates, numeric_version="refined"
    )
    source = _part("source", refined, geometry)
    rotation, translation = np.eye(3), np.zeros(3)
    target_rotation, target_translation = np.eye(3), np.zeros(3)
    if isometry is not None:
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        translation = np.asarray([2.0, -0.3, 0.7])
        if isometry == "reflection":
            rotation[2, 2] = -1.0
            target_rotation = np.diag(np.asarray([1.0, 1.0, -1.0]))
            target_translation = np.asarray([-0.2, 0.4, 0.1])
    world_sites = _sites() @ rotation.T + translation
    target_sites = (world_sites - target_translation) @ target_rotation
    target, boundary = _target(target_sites)
    registration = prepare_overset_connectivity(
        MeshAssembly((source, target)),
        (
            OversetPartSpec(
                source.name,
                image_rotation=None if isometry is None else rotation,
                image_translation=None if isometry is None else translation,
            ),
            OversetPartSpec(
                target.name,
                boundary=boundary,
                image_rotation=None if isometry != "reflection" else target_rotation,
                image_translation=None
                if isometry != "reflection"
                else target_translation,
            ),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )
    source = registration.assembly.part("source")
    element = {
        "nedelec1": form_element("tetrahedron", 1, 2, proxy="circulation"),
        "rt0": form_element("tetrahedron", 2, 1, twist="twisted", proxy="flux"),
        "bdm1": form_element(
            "tetrahedron", 2, 1, family="full", twist="twisted", proxy="flux"
        ),
    }[family]
    field = FiniteElementPlan(
        _carrier(source).mesh,
        FiniteElementFieldSpec("u", element),
        coordinate_spec=_carrier(source).geometry,
    ).prepare()
    # Identify the exact globally conforming coefficient representation of a
    # constant reference-compatible field, using independent volume tabulation.
    # Dense nonmonomial face transformations and orientation signs are included.
    rows, right = [], []
    vector = np.asarray([0.7, -0.2, 0.4])
    probe = np.asarray(
        [
            [0.11, 0.13, 0.17],
            [0.23, 0.17, 0.19],
            [0.09, 0.31, 0.13],
            [0.31, 0.07, 0.11],
            [0.13, 0.11, 0.29],
            [0.17, 0.23, 0.07],
            [0.21, 0.09, 0.23],
            [0.07, 0.19, 0.31],
        ]
    )
    vector_space = field.field_spaces[0].vector_space
    if not isinstance(vector_space, ArraySpace):
        raise TypeError("Overset FEM fixture requires an array vector space.")
    for block_index, block in enumerate(field.mesh.blocks):
        evaluated = field.evaluate_block_geometry(
            "u",
            block_index,
            field.default_runtime.coordinates,
            jnp.asarray(probe),
            jnp.ones(len(probe)),
        )
        basis = np.asarray(evaluated.basis_values)
        routes = np.asarray(field.dof_maps[0].cell_dofs[block_index])
        transforms = np.asarray(field.dof_maps[0].cell_transforms[block_index])
        for cell in range(block.cell_count):
            matrix = np.zeros((len(probe), 3, vector_space.shape[0]))
            matrix[:, :, routes[cell]] = np.moveaxis(
                np.einsum("qav,ai->qiv", basis[cell], transforms[cell]),
                1,
                2,
            )
            rows.append(matrix.reshape((-1, matrix.shape[-1])))
            # The analytic map x = xi + .1*y*z e_x has det J=1.
            physical = np.asarray(evaluated.physical_points)[cell]
            jacobian = np.broadcast_to(np.eye(3), (len(probe), 3, 3)).copy()
            jacobian[:, 0, 1] = 0.1 * physical[:, 2]
            jacobian[:, 0, 2] = 0.1 * physical[:, 1]
            if family == "nedelec1":
                expected = np.einsum(
                    "pij,j->pi", np.linalg.inv(jacobian).transpose(0, 2, 1), vector
                )
            else:
                expected = np.einsum("pij,j->pi", jacobian, vector)
            right.append(expected.reshape(-1))
    coefficients = {
        "source": jnp.asarray(
            np.linalg.lstsq(np.concatenate(rows), np.concatenate(right), rcond=None)[0]
        )
    }
    route = prepare_overset_field_transfer(
        registration,
        {"source": field},
        "u",
        value_action=None if isometry is None else "polar-vector",
    )
    sites = _sites()
    jacobian = np.broadcast_to(np.eye(3), (len(sites), 3, 3)).copy()
    jacobian[:, 0, 1], jacobian[:, 0, 2] = 0.1 * sites[:, 2], 0.1 * sites[:, 1]
    expected = (
        np.einsum("pij,j->pi", np.linalg.inv(jacobian).transpose(0, 2, 1), vector)
        if family == "nedelec1"
        else np.einsum("pij,j->pi", jacobian, vector)
    )
    values = route.apply(coefficients)
    np.testing.assert_allclose(
        values["target"], expected @ rotation.T @ target_rotation, atol=3e-9
    )
    np.testing.assert_allclose(
        values["target"] @ target_rotation.T, expected @ rotation.T, atol=3e-9
    )
    _check_transpose(route, coefficients, values)


def test_registered_mixed_quadratic_image_scalar_motion_and_restart() -> None:
    source = _mapped_source(("hexahedron", "prism"))
    sites = np.concatenate((_sites(), _sites(2.0)))
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.asarray([2.0, -0.3, 0.7])
    target_rotation = np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    target_translation = np.asarray([-0.2, 0.4, 0.1])
    target, boundary = _target(
        (sites @ rotation.T + translation - target_translation) @ target_rotation
    )
    registration = prepare_overset_connectivity(
        MeshAssembly((source, target)),
        (
            OversetPartSpec(
                source.name, image_rotation=rotation, image_translation=translation
            ),
            OversetPartSpec(
                target.name,
                boundary=boundary,
                image_rotation=target_rotation,
                image_translation=target_translation,
            ),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )
    field = FiniteElementPlan(
        _carrier(source).mesh,
        FiniteElementFieldSpec(
            "u",
            {
                block.name: lagrange_element(block.cell_kind, 2)
                for block in _carrier(source).mesh.blocks
            },
        ),
        coordinate_spec=_carrier(source).geometry,
    ).prepare()

    def polynomial(points: NDArray[np.float64]) -> NDArray[np.float64]:
        return (
            3
            + points[:, 0]
            + points[:, 1] ** 2
            - 2 * points[:, 2]
            + points[:, 1] * points[:, 2]
        )

    coefficients = {
        "source": jnp.asarray(polynomial(np.asarray(field.dof_maps[0].dof_coordinates)))
    }
    with pytest.raises(ValueError):
        prepare_overset_field_transfer(registration, {"source": field}, "u")
    route = prepare_overset_field_transfer(
        registration, {"source": field}, "u", value_action="invariant"
    )
    values = route.apply(coefficients)
    np.testing.assert_allclose(values["target"], polynomial(sites), atol=2e-10)
    _check_transpose(route, coefficients, values)
    restored = registration.registration().prepare()
    assert restored.connectivity_id == registration.connectivity_id
    candidate = restored.moved(
        {
            target.name: _carrier(target).mesh.coordinates
            + jnp.asarray([0.01, 0.0, 0.0]),
        }
    )
    moved_route = prepare_overset_field_transfer(
        candidate,
        {"source": field},
        "u",
        previous=restored,
        value_action="invariant",
    )
    points = np.asarray(_carrier(candidate.assembly.part("target")).mesh.coordinates)
    donor_sites = (
        points @ target_rotation.T + target_translation - translation
    ) @ rotation
    moved_values = moved_route.apply(coefficients)
    np.testing.assert_allclose(
        moved_values["target"], polynomial(donor_sites), atol=2e-10
    )
    _check_transpose(moved_route, coefficients, moved_values)


def test_registered_mapped_solid_image_preserves_source_wall_authority() -> None:
    from phydrax.discretization._mapped_locator import PreparedMappedCellLocator
    from phydrax.discretization._view_support import mapped_mesh_support_geometry
    from phydrax.meshing._overset import (
        _part_geometry,
        OversetCellStatus,
        OversetVertexStatus,
    )

    source = _mapped_source(("tetrahedron",))
    policy = OversetPolicy(fringe_layers=1)
    geometry = _part_geometry(source, OversetPartSpec(source.name), policy, None)
    locator = geometry.locators[0]
    if not isinstance(locator, PreparedMappedCellLocator):
        raise TypeError("The wall fixture requires its actual mapped source chart.")
    solid = mapped_mesh_support_geometry(locator, "closed-curved-image-solid")
    mesh = _carrier(source).mesh
    connectivity = mesh.connectivity
    if not isinstance(connectivity, TetrahedralConnectivity):
        raise TypeError("The wall fixture requires its actual tetrahedral incidence.")
    wall = source.scope(
        2,
        np.asarray(mesh.entity_set(2).entity_ids)[
            np.asarray(connectivity.boundary_faces)
        ],
    )
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.asarray([2.0, -0.3, 0.7])
    target, _ = _target(_sites() @ rotation.T + translation)
    registration = prepare_overset_connectivity(
        MeshAssembly((source, target)),
        (
            OversetPartSpec(
                source.name,
                wall=wall,
                solid_query=solid,
                image_rotation=rotation,
                image_translation=translation,
            ),
            OversetPartSpec(target.name),
        ),
        policy=policy,
    )
    registration.require_complete()
    np.testing.assert_array_equal(
        registration.blanking_of(target.name).vertex_status, int(OversetVertexStatus.HOLE)
    )
    np.testing.assert_array_equal(
        registration.blanking_of(target.name).cell_status, int(OversetCellStatus.HOLE)
    )
    np.testing.assert_array_equal(
        registration.blanking_of(source.name).vertex_status,
        int(OversetVertexStatus.ACTIVE),
    )


def test_common_numeric_affine_images_cannot_claim_polar_vector_isometry() -> None:
    source = _mapped_source(("tetrahedron",))
    target, boundary = _target(_sites())
    rotation = (1.0 + 2.0**-44) * np.eye(3, dtype=np.float64)
    translation = np.asarray([0.2, -0.3, 0.7], dtype=np.float64)
    registration = prepare_overset_connectivity(
        MeshAssembly((source, target)),
        (
            OversetPartSpec(
                source.name, image_rotation=rotation, image_translation=translation
            ),
            OversetPartSpec(
                target.name,
                boundary=boundary,
                image_rotation=rotation,
                image_translation=translation,
            ),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )
    field = FiniteElementPlan(
        _carrier(source).mesh,
        FiniteElementFieldSpec(
            "u", lagrange_element("tetrahedron", 2), component_shape=(3,)
        ),
        coordinate_spec=_carrier(source).geometry,
    ).prepare()
    with pytest.raises(ValueError):
        prepare_overset_field_transfer(
            registration, {"source": field}, "u", value_action="polar-vector"
        )
    route = prepare_overset_field_transfer(
        registration,
        {"source": field},
        "u",
        value_action="contravariant-vector",
    )
    nodes = np.asarray(field.dof_maps[0].dof_coordinates)
    coefficients = {
        "source": jnp.asarray(
            np.stack((1 + nodes[:, 0], 2 - nodes[:, 1], 0.5 + nodes[:, 2]), axis=1)
        )
    }
    sites = _sites()
    expected = np.stack((1 + sites[:, 0], 2 - sites[:, 1], 0.5 + sites[:, 2]), axis=1)
    values = route.apply(coefficients)
    np.testing.assert_allclose(values["target"], expected, atol=2e-10)
    _check_transpose(route, coefficients, values)
    coupling = route.field_couplings[0]
    if coupling.field_query is None:
        raise TypeError("The numeric affine route must retain its actual field query.")
    with pytest.raises(ValueError):
        OversetCoupling.from_field_query(
            source,
            target,
            coupling.source_scope,
            coupling.target_scope,
            coupling.field_query,
            rotation=coupling.rotation,
            translation=coupling.translation,
            source_image_rotation=rotation,
            target_image_rotation=rotation,
            value_action="polar-vector",
        )


def test_isometry_query_rejects_invalid_transform_and_nonvector_values() -> None:
    registration = _connect(_mapped_source(("tetrahedron",)), _sites())
    source = registration.assembly.part("source")
    field = FiniteElementPlan(
        _carrier(source).mesh,
        FiniteElementFieldSpec("u", lagrange_element("tetrahedron", 2)),
        coordinate_spec=_carrier(source).geometry,
    ).prepare()
    route = prepare_overset_field_transfer(registration, {"source": field}, "u")
    coupling = route.assembly.couplings[0]
    if not isinstance(coupling, OversetCoupling) or coupling.field_query is None:
        raise TypeError("The field transfer must retain its actual query.")
    args = (
        source,
        registration.assembly.part("target"),
        coupling.source_scope,
        coupling.target_scope,
        coupling.field_query,
    )
    with pytest.raises(ValueError):
        OversetCoupling.from_field_query(*args, rotation=np.eye(3))
    with pytest.raises(ValueError):
        OversetCoupling.from_field_query(
            *args, rotation=2 * np.eye(3), translation=np.zeros(3)
        )
    with pytest.raises(ValueError):
        OversetCoupling.from_field_query(
            *args, rotation=np.eye(3), translation=np.zeros(3)
        )


def test_fv_binding_refuses_same_coordinates_from_another_mesh_revision() -> None:
    from phydrax.discretization.finite_volume import PiecewiseConstantReconstruction

    old = _polyhedral_source()
    fv = UnstructuredFiniteVolumePlan.from_cell_mesh(_carrier(old).mesh).prepare()
    owner = prepare_finite_volume_field_reconstruction(
        fv,
        PiecewiseConstantReconstruction(),
        locator=PreparedPolyhedralCellLocator(_carrier(old).mesh),
    )
    candidate = _connect(_polyhedral_source(numeric_version="new-revision"), _sites())
    with pytest.raises(ValueError, match="another mesh/coordinate revision"):
        prepare_overset_field_transfer(candidate, {"source": owner}, "u")


def test_actual_nonlinear_fv_query_cannot_publish_an_algebraic_transpose() -> None:
    from phydrax.discretization.finite_volume import UnstructuredWENOZReconstructionPlan

    coordinates = np.asarray(
        [(x, y) for y in range(5) for x in range(5)], dtype=np.float64
    )
    cells = []
    for y in range(4):
        for x in range(4):
            a = 5 * y + x
            cells.extend(((a, a + 1, a + 6), (a, a + 6, a + 5)))
    mesh = CellMesh(coordinates, (CellBlock("cells", "triangle", np.asarray(cells)),))
    registration = _connect(_part("source", mesh), _sites()[:3, :2])
    source = registration.assembly.part("source")
    fv = UnstructuredFiniteVolumePlan.from_cell_mesh(_carrier(source).mesh).prepare()
    policy = UnstructuredWENOZReconstructionPlan(2, limiter="none").prepare(fv)
    owner = prepare_finite_volume_field_reconstruction(fv, policy)
    route = prepare_overset_field_transfer(registration, {"source": owner}, "u")
    state = jnp.ones(fv.state_shape) * 2.7
    values = route.apply({"source": state})
    np.testing.assert_allclose(values["target"], 2.7, atol=2e-12)
    assert not route.coefficient_linear
    assert not route.conservative
    dual = {name: jnp.ones(value.shape) for name, value in values.items()}
    with pytest.raises(ValueError, match="Nonlinear FV reconstruction"):
        route.transpose(dual)
    overlay = route.assembly.couplings[0]
    if not isinstance(overlay, OversetCoupling):
        raise TypeError(
            "The nonlinear FV transfer must publish its concrete overset coupling."
        )
    query = overlay.field_query
    if query is None:
        raise ValueError("The nonlinear FV coupling omitted its prepared field query.")
    with pytest.raises(ValueError, match="nonlinear"):
        overlay.transpose(jnp.ones(query.output_shape))


def test_moved_mixed_mapped_receptors_keep_actual_block_field_order() -> None:
    previous = _connect(
        _mapped_source(("hexahedron", "prism")), np.concatenate((_sites(), _sites(2.0)))
    )
    source = previous.assembly.part("source")
    field = FiniteElementPlan(
        _carrier(source).mesh,
        FiniteElementFieldSpec(
            "u",
            {
                block.name: lagrange_element(block.cell_kind, 2)
                for block in _carrier(source).mesh.blocks
            },
        ),
        coordinate_spec=_carrier(source).geometry,
    ).prepare()
    target = previous.assembly.part("target")
    candidate = previous.moved(
        {"target": _carrier(target).mesh.coordinates + jnp.asarray([0.02, 0.01, 0.0])}
    )
    route = prepare_overset_field_transfer(
        candidate, {"source": field}, "u", previous=previous
    )
    polynomial = lambda p: 3 + p[:, 0] + p[:, 1] ** 2 - 2 * p[:, 2] + p[:, 1] * p[:, 2]
    coefficients = {
        "source": jnp.asarray(polynomial(np.asarray(field.dof_maps[0].dof_coordinates)))
    }
    values = route.apply(coefficients)
    points = np.asarray(_carrier(candidate.assembly.part("target")).mesh.coordinates)
    rows = next(
        evidence.receptor_rows
        for evidence in route.receptors
        if evidence.part_name == "target"
    )
    np.testing.assert_allclose(values["target"], polynomial(points[rows]), atol=2e-10)
    _check_transpose(route, coefficients, values)


def test_discontinuous_field_uses_the_selected_donor_side_on_a_shared_facet() -> None:
    from phydrax.discretization.fem import discontinuous_element

    mesh = CellMesh(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, -0.3],
            ]
        ),
        (
            CellBlock(
                "cells",
                "tetrahedron",
                np.asarray([[0, 1, 2, 3], [0, 2, 1, 4]]),
                global_ids=np.asarray([10, 20]),
            ),
        ),
    )
    registration = _connect(
        _part("source", mesh),
        np.asarray(
            [[0.2, 0.3, 0.0], [0.2, 0.4, -0.1], [0.3, 0.3, -0.1], [0.2, 0.3, -0.12]]
        ),
    )
    source = registration.assembly.part("source")
    field = FiniteElementPlan(
        _carrier(source).mesh,
        FiniteElementFieldSpec("u", discontinuous_element("tetrahedron", 0)),
        coordinate_spec=_carrier(source).geometry,
    ).prepare()
    vector_space = field.field_spaces[0].vector_space
    if not isinstance(vector_space, ArraySpace):
        raise TypeError("Overset FEM fixture requires an array vector space.")
    coefficients = jnp.zeros(vector_space.shape)
    coefficients = coefficients.at[field.dof_maps[0].cell_dofs[0][0]].set(7.0)
    coefficients = coefficients.at[field.dof_maps[0].cell_dofs[0][1]].set(11.0)
    route = prepare_overset_field_transfer(registration, {"source": field}, "u")
    values = route.apply({"source": coefficients})
    np.testing.assert_allclose(values["target"], 11.0, atol=2e-12)
    _check_transpose(route, {"source": coefficients}, values)


def test_fv_binding_requires_actual_mapped_source_not_affine_corners() -> None:
    from phydrax.discretization.finite_volume import PiecewiseConstantReconstruction

    reference = np.asarray(
        [[0.3, 0.3, 0.398], [0.3, 0.29, 0.398], [0.29, 0.3, 0.398], [0.3, 0.3, 0.388]]
    )
    points = reference.copy()
    points[:, 0] += 0.1 * reference[:, 1] * reference[:, 2]
    # The first site lies beyond the affine tetrahedron's diagonal face but
    # inside the actual curved donor source.
    assert np.sum(points[0]) > 1.0
    registration = _connect(_mapped_source(("tetrahedron",)), points)
    source = registration.assembly.part("source")
    plan = UnstructuredFiniteVolumePlan.from_cell_mesh(_carrier(source).mesh)
    corners = plan.prepare()
    wrong = prepare_finite_volume_field_reconstruction(
        corners, PiecewiseConstantReconstruction()
    )
    with pytest.raises(ValueError, match="corner-only field"):
        prepare_overset_field_transfer(registration, {"source": wrong}, "u")
    actual = plan.prepare(cell_geometry=_carrier(source).geometry)
    owner = prepare_finite_volume_field_reconstruction(
        actual, PiecewiseConstantReconstruction()
    )
    route = prepare_overset_field_transfer(registration, {"source": owner}, "u")
    coefficients = {"source": jnp.full(actual.state_shape, 7.3)}
    values = route.apply(coefficients)
    np.testing.assert_allclose(values["target"], 7.3, atol=2e-12)
    _check_transpose(route, coefficients, values)
    coupling = route.assembly.couplings[0]
    np.testing.assert_allclose(coupling.transfer(coefficients["source"]), 7.3, atol=2e-12)
