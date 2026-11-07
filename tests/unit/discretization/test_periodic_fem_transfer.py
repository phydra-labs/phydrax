from itertools import permutations
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._periodic_topology import PeriodicMeshTopology
from phydrax.discretization.fem import FormElementFamily
from phydrax.discretization.fem._de_rham import FiniteElementDeRhamComplex
from phydrax.discretization.fem._form_elements import FormTwist
from phydrax.meshing._periodic import periodic_entity_orbit_map


def _periodic_tetrahedral_cover(
    periodic_axes: tuple[bool, bool, bool] = (True, True, True),
) -> CellMesh:
    shape = 3
    vertex_rows = np.arange(shape**3, dtype=np.int64).reshape((shape,) * 3)
    lattice_points = np.stack(
        np.meshgrid(*(np.arange(shape) for _ in range(3)), indexing="ij"), axis=-1
    ).reshape((-1, 3))
    coordinates = lattice_points.astype(np.float64) / 2.0
    indices = np.arange(shape, dtype=np.int64)
    representative_axes = tuple(
        indices % 2 if periodic else indices for periodic in periodic_axes
    )
    representatives = vertex_rows[
        representative_axes[0][:, None, None],
        representative_axes[1][None, :, None],
        representative_axes[2][None, None, :],
    ].reshape(-1)
    shift_axes = tuple(
        indices // 2 if periodic else np.zeros_like(indices) for periodic in periodic_axes
    )
    shifts = np.stack(np.meshgrid(*shift_axes, indexing="ij"), axis=-1).reshape((-1, 3))
    tetrahedra = []
    for cube in np.ndindex((shape - 1,) * 3):
        for order in permutations(range(3)):
            point = np.asarray(cube, dtype=np.int64)
            path = [point.copy()]
            for axis in order:
                point = point.copy()
                point[axis] += 1
                path.append(point)
            tetrahedron = np.asarray(
                [vertex_rows[tuple(row)] for row in path], dtype=np.int32
            )
            if (
                np.linalg.det(coordinates[tetrahedron[1:]] - coordinates[tetrahedron[0]])
                < 0.0
            ):
                tetrahedron[[1, 2]] = tetrahedron[[2, 1]]
            tetrahedra.append(tetrahedron)
    block = CellBlock(
        "periodic-tetrahedra",
        "tetrahedron",
        np.asarray(tetrahedra, dtype=np.int32),
        global_ids=np.arange(3000, 3000 + len(tetrahedra), dtype=np.int64),
    )
    vertex_ids = np.arange(shape**3, dtype=np.int64)
    lifted = CellMesh(coordinates, (block,), vertex_global_ids=vertex_ids)
    topology = PeriodicMeshTopology(
        lifted,
        PeriodicCell(np.eye(3, dtype=np.float64), periodic_axes=periodic_axes),
        representatives,
        shifts,
    )
    return CellMesh(
        coordinates,
        (block,),
        vertex_global_ids=vertex_ids,
        periodic_topology=topology,
    )


@pytest.mark.parametrize(
    ("family", "source_order", "target_order"),
    (("trimmed", 1, 2), ("full", 0, 1)),
)
def test_periodic_high_order_de_rham_transfer_descends_and_commutes(
    family: FormElementFamily, source_order: int, target_order: int
) -> None:
    mesh = _periodic_tetrahedral_cover()
    source = FiniteElementDeRhamComplex(mesh, family=family, order=source_order)
    target = FiniteElementDeRhamComplex(mesh, family=family, order=target_order)
    transfer = source.transfer_periodic_quotient(target)
    quotient_map = transfer.complex_map
    reports = transfer.support_report()
    assert len(reports) == 4
    assert all(report.rank_lower_bound <= report.rank_upper_bound for report in reports)

    assert transfer.lifted_transfer_result.entry_evidence
    assert transfer.quotient_transfer_result.complex_map.map_id == quotient_map.map_id
    np.testing.assert_allclose(
        transfer.commuting_error_bounds,
        transfer.quotient_transfer_result.commuting_error_bounds,
    )
    assert quotient_map.source.top_degree == 3
    assert quotient_map.target.top_degree == 3
    assert np.max(np.asarray(transfer.deck_descent_defects), initial=0.0) <= 1.0e-10
    assert np.max(np.asarray(transfer.commuting_defects), initial=0.0) <= 1.0e-10
    assert np.max(np.asarray(transfer.source_closure_defects), initial=0.0) <= 1.0e-10
    assert np.max(np.asarray(transfer.target_closure_defects), initial=0.0) <= 1.0e-10
    assert np.max(np.asarray(transfer.source_nilpotency_defects), initial=0.0) <= 1.0e-10
    assert np.max(np.asarray(transfer.target_nilpotency_defects), initial=0.0) <= 1.0e-10

    edge_orbits = periodic_entity_orbit_map(mesh, 1)
    assert np.unique(np.asarray(edge_orbits.lifted_to_quotient)).size < len(
        edge_orbits.lifted_entity_ids
    )
    assert np.any(np.asarray(edge_orbits.winding_shifts) != 0)

    generator = np.random.default_rng(19)
    for degree in range(4):
        source_space = quotient_map.source.space(degree)
        target_space = quotient_map.target.space(degree)
        source_values = jnp.asarray(
            generator.normal(size=source_space.size), dtype=jnp.float64
        )
        target_values = jnp.asarray(
            generator.normal(size=target_space.size), dtype=jnp.float64
        )
        action = quotient_map.map(degree)
        np.testing.assert_allclose(
            target_space.inner(action.mv(source_values), target_values),
            source_space.inner(
                source_values, quotient_map.adjoint().map(degree).mv(target_values)
            ),
            rtol=2.0e-10,
            atol=2.0e-10,
        )
        if degree < 3:
            left = quotient_map.target.differential(degree).mv(action.mv(source_values))
            right = quotient_map.map(degree + 1).mv(
                quotient_map.source.differential(degree).mv(source_values)
            )
            np.testing.assert_allclose(left, right, rtol=2.0e-10, atol=2.0e-10)


def test_periodic_relative_transfer_uses_quotient_physical_boundary() -> None:
    mesh = _periodic_tetrahedral_cover((True, False, False))
    source = FiniteElementDeRhamComplex(mesh, family="trimmed", order=1)
    target = FiniteElementDeRhamComplex(mesh, family="trimmed", order=2)
    transfer = source.transfer_periodic_quotient(target, boundary="relative")
    quotient_map = transfer.complex_map
    assert transfer.lifted_transfer_result.evidence_id
    assert transfer.quotient_transfer_result.entry_evidence

    assert transfer.boundary == "relative"
    assert len(transfer.target_active_indices[0]) < transfer.target_lifts[0].source.size
    generator = np.random.default_rng(23)
    for degree in range(4):
        source_space = quotient_map.source.space(degree)
        values = jnp.asarray(generator.normal(size=source_space.size), dtype=jnp.float64)
        mapped = quotient_map.map(degree).mv(values)
        lifted = transfer.lift_target(degree, mapped)
        np.testing.assert_allclose(
            transfer.collapse_target(degree, lifted), mapped, atol=2.0e-12
        )
        if degree < 3:
            np.testing.assert_allclose(
                quotient_map.target.differential(degree).mv(mapped),
                quotient_map.map(degree + 1).mv(
                    quotient_map.source.differential(degree).mv(values)
                ),
                atol=2.0e-10,
                rtol=2.0e-10,
            )


def test_periodic_prism_form_transfer_uses_normalized_entity_actions() -> None:
    from phydrax._owned_programs import run_owned_program
    from tools.meshing_qualification.scenarios.native_hybrid_profiles import (
        _layer_form_action,
        _layer_form_transpose,
    )

    coordinates = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1)),
        dtype=np.float64,
    )
    block = CellBlock("prisms", "prism", ((0, 1, 2, 3, 4, 5),), global_ids=(41,))
    lifted = CellMesh(coordinates, (block,))
    periodic = PeriodicMeshTopology(
        lifted,
        PeriodicCell(np.asarray(((0.0, 0.0, 1.0),))),
        np.asarray((0, 1, 2, 0, 1, 2), dtype=np.int32),
        np.asarray(((0,), (0,), (0,), (1,), (1,), (1,)), dtype=np.int32),
    )
    mesh = CellMesh(coordinates, (block,), periodic_topology=periodic)
    complex_ = FiniteElementDeRhamComplex(mesh, family="prism-trimmed", order=2)
    result = complex_.transfer_periodic_quotient(complex_)
    for degree in range(4):
        space = result.complex_map.source.space(degree)
        values = jnp.sin(jnp.arange(space.size, dtype=jnp.float64) + 0.23)
        np.testing.assert_allclose(
            result.map(degree).mv(values), values, atol=1e-10, rtol=1e-10
        )
        lifted_action = result.lifted_transfer_result.complex_map.map(degree)
        lifted_values = jnp.sin(jnp.arange(lifted_action.source.size, dtype=jnp.float64))
        np.testing.assert_allclose(
            run_owned_program(lifted_action, "layer-form-action", _layer_form_action, lifted_values),
            lifted_action.mv(lifted_values), atol=1e-10, rtol=1e-10,
        )
        dual = jnp.cos(jnp.arange(lifted_action.target.size, dtype=jnp.float64))
        pullback = jax.linear_transpose(lifted_action.mv, jnp.zeros_like(lifted_values))(dual)[0]
        compiled_pullback = run_owned_program(
            lifted_action, "layer-form-transpose", _layer_form_transpose, lifted_values, dual
        )
        np.testing.assert_allclose(compiled_pullback, pullback, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(
            jnp.vdot(lifted_action.mv(lifted_values), dual),
            jnp.vdot(lifted_values, pullback), atol=1e-10, rtol=1e-10,
        )
    assert len(result.support_report()) == 4


@pytest.mark.parametrize("twist", ("untwisted", "twisted"))
def test_periodic_mixed_prism_tetrahedron_quotient_uses_real_orbits(
    twist: FormTwist, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from copy import copy
    import pickle
    import weakref

    import phydrax.discretization.fem._de_rham as de_rham

    from phydrax.discretization.fem._de_rham import (
        _PeriodicFormTopology,
        _periodic_entity_coordinate_maps,
    )
    from phydrax.geometry._mesh_certificates import _same_tree_content

    from phydrax.discretization._cell_complex import PolyhedralConnectivity
    from phydrax.discretization._periodic_topology import _entity_vertex_rows
    coordinates = np.asarray(
        tuple((x, y, z) for z in (0, 1, 2) for x, y in ((0, 0), (1, 0), (0, 1))),
        dtype=np.float64,
    )
    blocks = (
        CellBlock("prisms", "prism", ((0, 1, 2, 3, 4, 5),), global_ids=(41,)),
        CellBlock(
            "tetrahedra",
            "tetrahedron",
            ((3, 4, 5, 6), (4, 5, 6, 7), (5, 7, 8, 6)),
            global_ids=(51, 52, 53),
        ),
    )
    lifted = CellMesh(coordinates, blocks)
    periodic = PeriodicMeshTopology(
        lifted,
        PeriodicCell(np.asarray(((0.0, 0.0, 2.0),))),
        np.asarray((0, 1, 2, 3, 4, 5, 0, 1, 2), dtype=np.int32),
        np.asarray(
            ((0,), (0,), (0,), (0,), (0,), (0,), (1,), (1,), (1,)), dtype=np.int32
        ),
    )
    mesh = CellMesh(coordinates, blocks, periodic_topology=periodic)
    assert isinstance(mesh.connectivity, PolyhedralConnectivity)
    edge_rows = _entity_vertex_rows(mesh)[1]
    assert all(row.dtype == np.dtype(np.int64) for row in edge_rows)
    np.testing.assert_array_equal(np.stack(edge_rows), np.asarray(mesh.connectivity.edges))
    orbit = periodic_entity_orbit_map(mesh, 1)
    np.testing.assert_array_equal(orbit.lifted_entity_ids, mesh.entity_set(1).entity_ids)
    malformed = copy(mesh)
    connectivity = copy(mesh.connectivity)
    object.__setattr__(connectivity, "edges", mesh.connectivity.edges[:-1])
    object.__setattr__(malformed, "connectivity", connectivity)
    with pytest.raises(ValueError, match="corner rows"):
        periodic_entity_orbit_map(malformed, 1)
    complex_ = FiniteElementDeRhamComplex(mesh, family="trimmed", order=2, twist=twist)
    prepared_topology = _PeriodicFormTopology(mesh, complex_.dimension)
    with pytest.raises(ValueError, match="different actual mesh"):
        prepared_topology.require(copy(mesh), complex_.dimension)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(prepared_topology)
    for degree in range(4):
        standalone = _periodic_entity_coordinate_maps(complex_, degree)
        reused = _periodic_entity_coordinate_maps(
            complex_, degree, _topology=prepared_topology
        )
        assert _same_tree_content(standalone[:2], reused[:2])
        assert standalone[2] == reused[2]
        np.testing.assert_array_equal(standalone[3], reused[3])
    result = complex_.transfer_periodic_quotient(complex_)
    target_complex = FiniteElementDeRhamComplex(
        copy(mesh), family="trimmed", order=2, twist=twist
    )
    source_row_refs = []
    original_actions = de_rham._periodic_entity_coordinate_maps

    def tracked_actions(complex_: Any, degree: int, /, *, _topology: Any) -> Any:
        if complex_ is complex_held and degree == 0:
            source_row_refs.append(weakref.ref(_topology.ordered_vertices[0][0]))
        elif complex_ is target_complex and degree == 0:
            assert source_row_refs and source_row_refs[0]() is None
        return original_actions(complex_, degree, _topology=_topology)

    complex_held = complex_
    with monkeypatch.context() as tracking:
        tracking.setattr(de_rham, "_periodic_entity_coordinate_maps", tracked_actions)
        distinct_result = complex_.transfer_periodic_quotient(target_complex)
    assert distinct_result.transfer_id == result.transfer_id
    assert distinct_result.lifted_transfer_result.evidence_id == result.lifted_transfer_result.evidence_id
    assert distinct_result.quotient_transfer_result.evidence_id == result.quotient_transfer_result.evidence_id
    original_matrix = de_rham._periodic_matrix
    matrices = []

    def tracked_matrix(operator: Any) -> Any:
        if matrices and len(matrices) % 4 == 0:
            assert all(reference() is None for reference in matrices[-4:])
        matrix = original_matrix(operator)
        matrices.append(weakref.ref(matrix))
        return matrix

    monkeypatch.setattr(de_rham, "_periodic_matrix", tracked_matrix)
    quotient = result.quotient_transfer_result
    replayed = de_rham._require_complex_map_commutes(
        quotient.complex_map, quotient.entry_evidence, quotient.integration_policy
    )
    assert _same_tree_content(replayed, quotient.commuting_evidence)
    for actual, expected in zip(replayed, quotient.commuting_evidence, strict=True):
        for name in (
            "target_indices", "source_indices", "observed_defects",
            "integration_error_bounds", "roundoff_error_bounds",
        ):
            np.testing.assert_array_equal(
                np.asarray(getattr(actual, name)).view(np.uint8),
                np.asarray(getattr(expected, name)).view(np.uint8),
            )
    assert all(reference() is None for reference in matrices)
    for degree in range(4):
        space = result.complex_map.source.space(degree)
        values = jnp.sin(jnp.arange(space.size, dtype=jnp.float64) + 0.39)
        np.testing.assert_allclose(
            result.map(degree).mv(values), values, atol=1e-10, rtol=1e-10
        )
    assert len(result.support_report()) == 4


def test_periodic_form_phases_preserve_full_transition_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fractions import Fraction
    from types import SimpleNamespace

    import phydrax as phx
    import weakref

    original_complex = phx.discretization.FiniteElementDeRhamComplex
    complex_refs = []

    def tracked_complex(*args: Any, **kwargs: Any) -> Any:
        complex_ = original_complex(*args, **kwargs)
        complex_refs.append(weakref.ref(complex_))
        return complex_

    monkeypatch.setattr(phx.discretization, "FiniteElementDeRhamComplex", tracked_complex)
    from phydrax.discretization._cell_geometry_transition_types import (
        CellGeometryTransition,
        CellGeometryTransitionEvidence,
    )
    from phydrax.discretization._nested_reference_types import ReferenceCellMapWitnesses
    from tools.meshing_qualification.scenarios.native_hybrid_profiles import (
        _layer_field_plan,
        _layer_form_transition,
    )

    coordinates = np.asarray(
        tuple((x, y, z) for z in (0, 1, 2) for x, y in ((0, 0), (1, 0), (0, 1))),
        dtype=np.float64,
    )
    blocks = (
        CellBlock("prisms", "prism", ((0, 1, 2, 3, 4, 5),), global_ids=(41,)),
        CellBlock(
            "tetrahedra", "tetrahedron", ((3, 4, 5, 6), (4, 5, 6, 7), (5, 7, 8, 6)),
            global_ids=(51, 52, 53),
        ),
    )
    lifted = CellMesh(coordinates, blocks)
    periodic = PeriodicMeshTopology(
        lifted, PeriodicCell(np.asarray(((0.0, 0.0, 2.0),))),
        np.asarray((0, 1, 2, 3, 4, 5, 0, 1, 2), dtype=np.int32),
        np.asarray(((0,), (0,), (0,), (0,), (0,), (0,), (1,), (1,), (1,)), dtype=np.int32),
    )
    mesh = CellMesh(coordinates, blocks, periodic_topology=periodic)
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    witnesses = []
    for block in blocks:
        vertices = tuple(
            tuple(Fraction(int(value)) for value in point)
            for point in phx.discretization.reference_cell_topology(block.cell_kind).vertices
        )
        corners = tuple(vertices for _ in range(block.cell_count))
        identifiers = np.asarray(block.global_ids)
        witnesses.append(ReferenceCellMapWitnesses(
            identifiers, identifiers, block.cell_kind, block.cell_kind,
            np.asarray(corners, dtype=np.float64), corners,
        ))
    transition = CellGeometryTransition(
        mesh, geometry, mesh, geometry, reference_map=tuple(witnesses),
        evidence=CellGeometryTransitionEvidence("nested_restriction", reference_map_count=4),
    )
    endpoint = SimpleNamespace(mesh=mesh, geometry=geometry)
    adaptation = SimpleNamespace(
        source=endpoint, target=endpoint,
        transition=SimpleNamespace(geometry_transition=transition),
    )
    declarations = _layer_field_plan(endpoint).fields
    banks = {}
    for spec in declarations:
        if spec.name in ("Hcurl", "Hdiv"):
            dofs = phx.discretization.FiniteElementDofMap(
                mesh, spec.resolve(mesh), component_shape=spec.component_shape,
                coordinate_spec=geometry,
            )
            banks[spec.name] = np.sin(np.arange(dofs.global_dof_count, dtype=np.float64))
    checkpoints = []
    def checkpoint(name: str, values: np.ndarray, record: Any) -> None:
        checkpoints.append((name, values.copy(), record))
        assert all(reference() is None for reference in complex_refs)

    full, full_records = _layer_form_transition(
        adaptation, declarations, declarations, banks, on_field=checkpoint, hodge_resources=None
    )
    assert tuple(item[0] for item in checkpoints) == ("Hcurl", "Hdiv")
    for name, values, record in checkpoints:
        np.testing.assert_array_equal(values, full[name])
        assert record == full_records[name]
        one, one_record = _layer_form_transition(
            adaptation, declarations, declarations, banks, fields=(name,), hodge_resources=None
        )
        np.testing.assert_array_equal(one[name], full[name])
        assert one_record[name] == full_records[name]
    for invalid in ((), ("Hcurl", "Hcurl"), ("H1",)):
        with pytest.raises(ValueError, match="field phases"):
            _layer_form_transition(adaptation, declarations, declarations, banks, fields=invalid, hodge_resources=None)
    with pytest.raises(ValueError, match="Hcurl and Hdiv"):
        _layer_form_transition(
            adaptation, tuple(item for item in declarations if item.name != "Hdiv"),
            declarations, banks, fields=("Hcurl",), hodge_resources=None,
        )
