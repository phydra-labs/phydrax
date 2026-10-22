# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Rational high-order interpolation, durable bindings and dynamic execution."""

import subprocess
import sys
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._model._structure import (
    model_from_array_recipe,
    model_recipe_array_values,
    model_structure_recipe,
)
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    reference_cell_topology,
)
from phydrax.discretization._cell_geometry import (
    coordinate_lagrange_element,
    coordinate_lagrange_reference_nodes,
    require_scalar_coordinate_element,
)
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization._exact_algebraic_geometry import rational_coefficient
from phydrax.discretization._exact_rational_geometry import (
    ExactRationalCellGeometrySource,
)
from phydrax.lifecycle._meshing_sources import (
    register_meshing_source_artifacts,
    validate_meshing_source_closure,
)
from phydrax.meshing._curving import straight_geometry
from phydrax.meshing._plc_mapped_support import _require_degree_one_image


@pytest.fixture
def budget() -> Iterator[CoordinateEnclosureBudget]:
    ticket = HostBatchTicket(10_000_000, bytes_upper=128_000_000)
    with (
        ticket,
        CoordinateEnclosureBudget(
            ticket, maximum_work_units=10_000_000, maximum_memory_bytes=128_000_000
        ).activate() as ledger,
    ):
        yield ledger


def _mesh(kind: str) -> CellMesh:
    nodes = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
    points = nodes * np.float64(0.7) + np.float64(0.1)
    return CellMesh(
        points,
        (CellBlock("cells", kind, np.arange(points.shape[0], dtype=np.int32)[None]),),
    )


@pytest.mark.parametrize(
    "first_import",
    (
        "import phydrax.discretization._exact_plc_geometry",
        "import phydrax.discretization._exact_rational_geometry",
        "import phydrax.discretization._cell_geometry_validity",
        "from phydrax.discretization import ExactPlcCellGeometrySource",
    ),
)
def test_exact_source_owners_support_cold_first_import(first_import: str) -> None:
    result = subprocess.run(
        (sys.executable, "-c", first_import),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "kind", ("triangle", "quadrilateral", "tetrahedron", "hexahedron", "prism", "pyramid")
)
@pytest.mark.parametrize("degree", (2, 3))
def test_nonbinary_exact_interpolation_preserves_degree_and_corner_proof(
    kind: str,
    degree: int,
    budget: CoordinateEnclosureBudget,
) -> None:
    mesh = _mesh(kind)
    parent = CellGeometrySpec.affine(mesh)
    geometry = straight_geometry(mesh, degree, source_geometry=parent)
    source = geometry.exact_source
    assert isinstance(source, ExactRationalCellGeometrySource)
    assert (
        require_scalar_coordinate_element(geometry.elements[0], "exact degree").degree
        == degree
    )
    bank = geometry.source_coordinates(budget=budget)
    original = parent.source_coordinates(budget=budget)
    for action, point in zip(source.action_rows(), bank, strict=True):
        assert point == tuple(
            sum(
                (Fraction(n, d) * original[index][axis] for index, n, d in action),
                Fraction(0),
            )
            for axis in range(mesh.coordinates.shape[1])
        )
    rounded = np.asarray(
        [[float(value) for value in row] for row in bank], dtype=np.float64
    )
    assert rounded.tobytes() == np.asarray(geometry.coordinates).tobytes()
    local = tuple(bank[int(index)] for index in np.asarray(geometry.geometry_dofs[0])[0])
    assert _require_degree_one_image(kind, geometry.elements[0], local, budget=budget) > 0
    assert any(
        Fraction(float(value)) != exact
        for row, numeric in zip(bank, rounded, strict=True)
        for exact, value in zip(row, numeric, strict=True)
    )
    # Reuse reconstructs the coefficient bank only once per request.
    before = budget.work_units
    assert source.prepare(geometry.coordinates, budget=budget).vertices == bank
    assert budget.work_units == before


def test_polynomial_differential_reuses_only_complete_request_authority(
    budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.discretization._cell_geometry import RestrictedCellGeometryElement
    from phydrax.discretization._cell_geometry_validity import (
        _prepare_polynomial_differential,
    )

    mesh = _mesh("tetrahedron")
    geometry = straight_geometry(mesh, 2, source_geometry=CellGeometrySpec.affine(mesh))
    element = require_scalar_coordinate_element(
        geometry.elements[0], "differential.element"
    )
    bank = geometry.source_coordinates(budget=budget)
    local = tuple(bank[int(index)] for index in np.asarray(geometry.geometry_dofs[0])[0])
    first = _prepare_polynomial_differential(
        element, local, "tetrahedron", 3, False, budget
    )
    work, retained = budget.work_units, budget.retained_basis_bytes
    repeated = _prepare_polynomial_differential(
        element, local, "tetrahedron", 3, False, budget
    )
    assert repeated is first
    assert budget.work_units == work
    assert budget.retained_basis_bytes == retained
    changed = ((local[0][0] + Fraction(1, 97), *local[0][1:]), *local[1:])
    foreign = _prepare_polynomial_differential(
        element, changed, "tetrahedron", 3, False, budget
    )
    assert foreign is not first
    assert foreign.polynomial != first.polynomial
    chart = RestrictedCellGeometryElement(
        element,
        "tetrahedron",
        np.eye(3, dtype=np.float64) / 2,
        np.zeros((3,), dtype=np.float64),
    )
    restricted = _prepare_polynomial_differential(
        chart, local, "tetrahedron", 3, False, budget
    )
    assert restricted is not first
    assert restricted.jacobian != first.jacobian


def test_exact_interpolation_refuses_curved_source_and_rounded_only_midpoints(
    budget: CoordinateEnclosureBudget,
) -> None:
    mesh = _mesh("tetrahedron")
    geometry = straight_geometry(mesh, 2, source_geometry=CellGeometrySpec.affine(mesh))
    bank = geometry.source_coordinates(budget=budget)
    route = np.asarray(geometry.geometry_dofs[0])[0]
    rounded = tuple(
        tuple(Fraction(float(value)) for value in row)
        for row in np.asarray(geometry.coordinates)[route]
    )
    with pytest.raises(ValueError, match="degree_one_exactness"):
        _require_degree_one_image(
            "tetrahedron", geometry.elements[0], rounded, budget=budget
        )
    curved = list(tuple(bank[int(index)] for index in route))
    interior = next(index for index, row in enumerate(curved) if row not in bank[:4])
    curved[interior] = (curved[interior][0] + Fraction(1, 100), *curved[interior][1:])
    with pytest.raises(ValueError, match="degree_one_exactness"):
        _require_degree_one_image(
            "tetrahedron", geometry.elements[0], tuple(curved), budget=budget
        )
    source = geometry.exact_source
    assert isinstance(source, ExactRationalCellGeometrySource)
    with pytest.raises(ValueError, match="representatives"):
        source.prepare(
            np.nextafter(np.asarray(geometry.coordinates), np.inf), budget=budget
        )
    foreign = eqx.tree_at(
        lambda value: value.parent.coordinates, source, source.parent.coordinates + 0.01
    )
    with pytest.raises(ValueError, match="authority_binding"):
        foreign.prepare(geometry.coordinates, budget=budget)
    with pytest.raises(ValueError, match="authority_binding"):
        ExactRationalCellGeometrySource.from_stored_fields(foreign)


def test_exact_interpolation_restore_retains_identity_and_shared_dynamic_bank(
    budget: CoordinateEnclosureBudget,
) -> None:
    mesh = _mesh("prism")
    geometry = straight_geometry(mesh, 3, source_geometry=CellGeometrySpec.affine(mesh))
    original_source = geometry.exact_source
    assert isinstance(original_source, ExactRationalCellGeometrySource)
    register_meshing_source_artifacts()
    recipe = model_structure_recipe((geometry, original_source.parent))
    arrays = model_recipe_array_values(
        (geometry, original_source.parent), recipe, prefix="exact-nodes"
    )
    restored, parent = model_from_array_recipe(recipe, arrays, prefix="exact-nodes")
    validate_meshing_source_closure(restored)
    source = restored.exact_source
    assert isinstance(source, ExactRationalCellGeometrySource)
    assert source.source_id == original_source.source_id
    assert source.parent is parent
    assert restored.geometry_layout_id == geometry.geometry_layout_id
    assert restored.source_coordinates(budget=budget) == geometry.source_coordinates(
        budget=budget
    )
    assert (
        np.asarray(restored.coordinates).tobytes()
        == np.asarray(geometry.coordinates).tobytes()
    )
    assert all(
        isinstance(value, jax.Array) for value in jax.tree_util.tree_leaves(source)
    )
    changed = eqx.tree_at(
        lambda value: value.coordinates, restored, restored.coordinates + 0.001
    )
    with pytest.raises(ValueError, match="representatives"):
        validate_meshing_source_closure(changed)


def test_geometry_restore_reproves_exact_source_on_actual_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.discretization._cell_geometry as geometry_owner

    mesh = _mesh("triangle")
    geometry = straight_geometry(mesh, 2, source_geometry=CellGeometrySpec.affine(mesh))
    source = geometry.exact_source
    assert isinstance(source, ExactRationalCellGeometrySource)

    def forbid_preparation(*args, **kwargs):
        raise AssertionError("explicit restoration must not open a helper preparation")

    monkeypatch.setattr(
        geometry_owner, "owned_coordinate_preparation", forbid_preparation
    )

    def restore(cap: int) -> int:
        memory = 128_000_000
        ticket = HostBatchTicket(cap, bytes_upper=memory)
        with (
            ticket,
            CoordinateEnclosureBudget(
                ticket, maximum_work_units=cap, maximum_memory_bytes=memory
            ).activate() as caller,
        ):
            restored = CellGeometrySpec.from_stored_fields(geometry, budget=caller)
            assert restored.exact_source is source
            assert restored.geometry_layout_id == geometry.geometry_layout_id
            np.testing.assert_array_equal(restored.coordinates, geometry.coordinates)
            assert restored.source_coordinates(budget=caller) == geometry.source_coordinates(
                budget=caller
            )
            return caller.work_units

    work = restore(10_000_000)
    assert work > 0
    assert restore(work) == work
    with pytest.raises(CoordinateEnclosureResourceError):
        restore(work - 1)
    changed = eqx.tree_at(
        lambda value: value.coordinates, geometry, geometry.coordinates + 0.001
    )
    memory = 128_000_000
    ticket = HostBatchTicket(10_000_000, bytes_upper=memory)
    with (
        ticket,
        CoordinateEnclosureBudget(
            ticket, maximum_work_units=10_000_000, maximum_memory_bytes=memory
        ).activate() as caller,
    ):
        with pytest.raises(ValueError, match="representatives"):
            CellGeometrySpec.from_stored_fields(changed, budget=caller)


def test_exact_interpolation_layout_orders_and_resource_refusals(
    budget: CoordinateEnclosureBudget,
) -> None:
    mesh = _mesh("tetrahedron")
    parent = CellGeometrySpec.affine(mesh)
    geometry = straight_geometry(mesh, 2, source_geometry=parent)
    source = geometry.exact_source
    assert isinstance(source, ExactRationalCellGeometrySource)
    with pytest.raises(ValueError, match="degrees from 1 through 10"):
        coordinate_lagrange_element("tetrahedron", 11)
    with pytest.raises(ValueError, match="coordinate families"):
        ExactRationalCellGeometrySource(
            parent,
            source.action_rows(),
            (("cells", "polyhedron", 2, "foreign", "foreign"),),
            maximum_work=1000,
            maximum_bits=4096,
        )
    with pytest.raises(ValueError, match="sorted unique"):
        ExactRationalCellGeometrySource(
            parent,
            (((0, 1, 2), (0, 1, 2)),),
            source.layouts,
            maximum_work=1000,
            maximum_bits=4096,
        )
    with pytest.raises(ValueError, match="ordered coefficient routes"):
        CellGeometrySpec(
            {"cells": geometry.elements[0]},
            {"cells": geometry.geometry_dofs[0][:, ::-1]},
            geometry.coordinates,
            exact_source=source,
            budget=budget,
        )
    with pytest.raises(CoordinateEnclosureResourceError):
        ExactRationalCellGeometrySource(
            parent,
            source.action_rows(),
            source.layouts,
            maximum_work=1,
            maximum_bits=4096,
        )
    ticket = HostBatchTicket(1_000_000, bytes_upper=1)
    with (
        ticket,
        CoordinateEnclosureBudget(
            ticket, maximum_work_units=1_000_000, maximum_memory_bytes=1
        ).activate() as narrow,
    ):
        with pytest.raises(CoordinateEnclosureResourceError):
            source.prepare(geometry.coordinates, budget=narrow)
    with pytest.raises(ValueError, match="resource limit"):
        straight_geometry(mesh, 3, source_geometry=parent, maximum_nodes=4)


def test_exact_source_keeps_runtime_coefficients_dynamic_for_jit_jvp_vjp() -> None:
    mesh = _mesh("tetrahedron")
    geometry = straight_geometry(mesh, 2, source_geometry=CellGeometrySpec.affine(mesh))
    element = require_scalar_coordinate_element(
        geometry.elements[0], "runtime derivative"
    )
    route = geometry.geometry_dofs[0][0]
    point = jnp.asarray(((0.2, 0.3, 0.1),), dtype=jnp.float64)

    @eqx.filter_jit
    def evaluate(value: CellGeometrySpec, xi: jax.Array) -> jax.Array:
        basis, _ = element.tabulate(xi)
        return basis @ value.coordinates[route]

    direction = jnp.ones_like(point)
    primal, tangent = jax.jvp(lambda xi: evaluate(geometry, xi), (point,), (direction,))
    _, pullback = jax.vjp(lambda xi: evaluate(geometry, xi), point)
    cotangent = jnp.ones_like(primal)
    np.testing.assert_allclose(
        jnp.sum(tangent * cotangent),
        jnp.sum(direction * pullback(cotangent)[0]),
        rtol=1e-14,
    )
    shifted = eqx.tree_at(
        lambda value: value.coordinates, geometry, geometry.coordinates + 1
    )
    np.testing.assert_allclose(
        evaluate(shifted, point) - primal, np.ones_like(primal), rtol=1e-14
    )


@pytest.mark.parametrize(
    "kind",
    (
        "interval",
        "triangle",
        "quadrilateral",
        "tetrahedron",
        "hexahedron",
        "prism",
        "pyramid",
    ),
)
def test_canonical_exact_nodes_preserve_all_existing_reference_bytes(kind: str) -> None:
    from itertools import product

    topology = reference_cell_topology(kind)
    for degree in range(1, 11):
        if degree == 1:
            previous = np.asarray(topology.vertices, dtype=np.float64)
        elif kind == "pyramid":
            previous = np.asarray(
                [
                    ((i + 0.5 * k) / degree, (j + 0.5 * k) / degree, k / degree)
                    for k in range(degree + 1)
                    for i in range(degree - k + 1)
                    for j in range(degree - k + 1)
                ],
                dtype=np.float64,
            )
        else:
            indices = tuple(product(range(degree + 1), repeat=topology.dimension))
            if kind in ("triangle", "tetrahedron"):
                indices = tuple(row for row in indices if sum(row) <= degree)
            elif kind == "prism":
                indices = tuple(row for row in indices if row[0] + row[1] <= degree)
            previous = np.asarray(indices, dtype=np.float64) / degree
        exact = coordinate_lagrange_reference_nodes(kind, degree)
        representatives = np.asarray(
            [[float(value) for value in row] for row in exact], dtype=np.float64
        )
        element = coordinate_lagrange_element(kind, degree)
        assert (
            previous.tobytes()
            == representatives.tobytes()
            == np.asarray(element.reference_nodes).tobytes()
        )


def test_odd_order_subnormal_actions_and_retained_storage_survive_phase_scope(
    budget: CoordinateEnclosureBudget,
) -> None:
    mesh = _mesh("tetrahedron")
    subnormal = np.nextafter(np.float64(0), np.float64(1))
    points = np.asarray(mesh.coordinates).copy()
    points[:, 0] = (subnormal, 2 * subnormal, 4 * subnormal, 7 * subnormal)
    # Geometry preparation requires no inverse of this intentionally thin cell.
    thin = CellMesh(points, mesh.blocks)
    geometry = straight_geometry(thin, 3, source_geometry=CellGeometrySpec.affine(thin))
    source = geometry.exact_source
    assert isinstance(source, ExactRationalCellGeometrySource)
    before = budget.retained_basis_bytes
    with budget.temporary_scope():
        bank = geometry.source_coordinates(budget=budget)
    assert budget.retained_basis_bytes > before
    assert any(
        rational.denominator.bit_length() > 1075
        for row in bank
        for value in row
        for rational in (rational_coefficient(value, role="fixture coordinate"),)
    )
    assert source.maximum_bits > 1075
    assert budget.retained_objects.get(id(bank)) is bank
    assert geometry.source_coordinates(budget=budget) is bank
    # An action allowance one unit short refuses before parent preparation.
    count = (
        2 * len(source.action_sources) + len(source.action_offsets) - 1
    ) * thin.coordinates.shape[1]
    short = ExactRationalCellGeometrySource(
        source.parent,
        source.action_rows(),
        source.layouts,
        maximum_work=count - 1,
        maximum_bits=source.maximum_bits,
    )
    ticket = HostBatchTicket(1_000_000, bytes_upper=128_000_000)
    with (
        ticket,
        CoordinateEnclosureBudget(
            ticket, maximum_work_units=1_000_000, maximum_memory_bytes=128_000_000
        ).activate() as narrow,
    ):
        with pytest.raises(CoordinateEnclosureResourceError):
            short.prepare(geometry.coordinates, budget=narrow)
        assert narrow.work_units == 0
        assert not narrow.source_bank_cache and not narrow.plc_witness_cache


def test_cold_exact_rational_archive_recreates_authority_sharing_and_rne(
    tmp_path: "Path",
    budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
    from phydrax.lifecycle import write_meshing_source_closure
    from tests._support.forced_devices import run_on_forced_devices

    mesh = _mesh("prism")
    parent = CellGeometrySpec.affine(mesh)
    geometry = straight_geometry(mesh, 3, source_geometry=parent)
    source = geometry.exact_source
    assert isinstance(source, ExactRationalCellGeometrySource)
    bank = geometry.source_coordinates(budget=budget)
    coefficients = canonical_fingerprint(
        tuple(
            tuple(
                (rational.numerator, rational.denominator)
                for value in row
                for rational in (rational_coefficient(value, role="fixture coordinate"),)
            )
            for row in bank
        )
    )
    representatives = canonical_fingerprint(array_tree_fingerprint(geometry.coordinates))
    receipt = write_meshing_source_closure(
        tmp_path / "exact-rational-source", (mesh, geometry, parent)
    )
    record = run_on_forced_devices(
        "tests._support.discretization_process_scenarios",
        "cold_exact_rational_geometry",
        (
            str(receipt.path),
            receipt.content_id,
            source.source_id,
            geometry.geometry_layout_id,
            coefficients,
            representatives,
        ),
        devices=1,
        env={},
        timeout_seconds=120,
    )
    for key in (
        "source_identity",
        "layout_identity",
        "parent_sharing",
        "exact_coefficients",
        "representative_bytes",
        "dynamic_arrays",
    ):
        assert record[key]
    assert record["retained_bytes"] > 0


@pytest.mark.parametrize("malformation", ("missing", "changed", "layout"))
def test_restored_exact_coefficients_authenticate_complete_actions(
    malformation: str,
) -> None:
    from tests._support.recipes import recipe_field_position

    mesh = _mesh("tetrahedron")
    source = straight_geometry(
        mesh, 2, source_geometry=CellGeometrySpec.affine(mesh)
    ).exact_source
    assert isinstance(source, ExactRationalCellGeometrySource)
    register_meshing_source_artifacts()
    recipe = model_structure_recipe(source)
    arrays = model_recipe_array_values(source, recipe, prefix="tampered-actions")
    if malformation == "layout":
        field, replacement = (
            "layouts",
            (
                (
                    source.layouts[0][0],
                    source.layouts[0][1],
                    source.layouts[0][2],
                    "foreign-element",
                    source.layouts[0][4],
                ),
            ),
        )
    else:
        field = "coefficients"
        replacement = (
            source.coefficients[:-1]
            if malformation == "missing"
            else (source.coefficients[0] + Fraction(1, 100), *source.coefficients[1:])
        )
    recipe["items"][recipe_field_position(recipe, field)] = model_structure_recipe(
        replacement
    )
    restored = model_from_array_recipe(recipe, arrays, prefix="tampered-actions")
    with pytest.raises(ValueError, match="authority_binding|coefficients|actions"):
        validate_meshing_source_closure(restored)


def test_nested_prism_restriction_preserves_exact_high_order_bank_and_restore(
    tmp_path: Path,
    budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.discretization._cell_geometry_transfer import (
        transition_nested_cell_geometry,
    )
    from phydrax.lifecycle import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )
    from phydrax.meshing._mixed_adaptation import adapt_mixed_mesh
    from phydrax.meshing._topology_edit import assemble_topology_edit

    mesh = _mesh("prism")
    parent = CellGeometrySpec.affine(mesh)
    geometry = straight_geometry(mesh, 2, source_geometry=parent)
    old_source = geometry.exact_source
    assert isinstance(old_source, ExactRationalCellGeometrySource)
    outcome = adapt_mixed_mesh(
        mesh, refine_cell_ids=np.asarray(mesh.blocks[0].global_ids)
    )
    target, _, _ = assemble_topology_edit(
        mesh, outcome.edit, numeric_version="refined", budget=budget
    )
    transition = transition_nested_cell_geometry(
        mesh,
        geometry,
        target,
        CellGeometrySpec.affine(target),
        refinement=outcome.edit.refinement,
        coarsening=outcome.edit.coarsening,
        budget=budget,
    )
    target = target.with_coordinates(
        transition.vertex_coordinates, numeric_version="restricted-source"
    )
    exact = transition.geometry.exact_source
    assert isinstance(exact, ExactRationalCellGeometrySource)
    assert exact.parent is parent and exact.parent_binding == old_source.parent_binding
    assert exact.source_id != old_source.source_id
    transition.geometry.resolve(target, budget=budget)
    bank = transition.geometry.source_coordinates(budget=budget)
    for element, route in zip(
        transition.geometry.elements, transition.geometry.geometry_dofs, strict=True
    ):
        assert require_scalar_coordinate_element(element, "nested degree").degree == 2
        for row in np.asarray(route):
            local = tuple(bank[int(index)] for index in row)
            _require_degree_one_image(element.cell_kind, element, local, budget=budget)
    receipt = write_meshing_source_closure(
        tmp_path / "nested-exact-source", (transition.geometry, parent)
    )
    restored, restored_parent = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert restored.exact_source.parent is restored_parent
    assert restored.source_coordinates(budget=budget) == bank


@pytest.mark.parametrize("displaced", (False, True))
def test_rational_nodes_use_exact_plc_authority_not_rounded_parent_vertices(
    budget: CoordinateEnclosureBudget,
    displaced: bool,
) -> None:
    from phydrax.discretization import ExactPlcCellGeometrySource

    retained_x = (
        np.nextafter(np.float64(1), np.float64(2)) if displaced else np.float64(1)
    )
    mesh = CellMesh.from_triangles(
        np.asarray(((0, 0), (retained_x, 1 / 3), (0, 1)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    source = ExactPlcCellGeometrySource(
        np.asarray(((0, 0), (3, 1), (0, 1)), dtype=np.float64),
        np.empty((0, 3), dtype=np.int64),
        np.asarray(((0, 1),), dtype=np.int64),
        np.asarray((0, 1, 0), dtype=np.int8),
        np.asarray((-1, 0, -1), dtype=np.int64),
        np.asarray(((0, 0), (1 / 3, 0), (0, 0)), dtype=np.float64),
        domain_source_id="exact-rational-node-plc",
        domain_source_revision="authored-third",
        source_triangle_ids=np.empty((0,), dtype=np.int64),
        source_triangle_bounds=np.empty((0,), dtype=np.float64),
        source_segment_ids=np.asarray((0,), dtype=np.int64),
        source_segment_bounds=np.asarray((1e-15,), dtype=np.float64),
        vertex_parameter_ratios=(((0, 1), (0, 1)), ((1, 3), (0, 1)), ((0, 1), (0, 1))),
        budget=budget,
    )
    parent = CellGeometrySpec.plc(mesh, source, budget=budget)
    geometry = straight_geometry(mesh, 3, source_geometry=parent)
    bank = geometry.source_coordinates(budget=budget)
    assert (Fraction(1, 3), Fraction(1, 9)) in bank
    assert Fraction(1, 9) != Fraction(float(Fraction(float(1 / 3)) / 3))
    local = tuple(bank[int(index)] for index in np.asarray(geometry.geometry_dofs[0])[0])
    _require_degree_one_image("triangle", geometry.elements[0], local, budget=budget)
    exact = geometry.exact_source
    assert isinstance(exact, ExactRationalCellGeometrySource)
    assert (
        np.asarray(geometry.coordinates)[:3].tobytes()
        == np.asarray(mesh.coordinates).tobytes()
    )
    error = abs(Fraction(float(retained_x)) - 1)
    assert (
        exact.prepare(geometry.coordinates, budget=budget).maximum_rounding_error >= error
    )
    if displaced:
        assert geometry.source_execution_error(mesh, budget=budget) >= float(error)
        changed = np.asarray(geometry.coordinates).copy()
        changed[1, 0] = 1
        with pytest.raises(ValueError, match="representatives"):
            exact.prepare(changed, budget=budget)


@pytest.mark.parametrize("kind", ("tetrahedron", "prism"))
def test_root_authority_requires_full_rational_law_not_only_corner_agreement(
    kind: str,
    budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.discretization._cell_geometry import (
        equivalent_coordinate_root,
        RestrictedCellGeometryElement,
    )

    mesh = _mesh(kind)
    parent = CellGeometrySpec.affine(mesh)
    geometry = straight_geometry(mesh, 2, source_geometry=parent)
    bank = geometry.source_coordinates(budget=budget)
    local = tuple(bank[int(index)] for index in np.asarray(geometry.geometry_dofs[0])[0])
    original = parent.source_coordinates(budget=budget)
    root = equivalent_coordinate_root(
        geometry.elements[0], local, parent.elements[0], original, budget=budget
    )
    assert root is geometry.elements[0]
    scalar_root = require_scalar_coordinate_element(root, "root degree")
    assert scalar_root.degree == 2
    restricted = RestrictedCellGeometryElement(
        scalar_root, kind, np.eye(3, dtype=np.float64) / 2, np.zeros(3, dtype=np.float64)
    )
    assert (
        equivalent_coordinate_root(
            restricted, local, parent.elements[0], original, budget=budget
        )
        is root
    )
    curved = list(local)
    node = next(index for index, point in enumerate(local) if point not in original)
    curved[node] = (curved[node][0] + Fraction(1, 100), *curved[node][1:])
    assert (
        equivalent_coordinate_root(
            restricted, tuple(curved), parent.elements[0], original, budget=budget
        )
        is None
    )


@pytest.mark.parametrize("degree", (2, 3))
def test_identity_rows_preserve_signed_zero_and_generated_rows_require_positive_zero(
    degree: int,
    tmp_path: Path,
    budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.lifecycle import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )

    points = np.asarray(
        ((-0.0, -0.0, 0.0), (0.7, -0.0, 0.0), (0.0, 0.9, 0.0), (0.0, 0.0, 1.1)),
        dtype=np.float64,
    )
    mesh = CellMesh.from_tetrahedra(points, np.asarray(((0, 1, 2, 3),), dtype=np.int32))
    parent = CellGeometrySpec.affine(mesh)
    geometry = straight_geometry(mesh, degree, source_geometry=parent)
    source = geometry.exact_source
    assert isinstance(source, ExactRationalCellGeometrySource)
    numeric = np.asarray(geometry.coordinates)
    assert numeric[:4].tobytes() == points.tobytes()
    generated = next(
        row
        for row, action in enumerate(source.action_rows())
        if len(action) > 1 and numeric[row, 2] == 0
    )
    assert not np.signbit(numeric[generated, 2])
    for row, axis, value in ((0, 0, 0.0), (generated, 2, -0.0)):
        changed = numeric.copy()
        changed[row, axis] = value
        with pytest.raises(ValueError, match="representatives"):
            source.prepare(changed, budget=budget)
    receipt = write_meshing_source_closure(tmp_path / "signed-zero-source", geometry)
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert np.asarray(restored.coordinates).tobytes() == numeric.tobytes()
    assert restored.source_coordinates(budget=budget) == geometry.source_coordinates(
        budget=budget
    )


def test_exact_source_public_facade_imports_before_other_geometry_owners() -> None:
    from tests._support.forced_devices import run_on_forced_devices

    record = run_on_forced_devices(
        "tests._support.exact_rational_startup",
        "source_facade_first",
        (),
        devices=1,
        env={},
        timeout_seconds=60,
    )
    assert record["source_available"]


@pytest.mark.parametrize("scale", (1.0, 1e-100, 1e-200, 1e-300, 1e300))
def test_actual_high_order_source_error_encloses_independent_exact_bernstein_bound(
    scale: float,
    budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.discretization import ExactPlcCellGeometrySource

    # Author a nonbinary third at every scale, including mantissas divisible by three.
    ordinate = (
        scale
        if Fraction(scale).numerator % 3
        else float(np.nextafter(np.float64(scale), np.inf))
    )
    source_points = np.asarray(
        ((0, 0), (3 * scale, ordinate), (0, scale)), dtype=np.float64
    )
    exact_vertex = tuple(Fraction(float(value)) / 3 for value in source_points[1])
    carrier = np.asarray(
        ((0, 0), tuple(float(value) for value in exact_vertex), (0, scale)),
        dtype=np.float64,
    )
    mesh = CellMesh.from_triangles(carrier, np.asarray(((0, 1, 2),), dtype=np.int32))
    source = ExactPlcCellGeometrySource(
        source_points,
        np.empty((0, 3), dtype=np.int64),
        np.asarray(((0, 1),), dtype=np.int64),
        np.asarray((0, 1, 0), dtype=np.int8),
        np.asarray((-1, 0, -1), dtype=np.int64),
        np.asarray(((0, 0), (1 / 3, 0), (0, 0)), dtype=np.float64),
        domain_source_id="scaled-rational-node-plc",
        domain_source_revision="authored-third",
        source_triangle_ids=np.empty((0,), dtype=np.int64),
        source_triangle_bounds=np.empty((0,), dtype=np.float64),
        source_segment_ids=np.asarray((0,), dtype=np.int64),
        source_segment_bounds=np.asarray(
            (16 * np.spacing(np.float64(scale)),), dtype=np.float64
        ),
        vertex_parameter_ratios=(((0, 1), (0, 1)), ((1, 3), (0, 1)), ((0, 1), (0, 1))),
        budget=budget,
    )
    parent = CellGeometrySpec.plc(mesh, source, budget=budget)
    geometry = straight_geometry(mesh, 2, source_geometry=parent)
    bank = geometry.source_coordinates(budget=budget)
    route = np.asarray(geometry.geometry_dofs[0])[0]
    numeric = np.asarray(geometry.coordinates)
    errors = tuple(
        tuple(
            bank[int(row)][axis] - Fraction(float(numeric[row, axis]))
            for axis in range(2)
        )
        for row in route
    )
    # Independent P2 nodal-to-Bernstein formula in the triangle node order.
    vertices = (0, 5, 2)
    edges = ((3, 0, 5), (4, 5, 2), (1, 2, 0))
    radii = tuple(
        max(
            *(abs(errors[row][axis]) for row in vertices),
            *(
                abs(
                    2 * errors[midpoint][axis]
                    - (errors[left][axis] + errors[right][axis]) / 2
                )
                for midpoint, left, right in edges
            ),
        )
        for axis in range(2)
    )
    squared = sum((value * value for value in radii), Fraction(0))
    result = geometry.source_execution_error(mesh, budget=budget)
    assert squared > 0 and np.isfinite(result)
    assert Fraction(result) ** 2 >= squared
    assert all(
        Fraction(result) ** 2 >= sum((value * value for value in row), Fraction(0))
        for row in errors
    )


@pytest.mark.parametrize(
    "malformation",
    (
        "offset-dtype",
        "offset-rank",
        "offset-start",
        "offset-end",
        "offset-empty-row",
        "source-negative",
        "source-foreign",
        "source-repeat",
        "weight-negative",
        "weight-foreign",
        "weight-coverage",
    ),
)
def test_csr_restore_rejects_noncanonical_routes(malformation: str) -> None:
    mesh = _mesh("tetrahedron")
    source = straight_geometry(
        mesh, 2, source_geometry=CellGeometrySpec.affine(mesh)
    ).exact_source
    assert isinstance(source, ExactRationalCellGeometrySource)
    offsets = np.asarray(source.action_offsets).copy()
    sources = np.asarray(source.action_sources).copy()
    weights = np.asarray(source.action_weights).copy()
    match malformation:
        case "offset-dtype":
            offsets = offsets.astype(np.int32)
        case "offset-rank":
            offsets = offsets[None]
        case "offset-start":
            offsets[0] = 1
        case "offset-end":
            offsets[-1] -= 1
        case "offset-empty-row":
            offsets[1] = 0
        case "source-negative":
            sources[0] = -1
        case "source-foreign":
            sources[0] = source.parent.coordinates.shape[0]
        case "source-repeat":
            start = int(offsets[np.flatnonzero(np.diff(offsets) > 1)[0]])
            sources[start + 1] = sources[start]
        case "weight-negative":
            weights[0] = -1
        case "weight-foreign":
            weights[0] = len(source.coefficients)
        case "weight-coverage":
            weights[:] = 0
        case _:
            raise ValueError(malformation)
    malformed = eqx.tree_at(
        lambda value: (value.action_offsets, value.action_sources, value.action_weights),
        source,
        tuple(jax.device_put(value) for value in (offsets, sources, weights)),
    )
    with pytest.raises(ValueError, match="actions"):
        malformed.action_rows()
    with pytest.raises(ValueError, match="actions"):
        ExactRationalCellGeometrySource.from_stored_fields(malformed)


def test_csr_storage_retains_only_dynamic_integer_routes_and_unique_exact_bank() -> None:
    mesh = _mesh("prism")
    source = straight_geometry(
        mesh, 3, source_geometry=CellGeometrySpec.affine(mesh)
    ).exact_source
    assert isinstance(source, ExactRationalCellGeometrySource)
    rows = source.action_rows()
    assert source.coefficients == tuple(
        sorted({Fraction(n, d) for row in rows for _, n, d in row})
    )
    assert tuple(np.asarray(source.action_offsets)) == tuple(
        np.cumsum((0, *(len(row) for row in rows)))
    )
    assert all(
        value.dtype == jnp.int64
        for value in (source.action_offsets, source.action_sources, source.action_weights)
    )
    leaves = jax.tree_util.tree_leaves(source)
    assert all(isinstance(value, jax.Array) for value in leaves)
    assert all(
        any(value is leaf for leaf in leaves)
        for value in (source.action_offsets, source.action_sources, source.action_weights)
    )


@pytest.fixture(scope="module")
def query_consuming_parent() -> tuple[CellMesh, CellGeometrySpec, int]:
    from phydrax.discretization import ExactPlcCellGeometrySource
    from phydrax.discretization._exact_plc_geometry import (
        ExactPlcCellGeometryConvexSource,
    )

    mesh = _mesh("tetrahedron")
    with HostBatchTicket(
        10_000_000, queries_upper=100, bytes_upper=128_000_000
    ) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=10_000_000, maximum_memory_bytes=128_000_000
        ).activate() as budget:
            root = ExactPlcCellGeometrySource(
                mesh.coordinates,
                np.empty((0, 3), dtype=np.int64),
                np.empty((0, 2), dtype=np.int64),
                np.zeros(4, dtype=np.int8),
                np.full(4, -1, dtype=np.int64),
                np.zeros((4, 2), dtype=np.float64),
                domain_source_id="rational-lift-query-parent",
                domain_source_revision="identity-convex-witness",
                source_triangle_ids=np.empty(0, dtype=np.int64),
                source_triangle_bounds=np.empty(0, dtype=np.float64),
                source_segment_ids=np.empty(0, dtype=np.int64),
                source_segment_bounds=np.empty(0, dtype=np.float64),
                budget=budget,
            )
            parent = CellGeometrySpec.plc(mesh, root, budget=budget)
            convex = ExactPlcCellGeometryConvexSource(
                mesh,
                parent,
                np.arange(5, dtype=np.int64),
                np.arange(4, dtype=np.int64),
                (Fraction(1),) * 4,
                mesh,
                np.asarray(mesh.blocks[0].global_ids),
                budget=budget,
            )
            geometry = CellGeometrySpec.plc(mesh, convex, budget=budget)
    with HostBatchTicket(
        10_000_000, queries_upper=100, bytes_upper=128_000_000
    ) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=10_000_000, maximum_memory_bytes=128_000_000
        ).activate() as budget:
            straight_geometry(mesh, 2, source_geometry=geometry, budget=budget)
        queries = ticket.geometry_queries
    assert queries > 0
    return mesh, geometry, queries


@pytest.mark.parametrize("short", (False, True), ids=("at-limit", "one-query-short"))
def test_lifting_queries_charge_the_remaining_original_ticket(
    query_consuming_parent: tuple[CellMesh, CellGeometrySpec, int], short: bool
) -> None:
    from phydrax._resource_ticket import HostBatchEstimateError

    mesh, parent, queries = query_consuming_parent
    with HostBatchTicket(
        10_000_000, queries_upper=3 + queries - int(short), bytes_upper=128_000_000
    ) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=10_000_000, maximum_memory_bytes=128_000_000
        ).activate() as budget:
            budget.reserve_geometry_queries(3)
            if short:
                with pytest.raises(HostBatchEstimateError):
                    with budget.measurement() as phase:
                        straight_geometry(mesh, 2, source_geometry=parent, budget=budget)
                assert ticket.geometry_queries == 3 + phase.geometry_queries
                assert phase.geometry_queries < queries
            else:
                with budget.measurement() as phase:
                    geometry = straight_geometry(
                        mesh, 2, source_geometry=parent, budget=budget
                    )
                assert geometry.exact_source is not None
                assert phase.geometry_queries == queries
                assert ticket.geometry_queries == queries + 3
                assert ticket.work_units == budget.work_units > 0
