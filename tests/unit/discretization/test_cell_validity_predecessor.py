# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Request-owned validity imports retain full certification evidence."""

from collections.abc import Callable
from dataclasses import replace
from typing import Literal, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax._geometry_predicates import PolygonSimplicityResult, PredicateMode
from phydrax.discretization import (
    _cell_geometry_predecessor as predecessor_owner,
    _cell_geometry_validity as validity_owner,
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    CellValidityCertificate,
    CellValidityPolicy,
    CellValidityStatus,
    certify_cell_geometry_validity,
    fem,
    reference_cell_topology,
)
from phydrax.discretization._cell_geometry_predecessor import (
    prepare_cell_geometry_predecessor,
)
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.geometry._embedding_state import MeshCertificateLimits
from phydrax.meshing._curving import straight_geometry
from tests._support.coordinate_references import coordinate_reference_budget


def _observed_hash[**P](
    function: Callable[P, str], observe: Callable[[], None]
) -> Callable[P, str]:
    def observed(*args: P.args, **kwargs: P.kwargs) -> str:
        observe()
        return function(*args, **kwargs)

    return observed


def _interleaved_predecessor[**P](
    function: Callable[P, predecessor_owner.CellGeometryPredecessor],
    after: Callable[
        [predecessor_owner.CellGeometryPredecessor],
        predecessor_owner.CellGeometryPredecessor,
    ],
) -> Callable[P, predecessor_owner.CellGeometryPredecessor]:
    def interleaved(
        *args: P.args, **kwargs: P.kwargs
    ) -> predecessor_owner.CellGeometryPredecessor:
        return after(function(*args, **kwargs))

    return interleaved


def _triangle_geometry(
    change: Literal["none", "invalid", "unresolved"],
    *,
    reorder: bool = False,
) -> tuple[CellMesh, CellGeometrySpec]:
    element = fem.lagrange_element("triangle", 2)
    reference = np.asarray(element.reference_nodes, dtype=np.float64)
    local = np.stack([reference + (3.0 * cell, 0.0) for cell in range(3)])
    if change == "invalid":
        local[1, :, 0] = 6.0 - local[1, :, 0]
    elif change == "unresolved":
        # The established curved triangle needs subdivision to prove positivity.
        local[1, 3:] = np.asarray(
            ((0.86, -0.17), (0.54, 0.29), (-0.06, 0.51)), dtype=np.float64
        ) + (3.0, 0.0)
    vertices = np.asarray(reference_cell_topology("triangle").vertices, dtype=np.float64)
    points = np.concatenate([vertices + (3.0 * cell, 0.0) for cell in range(3)])
    if change == "invalid":
        points[3:6, 0] = 6.0 - points[3:6, 0]
    cells = np.arange(9, dtype=np.int32).reshape((3, 3))
    routes = np.arange(local.size // 2, dtype=np.int32).reshape((3, -1))
    first = np.asarray((1, 0) if reorder else (0, 1), dtype=np.int32)
    blocks = (
        CellBlock("first", "triangle", cells[first], global_ids=101 + first),
        CellBlock("second", "triangle", cells[2:], global_ids=np.asarray((103,))),
    )
    mesh = CellMesh(points, blocks[::-1] if reorder else blocks)
    geometry = CellGeometrySpec(
        {"first": element, "second": element},
        {"first": routes[first], "second": routes[2:]},
        local.reshape((-1, 2)),
    )
    return mesh, geometry


def _polygon_geometry(*, changed: bool) -> tuple[CellMesh, CellGeometrySpec]:
    vertices = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.5, 0.5), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
    )
    points = np.concatenate([vertices + (3.0 * cell, 0.0) for cell in range(3)])
    if changed:
        points[5:10, 0] = 7.5 - points[5:10, 0]
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "cells",
                "polygon",
                np.arange(15, dtype=np.int32).reshape((3, 5)),
            ),
        ),
    )
    return mesh, CellGeometrySpec.affine(mesh)


def _assert_same_certificate(
    full: CellValidityCertificate, inherited: CellValidityCertificate
) -> None:
    np.testing.assert_array_equal(inherited.status, full.status)
    np.testing.assert_array_equal(inherited.determinant_lower, full.determinant_lower)
    np.testing.assert_array_equal(inherited.determinant_upper, full.determinant_upper)
    np.testing.assert_array_equal(inherited.depth, full.depth)
    for name in ("status", "determinant_lower", "determinant_upper", "depth"):
        assert np.asarray(getattr(inherited, name)).tobytes() == np.asarray(
            getattr(full, name)
        ).tobytes()
    assert inherited.block_names == full.block_names
    assert inherited.block_offsets == full.block_offsets
    assert inherited.unsupported_block_names == full.unsupported_block_names
    assert inherited.unresolved_reasons == full.unresolved_reasons
    assert inherited.certified_valid_count == full.certified_valid_count
    assert inherited.invalid_count == full.invalid_count
    assert inherited.unresolved_count == full.unresolved_count
    assert inherited.certificate_id == full.certificate_id


@pytest.mark.parametrize("kind", ("quadrilateral", "hexahedron"))
def test_rational_affine_chart_import_preserves_exact_nonbinary_restriction(
    kind: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fractions import Fraction

    from phydrax.discretization._cell_geometry import (
        CellGeometryRestrictionSource,
        coordinate_lagrange_element,
        PolynomialComposedCellGeometryElement,
        require_scalar_coordinate_element,
    )

    vertices = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
    source_mesh = CellMesh(vertices, (
        CellBlock("root", kind, np.arange(len(vertices)).reshape(1, -1), global_ids=np.asarray((101,))),
    ))
    source_geometry = CellGeometrySpec.affine(source_mesh)
    source_element = require_scalar_coordinate_element(source_geometry.elements[0], "root")
    source = certify_cell_geometry_validity(source_geometry, mesh=source_mesh)
    chart = coordinate_lagrange_element(kind, 1)
    target_mesh = CellMesh(vertices / 3, (
        CellBlock("child", kind, np.arange(len(vertices)).reshape(1, -1), global_ids=np.asarray((202,))),
    ))
    with coordinate_reference_budget() as budget:
        element = PolynomialComposedCellGeometryElement(
            source_element, chart,
            np.asarray(chart.reference_nodes, dtype=np.int64),
            np.full(np.asarray(chart.reference_nodes).shape, 3, dtype=np.int64),
            budget=budget,
        )
        geometry = CellGeometrySpec(
            {"child": element}, {"child": source_geometry.geometry_dofs[0]},
            source_geometry.coordinates,
            restriction_source=CellGeometryRestrictionSource(
                validity_owner.cell_geometry_id(source_geometry), source_mesh.topology_id,
                {"child": np.asarray((101,))},
                {"child": np.asarray(source_mesh.vertex_global_ids)[None, :]},
            ),
            budget=budget,
        )
        action = validity_owner._restriction_parent_chart(
            element, source_element, budget=budget
        )
        assert action is not None
        assert action[0][0][0] == Fraction(1, 3)
        assert action[0][0][0] != Fraction(float(1 / 3))
        predecessor = prepare_cell_geometry_predecessor(
            source_mesh, source_geometry, source, None, target_mesh, geometry, budget=budget
        )
        fresh = certify_cell_geometry_validity(geometry, mesh=target_mesh, budget=budget)

        def no_expansion(*args: object, **kwargs: object) -> None:
            raise AssertionError("A proved rational child must use the parent theorem")

        monkeypatch.setattr(validity_owner, "_certify_polynomial_block", no_expansion)
        inherited = certify_cell_geometry_validity(
            geometry, mesh=target_mesh, predecessor=predecessor, budget=budget
        )
        _assert_same_certificate(fresh, inherited)
        assert inherited.all_certified


@pytest.mark.parametrize(
    ("change", "expected"),
    (
        ("none", CellValidityStatus.CERTIFIED_VALID),
        ("invalid", CellValidityStatus.INVALID),
        ("unresolved", CellValidityStatus.UNRESOLVED),
    ),
)
def test_predecessor_preserves_polynomial_proof_and_target_block_order(
    change: Literal["none", "invalid", "unresolved"],
    expected: CellValidityStatus,
) -> None:
    source_mesh, source_geometry = _triangle_geometry("none")
    target_mesh, target_geometry = _triangle_geometry(change, reorder=True)
    policy = CellValidityPolicy(maximum_subdivision_depth=0)
    source = certify_cell_geometry_validity(
        source_geometry, mesh=source_mesh, policy=policy
    )
    with coordinate_reference_budget() as preparation:
        predecessor = prepare_cell_geometry_predecessor(
            source_mesh,
            source_geometry,
            source,
            None,
            target_mesh,
            target_geometry,
            budget=preparation,
        )
    assert predecessor.source_rows == ((2, 1, 0) if change == "none" else (2, -1, 0))
    with coordinate_reference_budget() as full_budget:
        full = certify_cell_geometry_validity(
            target_geometry, mesh=target_mesh, policy=policy, budget=full_budget
        )
    with coordinate_reference_budget() as inherited_budget:
        # Cold authentication and current-owner guards remain fully charged.
        # This small fixture proves certificate/block-order equivalence, not
        # an end-to-end cold-work improvement or a request-cap qualification.
        predecessor.require_bound(
            target_mesh, target_geometry, budget=inherited_budget
        )
        paid_prefix = inherited_budget.work_units
        with inherited_budget.measurement() as inherited_phase:
            inherited = certify_cell_geometry_validity(
                target_geometry,
                mesh=target_mesh,
                policy=policy,
                budget=inherited_budget,
                predecessor=predecessor,
            )
        assert inherited_budget.work_units == paid_prefix + inherited_phase.work_units
    _assert_same_certificate(full, inherited)
    assert inherited.block_names == ("second", "first")
    assert inherited.block_offsets == (0, 1, 3)
    assert tuple(np.asarray(inherited.status)) == (
        CellValidityStatus.CERTIFIED_VALID,
        expected,
        CellValidityStatus.CERTIFIED_VALID,
    )
    assert inherited_phase.work_units > 0


def test_predecessor_certifies_changed_polygon_without_rechecking_unchanged_loops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_mesh, source_geometry = _polygon_geometry(changed=False)
    target_mesh, target_geometry = _polygon_geometry(changed=True)
    source = certify_cell_geometry_validity(source_geometry, mesh=source_mesh)
    with coordinate_reference_budget() as preparation:
        predecessor = prepare_cell_geometry_predecessor(
            source_mesh,
            source_geometry,
            source,
            None,
            target_mesh,
            target_geometry,
            budget=preparation,
        )
    assert predecessor.source_rows == (0, -1, 2)
    counted_rows = 0
    predicate = validity_owner.polygon_simplicity_2d

    def counted(
        vertices: np.ndarray, /, *, mode: PredicateMode, budget: CoordinateEnclosureBudget
    ) -> PolygonSimplicityResult:
        nonlocal counted_rows
        counted_rows += vertices.shape[0]
        return predicate(vertices, mode=mode, budget=budget)

    monkeypatch.setattr(validity_owner, "polygon_simplicity_2d", counted)
    with coordinate_reference_budget() as full_budget:
        full = certify_cell_geometry_validity(
            target_geometry, mesh=target_mesh, budget=full_budget
        )
    full_rows = counted_rows
    counted_rows = 0
    with coordinate_reference_budget() as inherited_budget:
        inherited = certify_cell_geometry_validity(
            target_geometry,
            mesh=target_mesh,
            budget=inherited_budget,
            predecessor=predecessor,
        )
    _assert_same_certificate(full, inherited)
    assert inherited.invalid_count == 1
    assert int(inherited.status[1]) == CellValidityStatus.INVALID
    assert counted_rows < full_rows


@pytest.mark.parametrize(
    "select",
    (
        pytest.param(lambda value: value.status, id="status"),
        pytest.param(lambda value: value.determinant_lower, id="determinant-lower"),
        pytest.param(lambda value: value.determinant_upper, id="determinant-upper"),
        pytest.param(lambda value: value.depth, id="depth"),
    ),
)
def test_predecessor_refuses_tampered_source_payload_even_with_warm_proof_cache(
    select: Callable[[CellValidityCertificate], Array],
) -> None:
    mesh, geometry = _triangle_geometry("none")
    source = certify_cell_geometry_validity(geometry, mesh=mesh)
    with coordinate_reference_budget() as preparation:
        predecessor = prepare_cell_geometry_predecessor(
            mesh, geometry, source, None, mesh, geometry, budget=preparation
        )
    payload = select(source)
    altered = eqx.tree_at(select, source, payload.at[0].add(1))
    assert altered.certificate_id == source.certificate_id
    tampered = replace(predecessor, source_validity=altered)
    with coordinate_reference_budget() as budget:
        certify_cell_geometry_validity(geometry, mesh=mesh, budget=budget)
        with pytest.raises(ValueError, match="CellGeometryPredecessor.content_id"):
            certify_cell_geometry_validity(
                geometry, mesh=mesh, budget=budget, predecessor=tampered
            )


def test_predecessor_refuses_another_validity_policy() -> None:
    mesh, geometry = _triangle_geometry("none")
    source = certify_cell_geometry_validity(geometry, mesh=mesh)
    with coordinate_reference_budget() as preparation:
        predecessor = prepare_cell_geometry_predecessor(
            mesh, geometry, source, None, mesh, geometry, budget=preparation
        )
    with pytest.raises(ValueError, match="predecessor.policy_id"):
        certify_cell_geometry_validity(
            geometry,
            mesh=mesh,
            policy=CellValidityPolicy(maximum_subdivision_depth=0),
            predecessor=predecessor,
        )


def test_selected_predecessor_content_keys_preserve_identity_and_row_order() -> None:
    from phydrax.discretization._cell_geometry_predecessor import _cell_content_keys

    mesh, geometry = _triangle_geometry("none", reorder=True)
    with coordinate_reference_budget() as budget:
        complete = _cell_content_keys(mesh, geometry, budget)
    with coordinate_reference_budget() as budget:
        selected = _cell_content_keys(
            mesh, geometry, budget, selected_ids=np.asarray((101, 103), dtype=np.int64)
        )
    assert selected == tuple(key for key in complete if key[0] in (101, 103))
    assert tuple(key[0] for key in selected) == (103, 101)


@pytest.mark.parametrize("nonbinary", (False, True), ids=("immutable", "nonbinary"))
def test_predecessor_earned_content_reuses_actual_owners_on_one_ledger(
    nonbinary: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    mesh, geometry = _triangle_geometry("none")
    if nonbinary:
        mesh = CellMesh.from_triangles(
            np.asarray(((0.1, 0.1), (0.8, 0.1), (0.1, 0.8))),
            np.asarray(((0, 1, 2),), dtype=np.int32),
        )
        geometry = straight_geometry(
            mesh, 3, source_geometry=CellGeometrySpec.affine(mesh)
        )
        assert geometry.exact_source is not None
    source = certify_cell_geometry_validity(geometry, mesh=mesh)
    hashes = 0
    original: Callable[..., str] = predecessor_owner._predecessor_content_id

    def counted() -> None:
        nonlocal hashes
        hashes += 1

    monkeypatch.setattr(
        predecessor_owner, "_predecessor_content_id", _observed_hash(original, counted)
    )
    with coordinate_reference_budget() as budget:
        predecessor = prepare_cell_geometry_predecessor(
            mesh, geometry, source, None, mesh, geometry, budget=budget
        )
        assert hashes == 1
        admission = budget.predecessor_content_admission_cache[id(predecessor)]
        assert admission.predecessor is predecessor
        start = budget.work_units
        predecessor.require_bound(mesh, geometry, budget=budget)
        warm_work = budget.work_units - start
        inherited = certify_cell_geometry_validity(
            geometry, mesh=mesh, budget=budget, predecessor=predecessor
        )
        assert inherited.certificate_id == source.certificate_id
        assert hashes == 1
        # Equal IDs and equal owners on a new predecessor do not convey issuance.
        foreign = replace(predecessor)
        start = budget.work_units
        foreign.require_bound(mesh, geometry, budget=budget)
        full_work = budget.work_units - start
        assert hashes == 2
        assert 0 < warm_work < full_work
        print(
            f"predecessor-sourceauth nonbinary={nonbinary} "
            f"same-ledger-warm={warm_work} unissued-owner={full_work} "
            f"cumulative={budget.work_units}"
        )
    with coordinate_reference_budget() as independent:
        predecessor.require_bound(mesh, geometry, budget=independent)
        assert hashes == 3


@pytest.mark.parametrize(
    "change", ("rows", "certificate-scalar", "certificate-array", "geometry", "structure")
)
def test_earned_predecessor_refuses_object_setattr_tampering(
    change: str,
) -> None:
    mesh, geometry = _triangle_geometry("none")
    source = certify_cell_geometry_validity(geometry, mesh=mesh)
    with coordinate_reference_budget() as budget:
        predecessor = prepare_cell_geometry_predecessor(
            mesh, geometry, source, None, mesh, geometry, budget=budget
        )
        predecessor.require_bound(mesh, geometry, budget=budget)
        if change == "rows":
            object.__setattr__(predecessor, "source_rows", (1, 0, 2))
        elif change == "certificate-scalar":
            object.__setattr__(
                source, "certified_valid_count", source.certified_valid_count + 1
            )
        elif change == "certificate-array":
            object.__setattr__(
                source, "determinant_lower", source.determinant_lower.at[0].add(1)
            )
        elif change == "geometry":
            object.__setattr__(
                geometry, "coordinates", geometry.coordinates.at[0, 0].add(0.1)
            )
        else:
            object.__setattr__(geometry, "block_names", ("changed", "second"))
        with pytest.raises(ValueError, match="CellGeometryPredecessor.content_id"):
            predecessor.require_bound(mesh, geometry, budget=budget)


def test_earned_predecessor_equal_bytes_new_array_requires_full_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh, geometry = _triangle_geometry("none")
    source = certify_cell_geometry_validity(geometry, mesh=mesh)
    with coordinate_reference_budget() as budget:
        predecessor = prepare_cell_geometry_predecessor(
            mesh, geometry, source, None, mesh, geometry, budget=budget
        )
        original: Callable[..., str] = predecessor_owner._predecessor_content_id
        hashes = 0

        def counted() -> None:
            nonlocal hashes
            hashes += 1

        monkeypatch.setattr(
            predecessor_owner, "_predecessor_content_id", _observed_hash(original, counted)
        )
        previous = source.status
        replacement = jnp.array(previous, copy=True)
        assert replacement is not previous
        object.__setattr__(source, "status", replacement)
        predecessor.require_bound(mesh, geometry, budget=budget)
        assert hashes == 1
        predecessor.require_bound(mesh, geometry, budget=budget)
        assert hashes == 1


@pytest.mark.parametrize("readonly", (False, True), ids=("mutable", "readonly-toggled"))
def test_predecessor_numpy_never_earns_immutable_content_admission(
    readonly: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    mesh, geometry = _triangle_geometry("none")
    coordinates = np.array(geometry.coordinates, copy=True)
    coordinates.flags.writeable = not readonly
    object.__setattr__(geometry, "coordinates", coordinates)
    source = certify_cell_geometry_validity(geometry, mesh=mesh)
    with coordinate_reference_budget() as budget:
        predecessor = prepare_cell_geometry_predecessor(
            mesh, geometry, source, None, mesh, geometry, budget=budget
        )
        assert id(predecessor) not in budget.predecessor_content_admission_cache
        original: Callable[..., str] = predecessor_owner._predecessor_content_id
        hashes = 0

        def counted() -> None:
            nonlocal hashes
            hashes += 1

        monkeypatch.setattr(
            predecessor_owner, "_predecessor_content_id", _observed_hash(original, counted)
        )
        predecessor.require_bound(mesh, geometry, budget=budget)
        predecessor.require_bound(mesh, geometry, budget=budget)
        assert hashes == 2
        coordinates.flags.writeable = True
        coordinates[0, 0] += 0.1
        coordinates.flags.writeable = not readonly
        with pytest.raises(ValueError, match="CellGeometryPredecessor.content_id"):
            predecessor.require_bound(mesh, geometry, budget=budget)
        assert hashes == 3


class _RationalRestriction(NamedTuple):
    source_mesh: CellMesh
    source_geometry: CellGeometrySpec
    source_validity: CellValidityCertificate
    target_mesh: CellMesh
    target_geometry: CellGeometrySpec


def _rational_restriction(
    kind: str = "quadrilateral", *, reverse: bool = False,
) -> _RationalRestriction:
    from phydrax.discretization._cell_geometry import (
        CellGeometryRestrictionSource,
        coordinate_lagrange_element,
        PolynomialComposedCellGeometryElement,
        require_scalar_coordinate_element,
    )

    vertices = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
    route = np.arange(len(vertices)).reshape(1, -1)
    source_mesh = CellMesh(
        vertices, (CellBlock("root", kind, route, global_ids=np.asarray((101,))),)
    )
    source_geometry = CellGeometrySpec.affine(source_mesh)
    source_validity = certify_cell_geometry_validity(source_geometry, mesh=source_mesh)
    child_vertices = vertices.copy()
    chart = coordinate_lagrange_element(kind, 1)
    numerators = np.array(chart.reference_nodes, dtype=np.int64, copy=True)
    if reverse:
        child_vertices[:, 0] = 1 - child_vertices[:, 0]
        numerators[:, 0] = 1 - numerators[:, 0]
    target_mesh = CellMesh(
        child_vertices / 3,
        (CellBlock("child", kind, route, global_ids=np.asarray((202,))),),
    )
    with coordinate_reference_budget() as budget:
        element = PolynomialComposedCellGeometryElement(
            require_scalar_coordinate_element(source_geometry.elements[0], "root"),
            chart,
            numerators,
            np.full(np.asarray(chart.reference_nodes).shape, 3, dtype=np.int64),
            budget=budget,
        )
        target_geometry = CellGeometrySpec(
            {"child": element}, {"child": source_geometry.geometry_dofs[0]},
            source_geometry.coordinates,
            restriction_source=CellGeometryRestrictionSource(
                validity_owner.cell_geometry_id(source_geometry), source_mesh.topology_id,
                {"child": np.asarray((101,))},
                {"child": np.asarray(source_mesh.vertex_global_ids)[None, :]},
            ),
            budget=budget,
        )
    return _RationalRestriction(
        source_mesh, source_geometry, source_validity, target_mesh, target_geometry
    )


def _ordinary_restriction(
    inputs: _RationalRestriction, budget: CoordinateEnclosureBudget
) -> CellValidityCertificate:
    predecessor = prepare_cell_geometry_predecessor(
        inputs.source_mesh, inputs.source_geometry, inputs.source_validity, None,
        inputs.target_mesh, inputs.target_geometry, budget=budget,
    )
    return certify_cell_geometry_validity(
        inputs.target_geometry, mesh=inputs.target_mesh,
        predecessor=predecessor, budget=budget,
    )


@pytest.mark.parametrize("kind", ("quadrilateral", "hexahedron"))
def test_atomic_restriction_matches_ordinary_with_exact_work_and_boundary(
    kind: str,
) -> None:
    ordinary_inputs = _rational_restriction(kind)
    atomic_inputs = _rational_restriction(kind)
    with coordinate_reference_budget() as budget:
        ordinary = _ordinary_restriction(ordinary_inputs, budget)
        ordinary_work = budget.work_units
        (admission,) = budget.predecessor_content_admission_cache.values()
        # Both paths retain the lookup charge; only the second complete owner
        # traversal disappears. Issuer hash, owner sealing and chart work stay.
        traversal = sum(1 + len(names) for _, _, names, _ in admission.owner_fields)
    with coordinate_reference_budget() as budget:
        atomic = validity_owner._certify_restriction_cell_geometry_validity(
            *atomic_inputs, budget=budget
        )
        atomic_work = budget.work_units
    _assert_same_certificate(ordinary, atomic)
    assert atomic.all_certified
    assert ordinary_work - atomic_work == traversal > 0
    with coordinate_reference_budget(
        MeshCertificateLimits(maximum_work_units=atomic_work)
    ) as budget:
        at_limit = validity_owner._certify_restriction_cell_geometry_validity(
            *atomic_inputs, budget=budget
        )
        assert budget.work_units == atomic_work
    _assert_same_certificate(atomic, at_limit)
    with coordinate_reference_budget(
        MeshCertificateLimits(maximum_work_units=atomic_work - 1)
    ) as budget, pytest.raises(CoordinateEnclosureResourceError) as failure:
        validity_owner._certify_restriction_cell_geometry_validity(
            *atomic_inputs, budget=budget
        )
    assert failure.value.resource == "coefficient_work"
    assert failure.value.limit == atomic_work - 1
    print(
        f"atomic-restriction {kind}: ordinary={ordinary_work} atomic={atomic_work} "
        f"owner-traversal-saved={traversal} R={atomic_work} R-1=refused"
    )


@pytest.mark.parametrize(
    "change",
    ("source-mesh", "source-geometry", "source-validity", "target", "chart", "rows"),
)
def test_atomic_restriction_refuses_mutation_in_replaced_preparer(
    change: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _rational_restriction()
    original = validity_owner.prepare_cell_geometry_predecessor

    def changed(
        predecessor: predecessor_owner.CellGeometryPredecessor,
    ) -> predecessor_owner.CellGeometryPredecessor:
        # A preparer replacement is an external continuation, so atomic
        # consumption is disabled even when the original producer earned it.
        if change == "source-mesh":
            owner, name = predecessor.source_mesh, "coordinates"
        elif change == "source-geometry":
            owner, name = predecessor.source_geometry, "coordinates"
        elif change == "source-validity":
            owner, name = predecessor.source_validity, "determinant_lower"
        elif change == "target":
            owner, name = predecessor.target_geometry, "coordinates"
        elif change == "chart":
            element = predecessor.target_geometry.elements[0]
            coefficients = getattr(element, "chart_coefficients")
            object.__setattr__(element, "chart_coefficients", coefficients[::-1])
            return predecessor
        else:
            object.__setattr__(predecessor, "source_rows", (0,))
            return predecessor
        values = getattr(owner, name)
        object.__setattr__(owner, name, values + 1)
        return predecessor

    monkeypatch.setattr(
        validity_owner, "prepare_cell_geometry_predecessor",
        _interleaved_predecessor(original, changed),
    )
    with coordinate_reference_budget() as budget, pytest.raises(
        ValueError, match="CellGeometryPredecessor.content_id"
    ):
        validity_owner._certify_restriction_cell_geometry_validity(
            *inputs, budget=budget
        )


@pytest.mark.parametrize("readonly", (False, True))
def test_atomic_restriction_mutable_owners_keep_full_authentication(
    readonly: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _rational_restriction()
    coordinates = np.array(inputs.target_geometry.coordinates, copy=True)
    coordinates.flags.writeable = not readonly
    object.__setattr__(inputs.target_geometry, "coordinates", coordinates)
    hashes = 0
    original = predecessor_owner._predecessor_content_id

    def counted() -> None:
        nonlocal hashes
        hashes += 1

    monkeypatch.setattr(
        predecessor_owner, "_predecessor_content_id", _observed_hash(original, counted)
    )
    with coordinate_reference_budget() as budget:
        ordinary = _ordinary_restriction(inputs, budget)
    ordinary_hashes = hashes
    hashes = 0
    with coordinate_reference_budget() as budget:
        atomic = validity_owner._certify_restriction_cell_geometry_validity(
            *inputs, budget=budget
        )
        assert not budget.predecessor_content_admission_cache
    assert hashes == ordinary_hashes == 2
    _assert_same_certificate(ordinary, atomic)


def test_atomic_restriction_repeated_inputs_never_carry_an_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pickle

    inputs = _rational_restriction()
    hashes = 0
    original = predecessor_owner._predecessor_content_id

    def counted() -> None:
        nonlocal hashes
        hashes += 1

    monkeypatch.setattr(
        predecessor_owner, "_predecessor_content_id", _observed_hash(original, counted)
    )
    with coordinate_reference_budget() as budget:
        first = validity_owner._certify_restriction_cell_geometry_validity(
            *inputs, budget=budget
        )
        second = validity_owner._certify_restriction_cell_geometry_validity(
            *inputs, budget=budget
        )
        assert hashes == 2
        assert len(budget.predecessor_content_admission_cache) == 2
    _assert_same_certificate(first, second)
    # Only ordinary scientific inputs can be copied/restored. No issue, token,
    # continuation or predecessor escapes the atomic certificate API.
    for restored in (inputs, pickle.loads(pickle.dumps(inputs))):
        before = hashes
        with coordinate_reference_budget() as budget:
            cold = validity_owner._certify_restriction_cell_geometry_validity(
                *restored, budget=budget
            )
            assert hashes == before + 1
            assert len(budget.predecessor_content_admission_cache) == 1
        assert type(cold) is CellValidityCertificate
        _assert_same_certificate(first, cold)


@pytest.mark.parametrize("reuse", ("replay", "clone", "wrong-budget", "cold"))
def test_atomic_restriction_cannot_consume_a_returned_predecessor_as_fresh(
    reuse: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _rational_restriction()
    ordinary_require = predecessor_owner.CellGeometryPredecessor.require_bound
    guarded_work: list[int] = []

    def observed(
        predecessor: predecessor_owner.CellGeometryPredecessor,
        mesh: CellMesh, geometry: CellGeometrySpec, /,
        *, budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        assert budget is not None
        before = budget.work_units
        ordinary_require(predecessor, mesh, geometry, budget=budget)
        guarded_work.append(budget.work_units - before)

    monkeypatch.setattr(
        predecessor_owner.CellGeometryPredecessor, "require_bound", observed
    )
    with coordinate_reference_budget() as preparation:
        held = prepare_cell_geometry_predecessor(
            inputs.source_mesh, inputs.source_geometry, inputs.source_validity, None,
            inputs.target_mesh, inputs.target_geometry, budget=preparation,
        )
        if reuse == "clone":
            held = replace(held)

        def returned(
            *args: object, **kwargs: object,
        ) -> predecessor_owner.CellGeometryPredecessor:
            return held

        monkeypatch.setattr(
            validity_owner, "prepare_cell_geometry_predecessor", returned
        )
        if reuse in ("replay", "clone"):
            for _ in range(2):
                validity_owner._certify_restriction_cell_geometry_validity(
                    *inputs, budget=preparation
                )
            assert len(guarded_work) == 2
        elif reuse == "wrong-budget":
            with coordinate_reference_budget() as other:
                validity_owner._certify_restriction_cell_geometry_validity(
                    *inputs, budget=other
                )
    if reuse == "cold":
        with coordinate_reference_budget() as cold:
            validity_owner._certify_restriction_cell_geometry_validity(
                *inputs, budget=cold
            )
    assert guarded_work and all(work > 1 for work in guarded_work)


@pytest.mark.parametrize("kind", ("quadrilateral", "hexahedron"))
def test_atomic_restriction_retains_target_orientation_proof(kind: str) -> None:
    inputs = _rational_restriction(kind, reverse=True)
    with coordinate_reference_budget() as budget:
        ordinary = _ordinary_restriction(inputs, budget)
    with coordinate_reference_budget() as budget:
        atomic = validity_owner._certify_restriction_cell_geometry_validity(
            *inputs, budget=budget
        )
    _assert_same_certificate(ordinary, atomic)
    assert atomic.invalid_count == 1


def test_atomic_restriction_retains_requested_policy_binding() -> None:
    inputs = _rational_restriction()
    with coordinate_reference_budget() as budget, pytest.raises(
        ValueError, match="predecessor.policy_id"
    ):
        validity_owner._certify_restriction_cell_geometry_validity(
            *inputs, budget=budget,
            policy=CellValidityPolicy(maximum_subdivision_depth=0),
        )
