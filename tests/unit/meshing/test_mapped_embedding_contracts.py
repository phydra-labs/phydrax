# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Closed-cell hypotheses and resource ownership of mapped contact proofs."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
from phydrax.discretization._coordinate_enclosure import (
    affine_arguments,
    expression_evaluate,
    ExpressionComposition,
    Polynomial,
)
from phydrax.geometry._embedding_cells import BisectionWorklist, Cell, vertices
from phydrax.geometry._embedding_state import (
    ContactDecision,
    ContactStatus,
    CoordinateMapScope,
    CoordinateScope,
    MeshCertificateLimits,
)
from phydrax.geometry._mapped_embedding import pair_decision
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.mapped_embedding import complex_cube_map


def test_false_locally_injective_hint_does_not_earn_jordan_contact_check() -> None:
    # Cubing has determinant 9(x^2+y^2)^2; the reference charts add a positive
    # area factor of four. Both rectangle maps are oriented positively. The first
    # is nevertheless noninjective: 1 +/- i*sqrt(3) both cube to (-8, 0).
    # The second rectangle is a neighboring injective chart, not a repair of
    # the first rectangle's false closed-cell hypothesis.
    limits = MeshCertificateLimits(
        maximum_subdivision_depth=6, maximum_subdivision_pieces=4096
    )
    first = Cell(
        complex_cube_map(1),
        "box",
        "quadrilateral",
        2,
        (0, 1, 2, 3),
        vertices("quadrilateral"),
    )
    second = Cell(
        complex_cube_map(2),
        "box",
        "quadrilateral",
        2,
        (1, 4, 5, 2),
        vertices("quadrilateral"),
    )
    checks: list[str] = []
    with coordinate_reference_budget(limits) as budget:
        contact = pair_decision(
            first,
            second,
            limits,
            limits.maximum_subdivision_pieces,
            locally_injective=True,
            checks=checks,
            budget=budget,
        )
    assert "mapped_planar_jordan_boundary_contact" not in checks
    assert contact.work <= limits.maximum_subdivision_pieces


@pytest.mark.parametrize(
    ("second_coordinate", "expected"),
    (
        pytest.param({(1,): Fraction(-1)}, "separated", id="opposite-outgoing-arcs"),
        pytest.param({(1,): Fraction(1)}, "unresolved", id="same-ray-fold-back"),
        pytest.param(
            {(1,): Fraction(1), (2,): Fraction(-2)}, "unresolved", id="curved-fold-back"
        ),
    ),
)
def test_adjacent_edges_require_separated_outgoing_arcs(
    second_coordinate: Polynomial, expected: ContactStatus
) -> None:
    limits = MeshCertificateLimits(
        maximum_subdivision_depth=0, maximum_subdivision_pieces=16
    )
    first = Cell(
        ({(1,): Fraction(1)}, {}), "box", "interval", 1, (10, 11), vertices("interval")
    )
    second = Cell(
        (second_coordinate, {}), "box", "interval", 1, (10, 12), vertices("interval")
    )
    with coordinate_reference_budget(limits) as budget:
        contact = pair_decision(first, second, limits, 16, budget=budget)
    assert contact.status == expected
    if expected == "unresolved":
        assert contact.unresolved_check() == "mapped_intersection_subdivision_depth"


@pytest.mark.parametrize(
    ("fold_back", "expected"),
    ((False, "separated"), (True, "unresolved")),
    ids=("soft-cut-oblique-contact", "shared-ray-fold-back"),
)
def test_soft_cut_boundary_contact_requires_exact_oblique_support(
    fold_back: bool, expected: ContactStatus
) -> None:
    # Bilinear boundary patches with rational coefficients from the SOFT PLC cut.
    # Their support planes meet on the z axis, but only the common endpoint
    # is allowed. The normal (-1, 9, 1) separates their full corner hulls;
    # coordinate axes and support-plane normals alone cannot prove that.
    first = Cell(
        (
            {},
            {(1, 0): Fraction(1, 72), (1, 1): Fraction(-1, 540)},
            {
                (0, 0): Fraction(1),
                (0, 1): Fraction(-215, 432),
                (1, 0): Fraction(-1, 72),
                (1, 1): Fraction(-187, 2160),
            },
        ),
        "box",
        "quadrilateral",
        2,
        (3, 140, 225, 139),
        vertices("quadrilateral"),
    )
    direction = -1 if fold_back else 1
    second = Cell(
        (
            {(0, 1): Fraction(73, 1080), (1, 1): Fraction(-47, 720)},
            {},
            {
                (0, 0): Fraction(217, 432),
                (1, 0): direction * Fraction(-215, 432),
                (0, 1): direction * Fraction(-217, 2160),
                (1, 1): direction * Fraction(217, 2160),
            },
        ),
        "box",
        "quadrilateral",
        2,
        (139, 34, 64, 226),
        vertices("quadrilateral"),
    )
    limits = MeshCertificateLimits(
        maximum_subdivision_depth=0, maximum_subdivision_pieces=16
    )
    with coordinate_reference_budget(limits) as budget:
        contact = pair_decision(first, second, limits, 16, budget=budget)
    assert contact.status == expected
    if expected == "unresolved":
        assert contact.unresolved_check() == "mapped_intersection_subdivision_depth"


def test_soft_cut_coplanar_boundary_contact_uses_in_plane_edge_axes() -> None:
    # These SOFT PLC boundary patches share one corner in the x=0 plane.
    # The supporting normal (0, -10, 1) distinguishes their outgoing rays.
    first = Cell(
        (
            {},
            {(1, 0): Fraction(1, 72), (1, 1): Fraction(-1, 540)},
            {
                (0, 0): Fraction(1),
                (0, 1): Fraction(-215, 432),
                (1, 0): Fraction(-1, 72),
                (1, 1): Fraction(-187, 2160),
            },
        ),
        "box",
        "quadrilateral",
        2,
        (3, 140, 225, 139),
        vertices("quadrilateral"),
    )
    second = Cell(
        (
            {},
            {
                (0, 0): Fraction(1, 36),
                (1, 0): Fraction(-5, 432),
                (1, 1): Fraction(-1, 240),
            },
            {
                (0, 0): Fraction(1, 36),
                (1, 0): Fraction(-5, 432),
                (0, 1): Fraction(17, 36),
                (1, 1): Fraction(-187, 2160),
            },
        ),
        "box",
        "quadrilateral",
        2,
        (27, 49, 225, 103),
        vertices("quadrilateral"),
    )
    limits = MeshCertificateLimits(
        maximum_subdivision_depth=0, maximum_subdivision_pieces=16
    )
    with coordinate_reference_budget(limits) as budget:
        contact = pair_decision(first, second, limits, 16, budget=budget)
    assert contact.status == "separated"


def test_chart_composition_invalidates_corner_images_and_preserves_cell_identity() -> (
    None
):
    cell = Cell(
        ({(1,): Fraction(1)},),
        "box",
        "interval",
        1,
        (10, 11),
        vertices("interval"),
        ((Fraction(0),), (Fraction(1),)),
    )
    with coordinate_reference_budget() as budget:
        composition = ExpressionComposition(
            affine_arguments(
                np.asarray((0.25,), dtype=np.float64),
                np.asarray(((0.5,),), dtype=np.float64),
                budget=budget,
            ),
            budget=budget,
        )
        composed = cell.composed(composition, budget=budget)
        assert expression_evaluate(
            composed.coordinates[0], (Fraction(0),), budget=budget
        ) == Fraction(1, 4)
        assert expression_evaluate(
            composed.coordinates[0], (Fraction(1),), budget=budget
        ) == Fraction(3, 4)
    assert composed.corner_images is None
    assert composed.kind == cell.kind and composed.domain == cell.domain
    assert (
        composed.vertices == cell.vertices
        and composed.reference_vertices == cell.reference_vertices
    )
    assert cell.corner_images == ((Fraction(0),), (Fraction(1),))


def test_contact_decision_refuses_check_on_resolved_status() -> None:
    with pytest.raises(
        ValueError, match="ContactDecision.check must accompany unresolved status"
    ):
        ContactDecision("separated", 0, check="mapped_trace_mismatch")


def test_certificate_limits_replace_recomputes_identity_without_mutating_input() -> None:
    limits = MeshCertificateLimits(maximum_subdivision_depth=2)
    unchanged = replace(limits)
    changed = replace(limits, maximum_subdivision_depth=3)
    assert unchanged.limits_id == limits.limits_id
    assert changed.limits_id != limits.limits_id
    assert limits.maximum_subdivision_depth == 2
    assert changed.maximum_subdivision_depth == 3


def test_coordinate_scope_retains_prepared_array_and_source_bank() -> None:
    cells = np.asarray((1, 3), dtype=np.int64)
    bank = ((Fraction(0), Fraction(1)),)
    scope = CoordinateScope("mapped", cells, bank)
    assert scope.coordinate_scope == "mapped"
    assert scope.mapped_cells is cells
    assert scope.source_values is bank


@pytest.mark.parametrize(
    ("scope", "cells", "field"),
    (
        pytest.param(
            "mapped",
            np.empty((0,), dtype=np.int64),
            "coordinate_scope",
            id="mapped-empty",
        ),
        pytest.param(
            "affine",
            np.asarray((1,), dtype=np.int64),
            "coordinate_scope",
            id="affine-nonempty",
        ),
        pytest.param(
            "mapped",
            np.asarray((1,), dtype=np.int32),
            "mapped_cells",
            id="wrong-index-dtype",
        ),
        pytest.param(
            "mapped",
            np.asarray(((1,),), dtype=np.int64),
            "mapped_cells",
            id="wrong-index-rank",
        ),
    ),
)
def test_coordinate_scope_refuses_invalid_host_classification(
    scope: CoordinateMapScope, cells: np.ndarray, field: str
) -> None:
    with pytest.raises(ValueError, match=f"CoordinateScope.{field} must"):
        CoordinateScope(scope, cells, ())


def test_contact_worklist_storage_refusal_retains_measured_preallocation_work() -> None:
    # One dimension-one FIFO entry plus its in-flight owner requires 3136 bytes.
    # The cap is exactly one byte below that documented initial storage bound.
    limits = MeshCertificateLimits(maximum_scratch_bytes=3135)
    with coordinate_reference_budget(limits) as budget:
        with pytest.raises(CoordinateEnclosureResourceError) as caught:
            with budget.measurement(), budget.live_storage() as live:
                BisectionWorklist(1, 1, live)
        assert caught.value.resource == "polynomial_storage"
        assert caught.value.requested == 3136
        assert caught.value.phase_evidence is not None
        assert caught.value.phase_evidence.work_units == 0


@pytest.mark.parametrize("algebraic", (False, True), ids=("rational", "algebraic"))
@pytest.mark.parametrize("curved", (False, True), ids=("affine-p2", "curved-p2"))
def test_refined_p2_prism_contact_preserves_generic_verdict(
    curved: bool, algebraic: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from phydrax.discretization._cell_geometry import (
        coordinate_lagrange_element,
        RestrictedCellGeometryElement,
    )
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.discretization._coordinate_enclosure import coordinate_expressions
    from phydrax.discretization._exact_algebraic_geometry import RadicalElement
    from phydrax.geometry import _affine_mapped_contact as contact_owner
    from phydrax.geometry._affine_mapped_contact import ContactMethod

    source = coordinate_lagrange_element("prism", 2)
    nodes = np.asarray(source.reference_nodes, dtype=np.float64)
    shift = RadicalElement.sqrt(2) if algebraic else Fraction(0)
    controls = tuple(
        (
            Fraction(float(x))
            + shift
            + (Fraction(float(y)) * (1 - Fraction(float(y))) / 8 if curved else 0),
            Fraction(float(y)),
            Fraction(float(z)),
        )
        for x, y, z in nodes
    )
    limits = MeshCertificateLimits()
    pieces = []
    with coordinate_reference_budget(limits) as budget:
        for index, offset in enumerate((0.0, 0.5)):
            child = RestrictedCellGeometryElement(
                source,
                "prism",
                np.diag(np.asarray((0.25, 0.25, 0.5), dtype=np.float64)),
                np.asarray((offset, 0.0, 0.0), dtype=np.float64),
            )
            expressions = coordinate_expressions(child, controls, budget=budget)
            assert expressions is not None
            pieces.append(
                Cell(
                    expressions,
                    "prism",
                    "prism",
                    3,
                    tuple(range(6 * index, 6 * index + 6)),
                    vertices("prism"),
                )
            )
        checks: list[str] = []
        reduced = pair_decision(
            pieces[0],
            pieces[1],
            limits,
            limits.maximum_subdivision_pieces,
            checks=checks,
            budget=budget,
        )
    assert ("mapped_affine_polytope_contact" in checks) is not curved
    original = contact_owner.contact_decision

    def generic_contact(
        first: Cell,
        second: Cell,
        maximum_work: int,
        /,
        *,
        method: ContactMethod,
        budget: CoordinateEnclosureBudget,
        limits: MeshCertificateLimits | None = None,
        authenticated_first_traces: tuple[tuple[int, ...], ...] = (),
    ) -> ContactDecision | None:
        if method == "affine_polytope":
            return None
        return original(
            first,
            second,
            maximum_work,
            method=method,
            budget=budget,
            limits=limits,
            authenticated_first_traces=authenticated_first_traces,
        )

    monkeypatch.setattr(contact_owner, "contact_decision", generic_contact)
    with coordinate_reference_budget(limits) as budget:
        generic = pair_decision(
            pieces[0],
            pieces[1],
            limits,
            limits.maximum_subdivision_pieces,
            budget=budget,
        )
    assert reduced.status == generic.status == "separated"
