# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Closed-cell hypotheses and resource ownership of mapped contact proofs."""

from dataclasses import dataclass, replace
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._fingerprint import canonical_fingerprint
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._cell_geometry import coordinate_lagrange_element
from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
from phydrax.discretization._coordinate_enclosure import (
    affine_arguments,
    coordinate_expressions,
    expression_evaluate,
    Expression,
    ExpressionComposition,
    Polynomial,
)
from phydrax.geometry._embedding_cells import BisectionWorklist, Cell, vertices
from phydrax.geometry._embedding_state import (
    ContactDecision,
    ContactStatus,
    CoordinateMapScope,
    CoordinateScope,
    EmbeddingState,
    MeshCertificateLimits,
)
from phydrax.geometry._mapped_embedding import pair_decision, root_injective
from phydrax.geometry import _restricted_embedding as restricted_embedding
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.mapped_embedding import complex_cube_map


def test_root_injectivity_retains_proof_not_bernstein_workspace() -> None:
    element = coordinate_lagrange_element("quadrilateral", 2)
    local = tuple(
        (Fraction(float(u)) + Fraction(float(u)) ** 2 / 8, Fraction(float(v)))
        for u, v in np.asarray(element.reference_nodes)
    )
    limits = MeshCertificateLimits()
    with coordinate_reference_budget(limits) as budget:
        coordinates = coordinate_expressions(element, local, budget=budget)
        assert coordinates is not None
        cell = Cell(coordinates, "box", "quadrilateral", 2, (0, 1, 2, 3),
                    vertices("quadrilateral"))
        original = EmbeddingState([], [])
        before_memos = len(budget.bernstein_coefficient_cache)
        with budget.polynomial_preparation_scope():
            assert root_injective(cell, element, local, limits, original,
                                  budget=budget) is None
        assert len(budget.root_injectivity_cache) == 1
        assert len(budget.bernstein_coefficient_cache) == before_memos
        retained = budget.retained_basis_bytes
        work = budget.work_units
        repeated = EmbeddingState([], [])
        with budget.polynomial_preparation_scope():
            assert root_injective(cell, element, local, limits, repeated,
                                  budget=budget) is None
        assert repeated.subdivision_pieces == original.subdivision_pieces > 0
        assert repeated.checks == original.checks
        assert budget.retained_basis_bytes == retained
        assert budget.work_units - work < work
        full = EmbeddingState([], [])
        full.subdivision_pieces = limits.maximum_subdivision_pieces
        assert root_injective(cell, element, local, limits, full,
                              budget=budget) == "local_injectivity_piece_budget"
        assert full.subdivision_pieces == limits.maximum_subdivision_pieces


def test_root_injectivity_binds_coefficients_limits_and_incidence() -> None:
    element = coordinate_lagrange_element("quadrilateral", 2)
    limits = MeshCertificateLimits(maximum_subdivision_depth=0)
    with coordinate_reference_budget(limits) as budget:
        for offset, proof_limits, incidence in (
            (0, limits, (0, 1, 2, 3)),
            (1, limits, (0, 1, 2, 3)),
            (1, MeshCertificateLimits(maximum_subdivision_depth=1), (0, 1, 2, 3)),
            (1, limits, (4, 5, 6, 7)),
        ):
            local = tuple(
                (Fraction(float(u)) + Fraction(float(u)) ** 2 / 8 + offset,
                 Fraction(float(v)))
                for u, v in np.asarray(element.reference_nodes)
            )
            coordinates = coordinate_expressions(element, local, budget=budget)
            assert coordinates is not None
            cell = Cell(coordinates, "box", "quadrilateral", 2, incidence,
                        vertices("quadrilateral"))
            assert root_injective(cell, element, local, proof_limits,
                                  EmbeddingState([], []), budget=budget) is None
        assert len(budget.root_injectivity_cache) == 4
        folded = tuple(
            (Fraction(float(u)) - 2 * Fraction(float(u)) ** 2, Fraction(float(v)))
            for u, v in np.asarray(element.reference_nodes)
        )
        coordinates = coordinate_expressions(element, folded, budget=budget)
        assert coordinates is not None
        cell = Cell(coordinates, "box", "quadrilateral", 2, (0, 1, 2, 3),
                    vertices("quadrilateral"))
        assert root_injective(cell, element, folded, limits,
                              EmbeddingState([], []), budget=budget) is not None
        assert len(budget.root_injectivity_cache) == 4


@pytest.mark.parametrize("change", ("none", "coefficients", "incidence"))
def test_repeated_reference_box_contact_reuses_exact_proof(
    change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    points = np.asarray(
        ((0, 0), (.5, 0), (1, 0), (0, 1), (.5, 1), (1, 1)), dtype=np.float64
    )
    points = np.concatenate((points, points + (2, 0)))
    rows = np.asarray(
        ((0, 1, 4, 3), (1, 2, 5, 4), (6, 7, 10, 9), (7, 8, 11, 10)),
        dtype=np.int32,
    )
    mesh = CellMesh(points, (CellBlock("boxes", "quadrilateral", rows),))
    geometry = CellGeometrySpec.affine(mesh)
    children: dict[int, list[tuple[int, Cell]]] = {}
    for index, row in enumerate(rows):
        offset = Fraction(index % 2, 2)
        if index == 3 and change == "coefficients":
            offset = Fraction(1, 4)
        coordinates: tuple[Expression, ...] = (
            {(0, 0): offset, (1, 0): Fraction(1, 2)},
            {(0, 1): Fraction(1)},
        )
        corners = tuple((offset + u / 2, v) for u, v in vertices("quadrilateral"))
        incidence = tuple(map(int, row))
        if index == 3 and change == "incidence":
            incidence = tuple(vertex + 100 for vertex in incidence)
        chart = Cell(coordinates, "box", "quadrilateral", 2, incidence,
                     vertices("quadrilateral"), corners)
        children.setdefault(index // 2, []).append((index, chart))
    calls = []
    original = restricted_embedding.pair_decision

    def decide(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(restricted_embedding, "pair_decision", decide)
    limits = MeshCertificateLimits()
    with coordinate_reference_budget(limits) as budget:
        state = EmbeddingState([], [])
        request = restricted_embedding._RestrictedRequest(
            state, mesh, geometry, np.arange(4, dtype=np.int64), limits, budget, None
        )
        before = budget.temporary_bytes_upper
        with budget.live_storage() as live:
            cache = restricted_embedding._RestrictionChartCache({}, {}, {}, {}, live)
            restricted_embedding._certify_restriction_child_pairs(
                request, children, cache
            )
            assert len(calls) == (1 if change == "none" else 2)
            if change == "none":
                assert state.clean
                assert len(cache.separated) == 1
                pieces, _ = next(iter(cache.separated.values()))
                assert state.subdivision_pieces == 2 * pieces
            else:
                assert not state.clean
        assert budget.temporary_bytes_upper == before


def test_candidate_ray_reuse_preserves_original_sum_planes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry import _mapped_embedding as mapped

    first_corners = tuple(
        tuple(map(Fraction, point)) for point in ((0, 0), (1, -2), (0, -2), (-1, 0))
    )
    second_corners = tuple(
        tuple(map(Fraction, point)) for point in ((1, -2), (0, 0), (1, 0), (2, -2))
    )
    first = Cell(
        ({(1, 0): Fraction(1), (0, 1): Fraction(-1)}, {(1, 0): Fraction(-2)}),
        "box", "quadrilateral", 2, (0, 1, 2, 3), vertices("quadrilateral"),
        first_corners,
    )
    second = Cell(
        ({(0, 0): Fraction(1), (1, 0): Fraction(-1), (0, 1): Fraction(1)},
         {(0, 0): Fraction(-2), (1, 0): Fraction(2)}),
        "box", "quadrilateral", 2, (1, 0, 4, 5), vertices("quadrilateral"),
        second_corners,
    )
    # Every base plane fails. The ORIGINAL first sum (2, 1) supports the
    # common edge. Normalizing base vectors first would replace it by (1, 1),
    # which is not that edge's support plane.
    normals = [
        np.asarray(tuple(map(Fraction, row)), dtype=object)
        for row in ((2, 0), (0, 1), (4, 0), (-6, 0), (0, 2), (0, -3))
    ]
    candidates = mapped._PlaneCandidates(
        np.asarray(first_corners, dtype=object),
        np.asarray(second_corners, dtype=object), normals,
    )
    calls = []
    original = mapped._corner_plane_separates

    def separate(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(mapped, "_corner_plane_separates", separate)
    with coordinate_reference_budget() as budget:
        retained = budget.retained_basis_bytes
        temporary = budget.temporary_bytes_upper
        assert mapped._multi_affine_candidate_separation(
            first, second, {0, 1}, candidates, budget=budget
        )
        assert len(calls) == 5
        assert budget.retained_basis_bytes == retained
        assert budget.temporary_bytes_upper == temporary


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


def test_canonical_affine_corner_issue_binds_current_coefficients_not_public_hints() -> None:
    from phydrax.discretization._coordinate_enclosure import affine_simplex_corner_images
    from phydrax.geometry import _affine_mapped_contact as owner

    element = coordinate_lagrange_element("triangle", 1)
    local = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    with coordinate_reference_budget() as budget:
        coordinates = coordinate_expressions(element, local, budget=budget)
        corners = affine_simplex_corner_images(element, local, budget=budget)
        assert coordinates is not None and corners is not None
        hints = tuple(tuple(value + 100 for value in row) for row in corners)
        cell = Cell(coordinates, "simplex", "triangle", 2, (0, 1, 2), vertices("triangle"), hints)
        owner._retain_affine_cell_images(cell, corners, budget=budget)
        before = budget.work_units
        assert owner._cell_images(cell, budget=budget) is corners
        assert budget.work_units == before
        first_coordinate = cell.coordinates[0]
        assert isinstance(first_coordinate, dict)
        first_coordinate[(0, 0)] = Fraction(1)
        changed = owner._cell_images(cell, budget=budget)
        assert changed is not corners
        assert changed == tuple((row[0] + 1, row[1]) for row in corners)
        assert changed != hints


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


@dataclass(frozen=True)
class _AnalyticHomotopyTriangle:
    """Exact authored affine chart followed by an analytic spatial deformation."""

    corners: np.ndarray
    vertices: tuple[int, int, int]
    target: tuple[Polynomial, ...]
    amplitude: float = 0.0
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    x_scale: float = 1.0
    authority: str = "authored-analytic-chart"

    @property
    def map_id(self) -> str:
        return canonical_fingerprint((
            self.authority, self.corners, self.vertices, self.amplitude,
            self.translation, self.x_scale,
        ))

    @property
    def authored_deviation_bound(self) -> float:
        return 0.0

    def trace_identity(self, local_vertices: tuple[int, ...], /) -> str:
        # These are authored exact binary-rational affine endpoints, not sampled
        # values of the nonlinear source. Equal ordered charts followed by the
        # same analytic formula prove the complete parameterized trace identity.
        return canonical_fingerprint((
            self.authority,
            tuple(self.vertices[index] for index in local_vertices),
            self.corners[list(local_vertices)],
            self.amplitude, self.translation, self.x_scale,
        ))

    @classmethod
    def enclosures(cls, requests, /, *, image, jacobian, budget):
        from phydrax.geometry._embedding_cells import SurfaceMapEnclosures

        def stacked(rows):
            return np.stack([row[0] for row in rows]), np.stack([row[1] for row in rows])

        return [
            SurfaceMapEnclosures(
                stacked([source._image(row, budget=budget) for row in references]) if image else None,
                stacked([source._jacobian(row, budget=budget) for row in references]) if jacobian else None,
                stacked([source._orientation(row, budget=budget) for row in references]) if jacobian else None,
            )
            for source, references in requests
        ]

    def _image(self, reference, /, *, budget):
        from phydrax.geometry._interval_enclosure import (
            interval_add,
            interval_multiply,
            interval_sin,
        )
        from phydrax.geometry._mapped_embedding import _homotopy_target_bounds

        budget.reserve_geometry_queries(1)
        lower, upper = _homotopy_target_bounds(self.target, reference, budget=budget)
        x = (lower[0], upper[0])
        amplitude = np.asarray(self.amplitude)
        lower[2], upper[2] = interval_add(
            (lower[2], upper[2]),
            interval_multiply((amplitude, amplitude), interval_sin(x)),
        )
        factor = np.asarray(self.x_scale)
        lower[0], upper[0] = interval_multiply(x, (factor, factor))
        translation = np.asarray(self.translation)
        return interval_add((lower, upper), (translation, translation))

    def _jacobian(self, reference, /, *, budget):
        from phydrax.discretization._coordinate_enclosure import expression_derivative
        from phydrax.geometry._interval_enclosure import (
            interval_add,
            interval_cos,
            interval_multiply,
        )
        from phydrax.geometry._mapped_embedding import _homotopy_target_bounds

        budget.reserve_geometry_queries(1)
        jacobian = tuple(
            expression_derivative(value, axis, budget=budget)
            for value in self.target for axis in range(2)
        )
        lower, upper = _homotopy_target_bounds(jacobian, reference, budget=budget)
        lower, upper = lower.reshape(3, 2), upper.reshape(3, 2)
        x = _homotopy_target_bounds(self.target[:1], reference, budget=budget)
        amplitude = np.asarray(self.amplitude)
        factor = interval_multiply(
            (amplitude, amplitude), interval_cos((x[0][0], x[1][0]))
        )
        lower[2], upper[2] = interval_add(
            (lower[2], upper[2]), interval_multiply(factor, (lower[0], upper[0]))
        )
        scale = np.asarray(self.x_scale)
        lower[0], upper[0] = interval_multiply(
            (lower[0], upper[0]), (scale, scale)
        )
        return lower, upper

    def _orientation(self, reference, /, *, budget):
        return self._jacobian(reference, budget=budget)


def _analytic_homotopy_cell(
    points: np.ndarray,
    row: tuple[int, int, int],
    entity: int,
    *,
    amplitude: float = 0.0,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    x_scale: float = 1.0,
    orientation: int = 1,
):
    from phydrax.geometry._embedding_cells import IntervalBoundaryCell

    corners = points[list(row)]
    coordinates: list[Polynomial] = []
    for axis in range(3):
        coordinate: Polynomial = {
            (0, 0): Fraction(float(corners[0, axis])),
            (1, 0): Fraction(float(corners[1, axis])) - Fraction(float(corners[0, axis])),
            (0, 1): Fraction(float(corners[2, axis])) - Fraction(float(corners[0, axis])),
        }
        coordinates.append(coordinate)
    target = tuple(coordinates)
    source = _AnalyticHomotopyTriangle(
        corners, row, target, amplitude, translation, x_scale
    )
    return IntervalBoundaryCell(source, target, row, corners, entity, orientation)


def _analytic_octahedron(*, amplitude=0.0, x_scale=1.0, reverse_chart=False):
    points = np.asarray((
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
    ))
    rows = [(4, i, (i + 1) % 4) for i in range(4)]
    rows += [(5, (i + 1) % 4, i) for i in range(4)]
    if reverse_chart:
        rows[0] = (rows[0][0], rows[0][2], rows[0][1])
    return tuple(
        _analytic_homotopy_cell(
            points, row, index, amplitude=amplitude, x_scale=x_scale,
            orientation=-1 if reverse_chart and index == 0 else 1,
        )
        for index, row in enumerate(rows)
    )


@pytest.mark.parametrize("reverse_chart", (False, True))
def test_interval_boundary_homotopy_proves_nonlinear_source_and_moving_vertex(
    reverse_chart: bool,
) -> None:
    from phydrax.geometry._mapped_embedding import certify_interval_boundary_homotopy

    cells = _analytic_octahedron(amplitude=0.01, reverse_chart=reverse_chart)
    limits = MeshCertificateLimits()
    with coordinate_reference_budget(limits, queries_upper=limits.maximum_source_samples) as budget:
        result = certify_interval_boundary_homotopy(cells, limits, tolerance=0.05, budget=budget)
    assert result.complete
    assert result.reason == ""
    assert 0 < result.distance_upper <= 0.05
    assert result.proof_id
    assert result.piece_count > len(cells)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("trace", "interval_homotopy_vertex_trace_mismatch"),
        ("open", "interval_homotopy_boundary_not_closed_oriented"),
        ("fold", "interval_homotopy_monotonicity_unresolved"),
        ("distance", "interval_homotopy_deviation_unresolved"),
    ),
)
def test_interval_boundary_homotopy_never_publishes_partial_proof(
    mutation: str, reason: str,
) -> None:
    from phydrax.geometry._mapped_embedding import certify_interval_boundary_homotopy

    cells = _analytic_octahedron(
        amplitude=0.01 if mutation == "distance" else 0.0,
        x_scale=-1.0 if mutation == "fold" else 1.0,
    )
    if mutation == "trace":
        cells = (
            replace(cells[0], source=replace(cells[0].source, authority="unrelated-source")),
            *cells[1:],
        )
    elif mutation == "open":
        cells = cells[:-1]
    limits = MeshCertificateLimits(maximum_subdivision_depth=4)
    with coordinate_reference_budget(limits, queries_upper=limits.maximum_source_samples) as budget:
        result = certify_interval_boundary_homotopy(
            cells, limits, tolerance=0.001 if mutation == "distance" else 4.0, budget=budget
        )
    assert not result.complete
    assert result.reason == reason
    assert result.proof_id is None
    assert np.isinf(result.distance_upper)


@pytest.mark.parametrize("crossing", (False, True), ids=("synchronized-motion", "true-crossing"))
def test_interval_contact_refines_common_time_not_independent_swept_tubes(
    crossing: bool,
) -> None:
    from phydrax.geometry._mapped_embedding import (
        _homotopy_disjoint,
        _homotopy_p2_coordinates,
        _IntervalHomotopyRefusal,
        _IntervalHomotopyWork,
    )

    points = np.asarray((
        (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0),
    ))
    first = _analytic_homotopy_cell(points, (0, 1, 2), 0, translation=(2.0, 0.0, 0.0))
    second = _analytic_homotopy_cell(
        points, (3, 4, 5), 1, translation=(-2.0 if crossing else 2.0, 0.0, 0.0)
    )
    limits = MeshCertificateLimits(maximum_subdivision_depth=6)
    with coordinate_reference_budget(limits, queries_upper=limits.maximum_source_samples) as budget:
        with budget.live_storage() as live:
            work = _IntervalHomotopyWork(limits, budget, live, 8192)
            items = [(
                first, second,
                (_homotopy_p2_coordinates(first.target),
                 _homotopy_p2_coordinates(second.target)),
                (2.0, 2.0),
            )]
            if crossing:
                with pytest.raises(
                    _IntervalHomotopyRefusal,
                    match="interval_homotopy_nonadjacent_contact_unresolved",
                ):
                    _homotopy_disjoint(items, work)
            else:
                _homotopy_disjoint(items, work)
                assert work.pieces > 1


def test_interval_homotopy_sharp_edge_uses_convex_reference_atlas() -> None:
    from phydrax.geometry._mapped_embedding import certify_interval_boundary_homotopy

    points = np.asarray((
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
    ))
    rows = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
    cells = tuple(
        _analytic_homotopy_cell(points, row, index, amplitude=0.01)
        for index, row in enumerate(rows)
    )
    limits = MeshCertificateLimits()
    with coordinate_reference_budget(limits, queries_upper=limits.maximum_source_samples) as budget:
        result = certify_interval_boundary_homotopy(cells, limits, tolerance=0.05, budget=budget)
    assert result.complete
    assert result.proof_id


def test_interval_homotopy_authenticates_full_target_edge_not_just_corners() -> None:
    from phydrax.geometry._mapped_embedding import certify_interval_boundary_homotopy

    cells = _analytic_octahedron()
    coordinates = list(cells[0].target)
    changed = dict(coordinates[2])
    # u(1-u-v) vanishes at all three corners but not along the 0--1 edge.
    for exponent, coefficient in (
        ((1, 0), Fraction(1, 8)),
        ((2, 0), Fraction(-1, 8)),
        ((1, 1), Fraction(-1, 8)),
    ):
        changed[exponent] = changed.get(exponent, Fraction(0)) + coefficient
    coordinates[2] = changed
    cells = (replace(cells[0], target=tuple(coordinates)), *cells[1:])
    limits = MeshCertificateLimits()
    with coordinate_reference_budget(limits, queries_upper=limits.maximum_source_samples) as budget:
        result = certify_interval_boundary_homotopy(cells, limits, tolerance=1.0, budget=budget)
    assert not result.complete
    assert result.reason == "interval_homotopy_edge_trace_mismatch"
    assert result.proof_id is None
    assert np.isinf(result.distance_upper)


def test_interval_homotopy_propagates_real_shared_storage_refusal() -> None:
    from phydrax.geometry._mapped_embedding import certify_interval_boundary_homotopy

    limits = MeshCertificateLimits(maximum_scratch_bytes=4096)
    with coordinate_reference_budget(limits, queries_upper=limits.maximum_source_samples) as budget:
        with pytest.raises(CoordinateEnclosureResourceError) as caught:
            certify_interval_boundary_homotopy(
                _analytic_octahedron(), limits, tolerance=0.05, budget=budget
            )
        assert caught.value.resource == "polynomial_storage"
        assert caught.value.requested > caught.value.limit
        assert budget.work_units == 0


class _TranslatedAuthoredTriangle(_AnalyticHomotopyTriangle):
    """Raw authored map is the representative plus the constant vector (0,0,.02)."""

    @property
    def map_id(self) -> str:
        return canonical_fingerprint((super().map_id, "raw-z-translation", 0.02))

    @property
    def authored_deviation_bound(self) -> float:
        return 0.02


@pytest.mark.parametrize("tolerance", (0.01, 0.03))
def test_interval_homotopy_accounts_for_authored_representative_distance(
    tolerance: float,
) -> None:
    from phydrax.geometry._mapped_embedding import certify_interval_boundary_homotopy

    cells = tuple(
        replace(cell, source=_TranslatedAuthoredTriangle(
            cell.anchor, cell.vertices, cell.target
        ))
        for cell in _analytic_octahedron()
    )
    limits = MeshCertificateLimits()
    with coordinate_reference_budget(limits, queries_upper=limits.maximum_source_samples) as budget:
        result = certify_interval_boundary_homotopy(
            cells, limits, tolerance=tolerance, budget=budget
        )
    if tolerance < 0.02:
        assert not result.complete
        assert result.reason == "interval_homotopy_authored_deviation_exceeds_tolerance"
        assert result.proof_id is None
        assert np.isinf(result.distance_upper)
    else:
        assert result.complete
        assert 0.02 <= result.distance_upper <= tolerance
        assert result.proof_id


class _ReflectedAuthoredTriangle(_AnalyticHomotopyTriangle):
    """Raw authored map reflects representative x; representative itself is unchanged."""

    @property
    def map_id(self) -> str:
        return canonical_fingerprint((super().map_id, "raw-x-reflection"))

    @property
    def authored_deviation_bound(self) -> float:
        # These tests use the unit octahedron: |raw - representative|=2|x|<=2.
        return 2.0

    def _orientation(self, reference, /, *, budget):
        lower, upper = self._jacobian(reference, budget=budget)
        raw_lower, raw_upper = lower.copy(), upper.copy()
        raw_lower[0], raw_upper[0] = -upper[0], -lower[0]
        return np.minimum(lower, raw_lower), np.maximum(upper, raw_upper)


def test_interval_homotopy_requires_raw_authored_orientation_not_only_completion() -> None:
    from phydrax.geometry._mapped_embedding import certify_interval_boundary_homotopy

    cells = tuple(
        replace(cell, source=_ReflectedAuthoredTriangle(
            cell.anchor, cell.vertices, cell.target
        ))
        for cell in _analytic_octahedron()
    )
    limits = MeshCertificateLimits(maximum_subdivision_depth=2)
    with coordinate_reference_budget(limits, queries_upper=limits.maximum_source_samples) as budget:
        result = certify_interval_boundary_homotopy(
            cells, limits, tolerance=3.0, budget=budget
        )
    assert not result.complete
    assert result.reason == "interval_homotopy_monotonicity_unresolved"
    assert result.proof_id is None
    assert np.isinf(result.distance_upper)


@pytest.mark.parametrize("authored_bound", (-1.0, float("inf"), float("nan")))
def test_interval_homotopy_refuses_invalid_authored_distance_bound(
    authored_bound: float,
) -> None:
    from phydrax.geometry._mapped_embedding import certify_interval_boundary_homotopy

    class InvalidAuthoredBound(_AnalyticHomotopyTriangle):
        @property
        def authored_deviation_bound(self) -> float:
            return authored_bound

    cells = _analytic_octahedron()
    first = cells[0]
    cells = (
        replace(first, source=InvalidAuthoredBound(
            first.anchor, first.vertices, first.target
        )),
        *cells[1:],
    )
    limits = MeshCertificateLimits()
    with coordinate_reference_budget(limits, queries_upper=limits.maximum_source_samples) as budget:
        result = certify_interval_boundary_homotopy(
            cells, limits, tolerance=0.05, budget=budget
        )
    assert not result.complete
    assert result.reason == "interval_homotopy_invalid_authored_bound"
    assert result.proof_id is None
    assert np.isinf(result.distance_upper)


def _native_homotopy_chart_source(budget):
    """Build one complete, source-exact box-face chain without meshing."""
    import phydrax as phx
    from phydrax.geometry.brep._intersection_curve import exact_curve_parameter_point

    model = phx.geometry.brep_box(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0),
        coordinate_contract=phx.SpatialCoordinateContract.si(),
        tessellation=phx.geometry.BRepTessellationPolicy(realize=False),
        budget=budget,
    )
    domain = phx.geometry.MeshingDomain.from_brep(model, budget=budget)
    loop = tuple(
        use for use in domain.patches[0].loops[0]
        if isinstance(use, phx.geometry.PatchCurveUse)
    )
    assert len(loop) == len(domain.patches[0].loops[0]) == 4
    exact: list[tuple[Fraction, ...]] = []
    for use in loop:
        row = exact_curve_parameter_point(
            use.pcurve, Fraction(use.first), coordinate_budget=budget
        )
        assert row is not None and len(row) == 2
        fractions = tuple(value for value in row if isinstance(value, Fraction))
        assert len(fractions) == 2
        exact.append(fractions)
    chart = np.asarray(exact, dtype=np.float64).reshape((-1, 2))
    assert all(
        Fraction(x) == expected
        for row, values in zip(chart.tolist(), exact, strict=True)
        for x, expected in zip(row, values, strict=True)
    )
    cells = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    boundary = np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int64)
    provenance = np.asarray([(0, i, use.first, use.last) for i, use in enumerate(loop)])
    points = np.asarray(domain.patches[0].surface.evaluate(jnp.asarray(chart)))
    data = (
        0, chart, points, cells, boundary, provenance,
        np.zeros(4, dtype=np.bool_), np.empty(0, dtype=np.int64),
        np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.int64),
    )
    return phx.geometry.MeshingDomainBoundarySource(
        domain, (0,), chart_triangulations=(data,)
    )


def _native_homotopy_source_cell(budget):
    """Issue real native maps from a source-exact box face, without meshing."""
    from phydrax.geometry._domain_chart_chain import prepare_source_chart_maps

    source = _native_homotopy_chart_source(budget)
    data = source.chart_triangulations[0]
    mapping = prepare_source_chart_maps(source.domain, data, budget=budget)[0]
    target = _analytic_homotopy_cell(data[2], (0, 1, 2), 0)
    return replace(target, source=mapping)


def test_chart_source_preparation_reuses_only_complete_current_request_proofs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import equinox as eqx
    from phydrax.geometry import _source_fidelity as fidelity

    limits = MeshCertificateLimits()
    calls = []
    from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource

    original = MeshingDomainBoundarySource.boundary_chart_cover

    def counted(source, *args, **kwargs):
        calls.append(source)
        return original(source, *args, **kwargs)

    monkeypatch.setattr(MeshingDomainBoundarySource, "boundary_chart_cover", counted)
    with coordinate_reference_budget(
        limits, queries_upper=limits.maximum_source_samples
    ) as budget:
        source = _native_homotopy_chart_source(budget)
        first = fidelity._prepare_chart_boundary_source(source, 64, budget=budget)
        assert first.cover.complete and first.maps
        retained = budget.retained_basis_bytes
        assert retained > 0
        assert fidelity._prepare_chart_boundary_source(source, 64, budget=budget) is first
        assert len(calls) == 1
        assert budget.retained_basis_bytes == retained
        # Unchanged asserted source IDs cannot authorize altered chain arrays.
        missing = eqx.tree_at(
            lambda item: item.chart_triangulations[0][3],
            source,
            source.chart_triangulations[0][3][:1],
        )
        refused = fidelity._prepare_chart_boundary_source(missing, 64, budget=budget)
        assert not refused.cover.complete and not refused.maps
        repeated = fidelity._prepare_chart_boundary_source(missing, 64, budget=budget)
        assert repeated is not refused and len(calls) == 3
        # The original proof remains valid; refused proofs never replace it.
        assert fidelity._prepare_chart_boundary_source(source, 64, budget=budget) is first
    with coordinate_reference_budget(
        limits, queries_upper=limits.maximum_source_samples
    ) as cold:
        renewed = fidelity._prepare_chart_boundary_source(source, 64, budget=cold)
        assert renewed is not first and len(calls) == 4
        assert renewed.cover.cover_id == first.cover.cover_id
        assert tuple(item.map_id for item in renewed.maps[0]) == tuple(
            item.map_id for item in first.maps[0]
        )


def test_chart_source_proof_never_keeps_its_budget_alive() -> None:
    import gc
    import weakref
    from phydrax.geometry import _source_fidelity as fidelity

    limits = MeshCertificateLimits()
    with coordinate_reference_budget(
        limits, queries_upper=limits.maximum_source_samples
    ) as budget:
        source = _native_homotopy_chart_source(budget)
        prepared = fidelity._prepare_chart_boundary_source(source, 64, budget=budget)
        reference = weakref.ref(budget)
    del budget
    gc.collect()
    assert reference() is None
    assert prepared.cover.complete and prepared.maps


def test_native_source_definition_memo_is_proof_local_and_reauthenticates_banks() -> None:
    from phydrax.geometry._domain_chart_chain import SourceChartMapError, source_chart_enclosure_identity
    from dataclasses import fields, is_dataclass
    from phydrax._strict import StrictModule
    from phydrax.geometry._domain_chart_chain import _source_atom
    from phydrax.geometry.brep._carrier_pieces import carrier_payload

    def canonical_definition(value):
        if isinstance(value, StrictModule):
            return canonical_fingerprint(carrier_payload(value))
        if isinstance(value, Fraction):
            return _source_atom(value)
        if is_dataclass(value) and not isinstance(value, type):
            return canonical_fingerprint((
                type(value).__name__,
                tuple(
                    (member.name, canonical_definition(getattr(value, member.name)))
                    for member in fields(value)
                    if member.name not in ("preparations", "memo", "points", "enclosure_authority")
                ),
            ))
        if isinstance(value, tuple):
            return canonical_fingerprint(tuple(canonical_definition(child) for child in value))
        return value

    limits = MeshCertificateLimits()
    with coordinate_reference_budget(
        limits, queries_upper=limits.maximum_source_samples
    ) as budget:
        cell = _native_homotopy_source_cell(budget)
        source = cell.source
        carriers = {}
        before = budget.work_units
        identity = source_chart_enclosure_identity(source, budget, carriers)
        first_work = budget.work_units - before
        assert identity is not None
        assert identity == canonical_fingerprint({
            "map": source.map_id,
            "authority": source.enclosure_authority[:2],
            "definition": canonical_definition(source),
        })
        assert all(id(endpoint) in carriers for endpoint in source.endpoints)
        before = budget.work_units
        assert source_chart_enclosure_identity(source, budget, carriers) == identity
        assert budget.work_units - before < first_work
        assert source_chart_enclosure_identity(source, budget, {}) == identity
        # A new proof authenticates actual nested coefficients, not asserted map IDs.
        original = source.surface.translation
        object.__setattr__(source.surface, "translation", original + jnp.asarray((0.0, 0.0, 0.25)))
        try:
            with pytest.raises(SourceChartMapError, match="requires_reprepare"):
                source_chart_enclosure_identity(source, budget, {})
        finally:
            object.__setattr__(source.surface, "translation", original)
        with pytest.raises(SourceChartMapError, match="genuine_reprepare"):
            source_chart_enclosure_identity(source, budget, {})
        source = _native_homotopy_source_cell(budget).source
        assert source_chart_enclosure_identity(source, budget, {}) == identity
    with coordinate_reference_budget(
        limits, queries_upper=limits.maximum_source_samples
    ) as cold:
        assert source_chart_enclosure_identity(source, cold, {}) is None


def test_source_carrier_dag_preserves_owner_admission_and_mutable_leaf_checks() -> None:
    from phydrax.geometry._domain_chart_chain import _admit_source_carrier_dag
    from phydrax.geometry.brep._carrier_pieces import admit_carrier_fingerprint, carrier_payload
    from phydrax._fingerprint import array_tree_fingerprint

    limits = MeshCertificateLimits()
    with coordinate_reference_budget(limits, queries_upper=limits.maximum_source_samples) as budget:
        source = _native_homotopy_source_cell(budget).source.surface
        canonical = canonical_fingerprint(carrier_payload(source))
        roots = (source, source, (source,))
        with budget.temporary_scope():
            before = budget.temporary_bytes_upper
            admit_carrier_fingerprint(roots, budget)
            original_bytes = budget.temporary_bytes_upper - before
        memo = {}
        with budget.temporary_scope():
            before = budget.temporary_bytes_upper
            _admit_source_carrier_dag(roots, budget, memo)
            assert budget.temporary_bytes_upper - before == original_bytes + 512 * len(memo)
        assert canonical_fingerprint(carrier_payload(source)) == canonical
        assert memo[id(source)].owner is source
        public = np.asarray((1.0, 2.0))
        mutable = (source, public)
        with budget.temporary_scope():
            first = _admit_source_carrier_dag(mutable, budget, memo)
        assert not first[3] and id(public) not in memo and id(mutable) not in memo
        before_mutation = array_tree_fingerprint(public)
        public[0] = 3.0
        with budget.temporary_scope():
            changed = _admit_source_carrier_dag(mutable, budget, memo)
        assert not changed[3] and id(public) not in memo
        assert array_tree_fingerprint(public) != before_mutation
        cyclic = []
        cyclic.append(cyclic)
        with pytest.raises(ValueError, match="acyclic"):
            _admit_source_carrier_dag(cyclic, budget, memo)


def test_homotopy_reuses_only_native_source_rows_across_full_proof_work():
    from phydrax.geometry._mapped_embedding import _IntervalHomotopyWork

    limits = MeshCertificateLimits()
    reference = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    with coordinate_reference_budget(limits, queries_upper=limits.maximum_source_samples) as budget:
        cell = _native_homotopy_source_cell(budget)
        with budget.live_storage() as live:
            first = _IntervalHomotopyWork(limits, budget, live, 8192)
            image = first.source_interval(cell, reference, jet=False)
            jet = first.source_interval(cell, reference, jet=True)
            orientation = first.source_interval(cell, reference, jet=True, orientation=True)
            target = first.target_interval(cell, "target-image", cell.target, reference)
            assert first.source_queries == 2
        source_bytes = budget._source_homotopy_enclosure_storage.bound
        assert source_bytes > 0 and budget.temporary_bytes_upper >= source_bytes
        moved = replace(cell, target=tuple(
            {**expression, (0, 0): expression.get((0, 0), Fraction(0)) + 2}
            for expression in cell.target
        ))
        with budget.live_storage() as live:
            second = _IntervalHomotopyWork(limits, budget, live, 8192)
            second.prefetch(((moved, reference, False), (moved, reference, True)))
            np.testing.assert_array_equal(second.source_interval(moved, reference, jet=False), image)
            np.testing.assert_array_equal(second.source_interval(moved, reference, jet=True), jet)
            np.testing.assert_array_equal(
                second.source_interval(moved, reference, jet=True, orientation=True), orientation
            )
            assert second.source_queries == 0
            moved_target = second.target_interval(moved, "target-image", moved.target, reference)
            np.testing.assert_allclose(moved_target, np.asarray(target) + 2, atol=1e-14, rtol=0)
        assert {key[0] for key in budget.source_homotopy_enclosure_cache} == {"image", "jet"}


@pytest.mark.parametrize("mutation", ("surface", "endpoint", "trim", "authority"))
def test_homotopy_source_mutations_never_hit_asserted_map_identity(mutation):
    import phydrax as phx
    from phydrax.geometry._mapped_embedding import _IntervalHomotopyWork
    from phydrax.geometry._domain_chart_chain import SourceChartMapError

    limits = MeshCertificateLimits()
    reference = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    with coordinate_reference_budget(limits, queries_upper=limits.maximum_source_samples) as budget:
        cell = _native_homotopy_source_cell(budget)
        with budget.live_storage() as live:
            first = _IntervalHomotopyWork(limits, budget, live, 8192)
            first.source_interval(cell, reference, jet=False)
        source = cell.source
        if mutation == "surface":
            changed = replace(source, surface=phx.geometry.PlanePatch(
                (9.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
            ), memo={})
        elif mutation == "endpoint":
            endpoint = replace(source.endpoints[0], parameter=Fraction(1, 2))
            changed = replace(source, endpoints=(endpoint, *source.endpoints[1:]), memo={})
        elif mutation == "trim":
            i, j, arc = source.arcs[0]
            arc = replace(arc, last=(arc.first + arc.last) / 2, points={})
            changed = replace(source, arcs=((i, j, arc), *source.arcs[1:]), memo={})
        else:
            changed = replace(source, enclosure_authority=None, memo={})
        assert changed.map_id == source.map_id
        with budget.live_storage() as live:
            second = _IntervalHomotopyWork(limits, budget, live, 8192)
            with pytest.raises(SourceChartMapError, match="genuine_reprepare"):
                second.source_interval(replace(cell, source=changed), reference, jet=False)
            assert second.source_queries == 0


def test_protocol_source_assertions_never_earn_crossproof_native_memo():
    from phydrax.geometry._mapped_embedding import _IntervalHomotopyWork

    class AssertedIdentity(_AnalyticHomotopyTriangle):
        @property
        def map_id(self):
            return "same-untrusted-assertion"

    limits = MeshCertificateLimits()
    reference = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    original = _analytic_octahedron()[0]
    first_cell = replace(original, source=AssertedIdentity(
        original.anchor, original.vertices, original.target
    ))
    second_cell = replace(original, source=AssertedIdentity(
        original.anchor, original.vertices, original.target, translation=(0.0, 0.0, 1.0)
    ))
    with coordinate_reference_budget(limits, queries_upper=limits.maximum_source_samples) as budget:
        results = []
        for cell in (first_cell, second_cell):
            with budget.live_storage() as live:
                work = _IntervalHomotopyWork(limits, budget, live, 8192)
                results.append(work.source_interval(cell, reference, jet=False))
                assert work.source_queries == 1
        assert not budget.source_homotopy_enclosure_cache
        assert not np.array_equal(results[0], results[1])


def test_native_source_lru_owns_only_copied_rows_and_releases_evicted_backing():
    import weakref

    limits = MeshCertificateLimits()
    batch = np.arange(3000, dtype=np.float64).reshape(1000, 3)
    row = ((batch[0], batch[1]),)
    with coordinate_reference_budget(limits) as budget:
        retained = budget.retained_basis_bytes
        first = budget.source_homotopy_enclosure(
            ("image", "first", b"reference"), row, maximum_entries=2
        )
        reference = weakref.ref(first[0][0])
        assert first[0][0].base is None and not first[0][0].flags.writeable
        assert not np.shares_memory(first[0][0], batch)
        del first
        for identity in ("second", "third"):
            budget.source_homotopy_enclosure(
                ("image", identity, b"reference"), row, maximum_entries=2
            )
        cache = budget.source_homotopy_enclosure_cache
        assert len(cache) == 2 and reference() is None
        assert budget.retained_basis_bytes == retained
        assert budget._source_homotopy_enclosure_storage.bound == 256 + sum(
            budget._source_homotopy_entry_bytes(key, value) for key, value in cache.items()
        )




def test_native_completed_source_points_keep_exact_immutable_host_bank() -> None:
    import phydrax as phx
    from phydrax.discretization._coordinate_operands import _immutable_array_layout
    from phydrax.geometry._domain_chart_chain import prepare_source_chart_domain

    model = phx.geometry.brep_cone(
        1.0, 0.5, 2.0,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
        tessellation=phx.geometry.BRepTessellationPolicy(realize=False),
    )
    domain = phx.geometry.MeshingDomain.from_brep(model)
    original = np.asarray(model.require_geometry().vertex_points)
    limits = MeshCertificateLimits()
    with coordinate_reference_budget(
        limits, queries_upper=limits.maximum_source_samples
    ) as budget:
        prepared = prepare_source_chart_domain(domain, budget=budget)
        assert prepared.completion is not None
        backing = set()
        vertices = {
            id(vertex): vertex
            for edge in prepared.completion.edges for vertex in edge.vertices
        }
        for vertex in vertices.values():
            assert type(vertex.point) is np.ndarray
            vertex.require_point_layout()
            layout = _immutable_array_layout(vertex.point)
            backing.add(id(layout[-1]))
            assert any(
                canonical_fingerprint(vertex.point) == canonical_fingerprint(row)
                for row in original
            )
            with pytest.raises(ValueError):
                vertex.point.setflags(write=True)
        assert len(backing) == 1
        vertex = next(iter(vertices.values()))
        reshaped = replace(vertex, point=vertex.point.reshape((1, 3)))
        with pytest.raises(ValueError, match="point_layout_changed"):
            reshaped.require_point_layout()
        with pytest.raises(ValueError, match="point_layout_changed"):
            reshaped.bounds(budget)
        vertex.require_point_layout()
        readonly = np.array(vertex.point, copy=True)
        readonly.setflags(write=False)
        foreign = replace(vertex, point=readonly)
        with pytest.raises(ValueError, match="reversible NumPy write flag"):
            foreign.require_point_layout()
    np.testing.assert_array_equal(model.require_geometry().vertex_points, original)


def _parent_surface_partition_case(*, homogeneous: bool = False):
    from phydrax.discretization._cell_geometry import (
        BarycentricCellGeometryElement, CellGeometryRestrictionSource, FiniteElementSpec,
    )
    from phydrax.discretization._cell_geometry_validity import cell_geometry_id

    source_points = np.asarray(
        [(3 * parent + x, y, 0) for parent in range(4)
         for x, y in ((0, 0), (1, 0), (0, 1))],
        dtype=np.float64,
    )
    source_rows = np.arange(12, dtype=np.int32).reshape((-1, 3))
    parent_ids = np.asarray((10, 20, 30, 40), dtype=np.int64)
    source_mesh = CellMesh(source_points, (
        CellBlock("parent", "triangle", source_rows, global_ids=parent_ids),
    ))
    source_geometry = CellGeometrySpec.affine(source_mesh)
    root = source_geometry.elements[0]
    assert isinstance(root, FiniteElementSpec)
    fraction = Fraction(1, 3) if homogeneous else Fraction(1, 2)
    midpoint = source_points[source_rows[:, 0]].copy()
    midpoint[:, 1] = float(fraction)
    target_points = np.vstack((source_points, midpoint))
    first_rows, second_rows = source_rows.copy(), source_rows.copy()
    first_rows[:, 2] = np.arange(12, 16)
    second_rows[:, 0] = np.arange(12, 16)
    mesh = CellMesh(target_points, (
        CellBlock("first", "triangle", first_rows, global_ids=np.arange(100, 104)),
        CellBlock("second", "triangle", second_rows, global_ids=np.arange(104, 108)),
    ))
    first, second = np.eye(3), np.eye(3)
    first[2] = (float(1 - fraction), 0, float(fraction))
    second[0] = first[2]
    first_homogeneous = second_homogeneous = None
    if homogeneous:
        first_homogeneous, second_homogeneous = np.zeros((3, 3)), np.zeros((3, 3))
        first_homogeneous[2] = (2, 0, 1)
        second_homogeneous[0] = (2, 0, 1)
    geometry = CellGeometrySpec(
        {
            "first": BarycentricCellGeometryElement(
                root, first, barycentric_homogeneous=first_homogeneous,
            ),
            "second": BarycentricCellGeometryElement(
                root, second, barycentric_homogeneous=second_homogeneous,
            ),
        },
        {"first": source_rows, "second": source_rows},
        source_geometry.coordinates,
        restriction_source=CellGeometryRestrictionSource(
            cell_geometry_id(source_geometry), source_mesh.topology_id,
            {"first": parent_ids, "second": parent_ids},
            {"first": source_rows, "second": source_rows},
            block_source_blocks={"first": "parent", "second": "parent"},
        ),
    )
    return mesh, geometry, source_points, source_rows, fraction


@pytest.mark.parametrize("homogeneous", (False, True))
def test_parent_surface_partition_batches_actual_root_corner_laws(
    homogeneous: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from phydrax.geometry import _mapped_embedding as mapped

    mesh, geometry, points, routes, fraction = _parent_surface_partition_case(
        homogeneous=homogeneous
    )
    batch_calls, cell_calls = [], []
    original_batch = mapped.coordinate_enclosure_owner.coordinate_vertex_images
    original_cell = mapped.coordinate_enclosure_owner.coordinate_corner_images

    def batch(*args, **kwargs):
        if sys._getframe(1).f_code.co_name == "_prepare_parent_partition":
            batch_calls.append(args[0])
        return original_batch(*args, **kwargs)

    def cell(*args, **kwargs):
        if sys._getframe(1).f_code.co_name == "_prepare_parent_partition":
            cell_calls.append(args[0])
        return original_cell(*args, **kwargs)

    monkeypatch.setattr(mapped.coordinate_enclosure_owner, "coordinate_vertex_images", batch)
    monkeypatch.setattr(mapped.coordinate_enclosure_owner, "coordinate_corner_images", cell)
    with coordinate_reference_budget() as budget:
        with budget.temporary_scope():
            partition = mapped._prepare_parent_partition(mesh, geometry, budget=budget)
        assert partition is not None
        assert len(batch_calls) == len(mesh.blocks) == 2
        assert not cell_calls
        for parent, row in zip((10, 20, 30, 40), routes, strict=True):
            assert partition.vertices[parent] == tuple(row)
            assert partition.corners[parent] == tuple(
                tuple(Fraction(float(value)) for value in points[index]) for index in row
            )
            assert partition.children[parent] == [
                ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)),
                 (Fraction(0), fraction)),
                ((Fraction(0), fraction), (Fraction(1), Fraction(0)),
                 (Fraction(0), Fraction(1))),
            ]
        assert budget.temporary_bytes_upper == 0


def test_parent_surface_partition_does_not_alias_asserted_chart_ids() -> None:
    from phydrax.geometry import _mapped_embedding as mapped

    mesh, geometry, _, _, _ = _parent_surface_partition_case()
    first, second = geometry.elements
    original_id = second.element_id
    object.__setattr__(second, "element_id", first.element_id)
    try:
        with coordinate_reference_budget() as budget:
            partition = mapped._prepare_parent_partition(mesh, geometry, budget=budget)
        assert partition is not None
        assert partition.children[10][0] != partition.children[10][1]
    finally:
        object.__setattr__(second, "element_id", original_id)


@pytest.mark.parametrize("mutation", ("coordinates", "routes", "chart", "carrier"))
def test_parent_surface_partition_reauthenticates_stale_actual_owners(mutation) -> None:
    from phydrax.geometry import _mapped_embedding as mapped

    mesh, geometry, _, _, _ = _parent_surface_partition_case()
    with coordinate_reference_budget() as budget:
        assert mapped._prepare_parent_partition(mesh, geometry, budget=budget) is not None
        owner, field, old = geometry, "coordinates", geometry.coordinates
        changed_mesh = mesh
        if mutation == "coordinates":
            new = old.at[0, 2].set(0.25)
        elif mutation == "routes":
            field, old = "geometry_dofs", geometry.geometry_dofs
            new = (old[0].at[0, 0].set(1), *old[1:])
        elif mutation == "chart":
            owner, field = geometry.elements[0], "barycentric_weights"
            old = owner.barycentric_weights
            new = old.at[2].set(jnp.asarray((0.25, 0.0, 0.75)))
        else:
            changed = np.asarray(mesh.coordinates).copy()
            changed[12, 2] = 0.25
            changed_mesh = mesh.with_coordinates(changed, numeric_version="stale-carrier")
            new = old
        object.__setattr__(owner, field, new)
        try:
            with pytest.raises(ValueError):
                mapped._prepare_parent_partition(changed_mesh, geometry, budget=budget)
        finally:
            object.__setattr__(owner, field, old)


def test_parent_surface_partition_exact_work_boundary_releases_refused_scratch() -> None:
    from phydrax.geometry import _mapped_embedding as mapped

    mesh, geometry, _, _, _ = _parent_surface_partition_case(homogeneous=True)

    def run(budget):
        with budget.temporary_scope():
            result = mapped._prepare_parent_partition(mesh, geometry, budget=budget)
        assert result is not None
        return result

    with coordinate_reference_budget() as budget:
        expected = run(budget)
        required = budget.work_units
    with coordinate_reference_budget(MeshCertificateLimits(maximum_work_units=required)) as budget:
        actual = run(budget)
        assert actual.corners == expected.corners
        assert actual.children == expected.children
        assert budget.work_units == required
    with coordinate_reference_budget(MeshCertificateLimits(maximum_work_units=required - 1)) as budget:
        with pytest.raises(CoordinateEnclosureResourceError) as caught:
            run(budget)
        assert caught.value.resource == "coefficient_work"
        assert caught.value.requested > required - 1
        assert budget.work_units <= required - 1
        assert budget.temporary_bytes_upper == 0


def test_parent_surface_partition_keeps_unsupported_batch_corner_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from phydrax.geometry import _mapped_embedding as mapped

    mesh, geometry, _, _, _ = _parent_surface_partition_case(homogeneous=True)
    with coordinate_reference_budget() as budget:
        expected = mapped._prepare_parent_partition(mesh, geometry, budget=budget)
    assert expected is not None
    original = mapped.coordinate_enclosure_owner.coordinate_corner_images
    calls = []

    def cell(*args, **kwargs):
        if sys._getframe(1).f_code.co_name == "_prepare_parent_partition":
            calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(
        mapped.coordinate_enclosure_owner, "coordinate_vertex_images",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(mapped.coordinate_enclosure_owner, "coordinate_corner_images", cell)
    with coordinate_reference_budget() as budget:
        actual = mapped._prepare_parent_partition(mesh, geometry, budget=budget)
    assert actual is not None
    assert len(calls) == sum(block.cell_count for block in mesh.blocks)
    assert actual.vertices == expected.vertices
    assert actual.corners == expected.corners
    assert actual.children == expected.children


def test_prepared_native_map_rejects_changed_public_source_and_old_local_caches() -> None:
    import equinox as eqx
    from phydrax.geometry._domain_chart_chain import (
        prepare_source_chart_maps, SourceChartMapError, source_chart_enclosure_identity,
    )
    from phydrax.geometry._mapped_embedding import _IntervalHomotopyWork

    limits = MeshCertificateLimits()
    reference = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    with coordinate_reference_budget(limits, queries_upper=limits.maximum_source_samples) as budget:
        query = _native_homotopy_chart_source(budget)
        public = np.array(query.domain.patches[0].surface.translation, copy=True)
        public.setflags(write=False)
        query = eqx.tree_at(lambda value: value.domain.patches[0].surface.translation, query, public)
        data = query.chart_triangulations[0]
        source = prepare_source_chart_maps(query.domain, data, budget=budget)[0]
        cell = replace(_analytic_homotopy_cell(data[2], (0, 1, 2), 0), source=source)
        with budget.live_storage() as live:
            first = _IntervalHomotopyWork(limits, budget, live, 8192)
            first.source_interval(cell, reference, jet=False)
            first.source_interval(cell, reference, jet=True)
        old_identity = source_chart_enclosure_identity(source, budget, {})
        assert source.memo
        initial = float(public[0])
        public.setflags(write=True)
        public[0] = initial + 0.25
        public.setflags(write=False)
        with budget.live_storage() as live:
            second = _IntervalHomotopyWork(limits, budget, live, 8192)
            with pytest.raises(SourceChartMapError, match="requires_reprepare"):
                second.source_interval(cell, reference, jet=False)
            assert second.source_queries == 0
        public.setflags(write=True)
        public[0] = initial
        public.setflags(write=False)
        with pytest.raises(SourceChartMapError, match="genuine_reprepare"):
            source_chart_enclosure_identity(source, budget, {})
        fresh = prepare_source_chart_maps(query.domain, data, budget=budget)[0]
        assert source_chart_enclosure_identity(fresh, budget, {}) == old_identity
