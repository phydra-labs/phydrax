#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Invariant straight-piece compatible moments against the generic integral."""

from dataclasses import replace
from fractions import Fraction
from typing import Literal

from phydrax import config as pxconfig


pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.discretization as D
import phydrax.discretization.fem._sphere_chart_transfer as sphere
import phydrax.discretization.fem._surface_chart_compatible as compatible
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization import prepare_sphere_chart_deformation
from phydrax.discretization import _coordinate_enclosure as algebra
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._sphere_chart_records import (
    _SpherePieceGeometry,
    PreparedSphereChartDeformation,
)
from phydrax.discretization.fem import (
    form_element,
    prepare_sphere_chart_compatible_transfer,
)
from phydrax.discretization.fem._exact_form_moments import ExactFormMoments
from phydrax.discretization.fem._form_elements import form_entity_chart
from phydrax.discretization.fem._mapped_form_transfer import (
    derivative_cell,
    form_cells,
    FormCell,
    PreparationWork,
    transform_integrals,
)
from phydrax.geometry._rational import fraction_endpoint
from phydrax.sparse import SparseLinearMap
from tests._support.sphere_material import accepted_sphere_material


LIMIT = 10**10
STORAGE = 10**11


@pytest.fixture(scope="module")
def sphere_chart() -> PreparedSphereChartDeformation:
    # Two independently generated native sphere meshes: genuine projective
    # pieces whose target charts are rational, not affine corner fits.
    _, _, source_atlas = accepted_sphere_material(1.2)
    _, _, target_atlas = accepted_sphere_material(1.0)
    return prepare_sphere_chart_deformation(
        source_atlas, target_atlas, maximum_fidelity=0.3, maximum_displacement=0.5
    )


def _endpoints(
    chart: PreparedSphereChartDeformation,
    order: int,
    fields: tuple[tuple[str, Literal["untwisted", "twisted"]], ...],
) -> tuple[D.FiniteElementDiscretization, D.FiniteElementDiscretization]:
    specs = tuple(
        D.FiniteElementFieldSpec(
            name,
            form_element(
                "triangle",
                1,
                order,
                family="trimmed",
                twist=twist,
                proxy="circulation" if twist == "untwisted" else "flux",
            ),
        )
        for name, twist in fields
    )
    return (
        D.FiniteElementPlan(
            chart.source_mesh, specs, coordinate_spec=chart.source_geometry
        ).prepare(),
        D.FiniteElementPlan(
            chart.target_mesh, specs, coordinate_spec=chart.target_geometry
        ).prepare(),
    )


def _ledger() -> tuple[HostBatchTicket, CoordinateEnclosureBudget]:
    ticket = HostBatchTicket(LIMIT, bytes_upper=STORAGE)
    return ticket, CoordinateEnclosureBudget(
        ticket, maximum_work_units=LIMIT, maximum_memory_bytes=STORAGE
    )


@pytest.mark.parametrize("twist", ["untwisted", "twisted"])
def test_invariant_sphere_moments_equal_generic_common_entities(
    sphere_chart: PreparedSphereChartDeformation,
    twist: Literal["untwisted", "twisted"],
) -> None:
    source, target = _endpoints(sphere_chart, 1, (("material", twist),))
    ticket, budget = _ledger()
    with ticket, budget.activate():
        indices = sphere._validate_sphere(
            source,
            target,
            sphere_chart,
            sphere_chart.source_geometry,
            sphere_chart.target_geometry,
            budget=budget,
            validation=None,
        )
        sources, targets = form_cells(source, 0), form_cells(target, 0)
        work = PreparationWork(LIMIT, STORAGE, budget=budget)
        exact = ExactFormMoments(work.enclosure)
        derivatives: dict = {}
        derivative_sources: list[FormCell] = []
        companion_sources: list[FormCell] = []
        compatible._prepare_material_source_differentials(
            sources,
            derivatives,
            work,
            budget,
            [],
            {},
            derivative_sources,
            companion_sources,
            0,
        )
        plan = compatible._invariant_plan(
            sources,
            derivative_sources,
            targets,
            derivatives,
            work,
            exact,
            budget,
            compatible.MaterialInvariantCache(),
        )
        assert plan is not None
        pieces = tuple(
            compatible.MaterialCompatiblePiece(
                indices[0][piece.source_cell_global_id],
                indices[1][piece.target_cell_global_id],
                piece.exact_source_reference_vertices,
                piece.exact_target_reference_vertices,
                piece.source_reference_area,
                piece.target_reference_area,
                sphere._sphere_piece_charts(piece, budget=budget),
            )
            for piece in sphere_chart.pieces
        )
        compared = 0
        for row, cell in enumerate(targets):
            selected = tuple(piece for piece in pieces if piece.target_row == row)
            following = derivative_cell(
                FormCell(cell.basis, cell.routes, np.eye(cell.basis.local_dof_count)),
                derivatives,
                budget=budget,
            ).basis
            columns = np.unique(
                np.concatenate([sources[piece.source_row].routes for piece in selected])
            )
            companion_columns = np.unique(
                np.concatenate(
                    [companion_sources[piece.source_row].routes for piece in selected]
                )
            )
            for cells, basis, owned in (
                (sources, cell.basis, columns),
                (tuple(derivative_sources), following, columns),
                (tuple(companion_sources), following, companion_columns),
            ):
                # Whole-pass moments, continuity and coverage agree.
                generic, generic_defect = compatible._generic_common_moments(
                    cells, basis, selected, owned, work, exact
                )
                fast, fast_defect = compatible._invariant_common_moments(
                    cells, basis, selected, owned, work, plan
                )
                assert np.all(
                    np.abs(generic.value - fast.value) <= generic.error + fast.error
                )
                # The generic trace certificate also includes rounding error,
                # even when both traces represent the same exact integral.
                assert fast_defect <= generic_defect + np.finfo(np.float64).eps
                # Every native entity: same incidence, key and exact moment.
                for entity in plan.targets[basis.basis_id]:
                    natives = ((0, 1), (1, 2), (2, 0)) if entity.line else ((0, 1, 2),)
                    for piece in selected:
                        moving = cells[piece.source_row]
                        weights = compatible._route_weights(
                            moving, plan.sources[moving.basis.basis_id], work, plan.cache
                        )
                        sign = (
                            1
                            if twist == "untwisted"
                            or (piece.source_area > 0) == (piece.target_area > 0)
                            else -1
                        )
                        assert piece.charts is not None
                        for native in natives:
                            with (
                                work.enclosure.temporary_scope(),
                                work.enclosure.polynomial_preparation_scope(),
                            ):
                                reference = exact.common_entity(
                                    moving.basis,
                                    basis,
                                    entity.entity,
                                    "triangle",
                                    native,
                                    *piece.charts,
                                )
                            image = compatible._invariant_image(
                                moving,
                                weights,
                                piece,
                                sign,
                                native,
                                entity,
                                work,
                                plan.cache,
                            )
                            assert (reference is None) == (image is None)
                            if reference is None or image is None:
                                continue
                            raw, key = reference
                            images, contribution, _ = image
                            assert key == tuple(sorted(images))
                            transformed = transform_integrals(raw, moving.transform)
                            invariant = np.zeros_like(transformed.value)
                            for target_row, density in contribution.rows:
                                invariant[target_row] = [
                                    float(contribution.sign * density * value)
                                    for value in contribution.values
                                ]
                            assert np.all(
                                np.abs(transformed.value - invariant)
                                <= transformed.error + np.spacing(np.abs(invariant))
                            )
                            compared += 1
        assert compared > len(pieces)


def test_direct_projective_condition_never_exceeds_the_generic_enclosure(
    sphere_chart: PreparedSphereChartDeformation,
) -> None:
    ticket, budget = _ledger()
    with ticket, budget.activate():
        published = sphere._sphere_projective_condition(
            sphere_chart.pieces, budget=budget
        )
        generic_upper = 0.0
        largest = Fraction(0)
        for piece in sphere_chart.pieces:
            generic = sphere._generic_projective_condition(piece, budget=budget)
            direct = sphere._projective_controls(piece, budget=budget)
            # Same control net without the outward binary64 rounding.
            assert direct <= generic
            generic_upper = max(
                generic_upper, fraction_endpoint(generic, upper=True, saturate=True)
            )
            largest = max(largest, direct)
    # Pruning by the cheap bound never hides the largest piece.
    assert published == fraction_endpoint(largest, upper=True, saturate=True)
    assert published <= generic_upper


def test_nonconstant_or_interior_tests_keep_the_generic_rational_path(
    sphere_chart: PreparedSphereChartDeformation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"generic": 0, "charts": 0}
    generic = compatible._generic_common_moments
    charts = sphere._sphere_piece_charts

    def counted_generic(*args, **kwargs):
        calls["generic"] += 1
        return generic(*args, **kwargs)

    def counted_charts(*args, **kwargs):
        calls["charts"] += 1
        return charts(*args, **kwargs)

    monkeypatch.setattr(compatible, "_generic_common_moments", counted_generic)
    monkeypatch.setattr(sphere, "_sphere_piece_charts", counted_charts)

    # Order-two edge densities and interior tests reject the invariant rule.
    # Exercise their actual rational pullback on a whole projective triangle
    # whose modest denominator contrast fits the unchanged 32-term rule.
    # The native-remesh order-two curl/flux publications are independently
    # covered by test_material_chart_transfer; do not publish a third copy here.
    ticket, budget = _ledger()
    with ticket, budget.activate():
        exact = ExactFormMoments(budget)
        work = PreparationWork(LIMIT, STORAGE, budget=budget)
        _, piece, _ = _projective_triangle(
            1, "untwisted", False, False, work, exact, contrast=Fraction(1, 8)
        )
        basis = form_element(
            "triangle", 1, 2, family="trimmed", twist="untwisted", proxy="components"
        ).form_basis
        assert basis is not None
        assert compatible._invariant_target(basis, exact, work) is None
        cell = FormCell(
            basis, np.arange(basis.local_dof_count), np.eye(basis.local_dof_count)
        )
        moments, defect = compatible._material_common_moments(
            (cell,), basis, (piece,), cell.routes, work, exact, None
        )
        assert np.all(np.isfinite(moments.value))
        assert np.all(np.isfinite(moments.error))
        assert np.all(moments.error >= 0)
        assert defect == 0.0
    assert calls == {"generic": 1, "charts": 0}

    # Order one never builds a rational chart or a generic moment.
    calls.update(generic=0, charts=0)
    source, target = _endpoints(sphere_chart, 1, (("material", "untwisted"),))
    prepared = prepare_sphere_chart_compatible_transfer(
        source,
        target,
        sphere_chart,
        field_name="material",
        source_geometry=sphere_chart.source_geometry,
        target_geometry=sphere_chart.target_geometry,
    )
    assert prepared.field_transfer.evidence.passed
    assert calls == {"generic": 0, "charts": 0}


def test_request_proof_shares_condition_and_invariant_moments(
    sphere_chart: PreparedSphereChartDeformation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"condition": 0}
    condition = sphere._sphere_projective_condition

    def counted(*args, **kwargs):
        calls["condition"] += 1
        return condition(*args, **kwargs)

    monkeypatch.setattr(sphere, "_sphere_projective_condition", counted)
    source, target = _endpoints(
        sphere_chart, 1, (("curl", "untwisted"), ("flux", "twisted"))
    )
    ticket, budget = _ledger()
    work: dict[str, int] = {}
    published = {}
    with ticket, budget.activate():
        validation = sphere_chart.require_bound(
            sphere_chart.source_mesh,
            sphere_chart.source_geometry,
            sphere_chart.target_mesh,
            sphere_chart.target_geometry,
            budget=budget,
        )
        for name in ("curl", "flux"):
            before = budget.work_units
            prepared = prepare_sphere_chart_compatible_transfer(
                source,
                target,
                sphere_chart,
                field_name=name,
                source_geometry=sphere_chart.source_geometry,
                target_geometry=sphere_chart.target_geometry,
                budget=budget,
                validation=validation,
            )
            work[name] = budget.work_units - before
            assert prepared.field_transfer.evidence.passed
            published[name] = dict(prepared.field_transfer.evidence.estimates)
            index = source.field_index(name)
            old = jnp.sin(
                jnp.arange(source.dof_maps[index].global_dof_count, dtype=jnp.float64)
            )
            transfer = prepared.field_transfer.transfer
            actual = transfer.apply(old)
            assert isinstance(transfer.primal, SparseLinearMap)
            matrix = np.asarray(transfer.primal.as_dense())
            np.testing.assert_allclose(
                actual, matrix @ np.asarray(old), rtol=0, atol=1e-12
            )
            # Check every published column, not just the certificate's own
            # defect calculation or one favorable coefficient vector.
            source_d = np.asarray(prepared.source_differential.as_dense())
            target_d = np.asarray(prepared.target_differential.as_dense())
            companion = np.asarray(prepared.differential_companion.as_dense())
            np.testing.assert_allclose(
                target_d @ matrix, companion @ source_d, rtol=0, atol=1e-12
            )
            load = jnp.cos(jnp.arange(actual.size, dtype=jnp.float64))
            pulled = transfer.pullback(load)
            np.testing.assert_allclose(
                pulled, matrix.T @ np.asarray(load), rtol=0, atol=1e-12
            )
            np.testing.assert_allclose(
                jnp.vdot(load, actual), jnp.vdot(pulled, old), rtol=0, atol=1e-12
            )
            assert transfer.semantics == (
                "covariant-piola" if name == "curl" else "contravariant-piola"
            )
            assert dict(prepared.field_transfer.evidence.bounds)[
                "geometry-displacement"
            ] == sphere_chart.maximum_displacement_bound
    assert calls["condition"] == 1
    # Positive projective Jacobians make both orientation signs agree, so the
    # twisted field reuses every exact moment of the untwisted one.
    assert work["flux"] < work["curl"]
    assert (
        published["curl"]["moment_condition"] == published["flux"]["moment_condition"]
    )


def _projective_triangle(
    degree: int,
    twist: Literal["untwisted", "twisted"],
    reverse_source: bool,
    reverse_target: bool,
    work: PreparationWork,
    exact: ExactFormMoments,
    *,
    contrast: Fraction = Fraction(1),
) -> tuple[FormCell, compatible.MaterialCompatiblePiece, compatible._InvariantPlan]:
    """Independent rational chart of the full triangle, including reversals."""
    basis = form_element(
        "triangle", degree, 1, family="trimmed", twist=twist, proxy="components"
    ).form_basis
    assert basis is not None
    cell = FormCell(basis, np.arange(basis.local_dof_count), np.eye(basis.local_dof_count))
    vertices = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    source_vertices = (
        (vertices[0], vertices[2], vertices[1]) if reverse_source else vertices
    )
    target_vertices = (
        (vertices[0], vertices[2], vertices[1]) if reverse_target else vertices
    )
    # t=((1+c)u,(1+2c)v)/(1+cu+2cv) is genuinely projective for
    # c>0, has a positive denominator and fixes all three vertices.
    denominator: algebra.Polynomial = {
        (0, 0): Fraction(1), (1, 0): contrast, (0, 1): 2 * contrast
    }
    target_charts = (
        algebra.rational_expression(
            {(1, 0): 1 + contrast}, denominator, budget=work.enclosure
        ),
        algebra.rational_expression(
            {(0, 1): 1 + 2 * contrast}, denominator, budget=work.enclosure
        ),
    )
    if reverse_target:
        target_charts = (target_charts[1], target_charts[0])
    piece = compatible.MaterialCompatiblePiece(
        0,
        0,
        source_vertices,
        target_vertices,
        Fraction(-1 if reverse_source else 1, 2),
        Fraction(-1 if reverse_target else 1, 2),
        (
            compatible._material_affine_arguments(source_vertices, budget=work.enclosure),
            target_charts,
        ),
    )
    entities = compatible._invariant_target(basis, exact, work)
    components = compatible._linear_components(exact, basis)
    assert entities is not None and components is not None
    plan = compatible._InvariantPlan(
        {basis.basis_id: entities},
        {basis.basis_id: components},
        compatible.MaterialInvariantCache(),
    )
    return cell, piece, plan


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("twist", ["untwisted", "twisted"])
@pytest.mark.parametrize(
    "reverse_source,reverse_target",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_projective_line_and_area_invariance_with_both_chart_orientations(
    degree: int,
    twist: Literal["untwisted", "twisted"],
    reverse_source: bool,
    reverse_target: bool,
) -> None:
    ticket, budget = _ledger()
    with ticket, budget.activate():
        work = PreparationWork(LIMIT, STORAGE, budget=budget)
        exact = ExactFormMoments(budget)
        cell, piece, plan = _projective_triangle(
            degree, twist, reverse_source, reverse_target, work, exact
        )
        generic, _ = compatible._generic_common_moments(
            (cell,), cell.basis, (piece,), cell.routes, work, exact
        )
        invariant, defect = compatible._invariant_common_moments(
            (cell,), cell.basis, (piece,), cell.routes, work, plan
        )
        assert defect == 0.0
        assert np.all(
            np.abs(generic.value - invariant.value) <= generic.error + invariant.error
        )
        # Independent floating-point integration uses tabulated basis values,
        # not the exact component/route-weight primitives under test.
        numerical = np.zeros_like(invariant.value)
        native_source = np.asarray(piece.source_vertices, dtype=np.float64)
        native_target = np.asarray(piece.target_vertices, dtype=np.float64)
        twist_sign = -1 if twist == "twisted" and reverse_source != reverse_target else 1
        nodes, weights = np.polynomial.legendre.leggauss(4)
        for entity in plan.targets[cell.basis.basis_id]:
            _, _, chart = form_entity_chart(2, cell.basis.family, entity.entity)
            if degree == 1:
                reference = np.asarray(((0, 0), (1, 0), (0, 1)), dtype=np.float64)
                first = next(
                    index
                    for index, point in enumerate(native_target)
                    if np.array_equal(point, reference[entity.entity[0]])
                )
                last = next(
                    index
                    for index, point in enumerate(native_target)
                    if np.array_equal(point, reference[entity.entity[1]])
                )
                tangent = native_source[last] - native_source[first]
                points = native_source[first] + (nodes[:, None] + 1) * tangent / 2
                values, _ = cell.basis.tabulate_components(jnp.asarray(points))
                integrated = np.einsum(
                    "q,qbc,c->b", weights / 2, np.asarray(values), tangent
                )
            else:
                points = np.asarray(((1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3)))
                frame = (native_source[1:] - native_source[0]).T
                values, _ = cell.basis.tabulate_components(
                    jnp.asarray(native_source[0] + points @ frame.T)
                )
                # Source chart determinant and target/full entity orientation.
                sign = (-1 if reverse_target else 1) * np.sign(np.linalg.det(chart))
                integrated = (
                    np.asarray(values)[..., 0].sum(axis=0)
                    * np.linalg.det(frame)
                    * sign
                    / 6
                )
            for row, density in entity.rows:
                numerical[row] = float(density) * twist_sign * integrated
        np.testing.assert_allclose(invariant.value, numerical, rtol=0, atol=1e-11)


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("reverse_source,reverse_target", [(False, True), (True, False)])
def test_exact_image_reuse_is_independent_of_routes_densities_and_twist(
    degree: int, reverse_source: bool, reverse_target: bool
) -> None:
    ticket, budget = _ledger()
    with ticket, budget.activate():
        work = PreparationWork(LIMIT, STORAGE, budget=budget)
        exact = ExactFormMoments(budget)
        cell, piece, plan = _projective_triangle(
            degree, "untwisted", reverse_source, reverse_target, work, exact
        )
        entity = plan.targets[cell.basis.basis_id][0]
        weights = compatible._route_weights(
            cell, plan.sources[cell.basis.basis_id], work, plan.cache
        )
        natives = ((0, 1), (1, 2), (2, 0)) if degree == 1 else ((0, 1, 2),)
        admitted = [
            (
                native,
                compatible._invariant_image(
                    cell, weights, piece, 1, native, entity, work, plan.cache
                ),
            )
            for native in natives
        ]
        assert any(image is not None for _, image in admitted)
        before = budget.work_units
        moved = FormCell(cell.basis, cell.routes + 17, cell.transform)
        different_rows = entity._replace(
            rows=tuple((row + 5, density * 3) for row, density in entity.rows)
        )
        for native, previous in admitted:
            repeated = compatible._invariant_image(
                moved, weights, piece, -1, native, different_rows, work, plan.cache
            )
            if previous is None:
                assert repeated is None
                continue
            assert repeated is not None
            assert repeated[0] == previous[0]
            assert repeated[2] == previous[2]
            assert repeated[1].values is previous[1].values
            assert repeated[1].sign == -previous[1].sign
            assert repeated[1].rows == different_rows.rows
            np.testing.assert_array_equal(repeated[1].routes, moved.routes)
        # Only exact cached geometry/integrals are reused, not contributions.
        assert budget.work_units == before
        if degree == 2:
            changed = piece._replace(source_area=2 * piece.source_area)
            image = compatible._invariant_image(
                cell, weights, changed, 1, (0, 1, 2), entity, work, plan.cache
            )
            previous = admitted[0][1]
            assert image is not None and previous is not None
            assert image[1].values == tuple(2 * value for value in previous[1].values)
            assert budget.work_units > before


def test_edge_coverage_reuses_exact_partition_but_not_route_moments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticket, budget = _ledger()
    with ticket, budget.activate():
        work = PreparationWork(LIMIT, STORAGE, budget=budget)
        exact = ExactFormMoments(budget)
        cell, piece, plan = _projective_triangle(
            1, "untwisted", False, False, work, exact
        )
        first, _ = compatible._invariant_common_moments(
            (cell,), cell.basis, (piece,), cell.routes, work, plan
        )
        calls = 0
        original = compatible._invariant_edge_coverage

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(compatible, "_invariant_edge_coverage", counted)
        moved = FormCell(cell.basis, cell.routes + 13, cell.transform)
        repeated, defect = compatible._invariant_common_moments(
            (moved,), moved.basis, (piece,), moved.routes, work, plan
        )
        assert calls == 0
        assert defect == 0.0
        assert repeated is not first
        np.testing.assert_array_equal(repeated.value, first.value)
        np.testing.assert_array_equal(repeated.error, first.error)
        # The same cache must not certify a distinct, incomplete partition.
        partial = piece._replace(
            target_vertices=(
                (Fraction(0), Fraction(0)),
                (Fraction(1, 2), Fraction(0)),
                (Fraction(0), Fraction(1, 2)),
            ),
            target_area=Fraction(1, 8),
        )
        with pytest.raises(ValueError, match="material_entity_coverage"):
            compatible._invariant_common_moments(
                (moved,), moved.basis, (partial,), moved.routes, work, plan
            )
        assert calls > 0


def test_validation_fan_areas_reuse_exact_directed_edges() -> None:
    from phydrax.discretization._chart_geometry import area

    first = (Fraction(1, 7), Fraction(2, 11))
    second = (Fraction(3, 5), Fraction(1, 13))
    third = (Fraction(1, 17), Fraction(4, 5))
    triangle = (first, second, third)
    reversed_triangle = (first, third, second)
    ticket, budget = _ledger()
    with ticket, budget.activate():
        geometry = _SpherePieceGeometry()
        before = budget.work_units
        assert geometry.area(triangle, budget) == area(triangle)
        fresh_work = budget.work_units - before
        before = budget.work_units
        assert geometry.area(triangle, budget) == area(triangle)
        repeated_work = budget.work_units - before
        assert repeated_work == 3  # Only two sums and the final division.
        before = budget.work_units
        assert geometry.area(reversed_triangle, budget) == -area(triangle)
        reverse_work = budget.work_units - before
        assert reverse_work == 6  # Three cached-edge negations plus the sum.
        assert reverse_work < fresh_work
        assert repeated_work < fresh_work
        shifted = (Fraction(2, 17), Fraction(4, 5))
        changed = (first, second, shifted)
        assert geometry.area(changed, budget) == area(changed)
        assert geometry.area(changed, budget) != area(triangle)


def test_validation_jacobian_bounds_reuse_exact_denominator_values(
    sphere_chart: PreparedSphereChartDeformation,
) -> None:
    mapping = sphere_chart.pieces[0].projective_map
    ratio = mapping.orientation_ratio
    denominators = (Fraction(2, 3), Fraction(5, 7), Fraction(7, 9))
    ticket, budget = _ledger()
    with ticket, budget.activate():
        geometry = _SpherePieceGeometry()
        first = geometry.bounds(ratio, denominators, budget)
        assert first == mapping.triangle_bounds(denominators)
        before = budget.work_units
        assert geometry.bounds(
            ratio, (denominators[2], denominators[1], denominators[0]), budget
        ) == first
        assert budget.work_units == before
        changed = (Fraction(1, 2), denominators[1], denominators[2])
        assert geometry.bounds(ratio, changed, budget) == mapping.triangle_bounds(changed)
        assert budget.work_units - before == 3
        # A different map determinant must not alias a cached Jacobian value.
        doubled = geometry.bounds(2 * ratio, denominators, budget)
        assert doubled.denominator_lower == first.denominator_lower
        assert doubled.denominator_upper == first.denominator_upper
        assert doubled.jacobian_lower == 2 * first.jacobian_lower
        assert doubled.jacobian_upper == 2 * first.jacobian_upper
        with pytest.raises(ValueError, match="positive denominator and orientation"):
            geometry.bounds(
                ratio, (Fraction(0), denominators[1], denominators[2]), budget
            )


def test_completed_validation_releases_only_its_proof_scratch(
    sphere_chart: PreparedSphereChartDeformation,
) -> None:
    ticket, budget = _ledger()
    with ticket, budget.activate():
        before = budget.temporary_bytes_upper
        token = sphere_chart.require_bound(
            sphere_chart.source_mesh,
            sphere_chart.source_geometry,
            sphere_chart.target_mesh,
            sphere_chart.target_geometry,
            budget=budget,
        )
        assert budget.work_units > 0
        assert budget.temporary_bytes_upper == before
        token.require(
            sphere_chart,
            sphere_chart.source_mesh,
            sphere_chart.source_geometry,
            sphere_chart.target_mesh,
            sphere_chart.target_geometry,
            budget=budget,
        )
        before_work = budget.work_units
        sphere_chart.pieces.authenticate(budget=budget)
        authentication_work = budget.work_units - before_work
        repeated = sphere_chart.require_bound(
            sphere_chart.source_mesh,
            sphere_chart.source_geometry,
            sphere_chart.target_mesh,
            sphere_chart.target_geometry,
            budget=budget,
        )
        assert repeated.derived is token.derived
        assert budget.work_units == before_work + 2 * authentication_work


@pytest.mark.parametrize("degree", [1, 2])
def test_invariant_moments_refuse_incomplete_target_entities(degree: int) -> None:
    ticket, budget = _ledger()
    with ticket, budget.activate():
        work = PreparationWork(LIMIT, STORAGE, budget=budget)
        exact = ExactFormMoments(budget)
        cell, piece, plan = _projective_triangle(
            degree, "untwisted", False, False, work, exact
        )
        with pytest.raises(ValueError, match="material_entity_coverage"):
            compatible._invariant_common_moments(
                (cell,), cell.basis, (), cell.routes, work, plan
            )
        # A nonempty strict subset must refuse as well as an empty partition.
        half = piece._replace(
            target_vertices=(
                (piece.target_vertices[0][0] / 2, piece.target_vertices[0][1] / 2),
                (piece.target_vertices[1][0] / 2, piece.target_vertices[1][1] / 2),
                (piece.target_vertices[2][0] / 2, piece.target_vertices[2][1] / 2),
            ),
            target_area=piece.target_area / 4,
            charts=None,
        )
        with pytest.raises(ValueError, match="material_entity_coverage"):
            compatible._invariant_common_moments(
                (cell,), cell.basis, (half,), cell.routes, work, plan
            )


@pytest.mark.parametrize("quantity", ["denominator_lower", "jacobian_lower"])
def test_projective_condition_refuses_unresolved_whole_piece_orientation(
    sphere_chart: PreparedSphereChartDeformation, quantity: str
) -> None:
    piece = sphere_chart.pieces[0]
    changed = replace(
        piece,
        projective_bounds=replace(piece.projective_bounds, **{quantity: Fraction(0)}),
    )
    ticket, budget = _ledger()
    with (
        ticket,
        budget.activate(),
        pytest.raises(ValueError, match="positive whole-piece"),
    ):
        sphere._sphere_projective_condition(
            D.SphereChartPieceBank.from_pieces((changed,), budget=budget), budget=budget
        )


def test_sphere_compatible_transfer_refuses_changed_order(
    sphere_chart: PreparedSphereChartDeformation,
) -> None:
    source, _ = _endpoints(sphere_chart, 1, (("material", "untwisted"),))
    _, target = _endpoints(sphere_chart, 2, (("material", "untwisted"),))
    with pytest.raises(
        ValueError, match="cannot change degree, twist, proxy, order, or family"
    ):
        prepare_sphere_chart_compatible_transfer(
            source,
            target,
            sphere_chart,
            field_name="material",
            source_geometry=sphere_chart.source_geometry,
            target_geometry=sphere_chart.target_geometry,
        )
