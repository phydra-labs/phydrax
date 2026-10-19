from collections.abc import Callable, Iterator
from contextlib import ExitStack
from fractions import Fraction
from typing import NoReturn

import numpy as np
import pytest

from phydrax._bvh import PackedBVH
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    PeriodicCell,
    PeriodicMeshTopology,
)
from phydrax.discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    RestrictedCellGeometryElement,
)
from phydrax.discretization._cell_geometry_validity import (
    cell_geometry_id,
    certify_cell_geometry_validity,
)
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResource,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization._coordinate_enclosure import (
    coordinate_estimate_add,
    coordinate_estimate_bank,
    coordinate_estimate_multiply,
    coordinate_estimate_scale,
    coordinate_map_expression_estimates,
    coordinate_polynomial_estimate,
    coordinate_preparation_metadata_estimate,
    CoordinateEnclosureEstimate,
)
from phydrax.discretization.fem import FiniteElementSpec
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._embedding_cells import Cell
from phydrax.geometry._embedding_state import EmbeddingState, MeshCertificateLimits
from phydrax.geometry._global_embedding import certify_global_embedding
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
from phydrax.geometry._periodic_embedding import certify_periodic_mapped_embedding
from phydrax.geometry._source_fidelity import (
    certify_source_fidelity,
    MappedDomainBoundarySource,
)
from phydrax.geometry.brep._coordinate_request import coordinate_source_request
from phydrax.meshing._audit import audit_cell_mesh
from phydrax.meshing._certification import certify_meshing_acceptance
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing._curving import straight_geometry
from phydrax.meshing._failures import MeshingFailure
from phydrax.meshing._periodic import certify_periodic_embedding


def _budget(
    maximum_work_units: int, maximum_memory_bytes: int, /
) -> CoordinateEnclosureBudget:
    return CoordinateEnclosureBudget(
        HostBatchTicket(maximum_work_units, bytes_upper=maximum_memory_bytes),
        maximum_work_units=maximum_work_units,
        maximum_memory_bytes=maximum_memory_bytes,
    )


def _enter_budget(
    stack: ExitStack, budget: CoordinateEnclosureBudget, /
) -> CoordinateEnclosureBudget:
    stack.enter_context(budget.ticket)
    stack.enter_context(budget.activate())
    return budget


def _recipe_budget(
    values: tuple[CoordinateEnclosureEstimate, ...], /
) -> CoordinateEnclosureBudget:
    estimate = coordinate_estimate_bank(values)
    return _budget(estimate.work_upper, estimate.bytes_upper)


def _axis_recipe(dimension: int, /) -> CoordinateEnclosureEstimate:
    return coordinate_polynomial_estimate(
        terms=1,
        dimension=dimension,
        degree=1,
        numerator_bits=1,
        denominator_bits=1,
    )


def _axis_budget(
    dimension: int,
    /,
    *,
    additions: int = 0,
    scales: tuple[int, ...] = (),
) -> CoordinateEnclosureBudget:
    variable = _axis_recipe(dimension)
    return _recipe_budget(
        (
            *(variable for _ in range(dimension)),
            *(coordinate_estimate_add(variable, variable) for _ in range(additions)),
            *(coordinate_estimate_scale(variable, value) for value in scales),
        )
    )


def _coordinate_budget(
    element: FiniteElementSpec,
    banks: tuple[tuple[tuple[Fraction | int, ...], ...], ...],
    /,
) -> CoordinateEnclosureBudget:
    values = []
    for bank in banks:
        source = tuple(tuple(Fraction(value) for value in row) for row in bank)
        estimates = coordinate_map_expression_estimates(element, source)
        if estimates is None:
            raise ValueError(
                "element.coordinate_recipe must be supported; got an unsupported recipe."
            )
        values.extend(estimates)
        values.append(coordinate_preparation_metadata_estimate(element, source))
    return _recipe_budget(tuple(values))


def _periodic(
    mesh: CellMesh, representatives: np.ndarray, shifts: np.ndarray
) -> CellMesh:
    cell = PeriodicCell(np.eye(2, dtype=np.float64), periodic_axes=(True, False))
    topology = PeriodicMeshTopology(mesh, cell, representatives, shifts)
    return CellMesh(mesh.coordinates, mesh.blocks, periodic_topology=topology)


def _curved_hybrid() -> tuple[CellMesh, CellGeometrySpec, MappedReferenceDomain]:
    root_mesh = CellMesh(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        (
            CellBlock(
                "root", "quadrilateral", np.asarray(((0, 1, 2, 3),), dtype=np.int64)
            ),
        ),
    )
    root_mesh = _periodic(
        root_mesh, np.asarray((0, 0, 3, 3)), np.asarray(((0, 0), (1, 0), (1, 0), (0, 0)))
    )
    straight = straight_geometry(root_mesh, 2)
    elements, routes, coordinates = straight.resolve(root_mesh)
    if not isinstance(elements[0], FiniteElementSpec):
        raise TypeError(
            "straight.elements must contain a FiniteElementSpec; got a different coordinate-element kind."
        )
    values = np.asarray(coordinates).copy()
    values[:, 0] += 0.125 * values[:, 1] * (1.0 - values[:, 1])
    root_geometry = CellGeometrySpec({"root": elements[0]}, {"root": routes[0]}, values)
    reference_domain = PiecewiseLinearDomain(
        np.asarray(root_mesh.coordinates),
        np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int64),
        np.asarray(((0, -1), (0, -1), (0, -1), (0, -1)), dtype=np.int64),
        ("fluid",),
        source_id="periodic-curved-source",
    )
    domain = MappedReferenceDomain(
        reference_domain,
        root_mesh,
        root_geometry,
        np.asarray((0,), dtype=np.int64),
        source_id="periodic-curved-source",
        source_revision="authored-curved-strip",
    )
    points = np.asarray(
        ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 1.0), (1.0, 1.0))
    )
    blocks = (
        CellBlock(
            "lower",
            "triangle",
            np.asarray(((1, 2, 5),), dtype=np.int64),
            global_ids=np.asarray((1,), dtype=np.int64),
        ),
        CellBlock(
            "upper",
            "triangle",
            np.asarray(((1, 5, 4),), dtype=np.int64),
            global_ids=np.asarray((2,), dtype=np.int64),
        ),
        CellBlock(
            "quad",
            "quadrilateral",
            np.asarray(((0, 1, 4, 3),), dtype=np.int64),
            global_ids=np.asarray((0,), dtype=np.int64),
        ),
    )
    mesh = _periodic(
        CellMesh(points, blocks),
        np.asarray((0, 1, 0, 3, 4, 3)),
        np.asarray(((0, 0), (0, 0), (1, 0), (0, 0), (0, 0), (1, 0))),
    )
    charts = {
        "quad": (np.asarray(((0.5, 0.0), (0.0, 1.0))), np.asarray((0.0, 0.0))),
        "lower": (np.asarray(((0.5, 0.5), (0.0, 1.0))), np.asarray((0.5, 0.0))),
        "upper": (np.asarray(((0.5, 0.0), (1.0, 1.0))), np.asarray((0.5, 0.0))),
    }
    restriction = CellGeometryRestrictionSource(
        cell_geometry_id(root_geometry),
        root_mesh.topology_id,
        {block.name: np.asarray((0,), dtype=np.int64) for block in blocks},
        {block.name: np.asarray(((0, 1, 2, 3),), dtype=np.int64) for block in blocks},
    )
    geometry = CellGeometrySpec(
        {
            block.name: RestrictedCellGeometryElement(
                elements[0], block.cell_kind, *charts[block.name]
            )
            for block in blocks
        },
        {block.name: routes[0] for block in blocks},
        values,
        restriction_source=restriction,
    )
    return mesh, geometry, domain


def test_curved_periodic_hybrid_has_full_image_embedding_and_exact_source_cover() -> None:
    mesh, geometry, domain = _curved_hybrid()
    evidence = certify_periodic_embedding(mesh, geometry=geometry)
    certificate = evidence.global_embedding
    assert certificate is not None and certificate.status == "certified"
    assert certificate.periodic_image_count >= 3
    assert "periodic_mapped_trace_equivariance" in certificate.evaluated_checks
    # Curved seam, not an affine control: the physical x hull exceeds the carrier.
    assert np.max(np.asarray(geometry.coordinates)[:, 0]) > 1.0
    topology = mesh.periodic_topology
    assert topology is not None
    assert len(set(topology.entity_keys(1))) == len(topology.entity_keys(1))
    fidelity = certify_source_fidelity(
        mesh,
        geometry,
        MappedDomainBoundarySource(domain, np.asarray((0, 0, 0), dtype=np.int64)),
        tolerance=0.0,
    )
    assert fidelity.status == "certified"
    assert fidelity.mesh_to_source_upper == fidelity.source_to_mesh_upper == 0.0
    assert fidelity.domain_coverage is not None
    assert len(fidelity.domain_coverage.premise_certificate_ids) >= 3


@pytest.mark.parametrize("curved", (False, True))
def test_periodic_affine_validity_reuse_requires_full_source_affinity(
    curved: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import phydrax.geometry._periodic_embedding as owner

    mesh, curved_geometry, _ = _curved_hybrid()
    geometry = curved_geometry if curved else CellGeometrySpec.affine(mesh)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    original = owner.injective
    full_proof_kinds: list[str] = []
    original_bounds = owner._periodic_piece_bounds
    affine_bounds_calls = 0

    def piece_bounds(
        cell: Cell, affine: bool, *, budget: CoordinateEnclosureBudget
    ) -> tuple[np.ndarray, np.ndarray]:
        nonlocal affine_bounds_calls
        affine_bounds_calls += int(affine)
        result = original_bounds(cell, affine, budget=budget)
        with budget.temporary_scope():
            reference = owner.bounds(
                cell,
                np.zeros(cell.dimension, dtype=np.float64),
                np.eye(cell.dimension, dtype=np.float64),
                budget=budget,
            )
            np.testing.assert_array_equal(result[0], reference[0])
            np.testing.assert_array_equal(result[1], reference[1])
        return result

    def injective(
        cell: Cell,
        limits: MeshCertificateLimits,
        state: EmbeddingState,
        *,
        maximum_work: int | None = None,
        budget: CoordinateEnclosureBudget,
    ) -> str | None:
        full_proof_kinds.append(cell.kind)
        return original(cell, limits, state, maximum_work=maximum_work, budget=budget)

    monkeypatch.setattr(owner, "injective", injective)
    monkeypatch.setattr(owner, "_periodic_piece_bounds", piece_bounds)
    certificate = certify_global_embedding(mesh, geometry, validity)
    assert certificate.status == "certified"
    assert certificate.validity_certificate_id == validity.certificate_id
    assert ("periodic_affine_validity_reuse" in certificate.evaluated_checks) is (
        not curved
    )
    if curved:
        # These simplex pieces have an authored quadratic root. Their straight
        # corner carrier cannot discharge injectivity of the full source map.
        assert "triangle" in full_proof_kinds
        assert affine_bounds_calls == 0
    else:
        assert full_proof_kinds == ["quadrilateral"]
        assert affine_bounds_calls > 0


@pytest.mark.parametrize("curved", (False, True))
def test_inverse_contact_reuses_traces_only_without_source_facet_exceptions(
    curved: bool,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    import phydrax.geometry._periodic_embedding as owner

    mesh, curved_geometry, _ = _curved_hybrid()
    geometry = curved_geometry if curved else CellGeometrySpec.affine(mesh)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    original_traces = owner.cell_traces_equal
    trace_calls = 0

    def traces(first: Cell, second: Cell, *, budget: CoordinateEnclosureBudget) -> bool:
        nonlocal trace_calls
        trace_calls += 1
        return original_traces(first, second, budget=budget)

    monkeypatch.setattr(owner, "cell_traces_equal", traces)
    cached = certify_global_embedding(mesh, geometry, validity)
    cached_calls = trace_calls
    original_pair = owner._certify_periodic_pair

    def uncached_pair(
        request: owner._PeriodicRequest,
        prepared: owner._PeriodicCells,
        group: owner._PeriodicGroup,
        facets: owner._PeriodicFacets,
        sources: owner._PeriodicSourceFaces,
        images: owner._PeriodicImages,
        cache: owner._PeriodicContactCache,
        a: int,
        b: int,
        image: Cell,
        transform_: tuple[tuple[Fraction, ...], ...],
        exponents: tuple[int, ...],
    ) -> None:
        cache.decisions.clear()
        original_pair(
            request,
            prepared,
            group,
            facets,
            sources,
            images,
            cache,
            a,
            b,
            image,
            transform_,
            exponents,
        )

    monkeypatch.setattr(owner, "_certify_periodic_pair", uncached_pair)
    trace_calls = 0
    uncached = certify_global_embedding(mesh, geometry, validity)
    assert cached.status == uncached.status == "certified"
    assert cached.findings == uncached.findings
    assert cached.candidate_pair_count == uncached.candidate_pair_count
    assert cached.periodic_image_count == uncached.periodic_image_count
    request.node.user_properties.append(("cached_trace_calls", cached_calls))
    request.node.user_properties.append(("uncached_trace_calls", trace_calls))
    if curved:
        assert cached_calls == trace_calls
    else:
        assert cached_calls < trace_calls
        assert "periodic_inverse_pair_proof_reuse" in cached.evaluated_checks


@pytest.mark.parametrize("case", ("closed", "nonclosed", "curved"))
def test_full_source_affine_quotient_contacts_match_generic_exact_proof(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import phydrax.geometry._periodic_embedding as owner

    if case == "curved":
        mesh, geometry, _ = _curved_hybrid()
    else:
        carrier = CellMesh.from_triangles(
            np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
            np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        )
        if case == "closed":
            cell = PeriodicCell(np.eye(2, dtype=np.float64), periodic_axes=(True, True))
            topology = PeriodicMeshTopology(
                carrier,
                cell,
                np.zeros(4, dtype=np.int64),
                np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.int64),
            )
            mesh = CellMesh(
                carrier.coordinates, carrier.blocks, periodic_topology=topology
            )
        else:
            mesh = _periodic(
                carrier,
                np.asarray((0, 0, 3, 3), dtype=np.int64),
                np.asarray(((0, 0), (1, 0), (1, 0), (0, 0)), dtype=np.int64),
            )
        geometry = CellGeometrySpec.affine(mesh)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    original_record = owner._record_periodic_contact
    decisions: list[tuple[tuple[int, ...], int, int, str]] = []

    def record(
        request: owner._PeriodicRequest,
        prepared: owner._PeriodicCells,
        images: owner._PeriodicImages,
        first: Cell,
        image: Cell,
        transform_: tuple[tuple[Fraction, ...], ...],
        exponents: tuple[int, ...],
        a: int,
        b: int,
        decision: str,
    ) -> None:
        decisions.append((exponents, a, b, decision))
        original_record(
            request,
            prepared,
            images,
            first,
            image,
            transform_,
            exponents,
            a,
            b,
            decision,
        )

    monkeypatch.setattr(owner, "_record_periodic_contact", record)
    accelerated = certify_global_embedding(mesh, geometry, validity)
    accelerated_decisions = tuple(decisions)
    evidence = accelerated.periodic_affine
    if case == "closed":
        assert evidence is not None
        assert evidence.mesh_id == mesh.mesh_id
        assert evidence.geometry_id == validity.geometry_id
        assert evidence.validity_certificate_id == validity.certificate_id
        assert evidence.affinity_certificate_id
        assert evidence.facet_trace_certificate_id
        assert accelerated.boundary_facet_count == 0
    else:
        assert evidence is None

    def generic(
        request: owner._PeriodicRequest,
        prepared: owner._PeriodicCells,
        facets: owner._PeriodicFacets,
    ) -> None:
        # Retain the full-source/trace preparations; do not admit the optional
        # closed-quotient theorem, so every contact takes the generic route.
        assert request.state.periodic_affine is None

    monkeypatch.setattr(owner, "_prepare_periodic_affine_quotient", generic)
    decisions.clear()
    reference = certify_global_embedding(mesh, geometry, validity)
    assert reference.periodic_affine is None
    assert accelerated.status == reference.status == "certified"
    assert accelerated.findings == reference.findings
    assert accelerated.candidate_pair_count == reference.candidate_pair_count
    assert accelerated.periodic_image_count == reference.periodic_image_count
    assert accelerated_decisions == tuple(decisions)


def test_periodic_whole_source_box_transform_has_directed_exact_extrema() -> None:
    from itertools import product

    from phydrax.geometry._periodic_image_frame import transformed_source_box

    lower = np.asarray((-0.125, -2.0), dtype=np.float64)
    upper = np.asarray((1.0, 4.0), dtype=np.float64)
    matrix = (
        (Fraction(4, 5), Fraction(-3, 5), Fraction(1, 7)),
        (Fraction(3, 5), Fraction(4, 5), Fraction(-2, 7)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    ledger = _budget(1_000, 1_000_000)
    with ledger.ticket, ledger.activate():
        lo, hi = transformed_source_box(lower, upper, matrix, budget=ledger)
    corners = tuple(product(*zip(lower, upper, strict=True)))
    for axis, row in enumerate(matrix[:-1]):
        values = tuple(
            row[-1]
            + sum(
                (
                    weight * Fraction(value)
                    for weight, value in zip(row[:-1], point, strict=True)
                ),
                Fraction(0),
            )
            for point in corners
        )
        assert Fraction(float(lo[axis])) <= min(values)
        assert Fraction(float(hi[axis])) >= max(values)
        assert Fraction(float(np.nextafter(lo[axis], np.inf))) > min(values)
        assert Fraction(float(np.nextafter(hi[axis], -np.inf))) < max(values)


@pytest.mark.parametrize(
    "case,fault",
    (
        ("affine", False),
        ("curved", False),
        ("screw", False),
        ("overlap", False),
        ("curved", True),
    ),
)
def test_periodic_bvh_matches_complete_brute_force_decisions(
    case: str, fault: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import phydrax.geometry._periodic_embedding as owner

    if case == "overlap":
        mesh = CellMesh.from_triangles(
            np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
            np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        )
        mesh = _periodic(
            mesh,
            np.asarray((0, 0, 3, 3)),
            np.asarray(((0, 0), (1, 0), (1, 0), (0, 0))),
        )
        straight = straight_geometry(mesh, 4)
        coordinates = np.asarray(straight.coordinates).copy()
        x = coordinates[:, 0].copy()
        coordinates[:, 0] += 16.0 * x * x * (1.0 - x) * (1.0 - x)
        geometry = CellGeometrySpec(
            dict(zip(straight.block_names, straight.elements, strict=True)),
            dict(zip(straight.block_names, straight.geometry_dofs, strict=True)),
            coordinates,
        )
    elif case == "screw":
        mesh, geometry = _screw_mapped_source()
    else:
        mesh, geometry, _ = _curved_hybrid()
        if case == "affine":
            geometry = CellGeometrySpec.affine(mesh)
    original = owner.bvh_overlap_pair_blocks
    observed: list[tuple[tuple[int, int], ...]] = []
    expected: list[tuple[tuple[int, int], ...]] = []
    contacts: list[tuple[tuple[int, ...], int, int, int, tuple[str, ...]]] = []
    original_pair = owner._certify_periodic_pair

    def contact(
        request: owner._PeriodicRequest,
        prepared: owner._PeriodicCells,
        group: owner._PeriodicGroup,
        facets: owner._PeriodicFacets,
        sources: owner._PeriodicSourceFaces,
        images: owner._PeriodicImages,
        cache: owner._PeriodicContactCache,
        a: int,
        b: int,
        image: Cell,
        transform_: tuple[tuple[Fraction, ...], ...],
        exponents: tuple[int, ...],
    ) -> None:
        original_pair(
            request,
            prepared,
            group,
            facets,
            sources,
            images,
            cache,
            a,
            b,
            image,
            transform_,
            exponents,
        )
        contacts.append(
            (
                exponents,
                b,
                a,
                request.state.subdivision_pieces,
                tuple(finding.check for finding in request.state.findings),
            )
        )

    monkeypatch.setattr(owner, "_certify_periodic_pair", contact)

    def query(
        first: PackedBVH,
        second: PackedBVH,
        *,
        include_touching: bool,
        visit: Callable[[int], None],
        maximum_block_pairs: int,
        retain_owner: Callable[[object], None],
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        blocks = list(
            original(
                first,
                second,
                include_touching=include_touching,
                visit=visit,
                maximum_block_pairs=maximum_block_pairs,
                retain_owner=retain_owner,
            )
        )
        if fault and blocks:
            # A dropped touching/overlapping source box must fail the oracle.
            blocks[0] = (blocks[0][0][1:], blocks[0][1][1:])
        pairs = tuple(
            sorted(
                (int(a), int(b))
                for left, right in blocks
                for a, b in zip(left, right, strict=True)
            )
        )
        lo, hi = np.asarray(first.item_bbox_min), np.asarray(first.item_bbox_max)
        lower, upper = np.asarray(second.item_bbox_min), np.asarray(second.item_bbox_max)
        brute = tuple(
            (a, b)
            for a in range(lo.shape[0])
            for b in range(lower.shape[0])
            if not (np.any(hi[a] < lower[b]) or np.any(upper[b] < lo[a]))
        )
        observed.append(pairs)
        expected.append(brute)
        assert pairs == brute
        yield from blocks

    def prove() -> EmbeddingState:
        state = EmbeddingState([], [])
        limits = (
            MeshCertificateLimits(
                maximum_subdivision_depth=0, maximum_subdivision_pieces=1000
            )
            if case == "overlap"
            else MeshCertificateLimits()
        )
        ledger = _budget(limits.maximum_work_units, limits.maximum_scratch_bytes)
        ids = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
        with ledger.ticket, ledger.activate():
            certify_periodic_mapped_embedding(
                state, mesh, geometry, ids, limits, budget=ledger
            )
        return state

    monkeypatch.setattr(owner, "bvh_overlap_pair_blocks", query)
    if fault:
        with pytest.raises(AssertionError):
            prove()
        return
    accelerated = prove()
    if case == "overlap":
        assert "periodic_mapped_cell_overlap" in {
            finding.check for finding in accelerated.findings
        }
    accelerated_contacts = tuple(contacts)
    contacts.clear()
    assert observed and observed == expected

    def brute_query(
        first: PackedBVH,
        second: PackedBVH,
        *,
        include_touching: bool,
        visit: Callable[[int], None],
        maximum_block_pairs: int,
        retain_owner: Callable[[object], None],
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        # Exercise the predecessor's full b-by-a bank. Only the unchanged
        # rational transformed-map bound rejection may remove these pairs.
        selected = np.arange(first.item_bbox_min.shape[0], dtype=np.int64)
        for b in range(second.item_bbox_min.shape[0]):
            visit(selected.size)
            yield selected, np.full(selected.shape, b, dtype=np.int64)

    monkeypatch.setattr(owner, "bvh_overlap_pair_blocks", brute_query)
    brute = prove()
    assert tuple(contacts) == accelerated_contacts
    assert accelerated.candidate_pairs == brute.candidate_pairs > 0
    assert accelerated.subdivision_pieces == brute.subdivision_pieces
    assert accelerated.findings == brute.findings
    assert accelerated.checks == brute.checks


def test_periodic_image_memos_preserve_caller_storage_and_exact_memory_boundary() -> None:
    mesh, geometry, _ = _curved_hybrid()
    cell_ids = np.asarray((1, 2, 0), dtype=np.int64)
    limits = MeshCertificateLimits()

    def prove(memory: int) -> tuple[EmbeddingState, CoordinateEnclosureBudget]:
        ledger = _budget(limits.maximum_work_units, memory)
        state = EmbeddingState([], [])
        caller_basis = ({(0, 0): Fraction(7, 3)},)
        with ledger.ticket, ledger.activate():
            ledger.retain_basis(caller_basis)
            ledger.reserve(0, 512)
            affine = ledger.axis_affine_polynomial_cache
            bernstein = ledger.bernstein_coefficient_cache
            certify_periodic_mapped_embedding(
                state, mesh, geometry, cell_ids, limits, budget=ledger
            )
            assert ledger.axis_affine_polynomial_cache is affine
            assert ledger.bernstein_coefficient_cache is bernstein
            assert ledger.retained_objects[id(caller_basis)] is caller_basis
            assert ledger.temporary_bytes_upper >= 512
        return state, ledger

    reference, measured = prove(limits.maximum_scratch_bytes)
    assert not reference.findings
    assert measured.peak_bytes_upper < limits.maximum_scratch_bytes
    admitted, exact = prove(measured.peak_bytes_upper)
    assert not admitted.findings
    assert admitted.checks == reference.checks
    assert exact.work_units == measured.work_units
    assert exact.peak_bytes_upper == measured.peak_bytes_upper
    refused, short = prove(measured.peak_bytes_upper - 1)
    findings = tuple(
        finding for finding in refused.findings if finding.resource is not None
    )
    assert findings and short.peak_bytes_upper <= measured.peak_bytes_upper - 1
    assert any(
        dict(finding.requested)["limit"] == measured.peak_bytes_upper - 1
        and dict(finding.requested)["requested"] > measured.peak_bytes_upper - 1
        for finding in findings
    )


@pytest.mark.parametrize(
    "limits,check",
    (
        (MeshCertificateLimits(maximum_periodic_images=1), "periodic_image_budget"),
        (
            MeshCertificateLimits(maximum_candidate_pairs=1),
            "periodic_candidate_pair_budget",
        ),
        (
            MeshCertificateLimits(maximum_subdivision_pieces=1),
            "local_injectivity_piece_budget",
        ),
        (
            MeshCertificateLimits(maximum_work_units=1),
            "periodic_source_expression_resource_budget",
        ),
        (
            MeshCertificateLimits(maximum_scratch_bytes=1),
            "periodic_source_expression_resource_budget",
        ),
    ),
)
def test_mapped_quotient_resource_exhaustion_is_not_certification(
    limits: MeshCertificateLimits, check: str
) -> None:
    mesh, geometry, _ = _curved_hybrid()
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = certify_global_embedding(mesh, geometry, validity, limits=limits)
    assert certificate.status == "unresolved"
    if check != "periodic_source_expression_resource_budget":
        assert check in {value.check for value in certificate.findings}
        return
    refusals = tuple(
        value for value in certificate.findings if value.resource is not None
    )
    assert refusals
    for refusal in refusals:
        wanted, completed = dict(refusal.requested), dict(refusal.achieved)
        governing = (
            limits.maximum_work_units
            if refusal.resource == "coefficient_work"
            else limits.maximum_scratch_bytes
        )
        assert wanted["limit"] == governing == 1
        assert wanted["requested"] > wanted["limit"] >= completed["completed"] >= 0
        assert all(
            isinstance(value, int) for _, value in (*refusal.requested, *refusal.achieved)
        )
        assert completed["source_expression_work_units"] >= 0
        assert completed["source_expression_peak_bytes"] >= 0
    before = np.asarray(mesh.coordinates).copy()
    geometry_identity = cell_geometry_id(geometry)
    audit = audit_cell_mesh(mesh, geometry, prepared_validity=validity)
    report = certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=MeshCertificationSchedule("periodic"),
        limits=limits,
    )
    assert not report.passed
    proof = report.embedding
    assert proof is not None and proof.status == "unresolved"
    with pytest.raises(MeshingFailure) as failed:
        report.require_passed()
    for finding in proof.findings:
        if finding.resource is None:
            continue
        prefix = (
            f"certificate:{proof.certificate_id}:finding:{finding.finding_id}"
            f":{finding.check}:resource:{finding.resource}"
        )
        for key, value in finding.requested:
            assert dict(report.requested)[f"{prefix}:{key}"] == value
            assert dict(failed.value.evidence.requested)[f"{prefix}:{key}"] == value
        for key, value in finding.achieved:
            assert dict(report.achieved)[f"{prefix}:{key}"] == value
            assert dict(failed.value.evidence.achieved)[f"{prefix}:{key}"] == value
    np.testing.assert_array_equal(np.asarray(mesh.coordinates), before)
    assert cell_geometry_id(geometry) == geometry_identity


@pytest.mark.parametrize(
    "resource", ("coefficient_work", "polynomial_storage", "retained_basis")
)
def test_periodic_raw_refusal_preserves_each_actual_coordinate_resource_kind(
    resource: CoordinateEnclosureResource,
) -> None:
    ledger = _budget(
        0 if resource == "coefficient_work" else 2_000_000,
        0 if resource == "polynomial_storage" else 600,
    )
    with pytest.raises(CoordinateEnclosureResourceError) as refused:
        with ledger.ticket, ledger.activate():
            match resource:
                case "coefficient_work":
                    ledger.reserve(1)
                case "polynomial_storage":
                    ledger.reserve(0, 1)
                case "retained_basis":
                    ledger.retain_basis(({(0,): Fraction(1)},))
    raw = refused.value
    assert raw.resource == resource
    state = EmbeddingState([], [])
    state.add(
        "periodic_source_expression_resource_budget",
        "unresolved",
        "cell",
        (17001, 29003),
        resource_error=raw,
        expression_budget=ledger,
    )
    finding = state.findings[0]
    assert finding.resource == raw.resource
    assert finding.entity_ids == (17001, 29003)
    assert dict(finding.requested) == {"limit": raw.limit, "requested": raw.requested}
    measured = dict(finding.achieved)
    assert measured["completed"] == raw.completed
    assert measured["source_expression_work_units"] == ledger.work_units
    assert measured["source_expression_peak_bytes"] == ledger.peak_bytes_upper
    assert all(
        isinstance(value, int) for _, value in (*finding.requested, *finding.achieved)
    )


@pytest.mark.parametrize(
    "maximum_work,maximum_scratch", ((1, 81_920_000), (2_000_000, 1))
)
def test_periodic_source_proof_borrows_original_ledger_and_preserves_raw_refusal(
    maximum_work: int,
    maximum_scratch: int,
) -> None:
    mesh, geometry, _ = _curved_hybrid()
    ledger = _budget(maximum_work, maximum_scratch)
    state = EmbeddingState([], [])
    before = np.asarray(mesh.coordinates).copy()
    identity = cell_geometry_id(geometry)
    with ledger.ticket, ledger.activate():
        certify_periodic_mapped_embedding(
            state,
            mesh,
            geometry,
            np.asarray((1, 2, 0), dtype=np.int64),
            MeshCertificateLimits(),
            budget=ledger,
        )
    refusals = tuple(
        finding for finding in state.findings if finding.resource is not None
    )
    assert refusals
    for finding in refusals:
        requested, achieved = dict(finding.requested), dict(finding.achieved)
        original_limit = (
            maximum_work if finding.resource == "coefficient_work" else maximum_scratch
        )
        assert requested["limit"] == original_limit == 1
        assert requested["requested"] > original_limit >= achieved["completed"] >= 0
        assert achieved["source_expression_work_units"] == ledger.work_units
        assert achieved["source_expression_peak_bytes"] == ledger.peak_bytes_upper
        assert all(
            isinstance(value, int) for _, value in (*finding.requested, *finding.achieved)
        )
    np.testing.assert_array_equal(np.asarray(mesh.coordinates), before)
    assert cell_geometry_id(geometry) == identity


@pytest.mark.parametrize(
    "limits",
    (
        MeshCertificateLimits(maximum_work_units=1),
        MeshCertificateLimits(maximum_scratch_bytes=1),
    ),
)
def test_public_periodic_embedding_preserves_coordinate_refusal_fields(
    limits: MeshCertificateLimits,
) -> None:
    mesh, geometry, _ = _curved_hybrid()
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = certify_global_embedding(mesh, geometry, validity, limits=limits)
    with pytest.raises(MeshingFailure) as failed:
        certify_periodic_embedding(
            mesh, geometry=geometry, validity=validity, limits=limits
        )
    for finding in certificate.findings:
        if finding.resource is None:
            continue
        prefix = (
            f"certificate:{certificate.certificate_id}:finding:{finding.finding_id}"
            f":{finding.check}:resource:{finding.resource}"
        )
        for key, value in finding.requested:
            assert dict(failed.value.evidence.requested)[f"{prefix}:{key}"] == value
        for key, value in finding.achieved:
            assert dict(failed.value.evidence.achieved)[f"{prefix}:{key}"] == value


def test_positive_corner_control_does_not_admit_curved_cross_seam_overlap() -> None:
    with ExitStack() as stack:
        mesh = CellMesh.from_triangles(
            np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
            np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        )
        mesh = _periodic(
            mesh, np.asarray((0, 0, 3, 3)), np.asarray(((0, 0), (1, 0), (1, 0), (0, 0)))
        )
        assert certify_periodic_embedding(mesh).image_count == 3
        straight = straight_geometry(mesh, 4)
        coordinates = np.asarray(straight.coordinates).copy()
        x = coordinates[:, 0].copy()
        coordinates[:, 0] += 16.0 * x * x * (1.0 - x) * (1.0 - x)
        geometry = CellGeometrySpec(
            dict(zip(straight.block_names, straight.elements, strict=True)),
            dict(zip(straight.block_names, straight.geometry_dofs, strict=True)),
            coordinates,
        )
        # The derivative at both displayed x corners is 1, while the full map
        # crosses its translated image away from those corners.
        from phydrax.discretization._coordinate_enclosure import (
            coordinate_expressions,
            expression_derivative,
            expression_evaluate,
        )

        elements, routes, values = geometry.resolve(mesh)
        local = np.asarray(values)[np.asarray(routes[0])[0]]
        budget = stack.enter_context(
            coordinate_source_request((local, elements[0], mesh.topological_dimension))
        )
        expression = coordinate_expressions(
            elements[0], np.asarray(values)[np.asarray(routes[0])[0]], budget=budget
        )
        assert expression is not None
        assert (
            expression_evaluate(
                expression_derivative(expression[0], 0, budget=budget),
                (Fraction(0), Fraction(0)),
                budget=budget,
            )
            > 0
        )
        state = EmbeddingState([], [])
        certify_periodic_mapped_embedding(
            state,
            mesh,
            geometry,
            np.asarray(mesh.blocks[0].global_ids),
            MeshCertificateLimits(
                maximum_subdivision_depth=0, maximum_subdivision_pieces=1000
            ),
            budget=budget,
        )
        assert "periodic_mapped_cell_overlap" in {value.check for value in state.findings}


def _screw_mapped_source() -> tuple[CellMesh, CellGeometrySpec]:
    from phydrax.discretization._periodic_exact import PeriodicIsometryGroup

    generator = np.asarray(
        (
            (1.0, 0.0, 0.0, 1.0),
            (0.0, -1.0, 0.0, 1.0),
            (0.0, 0.0, -1.0, 1.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    group = PeriodicIsometryGroup(generator[None])
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
    mesh = CellMesh(
        points,
        (CellBlock("screw", "hexahedron", np.asarray(((0, 1, 2, 3, 4, 5, 6, 7),))),),
    )
    topology = PeriodicMeshTopology(
        mesh,
        group,
        np.asarray((0, 7, 4, 3, 4, 3, 0, 7)),
        np.asarray(((0,), (1,), (1,), (0,), (0,), (1,), (1,), (0,))),
    )
    mesh = CellMesh(points, mesh.blocks, periodic_topology=topology)
    straight = straight_geometry(mesh, 2)
    values = np.asarray(straight.coordinates).copy()
    # This interior bulge is invariant under (y,z) -> (1-y,1-z);
    # it preserves the authored screw seam without flattening the source.
    values[:, 0] += 0.125 * values[:, 1] * (1.0 - values[:, 1])
    geometry = CellGeometrySpec(
        dict(zip(straight.block_names, straight.elements, strict=True)),
        dict(zip(straight.block_names, straight.geometry_dofs, strict=True)),
        values,
    )
    return mesh, geometry


def test_original_screw_mapped_bank_keeps_negative_and_period_witnesses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.geometry._periodic_embedding as owner
    from phydrax.discretization._periodic_exact import (
        exact_periodic_element,
        PeriodicIsometryGroup,
    )

    mesh, geometry = _screw_mapped_source()
    source_bits = np.asarray(geometry.coordinates).copy()
    source_id = cell_geometry_id(geometry)
    original = exact_periodic_element
    observed: list[tuple[int, ...]] = []

    def record(
        matrices: tuple[tuple[tuple[Fraction, ...], ...], ...],
        orders: tuple[int, ...],
        exponents: tuple[int, ...],
        *,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[tuple[Fraction, ...], ...]:
        observed.append(exponents)
        return original(matrices, orders, exponents, budget=budget)

    monkeypatch.setattr(owner, "exact_periodic_element", record)
    state = EmbeddingState([], [])
    from tests._support.coordinate_references import coordinate_reference_budget

    with coordinate_reference_budget() as budget:
        count = owner.certify_periodic_mapped_embedding(
            state,
            mesh,
            geometry,
            np.asarray((0,)),
            MeshCertificateLimits(),
            budget=budget,
        )
    assert count >= 6
    assert (-1,) in observed and (2,) in observed
    assert not state.findings
    topology = mesh.periodic_topology
    if topology is None or not isinstance(topology.cell, PeriodicIsometryGroup):
        raise AssertionError(
            "mesh.periodic_topology.cell must be a PeriodicIsometryGroup; got missing topology or a different group kind."
        )
    group = topology.cell
    assert group.orders == (0,)
    assert group.linear_orders == (2,)
    assert group.translation_periods == (2,)
    np.testing.assert_array_equal(np.asarray(geometry.coordinates), source_bits)
    assert cell_geometry_id(geometry) == source_id


@pytest.mark.parametrize("cap,expected", ((1, 2), (5, 6)))
def test_original_screw_coset_cap_refuses_incomplete_bank(
    cap: int,
    expected: int,
) -> None:
    mesh, geometry = _screw_mapped_source()
    state = EmbeddingState([], [])
    from tests._support.coordinate_references import coordinate_reference_budget

    limits = MeshCertificateLimits(maximum_periodic_images=cap)
    with coordinate_reference_budget(limits) as budget:
        count = certify_periodic_mapped_embedding(
            state,
            mesh,
            geometry,
            np.asarray((0,)),
            limits,
            budget=budget,
        )
    assert count == expected
    assert [(finding.check, finding.status) for finding in state.findings] == [
        ("periodic_image_budget", "unresolved")
    ]


def test_overlap_image_bank_uses_whole_curved_source_support_and_original_screw_axis() -> (
    None
):
    with ExitStack() as stack:
        from phydrax.discretization._coordinate_enclosure import (
            add,
            axes,
            constant,
            multiply,
            scale,
        )
        from phydrax.discretization._periodic_exact import PeriodicIsometryGroup
        from phydrax.geometry._embedding_cells import bounds, Cell, vertices
        from phydrax.geometry._periodic_image_frame import (
            periodic_overlap_image_exponents,
            PeriodicEmbeddingResourceError,
            prepare_periodic_image_frame,
        )

        variable = _axis_recipe(3)
        one = coordinate_polynomial_estimate(
            terms=1, dimension=3, degree=0, numerator_bits=1, denominator_bits=1
        )
        x_recipe = coordinate_estimate_add(
            variable,
            coordinate_estimate_scale(
                coordinate_estimate_multiply(
                    variable,
                    coordinate_estimate_add(one, coordinate_estimate_scale(variable, -1)),
                ),
                5,
            ),
        )
        budget = _enter_budget(
            stack, _recipe_budget((variable, variable, variable, x_recipe))
        )
        u, v, w = axes(3, budget=budget)
        # Injective triangular map with an interior bulge in the translation
        # direction. Its corner box is not a whole-source support enclosure.
        x = add(
            u,
            scale(
                multiply(
                    v,
                    add(
                        constant(1, 3, budget=budget),
                        scale(v, -1, budget=budget),
                        budget=budget,
                    ),
                    budget=budget,
                ),
                5,
                budget=budget,
            ),
            budget=budget,
        )
        source = Cell(
            (x, v, w), "box", "hexahedron", 3, tuple(range(8)), vertices("hexahedron")
        )
        budget = stack.enter_context(
            coordinate_source_request((source.coordinates, source.dimension, 2))
        )
        box = bounds(source, np.zeros(3), np.eye(3), budget=budget)
        generator = np.diag((1.0, -1.0, -1.0, 1.0))
        generator[:3, 3] = 1.0
        group = PeriodicIsometryGroup(generator[None], tolerance=0.0)
        budget = stack.enter_context(
            coordinate_source_request(
                (group.generators, box, source.coordinates, 100), rows=100
            )
        )
        actions = periodic_overlap_image_exponents(group, (box,), 100, budget=budget)
        corner_actions = periodic_overlap_image_exponents(
            group,
            ((np.zeros(3), np.ones(3)),),
            100,
            budget=budget,
        )
        assert len(actions) > len(corner_actions)
        assert (-4,) in {tuple(row) for row in actions}
        assert (5,) in {tuple(row) for row in actions}
        with pytest.raises(PeriodicEmbeddingResourceError):
            periodic_overlap_image_exponents(
                group, (box,), len(actions) - 1, budget=budget
            )
        points = np.asarray(((0.25, 0.25, 0.25),))
        frame = prepare_periodic_image_frame(
            points,
            np.zeros(1, dtype=np.int64),
            np.zeros((1, 1), dtype=np.int64),
            group,
            100,
            source_support_boxes=(box,),
            budget=budget,
        )
        factor = Fraction(2) ** (3 * frame.exponent)
        for image, action in enumerate(actions):
            values = tuple(
                Fraction(int(value)) * factor
                for value in frame.image_points(image, budget=budget)[0]
            )
            n = int(action[0])
            yz = Fraction(1, 4) if n % 2 == 0 else Fraction(3, 4)
            assert values == (Fraction(1, 4) + n, yz, yz)
        np.testing.assert_array_equal(group.generators, generator[None])


def test_overlap_action_bank_pre_admits_original_source_storage_and_work() -> None:
    from phydrax.discretization._periodic_exact import PeriodicIsometryGroup
    from phydrax.geometry._periodic_image_frame import periodic_overlap_image_exponents

    generator = np.diag((1.0, -1.0, -1.0, 1.0))
    generator[:3, 3] = 1.0
    group = PeriodicIsometryGroup(generator[None], tolerance=0.0)
    support = ((np.zeros(3), np.ones(3)),)
    ledger = _budget(1_000_000, 1_000_000)
    with ledger.ticket, ledger.activate():
        actions = periodic_overlap_image_exponents(group, support, 100, budget=ledger)
    assert ledger.work_units > 0
    assert ledger.peak_bytes_upper >= actions.nbytes
    assert not actions.flags.writeable
    refused = _budget(1_000_000, 1)
    with (
        refused.ticket,
        refused.activate(),
        pytest.raises(CoordinateEnclosureResourceError) as caught,
    ):
        periodic_overlap_image_exponents(group, support, 100, budget=refused)
    assert caught.value.resource == "polynomial_storage"
    assert caught.value.limit == 1
    assert caught.value.requested > 1
    assert caught.value.completed == 0


def test_source_contact_diagnostics_preserve_large_exact_sci_integer_bits() -> None:
    with ExitStack() as stack:
        from phydrax.discretization._coordinate_enclosure import axes
        from phydrax.geometry._embedding_cells import Cell, vertices
        from phydrax.geometry._periodic_embedding import source_contact_quantities

        budget = _enter_budget(stack, _axis_budget(3))
        source = Cell(
            axes(3, budget=budget),
            "simplex",
            "tetrahedron",
            3,
            (0, 1, 2, 3),
            vertices("tetrahedron"),
        )
        numerator = (1 << 180) + 17
        denominator = 1 << 120
        matrix = (
            (Fraction(1), Fraction(0), Fraction(0), Fraction(-numerator, denominator)),
            (Fraction(0), Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
        )
        quantities = dict(
            source_contact_quantities(
                source,
                source,
                matrix,
                (-(1 << 80),),
                0,
                1,
                0,
                1,
            )
        )

        def original_integer(name: str) -> int:
            if name in quantities:
                return int(float(quantities[name]))
            return sum(
                int(float(quantities[f"{name}_limb_{index}"])) << (32 * index)
                for index in range(quantities[f"{name}_limb_count"])
            )

        assert original_integer("action_0_3_numerator_negative") == numerator
        assert original_integer("action_0_3_denominator") == denominator
        assert original_integer("image_exponent_0_negative") == 1 << 80
        assert quantities["shared_vertex_count"] == 4
        assert all(
            float(value).is_integer() and int(float(value)) == value
            for value in quantities.values()
        )
        from phydrax.geometry._embedding_state import MeshCertificateFinding

        finding = MeshCertificateFinding(
            "periodic_mapped_cell_overlap",
            "violated",
            "cell",
            (1, 2),
            observations=tuple(quantities.items()),
        )
        assert finding.resource is None
        assert not finding.requested and not finding.achieved
        assert dict(finding.observations) == quantities


@pytest.mark.parametrize("variant", ("strict", "parent", "wrong", "volume"))
def test_affine_closed_contact_requires_one_authenticated_parent_trace(
    variant: str,
) -> None:
    with ExitStack() as stack:
        from phydrax.discretization._coordinate_enclosure import add, axes, scale
        from phydrax.geometry._affine_mapped_contact import affine_simplex_contact
        from phydrax.geometry._embedding_cells import Cell, vertices

        budget = _enter_budget(stack, _axis_budget(3, additions=1, scales=(-1,)))
        u, v, w = axes(3, budget=budget)
        first = Cell(
            (add(u, v, budget=budget), v, w),
            "simplex",
            "tetrahedron",
            3,
            (10, 11, 12, 13),
            vertices("tetrahedron"),
        )
        # Opposite tetrahedron triangulates the same parent square with the other
        # diagonal. Their closed triangular contact is larger than shared edge.
        second = Cell(
            (u, v, scale(w, -1, budget=budget)),
            "simplex",
            "tetrahedron",
            3,
            (10, 11, 14, 15),
            vertices("tetrahedron"),
        )
        budget = stack.enter_context(
            coordinate_source_request(
                (first.coordinates, second.coordinates, first.dimension, 1000),
                rows=1000,
            )
        )
        if variant == "strict":
            contact = affine_simplex_contact(first, second, 1000, budget=budget)
            expected = "overlap"
        elif variant == "parent":
            contact = affine_simplex_contact(
                first,
                second,
                1000,
                authenticated_first_traces=((10, 11, 12),),
                budget=budget,
            )
            expected = "separated"
        elif variant == "wrong":
            contact = affine_simplex_contact(
                first,
                second,
                1000,
                authenticated_first_traces=((10, 11, 13),),
                budget=budget,
            )
            expected = "overlap"
            with pytest.raises(ValueError, match="authenticated_first_traces"):
                affine_simplex_contact(
                    first,
                    second,
                    1000,
                    authenticated_first_traces=((10, 11, 12, 13),),
                    budget=budget,
                )
        else:
            interior = Cell(
                (u, v, w),
                "simplex",
                "tetrahedron",
                3,
                (10, 11, 14, 13),
                vertices("tetrahedron"),
            )
            # One allowed face per candidate is insufficient: the closed convex
            # intersection must lie wholly in ONE proper source trace.
            all_faces = ((10, 11, 12), (10, 11, 13), (10, 12, 13), (11, 12, 13))
            contact = affine_simplex_contact(
                first,
                interior,
                1000,
                authenticated_first_traces=all_faces,
                budget=budget,
            )
            expected = "overlap"
        if contact is None:
            raise AssertionError(
                "affine_simplex_contact.result must contain a classified contact; got None."
            )
        assert contact.status == expected


@pytest.mark.parametrize("trace_dimension", (0, 1))
@pytest.mark.parametrize("authorized", (False, True))
def test_affine_closed_contact_authenticated_lower_source_trace(
    trace_dimension: int,
    authorized: bool,
) -> None:
    with ExitStack() as stack:
        from phydrax.discretization._coordinate_enclosure import axes, scale
        from phydrax.geometry._affine_mapped_contact import affine_simplex_contact
        from phydrax.geometry._embedding_cells import Cell, vertices

        budget = _enter_budget(
            stack, _axis_budget(3, scales=(-1, -1, -1 if trace_dimension == 0 else 1))
        )
        u, v, w = axes(3, budget=budget)
        first = Cell(
            (u, v, w),
            "simplex",
            "tetrahedron",
            3,
            (10, 11, 12, 13),
            vertices("tetrahedron"),
        )
        second = Cell(
            (
                scale(u, -1, budget=budget),
                scale(v, -1, budget=budget),
                scale(w, -1 if trace_dimension == 0 else 1, budget=budget),
            ),
            "simplex",
            "tetrahedron",
            3,
            (20, 21, 22, 23),
            vertices("tetrahedron"),
        )
        # The incident source trace is authored separately from auxiliary IDs.
        # A wrong source vertex/edge must not authorize the intersection.
        trace = (
            ((10,) if trace_dimension == 0 else (10, 13))
            if authorized
            else ((13,) if trace_dimension == 0 else (10, 11))
        )
        budget = stack.enter_context(
            coordinate_source_request(
                (first.coordinates, second.coordinates, first.dimension, 1000),
                rows=1000,
            )
        )
        contact = affine_simplex_contact(
            first,
            second,
            1000,
            authenticated_first_traces=(trace,),
            budget=budget,
        )
        if contact is None:
            raise AssertionError(
                "affine_simplex_contact.result must contain a classified contact; got None."
            )
        assert contact.status == ("separated" if authorized else "overlap")


@pytest.mark.parametrize(
    "variant", ("actual-subface", "concave-contained", "concave-crossing", "outside")
)
def test_closed_original_source_polygon_complete_containment(variant: str) -> None:
    with ExitStack() as stack:
        from phydrax.geometry._periodic_embedding import closed_source_polygon_contains

        source = tuple(
            tuple(Fraction(value) for value in point)
            for point in (
                (0, 0),
                (3, 0),
                (3, 3),
                (2, 3),
                (2, 1),
                (1, 1),
                (1, 3),
                (0, 3),
            )
        )
        if variant == "actual-subface":
            source = (
                (Fraction(0), Fraction(0)),
                (Fraction(1), Fraction(0)),
                (Fraction(1), Fraction(1)),
                (Fraction(0), Fraction(1)),
            )
            # The source translation intersection, projected to (x,z).
            subject = (
                (Fraction(1, 2), Fraction(1)),
                (Fraction(0), Fraction(1)),
                (Fraction(1, 2), Fraction(1, 2)),
            )
        elif variant == "concave-contained":
            subject = (
                (Fraction(0), Fraction(0)),
                (Fraction(3), Fraction(0)),
                (Fraction(1), Fraction(1)),
            )
        elif variant == "concave-crossing":
            # Every corner is inside; the connecting edge leaves the source.
            subject = (
                (Fraction(1, 2), Fraction(2)),
                (Fraction(5, 2), Fraction(2)),
                (Fraction(3, 2), Fraction(1, 2)),
            )
        else:
            subject = (
                (Fraction(-1), Fraction(0)),
                (Fraction(1), Fraction(0)),
                (Fraction(1), Fraction(1)),
            )
        rows = len(source) * len(subject) * (2 * len(source) + 3)
        budget = stack.enter_context(
            coordinate_source_request((source, subject, rows), rows=rows)
        )
        assert closed_source_polygon_contains(source, subject, budget=budget) is (
            variant in ("actual-subface", "concave-contained")
        )


@pytest.mark.parametrize("variant", ("collinear-prefix", "nonplanar"))
def test_original_source_polygon_plane_uses_complete_original_loop(variant: str) -> None:
    with ExitStack() as stack:
        from phydrax.geometry._periodic_embedding import source_polygon_plane

        points = tuple(
            tuple(Fraction(value) for value in point)
            for point in (
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
                (2, 0, 1),
                (0, 0 if variant == "collinear-prefix" else 1, 1),
            )
        )
        rows = len(points) * (len(points) - 1)
        budget = stack.enter_context(coordinate_source_request((points, rows), rows=rows))
        plane = source_polygon_plane(points, budget=budget)
        if variant == "collinear-prefix":
            assert plane is not None and plane[0] == (
                Fraction(0),
                Fraction(1),
                Fraction(0),
                Fraction(0),
            )
        else:
            assert plane is None


@pytest.mark.parametrize("original_trace", (False, True))
def test_actual_translation_triangle_uses_original_facet_not_fragment_corners(
    original_trace: bool,
) -> None:
    with ExitStack() as stack:
        from phydrax.discretization._cell_geometry import coordinate_lagrange_element
        from phydrax.discretization._coordinate_enclosure import coordinate_expressions
        from phydrax.geometry._affine_mapped_contact import affine_simplex_contact
        from phydrax.geometry._embedding_cells import Cell, vertices

        # Coordinate banks from the source translation failure.
        first_points = (
            (0, 0, 1),
            (Fraction(1, 2), 0, 1),
            (Fraction(1, 2), 0, 0),
            (Fraction(1, 2), Fraction(1, 2), 0),
        )
        second_points = (
            (0, -1, 1),
            (0, 0, 1),
            (Fraction(1, 2), 0, 1),
            (Fraction(1, 2), 0, Fraction(1, 2)),
        )
        element = coordinate_lagrange_element("tetrahedron", 1)
        budget = _enter_budget(
            stack, _coordinate_budget(element, (first_points, second_points))
        )
        coordinates = tuple(
            coordinate_expressions(
                element,
                tuple(tuple(Fraction(value) for value in point) for point in bank),
                budget=budget,
            )
            for bank in (first_points, second_points)
        )
        assert coordinates[0] is not None and coordinates[1] is not None
        first = Cell(
            coordinates[0],
            "simplex",
            "tetrahedron",
            3,
            (0, 10, 6, 3),
            vertices("tetrahedron"),
        )
        second = Cell(
            coordinates[1],
            "simplex",
            "tetrahedron",
            3,
            (32, 0, 10, 16),
            vertices("tetrahedron"),
        )
        # Published fragments offer an edge and an isolated vertex, whose union
        # cannot authorize a triangle. One source y0 facet trace does.
        traces = ((0, 10, 6),) if original_trace else ((0, 10), (6,))
        budget = stack.enter_context(
            coordinate_source_request(
                (first.coordinates, second.coordinates, first.dimension, 1000),
                rows=1000,
            )
        )
        contact = affine_simplex_contact(
            first,
            second,
            1000,
            authenticated_first_traces=traces,
            budget=budget,
        )
        if contact is None:
            raise AssertionError(
                "affine_simplex_contact.result must contain a classified contact; got None."
            )
        assert contact.status == ("separated" if original_trace else "overlap")


@pytest.mark.parametrize(
    ("apex_height", "shared_topology", "maximum_work", "expected"),
    (
        pytest.param(2, True, 1_000, "separated", id="shared-prism-tetra-face"),
        pytest.param(2, False, 1_000, "overlap", id="unowned-coincident-face"),
        pytest.param(Fraction(1, 2), True, 1_000, "overlap", id="volume-overlap"),
        pytest.param(2, True, 0, "mapped_intersection_piece_budget", id="work-refusal"),
    ),
)
def test_layer_fiber_affine_reference_contact_is_exact_and_fail_closed(
    apex_height: Fraction | int,
    shared_topology: bool,
    maximum_work: int,
    expected: str,
) -> None:
    with ExitStack() as stack:
        from phydrax.discretization._cell_geometry import coordinate_lagrange_element
        from phydrax.discretization._coordinate_enclosure import coordinate_expressions
        from phydrax.geometry._embedding_cells import Cell, vertices
        from phydrax.geometry._periodic_embedding import affine_reference_contact

        prism_points = tuple(
            tuple(Fraction(value) for value in point)
            for point in (
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 0, 1),
                (0, 1, 1),
            )
        )
        tetrahedron_points = (
            prism_points[3],
            prism_points[4],
            prism_points[5],
            (Fraction(0), Fraction(0), Fraction(apex_height)),
        )
        prism_element = coordinate_lagrange_element("prism", 1)
        tetrahedron_element = coordinate_lagrange_element("tetrahedron", 1)
        budget = _enter_budget(stack, _coordinate_budget(prism_element, (prism_points,)))
        prism_coordinates = coordinate_expressions(
            prism_element, prism_points, budget=budget
        )
        budget = _enter_budget(
            stack, _coordinate_budget(tetrahedron_element, (tetrahedron_points,))
        )
        tetrahedron_coordinates = coordinate_expressions(
            tetrahedron_element, tetrahedron_points, budget=budget
        )
        if prism_coordinates is None or tetrahedron_coordinates is None:
            raise AssertionError(
                "coordinate_expressions.result must contain both P1 sources; got a missing prism or tetrahedron source."
            )
        prism = Cell(
            prism_coordinates,
            "prism",
            "prism",
            3,
            (0, 1, 2, 3, 4, 5),
            vertices("prism"),
        )
        tetrahedron = Cell(
            tetrahedron_coordinates,
            "simplex",
            "tetrahedron",
            3,
            (3, 4, 5, 6) if shared_topology else (7, 8, 9, 6),
            vertices("tetrahedron"),
        )
        budget = stack.enter_context(
            coordinate_source_request(
                (
                    prism.coordinates,
                    tetrahedron.coordinates,
                    prism.dimension,
                    maximum_work,
                ),
                rows=maximum_work,
            )
        )
        decision = affine_reference_contact(
            prism, tetrahedron, maximum_work, budget=budget
        )
        if decision is None:
            raise AssertionError(
                "affine_reference_contact.result must contain a classification; got None."
            )
        expected_status = (
            "unresolved" if expected == "mapped_intersection_piece_budget" else expected
        )
        assert decision.status == expected_status
        assert decision.check == (
            "mapped_intersection_piece_budget"
            if expected_status == "unresolved"
            else None
        )


def test_periodic_wrapper_refusal_measures_only_its_source_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh, _, _ = _curved_hybrid()
    ledger = _budget(1_000, 1_000_000)
    coordinates = np.asarray(mesh.coordinates).copy()

    def refuse_source(candidate: CellMesh, /) -> NoReturn:
        assert candidate is mesh
        ledger.reserve(3, 96)
        raise CoordinateEnclosureResourceError("coefficient_work", 10, 11, 3)

    # Refuse source preparation after predecessor work and storage are settled.
    monkeypatch.setattr(CellGeometrySpec, "affine", staticmethod(refuse_source))
    with ledger.ticket, ledger.activate():
        ledger.reserve(7, 256)

        with pytest.raises(MeshingFailure) as failed:
            certify_periodic_embedding(
                mesh,
                limits=MeshCertificateLimits(
                    maximum_work_units=1_000, maximum_scratch_bytes=1_000_000
                ),
                budget=ledger,
            )
    prefix = "periodic:coordinate_source:resource:coefficient_work"
    achieved = dict(failed.value.evidence.achieved)
    assert achieved[f"{prefix}:completed"] == 3
    assert achieved[f"{prefix}:source_expression_work_units"] == 3
    assert achieved[f"{prefix}:source_expression_peak_bytes"] == 96
    assert achieved[f"{prefix}:source_expression_native_charged_work_units"] == 0
    assert ledger.work_units == 10
    np.testing.assert_array_equal(mesh.coordinates, coordinates)
