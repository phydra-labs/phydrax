from fractions import Fraction

import numpy as np
import pytest

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
from phydrax.discretization._coordinate_enclosure import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResource,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization.fem import FiniteElementSpec
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
from phydrax.geometry._mesh_certificates import (
    _EmbeddingState,
    certify_global_embedding,
    certify_source_fidelity,
    MappedDomainBoundarySource,
    MeshCertificateLimits,
    PiecewiseLinearDomain,
)
from phydrax.geometry._periodic_embedding import certify_periodic_mapped_embedding
from phydrax.meshing._audit import audit_cell_mesh
from phydrax.meshing._certification import (
    certify_meshing_acceptance,
    MeshCertificationSchedule,
)
from phydrax.meshing._contracts import MeshingFailure
from phydrax.meshing._curving import _straight_geometry
from phydrax.meshing._periodic import certify_periodic_embedding


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
    straight = _straight_geometry(root_mesh, 2)
    elements, routes, coordinates = straight.resolve(root_mesh)
    if not isinstance(elements[0], FiniteElementSpec):
        raise TypeError(
            "The authored source must retain its canonical coordinate element."
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
    ledger = CoordinateEnclosureBudget(
        0 if resource == "coefficient_work" else 2_000_000,
        0 if resource == "polynomial_storage" else 600,
    )
    with pytest.raises(CoordinateEnclosureResourceError) as refused:
        with ledger.activate():
            match resource:
                case "coefficient_work":
                    ledger.reserve(1)
                case "polynomial_storage":
                    ledger.reserve(0, 1)
                case "retained_basis":
                    ledger.retain_basis(({(0,): Fraction(1)},))
    raw = refused.value
    assert raw.resource == resource
    state = _EmbeddingState([], [])
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
    ledger = CoordinateEnclosureBudget(maximum_work, maximum_scratch)
    state = _EmbeddingState([], [])
    before = np.asarray(mesh.coordinates).copy()
    identity = cell_geometry_id(geometry)
    with ledger.activate():
        certify_periodic_mapped_embedding(
            state,
            mesh,
            geometry,
            np.asarray((1, 2, 0), dtype=np.int64),
            MeshCertificateLimits(),
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
    mesh = CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
    )
    mesh = _periodic(
        mesh, np.asarray((0, 0, 3, 3)), np.asarray(((0, 0), (1, 0), (1, 0), (0, 0)))
    )
    assert certify_periodic_embedding(mesh).image_count == 3
    straight = _straight_geometry(mesh, 4)
    coordinates = np.asarray(straight.coordinates).copy()
    x = coordinates[:, 0].copy()
    coordinates[:, 0] += 16.0 * x * x * (1.0 - x) * (1.0 - x)
    geometry = CellGeometrySpec(
        dict(zip(straight.block_names, straight.elements, strict=True)),
        dict(zip(straight.block_names, straight.geometry_dofs, strict=True)),
        coordinates,
    )
    # Exact derivative at both displayed x corners is 1, while the full map
    # crosses its translated image away from those corners.
    from phydrax.discretization._coordinate_enclosure import (
        coordinate_expressions,
        expression_derivative,
        expression_evaluate,
    )

    elements, routes, values = geometry.resolve(mesh)
    expression = coordinate_expressions(
        elements[0], np.asarray(values)[np.asarray(routes[0])[0]]
    )
    assert expression is not None
    assert (
        expression_evaluate(
            expression_derivative(expression[0], 0), (Fraction(0), Fraction(0))
        )
        > 0
    )
    state = _EmbeddingState([], [])
    certify_periodic_mapped_embedding(
        state,
        mesh,
        geometry,
        np.asarray(mesh.blocks[0].global_ids),
        MeshCertificateLimits(
            maximum_subdivision_depth=0, maximum_subdivision_pieces=1000
        ),
    )
    assert "periodic_mapped_cell_overlap" in {value.check for value in state.findings}


def _screw_mapped_source() -> tuple[CellMesh, CellGeometrySpec]:
    from phydrax.discretization._periodic_topology import PeriodicIsometryGroup

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
    straight = _straight_geometry(mesh, 2)
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
    from phydrax.discretization._periodic_topology import PeriodicIsometryGroup

    mesh, geometry = _screw_mapped_source()
    source_bits = np.asarray(geometry.coordinates).copy()
    source_id = cell_geometry_id(geometry)
    original = owner._exact_periodic_element
    observed: list[tuple[int, ...]] = []

    def record(
        matrices: tuple[tuple[tuple[Fraction, ...], ...], ...],
        orders: tuple[int, ...],
        exponents: tuple[int, ...],
    ) -> tuple[tuple[Fraction, ...], ...]:
        observed.append(exponents)
        return original(matrices, orders, exponents)

    monkeypatch.setattr(owner, "_exact_periodic_element", record)
    state = _EmbeddingState([], [])
    count = owner.certify_periodic_mapped_embedding(
        state, mesh, geometry, np.asarray((0,)), MeshCertificateLimits()
    )
    assert count >= 6
    assert (-1,) in observed and (2,) in observed
    assert not state.findings
    topology = mesh.periodic_topology
    if topology is None or not isinstance(topology.cell, PeriodicIsometryGroup):
        raise AssertionError("The screw source must retain its periodic isometry group.")
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
    state = _EmbeddingState([], [])
    count = certify_periodic_mapped_embedding(
        state,
        mesh,
        geometry,
        np.asarray((0,)),
        MeshCertificateLimits(maximum_periodic_images=cap),
    )
    assert count == expected
    assert [(finding.check, finding.status) for finding in state.findings] == [
        ("periodic_image_budget", "unresolved")
    ]


def test_overlap_image_bank_uses_whole_curved_source_support_and_original_screw_axis() -> (
    None
):
    from phydrax.discretization._coordinate_enclosure import (
        add,
        axes,
        constant,
        multiply,
        scale,
    )
    from phydrax.discretization._periodic_topology import PeriodicIsometryGroup
    from phydrax.geometry._mapped_embedding import _bounds, _Cell, _vertices
    from phydrax.meshing._periodic import (
        _periodic_overlap_image_exponents,
        _PeriodicEmbeddingResourceError,
        _prepare_periodic_image_frame,
    )

    u, v, w = axes(3)
    # Injective triangular map with a genuine interior bulge in the translation
    # direction. Its corner box is not a whole-source support enclosure.
    x = add(u, scale(multiply(v, add(constant(1, 3), scale(v, -1))), 5))
    source = _Cell(
        (x, v, w), "box", "hexahedron", 3, tuple(range(8)), _vertices("hexahedron")
    )
    box = _bounds(source, np.zeros(3), np.eye(3))
    generator = np.diag((1.0, -1.0, -1.0, 1.0))
    generator[:3, 3] = 1.0
    group = PeriodicIsometryGroup(generator[None], tolerance=0.0)
    actions = _periodic_overlap_image_exponents(group, (box,), 100)
    corner_actions = _periodic_overlap_image_exponents(
        group,
        ((np.zeros(3), np.ones(3)),),
        100,
    )
    assert len(actions) > len(corner_actions)
    assert (-4,) in {tuple(row) for row in actions}
    assert (5,) in {tuple(row) for row in actions}
    with pytest.raises(_PeriodicEmbeddingResourceError):
        _periodic_overlap_image_exponents(group, (box,), len(actions) - 1)
    points = np.asarray(((0.25, 0.25, 0.25),))
    frame = _prepare_periodic_image_frame(
        points,
        np.zeros(1, dtype=np.int64),
        np.zeros((1, 1), dtype=np.int64),
        group,
        100,
        source_support_boxes=(box,),
    )
    factor = Fraction(2) ** (3 * frame.exponent)
    for image, action in enumerate(actions):
        values = tuple(
            Fraction(int(value)) * factor for value in frame.image_points(image)[0]
        )
        n = int(action[0])
        yz = Fraction(1, 4) if n % 2 == 0 else Fraction(3, 4)
        assert values == (Fraction(1, 4) + n, yz, yz)
    np.testing.assert_array_equal(group.generators, generator[None])


def test_overlap_action_bank_pre_admits_original_source_storage_and_work() -> None:
    from phydrax.discretization._periodic_topology import PeriodicIsometryGroup
    from phydrax.meshing._periodic import _periodic_overlap_image_exponents

    generator = np.diag((1.0, -1.0, -1.0, 1.0))
    generator[:3, 3] = 1.0
    group = PeriodicIsometryGroup(generator[None], tolerance=0.0)
    support = ((np.zeros(3), np.ones(3)),)
    ledger = CoordinateEnclosureBudget(1_000_000, 1_000_000)
    with ledger.activate():
        actions = _periodic_overlap_image_exponents(group, support, 100)
    assert ledger.work_units > 0
    assert ledger.peak_bytes_upper >= actions.nbytes
    assert not actions.flags.writeable
    refused = CoordinateEnclosureBudget(1_000_000, 1)
    with refused.activate(), pytest.raises(CoordinateEnclosureResourceError) as caught:
        _periodic_overlap_image_exponents(group, support, 100)
    assert caught.value.resource == "polynomial_storage"
    assert caught.value.limit == 1
    assert caught.value.requested > 1
    assert caught.value.completed == 0


def test_source_contact_diagnostics_preserve_large_exact_sci_integer_bits() -> None:
    from phydrax.discretization._coordinate_enclosure import axes
    from phydrax.geometry._mapped_embedding import _Cell, _vertices
    from phydrax.geometry._periodic_embedding import _source_contact_quantities

    source = _Cell(
        axes(3), "simplex", "tetrahedron", 3, (0, 1, 2, 3), _vertices("tetrahedron")
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
        _source_contact_quantities(
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
    from phydrax.geometry._mesh_certificates import MeshCertificateFinding

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
    from phydrax.discretization._coordinate_enclosure import add, axes, scale
    from phydrax.geometry._affine_mapped_contact import affine_simplex_contact
    from phydrax.geometry._mapped_embedding import _Cell, _vertices

    u, v, w = axes(3)
    first = _Cell(
        (add(u, v), v, w),
        "simplex",
        "tetrahedron",
        3,
        (10, 11, 12, 13),
        _vertices("tetrahedron"),
    )
    # Opposite tetrahedron triangulates the same parent square with the other
    # diagonal. Their closed triangular contact is larger than shared edge.
    second = _Cell(
        (u, v, scale(w, -1)),
        "simplex",
        "tetrahedron",
        3,
        (10, 11, 14, 15),
        _vertices("tetrahedron"),
    )
    if variant == "strict":
        contact = affine_simplex_contact(first, second, 1000)
        expected = "overlap"
    elif variant == "parent":
        contact = affine_simplex_contact(
            first,
            second,
            1000,
            authoritative_first_traces=((10, 11, 12),),
        )
        expected = "separated"
    elif variant == "wrong":
        contact = affine_simplex_contact(
            first,
            second,
            1000,
            authoritative_first_traces=((10, 11, 13),),
        )
        expected = "overlap"
        with pytest.raises(ValueError, match="proper first-simplex trace"):
            affine_simplex_contact(
                first,
                second,
                1000,
                authoritative_first_traces=((10, 11, 12, 13),),
            )
    else:
        interior = _Cell(
            (u, v, w),
            "simplex",
            "tetrahedron",
            3,
            (10, 11, 14, 13),
            _vertices("tetrahedron"),
        )
        # One allowed face per candidate is insufficient: the closed convex
        # intersection must lie wholly in ONE proper source trace.
        all_faces = ((10, 11, 12), (10, 11, 13), (10, 12, 13), (11, 12, 13))
        contact = affine_simplex_contact(
            first,
            interior,
            1000,
            authoritative_first_traces=all_faces,
        )
        expected = "overlap"
    if contact is None:
        raise AssertionError("The affine fixture requires a classified contact.")
    assert contact[0] == expected


@pytest.mark.parametrize("trace_dimension", (0, 1))
@pytest.mark.parametrize("authorized", (False, True))
def test_affine_closed_contact_authenticated_lower_source_trace(
    trace_dimension: int,
    authorized: bool,
) -> None:
    from phydrax.discretization._coordinate_enclosure import axes, scale
    from phydrax.geometry._affine_mapped_contact import affine_simplex_contact
    from phydrax.geometry._mapped_embedding import _Cell, _vertices

    u, v, w = axes(3)
    first = _Cell(
        (u, v, w),
        "simplex",
        "tetrahedron",
        3,
        (10, 11, 12, 13),
        _vertices("tetrahedron"),
    )
    second = _Cell(
        (scale(u, -1), scale(v, -1), scale(w, -1 if trace_dimension == 0 else 1)),
        "simplex",
        "tetrahedron",
        3,
        (20, 21, 22, 23),
        _vertices("tetrahedron"),
    )
    # The incident source trace is authored separately from auxiliary IDs.
    # A wrong source vertex/edge must not authorize the actual intersection.
    trace = (
        ((10,) if trace_dimension == 0 else (10, 13))
        if authorized
        else ((13,) if trace_dimension == 0 else (10, 11))
    )
    contact = affine_simplex_contact(
        first,
        second,
        1000,
        authoritative_first_traces=(trace,),
    )
    if contact is None:
        raise AssertionError(
            "The lower-dimensional fixture requires a classified contact."
        )
    assert contact[0] == ("separated" if authorized else "overlap")


@pytest.mark.parametrize(
    "variant", ("actual-subface", "concave-contained", "concave-crossing", "outside")
)
def test_closed_original_source_polygon_complete_containment(variant: str) -> None:
    from phydrax.geometry._periodic_embedding import _closed_source_polygon_contains

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
        # The actual original translation intersection, projected to (x,z).
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
    assert _closed_source_polygon_contains(source, subject) is (
        variant in ("actual-subface", "concave-contained")
    )


@pytest.mark.parametrize("variant", ("collinear-prefix", "nonplanar"))
def test_original_source_polygon_plane_uses_complete_original_loop(variant: str) -> None:
    from phydrax.geometry._periodic_embedding import _source_polygon_plane

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
    plane = _source_polygon_plane(points)
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
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element
    from phydrax.discretization._coordinate_enclosure import coordinate_expressions
    from phydrax.geometry._affine_mapped_contact import affine_simplex_contact
    from phydrax.geometry._mapped_embedding import _Cell, _vertices

    # Complete exact banks from the genuine original translation failure.
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
    coordinates = tuple(
        coordinate_expressions(
            element, tuple(tuple(Fraction(value) for value in point) for point in bank)
        )
        for bank in (first_points, second_points)
    )
    assert coordinates[0] is not None and coordinates[1] is not None
    first = _Cell(
        coordinates[0],
        "simplex",
        "tetrahedron",
        3,
        (0, 10, 6, 3),
        _vertices("tetrahedron"),
    )
    second = _Cell(
        coordinates[1],
        "simplex",
        "tetrahedron",
        3,
        (32, 0, 10, 16),
        _vertices("tetrahedron"),
    )
    # Published fragments offer an edge and an isolated vertex, whose union
    # cannot authorize a triangle. ONE original y0 facet trace does.
    traces = ((0, 10, 6),) if original_trace else ((0, 10), (6,))
    contact = affine_simplex_contact(
        first,
        second,
        1000,
        authoritative_first_traces=traces,
    )
    if contact is None:
        raise AssertionError("The translation fixture requires a classified contact.")
    assert contact[0] == ("separated" if original_trace else "overlap")


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
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element
    from phydrax.discretization._coordinate_enclosure import coordinate_expressions
    from phydrax.geometry._mapped_embedding import _Cell, _vertices
    from phydrax.geometry._periodic_embedding import _affine_reference_contact

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
    prism_coordinates = coordinate_expressions(
        coordinate_lagrange_element("prism", 1), prism_points
    )
    tetrahedron_coordinates = coordinate_expressions(
        coordinate_lagrange_element("tetrahedron", 1), tetrahedron_points
    )
    if prism_coordinates is None or tetrahedron_coordinates is None:
        raise AssertionError("The affine contact fixture requires exact P1 sources.")
    prism = _Cell(
        prism_coordinates,
        "prism",
        "prism",
        3,
        (0, 1, 2, 3, 4, 5),
        _vertices("prism"),
    )
    tetrahedron = _Cell(
        tetrahedron_coordinates,
        "simplex",
        "tetrahedron",
        3,
        (3, 4, 5, 6) if shared_topology else (7, 8, 9, 6),
        _vertices("tetrahedron"),
    )
    decision = _affine_reference_contact(prism, tetrahedron, maximum_work)
    if decision is None:
        raise AssertionError("The layer fiber contact requires an exact classification.")
    assert decision[0] == expected
