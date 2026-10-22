#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from copy import deepcopy
from itertools import product
from typing import Any

import numpy as np
import pytest

from examples.native_tetrahedral_meshing import complex_with_cavity
from phydrax.discretization import CellBlock, CellMesh, reference_cell_topology
from phydrax.meshing.qualification import curved_occupied_source
from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._runtime import QualificationFailure
from tools.meshing_qualification.scenarios.family import (
    _AUTHORED_ROOT_CELLS,
    _family_authored_planar_boundary,
    _family_authored_topology,
    _family_completion_obligations,
    _family_obligation_class,
    _family_quad_feature_acceptance,
    _family_topology_acceptance,
    general_all_hex_family_cases,
)


def test_general_all_hex_gate_requires_a_declared_product_obstruction() -> None:
    assert general_all_hex_family_cases() == ("nonsweepable-dual-hex",)


def _voxel_mesh(occupied: set[tuple[int, int, int]]) -> CellMesh:
    offsets = np.asarray(reference_cell_topology("hexahedron").vertices, dtype=np.int64)
    origins = sorted(occupied)
    keys = sorted(
        {tuple(np.asarray(origin) + offset) for origin in origins for offset in offsets}
    )
    index = {key: row for row, key in enumerate(keys)}
    cells = np.asarray(
        [
            [index[tuple(np.asarray(origin) + offset)] for offset in offsets]
            for origin in origins
        ],
        dtype=np.int64,
    )
    return CellMesh(
        np.asarray(keys, dtype=np.float64),
        (CellBlock("cells", "hexahedron", cells),),
        numeric_version="family-gate-test",
    )


def _authored(source: str) -> dict[str, Any]:
    occupied, materials = _AUTHORED_ROOT_CELLS[source]
    return _family_authored_topology(
        curved_occupied_source(set(occupied), materials=materials).reference.complex
    )


def test_one_cavity_does_not_exclude_a_closed_surface_sweep() -> None:
    topology = _family_authored_topology(complex_with_cavity())
    assert topology["boundary"] == {
        "closed_oriented_manifold": True,
        "components": [
            {"shell": "cavity", "euler_characteristic": 2},
            {"shell": "outer", "euler_characteristic": 2},
        ],
        "outer_shells": 1,
        "enclosed_cavities": 1,
    }
    assert topology["interface_components"] == 1
    assert topology["nonsweepable_witness"]["witness"] is None
    assert topology["nonsweepable_witness"]["interval_product_excluded"] is False


def test_axis_caps_and_one_cavity_do_not_prove_general_nonsweepability() -> None:
    junction = _authored("NativeMappedHexSource:curved-six-arm-junction")
    assert junction["boundary"]["enclosed_cavities"] == 0
    assert junction["nonsweepable_witness"]["axes_with_intermediate_caps"] == [0, 1, 2]
    assert junction["nonsweepable_witness"]["witness"] is None
    cavity = _authored("NativeMappedHexSource:curved-enclosed-cavity")
    assert cavity["nonsweepable_witness"]["witness"] is None
    assert cavity["interface_components"] == 0
    material = _authored("NativeMappedHexSource:curved-two-material-cavity")
    assert material["nonsweepable_witness"]["witness"] is None
    assert material["interface_components"] == 1


@pytest.mark.parametrize(
    "occupied",
    [
        {(0, 0, 0)},
        {(0, 0, 0), (1, 0, 0), (0, 1, 0)},
        {(0, 0, z) for z in range(3)} | {(1, 0, z) for z in range(3)},
    ],
)
def test_straight_extrusions_never_receive_a_nonsweepable_witness(
    occupied: set[tuple[int, int, int]],
) -> None:
    source = curved_occupied_source(occupied)
    assert (
        _family_authored_topology(source.reference.complex)["nonsweepable_witness"][
            "witness"
        ]
        is None
    )


def test_published_topology_matches_and_detects_a_lost_cell() -> None:
    occupied = set(
        _AUTHORED_ROOT_CELLS["NativeMappedHexSource:curved-enclosed-cavity"][0]
    )
    case = {
        "authored_topology": _authored("NativeMappedHexSource:curved-enclosed-cavity")
    }
    mesh = _voxel_mesh(occupied)
    regions = np.zeros((len(occupied),), dtype=np.int64)
    accepted = _family_topology_acceptance(mesh, regions, case)
    assert accepted["authored_topology_match"] is True
    assert accepted["nonsweepability_proved"] is False
    # Removing a wall cell opens the cavity: one boundary shell, no enclosed void.
    damaged = _voxel_mesh(occupied - {(1, 1, 0)})
    with pytest.raises(QualificationFailure):
        _family_topology_acceptance(damaged, regions[:-1], case)


def test_three_spherical_boundaries_exclude_interval_and_closed_sweeps() -> None:
    occupied = set(product(range(5), range(3), range(3))) - {
        (1, 1, 1),
        (3, 1, 1),
    }
    source = curved_occupied_source(occupied)
    case = {
        "authored_topology": _family_authored_topology(source.reference.complex),
        "nonsweepability_required": True,
        # The two isolated interior cubes leave the connected outer wall intact.
        "source_construction_proof": {"solid_connected": True},
    }
    accepted = _family_topology_acceptance(
        _voxel_mesh(occupied), np.zeros(len(occupied), dtype=np.int64), case
    )
    assert accepted["nonsweepability_proved"] is True
    topology = accepted["published_topology"]
    assert isinstance(topology, dict)
    assert topology["solid_components"] == 1
    assert topology["boundary"]["components"] == [
        {"shell": "cavity", "euler_characteristic": 2},
        {"shell": "cavity", "euler_characteristic": 2},
        {"shell": "outer", "euler_characteristic": 2},
    ]


def test_cavity_profile_cannot_claim_the_required_product_obstruction() -> None:
    occupied = set(
        _AUTHORED_ROOT_CELLS["NativeMappedHexSource:curved-enclosed-cavity"][0]
    )
    case = {
        "authored_topology": _authored("NativeMappedHexSource:curved-enclosed-cavity"),
        "nonsweepability_required": True,
        "source_construction_proof": {"solid_connected": True},
    }
    with pytest.raises(QualificationFailure):
        _family_topology_acceptance(
            _voxel_mesh(occupied), np.zeros(len(occupied), dtype=np.int64), case
        )


def test_published_material_interface_is_checked() -> None:
    source = "NativeMappedHexSource:curved-two-material-cavity"
    occupied = sorted(_AUTHORED_ROOT_CELLS[source][0])
    case = {"authored_topology": _authored(source)}
    mesh = _voxel_mesh(set(occupied))
    regions = np.asarray([int(cell[0] >= 2) for cell in occupied], dtype=np.int64)
    assert _family_topology_acceptance(mesh, regions, case)["authored_topology_match"]
    with pytest.raises(QualificationFailure):
        _family_topology_acceptance(mesh, np.zeros_like(regions), case)


def _holed_quads(drop: tuple[int, int] | None = None) -> CellMesh:
    keys = [(float(x), float(y)) for y, x in product(range(4), range(4))]
    cells = [
        (x + 4 * y, x + 1 + 4 * y, x + 1 + 4 * (y + 1), x + 4 * (y + 1))
        for y, x in product(range(3), range(3))
        if (x, y) != (1, 1) and (x, y) != drop
    ]
    return CellMesh(
        np.asarray(keys, dtype=np.float64),
        (CellBlock("quads", "quadrilateral", np.asarray(cells, dtype=np.int64)),),
        numeric_version="family-gate-test",
    )


def test_quad_feature_loops_match_and_detect_a_missing_feature_cell() -> None:
    vertices = np.asarray(
        ((0, 0), (3, 0), (3, 3), (0, 3), (1, 1), (1, 2), (2, 2), (2, 1)),
        dtype=np.float64,
    )
    case = {
        "authored_boundary": _family_authored_planar_boundary(
            vertices, ((0, 1, 2, 3), (4, 5, 6, 7))
        )
    }
    assert case["authored_boundary"]["outer_loops"] == 1
    assert case["authored_boundary"]["holes"] == 1
    accepted = _family_quad_feature_acceptance(_holed_quads(), case)
    assert accepted["authored_feature_match"] is True
    assert accepted["boundary_loops"] == 2
    with pytest.raises(QualificationFailure, match="authored feature loops"):
        _family_quad_feature_acceptance(_holed_quads(drop=(0, 1)), case)


def test_obligation_classes_require_explicit_profile_obligations() -> None:
    assert (
        _family_obligation_class(
            {"family": "hexahedron", "route": "mapped_frame_grid_hex"}
        )
        == "declared-exact-all-hex"
    )
    assert (
        _family_obligation_class(
            {
                "family": "hexahedron",
                "route": "plc_dual_hex",
                "nonsweepability_required": True,
            }
        )
        == "general-nonsweepable-all-hex"
    )
    assert (
        _family_obligation_class({"family": "hexahedron", "route": "sweep"})
        == "declared-sweep"
    )
    assert (
        _family_obligation_class(
            {"family": "hexahedron", "route": "plc_hex_dominant", "family_policy": {}}
        )
        == "declared-mixed-family-policy"
    )


def _acceptance() -> dict[str, Any]:
    return {
        "passed": True,
        "actual_family_counts": {"hexahedron": 8},
        "embedding_id": "embedding",
        "coverage_id": "coverage",
        "source_fidelity": {"status": "certified"},
        "minimum_scaled_jacobian": 0.5,
        "size_passed": True,
        "topology": {"authored_topology_match": True, "nonsweepability_proved": True},
    }


def _complete_record() -> dict[str, Any]:
    epoch = {
        "physical": {"l2_error": 0.01},
        "embedded_fv_content": {"stationary_euler_maximum_content_rate": 0.0},
        "result": {"cells": 8},
    }
    transition = {
        "adaptation_id": "adaptation",
        "lineage": {},
        "transfer_id": "transfer",
        "target_acceptance": _acceptance(),
        "allfield_transfer": {},
    }
    return {
        "initial_acceptance": _acceptance(),
        "epochs": [
            {**epoch, **transition, "operation": "refine", "seed_origin": "zero"},
            {
                **epoch,
                **transition,
                "operation": "coarsen",
                "seed_origin": "actual-transferred-accepted-scalar",
            },
            {**epoch, "seed_origin": "actual-transferred-accepted-scalar"},
        ],
        "checkpoint": {
            "passed": True,
            "process_id": -1,
            "history_epochs": 3,
            "epoch_acceptance": [{}, {}, {}],
            "continued_physical": {"l2_error": 0.01},
            "continued_fv": {},
            "cold_native_field_transfer_replay": [{}, {}],
        },
        "resource_refusals": [
            {"control": name}
            for name in (
                "maximum_cells",
                "maximum_geometry_queries",
                "maximum_scratch_bytes",
                "maximum_work_units",
            )
        ],
    }


_CASE = {
    "family": "hexahedron",
    "route": "plc_dual_hex",
    "nonsweepability_required": True,
    "dimension": 3,
    "authored_topology": {
        "nonsweepable_witness": {"witness": "boundary-component-product-obstruction"}
    },
}
_ARGS = QualificationControls(family_case="nonsweepable-dual-hex")


def test_complete_evidence_satisfies_every_obligation() -> None:
    obligations = _family_completion_obligations(
        _complete_record(), _CASE, _ARGS, 3, 10.0
    )
    assert all(item["passed"] for item in obligations.values()), obligations
    assert "nonsweepable_topology_each_publication" in obligations
    assert "all_field_transfer_each_transition" in obligations


@pytest.mark.parametrize(
    ("damage", "obligation"),
    [
        (lambda record: record.pop("checkpoint"), "cold_fresh_process_reanalysis"),
        (
            lambda record: record["epochs"][1]["target_acceptance"][
                "actual_family_counts"
            ].update({"tetrahedron": 1}),
            "exact_required_family",
        ),
        (
            lambda record: record["epochs"][1].update({"operation": "refine"}),
            "refine_then_coarsen_each_round",
        ),
        (
            lambda record: record["epochs"][2].update({"physical": {"l2_error": 0.06}}),
            "physical_error_each_epoch",
        ),
        (
            lambda record: record["epochs"][0]["target_acceptance"]["topology"].update(
                {"authored_topology_match": False}
            ),
            "nonsweepable_topology_each_publication",
        ),
        (
            lambda record: record["resource_refusals"].pop(),
            "controlled_resource_refusals",
        ),
        (
            lambda record: record["epochs"][2].update({"embedded_fv_content": None}),
            "fv_consumer_each_epoch",
        ),
    ],
)
def test_missing_evidence_blocks_completion(damage: Any, obligation: str) -> None:
    record = deepcopy(_complete_record())
    damage(record)
    obligations = _family_completion_obligations(
        record, _CASE, _ARGS, 3, 10.0
    )
    assert obligations[obligation]["passed"] is False


def test_status_alone_and_late_completion_are_incomplete() -> None:
    record = {**_complete_record(), "epochs": [], "checkpoint": None}
    obligations = _family_completion_obligations(record, _CASE, _ARGS, 0, 10.0)
    assert not obligations["accepted_epochs"]["passed"]
    late = _family_completion_obligations(
        _complete_record(), _CASE, _ARGS, 3, _ARGS.timeout + 1.0
    )
    assert not late["wall_time"]["passed"]


@pytest.fixture(scope="module")
def published_frame_source() -> tuple[Any, dict[str, Any]]:
    import phydrax as phx
    from tools.meshing_qualification.scenarios.family import (
        _family_qualification_fixture,
    )

    source, request, options, case = _family_qualification_fixture(
        QualificationControls(family_case="curved-frame-sweep-hex", resolution=4)
    )
    result = phx.meshing.NativeMeshingProvider(options).plan(
        source, request, coordinate_contract=phx.SpatialCoordinateContract.si()
    ).execute()
    assert result.certification is not None
    assert result.certification.passed
    return result, case


def test_frame_source_renewal_reproves_roots_in_the_consuming_budget(
    published_frame_source: tuple[Any, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry import _mapped_reference_coverage as mapped
    from phydrax.geometry._embedding_state import coordinate_certificate_request
    from tools.meshing_qualification.scenarios import family

    result, case = published_frame_source
    certificate = result.certification
    authority = certificate.request
    assert authority.limits is not None
    root_budgets, renewal_budgets = [], []
    prove_roots = mapped.certify_mapped_reference_source
    renew_source = family.certify_meshing_acceptance

    def record_roots(*args: Any, **kwargs: Any) -> Any:
        root_budgets.append(kwargs["budget"])
        return prove_roots(*args, **kwargs)

    def record_renewal(*args: Any, **kwargs: Any) -> Any:
        renewal_budgets.append(kwargs["budget"])
        return renew_source(*args, **kwargs)

    monkeypatch.setattr(mapped, "certify_mapped_reference_source", record_roots)
    monkeypatch.setattr(family, "certify_meshing_acceptance", record_renewal)
    with coordinate_certificate_request(authority.limits) as budget:
        accepted = family._family_source_acceptance(
            authority, result, case, certificate, budget=budget
        )
        assert root_budgets and all(value is budget for value in root_budgets)
        assert renewal_budgets == [budget]
    renewed = accepted.renewed
    assert renewed.passed
    assert renewed.embedding is not None
    assert renewed.fidelity is not None
    assert accepted.measured is not None
    assert renewed.embedding.restriction_root_embedding is not None
    assert renewed.fidelity.status == "certified"
    assert renewed.fidelity.mesh_to_source_upper == 0.0
    assert renewed.fidelity.source_to_mesh_upper == 0.0
    assert accepted.measured["solid"] == pytest.approx(
        case["expected_regions"]["solid"], abs=1e-10
    )


def test_frame_source_renewal_refuses_changed_target_control_bank(
    published_frame_source: tuple[Any, dict[str, Any]],
) -> None:
    import equinox as eqx
    import jax.numpy as jnp

    from tools.meshing_qualification.scenarios import family

    result, case = published_frame_source
    certificate = result.certification
    changed = eqx.tree_at(
        lambda value: value.geometry.coordinates,
        result,
        jnp.asarray(result.geometry.coordinates).at[0, 0].add(1 / 1024),
    )
    with pytest.raises(ValueError, match="geometry|bound|source|coordinate"):
        family._family_source_acceptance(
            certificate.request, changed, case, certificate
        )


def test_family_q1_galerkin_physical_error_has_exact_affine_reference() -> None:
    from types import SimpleNamespace

    import jax.numpy as jnp

    import phydrax as phx
    from tools.meshing_qualification.scenarios import family

    lattice = _voxel_mesh({(i, j, k) for i in range(4) for j in range(4) for k in range(4)})
    mesh = CellMesh(
        np.asarray(lattice.coordinates) / 4,
        lattice.blocks,
        numeric_version="family-Q1-affine-Galerkin-reference",
    )
    result = SimpleNamespace(
        mesh=mesh, geometry=phx.discretization.CellGeometrySpec.affine(mesh)
    )
    case = {"name": "sweep-hex", "dimension": 3, "family": "hexahedron"}
    space, problem, _ = family._family_qualification_solver(result, case)
    system, rhs = problem.linear_system()
    solved = phx.linalg.solve(system, rhs)
    assert bool(jnp.all(solved.successful))
    values = problem.expand(solved.value)
    nodal = space.project("u", lambda points, args: jnp.sum(points**2, axis=-1))
    np.testing.assert_allclose(values, nodal, rtol=1e-9, atol=1e-10)
    physical = family._family_qualification_physical(space, values, case)
    # On each affine cell, grad(I_Q1 u - u) is orthogonal to every
    # Q1 test gradient. Thus this is the actual Galerkin solution, not an
    # interpolation-error surrogate for a curved-source target assertion.
    # Integral (sum_i t_i(1-t_i))^2 = 3/30 + 6/36 = 4/15.
    assert physical["l2_error"] ** 2 == pytest.approx(1 / 960, abs=1e-10)
    assert physical["inventory"] == pytest.approx(1 + 1 / 32, abs=1e-9)
    invalid = family._family_qualification_physical(
        space, jnp.full_like(values, jnp.nan), case
    )
    assert np.isnan(invalid["l2_error"])


def test_frame_quality_uses_the_same_explicit_cold_request_budget(
    published_frame_source: tuple[Any, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.meshing.qualification import _family as preparations
    from phydrax.meshing.qualification._native_lifecycle import execute_native_request
    from tools.meshing_qualification.scenarios import family

    result, case = published_frame_source
    _, request, _, _ = family._family_qualification_fixture(
        QualificationControls(family_case=case["name"], resolution=4)
    )
    original_edges = preparations.mapped_edge_evidence
    received = []

    def edges(*args: Any, **kwargs: Any) -> Any:
        received.append(kwargs["budget"])
        return original_edges(*args, **kwargs)

    def no_replacement(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Explicit family quality budget must not be replaced.")

    monkeypatch.setattr(preparations, "mapped_edge_evidence", edges)
    monkeypatch.setattr(preparations, "coordinate_budget", no_replacement)

    def consume(budget: Any) -> Any:
        measured = preparations.prepare_family_quality_measurements(
            result,
            request,
            profile=case["name"],
            limits=request.limits,
            budget=budget,
        )
        assert received == [budget]
        assert np.all(np.isfinite(measured.lengths))
        assert measured.edge_lower is not None
        assert np.all(measured.edge_lower > 0.0)
        return measured

    continued = execute_native_request(
        result,
        request.limits,
        consume,
        target=lambda _: result,
        failure_factory=family.qualification_failure,
    )
    # The physical carrier owner is unchanged; its independently ended receipt
    # is new and contains this request's actual work and source-query charges.
    receipt = continued.result.execution_evidence
    previous_receipt = result.execution_evidence
    assert receipt is not None and previous_receipt is not None
    assert receipt.owner_id == previous_receipt.owner_id
    assert not np.array_equal(
        receipt.receipt_id,
        previous_receipt.receipt_id,
    )


@pytest.mark.meshcore
def test_original_r4_family_epoch_keeps_the_paid_request_and_actual_history() -> None:
    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification.scenarios import family

    controls = QualificationControls(family_case="transfinite-quad", resolution=4)
    initial = family._family_lifecycle_start(controls)
    prepared = family._family_prepare_publication(
        controls, initial.record, initial.phases, deadline=initial.deadline
    )
    publication = family._family_execute_publication(
        prepared, initial.record, initial.phases, deadline=initial.deadline
    )
    result = publication.result
    represented = bool(result.associations)
    audit = family.phx.meshing.CellMeshAuditPolicy(
        require_complete_association=represented,
        watertight_boundary=(
            family.phx.meshing.CellMeshAuditDisposition.REJECT
            if represented else family.phx.meshing.CellMeshAuditDisposition.SKIP
        ),
    )
    assert audit.policy_id == result.audit.policy_id
    inputs = family._FamilyEpochInputs(
        prepared.request, publication.authority, audit,
        publication.association_transfer, controls, prepared.case,
        initial.record, family.StageLedger(initial.record, deadline=initial.deadline),
        initial.started,
    )
    consumed = family.execute_native_consumer(
        result, prepared.request.limits,
        lambda budget: family._family_consume_epoch(
            inputs, result, 0, None, None, budget=budget
        ),
        failure_factory=family.qualification_failure,
    )
    paid = consumed.value
    epoch = family.rebind_meshing_accepted_epoch_execution(
        paid.solved.accepted, consumed.result, transition=None
    )
    previous_receipt = result.execution_evidence
    receipt = consumed.result.execution_evidence
    assert previous_receipt is not None and receipt is not None
    assert receipt.owner_id == previous_receipt.owner_id
    assert int(np.asarray(receipt.total_work_units)) > int(np.asarray(previous_receipt.total_work_units))
    assert not np.array_equal(receipt.receipt_id, previous_receipt.receipt_id)
    assert paid.solved.physical["l2_error"] <= controls.target_error
    assert paid.entry["embedded_fv_content"]["stationary_euler_maximum_content_rate"] <= 1e-10
    assert set(paid.fields) == {*family.LAYER_CORE_FIELD_KINDS, "material_content"}
    for kind, owner in paid.owners.field_spaces.items():
        assert owner.mesh is result.mesh
        for index, name in enumerate(("current", "history")):
            np.testing.assert_array_equal(
                epoch.fields[f"{kind}/{name}/coefficients"], paid.fields[kind][:, index]
            )
    np.testing.assert_array_equal(
        epoch.fields["FV/state/content"], paid.fields["material_content"]
    )
    assert paid.entry["embedded_fv_content"]["preparation"]["reused_prepared_owner"] is True


@pytest.mark.meshcore
@pytest.mark.parametrize("fault", ("none", "limits", "scratch", "foreign"))
def test_family_association_requires_exact_live_request_allowance(fault: str) -> None:
    import phydrax as phx
    from phydrax._meshcore import NativeExecutionScope
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.meshing.qualification import prepare_family_association_transfer
    from tests._support.cell_meshes import triangle_grid
    from tools.meshing_qualification.scenarios.family import _family_qualification_fixture

    source, request, _, case = _family_qualification_fixture(
        QualificationControls(
            family_case="dual-quad", resolution=1, capacity=10_000,
            maximum_scratch_bytes=1_536_000_000,
        )
    )
    result = phx.meshing.certify_cell_mesh(
        triangle_grid(1, 1), phx.SpatialCoordinateContract.si(),
    )
    limits = request.limits
    with NativeExecutionScope(limits.native_limits()) as scope:
        with scope.ticket_view() as ticket:
            budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=(
                    536_870_912 if fault == "scratch" else limits.maximum_scratch_bytes
                ),
            )

            def prepare(receiving_scope: NativeExecutionScope) -> None:
                assert prepare_family_association_transfer(
                    source, result, request, domain=case["domain"],
                    limits=(
                        phx.meshing.MeshingLimits(maximum_scratch_bytes=536_870_912)
                        if fault == "limits" else limits
                    ),
                    record_phase=lambda _: None, budget=budget, scope=receiving_scope,
                ) is None

            if fault == "foreign":
                with NativeExecutionScope(limits.native_limits()) as foreign:
                    with pytest.raises(ValueError, match="live scope and limits"):
                        prepare(foreign)
            elif fault != "none":
                with pytest.raises(ValueError, match="authored request"):
                    prepare(scope)
            else:
                prepare(scope)
