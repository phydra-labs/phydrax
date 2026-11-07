from __future__ import annotations

import importlib
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.meshing_qualification import cli


_TEST_SCENARIO = "native-dispatch-contract-test"


def _invoke_with_result(
    monkeypatch: pytest.MonkeyPatch, result: dict[str, object]
) -> dict[str, object]:
    monkeypatch.setitem(
        cli.NATIVE_CASE_REGISTRY, _TEST_SCENARIO, (".test_scenario", "run")
    )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _module, _package: SimpleNamespace(run=lambda: result),
    )
    monkeypatch.setattr(cli, "runtime_identity", lambda: {"engine": "unit-test"})
    return cli._invoke(_TEST_SCENARIO, Namespace())


def test_dispatch_rejects_missing_explicit_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _invoke_with_result(monkeypatch, {"evidence": {"oracle": "checked"}})

    assert record["status"] == "failed"
    assert record["error_type"] == "RuntimeError"
    assert "explicit nonempty status" in str(record["detail"])


def test_dispatch_rejects_noncanonical_scenario_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _invoke_with_result(
        monkeypatch,
        {"status": "passed", "scenario": "legacy-generation-alias"},
    )

    assert record["status"] == "failed"
    assert record["error_type"] == "RuntimeError"
    assert "noncanonical identity" in str(record["detail"])


def test_dispatch_preserves_explicit_status_and_canonical_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _invoke_with_result(
        monkeypatch,
        {"status": "passed", "evidence": {"oracle": "checked"}},
    )

    assert record["status"] == "passed"
    assert record["scenario"] == _TEST_SCENARIO


def test_family_checkpoint_survives_later_profile_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tools.meshing_qualification.scenarios import native_family_matrix

    monkeypatch.setattr(native_family_matrix, "_profile_ids", lambda: ("first", "second"))

    def run_profile(profile_id: str, *, checkpoint: object) -> dict[str, object]:
        if profile_id == "second":
            raise RuntimeError("later profile interrupted")
        return {"profile_id": profile_id, "status": "passed"}

    monkeypatch.setattr(native_family_matrix, "_run_profile", run_profile)
    monkeypatch.setattr(cli, "runtime_identity", lambda: {"engine": "unit-test"})
    output = tmp_path / "result.json"
    result = cli._invoke("native-general-families", Namespace(output=output))
    assert result["status"] == "failed"
    events = [
        json.loads(line)
        for line in (tmp_path / "result.json.profiles.jsonl").read_text().splitlines()
    ]
    assert events == [
        {
            "profile_id": "first",
            "event": "profile-complete",
            "record": {"profile_id": "first", "status": "passed"},
        }
    ]


def test_records_and_checkpoints_are_byte_identical_line_feed_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tools.meshing_qualification.scenarios import native_family_matrix

    monkeypatch.setattr(native_family_matrix, "_profile_ids", lambda: ("first",))

    def run_profile(
        profile_id: str, *, checkpoint: object, capture_result: object
    ) -> dict[str, object]:
        return {"profile_id": profile_id, "status": "passed"}

    monkeypatch.setattr(native_family_matrix, "_run_profile", run_profile)
    monkeypatch.setattr(cli, "runtime_identity", lambda: {"engine": "unit-test"})
    output = tmp_path / "result.json"
    status = cli.main(["--scenario", "native-general-families", "--output", str(output)])
    assert status == 0
    for path in (output, tmp_path / "result.json.profiles.jsonl"):
        payload = path.read_bytes()
        assert payload.endswith(b"\n")
        assert b"\r" not in payload


def test_run_rejects_empty_scenario_selection() -> None:
    with pytest.raises(ValueError, match="at least one explicit scenario"):
        cli.run((), Namespace())


def test_native_case_migration_covers_every_authored_profile() -> None:
    migration_path = (
        Path(__file__).resolve().parents[2]
        / "tools/meshing_qualification/native_case_migration.json"
    )
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    inventory = migration["inventory"]
    contracts = migration["workflow_contracts"]
    profiles = [profile for contract in contracts for profile in contract["profiles"]]

    assert migration["provenance"]["source_branch"] == "feature/native-meshing-20260929"
    assert (
        migration["provenance"]["source_commit"]
        == "c56614a9fe52c4a8f56484874e91fde6498ec9ff"
    )
    assert inventory["family_count"] == len(contracts) == 13
    assert inventory["profile_count"] == len(profiles) == 114
    assert inventory["mapped_profile_count"] == len(set(profiles)) == 114
    assert inventory["unmapped_profile_count"] == 0
    assert migration["provenance"]["expected_admission"] == "positive"
    assert migration["provenance"]["refusal_counts_as_positive_completion"] is False
    assert all(
        contract.get("target_scenario") or contract.get("target_scenarios")
        for contract in contracts
    )
    targets: set[str] = set()
    for contract in contracts:
        target = contract.get("target_scenario")
        if isinstance(target, str):
            targets.add(target)
        targets.update(contract.get("target_scenarios", ()))
        targets.update((contract.get("profile_target_overrides") or {}).values())
    for workflow in migration["additional_mandatory_workflows"]:
        target = workflow.get("target_scenario")
        if isinstance(target, str):
            targets.add(target)
        targets.update(workflow.get("target_scenarios", ()))
    assert targets <= set(cli.NATIVE_SCENARIOS)
    assert all(
        all(
            contract.get(field)
            for field in (
                "source",
                "source_provenance",
                "required_feature_combinations",
                "independent_oracle",
                "tolerances",
                "target_gate_status",
            )
        )
        for contract in contracts
    )
    assert all(
        all(
            workflow.get(field)
            for field in (
                "source",
                "source_provenance",
                "independent_oracle",
                "tolerances",
                "coverage_status",
            )
        )
        for workflow in migration["additional_mandatory_workflows"]
    )


def test_periodic_cone_payload_interpolates_physical_corner_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fractions import Fraction as F

    import numpy as np

    from phydrax.meshing import _distribution
    from phydrax.meshing._lineage import (
        EntityLineage,
        EntityLineageKind,
        MeshLineage,
        VertexInterpolationStencil,
    )
    from tools.meshing_qualification.scenarios.cold_owner_local import (
        _transfer_periodic_cone_kinematics,
    )

    def view(cell: int, ids: tuple[int, ...], points: tuple[tuple[F, F], ...]) -> object:
        block = SimpleNamespace(
            cell_kind="triangle",
            cell_ids=np.asarray((cell,), dtype=np.int64),
            vertices=np.asarray(((0, 1, 2),), dtype=np.int64),
            vertex_valid=np.ones((1, 3), dtype=bool),
        )
        return SimpleNamespace(
            cell_ids=block.cell_ids,
            vertex_ids=np.asarray(ids, dtype=np.int64),
            exact_vertex_coordinates=points,
            blocks=(block,),
        )

    parent = ((F(1), F(0)), (F(2), F(0)), (F(1), F(1)))
    child = ((F(1), F(0)), (F(3, 2), F(0)), (F(1), F(1)))
    stencil = VertexInterpolationStencil(
        "source",
        "target",
        np.asarray((10, 13, 12), dtype=np.int64),
        np.asarray(((10, -1), (10, 11), (12, -1)), dtype=np.int64),
        np.asarray(((1.0, 0.0), (0.5, 0.5), (1.0, 0.0))),
        np.asarray(((True, False), (True, True), (True, False))),
        exact_weights=((F(1), F(0)), (F(1, 2), F(1, 2)), (F(1), F(0))),
    )
    lineage = MeshLineage(
        "source",
        "target",
        (
            EntityLineage(
                2, "source", "target", (7,), (8,), (int(EntityLineageKind.REFINED_FROM),)
            ),
        ),
    )
    values = np.arange(12, dtype=np.float64).reshape((1, 4, 3))
    context = SimpleNamespace(
        source_geometry_view=view(7, (10, 11, 12), parent),
        target_geometry_view=view(8, (10, 13, 12), child),
        execution_group=None,
        axis_name="cone_test",
    )

    def request(keys: np.ndarray, *, axis_name: str) -> np.ndarray:
        assert axis_name == "cone_test_cone_parents"
        np.testing.assert_array_equal(keys, ((2, 7, 0),))
        return values

    action = SimpleNamespace(
        repartition=context,
        target_ownership=SimpleNamespace(dof_keys=np.asarray(((2, 8, 0),))),
        request_source_rows=request,
    )
    monkeypatch.setattr(
        _distribution,
        "_owner_group_fail_if_any",
        lambda error, _group, _axis, _description: (
            None if error is None else pytest.fail(error)
        ),
    )
    result = _transfer_periodic_cone_kinematics(
        values, None, SimpleNamespace(lineage=lineage, vertex_stencil=stencil), action
    )
    np.testing.assert_allclose(
        result[0],
        (values[0, 0], 0.5 * (values[0, 0] + values[0, 1]), values[0, 2], values[0, 3]),
    )
    empty_view = SimpleNamespace(cell_ids=np.empty((0,), dtype=np.int64))
    empty_context = SimpleNamespace(
        source_geometry_view=empty_view,
        target_geometry_view=empty_view,
        execution_group=None,
        axis_name="cone_test",
    )
    calls = []

    def empty_request(keys: np.ndarray, *, axis_name: str) -> np.ndarray:
        calls.append((keys.shape, axis_name))
        return np.empty((0, 4, 3))

    empty_action = SimpleNamespace(
        repartition=empty_context,
        target_ownership=SimpleNamespace(dof_keys=np.empty((0, 3), dtype=np.int64)),
        request_source_rows=empty_request,
    )
    empty_result = _transfer_periodic_cone_kinematics(
        np.empty((0, 4, 3)), None, None, empty_action
    )
    assert empty_result.shape == (0, 4, 3)
    assert calls == [((0, 3), "cone_test_cone_parents")]


@pytest.mark.strict_jax
def test_periodic_cone_newmark_consumes_j2_and_rates() -> None:
    import jax.numpy as jnp
    import numpy as np

    import phydrax as phx
    from tools.meshing_qualification.scenarios.cold_fe_cone_physics import (
        ConePhysicalWorkset,
    )
    from tools.meshing_qualification.scenarios.cold_owner_local import (
        _advance_periodic_cone,
    )

    vertices = np.asarray(
        ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    law = phx.applications.solid_mechanics.FiniteStrainJ2MPMConstitutivePlan()
    material = phx.equations.MaterialState(
        phx.equations.MaterialSiteId.for_cell_quadrature(
            10, "centroid", domain_id="periodic-cone", quadrature_rule_id="tetra-centroid"
        ),
        "j2",
        law.initialize_state((), jnp.float64),
    )
    velocity = vertices @ np.diag((4.0, -2.0, -2.0))
    zero = np.zeros((4, 3))
    workset = ConePhysicalWorkset()
    first, pending = _advance_periodic_cone(
        vertices,
        zero,
        velocity,
        zero,
        material,
        0,
        0,
        0,
        0.05,
        pending_increment=0.01 * velocity,
        workset=workset,
    )
    assert not np.array_equal(
        np.asarray(pending), np.asarray(first.materials.states[0].committed)
    )
    assert len(workset._programs) == 1
    assert first.materials is not None
    assert float(first.materials.states[0].committed[-1]) > 0.0
    second = _advance_periodic_cone(
        vertices,
        first.displacement,
        first.velocity,
        first.acceleration,
        first.materials.states[0],
        float(first.time),
        int(first.step),
        int(first.state_version),
        0.05,
        workset=workset,
    )
    assert len(workset._programs) == 1
    np.testing.assert_allclose(
        second.displacement,
        first.displacement
        + 0.05 * first.velocity
        + 0.25 * 0.05**2 * (first.acceleration + second.acceleration),
        rtol=1e-7,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.sum(second.velocity, axis=0), np.sum(first.velocity, axis=0), atol=1e-8
    )
    assert second.materials is not None
    assert second.materials.states[0].state_version == 2


@pytest.mark.strict_jax
def test_periodic_cone_grouped_sites_and_duplicate_vertices() -> None:
    import jax.numpy as jnp
    import numpy as np

    import phydrax as phx
    from tools.meshing_qualification.scenarios.cold_fe_cone_physics import (
        ConePhysicalWorkset,
    )

    tetra = np.asarray(
        ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    vertices = np.stack((tetra, tetra))
    law = phx.applications.solid_mechanics.FiniteStrainJ2MPMConstitutivePlan()
    materials = tuple(
        phx.equations.MaterialState(
            phx.equations.MaterialSiteId.for_cell_quadrature(
                cell,
                "centroid",
                domain_id="periodic-cone",
                quadrature_rule_id="tetra-centroid",
            ),
            "j2",
            law.initialize_state((), jnp.float64),
        )
        for cell in (23, 7)
    )
    velocity = vertices @ np.diag((4.0, -2.0, -2.0))
    velocity[1] *= 1.1
    zero = np.zeros_like(vertices)
    workset = ConePhysicalWorkset()
    result, trials = workset.advance(
        vertices,
        zero,
        velocity,
        zero,
        materials,
        0.0,
        0,
        0,
        0.05,
        pending_increment=0.01 * velocity,
    )
    assert len(workset._programs) == 1
    for index, material in enumerate(materials):
        accepted = result.materials.state(material.site_id)
        assert accepted.state_version == 1
        assert not np.array_equal(
            np.asarray(accepted.committed),
            np.asarray(trials.state(material.site_id).trial),
        )
        np.testing.assert_allclose(
            result.displacement[index],
            0.05 * velocity[index] + 0.25 * 0.05**2 * result.acceleration[index],
            rtol=1e-8,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            np.sum(result.velocity[index], axis=0),
            np.sum(velocity[index], axis=0),
            rtol=1e-8,
            atol=1e-10,
        )
    assert not np.array_equal(
        np.asarray(result.materials.state(materials[0].site_id).committed),
        np.asarray(result.materials.state(materials[1].site_id).committed),
    )
    differences = []
    newmark_errors = []
    momentum_errors = []
    for index, material in enumerate(materials):
        independent = ConePhysicalWorkset().advance(
            vertices[index],
            zero[index],
            velocity[index],
            zero[index],
            material,
            0.0,
            0,
            0,
            0.05,
        )
        differences.append(
            (
                float(
                    np.max(
                        np.abs(
                            np.asarray(result.displacement[index])
                            - np.asarray(independent.displacement)
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            np.asarray(result.materials.state(material.site_id).committed)
                            - np.asarray(
                                independent.materials.state(material.site_id).committed
                            )
                        )
                    )
                ),
            )
        )
        newmark_errors.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(independent.displacement)
                        - 0.05 * velocity[index]
                        - 0.25 * 0.05**2 * np.asarray(independent.acceleration)
                    )
                )
            )
        )
        momentum_errors.append(
            float(
                np.max(
                    np.abs(
                        np.sum(np.asarray(independent.velocity), axis=0)
                        - np.sum(velocity[index], axis=0)
                    )
                )
            )
        )
        np.testing.assert_allclose(
            independent.displacement,
            0.05 * velocity[index] + 0.25 * 0.05**2 * independent.acceleration,
            rtol=1e-8,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            np.sum(independent.velocity, axis=0),
            np.sum(velocity[index], axis=0),
            rtol=1e-8,
            atol=1e-10,
        )
        assert float(independent.materials.state(material.site_id).committed[-1]) > 0.0
    print(
        "grouped/independent displacement and J2 differences:",
        differences,
        "independent Newmark errors:",
        newmark_errors,
        "momentum errors:",
        momentum_errors,
    )


@pytest.mark.strict_jax
def test_periodic_cone_material_refinement_retains_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jax.numpy as jnp
    import numpy as np

    import phydrax as phx
    from phydrax.meshing import _distribution
    from phydrax.meshing._lineage import EntityLineage, EntityLineageKind, MeshLineage
    from tools.meshing_qualification.scenarios.cold_owner_local import (
        _refine_periodic_cone_materials,
    )

    equations = phx.equations
    mechanics = phx.applications.solid_mechanics
    law = mechanics.FiniteStrainJ2MPMConstitutivePlan()
    model = mechanics.FiniteStrainConstitutiveModel(law, 1.0)
    site = lambda cell: equations.MaterialSiteId.for_cell_quadrature(
        cell, "centroid", domain_id="cone-test", quadrature_rule_id="tetra-centroid"
    )
    parent, child = site(7), site(8)
    source_owner = equations.MaterialSiteOwnership(((parent, 0),), part_count=3)
    target_owner = equations.MaterialSiteOwnership(((child, 0),), part_count=3)
    target_plan = equations.MaterialIntegrationPlan(
        ((child, model),), ownership=target_owner, owner_rank=0
    )
    committed = law.initialize_state((), jnp.float64)
    trial = committed.at[-1].set(0.25)
    source = equations.MaterialTransaction(
        (
            equations.MaterialState(
                parent, model.model_id, committed, trial=trial, state_version=3
            ),
        ),
        ownership=source_owner,
        owner_rank=0,
    )
    lineage = MeshLineage(
        "source",
        "target",
        (
            EntityLineage(
                2, "source", "target", (7,), (8,), (int(EntityLineageKind.REFINED_FROM),)
            ),
        ),
    )
    context = SimpleNamespace(
        target_material_plan=target_plan, execution_group=None, axis_name="cone_test"
    )
    monkeypatch.setattr(
        _distribution,
        "_owner_group_fail_if_any",
        lambda error, _group, _axis, _description: (
            None if error is None else pytest.fail(error)
        ),
    )
    moved = _refine_periodic_cone_materials(source, lineage, context)
    assert moved.state(child).state_version == 3
    np.testing.assert_array_equal(moved.state(child).committed, committed)
    np.testing.assert_array_equal(moved.state(child).trial, trial)


@pytest.mark.strict_jax
@pytest.mark.parametrize("rank", [0, 1, 2])
def test_physical_cone_writer_owned_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rank: int
) -> None:
    """Exercise real cone dynamics while isolating the three-rank transport."""
    import numpy as np

    import phydrax as phx
    from phydrax.discretization.fem._distributed import FiniteElementDofOwnership
    from phydrax.meshing import _distribution, _model_codec
    from tools.meshing_qualification.scenarios import cold_fe_cone_writer as writer

    roles = phx.solver.FiniteElementFieldTransferSemantics
    layouts = (
        phx.solver.FiniteElementAcceptedFieldLayout(
            phx.discretization.FiniteElementFieldSpec(
                "u", phx.discretization.lagrange_element("triangle", 1)
            ),
            transfer_semantics=roles.NONCONSERVATIVE_INTERPOLATION,
        ),
        phx.solver.FiniteElementAcceptedFieldLayout(
            phx.discretization.FiniteElementFieldSpec(
                "content", phx.discretization.discontinuous_element("triangle", 0)
            ),
            transfer_semantics=roles.CONSERVATIVE_CELL_AVERAGE,
        ),
    )
    cell = 10 + rank
    count = int(rank != 2)
    keys = (
        np.asarray([[0, 20 + rank, 0]] if count else [], dtype=np.int64).reshape(-1, 3),
        np.asarray([[2, cell, 0]] if count else [], dtype=np.int64).reshape(-1, 3),
    )
    ownerships = tuple(
        FiniteElementDofOwnership(
            key,
            np.full((count,), rank, dtype=np.int32),
            field_layout_id=layout.layout_id,
            identity_id=f"identity-{index}",
            orientation_action_id="orientation",
            distribution_id="distribution",
            topology_id="topology",
            geometry_id="geometry",
            local_rank=rank,
            part_count=3,
        )
        for index, (key, layout) in enumerate(zip(keys, layouts, strict=True))
    )
    identities = tuple(
        SimpleNamespace(ownership=lambda value=value: value) for value in ownerships
    )
    fields = (np.full((count,), 0.3), np.full((count,), -0.7))
    view = SimpleNamespace(
        owned_cell_ids=np.asarray([cell] if count else [], dtype=np.int64),
        source_topology_id="topology",
    )
    group = SimpleNamespace(spec=SimpleNamespace(process_indices=(0, 1, 2)))
    monkeypatch.setattr(
        writer,
        "_periodic_cone_triangles",
        lambda _view: {cell: ((0, 1, 2), ((1, 0), (2, 0), (1, 1)))} if count else {},
    )
    monkeypatch.setattr(
        _distribution,
        "_owner_group_allgather",
        lambda _values, _group, axis: (
            np.asarray([[1], [1], [0]], dtype=np.int64)
            if axis.endswith("_counts")
            else np.asarray([[10], [11], [-1]], dtype=np.int64)
        ),
    )
    monkeypatch.setattr(
        _distribution,
        "_owner_group_fail_if_any",
        lambda error, *_args: (
            (_ for _ in ()).throw(RuntimeError(error)) if error else None
        ),
    )
    captured = []
    monkeypatch.setattr(writer, "_source_composition", lambda *args, **kwargs: args[2])
    monkeypatch.setattr(
        _model_codec,
        "write_model_artifact",
        lambda *args, **kwargs: captured.append(args),
    )
    manifest, plan, composition = writer.write_physical_cone_restart(
        tmp_path / "cone.phx",
        view,
        None,
        SimpleNamespace(prepared_id="prepared"),
        SimpleNamespace(compilation_id="compiled"),
        identities,
        layouts,
        fields,
        None,
        None,
        None,
        None,
        group=group,
        rank=rank,
    )
    assert composition is manifest and captured[0][1] is manifest
    assert plan.ownership.part_count == 3
    assert {
        site.cell_id: owner
        for site, owner in zip(
            plan.ownership.site_ids, plan.ownership.owners, strict=True
        )
    } == {10: 0, 11: 1}
    assert len(plan.site_ids) == count
    assert manifest.state.step == manifest.state.state_version == 1
    for actual, expected in zip(manifest.state.fields, fields, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert tuple(name for name, _ in manifest.auxiliary_state) == (
        "cone/displacement",
        "cone/velocity",
        "cone/acceleration",
    )
    for (_, value), layout in zip(
        manifest.auxiliary_state, manifest.auxiliary_layouts, strict=True
    ):
        assert value.shape == (count, 4, 3)
        assert (
            layout.association
            is phx.solver.FiniteElementRestartStateAssociation.OWNED_ROWS
        )
        assert layout.ownership.ownership_id == ownerships[1].ownership_id
    assert float(manifest.integrator_state[0][1]) == pytest.approx(0.05)
    assert (
        manifest.integrator_layouts[0].association
        is phx.solver.FiniteElementRestartStateAssociation.REPLICATED
    )
    assert len(manifest.material_layouts) == count
    for material, layout in zip(
        manifest.state.materials.states, manifest.material_layouts, strict=True
    ):
        assert material.site_id.cell_id == cell
        assert material.state_version == 1
        assert float(material.committed[-1]) > 0
        assert not np.array_equal(material.committed, material.trial)
        assert (
            layout.association
            is phx.solver.FiniteElementRestartStateAssociation.OWNED_SITES
        )
        np.testing.assert_array_equal(material.rollback().trial, material.committed)


def test_cold_fe_enumerates_all_false_transfer_claims() -> None:
    from typing import get_args

    from tools.meshing_qualification.scenarios.cold_owner_local import _FalseTransferClaim

    assert get_args(_FalseTransferClaim) == ("stencil", "measures")


def test_cold_fe_serial_reference_runs_in_a_finished_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tools.meshing_qualification.scenarios import cold_owner_local

    calls = []
    evidence = {"pid": cold_owner_local.os.getpid() + 1, "geometry_identity": "proof"}

    def run(command: object, **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(evidence), stderr="")

    monkeypatch.setattr(cold_owner_local.subprocess, "run", run)
    environment = {"PYTHONPATH": "example"}
    assert (
        cold_owner_local._launch_serial_reference(tmp_path, environment, "routine")
        == evidence
    )
    command, kwargs = calls[0]
    assert command[-3:] == ("serial-reference", str(tmp_path), "routine")
    assert kwargs["env"] is environment
    assert kwargs["timeout"] == 900.0

    evidence["pid"] = cold_owner_local.os.getpid()
    with pytest.raises(RuntimeError, match="malformed evidence"):
        cold_owner_local._launch_serial_reference(tmp_path, environment, "routine")


@pytest.mark.parametrize(
    ("profile", "entity_count"),
    (
        ("planar-hole-feature", 20),
        ("planar-hole-feature:larger", 20),
        ("planar-thin-channel", 25),
    ),
)
def test_native_planar_profiles_preserve_all_authored_strata(
    profile: str, entity_count: int
) -> None:
    from tools.meshing_qualification.scenarios.native_planar_profiles import (
        _qualify_planar,
    )

    record = _qualify_planar(profile)
    association = record["source_association"]
    assert isinstance(association, dict)
    assert association["complete"]
    assert association["expected_source_entity_count"] == entity_count
    assert association["associated_source_entity_count"] == entity_count
    fidelity = record["source_fidelity"]
    assert isinstance(fidelity, dict)
    assert fidelity["outside_source_domain_points"] == 0


def test_native_junction_network_preserves_branch_and_endpoint_strata() -> None:
    from tools.meshing_qualification.scenarios.native_planar_profiles import (
        _qualify_junction_network,
    )

    record = _qualify_junction_network()
    association = record["source_association"]
    assert isinstance(association, dict)
    assert association["expected_source_entity_count"] == 7
    assert association["associated_source_entity_count"] == 7
    assert record["junction_valence"] == 3


def test_interval_network_preserves_kirchhoff_diffusion() -> None:
    import jax.numpy as jnp
    import numpy as np

    import phydrax as phx
    from phydrax.discretization._cell_complex import IntervalConnectivity

    coordinates = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)))
    cells = jnp.asarray(((0, 1), (0, 2), (0, 3)), dtype=jnp.int32)
    mesh = phx.discretization.CellMesh(
        coordinates, (phx.discretization.CellBlock("branches", "interval", cells),)
    )
    connectivity = mesh.connectivity
    assert isinstance(connectivity, IntervalConnectivity)
    np.testing.assert_array_equal(connectivity.vertex_cell_counts, (3, 1, 1, 1))
    np.testing.assert_array_equal(
        connectivity.boundary_vertices, (False, True, True, True)
    )
    space = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("interval", 1)
        ),
    ).prepare()
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "junction-diffusion", "u", (phx.equations.DiffusionAction("u"),)
        ),
        space,
    )
    points = np.asarray(space.dof_maps[0].dof_coordinates)
    residual = np.asarray(problem.full_residual(jnp.asarray(points[:, 0])))
    np.testing.assert_allclose(residual, points[:, 0], atol=1e-12)
    assert abs(float(residual.sum())) < 1e-12


@pytest.mark.parametrize(
    ("profile", "geometry"),
    (
        ("native-surface-pde:folded_plates", "folded_plates"),
        ("native-surface-pde:torus", "torus"),
        ("sphere-feature-surface", "sphere"),
        ("sphere-feature-surface:larger", "sphere"),
        ("native-surface-pde:capped_cylinder", "capped_cylinder"),
        ("native-surface-pde:capped_cylinder_features", "capped_cylinder"),
    ),
)
def test_native_surface_profiles_preserve_complete_source_and_fidelity(
    profile: str, geometry: str
) -> None:
    from tools.meshing_qualification.scenarios.native_planar_profiles import (
        _qualify_surface,
    )

    record = _qualify_surface(profile, geometry)
    association = record["source_association"]
    assert isinstance(association, dict)
    assert association["complete"]
    assert (
        association["associated_source_entity_count"]
        == association["expected_source_entity_count"]
    )
    fidelity = record["source_fidelity"]
    assert isinstance(fidelity, dict)
    assert fidelity["certificate_id"]
    assert association["declared_source_entity_count"] == (
        association["associated_source_entity_count"]
        + association["collapsed_pole_edge_count"]
    )
    if geometry in ("sphere", "capped_cylinder"):
        assert association["collapsed_pole_edge_count"] == 2
        poles = association["collapsed_pole_edges"]
        assert isinstance(poles, list)
        assert all(
            isinstance(pole, dict) and pole["proved_coedge_uses"] > 0 for pole in poles
        )
    if geometry in ("sphere", "torus", "capped_cylinder"):
        consumer = record["finite_element_consumer"]
        assert isinstance(consumer, dict)
        accuracy = consumer["closed_surface_accuracy"]
        assert isinstance(accuracy, dict)
        assert accuracy["energy_accuracy_passed"]
        error = accuracy["energy_error"]
        interpolant = accuracy["interpolant_energy_error"]
        allowance = accuracy["energy_error_allowance"]
        difference = accuracy["reference_quadrature_difference"]
        assert isinstance(error, (int, float))
        assert isinstance(interpolant, (int, float))
        assert isinstance(allowance, (int, float))
        assert isinstance(difference, (int, float))
        assert error <= interpolant + allowance
        assert difference < 1e-4 * max(error, interpolant)


def test_cylinder_pole_proof_rejects_a_forged_coedge() -> None:
    from phydrax._physical import SpatialCoordinateContract
    from phydrax.geometry.brep import brep_cylinder, LineCurve
    from phydrax.geometry.brep._correspondence import _exact_brep_pole_collapse

    model = brep_cylinder(0.8, 1.5, coordinate_contract=SpatialCoordinateContract.si())
    geometry = model.geometry
    face, coedge = next(
        (face, coedge)
        for face, loops in enumerate(geometry.face_loops)
        for loop in loops
        for coedge in loop
        if geometry.edge_curves[geometry.coedge_edges[coedge]] < 0
    )
    edge = geometry.coedge_edges[coedge]
    vertex = geometry.edge_vertices[edge][0]
    first, last = map(float, geometry.edge_ranges[edge])
    args = (
        first,
        last,
        geometry.vertex_exact_points[vertex],
        geometry.vertex_roots[vertex],
        geometry.coedge_endpoint_roots[coedge],
    )
    assert _exact_brep_pole_collapse(
        None, model.patches[face], geometry.pcurves[coedge], *args
    )
    assert not _exact_brep_pole_collapse(
        None, model.patches[face], LineCurve((1.0, 1.0), (1.0, 0.0)), *args
    )


@pytest.mark.parametrize(
    "failed", (None, "serial_fe", "owner_fe", "serial_fv", "owner_fv")
)
def test_combined_cold_continuation_requires_each_component(
    monkeypatch: pytest.MonkeyPatch, failed: str | None
) -> None:
    from tools.meshing_qualification.scenarios import cold_fe_fv, cold_owner_local

    fe = {
        "status": "passed",
        "fresh_process": True,
        "producer_process_exited_before_consumer": True,
    }
    owner_fe = {
        "status": "passed",
        "producer_processes_exited_before_each_reader": True,
        "profiles": [
            {
                "status": "passed",
                "physical_cone_continuation_by_rank": [
                    {
                        key: True
                        for key in (
                            "changed_placement",
                            "independent_reference",
                            "newmark",
                            "linear_momentum",
                            "material_commit_trial_version",
                            "rollback",
                            "history_velocity_acceleration_timestep_sensitive",
                        )
                    }
                    for _ in range(3)
                ],
                "collective_oracles": {
                    key: True
                    for key in (
                        "stale_migration_refused_without_partial_commit",
                        "one_rank_restart_layout_refusal_without_partial_commit",
                        "saved_source_reused_after_both_refusals",
                        "history_clock_integrator_rng_material_rebound",
                        "next_compiled_fe_solve_accepted",
                    )
                },
            }
        ],
    }
    owner_fv = {
        "status": "passed",
        "producer_processes_exited_before_reader": True,
    }
    if failed == "serial_fe":
        fe["status"] = "failed"
    if failed == "owner_fe":
        owner_fe["status"] = "failed"
    if failed == "owner_fv":
        owner_fv["status"] = "failed"
    monkeypatch.setattr(cold_fe_fv, "native_cold_restore", lambda: fe)
    monkeypatch.setattr(cold_owner_local, "native_cold_owner_local", lambda: owner_fe)
    monkeypatch.setattr(cold_fe_fv, "_owner_local_fv_continuation", lambda: owner_fv)

    def child(phase: str, archive: Path) -> dict[str, object]:
        if phase == "write":
            archive.write_bytes(b"archive")
            return {
                "pid": 1,
                "runtime_id": "r",
                "epoch_id": "e",
                "state_id": "s",
                "cell_ids": [1009],
            }
        return {
            "pid": 2,
            "phase": "read",
            "runtime_id": "r",
            "epoch_id": "e",
            "restored_state_id": "s",
            "source_cell_ids": [1009],
            "accepted": failed != "serial_fv",
            "positivity_valid": True,
        }

    monkeypatch.setattr(cold_fe_fv, "_run_child", child)
    result = cold_fe_fv.native_cold_fe_fv_continuation()
    assert result["status"] == ("passed" if failed is None else "failed")
    assert bool(result["component_failures"]) == (failed is not None)
