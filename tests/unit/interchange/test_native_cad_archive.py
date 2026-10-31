from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from phydrax._array_archive import ArrayArchiveCorruptionError
from phydrax._meshcore import meshcore_available
from phydrax._model._structure import (
    model_from_structure_recipe,
    model_recipe_array_inventory,
    model_structure_recipe,
)
from phydrax._physical import SpatialCoordinateContract
from phydrax.exterior import FormType, FormValueSpec
from phydrax.geometry.brep._constructors import brep_sphere
from phydrax.geometry.brep._model import BRepEntityId
from phydrax.interchange._cad_archive import (
    _reconstruct_brep_graph,
    load_brep_archive,
    save_brep_archive,
)
from phydrax.meshing._model_codec import (
    _register_model_archive_types,
    MODEL_ARCHIVE_LIMITS,
    read_model_archive,
    write_model_archive,
)


_WORKER = Path(__file__).with_name("_native_cad_archive_worker.py")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_form_values_round_trip_through_registered_structure_recipes() -> None:
    _register_model_archive_types()
    form_type = FormType(3, 1, twist="twisted", fiber_shape=(2,), ambient_dimension=4)
    form_value = FormValueSpec(form_type, proxy="circulation")
    cases = (
        (
            form_type,
            "phydrax.exterior:FormType",
            "form_type_id",
            {
                "dimension",
                "degree",
                "twist",
                "fiber_shape",
                "ambient_dimension",
                "form_type_id",
            },
        ),
        (
            form_value,
            "phydrax.exterior:FormValueSpec",
            "value_spec_id",
            {"form_type", "proxy", "value_spec_id"},
        ),
    )
    for value, type_id, identity_field, fields in cases:
        recipe = model_structure_recipe(value)
        assert recipe["kind"] == "strict_value"
        assert recipe["type"] == type_id
        assert set(recipe["fields"]) == fields
        restored = model_from_structure_recipe(recipe, limits=None)
        assert type(restored) is type(value)
        assert restored == value

        tampered = copy.deepcopy(recipe)
        identity = tampered["fields"][identity_field]["value"]
        tampered["fields"][identity_field]["value"] = (
            "0" * 64 if identity != "0" * 64 else "1" * 64
        )
        with pytest.raises(ValueError, match="identity does not reproduce"):
            model_from_structure_recipe(tampered, limits=None)
    from phydrax.discretization.fem import lagrange_element

    element = lagrange_element("interval", 1)
    projection_recipe = model_structure_recipe(element)
    assert projection_recipe["kind"] == "finite_element_spec"
    projection = model_from_structure_recipe(projection_recipe, limits=None)
    restored_element = projection.rebuild()
    assert type(restored_element) is type(element)
    assert restored_element.element_id == element.element_id
    assert restored_element.tabulator_id == element.tabulator_id
    np.testing.assert_array_equal(
        np.asarray(restored_element.reference_nodes), np.asarray(element.reference_nodes)
    )

    high_order = lagrange_element("triangle", 3)
    high_order_recipe = model_structure_recipe(high_order)
    high_order_projection = model_from_structure_recipe(high_order_recipe, limits=None)
    restored_high_order = high_order_projection.rebuild()
    assert restored_high_order.element_id == high_order.element_id
    assert restored_high_order.tabulator_id == high_order.tabulator_id
    np.testing.assert_array_equal(
        np.asarray(restored_high_order.reference_nodes),
        np.asarray(high_order.reference_nodes),
    )
    from phydrax.discretization.fem import form_element

    form = form_element("triangle", 1, 1, proxy="circulation")
    form_recipe = model_structure_recipe(form)
    assert form_recipe["kind"] == "finite_element_spec"
    restored_form = model_from_structure_recipe(form_recipe, limits=None).rebuild()
    assert type(restored_form) is type(form)
    assert restored_form.element_id == form.element_id
    assert restored_form.tabulator_id == form.tabulator_id
    form_basis = form.form_basis
    restored_basis = restored_form.form_basis
    assert form_basis is not None and restored_basis is not None
    assert restored_basis.basis_id == form_basis.basis_id
    assert restored_basis.exact_generators == form_basis.exact_generators
    assert restored_basis.exact_generator_labels == form_basis.exact_generator_labels
    assert restored_basis.exact_coefficients == form_basis.exact_coefficients
    assert restored_basis.exact_functional_moments == form_basis.exact_functional_moments
    assert restored_basis.coefficient_projection_errors == (
        form_basis.coefficient_projection_errors
    )
    assert restored_basis.functional_moment_projection_errors == (
        form_basis.functional_moment_projection_errors
    )
    vertices = (0, 1, 2)
    assert restored_basis.exact_permutation_matrix(vertices) == (
        form_basis.exact_permutation_matrix(vertices)
    )
    global_vertices = (11, 29, 47)
    assert restored_basis.exact_entity_permutation_matrix(global_vertices) == (
        form_basis.exact_entity_permutation_matrix(global_vertices)
    )


def test_brep_archive_cold_restores_source_and_exact_spline_in_new_process(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [sys.executable, str(_WORKER), "write", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    restored = subprocess.run(
        [sys.executable, str(_WORKER), "read", str(tmp_path), json.dumps(expected)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert restored.stdout.strip() == "cold-restored"


def test_brep_projection_result_archive_recomputes_rows_cold(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [sys.executable, str(_WORKER), "write-brep-projection", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-brep-projection",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert restored.stdout.strip() == "brep-projection-cold-restored"


def test_brep_projection_result_archive_rejects_tampered_frame_binding(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [sys.executable, str(_WORKER), "write-brep-projection", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    manifest, arrays = read_model_archive(
        tmp_path / "brep-projection.phx", limits=MODEL_ARCHIVE_LIMITS
    )

    def find_projection(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if node.get("kind") == "dataclass" and str(node.get("type", "")).endswith(
                ":BRepProjectionResult"
            ):
                return node
            for child in node.values():
                found = find_projection(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_projection(child)
                if found is not None:
                    return found
        return None

    projection = find_projection(manifest["model_recipe"])
    assert projection is not None
    frame = projection["fields"]["frame_id"]["value"]
    projection["fields"]["frame_id"]["value"] = (
        "tampered-frame" if frame != "tampered-frame" else "changed-frame"
    )
    corrupt_directory = tmp_path / "changed-projection-frame"
    corrupt_directory.mkdir()
    write_model_archive(
        corrupt_directory / "brep-projection.phx",
        manifest=manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-brep-projection",
            str(corrupt_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_brep_archive_rejects_tampered_tessellation_chart_tokens(
    tmp_path: Path,
) -> None:
    subprocess.run(
        [sys.executable, str(_WORKER), "write", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    source = tmp_path / "tessellated-box.phx"
    corrupted = tmp_path / "tampered-tessellation-chart-tokens.phx"
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    def find_surface_witness(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if node.get("kind") == "dataclass" and str(node.get("type", "")).endswith(
                ":SurfaceChartWitness"
            ):
                return node
            for child in node.values():
                found = find_surface_witness(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_surface_witness(child)
                if found is not None:
                    return found
        return None

    witness = find_surface_witness(manifest["model_recipe"])
    assert witness is not None
    token_bank = witness["fields"]["chart_rational_tokens"]

    def tamper_first_integer(node: object) -> bool:
        if isinstance(node, dict):
            if node.get("kind") == "literal" and type(node.get("value")) is int:
                node["value"] += 1
                return True
            return any(tamper_first_integer(child) for child in node.values())
        if isinstance(node, list):
            return any(tamper_first_integer(child) for child in node)
        return False

    assert tamper_first_integer(token_bank)
    write_model_archive(
        corrupted, manifest=manifest, arrays=arrays, limits=MODEL_ARCHIVE_LIMITS
    )
    refused = subprocess.run(
        [sys.executable, str(_WORKER), "read-brep", str(corrupted)],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_brep_archive_recomputes_serialized_identity(tmp_path: Path) -> None:
    model = brep_sphere(1.0, coordinate_contract=SpatialCoordinateContract.si())
    source = tmp_path / "sphere.phx"
    corrupted = tmp_path / "changed-identity.phx"
    save_brep_archive(model, source)

    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    root_recipe = manifest["model_recipe"]
    identity = root_recipe["fields"]["model_id"]["value"]
    root_recipe["fields"]["model_id"]["value"] = (
        "0" * 64 if identity != "0" * 64 else "1" * 64
    )
    write_model_archive(
        corrupted, manifest=manifest, arrays=arrays, limits=MODEL_ARCHIVE_LIMITS
    )

    with pytest.raises(ArrayArchiveCorruptionError):
        load_brep_archive(corrupted)


def test_composition_archive_recomputes_identity_in_a_cold_process(
    tmp_path: Path,
) -> None:
    subprocess.run(
        [sys.executable, str(_WORKER), "write", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    source = tmp_path / "continuation.phx"
    corrupted = tmp_path / "changed-composition-identity.phx"
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    recipe = manifest["model_recipe"]
    identity = recipe["fields"]["composition_id"]["value"]
    recipe["fields"]["composition_id"]["value"] = (
        "0" * 64 if identity != "0" * 64 else "1" * 64
    )
    write_model_archive(
        corrupted, manifest=manifest, arrays=arrays, limits=MODEL_ARCHIVE_LIMITS
    )

    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-continuation",
            str(corrupted),
            "{}",
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_polyhedral_periodic_archive_cold_restores_ragged_connectivity(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "write-polyhedral-periodic",
            str(tmp_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    source = tmp_path / "periodic-polyhedral.phx"
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-polyhedral-periodic",
            str(source),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert restored.stdout.strip() == "polyhedral-periodic-cold-restored"

    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    def find_periodic_topology(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") == "dataclass"
                and str(node.get("type", "")).rsplit(":", 1)[-1] == "PeriodicMeshTopology"
            ):
                return node
            for child in node.values():
                found = find_periodic_topology(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_periodic_topology(child)
                if found is not None:
                    return found
        return None

    periodic = find_periodic_topology(manifest["model_recipe"])
    assert periodic is not None
    periodic["fields"]["periodic_topology_id"]["value"] = "0" * 64
    corrupted = tmp_path / "changed-periodic-polyhedral-topology.phx"
    write_model_archive(
        corrupted, manifest=manifest, arrays=arrays, limits=MODEL_ARCHIVE_LIMITS
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-polyhedral-periodic",
            str(corrupted),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_composition_archive_refuses_tampered_fe_layout_identity(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [sys.executable, str(_WORKER), "write", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)["continuation"]
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-continuation",
            str(tmp_path / "continuation.phx"),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert restored.returncode == 0, restored.stderr
    source = tmp_path / "continuation.phx"
    corrupted = tmp_path / "changed-fe-layout-identity.phx"
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    original_manifest = copy.deepcopy(manifest)

    def find_layout(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") == "dataclass"
                and str(node.get("type", "")).rsplit(":", 1)[-1]
                == "FiniteElementAcceptedFieldLayoutRecipe"
            ):
                return node
            for child in node.values():
                found = find_layout(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_layout(child)
                if found is not None:
                    return found
        return None

    layout = find_layout(manifest["model_recipe"])
    assert layout is not None
    layout["fields"]["layout_id"]["value"] = "0" * 64
    write_model_archive(
        corrupted, manifest=manifest, arrays=arrays, limits=MODEL_ARCHIVE_LIMITS
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-continuation",
            str(corrupted),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr

    restart_source = tmp_path / "finite-element-restart.phx"
    restart_manifest, restart_arrays = read_model_archive(restart_source, limits=None)

    def find_restart_nodes(node: object, type_name: str, /) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        if isinstance(node, dict):
            if (
                node.get("kind") in ("dataclass", "strict_value", "namedtuple")
                and str(node.get("type", "")).rsplit(":", 1)[-1] == type_name
            ):
                found.append(node)
            for child in node.values():
                found.extend(find_restart_nodes(child, type_name))
        elif isinstance(node, list):
            for child in node:
                found.extend(find_restart_nodes(child, type_name))
        return found

    def refuse_restart_tamper(
        manifest_: dict[str, Any], directory_name: str, /
    ) -> subprocess.CompletedProcess[str]:
        directory = tmp_path / directory_name
        directory.mkdir()
        write_model_archive(
            directory / "finite-element-restart.phx",
            manifest=manifest_,
            arrays={
                name: np.array(array, copy=True) for name, array in restart_arrays.items()
            },
            limits=None,
        )
        shutil.copyfile(source, directory / "continuation.phx")
        return subprocess.run(
            [
                sys.executable,
                str(_WORKER),
                "read-continuation",
                str(directory / "continuation.phx"),
                json.dumps(expected),
            ],
            cwd=_REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    ownership_manifest = copy.deepcopy(restart_manifest)
    ownership_node = find_restart_nodes(
        ownership_manifest["model_recipe"], "FiniteElementDofOwnership"
    )
    assert ownership_node
    ownership_id = ownership_node[0]["fields"]["ownership_id"]
    assert ownership_id["kind"] == "literal"
    stored_ownership_id = ownership_id["value"]
    ownership_id["value"] = "0" * 64 if stored_ownership_id != "0" * 64 else "1" * 64
    refused_ownership = refuse_restart_tamper(
        ownership_manifest, "changed-fe-restart-ownership"
    )
    assert refused_ownership.returncode != 0
    assert "incompatible with its registered recipe" in refused_ownership.stderr

    rng_manifest = copy.deepcopy(restart_manifest)
    rng_layout = next(
        (
            node
            for node in find_restart_nodes(
                rng_manifest["model_recipe"], "FiniteElementRestartStateLayout"
            )
            if node["fields"]["name"]["value"] == "rng"
        ),
        None,
    )
    assert rng_layout is not None
    prng_impl = rng_layout["fields"]["prng_impl"]
    assert prng_impl["kind"] == "literal"
    prng_impl["value"] = "changed-prng-implementation"
    refused_rng = refuse_restart_tamper(rng_manifest, "changed-fe-restart-prng-role")
    assert refused_rng.returncode != 0
    assert "incompatible with its registered recipe" in refused_rng.stderr

    semantic_manifest = copy.deepcopy(original_manifest)
    semantic_layout = find_layout(semantic_manifest["model_recipe"])
    assert semantic_layout is not None
    stored_layout_id = semantic_layout["fields"]["layout_id"]["value"]
    semantics = semantic_layout["fields"]["transfer_semantics"]
    assert semantics["kind"] == "enum"
    assert semantics["name"] == "NONCONSERVATIVE_INTERPOLATION"
    semantics["name"] = "MATERIAL_FOLLOWING_CONTENT"
    assert semantic_layout["fields"]["layout_id"]["value"] == stored_layout_id
    semantic_directory = tmp_path / "changed-fe-layout-transfer-semantics"
    semantic_directory.mkdir()
    semantic_archive = semantic_directory / "continuation.phx"
    write_model_archive(
        semantic_archive,
        manifest=semantic_manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    refused_semantics = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-continuation",
            str(semantic_archive),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_semantics.returncode != 0
    assert "incompatible with its registered recipe" in refused_semantics.stderr


def test_composition_archive_rejects_tampered_material_deformation_identity(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [sys.executable, str(_WORKER), "write", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)["continuation"]
    source = tmp_path / "continuation.phx"
    corrupted = tmp_path / "changed-material-deformation-identity.phx"
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    def find_deformation(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") == "dataclass"
                and str(node.get("type", "")).rsplit(":", 1)[-1]
                == "VolumeMaterialDeformation"
            ):
                return node
            for child in node.values():
                found = find_deformation(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_deformation(child)
                if found is not None:
                    return found
        return None

    deformation = find_deformation(manifest["model_recipe"])
    assert deformation is not None
    deformation["fields"]["deformation_id"]["value"] = "0" * 64
    write_model_archive(
        corrupted, manifest=manifest, arrays=arrays, limits=MODEL_ARCHIVE_LIMITS
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-continuation",
            str(corrupted),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    for field_name in ("source_chart_witnesses", "target_chart_witnesses"):
        manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

        def find_material_witness(node: object) -> dict[str, Any] | None:
            if isinstance(node, dict):
                if (
                    node.get("kind") == "dataclass"
                    and str(node.get("type", "")).rsplit(":", 1)[-1]
                    == "MaterialReferenceVolumeWitness"
                ):
                    return node
                for child in node.values():
                    found = find_material_witness(child)
                    if found is not None:
                        return found
            elif isinstance(node, list):
                for child in node:
                    found = find_material_witness(child)
                    if found is not None:
                        return found
            return None

        def tamper_first_chart_string(node: object) -> bool:
            if isinstance(node, dict):
                if node.get("kind") == "literal" and isinstance(node.get("value"), str):
                    node["value"] = "tampered-material-chart-witness"
                    return True
                return any(tamper_first_chart_string(child) for child in node.values())
            if isinstance(node, list):
                return any(tamper_first_chart_string(child) for child in node)
            return False

        witness = find_material_witness(manifest["model_recipe"])
        assert witness is not None
        fields = witness.get("fields")
        assert isinstance(fields, dict)
        assert tamper_first_chart_string(fields[field_name])
        chart_corruption = tmp_path / f"changed-material-{field_name}.phx"
        write_model_archive(
            chart_corruption,
            manifest=manifest,
            arrays=arrays,
            limits=MODEL_ARCHIVE_LIMITS,
        )
        refused_chart = subprocess.run(
            [
                sys.executable,
                str(_WORKER),
                "read-continuation",
                str(chart_corruption),
                json.dumps(expected),
            ],
            cwd=_REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert refused_chart.returncode != 0


def test_layer_curving_source_evidence_archive_revalidates_in_cold_process(
    tmp_path: Path,
) -> None:
    if not meshcore_available():
        pytest.skip("native boundary-layer generation requires meshcore")
    written = subprocess.run(
        [sys.executable, str(_WORKER), "write-layer-curving-evidence", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    assert expected["trace_id"] is None
    assert expected["chart_witness_id"]
    assert expected["fidelity_certificate_id"]
    source = tmp_path / "layer-curving-source-evidence.phx"
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-layer-curving-evidence",
            str(source),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert restored.stdout.strip() == "layer-curving-evidence-cold-restored"

    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    def find_source_chart_witness(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") == "dataclass"
                and str(node.get("type", "")).rsplit(":", 1)[-1]
                == "SourceBoundaryChartWitness"
            ):
                return node
            for child in node.values():
                found = find_source_chart_witness(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_source_chart_witness(child)
                if found is not None:
                    return found
        return None

    witness = find_source_chart_witness(manifest["model_recipe"])
    assert witness is not None
    witness["fields"]["witness_id"]["value"] = "0" * 64
    corrupted = tmp_path / "changed-layer-curving-trace.phx"
    write_model_archive(
        corrupted, manifest=manifest, arrays=arrays, limits=MODEL_ARCHIVE_LIMITS
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-layer-curving-evidence",
            str(corrupted),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    _register_model_archive_types()
    row_leaves = tuple(
        entry
        for entry in model_recipe_array_inventory(
            manifest["model_recipe"], prefix="model/leaves", limits=None
        )
        if "source_root_cell_ids" in entry.path and arrays[entry.name].size
    )
    assert row_leaves
    row_payload = arrays[row_leaves[0].name]
    assert row_payload.size
    row_payload[0] += 1
    changed_rows = tmp_path / "changed-layer-source-root-cell-ids.phx"
    write_model_archive(
        changed_rows, manifest=manifest, arrays=arrays, limits=MODEL_ARCHIVE_LIMITS
    )
    refused_rows = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-layer-curving-evidence",
            str(changed_rows),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_rows.returncode != 0
    assert "incompatible with its registered recipe" in refused_rows.stderr
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    witness = find_source_chart_witness(manifest["model_recipe"])
    assert witness is not None
    fields = witness["fields"]

    def tamper_first_integer(node: object) -> bool:
        if isinstance(node, dict):
            if node.get("kind") == "literal" and type(node.get("value")) is int:
                node["value"] += 1
                return True
            return any(tamper_first_integer(child) for child in node.values())
        if isinstance(node, list):
            return any(tamper_first_integer(child) for child in node)
        return False

    assert tamper_first_integer(fields["source_parameter_rational_tokens"])
    changed_reference = tmp_path / "changed-layer-source-rational-tokens.phx"
    write_model_archive(
        changed_reference, manifest=manifest, arrays=arrays, limits=MODEL_ARCHIVE_LIMITS
    )
    refused_reference = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-layer-curving-evidence",
            str(changed_reference),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_reference.returncode != 0
    assert "incompatible with its registered recipe" in refused_reference.stderr


def test_mapped_layer_curving_composition_rebuilds_source_trace_cold(
    tmp_path: Path,
) -> None:
    if not meshcore_available():
        pytest.skip("native mapped layer generation requires meshcore")
    written = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "write-mapped-layer-curving-evidence",
            str(tmp_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    assert expected["mapped_trace_id"]
    assert expected["mapped_reference_map_bindings"]
    assert expected["mapped_source_parameters"]
    source = tmp_path / "mapped-layer-curving-source-evidence.phx"
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-layer-curving-evidence",
            str(source),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert restored.stdout.strip() == "layer-curving-evidence-cold-restored"

    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    def find_mapped_trace(node: object) -> dict[object, object] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") == "dataclass"
                and str(node.get("type", "")).rsplit(":", 1)[-1]
                == "LayerCurvingMappedChartTrace"
            ):
                return node
            for child in node.values():
                found = find_mapped_trace(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_mapped_trace(child)
                if found is not None:
                    return found
        return None

    def tamper_first_fraction(node: object) -> bool:
        if isinstance(node, dict):
            if node.get("kind") == "fraction":
                numerator = node.get("numerator")
                if not isinstance(numerator, int):
                    return False
                node["numerator"] = numerator + 1
                return True
            return any(tamper_first_fraction(child) for child in node.values())
        if isinstance(node, list):
            return any(tamper_first_fraction(child) for child in node)
        return False

    trace = find_mapped_trace(manifest["model_recipe"])
    assert trace is not None
    fields = trace.get("fields")
    assert isinstance(fields, dict)
    assert tamper_first_fraction(fields["source_parameters"])
    corrupted = tmp_path / "changed-mapped-layer-source-parameters.phx"
    write_model_archive(
        corrupted, manifest=manifest, arrays=arrays, limits=MODEL_ARCHIVE_LIMITS
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-layer-curving-evidence",
            str(corrupted),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0


def test_restricted_layer_curving_archive_replays_exact_reference_points(
    tmp_path: Path,
) -> None:
    if not meshcore_available():
        pytest.skip("native boundary-layer generation requires meshcore")
    written = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "write-restricted-layer-curving-evidence",
            str(tmp_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    assert expected["target_reference_points"]
    assert np.all(np.asarray(expected["geometry_node_rows"]) == -1)
    source = tmp_path / "restricted-layer-curving-source-evidence.phx"
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-restricted-layer-curving-evidence",
            str(source),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert restored.stdout.strip() == "layer-curving-evidence-cold-restored"


def test_implicit_layer_curving_archive_revalidates_primitive_chart_evidence_cold(
    tmp_path: Path,
) -> None:
    if not meshcore_available():
        pytest.skip("native boundary-layer generation requires meshcore")
    written = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "write-implicit-layer-curving-evidence",
            str(tmp_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    source = tmp_path / "implicit-layer-curving-source-evidence.phx"
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-implicit-layer-curving-evidence",
            str(source),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert restored.stdout.strip() == "layer-curving-evidence-cold-restored"

    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    def find_implicit_trace(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") == "dataclass"
                and str(node.get("type", "")).rsplit(":", 1)[-1]
                == "LayerCurvingImplicitRootTrace"
            ):
                return node
            for child in node.values():
                found = find_implicit_trace(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_implicit_trace(child)
                if found is not None:
                    return found
        return None

    trace = find_implicit_trace(manifest["model_recipe"])
    assert trace is not None
    chart_rows = trace["fields"]["chart_evidence"]["items"]
    assert chart_rows and chart_rows[0]["kind"] == "mapping"
    primitive_fields = {pair[0]["value"]: pair[1] for pair in chart_rows[0]["items"]}
    lower_bound = primitive_fields["target_to_root_lower_bound"]
    assert lower_bound["kind"] == "literal"
    lower_bound["value"] += 0.125
    corrupted = tmp_path / "changed-implicit-layer-chart-evidence.phx"
    write_model_archive(
        corrupted, manifest=manifest, arrays=arrays, limits=MODEL_ARCHIVE_LIMITS
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-implicit-layer-curving-evidence",
            str(corrupted),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_seed_transition_preserves_material_reference_without_maps(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "write-material-reference-seed-transition",
            str(tmp_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    source = tmp_path / "material-reference-seed-transition.phx"
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-material-reference-seed-transition",
            str(source),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert restored.stdout.strip() == "material-reference-seed-transition-cold-restored"

    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    def find_reference(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") == "dataclass"
                and str(node.get("type", "")).rsplit(":", 1)[-1]
                == "MaterialReferenceDeclaration"
            ):
                return node
            for child in node.values():
                found = find_reference(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_reference(child)
                if found is not None:
                    return found
        return None

    reference = find_reference(manifest["model_recipe"])
    assert reference is not None
    reference["fields"]["declaration_id"]["value"] = "0" * 64
    corrupted = tmp_path / "changed-material-reference-seed-transition.phx"
    write_model_archive(
        corrupted, manifest=manifest, arrays=arrays, limits=MODEL_ARCHIVE_LIMITS
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-material-reference-seed-transition",
            str(corrupted),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0


def test_mapped_reference_domain_archive_replays_root_vertex_relation(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "write-mapped-reference-domain",
            str(tmp_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    assert expected["reference_domain_region_occurrence_paths"] == [[]]
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-mapped-reference-domain",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert restored.stdout.strip() == "mapped-reference-domain-cold-restored"

    source = tmp_path / "mapped-reference-domain.phx"
    corrupted_directory = tmp_path / "changed-reference-root-vertex-relation"
    corrupted_directory.mkdir()
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    def find_mapped_domain(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") == "dataclass"
                and str(node.get("type", "")).rsplit(":", 1)[-1]
                == "MappedReferenceDomain"
            ):
                return node
            for child in node.values():
                found = find_mapped_domain(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_mapped_domain(child)
                if found is not None:
                    return found
        return None

    mapped_domain = find_mapped_domain(manifest["model_recipe"])
    assert mapped_domain is not None
    mapped_domain["fields"]["reference_domain_vertex_root_ids"]["items"][0]["value"] = 999
    write_model_archive(
        corrupted_directory / "mapped-reference-domain.phx",
        manifest=manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-mapped-reference-domain",
            str(corrupted_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr

    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    mapped_domain = find_mapped_domain(manifest["model_recipe"])
    assert mapped_domain is not None
    coordinate_contract = mapped_domain["fields"]["coordinate_contract"]
    assert (
        coordinate_contract["kind"] == "dataclass"
        and str(coordinate_contract["type"]).rsplit(":", 1)[-1]
        == "SpatialCoordinateContract"
    )
    stored_spatial_id = coordinate_contract["fields"]["spatial_id"]["value"]
    coordinate_contract["fields"]["spatial_id"]["value"] = (
        "0" * 64 if stored_spatial_id != "0" * 64 else "1" * 64
    )
    spatial_directory = tmp_path / "changed-mapped-domain-spatial-contract"
    spatial_directory.mkdir()
    write_model_archive(
        spatial_directory / "mapped-reference-domain.phx",
        manifest=manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    refused_spatial = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-mapped-reference-domain",
            str(spatial_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_spatial.returncode != 0
    assert "incompatible with its registered recipe" in refused_spatial.stderr
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    def find_piecewise_domain(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if node.get("kind") == "dataclass" and str(node.get("type", "")).endswith(
                ":PiecewiseLinearDomain"
            ):
                return node
            for child in node.values():
                found = find_piecewise_domain(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_piecewise_domain(child)
                if found is not None:
                    return found
        return None

    reference_domain = find_piecewise_domain(manifest["model_recipe"])
    assert reference_domain is not None
    paths = reference_domain["fields"]["region_occurrence_paths"]
    assert paths["kind"] == "tuple"
    assert paths["items"][0]["kind"] == "tuple"
    paths["items"][0]["items"] = [{"kind": "literal", "value": "wrong-region-occurrence"}]
    path_directory = tmp_path / "changed-reference-region-occurrence"
    path_directory.mkdir()
    write_model_archive(
        path_directory / "mapped-reference-domain.phx",
        manifest=manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    refused_path = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-mapped-reference-domain",
            str(path_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_path.returncode != 0


def test_piecewise_linear_domain_archives_duplicate_region_occurrences_cold(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "write-piecewise-region-occurrences",
            str(tmp_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    assert expected["region_ids"] == ["solid", "solid"]
    assert expected["region_occurrence_paths"] == [
        ["assembly", "left"],
        ["assembly", "right"],
    ]
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-piecewise-region-occurrences",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert restored.stdout.strip() == "piecewise-region-occurrences-cold-restored"

    source = tmp_path / "piecewise-region-occurrences.phx"
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    def find_piecewise_domain(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if node.get("kind") == "dataclass" and str(node.get("type", "")).endswith(
                ":PiecewiseLinearDomain"
            ):
                return node
            for child in node.values():
                found = find_piecewise_domain(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_piecewise_domain(child)
                if found is not None:
                    return found
        return None

    domain_recipe = find_piecewise_domain(manifest["model_recipe"])
    assert domain_recipe is not None
    path_bank = domain_recipe["fields"]["region_occurrence_paths"]
    assert path_bank["kind"] == "tuple"
    path_bank["items"][0]["items"][1]["value"] = "changed-occurrence"
    corrupt_directory = tmp_path / "changed-piecewise-region-occurrence"
    corrupt_directory.mkdir()
    write_model_archive(
        corrupt_directory / "piecewise-region-occurrences.phx",
        manifest=manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-piecewise-region-occurrences",
            str(corrupt_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0


def test_mapped_reference_boundary_query_archive_rebuilds_root_authority_cold(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "write-mapped-reference-boundary-query",
            str(tmp_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    source = tmp_path / "mapped-reference-boundary-query.phx"
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-mapped-reference-boundary-query",
            str(source),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert restored.stdout.strip() == "mapped-reference-boundary-query-cold-restored"

    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    def find_query_recipe(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") == "namedtuple"
                and str(node.get("type", "")).rsplit(":", 1)[-1]
                == "MappedReferenceBoundarySourceArchiveRecipe"
            ):
                return node
            for child in node.values():
                found = find_query_recipe(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_query_recipe(child)
                if found is not None:
                    return found
        return None

    recipe = find_query_recipe(manifest["model_recipe"])
    assert recipe is not None
    stored_query_id = recipe["items"][3]["value"]
    recipe["items"][3]["value"] = "0" * 64 if stored_query_id != "0" * 64 else "1" * 64
    corrupted = tmp_path / "changed-mapped-reference-boundary-query.phx"
    write_model_archive(
        corrupted, manifest=manifest, arrays=arrays, limits=MODEL_ARCHIVE_LIMITS
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-mapped-reference-boundary-query",
            str(corrupted),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    def find_authority_recipe(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") == "namedtuple"
                and str(node.get("type", "")).rsplit(":", 1)[-1]
                == "NativeMappedHexSourceArchiveRecipe"
            ):
                return node
            for child in node.values():
                found = find_authority_recipe(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_authority_recipe(child)
                if found is not None:
                    return found
        return None

    authority = find_authority_recipe(manifest["model_recipe"])
    assert authority is not None
    stored_relation_id = authority["items"][6]["value"]
    authority["items"][6]["value"] = (
        "0" * 64 if stored_relation_id != "0" * 64 else "1" * 64
    )
    changed_relation = tmp_path / "changed-mapped-reference-root-relation.phx"
    write_model_archive(
        changed_relation, manifest=manifest, arrays=arrays, limits=MODEL_ARCHIVE_LIMITS
    )
    refused_relation = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-mapped-reference-boundary-query",
            str(changed_relation),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_relation.returncode != 0
    assert "incompatible with its registered recipe" in refused_relation.stderr


def test_brep_archive_revalidates_occurrence_entity_membership_cold(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [sys.executable, str(_WORKER), "write-brep-entity", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-brep-entity",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert restored.stdout.strip() == "qualified-brep-entity-cold-restored"

    source = tmp_path / "qualified-brep-entity.phx"
    corrupted_directory = tmp_path / "changed-solid-face-membership"
    corrupted_directory.mkdir()
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    def find_entity(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") == "dataclass"
                and str(node.get("type", "")).rsplit(":", 1)[-1] == "BRepEntityId"
            ):
                return node
            for child in node.values():
                found = find_entity(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_entity(child)
                if found is not None:
                    return found
        return None

    entity = find_entity(manifest["model_recipe"])
    assert entity is not None
    entity["fields"]["index"]["value"] = 1
    write_model_archive(
        corrupted_directory / "qualified-brep-entity.phx",
        manifest=manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-brep-entity",
            str(corrupted_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_composition_archive_rejects_payload_checksum_corruption(
    tmp_path: Path,
) -> None:
    subprocess.run(
        [sys.executable, str(_WORKER), "write", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    source = tmp_path / "continuation.phx"
    corrupted = tmp_path / "corrupt-continuation.phx"
    with zipfile.ZipFile(source, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        members = {name: archive.read(name) for name in archive.namelist()}
    member_name = next(iter(manifest["arrays"].values()))["member"]
    payload = members[member_name]
    members[member_name] = payload[:-1] + bytes((payload[-1] ^ 1,))
    with zipfile.ZipFile(corrupted, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, member_payload in members.items():
            archive.writestr(name, member_payload)

    refused = subprocess.run(
        [sys.executable, str(_WORKER), "read-continuation", str(corrupted), "{}"],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "checksum failed" in refused.stderr.lower()


def test_t_spline_archive_recomputes_basis_certificate_in_a_cold_process(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "write-tspline-cold-archive",
            str(tmp_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert written.returncode == 0, written.stderr
    expected = json.loads(written.stdout)
    source = tmp_path / "refined.phx"
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-tspline-cold-archive",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert restored.returncode == 0, restored.stderr
    assert restored.stdout.strip() == "tspline-cold-restored"
    corrupted = tmp_path / "changed-t-spline-certificate.phx"
    manifest, arrays = read_model_archive(source, limits=None)

    original_manifest = copy.deepcopy(manifest)
    patch = manifest["model_recipe"]["fields"]["patches"]["items"][0]
    certificate = patch["fields"]["source_basis_certificate"]["fields"]
    identity = certificate["certificate_id"]["value"]
    certificate["certificate_id"]["value"] = (
        "0" * 64 if identity != "0" * 64 else "1" * 64
    )
    write_model_archive(corrupted, manifest=manifest, arrays=arrays, limits=None)
    altered_piece = copy.deepcopy(original_manifest)
    piece_fields = altered_piece["model_recipe"]["fields"]["patches"]["items"][0][
        "fields"
    ]["realization"]["fields"]["extractions"]["items"][0]["fields"]
    piece_fields["piece_id"]["value"] += 1
    piece_fields["rank"]["value"] += 1
    piece_corrupted = tmp_path / "changed-t-spline-extraction-piece-and-status.phx"
    write_model_archive(
        piece_corrupted, manifest=altered_piece, arrays=arrays, limits=None
    )
    refused_piece = subprocess.run(
        [sys.executable, str(_WORKER), "read-brep", str(piece_corrupted)],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_piece.returncode != 0
    assert "incompatible with its registered recipe" in refused_piece.stderr
    altered_mesh = copy.deepcopy(original_manifest)
    mesh_fields = altered_mesh["model_recipe"]["fields"]["patches"]["items"][0]["fields"][
        "mesh"
    ]["fields"]
    mesh_identity = mesh_fields["mesh_id"]["value"]
    mesh_fields["mesh_id"]["value"] = "0" * 64 if mesh_identity != "0" * 64 else "1" * 64
    mesh_corrupted = tmp_path / "changed-t-spline-mesh-identity.phx"
    write_model_archive(mesh_corrupted, manifest=altered_mesh, arrays=arrays, limits=None)
    refused_mesh = subprocess.run(
        [sys.executable, str(_WORKER), "read-brep", str(mesh_corrupted)],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_mesh.returncode != 0
    assert "incompatible with its registered recipe" in refused_mesh.stderr

    altered_connectivity = copy.deepcopy(original_manifest)
    mesh_fields = altered_connectivity["model_recipe"]["fields"]["patches"]["items"][0][
        "fields"
    ]["mesh"]["fields"]
    cell = mesh_fields["cells"]["items"][0]["fields"]
    cell["vertex_ids"]["items"][0]["value"] += 1000
    connectivity_corrupted = tmp_path / "changed-t-spline-connectivity.phx"
    write_model_archive(
        connectivity_corrupted, manifest=altered_connectivity, arrays=arrays, limits=None
    )
    refused_connectivity = subprocess.run(
        [sys.executable, str(_WORKER), "read-brep", str(connectivity_corrupted)],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_connectivity.returncode != 0
    assert "incompatible with its registered recipe" in refused_connectivity.stderr

    def find_local_knots(node: object) -> object | None:
        if isinstance(node, dict):
            fields = node.get("fields")
            if isinstance(fields, dict) and "local_knots" in fields:
                return fields["local_knots"]
            for child in node.values():
                found = find_local_knots(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_local_knots(child)
                if found is not None:
                    return found
        return None

    def tamper_first_fraction(node: object) -> bool:
        if isinstance(node, dict):
            if node.get("kind") == "fraction" and type(node.get("numerator")) is int:
                node["numerator"] += 1
                return True
            return any(tamper_first_fraction(child) for child in node.values())
        if isinstance(node, list):
            return any(tamper_first_fraction(child) for child in node)
        return False

    altered_knots = copy.deepcopy(original_manifest)
    local_knots = find_local_knots(altered_knots["model_recipe"])
    assert local_knots is not None
    assert tamper_first_fraction(local_knots)
    knot_corrupted = tmp_path / "changed-t-spline-local-knots.phx"
    write_model_archive(
        knot_corrupted, manifest=altered_knots, arrays=arrays, limits=None
    )
    refused_knots = subprocess.run(
        [sys.executable, str(_WORKER), "read-brep", str(knot_corrupted)],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_knots.returncode != 0
    assert "incompatible with its registered recipe" in refused_knots.stderr

    refused = subprocess.run(
        [sys.executable, str(_WORKER), "read-brep", str(corrupted)],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_plane_cylinder_section_pcurve_archive_recomputes_exact_phase_cold(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "write-plane-cylinder-section-pcurve",
            str(tmp_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert written.returncode == 0, written.stderr
    expected = json.loads(written.stdout)
    source = tmp_path / "plane-cylinder-section-pcurve.phx"
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-plane-cylinder-section-pcurve",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert restored.returncode == 0, restored.stderr
    assert restored.stdout.strip() == "plane-cylinder-section-pcurve-cold-restored"

    manifest, arrays = read_model_archive(source, limits=None)

    recipe = manifest["model_recipe"]
    assert str(recipe["type"]).endswith(":PlaneCylinderSectionPCurve")
    original_manifest = copy.deepcopy(manifest)

    def find_pcurve(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") == "dataclass"
                and str(node.get("type", "")).rsplit(":", 1)[-1]
                == "PlaneCylinderSectionPCurve"
            ):
                return node
            for child in node.values():
                found = find_pcurve(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_pcurve(child)
                if found is not None:
                    return found
        return None

    def refuse(
        manifest: dict[str, Any], payloads: dict[str, np.ndarray], name: str
    ) -> None:
        directory = tmp_path / name
        directory.mkdir()
        archive = directory / "plane-cylinder-section-pcurve.phx"
        write_model_archive(archive, manifest=manifest, arrays=payloads, limits=None)
        refused = subprocess.run(
            [
                sys.executable,
                str(_WORKER),
                "read-plane-cylinder-section-pcurve",
                str(directory),
                json.dumps(expected),
            ],
            cwd=_REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert refused.returncode != 0
        assert "incompatible with its registered recipe" in refused.stderr

    exact_manifest = copy.deepcopy(original_manifest)
    exact_node = find_pcurve(exact_manifest["model_recipe"])
    assert exact_node is not None
    exact_node["fields"]["cosine_exact"]["items"][1]["numerator"] += 1
    refuse(exact_manifest, arrays, "changed-plane-cylinder-section-coefficient")

    phase_manifest = copy.deepcopy(original_manifest)
    phase_node = find_pcurve(phase_manifest["model_recipe"])
    assert phase_node is not None
    phase_node["fields"]["angle_offset_root"]["fields"]["turns"]["value"] += 1
    refuse(phase_manifest, arrays, "changed-plane-cylinder-section-root-phase")

    _register_model_archive_types()
    shadow_inventory = model_recipe_array_inventory(
        original_manifest["model_recipe"],
        prefix="model/leaves",
        limits=None,
    )
    angle_offset = next(
        entry for entry in shadow_inventory if entry.path.endswith("angle_offset")
    )
    changed_shadows = {name: payload.copy() for name, payload in arrays.items()}
    changed_shadows[angle_offset.name].flat[0] += 1.0
    refuse(
        copy.deepcopy(original_manifest),
        changed_shadows,
        "changed-plane-cylinder-section-angle-shadow",
    )


def test_brep_curve_surface_lift_archive_recomputes_chart_bounds_cold(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "write-brep-curve-surface-lift",
            str(tmp_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert written.returncode == 0, written.stderr
    expected = json.loads(written.stdout)
    source = tmp_path / "brep-curve-surface-lift.phx"
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-brep-curve-surface-lift",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert restored.returncode == 0, restored.stderr
    assert restored.stdout.strip() == "brep-curve-surface-lift-cold-restored"

    manifest, arrays = read_model_archive(source, limits=None)

    lift = manifest["model_recipe"]["fields"]["parameter_box"]
    assert [entry["value"] for entry in lift["items"][0]["items"]] == [0.25, 0.0]
    original_manifest = copy.deepcopy(manifest)
    changed_manifest = copy.deepcopy(original_manifest)
    changed_lift = changed_manifest["model_recipe"]["fields"]["parameter_box"]
    changed_lift["items"][0]["items"][0]["value"] = 0.3
    corrupted_directory = tmp_path / "changed-brep-lift-chart-bounds"
    corrupted_directory.mkdir()
    corrupted = corrupted_directory / "brep-curve-surface-lift.phx"
    write_model_archive(corrupted, manifest=changed_manifest, arrays=arrays, limits=None)
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-brep-curve-surface-lift",
            str(corrupted_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_sphere_chamfer_archive_cold_restores_exact_isoparametric_rail(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "write-sphere-chamfer-isoparametric",
            str(tmp_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert written.returncode == 0, written.stderr
    expected = json.loads(written.stdout)
    source = tmp_path / "sphere-chamfer.phx"
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-sphere-chamfer-isoparametric",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert restored.returncode == 0, restored.stderr
    assert restored.stdout.strip() == "sphere-chamfer-isoparametric-cold-restored"

    manifest, arrays = read_model_archive(source, limits=None)

    def find_exact_isoparametric_curve(node: object, /) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") == "dataclass"
                and str(node.get("type", "")).rsplit(":", 1)[-1]
                == "SurfaceIsoparametricCurve"
            ):
                exact = node["fields"]["fixed_value_exact"]
                if not (exact.get("kind") == "literal" and exact.get("value") is None):
                    return node
            for child in node.values():
                found = find_exact_isoparametric_curve(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_exact_isoparametric_curve(child)
                if found is not None:
                    return found
        return None

    curve = find_exact_isoparametric_curve(manifest["model_recipe"])
    assert curve is not None
    tampered = copy.deepcopy(manifest)
    tampered_curve = find_exact_isoparametric_curve(tampered["model_recipe"])
    assert tampered_curve is not None
    exact = tampered_curve["fields"]["fixed_value_exact"]
    if exact["kind"] == "fraction":
        exact["numerator"] += exact["denominator"]
    elif exact["kind"] == "radical":
        terms = exact["terms"]
        assert terms
        terms[0][1] += terms[0][2]
        if terms[0][1] == 0:
            terms[0][1] += terms[0][2]
    else:
        raise ValueError("Sphere chamfer archive has an unsupported exact rail scalar.")
    corrupted_directory = tmp_path / "changed-sphere-chamfer-exact-rail"
    corrupted_directory.mkdir()
    corrupted = corrupted_directory / "sphere-chamfer.phx"
    write_model_archive(corrupted, manifest=tampered, arrays=arrays, limits=None)
    refused = subprocess.run(
        [sys.executable, str(_WORKER), "read-brep", str(corrupted)],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_parallel_cylinder_chamfer_archive_cold_restores_witness_and_refuses_tamper(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "write-parallel-cylinder-chamfer",
            str(tmp_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert written.returncode == 0, written.stderr
    expected = json.loads(written.stdout)
    source = tmp_path / "parallel-cylinder-chamfer.phx"
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-parallel-cylinder-chamfer",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert restored.returncode == 0, restored.stderr
    assert restored.stdout.strip() == "parallel-cylinder-chamfer-cold-restored"

    manifest, arrays = read_model_archive(source, limits=None)

    def find_nodes(node: object, type_name: str, /) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        if isinstance(node, dict):
            if (
                node.get("kind") in ("dataclass", "strict_value", "namedtuple")
                and str(node.get("type", "")).rsplit(":", 1)[-1] == type_name
            ):
                found.append(node)
            for child in node.values():
                found.extend(find_nodes(child, type_name))
        elif isinstance(node, list):
            for child in node:
                found.extend(find_nodes(child, type_name))
        return found

    witness_name = "ConicRootLineEndpointWitnessArchiveRecipe"
    witness_nodes = find_nodes(manifest["model_recipe"], witness_name)
    assert witness_nodes
    assert not find_nodes(manifest["model_recipe"], "ConicRootLineEndpointWitness")
    from phydrax.interchange._cad_archive import (
        ConicRootLineEndpointWitnessArchiveRecipe,
    )

    witness_fields = tuple(ConicRootLineEndpointWitnessArchiveRecipe._fields)
    assert set(witness_fields) == {
        "source_curve",
        "source_pcurve",
        "source_region",
        "parameter_exact",
        "source_interval",
        "trim_curve",
        "trim_interval",
    }
    assert all(
        node["kind"] == "namedtuple" and len(node["items"]) == len(witness_fields)
        for node in witness_nodes
    )

    def refuse(manifest_: dict[str, Any], name: str) -> subprocess.CompletedProcess[str]:
        directory = tmp_path / name
        directory.mkdir()
        write_model_archive(
            directory / "parallel-cylinder-chamfer.phx",
            manifest=manifest_,
            arrays={key: np.array(value, copy=True) for key, value in arrays.items()},
            limits=None,
        )
        return subprocess.run(
            [
                sys.executable,
                str(_WORKER),
                "read-parallel-cylinder-chamfer",
                str(directory),
                json.dumps(expected),
            ],
            cwd=_REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    witness_manifest = copy.deepcopy(manifest)
    witness = find_nodes(witness_manifest["model_recipe"], witness_name)[0]
    parameter_index = witness_fields.index("parameter_exact")
    parameter = witness["items"][parameter_index]
    assert parameter["kind"] == "fraction"
    parameter["numerator"], parameter["denominator"] = 1, 2
    refused_witness = refuse(witness_manifest, "changed-conic-endpoint-parameter")
    assert refused_witness.returncode != 0
    assert "incompatible with its registered recipe" in refused_witness.stderr

    root_names = ("TrimIntersectionRoot", "CurveSurfaceIntersectionRoot")

    def find_witness_roots(recipe: object, /) -> list[dict[str, Any]]:
        return [
            root
            for root_name in root_names
            for root in find_nodes(recipe, root_name)
            if root["fields"]["endpoint_witness"]["kind"] == "namedtuple"
            and str(root["fields"]["endpoint_witness"]["type"]).endswith(
                f":{witness_name}"
            )
        ]

    witness_roots = find_witness_roots(manifest["model_recipe"])
    assert witness_roots
    root_manifest = copy.deepcopy(manifest)
    witnessed_root = find_witness_roots(root_manifest["model_recipe"])[0]
    root_id = witnessed_root["fields"]["root_id"]
    assert root_id["kind"] == "literal"
    root_id["value"] = "0" * 64 if root_id["value"] != "0" * 64 else "1" * 64
    refused_root = refuse(root_manifest, "changed-conic-endpoint-root-identity")
    assert refused_root.returncode != 0
    assert "root identity" in refused_root.stderr


def test_torus_exact_source_archive_cold_preserves_coefficients_and_refuses_tamper(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "write-torus-exact-source",
            str(tmp_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert written.returncode == 0, written.stderr
    expected = json.loads(written.stdout)
    identity = expected["torus_patch_identity"]
    assert set(identity) == {
        "kind",
        "center",
        "first_axis",
        "second_axis",
        "axis",
        "major_radius",
        "minor_radius",
    }
    placed_torus_identity = expected["placed_torus_patch_identity"]
    assert set(placed_torus_identity) == set(identity)
    assert (
        expected["placed_surface_identity"]["definition"]["definition"]
        == placed_torus_identity
    )
    assert identity["center"] == [
        ["rational", 1, 10],
        ["rational", -1, 2],
        ["rational", 5, 4],
    ]
    assert identity["first_axis"] == [
        ["rational", 3, 5],
        ["rational", 4, 5],
        ["rational", 0, 1],
    ]
    assert identity["second_axis"] == [
        ["rational", 0, 1],
        ["rational", 0, 1],
        ["rational", 1, 1],
    ]
    assert identity["axis"] == [
        ["rational", 4, 5],
        ["rational", -3, 5],
        ["rational", 0, 1],
    ]
    assert identity["major_radius"] == ["rational", 21, 10]
    assert identity["minor_radius"] == ["rational", 1, 5]
    placed_circle_identity = expected["placed_curve_identity"]["definition"]["definition"]
    assert set(placed_circle_identity) == {
        "kind",
        "center",
        "first_axis",
        "second_axis",
        "radius",
    }
    assert placed_circle_identity["center"] == [
        ["rational", 1, 10],
        ["rational", -1, 2],
        ["rational", 5, 4],
    ]
    assert placed_circle_identity["radius"] == ["rational", 1, 10]
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-torus-exact-source",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert restored.returncode == 0, restored.stderr
    assert restored.stdout.strip() == "exact-torus-source-cold-restored"

    source = tmp_path / "exact-torus-source.phx"
    manifest, arrays = read_model_archive(source, limits=None)

    def find_archive_node(node: object, type_name: str, /) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") in ("dataclass", "strict_value")
                and str(node.get("type", "")).rsplit(":", 1)[-1] == type_name
            ):
                return node
            for child in node.values():
                found = find_archive_node(child, type_name)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_archive_node(child, type_name)
                if found is not None:
                    return found
        return None

    tampered = copy.deepcopy(manifest)
    torus = find_archive_node(tampered["model_recipe"], "TorusPatch")
    assert torus is not None
    exact_major_radius = torus["fields"]["major_radius_exact"]
    assert exact_major_radius["kind"] == "fraction"
    exact_major_radius["numerator"] = int(exact_major_radius["numerator"]) + 1
    tampered_directory = tmp_path / "changed-exact-torus-source"
    tampered_directory.mkdir()
    write_model_archive(
        tampered_directory / "exact-torus-source.phx",
        manifest=tampered,
        arrays={name: np.array(array, copy=True) for name, array in arrays.items()},
        limits=None,
    )
    shutil.copyfile(
        tmp_path / "placed-torus-surface.phx",
        tampered_directory / "placed-torus-surface.phx",
    )
    shutil.copyfile(
        tmp_path / "placed-exact-circle.phx",
        tampered_directory / "placed-exact-circle.phx",
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-torus-exact-source",
            str(tampered_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_brep_archive_cold_restores_orthogonal_quadric_intersection_period_shifts(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "write-orthogonal-quadric-boolean",
            str(tmp_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert written.returncode == 0, written.stderr
    expected = json.loads(written.stdout)
    assert expected["branches"]
    assert any(
        np.any(np.asarray(branch["node_period_shifts"], dtype=np.int64))
        for branch in expected["branches"]
    )
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-orthogonal-quadric-boolean",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert restored.returncode == 0, restored.stderr
    assert restored.stdout.strip() == "orthogonal-quadric-boolean-cold-restored"

    source = tmp_path / "orthogonal-quadric-boolean.phx"
    manifest, arrays = read_model_archive(source, limits=None)

    _register_model_archive_types()
    shift_leaf = next(
        (
            entry
            for entry in model_recipe_array_inventory(
                manifest["model_recipe"], prefix="model/leaves", limits=None
            )
            if entry.path.endswith(".node_period_shifts") and np.any(arrays[entry.name])
        ),
        None,
    )
    assert shift_leaf is not None
    damaged_arrays = {name: array.copy() for name, array in arrays.items()}
    damaged_shifts = damaged_arrays[shift_leaf.name]
    damaged_shifts.flat[0] += 1
    corrupted_directory = tmp_path / "changed-orthogonal-quadric-period-shift"
    corrupted_directory.mkdir()
    write_model_archive(
        corrupted_directory / "orthogonal-quadric-boolean.phx",
        manifest=manifest,
        arrays=damaged_arrays,
        limits=None,
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-orthogonal-quadric-boolean",
            str(corrupted_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_adaptive_implicit_domain_archive_recomputes_proof_in_a_cold_process(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [sys.executable, str(_WORKER), "write-implicit", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-implicit",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert restored.stdout.strip() == "implicit-cold-restored"
    source = tmp_path / "implicit-domain.phx"
    corrupt_directory = tmp_path / "changed-implicit-domain"
    corrupt_directory.mkdir()
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    recipe = manifest["model_recipe"]["fields"]["entries"]["items"][0]["fields"]["value"]
    recipe["fields"]["source_id"]["value"] = "tampered-implicit-source"
    corrupted = corrupt_directory / "implicit-domain.phx"
    write_model_archive(
        corrupted, manifest=manifest, arrays=arrays, limits=MODEL_ARCHIVE_LIMITS
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-implicit",
            str(corrupt_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_reconstructed_geometry_source_recipe_restores_stable_rows_cold(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [sys.executable, str(_WORKER), "write-reconstructed", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-reconstructed",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert restored.stdout.strip() == "reconstructed-source-cold-restored"

    source = tmp_path / "reconstructed-source.phx"
    corrupt_directory = tmp_path / "changed-reconstructed-source"
    corrupt_directory.mkdir()
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    recipe = manifest["model_recipe"]["fields"]["entries"]["items"][0]["fields"]["value"]
    assert recipe["kind"] == "namedtuple"
    # ReconstructedGeometrySourceArchiveRecipe field 7: expected_source_revision.
    revision_field = recipe["items"][7]
    revision = revision_field["value"]
    revision_field["value"] = "0" * 64 if revision != "0" * 64 else "1" * 64
    corrupted = corrupt_directory / "reconstructed-source.phx"
    write_model_archive(
        corrupted, manifest=manifest, arrays=arrays, limits=MODEL_ARCHIVE_LIMITS
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-reconstructed",
            str(corrupt_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr
    domain_archive = tmp_path / "reconstructed-meshing-domain.phx"
    domain_manifest, domain_arrays = read_model_archive(domain_archive, limits=None)

    def find_reconstructed_patch(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if node.get("kind") == "dataclass" and str(node.get("type", "")).endswith(
                ":_ReconstructedTrianglePatch"
            ):
                return node
            for child in node.values():
                found = find_reconstructed_patch(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_reconstructed_patch(child)
                if found is not None:
                    return found
        return None

    patch_recipe = find_reconstructed_patch(domain_manifest["model_recipe"])
    assert patch_recipe is not None
    first_coordinate = patch_recipe["fields"]["exact_corners"]["items"][0]["items"][0]
    assert first_coordinate["kind"] == "fraction"
    first_coordinate["numerator"] += 1
    corrupted_domain_directory = tmp_path / "changed-reconstructed-triangle-corners"
    corrupted_domain_directory.mkdir()
    shutil.copyfile(source, corrupted_domain_directory / "reconstructed-source.phx")
    write_model_archive(
        corrupted_domain_directory / "reconstructed-meshing-domain.phx",
        manifest=domain_manifest,
        arrays=domain_arrays,
        limits=None,
    )
    refused_corners = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-reconstructed",
            str(corrupted_domain_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_corners.returncode != 0


def test_parametric_curve_query_archive_rebuilds_source_in_a_cold_process(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [sys.executable, str(_WORKER), "write-curve-query", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-curve-query",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert restored.stdout.strip() == "parametric-curve-query-cold-restored"

    source = tmp_path / "parametric-curve-query.phx"
    corrupt_directory = tmp_path / "changed-parametric-curve-query"
    corrupt_directory.mkdir()
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    query_entry = next(
        entry
        for entry in manifest["model_recipe"]["fields"]["entries"]["items"]
        if entry["fields"]["entry_id"]["value"] == "geometry/curve-query"
    )
    query_recipe = query_entry["fields"]["value"]
    revision = query_recipe["fields"]["source_revision"]["value"]
    query_recipe["fields"]["source_revision"]["value"] = (
        "0" * 64 if revision != "0" * 64 else "1" * 64
    )
    corrupted = corrupt_directory / "parametric-curve-query.phx"
    write_model_archive(
        corrupted, manifest=manifest, arrays=arrays, limits=MODEL_ARCHIVE_LIMITS
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-curve-query",
            str(corrupt_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_brep_archive_recomputes_exact_curve_trim_identity(
    tmp_path: Path,
) -> None:
    subprocess.run(
        [sys.executable, str(_WORKER), "write", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    source = tmp_path / "boolean-carriers.phx"
    corrupted = tmp_path / "changed-curve-trim-domain.phx"
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    trim = manifest["model_recipe"]["fields"]["trim_domains"]["items"][0]
    segment = trim["fields"]["curve_loops"]["items"][0]["items"][0]
    segment["fields"]["first"]["value"] = 0.125
    write_model_archive(
        corrupted, manifest=manifest, arrays=arrays, limits=MODEL_ARCHIVE_LIMITS
    )

    refused = subprocess.run(
        [sys.executable, str(_WORKER), "read-brep", str(corrupted)],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_brep_archive_recomputes_conic_root_source_identity(
    tmp_path: Path,
) -> None:
    subprocess.run(
        [sys.executable, str(_WORKER), "write", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    source = tmp_path / "boolean-carriers.phx"
    corrupted = tmp_path / "changed-conic-root.phx"
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    original_manifest = copy.deepcopy(manifest)
    pcurve = manifest["model_recipe"]["fields"]["geometry"]["fields"]["pcurves"]["items"][
        3
    ]
    root = pcurve["fields"]["root"]
    root["fields"]["cosine_exact"]["numerator"] = 2
    write_model_archive(
        corrupted, manifest=manifest, arrays=arrays, limits=MODEL_ARCHIVE_LIMITS
    )

    refused = subprocess.run(
        [sys.executable, str(_WORKER), "read-brep", str(corrupted)],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr
    endpoint_manifest = copy.deepcopy(original_manifest)
    geometry_fields = endpoint_manifest["model_recipe"]["fields"]["geometry"]["fields"]
    endpoint = geometry_fields["edge_endpoint_roots"]["items"][0]["items"][1]
    endpoint["fields"]["point_exact"]["items"][0]["numerator"] = 2
    endpoint_corrupted = tmp_path / "changed-conic-endpoint-point.phx"
    write_model_archive(
        endpoint_corrupted,
        manifest=endpoint_manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    refused_endpoint = subprocess.run(
        [sys.executable, str(_WORKER), "read-brep", str(endpoint_corrupted)],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_endpoint.returncode != 0
    assert "incompatible with its registered recipe" in refused_endpoint.stderr
    phase_manifest = copy.deepcopy(original_manifest)
    conic_curve = phase_manifest["model_recipe"]["fields"]["geometry"]["fields"][
        "pcurves"
    ]["items"][3]
    phase_shift = conic_curve["fields"]["phase_shift_exact"]
    assert phase_shift["kind"] == "fraction"
    assert phase_shift["numerator"] == 0
    phase_shift["numerator"] = 1
    phase_corrupted = tmp_path / "changed-conic-phase-shift.phx"
    write_model_archive(
        phase_corrupted,
        manifest=phase_manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    refused_phase = subprocess.run(
        [sys.executable, str(_WORKER), "read-brep", str(phase_corrupted)],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_phase.returncode != 0
    assert "incompatible with its registered recipe" in refused_phase.stderr
    spline_manifest = copy.deepcopy(original_manifest)

    def find_spline(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") == "dataclass"
                and str(node.get("type", "")).rsplit(":", 1)[-1] == "BSplineCurve"
            ):
                return node
            for child in node.values():
                found = find_spline(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_spline(child)
                if found is not None:
                    return found
        return None

    spline = find_spline(spline_manifest["model_recipe"])
    assert spline is not None
    spline["fields"]["exact_control_points"]["items"][1]["items"][0]["numerator"] = 2
    spline_corrupted = tmp_path / "changed-exact-spline-controls.phx"
    write_model_archive(
        spline_corrupted,
        manifest=spline_manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    refused_spline = subprocess.run(
        [sys.executable, str(_WORKER), "read-brep", str(spline_corrupted)],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_spline.returncode != 0
    assert "incompatible with its registered recipe" in refused_spline.stderr


def test_brep_archive_refuses_tampered_exact_bspline_surface_bank(
    tmp_path: Path,
) -> None:
    subprocess.run(
        [sys.executable, str(_WORKER), "write", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    source = tmp_path / "bspline-surface.phx"
    corrupted = tmp_path / "changed-exact-bspline-surface.phx"
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    def find_surface(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") == "dataclass"
                and str(node.get("type", "")).rsplit(":", 1)[-1] == "BSplineSurfacePatch"
            ):
                return node
            for child in node.values():
                found = find_surface(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_surface(child)
                if found is not None:
                    return found
        return None

    surface = find_surface(manifest["model_recipe"])
    assert surface is not None
    surface["fields"]["exact_control_points"]["items"][0]["items"][0]["items"][0][
        "numerator"
    ] = 1
    write_model_archive(
        corrupted, manifest=manifest, arrays=arrays, limits=MODEL_ARCHIVE_LIMITS
    )
    refused = subprocess.run(
        [sys.executable, str(_WORKER), "read-brep", str(corrupted)],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_brep_archive_rejects_tampered_exact_offset_carrier_distance(
    tmp_path: Path,
) -> None:
    subprocess.run(
        [sys.executable, str(_WORKER), "write", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    source = tmp_path / "offset-ruled.phx"
    corrupted = tmp_path / "changed-exact-offset-distance.phx"
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    def find_offset(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") == "dataclass"
                and str(node.get("type", "")).rsplit(":", 1)[-1] == "OffsetCurve"
            ):
                return node
            for child in node.values():
                found = find_offset(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_offset(child)
                if found is not None:
                    return found
        return None

    offset = find_offset(manifest["model_recipe"])
    assert offset is not None
    offset["fields"]["distance_exact"]["numerator"] = 2
    write_model_archive(
        corrupted, manifest=manifest, arrays=arrays, limits=MODEL_ARCHIVE_LIMITS
    )
    refused = subprocess.run(
        [sys.executable, str(_WORKER), "read-brep", str(corrupted)],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_brep_archive_rejects_forged_absent_sphere_revolution_witness(
    tmp_path: Path,
) -> None:
    subprocess.run(
        [sys.executable, str(_WORKER), "write", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    source = tmp_path / "offset.phx"
    corrupted = tmp_path / "changed-absent-offset-witness.phx"
    manifest, arrays = read_model_archive(source, limits=None)

    def find_offset(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") == "dataclass"
                and str(node.get("type", "")).rsplit(":", 1)[-1] == "OffsetSurface"
            ):
                return node
            for child in node.values():
                found = find_offset(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_offset(child)
                if found is not None:
                    return found
        return None

    offset = find_offset(manifest["model_recipe"])
    assert offset is not None
    witness = offset["fields"]["_sphere_revolution_witness"]
    assert witness["kind"] == "tuple" and len(witness["items"]) == 4
    offset["fields"]["_sphere_revolution_witness"] = {
        "kind": "literal",
        "value": None,
    }
    write_model_archive(corrupted, manifest=manifest, arrays=arrays, limits=None)
    refused = subprocess.run(
        [sys.executable, str(_WORKER), "read-brep", str(corrupted)],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr

    missing_directory = tmp_path / "missing-absent-offset-witness"
    missing_directory.mkdir()
    missing_manifest, missing_arrays = read_model_archive(source, limits=None)

    missing_offset = find_offset(missing_manifest["model_recipe"])
    assert missing_offset is not None
    del missing_offset["fields"]["_sphere_revolution_witness"]
    write_model_archive(
        missing_directory / "offset.phx",
        manifest=missing_manifest,
        arrays=missing_arrays,
        limits=None,
    )
    refused_missing = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-brep",
            str(missing_directory / "offset.phx"),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_missing.returncode != 0


def test_brep_archive_recomputes_revolution_offset_witness_cold(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [sys.executable, str(_WORKER), "write-revolution-offset", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-revolution-offset",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert restored.stdout.strip() == "revolution-offset-cold-restored"

    source = tmp_path / "revolution-offset.phx"
    corrupted_directory = tmp_path / "changed-revolution-offset-witness"
    corrupted_directory.mkdir()
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    def find_offset(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") == "dataclass"
                and str(node.get("type", "")).rsplit(":", 1)[-1] == "OffsetSurface"
            ):
                return node
            for child in node.values():
                found = find_offset(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_offset(child)
                if found is not None:
                    return found
        return None

    offset = find_offset(manifest["model_recipe"])
    assert offset is not None
    witness = offset["fields"]["_sphere_revolution_witness"]
    assert witness["kind"] == "tuple" and len(witness["items"]) == 4
    frame_sign = witness["items"][3]
    assert frame_sign["kind"] == "literal"
    frame_sign["value"] = -frame_sign["value"]
    write_model_archive(
        corrupted_directory / "revolution-offset.phx",
        manifest=manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-revolution-offset",
            str(corrupted_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_computed_mesh_certification_archive_recomputes_acceptance_cold(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [sys.executable, str(_WORKER), "write-certification", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-certification",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert restored.stdout.strip() == "certification-cold-restored"

    source = tmp_path / "certification.phx"
    corrupt_directory = tmp_path / "changed-certification-report"
    corrupt_directory.mkdir()
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    report = manifest["model_recipe"]["fields"]["certification"]
    report["fields"]["report_id"]["value"] = "0" * 64
    write_model_archive(
        corrupt_directory / "certification.phx",
        manifest=manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-certification",
            str(corrupt_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr

    evidence_directory = tmp_path / "changed-certification-embedding"
    evidence_directory.mkdir()
    evidence_manifest, evidence_arrays = read_model_archive(
        source, limits=MODEL_ARCHIVE_LIMITS
    )

    evidence_report = evidence_manifest["model_recipe"]["fields"]["certification"]
    evidence_report["fields"]["embedding"]["fields"]["certificate_id"]["value"] = "0" * 64
    write_model_archive(
        evidence_directory / "certification.phx",
        manifest=evidence_manifest,
        arrays=evidence_arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    refused_evidence = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-certification",
            str(evidence_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_evidence.returncode != 0
    assert "incompatible with its registered recipe" in refused_evidence.stderr

    def quality_recipes(node: object, /) -> tuple[dict[str, Any], ...]:
        found: list[dict[str, Any]] = []
        pending = [node]
        while pending:
            current = pending.pop()
            if isinstance(current, dict):
                if current.get("kind") == "dataclass" and str(
                    current.get("type", "")
                ).endswith(":CellQualityReportArchiveRecipe"):
                    found.append(current)
                pending.extend(current.values())
            elif isinstance(current, list):
                pending.extend(current)
        return tuple(found)

    quality_manifest, quality_arrays = read_model_archive(
        source, limits=MODEL_ARCHIVE_LIMITS
    )

    stored_quality_recipes = quality_recipes(quality_manifest["model_recipe"])
    assert stored_quality_recipes
    for quality_recipe in stored_quality_recipes:
        minimum_measure = quality_recipe["fields"]["minimum_measure"]
        assert minimum_measure["kind"] == "literal"
        assert type(minimum_measure["value"]) in (int, float)
        assert np.isfinite(minimum_measure["value"])
        minimum_measure["value"] = None
    quality_directory = tmp_path / "changed-quality-finite-summary-to-none"
    quality_directory.mkdir()
    write_model_archive(
        quality_directory / "certification.phx",
        manifest=quality_manifest,
        arrays=quality_arrays,
        limits=None,
    )
    refused_quality = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-certification",
            str(quality_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_quality.returncode != 0
    assert "incompatible with its registered recipe" in refused_quality.stderr

    validity_fields = (
        "status",
        "determinant_lower",
        "determinant_upper",
        "depth",
        "execution_status",
        "source_status",
        "source_determinant_lower",
        "source_determinant_upper",
        "source_depth",
        "maximum_coefficient_rounding_error",
        "maximum_element_coefficient_rounding_error",
    )
    proof_manifest, proof_arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    _register_model_archive_types()
    validity_inventory = model_recipe_array_inventory(
        proof_manifest["model_recipe"], prefix="model/leaves", limits=None
    )
    for validity_field in validity_fields:
        matches = tuple(
            entry
            for entry in validity_inventory
            if "validity" in entry.path and entry.path.endswith(f".{validity_field}")
        )
        assert len(matches) == 1, validity_field
        damaged_arrays = {name: array.copy() for name, array in proof_arrays.items()}
        damaged = damaged_arrays[matches[0].name]
        assert damaged.size
        if np.issubdtype(damaged.dtype, np.integer):
            damaged.flat[0] += 1
        else:
            damaged.flat[0] += 1.0
        field_directory = tmp_path / f"changed-validity-{validity_field}"
        field_directory.mkdir()
        write_model_archive(
            field_directory / "certification.phx",
            manifest=copy.deepcopy(proof_manifest),
            arrays=damaged_arrays,
            limits=None,
        )
        refused_field = subprocess.run(
            [
                sys.executable,
                str(_WORKER),
                "read-certification",
                str(field_directory),
                json.dumps(expected),
            ],
            cwd=_REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert refused_field.returncode != 0, validity_field
        assert "incompatible with its registered recipe" in refused_field.stderr, (
            validity_field
        )


def test_restricted_coverage_witness_archive_replays_proof_cold(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [sys.executable, str(_WORKER), "write-source-coverage", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-source-coverage",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert restored.stdout.strip() == "source-coverage-cold-restored"

    source = tmp_path / "source-coverage.phx"
    corrupted_directory = tmp_path / "changed-source-coverage-witness"
    corrupted_directory.mkdir()
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    def find_witness(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") == "dataclass"
                and str(node.get("type", "")).rsplit(":", 1)[-1]
                == "RestrictedDelaunayBoundaryWitness"
            ):
                return node
            for child in node.values():
                found = find_witness(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_witness(child)
                if found is not None:
                    return found
        return None

    witness = find_witness(manifest["model_recipe"])
    assert witness is not None
    assert set(witness["fields"]) == {
        "mesh_id",
        "geometry_id",
        "topology_id",
        "profile_id",
        "source_id",
        "source_revision",
        "domain_id",
        "distance_witness_id",
        "witness_id",
    }
    witness["fields"]["witness_id"]["value"] = "0" * 64
    write_model_archive(
        corrupted_directory / "source-coverage.phx",
        manifest=manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-source-coverage",
            str(corrupted_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr
    assert expected["coverage_region_occurrence_paths"] == [[]]
    path_manifest, path_arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    def find_coverage_certificate(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if node.get("kind") == "dataclass" and str(node.get("type", "")).endswith(
                ":DomainCoverageCertificate"
            ):
                return node
            for child in node.values():
                found = find_coverage_certificate(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_coverage_certificate(child)
                if found is not None:
                    return found
        return None

    coverage = find_coverage_certificate(path_manifest["model_recipe"])
    assert coverage is not None
    path_bank = coverage["fields"]["region_occurrence_paths"]
    assert path_bank["kind"] == "tuple"
    path_bank["items"][0]["items"] = [
        {"kind": "literal", "value": "stale-region-occurrence"}
    ]
    path_directory = tmp_path / "changed-coverage-region-occurrence"
    path_directory.mkdir()
    write_model_archive(
        path_directory / "source-coverage.phx",
        manifest=path_manifest,
        arrays=path_arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    refused_paths = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-source-coverage",
            str(path_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_paths.returncode != 0


def test_region_boundary_archive_preserves_source_occurrence_identity_cold(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "write-region-boundary-evidence",
            str(tmp_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    assert expected["source_occurrence_path"] == []
    assert expected["patch_source_adjacent_region_occurrence_paths"] == [[], None]
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-region-boundary-evidence",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert restored.stdout.strip() == "region-boundary-evidence-cold-restored"

    source = tmp_path / "region-boundary-evidence.phx"
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    def find_region_evidence(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if node.get("kind") == "dataclass" and str(node.get("type", "")).endswith(
                ":RegionBoundaryEvidence"
            ):
                return node
            for child in node.values():
                found = find_region_evidence(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_region_evidence(child)
                if found is not None:
                    return found
        return None

    evidence = find_region_evidence(manifest["model_recipe"])
    assert evidence is not None
    assert "source_occurrence_path" in evidence["fields"]

    def find_mesh_patch(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if node.get("kind") == "dataclass" and str(node.get("type", "")).endswith(
                ":MeshPatch"
            ):
                return node
            for child in node.values():
                found = find_mesh_patch(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_mesh_patch(child)
                if found is not None:
                    return found
        return None

    patch = find_mesh_patch(manifest["model_recipe"])
    assert patch is not None
    assert "source_adjacent_region_occurrence_paths" in patch["fields"]
    evidence["fields"]["source_occurrence_path"] = {
        "kind": "tuple",
        "items": [{"kind": "literal", "value": "renamed-occurrence"}],
    }
    corrupt_directory = tmp_path / "changed-region-occurrence-identity"
    corrupt_directory.mkdir()
    write_model_archive(
        corrupt_directory / "region-boundary-evidence.phx",
        manifest=manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-region-boundary-evidence",
            str(corrupt_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0


def test_source_chart_rational_banks_restore_exactly_in_a_cold_process(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [sys.executable, str(_WORKER), "write-source-chart-rational", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-source-chart-rational",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert restored.stdout.strip() == "source-chart-rational-cold-restored"

    source = tmp_path / "source-chart-rational-witness.phx"
    corrupted_directory = tmp_path / "changed-source-chart-rational-bank"
    corrupted_directory.mkdir()
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    witness = manifest["model_recipe"]
    tokens = witness["fields"]["source_parameter_rational_tokens"]["items"][0]
    tokens["items"][1]["items"][0]["items"][0]["value"] = 2
    write_model_archive(
        corrupted_directory / "source-chart-rational-witness.phx",
        manifest=manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-source-chart-rational",
            str(corrupted_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_coordinate_map_composition_cold_replays_native_source_maps(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [sys.executable, str(_WORKER), "write-coordinate-maps", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-coordinate-maps",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert restored.stdout.strip() == "coordinate-map-cold-restored"

    source = tmp_path / "coordinate-maps.phx"
    corrupt_directory = tmp_path / "changed-cut-map-identity"
    corrupt_directory.mkdir()
    manifest, arrays = read_model_archive(source, limits=None)

    def replace_cut_map_identity(value: object) -> bool:
        if isinstance(value, dict):
            if str(value.get("type", "")).endswith(":_CutCoordinateMap"):
                identity = value["fields"]["source_map_id"]["value"]
                value["fields"]["source_map_id"]["value"] = (
                    "0" * 64 if identity != "0" * 64 else "1" * 64
                )
                return True
            return any(replace_cut_map_identity(item) for item in value.values())
        if isinstance(value, list):
            return any(replace_cut_map_identity(item) for item in value)
        return False

    assert replace_cut_map_identity(manifest["model_recipe"])
    write_model_archive(
        corrupt_directory / "coordinate-maps.phx",
        manifest=manifest,
        arrays=arrays,
        limits=None,
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-coordinate-maps",
            str(corrupt_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr
    manifest, arrays = read_model_archive(source, limits=None)

    def find_cut_map(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if str(node.get("type", "")).endswith(":_CutCoordinateMap"):
                return node
            for child in node.values():
                found = find_cut_map(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_cut_map(child)
                if found is not None:
                    return found
        return None

    cut_map = find_cut_map(manifest["model_recipe"])
    assert cut_map is not None
    node_keys = cut_map["fields"]["node_keys"]
    first_key = node_keys["items"][0]
    root_vertex_id = first_key["items"][2]
    assert root_vertex_id["kind"] == "literal"
    root_vertex_id["value"] += 1
    node_directory = tmp_path / "changed-cut-node-keys"
    node_directory.mkdir()
    write_model_archive(
        node_directory / "coordinate-maps.phx",
        manifest=manifest,
        arrays=arrays,
        limits=None,
    )
    refused_node_keys = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-coordinate-maps",
            str(node_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_node_keys.returncode != 0
    assert "incompatible with its registered recipe" in refused_node_keys.stderr
    bank_manifest, bank_arrays = read_model_archive(source, limits=None)

    def find_exact_coordinate_bank(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") == "dataclass"
                and str(node.get("type", "")).rsplit(":", 1)[-1] == "CellGeometrySpec"
            ):
                fields = node.get("fields")
                if isinstance(fields, dict):
                    bank = fields.get("exact_coordinate_bank")
                    if isinstance(bank, dict) and bank.get("kind") == "tuple":
                        return bank
            for child in node.values():
                found = find_exact_coordinate_bank(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_exact_coordinate_bank(child)
                if found is not None:
                    return found
        return None

    def tamper_first_exact_fraction(node: object) -> bool:
        if isinstance(node, dict):
            if node.get("kind") == "fraction" and type(node.get("numerator")) is int:
                node["numerator"] += 1
                return True
            return any(tamper_first_exact_fraction(child) for child in node.values())
        if isinstance(node, list):
            return any(tamper_first_exact_fraction(child) for child in node)
        return False

    bank = find_exact_coordinate_bank(bank_manifest["model_recipe"])
    assert bank is not None
    assert tamper_first_exact_fraction(bank)
    bank_directory = tmp_path / "changed-cell-geometry-exact-coordinate-bank"
    bank_directory.mkdir()
    write_model_archive(
        bank_directory / "coordinate-maps.phx",
        manifest=bank_manifest,
        arrays=bank_arrays,
        limits=None,
    )
    refused_bank = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-coordinate-maps",
            str(bank_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_bank.returncode != 0
    assert "incompatible with its registered recipe" in refused_bank.stderr


def test_periodic_power_source_context_archive_cold_restores_and_refuses_tamper(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "write-periodic-power-map",
            str(tmp_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert written.returncode == 0, written.stderr
    expected = json.loads(written.stdout)
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-periodic-power-map",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert restored.returncode == 0, restored.stderr
    assert restored.stdout.strip() == "periodic-power-source-context-cold-restored"
    no_mesh = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-periodic-power-map-without-mesh",
            str(tmp_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert no_mesh.returncode != 0
    assert "incompatible with its registered recipe" in no_mesh.stderr
    assert expected["output_vertex_indices"]
    assert expected["context_id"]
    source = tmp_path / "periodic-power.phx"
    manifest, arrays = read_model_archive(source, limits=None)

    def find_periodic_context(node: object, /) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") == "dataclass"
                and str(node.get("type", "")).rsplit(":", 1)[-1]
                == "ExactPeriodicPowerSourceContext"
            ):
                return node
            for child in node.values():
                found = find_periodic_context(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_periodic_context(child)
                if found is not None:
                    return found
        return None

    context = find_periodic_context(manifest["model_recipe"])
    assert context is not None
    context_id = context["fields"]["context_id"]
    assert context_id["kind"] == "literal"
    original_id = context_id["value"]
    context_id["value"] = "0" * 64 if original_id != "0" * 64 else "1" * 64
    corrupted_directory = tmp_path / "changed-periodic-power-source-context"
    corrupted_directory.mkdir()
    write_model_archive(
        corrupted_directory / "periodic-power.phx",
        manifest=manifest,
        arrays=arrays,
        limits=None,
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-periodic-power-map",
            str(corrupted_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr
    shard_manifest, shard_arrays = read_model_archive(source, limits=None)

    _register_model_archive_types()
    shard_leaf = next(
        (
            entry
            for entry in model_recipe_array_inventory(
                shard_manifest["model_recipe"], prefix="model/leaves", limits=None
            )
            if entry.path.endswith(".source_directory_ids")
        ),
        None,
    )
    assert shard_leaf is not None
    damaged_arrays = {name: array.copy() for name, array in shard_arrays.items()}
    damaged_shard = damaged_arrays[shard_leaf.name]
    assert damaged_shard.size
    damaged_shard.flat[0] += 1
    shard_directory = tmp_path / "changed-periodic-owner-directory-shard"
    shard_directory.mkdir()
    write_model_archive(
        shard_directory / "periodic-power.phx",
        manifest=shard_manifest,
        arrays=damaged_arrays,
        limits=None,
    )
    refused_shard = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-periodic-power-map",
            str(shard_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_shard.returncode != 0
    assert "incompatible with its registered recipe" in refused_shard.stderr


def test_owner_local_periodic_remap_archive_cold_restores_and_refuses_tamper(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "write-owner-local-remap",
            str(tmp_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert written.returncode == 0, written.stderr
    expected = json.loads(written.stdout)
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-owner-local-remap",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert restored.returncode == 0, restored.stderr
    assert restored.stdout.strip() == "owner-local-periodic-remap-cold-restored"
    assert expected["owned_cell_count"] == len(expected["cell_ids"]) > 0
    assert expected["remap_id"] and expected["evidence_id"]
    assert expected["specification_id"]
    assert expected["view_specification_id"] == expected["specification_id"]
    assert expected["coverage_complete"] is True
    assert expected["owner_rank"] == expected["local_rank"] == 0
    assert expected["source_context_id"]
    assert expected["source_map_id"]
    assert expected["periodic_topology_id"]
    assert expected["periodic_face_coverage_id"]
    assert expected["local_mesh_id"]
    assert expected["local_topology_id"]
    assert expected["local_geometry_id"]

    source = tmp_path / "owner-local-remap.phx"
    manifest, arrays = read_model_archive(source, limits=None)

    def find_archive_node(node: object, type_name: str, /) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") in ("dataclass", "strict_value")
                and str(node.get("type", "")).rsplit(":", 1)[-1] == type_name
            ):
                return node
            for child in node.values():
                found = find_archive_node(child, type_name)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_archive_node(child, type_name)
                if found is not None:
                    return found
        return None

    def refuse_archive(
        mutated_manifest: dict[str, Any],
        directory_name: str,
        expected_error: str,
        /,
    ) -> None:
        directory = tmp_path / directory_name
        directory.mkdir()
        write_model_archive(
            directory / "owner-local-remap.phx",
            manifest=mutated_manifest,
            arrays={name: np.array(array, copy=True) for name, array in arrays.items()},
            limits=None,
        )
        refused = subprocess.run(
            [
                sys.executable,
                str(_WORKER),
                "read-owner-local-remap",
                str(directory),
                json.dumps(expected),
            ],
            cwd=_REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert refused.returncode != 0
        assert expected_error in refused.stderr

    rank_manifest = copy.deepcopy(manifest)
    owner_view = find_archive_node(
        rank_manifest["model_recipe"], "OwnerLocalCellGeometry"
    )
    assert owner_view is not None
    owner_rank = owner_view["fields"]["owner_rank"]
    assert owner_rank["kind"] == "literal"
    owner_rank["value"] = int(owner_rank["value"]) + 1
    rank_directory = tmp_path / "changed-owner-local-rank"
    rank_directory.mkdir()
    write_model_archive(
        rank_directory / "owner-local-remap.phx",
        manifest=rank_manifest,
        arrays={name: np.array(array, copy=True) for name, array in arrays.items()},
        limits=None,
    )
    refused_rank = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-owner-local-remap",
            str(rank_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_rank.returncode != 0
    assert "current group rank" in refused_rank.stderr

    stale_spec = copy.deepcopy(manifest)
    volume_spec = find_archive_node(stale_spec["model_recipe"], "VolumeMeshingSpec")
    assert volume_spec is not None
    maximum_deviation = volume_spec["fields"]["maximum_deviation"]
    assert maximum_deviation["kind"] == "literal"
    assert maximum_deviation["value"] is None
    maximum_deviation["value"] = 0.1
    refuse_archive(
        stale_spec,
        "changed-owner-local-current-specification",
        "incompatible with its registered recipe",
    )

    stale_view_binding = copy.deepcopy(manifest)
    owner_view = find_archive_node(
        stale_view_binding["model_recipe"], "OwnerLocalCellGeometry"
    )
    assert owner_view is not None
    specification_id = owner_view["fields"]["specification_id"]
    assert specification_id["kind"] == "literal"
    assert specification_id["value"] == expected["specification_id"]
    specification_id["value"] = (
        "0" * 64 if specification_id["value"] != "0" * 64 else "1" * 64
    )
    refuse_archive(
        stale_view_binding,
        "changed-owner-local-view-specification",
        "current exact meshing specification",
    )

    remap_manifest = copy.deepcopy(manifest)
    owner_remap = find_archive_node(
        remap_manifest["model_recipe"], "PreparedOwnerLocalConservativeRemap"
    )
    assert owner_remap is not None
    remap_id = owner_remap["fields"]["remap_id"]
    assert remap_id["kind"] == "literal"
    original_remap_id = remap_id["value"]
    remap_id["value"] = "0" * 64 if original_remap_id != "0" * 64 else "1" * 64
    remap_directory = tmp_path / "changed-owner-local-remap-identity"
    remap_directory.mkdir()
    write_model_archive(
        remap_directory / "owner-local-remap.phx",
        manifest=remap_manifest,
        arrays={name: np.array(array, copy=True) for name, array in arrays.items()},
        limits=None,
    )
    refused_remap = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-owner-local-remap",
            str(remap_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_remap.returncode != 0
    assert "native geometry replay" in refused_remap.stderr


def test_owner_local_periodic_adaptive_archive_cold_reattaches_new_group_with_spec_and_plan(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "write-owner-local-periodic-adaptive",
            str(tmp_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert written.returncode == 0, written.stderr
    expected = json.loads(written.stdout)
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-owner-local-periodic-adaptive",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert restored.returncode == 0, restored.stderr
    assert (
        restored.stdout.strip()
        == "owner-local-periodic-adaptive-reattached-cold-restored"
    )
    assert expected["specification_id"]
    assert expected["graph_partition_plan_id"]
    assert expected["adaptive_result_id"]
    assert expected["candidate_id"]
    assert expected["distribution_id"]
    assert expected["local_mesh_id"]
    assert expected["local_topology_id"]
    assert expected["local_geometry_id"]
    assert expected["periodic_topology_id"]
    assert expected["periodic_face_coverage_id"]
    assert expected["owned_cell_count"] > 0
    assert expected["owner_rank"] == 0
    assert expected["graph_migration_evidence_id"]
    assert expected["graph_partition_result_id"]
    assert expected["graph_source_candidate_id"]
    assert expected["graph_source_part_id"]
    assert expected["graph_source_directory_id"]
    assert expected["graph_target_part_id"]
    assert expected["graph_target_directory_id"]
    assert expected["graph_group_id"]
    assert expected["reattachment_group_id"]
    assert expected["graph_group_id"] != expected["reattachment_group_id"]
    assert expected["graph_local_rank"] == expected["owner_rank"] == 0
    assert expected["graph_part_count"] == 1
    assert expected["graph_target_cell_ids"]

    source = tmp_path / "owner-local-periodic-adaptive.phx"
    manifest, arrays = read_model_archive(source, limits=None)

    def find_archive_node(node: object, type_name: str, /) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") in ("dataclass", "strict_value")
                and str(node.get("type", "")).rsplit(":", 1)[-1] == type_name
            ):
                return node
            for child in node.values():
                found = find_archive_node(child, type_name)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_archive_node(child, type_name)
                if found is not None:
                    return found
        return None

    stale_spec = copy.deepcopy(manifest)
    specification = find_archive_node(stale_spec["model_recipe"], "SurfaceMeshingSpec")
    assert specification is not None
    maximum_deviation = specification["fields"]["maximum_deviation"]
    assert maximum_deviation["kind"] == "literal"
    assert maximum_deviation["value"] is None
    maximum_deviation["value"] = 0.1
    stale_spec_directory = tmp_path / "changed-owner-local-current-specification"
    stale_spec_directory.mkdir()
    write_model_archive(
        stale_spec_directory / "owner-local-periodic-adaptive.phx",
        manifest=stale_spec,
        arrays={name: np.array(array, copy=True) for name, array in arrays.items()},
        limits=None,
    )
    refused_spec = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-owner-local-periodic-adaptive",
            str(stale_spec_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_spec.returncode != 0
    assert "incompatible with its registered recipe" in refused_spec.stderr

    stale_plan = copy.deepcopy(manifest)
    plan = find_archive_node(stale_plan["model_recipe"], "GraphPartitionPlan")
    assert plan is not None
    refinement_passes = plan["fields"]["refinement_passes"]
    assert refinement_passes["kind"] == "literal"
    refinement_passes["value"] = int(refinement_passes["value"]) + 1
    stale_plan_directory = tmp_path / "changed-owner-local-current-graph-plan"
    stale_plan_directory.mkdir()
    write_model_archive(
        stale_plan_directory / "owner-local-periodic-adaptive.phx",
        manifest=stale_plan,
        arrays={name: np.array(array, copy=True) for name, array in arrays.items()},
        limits=None,
    )
    refused_plan = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-owner-local-periodic-adaptive",
            str(stale_plan_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_plan.returncode != 0
    assert "incompatible with its registered recipe" in refused_plan.stderr
    stale_evidence = copy.deepcopy(manifest)
    graph_evidence = find_archive_node(
        stale_evidence["model_recipe"], "OwnerLocalGraphMigrationEvidence"
    )
    assert graph_evidence is not None
    evidence_id = graph_evidence["fields"]["evidence_id"]
    assert evidence_id["kind"] == "literal"
    evidence_id["value"] = "0" * 64 if evidence_id["value"] != "0" * 64 else "1" * 64
    stale_evidence_directory = tmp_path / "changed-owner-local-graph-evidence"
    stale_evidence_directory.mkdir()
    write_model_archive(
        stale_evidence_directory / "owner-local-periodic-adaptive.phx",
        manifest=stale_evidence,
        arrays={name: np.array(array, copy=True) for name, array in arrays.items()},
        limits=None,
    )
    refused_evidence = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-owner-local-periodic-adaptive",
            str(stale_evidence_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_evidence.returncode != 0
    assert "incompatible with its registered recipe" in refused_evidence.stderr

    stale_group = copy.deepcopy(manifest)
    group_evidence = find_archive_node(
        stale_group["model_recipe"], "OwnerLocalGraphMigrationEvidence"
    )
    assert group_evidence is not None
    group_id = group_evidence["fields"]["group_id"]
    assert group_id["kind"] == "literal"
    group_id["value"] = "0" * 64 if group_id["value"] != "0" * 64 else "1" * 64
    stale_group_directory = tmp_path / "changed-owner-local-graph-group"
    stale_group_directory.mkdir()
    write_model_archive(
        stale_group_directory / "owner-local-periodic-adaptive.phx",
        manifest=stale_group,
        arrays={name: np.array(array, copy=True) for name, array in arrays.items()},
        limits=None,
    )
    refused_group = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-owner-local-periodic-adaptive",
            str(stale_group_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_group.returncode != 0
    assert "incompatible with its registered recipe" in refused_group.stderr

    _register_model_archive_types()
    graph_arrays = model_recipe_array_inventory(
        manifest["model_recipe"], prefix="model/leaves", limits=None
    )
    target_cell_arrays = tuple(
        entry
        for entry in graph_arrays
        if "graph_migration_evidence" in entry.path and "target_cell_ids" in entry.path
    )
    assert len(target_cell_arrays) == 1
    changed_graph_arrays = {
        name: np.array(array, copy=True) for name, array in arrays.items()
    }
    changed_target_ids = changed_graph_arrays[target_cell_arrays[0].name]
    assert changed_target_ids.size > 0
    changed_target_ids[0] = int(np.max(changed_target_ids)) + 1
    target_cell_directory = tmp_path / "changed-owner-local-target-cell-coverage"
    target_cell_directory.mkdir()
    write_model_archive(
        target_cell_directory / "owner-local-periodic-adaptive.phx",
        manifest=manifest,
        arrays=changed_graph_arrays,
        limits=None,
    )
    refused_target_cells = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-owner-local-periodic-adaptive",
            str(target_cell_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_target_cells.returncode != 0
    assert "incompatible with its registered recipe" in refused_target_cells.stderr


def test_meshing_spec_archive_cold_rebuilds_background_metric_and_all_spec_types(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [sys.executable, str(_WORKER), "write-meshing-specs", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert written.returncode == 0, written.stderr
    expected = json.loads(written.stdout)
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-meshing-specs",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert restored.returncode == 0, restored.stderr
    assert restored.stdout.strip() == "meshing-specs-cold-restored"
    assert expected["surface_specification_id"]
    assert expected["background_metric_control_id"]
    assert expected["background_mesh_id"]
    assert expected["background_metric_id"]
    assert expected["volume_specification_id"]
    assert expected["planar_specification_id"]
    assert expected["planar_embedding_id"]

    source = tmp_path / "meshing-specs.phx"
    manifest, arrays = read_model_archive(source, limits=None)

    def find_archive_node(node: object, type_name: str, /) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") in ("dataclass", "strict_value")
                and str(node.get("type", "")).rsplit(":", 1)[-1] == type_name
            ):
                return node
            for child in node.values():
                found = find_archive_node(child, type_name)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_archive_node(child, type_name)
                if found is not None:
                    return found
        return None

    tampered = copy.deepcopy(manifest)
    background = find_archive_node(tampered["model_recipe"], "BackgroundMetricControl")
    assert background is not None
    maximum_length = background["fields"]["maximum_metric_edge_length"]
    assert maximum_length["kind"] == "literal"
    maximum_length["value"] = float(maximum_length["value"]) + 0.25
    tampered_directory = tmp_path / "changed-meshing-background-metric-control"
    tampered_directory.mkdir()
    write_model_archive(
        tampered_directory / "meshing-specs.phx",
        manifest=tampered,
        arrays={name: np.array(array, copy=True) for name, array in arrays.items()},
        limits=None,
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-meshing-specs",
            str(tampered_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr

    _register_model_archive_types()
    inventory = model_recipe_array_inventory(
        manifest["model_recipe"], prefix="model/leaves", limits=None
    )
    metric_arrays = tuple(entry for entry in inventory if entry.shape == (3, 3, 3))
    assert len(metric_arrays) == 1
    changed_metric_arrays = {
        name: np.array(array, copy=True) for name, array in arrays.items()
    }
    changed_metric_arrays[metric_arrays[0].name][0, 0, 0] += 0.125
    metric_directory = tmp_path / "changed-background-metric-tensor"
    metric_directory.mkdir()
    write_model_archive(
        metric_directory / "meshing-specs.phx",
        manifest=manifest,
        arrays=changed_metric_arrays,
        limits=None,
    )
    refused_metric = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-meshing-specs",
            str(metric_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_metric.returncode != 0
    assert "incompatible with its registered recipe" in refused_metric.stderr

    coordinate_arrays = tuple(
        entry
        for entry in inventory
        if entry.shape == (3, 3) and "coordinates" in entry.path
    )
    assert len(coordinate_arrays) == 1
    changed_mesh_arrays = {
        name: np.array(array, copy=True) for name, array in arrays.items()
    }
    changed_mesh_arrays[coordinate_arrays[0].name][2, 1] += 0.05
    mesh_directory = tmp_path / "changed-background-mesh-coordinates"
    mesh_directory.mkdir()
    write_model_archive(
        mesh_directory / "meshing-specs.phx",
        manifest=manifest,
        arrays=changed_mesh_arrays,
        limits=None,
    )
    refused_mesh = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-meshing-specs",
            str(mesh_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_mesh.returncode != 0
    assert "incompatible with its registered recipe" in refused_mesh.stderr


def test_analytic_implicit_profile_archive_rebuilds_query_cold(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [sys.executable, str(_WORKER), "write-analytic-profile", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-analytic-profile",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert restored.stdout.strip() == "analytic-profile-cold-restored"

    source = tmp_path / "analytic-profile.phx"
    corrupt_directory = tmp_path / "changed-analytic-profile-identity"
    corrupt_directory.mkdir()
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    def replace_profile_identity(value: object) -> bool:
        if isinstance(value, dict):
            if value.get("kind") == "namedtuple" and str(value.get("type", "")).endswith(
                ":AnalyticImplicitProfileArchiveRecipe"
            ):
                identity = value["items"][-1]["value"]
                value["items"][-1]["value"] = (
                    "0" * 64 if identity != "0" * 64 else "1" * 64
                )
                return True
            return any(replace_profile_identity(item) for item in value.values())
        if isinstance(value, list):
            return any(replace_profile_identity(item) for item in value)
        return False

    assert replace_profile_identity(manifest["model_recipe"])
    write_model_archive(
        corrupt_directory / "analytic-profile.phx",
        manifest=manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-analytic-profile",
            str(corrupt_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_surface_chart_boundary_query_archive_cold_restores_sphere_authority(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "write-surface-chart-query",
            str(tmp_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-surface-chart-query",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    continuation = json.loads(restored.stdout)
    assert continuation["restored_complete"] is True
    assert continuation["restored_semantics"] == "certified"
    assert continuation["restored_chart_count"] > 0
    assert expected["source_result_id"] != expected["result_id"]
    assert expected["source_domain_id"] != expected["material_domain_id"]
    assert expected["source_query_id"]
    assert expected["deformation_id"]

    manifest, arrays = read_model_archive(
        tmp_path / "surface-chart-boundary-query.phx", limits=MODEL_ARCHIVE_LIMITS
    )

    def find_query_recipe(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if node.get("kind") == "namedtuple" and str(node.get("type", "")).endswith(
                ":SurfaceChartBoundarySourceArchiveRecipe"
            ):
                return node
            for child in node.values():
                found = find_query_recipe(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_query_recipe(child)
                if found is not None:
                    return found
        return None

    recipe = find_query_recipe(manifest["model_recipe"])
    assert recipe is not None
    query_identity = recipe["items"][6]
    assert query_identity["kind"] == "literal"
    query_identity["value"] = "tampered-surface-query-identity"
    corrupt_directory = tmp_path / "changed-surface-chart-query-identity"
    corrupt_directory.mkdir()
    write_model_archive(
        corrupt_directory / "surface-chart-boundary-query.phx",
        manifest=manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-surface-chart-query",
            str(corrupt_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0


def test_brep_archive_rejects_tampered_source_piece_witness(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [sys.executable, str(_WORKER), "write", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    corrupt_directory = tmp_path / "changed-source-piece-witness"
    corrupt_directory.mkdir()
    shutil.copyfile(tmp_path / "refined.phx", corrupt_directory / "refined.phx")
    manifest, arrays = read_model_archive(
        tmp_path / "surface-chart-witness.phx", limits=MODEL_ARCHIVE_LIMITS
    )

    pieces = manifest["model_recipe"]["fields"]["source_piece_ids"]["items"]
    pieces[0]["items"][-1]["value"] += 1
    write_model_archive(
        corrupt_directory / "surface-chart-witness.phx",
        manifest=manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-surface-witness",
            str(corrupt_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_brep_archive_rejects_tampered_exact_surface_chart_bank(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [sys.executable, str(_WORKER), "write", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    corrupt_directory = tmp_path / "changed-surface-chart-rational-bank"
    corrupt_directory.mkdir()
    shutil.copyfile(tmp_path / "refined.phx", corrupt_directory / "refined.phx")
    manifest, arrays = read_model_archive(
        tmp_path / "surface-chart-witness.phx", limits=MODEL_ARCHIVE_LIMITS
    )

    token_bank = manifest["model_recipe"]["fields"]["chart_rational_tokens"]

    def tamper_first_integer(node: object) -> bool:
        if isinstance(node, dict):
            if node.get("kind") == "literal" and type(node.get("value")) is int:
                node["value"] += 1
                return True
            return any(tamper_first_integer(item) for item in node.values())
        if isinstance(node, list):
            return any(tamper_first_integer(item) for item in node)
        return False

    assert tamper_first_integer(token_bank)
    write_model_archive(
        corrupt_directory / "surface-chart-witness.phx",
        manifest=manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-surface-witness",
            str(corrupt_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_brep_archive_rejects_tampered_surface_root_mesh_binding(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [sys.executable, str(_WORKER), "write", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    corrupt_directory = tmp_path / "changed-surface-root-mesh-binding"
    corrupt_directory.mkdir()
    shutil.copyfile(tmp_path / "refined.phx", corrupt_directory / "refined.phx")
    manifest, arrays = read_model_archive(
        tmp_path / "surface-chart-witness.phx", limits=MODEL_ARCHIVE_LIMITS
    )

    def find_surface_witness(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if node.get("kind") == "dataclass" and str(node.get("type", "")).endswith(
                ":SurfaceChartWitness"
            ):
                return node
            for child in node.values():
                found = find_surface_witness(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_surface_witness(child)
                if found is not None:
                    return found
        return None

    witness = find_surface_witness(manifest["model_recipe"])
    assert witness is not None and "root_mesh_cell_bindings" in witness["fields"]
    bindings = witness["fields"]["root_mesh_cell_bindings"]

    def tamper_binding(node: object) -> bool:
        if isinstance(node, dict):
            if node.get("kind") == "namedtuple" and str(node.get("type", "")).endswith(
                ":SurfaceRootMeshCellBinding"
            ):
                cell_id = node["items"][2]
                if cell_id.get("kind") == "literal" and type(cell_id.get("value")) is int:
                    cell_id["value"] += 1
                    return True
            if node.get("kind") == "literal" and node.get("value") is None:
                node["value"] = "fabricated-root-binding"
                return True
            return any(tamper_binding(child) for child in node.values())
        if isinstance(node, list):
            return any(tamper_binding(child) for child in node)
        return False

    assert tamper_binding(bindings)

    write_model_archive(
        corrupt_directory / "surface-chart-witness.phx",
        manifest=manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-surface-witness",
            str(corrupt_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_brep_entity_archive_revalidates_source_owner() -> None:
    model = brep_sphere(1.0, coordinate_contract=SpatialCoordinateContract.si())
    entity = BRepEntityId(model.source_id, model.source_revision, "face", 0)
    rebuilt_model, rebuilt_entity = _reconstruct_brep_graph((model, entity))
    assert rebuilt_model.source_id == model.source_id
    assert rebuilt_model.source_revision == model.source_revision
    assert rebuilt_entity == entity

    foreign_entity = BRepEntityId("foreign-source", model.source_revision, "face", 0)
    with pytest.raises(ValueError, match="no retained owning source model"):
        _reconstruct_brep_graph((model, foreign_entity))


def _write_pwl_boundary_query(directory: Path, /) -> Path:
    """Write the PWL query archive; return its per-entity oracle as a JSON file."""
    written = subprocess.run(
        [sys.executable, str(_WORKER), "write-pwl-boundary-query", str(directory)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = directory / "pwl-boundary-query-expected.json"
    expected.write_text(written.stdout, encoding="utf-8")
    return expected


def _read_pwl_boundary_query(
    directory: Path, expected: Path, /, *, check: bool
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-pwl-boundary-query",
            str(directory),
            str(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def test_piecewise_boundary_query_archive_cold_rebuilds_source_owner(
    tmp_path: Path,
) -> None:
    expected = _write_pwl_boundary_query(tmp_path)
    restored = _read_pwl_boundary_query(tmp_path, expected, check=True)
    assert restored.stdout.strip() == "pwl-boundary-query-cold-restored"


@pytest.mark.parametrize("tampered", ("query-id", "entity-ids"))
def test_piecewise_boundary_query_archive_rejects_tampered_chart_ancestry(
    tmp_path: Path, tampered: str
) -> None:
    expected = _write_pwl_boundary_query(tmp_path)
    oracle = json.loads(expected.read_text(encoding="utf-8"))
    source = tmp_path / "pwl-boundary-query.phx"
    corrupted_directory = tmp_path / f"changed-pwl-boundary-{tampered}"
    corrupted_directory.mkdir()
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    def find_query(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") == "namedtuple"
                and str(node.get("type", "")).endswith(
                    ":PiecewiseLinearBoundarySourceArchiveRecipe"
                )
                and isinstance(node.get("items"), list)
                and node["items"][1].get("kind") == "literal"
                and node["items"][1].get("value") == 2
            ):
                return node
            for child in node.values():
                found = find_query(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_query(child)
                if found is not None:
                    return found
        return None

    recipe = find_query(manifest["model_recipe"])
    assert recipe is not None
    if tampered == "query-id":
        # The query ID fingerprints the selected entities and their source keys,
        # from which the owner recomputes every chart ID.
        field = recipe["items"][3]
        assert field["kind"] == "literal" and isinstance(field["value"], str)
        field["value"] += ":tampered"
    else:
        # Same selected set, different stored order: the owner canonicalizes the
        # selection, so the stored chart order no longer matches its rebuild.
        facet_ids = next(
            query["entity_ids"] for query in oracle["queries"] if query["dimension"] == 2
        )
        names = [
            name
            for name, array in arrays.items()
            if np.array_equal(np.asarray(array), np.asarray(facet_ids))
        ]
        assert len(names) == 1 and len(facet_ids) > 1
        arrays[names[0]] = np.ascontiguousarray(np.asarray(arrays[names[0]])[::-1])
    write_model_archive(
        corrupted_directory / "pwl-boundary-query.phx",
        manifest=manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    refused = _read_pwl_boundary_query(corrupted_directory, expected, check=False)
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr
    assert "PWL boundary query ancestry differs" in refused.stderr


def test_reference_chart_rational_archive_cold_restores_exact_maps(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "write-reference-chart-arguments",
            str(tmp_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-reference-chart-arguments",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert restored.stdout.strip() == "reference-chart-arguments-cold-restored"


def test_reference_chart_rational_archive_rejects_tampered_expression(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "write-reference-chart-arguments",
            str(tmp_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(written.stdout)
    source = tmp_path / "reference-chart-arguments.phx"
    corrupted = tmp_path / "changed-reference-chart-arguments.phx"
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    def find_rational(node: object) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if node.get("kind") == "rational_polynomial":
                return node
            for child in node.values():
                found = find_rational(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_rational(child)
                if found is not None:
                    return found
        return None

    polynomial = find_rational(manifest["model_recipe"])
    assert polynomial is not None
    coefficient = polynomial["numerator"]["items"][0][1]
    assert coefficient["kind"] == "fraction"
    coefficient["numerator"] += 1
    write_model_archive(
        corrupted, manifest=manifest, arrays=arrays, limits=MODEL_ARCHIVE_LIMITS
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-reference-chart-arguments",
            str(corrupted),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_finite_element_transfer_archive_rebuilds_sparse_evidence_cold(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [sys.executable, str(_WORKER), "write-finite-element-transfer", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert written.returncode == 0, written.stderr
    expected = json.loads(written.stdout)
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-finite-element-transfer",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert restored.returncode == 0, restored.stderr
    assert restored.stdout.strip() == "finite-element-transfer-cold-restored"

    source = tmp_path / "finite-element-transfer.phx"
    corrupted = tmp_path / "changed-finite-element-transfer-coefficient"
    corrupted.mkdir()
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)
    _register_model_archive_types()
    # Tamper a transfer-map coefficient: the entry evidence binds it. Space
    # pairings also carry coefficient leaves, so select the map path exactly.
    coefficient_leaf = next(
        (
            entry
            for entry in model_recipe_array_inventory(
                manifest["model_recipe"], prefix="model/leaves", limits=None
            )
            if entry.path.startswith(".complex_map.maps[")
            and entry.path.endswith("].coefficients")
            and arrays[entry.name].size
        ),
        None,
    )
    assert coefficient_leaf is not None
    damaged_arrays = {name: array.copy() for name, array in arrays.items()}
    damaged_arrays[coefficient_leaf.name].flat[0] += 1.0
    write_model_archive(
        corrupted / "finite-element-transfer.phx",
        manifest=manifest,
        arrays=damaged_arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-finite-element-transfer",
            str(corrupted),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr

    # Space-pairing mass coefficients sit outside every sparse operator ID; the
    # archived pairing binds its operator content, so tampering one is refused.
    pairing_leaf = next(
        (
            entry
            for entry in model_recipe_array_inventory(
                manifest["model_recipe"], prefix="model/leaves", limits=None
            )
            if entry.path.startswith(".complex_map.target.spaces[")
            and entry.path.endswith("].pairing.operator.coefficients")
            and arrays[entry.name].size
        ),
        None,
    )
    assert pairing_leaf is not None
    pairing_corrupted = tmp_path / "changed-finite-element-space-pairing"
    pairing_corrupted.mkdir()
    pairing_arrays = {name: array.copy() for name, array in arrays.items()}
    pairing_arrays[pairing_leaf.name].flat[0] *= 2.0
    write_array_archive(
        pairing_corrupted / "finite-element-transfer.phx",
        manifest=manifest,
        arrays=pairing_arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    pairing_refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-finite-element-transfer",
            str(pairing_corrupted),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert pairing_refused.returncode != 0
    assert "Stored Riesz operator content changed." in pairing_refused.stderr


def test_vertex_interpolation_stencil_archive_preserves_exact_fraction_weights_cold(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [sys.executable, str(_WORKER), "write-vertex-stencil", str(tmp_path)],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert written.returncode == 0, written.stderr
    expected = json.loads(written.stdout)
    assert expected["exact_weights"] == [[[1, 3], [2, 3]]]
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-vertex-stencil",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert restored.returncode == 0, restored.stderr
    assert restored.stdout.strip() == "vertex-interpolation-stencil-cold-restored"

    source = tmp_path / "vertex-interpolation-stencil.phx"
    manifest, arrays = read_model_archive(source, limits=None)

    recipe = manifest["model_recipe"]
    assert str(recipe["type"]).endswith(":VertexInterpolationStencil")
    first_weight = recipe["fields"]["exact_weights"]["items"][0]["items"][0]
    assert first_weight["kind"] == "fraction"
    assert first_weight["numerator"] == 1
    assert first_weight["denominator"] == 3
    first_weight["numerator"] = 2
    corrupted_directory = tmp_path / "changed-exact-stencil-weight"
    corrupted_directory.mkdir()
    write_model_archive(
        corrupted_directory / "vertex-interpolation-stencil.phx",
        manifest=manifest,
        arrays=arrays,
        limits=None,
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-vertex-stencil",
            str(corrupted_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_periodic_distributed_fe_transfer_rebuilds_from_mesh_and_rejects_corruption(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "write-periodic-distributed-fe-transfer",
            str(tmp_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert written.returncode == 0, written.stderr
    expected = json.loads(written.stdout)
    assert expected["settings"] == ["trimmed", 1, "trimmed", 2, "relative", 2]
    assert expected["periodic_transfer_id"]
    assert expected["distributed_transfer_id"]
    source = tmp_path / "periodic-distributed-fe-transfer.phx"
    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-periodic-distributed-fe-transfer",
            str(source),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert restored.returncode == 0, restored.stderr
    assert restored.stdout.strip() == "periodic-distributed-fe-transfer-cold-restored"

    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)

    _register_model_archive_types()
    inventory = model_recipe_array_inventory(
        manifest["model_recipe"], prefix="model/leaves", limits=None
    )
    # Derived quotient operators and evidence are recomputed from the mesh and
    # settings on restore; only the authoritative owner rows are stored.
    assert not any("coefficients" in entry.path for entry in inventory)
    owner_leaf = next(
        (
            entry
            for entry in inventory
            if "source_owners" in entry.path and arrays[entry.name].size
        ),
        None,
    )
    assert owner_leaf is not None
    damaged_arrays = {name: array.copy() for name, array in arrays.items()}
    damaged_owners = damaged_arrays[owner_leaf.name]
    damaged_owners.flat[0] = (damaged_owners.flat[0] + 1) % expected["settings"][5]
    corrupted = tmp_path / "changed-periodic-distributed-fe-owner-row.phx"
    write_model_archive(
        corrupted,
        manifest=manifest,
        arrays=damaged_arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-periodic-distributed-fe-transfer",
            str(corrupted),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_brep_assembly_placement_evidence_archive_round_trips_cold_and_refuses_tamper(
    tmp_path: Path,
) -> None:
    written = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "write-brep-assembly-placement-evidence",
            str(tmp_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert written.returncode == 0, written.stderr
    expected = json.loads(written.stdout)
    assert expected["occurrence_evidence"][0][0] == ["placed-box"]
    bounds = expected["occurrence_evidence"][0][1]
    assert bounds["model_extent"] > 0.0
    assert bounds["rotation_deviation_bound"] > 0.0
    assert bounds["point_deviation_bound"] > 0.0

    restored = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-brep-assembly-placement-evidence",
            str(tmp_path),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert restored.returncode == 0, restored.stderr
    assert restored.stdout.strip() == "brep-assembly-placement-evidence-cold-restored"

    source = tmp_path / "brep-assembly-placement-evidence.phx"
    manifest, arrays = read_model_archive(source, limits=None)

    def find_placement_evidence(node: object, /) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if (
                node.get("kind") == "dataclass"
                and str(node.get("type", "")).rsplit(":", 1)[-1]
                == "BRepPlacementEvidence"
            ):
                return node
            for child in node.values():
                found = find_placement_evidence(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find_placement_evidence(child)
                if found is not None:
                    return found
        return None

    evidence_node = find_placement_evidence(manifest["model_recipe"])
    assert evidence_node is not None
    point_bound = evidence_node["fields"]["point_deviation_bound"]
    assert isinstance(point_bound, dict)
    assert type(point_bound.get("value")) is float
    point_bound["value"] += 1.0e-15
    tampered_directory = tmp_path / "tampered-placement-evidence"
    tampered_directory.mkdir()
    write_model_archive(
        tampered_directory / source.name,
        manifest=manifest,
        arrays=arrays,
        limits=None,
    )
    refused = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "read-brep-assembly-placement-evidence",
            str(tampered_directory),
            json.dumps(expected),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr
