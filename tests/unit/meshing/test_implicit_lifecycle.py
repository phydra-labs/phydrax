# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax._meshcore import meshcore_available
from phydrax.lifecycle._meshing_source_families import is_native_generation_source
from phydrax.meshing.qualification import (
    envelope_transport_field_templates,
    prepare_envelope_accepted_epoch,
)
from phydrax.units import METER
from tests._support.native_family_generation import native_family_generation


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)


def _carrier(shift: float = 0.0) -> phx.meshing.CellMeshingResult:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    points[:, 0] += shift
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        points, np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    )
    return phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())


@pytest.fixture(scope="module")
def native_generation() -> tuple[
    phx.meshing.CellMeshingResult,
    phx.meshing.NativePlcSource,
    phx.meshing.VolumeMeshingSpec,
    phx.meshing.NativeMeshingOptions,
]:
    part, source, request, options = native_family_generation("plc")
    assert isinstance(part.carrier, phx.meshing.CellMeshingResult)
    assert isinstance(source, phx.meshing.NativePlcSource)
    assert isinstance(request, phx.meshing.VolumeMeshingSpec)
    return part.carrier, source, request, options


@pytest.fixture(scope="module")
def accepted_scalar(
    native_generation: tuple[
        phx.meshing.CellMeshingResult,
        phx.meshing.NativePlcSource,
        phx.meshing.VolumeMeshingSpec,
        phx.meshing.NativeMeshingOptions,
    ],
) -> tuple[
    phx.meshing.CellMeshingResult,
    phx.discretization.FiniteElementDiscretization,
    Array,
    tuple[Array, Array],
]:
    carrier = native_generation[0]
    scalar = phx.discretization.FiniteElementPlan(
        carrier.mesh,
        envelope_transport_field_templates("tetrahedron")[0][1],
        coordinate_spec=carrier.geometry,
    ).prepare()
    coordinates = scalar.dof_maps[0].dof_coordinates
    values = jnp.sum(coordinates**2, axis=-1)
    auxiliary = (
        jnp.full(values.shape, 2.0, dtype=jnp.float64),
        coordinates[:, 0] + 0.5,
    )
    return carrier, scalar, values, auxiliary


def test_accepted_epoch_binds_existing_field_templates_and_restores_history(
    accepted_scalar: tuple[
        phx.meshing.CellMeshingResult,
        phx.discretization.FiniteElementDiscretization,
        Array,
        tuple[Array, Array],
    ],
    tmp_path: Path,
    native_generation: tuple[
        phx.meshing.CellMeshingResult,
        phx.meshing.NativePlcSource,
        phx.meshing.VolumeMeshingSpec,
        phx.meshing.NativeMeshingOptions,
    ],
) -> None:
    carrier, scalar, values, auxiliary = accepted_scalar
    accepted = prepare_envelope_accepted_epoch(
        carrier, scalar, values, auxiliary, 0, part_name="implicit"
    )
    declaration = accepted.declarations[0]
    assert isinstance(declaration, phx.lifecycle.MeshingFiniteElementFieldDeclaration)
    expected = tuple(
        field for _, field in envelope_transport_field_templates("tetrahedron")
    )
    assert tuple(field.field_spec_id for field in declaration.finite_element_fields) == (
        tuple(field.field_spec_id for field in expected)
    )
    owner = phx.discretization.FiniteElementPlan(
        carrier.mesh, expected, coordinate_spec=carrier.geometry
    ).prepare()
    assert owner.dof_maps[0].dof_map_id == scalar.dof_maps[0].dof_map_id
    assert declaration.field_space_ids["density"] == owner.field_spaces[1].field_space_id
    assert owner.field_spaces[1].layout.size == carrier.mesh.entity_set(0).count
    assert declaration.geometry_layout_id == carrier.geometry.geometry_layout_id
    assert declaration.history_policy == "material-history"
    history = tuple(
        role for role in declaration.state_roles if role.role == "material-history"
    )
    assert len(history) == 1 and history[0].field_name == "history"
    np.testing.assert_array_equal(accepted.fields[history[0].entry_name], auxiliary[1])
    certification = carrier.certification
    assert certification is not None
    records = {
        "certification_inputs": certification.request,
        "report": certification,
        "associations": carrier.associations,
        "generation_source": native_generation[1],
        "generation_specification": native_generation[2],
        "generation_options": native_generation[3],
        "accepted_target": carrier,
        "accepted_data": {"accepted_epochs": (accepted,)},
    }
    receipt = phx.lifecycle.write_meshing_source_closure(tmp_path / "accepted", records)
    restored = phx.lifecycle.read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    restored_epoch = phx.lifecycle.validate_meshing_accepted_epochs(restored)[0]
    assert restored_epoch.epoch_id == accepted.epoch_id
    assert restored_epoch.carrier.result_id == carrier.result_id
    for name in ("u", "density", "history", "accepted-history"):
        np.testing.assert_array_equal(restored_epoch.fields[name], accepted.fields[name])


@pytest.mark.parametrize("mismatch", ("carrier", "element", "geometry"))
def test_accepted_epoch_refuses_a_foreign_scalar_coefficient_owner(
    accepted_scalar: tuple[
        phx.meshing.CellMeshingResult,
        phx.discretization.FiniteElementDiscretization,
        Array,
        tuple[Array, Array],
    ],
    mismatch: str,
) -> None:
    carrier, _, values, auxiliary = accepted_scalar
    foreign = _carrier(0.25) if mismatch == "carrier" else carrier
    degree = 2 if mismatch == "element" else 1
    space = phx.discretization.FiniteElementPlan(
        foreign.mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("tetrahedron", degree)
        ),
        coordinate_spec=foreign.geometry,
    ).prepare()
    if mismatch == "geometry":
        space = eqx.tree_at(
            lambda value: value.coordinate_spec.coordinates,
            space,
            space.coordinate_spec.coordinates
            + jnp.asarray((0.125, 0.0, 0.0), dtype=jnp.float64),
        )
    with pytest.raises(ValueError, match="accepted_epoch.scalar_space"):
        prepare_envelope_accepted_epoch(
            carrier, space, values, auxiliary, 0, part_name="implicit"
        )


def test_accepted_epoch_refuses_a_truncated_density_bank(
    accepted_scalar: tuple[
        phx.meshing.CellMeshingResult,
        phx.discretization.FiniteElementDiscretization,
        Array,
        tuple[Array, Array],
    ],
) -> None:
    carrier, scalar, values, auxiliary = accepted_scalar
    with pytest.raises(ValueError):
        prepare_envelope_accepted_epoch(
            carrier,
            scalar,
            values,
            (auxiliary[0][:-1], auxiliary[1]),
            0,
            part_name="implicit",
        )


@pytest.fixture(scope="module")
def implicit_source() -> phx.meshing.NativeImplicitSource:
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 0.371, feature_id="source-family-sphere"
    ).compile(coordinate_unit=METER)
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(3) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((-0.5,) * 3, (0.5,) * 3), dtype=jnp.float64))
    return phx.meshing.NativeImplicitSource(
        geometry, grid, "source-family-sphere", "authored-r1"
    )


def test_implicit_generation_source_admission_survives_source_archive_restore(
    implicit_source: phx.meshing.NativeImplicitSource, tmp_path: Path
) -> None:
    receipt = phx.lifecycle.write_meshing_source_closure(
        tmp_path / "implicit-source", implicit_source
    )
    restored = phx.lifecycle.read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert isinstance(restored, phx.meshing.NativeImplicitSource)
    assert restored.source_id == implicit_source.source_id
    assert restored.source_revision == implicit_source.source_revision
    assert restored.grid.prepared_id == implicit_source.grid.prepared_id
    assert phx.array_tree_fingerprint(restored) == phx.array_tree_fingerprint(
        implicit_source
    )
    assert is_native_generation_source(implicit_source)
    assert is_native_generation_source(restored)


@pytest.mark.parametrize("kind", ("geometry", "lookalike"))
def test_implicit_generation_admission_requires_the_declared_source_owner(
    implicit_source: phx.meshing.NativeImplicitSource, kind: str
) -> None:
    candidate = (
        implicit_source.geometry
        if kind == "geometry"
        else SimpleNamespace(
            geometry=implicit_source.geometry,
            grid=implicit_source.grid,
            source_id=implicit_source.source_id,
            source_revision=implicit_source.source_revision,
        )
    )
    assert not is_native_generation_source(candidate)


@pytest.fixture(scope="module")
def analytic_generation() -> tuple[
    phx.meshing.NativeImplicitSource,
    phx.meshing.CellMeshingResult,
    phx.meshing.VolumeMeshingSpec,
    phx.meshing.NativeMeshingOptions,
]:
    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification.scenarios.implicit import (
        _implicit_qualification_fixture,
    )

    _, source, specification, options, _, _ = _implicit_qualification_fixture(
        QualificationControls()
    )
    result = (
        phx.meshing.NativeMeshingProvider(options)
        .plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    assert result.certification is not None and result.certification.passed
    return source, result, specification, options


def test_registered_implicit_generation_restores_field_grid_and_raw_controls(
    analytic_generation: tuple[
        phx.meshing.NativeImplicitSource,
        phx.meshing.CellMeshingResult,
        phx.meshing.VolumeMeshingSpec,
        phx.meshing.NativeMeshingOptions,
    ],
    tmp_path: Path,
) -> None:
    from phydrax.lifecycle import read_meshing_source_execution_controls
    from tests._support.forced_devices import run_on_forced_devices

    source, result, specification, options = analytic_generation
    report = result.certification
    assert report is not None
    records = {
        "certification_inputs": report.request,
        "report": report,
        "associations": result.associations,
        "generation_part": phx.meshing.MeshPart("implicit", result),
        "generation_source": source,
        "generation_specification": specification,
        "generation_options": options,
    }
    receipt = phx.lifecycle.write_meshing_source_closure(
        tmp_path / "registered-implicit", records
    )
    controls = read_meshing_source_execution_controls(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert controls.source_id == source.source_id
    assert controls.source_revision == source.source_revision
    assert controls.specification_id == specification.specification_id
    assert controls.source_binding_id
    assert controls.limits.limits_id == specification.limits.limits_id
    assert result.execution_evidence is not None
    np.testing.assert_array_equal(
        controls.execution_evidence.receipt_id, result.execution_evidence.receipt_id
    )
    cold = run_on_forced_devices(
        "tests._support.discretization_process_scenarios",
        "cold_native_implicit_generation",
        (
            str(receipt.path),
            receipt.content_id,
            source.source_id,
            result.result_id,
            source.grid.prepared_id,
            controls.source_binding_id,
        ),
        devices=1,
        env={},
        timeout_seconds=180,
    )
    assert all(cold.values())


@pytest.mark.parametrize("foreign", ("geometry", "grid", "options"))
def test_registered_implicit_generation_refuses_foreign_source_controls(
    analytic_generation: tuple[
        phx.meshing.NativeImplicitSource,
        phx.meshing.CellMeshingResult,
        phx.meshing.VolumeMeshingSpec,
        phx.meshing.NativeMeshingOptions,
    ],
    foreign: str,
) -> None:
    from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
    from phydrax.geometry.design import ParameterId
    from phydrax.lifecycle._meshing_sources import validate_registered_native_part

    source, result, specification, options = analytic_generation
    match foreign:
        case "geometry":
            changed = source.geometry.with_parameters(
                {ParameterId("qualification-implicit-sphere", "radius"): 0.4}
            )
            source = phx.meshing.NativeImplicitSource(
                changed, source.grid, source.source_id, source.source_revision
            )
        case "grid":
            grid = phx.discretization.TensorGridPlan(
                tuple(phx.discretization.UniformAxisSpec(5) for _ in range(3)),
                axis_names=("x", "y", "z"),
            ).prepare(jnp.asarray(((-0.6,) * 3, (0.6,) * 3), dtype=jnp.float64))
            source = phx.meshing.NativeImplicitSource(
                source.geometry, grid, source.source_id, source.source_revision
            )
        case "options":
            options = phx.meshing.NativeMeshingOptions(
                "implicit_restricted_delaunay",
                volume_schedule=phx.meshing.NativeVolumeSchedule(
                    radius_edge_bound=2.0,
                    minimum_dihedral_degrees=10.0,
                    refinement_rounds=7,
                    improvement_passes=8,
                ),
            )
        case _:
            raise ValueError(
                f"implicit_generation.foreign must name a source control; got {foreign!r}"
            )
    with pytest.raises(ValueError, match="source-bound generation plan|retained field"):
        validate_registered_native_part(
            phx.meshing.MeshPart("implicit", result),
            source,
            specification,
            options,
            limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
        )


def test_native_plc_raw_controls_preserve_declared_binding(
    native_generation: tuple[
        phx.meshing.CellMeshingResult,
        phx.meshing.NativePlcSource,
        phx.meshing.VolumeMeshingSpec,
        phx.meshing.NativeMeshingOptions,
    ],
    tmp_path: Path,
) -> None:
    from phydrax.lifecycle import read_meshing_source_execution_controls

    result, source, specification, options = native_generation
    report = result.certification
    assert report is not None
    records = {
        "certification_inputs": report.request,
        "report": report,
        "associations": result.associations,
        "generation_part": phx.meshing.MeshPart("plc", result),
        "generation_source": source,
        "generation_specification": specification,
        "generation_options": options,
    }
    receipt = phx.lifecycle.write_meshing_source_closure(
        tmp_path / "plc-controls", records
    )
    controls = read_meshing_source_execution_controls(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert controls.source_binding_id == source.binding_id
    assert controls.specification_id == specification.specification_id


def test_raw_implicit_controls_refuse_a_foreign_published_specification(
    analytic_generation: tuple[
        phx.meshing.NativeImplicitSource,
        phx.meshing.CellMeshingResult,
        phx.meshing.VolumeMeshingSpec,
        phx.meshing.NativeMeshingOptions,
    ],
) -> None:
    from phydrax._array_archive import (
        ArrayArchiveCorruptionError,
        DEFAULT_ARRAY_ARCHIVE_LIMITS,
    )
    from phydrax._model._structure import (
        model_recipe_array_values,
        model_structure_recipe,
    )
    from phydrax.lifecycle._meshing_sources import (
        _require_execution_control_binding,
        _restore_execution_controls,
        _select_execution_controls,
    )

    source, result, specification, options = analytic_generation
    report = result.certification
    assert report is not None
    records = {
        "certification_inputs": report.request,
        "report": report,
        "associations": result.associations,
        "generation_part": phx.meshing.MeshPart("implicit", result),
        "generation_source": source,
        "generation_specification": specification,
        "generation_options": options,
    }
    recipe = model_structure_recipe(records)
    arrays = model_recipe_array_values(records, recipe, prefix="source-record")
    selection = _select_execution_controls(recipe, DEFAULT_ARRAY_ARCHIVE_LIMITS)
    published = selection.selected["published_specification_id"]
    selection.table.nodes[published] = {
        "kind": "literal",
        "value": "foreign-published-specification",
    }
    controls = _restore_execution_controls(
        selection, {name: np.asarray(value) for name, value in arrays.items()}
    )
    with pytest.raises(ArrayArchiveCorruptionError, match="another plan/specification"):
        _require_execution_control_binding(controls)
