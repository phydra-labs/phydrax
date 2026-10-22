# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import jax
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
from tools.meshing_qualification._controls import ImplicitCase


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


@pytest.mark.parametrize("degree", (1, 2))
def test_accepted_epoch_preserves_actual_mixed_degree_owners_on_explicit_budget(
    native_generation: tuple[
        phx.meshing.CellMeshingResult, phx.meshing.NativePlcSource,
        phx.meshing.VolumeMeshingSpec, phx.meshing.NativeMeshingOptions,
    ],
    degree: int,
) -> None:
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget

    carrier = native_generation[0]
    fields = tuple(
        field for _, field in envelope_transport_field_templates(
            "tetrahedron", solution_degree=degree,
        )
    )
    with HostBatchTicket(2_000_000, bytes_upper=1 << 26, queries_upper=2_000_000) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=2_000_000, maximum_memory_bytes=1 << 26,
        ).activate() as budget:
            transport = phx.discretization.FiniteElementPlan(
                carrier.mesh, fields, coordinate_spec=carrier.geometry, budget=budget,
            ).prepare(budget=budget)
            scalar = phx.discretization.FiniteElementPlan(
                carrier.mesh, fields[0], coordinate_spec=carrier.geometry, budget=budget,
            ).prepare(budget=budget)
            coordinates = scalar.dof_maps[0].dof_coordinates
            values = jnp.sum(coordinates**2, axis=-1)
            auxiliary = (
                jnp.full((transport.dof_maps[1].global_dof_count,), 2.0),
                coordinates[:, 0] + 0.5,
            )
            accepted = prepare_envelope_accepted_epoch(
                carrier, scalar, values, auxiliary, 0, part_name="implicit",
                transport_space=transport, budget=budget,
            )
    declaration = accepted.declarations[0]
    assert isinstance(declaration, phx.lifecycle.MeshingFiniteElementFieldDeclaration)
    assert tuple(field.field_spec_id for field in declaration.finite_element_fields) == (
        tuple(field.field_spec_id for field in fields)
    )
    assert accepted.fields["u"] is values
    assert accepted.fields["density"] is auxiliary[0]
    assert accepted.fields["history"] is auxiliary[1]


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
    controls = phx.lifecycle.read_meshing_source_execution_controls(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert carrier.execution_evidence is not None
    assert (
        controls.execution_evidence.to_record() == carrier.execution_evidence.to_record()
    )
    assert controls.source_binding_id == native_generation[1].binding_id
    assert controls.specification_id == native_generation[2].specification_id


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


@pytest.mark.parametrize("degree", (1, 2))
@pytest.mark.parametrize("extent", ("short", "long", "rank"))
def test_accepted_epoch_refuses_values_outside_declared_scalar_owner(
    accepted_scalar: tuple[
        phx.meshing.CellMeshingResult,
        phx.discretization.FiniteElementDiscretization,
        Array,
        tuple[Array, Array],
    ],
    degree: int,
    extent: str,
) -> None:
    carrier, _, _, auxiliary = accepted_scalar
    scalar = phx.discretization.FiniteElementPlan(
        carrier.mesh,
        envelope_transport_field_templates("tetrahedron", solution_degree=degree)[0][1],
        coordinate_spec=carrier.geometry,
    ).prepare()
    count = scalar.dof_maps[0].global_dof_count
    shape = (count, 1) if extent == "rank" else (
        count + (1 if extent == "long" else -1),
    )
    values = jnp.zeros(shape, dtype=jnp.float64)
    history = jnp.zeros((count,), dtype=jnp.float64)
    with pytest.raises(ValueError, match="accepted_epoch.scalar_space"):
        prepare_envelope_accepted_epoch(
            carrier, scalar, values, (auxiliary[0], history), 0, part_name="implicit"
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


def test_raw_controls_preserve_completed_request_history(
    native_generation: tuple[
        phx.meshing.CellMeshingResult,
        phx.meshing.NativePlcSource,
        phx.meshing.VolumeMeshingSpec,
        phx.meshing.NativeMeshingOptions,
    ],
    tmp_path: Path,
) -> None:
    from phydrax.lifecycle import read_meshing_source_execution_controls
    from phydrax.meshing.qualification._native_lifecycle import execute_native_request
    from tools.meshing_qualification._runtime import qualification_failure

    previous, source, specification, options = native_generation
    plan = phx.meshing.NativeMeshingProvider(options).plan(
        source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
    )
    regenerated = execute_native_request(
        previous,
        specification.limits,
        lambda _budget: plan.execute(),
        target=lambda value: value,
        failure_factory=qualification_failure,
    ).result
    report = regenerated.certification
    execution = regenerated.execution_evidence
    assert report is not None and execution is not None
    assert previous.execution_evidence is not None
    assert tuple(item.receipt_hex() for item in execution.history_evidence) == (
        previous.execution_evidence.receipt_hex(),
    )
    receipt = phx.lifecycle.write_meshing_source_closure(
        tmp_path / "regenerated-plc-controls",
        {
            "certification_inputs": report.request,
            "report": report,
            "associations": regenerated.associations,
            "generation_part": phx.meshing.MeshPart("plc", regenerated),
            "generation_source": source,
            "generation_specification": specification,
            "generation_options": options,
        },
    )
    controls = read_meshing_source_execution_controls(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert controls.execution_evidence.to_record() == execution.to_record()
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
        register_meshing_source_artifacts,
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
    register_meshing_source_artifacts()
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


def _metered_discovery(
    run: Any, work_capacity: int, query_capacity: int
) -> tuple[Any, int, int]:
    """Run one discovery on a fresh request ledger; return its charged totals."""
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget

    with HostBatchTicket(work_capacity, queries_upper=query_capacity) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=work_capacity, maximum_memory_bytes=0
        ).activate() as budget:
            value = run(budget)
        work = work_capacity - ticket.remaining_work_units
        queries = query_capacity - ticket.remaining_geometry_queries
    return value, work, queries


def _counter_boundary_proof(
    run: Any, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, int, int]:
    """Charged totals are the exact capacity boundary of a discovery.

    Conditional stages admit their upper bound before dispatch, so the minimal
    capacity is the larger of the charged total and the peak of (charged so
    far + admitted bound). That capacity succeeds; one unit less of either
    resource refuses with typed resource evidence and publishes nothing.
    """
    import phydrax.geometry.implicit._adaptive_discovery as adaptive
    import phydrax.geometry.implicit._discovery as discovery
    from phydrax._resource_ticket import HostBatchEstimateError
    from phydrax.discretization._coordinate_budget import (
        CoordinateEnclosureResourceError,
    )

    peaks = [0, 0]
    original = discovery._admitted_batch

    def recording(budget: Any, work_upper: int, queries_upper: int, /) -> Any:
        ticket = budget.ticket
        used_work = ticket.work_upper - ticket.remaining_work_units
        used_queries = ticket.queries_upper - ticket.remaining_geometry_queries
        peaks[0] = max(peaks[0], used_work + work_upper)
        peaks[1] = max(peaks[1], used_queries + queries_upper)
        return original(budget, work_upper, queries_upper)

    monkeypatch.setattr(discovery, "_admitted_batch", recording)
    monkeypatch.setattr(adaptive, "_admitted_batch", recording)
    capacity = 10**12
    value, work, queries = _metered_discovery(run, capacity, capacity)
    required_work, required_queries = max(work, peaks[0]), max(queries, peaks[1])
    _, exact_work, exact_queries = _metered_discovery(
        run, required_work, required_queries
    )
    assert (exact_work, exact_queries) == (work, queries)
    for work_capacity, query_capacity in (
        (required_work - 1, required_queries),
        (required_work, required_queries - 1),
    ):
        with pytest.raises((HostBatchEstimateError, CoordinateEnclosureResourceError)):
            _metered_discovery(run, work_capacity, query_capacity)
    return value, work, queries


def test_fixed_lattice_discovery_charges_realized_root_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry.implicit._discovery import (
        _EDGE_ROOT_MAXIMUM_STEPS,
        _QEF_REGULARIZATION_LEVELS,
        implicit_realization_operations,
    )

    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 0.75, feature_id="metered-sphere"
    ).compile(coordinate_unit=METER)
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(9) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[-1.4, -1.4, -1.4], [1.4, 1.4, 1.4]]))
    policy = phx.geometry.ImplicitSurfacePolicy(
        projection=phx.geometry.ImplicitProjectionPolicy(trust_fraction=0.45),
        maximum_intersection_pairs=500_000,
    )
    plan, work, queries = _counter_boundary_proof(
        lambda budget: phx.geometry.discover_implicit_surface(
            geometry, grid, policy=policy, source_id="metered", budget=budget
        ),
        monkeypatch,
    )

    lattice = int(plan.grid_points.shape[0])
    lanes = int(plan.projection.anchors.shape[0])
    faces = int(plan.faces.shape[0])
    vertices = int(plan.base_vertices.shape[0])
    realization_work, realization_queries = implicit_realization_operations(plan)
    assert realization_queries == lattice + lanes * (policy.projection.maximum_steps + 1)
    # Lattice samples, anchor and face-orientation gradients and the base
    # realization are fixed; the remainder is the shared ITP batch, which
    # evaluates every lane per iteration plus one final residual.
    root_queries = queries - lattice - lanes - faces - realization_queries
    assert root_queries % lanes == 0
    iterations = root_queries // lanes - 1
    assert 1 <= iterations < _EDGE_ROOT_MAXIMUM_STEPS
    regularization = np.asarray(plan.qef_regularization)
    levels = 1 + max(
        level
        for level in range(_QEF_REGULARIZATION_LEVELS)
        if np.any(regularization == policy.qef_regularization * 10.0**level)
    )
    # The lattice-window candidate scan must reproduce the BVH self query on
    # the same padded discovery-cell boxes exactly, with fewer box tests.
    from phydrax._bvh import bvh_overlap_pair_blocks, prepare_bvh

    face_rows = np.asarray(plan.faces)
    tolerance = policy.projection.root_tolerance
    box_min = np.min(np.asarray(plan.cell_lower)[face_rows], axis=1) - tolerance
    box_max = np.max(np.asarray(plan.cell_upper)[face_rows], axis=1) + tolerance
    diagonal = float(np.max(np.linalg.norm(box_max - box_min, axis=1)))
    tests: list[int] = []
    hierarchy = prepare_bvh(box_min, box_max, dtype=jnp.float64)
    blocks = list(
        bvh_overlap_pair_blocks(
            hierarchy,
            hierarchy,
            include_touching=True,
            absolute_tolerance=tolerance * (1.0 + 2.0 * diagonal),
            visit=tests.append,
        )
    )
    first = np.concatenate([block[0] for block in blocks])
    second = np.concatenate([block[1] for block in blocks])
    unordered = first < second
    first, second = first[unordered], second[unordered]
    order = np.lexsort((second, first))
    first, second = first[order], second[order]
    shares = np.any(
        face_rows[first][:, :, None] == face_rows[second][:, None, :], axis=(1, 2)
    )
    assert np.array_equal(
        np.asarray(plan.intersection_pairs),
        np.stack((first[~shares], second[~shares]), axis=1),
    )
    # The remainder is one unit per exact candidate box test: at least one per
    # overlapping pair, and fewer than the BVH traversal's AABB tests.
    scan = work - (lanes * iterations + vertices * levels + realization_work)
    assert first.size <= scan < sum(tests)


def test_lattice_candidate_pairs_match_bvh_on_anisotropic_offset_torus() -> None:
    from phydrax._bvh import bvh_overlap_pairs_host, prepare_bvh

    geometry = phx.geometry.Torus(
        (0.031, -0.019, 0.047), 0.35, 0.95, feature_id="candidate-torus"
    ).compile(coordinate_unit=METER)
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(n) for n in (23, 21, 13)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[-1.21, -1.17, -0.43], [1.19, 1.23, 0.41]]))
    policy = phx.geometry.ImplicitSurfacePolicy(
        projection=phx.geometry.ImplicitProjectionPolicy(trust_fraction=0.45),
        maximum_intersection_pairs=2_000_000,
    )
    plan = phx.geometry.discover_implicit_surface(
        geometry, grid, policy=policy, source_id="candidate-torus"
    )
    faces = np.asarray(plan.faces)
    tolerance = policy.projection.root_tolerance
    box_min = np.min(np.asarray(plan.cell_lower)[faces], axis=1) - tolerance
    box_max = np.max(np.asarray(plan.cell_upper)[faces], axis=1) + tolerance
    diagonal = float(np.max(np.linalg.norm(box_max - box_min, axis=1)))
    hierarchy = prepare_bvh(box_min, box_max, dtype=jnp.float64)
    first, second = bvh_overlap_pairs_host(
        hierarchy,
        hierarchy,
        include_touching=True,
        absolute_tolerance=tolerance * (1.0 + 2.0 * diagonal),
    )
    unordered = first < second
    first, second = first[unordered], second[unordered]
    shares = np.any(faces[first][:, :, None] == faces[second][:, None, :], axis=(1, 2))
    assert np.array_equal(
        np.asarray(plan.intersection_pairs),
        np.stack((first[~shares], second[~shares]), axis=1),
    )


def test_adaptive_discovery_charges_realized_root_iterations_and_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax._bvh import bvh_overlap_pair_blocks, prepare_bvh
    from phydrax.geometry.implicit._adaptive_discovery import _PAIR_CHUNK
    from phydrax.geometry.implicit._discovery import _EDGE_ROOT_MAXIMUM_STEPS

    geometry = phx.geometry.Sphere(
        (0.013, -0.021, 0.007), 0.75, feature_id="metered-adaptive"
    ).compile(coordinate_unit=METER)
    domain = 1.4 * np.asarray([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    policy = phx.geometry.AdaptiveImplicitSurfacePolicy(maximum_level=6)
    surface, work, queries = _counter_boundary_proof(
        lambda budget: phx.geometry.discover_adaptive_implicit_surface(
            geometry,
            domain=domain,
            policy=policy,
            source_id="metered-adaptive",
            budget=budget,
        ),
        monkeypatch,
    )
    assert surface.mesh is not None
    evidence = surface.evidence
    lanes = int(evidence.root_solves)
    vertex_count = int(np.asarray(surface.mesh.vertices).shape[0])
    # Extraction evaluations: endpoint samples, the realized shared ITP batch
    # (every lane per iteration plus the final residual), the first jet and
    # the published-vertex residuals.
    root_queries = evidence.extraction_evaluations - 2 * lanes - 4 * lanes - vertex_count
    assert lanes > 0 and root_queries % lanes == 0
    assert 1 <= root_queries // lanes - 1 < _EDGE_ROOT_MAXIMUM_STEPS
    assert queries >= evidence.extraction_evaluations

    # Independent re-traversal of the self-intersection candidates.
    triangles = np.asarray(surface.mesh.vertices)[np.asarray(surface.mesh.faces)]
    tests: list[int] = []
    blocks = list(
        bvh_overlap_pair_blocks(
            prepare_bvh(
                np.min(triangles, axis=1), np.max(triangles, axis=1), dtype=jnp.float64
            ),
            prepare_bvh(
                np.min(triangles, axis=1), np.max(triangles, axis=1), dtype=jnp.float64
            ),
            include_touching=True,
            visit=tests.append,
        )
    )
    faces = np.asarray(surface.mesh.faces)
    first = np.concatenate([block[0] for block in blocks])
    second = np.concatenate([block[1] for block in blocks])
    ordered = first < second
    first, second = first[ordered], second[ordered]
    shares = np.any(faces[first][:, :, None] == faces[second][:, None, :], axis=(1, 2))
    chunks = -(-int(np.sum(~shares)) // _PAIR_CHUNK)
    assert work >= lanes + sum(tests) + chunks * _PAIR_CHUNK


def test_implicit_exact_volumes_match_independent_quadrature() -> None:
    from tools.meshing_qualification.scenarios.implicit import _analytic_volume

    radius, major = 0.371, 0.742
    z = np.linspace(-radius, radius, 400_001)
    half_chord = np.sqrt(np.maximum(radius**2 - z**2, 0.0))
    # Washer areas pi((R + c)^2 - (R - c)^2) and disk areas pi c^2 per slice.
    torus = np.trapezoid(4.0 * np.pi * major * half_chord, z)
    ball = np.trapezoid(np.pi * half_chord**2, z)
    assert _analytic_volume("ring-torus", radius, major) == pytest.approx(
        torus, rel=1.0e-8
    )
    assert _analytic_volume("sphere", radius, 0.0) == pytest.approx(ball, rel=1.0e-8)
    with pytest.raises(ValueError, match="sphere or ring-torus"):
        _analytic_volume("ellipsoid", radius, major)


def test_implicit_acceptance_requires_exact_declared_domain_coverage() -> None:
    from tools.meshing_qualification._runtime import QualificationFailure
    from tools.meshing_qualification.scenarios.implicit import (
        _implicit_qualification_acceptance,
    )

    # A certified approximation or lookalike cannot replace the exact theorem.
    coverage = SimpleNamespace(status="certified", domain_id="same-domain")
    publication: Any = SimpleNamespace(
        audit=SimpleNamespace(passed=True),
        compliance=SimpleNamespace(passed=True),
        certification=SimpleNamespace(
            passed=True,
            embedding=SimpleNamespace(status="certified"),
            coverage=coverage,
            fidelity=SimpleNamespace(status="certified"),
        ),
    )
    unused: Any = None
    with pytest.raises(QualificationFailure, match="declared-domain coverage"):
        _implicit_qualification_acceptance(
            publication, publication, unused, unused, unused
        )


def test_implicit_convergence_order_uses_realized_carrier_sizes() -> None:
    from tools.meshing_qualification.scenarios.implicit import (
        _physical_error_convergence,
        _realized_carrier_size,
    )

    acceptances = (
        {"mesh_volume": 0.2, "cells": 1000},
        {"mesh_volume": 0.2, "cells": 8000},
        {"mesh_volume": 0.2, "cells": 64000},
    )
    sizes = [_realized_carrier_size(value) for value in acceptances]
    assert sizes[0] / sizes[1] == pytest.approx(2.0)
    errors = [3.0 * size**2 for size in sizes]
    converged = _physical_error_convergence(errors, sizes, 1.0, errors[-1])
    assert converged["observed_orders"] == pytest.approx([2.0, 2.0])
    assert converged["order_met"] and converged["strictly_decreasing"]
    assert converged["target_met"]

    stalled = _physical_error_convergence(
        [errors[0], 0.5 * errors[0]], [sizes[0], sizes[0]], 1.0, 1.0
    )
    assert stalled["observed_orders"] == [None]
    assert not stalled["order_met"]
    assert _physical_error_convergence(errors[:2], sizes[:2], None, 0.0)["order_met"]
    assert not _physical_error_convergence(errors[:2], sizes[:2], None, 0.0)["target_met"]


@pytest.mark.parametrize(
    "case", ("sphere", "sphere-refinement", "offset-ring-torus-anisotropic")
)
def test_implicit_domain_probe_boxes_follow_the_authored_field(
    case: ImplicitCase,
) -> None:
    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification.scenarios.implicit import (
        _domain_probes,
        _implicit_case,
        _implicit_qualification_source,
    )

    args = QualificationControls(implicit_case=case)
    declared = _implicit_case(args)
    geometry, _, domain, witness = _implicit_qualification_source(args, declared)
    names, lower, upper, classes = _domain_probes(args, declared, domain)
    assert len(names) == lower.shape[0] == upper.shape[0] == len(classes)
    assert np.all(upper > lower)
    assert float(geometry.boundary_field(jnp.asarray(witness[None, :]))[0]) < 0.0
    offsets = np.stack(
        np.meshgrid(*((np.linspace(0.0, 1.0, 5),) * 3), indexing="ij"), axis=-1
    ).reshape((-1, 3))
    inside = int(phx.geometry.ImplicitVolumeClass.INSIDE)
    for name, low, high, expected in zip(names, lower, upper, classes, strict=True):
        samples = low + offsets * (high - low)
        field = np.asarray(geometry.boundary_field(jnp.asarray(samples)))
        if expected == inside:
            assert np.all(field < 0.0), name
        else:
            assert np.all(field > 0.0), name


def test_implicit_fidelity_obligation_recomputes_offset_volume_bracket() -> None:
    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification.scenarios.implicit import (
        _analytic_volume,
        _implicit_case,
        _source_fidelity_holds,
    )

    args = QualificationControls(implicit_case="offset-ring-torus-anisotropic")
    case = _implicit_case(args)
    radius = args.implicit_radius
    major = case.major_ratio * radius
    epsilon = 0.5 * args.implicit_fidelity
    evidence = {
        "source_family": "ring-torus",
        "source_radius": radius,
        "source_major_radius": major,
        "source_reach_lower": radius,
        "source_fidelity_upper": epsilon,
        "minimum_signed_cell_volume": 1.0e-9,
        "mesh_volume": _analytic_volume("ring-torus", radius, major),
    }
    tolerance = args.implicit_fidelity
    assert _source_fidelity_holds(evidence, tolerance, args, case)
    outer = _analytic_volume("ring-torus", radius + epsilon, major)
    assert not _source_fidelity_holds(
        {**evidence, "mesh_volume": outer * (1.0 + 1.0e-9)}, tolerance, args, case
    )
    assert not _source_fidelity_holds(
        {**evidence, "source_fidelity_upper": 2.0 * tolerance}, tolerance, args, case
    )
    assert not _source_fidelity_holds(
        {**evidence, "source_major_radius": 1.5 * radius}, tolerance, args, case
    )


@pytest.fixture(scope="module")
def retained_surface() -> tuple[
    phx.geometry.CompiledGeometry, phx.geometry.ImplicitSurfacePlan
]:
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 0.75, feature_id="retained-sphere"
    ).compile(coordinate_unit=METER)
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(9) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[-1.4, -1.4, -1.4], [1.4, 1.4, 1.4]]))
    plan = phx.geometry.discover_implicit_surface(
        geometry,
        grid,
        policy=phx.geometry.ImplicitSurfacePolicy(
            projection=phx.geometry.ImplicitProjectionPolicy(trust_fraction=0.45),
            maximum_intersection_pairs=500_000,
        ),
        source_id="retained-sphere",
    )
    return geometry, plan


def _radius_state(geometry: phx.geometry.CompiledGeometry, radius: Any) -> Any:
    index = geometry.schema.index(phx.geometry.ParameterId("retained-sphere", "radius"))
    return geometry.state.replace_at(index, radius)


def test_retained_initial_realization_equals_fresh_realization_bitwise(
    retained_surface: tuple[
        phx.geometry.CompiledGeometry, phx.geometry.ImplicitSurfacePlan
    ],
) -> None:
    geometry, plan = retained_surface
    fresh = plan.realize(geometry.state)
    retained = plan.retained_initial_realization(geometry.state)
    vertices = retained.vertices

    assert plan.binds_initial_state(geometry.state)
    proof_paths = {
        jax.tree_util.keystr(path)
        for path, _ in jax.tree_util.tree_flatten_with_path(plan)[0]
    }
    assert ".initial_evidence.status" in proof_paths
    assert ".initial_evidence.minimum_face_area" in proof_paths
    assert ".initial_evidence.projection.root_residual" in proof_paths
    assert ".initial_vertices" not in proof_paths
    assert np.array_equal(np.asarray(vertices), np.asarray(fresh.vertices))
    fresh_leaves, fresh_structure = jax.tree_util.tree_flatten(fresh.evidence)
    retained_leaves, retained_structure = jax.tree_util.tree_flatten(retained.evidence)
    assert fresh_structure == retained_structure
    for expected, actual in zip(fresh_leaves, retained_leaves, strict=True):
        assert np.array_equal(np.asarray(expected), np.asarray(actual))
    # A changed value, even by one ulp, binds nothing and realizes afresh.
    radius = geometry.state.values[
        geometry.schema.index(phx.geometry.ParameterId("retained-sphere", "radius"))
    ]
    nudged = _radius_state(geometry, jnp.nextafter(radius, jnp.inf))
    assert not plan.binds_initial_state(nudged)
    assert not np.array_equal(
        np.asarray(plan.realize(_radius_state(geometry, 0.76)).vertices),
        np.asarray(vertices),
    )


def test_retained_proof_preserves_the_discovery_fallback_bank(
    retained_surface: tuple[
        phx.geometry.CompiledGeometry, phx.geometry.ImplicitSurfacePlan
    ],
) -> None:
    from phydrax.geometry.implicit._discovery import (
        _QEF_REGULARIZATION_LEVELS,
        field_and_gradient,
        qef_vertices,
    )

    geometry, plan = retained_surface
    _, gradients = field_and_gradient(
        geometry.kernel, geometry.state, plan.projection.anchors
    )
    (discovery_vertices, _), _, _ = _metered_discovery(
        lambda budget: qef_vertices(
            np.asarray(plan.projection.anchors),
            np.asarray(gradients),
            np.asarray(plan.vertex_anchor_indices),
            np.asarray(plan.vertex_anchor_mask),
            np.asarray(plan.cell_lower),
            np.asarray(plan.cell_upper),
            plan.policy.qef_regularization,
            plan.policy.projection.root_tolerance,
            budget=budget,
        ),
        int(plan.base_vertices.shape[0]) * _QEF_REGULARIZATION_LEVELS,
        0,
    )
    assert np.array_equal(discovery_vertices, np.asarray(plan.base_vertices))
    rejected = plan.realize(_radius_state(geometry, 1.2))
    assert not bool(rejected.accepted)
    assert np.array_equal(np.asarray(rejected.vertices), np.asarray(plan.base_vertices))


def test_traced_discovery_state_never_binds_retained_realization(
    retained_surface: tuple[
        phx.geometry.CompiledGeometry, phx.geometry.ImplicitSurfacePlan
    ],
) -> None:
    geometry, plan = retained_surface
    bound: list[bool] = []

    def vertex_sum(radius: Array) -> Array:
        state = _radius_state(geometry, radius)
        bound.append(plan.binds_initial_state(state))
        return jnp.sum(plan.realize(state).vertices ** 2)

    derivative = jax.grad(vertex_sum)(jnp.asarray(0.75))
    assert bound == [False]
    # d/dr sum |v|^2 = 2 r V on the sphere: the fixed route stays differentiable.
    assert float(derivative) > 0.0


def test_edited_retained_realization_is_refused(
    retained_surface: tuple[
        phx.geometry.CompiledGeometry, phx.geometry.ImplicitSurfacePlan
    ],
) -> None:
    geometry, plan = retained_surface
    edited = eqx.tree_at(
        lambda value: value.initial_evidence.projection.root_residual,
        plan,
        plan.initial_evidence.projection.root_residual + 1.0e-12,
    )
    with pytest.raises(ValueError, match="edited retained leaves"):
        edited.retained_initial_realization(geometry.state)
    with pytest.raises(ValueError, match="mismatched retained evidence"):
        phx.geometry.ImplicitSurfacePlan.from_stored_fields(edited, geometry=geometry)
    edited_area = eqx.tree_at(
        lambda value: value.initial_evidence.minimum_face_area,
        plan,
        2.0 * plan.initial_evidence.minimum_face_area,
    )
    with pytest.raises(ValueError, match="edited retained leaves"):
        edited_area.retained_initial_realization(geometry.state)
    edited_fallback = eqx.tree_at(
        lambda value: value.base_vertices, plan, plan.base_vertices + 1.0e-12
    )
    with pytest.raises(ValueError, match="mismatched retained evidence"):
        phx.geometry.ImplicitSurfacePlan.from_stored_fields(
            edited_fallback, geometry=geometry
        )


def test_retained_realization_restores_only_for_its_bound_source_state(
    retained_surface: tuple[
        phx.geometry.CompiledGeometry, phx.geometry.ImplicitSurfacePlan
    ],
) -> None:
    geometry, plan = retained_surface
    restored = phx.geometry.ImplicitSurfacePlan.from_stored_fields(
        plan, geometry=geometry
    )
    assert (restored.plan_id, restored.initial_evidence_id) == (
        plan.plan_id,
        plan.initial_evidence_id,
    )
    assert np.array_equal(
        np.asarray(restored.retained_initial_realization(geometry.state).vertices),
        np.asarray(plan.realize(geometry.state).vertices),
    )
    moved = geometry.with_state(_radius_state(geometry, 0.76))
    with pytest.raises(ValueError, match="mismatched retained evidence"):
        phx.geometry.ImplicitSurfacePlan.from_stored_fields(plan, geometry=moved)


def test_native_implicit_execution_reuses_the_retained_discovery_realization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.meshing.providers._implicit as implicit_route
    from phydrax.geometry.implicit._discovery import implicit_realization_operations
    from tests._support.native_provider_cases import implicit_case, implicit_plan

    geometry, _, specification = implicit_case(
        size_strength=phx.meshing.SizeControlStrength.SOFT
    )
    plan = implicit_plan(geometry, specification)
    prepared = plan.prepared
    assert isinstance(prepared, phx.geometry.ImplicitSurfacePlan)
    charges: list[tuple[int, int]] = []
    original = implicit_route.consume_queries

    def recording(budget: Any, count: int, /, *, work_units: int = 0) -> None:
        charges.append((count, work_units))
        original(budget, count, work_units=work_units)

    monkeypatch.setattr(implicit_route, "consume_queries", recording)
    work, queries = implicit_realization_operations(prepared)
    retained_work, retained_queries = implicit_realization_operations(
        prepared, retained_proof=True
    )
    assert 0 < retained_work < work
    assert 0 < retained_queries < queries

    reused = plan.execute()
    assert (queries, work) not in charges
    assert charges.count((retained_queries, retained_work)) == 1
    assert np.array_equal(
        np.asarray(reused.geometry.coordinates),
        np.asarray(prepared.realize(geometry.state).vertices),
    )

    charges.clear()
    index = geometry.schema.index(phx.geometry.ParameterId("sphere", "radius"))
    moved = plan.execute(geometry.state.replace_at(index, jnp.asarray(0.76)))
    assert charges.count((queries, work)) == 1
    assert not np.array_equal(
        np.asarray(moved.geometry.coordinates), np.asarray(reused.geometry.coordinates)
    )


@pytest.mark.parametrize("route", ("fixed-lattice", "adaptive"))
def test_implicit_plan_without_preparation_receipt_refuses_execution(
    route: str,
) -> None:
    from tests._support.native_provider_cases import (
        adaptive_plan,
        implicit_case,
        implicit_plan,
    )

    geometry, _, specification = implicit_case(
        size_strength=phx.meshing.SizeControlStrength.SOFT,
        fidelity=None if route == "fixed-lattice" else 0.2,
    )
    plan = (
        implicit_plan(geometry, specification)
        if route == "fixed-lattice"
        else adaptive_plan(
            geometry,
            specification,
            phx.geometry.implicit.AdaptiveImplicitSurfacePolicy(maximum_level=6),
        )
    )
    assert plan.preparation_evidence is not None
    stripped = eqx.tree_at(
        lambda value: value.preparation_evidence,
        plan,
        None,
        is_leaf=lambda value: value is None,
    )
    direct = phx.meshing.NativeMeshingPlan(
        plan.source,
        plan.specification,
        plan.options,
        plan.support,
        plan.coordinate_contract,
        plan.prepared,
    )
    for receiptless in (stripped, direct):
        with pytest.raises(ValueError, match="measured native preparation receipt"):
            receiptless.execute()


def _metric_epoch_records(count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "result": {"result_id": f"carrier-{index}", "audit_passed": True},
            "acceptance": {"mesh_volume": 1.0, "cells": 8 ** (index + 1)},
            "physical_L2_error": 0.01 / (index + 1),
            "transferred_state_L2_error": 0.02,
            "seed_origin": "actual-transferred-prior-diffusion-solution",
        }
        for index in range(count + 1)
    ]
    for index, row in enumerate(rows[:-1]):
        row["adaptation"] = {
            "status": "complete",
            "compliance_passed": True,
            "target": rows[index + 1]["result"],
            "acceptance": rows[index + 1]["acceptance"],
            "metric_evidence": {
                "out_of_range_edges": 1,
                "unmet_metric_edges": 0,
                "source_lower_exemptions": [[0, 1, 0.5]],
                "source_lower_binding": "certified-source-binding",
            },
        }
        row["transfer"] = {
            "receipt_id": f"transfer-{index}",
            "field_names": ["u", "density", "history"],
            "transport_accepted": [True, True, True],
            "field_transfers": [{"passed": True} for _ in range(3)],
        }
    return rows


@pytest.mark.parametrize(
    ("case", "rounds", "completed", "expected"),
    (
        ("sphere", 0, 0, False),
        ("sphere", 1, 1, True),
        ("sphere-refinement", 2, 2, True),
        ("sphere-refinement", 2, 1, False),
        ("sphere-refinement", 2, 3, False),
        ("offset-ring-torus-anisotropic", 2, 2, True),
        ("offset-ring-torus-anisotropic", 2, 1, False),
    ),
)
def test_native_metric_obligation_counts_completed_transitions(
    case: ImplicitCase, rounds: int, completed: int, expected: bool
) -> None:
    from copy import deepcopy

    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification.scenarios.implicit import (
        _completed_native_metric_epochs,
        _implicit_qualification_obligations,
    )

    record = {"epochs": _metric_epoch_records(completed)}
    original = deepcopy(record)
    obligations = _implicit_qualification_obligations(
        QualificationControls(implicit_case=case, adaptation_rounds=rounds), record
    )
    assert _completed_native_metric_epochs(record["epochs"]) == completed
    assert obligations["native_metric_adaptation_epochs"] is expected
    assert record == original


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("adaptation",), {}),
        (("adaptation", "status"), "failed"),
        (("adaptation", "status"), "stalled"),
        (("adaptation", "compliance_passed"), False),
        (("adaptation", "metric_evidence", "unmet_metric_edges"), 1),
        (("adaptation", "metric_evidence"), {"out_of_range_edges": 0}),
        (("adaptation", "target", "audit_passed"), False),
        (("adaptation", "acceptance"), {}),
        (("transfer",), {}),
        (("transfer", "transport_accepted"), [True, False, True]),
        (("transfer", "field_transfers"), [{"passed": False}] * 3),
    ),
)
def test_incomplete_metric_transition_does_not_count(
    path: tuple[str, ...], value: Any
) -> None:
    from tools.meshing_qualification.scenarios.implicit import (
        _completed_native_metric_epochs,
    )

    rows = _metric_epoch_records(2)
    owner = rows[0]
    for key in path[:-1]:
        owner = owner[key]
    owner[path[-1]] = value
    assert _completed_native_metric_epochs(rows) == 1


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("result", {"result_id": "wrong-carrier", "audit_passed": True}),
        ("physical_L2_error", float("nan")),
        ("physical_L2_error", None),
        ("transferred_state_L2_error", float("inf")),
        ("seed_origin", "authored-initial-state"),
        ("acceptance", {}),
    ),
)
def test_metric_epoch_requires_continued_pde(key: str, value: Any) -> None:
    from tools.meshing_qualification.scenarios.implicit import (
        _completed_native_metric_epochs,
    )

    rows = _metric_epoch_records(2)
    rows[-1][key] = value
    assert _completed_native_metric_epochs(rows) == 1
    assert _completed_native_metric_epochs(rows[:1]) == 0
