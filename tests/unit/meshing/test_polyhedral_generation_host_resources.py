from contextlib import ExitStack, nullcontext
from typing import Any, Never

import numpy as np
import pytest
from numpy.typing import DTypeLike

import phydrax._meshcore as meshcore
from phydrax.meshing._polyhedral_generation import (
    _freeze,
    _recover_polyhedral_source,
    _tet_facets,
    NativePolyhedralSchedule,
)
from tests._support.polyhedral_meshes import voxel_plc


@pytest.mark.parametrize("refuse", (False, True), ids=("realized", "before-output"))
def test_periodic_reconciliation_admits_realized_retained_banks(
    monkeypatch: pytest.MonkeyPatch, refuse: bool,
) -> None:
    from dataclasses import replace

    import phydrax as phx
    from phydrax.meshing import _polyhedral_generation as generation
    from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory
    from tools._meshing_cases import prepare_polyhedral_case
    from tools.meshing_qualification._controls import MeshingLimitControls

    source, request, options = prepare_polyhedral_case(
        "polyhedral-periodic-cube", 4, 20_000, 120.0, source=True
    )
    original_reconcile = generation._reconcile_periodic_polyhedral
    original_admit = generation._admit_polyhedral_mesh_capacity
    original_builder = phx.discretization.CellMesh.from_polyhedra
    reconciling = False
    admissions = 0
    output_calls = 0
    worst_corners = 0

    def reconcile(*args: Any, **kwargs: Any) -> Any:
        nonlocal reconciling, worst_corners
        old = generation.require_polyhedral_connectivity(args[0])
        fragments = args[3]
        fragment_corners = sum(len(fragment.vertices) for fragment in fragments)
        possible_vertices = old.vertex_count + 2 * fragment_corners
        worst_corners = (
            2 * (old.face_vertex_values.size + 2 * fragment_corners)
            * max(1, possible_vertices - 1)
        )
        worst_slots, _ = generation._polyhedral_numeric_publication_bounds(
            possible_vertices, old.cell_count,
            2 * (old.face_count + 2 * len(fragments)), worst_corners,
        )
        assert worst_slots > request.limits.maximum_connectivity_entries
        reconciling = True
        try:
            return original_reconcile(*args, **kwargs)
        finally:
            reconciling = False

    def build(*args: Any, **kwargs: Any) -> Any:
        nonlocal output_calls
        if reconciling:
            assert admissions == 1
            output_calls += 1
        return original_builder(*args, **kwargs)

    def admit(
        vertices: int, cells: int, loops: int, corners: int,
        limits: phx.meshing.MeshingLimits,
    ) -> None:
        nonlocal admissions
        if reconciling:
            assert output_calls == 0
            assert corners < worst_corners
            admissions += 1
            slots, data = generation._polyhedral_numeric_publication_bounds(
                vertices, cells, loops, corners
            )
            for name, required in (
                ("vertices", vertices), ("edges", corners), ("faces", loops),
                ("cells", cells), ("connectivity_entries", slots), ("data_bytes", data),
            ):
                exact = replace(
                    MeshingLimitControls.from_limits(limits),
                    **{f"maximum_{name}": required},
                )
                original_admit(vertices, cells, loops, corners, exact.to_limits())
                with pytest.raises(MeshingFailure) as caught:
                    original_admit(
                        vertices, cells, loops, corners,
                        replace(exact, **{f"maximum_{name}": required - 1}).to_limits(),
                    )
                assert caught.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
            if refuse:
                limits = replace(
                    MeshingLimitControls.from_limits(limits),
                    maximum_connectivity_entries=slots - 1,
                ).to_limits()
        original_admit(vertices, cells, loops, corners, limits)

    monkeypatch.setattr(generation, "_reconcile_periodic_polyhedral", reconcile)
    monkeypatch.setattr(generation, "_admit_polyhedral_mesh_capacity", admit)
    monkeypatch.setattr(phx.discretization.CellMesh, "from_polyhedra", build)
    expectation = pytest.raises(MeshingFailure) if refuse else nullcontext()
    with expectation:
        result = phx.meshing.NativeMeshingProvider(options).plan(
            source, request, coordinate_contract=phx.SpatialCoordinateContract.si()
        ).execute()
        assert result.audit.passed
    assert admissions == 1
    assert output_calls == (0 if refuse else 1)


def test_recovery_host_export_exact_capacity_before_first_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real recovery export admits its complete host bank before copying it."""
    source = voxel_plc([(0, 0, 0)])
    schedule = NativePolyhedralSchedule()
    original_zeros = np.zeros
    observed_bounds: list[int] = []
    successful_points: np.ndarray | None = None
    for shortfall in (0, 1):
        output_allocations: list[tuple[int, ...]] = []
        export_started = False

        def tracked_zeros(shape: tuple[int, ...], dtype: DTypeLike = float) -> np.ndarray:
            if export_started:
                output_allocations.append(shape)
            return original_zeros(shape, dtype=dtype)

        expected = pytest.raises(meshcore.MeshcoreError) if shortfall else nullcontext()
        with expected as error, monkeypatch.context() as patch, ExitStack() as lifetime:
            patch.setattr(meshcore.np, "zeros", tracked_zeros)
            limits = schedule.native_limits()
            lifetime.enter_context(meshcore.NativeExecutionScope(limits))

            def admit_host_output(bound: int) -> None:
                nonlocal export_started
                observed_bounds.append(bound)
                child = lifetime.enter_context(meshcore.NativeExecutionScope(limits))
                workspace = lifetime.enter_context(meshcore.NativeHostStorageWorkspace(child))
                workspace.reserve_empty()
                ballast = lifetime.enter_context(meshcore.NativeHostStorageWorkspace(child))
                ballast.reserve_empty()
                available = child.remaining().remaining_managed_bytes
                ballast.set_bound(available - bound + shortfall)
                assert child.remaining().remaining_managed_bytes == bound - shortfall
                workspace.set_bound(bound)
                export_started = True

            def recover() -> meshcore.PlcRecovery3D:
                return meshcore.recover_plc_3d(
                    source.vertices,
                    source.polygon_offsets,
                    source.polygon_vertices,
                    source.polygon_facets,
                    source.facet_regions,
                    segments=source.segments,
                    boundary_policy=source.boundary,
                    max_vertices=schedule.maximum_vertices,
                    max_tetrahedra=schedule.maximum_cells,
                    work_limit=schedule.maximum_work_units,
                    max_scratch_bytes=schedule.maximum_scratch_bytes,
                    admit_host_output=admit_host_output,
                )

            result = recover()
            successful_points = result.points.copy()
            assert len(output_allocations) == 16
            active = meshcore.current_native_execution_scope()
            assert active is not None
            assert active.remaining().remaining_managed_bytes < limits.maximum_managed_bytes
        if shortfall:
            assert error is not None
            assert error.value.status is meshcore.MeshcoreStatus.CAPACITY_EXCEEDED
            assert output_allocations == []
    assert observed_bounds[0] == observed_bounds[1]
    assert successful_points is not None
    np.testing.assert_array_equal(successful_points[: len(source.vertices)], source.vertices)


def test_recovery_host_storage_survives_export_and_freeze_is_native_owned() -> None:
    source = voxel_plc([(0, 0, 0)])
    schedule = NativePolyhedralSchedule()
    with meshcore.NativeExecutionScope(schedule.native_limits()) as scope:
        before = scope.remaining().remaining_managed_bytes
        recovery = _recover_polyhedral_source(
            source, None, None, schedule, scope=scope, completed_work=0
        )
        retained = scope.remaining().remaining_managed_bytes
        assert retained < before
        frozen = _freeze(recovery.tetrahedra, scope=scope)
        assert not frozen.flags.writeable
        np.testing.assert_array_equal(frozen, recovery.tetrahedra)
        assert scope.remaining().remaining_managed_bytes < retained


def test_recovered_facet_keys_charge_only_actual_visits() -> None:
    source = voxel_plc([(0, 0, 0)])
    schedule = NativePolyhedralSchedule()
    with meshcore.NativeExecutionScope(schedule.native_limits()) as recovery_scope:
        recovery = _recover_polyhedral_source(
            source, None, None, schedule, scope=recovery_scope, completed_work=0
        )
    required = len(recovery.faces) + 4 * len(recovery.tetrahedra)
    expected = {
        tuple(sorted(map(int, face))): index
        for index, face in enumerate(recovery.faces)
    }
    limits = schedule.native_limits()
    for shortfall in (0, 1):
        bounded = meshcore.NativeExecutionLimits(
            required - shortfall, 0, limits.maximum_cavity_cells,
            limits.maximum_managed_bytes, limits.maximum_seconds,
        )
        expectation = pytest.raises(meshcore.MeshcoreError) if shortfall else nullcontext()
        with expectation as caught, meshcore.NativeExecutionScope(bounded) as scope:
            facets = _tet_facets(recovery, scope=scope)
            assert scope.remaining().remaining_work_units == 0
            for row, tet in enumerate(recovery.tetrahedra):
                for opposite in range(4):
                    key = tuple(sorted(int(tet[v]) for v in range(4) if v != opposite))
                    assert facets[row, opposite] == expected.get(key, -1)
        if shortfall:
            assert caught is not None
            assert caught.value.status is meshcore.MeshcoreStatus.CAPACITY_EXCEEDED
        evidence = scope.completed_evidence()
        assert evidence.externally_charged_work == required - shortfall
        assert evidence.externally_charged_geometry_queries == 0


@pytest.mark.parametrize("resolution", (4, 8))
def test_original_periodic_request_derives_and_exercises_admission_boundary(
    resolution: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Earn R from this request; never hardcode another cohort's boundary."""
    from dataclasses import replace
    from time import monotonic, sleep

    import phydrax as phx
    from phydrax._meshcore import NativeParallelExecution
    from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory
    from phydrax.meshing.qualification import execute_native_capacity_trial
    from tools._meshing_cases import prepare_polyhedral_case
    from tools.meshing_qualification._controls import (
        MeshingLimitControls,
        with_meshing_limits,
    )

    source, request, options = prepare_polyhedral_case(
        "polyhedral-periodic-cube", resolution, 20000, 120, source=True
    )
    original_source = source.complex.vertices.tobytes()

    operation_started = monotonic()
    original_phases = []
    def execute(capacity: int) -> phx.meshing.CellMeshingResult:
        limits = replace(
            MeshingLimitControls.from_limits(request.limits),
            maximum_scratch_bytes=capacity,
        ).to_limits()
        return phx.meshing.NativeMeshingProvider(options).plan(
            source,
            with_meshing_limits(request, limits),
            coordinate_contract=phx.SpatialCoordinateContract.si(),
            record_phase=original_phases.append,
        ).execute(record_phase=original_phases.append)

    with NativeParallelExecution(None, request.limits.maximum_wall_seconds):
        admitted = execute(request.limits.maximum_scratch_bytes)
    receipt = admitted.execution_evidence
    assert receipt is not None
    observed = receipt.total_managed_admission_peak_bytes_upper
    assert observed is not None
    boundary = int(np.asarray(observed))
    assert 0 < boundary <= request.limits.maximum_scratch_bytes
    def trial(capacity: int) -> phx.meshing.CellMeshingResult:
        limits = replace(
            MeshingLimitControls.from_limits(request.limits),
            maximum_scratch_bytes=capacity,
        ).to_limits()
        phases = []
        return execute_native_capacity_trial(
            source, with_meshing_limits(request, limits), options, admitted,
            operation_started=operation_started, record_phase=phases.append,
        )

    exact = trial(boundary)
    assert exact.execution_evidence is not None
    assert exact.execution_evidence.total_managed_admission_peak_bytes_upper is not None
    np.testing.assert_array_equal(exact.mesh.coordinates, admitted.mesh.coordinates)
    assert isinstance(exact.mesh.connectivity, phx.discretization.PolyhedralConnectivity)
    assert isinstance(admitted.mesh.connectivity, phx.discretization.PolyhedralConnectivity)
    np.testing.assert_array_equal(
        exact.mesh.connectivity.face_vertex_values,
        admitted.mesh.connectivity.face_vertex_values,
    )
    with pytest.raises(MeshingFailure) as refused:
        trial(boundary - 1)
    assert refused.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert source.complex.vertices.tobytes() == original_source
    assert "scratch_bytes" in exact.runtime.enforced_limits
    assert "scratch_bytes:unmanaged_host_device_compiler" in exact.runtime.unenforced_limits

    def unreachable_plan(*args: object, **kwargs: object) -> Never:
        raise AssertionError("An expired workflow must refuse before planning.")

    with monkeypatch.context() as patch:
        patch.setattr(phx.meshing.NativeMeshingProvider, "plan", unreachable_plan)
        with pytest.raises(MeshingFailure) as expired:
            execute_native_capacity_trial(
                source, request, options, admitted,
                operation_started=monotonic() - 2 * request.limits.maximum_wall_seconds,
            )
        assert expired.value.category is MeshingFailureCategory.TIMED_OUT
        with pytest.raises(MeshingFailure) as inherited:
            with NativeParallelExecution(None, 0.01):
                sleep(0.02)
                execute_native_capacity_trial(
                    source, request, options, admitted, operation_started=monotonic()
                )
        assert inherited.value.category is MeshingFailureCategory.TIMED_OUT


def test_material_source_prefix_retains_original_plan_and_exact_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay measured R and R-1 without changing any scientific certificate limit."""
    from copy import deepcopy
    from time import monotonic, sleep

    import phydrax as phx
    from phydrax.discretization._coordinate_operands import CoordinateOperandCollector
    from phydrax.meshing.qualification import execute_native_capacity_trial
    from tools._meshing_cases import prepare_polyhedral_case
    from tools.meshing_qualification._records import _require_source_prefix_trial
    from tools.meshing_qualification.scenarios.polyhedral import (
        _polyhedral_source_prefix_resource_refusals,
    )

    source, request, options = prepare_polyhedral_case(
        "polyhedral-material-L", 4, 20000, 120, source=True
    )
    provider = phx.meshing.NativeMeshingProvider(options)
    started = monotonic()
    plan = provider.plan(
        source, request, coordinate_contract=phx.SpatialCoordinateContract.si()
    )
    publication = plan.execute(coordinate_operands=CoordinateOperandCollector())
    assert publication.execution_evidence is not None
    assert publication.execution_evidence.total_managed_admission_peak_bytes_upper is not None
    boundary = int(np.asarray(publication.execution_evidence.total_work_units))
    original_science = (plan.plan_id, request.specification_id, publication.result_id)
    record = _polyhedral_source_prefix_resource_refusals(
        source, request, options, publication, prepared_plan=plan,
        campaign_started=started,
    )
    work = next(item for item in record["refusals"] if item["control"] == "maximum_work_units")
    assert work["reduced_limit"] == boundary - 1
    assert work["measured_R_positive"]["observed_work_units"] == boundary
    assert original_science == (plan.plan_id, request.specification_id, publication.result_id)
    assert plan.specification is request
    assert work["frozen_request"] == record["frozen_request"]
    print({
        "source_prefix_status": record["status"], "R": boundary,
        "R_minus_one": work["reduced_limit"], "science_ids": original_science,
        "managed_admission_peak_bytes_upper": int(np.asarray(
            publication.execution_evidence.total_managed_admission_peak_bytes_upper
        )),
        "remaining_gaps": record["unobserved_or_unenforced_original_bounds"],
        "refusals": [(item["control"], item["failure_category"]) for item in record["refusals"]],
    })

    copied = deepcopy(work)
    copied["measured_R_positive"]["result"] = record["positive"]["result"]
    with pytest.raises(ValueError, match="fresh replay"):
        _require_source_prefix_trial(
            copied, boundary, source, request, options, record["positive"], None
        )
    replanned = provider.plan(
        source, request, coordinate_contract=publication.coordinate_contract
    )
    with pytest.raises(ValueError, match="original source"):
        execute_native_capacity_trial(
            source, request, options, publication, operation_started=started,
            prepared_plan=replanned, root_maximum_work_units=boundary,
        )
    changed = deepcopy(work)
    changed["frozen_request"] = {"specification_id": "replanned"}
    with pytest.raises(ValueError, match="frozen source"):
        _require_source_prefix_trial(
            changed, boundary, source, request, options, record["positive"], None
        )
    stale = deepcopy(work)
    stale["measured_R_positive"]["observed_work_units"] = boundary - 1
    with pytest.raises(ValueError, match="fresh replay"):
        _require_source_prefix_trial(
            stale, boundary, source, request, options, record["positive"], None
        )

    def unreachable(*args: object, **kwargs: object) -> Never:
        raise AssertionError("An expired workflow cannot execute the retained plan.")

    with monkeypatch.context() as patch:
        patch.setattr(phx.meshing.NativeMeshingPlan, "_execute_with_root_work_admission", unreachable)
        with pytest.raises(phx.meshing.MeshingFailure) as expired:
            execute_native_capacity_trial(
                source, request, options, publication,
                operation_started=started - 2 * request.limits.maximum_wall_seconds,
                prepared_plan=plan, root_maximum_work_units=boundary,
            )
        assert expired.value.category is phx.meshing.MeshingFailureCategory.TIMED_OUT
        with pytest.raises(phx.meshing.MeshingFailure) as inherited:
            with meshcore.NativeParallelExecution(None, 0.01):
                sleep(0.02)
                execute_native_capacity_trial(
                    source, request, options, publication, operation_started=started,
                    prepared_plan=plan, root_maximum_work_units=boundary,
                )
        assert inherited.value.category is phx.meshing.MeshingFailureCategory.TIMED_OUT


def test_mechanical_root_work_refusal_retains_owning_limit() -> None:
    import phydrax as phx
    from phydrax.meshing._volume_generation import native_volume_execution_budget

    limits = phx.meshing.MeshingLimits(maximum_work_units=100)
    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        with native_volume_execution_budget(limits, root_maximum_work_units=2):
            with native_volume_execution_budget(limits) as child:
                child.charge(work=3)
    assert failure.value.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert dict(failure.value.evidence.requested)["maximum_work_units"] == 2
    assert limits.maximum_work_units == 100
    with native_volume_execution_budget(limits):
        with pytest.raises(ValueError, match="execution root"):
            with native_volume_execution_budget(limits, root_maximum_work_units=2):
                pytest.fail("A nested override cannot replace the root owner.")


def test_unmapped_native_predicate_does_not_invent_managed_envelope() -> None:
    """Native predicate reservations are not allocation-free control queries."""
    scope = meshcore.NativeExecutionScope(
        meshcore.NativeExecutionLimits(1000, 1000, 1000, 1 << 20, 120)
    )
    with scope:
        scope.observe_managed_admission()
        points = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
             (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        )
        signs = np.zeros((1,), dtype=np.int8)
        library = meshcore.load_meshcore()
        status = library["phx_mc_orient3d"](
            1, *(point.ctypes.data for point in points), signs.ctypes.data
        )
        assert status == int(meshcore.MeshcoreStatus.OK)
        assert signs[0] == 1
        scope.observe_managed_admission()
    assert scope.completed_evidence().managed_admission_peak_bytes_upper is None


@pytest.mark.parametrize(
    ("wrapper", "executor"),
    (
        ("_corpus_native_consumer", "execute_native_consumer"),
        ("_corpus_native_request", "execute_native_request"),
        ("cold_worker", "execute_native_request"),
    ),
)
def test_corpus_resource_wrappers_opt_into_managed_observation(
    wrapper: str, executor: str,
) -> None:
    """Guard call-site observation policy without rerunning cold science."""
    import ast
    import inspect

    from tools.meshing_qualification.scenarios import native_lifecycle

    tree = ast.parse(inspect.getsource(getattr(native_lifecycle, wrapper)))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == executor
    ]
    assert len(calls) == 1
    observation = [
        keyword.value for keyword in calls[0].keywords
        if keyword.arg == "observe_managed_admission"
    ]
    assert len(observation) == 1
    assert isinstance(observation[0], ast.Constant)
    assert observation[0].value is True


def test_small_generation_measured_query_boundary() -> None:
    """Replay the same scientific construction at its own measured query R/R-1."""
    from dataclasses import replace

    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
    from phydrax.meshing._polyhedral_generation import generate_polyhedral_volume

    source = voxel_plc([(0, 0, 0)])
    original = source.vertices.copy()
    schedule = NativePolyhedralSchedule()
    limits = schedule.native_limits()
    with meshcore.NativeExecutionScope(limits) as measured:
        reference = generate_polyhedral_volume(source, schedule=schedule, scope=measured)
        remaining = measured.remaining()
        required = limits.maximum_geometry_queries - remaining.remaining_geometry_queries
    assert required > 0
    for shortfall in (0, 1):
        bounded = replace(limits, maximum_geometry_queries=required - shortfall)
        expectation = (
            pytest.raises((meshcore.MeshcoreError, CoordinateEnclosureResourceError))
            if shortfall else nullcontext()
        )
        with expectation, meshcore.NativeExecutionScope(bounded) as scope:
            result = generate_polyhedral_volume(source, schedule=schedule, scope=scope)
            assert result.mesh.topology_id == reference.mesh.topology_id
            np.testing.assert_array_equal(result.mesh.coordinates, reference.mesh.coordinates)
            np.testing.assert_array_equal(
                result.mesh.vertex_global_ids, reference.mesh.vertex_global_ids
            )
            np.testing.assert_array_equal(result.cell_sites, reference.cell_sites)
            assert scope.remaining().remaining_geometry_queries == 0
    np.testing.assert_array_equal(source.vertices, original)


def test_small_generation_consumed_work_is_its_admission_boundary() -> None:
    """The complete request's measured work admits it exactly and refuses R-1."""
    from dataclasses import replace

    from phydrax.meshing._polyhedral_generation import generate_polyhedral_volume

    source = voxel_plc([(0, 0, 0)])
    schedule = NativePolyhedralSchedule()
    limits = schedule.native_limits()
    with meshcore.NativeExecutionScope(limits) as measured:
        reference = generate_polyhedral_volume(source, schedule=schedule, scope=measured)
        consumed = limits.maximum_work_units - measured.remaining().remaining_work_units
    with meshcore.NativeExecutionScope(
        replace(limits, maximum_work_units=consumed)
    ) as exact:
        result = generate_polyhedral_volume(source, schedule=schedule, scope=exact)
        assert exact.remaining().remaining_work_units == 0
    assert result.mesh.topology_id == reference.mesh.topology_id
    np.testing.assert_array_equal(result.cell_sites, reference.cell_sites)
    with pytest.raises(meshcore.MeshcoreError) as caught:
        with meshcore.NativeExecutionScope(
            replace(limits, maximum_work_units=consumed - 1)
        ) as scope:
            generate_polyhedral_volume(source, schedule=schedule, scope=scope)
    assert caught.value.status is meshcore.MeshcoreStatus.CAPACITY_EXCEEDED
    assert caught.value.batch_requested_upper is not None


def test_refinement_insertion_work_boundary_precedes_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real restricted-cell insertion retains its inputs at the R-1 boundary.

    A nested refusal terminates the shared parent pool, so each boundary runs
    in its own root scope over the captured live insertion inputs.
    """
    from dataclasses import replace

    from phydrax.meshing import _polyhedral_generation as generation

    source = voxel_plc([(0, 0, 0)])
    source_vertices = source.vertices.copy()
    schedule = NativePolyhedralSchedule(maximum_cell_diameter=1.5)
    original = generation._insert_polyhedral_sites
    captured: list[tuple[Any, Any, Any, Any]] = []

    class CapturedInsertion(Exception):
        pass

    def insert(publication: Any, state: Any, measures: Any, complex_: Any, *, scope: Any) -> Any:
        captured.append((publication, state, measures, complex_))
        raise CapturedInsertion

    monkeypatch.setattr(generation, "_insert_polyhedral_sites", insert)
    with pytest.raises(CapturedInsertion), meshcore.NativeExecutionScope(
        schedule.native_limits()
    ) as scope:
        generation.generate_polyhedral_volume(
            source, sites=np.asarray(((0.2, 0.2, 0.2),)), schedule=schedule, scope=scope
        )
    np.testing.assert_array_equal(source.vertices, source_vertices)
    ((publication, state, measures, complex_),) = captured
    connectivity = generation.require_polyhedral_connectivity(publication.mesh)
    added = len(measures.large)
    total = len(state.points) + added
    required = (
        publication.mesh.coordinates.size + 7 * connectivity.cell_vertex_values.size
        + len(connectivity.cell_vertex_offsets)
        + 3 * total * (total.bit_length() + 2) + 12 * total + 4 * added
    )
    saved = tuple(bank.copy() for bank in (state.points, state.weights, state.parents, state.split))
    with (
        pytest.raises(meshcore.MeshcoreError) as caught,
        meshcore.NativeExecutionScope(
            replace(schedule.native_limits(), maximum_work_units=required - 1)
        ) as refused,
    ):
        original(publication, state, measures, complex_, scope=refused)
    assert caught.value.status is meshcore.MeshcoreStatus.CAPACITY_EXCEEDED
    assert refused.completed_evidence().work_counters.construction_work == 0
    for actual, expected in zip(
        (state.points, state.weights, state.parents, state.split), saved, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    with meshcore.NativeExecutionScope(schedule.native_limits()) as measured:
        reference = original(publication, state, measures, complex_, scope=measured)
        complete_required = (
            schedule.native_limits().maximum_work_units
            - measured.remaining().remaining_work_units
        )
    with meshcore.NativeExecutionScope(
        replace(schedule.native_limits(), maximum_work_units=complete_required)
    ) as exact:
        result = original(publication, state, measures, complex_, scope=exact)
        assert exact.remaining().remaining_work_units == 0
    np.testing.assert_array_equal(result.points, reference.points)
    with (
        pytest.raises(meshcore.MeshcoreError),
        meshcore.NativeExecutionScope(
            replace(schedule.native_limits(), maximum_work_units=complete_required - 1)
        ) as short,
    ):
        original(publication, state, measures, complex_, scope=short)
    failed = short.completed_evidence()
    assert failed.memory_counters.live_bytes == 0
    assert failed.host_storage_live_bytes_upper == 0
