"""Real owner-bound regressions for polyhedral provider host storage admission."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax._host_storage import retained_host_storage_profile
from phydrax._meshcore import NativeExecutionScope
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._cell_complex import PolyhedralConnectivity
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.meshing._polyhedral_generation import (
    generate_polyhedral_volume,
    NativePolyhedralSchedule,
    PolyhedralConstruction,
)
from phydrax.meshing.providers import _native_polyhedral as owner
from phydrax.meshing.providers._native_publication import coordinate_budget
from tests._support.polyhedral_meshes import periodic_cube_source, voxel_plc


M = phx.meshing


@contextmanager
def _budget(memory: int) -> Iterator[CoordinateEnclosureBudget]:
    work = 1 << 40
    with HostBatchTicket(work, bytes_upper=memory) as ticket:
        yield CoordinateEnclosureBudget(
            ticket, maximum_work_units=work, maximum_memory_bytes=memory
        )


def _specification(
    source: M.NativePolyhedralSource,
    periodic_constraints: tuple[M.PeriodicConstraint, ...] = (),
) -> M.VolumeMeshingSpec:
    scope = M.MeshingScope(
        source.source_id, source.source_revision, M.MeshingEntityKind.GEOMETRY,
        2, "polyhedral-facets", np.arange(source.complex.facet_count),
    )
    return M.VolumeMeshingSpec(
        M.CellMeshingTarget(3, 3, M.CellFamilyPolicy(required=("polyhedron",))),
        scope, M.VolumeFillStrategy.POLYHEDRAL,
        size_controls=(M.UniformSizeControl(
            scope, 10.0, strength=M.SizeControlStrength.SOFT,
        ),),
        periodic_constraints=periodic_constraints,
    )


@pytest.fixture(scope="module")
def provider_inputs() -> tuple[
    M.NativePolyhedralSource, M.VolumeMeshingSpec,
    owner.PreparedPolyhedralVolume, PolyhedralConstruction,
]:
    source = M.NativePolyhedralSource(
        voxel_plc([(0, 0, 0)]), "polyhedral-domain", "r1",
        sites=np.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]]),
    )
    specification = _specification(source)
    schedule = NativePolyhedralSchedule()
    prepared = owner.PreparedPolyhedralVolume(source, specification, schedule)
    with NativeExecutionScope(schedule.native_limits()) as scope:
        construction = generate_polyhedral_volume(
            source.complex, sites=source.sites, weights=source.weights,
            schedule=schedule, source_id=source.source_id, scope=scope,
        )
    return source, specification, prepared, construction


def _operation(
    name: str,
    inputs: tuple[
        M.NativePolyhedralSource, M.VolumeMeshingSpec,
        owner.PreparedPolyhedralVolume, PolyhedralConstruction,
    ],
) -> Callable[[CoordinateEnclosureBudget], Any]:
    source, specification, prepared, construction = inputs
    if name == "organization":
        return lambda budget: owner.organization(
            source, specification, construction, prepared.schedule.feature_tolerance,
            budget=budget,
        )
    if name == "compliance":
        return lambda budget: owner._polyhedral_compliance(
            construction, specification, None, budget=budget,
        )
    if name == "provenance":
        return lambda budget: owner._polyhedral_provenance(
            source, prepared, "provider-storage-plan", construction, budget=budget,
        )
    if name == "inputs":
        return lambda budget: owner._polyhedral_construction_inputs(
            source, specification, budget=budget,
        )
    if name == "reports":
        return lambda budget: owner._polyhedral_stage_reports(
            source, prepared, construction, budget=budget,
        )
    raise AssertionError(name)


@pytest.mark.parametrize("phase", ["organization", "compliance", "provenance", "inputs", "reports"])
def test_real_provider_owner_memory_r_and_r_minus_one(
    provider_inputs: tuple[
        M.NativePolyhedralSource, M.VolumeMeshingSpec,
        owner.PreparedPolyhedralVolume, PolyhedralConstruction,
    ],
    phase: str,
) -> None:
    operation = _operation(phase, provider_inputs)
    with _budget(1 << 30) as measured:
        reference = operation(measured)
        required = measured.peak_bytes_upper
        retained = measured.retained_basis_bytes
        assert measured.temporary_bytes_upper == 0
        assert retained > 0
        assert any(value is reference for value in measured.retained_owner_objects.values())
    with _budget(required) as exact:
        accepted = operation(exact)
        assert exact.peak_bytes_upper == required
        assert exact.retained_basis_bytes == retained
    if phase == "provenance":
        assert canonical_fingerprint(accepted) == canonical_fingerprint(reference)
    elif phase == "organization":
        assert tuple(value.association_id for value in accepted.associations) == tuple(
            value.association_id for value in reference.associations
        )
    elif phase == "compliance":
        assert accepted.compliance.report_id == reference.compliance.report_id
    with _budget(required - 1) as refused:
        with pytest.raises(CoordinateEnclosureResourceError) as captured:
            operation(refused)
        assert captured.value.requested == required


def test_organization_refuses_before_constructing_source_arrays(
    provider_inputs: tuple[
        M.NativePolyhedralSource, M.VolumeMeshingSpec,
        owner.PreparedPolyhedralVolume, PolyhedralConstruction,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, specification, prepared, construction = provider_inputs
    output, scratch, _ = owner._polyhedral_organization_bounds(
        source, specification, construction
    )

    def no_construction(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("source association arrays allocated before admission")

    monkeypatch.setattr(owner, "_organization", no_construction)
    with _budget(output + scratch - 1) as budget:
        with pytest.raises(CoordinateEnclosureResourceError):
            owner.organization(
                source, specification, construction,
                prepared.schedule.feature_tolerance, budget=budget,
            )
        assert budget.retained_basis_bytes == 0
        assert budget.temporary_bytes_upper == 0


@pytest.mark.parametrize("shape", [(0,), (7,), (2, 3), (3, 0, 2), (2, 3, 4)])
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float64])
def test_tolist_bound_covers_actual_nested_payload(
    shape: tuple[int, ...], dtype: type[np.generic],
) -> None:
    array = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    payload = array.tolist()
    _, storage, _, _ = retained_host_storage_profile(payload, {}, lambda value: False)
    assert storage <= owner._tolist_storage_upper(shape, array.dtype)


def test_provenance_keeps_concrete_selected_bank_values(
    provider_inputs: tuple[
        M.NativePolyhedralSource, M.VolumeMeshingSpec,
        owner.PreparedPolyhedralVolume, PolyhedralConstruction,
    ],
) -> None:
    source, _, prepared, construction = provider_inputs
    with _budget(1 << 30) as budget:
        payload = owner._polyhedral_provenance(
            source, prepared, "provider-storage-plan", construction, budget=budget,
        )
        first_peak, first_retained = budget.peak_bytes_upper, budget.retained_basis_bytes
        second = owner._polyhedral_provenance(
            source, prepared, "provider-storage-plan", construction, budget=budget,
        )
        assert canonical_fingerprint(second) == canonical_fingerprint(payload)
        # The second payload must coexist with the first, not renew its ledger.
        assert budget.peak_bytes_upper >= first_peak + first_retained
    data = construction.diagram.construction
    connectivity = construction.mesh.connectivity
    assert isinstance(connectivity, PolyhedralConnectivity)
    order = np.asarray(connectivity.cell_global_ids)
    assert payload["site_ancestry"] == construction.site_parents.tolist()
    assert payload["carrier_points"] == construction.recovery.points.tolist()
    assert payload["construction_cell_first_moments"] == data.cell_moments[order].tolist()
    assert payload["cell_image_sites"] == data.cell_sites[order].tolist()
    assert payload["source"] == source.binding_id
    assert payload["construction"] == construction.construction_id


def test_preparation_defers_exact_periodic_admission_to_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.meshing import _polyhedral_generation as generation

    plc, constraints = periodic_cube_source()
    source = M.NativePolyhedralSource(
        plc, "polyhedral-domain", "r1", sites=np.asarray([[0.5, 0.5, 0.5]]),
    )
    specification = _specification(source, constraints)

    def execution_only(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("exact periodic proof must use execution's request ledger")

    monkeypatch.setattr(generation, "polyhedral_periodic_group", execution_only)
    # Catch a stale directly imported provider alias as well as a fresh lookup.
    monkeypatch.setattr(owner, "polyhedral_periodic_group", execution_only, raising=False)
    prepared = owner.PreparedPolyhedralVolume(source, specification)
    assert prepared.source_binding_id == source.binding_id
    assert prepared.specification_id == specification.specification_id


def test_provider_host_phases_keep_one_actual_request_ledger(
    provider_inputs: tuple[
        M.NativePolyhedralSource, M.VolumeMeshingSpec,
        owner.PreparedPolyhedralVolume, PolyhedralConstruction,
    ],
) -> None:
    source, specification, prepared, construction = provider_inputs
    budget = coordinate_budget(specification.limits)
    limits = prepared.schedule.native_limits()
    with NativeExecutionScope(limits) as scope:
        before = scope.remaining()
        with budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)):
            with budget.bind_ticket(scope.ticket_view()):
                metadata = owner.organization(
                    source, specification, construction,
                    prepared.schedule.feature_tolerance, budget=budget,
                )
            first_work, first_retained = budget.work_units, budget.retained_basis_bytes
            assert budget.ticket.settled
            assert budget.storage_ticket is not None
            assert budget.storage_ticket.bytes_upper >= first_retained
            scope.charge()
            with budget.bind_ticket(scope.ticket_view()):
                provenance = owner._polyhedral_provenance(
                    source, prepared, "provider-storage-plan", construction,
                    budget=budget,
                )
            assert budget.work_units > first_work
            assert budget.retained_basis_bytes > first_retained
            remaining = scope.remaining()
            assert before.remaining_work_units - remaining.remaining_work_units == budget.work_units
            assert remaining.remaining_managed_bytes < before.remaining_managed_bytes
            assert any(value is metadata for value in budget.retained_owner_objects.values())
            assert any(value is provenance for value in budget.retained_owner_objects.values())
    assert scope.evidence is not None


def test_reports_preserve_named_native_counter_order(
    provider_inputs: tuple[
        M.NativePolyhedralSource, M.VolumeMeshingSpec,
        owner.PreparedPolyhedralVolume, PolyhedralConstruction,
    ],
) -> None:
    source, _, prepared, construction = provider_inputs
    counters = construction.diagram.construction.counters
    with _budget(1 << 30) as budget:
        stages = owner._polyhedral_stage_reports(
            source, prepared, construction, budget=budget,
        )
    assert stages[1].diagnostics[0].quantities == tuple(
        (f"power_counter_{index}", float(value), float(value))
        for index, (_, value) in enumerate(counters.items())
    )
