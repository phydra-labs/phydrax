# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Restore-local owner reuse requires freshly identical content in every field."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace

import equinox as eqx
import jax
import numpy as np
import pytest

from phydrax import config as pxconfig
from phydrax._array_archive import ArrayArchiveLimits
from phydrax._fingerprint import logical_array_value_collection_digest
from phydrax._model._artifacts import register_artifact_value
from phydrax.discretization import CellBlock
from phydrax.lifecycle import _meshing_sources as sources


@dataclass(frozen=True)
class _MutableSourceOwner:
    values: np.ndarray
    marker: float = 0.0
    flags: frozenset[str] = frozenset(("a", "b"))

    def __post_init__(self) -> None:
        if not np.all(np.isfinite(self.values)):
            raise ValueError(
                "owner.values must contain finite values; got nonfinite input"
            )


def test_identical_context_free_owner_reuses_only_one_restore_scope() -> None:
    sources.register_meshing_source_artifacts()
    first = CellBlock("cells", "triangle", np.asarray(((0, 1, 2),), dtype=np.int64))
    second = CellBlock.from_stored_fields(first)
    limits = ArrayArchiveLimits()
    calls = 0

    def reconstruct(node: CellBlock) -> CellBlock:
        nonlocal calls
        calls += 1
        return CellBlock.from_stored_fields(node)

    with sources.meshing_source_owner_reconstruction_scope():
        sources._validate_context_free_source_owner(
            first, reconstruct, "owner", limits, None
        )
        sources._validate_context_free_source_owner(
            second, reconstruct, "owner", limits, None
        )
        assert calls == 1
        changed = eqx.tree_at(
            lambda value: value.vertices,
            second,
            np.asarray(((0, 1, 3),), dtype=np.int64),
        )
        with pytest.raises(ValueError, match="owner.stored_fields"):
            sources._validate_context_free_source_owner(
                changed, reconstruct, "owner", limits, None
            )
        assert calls == 2
    with sources.meshing_source_owner_reconstruction_scope():
        sources._validate_context_free_source_owner(
            second, reconstruct, "owner", limits, None
        )
        assert calls == 3
    assert (
        sources._validate_context_free_source_owner(
            first, reconstruct, "owner", limits, None
        )
        is not None
    )
    assert calls == 4


def test_public_source_validation_scopes_exact_owner_reuse_and_actual_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget

    sources.register_meshing_source_artifacts()
    first = CellBlock("cells", "triangle", np.asarray(((0, 1, 2),), dtype=np.int64))
    second = CellBlock.from_stored_fields(first)
    admissions = dict(sources._source_admissions())
    calls = 0
    expected_budget: CoordinateEnclosureBudget | None = None
    previous_scope = sources._source_owner_reconstruction_scope.get()

    def reconstruct(
        node: CellBlock, *, budget: CoordinateEnclosureBudget | None = None
    ) -> CellBlock:
        nonlocal calls
        scope = sources._source_owner_reconstruction_scope.get()
        assert scope is not None and scope.budget is expected_budget
        assert budget is expected_budget
        calls += 1
        return CellBlock.from_stored_fields(node)

    admissions[CellBlock] = replace(admissions[CellBlock], reconstruct=reconstruct)
    monkeypatch.setattr(sources, "_source_admissions", lambda: admissions)
    sources.validate_meshing_source_closure((first, second))
    assert calls == 1
    assert sources._source_owner_reconstruction_scope.get() is previous_scope
    sources.validate_meshing_source_closure((first, second))
    assert calls == 2  # Independent public validation cannot borrow an earlier scope.
    changed = eqx.tree_at(
        lambda value: value.vertices, second,
        np.asarray(((0, 1, 3),), dtype=np.int64),
    )
    with pytest.raises(ValueError, match="stored_fields"):
        sources.validate_meshing_source_closure((first, changed))
    assert calls == 4
    assert sources._source_owner_reconstruction_scope.get() is previous_scope
    with HostBatchTicket(100_000, bytes_upper=1_000_000) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000, maximum_memory_bytes=1_000_000
        )
        expected_budget = budget
        with budget.activate():
            with sources.meshing_source_owner_reconstruction_scope(budget=budget):
                sources.validate_meshing_source_closure((first, second))
                assert calls == 5
                sources.validate_meshing_source_closure((first, second))
                assert calls == 5
    assert sources._source_owner_reconstruction_scope.get() is previous_scope


def test_mutated_numpy_owner_and_foreign_context_do_not_reuse() -> None:
    sources.register_meshing_source_artifacts()
    original = CellBlock("cells", "triangle", np.asarray(((0, 1, 2),), dtype=np.int64))
    register_artifact_value("test:MutableSourceOwner", _MutableSourceOwner)
    vertices = np.asarray(((0.0, 1.0, 2.0),), dtype=np.float64)
    node = _MutableSourceOwner(vertices)
    digest = logical_array_value_collection_digest({})
    first_context = sources.MeshingSourceValidation((), {}, digest)
    second_context = sources.MeshingSourceValidation((), {}, digest)
    limits = ArrayArchiveLimits()
    calls = 0

    def reconstruct(value: CellBlock) -> CellBlock:
        nonlocal calls
        calls += 1
        return CellBlock.from_stored_fields(value)

    def reconstruct_mutable(value: _MutableSourceOwner) -> _MutableSourceOwner:
        nonlocal calls
        calls += 1
        return _MutableSourceOwner(value.values.copy())

    with sources.meshing_source_owner_reconstruction_scope():
        for context in (first_context, first_context, second_context):
            sources._validate_context_free_source_owner(
                original, reconstruct, "owner", limits, context
            )
        assert calls == 2
        for _ in range(2):
            sources._validate_context_free_source_owner(
                node, reconstruct_mutable, "owner", limits, None
            )
        assert calls == 3
        vertices[0, 2] = np.nan
        with pytest.raises(ValueError, match="owner.values"):
            sources._validate_context_free_source_owner(
                node, reconstruct_mutable, "owner", limits, None
            )
        assert calls == 4


def test_readonly_numpy_bank_mutation_reconstructs_and_refuses_stale_owner() -> None:
    sources.register_meshing_source_artifacts()
    register_artifact_value("test:MutableSourceOwner", _MutableSourceOwner)
    bank = np.asarray(((0.0, 1.0, 2.0),), dtype=np.float64)
    bank.flags.writeable = False
    stored = _MutableSourceOwner(bank)
    with sources.meshing_source_owner_reconstruction_scope():
        sources._validate_context_free_source_owner(
            stored, replace, "owner", ArrayArchiveLimits(), None
        )
        sources._validate_context_free_source_owner(
            stored, replace, "owner", ArrayArchiveLimits(), None
        )
        bank.flags.writeable = True
        bank[0, 2] = np.nan
        bank.flags.writeable = False
        with pytest.raises(ValueError):
            sources._validate_context_free_source_owner(
                stored, replace, "owner", ArrayArchiveLimits(), None
            )


def test_reconstruction_callable_and_archive_limits_bind_reuse() -> None:
    sources.register_meshing_source_artifacts()
    node = CellBlock("cells", "triangle", np.asarray(((0, 1, 2),), dtype=np.int64))
    calls = [0, 0]

    def first(value: CellBlock) -> CellBlock:
        calls[0] += 1
        return CellBlock.from_stored_fields(value)

    def second(value: CellBlock) -> CellBlock:
        calls[1] += 1
        return CellBlock.from_stored_fields(value)

    with sources.meshing_source_owner_reconstruction_scope():
        for reconstruct in (first, first, second, second):
            sources._validate_context_free_source_owner(
                node, reconstruct, "owner", ArrayArchiveLimits(), None
            )
        assert calls == [1, 1]
        sources._validate_context_free_source_owner(
            node, first, "owner", ArrayArchiveLimits(max_manifest_nesting=15), None
        )
        assert calls == [2, 1]


def test_static_nan_payloads_and_signed_zero_bytes_revalidate() -> None:
    register_artifact_value("test:MutableSourceOwner", _MutableSourceOwner)
    values = np.asarray((-0.0,), dtype=np.float64)
    node = _MutableSourceOwner(values)
    nan_values = np.asarray(
        (0x7FF8000000000001, 0x7FF8000000000002), dtype=np.uint64
    ).view(np.float64)
    nonfinite = _MutableSourceOwner(values, nan_values[0].item())
    changed_payload = _MutableSourceOwner(values, nan_values[1].item())
    calls = 0

    def reconstruct(value: _MutableSourceOwner) -> _MutableSourceOwner:
        nonlocal calls
        calls += 1
        return _MutableSourceOwner(value.values.copy(), value.marker)

    with sources.meshing_source_owner_reconstruction_scope():
        for _ in range(2):
            sources._validate_context_free_source_owner(
                nonfinite, reconstruct, "owner", ArrayArchiveLimits(), None
            )
        assert calls == 1
        sources._validate_context_free_source_owner(
            changed_payload, reconstruct, "owner", ArrayArchiveLimits(), None
        )
        assert calls == 2
        sources._validate_context_free_source_owner(
            node, reconstruct, "owner", ArrayArchiveLimits(), None
        )
        values[0] = 0.0
        sources._validate_context_free_source_owner(
            node, reconstruct, "owner", ArrayArchiveLimits(), None
        )
        assert calls == 4


def test_unordered_static_nan_payloads_have_canonical_snapshot_order() -> None:
    bits = np.asarray(
        (0x7FF8000000000001, 0x7FF8000000000002, 0x7FF8000000000003),
        dtype=np.uint64,
    ).view(np.float64)
    first = frozenset((("value", bits[0].item()), ("value", bits[1].item())))
    second = frozenset((("value", bits[1].item()), ("value", bits[0].item())))
    changed = frozenset((("value", bits[0].item()), ("value", bits[2].item())))
    first_key = sources._source_owner_static_scalar_bits(first)
    assert first_key is not None
    assert first_key == sources._source_owner_static_scalar_bits(second)
    assert first_key != sources._source_owner_static_scalar_bits(changed)


def test_numerical_configuration_context_requires_fresh_owner_validation() -> None:
    sources.register_meshing_source_artifacts()
    node = CellBlock("cells", "triangle", np.asarray(((0, 1, 2),), dtype=np.int64))
    calls = 0

    def reconstruct(value: CellBlock) -> CellBlock:
        nonlocal calls
        calls += 1
        return CellBlock.from_stored_fields(value)

    with sources.meshing_source_owner_reconstruction_scope():
        sources._validate_context_free_source_owner(
            node, reconstruct, "owner", ArrayArchiveLimits(), None
        )
        assert calls == 1
        with pxconfig.context(pxconfig.NumericsConfig(disable_jit=True)):
            sources._validate_context_free_source_owner(
                node, reconstruct, "owner", ArrayArchiveLimits(), None
            )
        assert calls == 2
        sources._validate_context_free_source_owner(
            node, reconstruct, "owner", ArrayArchiveLimits(), None
        )
        assert calls == 2


def test_only_immutable_live_array_hashes_reuse_within_the_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = jax.device_put(np.asarray((1.0, 2.0), dtype=np.float64))
    host = np.asarray((-0.0, 1.0), dtype=np.float64)
    original = sources.logical_array_value_collection_digest
    calls = 0

    def counted(
        arrays: Mapping[str, jax.Array | np.ndarray],
        /,
        *,
        maximum_chunk_bytes: int = 1 << 20,
    ) -> str:
        nonlocal calls
        calls += 1
        return original(arrays, maximum_chunk_bytes=maximum_chunk_bytes)

    monkeypatch.setattr(sources, "logical_array_value_collection_digest", counted)
    for repeat in (1, 2):
        with sources.meshing_source_owner_reconstruction_scope():
            scope = sources._source_owner_reconstruction_scope.get()
            assert scope is not None
            first = sources._source_owner_array_digest(
                {"jax": device, "host": host}, scope, 8, None
            )
            assert calls == 4 * repeat - 2
            repeated = sources._source_owner_array_digest(
                {"jax": device, "host": host}, scope, 8, None
            )
            assert first == repeated
            assert calls == 4 * repeat - 1
            host[0] = 0.0 if np.signbit(host[0]) else -0.0
            changed = sources._source_owner_array_digest(
                {"jax": device, "host": host}, scope, 8, None
            )
            assert first != changed
            assert calls == 4 * repeat
            renamed = sources._source_owner_array_digest(
                {"renamed": device}, scope, 8, None
            )
            named = sources._source_owner_array_digest({"jax": device}, scope, 8, None)
            assert renamed != named
            assert calls == 4 * repeat


def test_deleted_immutable_array_cannot_use_an_earned_hash() -> None:
    device = jax.device_put(np.asarray((1.0,), dtype=np.float64))
    with sources.meshing_source_owner_reconstruction_scope():
        scope = sources._source_owner_reconstruction_scope.get()
        assert scope is not None
        sources._source_owner_array_digest({"value": device}, scope, 8, None)
        device.delete()
        with pytest.raises(RuntimeError, match="deleted"):
            sources._source_owner_array_digest({"value": device}, scope, 8, None)


def test_nested_source_validation_scope_retains_only_its_outer_operation() -> None:
    assert sources._source_owner_reconstruction_scope.get() is None
    with sources.meshing_source_owner_reconstruction_scope():
        outer = sources._source_owner_reconstruction_scope.get()
        assert outer is not None
        with sources.meshing_source_owner_reconstruction_scope():
            assert sources._source_owner_reconstruction_scope.get() is outer
        assert sources._source_owner_reconstruction_scope.get() is outer
    assert sources._source_owner_reconstruction_scope.get() is None
    with sources.meshing_source_owner_reconstruction_scope():
        assert sources._source_owner_reconstruction_scope.get() is not outer


def test_source_recertification_reuses_full_binding_only_inside_one_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace
    from typing import Any

    import phydrax as phx
    from phydrax.meshing._certification import MeshCertificationReport
    from tests._support.periodic_meshing import periodic_seed_request

    source, specification = periodic_seed_request()
    result = (
        phx.meshing.NativeMeshingProvider(
            phx.meshing.NativeMeshingOptions("periodic_delaunay")
        )
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )
    certification = result.certification
    assert certification is not None
    original = sources.certify_meshing_acceptance
    calls = 0

    def counted(*args: Any, **kwargs: Any) -> MeshCertificationReport:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(sources, "certify_meshing_acceptance", counted)
    limits = ArrayArchiveLimits()

    def renew(policy: ArrayArchiveLimits) -> MeshCertificationReport:
        return sources.recertify_restored_meshing_source(
            certification.request,
            result.mesh,
            result.geometry,
            result.audit,
            archive_limits=policy,
        )

    with sources.meshing_source_owner_reconstruction_scope():
        first = renew(limits)
        assert first.report_id == certification.report_id
        assert renew(limits) is first
        with sources.meshing_source_owner_reconstruction_scope():
            assert renew(limits) is first
        assert calls == 1
        changed = replace(limits, max_member_bytes=limits.max_member_bytes - 1)
        second = renew(changed)
        assert second.report_id == first.report_id
        assert calls == 2
        changed_geometry = eqx.tree_at(
            lambda value: value.coordinates,
            result.geometry,
            result.geometry.coordinates + np.asarray((0.125, 0.0), dtype=np.float64),
        )
        with pytest.raises(ValueError):
            sources.recertify_restored_meshing_source(
                certification.request,
                result.mesh,
                changed_geometry,
                result.audit,
                archive_limits=limits,
            )
        assert calls == 3
    renew(limits)
    assert calls == 4


def test_cached_recertification_refuses_corrupted_supplied_preparation() -> None:
    from phydrax.discretization import CellValidityStatus
    from tests._support.mesh_certificate_fixtures import prepared_tetrahedral_acceptance

    mesh, geometry, audit, prepared = prepared_tetrahedral_acceptance()
    corrupted = eqx.tree_at(
        lambda value: value.validity.status,
        prepared,
        np.full_like(np.asarray(prepared.validity.status), CellValidityStatus.INVALID),
    )
    with sources.meshing_source_owner_reconstruction_scope():
        report = sources.recertify_restored_meshing_source(
            prepared.request, mesh, geometry, audit, prepared=prepared
        )
        assert report.passed
        with pytest.raises(ValueError):
            sources.recertify_restored_meshing_source(
                prepared.request, mesh, geometry, audit, prepared=corrupted
            )


def test_large_live_dependency_identity_keeps_mutation_and_archive_guards() -> None:
    from phydrax._model._structure import (
        model_recipe_wire_json,
        prepare_model_recipe,
    )
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget

    values = np.asarray((1.0, 2.0), dtype=np.float64)
    dependencies = (values, "live-dependency-metadata" * 60_000)
    limits = ArrayArchiveLimits()
    sources.register_meshing_source_artifacts()
    with HostBatchTicket(100_000_000, bytes_upper=128 << 20) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=100_000_000,
            maximum_memory_bytes=64 << 20,
        )
        with budget.activate(), sources.meshing_source_owner_reconstruction_scope(
            budget=budget
        ):
            scope = sources._source_owner_reconstruction_scope.get()
            assert scope is not None
            original = sources._source_owner_reconstruction_key(
                dependencies, limits, None, tuple, scope
            )
            values[0] = 3.0
            changed = sources._source_owner_reconstruction_key(
                dependencies, limits, None, tuple, scope
            )
            assert changed != original
            values[0] = 1.0
            restored = sources._source_owner_reconstruction_key(
                dependencies, limits, None, tuple, scope
            )
            assert restored == original
    recipe, _ = prepare_model_recipe(dependencies, prefix="archive")
    with pytest.raises(ValueError):
        model_recipe_wire_json(recipe, limits=limits)


def test_owner_recipe_admission_refuses_before_large_canonical_workspace() -> None:
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import (
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
    )

    register_artifact_value("test:MutableSourceOwner", _MutableSourceOwner)
    owner = _MutableSourceOwner(
        np.asarray((1.0,), dtype=np.float64),
        flags=frozenset(("admission-metadata-" * 20_000,)),
    )
    calls = 0

    def reconstruct(value: _MutableSourceOwner) -> _MutableSourceOwner:
        nonlocal calls
        calls += 1
        return _MutableSourceOwner(value.values.copy(), value.marker, value.flags)

    with HostBatchTicket(10_000_000, bytes_upper=64 << 10) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=10_000_000,
            maximum_memory_bytes=64 << 10,
        )
        with budget.activate(), sources.meshing_source_owner_reconstruction_scope(
            budget=budget
        ):
            with pytest.raises(CoordinateEnclosureResourceError):
                sources._validate_context_free_source_owner(
                    owner,
                    reconstruct,
                    "owner",
                    ArrayArchiveLimits(),
                    None,
                )
    assert calls == 0


def test_field_preparation_reuses_only_complete_operation_local_inputs() -> None:
    from dataclasses import replace

    import phydrax as phx
    from phydrax.discretization.fem import FiniteElementPlan
    from phydrax.lifecycle import prepare_meshing_field_owner
    from phydrax.meshing.qualification import (
        periodic_field_templates,
        prepare_periodic_accepted_epoch,
    )
    from tests._support.periodic_meshing import periodic_seed_request
    from tools.meshing_qualification._runtime import qualification_failure

    source, specification = periodic_seed_request()
    result = (
        phx.meshing.NativeMeshingProvider(
            phx.meshing.NativeMeshingOptions("periodic_delaunay")
        )
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )
    spaces = {
        field.name: FiniteElementPlan(
            result.mesh, field, coordinate_spec=result.geometry
        ).prepare()
        for _, field in periodic_field_templates()
    }
    state = {
        name: jax.device_put(
            np.zeros((space.dof_maps[0].global_dof_count,), dtype=np.float64)
        )
        for name, space in spaces.items()
    }
    epoch = prepare_periodic_accepted_epoch(
        result, spaces, state, 0, failure_factory=qualification_failure
    )
    declaration = epoch.declarations[0]
    changed_carrier = phx.meshing.certify_cell_mesh(
        result.mesh, phx.SpatialCoordinateContract.si(), geometry=result.geometry
    )
    with sources.meshing_source_owner_reconstruction_scope():
        first, _ = prepare_meshing_field_owner(declaration, result)
        same, _ = prepare_meshing_field_owner(declaration, result)
        assert same is first
        changed_declaration = replace(declaration, part_name="another-part")
        second, _ = prepare_meshing_field_owner(changed_declaration, result)
        assert second is not first
        with pytest.raises(ValueError, match="MeshingFieldStateRole.index_binding"):
            prepare_meshing_field_owner(declaration, changed_carrier)
    fourth, _ = prepare_meshing_field_owner(declaration, result)
    assert fourth is not first


def test_recertification_integrity_and_canonical_proof_share_caller_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import contextmanager
    from typing import Any

    import phydrax as phx
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import (
        CoordinateEnclosureBudget,
    )
    from phydrax.geometry._embedding_state import MeshCertificateLimits
    from phydrax.meshing._certification import MeshCertificationReport
    from phydrax.meshing._certification_inputs import MeshCertificationInputs
    from tests._support.periodic_meshing import periodic_seed_request

    source, specification = periodic_seed_request()
    result = (
        phx.meshing.NativeMeshingProvider(
            phx.meshing.NativeMeshingOptions("periodic_delaunay")
        )
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )
    certification = result.certification
    assert certification is not None
    validate = MeshCertificationInputs.validate_source_integrity
    integrity_budgets: list[CoordinateEnclosureBudget] = []
    proof_budgets: list[CoordinateEnclosureBudget] = []

    def integrity(
        self: MeshCertificationInputs,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        assert budget is not None
        integrity_budgets.append(budget)
        budget.reserve(3, 0)
        validate(self, budget=budget)

    def proof(*args: Any, **kwargs: Any) -> MeshCertificationReport:
        budget = kwargs["budget"]
        proof_budgets.append(budget)
        budget.reserve(5, 0)
        return certification

    request = sources.coordinate_certificate_request

    @contextmanager
    def caller_only(
        limits: MeshCertificateLimits,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> Iterator[CoordinateEnclosureBudget]:
        assert budget is not None, "restored certification must not create a helper quota"
        with request(limits, budget=budget) as ledger:
            assert ledger is budget
            yield ledger

    monkeypatch.setattr(MeshCertificationInputs, "validate_source_integrity", integrity)
    monkeypatch.setattr(sources, "certify_meshing_acceptance", proof)
    monkeypatch.setattr(sources, "coordinate_certificate_request", caller_only)
    with sources.meshing_source_owner_reconstruction_scope():
        with HostBatchTicket(100_000_000, bytes_upper=1 << 30) as ticket:
            budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=100_000_000,
                maximum_memory_bytes=1 << 30,
            )
            renewed = sources.recertify_restored_meshing_source(
                certification.request,
                result.mesh,
                result.geometry,
                result.audit,
                budget=budget,
            )
            assert renewed is certification
            assert budget.work_units > 8
            assert sources.recertify_restored_meshing_source(
                certification.request,
                result.mesh,
                result.geometry,
                result.audit,
                budget=budget,
            ) is renewed
        assert len(integrity_budgets) == len(proof_budgets) == 1
        assert integrity_budgets[0] is proof_budgets[0] is budget


def test_same_budget_object_cannot_borrow_another_native_request_admission() -> None:
    from phydrax._meshcore import NativeExecutionLimits, NativeExecutionScope
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget

    node = CellBlock("cells", "triangle", np.asarray(((0, 1, 2),), dtype=np.int64))
    calls = 0

    def reconstruct(value: CellBlock) -> CellBlock:
        nonlocal calls
        calls += 1
        return CellBlock.from_stored_fields(value)

    with HostBatchTicket(1_000_000, bytes_upper=64 << 20) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=1_000_000, maximum_memory_bytes=64 << 20
        )
        with sources.meshing_source_owner_reconstruction_scope(budget=budget):
            for _ in range(2):
                with NativeExecutionScope(
                    NativeExecutionLimits(1_000_000, 0, 0, 64 << 20, 120.0)
                ):
                    sources._validate_context_free_source_owner(
                        node, reconstruct, "owner", ArrayArchiveLimits(), None
                    )
                    sources._validate_context_free_source_owner(
                        node, reconstruct, "owner", ArrayArchiveLimits(), None
                    )
            assert calls == 2


def test_source_memo_binds_nested_current_owning_constructor_not_only_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources.register_meshing_source_artifacts()
    block = CellBlock("cells", "triangle", np.asarray(((0, 1, 2),), dtype=np.int64))
    value = (block,)
    calls = 0
    original = CellBlock.from_stored_fields

    def reconstruct(record: tuple[CellBlock, ...]) -> tuple[CellBlock, ...]:
        nonlocal calls
        calls += 1
        return tuple(CellBlock.from_stored_fields(item) for item in record)

    def changed(cls: type[CellBlock], item: CellBlock) -> CellBlock:
        return original(item)

    with sources.meshing_source_owner_reconstruction_scope():
        sources._validate_context_free_source_owner(
            value, reconstruct, "owner", ArrayArchiveLimits(), None
        )
        sources._validate_context_free_source_owner(
            value, reconstruct, "owner", ArrayArchiveLimits(), None
        )
        assert calls == 1
        with monkeypatch.context() as patch:
            patch.setattr(CellBlock, "from_stored_fields", classmethod(changed))
            sources._validate_context_free_source_owner(
                value, reconstruct, "owner", ArrayArchiveLimits(), None
            )
        assert calls == 2


@pytest.mark.parametrize("registered_forwarder", (False, True))
def test_registered_direct_owner_uses_current_constructor_and_reuses_its_admission(
    monkeypatch: pytest.MonkeyPatch,
    registered_forwarder: bool,
) -> None:
    sources.register_meshing_source_artifacts()
    block = CellBlock("cells", "triangle", np.asarray(((0, 1, 2),), dtype=np.int64))
    original = CellBlock.from_stored_fields
    registered = sources._source_admissions()[CellBlock].reconstruct
    assert registered is not None
    calls: list[str] = []

    def first(cls: type[CellBlock], value: CellBlock) -> CellBlock:
        calls.append("first")
        return original(value)

    def second(cls: type[CellBlock], value: CellBlock) -> CellBlock:
        calls.append("second")
        return original(value)

    monkeypatch.setattr(CellBlock, "from_stored_fields", classmethod(first))
    if registered_forwarder:
        # A canonical slot may hold a forwarding method named "first", not
        # "from_stored_fields", when the admission table first captures it.
        admissions = dict(sources._source_admissions())
        admissions[CellBlock] = replace(
            admissions[CellBlock], reconstruct=CellBlock.from_stored_fields
        )
        monkeypatch.setattr(sources, "_source_admissions", lambda: admissions)
    with sources.meshing_source_owner_reconstruction_scope():
        for _ in range(2):
            sources.reconstruct_meshing_source(block)
        assert calls == ["first"]
        with monkeypatch.context() as patch:
            patch.setattr(CellBlock, "from_stored_fields", classmethod(second))
            for _ in range(2):
                sources.reconstruct_meshing_source(block)
            assert calls == ["first", "second"]
        sources.reconstruct_meshing_source(block)
        assert calls == ["first", "second"]
    sources.reconstruct_meshing_source(block)
    assert calls == ["first", "second", "first"]


def test_actual_receiving_limits_bind_preparation_and_restore_nested_policy() -> None:
    from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS

    receiving = replace(DEFAULT_ARRAY_ARCHIVE_LIMITS, max_member_bytes=4096)
    inner = replace(receiving, max_member_bytes=2048)
    block = CellBlock("cells", "triangle", np.asarray(((0, 1, 2),), dtype=np.int64))
    sources.register_meshing_source_artifacts()
    constructor = sources._source_admissions()[CellBlock].reconstruct
    assert constructor is not None
    with sources.meshing_source_owner_reconstruction_scope():
        scope = sources._source_owner_reconstruction_scope.get()
        assert scope is not None
        previous = scope.archive_limits
        with sources.meshing_source_owner_reconstruction_scope(archive_limits=receiving):
            assert scope.archive_limits is receiving
            before = sources._source_owner_reconstruction_key(
                block, scope.archive_limits, None, constructor, scope
            )
            with sources.meshing_source_owner_reconstruction_scope():
                assert scope.archive_limits is receiving
            with sources.meshing_source_owner_reconstruction_scope(archive_limits=inner):
                assert scope.archive_limits is inner
            assert scope.archive_limits is receiving
            sources.validate_meshing_source_closure(block, limits=receiving)
            after = sources._source_owner_reconstruction_key(
                block, scope.archive_limits, None, constructor, scope
            )
            assert after == before
        assert scope.archive_limits is previous
