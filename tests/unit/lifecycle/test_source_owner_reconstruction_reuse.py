# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Restore-local owner reuse requires freshly identical complete content."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

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
                {"jax": device, "host": host}, scope, 8
            )
            assert calls == 4 * repeat - 2
            repeated = sources._source_owner_array_digest(
                {"jax": device, "host": host}, scope, 8
            )
            assert first == repeated
            assert calls == 4 * repeat - 1
            host[0] = 0.0 if np.signbit(host[0]) else -0.0
            changed = sources._source_owner_array_digest(
                {"jax": device, "host": host}, scope, 8
            )
            assert first != changed
            assert calls == 4 * repeat
            renamed = sources._source_owner_array_digest({"renamed": device}, scope, 8)
            named = sources._source_owner_array_digest({"jax": device}, scope, 8)
            assert renamed != named
            assert calls == 4 * repeat


def test_deleted_immutable_array_cannot_use_an_earned_hash() -> None:
    device = jax.device_put(np.asarray((1.0,), dtype=np.float64))
    with sources.meshing_source_owner_reconstruction_scope():
        scope = sources._source_owner_reconstruction_scope.get()
        assert scope is not None
        sources._source_owner_array_digest({"value": device}, scope, 8)
        device.delete()
        with pytest.raises(RuntimeError, match="deleted"):
            sources._source_owner_array_digest({"value": device}, scope, 8)


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


def test_field_preparation_reuses_only_complete_operation_local_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace
    from typing import Any

    import phydrax as phx
    from phydrax.discretization.fem import FiniteElementDiscretization, FiniteElementPlan
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
    original = FiniteElementPlan.prepare
    calls = 0

    def counted(
        plan: FiniteElementPlan, *args: Any, **kwargs: Any
    ) -> FiniteElementDiscretization:
        nonlocal calls
        calls += 1
        return original(plan, *args, **kwargs)

    monkeypatch.setattr(FiniteElementPlan, "prepare", counted)
    with sources.meshing_source_owner_reconstruction_scope():
        first, _ = prepare_meshing_field_owner(declaration, result)
        same, _ = prepare_meshing_field_owner(declaration, result)
        assert same is first and calls == 1
        changed_declaration = replace(declaration, part_name="another-part")
        second, _ = prepare_meshing_field_owner(changed_declaration, result)
        assert second is not first and calls == 2
        with pytest.raises(ValueError, match="MeshingFieldStateRole.index_binding"):
            prepare_meshing_field_owner(declaration, changed_carrier)
        assert calls == 3
    fourth, _ = prepare_meshing_field_owner(declaration, result)
    assert fourth is not first and calls == 4
