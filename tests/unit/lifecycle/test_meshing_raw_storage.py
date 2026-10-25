"""Raw archive banks remain admitted alongside genuine reconstructed owners."""

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

from phydrax._array_archive import ArrayArchiveLimits, DEFAULT_ARRAY_ARCHIVE_LIMITS
from phydrax._host_storage import retained_host_storage_profile
from phydrax._meshcore import is_native_host_buffer
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.lifecycle import (
    _meshing_field_records as fields,
    _meshing_sources as sources,
)
from phydrax.meshing.qualification._native_lifecycle import execute_native_request
from tests.unit.lifecycle.test_meshing_polyhedral_fields import _fail, actual_epoch


@pytest.mark.parametrize("inherited_budget", (False, True), ids=("explicit", "scope"))
def test_raw_archive_admitted_before_real_decoder(
    actual_epoch: tuple[fields.MeshingAcceptedEpoch, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inherited_budget: bool,
) -> None:
    epoch, records = actual_epoch
    receipt = sources.write_meshing_source_closure(tmp_path / "raw.phx", records)
    controls = sources.read_meshing_source_execution_controls(
        receipt.path, expected_content_id=receipt.content_id
    )
    limits = replace(
        sources.DEFAULT_ARRAY_ARCHIVE_LIMITS,
        max_member_bytes=sources.DEFAULT_ARRAY_ARCHIVE_LIMITS.max_member_bytes // 2,
    )
    archive = sources._read_meshing_source_archive(
        receipt.path, expected_content_id=receipt.content_id, limits=limits
    )
    recipe, arrays, _ = archive
    raw_owners, _, _, _ = retained_host_storage_profile(
        archive, {}, is_native_host_buffer
    )
    bases = []
    for array in arrays.values():
        assert isinstance(array, np.ndarray)
        assert array.base is not None
        base = array
        while isinstance(base.base, np.ndarray):
            base = base.base
        bases.append(base)
    assert len({id(base) for base in bases}) < len(bases)
    decoder = sources.model_from_logical_array_recipe
    decoder_calls = 0
    request_budget: CoordinateEnclosureBudget | None = None

    def decode(
        actual_recipe: Mapping[str, Any],
        actual_arrays: Mapping[str, jax.Array | np.ndarray],
        /,
        *,
        prefix: str,
        limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
    ) -> Any:
        nonlocal decoder_calls
        decoder_calls += 1
        assert actual_recipe is recipe
        assert actual_arrays is arrays
        scope = sources._source_owner_reconstruction_scope.get()
        assert scope is not None and scope.budget is request_budget
        assert scope.archive_limits is limits
        assert request_budget is not None
        retained = request_budget.retained_owner_objects
        assert retained[id(archive)] is archive
        for identity, owner in raw_owners.items():
            assert retained[identity] is owner
        # The existing profiler sees neither additional bank bytes nor alias owners.
        assert retained_host_storage_profile(
            archive, retained, is_native_host_buffer
        ) == ({}, 0, 0, 0)
        before = (request_budget.retained_basis_bytes,
                  request_budget.logical_array_bytes_upper, request_budget.work_units)
        for array in (*arrays.values(), *bases):
            request_budget.retain_owner(array)
        assert (request_budget.retained_basis_bytes,
                request_budget.logical_array_bytes_upper,
                request_budget.work_units) == before
        return decoder(actual_recipe, actual_arrays, prefix=prefix, limits=limits)

    monkeypatch.setattr(sources, "model_from_logical_array_recipe", decode)

    def restore(budget: CoordinateEnclosureBudget) -> fields.MeshingAcceptedEpoch:
        nonlocal request_budget
        request_budget = budget
        with sources.meshing_source_owner_reconstruction_scope(budget=budget):
            restored = sources._restore_meshing_source_archive(
                archive, limits=limits,
                budget=None if inherited_budget else budget,
            )
            scope = sources._source_owner_reconstruction_scope.get()
            assert scope is not None and scope.budget is budget
        # Scope exit does not retire captures still alive in this fresh request.
        assert budget.retained_owner_objects[id(archive)] is archive
        (current,) = fields.validate_meshing_accepted_epochs(restored)
        for name, value in epoch.fields.items():
            np.testing.assert_array_equal(current.fields[name], value)
        return current

    result = execute_native_request(
        controls.execution_evidence, controls.limits, restore,
        target=lambda current: current.carrier, failure_factory=_fail,
    )
    assert decoder_calls == 1
    assert result.value.epoch_id == epoch.epoch_id
    assert result.result.execution_evidence is not None
    assert result.result.execution_evidence.history_evidence
