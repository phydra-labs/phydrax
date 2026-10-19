# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Whole-source publication admission remains tied to live scientific content."""

import copy
import gc
import os
import pickle
import weakref
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np
import pytest

from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
from phydrax._fingerprint import canonical_json
from phydrax.lifecycle import _meshing_checkpoint_binding as binding
from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts
from tests._support.distributed_meshing.restart_builders import native_source_closure
from tests._support.meshing_field_declarations import (
    prepare_single_native_field_scenario,
)


@pytest.fixture(scope="module")
def source_records() -> dict[str, Any]:
    register_meshing_source_artifacts()
    return native_source_closure()[3]


def _admit(
    records: Mapping[str, Any],
) -> tuple[binding.MeshingSourceBinding, dict[str, str]]:
    _, _, values = binding.prepare_model_recipe_catalog(
        records, prefix=binding.MESHING_SOURCE_PREFIX
    )
    aliases = {
        name: f"shared-{position}"
        for position, (name, value) in enumerate(values.items())
        if isinstance(value, jax.Array)
    }
    bank = {
        aliases[name]: value
        for name, value in values.items()
        if isinstance(value, jax.Array)
    }
    admitted = binding.prepare_meshing_source_binding(
        records,
        None,
        None,
        owner_index=3,
        array_bindings=aliases,
        logical_arrays=bank,
        logical_array_names=(),
        logical_content_digest=None,
    )
    return admitted, aliases


def _arguments(
    admitted: binding.MeshingSourceBinding, aliases: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "owner_index": 3,
        "array_bindings": aliases,
        "logical_arrays": admitted.logical_arrays,
        "logical_array_names": admitted.logical_array_names,
        "logical_content_digest": admitted.logical_content_digest,
    }


def _observe_semantics(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    calls: list[object] = []
    validate = binding.validate_meshing_source_closure

    def observe(value: Any, /, **kwargs: Any) -> None:
        calls.append(value)
        validate(value, **kwargs)

    monkeypatch.setattr(binding, "validate_meshing_source_closure", observe)
    return calls


def test_identical_admission_rehashes_live_content_without_semantic_revalidation(
    source_records: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted, aliases = _admit(source_records)
    semantics = _observe_semantics(monkeypatch)
    digests: list[object] = []
    content_digest = binding.meshing_source_content_digest

    def observe_digest(*args: Any, **kwargs: Any) -> str:
        digests.append(args[0])
        return content_digest(*args, **kwargs)

    monkeypatch.setattr(binding, "meshing_source_content_digest", observe_digest)
    reused = binding.authenticate_meshing_source_publication(
        source_records,
        admitted.recipe_json,
        admitted.content_digest,
        **_arguments(admitted, aliases),
    )
    assert semantics == []
    assert len(digests) == 1
    assert reused.recipe_json == admitted.recipe_json
    assert reused.content_digest == admitted.content_digest
    assert reused.array_roles == admitted.array_roles
    assert reused.logical_arrays is admitted.logical_arrays
    assert reused.logical_arrays is not None
    assert reused.array_values is digests[0]
    assert reused.array_values is not None
    for name, alias in aliases.items():
        assert reused.array_values[name] is reused.logical_arrays[alias]


def test_public_bank_copy_drops_whole_source_admission(
    source_records: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted, aliases = _admit(source_records)
    bank = admitted.logical_arrays
    assert bank is not None
    copied = binding.AdmittedMeshingSourceArrays(
        bank,
        maximum_chunk_bytes=bank.maximum_chunk_bytes,
        source_validation=bank.source_validation,
    )
    assert copied._arrays is bank._arrays
    assert copied.source_validation is bank.source_validation
    assert copied._publication_slot.admission is None
    semantics = _observe_semantics(monkeypatch)
    arguments = _arguments(admitted, aliases)
    arguments["logical_arrays"] = copied
    result = binding.authenticate_meshing_source_publication(
        source_records, admitted.recipe_json, admitted.content_digest, **arguments
    )
    assert len(semantics) == 1
    assert result.content_digest == admitted.content_digest


def test_recipe_only_binding_carries_no_publication_authority(
    source_records: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted, aliases = _admit(source_records)
    semantics = _observe_semantics(monkeypatch)
    cold = binding.authenticate_meshing_source_publication(
        None,
        admitted.recipe_json,
        admitted.content_digest,
        **_arguments(admitted, aliases),
    )
    assert semantics == []
    assert cold.array_values is None
    assert cold.logical_arrays is not None
    assert cold.logical_arrays._publication_slot.admission is None


def test_mutated_numpy_payload_uses_full_validator_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_meshing_source_artifacts()
    scenario = prepare_single_native_field_scenario()
    source_records = scenario.closure
    host = scenario.coefficients
    writable = host.flags.writeable
    saved = host.copy()
    host.setflags(write=True)
    try:
        admitted, aliases = _admit(source_records)
        arguments = _arguments(admitted, aliases)
        semantics = _observe_semantics(monkeypatch)
        host.flat[0] = np.nan
        with pytest.raises(ValueError) as full_error:
            binding.prepare_meshing_source_binding(
                source_records, admitted.recipe_json, admitted.content_digest, **arguments
            )
        semantics.clear()
        with pytest.raises(type(full_error.value)) as reused_error:
            binding.authenticate_meshing_source_publication(
                source_records, admitted.recipe_json, admitted.content_digest, **arguments
            )
        assert str(reused_error.value) == str(full_error.value)
        assert len(semantics) == 1
    finally:
        host[...] = saved
        host.setflags(write=writable)


def test_replaced_alias_value_preserves_original_first_identity_refusal(
    source_records: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted, aliases = _admit(source_records)
    bank = admitted.logical_arrays
    assert bank is not None
    name = next(iter(aliases))
    alias = aliases[name]
    changed = dict(bank)
    changed[alias] = bank[alias].copy()
    replacement = binding.AdmittedMeshingSourceArrays(
        changed, maximum_chunk_bytes=bank.maximum_chunk_bytes
    )
    replacement._publication_slot.admission = bank._publication_slot.admission
    arguments = _arguments(admitted, aliases)
    arguments["logical_arrays"] = replacement
    semantics = _observe_semantics(monkeypatch)
    with pytest.raises(ValueError) as full_error:
        binding.prepare_meshing_source_binding(
            source_records, admitted.recipe_json, admitted.content_digest, **arguments
        )
    semantics.clear()
    with pytest.raises(type(full_error.value)) as reused_error:
        binding.authenticate_meshing_source_publication(
            source_records, admitted.recipe_json, admitted.content_digest, **arguments
        )
    assert str(reused_error.value) == str(full_error.value)
    assert "value identity" in str(reused_error.value)
    assert len(semantics) == 1


def test_replaced_immutable_record_requires_fresh_semantic_admission(
    source_records: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted, aliases = _admit(source_records)
    changed = dict(source_records)
    changed["report"] = copy.copy(source_records["report"])
    recipe, _, _ = binding.prepare_model_recipe_catalog(
        changed, prefix=binding.MESHING_SOURCE_PREFIX
    )
    assert changed["report"] is not source_records["report"]
    assert canonical_json(recipe) == admitted.recipe_json
    semantics = _observe_semantics(monkeypatch)
    result = binding.authenticate_meshing_source_publication(
        changed,
        admitted.recipe_json,
        admitted.content_digest,
        **_arguments(admitted, aliases),
    )
    assert len(semantics) == 1
    assert result.content_digest == admitted.content_digest


@pytest.mark.parametrize("copied_slots", (False, True), ids=("empty", "copied-slots"))
def test_unissued_exact_type_token_falls_back(
    source_records: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    copied_slots: bool,
) -> None:
    admitted, aliases = _admit(source_records)
    bank = admitted.logical_arrays
    assert bank is not None
    issued = bank._publication_slot.admission
    assert issued is not None
    forged = object.__new__(type(issued))
    if copied_slots:
        object.__setattr__(forged, "snapshot", issued.snapshot)
        object.__setattr__(forged, "process_id", issued.process_id)
    bank._publication_slot.admission = forged
    semantics = _observe_semantics(monkeypatch)
    result = binding.authenticate_meshing_source_publication(
        source_records,
        admitted.recipe_json,
        admitted.content_digest,
        **_arguments(admitted, aliases),
    )
    assert len(semantics) == 1
    assert result.content_digest == admitted.content_digest


def test_foreign_owner_token_falls_back_to_valid_full_outcome(
    source_records: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted, aliases = _admit(source_records)
    arguments = _arguments(admitted, aliases)
    arguments["owner_index"] = 9
    foreign = binding.prepare_meshing_source_binding(
        source_records, admitted.recipe_json, admitted.content_digest, **arguments
    )
    assert admitted.logical_arrays is not None
    assert foreign.logical_arrays is not None
    admitted.logical_arrays._publication_slot.admission = (
        foreign.logical_arrays._publication_slot.admission
    )
    semantics = _observe_semantics(monkeypatch)
    result = binding.authenticate_meshing_source_publication(
        source_records,
        admitted.recipe_json,
        admitted.content_digest,
        **_arguments(admitted, aliases),
    )
    assert len(semantics) == 1
    assert result.array_roles == admitted.array_roles


def test_foreign_process_token_falls_back(
    source_records: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted, aliases = _admit(source_records)
    process_id = os.getpid()
    monkeypatch.setattr(binding, "os", SimpleNamespace(getpid=lambda: process_id + 1))
    semantics = _observe_semantics(monkeypatch)
    result = binding.authenticate_meshing_source_publication(
        source_records,
        admitted.recipe_json,
        admitted.content_digest,
        **_arguments(admitted, aliases),
    )
    assert len(semantics) == 1
    assert result.content_digest == admitted.content_digest


def test_token_cannot_be_constructed_or_serialized(
    source_records: dict[str, Any],
) -> None:
    admitted, _ = _admit(source_records)
    assert admitted.logical_arrays is not None
    token = admitted.logical_arrays._publication_slot.admission
    assert token is not None
    with pytest.raises(TypeError, match="issued by full source validation"):
        type(token)()
    with pytest.raises(TypeError, match="remain process-local"):
        pickle.dumps(token)


def test_issuance_registry_does_not_retain_publication_requests(
    source_records: dict[str, Any],
) -> None:
    admitted, _ = _admit(source_records)
    assert admitted.logical_arrays is not None
    token = admitted.logical_arrays._publication_slot.admission
    assert token is not None
    reference = weakref.ref(token)
    del token, admitted
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize(
    "field, replacement",
    (
        ("recipe_json", "{}"),
        ("content_digest", "0" * 64),
        ("logical_array_names", ("undeclared",)),
        ("logical_content_digest", "0" * 64),
        ("array_bindings", {"undeclared": "shared"}),
        ("limits", replace(DEFAULT_ARRAY_ARCHIVE_LIMITS, max_manifest_bytes=1)),
    ),
    ids=("recipe", "content", "bank-names", "bank-digest", "aliases", "limits"),
)
def test_mismatch_preserves_full_validation_error_order(
    source_records: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: Any,
) -> None:
    admitted, aliases = _admit(source_records)
    arguments = _arguments(admitted, aliases)
    recipe_json = admitted.recipe_json
    content_digest = admitted.content_digest
    if field == "recipe_json":
        recipe_json = replacement
    elif field == "content_digest":
        content_digest = replacement
    else:
        arguments[field] = replacement
    semantics = _observe_semantics(monkeypatch)
    with pytest.raises((ValueError, binding.RepositoryCorruptionError)) as full_error:
        binding.prepare_meshing_source_binding(
            source_records, recipe_json, content_digest, **arguments
        )
    semantics.clear()
    with pytest.raises(type(full_error.value)) as reused_error:
        binding.authenticate_meshing_source_publication(
            source_records, recipe_json, content_digest, **arguments
        )
    assert str(reused_error.value) == str(full_error.value)
    assert len(semantics) == 1


def test_unrelated_catalog_bug_is_not_hidden_by_full_validation(
    source_records: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted, aliases = _admit(source_records)
    semantics = _observe_semantics(monkeypatch)

    def broken_catalog(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "catalog.program must execute successfully; got injected failure"
        )

    monkeypatch.setattr(binding, "prepare_model_recipe_catalog", broken_catalog)
    with pytest.raises(RuntimeError, match="got injected failure"):
        binding.authenticate_meshing_source_publication(
            source_records,
            admitted.recipe_json,
            admitted.content_digest,
            **_arguments(admitted, aliases),
        )
    assert semantics == []


def test_public_bank_constructor_cannot_import_publication_token(
    source_records: dict[str, Any],
) -> None:
    admitted, _ = _admit(source_records)
    bank = admitted.logical_arrays
    assert bank is not None
    arguments: dict[str, Any] = {
        "maximum_chunk_bytes": bank.maximum_chunk_bytes,
        "publication_admission": bank._publication_slot.admission,
    }
    with pytest.raises(TypeError, match="publication_admission"):
        binding.AdmittedMeshingSourceArrays(bank, **arguments)


@pytest.mark.parametrize("identity", ("map", "validation"))
def test_replaced_common_bank_authority_requires_full_admission(
    source_records: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    identity: str,
) -> None:
    admitted, aliases = _admit(source_records)
    bank = admitted.logical_arrays
    assert bank is not None
    arrays = dict(bank) if identity == "map" else bank
    validation = (
        copy.copy(bank.source_validation)
        if identity == "validation"
        else bank.source_validation
    )
    changed = binding.AdmittedMeshingSourceArrays(
        arrays,
        maximum_chunk_bytes=bank.maximum_chunk_bytes,
        source_validation=validation,
    )
    changed._publication_slot.admission = bank._publication_slot.admission
    arguments = _arguments(admitted, aliases)
    arguments["logical_arrays"] = changed
    semantics = _observe_semantics(monkeypatch)
    result = binding.authenticate_meshing_source_publication(
        source_records, admitted.recipe_json, admitted.content_digest, **arguments
    )
    assert len(semantics) == 1
    assert result.content_digest == admitted.content_digest


def test_recipe_equivalent_unadmitted_mapping_runs_full_validation(
    source_records: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted, aliases = _admit(source_records)
    replacement = defaultdict(None, source_records)
    arguments = _arguments(admitted, aliases)
    semantics = _observe_semantics(monkeypatch)
    with pytest.raises(TypeError) as full_error:
        binding.prepare_meshing_source_binding(
            replacement, admitted.recipe_json, admitted.content_digest, **arguments
        )
    semantics.clear()
    with pytest.raises(type(full_error.value)) as reused_error:
        binding.authenticate_meshing_source_publication(
            replacement, admitted.recipe_json, admitted.content_digest, **arguments
        )
    assert str(reused_error.value) == str(full_error.value)
    assert len(semantics) == 1


def test_malformed_alias_map_preserves_full_validation_error_order(
    source_records: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted, aliases = _admit(source_records)
    arguments = _arguments(admitted, aliases)
    arguments["array_bindings"] = {**aliases, 3: "undeclared"}
    semantics = _observe_semantics(monkeypatch)
    with pytest.raises(ValueError) as full_error:
        binding.prepare_meshing_source_binding(
            source_records, admitted.recipe_json, admitted.content_digest, **arguments
        )
    semantics.clear()
    with pytest.raises(type(full_error.value)) as reused_error:
        binding.authenticate_meshing_source_publication(
            source_records, admitted.recipe_json, admitted.content_digest, **arguments
        )
    assert str(reused_error.value) == str(full_error.value)
    assert len(semantics) == 1
