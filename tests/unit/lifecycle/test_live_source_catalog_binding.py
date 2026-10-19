# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Live source catalogs preserve checkpoint framing and numerical admission."""

from dataclasses import replace
from typing import Any

import jax
import pytest

from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
from phydrax._fingerprint import canonical_json
from phydrax._model._structure import (
    model_recipe_array_inventory,
    model_recipe_array_values,
    model_structure_recipe,
)
from phydrax.lifecycle import _meshing_checkpoint_binding as binding
from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts
from tests._support.distributed_meshing.restart_builders import native_source_closure


@pytest.fixture(scope="module")
def live_closure() -> dict[str, Any]:
    register_meshing_source_artifacts()
    return native_source_closure()[3]


def test_live_binding_matches_legacy_recipe_roles_and_content(
    live_closure: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = model_structure_recipe(live_closure)
    inventory = model_recipe_array_inventory(recipe, prefix="source-record")
    values = model_recipe_array_values(live_closure, recipe, prefix="source-record")
    expected_digest = binding.meshing_source_content_digest(
        values,
        {},
        None,
        None,
        maximum_chunk_bytes=1 << 20,
        owner_index=3,
        array_paths={entry.name: entry.path for entry in inventory},
    )
    expected = binding.prepare_meshing_source_binding(
        None,
        canonical_json(recipe),
        expected_digest,
        owner_index=3,
        array_bindings={},
        logical_arrays=None,
        logical_array_names=(),
        logical_content_digest=None,
    )

    def reject_descriptor_catalog(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "live_catalog.descriptor must remain unused during live binding; got a descriptor catalog call"
        )

    monkeypatch.setattr(
        binding, "model_recipe_array_inventory", reject_descriptor_catalog
    )
    actual = binding.prepare_meshing_source_binding(
        live_closure,
        canonical_json(recipe),
        expected_digest,
        owner_index=3,
        array_bindings={},
        logical_arrays=None,
        logical_array_names=(),
        logical_content_digest=None,
    )
    assert actual.recipe_json == expected.recipe_json
    assert actual.content_digest == expected.content_digest
    assert actual.array_roles == expected.array_roles
    assert actual.logical_array_names == expected.logical_array_names
    assert actual.logical_content_digest == expected.logical_content_digest


def test_live_binding_retains_recipe_mismatch(
    live_closure: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="must match the retained source records"):
        binding.prepare_meshing_source_binding(
            live_closure,
            "{}",
            None,
            owner_index=0,
            array_bindings={},
            logical_arrays=None,
            logical_array_names=(),
            logical_content_digest=None,
        )


def test_live_binding_retains_manifest_limit(live_closure: dict[str, Any]) -> None:
    limits = replace(DEFAULT_ARRAY_ARCHIVE_LIMITS, max_manifest_bytes=1)
    with pytest.raises((ValueError, binding.RepositoryCorruptionError)):
        binding.prepare_meshing_source_binding(
            live_closure,
            None,
            None,
            owner_index=0,
            array_bindings={},
            logical_arrays=None,
            logical_array_names=(),
            logical_content_digest=None,
            limits=limits,
        )


def test_source_digest_rejects_equal_but_distinct_alias_value(
    live_closure: dict[str, Any],
) -> None:
    recipe = model_structure_recipe(live_closure)
    inventory = model_recipe_array_inventory(recipe, prefix="source-record")
    values = model_recipe_array_values(live_closure, recipe, prefix="source-record")
    name, value = next(
        (name, value) for name, value in values.items() if isinstance(value, jax.Array)
    )
    with pytest.raises(ValueError, match="must retain logical binding.*value identity"):
        binding.meshing_source_content_digest(
            values,
            {name: "shared"},
            {"shared": value.copy()},
            "unused",
            maximum_chunk_bytes=1 << 20,
            owner_index=0,
            array_paths={entry.name: entry.path for entry in inventory},
        )


def test_live_binding_authenticates_supplied_content_digest(
    live_closure: dict[str, Any],
) -> None:
    recipe = model_structure_recipe(live_closure)
    values = model_recipe_array_values(live_closure, recipe, prefix="source-record")
    name, value = next(
        (name, value)
        for name, value in values.items()
        if value.size and value.dtype.kind == "f"
    )
    changed = dict(values)
    changed[name] = value + 1
    inventory = model_recipe_array_inventory(recipe, prefix="source-record")
    changed_digest = binding.meshing_source_content_digest(
        changed,
        {},
        None,
        None,
        maximum_chunk_bytes=1 << 20,
        owner_index=0,
        array_paths={entry.name: entry.path for entry in inventory},
    )
    with pytest.raises(ValueError, match="source numerical content changed"):
        binding.prepare_meshing_source_binding(
            live_closure,
            canonical_json(recipe),
            changed_digest,
            owner_index=0,
            array_bindings={},
            logical_arrays=None,
            logical_array_names=(),
            logical_content_digest=None,
        )
