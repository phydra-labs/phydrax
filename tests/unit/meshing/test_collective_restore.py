"""Authenticated restoration retains proofs without consuming native reservations."""

import json
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._fingerprint import tree_content_equal
from phydrax._model._structure import (
    model_recipe_array_values,
    model_structure_recipe,
    prepare_model_recipe,
    restore_registered_fields,
)
from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts
from tests._support.forced_devices import run_on_forced_devices


@pytest.fixture(scope="module")
def restore_receipt(
    tmp_path_factory: pytest.TempPathFactory,
    record_testsuite_property: Callable[[str, object], None],
) -> dict[str, Any]:
    archive = tmp_path_factory.mktemp("collective-restore") / "source.npz"
    receipt = run_on_forced_devices(
        "tests._support.distributed_meshing.native_distributed_restore",
        "collective_restore",
        (str(archive),),
        devices=2,
        env={},
    )
    record_testsuite_property("restore_receipt", json.dumps(receipt, sort_keys=True))
    return receipt


def test_restore_inside_consumed_native_reservation(
    restore_receipt: dict[str, Any],
) -> None:
    assert restore_receipt["equal"]
    assert restore_receipt["domain_equal"]
    assert restore_receipt["reservation_work"] == 1
    assert restore_receipt["restore_native_work"] == 0
    assert restore_receipt["restore_native_queries"] == 0
    assert restore_receipt["restore_external_work"] == 1


@pytest.mark.parametrize("fault", ("proof", "bank", "id", "anchored_forgery"))
def test_collective_restore_refuses_tampering(
    restore_receipt: dict[str, Any], fault: str
) -> None:
    assert restore_receipt["faults"][fault]


def test_explicit_validator_replays_restored_mathematics(
    restore_receipt: dict[str, Any],
) -> None:
    assert restore_receipt["replay_native_queries"] > 0
    assert restore_receipt["replay_external_work"] > 0


def test_authenticated_records_persist_roundtrip(
    restore_receipt: dict[str, Any],
) -> None:
    assert restore_receipt["roundtrip"]
    assert restore_receipt["archive_content_id"]


def test_owner_local_partition_restores_its_storage_binding(
    restore_receipt: dict[str, Any],
) -> None:
    assert restore_receipt["partition_equal"]


def test_owner_local_partition_restore_refuses_missing_storage(
    restore_receipt: dict[str, Any],
) -> None:
    assert restore_receipt["partition_missing_storage_refused"]


def test_owner_local_distribution_restores_without_replacing_inventories(
    restore_receipt: dict[str, Any],
) -> None:
    assert restore_receipt["distribution_equal"]


def test_owner_local_distribution_restore_refuses_tampered_bank(
    restore_receipt: dict[str, Any],
) -> None:
    assert restore_receipt["distribution_tampered_bank_refused"]


def test_collective_storage_binding_preserves_scientific_result_identity(
    restore_receipt: dict[str, Any],
) -> None:
    # Main ID-15 replaces the abandoned result-ID anchor with a storage anchor.
    assert restore_receipt["result_id"] == restore_receipt["prior_result_id"]
    assert (
        restore_receipt["parent_stored_content_id"]
        == restore_receipt["stored_content_id"]
    )
    assert restore_receipt["stored_content_id"] != restore_receipt["evidence_id"]


@pytest.mark.parametrize("field", ("vertices", "facets", "facet_regions"))
def test_domain_restore_refuses_changed_bank(field: str) -> None:
    vertices = np.asarray(((0, 0), (1, 0), (0, 1)), dtype=np.float64)
    facets = np.asarray(((0, 1), (1, 2), (2, 0)), dtype=np.int64)
    regions = np.tile(np.asarray((0, -1), dtype=np.int64), (3, 1))
    domain = phx.geometry.PiecewiseLinearDomain(
        vertices, facets, regions, ("region",), source_id="restore-domain"
    )
    altered = eqx.tree_at(
        lambda value: getattr(value, field), domain, getattr(domain, field) + 1
    )
    with pytest.raises(ValueError, match=r"PiecewiseLinearDomain\.source_revision"):
        phx.geometry.PiecewiseLinearDomain.from_stored_fields(altered)


@pytest.mark.parametrize("field", ("source_revision", "domain_id"))
def test_domain_restore_refuses_changed_identity(field: str) -> None:
    vertices = np.asarray(((0, 0), (1, 0), (0, 1)), dtype=np.float64)
    facets = np.asarray(((0, 1), (1, 2), (2, 0)), dtype=np.int64)
    regions = np.tile(np.asarray((0, -1), dtype=np.int64), (3, 1))
    domain = phx.geometry.PiecewiseLinearDomain(
        vertices, facets, regions, ("region",), source_id="restore-domain"
    )
    register_meshing_source_artifacts()
    stored = {
        "vertices": domain.vertices,
        "facets": domain.facets,
        "facet_regions": domain.facet_regions,
        "region_ids": domain.region_ids,
        "source_id": domain.source_id,
        "source_revision": domain.source_revision,
        "domain_id": domain.domain_id,
    }
    stored[field] = "tampered-domain-identity"
    altered = restore_registered_fields(phx.geometry.PiecewiseLinearDomain, stored)
    with pytest.raises(ValueError, match=rf"PiecewiseLinearDomain\.{field}"):
        phx.geometry.PiecewiseLinearDomain.from_stored_fields(altered)


def test_live_recipe_preparation_preserves_canonical_array_bindings() -> None:
    vertices = np.asarray(((0, 0), (1, 0), (0, 1)), dtype=np.float64)
    facets = np.asarray(((0, 1), (1, 2), (2, 0)), dtype=np.int64)
    regions = np.tile(np.asarray((0, -1), dtype=np.int64), (3, 1))
    domain = phx.geometry.PiecewiseLinearDomain(
        vertices, facets, regions, ("region",), source_id="recipe-domain"
    )
    register_meshing_source_artifacts()
    value = (
        domain,
        domain,
        jnp.asarray((0.5, 0.25), dtype=jnp.float64),
        {"bank": np.asarray((1, 2), dtype=np.int64)},
    )
    expected = model_structure_recipe(value)
    expected_arrays = model_recipe_array_values(value, expected, prefix="stored_fields")
    recipe, arrays = prepare_model_recipe(value, prefix="stored_fields")
    assert tree_content_equal(recipe, expected)
    assert tree_content_equal(arrays, expected_arrays)
