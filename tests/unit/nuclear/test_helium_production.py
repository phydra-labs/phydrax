from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.nuclear import (
    AVOGADRO_PER_MOL,
    FISSIONING_NUCLIDES,
    HeliumProductionPlan,
    InventoryTransition,
    NuclearDataProvenance,
    NuclideInventory,
    NuclideKey,
)
from phydrax.qualification import ReferenceArtifactManifest


def _source() -> NuclearDataProvenance:
    payload = b"synthetic-helium-production-reference"
    artifact = ReferenceArtifactManifest(
        "synthetic-helium-production",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="synthetic-owned-test-input",
        commercial_use_permitted=False,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=False,
        export_classification="fixture",
        nondimensionalization={"identity": 1.0},
        uncertainty=None,
        lineage_ids=("synthetic",),
    )
    return NuclearDataProvenance(
        artifact,
        "https://example.invalid/synthetic-helium-production",
        "synthetic-library",
        "synthetic-release",
        "synthetic-helium-production",
    )


def _plan() -> tuple[HeliumProductionPlan, NuclideInventory]:
    parent = NuclideKey(92, 238)
    daughter = NuclideKey(90, 234)
    helium = NuclideKey(2, 4)
    source = _source()
    transition = InventoryTransition(
        "declared-alpha-control",
        parent,
        ((daughter, 1.0), (helium, 1.0)),
        0.01,
        np.zeros((1,), dtype=np.float64),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        source,
    )
    inventory = NuclideInventory(
        jnp.asarray([2.0], dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
        (parent.nuclide_id,),
        "synthetic-alpha-network",
    )
    plan = HeliumProductionPlan(
        {nuclide: 0.001 for nuclide in FISSIONING_NUCLIDES},
        inventory,
        (transition,),
        source,
        helium=helium,
    )
    return plan, inventory


def test_helium_sources_count_fission_and_native_decay_events_in_reference_volume() -> (
    None
):
    plan, inventory = _plan()
    fractions = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    rate = jax.jit(
        lambda fission_density: plan.evaluate(fission_density, fractions, inventory, 0.5)
    )(jnp.asarray(1.0e20, dtype=jnp.float64))
    np.testing.assert_allclose(rate.ternary, 1.0e17, rtol=1.0e-15)
    np.testing.assert_allclose(rate.decay, 0.04 * AVOGADRO_PER_MOL, rtol=1.0e-15)
    np.testing.assert_allclose(rate.total, 1.0e17 + 0.04 * AVOGADRO_PER_MOL, rtol=1e-15)


def test_helium_source_rejects_equal_shape_wrong_inventory_identity() -> None:
    plan, inventory = _plan()
    wrong = NuclideInventory(
        inventory.amounts_mol,
        inventory.time_s,
        (NuclideKey(94, 239).nuclide_id,),
        inventory.network_id,
    )
    with pytest.raises(ValueError, match="inventory identity"):
        plan.evaluate(1e20, jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), wrong, 0.5)
