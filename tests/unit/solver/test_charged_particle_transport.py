#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx


def _manifest():
    return phx.qualification.ReferenceArtifactManifest(
        "charged-material",
        checksum_algorithm="sha256",
        checksum="f" * 64,
        size_bytes=1,
        license_id="synthetic-permissive",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=False,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"energy_eV": 1.0},
        uncertainty={"relative": 0.0},
        lineage_ids=("synthetic:charged",),
    )


def _plan(*, stopping=1000.0, box_length=2.0):
    materials = phx.equations.ChargedRadiationMaterialLibrary(
        jnp.asarray((10.0, 2000.0)),
        jnp.full((1, 2), stopping),
        jnp.zeros((1, 2)),
        jnp.zeros((1, 2)),
        ("material",),
        _manifest(),
    )
    geometry = phx.discretization.VoxelRadiationGeometryPlan(
        jnp.asarray((0.0, 0.0, 0.0)),
        jnp.asarray((1.0, 1.0, box_length)),
        jnp.zeros((1, 1, 2), dtype=jnp.int32),
        material_count=1,
    )
    return phx.solver.ChargedParticleTransportPlan(
        geometry,
        materials,
        maximum_steps=256,
        maximum_step_length=0.05,
        maximum_fractional_energy_loss=0.05,
        cutoff_energy_ev=10.0,
    )


def test_condensed_history_csda_range_and_kinetic_energy_ledger():
    plan = _plan()
    result = plan.simulate(
        jnp.asarray(((0.5, 0.5, 0.1),)),
        jnp.asarray(((0.0, 0.0, 1.0),)),
        jnp.asarray((1000.0,)),
        jnp.asarray((int(phx.equations.ChargedRadiationParticleKind.ELECTRON),)),
        jr.key(3),
    )

    assert bool(result.all_successful)
    np.testing.assert_allclose(result.deposited_energy, 1000.0, atol=1e-8)
    np.testing.assert_allclose(result.path_length, 0.99, atol=2e-3)
    np.testing.assert_allclose(result.maximum_kinetic_ledger_residual, 0.0, atol=1e-9)


def test_boundary_escape_and_positron_annihilation_are_distinct_ledgers():
    escaping = _plan(stopping=1.0, box_length=0.5).simulate(
        jnp.asarray(((0.5, 0.5, 0.1),)),
        jnp.asarray(((0.0, 0.0, 1.0),)),
        jnp.asarray((1000.0,)),
        jnp.asarray((int(phx.equations.ChargedRadiationParticleKind.ELECTRON),)),
        jr.key(4),
    )
    annihilating = _plan(stopping=1000.0).simulate(
        jnp.asarray(((0.5, 0.5, 0.1),)),
        jnp.asarray(((0.0, 0.0, 1.0),)),
        jnp.asarray((20.0,)),
        jnp.asarray((int(phx.equations.ChargedRadiationParticleKind.POSITRON),)),
        jr.key(5),
    )

    assert bool(escaping.all_successful)
    assert escaping.escaped_energy[0] > 999.0
    assert bool(annihilating.all_successful)
    np.testing.assert_allclose(annihilating.annihilation_photon_energy, 2.0 * 510998.95)
    np.testing.assert_allclose(
        annihilating.deposited_energy + annihilating.escaped_energy,
        20.0,
        atol=1e-8,
    )
