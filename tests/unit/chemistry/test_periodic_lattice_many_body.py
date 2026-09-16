import numpy as np

import phydrax as phx


periodic = phx.chemistry.periodic


def test_supplied_diagonal_gw_and_bse_retain_provider_provenance():
    gw_manifest = periodic.PeriodicProvenanceManifest.for_bytes(
        b"analytic-diagonal-self-energy",
        "analytic-gw-provider",
        "test-redistributable",
    )
    gw = periodic.DiagonalGWPlan(
        [-0.5, 0.3],
        [0.0, 0.0],
        [[-0.6, -0.2], [0.2, 0.6]],
        lambda index, energy: 0.1 + 0.0 * index + 0.0 * energy,
        "constant-self-energy",
        "analytic-gw-provider",
        gw_manifest,
        phx.units.HARTREE,
    ).evaluate()
    dipole_unit = phx.units.derived_unit(
        "e*bohr-periodic-bse",
        ((phx.units.ELEMENTARY_CHARGE, 1), (phx.units.BOHR, 1)),
    )
    bse_manifest = periodic.PeriodicProvenanceManifest.for_bytes(
        b"analytic-transition-kernel",
        "analytic-bse-provider",
        "test-redistributable",
    )
    bse = periodic.BetheSalpeterPlan(
        ("v0-c0",),
        [0.5],
        [[0.1]],
        [[0.0]],
        [[1.0, 0.0, 0.0]],
        -1.0,
        phx.units.HARTREE,
        dipole_unit,
        "analytic-bse-provider",
        bse_manifest,
    )
    tda = bse.tda(1)
    full = bse.full(1)

    assert bool(gw.successful) and bool(tda.successful) and bool(full.successful)
    np.testing.assert_allclose(gw.quasiparticle_energies, [-0.4, 0.4], atol=1.0e-9)
    np.testing.assert_allclose(gw.renormalization_factors, 1.0, atol=1.0e-13)
    np.testing.assert_allclose(tda.manifold.excitation_energies, [0.4], atol=1.0e-12)
    np.testing.assert_allclose(full.manifold.excitation_energies, [0.4], atol=1.0e-12)
    assert gw.source_manifest_id == gw_manifest.manifest_id
    assert tda.evidence.source_manifest_id == bse_manifest.manifest_id
    assert tda.evidence.transition_order_id == bse.transition_order_id
