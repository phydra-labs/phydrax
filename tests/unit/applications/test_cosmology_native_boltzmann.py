import jax.numpy as jnp
import numpy as np

import phydrax as phx


cosmology = phx.applications.cosmology


def _artifact(kind):
    return cosmology.ScientificArtifactEnvelope(
        artifact_kind=kind,
        content_digest=f"{kind}-fixture",
        producer="test",
        producer_version="current",
        build_id="fixture",
        license_id="internal",
        resource_id="static",
        status="complete",
    )


def _profile():
    return cosmology.ParityProfile(
        name="restricted-scalar-flat",
        equations=("Einstein-Boltzmann-linear",),
        species=("photon", "baryon", "cdm", "massless-relic"),
        geometry="flat-FLRW",
        approximations=("scalar-adiabatic", "fixed-layout"),
        outputs=("transfer", "unlensed-TT-TE-EE"),
        references=("independent-fixture",),
        metrics=("state-residual", "spectrum-residual"),
        negative_boundaries=("massive-relic", "curvature", "lensing"),
    )


def test_native_thermodynamics_and_scalar_transfer_are_finite():
    scale = jnp.linspace(0.1, 1.0, 16)
    rates = cosmology.ThermodynamicsRateTable(
        scale,
        1.0e-3 * jnp.ones_like(scale),
        1.0e-4 * jnp.ones_like(scale),
        1.0e-2 * jnp.ones_like(scale),
        2.7255 / scale,
        _artifact("thermodynamics-rates"),
    )
    thermo = cosmology.NativeThermodynamicsPlan(
        rates,
        hydrogen_number_density_today=1.0,
        thomson_cross_section=1.0e-3,
        speed_of_light=1.0,
    ).solve(jnp.ones_like(scale))
    assert bool(thermo.successful)
    assert jnp.all(
        (thermo.ionization_fraction >= 0.0) & (thermo.ionization_fraction <= 1.0)
    )

    layout = cosmology.ScalarHierarchyLayout(
        photon_order=3, polarization_order=3, relic_order=3
    )
    k = jnp.asarray([0.1, 0.2])
    matrices = -0.1 * jnp.broadcast_to(
        jnp.eye(layout.state_size),
        (scale.size, k.size, layout.state_size, layout.state_size),
    )
    sources = jnp.zeros((scale.size, k.size, layout.state_size))
    operators = cosmology.ScalarEvolutionOperatorTable(
        scale, k, matrices, sources, layout, _artifact("scalar-operators")
    )
    transitions = cosmology.ApproximationTransitionPolicy(
        tight_coupling_exit=0.2,
        radiation_streaming_entry=0.8,
        overlap_tolerance=1.0e-5,
    )
    result = cosmology.RestrictedScalarTransferPlan(
        operators, transitions, _profile()
    ).solve(jnp.ones((k.size, layout.state_size)))
    assert bool(result.successful)
    assert result.states.shape == (scale.size, k.size, layout.state_size)
    assert set(np.asarray(result.transition_phases).tolist()) == {0, 1, 2}


def test_flat_radial_and_line_of_sight_projection():
    radial = cosmology.FlatRadialKernelPlan(6)
    values = radial.evaluate(jnp.asarray([0.0, 1.0]))
    np.testing.assert_allclose(values[0, 0], 1.0)
    np.testing.assert_allclose(values[1, 0], 0.0)
    time = jnp.linspace(0.0, 1.0, 32)
    k = jnp.geomspace(0.1, 2.0, 24)
    source = jnp.exp(-(((time[None, :] - 0.5) / 0.1) ** 2)) * jnp.ones((k.size, 1))
    result = cosmology.LineOfSightSpectraPlan(radial, [2, 3, 4]).project(
        time, k, source, 2.1e-9 * jnp.ones_like(k)
    )
    assert bool(result.successful)
    assert jnp.all(result.spectra >= 0.0)


def test_parity_evidence_enforces_metric_limits():
    profile = _profile()
    evidence = cosmology.ParityEvidence(
        profile,
        [1.0e-5, 2.0e-5],
        [2.0e-5, 2.0e-5],
        _artifact("parity-corpus"),
    )
    assert bool(evidence.successful)
