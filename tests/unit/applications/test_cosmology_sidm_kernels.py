import hashlib

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.applications.cosmology._dark_sector_species import DarkSectorSpeciesPlan
from phydrax.applications.cosmology._sidm_kernels import (
    angles_from_direction,
    directions_from_angles,
    SmallAngleSplitPlan,
    TwoBodyDifferentialKernelPlan,
)
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.qualification import ReferenceArtifactManifest


def _species():
    return DarkSectorSpeciesPlan(
        "chi",
        2.0,
        internal_energy=0.25,
        degeneracy=3,
        charge_names=("dark-number", "parity"),
        charges=(1.0, -1.0),
    )


def _source_kwargs(
    speeds,
    cosines,
    differential,
    *,
    azimuths=None,
    permit_commercial=True,
    request_commercial=True,
):
    payload = TwoBodyDifferentialKernelPlan.canonical_table_bytes(
        speeds, cosines, differential, azimuths=azimuths
    )
    digest = hashlib.sha256(payload).hexdigest()
    lineage = ("synthetic-sidm-kernel-fixture",)
    manifest = ReferenceArtifactManifest(
        "synthetic-sidm-kernel-table",
        checksum_algorithm="sha256",
        checksum=digest,
        size_bytes=len(payload),
        license_id="CC0-1.0",
        commercial_use_permitted=permit_commercial,
        redistribution_permitted=True,
        training_use_permitted=False,
        export_permitted=False,
        export_classification="unrestricted",
        nondimensionalization={"speed": 1.0, "cross_section": 1.0},
        uncertainty={"tabulation": 0.0},
        lineage_ids=lineage,
    )
    artifact = ScientificArtifactEnvelope(
        artifact_kind="synthetic-differential-kernel",
        content_digest=digest,
        producer="unit-fixture",
        producer_version="native",
        build_id="hand-authored",
        license_id=manifest.license_id,
        parent_artifact_ids=lineage,
        resource_id="synthetic-sidm-kernel-table",
        status="complete",
    )
    return {
        "source_artifact": artifact,
        "reference_manifest": manifest,
        "commercial_use": request_commercial,
        "redistribution": False,
        "training_use": False,
        "export": False,
    }


def _kernel(species, speeds, cosines, differential, *, azimuths=None, **kwargs):
    return TwoBodyDifferentialKernelPlan(
        species,
        species,
        speeds,
        cosines,
        differential,
        azimuths=azimuths,
        **_source_kwargs(speeds, cosines, differential, azimuths=azimuths),
        **kwargs,
    )


def test_species_and_isotropic_analytic_moments_are_explicit():
    species = _species()
    kernel = TwoBodyDifferentialKernelPlan.constant_isotropic(species, 6.0)
    moments = kernel.moments(jnp.asarray((0.0, 2.0, 100.0)))

    assert species.species_id == "chi"
    assert species.mass == 2.0
    assert species.internal_energy == 0.25
    assert species.degeneracy == 3
    np.testing.assert_array_equal(species.charges, (1.0, -1.0))
    np.testing.assert_allclose(moments.total, 6.0, rtol=2e-15)
    np.testing.assert_allclose(moments.transfer, 6.0, rtol=2e-15)
    np.testing.assert_allclose(moments.viscosity, 4.0, rtol=2e-15)
    np.testing.assert_allclose(moments.modified_transfer, 3.0, rtol=2e-15)
    assert np.all(moments.supported)

    with pytest.raises(ValueError, match="one value per charge"):
        DarkSectorSpeciesPlan("bad", 1.0, charge_names=("q",), charges=())


def test_full_sphere_and_exchange_quotient_have_one_physical_normalization():
    species = _species()
    speeds = jnp.asarray((0.0, 2.0))
    full = _kernel(
        species,
        speeds,
        jnp.asarray((-1.0, -0.25, 0.0, 0.5, 1.0)),
        jnp.full((2, 5), 8.0 / (4.0 * jnp.pi)),
        identical_particle_convention="labelled-full-sphere",
    )
    quotient = _kernel(
        species,
        speeds,
        jnp.asarray((0.0, 0.25, 0.5, 0.75, 1.0)),
        jnp.full((2, 5), 8.0 / (2.0 * jnp.pi)),
        identical_particle_convention="exchange-quotient",
    )

    full_moments = full.moments(1.0)
    quotient_moments = quotient.moments(1.0)
    np.testing.assert_allclose(quotient_moments.total, full_moments.total, rtol=2e-15)
    np.testing.assert_allclose(
        quotient_moments.modified_transfer,
        full_moments.modified_transfer,
        rtol=2e-15,
    )
    np.testing.assert_allclose(
        quotient_moments.viscosity, full_moments.viscosity, rtol=2e-15
    )


def test_screening_azimuth_and_speed_table_support_fail_closed():
    species = _species()
    cutoff = 0.2
    cosines = jnp.asarray((-1.0, 0.0, np.cos(cutoff)))
    azimuths = jnp.asarray((0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi))
    base = jnp.asarray((1.0, 2.0, 8.0))[None, :, None]
    azimuthal = 1.0 + 0.25 * jnp.cos(azimuths)[None, None, :]
    values = jnp.concatenate((base * azimuthal, 3.0 * base * azimuthal), axis=0)
    kernel = _kernel(
        species,
        jnp.asarray((1.0, 3.0)),
        cosines,
        values,
        azimuths=azimuths,
        identical_particle_convention="labelled-full-sphere",
        screening_convention="hard-angular-cutoff",
        minimum_scattering_angle=cutoff,
    )

    at_middle = kernel.evaluate(2.0, 0.0, 0.0)
    at_left = kernel.evaluate(1.0, 0.0, 0.0)
    np.testing.assert_allclose(
        at_middle.differential_cross_section, 2.0 * at_left.differential_cross_section
    )
    assert bool(at_middle.supported)
    assert not bool(kernel.evaluate(0.5, 0.0, 0.0).supported)
    assert not bool(kernel.evaluate(2.0, 1.0, 0.0).supported)
    assert not bool(kernel.evaluate(2.0, 0.0, -0.1).supported)


def test_inverse_cdf_sampling_reproduces_angular_moments():
    species = _species()
    cosines = jnp.linspace(-1.0, 1.0, 65)
    # p(mu) is proportional to 1 + mu on the full sphere.
    values = jnp.broadcast_to((1.0 + cosines)[None, :], (2, cosines.size))
    kernel = _kernel(
        species,
        jnp.asarray((0.0, 2.0)),
        cosines,
        values,
        identical_particle_convention="labelled-full-sphere",
    )
    keys = jr.split(jr.key(11), 20_000)
    samples = jax.jit(jax.vmap(lambda key: kernel.sample_angles(key, 1.0)))(keys)

    assert np.all(samples.supported)
    np.testing.assert_allclose(jnp.mean(samples.cosine), 1.0 / 3.0, atol=0.015)
    np.testing.assert_allclose(jnp.mean(samples.cosine**2), 1.0 / 3.0, atol=0.015)
    assert float(jnp.max(samples.normalization_residual)) < 1.0e-14


def test_small_angle_split_has_no_gap_or_overlap_and_reconstructs_every_moment():
    species = _species()
    cosines = jnp.linspace(-1.0, 1.0, 33)
    differential = jnp.stack(
        (
            1.0 / (1.1 - cosines) ** 2,
            2.0 / (1.1 - cosines) ** 2,
        )
    )
    kernel = _kernel(
        species,
        jnp.asarray((1.0, 2.0)),
        cosines,
        differential,
        identical_particle_convention="labelled-full-sphere",
    )
    split = SmallAngleSplitPlan(kernel, 0.75)
    evidence = split.moments(jnp.asarray((1.0, 1.5, 2.0)))

    assert bool(split.no_gap)
    assert bool(split.no_overlap)
    assert np.all(evidence.supported)
    assert np.all(evidence.successful)
    np.testing.assert_allclose(
        evidence.small.total + evidence.rare.total, evidence.total.total, rtol=2e-14
    )
    np.testing.assert_allclose(
        evidence.small.transfer + evidence.rare.transfer,
        evidence.total.transfer,
        rtol=2e-14,
    )
    np.testing.assert_allclose(
        evidence.small.viscosity + evidence.rare.viscosity,
        evidence.total.viscosity,
        rtol=2e-14,
    )
    np.testing.assert_allclose(
        evidence.small.modified_transfer + evidence.rare.modified_transfer,
        evidence.total.modified_transfer,
        rtol=2e-14,
    )
    rare_sample = split.sample_rare_angles(jr.key(8), 1.5)
    interpolated_split = jnp.interp(1.5, kernel.relative_speeds, split.split_cosines)
    assert bool(rare_sample.supported)
    assert float(rare_sample.cosine) <= float(interpolated_split)


def test_reference_rights_denial_and_table_substitution_fail_closed():
    species = _species()
    speeds = jnp.asarray((0.0, 2.0))
    cosines = jnp.asarray((-1.0, 0.0, 1.0))
    differential = jnp.ones((2, 3))
    with pytest.raises(TypeError, match="source_artifact"):
        TwoBodyDifferentialKernelPlan(
            species,
            species,
            speeds,
            cosines,
            differential,
            identical_particle_convention="labelled-full-sphere",
        )
    denied = _source_kwargs(
        speeds,
        cosines,
        differential,
        permit_commercial=False,
        request_commercial=True,
    )
    with pytest.raises(PermissionError, match="commercial-use-not-permitted"):
        TwoBodyDifferentialKernelPlan(
            species,
            species,
            speeds,
            cosines,
            differential,
            **denied,
            identical_particle_convention="labelled-full-sphere",
        )

    admitted = _source_kwargs(speeds, cosines, differential)
    commercial = TwoBodyDifferentialKernelPlan(
        species,
        species,
        speeds,
        cosines,
        differential,
        **admitted,
        identical_particle_convention="labelled-full-sphere",
    )
    noncommercial = TwoBodyDifferentialKernelPlan(
        species,
        species,
        speeds,
        cosines,
        differential,
        **_source_kwargs(speeds, cosines, differential, request_commercial=False),
        identical_particle_convention="labelled-full-sphere",
    )
    assert commercial.reference_manifest.manifest_id == (
        noncommercial.reference_manifest.manifest_id
    )
    assert commercial.requested_use_id != noncommercial.requested_use_id
    assert commercial.kernel_id != noncommercial.kernel_id
    with pytest.raises(ValueError, match="checksum mismatch"):
        TwoBodyDifferentialKernelPlan(
            species,
            species,
            speeds,
            cosines,
            differential.at[1, 1].set(2.0),
            **admitted,
            identical_particle_convention="labelled-full-sphere",
        )


def test_inverse_cdf_is_scale_invariant_and_rejects_bounds_outside_support():
    species = _species()
    speeds = jnp.asarray((0.0, 2.0))
    cosines = jnp.linspace(-1.0, 1.0, 33)
    shape = 1.0 + cosines
    ordinary = _kernel(
        species,
        speeds,
        cosines,
        jnp.broadcast_to(shape[None, :], (2, cosines.size)),
        identical_particle_convention="labelled-full-sphere",
    )
    tiny = _kernel(
        species,
        speeds,
        cosines,
        jnp.broadcast_to((1.0e-280 * shape)[None, :], (2, cosines.size)),
        identical_particle_convention="labelled-full-sphere",
    )
    keys = jr.split(jr.key(23), 256)
    ordinary_samples = jax.vmap(lambda key: ordinary.sample_angles(key, 1.0))(keys)
    tiny_samples = jax.vmap(lambda key: tiny.sample_angles(key, 1.0))(keys)
    np.testing.assert_allclose(
        tiny_samples.cosine, ordinary_samples.cosine, rtol=0.0, atol=2e-15
    )
    assert np.all(tiny_samples.supported)
    assert not bool(
        ordinary.sample_angles(
            jr.key(0), 1.0, cosine_minimum=-1.1, cosine_maximum=0.5
        ).supported
    )

    varying = jnp.stack((1.0e-280 * shape, 2.0e-280 * shape))
    with pytest.raises(ValueError, match="speed independent"):
        _kernel(
            species,
            speeds,
            cosines,
            varying,
            identical_particle_convention="labelled-full-sphere",
            unbounded_speed=True,
        )


def test_canonical_recoil_frame_round_trips_and_split_is_speed_independent():
    relative = jnp.asarray(((1.0, 2.0, 3.0), (-2.0, 1.0, 0.5)))
    cosine = jnp.asarray((0.25, -0.6))
    azimuth = jnp.asarray((0.7, 5.1))
    direction = directions_from_angles(relative, cosine, azimuth)
    actual_cosine, actual_azimuth = angles_from_direction(relative, direction)
    np.testing.assert_allclose(actual_cosine, cosine, atol=2e-15)
    np.testing.assert_allclose(actual_azimuth, azimuth, atol=2e-15)

    species = _species()
    kernel = TwoBodyDifferentialKernelPlan.constant_isotropic(species, 1.0)
    with pytest.raises(ValueError, match="speed independent"):
        SmallAngleSplitPlan(kernel, jnp.asarray((0.5, 0.6)))
