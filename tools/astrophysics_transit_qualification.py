"""Qualification evidence for occultation and photon-counting observations."""

import json

import jax
import jax.numpy as jnp

import phydrax as phx


def main():
    physics = phx.applications.astrophysics
    disk = physics.PolynomialLimbDarkenedDisk(jnp.asarray([0.3, 0.2]))
    reference = physics.CircularOccultationPlan(disk, quadrature_order=512)
    candidate = physics.CircularOccultationPlan(disk, quadrature_order=192)
    separation = jnp.linspace(0.0, 1.2, 257)
    reference_flux = reference.evaluate(separation, 0.1).relative_flux
    candidate_flux = candidate.evaluate(separation, 0.1).relative_flux
    derivative = jax.jvp(
        lambda value: candidate.evaluate(value, 0.1).relative_flux,
        (jnp.asarray(0.5),),
        (jnp.asarray(1.0),),
    )[1]
    wavelength = jnp.asarray([4.0e-7, 5.0e-7, 6.0e-7])
    band = physics.PhotonCountingBandpass(
        wavelength,
        jnp.ones(3),
        physics.ObservationDataProvenance.native("qualification-band"),
        band_id="qualification",
    )
    flux = jnp.full((3,), 1.0e-9)
    report = {
        "kind": "astrophysics-transit-qualification",
        "dtype": str(reference_flux.dtype),
        "maximum_quadrature_difference": float(
            jnp.max(jnp.abs(candidate_flux - reference_flux))
        ),
        "disjoint_flux_error": float(
            jnp.abs(candidate.evaluate(2.0, 0.1).relative_flux - 1.0)
        ),
        "full_cover_flux_error": float(
            jnp.abs(candidate.evaluate(0.0, 2.0).relative_flux)
        ),
        "interior_derivative_finite": bool(jnp.isfinite(derivative)),
        "uniform_band_photon_rate": float(band.photon_rate(flux)),
        "passed": bool(
            (jnp.max(jnp.abs(candidate_flux - reference_flux)) < 2.0e-5)
            & jnp.isfinite(derivative)
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
