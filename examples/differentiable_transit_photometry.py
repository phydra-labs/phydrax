"""Differentiable circular occultation, photon counts, and Poisson likelihood."""

import jax
import jax.numpy as jnp

import phydrax as phx


def build_workflow():
    astrophysics = phx.applications.astrophysics
    disk = astrophysics.PolynomialLimbDarkenedDisk(jnp.asarray([0.3, 0.2]))
    occultation = astrophysics.CircularOccultationPlan(disk)
    wavelength = jnp.asarray([4.0e-7, 5.0e-7, 6.0e-7])
    provenance = astrophysics.ObservationDataProvenance.native("synthetic-band")
    bandpass = astrophysics.PhotonCountingBandpass(
        wavelength,
        jnp.asarray([0.2, 1.0, 0.2]),
        provenance,
        band_id="synthetic",
    )
    cadence = jnp.linspace(-1.2, 1.2, 64)
    plan = astrophysics.TransitPhotometryPlan(
        (bandpass,),
        jnp.zeros(cadence.shape, dtype=jnp.int32),
        jnp.full(cadence.shape, 60.0),
        collecting_area=1.0,
        background_rate=0.1,
    )
    spectrum = jnp.full((cadence.size, wavelength.size), 1.0e-9)
    return cadence, occultation, plan, spectrum


def main():
    cadence, occultation, plan, spectrum = build_workflow()

    def expected_counts(radius_ratio):
        relative = occultation.evaluate(jnp.abs(cadence), radius_ratio).relative_flux
        return plan.evaluate(relative, spectrum).expected_counts

    counts, tangent = jax.jvp(
        expected_counts,
        (jnp.asarray(0.1),),
        (jnp.asarray(1.0),),
    )
    result = plan.evaluate(
        occultation.evaluate(jnp.abs(cadence), 0.1).relative_flux, spectrum
    )
    log_likelihood = phx.applications.astrophysics.transit_poisson_log_prob(
        result, jnp.floor(counts)
    )
    print(counts)
    print(tangent)
    print(log_likelihood)


if __name__ == "__main__":
    main()
