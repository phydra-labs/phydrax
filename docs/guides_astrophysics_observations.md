# Astrophysical observations

The native observation path separates observer projection, source occultation,
instrument response, and normalized likelihoods. It reuses Phydrax observation,
posterior, state-space, and BlackJAX contracts rather than introducing an astronomy
inference runtime.

## Transit photon counts

```python
import jax.numpy as jnp
import phydrax as phx

physics = phx.applications.astrophysics
disk = physics.PolynomialLimbDarkenedDisk(jnp.asarray([0.3, 0.2]))
occultation = physics.CircularOccultationPlan(disk)
relative_flux = occultation.evaluate(jnp.asarray([2.0, 0.0]), 0.1).relative_flux

wavelength = jnp.asarray([4.0e-7, 5.0e-7, 6.0e-7])
provenance = physics.ObservationDataProvenance.native("synthetic")
band = physics.PhotonCountingBandpass(
    wavelength, jnp.ones(3), provenance, band_id="synthetic"
)
plan = physics.TransitPhotometryPlan(
    (band,),
    jnp.asarray([0, 0]),
    jnp.asarray([30.0, 30.0]),
    collecting_area=1.0,
    background_rate=0.1,
)
result = plan.evaluate(relative_flux, jnp.full((2, 3), 1.0e-9))
assert bool(jnp.all(result.valid))
```

Wavelength is measured in metres and source spectral energy flux density in
`W m^-2 m^-1`. Photon rates use the immutable bandpass nodes and trapezoidal weights;
there is no evaluation-time interpolation or extrapolation. Response curves require
source/version/checksum/license provenance.

The occultation law is `I(mu) = 1 - sum u_n (1 - mu)^(n+1)`. Construction rejects a
negative intensity law or non-positive total stellar flux. Circular contact is a real
piecewise-smooth boundary and is not replaced by a smoothing surrogate.

## Poisson observations

`transit_poisson_log_prob` delegates to the normalized scalar Poisson likelihood. A
zero physical photon rate remains zero. Because the existing Poisson natural coordinate
is a finite log rate, inference requires a strictly positive expected mean, normally
through an explicit physical background. No hidden floor is added. Invalid model rows
produce negative infinity; user-masked missing observations contribute zero through the
existing observation mask.

## Other concrete operators

The same package includes fixed binned responses, normalized PSF convolution,
frequency-domain detector response, fixed-ray emission/absorption transfer, and static
complex-field operator sequences. These are concrete array boundaries for external
X-ray, imaging, waveform, ray-transfer, and optics packages; they are not a provider
registry or a second operator framework.

`BinnedResponsePlan` now adapts the shared `LinearObservationPlan`; CMB bandpower and
survey windows use the same labelled response algebra. Core Cholesky/precision
covariance actions and Gaussian likelihoods are shared with orbit determination and
cosmology, while PSF, bandpass, ray, occultation, antenna, and measurement geometry
remain domain-specific. Observation provenance uses the shared dependency-aware
differentiation contract rather than a second three-valued vocabulary.
