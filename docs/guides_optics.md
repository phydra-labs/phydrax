# Optics

Phydrax separates optical models by physical state and approximation. It does not
automatically switch between rays, scalar fields, Maxwell fields, beamlets, or
radiative packets.

## Model selection

Use `phydrax.solver.maxwell.fourier_modal` for periodic vector electromagnetics,
evanescent diffraction orders, metasurfaces, and bianisotropic stacks. Use compatible
Maxwell for general full-wave time-domain problems. Use `phydrax.optics.geometric`
when a fixed optical path and surface intersections are the model. Use
`phydrax.optics.wave` for coherent sampled-plane propagation. Use beamlets only while
the chief path and its differential neighbourhood share one regular topology. Use
`phydrax.optics.transport` only after coherence and diffraction have intentionally
been discarded.

The canonical time-harmonic convention is `exp(-i omega t)`. Angular frequency is
stored explicitly. Any conversion from angular frequency to wavelength or medium
wavenumber names its reference wave speed; no global value of the speed of light is
implied.

## Geometric optics

`intersect_ray_plane` and the triangle query plans live in `phydrax.geometry`; they
return geometric hit facts only. `evaluate_refractive_interface` owns real-isotropic
Snell and Fresnel physics. Its fixed result contains reflected and transmitted
branches, complex s/p amplitudes, flux coefficients, margins, validity, and status.
Total internal reflection keeps a valid reflected branch and reports no real
transmitted ray.

`PlanarRefractiveStack` is the fixed-capacity transmitted-only path used by camera
models. `SequentialOpticsPlan` lowers an ordered fixed prescription of plane, sphere,
conic, and even-asphere surfaces with circular apertures and declared transmit or
reflect routes. `PreparedSequentialOptics.execute` performs bounded work and reports
misses, aperture clipping, tangency, root exhaustion, TIR, and numerical failure. It
does not discover objects or split a ray tree.

`linearize_sequential_optics` differentiates one regular fixed branch in canonical
`(u, v, n theta_u, n theta_v)` coordinates. `ParaxialOpticsPlan` caches that affine
map and refuses queries outside its declared transverse and angular trust envelope.
A topology margin is part of the result; aperture, root, TIR, and route changes are
not smooth gradients.

`ParaxialResonatorPlan` composes a closed ordered tuple of compatible
`DifferentialRayMap` legs. It solves the affine closed orbit, audits the full
canonical four-dimensional symplectic/Floquet map, and distinguishes certified
stable, cleanly unstable, marginal, singular, and invalid analyses. A stable
analysis returns a positive invariant complex Lagrangian plane rather than
assuming independent sagittal and tangential scalar q parameters. Mode
selection is host-prepared and fixed; crossings, marginal multipliers, route
changes, and topology changes are derivative boundaries.

`gaussian_beamlet_from_resonator_mode` explicitly lowers only a certified stable
mode to `GaussianBeamletState` while preserving frame, coordinate, prescription,
and resonator identities. An unstable analysis remains useful evidence but
cannot seed a physical Gaussian beamlet.

`NonSequentialOpticsPlan` is a separate bounded branch-tree model over conservative
oriented triangle queries. Optional per-medium passive power attenuation is applied
only along a valid finite hit segment, before its surface action. Results retain
volume, per-medium, and per-physical-surface absorption while keeping detector,
escape, discard, ambiguity, truncation, live-power, and complete-ledger channels
separate. Misses and unresolved paths are never assigned an invented travel length.

## Plane fields and propagation

`PlaneFieldSpace` composes a two-dimensional `PreparedTensorGrid` with a
three-dimensional `RigidFrame`. The tensor grid remains the sole owner of coordinates,
quadrature weights, measures, and topology. A space is explicitly either a
`finite-window` or `periodic-cell` support.

`ScalarPlaneField`, `TangentialPlaneField`, and `IntensityPlane` are distinct concrete
states. `ideal_square_law` returns intensity in the caller's amplitude convention;
it is not an implicit electromagnetic impedance conversion.

`AngularSpectrumPlan` prepares same-grid Fourier geometry. Periodic cells use an
un-padded periodic transform. Finite windows require explicit positive padding on
both sides of both axes and report boundary leakage and cropped energy. Execution
takes an explicit complex medium wavenumber and nonnegative propagation distance.
The outgoing branch has nonnegative real and imaginary longitudinal wavenumber.
Approximation failure returns the computed field with a non-success status.

`DirectFresnelPlan` is the explicit different-grid finite-plane alternative. It
preflights the two separable weighted kernels, admits only aligned uniform finite
planes, propagates scalar or tangential fields componentwise, and reports chirp
sampling, paraxial angle, finite-aperture power capture, and workspace. Zero
distance is an identity only for the identical space; no resampling or automatic
fallback is hidden in that branch.

`ScalarThinTransmission` and `JonesThinTransmission` are concrete multiplicative
operators. Sampled complex transmissions represent apertures, phase screens, OPD,
and amplitude masks without a type-erased stage graph. `thin_lens` creates the
quadratic phase for an explicit medium wavenumber and transverse optical-power
matrix.

`coherent_mode_intensity` applies declared nonnegative weights to squared coherent
modes. It does not normalize weights or introduce cross-mode coherence. Inactive
fixed-capacity lanes are masked before magnitude evaluation.

## Materials and regime lowering

`phydrax.optics.materials` contains scalar isotropic refractive-index laws, not a
universal material model. Constant, Cauchy, Sellmeier, Lorentz-Drude, and tabulated
complex laws expose validity, extrapolation, passivity-branch, and provenance
evidence. `lower_to_geometric_index` rejects a nonzero imaginary index rather than
discarding absorption. `lower_to_passive_ray_attenuation` is the separate explicit
lowering for real ray kinematics plus the passive power coefficient
`alpha = 2 omega Im(n) / reference_speed`; gain and unprovenanced loss are rejected.
Maxwell lowering is restricted to `epsilon_r = n**2`, `mu_r = 1`, and zero
magnetoelectric coupling.

No glass or crystal catalog is bundled. External records must carry an
`ArtifactManifest` and valid angular-frequency interval.

## Maxwell and pupil adapters

Fourier-modal field conversion produces a periodic-cell tangential electromagnetic
plane and preserves both E and H. It does not silently turn an infinite periodic cell
into a finite aperture. `tile_periodic_plane_to_finite_window` performs that change
explicitly and returns tiling/window evidence.

`sequential_pupil_to_scalar_field` requires one-to-one ordered ray samples, finite
optical paths, a regular area map, and a finite-window output. Folds and caustics are
typed failures rather than interpolation accidents.

## Beamlets and imaging

A `GaussianBeamletState` combines a chief ray with a complex coupled H/U Lagrangian
state and deterministic moving transverse frame. Production transport uses
`DifferentialRayMap`; the nine-ray construction is qualification evidence only.
Reconstruction is tiled and reports invariant, topology, conditioning, and caustic
evidence.

Noll Zernike coefficients are physical OPD lengths on a unit disk. Fraunhofer imaging
returns a focal intensity, sampling evidence, normalized OTF/MTF, and Strehl ratio.
Broadband modes are propagated separately and reduced after measurement unless the
caller explicitly declares coherent superposition.

## Atmosphere and statistical AO

Von Karman phase-screen plans use explicit JAX keys and Hermitian spectral sampling.
Frozen flow is an exact spectral translation. Layered atmosphere records retain
individual altitudes, velocities, strengths, and provenance.

The residual-AO implementation is explicitly a statistical frozen-flow PSD model.
Fitting, anisoplanatic, temporal, alias, and noise terms remain separate in
`ResidualAOErrorBudget`. It is not a physical wavefront-sensor/deformable-mirror
control-loop simulator.

## Pulse time and envelopes

`PulseTimeSpace` is the sole owner of a one-dimensional pulse-time grid and its
`finite-window` or `periodic-cell` topology. Carrier-resolved analytic fields and
unidirectional spectral propagation require periodic time. `PulseEnvelopeField`
stores a slowly varying scalar or local tangential electric-field envelope and may
also use a finite window.

`PulseEnvelopeBridgePlan` performs only an exact grid-aligned carrier-bin shift.
It rejects finite-window time, off-grid carriers, nonpositive absolute frequencies,
Nyquist collisions, excessive unsupported energy, and nonfinite payloads.
`GaussianPulseEnvelopePlan` samples an explicitly parameterized intensity-RMS
Gaussian and reports spatial/temporal boundary tails and spectral-edge support.
Neither API infers pulse energy, waist convention, polarization, or carrier.


## Nonlinear propagation

`AnalyticPulseField` is a carrier-resolved positive-frequency analytic field over
one plane and periodic `PulseTimeSpace`. Nonpositive and Nyquist-inactive bins are
never sent to a material law. `AbstractCarrierResolvedResponse` prepares one
fixed-shape finite-pulse material evaluation and returns analytic nonlinear
polarization, analytic free current, physical material history, a work ledger,
provenance, and fail-closed response evidence. Existing instantaneous chi2/chi3
responses retain their represented-band projection through that contract.

`DelayedRamanResponsePlan` is a causal held-drive damped-oscillator response.
`MultiphotonIonizationRatePlan` declares an `E**(2*K)` rate rather than claiming an
ADK/PPT/Keldysh model. `IonizingDrudeResponsePlan` combines exact bounded neutral
depletion with a causal Drude ADE and reports electron inventory, ionization
potential, current work, collision dissipation, terminal kinetic storage, and
closure. These initial material responses are scalar.

`UnidirectionalPropagationPlan` uses fixed-step interaction-picture RK4 with
explicit dispersion, de-aliasing, edge-energy, refinement, response, current-source,
and backward-wave evidence. `CylindricalUnidirectionalPropagationPlan` is the
separate scalar azimuthal-order-zero route over a certified finite-radius
Bessel-zero Hankel pair. It reports radial boundary/high-mode fractions, transform
defects, cutoff, temporal, refinement, response, and resource evidence. Nonzero
azimuthal order, vector cylindrical fields, reflection, and full Maxwell behavior
are not claimed.

The passive `BidirectionalCoupledModePlan` is the distinct two-direction model for
piecewise-constant Bragg, DBR, and DFB sections. Complex amplitudes are normalized
so squared magnitude is power. Exact local section maps are composed in a stable
scattering representation; negative attenuation is rejected as gain. The result
retains explicit left/right port ordering, interface amplitudes, distributed loss,
reciprocity, passivity, power balance, conditioning, and resource evidence.

## Radiative transport

`TissueTransportPlan` uses concrete absorption, scattering, Henyey-Greenstein, and
real-index records. Packets retain remaining optical depth across interfaces, use
implicit capture and unbiased roulette, and produce fixed tallies with standard
errors. Pathwise differentiability is explicitly not claimed.

## Guided modes and SBS

`FixedFrequencyGuidedModePlan` solves a propagation-constant polynomial at fixed
angular frequency and retains left/right traces, flux normalization, residuals,
classification, gap evidence, and mode identities. `GuidedElasticModePlan` solves the
fixed-longitudinal-wavenumber elastic eigenproblem.

`SBSOverlapPlan` maps native optical and acoustic representations to an explicit
shared quadrature. Photoelastic and moving-boundary overlaps remain complex and are
added before magnitude evaluation, preserving physical cancellation or reinforcement.

## Optional OpticStudio boundary

`phydrax.interchange.opticstudio` is lazy, host-only, and optional. It opens only an
owned standalone process, rejects unsupported native sequential features before
vendor mutation, returns detached immutable results, and is nondifferentiable. The
core package contains no vendor DLL, catalog, constant table, or session handle.

## Neutral laser-envelope HDF5 boundary

`phydrax.interchange` reads and writes one bounded Cartesian temporal electric-field
profile from the pinned upcoming openPMD LaserEnvelope HDF5 draft. The adapter uses
the public data schema through `h5py`; it does not import or depend on openPMD-api.
The exact draft commit is part of the profile identity. Vector potential, theta
modes, nonuniform axes, nonidentity export frames, nonzero export longitudinal
coordinates, and unpreserved required semantics are rejected with `AdapterReport`.

## Laser-domain composition and boundaries

Laser workflows compose explicit optical representations rather than selecting a
solver from one generic laser object. Classical paraxial resonators consume
fixed-branch differential ray maps. Passive Bragg and DFB/DBR sections use
power-normalized bidirectional coupled-mode amplitudes. Slowly varying pulse
envelopes remain distinct from carrier-resolved analytic fields, and both remain
distinct from full Maxwell fields. Semiconductor gain, carrier storage, plasma
state, and thermal deposition stay with their authoritative application owners.

Long-pulse collisional laser-plasma simulation is not currently a production
capability. It requires a separately qualified magnetized Braginskii transport
closure, a paraxial plasma-envelope model, conservative optical/plasma transfer,
and an ionized-continuum advance that actually applies transport and
electromagnetic source terms. A field-free collisional branch must be named
explicitly rather than treating an absent magnetic direction as arbitrary.

Full laser-processing CFD is also outside the native optics contract. The
current incompressible VOF state has no temperature, enthalpy, phase change,
evaporation, recoil, or shielding-gas thermodynamics. Optical heat must not be
injected into that state. External manufacturing workflows may exchange only
provider-neutral, producer-identified mesh/field artifacts managed outside Phydrax; no
copyleft solver runtime, case, mesh, or executable is bundled or launched.
