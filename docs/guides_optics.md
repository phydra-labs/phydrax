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

`NonSequentialOpticsPlan` is a separate bounded branch-tree model over conservative
oriented triangle queries. It accounts for detected, absorbed, escaped, deliberately
discarded, ambiguous, truncated, and still-live power. It never hides dropped
branches or traversal exhaustion.

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
evidence. Geometry lowering rejects a nonzero imaginary index. Maxwell lowering is
restricted to the explicit isotropic nonmagnetic relation `epsilon_r = n**2`,
`mu_r = 1`, and zero magnetoelectric coupling.

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

## Nonlinear propagation

`AnalyticPulseField` is a carrier-resolved positive-frequency analytic field over one
plane and temporal tensor grid. Nonpositive and Nyquist-inactive bins are never sent
to a material law. Instantaneous chi2/chi3 response reconstructs the real field,
forms nonlinear polarization in time, transforms back, and retains the positive
spectrum.

`UnidirectionalPropagationPlan` uses fixed-step interaction-picture RK4 with explicit
dispersion, de-aliasing, edge-energy, refinement, and backward-wave applicability
evidence. It is not a reflected-wave solver and does not duplicate compatible
Maxwell.

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
