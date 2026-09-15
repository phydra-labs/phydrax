# General-relativistic rays, transfer, and imaging

The black-hole observation path lives in `phydrax.applications.astrophysics`. It
composes `phydrax.metrix` metrics/connections with existing differential, units,
interpolation, imaging, observation, likelihood, and inverse owners. It does not add a
second ray solver, image type hierarchy, or inference runtime.

## Observer screens and ray states

`GRObserverScreenPlan` binds a four-dimensional Lorentzian metric, observer event,
observer velocity, line-of-sight and up seeds, fixed pixel coordinates/mask, ray energy,
and future- or past-directed convention. `initialize_gr_observer_screen` constructs a
metric-orthonormal oriented tetrad and one null initial state and two-vector screen
basis per pixel. Tetrad and null residuals remain in `GRObserverScreenResult`; an
invalid pixel stays visible in its mask.

`GRRayState` stores a fixed batch of four-coordinates and tangents, optional screen
basis, and active lanes. `GRRayPlan` supports null and timelike geodesics, a fixed
reported affine grid, bounded adaptive solver work, optional parallel transport of the
screen basis, optional Jacobi fields, and caller-selected constants of motion. It uses
the native differential backend and `LeviCivitaConnection`.

## Chart identity is not metric identity

`gr_chart_identity(metric)` hashes the four-dimensional chart name and ordered
coordinate names. It answers whether products use the same coordinate schema.
`gr_metric_identity(metric)` separately hashes that chart ID, Lorentzian convention,
and canonical content of the metric matrix callable. Two Kerr metrics with identical
$(t,r,\theta,\phi)$ coordinates but different mass or spin therefore share a chart ID
and retain different metric IDs. Same-chart parameter substitution is not admitted.

Inspectable metric callables use the canonical callable payload, including bound
content. Opaque callables must be assigned nonempty `semantic_id` and `numeric_id`
together when identity is formed; either alone is rejected. Observer screens,
`GRRayPlan`/`GRRayResult`, and `PolarizedRayPath` retain and compare metric and chart
identities in addition to convention, scale and unit identities. A user-facing label,
equal coordinate names, or equal output shape cannot replace that content binding.

## Ordered terminal events and statuses

`GRRayEventSurfaces` accepts optional signed capture, escape, and metric-domain margins.
The admissible side has positive margin and a down-crossing triggers. Simultaneous
terminal events use the fixed priority

```text
capture > escape > domain exit > work exhaustion
```

`GRRayEventLedger` retains event code, affine location, history index, whether a slot
was recorded, and simultaneity. No event is inferred from a coordinate named `r`.

`GRRayStatus` distinguishes endpoint success, capture, escape, chart exit, exhausted
work, nonfinite state, invalid initial state, mass-shell violation, numerical failure,
and inactive lanes. `GRRayStatusEvidence` keeps `finite`, `converged`,
`physically_valid`, `qualified`, and `derivative_valid` separate. Event identity,
solver termination, and chart exit are not disguised as ordinary endpoint success.

`GRRayResult` retains the coordinate/tangent history, transported basis, event ledger,
mass-shell residuals, constant-of-motion residuals, and optional bundle/Jacobi evidence.
`build_gr_ray_bundle_evidence` evaluates expansion, shear, rotation, area/distance
behavior, and caustic evidence without turning a finite ray bundle into a global
lensing theorem.

## Fast-light and slow-light plasma fields

`GRMediumFieldUnits` binds density, electron number density and temperature, magnetic
field, coordinate, and time units to one relativity scale. `FastLightSnapshot` is an
immutable rectilinear spatial snapshot with an explicit `chart_id`.
`prepare_path_sampling(path)` requires that exact ID to match the
`PolarizedRayPath.chart_id`, derives active/valid segment midpoints from the exact ray
result, and fingerprints both path and snapshot into the fixed eight-corner sampling
plan. Raw coordinate preparation remains a separate lower-level route.
`MonotoneSlowLightWorldtube` stores strictly ordered snapshots and
`FixedGRWorldtubeSamplingPlan` prepares sixteen-corner spacetime interpolation.

Samples retain support, finiteness, physical field values, chart, convention, units,
and source identity. Trilinear interpolation derivatives are admitted only away from
outer boundaries and interior cell-knot crossings. Four-vector components are declared
contravariant in the bound chart, but a snapshot does not claim metric normalization
unless separately evaluated.

## Thermal synchrotron microphysics

`ThermalSynchrotronModel` is capability-scoped. Its Stokes-$I$ shape is
Mahadevan--Narayan--Yi (1996), equation 31, only for
$T_e\ge3.2\times10^{10}\,\mathrm K$, with the published maximum relative shape error
$0.027$ at normalized frequency $160$. The default numeric intersection additionally
requires $10^{-3}\le\Theta_e\le10^3$ and
$10^{-6}\le\nu/\nu_c\le10^6$; bounds are evidence, not clipping.

`ThermalSynchrotronModel.reference` retains authors, title, DOI
`10.1086/177422`, arXiv `astro-ph/9601073`, equation and support/error metadata.
Its independent piecewise degree-12 Chebyshev approximation to
$\log K_2(z)$ covers $10^{-3}\le z\le10^3$ and records uniform log-error bounds below
$10^{-10}$ in float64 and $2\times10^{-4}$ after float32 rounding against 600,006
validation nodes. Panel boundaries remain derivative boundaries.
`ThermalSynchrotronUnitContract` requires the exact SI relativity scale and reports
emissivity, absorption, intensity, frequency, temperature, magnetic-field, and number-
density units.

The local Stokes convention is $(I,Q,U,V)$ with the first screen axis parallel to the
projected magnetic field. Kirchhoff absorption derives from the supported Stokes-$I$
emission and the full Planck function. Linear polarization/dichroism and Faraday
rotation/conversion are independent approximations marked
`unqualified-independent-approximation`. When either is active,
`polarization_reference_valid` or `faraday_reference_valid` is false, so the composite
polarized coefficients are not reference-qualified even if finite and in the Stokes-$I$
domain. `invariant_synchrotron_coefficients` preserves that disposition.

## Invariant transfer

`InvariantTransferUnitContract` fixes the path-parameter and $I_\nu/\nu^3$ units.
`InvariantScalarTransferPlan` applies the exact piecewise-constant slab update in
incident-to-observer order and reports history, optical depth, support, nonnegativity,
and derivative evidence.

`PolarizedRayPath` is the typed bridge from one null `GRRayResult` lane and the exact
four-dimensional metric to transfer. It inserts a recorded partial event segment,
recomputes basis Gram/tangent residuals, and compares them with the ray bundle evidence.
`PolarizedInvariantTransferPlan` then evolves invariant Stokes vectors by matrix
exponentials in that transported basis. `stokes_basis_rotation` and
`rotate_stokes_coefficients` rotate local coefficients without rotating the transported
state twice. Evidence checks finite propagation, support, ray-path/basis transport, the
propagation-matrix convention, and the physical Stokes cone. The transfer plans do not
infer a plasma model, solve scattering transport, or promote reference-unqualified
polarization/Faraday coefficients.

## Images and interferometry

`GRImageScreen` records physical angular coordinates and per-pixel solid angle.
`StokesImage` binds $(I,Q,U,V)$, disjoint lensing masks, positive observer/emitter
redshift ratio, monochromatic frequency, explicit intensity and flux-density units,
observation provenance, and parent product IDs. `phydrax.units.JANSKY` is the canonical
Jy unit for physical visibility products. Neutral FITS payload conversion preserves
arrays and canonical metadata but performs no file I/O in traced execution.

`VisibilitySampling` binds station IDs, ordered baseline pairs, $(u,v)$ coordinates in
wavelengths, and positive frequencies. `direct_stokes_visibilities` computes the signed
kernel $\exp[-2\pi i(ul+vm)]$ with solid-angle quadrature. Station gains are applied as
$g_i\overline{g_j}$. Polarization products and `ClosureTopology` retain fixed baseline,
triangle, and quadrangle routes; `closure_products` returns bispectra, closure phases,
and safe closure amplitudes/log-amplitudes with zero/nonfinite statuses. These gain-
invariant products are not calibration, atmospheric correction, array simulation, or
an observational validation claim.

FITS- and UVFITS-neutral payload helpers preserve declared array/metadata contracts.
They do not bundle Astropy, open files, or claim lossless support for every external
header convention.

## Fixed-branch inference

`FixedBranchRayInferencePlan` evaluates a Gaussian likelihood only for a frozen ray
event/topology program. `gr_ray_model_evaluation` refuses derivatives through ray
events, and `fixed_branch_ray_inverse_adapter` binds an event-free, fixed-topology
realization to the shared inverse contract. `GRPosteriorRealizationBinding` and
`GRPosteriorPrediction` preserve chain/draw identity and exact forward realization IDs.
They do not create a posterior, diagnose sampler convergence, or validate an
astrophysical model.

## Differentiation boundary

Metric, geodesic, basis-transport, smooth interpolation, fixed-segment transfer,
direct Fourier, and fixed-branch likelihood kernels retain JAX derivatives where their
own evidence admits them. Pixel masks, ray activation, event ordering, capture/escape,
chart exit, work exhaustion, caustics, adaptive step history, interpolation-cell
selection, coefficient-domain boundaries, closure topology, and posterior/forward
binding are discrete boundaries. A visually plausible image is not derivative,
convergence, or physical qualification evidence.