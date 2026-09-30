# Maxwell solvers

Phydrax provides complementary Maxwell substrates.

- `phydrax.solver.CompatibleMaxwellPlan` advances compatible cochain D/B state and
  owns the general time-domain Maxwell lifecycle.
- `phydrax.solver.FrequencyMaxwellOperator` solves a prepared cochain curl-curl
  problem.
- `phydrax.solver.maxwell.fourier_modal` solves transversely periodic layered
  frequency-domain problems with boundary-field propagation.
- `phydrax.solver.maxwell.spectral` advances Cartesian PSATD/Galilean fields for
  explicit PIC ([Spectral Maxwell](spectral_maxwell.md)).

## Compatible time-domain lifecycle

::: phydrax.solver.maxwell.MaxwellCochainLayout

---

::: phydrax.solver.maxwell.MaxwellResourcePolicy

---

::: phydrax.solver.maxwell.MaxwellMagneticConstraintPolicy

---

::: phydrax.solver.maxwell.CompatibleMaxwellPlan

---

::: phydrax.solver.maxwell.PreparedCompatibleMaxwell

`MaxwellCochainLayout` accepts any `AbstractDeRhamComplex` and records scientific
role types: E/B are untwisted, H/D/J/charge are twisted dual roles even when
stored through their primal inverse-star layout. `CompatibleMaxwellPlan` accepts
cell realizations; CPML still explicitly requires a structured bridge.
`PreparedCompatibleMaxwell.magnetic_flux(state)` returns constrained raw B for
PIC's Lorentz force, not the constitutive H field. Metric and material operators
remain separate.

Linear Maxwell materials on real metric cochain spaces admit both real and
native-complex fields and charges. The prepared runtime extends real magnetic
incidence componentwise with `linalg.apply_real_map_componentwise`; it does not
change the native paired-space dtype or discard imaginary charge. Both components
participate in conservation constraints, and complex projection preserves the
status and work of the real and imaginary native solves. For compiled constitutive
frequency responses, pass the response as a dynamic PyTree to a plain JIT helper
(see [the compatible Maxwell guide](../../guides_compatible_maxwell.md#cochain-roles)).


---

::: phydrax.solver.maxwell.CompatibleMaxwellRefreshSpec

---

::: phydrax.solver.maxwell.refresh_compatible_maxwell

---

::: phydrax.solver.maxwell.solve_compatible_maxwell

## Sources, observers, and ports

::: phydrax.solver.maxwell.MaxwellElectricCurrentSourcePlan

---

::: phydrax.solver.maxwell.MaxwellPairedCurrentSourcePlan

---

::: phydrax.solver.maxwell.MaxwellHuygensSourcePlan

---


::: phydrax.solver.maxwell.MaxwellModePortPlan

---

::: phydrax.solver.maxwell.MaxwellSpectralAcquisition

---

::: phydrax.solver.maxwell.DFTObserverPlan

## One-way plane antennas

A sampled plane antenna is a solver-native equivalent source: an electric sheet
`K = s â × H'` on node planes and a magnetic sheet `K_m = −s â × E'` half a cell
upstream on interval centers (the total-field/scattered-field placement), driven by
sampled rest-frame envelopes `Re[A(τ) e^{−iω₀τ}]`. A moving antenna (velocity `β`
along its normal, vacuum of a declared `ElectromagneticScaleContract`) is a smoothed
moving total-field/scattered-field boundary: quadratic B-spline pairs driven by the
Lorentz-transformed incident fields at each event's rest-frame retarded time plus the
convective currents of the moving boundary, with second-order backward leakage. The
magnetic sheet's surface divergence is declared magnetic charge
(`MaxwellAuxiliaryState.magnetic_charge`), so antennas never trigger the global
magnetic projection. Runtime preparation refuses sheets outside the declared
homogeneous medium or inside CPML. `MaxwellAntennaWorkObserverPlan` streams the work
the sheets do on the field.
Optical envelopes reach antennas through `phydrax.optics.wave.pulse_envelope_antenna`
and `phydrax.optics.wave.openpmd_laser_envelope_antenna`.

::: phydrax.solver.maxwell.SampledPlaneCurrentAntennaPlan

---

::: phydrax.solver.maxwell.PreparedSampledPlaneCurrentAntenna

---

::: phydrax.solver.maxwell.SampledPlaneAntennaEvidence

---

::: phydrax.solver.maxwell.MaxwellAntennaWorkObserverPlan

---

::: phydrax.solver.maxwell.MaxwellAntennaWorkEvidence

## Prescribed moving charges

`PrescribedChargeMaxwellPlan(prepared_maxwell, current_plan, trajectory, charge)`
drives a prepared compatible runtime with point charges whose positions
`PrescribedChargeTrajectory` samples on the Maxwell step grid (uniform times, step
no larger than `stable_dt`). Each step deposits the charge-conserving
tail-to-head current of the straight path between samples, so the Maxwell
charge cochain follows the deposited charge exactly. Each charge starts
coincident with a static compensating charge (zero initial field, Gauss's law
without an electrostatic solve) and is frozen after the last sample or after it
leaves the box through a nonperiodic face (`boundary_exit`). The runtime must
carry the matching `PrescribedChargeCurrentSourcePlan`, whose preparation
certifies the electric support of every step so Huygens boxes can verify
`J = 0` on their surfaces. Any linear runtime is accepted: dispersive, lossy,
heterogeneous, magnetized-plasma, and negative-index media, CPML, and
PEC/PMC/impedance boundaries including interior conductors
(`MaxwellBoundaryPlan(kind, support=mask)`). `solve_prescribed_charge_maxwell`
runs one `lax.scan` and returns per-step leapfrog energy, source work
`−Δt⟨(E_n+E_{n+1})/2, ⋆J⟩`, trapezoidal losses, and `PrescribedChargeEvidence`
(continuity, Gauss against the prescribed charge on boundary-free vertices,
runtime Gauss constraint, magnetic, support-leak, ledger, and exit evidence;
`PrescribedChargeStatus` flags).

With CPML the power ledger is an `O(Δt²)` residual of the absorber: at CFL 0.9
it typically exceeds the default `1e-2` tolerance (`LEDGER_OPEN`) while every
constraint bit stays clear, and it falls by ≈ 4 per halving of Δt. The example
and the benchmark therefore run below CFL 0.45.

Prescribed runs match the `FrequencyMovingChargePlan` solve of the same bridge,
material, absorber, and conductors. In a periodic cell with the charge
advancing `h/r` per step, harmonic `m` of `T = L/v` is an exact Bloch wave and
`T` times its whole-period sample-mean phasor is the single-charge transform
the frequency-domain route solves for. Projecting both onto `e^{ikz}`
(each Smith–Purcell order separately) removes the start transient, and
averaging whole-period windows that start one period apart removes waves that
graze along the absorber-free axis. The routes then differ only by
`O((ωΔt)²)`: Cherenkov and Smith–Purcell fields and spectral Poynting fluxes
agree to about `1e-3` at `Δt = h/(4β)` and `h/(6β)`. Diffraction radiation of
a slit in an open box needs a finite path. Start and stop at rest with slow
ramps, since ramp radiation reaches the screen Doppler-compressed by
`1 − β`. Add a delayed opposite charge on the same path so that no static
field remains. Taper the time cut, since a line charge's 2-D wake decays only
algebraically. The result then matches the scattered-field reference to the
`O((kh)²)` difference between the lattice and the analytic incident field.
For line charges use the planar `tez` layout for the scattered-field route: in
a `full_3d` bridge with periodic transverse axes its analytic incident field
is a single point charge, not the periodic image line.

::: phydrax.solver.maxwell.PrescribedChargeTrajectory

---

::: phydrax.solver.maxwell.PrescribedChargeCurrentSourcePlan

---

::: phydrax.solver.maxwell.PrescribedChargeMaxwellPlan

---

::: phydrax.solver.maxwell.PrescribedChargeMaxwellResult

---

::: phydrax.solver.maxwell.PrescribedChargeEvidence

---

::: phydrax.solver.maxwell.PrescribedChargeStatus

---

::: phydrax.solver.maxwell.solve_prescribed_charge_maxwell

## Frequency-domain moving charges

`MaxwellMovingChargePlan(bridge, layout, charge=, speed=, origin=, direction=)`
describes one charge in uniform rectilinear motion: a point charge on
`full_3d` layouts, a line charge per unit `z` length on `tez` layouts. Its
preparation integrates the transformed current `q d̂ δ_⊥ exp(iωs/v)` exactly on
the Whitney edge forms (closed-form moments of the polynomial Whitney factors on
every crossed cell) and the transformed charge on the Whitney node forms, so
`d₀ᵀ b + iω b₀ = 0` to roundoff off the path ends. A path along a periodic axis
is one closed pass and requires `ωL/v ∈ 2πℤ`; it then equals the transform of
one charge on an infinite line. `FrequencyMovingChargePlan` solves
`SourceFormulation` `"total-field"` (`A E = iω J̃`) or `"scattered-field"` (the
analytic `UniformMotionFieldPlan` field of a declared homogeneous background is
the incident field; only material contrast and conductor surfaces radiate) with
the `FrequencyMaxwellOperator` Krylov or sparse-direct route, the B2a
stretching, and the shared `MaxwellBoundaryPlan` vocabulary.
`FrequencyMovingChargeEvidence` reports the branch (`radiating`, transverse
wavenumber), the bound-field reach `γβλ = 2π/Im k_ρ` against the transverse
clearance to the absorbers, the Whitney continuity defect, the closest
scattered-field source distance, open endpoints, and solve convergence.

::: phydrax.solver.maxwell.SourceFormulation

---

::: phydrax.solver.maxwell.MaxwellMovingChargePlan

---

::: phydrax.solver.maxwell.PreparedMaxwellMovingCharge

---

::: phydrax.solver.maxwell.FrequencyMovingChargePlan

---

::: phydrax.solver.maxwell.PreparedFrequencyMovingCharge

---

::: phydrax.solver.maxwell.FrequencyMovingChargeResult

---

::: phydrax.solver.maxwell.FrequencyMovingChargeEvidence

## Huygens surfaces and far fields

Phasors follow `exp(-iωt)`. A Huygens sampler accumulates the transient
spectrum `∫ f(t) e^{+iωt} dt` of the tangential fields on a closed surface, so
its acquisition must use `measure="time-integral"` and `sign="positive"`. The
far field of the equivalent currents `J = n̂ × H̃`, `M = −n̂ × Ẽ` in the declared
homogeneous exterior uses `k = ω√(εμ)` and `η = √(μ/ε)`; `spectral_energy` is the
one-sided `d²W/(dω dΩ) = εc|rẼ|²/π` and `spectral_poynting_energy` the matching
surface integral `(1/π) Re ∫ (Ẽ × H̃*)·n̂ dS`.

Admissibility is refused, not approximated: the surface entities must carry the
declared lossless homogeneous exterior (diagonal, or conductive with zero
conductivity), no CPML term or boundary constraint may touch the surface, and no
electric or magnetic current may drive it during the acquisition window (dynamic
PIC currents are refused because they cannot certify `J = 0`; prescribed-charge
sources are admitted when their certified path support avoids the surface).
Structured boxes require the `full_3d` polarization.

::: phydrax.solver.maxwell.HomogeneousMaxwellExterior

---

::: phydrax.solver.maxwell.MaxwellHuygensSampler

---

::: phydrax.solver.maxwell.MaxwellHuygensBoxPlan

---

::: phydrax.solver.maxwell.MaxwellHuygensSurfacePlan

---

::: phydrax.solver.maxwell.HuygensSurfacePhasors

---

::: phydrax.solver.maxwell.MaxwellFarFieldPlan

---

::: phydrax.solver.maxwell.MaxwellFarFieldResult

---

::: phydrax.solver.maxwell.spectral_poynting_energy

## Harmonic and material evidence

::: phydrax.solver.maxwell.MaxwellHarmonicDefectReport

---

::: phydrax.solver.maxwell.compatible_maxwell_harmonic_defect

---

::: phydrax.solver.maxwell.MaxwellScalarMaterialAssemblyPolicy

---

::: phydrax.solver.maxwell.assemble_scalar_maxwell_material

## Dispersive and magnetized media

::: phydrax.solver.maxwell.MaxwellLorentzPoles

---

::: phydrax.solver.maxwell.PreparedLorentzDrudeMaxwellConstitutive

---

::: phydrax.solver.maxwell.drude_maxwell_constitutive

---

::: phydrax.solver.maxwell.MagnetizedColdPlasmaMaxwellConstitutivePlan

---

::: phydrax.solver.maxwell.PreparedMagnetizedColdPlasmaMaxwellConstitutive

---

::: phydrax.solver.maxwell.MagnetizedColdPlasmaState

## Frequency response and stretched coordinates

::: phydrax.solver.maxwell.AbstractMaxwellFrequencyResponse

---

::: phydrax.solver.maxwell.DiagonalMaxwellFrequencyResponse

---

::: phydrax.solver.maxwell.InstantaneousMaxwellFrequencyResponse

---

::: phydrax.solver.maxwell.MagnetizedColdPlasmaFrequencyResponse

---

::: phydrax.solver.maxwell.FrequencyMaxwellPowerLedger

---

::: phydrax.solver.maxwell.FrequencyMaxwellSolveMethod

## Discrete-dispersion audit

::: phydrax.solver.maxwell.MaxwellMaterialRegion

---

::: phydrax.solver.maxwell.CompatibleMaxwellDispersionAudit

---

::: phydrax.solver.maxwell.MaxwellDispersionResult

---

::: phydrax.solver.maxwell.CherenkovRegimePlan

---

::: phydrax.solver.maxwell.CherenkovRegimeEvidence

## Independent case batching

::: phydrax.solver.maxwell.PreparedCompatibleMaxwellCaseBatch

---

::: phydrax.solver.maxwell.prepare_compatible_maxwell_case_batch

---

::: phydrax.solver.maxwell.solve_compatible_maxwell_case_batch

## Reduced-dimensional compatible Maxwell

::: phydrax.solver.CompatibleMaxwell1DPlan

---

::: phydrax.solver.CompatibleMaxwell1DState

---

::: phydrax.solver.CompatibleMaxwell2DPlan

---

::: phydrax.solver.CompatibleMaxwell2DState

---

::: phydrax.solver.PreparedReducedMaxwellCPML

## Fourier-modal lifecycle

::: phydrax.solver.maxwell.fourier_modal.FourierModalMaxwellProblem

::: phydrax.solver.maxwell.fourier_modal.FourierModalSolvePolicy

::: phydrax.solver.maxwell.fourier_modal.FourierModalSolvePlan

::: phydrax.solver.maxwell.fourier_modal.PreparedFourierModalMaxwell

::: phydrax.solver.maxwell.fourier_modal.plan_fourier_modal_maxwell

::: phydrax.solver.maxwell.fourier_modal.prepare_fourier_modal_maxwell

::: phydrax.solver.maxwell.fourier_modal.refresh_fourier_modal_maxwell

::: phydrax.solver.maxwell.fourier_modal.solve_fourier_modal_maxwell

---

::: phydrax.solver.maxwell.fourier_modal.fourier_modal_numeric_revision

---

::: phydrax.solver.maxwell.fourier_modal.fourier_modal_physical_state_digest

---

::: phydrax.solver.maxwell.fourier_modal.fourier_modal_physical_stack_digest

---

::: phydrax.solver.maxwell.fourier_modal.require_fourier_modal_numeric_revision

::: phydrax.solver.maxwell.fourier_modal.PreparedFourierModalCaseBatch

::: phydrax.solver.maxwell.fourier_modal.FourierModalCaseBatchResult

::: phydrax.solver.maxwell.fourier_modal.prepare_brillouin_zone_maxwell

::: phydrax.solver.maxwell.fourier_modal.solve_fourier_modal_case_batch

## Materials, layers, and ports

::: phydrax.solver.maxwell.fourier_modal.FrequencyMaxwellMaterial

::: phydrax.solver.maxwell.fourier_modal.HomogeneousMaxwellPort
::: phydrax.solver.maxwell.fourier_modal.PeriodicMaxwellPort


::: phydrax.solver.maxwell.fourier_modal.FourierModalLayer
::: phydrax.solver.maxwell.fourier_modal.ContinuousFourierModalLayer

::: phydrax.solver.maxwell.fourier_modal.ContinuousZIntegrationPolicy

::: phydrax.solver.maxwell.fourier_modal.LateralTransformationOpticsPMLPlan

::: phydrax.solver.maxwell.fourier_modal.transform_fourier_modal_material


::: phydrax.solver.maxwell.fourier_modal.FourierModalSourcePlane

## Geometry rasterization

::: phydrax.solver.maxwell.fourier_modal.FourierModalRasterizationPolicy

::: phydrax.solver.maxwell.fourier_modal.FourierModalRasterizationPlan

::: phydrax.solver.maxwell.fourier_modal.FourierModalRasterizationResult

::: phydrax.solver.maxwell.fourier_modal.FourierModalRasterizationEvidence

::: phydrax.solver.maxwell.fourier_modal.rasterize_fourier_modal_material

## Factorization and propagation

::: phydrax.solver.maxwell.fourier_modal.DirectFourierFactorizationPlan

::: phydrax.solver.maxwell.fourier_modal.InverseFourierFactorizationPlan

::: phydrax.solver.maxwell.fourier_modal.VectorFourierFactorizationPlan

::: phydrax.solver.maxwell.fourier_modal.AnalyticInterfaceFramePlan

::: phydrax.solver.maxwell.fourier_modal.JonesDirectFramePlan

::: phydrax.solver.maxwell.fourier_modal.BoundaryCascadePolicy

::: phydrax.solver.maxwell.fourier_modal.ModalPropagationPolicy

## Excitations and observables

::: phydrax.solver.maxwell.fourier_modal.FourierModalExcitation

::: phydrax.solver.maxwell.fourier_modal.plane_wave_excitation
::: phydrax.solver.maxwell.fourier_modal.port_mode_excitation

A `MovingLineChargeSource` in a `FourierModalSourcePlane` is the single
zeroth-harmonic surface current `λ d̂ exp(i k_B·r)` of a line charge moving at
`v`; the problem's Bloch wavevector must be `k_B = (ω/v) d̂ + k_⊥ ê_⊥`, and
`moving_line_charge_excitation` builds its source-only excitation.
`MovingPointChargeQuadrature` decomposes a point charge into `k_⊥` components
with decay `Γ = √(ω²/(β²γ²c²) + k_⊥²)`, truncates where `exp(−2Γ h)` reaches the
tolerance, and integrates per-component results with cosine-mapped embedded
Gauss–Kronrod panels (`MovingPointChargeIntegral` carries the Kronrod–Gauss
difference and the truncation bound). The one-sided spectral energy per cell is
`(2/π)` times the outgoing port power of the transformed fields.

::: phydrax.solver.maxwell.fourier_modal.MovingLineChargeSource
::: phydrax.solver.maxwell.fourier_modal.moving_line_charge_excitation
::: phydrax.solver.maxwell.fourier_modal.MovingPointChargeQuadrature
::: phydrax.solver.maxwell.fourier_modal.MovingPointChargeIntegral


::: phydrax.solver.maxwell.fourier_modal.fields_in_layer

::: phydrax.solver.maxwell.fourier_modal.diffraction_order_far_field
::: phydrax.solver.maxwell.fourier_modal.FiniteApertureFarFieldPlan

::: phydrax.solver.maxwell.fourier_modal.finite_aperture_far_field

::: phydrax.solver.maxwell.fourier_modal.FourierModalHarmonicAdaptationPolicy

::: phydrax.solver.maxwell.fourier_modal.solve_adaptive_fourier_modal_case


::: phydrax.solver.maxwell.fourier_modal.FourierModalSolveResult

::: phydrax.solver.maxwell.fourier_modal.FourierModalDiagnostics

## Directional power and independent physical loss

::: phydrax.solver.maxwell.fourier_modal.FourierModalLossPolicy

---

::: phydrax.solver.maxwell.fourier_modal.FourierModalLossEvidence

---

::: phydrax.solver.maxwell.fourier_modal.FourierModalLossStatus

---

::: phydrax.solver.maxwell.fourier_modal.evaluate_fourier_modal_loss

---

::: phydrax.solver.maxwell.fourier_modal.FourierModalLossConvergenceEvidence

---

::: phydrax.solver.maxwell.fourier_modal.assess_fourier_modal_loss_convergence

## Fixed-frequency guided modes

::: phydrax.solver.maxwell.FixedFrequencyGuidedModePlan

---

::: phydrax.solver.maxwell.PreparedFixedFrequencyGuidedModes

---

::: phydrax.solver.maxwell.FixedFrequencyGuidedModeResult

---

::: phydrax.solver.maxwell.solve_fixed_frequency_guided_modes

---

::: phydrax.solver.maxwell.guided_mode_beta_derivative

## Equivalent-slab retrieval and local-isotropic qualification

::: phydrax.solver.maxwell.fourier_modal.MaxwellModalSweep

---

::: phydrax.solver.maxwell.fourier_modal.prepare_maxwell_modal_sweep

---

::: phydrax.solver.maxwell.fourier_modal.EquivalentSlabRetrievalPlan

---

::: phydrax.solver.maxwell.fourier_modal.EquivalentSlabRetrieval

---

::: phydrax.solver.maxwell.fourier_modal.EquivalentSlabRetrievalStatus

---

::: phydrax.solver.maxwell.fourier_modal.retrieve_equivalent_slab

---

::: phydrax.solver.maxwell.fourier_modal.LocalIsotropicQualificationPolicy

---

::: phydrax.solver.maxwell.fourier_modal.LocalIsotropicMediumQualification

---

::: phydrax.solver.maxwell.fourier_modal.LocalIsotropicQualificationStatus

---

::: phydrax.solver.maxwell.fourier_modal.qualify_local_isotropic_medium

## Sparse finite-element Maxwell

`UnstructuredMaxwellPlan(complex, constitutive, /, *, spectral_upper_bound=None,
courant_factor, boundary="absolute")` uses `FiniteElementDeRhamComplex` and
separate `FiniteElementMaxwellConstitutivePlan` weighted coordinate Grams.
Automatic CFL preparation applies native Lanczos to M1⁻¹K1 in the M1 Hilbert
pairing, a faithful Riesz realization of the generalized (K1,M1) pencil—not a
Euclidean self-adjoint relabeling. A guaranteed trace upper certificate bounds
the spectrum; a largest Ritz value alone is not an upper bound.
Repeated stepping reuses native device solve preparation.
`PreparedUnstructuredMaxwell.mesh_quality` retains canonical meshing
`CellQualityEvaluation` for FE geometry; a meshless abstract cochain has no
fabricated quality report. The trace bound uses linear coordinate workspace,
not a constant-byte claim. Cell/tensor material solves retain native solve
evidence and explicit failure checks.

Unstructured conducting PIC requires `boundary="relative"`; its Gauss correction
uses the restricted metric inverse, never a masked full inverse. Whitney-2
reconstruction gathers physical B without per-cell least squares.

::: phydrax.solver.maxwell.UnstructuredMaxwellPlan

::: phydrax.solver.maxwell.FiniteElementMaxwellConstitutivePlan

## Matching and nonmatching Maxwell FEM–BEM

`prepare_matching_maxwell_fem_bem_3d(complex, interior_operator, /, *,
wavenumber, wave_impedance=1, boundary_policy=None, policy=None,
residual_tolerance=1e-5)` automatically assembles the lowest-order trimmed
tetrahedral n×E RWG trace, BC dual conormal and genuine BC EFIE boundary matrix.
Its prepared artifact retains `rwg_trace`, `dual_trace`, `dual_conormal` and the
physical `magnetic_conormal`, both block residuals and solver/geometry evidence.
The upper block includes i k η Qᴴ G_BC⁻¹(G_BC/2 + K_BC), with the actual outgoing
MFIE kernel and half jump; it is not Qᴴ alone. The BC Gram inverse uses native
prepared PCG with failure propagated to outer status. Coordinate adjoint, finite
quadrature envelope and boundary evidence remain explicit.

`prepare_maxwell_fem_bem_3d(interior_operator, boundary, mortar, /, *, policy=None,
residual_tolerance=1e-5)` consumes an explicitly typed volume/boundary mortar;
`volume_complex` and `boundary_space` identities are required by its owner.
The upper block uses magnetic conormal adjoint and the lower block weak trace.

Caller-built periodic bounded-image boundary coupling remains supported with
its finite-envelope evidence; it is not automatic periodic matching or
infinite-lattice certification. A supplied periodic block alone is not a claim
of automatic physical trace/conormal construction.

::: phydrax.solver.prepare_matching_maxwell_fem_bem_3d

::: phydrax.solver.prepare_maxwell_fem_bem_3d

## Hodge–Laplace and cavity recipes

::: phydrax.solver.HodgeLaplacePlan

::: phydrax.solver.maxwell_cavity_modes

`maxwell_cavity_modes` uses resource-bounded eager native full `DenseEigh`
admission and selects positive/kernel-complement modes. It retains actual dense
provenance, status, iterations/matvecs, per-mode convergence, effective count and
conservative full-spectrum orthogonality evidence. It does not run a redundant
LOBPCG stage or silently fall back from a failed iterative solve.
