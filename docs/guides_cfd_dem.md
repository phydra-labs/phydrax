# CFD–DEM coupling

## Conservative transfer

`MeshSplatTarget` plus `MeshCompactKernelSplatAssignment` builds the canonical
fixed-capacity normalized particle-to-cell route on a declared `CellMesh`.
Gather and extensive deposition use one adjoint-paired stencil. Particle volume,
momentum, force, impulse, and heat remain extensive until division by the
receiving cell measure. Empty support, route overflow, or failed normalization
is explicit evidence; inactive runtime population slots contribute exactly zero.

## Unresolved coupling

`UnresolvedCFDEMCouplingPlan` combines prepared DEM, conservative transfer,
porosity bounds, and a `drag_law: AbstractInterphaseDragLaw`. Construct a law with
`InterphaseDragLawPlan(correlation, /, *, maximum_reynolds=None)`. The closed
correlation set is `stokes`, `schiller_naumann`, `di_felice`, `ergun`, `wen_yu`,
`gidaspow`, `huilin_gidaspow`, and `beetstra`.

Pressure-gradient force is separate from drag:
`include_pressure_gradient=True` belongs to the coupling plan, not the law.
The particle force and fluid momentum source are equal and opposite through
the existing transfer adjoint. `advance_cfd_dem_window` requires an integer
number of DEM substeps per fluid macro-step. Hydrodynamic impulse is accumulated
over the accepted window; fluid and DEM candidates commit atomically or both
roll back.

### Bulk exchange versus particle resistance

Evaluate a law with the positional arguments
`evaluate(porosity, relative_speed, particle_diameter, dynamic_viscosity,
fluid_density, particle_volume, /)`. The speed is the magnitude of the
**interstitial** fluid–solid slip, not the superficial velocity.

Let ε be porosity, φ = 1 − ε, w the slip speed, d the diameter, μ the dynamic
viscosity, ρ the intrinsic fluid density, and Vp the supplied particle volume.
`InterphaseDragEvaluation.beta` has units kg m⁻³ s⁻¹:
β (vf − vs) is force per mixture volume. In contrast,
`particle_drag_coefficient` has units kg s⁻¹ (N s m⁻¹):
Kp (vf − vp) is the drag force on one particle.

For isolated-sphere Stokes and Schiller–Naumann laws, Kp is based on diameter:
Kp = 3πμd B, where β = 18μφ B/d². Di Felice uses the same sphere-based conversion
with its voidage resistance B. These coefficients must not be replaced by
Vp β/φ: a planar DEM model may supply Vp independently of the geometric sphere
volume. The packed-bed laws use Kp = Vp 18μ B/d², equivalently Vp β/φ when φ > 0;
the first expression also defines the algebraic pure-fluid limit without 0/0.
Two-phase continuum consumers use β, not Kp.

### Equations and Reynolds convention

The returned `reynolds_number` is always Re = ρdw/μ. The Wen–Yu, Di Felice,
and Beetstra fits instead use **Rs = ε Re**, based on superficial slip U = εw.
`maximum_reynolds` limits the Reynolds number used by the selected fit, not
necessarily the returned interstitial Re. The following normalized resistance
forms make the distinction explicit: β = 18μφ B/d².
Gidaspow applies this cap to interstitial Re in its dense Ergun branch and to
superficial Rs in its dilute Wen–Yu branch. Huilin–Gidaspow blends both branches
with nonzero weights, so both Re and Rs must satisfy the cap.

Define S(x) = 1 + 0.15 x^0.687. For Wen–Yu, its isolated-sphere resistance is
S(x) for x < 1000 and 0.44 x/24 for x ≥ 1000.

| Correlation | Normalized resistance B |
| --- | --- |
| `stokes` | 1 |
| `schiller_naumann` | S(Re), within the admitted finite-Re range |
| `di_felice` | (0.63 √Rs + 4.8)² ε^(1−χ)/24 |
| `ergun` | (150/18) φ/ε + (1.75/18) Re |
| `wen_yu` | S(Rs) ε^−2.65, using the high-Rs branch above when applicable |
| `gidaspow` | B_Ergun for ε ≤ 0.8; B_Wen–Yu for ε > 0.8 |
| `huilin_gidaspow` | s B_Ergun + (1−s) B_Wen–Yu |
| `beetstra` | ε (f0 + fR) |

Di Felice's voidage exponent is
χ = 3.7 − 0.65 exp[−(1.5 − log₁₀ Rs)²/2]. Its zero-speed value is χ = 3.7.
The Dallavalle isolated-sphere coefficient underlying this form is
Cd = (0.63 + 4.8/√Rs)², not Schiller–Naumann. In particular, its normalized
zero-speed resistance is 0.96 ε^−2.7, not the unit Stokes resistance.

The Huilin–Gidaspow Ergun weight is
s = 1/2 + arctan[150 × 1.75 × (φ − 0.2)]/π.
The numerical constants and transition center are part of the correlation;
there is no tunable smoothing width.

Beetstra's monodisperse fit is
f0 = 10φ/ε² + ε²(1 + 1.5√φ), and
fR = [0.413 Rs/(24ε²)]
\[ε⁻¹ + 3φε + 8.4 Rs^−0.343\] /
[1 + 10^(3φ) Rs^−(1+4φ)/2].
At zero speed fR = 0. This is the frictional drag definition from the paper,
excluding the separately modeled pressure-gradient force; no bidisperse or
polydisperse correction is implied by the `beetstra` selector.

### Admission, evidence, and differentiation

The inputs must be finite, with 0 < ε ≤ 1, w ≥ 0, and positive d, μ, ρ, and Vp.
The caller masks phase-absent cells; the law does not clip porosity or floor a
physical denominator. Defaults cap Stokes at Re = 1 and Schiller–Naumann at
Re = 1000. Beetstra admits ε in [0.35, 0.9] and Rs ≤ 1000, reflecting the
published random-array data range. Wen–Yu is the dilute branch, ε > 0.8.
For correlations without a source-specific upper Reynolds bound, an uncapped
default does not constitute validation of arbitrary extrapolation.
An explicit `maximum_reynolds` may tighten, but never widen, a finite default.
Stokes's finite cap is an admission policy, not a claim that creeping-flow
asymptotics are exact at its endpoint.

`CFDEMCouplingEvaluation.drag` preserves all law evidence:
`beta`, `particle_drag_coefficient`, `reynolds_number`, `branch_code`,
`branch_margin`, `derivative_valid`, `admissible`, `successful`, and static
`law_id`. Admission and numerical success are not derivative qualification.
JAX AD differentiates the implemented formulas within a fixed smooth branch.
The signed Gidaspow porosity margin is ε − 0.8; its hard switch is not
derivative-qualified at zero margin. Huilin–Gidaspow is smooth through that
porosity transition but still inherits any Reynolds branch boundary.
Zero-speed analytic value limits do not imply valid speed derivatives for
fractional-power or square-root correlations. Consult `derivative_valid`
before using a JVP/VJP; no custom derivative or second-order guarantee is
claimed. Status flags and branch selection are diagnostics, not smooth
observables.

### Clean API cutover and sources

`InterphaseDragLawPlan`, `AbstractInterphaseDragLaw`, and
`InterphaseDragEvaluation` replace the old particle-force closure API.
The canonical coupling argument and law identity are `drag_law` and `law_id`.
There are no compatibility aliases for the removed particle-force closure API.

The formulas above specify the implemented normalization. Primary-source access
is incomplete for two details: the original Di Felice full text containing the
Dallavalle constants and the original Lu–Gidaspow full text containing the
blending constants were not available during source review. Their bibliographic
records and the Di Felice volumetric-flux convention were recovered, but the
0.63/4.8 and 150/1.75 constants must not be described as independently verified
against those original full texts. Numerical agreement with these formulas
alone is not publication-level qualification.

Secondary evidence is recorded separately from those primary references:

- The [NETL-hosted Di Felice source-examination discussion](https://mfix.netl.doe.gov/forum/t/di-felice-drag-correlation/4585)
  identifies Dallavalle, rather than Schiller–Naumann, as the original
  isolated-sphere correlation. This is a secondary discussion, not an original
  equation transcription.
- The [Ansys Fluent Theory Guide, 2024 R2, §14.5.7.2](https://ansyshelp.ansys.com/public/Views/Secured/corp/v242/en/flu_th/flu_th_sec_eulermp_theory_exchange.html#flu_th_sec_fl_sol_ex)
  gives the Huilin–Gidaspow weighting in equation (14–286) and the stitching
  function in equation (14–287). Its
  [equation graphic](https://ansyshelp.ansys.com/public/Views/Secured/corp/v242/en/flu_th/graphics/flu_th_eqn_huilin_2.svg)
  uses 262.5 (φ − 0.2), algebraically identical to 150 × 1.75 (φ − 0.2).
  This is independently accessible secondary equation evidence, not verification
  against the original 2003 article.

Primary references:

- Stokes, G. G. (1851), *On the effect of the internal friction of fluids on
  the motion of pendulums*, Transactions of the Cambridge Philosophical Society
  9, 8–106. The isolated-sphere coefficient is also stated explicitly in
  Beetstra et al.'s introduction.
- Schiller, L. and Naumann, A. (**1933**), *Über die grundlegenden Berechnungen
  bei der Schwerkraftaufbereitung*, Zeitschrift des Vereines Deutscher
  Ingenieure 77, 318–320. The [primary library holdings](https://catalog.hathitrust.org/Record/009039143)
  identify volume 77 as 1933; the frequently repeated 1935 date is inconsistent
  with those holdings.
- Di Felice, R. (1994), [*The voidage function for fluid-particle interaction
  systems*](https://doi.org/10.1016/0301-9322(94)90011-6), International Journal
  of Multiphase Flow 20(1), 153–159. The original abstract defines the reference
  drag at the same volumetric flux of fluid.
- Ergun, S. (1952), *Fluid flow through packed columns*, Chemical Engineering
  Progress 48, 89–94.
- Wen, C. Y. and Yu, Y. H. (1966), *Mechanics of fluidization*, Chemical
  Engineering Progress Symposium Series 62, 100–111.
- Gidaspow, D. (1994), *Multiphase Flow and Fluidization: Continuum and Kinetic
  Theory Descriptions*, Academic Press. The author's earlier
  [1992 conference paper](https://www.osti.gov/servlets/purl/5896246-VUbD40),
  Table 1, gives the gas–solid drag branches and superficial Reynolds convention.
- Lu, H. and Gidaspow, D. (2003), [*Hydrodynamics of binary fluidization in a
  riser: CFD simulation using two granular temperatures*](https://doi.org/10.1016/S0009-2509(03)00238-0),
  Chemical Engineering Science 58(16), 3777–3792.
- Beetstra, R., van der Hoef, M. A. and Kuipers, J. A. M. (2007),
  [*Drag force of intermediate Reynolds number flow past mono- and bidisperse
  arrays of spheres*](https://doi.org/10.1002/aic.11065), AIChE Journal 53(2),
  489–501. Equations (1)–(2) define normalized frictional drag and superficial
  Reynolds number; equation (17) is the monodisperse fit. The study combines
  finite-Re data with the earlier zero-Re fit; it is not a universal packed-bed
  validation.

## Resolved immersed-boundary penalty coupling

`MACPenaltyIBCFDEMCouplingPlan` binds fixed material marker quadrature to
rigid-sphere DEM ownership. `MACMarkerTransferPlan` uses local cubic tensor
B-spline routes on the actual staggered MAC face layouts. Gather and spread are
material/face-measure Hilbert adjoints; force, torque, virtual work, support,
and fixed-body reaction are explicit evidence.

`advance_mac_penalty_ib_cfd_dem_window` subcycles DEM contact dynamics,
accumulates trapezoidal hydrodynamic impulses, inserts the penalty source before
pressure projection, and commits fluid and particle states atomically.
`IBPenaltyPlan` reports numerical validity separately from slip qualification;
strict slip qualification remains the default acceptance policy. This path is
not the exact pressure-plus-marker multiplier method described in the dedicated
[immersed-boundary guide](guides_immersed_boundary.md).

## Volume-fraction immersed-boundary coupling

`MACVolumeFractionIBCFDEMCouplingPlan` uses a fixed intrinsic `solid_density`.
`MACVolumeFractionIBState.initialize` refuses active DEM bodies whose runtime
mass differs from `solid_density * reference_volume * shape_scale**dimension`,
or whose inverse mass is inconsistent with that mass (zero for fixed bodies).
Uniform shape scaling is supported only with consistent runtime mass.
Every step checks both accepted and candidate runtime properties; a mismatch
sets `MACVolumeFractionIBStepStatus.GEOMETRY_FAILED` and atomically retains
both phase states and impulse ledgers.

## Reactive heat and species coupling

`ParticleContinuumExchangePlan` samples fluid temperature and species concentration and deposits the exact opposite extensive heat and species sources through the same conservative transfer. `ParticleContactExchangePlan` handles reciprocal contact heat independently. `ReciprocalPairRadiationPlan` adds reciprocal particle and optional wall radiation.

`ReactiveCFDDEMCouplingPlan` combines prepared DEM, radial particle conversion, continuum exchange, optional contact exchange, morphology, and radiation. `advance_reactive_cfd_dem_window` uses Strang or fixed-iteration strong coupling and commits fluid, DEM, conversion, and morphology candidates atomically.

Distributed ownership, turbulence modulation, added mass, and monolithic fluid–particle Newton solves remain unsupported. Contact-scale near-gap lubrication is available through the DEM cohesion channel; it is not a bulk hydrodynamic closure.
