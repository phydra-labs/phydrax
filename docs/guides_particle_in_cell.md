# Particle-in-cell methods

Phydrax particle-in-cell methods compose stable material-particle supports, measure-aware
structured splats, compatible cochains, native linear solves, and transactional fixed-step
state. They do not introduce a second particle or mesh runtime.

## Charged particle support

`ChargedParticlePlan` attaches one extensive macrocharge to every slot of an existing
`ParticleDiscretization`. Active charges are finite and nonzero; inactive charges are exactly
zero. Macro mass, activity, stable ID, support, and position/velocity field-space identity remain
owned by `ParticleDiscretization`.

PIC state stores spatial position and three-component proper velocity. `RelativisticPushPlan`
converts proper velocity to physical velocity, applies one relativistic pusher selected by
`method` (`RelativisticPusher`), and reports finite and subluminal evidence in
`RelativisticPushResult`:

- `"boris"`: the relativistic Boris map; it evaluates the magnetic rotation at the Lorentz factor
  after the first electric half kick, so crossed fields with E + v×B = 0 acquire a spurious
  drift-frame velocity when the rest-mass cyclotron frequency is unresolved;
- `"vay"`: Vay (Phys. Plasmas 15, 056701, 2008), which keeps the E×B drift exact at any γ but is
  not phase-space volume preserving;
- `"higuera-cary"`: Higuera and Cary (Phys. Plasmas 24, 052104, 2017), which keeps the E×B drift
  exact and preserves phase-space volume.

All three conserve |u| in a pure magnetic field. The speed of light is the exact
`speed_of_light` of the plan's `RelativityScaleContract`; fields, specific charge, and step size
are expressed in that scale and no unit conversion is inferred. Callers holding an
`ElectromagneticScaleContract` pass its `relativity`. PIC plans constructed without a pusher use
`"boris"` in `PIC_CODE_RELATIVITY`, a declared code-unit system with c = 1.

`RelativisticPushPlan.precess(spin, u, u', E, B, q/m, anomaly, active, dt)` advances rest-frame
spin vectors over the same step with the Thomas–Bargmann–Michel–Telegdi equation, evaluated at the
time-centered velocity and the Lorentz factor the selected method uses for its magnetic rotation
and applied with the same norm-exact Cayley rotation: with a zero anomaly the spin stays locked to
the momentum in a magnetic field, and with `a = (g − 2)/2` it turns relative to it at `aγω_c`.
Processes receive the run's pusher and the step-start proper velocities in `PICProcessContext`
(`pusher`, `step_start_proper_velocity`), together with the Galilean `grid_velocity` of the field
grid; the polarized strong-field QED cascade uses them (see the strong-field QED guide).

## Instantaneous particle–cochain transfer

`PICParticleCochainTransferPlan(bridge, *, shape_order=1)` prepares only existing
`ParticleGridSplatPlan` instances:

- vertices for degree-zero charge;
- oriented edges for electric field gather;
- oriented faces for magnetic field gather.

Charge deposition returns extensive vertex charge, charge density under the vertex dual measure,
and the packed degree-zero cochain. Electric and magnetic gather first use
`StructuredCochainBridge` to recover physical edge-tangent and face-normal fields, then gather each
component from its exact tensor location.

`shape_order` (`PICShapeOrder`, 1–3) selects the spline-Whitney family. Order one is the
lowest-order Whitney transfer: multilinear charge, with degree-zero interpolation along
each edge/face span and linear interpolation across it. Order `p = 2, 3` deposits charge
with the degree-`p` cardinal B-spline and gathers each oriented component with degree
`p − 1` along the axes the entity spans and degree `p` across the others
(`TensorBSplineSplatAssignment` with per-axis degrees). Because
`dN^p(x − i)/dx = N^{p−1}(x − i + ½) − N^{p−1}(x − i − ½)`, the gathered `E` of a discrete
gradient field is exactly the gradient of the degree-`p` charge-shape interpolant of the potential.

Ordinary endpoint splatting is not current deposition.

## Compatible electrostatics

`CochainElectrostaticPlan` solves

```text
-delta(epsilon d phi) = rho
```

with the supplied cochain exterior derivative, codifferential, Hodge pairing, and `phydrax.linalg`.
Fully periodic problems require explicitly neutral particle-plus-background charge and use a
zero-mean potential gauge. Bounded problems initially support homogeneous Dirichlet potential.
Nonneutral periodic charge is rejected rather than silently mean-subtracted.

`ElectrostaticPICPlan` stores synchronized particle and field state and advances it with
kick–drift–kick. Each step deposits new endpoint charge, solves one new electrostatic field,
gathers E, reports kinetic/field energy, and commits only if transfer, solve, pusher, displacement,
and finite-state checks all pass.

## Charge-conserving electromagnetic coupling

`ChargeConservingCurrentPlan` prepares fixed-capacity exact spline-Whitney chain
integration on periodic or wall-bounded structured grids. Paths may cross multiple
cells; the declared positive segment capacity determines admission, with explicit
overflow/exit evidence rather than a one-cell-per-axis rule.
Order one uses actual nonuniform widths and exact facet intervals. Orders two and
three on uniform axes split at the common knot lattice and integrate the polynomial
chain moments exactly. Prepared gather/deposit routes are conjugate adjoints and
reuse canonical DOF offsets. Current content is +∫W and every order satisfies

```text
(rho_new - rho_old) / dt - delta(J_mid) = 0
```

to roundoff under the exact `StructuredCochainBridge.codifferential`. Segment overflow, dropped
support, or a continuity defect rejects the step.

`PICMaxwellCurrentSourcePlan` is the native dynamic Maxwell source plan. The
same `PICMaxwellCurrentArguments` instance is passed to Maxwell stepping and
diagnostics, so both see the same deposited midpoint current through the
prepared `sources` collection.

## Field-solver protocol

Explicit electromagnetic PIC is one runtime, `ElectromagneticPICPlan`, over any
`AbstractPreparedPICFieldSolver`. The field solver owns the discrete field, its Gauss charge,
and the particle↔grid transfers bound to its discretization; its minimal core is

- `deposit`: charge-conserving current of one species along a step, with start/end charge in
  the solver's own Gauss-charge layout;
- `advance`: one field step driven by the summed current;
- `gather(derivative_order)`: physical E and B at particle positions, and for
  `derivative_order=1` their spatial gradients `∂F_i/∂x_j` as exact forward-mode derivatives of
  the solver's own interpolant.

The deposit↔Gauss pairing is verified numerically when the PIC plan is prepared: every species
deposits a probe path, the solver advances a zero field carrying the start charge with that
current, and the current-driven Gauss charge (`PICFieldAdvance.charge`, the field's own discrete
divergence applied to the current) must equal the deposited end charge
(`ElectromagneticPICPlan.pairing_defect`). A solver whose Gauss charge does not follow its own
deposited current is refused at construction. Charge the field acquires by itself — induced
charge of conducting walls, conduction or plasma charge of the medium, and the bookkeeping
divergence of absorbing layers — stays in the field state and is not a pairing defect.
The probe runs through one module-level compiled function reused by every plan over the same
solver structure, so preparation does not dispatch the deposit and field update op by op.

| Solver | Field | Transfer |
|---|---|---|
| `CochainMaxwellPICFieldSolver` | 3-D `PreparedCompatibleMaxwell`, periodic or bounded axes, boundaries, CPML, linear passive media | cochain splats + `ChargeConservingCurrentPlan` per species |
| `ReducedMaxwellPICFieldSolver` | `CompatibleMaxwell1DPlan`/`CompatibleMaxwell2DPlan`, periodic or bounded axes | `ReducedPICTransferPlan` |
| `UnstructuredMaxwellPICFieldSolver` | tetrahedral `PreparedUnstructuredMaxwell` | `UnstructuredWhitneyCurrentPlan` |
| `PreparedSpectralMaxwell` | Cartesian PSATD, global-FFT or local-guarded ([Spectral PIC](guides_spectral_pic.md)) | cochain splats + `ChargeConservingCurrentPlan` per species |
| `PreparedQuasiCylindricalMaxwell` | azimuthal-mode PSATD | `AzimuthalTransferPlan` |

Optional capabilities are structural protocols, each named by a `PICFieldSolverCapability`:
`PICTensorLayout` (`"tensor-layout"`, the structured component layout `PICFilterPlan` filters),
`PICSpectralSymbol` (`"spectral-symbol"`, vacuum numerical dispersion `ω(k)`), `PICHuygensSampling`
(`"huygens-sampling"`, phasors of the solver's Huygens observers), `PICMultiDeposit`
(`"multi-deposit"`, one fused deposit of every species), `PICWindowShift` (`"window-shift"`,
integer-cell translation, consumed by `PICMovingWindowPlan`), `PICGalileanGrid`
(`"galilean-grid"`, a grid translating at `grid_velocity`), `PICEnergyAccounting`
(`"energy-accounting"`, field, magnetic, and medium energy split plus the source-free loss power the
energy ledger integrates), `PICOpenDomain` (`"open-domain"`, periodic and wall-bounded axes of the
field box and the wall inset each species' stencil needs), `PICRestartState` (`"restart-state"`),
`PICGaussProjection` (`"gauss-projection"`, curl-free Poisson projection of the field onto a
prescribed Gauss charge, reporting `divergence_before`/`divergence_after`, the added field energy,
and its `"cochain-poisson"` or `"spectral-poisson"` route), and `PICRelativisticSelfFields`
(`"relativistic-self-fields"`, boosted-Coulomb initial fields of drifting species). Prescribed
fields enter through `phydrax.discretization.pic.ExternalFieldSource` and are added to every
gather. The semi-implicit ECSIM runtime `SemiImplicitPICPlan` solves particles and field jointly
and remains a separate orchestrator.

Every prepared solver declares its capability matrix: `solver.pic_capabilities` is a
`PICFieldSolverCapabilities` holding one `PICCapabilityRecord` per capability, in canonical order,
with `published` (the solver structurally implements the protocol the runtime checks), `admitted`
(this configuration executes the route), and `basis` (the route, or the refusal reason).
`published` is verified against the structural protocol check whenever the matrix is built, so a
record cannot claim a protocol the runtime would not find or hide one it would use. A published but
unadmitted protocol refuses when called for the configuration reason in `basis`.

| Capability | cochain 3-D | reduced 1-D/2-D | unstructured Whitney | Cartesian PSATD (global, local) | quasi-cylindrical PSATD |
|---|---|---|---|---|---|
| `tensor-layout` | admitted | admitted | — | — | — |
| `spectral-symbol` | admitted for homogeneous diagonal media (Yee) | admitted (Yee) | — | admitted (`c\|[k]\|`; local-guarded steps differ by at most `guard_truncation(dt)`) | admitted |
| `huygens-sampling` | published; refused: Maxwell refuses Huygens boxes beside the dynamic PIC current, so no phasors exist | — | — | admitted with observers (standard, staggered, no antennas); published and refused without observers | admitted with observers; published and refused without |
| `multi-deposit` | — | admitted | — | admitted | — |
| `window-shift` | admitted (uniform axis) | admitted | — | — (periodic box) | admitted (axis 2) |
| `galilean-grid` | — | — | — | admitted for the Galilean variants; published and refused for `"standard"` (lab-fixed grid) | admitted for the Galilean variants; published and refused for `"standard"` |
| `energy-accounting` | admitted | — | — | — | — |
| `open-domain` | admitted | — | — | — | — |
| `restart-state` | admitted | admitted | admitted | admitted | admitted |
| `gauss-projection` | admitted (cochain Poisson) | admitted | admitted | admitted (spectral Poisson) | admitted |
| `relativistic-self-fields` | published; admitted only with a zero-valued grounded electrostatic boundary | — | — | — | — |

"—" is unpublished. Without `open-domain`, periodic particle faces are refused and bounded faces are
not inset-checked; without `energy-accounting` the ledger uses the total field energy.

The Whitney transfer returns endpoint nodal charge and integrated edge flow.
Unstructured conducting PIC explicitly requires `UnstructuredMaxwellPlan(...,
boundary="relative")`; the degree0/1 metric inverses are genuinely restricted,
then zero-extended onto Maxwell's layout. Charge lives on interior vertices.
On a nonperiodic reduced axis the stored layout keeps the upper wall face
and omits the lower one: the backward difference reads zero below the lower wall and the forward
difference zero beyond the last cell, a skew-adjoint pair. The reduced 1-D and 2-D curl updates
therefore conserve energy exactly, the divergence of the curl vanishes axis by axis, and Gauss's
law uses the transfer's continuity divergence, so reduced 2-D fields pair to roundoff on any mix of
periodic and bounded axes. The reduced transfer projects the raw midpoint current onto continuity
with the same operator (a cumulative sum in 1-D, the per-axis eigenbasis of the reduced Poisson
operator in 2-D); the corrected current on an upper wall face is the physical outflow.

### Field-solver state handoff

Replacing the field solver of a declaration is construction of a new prepared solver (see
[Spectral PIC](guides_spectral_pic.md#substituting-the-cochain-solver)). Converting a running state
is a separate explicit operation: `hand_off_pic_state(source, target, state, step_size)` returns a
`PICFieldHandoffResult` with the converted `ElectromagneticPICState` for `target` and its
`PICFieldHandoffEvidence`. It is defined only between `CochainMaxwellPICFieldSolver` and
`PreparedSpectralMaxwell` (`"cochain-to-spectral"` and `"spectral-to-cochain"`), for both plans
declaring the same run (species, processes, boundaries, recorders, filters, guards, external fields,
ownership, precision, pusher, key), the same bridge, transfers, and current plans, on a periodic
uniform grid with a lossless stateless homogeneous medium and the PIC current as the only Maxwell
source. On that grid the degree-1 and degree-2 cochains are the edge circulations `E_a Δ_a` and
face fluxes `B_c Δ_a Δ_b` at the Yee positions where `grid="staggered"` PSATD stores `E_a` and
`B_c`, and the order-2 finite PSATD stencil `[k] = 2 sin(kΔ/2)/Δ` is exactly the Yee node
difference, so the conversion preserves Gauss's law and `∇·B` identically; both runtimes hold `E`
and `B` at the same integer time. The evidence reports one mean-free relative Gauss residual on
both sides (`−δD − (ρ − ρ̄)` and `ε∇⁻·E − (ρ − ρ̄)`, the mean being invisible to PSATD), relative
magnetic residuals, the synchronized field energies and their relative difference, total charges,
and `leapfrog_energy_correction`, the Yee half-kick term by which the ledger's field energy jumps
across the handoff. Preservation carries a violated constraint over unchanged, so
`constraint_satisfied` separately checks the converted state against the target plan's absolute
`constraint_tolerance` with the target solver's own per-step residuals: Gauss's law including the
mean charge (a non-neutral periodic state fails in either direction) and `∇·B`. `successful`
requires finite fields, `constraint_satisfied`, and target residuals, energy, and charge
preserved to roundoff. Infinite-order or higher-order stencils and collocated grids are refused
(their Gauss law differs at `O((kΔ)²)` and restoring it would change the field), as are Galilean
variants, PSATD PML, antennas, and observers, and cochain boundaries, CPML, observers, harmonic
constraints, and conductive, dispersive, or plasma media (memory with no counterpart). Distributed
solvers are not converted. `examples/pic_field_handoff.py` hands a cochain plasma run to PSATD
mid-run (Gauss residual `3.4e-15` relative, energy difference `3.1e-16`).

## Electromagnetic PIC runtime

`ElectromagneticPICPlan(solver, *, species, processes, boundaries, recorders, filters, ownership,
precision)` owns:

- `species`: `PICSpeciesPlan(population, charge_model)` per species. Particles are a runtime
  `ParticlePopulationState` with persistent `(id_hi, id_lo)` identities and parent lineage;
  macrocharge is `mass × base_specific_charge × charge_number`.
- `processes`: `AbstractPICProcess` values at the `"momentum"` stage (after the push, proper
  velocity only), the `"creation"` stage (after the momentum stage and before the drift; may
  change proper velocities and create particles at step-start positions, which then drift and
  deposit in the same step — strong-field QED, see [Strong-field QED](guides_strong_field_qed.md))
  or the `"population"` stage (after the field advance, may create particles and
  change charge numbers). Creation and population processes must preserve deposited charge pointwise, which is
  verified by redeposition (`process_charge_defect`), unless they declare
  `redistributes_charge` (particle resampling): those conserve total charge, require a solver
  implementing `PICGaussProjection`, and the runtime projects the advanced field onto the
  redeposited charge (`ElectromagneticPICDiagnostics.gauss_projection`). Ported processes:
  `collisions.CoulombCollisionProcess`, `collisions.BackgroundCollisionProcess`,
  `ionization.FieldIonizationProcess`, `ionization.ImpactIonizationProcess`; resampling
  processes are `ParticleMergePlan` and `ParticleSplitPlan`; radiation reaction is
  `RadiationReactionProcess` (see below). Each reports a
  `PICProcessLedger`; stochastic processes draw stateless keys
  `derive_key(key, address(process_id), step)`, so restart stores no RNG counters. At construction
  each process's `validate_run(species, relativity)` may refuse species or pusher units it cannot
  act on.
- `boundaries`: an optional `PICOpenBoundaryPlan`. Reflected paths deposit two straight segments
  joined at the wall; absorbed charge stays immobile on the grid as `wall_charge`, and the
  absorbed macrocharge, mass, and kinetic energy are recorded per face (see
  [Open, dispersive, and magnetized PIC](#open-dispersive-and-magnetized-pic)). Periodic axes
  carry `PICBoundaryKind.PERIODIC` faces.
- `recorders` (`AbstractPICRecorder`), `filters` (`AbstractPICFieldFilter`, applied identically to
  charge, current, and the gathered field; see below), `cherenkov_guards` (`PICCherenkovGuard`),
  `external_fields`.
- `ownership`: the run's `RadiationOwnership`. Processes claiming `"resolved-field"` are refused;
  `"subgrid-reaction"` ownership requires exactly one claiming process and otherwise none. The
  field solver advances the field of the deposited current, so it claims the resolved radiation;
  `"diagnostic-only"` ownership overlaps that claim and is refused at construction. At
  runtime a claiming process reports `PICProcessLedger.radiation` (`PICProcessRadiation`): the
  energy handed to unresolved radiation, which enters `PICEnergyLedger.radiated` (the ledger
  defect is `total + radiated − previous_total`), and the scale separation between emission and
  the grid cutoff `min(π c / min Δx, π / Δt)`; a step whose emission the grid would resolve is
  rejected with `PICRejectionReason.RADIATION_OWNERSHIP`.
- `precision`: `PICPrecisionPolicy(field_dtype, particle_dtype, accumulation_dtype)`; the field
  dtype must match the solver.

One step gathers E and B at integer-time positions, pushes half-step proper velocities, runs
momentum processes, drifts, applies particle boundaries, deposits current, advances the field,
runs population processes, and commits the whole candidate only if transfer, pusher, continuity,
particle↔field charge, Gauss, magnetic, displacement, CFL, process, radiation-ownership, and
finiteness checks pass. `PICRejectionReason` flags record every failed gate. Continuity and
particle↔field charge are certified relative to the deposit's unsigned charge-rate magnitude
(`PICFieldDeposit.continuity_scale`, reported as `diagnostics.continuity_scale`) with relative
tolerance `max(continuity_tolerance, 64 ε)`, so the certificate is independent of units, grid
size, and particle count. A momentum process
declaring `requires_field_derivatives` (full Landau–Lifshitz) receives, per species, the gradients
of the gathered fields (order-one gather plus exact forward-mode derivatives of external fields)
and their time derivatives; grid-field time derivatives come from a staggered field history, the
previous accepted field (`ElectromagneticPICState.field_history`, a `PICFieldHistory`) gathered at
the current positions. Initialization declares the history equal to the initial field one step
earlier, the same static-field assumption as the backward half-push bootstrap.

`ElectromagneticPICPlan.checkpoint(state)` returns per-component restart data: field (owned by the
solver), clock, particle boundaries, each species, each recorder, and the field history when kept.
`restore` admits every component only against the same owner identity, so a plan with different
processes restarts from the same field and species, while a different field solver or species is
refused.

See [Advanced particle-grid physics](guides_advanced_particle_grid.md) for population changes,
collisions, ionization, moving windows, and semi-implicit response.

## Open, dispersive, and magnetized PIC

`CochainMaxwellPICFieldSolver` accepts linear passive `PreparedCompatibleMaxwell`
on a `StructuredCochainBridge`: bounded axes with
`MaxwellCPMLPlan` absorbers and PEC/PMC/impedance `MaxwellBoundaryPlan`s, lossy
conductors (`ConductiveMaxwellConstitutivePlan`), Lorentz–Drude electric poles and magnetic poles
(`LorentzDrudeMaxwellConstitutivePlan`, negative index when both are resonant), and the magnetized
cold plasma (`MagnetizedColdPlasmaMaxwellConstitutivePlan`). Nonlinear or active media are refused.
Particles deposit through `ChargeConservingCurrentPlan`, which clips paths at wall-bounded faces,
and the gather is its exact Galerkin transpose (Whitney forms at order one), so the gathered field
does exactly the work `⟨E, ⋆J⟩` the deposited current does on the grid.
The magnetic gather is constrained raw physical B from `magnetic_flux(state)`,
not constitutive H; this remains true for passive magnetic dispersion and μ≠1.

The electrostatic plan initializes Gauss-consistent fields from the filtered deposited charge: a
periodic boundary on periodic grids, a Dirichlet (grounded) or mixed boundary on bounded grids,
whose fixed vertices carry the induced charge `−δD`. Its permittivity must equal the medium's
instantaneous electric response (checked at construction). `CochainElectrostaticPlan` solves with
native PCG, whose stopping test is the Hodge-norm true residual the linear runtime certifies, and
its `tolerance` is relative to the assembled right-hand side (charge plus Neumann source minus the
Dirichlet lift), independent of units: non-neutral charge on bounded 3-D grids of any size, and
charge-free Dirichlet-driven solves with SI permittivities, converge to that relative residual.
Reduced 2-D fields initialize on any mix of periodic and bounded axes through
the per-axis eigenbasis of their Poisson operator.

A static field with `B = 0` around a relativistic beam is not the field of the drifting beam and
radiates a start-up pulse that contaminates radiation diagnostics.
`ElectromagneticPICPlan.initialize(..., self_fields="relativistic-per-species", drifts=None)`
instead gives every species the lab-frame field of its rest-frame Coulomb field: for drift `β_s`
along one grid axis (explicit `drifts[s]`, or the mean velocity of its active particles, read on
the host) the cochain solver solves `−∇·(ε(∇⊥ + γ_s⁻² ê∥∂∥)φ_s) = ρ_s` (the electrostatic
operator with `ε/γ_s²` on the drift edges, with a Krylov budget scaled by `γ_s`), and superposes
`E_s = −(∇⊥φ_s + γ_s⁻² ê∥∂∥φ_s)` and `B_s = d(β_s φ_s/c) = β_s × E_s/c`. Gauss's law holds to
the certified solve tolerance per species and `d(B) = 0` exactly; `initialize_relativistic_field`
reports both (`PICRelativisticFieldResult`). The grounded boundary must have zero potential values
and no Neumann source, and `magnetic` must be `None`. A zero drift is the electrostatic field.

Particle boundaries (`PICOpenBoundaryPlan`) must lie inside the field box, inset on bounded axes by
the wall distance each species' spline stencil needs (`PICOpenDomain.boundary_inset`: zero at shape
order one, `(p + 1)/2` cells at orders two and three); faces on periodic axes are `PERIODIC`.
Absorbed particles leave their charge frozen at the exit point (`wall_charge`), so Gauss's law
holds on every vertex, and each step's `ElectromagneticPICDiagnostics.exit` (`PICExitLedger`)
reports the charge, mass, and kinetic energy taken out; the energy is `(γ − 1)mc²` interpolated to
the hit fraction between the path's start and end time. `charge_ledger_defect` is the relative
balance `|Q_particles(t + Δt) + Q_exited − Q_particles(t)|`, and `medium_charge` is the largest
difference between the field's Gauss charge and the particle plus wall charge: induced wall charge,
conduction or plasma charge of the medium, and the CPML bookkeeping divergence, zero in vacuum away
from conducting walls and absorbers.

The energy ledger splits the field energy for solvers implementing `PICEnergyAccounting`:
`electric_field` is `½⟨E, ⋆ε∞E⟩`, `magnetic_field` is `½⟨H, ⋆μ∞H⟩` minus the leapfrog half-kick
term, `material` is the energy the medium stores (polarization and magnetization oscillators,
cold-plasma current), and `dissipated` is the trapezoidal integral of the source-free loss power
(conduction, pole damping, collisions, impedance boundaries, and CPML absorption). With `exited`
the kinetic energy particles carried out,
`defect = total + radiated + created_rest_energy + dissipated + exited − field_exchange −
previous_total`. Per-step ledgers pair half-step kinetic energies with integer-time fields, so their
sum carries a first-order endpoint term; `ElectromagneticPICPlan.synchronized_energy(state,
step_size)` returns a `PICEnergySnapshot` at the state's integer time, whose differences plus the
accumulated dissipated, exited, and radiated energy close at second order as `Δt` is halved at
fixed `h`, in vacuum, with CPML, and in every passive medium (the defect falls fourfold per halving
for Lorentz–Drude, negative-index, magnetized-plasma, and conducting media). This relies on the
Maxwell step advancing medium memory with the boundary-constrained fluxes the kicks see and
evaluating conduction and impedance-wall currents at the kick's midpoint field. Particles
crossing cell faces see the piecewise-constant order-one Whitney field along their motion, a
first-order work defect per crossing; shape orders two and three give continuous gathers and
keep the ledger second order for paths that cross cells.

A weakly coupled relativistic macroparticle (huge mass, immobile compensating charge at its start)
reproduces the prescribed-charge runtime `solve_prescribed_charge_maxwell` field for field, so the
PIC Cherenkov cone and spectrum in a dielectric equal the B4 reference.
`examples/dispersive_pic_cherenkov.py` runs a beam at `β = 0.9` through a Lorentz dielectric with
quadratic shapes at `Δt` and `Δt/2`: the energy ledger falls 4.45× (−3.94 then −0.886 against
13.5 dissipated), within its Richardson second-order bound, and the first-harmonic cone at `Δt/2`
is 43.26° against the dispersion audit's discrete cone 43.10° at the same `Δt` (continuum 42.50°;
the gap to the continuum is the lattice dispersion at `kh ≈ 0.7`). With order-one shapes the
beam's own grid-scale field jumps at every face it crosses, and the ledger converges only at
first order.

## Filters, cell binning, and the numerical-Cherenkov guard

`PICFilterPlan(*, passes=1, compensation=True, axes=None)` applies `passes` binomial
`(1/4, 1/2, 1/4)` passes per filtered axis and one compensation pass `(−n/4, 1 + n/2, −n/4)`, with
per-axis response `cos²ⁿ(kΔ/2)(1 + n sin²(kΔ/2))` (`cos²ⁿ(kΔ/2)` uncompensated). The same
operator acts on deposited charge, deposited current, and the field view used by the gather,
component by component through the solver's `PICTensorLayout` capability (cochain and reduced
solvers; the unstructured solver has no structured layout and is refused). On periodic axes the
stencil commutes with the discrete divergence, so the Gauss charge advanced by filtered current is
exactly the filtered deposited charge, and the symmetric stencil makes the gather filter the
transpose of the deposit filter, so filtering adds no self-force and no particle/field power
mismatch. Near nonperiodic walls each pass extends a component by mirror reflection with its
declared parity (polar normal and axial tangential components odd); the mirror commutes with the
divergence while no current crosses the wall. `ElectromagneticPICPlan.filter_reports` holds each
filter's `PICFilterContinuityReport`: the interior and wall-normal commutation defects measured
through the solver's own step and the relative Gauss residual of the field initialized from
filtered neutral charge (the Gauss-initialization hook for open PIC). A filter whose interior
commutation defect exceeds the pairing tolerance is refused; a wall-normal defect surfaces as the
runtime `particle_field_charge_defect`.

`phydrax.discretization.pic.PICCellBinningPlan(lower, upper, shape, periodic, *,
maximum_particles_per_cell, maximum_occupied_cells)` groups particles by uniform cell in canonical
`(cell, identity)` order through `phydrax.sparse.KeyGroupPlan`. The identity key is the persistent
`(id_hi, id_lo)` pair for populations (positions otherwise), so bins, per-cell membership, and every
reduction in bin order are invariant to slot order. Work is one sort over the particle capacity
plus fixed group arrays; exceeding the occupied-cell or per-cell capacity sets `overflow`, and
`outside` counts active particles beyond nonperiodic bounds.

`PICCherenkovGuard(regime, *, species)` binds a species to a
`phydrax.solver.maxwell.CherenkovRegimePlan` whose `CompatibleMaxwellDispersionAudit` is the
cochain solver's own prepared Maxwell update. Construction refuses a foreign audit, a different
speed of light, unconverged resonance roots, and any numerical-only emission (numerical Cherenkov
radiation into grid modes the medium forbids); the evidence is kept in
`ElectromagneticPICPlan.cherenkov_evidence`. Each step is rejected with
`PICRejectionReason.NUMERICAL_CHERENKOV` unless it uses the audited step size and the guarded
species stays at or below the audited drift speed. The numerical Cherenkov *instability* of
drifting plasmas belongs to the spectral-PIC NCI analysis.

## Macroparticle resampling

`ParticleMergePlan(binning, relativity, *, species, maximum_per_cell, method="vranic-momentum-cell",
momentum_bins, minimum_packet_size, maximum_packet_size, minimum_occupancy=0.0)` implements the
momentum-cell merging of Vranic et al. (Comput. Phys. Commun. 191, 65, 2015); with
`minimum_occupancy` it merges only while a species' active fraction of its capacity reaches that
value (the QED cascade merge trigger). `binning` is a `PICCellBinningPlan`; in every
spatial cell holding more than `maximum_per_cell` particles, particles of one charge state are
grouped into spherical momentum cells (magnitude between the cell's extreme speeds, polar cosine,
azimuth), each momentum cell is cut into packets of `maximum_packet_size` in global-identity order
(a trailing packet merges only with at least `minimum_packet_size ≥ 3` members), and a packet of
mass `W`, momentum `P`, and kinetic energy `K` becomes two particles of mass `W/2` at its center of
mass with

$$
\gamma_t - 1 = \frac{K}{W c^2},\quad u_t = c\sqrt{(\gamma_t-1)(\gamma_t+1)},\quad
\cos\omega = \frac{|P|}{W u_t},\quad u_\pm = u_t(\cos\omega\,\hat e_1 \pm \sin\omega\,\hat e_2),
$$

`ê₁ = P/|P|` and `ê₂` the unit component of the packet's lowest-identity momentum orthogonal to
`ê₁`. Charge, mass, momentum, energy, and the charge dipole are conserved to roundoff.

`ParticleSplitPlan(binning, relativity, *, species, minimum_per_cell, minimum_child_mass,
maximum_splits, displacement_fraction=0.25)` splits the heaviest particles of cells holding fewer
than `minimum_per_cell` particles (identity breaking ties) until the cell reaches the threshold.
Each split particle becomes `2d` children of equal mass and momentum displaced by
`±displacement_fraction` cell widths along each axis; charge, momentum, energy, and the dipole are
conserved, and splits whose children would leave a nonperiodic binning box are not made.

Both are deterministic population processes with `redistributes_charge`, so every step that runs
them Gauss-projects the field onto the redeposited charge. Products and children receive fresh
`(id_hi, id_lo)` identities in canonical event order with the lowest merged identity (merge) or
the split particle (split) as parent. Cells, packets, reductions, and identity assignment follow
cell and global identity, never storage slots, so the resampled population is identical under any
slot permutation. Events beyond the population's free slots or allocation capacity are refused and
counted rather than truncated; a failed allocation or a conservation defect above
`conservation_tolerance` leaves the species unchanged. `validate_run` refuses a relativity scale
other than the pusher's. Each application returns one `ParticleResamplingEvidence` per species
(`PICProcessResult.evidence`, surfaced as `ElectromagneticPICDiagnostics.process_evidence`):
event, removal, creation, refusal, and unsupported counts; the largest cell occupancy before and
after; relative charge, momentum, energy, and dipole defects; the Frobenius distortion of
`Σ m u⊗u` and of the central spatial second moment; and `ParticleResamplingStatus` flags.
Resampling is a discrete event and carries no derivative. `examples/pic_resampling.py` merges a
crowded electron cloud and splits a lone electron inside a reduced 1-D run.

## Radiation reaction

`phydrax.discretization.pic.RadiationReactionPlan(model, scale, physical_charge, physical_mass, *,
tables=None, maximum_chi, minimum_gamma, maximum_relative_step_loss=0.1,
minimum_scale_separation=10.0)` applies one explicit radiation-reaction step to the proper velocities
of one physical species in the units of an `ElectromagneticScaleContract`. With
`τ = q²/(6π ε₀ m c³)`, `L = E + v×B` and `Q² = L² − (v·E)²/c²`:

| `model` | Force |
|---|---|
| `"landau-lifshitz-reduced"` | `F = τ (q²/m)[L×B + (v·E)E/c²] − τ (q²/m)(γ²/c²) Q² v` (Tamburini et al. 2010) |
| `"landau-lifshitz"` | adds `τ q γ [(∂ₜ + v·∇)E + v × (∂ₜ + v·∇)B]` (Landau & Lifshitz §76) |
| `"quantum-corrected-landau-lifshitz"` | `g(χ)` times the reduced force |
| `"stochastic-fokker-planck"` | quantum-corrected drift plus diffusion of `γ` along `u` (Niel et al. 2018) |

The quantum parameter is `χ = γ √Q² / E_S` with the species critical field `E_S = m²c³/(|q|ħ)`,
and the critical angular frequency `ω_c = 3χγmc²/(2ħ)`. The Fokker–Planck model advances
`γ ← γ + A Δt + √(B Δt) ξ` with `A` the drift `dγ/dt` and
`B = γ (P_cl/mc²)(55/(16√3)) χ h(χ)`, `P_cl = (2/3) α_q m²c⁴ χ²/ħ`; `ξ` are standard normal
Wiener increments addressed by persistent identity (`wiener_increments(key, id_hi, id_lo)`), so a
particle's noise does not depend on its storage slot. `RadiationReactionTables(*, maximum_chi,
nodes_per_decade=48, quadrature_panels=64)` tabulates `g = P_quantum/P_classical` and the
normalized second photon-energy moment `h` (Niel's `h_N = (55/(16√3)) χ³ h`) from the
Baier–Katkov spectrum written with the `phydrax.special` kernels `F` and `G`, by composite
Gauss–Legendre quadrature in `ν^{1/3}` with exact `d/d log χ` slopes and cubic Hermite
interpolation; tables whose midpoint interpolation error exceeds `1e-6` are refused. Both tend to
one as `χ → 0`, where the quantum models reduce to the classical one.

One step is the first-order split `u ← u + Δt F/m` after the push, so the radiated energy is exactly
the kinetic energy removed, `m c²(γ_before − γ_after)`, and planar cooling converges at first
order in `Δt`. `RadiationReactionResult` reports per particle the updated proper velocity,
radiated energy, `χ`, `ω_c`, the separation `ω_c/ω_grid`, the Fokker–Planck `A` and `B`, and
`RadiationReactionFlag` bits: particles below `minimum_gamma` are exempt (unchanged, flagged);
`χ > maximum_chi`, a relative kinetic-energy change above `maximum_relative_step_loss` (for the
Fokker–Planck model `|A|Δt + √(BΔt)`), `γ < 1`, and nonfinite results are support failures;
emitting particles with `ω_c < minimum_scale_separation · ω_grid` are ownership conflicts.
Classical models refuse tables and quantum models require tables covering `maximum_chi`; the
full model requires field gradients and time derivatives, and only the Fokker–Planck model takes
Wiener increments.

`RadiationReactionProcess(plan, species)` is the momentum-stage PIC process. It claims
`"subgrid-reaction"` ownership, requests field derivatives for the full model, and refuses at run
construction a species whose charge number is not fixed or whose charge-to-mass ratio differs from
`physical_charge/physical_mass`, and a pusher whose units or speed of light differ from `scale`.
Each macroparticle represents `mass / physical_mass` physical particles; the ledger's
`energy_defect` is the radiated energy minus the kinetic energy removed, `radiation` carries the
radiated energy and scale separation, and the per-particle result is the process evidence. A step
with a support failure leaves the species unchanged and is rejected (`PICRejectionReason.PROCESS`);
an unseparated step is rejected with `RADIATION_OWNERSHIP`. The deterministic models are smooth
explicit updates away from the exemption and support thresholds; the stochastic model's draws
carry no derivative claim.

`phydrax.applications.detector.ChargedPropagationPlan(..., radiation_reaction=plan,
radiation_key=None)` applies the same step after each constant-field push (exactly zero field
derivatives) and reports `radiated_energy_history` and `radiation_flags_history`; see
[Detector and calorimeter production](guides_detector_calorimetry.md).

## Particle tracks and streamed radiation

`PICTrackRecorder(species, lane_species, identities, *, relativity, sample_capacity, overflow,
radiation)` is a diagnostic-only recorder that follows declared `(id_hi, id_lo)` identities, not
slots. `ElectromagneticPICPlan` calls `AbstractPICRecorder.validate_run(species, relativity)` at
construction and refuses a recorder declared for other species plans or another pusher relativity
scale. Each accepted step locates every tracked identity among the active slots of its species
(identities are sorted once on the host; each slot searches them in `O(log L)`), so a track
survives slot permutation and migration, a reused slot never continues its previous occupant's
track, identities created later by a process are inactive until they appear, and dead particles
become inactive. An identity found in more than one active slot is recorded inactive with
`duplicate` evidence.

Sample `k` is `(t_k, x^k, (u^{k−1/2} + u^{k+1/2}) / 2)`: the time-centered proper velocity makes
each sampled segment's midpoint velocity equal the leapfrog drift to second order, so tracks lag
the PIC state by one step. A particle dying during step `k → k+1` keeps `u^{k−1/2}` at its last
sample. Reduced 1-D/2-D runs resolve only the leading position axes; the other coordinates start
at zero when a lane first appears and follow the drift `u/γ`. `PICMovingWindowPlan` reports each
shift through `AbstractPICRecorder.shift_frame`; the recorder adds the accumulated offset to
window-local positions, so tracks stay in the fixed (lab) frame without artificial jumps. Each lane
keeps its first-sighting charge number and macroparticle mass; a later change ends the lane with
`property_transition` evidence. Per-lane `seen`, `first_step`, `last_step`, `activations`, parent
lineage, and per-sample slot and incarnation are reported in `PICTrackRecorderState`.

`sample_capacity` rows are kept in a ring. `to_charged_trajectory(state, scale)` returns the stored
samples in chronological order as a `ChargedTrajectory` (particle charge `sign(q/m) Z e`,
multiplicity `mass |q/m| / e`, so their product is the macrocharge; `scale` must share the run's
dimensional scale and speed of light). With `overflow="refuse"` a ring that lost samples is
refused; `"keep-latest"` returns the latest window, and `dropped_samples(state)` reports the loss.
With `radiation=PreparedTrajectoryRadiation` the recorder folds every emitted sample into the
streaming far-field accumulator inside the run, with or without stored tracks
(`sample_capacity=None`); `finalize_radiation(state)` equals the offline evaluation of the same
samples. The Hermite route needs sampled accelerations and is refused. The recorder state is one
restart component admitted by `recorder_id`.

## openPMD output and restart interchange

`phydrax.interchange.OpenPMDPICLayout(plan, scale, particle_masses)` binds a run over the periodic
3-D cochain or the reduced 1-D/2-D field solver to openPMD 1.1.0 records. The scale declares the
run's code units (its speed of light must equal the pusher's) and supplies every `unitSI`;
`particle_masses` gives one real particle's rest mass per species, so `weighting` is the
macroparticle mass over it.

```python
from phydrax.interchange import OpenPMDPICLayout, OpenPMDPICStreamWriter

layout = OpenPMDPICLayout(pic, scale, (electron_mass, ion_mass))
writer = OpenPMDPICStreamWriter(
    run_directory, layout, limits=limits, maximum_iterations=200,
    maximum_total_bytes=2**30, interval=10,
)
writer.write(state, step_size=dt)  # initial state, no J
for _ in range(steps):
    result = pic.step_detailed(state, dt)
    writer.write(result, step_size=dt)  # rejected steps publish nothing
    state = result.accepted_state
```

Each due step becomes one atomically published file `pic_<step>.h5` of the `fileBased` series
`pic_%T.h5`: `E`, `B`, `rho`, and the step's filtered `J` with the solver staggering as component
`position` (cochain: `E`/`J` on edges, `B` on faces, `rho` on vertices; reduced: cell centers), and
one species per PIC species with `id` and `parentId`. Positions, fields, and charge sit at the
iteration time; momenta and `J` carry `timeOffset = -dt/2`. `read_openpmd_pic_state` rebuilds a
state that continues the run (wall charge is `rho` minus the deposited particle charge); slot
bookkeeping, charge-transition history, boundary ledgers, and recorder states restart and are
reported as losses, so bitwise restarts remain the job of `ElectromagneticPICPlan.checkpoint`.
With `provider=OpenPMDADIOS2Provider(...)` the writer publishes ADIOS2 BP4 directories through a
pinned openPMD-api `openpmd-pipe`.

## External PIC oracles

`phydrax.solver.pic_oracle_case(plan, scale, positions, velocities, dt, *, particle_masses,
steps, output_interval)` binds one `ElectromagneticPICPlan` and exactly the arguments its
`initialize` receives to a `PICOracleCase` for a pinned external code. The scale declares the
plan's code units (its `c`, `ε₀`, and `μ₀` must match the pusher and field solver) and supplies
every SI conversion; `particle_masses` fixes the real particle charge, mass, and weight per
species. The scenario follows the field solver: `"periodic-psatd-plasma"` for
`PreparedSpectralMaxwell` and `"periodic-yee-plasma"` for the cochain Yee solver, on a fully
periodic box without processes, particle boundaries, filters, or external fields.

- WarpX (`WarpXProvider`, pinned `warpx.3d`; `PHYDRAX_WARPX`, `PHYDRAX_WARPX_VERSION`): both
  scenarios from a native `inputs` deck with explicit `MultipleParticles`; openPMD HDF5 snapshots
  are imported with `read_openpmd_meshes_hdf5`. Declared losses: no Gauss-consistent initial
  field, cell-centered output averaging, and direct deposition on collocated PSATD grids.
- Smilei (`SmileiProvider`, pinned `smilei`; `PHYDRAX_SMILEI`, `PHYDRAX_SMILEI_VERSION`): Yee
  only, shape order 2, integer charge states; its `Fields0.h5` (catalog format `smilei-fields`)
  is read by `read_smilei_fields`. Declared losses: Smilei's own field interpolator and the
  dropped periodic duplicate samples.
- PIConGPU (`PIConGPUProvider`, a pinned per-setup build driver; `PHYDRAX_PICONGPU`,
  `PHYDRAX_PICONGPU_VERSION`): Yee only; `picongpu_input` writes the setup's `.param` files,
  the driver compiles `setup/bin/picongpu`, which is pinned by digest and run. Each species must
  be a one-per-cell lattice with one drift and weight, counts multiples of `(8, 8, 4)` spanning
  three supercells.

`run_warpx`, `run_smilei`, and `run_picongpu` return a `PICOracleResult`: `E`/`B` snapshots in
plan units with their staggering and time offsets, provider version, executable and output
digests, license, and an `AdapterReport` enumerating the losses. Measured: the uniform
cold-plasma oscillation (`ω_pe = 1`, `Δt = 0.1`, 126 steps) of WarpX 26.01 matches Phydrax's
electric field energy to 1.3e-13 relative and Smilei 5.1 to 2.6e-7, both oscillating at the
leapfrog frequency 1.00067, and PIConGPU 0.8.0 reproduces the same energy per volume on its
`(24, 24, 12)` box to 1e-6 of the peak; the staggered-PSATD numerical Cherenkov growth of a γ = 10 drifting
plasma in WarpX (0.205) matches Phydrax (0.228) to 11% once both are sampled cell-centered.

Two further scenarios run on WarpX only (Smilei and PIConGPU refuse them):

- `"single-particle-radiation"`: a plan with one `PICTrackRecorder` following the only
  macroparticle of its species on the Yee cochain solver, with external fields that sum to a
  uniform, static field (probed at every cell center and particle at the start, middle, and end of
  the run). `pic_oracle_case` stores it in `PICOracleCase.track` (`PICOracleTrack`); the WarpX
  deck applies it as constant particle fields and writes the tracked species as a group-based
  openPMD series, which `run_warpx_track`/`read_pic_oracle_track` import through the openPMD
  particle-track reader into a `ChargedTrajectory` under the recorder identity
  (`PICOracleTrackResult`). Smilei's track output (openPMD 1.0.0 without ED-PIC weighting
  metadata) is not readable by that reader; PIConGPU loads species only as lattices. Measured
  (four cyclotron periods, A1 segment-exact spectra): WarpX vs the Phydrax recorder differ by
  1.8e-3/2.7e-3 (fundamental/second harmonic) at 192 steps and 4.5e-4/6.7e-4 at 384 steps,
  below the half-push-vs-mean velocity model `m(ΩΔt)²/8`; both are 5.0e-3/1.1e-2 from the exact
  helix at 384 steps (Boris phase lag).
- `"laser-wakefield-stage"`: `pic_oracle_wakefield_case(plan, scale, positions, velocities,
  step, laser=PICOracleLaser(...), ...)` binds a quasi-cylindrical P3 plan (standard PSATD,
  spectral charge conservation, no absorber, antennas, or observers) and the focused Gaussian
  pulse it adds to its initial field into a `PICOracleWakefieldCase`. `run_warpx_wakefield`
  (pinned `warpx.rz`; `PHYDRAX_WARPX_RZ`, `PHYDRAX_WARPX_VERSION`) runs RZ PSATD with the
  explicit particles and a Gaussian antenna at the pulse center (`warpx_preroll_steps` steps
  of pre-roll), and `read_warpx_wakefield` imports the `thetaMode` `E`/`B` snapshots onto the
  case's nodes (`PICOracleWakefieldResult`). Declared losses: antenna injection and its
  backward pulse (damped/absorbing, not periodic, axial boundaries), local FFTs with
  `psatd.noz = 32`, direct deposition with the rho update (WarpX's current correction needs a
  global FFT RZ lacks), cell-centered output. Measured on the FBPIC test's linear LWFA: the
  on-axis wake differs from P3 by 13.1%, 9.8%, and 8.1% (relative L²) at Δt, Δt/2, and Δt/4
  (WarpX step error of order ≈ 0.7; P3 is step-converged to 0.7% and agrees with FBPIC 0.27.0
  to 5.6%).

## Distributed PIC and restart

`distribute_pic_field_solver(base, mesh, guard_cells=3, particle_margin=1.0, identity_tiles=None)`
runs the periodic 3-D cochain, reduced 1-D/2-D (periodic or bounded axes), or Cartesian PSATD
solver on a device mesh of one to three axes. Mesh axis `a` splits grid axis `a` into equal blocks
of whole cells (`PICDomainDecomposition`): a one-axis mesh gives slabs, a two-axis mesh pencils, a
three-axis mesh blocks (every mesh axis holds at least two devices). Device `(p₀, …)` with
row-major linear index `p` owns slot block `[p C/P, (p+1) C/P)` of
every species, so particle ownership is aligned with the field decomposition. The result is an
`AbstractDistributedPICFieldSolver`, an ordinary `AbstractPreparedPICFieldSolver`, so
`ElectromagneticPICPlan` runs over it unchanged:

- each device deposits its own particles on its block window (owned cells plus `guard_cells`
  cells on each side of every decomposed axis); one plane-level
  `phydrax.discretization.DistributedHaloPlan` per decomposed axis adds guard contributions to
  their owners axis by axis. A component uniform along the decomposed axes — the mean current of
  periodic continuity-projected reduced grids — is summed over devices instead;
- gathers read the owned cells plus guards exchanged axis by axis, each exchange carrying the
  previous axes' guards, so edge and corner guards of diagonal neighbors arrive without extra
  messages; particles more than `particle_margin` cells outside their block are refused;
- cochain, reduced, and global-FFT PSATD updates are the base solver's own update,
  SPMD-partitioned over the same mesh; PSATD global FFTs must run through a
  `DistributedSpectralExecutionPlan` built on the same mesh (slab schedule on one axis, pencil on
  two; three-axis meshes are refused for PSATD). Reduced and spectral field arrays are sharded over
  the blocks; packed cochains are replicated;
- local-guarded PSATD (Kirchen et al., Phys. Rev. E 102, 063215 (2020)) runs per device: the owned
  fields, current, and charge receive the plan's `guard_cells` cells exchanged through the same
  halo substrate on every decomposed axis, and each of the device's local-guarded subdomains
  (which must tile the device block: the mesh part counts divide `subdomains`) is transformed,
  advanced with the finite-order stencil, and inverted locally; only the Gauss residual maxima and
  energies are reduced over the mesh. Vay deposition or update-with-ρ conserve charge (the
  spectral correction needs the global spectrum). A step equals the single-device local-guarded
  step to reduction order and the global-FFT step up to the stencil truncation reported by
  `PreparedSpectralMaxwell.guard_truncation(dt)`, the relative real-space kernel mass of one
  vacuum step beyond the guards.

Preparation refuses a guard narrower than the transfer footprint: probe paths leaving every
interior block face by `particle_margin` cells must deposit and gather inside the window. A solver
whose deposit is not window-local is refused by the same probe: collocated PSATD with even extents
projects checkerboard charge over the whole grid, and the continuity-projected reduced 2-D current
spreads over the whole grid, so a reduced 2-D run decomposes only when the guard windows cover the
grid (reduced 1-D currents are window-local up to their mean current). The distributed solver shares its base
solver's identity — decomposition changes reduction order, not the discretization — and reports
execution in `PICDistributedEvidence` (`distributed=True`, mesh shape, local-guarded guards; for
cochain Maxwell the capability set with `distributed` and `spatial_distribution` set).

`pic_distribution_support(base)` returns the `PICDistributionSupport` (`route` or `None`, and the
`basis`) that `distribute_pic_field_solver` enforces: quasi-cylindrical PSATD is refused (its
radial Hankel transforms couple every radius of every mode), unstructured Whitney PIC is refused
(tetrahedral cochains have no block decomposition or halo plan), and a cochain grid with a bounded
axis is refused (the wall vertex plane does not tile the cell blocks). The route-specific
distributed solver publishes exactly the protocols whose distributed route it executes
(`pic_capabilities`, configuration `"distributed-<base>"`) and states every base protocol it
withholds; a published distributed protocol is admitted exactly when the base admits it:

| Capability | distributed cochain 3-D | distributed reduced 1-D/2-D | distributed PSATD (global-FFT, local-guarded) |
|---|---|---|---|
| `tensor-layout` | forwarded: filters act on the global arrays of the partitioned update | forwarded | — (base unpublished) |
| `spectral-symbol` | forwarded (the distributed update is the base update) | forwarded | forwarded |
| `huygens-sampling` | withheld: the cochain base never carries Huygens boxes beside the PIC current | — | forwarded: observers accumulate from the global fields when each step completes |
| `multi-deposit` | wrapper route: every species' block-window deposit summed before one halo accumulation | same | same |
| `window-shift` | withheld: `PICMovingWindowPlan` shifts particles without migrating them to their new owners | withheld (same) | — (base unpublished) |
| `galilean-grid` | — | — | forwarded: particles drift and migrate in grid coordinates |
| `energy-accounting` | forwarded (replicated cochains) | — | — |
| `open-domain` | forwarded (every distributed cochain axis is periodic) | — | — |
| `restart-state` | forwarded; components restart across topologies | forwarded | forwarded |
| `gauss-projection` | forwarded on the halo-accumulated charge | forwarded | forwarded (FFTs on the PIC mesh) |
| `relativistic-self-fields` | withheld: needs a grounded boundary on bounded axes, and distributed cochain axes are periodic | — | — |

Each published route is exercised on four forced host devices against the single-device base or an
analytic reference in `tests/unit/solver/test_distributed_pic_capabilities.py` (functional
evidence, not hardware scaling).

```python
from jax.sharding import Mesh

mesh = Mesh(np.asarray(jax.devices()[:4], dtype=object).reshape(2, 2), ("x", "y"))
solver = phx.solver.distribute_pic_field_solver(base_solver, mesh)
pic = phx.solver.ElectromagneticPICPlan(solver, species=species, processes=processes)
run = phx.solver.DistributedElectromagneticPICPlan(pic, packet_capacity=64, reach=1)
state = run.initialize(positions, velocities, dt, active_masks=masks, masses=masses)
step = eqx.filter_jit(lambda value: run.step_detailed(value, dt))
result = step(state)  # result.migration, result.rejection_reason
```

`initialize` places every particle in its owner's slot block while persistent identities follow
the caller's slot order, so identity-addressed randomness and diagnostics do not depend on the
decomposition. `PICMigrationPlan` routes particles whose owner changed through fixed-capacity
`lax.ppermute` packets, one per mesh offset in `{−reach, …, reach}^k` (diagonal neighbors
included): the source slot is deactivated (it keeps its occupant's identity) and a free destination
slot is allocated with the particle's identity, lineage, charge state, and slot-aligned process
state. A packet overflow, an owner beyond `reach`, or a destination without free slots on any
device rejects the whole step: the accepted state is the step-start state and
`PICRejectionReason.MIGRATION` is set.

`DistributedElectromagneticPICPlan` installs a `DistributedPICExecutor` in the PIC plan.
Creation-stage, population-stage, stateful, and charge-redistributing processes must implement
`PICDistributedProcess` (the QED cascade with polarization, merging and splitting, field and impact
ionization do); they run per device on the device's slot blocks, their slot-aligned state (QED
lepton optical depths and spins) moves with the particles, and process-owned banks (the QED photon
bank, whose capacity must equal the gather species' capacity) are decomposed and migrated by their
own positions. Additive process totals (escape histograms) are accumulated per device and summed;
ledgers and evidence are combined by the process. The step order is gather → push → momentum and
creation processes → drift, deposit, field advance → migration → population processes → migration,
so resampling groups and ionization act on particles held by their owners, and every step ends
with every particle on its owner. Merge and split binning cells must nest in the device blocks.

Created particles are allocated inside the creating device's slot block by `PICIdentityAllocator`,
whose identities do not depend on the decomposition and need no communication: every allocation
call reserves `B·R` identities after the population counter (`B` identity tiles — `identity_tiles`,
by default one per decomposed cell, a partition nested in every admissible block — and `R` the
population's global capacity); the particle created by an event in tile `t` with rank `r` among the
call's events in `t` (in the process's canonical, identity-keyed event order) receives
`counter + t R + r`. An event's tile is that of its creating event's position (emitter, decaying
photon, merged packet, split particle, ionized ion), which always lies in the creating device's
block, so each tile's identities come from exactly one device, and a run on `N` devices creates the
same identities, lineage, and counts as on one device. Single-device `ElectromagneticPICPlan` runs
keep the consecutive per-event F6 counter.

`PICRestartPlan(run, window=None, auxiliaries=())` composes the run's per-component restart leaves
— species with identities and lineage, the field state with its ADE/plasma/CPML/PML memory, the
boundary ledger, recorders, process states (radiation accumulators, QED photon and pair banks),
the field history, the moving-window epoch, and auxiliary `PICRestartState` owners such as
boosted-frame buffers — into a `PICRestartManifest`, publishes this process's addressable shards
through `phydrax.lifecycle` checkpoints, and restores them:

```python
plan = phx.solver.PICRestartPlan(run)
plan.publish(repository, state, checkpoint_id="pic-0100", writer_id="rank-0")
checkpoint = plan.assemble(repository, "pic-0100", expected_process_count=1)
restored = phx.solver.PICRestartPlan(other_run).restore(repository, checkpoint)
restored.state, restored.restart_class  # "bitwise" or "tolerance"
```

A restart on the topology that wrote the checkpoint continues bitwise. The decomposition is
static during a run and changes only at restart: a different mesh (device count, or slabs versus
blocks) repartitions particles, with their slot-aligned process state, into the new slot blocks
and process banks by their own positions (slot permutations, so the restored state is exact) and
is a
`"tolerance"` restart within `repartition_tolerance`, because the continued run sums deposits in a
different order. The lifecycle `TopologyRestartPolicy` admits or refuses the relation; the default
admits tolerance restarts. Components are admitted only by owners with the same identity.

## Differentiation and limits

Weights and payloads differentiate inside a fixed route and segment program. Cell crossings,
periodic-image selection, segment count, support changes, solver failure, and step acceptance are
stopped branch decisions. No derivative is claimed through particle creation/deletion, collisions,
ionization, moving windows, repartitioning, or adaptive topology.

Support is configuration-specific rather than inherited across those plans. Quasi-cylindrical
PSATD, unstructured Whitney PIC, and bounded cochain grids are not distributed, distributed runs
refuse moving windows and relativistic self-field initialization, the decomposition is static
between restarts (no dynamic load balancing or ownership migration during a run), and each advanced
configuration requires its own conservation, capacity, solver, and differentiation evidence.
