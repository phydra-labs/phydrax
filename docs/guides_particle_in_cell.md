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
lowest-order Whitney transfer: multilinear charge and multilinear interpolation of each edge/face
component. Order `p = 2, 3` deposits charge with the degree-`p` cardinal B-spline and gathers
each oriented component with degree `p − 1` along the axes the entity spans and degree `p` across
the others (`TensorBSplineSplatAssignment` with per-axis degrees). Because
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

`ChargeConservingCurrentPlan` currently supports uniform periodic 3-D grids and trajectories that
cross at most one cell per axis in one step, at the transfer's shape order. Order one splits a
straight trajectory at crossed faces and integrates cubical Whitney edge forms in closed form.
Orders two and three split the path at the common knot lattice of the spline-Whitney forms
(half-integers for `p = 2`, integers for `p = 3`) and integrate the path integrals
`q/Δt ∫ N^{p−1}(q_a − e − ½) N^p(q_b − j) N^p(q_c − k) dq_a`, polynomials of degree `3p − 1`
per segment, exactly with `⌈3p/2⌉`-point Gauss–Legendre; their contributions are reduced by
`phydrax.sparse` in canonical cell-binned particle order, so the current is bitwise invariant to
particle slot order. Every order satisfies

```text
(rho_new - rho_old) / dt + delta(J_mid) = 0
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
current, and the resulting Gauss charge must equal the deposited end charge
(`ElectromagneticPICPlan.pairing_defect`). A solver whose Gauss charge does not follow its own
deposited current is refused at construction.

| Solver | Field | Transfer | Optional capabilities |
|---|---|---|---|
| `CochainMaxwellPICFieldSolver` | periodic 3-D `PreparedCompatibleMaxwell` | cochain splats + `ChargeConservingCurrentPlan` per species | spectral symbol, Huygens sampling, Gauss projection (cochain Poisson), window shift, restart |
| `ReducedMaxwellPICFieldSolver` | `CompatibleMaxwell1DPlan`/`CompatibleMaxwell2DPlan` | `ReducedPICTransferPlan` | spectral symbol, multi-deposit, Gauss projection (1-D cochain, periodic 2-D spectral Poisson), window shift, restart |
| `UnstructuredMaxwellPICFieldSolver` | tetrahedral `PreparedUnstructuredMaxwell` | `UnstructuredWhitneyCurrentPlan` | Gauss projection (cochain Poisson), restart |

Optional capabilities are structural protocols: `PICSpectralSymbol` (vacuum numerical dispersion
`ω(k)`), `PICHuygensSampling` (phasors of the solver's Huygens observers), `PICMultiDeposit` (one
fused deposit of every species), `PICWindowShift` (integer-cell translation, consumed by
`PICMovingWindowPlan`), `PICGaussProjection` (curl-free Poisson projection of the field onto a
prescribed Gauss charge, reporting `divergence_before`/`divergence_after`, the added field energy,
and its `"cochain-poisson"` or `"spectral-poisson"` route), and `PICRestartState`. Prescribed
fields enter through
`phydrax.discretization.pic.ExternalFieldSource` and are added to every gather. The semi-implicit
ECSIM runtime `SemiImplicitPICPlan` solves particles and field jointly and remains a separate
orchestrator.

The Whitney transfer returns nodal charge content and integrated edge flow; the unstructured
solver maps them through the inverse degree-0/degree-1 Hodge stars onto Maxwell's charge density
and current on Maxwell's edge order. Boundary vertices are absolutely constrained, so charge lives
on interior vertices. The reduced 1-D Gauss divergence gives the nonperiodic lower wall zero flux,
matching the transfer's continuity operator; reduced 2-D fields with a nonperiodic axis do not pair
their Gauss charge with the deposited current and are refused.

## Electromagnetic PIC runtime

`ElectromagneticPICPlan(solver, *, species, processes, boundaries, recorders, filters, ownership,
precision)` owns:

- `species`: `PICSpeciesPlan(population, charge_model)` per species. Particles are a runtime
  `ParticlePopulationState` with persistent `(id_hi, id_lo)` identities and parent lineage;
  macrocharge is `mass × base_specific_charge × charge_number`.
- `processes`: `AbstractPICProcess` values at the `"momentum"` stage (after the push, proper
  velocity only) or the `"population"` stage (after the field advance, may create particles and
  change charge numbers). Population processes must preserve deposited charge pointwise, which is
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
  joined at the wall; absorbed charge stays immobile on the grid as `wall_charge`.
- `recorders` (`AbstractPICRecorder`), `filters` (`AbstractPICFieldFilter`, applied identically to
  charge, current, and the gathered field; see below), `cherenkov_guards` (`PICCherenkovGuard`),
  `external_fields`.
- `ownership`: the run's `RadiationOwnership`. Processes claiming `"resolved-field"` are refused;
  `"subgrid-reaction"` ownership requires exactly one claiming process and otherwise none. At
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
finiteness checks pass. `PICRejectionReason` flags record every failed gate. A momentum process
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

The full 3-D cochain solver requires every axis periodic, and `PreparedMaxwellCPML` rejects
nonzero CPML width on a periodic axis, so it cannot carry an absorbing CPML layer; reduced 1-D
fields are the PIC route that accepts `MaxwellCPMLPlan`. See
[Advanced particle-grid physics](guides_advanced_particle_grid.md) for population changes,
collisions, ionization, moving windows, and semi-implicit response.

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
momentum_bins, minimum_packet_size, maximum_packet_size)` implements the momentum-cell merging of
Vranic et al. (Comput. Phys. Commun. 191, 65, 2015). `binning` is a `PICCellBinningPlan`; in every
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

## Differentiation and limits

Weights and payloads differentiate inside a fixed route and segment program. Cell crossings,
periodic-image selection, segment count, support changes, solver failure, and step acceptance are
stopped branch decisions. No derivative is claimed through particle creation/deletion, collisions,
ionization, moving windows, repartitioning, or adaptive topology.

Support is configuration-specific rather than inherited across those plans. Quasi-cylindrical
PSATD and cross-device particle sharding remain unsupported; each advanced configuration requires
its own conservation, capacity, solver, and differentiation evidence.
