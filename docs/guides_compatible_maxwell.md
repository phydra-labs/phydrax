# Compatible time-domain Maxwell

`CompatibleMaxwellPlan` evolves conservative electric displacement, magnetic flux, and
charge cochains on one exact complex. Electric and magnetic fields are constitutive
outputs; material, CPML, source, boundary, and observer state remain auxiliary.

## Cochain roles

The plan owns one explicit role layout.

- Full three-dimensional Maxwell stores `D`, `E`, and electric current on degree one,
  `B` and `H` on degree two, and charge on degree zero.
- TEz on a genuine two-dimensional bridge retains in-plane electric degree-one fields,
  an out-of-plane magnetic degree-two field, and degree-zero charge. Magnetic divergence
  is absent because there is no following degree.
- TMz retains out-of-plane electric degree-zero fields and in-plane magnetic degree-one
  fields. It has no retained charge degree; magnetic closedness is nontrivial.

A one-cell-thick three-dimensional grid is neither the implementation nor the
qualification oracle for a reduced model. Material tensors that couple retained and
suppressed components fail during preparation.

## Magnetic closedness

Pure Faraday forcing preserves the magnetic Gauss law because the next exterior
derivative composed with the electric derivative is zero. Every non-curl magnetic
forcing carries magnetic charge: an equivalent (Huygens, antenna) sheet has physical
surface divergence, the jump of the normal `B` across it, and CPML coordinate
stretching and magnetic conductivity act as effective magnetic currents inside their
supports. The runtime therefore tracks declared magnetic charge `q_m` in
`MaxwellAuxiliaryState.magnetic_charge`, advanced with the same half kicks as `B`
(`∂q_m/∂t = d(Ḃ − curl E)`, the divergence of exactly the non-curl forcing `B`
receives), and the constraint is `d(B) = q_m`; `magnetic_constraint(state)` and the
diagnostics report the defect `d(B) − q_m`. Only PMC masking, which overwrites `B`,
breaks that bookkeeping.

The automatic constraint policy elides projection when no PMC boundary is present
and the initial state matches its declared charge. Otherwise Phydrax computes the
Euclidean minimum-norm correction onto
`d(B) = q_m` through a resource-bounded sparse native solve, restores the declared
harmonic periods, and reports original residual and solver work.
No production path materializes the incidence matrix densely.

## Sources and ports

Sources are prepared plans. Static support and spatial profiles are lowered once;
waveform amplitudes and controls remain dynamic. Electric current enters the full D
step at the midpoint. Magnetic current enters the two B half steps at their respective
times. Charge uses the complete electric forcing, so the same discrete continuity law
is audited.

A source `envelope(time, args)` replaces the harmonic temporal factor and is part of
the static source identity. StrictModule envelopes and module-level plain functions
are content-addressed. Lambdas, closures, nested functions, methods, partials, and
other callable objects are opaque: pass both `envelope_semantic_id` and
`envelope_numeric_id`, or construction raises `TypeError`. Both identities enter the
source identity; only the semantic identity enters the refresh signature.

A one-way mode or Huygens launch uses paired electric and magnetic trace forcing from
one oriented surface. Production mode ports initially require propagating, lossless,
nondegenerate modes with finite nonzero signed surface power. The same mode basis and
reference-plane identity drive launch, DFT observation, decomposition, power, and the
circuit scattering adapter.

## One-way plane antennas

`SampledPlaneCurrentAntennaPlan(bridge, normal_axis, plane_coordinate,
first_coordinates, second_coordinates, times, electric, *, magnetic=None,
carrier_angular_frequency=..., direction=..., medium=None, beta=0.0, scale=None)`
launches a prescribed forward wave from a sheet normal to one uniformly spaced axis
(nonperiodic and `full_3d` for the cochain runtime). `electric[b, c, t, 2]` holds the
complex tangential envelope `(E_b, E_c)` of the wave at the sheet on a tensor of sample
coordinates along the two remaining axes in increasing order; the physical field is
`Re[A(τ) e^{−iω₀τ}]` and the sampled window is the aperture (zero outside). `magnetic`
defaults to the plane-wave relation `H' = s â × E'/η` of `medium`. The same plan drives
Cartesian PSATD through `SpectralMaxwellPlan(antennas=...)` on a periodic grid (see the
spectral PIC guide).

With emission sign `s` the equivalence principle gives `K = s â × H'` and
`K_m = −s â × E'`, radiating the wave ahead of the sheet and nothing behind it. On the
cochain lattice the electric sheet lies on node planes (`j = K ℓ/h`) and the magnetic
sheet half a cell upstream on interval centers (`m = K_m ℓ`); each is driven by the
incident field at the other sheet, the total-field/scattered-field pairing, so the
backward leakage of a static sheet is the lattice/continuum plane-wave mismatch
(energy ratio ~10⁻⁷ at 20 cells per wavelength). Envelopes are interpolated with
cubic Hermite segments in time and bilinearly onto the staggered sheet points.

`beta` moves the sheet along its normal in the simulation frame, as for a
laboratory-frame laser antenna in a boosted-frame run. Surface currents are
four-vector densities, so the sheet deposits `K'(τ)/γ` (`phydrax.boost_event`). The
sheet is spread over its three nearest node planes with quadratic B-spline weights, and
each plane carries one exact lattice pair: `J` on node `k` driven by `H'` at the
upstream interval center, `M` on that interval by `E'` on node `k`, each at the
rest-frame retarded time `τ = t' − s z'/c` of its own event. A static sheet is a
constant superposition of exact one-way pairs; a moving sheet's backward leakage
converges at second order in the cell size. The launched wave is Doppler-shifted by
`γ(1 + sβ)`. A moving antenna requires `scale`, whose vacuum must be the antenna
medium.

Prepared antennas report `SampledPlaneAntennaEvidence`: aperture support, emitted
carrier and cells per wavelength, Lorentz factor, active window, touched planes, and
the relative discrete divergence of the magnetic sheet, which is nonzero exactly when
the launched field has a normal `B` (fully three-dimensional beams). That divergence is
declared magnetic charge, so antennas never trigger the global magnetic projection.
Runtime preparation refuses sheets whose entities do not carry the
declared lossless homogeneous medium or intersect CPML, and Huygens boxes refuse
antennas whose support reaches their surface. `MaxwellAntennaWorkObserverPlan(antenna)`
streams the trapezoidal work `−∫(E·J + H·M) dV dt` the sheets do on the field; for a
pulse emitted into a closed lossless run it equals the field energy.

`phydrax.optics.wave.pulse_envelope_antenna` lowers a tangential `PulseEnvelopeField`
on a grid-aligned plane (normal = emission direction) to an antenna;
`sample_focused_gaussian_pulse_envelope` samples a paraxial Gaussian a distance
upstream of its waist, and `openpmd_laser_envelope_antenna` binds the antenna identity
to an imported openPMD LaserEnvelope record. `examples/maxwell_antenna_gaussian_beam.py`
launches a focused beam and compares its waist and Gouy phase with paraxial theory.

## Spectral acquisition, Huygens surfaces, and far fields

Every spectral observer takes an explicit `MaxwellSpectralAcquisition(angular_frequencies,
sign=..., measure=..., start_time=..., stop_time=...)`. `sign="negative"` accumulates
`exp(-iωt) f(t)` and `"positive"` accumulates `exp(+iωt) f(t)`. The `"sample-mean"`
measure divides the windowed sum by the number of samples inside the window (the
normalization `DFTObserverPlan` used before the acquisition became explicit); the
`"time-integral"` measure applies the trapezoid rule to consecutive samples that both
lie in the window, carrying the previous payload and time in `DFTObserverState`.

Phydrax phasors follow `exp(-iωt)`, so the transient spectrum of a field is
`F(ω) = ∫ f(t) e^{+iωt} dt`: Huygens samplers require the time-integral measure with the
positive exponent. `MaxwellHuygensBoxPlan(bridge, lower_nodes, upper_nodes, acquisition,
exterior)` samples a closed axis-aligned box on node planes of a `full_3d` bridge at
surface-cell centers. Tangential `E` is the mean of the two bounding edge circulations
per unit length, and tangential `H` the mean of the four face fluxes that straddle the
surface plane, per unit area; both are gathered through prepared
`phydrax.sparse.SparseLinearMap` operators. On tetrahedral meshes
`MaxwellHuygensSurfacePlan(hodge, faces, acquisition, exterior)` takes a closed,
consistently oriented set of interior faces and reconstructs the tangential fields with
Whitney edge and face elements averaged over the two adjacent cells.

`MaxwellFarFieldPlan(directions, reference_axis, exterior)` forms the equivalent
currents `J = n̂ × H̃` and `M = −n̂ × Ẽ`, their radiation vectors `N` and `L`, and

```text
F_θ = (ik/4π)(L_φ + η N_θ),   F_φ = −(ik/4π)(L_θ − η N_φ),   k = ω√(εμ),  η = √(μ/ε),
```

with `Ẽ(r r̂) ≈ e^{ikr} F(r̂)/r`. The result carries `field_spectrum[F, D, 2]` on
`(θ̂, φ̂)` with `φ̂ = â × r̂/|â × r̂|`, the coherency `F_i F_j*`, Stokes parameters with
`U = 2 Re(F_θ F_φ*)` and `V = −2 Im(F_θ F_φ*)`, and the one-sided spectral energy
`d²W/(dω dΩ) = εc|F|²/π`. `spectral_poynting_energy` integrates
`(1/π) Re ∫ (Ẽ × H̃*)·n̂ dS` over all or selected surface cells; over a closed surface
it equals the far-field energy integrated over the sphere. The Hertzian-dipole test
fixes every sign against `−iωμ m sin θ e^{-ik r̂·r₀}/(4π)`.

The equivalence theorem only holds when the surface lies in the declared homogeneous,
lossless exterior with no sources on it and no absorber. Preparation refuses CPML terms
on surface entities, any source whose static support reaches the surface, dynamic PIC
currents, and every constitutive law other than a diagonal one or a conductive one with
zero conductivity whose surface values equal the declared exterior. The tetrahedral
sampler checks the runtime's constitutive factors composed with the Hodge factors, and
its `update(..., electric_current=...)` raises when a supplied current is nonzero on the
surface inside the acquisition window. `examples/maxwell_dipole_far_field.py` runs the
dipole end to end and `benchmarks/maxwell_far_field.py` records the phase costs.

## CPML

CPML retains one memory for each active directional derivative and cochain support.
Only exact lower/upper boundary slabs are stored; corner degrees of freedom intentionally
appear in several directional terms. The ordinary curl is evaluated on the logical
interior, and packed corrections are scattered back in a deterministic order.

The graded profile is `σ = σ_max d^m`, `κ = 1 + (κ_max − 1) d^m`,
`α = α_max (1 − d)` in the normalized depth `d`, with
`σ_max = (m + 1) c ln(1/R) / (2 L)` for the physical layer thickness `L` and the
runtime's material wave-speed bound `c`. That is the design for a continuum
normal-incidence round-trip reflection of `R` (`target_reflection`).
`MaxwellCPMLPlan.prepare(bridge, layout, wave_speed)` takes `c` explicitly.

Fixed runs prepare distinct recurrence coefficients for electric full steps and magnetic
half steps. Changing the time step requires explicit refresh or preparation; stale
coefficients cannot be reused. Variable public stepping computes the same recurrence on
packed terms.

## Dispersive and magnetized media

`LorentzDrudeMaxwellConstitutivePlan(electric_poles, magnetic_poles=...)` evolves
`D = ε∞E + ΣP`, `P̈ + γṖ + ω₀²P = f E`, on electric cochains. It also evolves the
magnetization `B = μ∞H + ΣM`, `M̈ + γṀ + ω₀²M = f H`, on magnetic cochains. Each
`MaxwellLorentzPoles` strength is `(poles,)` or spatial `(poles, entities)`. A zero
strength removes the pole from an entity: its state stays exactly zero, and its
energy and dissipation are masked. `drude_maxwell_constitutive` builds `ω₀ = 0` poles
with `f = ωₚ²` and accepts spatial plasma frequencies. Each oscillator half step is a
symmetric kick-drift-kick with Crank–Nicolson damping, driven by the self-consistent
field at the held primary flux. The composed step is second order in `Δt`. `continuum_relative_permittivity(ω)` and
`continuum_relative_permeability(ω)` return the continuous pole sums.

`MagnetizedColdPlasmaMaxwellConstitutivePlan(plasma_frequency, cyclotron_frequency)`
evolves one current per species, `J̇ = ε₀ωₚ²E + J × ω_c − νJ`, with the signed vector
`ω_c = qB₀/m`. It requires `full_3d`, and the plasma frequency can be spatial per
vertex. Currents are Cartesian vectors at vertices. Vertex fields and edge currents
form a Hodge-adjoint averaging pair, so the plasma energy `|J|²/(2ε₀ωₚ²)` exchanges
exactly with the field energy. Each half step integrates the linear current ODE
exactly for the field held at that half step's endpoint. This is an exponential
rotation, not a Cayley rotation, and the pair of half steps couples to `E`
symmetrically about the step midpoint. The gyration phase is therefore exact for any
`|ω_c|Δt`. For both laws, `energy_rate` is the rate of the complete stored energy, so
`power_balance_residual` closes with collisional and pole dissipation.

## Frequency response and stretched coordinates

Every constitutive law implements `frequency_response(ω)` (`exp(-iωt)`). It returns
`ε(ω)` with conduction folded in as `iσ/ω`, and `μ(ω)` with magnetic conduction folded
in as `iσₘ/ω`. The response is diagonal for instantaneous, conductive, gain, and
Lorentz–Drude laws; the matrix law uses its own maps. The magnetized plasma uses the
per-vertex cold-plasma conductivity `ε₀ωₚ²((ν − iω)I + |ω_c| n̂×)⁻¹`. Nonlinear laws
refuse. `FrequencyMaxwellOperator(bridge, layout, constitutive, ω, stretching=...)`
applies `curl_s μ(ω)⁻¹ curl_s − ω²ε(ω)`. Here `curl_s` scales each directional
derivative by `1/s`, with `s = κ + σ/(α − iω)` on the same graded profile that
`MaxwellCPMLPlan` compiles for the time domain.

`power_ledger(E, source)` reports the impressed source power `−½Re⟨E, J⟩`, the
electric and magnetic material losses `½ω Im⟨E, εE⟩` and `½ω Im⟨H, μH⟩`, and the work
of the equivalent stretched-coordinate currents, plus the impedance-boundary loss
`½Re⟨E, YE⟩`. On a closed domain the ledger closes to the solve residual. The
Hermitian eigen path is refused when stretching is active, when the response is
dispersive or lossy, or when perfect-conductor or impedance boundaries are present.

`boundaries=` takes the time-domain `MaxwellBoundaryPlan` vocabulary: the domain trace
or an explicit `support` mask (interior plates, gratings, thick conductors).
Perfect-conductor entries become identity rows (`E` equals the right-hand side
there), perfect-magnetic-conductor entries zero `H`, and impedance entries add the
surface conduction current `YE`. `solve(source, method="krylov")` runs restarted
GMRES on the matrix-free operator; `method="direct"` assembles the exact sparse
operator by structural coloring of the traced curl-curl pattern (`phydrax.sparse`
structure detection), reuses that coloring across frequencies, and factors it with
native sparse LU, whose resource estimate is the symbolic fill bound of its owned
column ordering.

## Moving charges in the frequency domain

A charge in uniform motion has the transformed current `J̃ = q d̂ δ_⊥ exp(iωs/v)`
(`s` the path coordinate from the position at `t = 0`). `MaxwellMovingChargePlan`
integrates it exactly on the Whitney edge forms: on every crossed cell the Whitney
factor `W_e·d̂` is a polynomial of degree `dimension − 1` in the path parameter, so each
segment contributes closed-form moments `∫₀¹ τᵐ exp(iθτ) dτ`. The Whitney node load of
`ρ̃ = (q/v) δ_⊥ exp(iωs/v)` satisfies `d₀ᵀ b + iωb₀ = 0` to roundoff away from path
ends. Full 3-D layouts carry a point charge; `tez` layouts a line charge per unit `z`
length. A path parallel to a periodic axis of length `L` is one closed pass and needs
`ωL/v ∈ 2πℤ`; it then equals the transform of a single charge on an infinite line, so
a commensurate periodic cell is the natural Cherenkov and Smith–Purcell domain. Any
other path is clipped to the box; its ends are reported as open endpoints where charge
appears or vanishes (harmless on a conductor or deep in an absorber).

`FrequencyMovingChargePlan(source, constitutive, formulation=...)`:

- `"total-field"` solves `A E = iωJ̃`. `(2/π)·ledger.source_power` is the one-sided
  energy per unit frequency the field extracts from the charge (Frank–Tamm per unit
  length in a periodic cell).
- `"scattered-field"` requires a prepared homogeneous isotropic `background`. The
  incident field is the analytic `phydrax.electromagnetics.UniformMotionFieldPlan`
  field in that background, integrated on edges; free rows carry `A_b E_inc − A P E_inc`
  and conductor surfaces `E_s = −E_inc`, so only material contrast and conductor
  surfaces radiate and thick conductors may enclose the path. Sources must stay at
  least a quarter edge from the path. `(2/π)·ledger.absorbed_power` is the radiated
  (for example transition) energy per unit frequency.

`FrequencyMovingChargeEvidence` reports the branch (`radiating`, `k_ρ`), the bound-field
reach `γβλ = 2π/Im k_ρ` against the transverse clearance to the absorbers
(`bound_field_contained`), the continuity defect, the scattered-field source distance,
open endpoints, and the solve residual, convergence, and iterations. Resolving the
bound field needs cells well below `γβλ/2π`; slow charges (`γβλ/2π` below a cell)
under-resolve transition radiation.

The Fourier-modal solver carries the same physics for periodic layer stacks:
`fourier_modal.MovingLineChargeSource` is the zeroth-harmonic sheet
`λ d̂ exp(i k_B·r)` with Bloch wavevector `k_B = (ω/v) d̂`, which produces the
Smith–Purcell orders `λ = (d/|n|)(1/β − cosθ)` of a grating directly.
`MovingPointChargeQuadrature` decomposes a point charge into `k_⊥` components with
`Γ = √(ω²/(β²γ²c²) + k_⊥²)`, truncates at `exp(−2Γh) ≤ tolerance`, and integrates with
cosine-mapped Gauss–Kronrod panels that absorb the square-root cutoffs of the orders;
its evidence is the Kronrod–Gauss difference and the truncation bound.

## Discrete-dispersion audit and Cherenkov regimes

`CompatibleMaxwellDispersionAudit(prepared, region, step_size)` linearizes the executed
leapfrog step of a prepared runtime. It extracts the translation-invariant stencil at
the centre of a `MaxwellMaterialRegion` and forms the exact one-step Bloch map `M(k)`
on the local electric, magnetic, and auxiliary material components. Multipliers
`λ = exp(−iωΔt)` give the numerical dispersion `ω = i log(λ)/Δt`. `dispersion(k)` uses
the native dense general eigensolver and reports `stable = max|λ| ≤ 1` at any positive
step, including steps beyond the runtime Courant limit. `bloch_state` synthesizes the
corresponding runtime state.

The audit refuses nonuniform axes, regions narrower than the stencil, nonlinear or
time-varying dynamics, and global magnetic projection. A global Bloch wave checks every
cell of the region, eroded by the stencil radius except along fully covered periodic
axes. A mismatch refuses the region as heterogeneous or as touching a boundary,
source, or absorber. For magnetized plasmas the audit reports
`cyclotron_resonance_shift` (executed minus exact gyration frequency) and
`cyclotron_step_phase = |ω_c|Δt`.

`CherenkovRegimePlan(audit, velocity, angular_frequencies, azimuth_count).evaluate()`
solves the resonance `k·v = ω` with the native bracketed root. The continuum root uses
the Booker indices of `frequency_response` and requires an isotropic `μ`; a
negative-index band reports a negative `continuum_index` and a forward wavevector cone.
The numerical root uses the audited Bloch branches inside the inscribed Brillouin
sphere. `numerical_only` flags numerical Cherenkov radiation: a physical source
radiating into grid modes slowed by discrete dispersion. It is a steady property of
the field solver and is distinct from the numerical Cherenkov instability of drifting
PIC plasmas, which is an aliasing instability of the coupled particle-field loop and is
analysed with spectral PIC.

## Execution and resources

The public state always uses logical one-dimensional cochains. Orientation tensors,
padding, case axes, and shards are private execution layouts. The resource policy counts
primary and auxiliary state, projection workspace, observers, checkpoints, padding,
case axes, per-device state, and requested acquisition before allocation.

Potentially promoted primary, material, observer, CPML, and projection arrays reserve
complex-128 width even when their zero state is initially real.


`solve_compatible_maxwell` scans the same private step core used by
`PreparedCompatibleMaxwell.leapfrog_step`, returns the final state and streaming
observations, and does not implicitly retain a trajectory. Numeric refresh is allowed
only when topology, role layout, prepared array/state shapes and dtypes, static
execution semantics, source envelope semantic identity, PML term layout, dtype, and
backend signature remain unchanged.

## Harmonic defects

The semi-discrete frequency residual evaluates the actual prepared cochain operator and
reports degree-paired absolute and relative norms. A fixed-step harmonic defect is
stronger: it compares the complete affine one-step state map with multiplication by
`exp(-i ω dt)`, including physical and eligible linear auxiliary state and the exact
source phases. Nonlinear, time-varying, or otherwise ineligible systems fail closed
rather than receiving a misleading frequency residual.

## Evidence boundary

Scientific qualification includes chain identities, Gauss continuity, magnetic
closedness, harmonic periods, energy/power balance, TEz/TMz analytic and invariant-3-D
convergence, CPML reflection over frequency/angle/polarization/corners, paired-source
directionality and power, and directional derivatives for every advertised control.
Finite output alone is characterization, not validation.
