# Free-electron lasers

`phydrax.applications.accelerator.fel` integrates the period-averaged
(Kroll–Morton–Rosenbluth) free-electron laser in two forms. `FELPlan` is the
time-independent model: every beam slice is one radiation wavelength long and
evolves independently of the others, so it describes seeded amplifiers,
steady-state gain, harmonic generation, taper, and saturation of a
monochromatic field. `FELTimeDependentPlan` couples the slices through
slippage and resolves SASE start-up, pulse seeding, HGHG/EEHG prebunching,
temporal and spectral structure, and per-slice collective effects.
`FELFullWavePlan` is the full-wave counterpart: the same lattice and beam
through a self-consistent electromagnetic PIC run in a Lorentz-boosted frame,
without period averaging or a slowly varying envelope (see
[Full-wave FEL](#full-wave-fel)).

## Model

With phasors `E_x = Re[Ẽ_h e^{i(hkz − hωt)}]` (the repository's `exp(−iωt)`
convention) and ponderomotive phase `θ = (k + k_u) z − ωt`, each macroparticle
`j` and harmonic `h` obey

- `dθ_j/dz = k_u − k (1 + a_w² + u_x² + u_y²) / (2γ_j²)`,
- `dγ_j/dz = −(e/mₑc²) Σ_h κ_h Re[Ẽ_h(x_j) e^{ihθ_j}] / γ_j`,
- `∂_z Ẽ_h = (i/2hk) ∇⊥² Ẽ_h + (κ_h I / (ε₀ c N ΔA)) Σ_j W_j e^{−ihθ_j} / γ_j`,

with coupling `κ_h = a_w [JJ]_h / √2`, `u = γ(x′, y′)`, and multilinear
deposit weights `W_j`. A planar device has `a_w = K/√2` and
`[JJ]_h = (−1)^((h−1)/2)[J_((h−1)/2)(hξ) − J_((h+1)/2)(hξ)]`,
`ξ = K²/(4 + 2K²)`; a helical device has `a_w = K` and couples only its
fundamental. Even planar harmonics and helical harmonics above one have no
on-axis coupling in this model and are refused.

The Pierce parameter of these equations is
`ρ³ = e²κ²n_e / (8ε₀mₑc²γ³k_u²)` with `n_e = I/(ec·2πσ_xσ_y)`; the cold
one-dimensional field grows as `exp(√3ρk_u z)` (power gain length
`λ_u/(4π√3ρ)`). `fel_scaling_estimate` also evaluates Ming Xie's fitted 3-D
gain length and saturation estimate for every slice.

## Lattice

A lattice is a sequence of `FELUndulatorSegment`s bound to one
`ElectromagneticScaleContract`. Each segment wraps an X1a
`InsertionDeviceField`; its peak field sets `K`, so a stepwise taper is a
sequence of segments with decreasing peak fields. The break after a segment
can carry a thin quadrupole (`quadrupole_integrated_gradient`) and a phase
shifter at its entrance. Focusing inside a module is the device's natural
focusing (`k² = a_w²k_u²/γ²` in `y` for planar, `a_w²k_u²/(2γ²)` per plane for
helical) plus an optional symmetric FODO-averaged `smooth_focusing_gradient`.
The averaged model uses each module's flat-top length `N λ_u`.

```python
import math

import jax
import phydrax as phx
from phydrax.applications.accelerator import InsertionDeviceField, fel

scale = phx.ElectromagneticScaleContract.si()
period, gamma, deflection = 0.03, 1.0e4, 3.5
peak = deflection * 2 * math.pi * float(scale.electron_mass) * float(
    scale.speed_of_light
) / (float(scale.elementary_charge) * period)
device = InsertionDeviceField(
    peak, period, 1200, polarization="planar", center=0.0, aperture=(1e-2, 1e-2)
)
lattice = fel.FELUndulatorLattice(
    scale, (fel.FELUndulatorSegment(device),), step_length=0.05
)
wavelength = lattice.resonant_wavelength(gamma)
```

## Beam slices and loading

`FELBeamSlices` declares per-slice position, current, energy, relative energy
spread, and Gaussian transverse phase space (normalized emittance, Twiss `β`,
`α`) together with a `uint32` identity per slice. `FELLoading` represents a
slice by `B` beamlets of `M` particles at equally spaced phases (quiet
start); `M ≥ 2 max(h)` is enforced. `shot_noise="fawley"` adds Fawley's
phase perturbations so that each slice carries the Poisson bunching
`⟨|b_h|²⟩ = 1/N_λ` of its `N_λ = Iλ/(ec)` electrons. All random draws use
`derive_key(key, address, 0, slice_identity, beamlet, event)`, so a solve is
invariant to slice order and `slice_batch`.

## Solving

```python
slices = fel.FELBeamSlices(
    [0.0], [3000.0], [gamma], [1e-4], [[1e-6, 1e-6]], [[10.0, 10.0]], [[0.0, 0.0]]
)
plan = fel.FELPlan(
    lattice,
    wavelength,
    loading=fel.FELLoading(256, 6, shot_noise="quiet"),
    harmonics=(1, 3),
    seed=fel.FELSeed(1e3, waist=30e-6),
)
result = plan.solve(slices, jax.random.key(0))
```

`transverse="one-dimensional"` couples one field value per harmonic over the
slice area `2πσ_xσ_y`. `transverse="angular-spectrum"` evolves the field on a
uniform `PlaneFieldSpace` and diffracts it with the optics
`AngularSpectrumPlan` (exact Helmholtz propagator with the carrier removed;
finite windows need padding and report cropped power). Each step is the
symmetric Strang sequence `T(Δz/2) D(Δz/2) S(Δz) D(Δz/2) T(Δz/2)`: exact
betatron rotation, diffraction, and a fourth-order Runge–Kutta source kick of
the coupled particle/field system; the deposit is the exact transpose of the
gather.

`FELResult` reports `power[s, z, h]`, complex bunching, beam power, mean
energy and energy spread at every step; the exit field; the harmonic
frequencies `hω` and, on the grid route, the exit far field `dP/dΩ`; initial
and exit particles; `FELGainEvidence` (fitted power gain length over a
declared window of the peak power, saturation power and position, and the
analytic scaling); and the `FELEnergyLedger`
`beam_power_change + field_power_change + diffraction_loss + wake_loss`,
whose relative defect is checked against `ledger_tolerance`. `FELStatus`
flags nonfinite output, diffraction leakage, particles leaving the field
window, ledger defects, unresolved detuning phase steps, and nonlinear
shot-noise loading.

## Collective effects

`FELWakeLoss` applies an X2 longitudinal `WakeFunctionPlan` per slice: the
causal wake potential of the slice charges `q_s = I_s Δζ/c` (half self term)
per declared structure length becomes a constant energy-loss rate along the
lattice, and the removed power appears as `wake_loss` in the ledger. Slices
must be uniformly spaced. Space charge is evaluated from the current particles
every step by the time-dependent plan through `FELSpaceCharge` (below); the
time-independent `FELPlan` does not apply it (a uniform, infinitely long beam
is the periodic time-dependent window, which models the intra-slice part).

## Time-dependent FEL

`FELTimeDependentPlan(core, slippage=..., boundary=...)` composes a
time-independent `FELPlan` (lattice, wavelength, loading, harmonics,
transverse model, wakes) and couples its slices. The beam is a chain of
uniformly spaced slices `Δζ` apart; slice `s` represents
`N_s = I_s Δζ/(ec)` electrons, so Fawley shot noise gives the Poisson SASE
start-up of that many electrons (still identity-addressed per slice). The
radiation lives on a window of `head_padding` field-only slots ahead of the
beam plus one slot per slice. Radiation slips ahead of the electrons by one
wavelength per undulator period (`λ/λ_u` per unit length) and by
`1/(2γ_r²)` per unit length in breaks, with `2γ_r² = λ_u(1 + a_w²)/λ` of the
first module.

Each step is the Strang sequence
`T(Δz/2) S(Δz/2) F(Δz) K(Δz) S(Δz/2) T(Δz/2)` in which `F = D X` combines the
diffraction of every slot with the slippage translation `X` by one step of
slip. Consecutive source kicks see the field one step's slip apart, so an
undulator step must slip at most one slot or the field jumps over slices
(`FELStatus.SLIPPAGE_UNRESOLVED`; breaks carry no coupling and are exempt).
`K` is the space-charge kick (below); it acts on particles only and commutes
with `F`.
`X` never interpolates: `slippage="commensurate"` requires every step's slip
to be an integer number of slots (for example one-period steps with
`Δζ = λ`) and rolls the window; `slippage="spectral"` multiplies the
window's discrete Fourier transform by the exact phase ramp and accepts any
slip, including breaks. `boundary="periodic"` models an infinitely long
periodic beam; for a uniform beam it reproduces `FELPlan` slice by slice.
`boundary="open"` lets radiation leave through the head (zero field enters at
the tail); the energy that leaves is ledgered as `exit_energy`, and
`FELTimeDependentEvidence` reports the total slippage, whether the head
padding covers it (`padding_sufficient`), and the exit fraction
(`FELStatus.WINDOW_TRUNCATED` above `window_tolerance`). Open windows refuse
the continuous-wave `FELSeed`.

```python
plan = fel.FELTimeDependentPlan(
    fel.FELPlan(lattice, wavelength, loading=fel.FELLoading(64, 4, shot_noise="fawley")),
    slippage="spectral",
    boundary="open",
    head_padding=128,
)
result = plan.solve(slices, jax.random.key(0))
```

Seeds come from optics pulse envelopes: `FELPulseSeed(envelope,
reference_position=...)` places a scalar `PulseEnvelopeField` at the lattice
entrance, mapping pulse time `t` to `ζ = reference_position + c t`. Samples
must be spaced `c Δt = Δζ` and land on slots (misalignment or truncation
raises `FELStatus.SEED_UNREPRESENTED`); a carrier `ω′ ≠ ω` is carried exactly
by the ramp `e^{−i(ω′−ω)t}`. The one-dimensional model accepts only
transversely uniform (plane-wave) envelopes; the grid model requires the
plan's `PlaneFieldSpace`.

`FELPrebunching(stages, harmonic=h, reference_lorentz_factor=γ₀)` prepares
HGHG and EEHG beams. Particles are loaded at the base wavelength `hλ`;
`FELModulator(Δγ, harmonic=p, phase=φ)` applies `γ ← γ + Δγ sin(pψ + φ)`,
and an existing `SymplecticMapPlan` (for example a chicane with `R₅₆`) maps
`(x, p_x/p₀, y, p_y/p₀, ζ, δ)`, so a positive-late chicane shifts
`ψ ← ψ − k_b R₅₆ δ`. The radiator receives `θ = hψ`; loading needs
`particles_per_beamlet ≥ 2h max(harmonics)`. Particles stay in their slice
(the slice is the periodic representative of a locally uniform beam) and the
largest displacement is reported.

`FELSpaceCharge(bunch=None, *, transverse, harmonics=0, radial_extent=0.0,
radial_cells=100, azimuthal_modes=0, transverse_tolerance=1e-2)` evaluates
space charge from the current particles at every step:

- Intra-slice (wavelength-scale) longitudinal space charge from the bunching
  harmonics `l = 1 … harmonics` of each slice, the Genesis 1.3 version 4
  short-range model: in the frame of the mean longitudinal motion
  (`γ_z² = γ_r²/(1 + a_w²)` of the step) the field solves
  `[1 − (γ_z/lk)²∇⊥²] E_l = −i (I/(ε₀ c l k)) (e/mₑc²) ρ̂_l` and kicks
  `dγ/dz = Σ_l 2 Re(E_l e^{ilθ})`. The `"angular-spectrum"` model solves it on
  a radial grid (`radial_cells` annuli out to
  `max(radial_extent, 1.5 r_max)`, azimuthal modes `|m| ≤ azimuthal_modes`)
  centered on the slice centroid; the `"one-dimensional"` model treats the
  slice as a uniform disk of its current area `A = 2πσ_xσ_y` with the
  transversely averaged reduction `F_l = 1 − 2I₁(ξ)K₁(ξ)`,
  `ξ = lk√(A/π)/γ_z`. A cold bunched beam then oscillates at
  `λ_p = 2πγ^{3/2}c/(ω_p√(F(1 + a_w²)))`. Loading needs
  `particles_per_beamlet ≥ 2 harmonics`.
- Bunch-scale (inter-slice) fields from the X3 `SpaceChargeIGFPlan`
  (`bunch`; capacity slices × particles per slice, open windows only): every
  particle is deposited at its slice center in the mean-motion frame `γ_z`
  of the step, and the kick `Δ(pc) = q(E + v × B)Δz/β_z` changes the energy by
  `β · Δ(pc)`. Its grid must cover `γ_z` (up to `γ_r` in breaks) times the
  window plus one guard cell, and resolve the bunch (`cells_per_sigma`); a
  refused kick is skipped and raises `FELStatus.SPACE_CHARGE_REFUSED`.
- Transverse space charge: the X3 transverse kick `Δu⊥ = qE⊥Δz/(mₑc²γ_z²)`
  (∝ 1/γ_z², negligible for most FELs) is applied for `transverse="applied"`
  and only measured for `"omitted"`. `FELTimeDependentEvidence` reports the
  accumulated rms kick per slice relative to the entrance rms `u⊥`
  (`transverse_space_charge_ratio`, NaN without `bunch`); an omitted kick above
  `transverse_tolerance` raises `FELStatus.TRANSVERSE_SPACE_CHARGE_OMITTED`.

The result reports the slice-mean Lorentz factor gained from space charge up
to every record (`space_charge_gain`); the smallest X3 `cells_per_sigma` over
the run is evidence. Wakes apply per slice exactly as in `FELPlan`.

`FELTimeDependentResult` reports the temporal power `power[z, j, h]` on every
slot and record, the pulse energy, per-slice bunching, beam power, energy and
energy spread, the exit field, the collective wake rate and space-charge gain,
and `FELSpectrum`: the
one-sided exit spectral energy `dW/dω` at `hω + Δω`
(`F(ω) = ∫ f(t) e^{+iωt} dt`), the spike count above `spike_threshold` of the
peak, the single-shot coherence time `∫|g₁|² dτ`, and the rms bandwidth. The
`FELTimeDependentLedger` closes
`beam + field + diffraction + exit + wake + space charge` energy over the
whole window (space charge exchanges kinetic with untracked electrostatic
energy; its internal field exerts no net force).

`run_genesis4` translates a supported subset (SI scale, angular-spectrum
grid, fundamental only, open unpadded window, uniform slices, modules and
drifts, longitudinal space charge mapped onto Genesis `&efield`) into a
Genesis 1.3 version 4 deck and runs a caller-pinned executable as an external
oracle; Genesis is GPL-licensed and is never bundled.

`run_puffin` is the unaveraged counterpart: Puffin (Campbell & McNeil, Phys.
Plasmas 19, 093119, 2012; BSD-3-Clause, run only as a caller-pinned binary)
resolves the radiation carrier and the electron wiggle. `puffin_input` writes
Puffin's scaled frame from the plan (`ρ` the plan's Pierce parameter,
`l_g = λ_u/(4πρ)`, `l_c = λ/(4πρ)`, `γ_r` at Puffin's exact resonance) for the
one-dimensional model: fundamental only, open unpadded window, identical
slices spaced by whole wavelengths, contiguous modules of one period (stepwise
`K` as module tuning), and a declared `PuffinGaussianSeed`. `read_puffin_power`
averages Puffin's carrier-resolved power over the window slots (head first).
The `PuffinResult` report declares the losses: sub-wavelength structure
averaged away, every harmonic and flat-top-edge coherent spontaneous emission
included in the slot power, transverse phase space reduced to the slice area,
and Puffin's own steps and macroparticles. On a seeded `ρ = 0.005` helical
amplifier (Puffin 2.1.0a) the fitted gain length (0.281 m vs 0.280 m) and the
saturated peak power (2.84 MW) agree with `FELTimeDependentPlan` within 0.3 %.

## Full-wave FEL

`FELFullWavePlan(lattice, wavelength, *, boost_lorentz_factor, transverse_size,
...)` runs the FEL as a self-consistent electromagnetic PIC in the frame moving
at `β_b ẑ` (`phydrax.solver.BoostedFramePlan`). At the undulator resonance
`γ_b = γ_z = γ/√(1 + a_w²)` the beam is on average at rest, the period contracts
to `λ_u′ = λ_u/γ_b`, and the radiation wavelength stretches to
`λ′ = λγ_b(1 + β_b)`: one scale, so the cost does not grow with `γ²` (Vay 2007;
Fawley and Vay 2009). The lattice must use PIC code units (the PIC relativity
scale's `c` and `ε₀ = 1`).

- Field: a standard staggered PSATD grid with the boosted run's NCI guard,
  periodic over the transverse cross section (`transverse_size`,
  `transverse_cells`; a thin cross section with a transversely uniform beam is
  the one-dimensional limit) and open along the boost axis: `absorber_cells`
  PSATD PML cells at both ends absorb whatever leaves the interior.
  `cells_per_wavelength` resolves `λ′`; `steps_per_period` resolves the passage
  of one boosted period (never above the unaliased PSATD step).
- Undulator: each lattice device is gathered through `BoostedExternalField`,
  never deposited and never radiating on the grid.
- Beam: `FELFullWaveBeam` holds lab positions and velocities upstream of the
  lattice support; particles reach it ballistically. `species="electron-positron"`
  pairs every electron with a co-moving positron: the positron couples to the
  undulator and radiation like its electron, so the neutral pair beam is the
  full-wave counterpart of a beam without space charge (the periodic Gauss
  solve refuses a charged box). `flat_top_beam(γ, I, wavelengths=...)` loads a
  quiet-start, one-dimensional flat-top beam (planar lattice, two vertical
  cells) with optional prebunching `θ = ψ − a sin(ψ − φ)` (bunching `J₁(a)`).
- Seed: `FELFullWaveSeed(E₀, phase=..., ramp_wavelengths=4, margin_wavelengths=2)`
  is the lab plane wave `E_x = cB_y = E₀ g(ξ) cos(kξ + φ)`, `ξ = z − ct`, whose
  flat top covers every electron's slippage interval plus the margins, with
  `C^∞` ramps. A one-way B5 `SampledPlaneCurrentAntennaPlan` at rest on the lab
  undulator entrance plane launches it; `BoostedFramePlan.boost_antenna` turns
  it into a sheet moving at `−β_b c` on the spectral grid. The run starts before
  the antenna emits, so the PIC initialization sees no seed.
- Radiation: `FELFullWaveTracks(radiation, lanes)` records lab tracks and
  evaluates A1 trajectory radiation (`TrajectoryRadiationPlan`);
  `FELFullWaveHuygens(frequencies, directions)` closes a spectral Huygens box
  around the beam's boosted envelope and relabels its far field into the lab
  (complete emission). A Huygens box is refused together with a seed antenna
  (the sheet has current at every node along the axis), and when transverse
  periodic images of the radiation reach it before its window closes.

```python
lattice = fel.FELUndulatorLattice(code_scale, (fel.FELUndulatorSegment(device),), step_length=0.0625)
plan = fel.FELFullWavePlan(
    lattice,
    wavelength,
    boost_lorentz_factor=gamma / math.sqrt(1.5),
    transverse_size=(width, width),
    cells_per_wavelength=24,
    steps_per_period=48,
    seed=fel.FELFullWaveSeed(0.3),
)
beam = plan.flat_top_beam(gamma, 4e-3, wavelengths=22, taper_wavelengths=3.0)
result = plan.prepare(beam).run()
```

`FELFullWaveResult` reports the lab Lorentz factors, the lab envelope of the
forward, transversely uniform wave on the co-moving coordinate `ξ` (Gaussian
band of relative width 0.3 about `k′`), the steady amplitude over the window
of field elements that slipped across flat-top beam only, the seed amplitude
measured on the grid over the second half of the tail margin (which no
electron crosses inside the undulator), the gain profile and steady gain
against it, the A1 and Huygens spectra, and the lab `FELFullWaveLedger`:
`beam + field + escaped − injected`, where the grid-field, PML-absorbed, and
antenna four-momenta are carried to the lab by `U = γ_b(U′ + β_b cP′_z)`.
`FELFullWaveFrameEvidence` reports `γ_b` against `γ_z`, the mean boosted beam
velocity, cells per `λ′` and per `λ_u′`, the bunching resolution, the
position-averaged spline factor, the grid origin and spacing, and the
schedule. `FELFullWaveStatus` flags nonfinite output, rejected steps (and among
them NCI refusals), particles still inside the lattice, ledger defects, and
unresolved trajectory radiation.

Qualified behavior (`tests/unit/applications/test_accelerator_fel_full_wave.py`):
the seeded small-signal gain of a `γ = 20`, six-period, 1-D beam agrees with
`FELPlan` within 5 % (+3.6 %, the averaged model's `γ ≫ 1` resonance and end
ramps); the spontaneous boosted Huygens spectrum equals A1 on the same tracks
times the order-one spline deposit factor of the track within 6 % over
±20 % of the peak, for an electron on a cell center and on a node; prebunched coherent
emission matches the KMR field `κI L J₁(a)/(ε₀cAγ)` within 3 % and its power
scales as `(I J₁(a))²`; and the boosted run matches a lab-frame PIC run of the
same beam within 5 % in steady amplitude.

## Validity and non-claims

- Slowly varying envelope, period averaging, and `γ ≫ 1` (the averaged
  plans); steps must resolve the detuning phase (`maximum_phase_step`).
- `FELPlan` slices are time independent: no slippage, no SASE spectrum, no
  slice-to-slice coupling other than the declared wake potential.
- Time-dependent slippage uses the reference-energy group delay; slices do
  not exchange particles, so compression or chicane displacements comparable
  to the length over which current or energy varies are outside the model.
  The spectral route
  treats the window as band limited; sharp field edges produce Gibbs ringing
  whose leakage out of an open window is ledgered.
- One-dimensional time-dependent runs couple every slot to one radiation
  area, the current-weighted mean slice area.
- Space charge acts on the fixed slice current profile (particles stay in
  their slice); the intra-slice field keeps the `radial_cells` binning of the
  slice particles and the bunch-scale field the X3 grid resolution. Wakes are
  frozen per-slice rates.
- Stepwise taper only; module terminations are not resolved.
- Ming Xie's formula is a fit to the exact 3-D eigenmode growth, stated to be
  accurate within about ten percent for a round, matched beam in smooth
  focusing; it is reported as evidence, not used by the solver.
- Full-wave runs are order-one spline, standard PSATD PIC in one boosted
  frame: a particle at rest in that frame radiates through the spline factor of
  its own sub-cell position (the frame evidence reports the position average),
  and the Huygens quadrature and grid dispersion leave `O((k′h)²)` errors (3 %
  at eight cells per `λ′`). The beam's lab energy carries the boosted push's
  discretization error, which bounds the ledger closure. Transverse
  boundaries are periodic.

Runnable examples: `examples/free_electron_laser_averaged.py`,
`examples/free_electron_laser_time_dependent.py`,
`examples/free_electron_laser_full_wave.py`. Benchmarks:
`benchmarks/fel_averaged.py`, `benchmarks/fel_time_dependent.py`,
`benchmarks/fel_full_wave.py`.
