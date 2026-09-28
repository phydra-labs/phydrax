# Spectral PIC field solvers

`phydrax.solver.maxwell.spectral` provides the Cartesian pseudo-spectral analytical time-domain
(PSATD) field solver for explicit electromagnetic PIC. `SpectralMaxwellPlan` declares one PSATD
configuration on a periodic uniform 3-D `StructuredCochainBridge`, and
`prepare(transfers, currents)` binds the species' particle–cochain transfers and
charge-conserving current plans. The result, `PreparedSpectralMaxwell`, is an
`AbstractPreparedPICFieldSolver`, so the same `ElectromagneticPICPlan` runtime that drives the
cochain solvers drives it (see [Particle-in-cell methods](guides_particle_in_cell.md)). It also
implements `PICSpectralSymbol`, `PICHuygensSampling`, `PICMultiDeposit`, `PICRestartState`,
`PICGaussProjection`, and `PICGalileanGrid`.

`QuasiCylindricalMaxwellPlan` is the azimuthal-mode (quasi-cylindrical) counterpart for
laser–plasma geometries; see [Quasi-cylindrical spectral PIC](#quasi-cylindrical-spectral-pic).

`phydrax.solver.BoostedFramePlan` runs Lorentz-boosted-frame PIC on Galilean PSATD; see
[Boosted-frame PIC](#boosted-frame-pic).

## Discretization

Fields `E` and `B` are stored in real space as `[N₀, N₁, N₂, 3]` arrays at their grid locations:
all components on the nodes for `grid="collocated"`, or on the Yee positions for
`grid="staggered"` (`E_c` half a cell along axis `c`, `B_c` half a cell along the two other axes).
The Gauss charge is the node charge density. Each species deposits the spline-Whitney
(Esirkepov) path current of `ChargeConservingCurrentPlan` per current sub-interval together with
its node-charge change; the solver owns the spectral treatment of that current: staggering,
charge-conservation mode, and the Galilean lab-frame current.

Derivatives are the Fourier symbols `D±` of the declared stencil. `stencil="infinite-order"`
uses the exact wavenumber `k`; `stencil="finite-order"` with an even `stencil_order` uses the
centered Fornberg weights `stencil_coefficients(order, staggered)` and the modified wavenumber

```text
[k] = Σ_m 2 w_m sin((m − s) k Δ) / Δ,   s = ½ (staggered) or 0 (collocated),
```

returned by `modified_wavenumber(k, Δ, order, staggered)`. In every Fourier mode the vacuum
system `∂ₜU = A U + F`, `U = (E, B)`, `A U = (c² D⁻×B, −D⁺×E)`, is integrated analytically. With
Galilean coordinates translating at `v_gal` the shifted operator `A + iκ`, `κ = [k]·v_gal`, obeys
`A² = −c²[k]²` on transverse fields, so every analytic function of it is

```text
f(A + iκ) = f(iκ) P∥ + ½[f(z₊) + f(z₋)] P⊥ + ½[f(z₊) − f(z₋)] A/(ic[k]),   z± = i(κ ± c[k]).
```

With `φ₀ = exp` and `φ_{j+1}(z) = (φ_j(z) − 1/j!)/z`, one interval with `J(τ) = J₀ + J₁τ` is

```text
U(h) = φ₀(Ah) U₀ − h φ₁(Ah) (J₀/ε, 0) − h² φ₂(Ah) (J₁/ε, 0).
```

For `κ = 0` and constant `J` this is Haber's PSATD update; for `κ ≠ 0` it is the Galilean PSATD;
for linear `J` it is the time-polynomial current formulation. The φ-functions switch to their
Taylor series near `z = 0`, so the update stays regular at `k = 0`.
Window integrals `∫ U dτ` of the same closed form give the averaged Galilean fields and the PML
split-field increments.

## Options

| Keyword | Values | Meaning |
|---|---|---|
| `variant` | `"standard"`, `"galilean"`, `"averaged-galilean"` | lab-frame PSATD; Galilean coordinates moving at `galilean_velocity`; Galilean with gathered fields averaged over the step |
| `time_dependency` | `"constant-j"`, `"linear-j"`, `"multi-j"` | current constant, linear in time over the step, or piecewise constant over `current_substeps` sub-intervals |
| `charge_conservation` | `"spectral-correction"`, `"vay-deposition"`, `"update-with-rho"` | projects the longitudinal current onto the deposited charge change; maps the Esirkepov current onto the solver's own divergence; drives with the transverse current and lets `E∥` follow the deposited charge |
| `stencil` / `stencil_order` | `"infinite-order"` / `None`; `"finite-order"` / even `≥ 2` | exact or finite-difference derivative symbol |
| `decomposition` | `"global-fft"`, `"local-guarded"` | one distributed transform of the whole box (`DistributedSpectralExecutionPlan` over `topology`); per-block local transforms with `subdomains` blocks and `guard_cells` periodic guard cells |
| `grid` | `"collocated"`, `"staggered"` | nodal or Yee component locations |
| `absorber` / `pml` | `"none"`; `"psatd-pml"` with `SpectralPMLPlan` | periodic box; split-field PML layers inside it |
| `observers` | `SpectralHuygensBoxPlan` values | streaming Huygens boxes for far fields |
| `antennas` | `phydrax.solver.maxwell.SampledPlaneCurrentAntennaPlan` values on the plan's bridge | one-way sheet antennas, stationary or moving along their normal |
| `permittivity`, `permeability` | positive floats | vacuum medium; `c = 1/√(εμ)` |

Selectors are closed `Literal` aliases (`SpectralMaxwellVariant`, `SpectralTimeDependency`,
`SpectralChargeConservation`, `SpectralStencil`, `SpectralDecomposition`, `SpectralGrid`,
`SpectralAbsorber`) parsed at construction.

## Compatibility

Every combination below is refused when the plan is constructed, never at runtime:

| Configuration | Rule |
|---|---|
| `"spectral-correction"` | constant-J with `"global-fft"` only; standard or Galilean form |
| `"vay-deposition"` | standard constant-J only; collocated grids need odd spectral extents (the Nyquist-plane derivative symbol vanishes) |
| `"update-with-rho"` | every variant and time dependency |
| `"linear-j"`, `"multi-j"` | require `"update-with-rho"` |
| Galilean variants | `"update-with-rho"`, or the Galilean-form `"spectral-correction"` for constant-J with `"global-fft"`; `galilean_velocity` must be a nonzero subluminal 3-vector |
| `"averaged-galilean"` | refuses `"linear-j"`: the averaged fields are defined for piecewise-constant currents |
| `"local-guarded"` | finite-order stencils, `subdomains` dividing every axis, `guard_cells` covering the stencil half-width; refuses `"spectral-correction"` |
| `"psatd-pml"`, Huygens observers | standard variant only; boxes lie inside the PML-free interior and declare the solver's vacuum; an infinite-order PML needs its thickest layer at least two cells thick |
| Antennas | `"global-fft"`, `"constant-j"` or `"multi-j"`, the solver's vacuum as medium, a Galilean velocity (if any) along every antenna normal, no Huygens observers; preparation refuses emitted carriers under eight cells per wavelength along the normal and sheets entering a PML layer |

`current_substeps` is accepted only with `"multi-j"` (at least two), `galilean_velocity` only
with a Galilean variant, and `subdomains`/`guard_cells` only with `"local-guarded"`.

## Minimal usage

The spectral solver replaces the cochain field solver of `examples/electromagnetic_pic.py`; the
transfers, current plans, and species are unchanged.

```python
import jax.numpy as jnp

import phydrax as phx
from phydrax.solver.maxwell import spectral

grid = phx.discretization.TensorGridPlan(
    tuple(phx.discretization.UniformCellAxisSpec(8, periodic=True) for _ in range(3)),
    axis_names=("x", "y", "z"),
).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
bridge = phx.discretization.StructuredCochainBridge(grid)

species, charged = [], []
for offset, charge, name in ((0, -1.0, "electrons"), (100, 1.0, "ions")):
    support = phx.discretization.ParticleSetPlan(
        jnp.arange(offset, offset + 4), jnp.ones((4,)), ambient_dimension=3
    ).prepare()
    charged.append(
        phx.discretization.ChargedParticlePlan(charge * jnp.ones((4,)), name).prepare(
            support
        )
    )
    species.append(
        phx.discretization.pic.PICSpeciesPlan(
            phx.discretization.ParticlePopulationPlan(support),
            phx.discretization.pic.PICChargeModelPlan(
                charge,
                name,
                minimum_charge_number=1,
                maximum_charge_number=1,
                initial_charge_number=1,
            ),
        )
    )

transfer_plan = phx.discretization.pic.PICParticleCochainTransferPlan(
    bridge, shape_order=2
)
transfers = tuple(transfer_plan.prepare(value) for value in charged)
currents = tuple(
    phx.discretization.pic.ChargeConservingCurrentPlan(value) for value in transfers
)
solver = spectral.SpectralMaxwellPlan(
    bridge,
    variant="galilean",
    galilean_velocity=(0.0, 0.0, 0.9),
    charge_conservation="update-with-rho",
).prepare(transfers, currents)
pic = phx.solver.ElectromagneticPICPlan(solver, species=species)

position = jnp.asarray(
    [[0.20, 0.20, 0.20], [0.35, 0.45, 0.55], [0.60, 0.30, 0.70], [0.80, 0.75, 0.40]]
)
velocity = jnp.zeros((4, 3))
dt = 0.5 * solver.stable_step
state = pic.initialize((position, position), (velocity, velocity), dt)
result = pic.step_detailed(state, dt)
```

`ElectromagneticPICPlan` verifies the deposit↔Gauss pairing at construction
(`pic.pairing_defect`, zero for every spectral configuration), and each step reports the Gauss
residual as `result.diagnostics.electric_constraint`. `solver.initialize_field(charge)` returns
the Coulomb field `E = −D⁺ρ/(ε[k]²)` of a periodic charge with a neutrality flag, and
`project_gauss` performs the `"spectral-poisson"` projection that particle resampling requires.

## Galilean coordinates and NCI suppression

A plasma drifting relativistically through a PSATD grid couples its aliased beam modes
`ω = (k + 2πm/Δ)·v` to the grid's electromagnetic modes and grows the numerical Cherenkov
instability (NCI). In Galilean coordinates `x′ = x − v_gal t` moving with the drift, the drifting
plasma is at rest on the grid and the instability is eliminated when `v_gal` equals the drift
velocity (Lehe et al. 2016; Kirchen et al. 2016).

`PreparedSpectralMaxwell.grid_velocity` implements the core `phydrax.solver.PICGalileanGrid`
protocol. Particle positions are grid coordinates: `ElectromagneticPICPlan` drifts them by
`(v − v_grid)Δt` and samples external fields at the lab position `x + v_grid t`. The deposit
therefore returns the convective current `J′` in the grid frame, and the solver forms the lab
current `J = J′ + v_gal ρ` from it and the time-centered deposited charge. The averaged
Galilean variant additionally gathers fields averaged over the step (Shapoval et al. 2021),
which keeps the scheme stable at steps with `cΔt` larger than the cell size. The Galilean
variants refuse the PML and Huygens observers, which are defined in the lab frame.

`examples/galilean_drifting_plasma.py` drifts a neutral cold plasma at `γ = 10` through a periodic
box (`ω_p²/γ = 4`, `cΔt = 0.45Δ`, quadratic shapes). The standard PSATD's high-|k| energy
(`SpectralNCIMonitorPlan`) grows about 400-fold within 70 steps; its fitted amplitude rate `0.30`
is of the order of the Godfrey–Vay linear peak `0.48` (`godfrey_vay_growth_rate`; the box's
discrete modes sample the resonance only approximately, and the tests accept 0.5–1.5 times the
prediction). The Galilean run comoving with the plasma stays at the noise level (fitted rate ≈ 0).

## Charge conservation

The Esirkepov current satisfies the discrete continuity equation for the node-difference
divergence, which differs from the solver's divergence `D⁺·` except for a second-order staggered
stencil. The three modes restore Gauss's law with respect to the solver's own operator:

- `"spectral-correction"` replaces the longitudinal part of `J` so that the spectral continuity
  equation holds with the deposited charge change (in Galilean form with the advected charge);
- `"vay-deposition"` multiplies each current component by `(2/Δ) sin(kΔ/2)/[k]` along its axis, so
  `D⁺·J` equals the Esirkepov divergence identically (Vay et al. 2013); the map needs `[k] ≠ 0`
  off `k = 0`, hence odd extents on collocated grids;
- `"update-with-rho"` drives the fields with the transverse current only and adds the exact
  increment that makes `E∥` follow the deposited charge from the start to the end of each
  sub-interval; it is the one mode valid for every variant and current time dependency.

For every global mode, variant, and time dependency the PIC Gauss residual stays below
`2e-13` and the pairing defect is zero.

## Local guarded transforms

`decomposition="local-guarded"` splits the periodic box into `subdomains` blocks, extends each by
`guard_cells` cells taken periodically from its neighbors, transforms each extended block as its
own periodic box, keeps only its interior, and re-exchanges the guards every step. It requires a
finite-order stencil whose half-width the guards cover. The analytic propagator of a
finite-order stencil is not compactly supported, but its tails decay rapidly with distance, so
the guard cells truncate them to an error that shrinks with guard width (for order 4, four
guards, and half the stable step, `benchmarks/spectral_maxwell.py` records a relative vacuum
energy change of order `1e-7` per step, against roundoff for `"global-fft"`). With
`"update-with-rho"` a current that is not consistent with the finite stencil makes the
longitudinal correction nonlocal, so the truncation error is no longer small; `"vay-deposition"`
keeps the update local and is the mode to use with local transforms.

`guard_truncation(dt)` reports the stencil truncation: the relative real-space mass of the
one-step vacuum propagator kernels (`cos(ωΔt)`, `c[k]_a sin(ωΔt)/ω`) beyond the guard cells, which
bounds the per-step relative difference from `"global-fft"`. On one device the blocks are
transformed in turn; `DistributedPICFieldSolver` runs each device's blocks on that device, with the
guards exchanged through the halo substrate (`guarded_update` on the owned block, then
`complete_advance` for the evidence), so an `N`-device step equals the one-device local-guarded
step to reduction order (see the particle-in-cell guide).

## PSATD PML

`absorber="psatd-pml"` with `SpectralPMLPlan(thickness, *, reflection=1e-6, profile_power=3.0)`
places absorbing layers in the outer `thickness[a]` cells at both ends of axis `a` inside the
periodic box, so the two layers of an axis meet across the periodic seam. Each field component is
split by the derivative that drives it, `F_c = Σ_a F_{c,a}`. Step one advances the splits by the
exact integrals of the analytic PSATD solution, whose sum is the exact update of the totals;
step two damps `F_{c,a}` by `exp(−σ_a h)` in real space with the graded conductivity
`σ_a = σ_max (d/L)^m` at the component's own location (Shapoval, Vay & Vincenti 2019). `σ_max` is
chosen from the declared normal-incidence continuum reflection `exp(−2σ_max L/(c(m + 1)))`.
`SpectralMaxwellDiagnostics.absorbed_energy` reports the energy removed per step. The layers
must stay current-free: a deposit with any current inside them fails the step.

The split-field layer is not divergence-preserving: damping `F_{c,a}` with different `σ_a` changes
`∇⁻·E` and `∇⁺·B` inside the layers, and a spectral derivative carries that change into the
whole box, so the Gauss residual of the interior would grow step by step. The solver books it
instead as absorber charge, `SpectralMaxwellState.absorber_charge` (over `ε`) and
`absorber_magnetic_charge`, supported in `PreparedSpectralMaxwell.absorber_support`:

- finite-order stencils create it within the stencil half-width of the layers, which is the
  support (no correction is needed; local-guarded runs keep every transform local);
- infinite-order divergences are global, so each step confines the created divergence to the
  layers, moves its content along the modes the derivative cannot see (the mean and, on collocated
  grids, the checkerboards) onto the two outermost planes of the thickest layer, where those
  patterns are orthogonal, and adds the curl-free field of the difference to `E`/`B` (a static
  field that does not radiate).

The electric and magnetic constraints are then the full-grid residuals
`∇⁻·E − (ρ + ρ_absorber + ρ_antenna)/ε` and `∇⁺·B − ρ_m`, and `∇⁻·E − ρ/ε` vanishes to roundoff
outside the support over arbitrarily long runs. `project_gauss` keeps the declared charges.

Measured (`tests/unit/solver/test_spectral_maxwell.py`): a static dipole whose Coulomb field fills
four-cell layers keeps the independent interior Gauss residual below `1e-13` over 300 steps for
staggered and collocated infinite order and staggered order 4 (without the bookkeeping it grew to
`3e-3` of the charge density); an oblique 3-D burst leaves `1.3e-5` of its energy after crossing
the layers (`1.26e-5` before the confinement) with the absorbed-energy ledger closed to `1e-12`;
PIC with the PML is accepted over 200 steps with its constraint at roundoff.

## Sheet antennas

`SpectralMaxwellPlan(..., antennas=(antenna, ...))` takes the same
`SampledPlaneCurrentAntennaPlan` that drives compatible Maxwell runs (its normal axis may be
periodic here) and prepares it as `PreparedSpectralPlaneAntenna`
(`PreparedSpectralMaxwell.antennas`, with the shared `SampledPlaneAntennaEvidence`). The sheet is
a moving total-field/scattered-field boundary radiating `Θ·(E, H)` of the Lorentz-transformed
rest-frame wave through

```text
J = s δ_w(x_n − x_s(t)) (n̂ × H + v εE),   M = s δ_w(x_n − x_s(t)) (−n̂ × E + v μH),
```

evaluated on the sheet at rest-frame time `t/γ` and held constant over each step
(`∂ₜB = −∇×E − M`, integrated with the same exact interval propagator). `δ_w` is a
band-limited delta along the normal, flat up to a quarter of the Nyquist wavenumber and rolled off
with `cos²` to zero at half of it: a sheet radiates only at `k = ±ω/c`, so admitted emission (at
least eight cells per emitted wavelength) is exact and one-way, while the empty upper half of the
spectrum keeps the sheet's bound near field out of the high-`|k|` shells the NCI monitor
watches. The sheet's divergences are declared charges, `SpectralMaxwellState.antenna_charge` and
`antenna_magnetic_charge`, advanced with the exact longitudinal propagator; the constraints
include them. `SpectralMaxwellDiagnostics.antenna_work[a]` is the work `−∫∫(J·E + M·H)` antenna
`a` does on the field over the step, from the analytic in-step fields.

On a Galilean grid the sheet moves at `v − v_gal`: `BoostedFramePlan.boost_antenna` of a lab
antenna at rest is stationary on the boosted Galilean grid.

Measured (`tests/unit/solver/test_spectral_antenna.py`): a plane pulse at 20 cells per
wavelength matches its retarded analytic waveform to `3e-3` (the constant-in-step current) with
a backward/forward energy ratio below `1e-12`, and the work ledger equals the field energy to
`3e-14`; a sheet receding at `β = −0.9` emits at `γ(1 + β)ω₀` (spectral centroid within `6e-5`,
peak `F(ω) = F′(ω/D)` within `8e-4`); a paraxial Gaussian beam (`kw₀ = 12.6`) reaches its waist
within `0.6%` in width and `5e-3` rad in Gouy phase; a focused 3-D sheet's declared charges equal
the NumPy spectral divergences of `E` and `B` to roundoff.

## Huygens surfaces and far fields

`SpectralHuygensBoxPlan(lower, upper, acquisition, exterior)` declares a closed axis-aligned box
on node planes of a `grid="staggered"` solver (collocated grids are refused: their half-cell
spectral centering of the deposited edge current leaves current on every node along each
current's axis, so no surface is current-free). Each tangential `E` component is sampled at its
own edge midpoints in the face, without interpolation (midpoint rule along the edge, trapezoid
across it); the paired `H = B/μ` is interpolated across the face by the fourth-order stencil
`(−1, 9, 9, −1)/16`, which reaches `3h/2`, so the box must stay two cells clear of any PML. The
samples are folded into the acquisition's
phasors inside every `advance`. The acquisition must be the time-integral measure with positive
exponent, and the exterior must be the solver's vacuum. `SpectralMaxwellDiagnostics.surface_current`
reports the largest current sampled on the surface inside the acquisition window, which must stay
zero for the equivalence theorem to hold.

On a `16³` bridge with three-cell layers:

```python
acquisition = phx.solver.maxwell.MaxwellSpectralAcquisition(
    omegas, sign="positive", measure="time-integral"
)
exterior = phx.solver.maxwell.HomogeneousMaxwellExterior()
box = spectral.SpectralHuygensBoxPlan((5, 5, 5), (11, 11, 11), acquisition, exterior)
solver = spectral.SpectralMaxwellPlan(
    bridge,
    grid="staggered",
    absorber="psatd-pml",
    pml=spectral.SpectralPMLPlan((3, 3, 3)),
    observers=(box,),
).prepare(transfers, currents)
# ... run ElectromagneticPICPlan ...
(phasors,) = solver.huygens_phasors(state.field)
far_field = phx.solver.maxwell.MaxwellFarFieldPlan(
    directions, (0.0, 0.0, 1.0), exterior
).evaluate(phasors)
```

`MaxwellFarFieldPlan` is the same radiation-vector transform the cochain Huygens samplers use;
see [Compatible Maxwell](guides_compatible_maxwell.md) for the far-field conventions.

## NCI monitor and linear reference

`SpectralNCIMonitorPlan(solver, *, shell_count=8, high_fraction=0.5)` reduces the field energy
`½∫(ε|E|² + |B|²/μ)` of a state into isotropic `|k|` shells by Parseval
(`PeriodicFourierShellPlan`) and reports the energy above `high_fraction` of the largest grid
wavenumber, where the NCI first appears. `SpectralNCIMonitorPlan.fit(times, energies, start=...,
stop=...)` fits `ln W(t) ≈ a + 2Γt` and returns `NCIGrowthFit` with the field-amplitude rate `Γ`.

`godfrey_vay_growth_rate(solver, *, drift_axis, transverse_axis, drift_speed, plasma_frequency,
step_size, aliases=2)` solves the linear PSATD–Esirkepov cold-beam dispersion relation of Godfrey,
Vay & Haber (2014, Eqs. 14–25) with the solver's current scaling on every resolved grid mode of
the drift/transverse plane, by Newton iteration seeded at the beam aliases, and returns
`GodfreyVayReference` with per-mode growth rates and `maximum_growth_rate`. It covers the
configuration that analysis describes (standard, constant-J, infinite-order, collocated,
global-FFT PSATD with one shared shape order) and refuses others.

```python
monitor = spectral.SpectralNCIMonitorPlan(solver)
energies = [monitor.sample(state.field).high_energy for state in history]
measured = spectral.SpectralNCIMonitorPlan.fit(times, energies, start=t0, stop=t1)
predicted = spectral.godfrey_vay_growth_rate(
    solver, drift_axis=2, transverse_axis=0, drift_speed=0.9,
    plasma_frequency=omega_p, step_size=dt,
)
```

The WarpX oracle (`phydrax.solver.run_warpx`, see the particle-in-cell guide) runs the same
Cartesian PSATD case in WarpX 26.01. With the staggered grid both codes use Esirkepov current
and spectral correction; the monitored cell-centered high-`|k|` energy of a γ = 10 drifting
plasma then grows at 0.205 (WarpX) against 0.228 (Phydrax) over `7 ≤ t ≤ 13.9`. WarpX refuses
charge-conserving deposition on collocated grids and Vay deposition with a global FFT, so the
collocated case is a declared-loss comparison and Vay is refused.

## Boosted-frame PIC

A laser wakefield stage or an undulator spans scales from the laser (or radiation) wavelength
to the plasma (or device) length. In a frame moving with `v_b = β_b c ê` along the propagation
axis (Vay 2007) the laser wavelength stretches by `γ_b(1 + β_b)` and lab-resting lengths contract
by `γ_b`, so the cells and steps of a stage shrink roughly by `γ_b²(1 + β_b)`.
`phydrax.solver.BoostedFramePlan(frame, lab_domain, *, relativity=PIC_CODE_RELATIVITY)` owns the
frame and the lab region of interest; `prepare(pic, *, snapshots, ...)` binds an
`ElectromagneticPICPlan` on a `PreparedSpectralMaxwell` and returns `PreparedBoostedFrame`.

### Frame and transforms

`frame` is a `phydrax.LorentzFrame` holding the passive map from lab to boosted four-vectors,
`LorentzFrame(boost_matrix(β_b ê))` (equivalently `LorentzFrame.boost(−β_b ê)`); rotations,
oblique boosts, and the identity are refused. `lab_domain` is a `phydrax.geometry.Box`.
Every transform is an exact Lorentz transform in the scale's units:

- `to_lab`/`from_lab` map events, `boost_fields`/`lab_fields` map `(E, B)` by `F′ = ΛFΛᵀ`, and
  `boost_proper_velocity`/`lab_proper_velocity` map `u = γv`.
- `boost_particles(lab_positions, lab_velocities, *, boosted_time, lab_times=0)` boosts each
  particle's proper velocity and drifts it ballistically from its own boosted event onto the slice
  `t′ = boosted_time` (`BoostedParticles.drift_intervals` reports the interval). Lab-resting plasma
  becomes the boosted plasma of density `γ_b n` (its spacing contracts at unchanged macroparticle
  weights) drifting at `−v_b`; beams receive the relativistic velocity addition. The drift assumes
  free streaming, so beams are loaded in vacuum.
- `boost_external_field(source)` returns a `BoostedExternalField`, an `ExternalFieldSource` that
  samples the lab source at each particle's own lab event and boosts the result. External fields
  are gathered by the pusher and never deposited: a static undulator boosts into a field pattern
  moving at `−β_b c` with `E·B = 0` and `E² − c²B² = −c²B_lab² < 0`, a sub-luminal, magnetic-type
  (evanescent) field that no free vacuum mode carries, so it cannot radiate on the grid.
- `boost_antenna(antenna, bridge, *, scale=None)` turns a lab
  `SampledPlaneCurrentAntennaPlan` normal to the boost axis into the moving antenna of the boosted
  frame on a boosted `bridge`: the rest-frame samples are unchanged, the sheet velocity composes
  relativistically, and the rest-frame time origin moves to the sheet's crossing of `t′ = 0`, so
  each sheet event keeps its lab sample. Moving antennas need a scale's vacuum. The result drives
  compatible (cochain) Maxwell runs and, through `SpectralMaxwellPlan(antennas=...)`, the boosted
  PSATD grid (see [Sheet antennas](#sheet-antennas)).

### Grid, loading, and the NCI guard

The field grid is either Galilean comoving with the boosted lab plasma
(`galilean_velocity=frame.galilean_velocity`, that is `−v_b`) or standard. In Galilean coordinates
`z = z′ + v_b t′` the lab plasma is at rest, `z_lab = γ_b z`, and its NCI is suppressed; the grid
must cover the contracted lab domain (`frame.grid_lower`/`grid_upper`). The moving plasma
boundary of the boosted frame is stationary on this grid, so the lab plasma column is injected
once, on the initial slice. The standard grid serves vacuum and beam-only runs (the Huygens
route); `initialize` refuses plasma drifting at `−v_b` on it.

`PreparedBoostedFrame.initialize(positions, velocities, step_size, *, time, masses=...,
vacuum_fields=(...))` takes the boosted positions and velocities on the slice `t′ = time`.
`vacuum_fields` are lab vacuum solutions (laser pulses as `ExternalFieldSource`s) sampled at the lab
events of the slice, boosted, added to the grid field, and Gauss-projected
(`BoostedFrameEvidence.vacuum_divergence` reports the removed residual). They must not overlap
particles, because the leapfrog bootstrap of the PIC initialization does not see them; choose the
slice through the lab plasma entrance at the lab time the laser is still in vacuum, or launch the
laser through a boosted antenna instead.

The NCI guard is mandatory. Every `step_detailed` samples the high-`|k|` shell energy `W_k` of the
candidate field with `SpectralNCIMonitorPlan` (`nci_shell_count`, `nci_high_fraction`) and rejects
the step when `W_k` has grown beyond `nci_growth_limit` (default `1e2`) times its reference (the
initial high-`|k|` energy, or the first nonzero one) while also exceeding `nci_energy_fraction`
(default `1e-2`) of the electromagnetic energy. `BoostedFrameEvidence` carries the reference, the
largest growth and fraction, and the count of rejected steps.

### Back-transformed diagnostics and tracks

`BoostedSnapshotPlan(lab_times, *, ring_capacity=None, species=())` requests lab snapshots. Lab
plane `z` of snapshot `T` is reached at boosted time `t′ = γ_b(T − β_b z/c)`; each step fills the
planes whose time falls in `[t′ₙ, t′ₙ₊₁)` from the two bracketing fields, linearly in time and
along the axis in grid coordinates, and transforms them back. Planes are the lab images of the
grid's node planes inside the lab domain. A particle enters snapshot `T` when its lab time crosses
`T` inside the lab domain. Snapshots occupy a ring of `ring_capacity` slots; preparation refuses a
ring whose next occupant of a slot starts before the previous one completes, and
`BoostedSnapshotState.conflicts` counts runtime evictions of incomplete snapshots. A snapshot is
read with `lab_field_snapshot(state, index)` or `lab_particle_snapshot(state, index, species)`
while resident.

`lab_trajectory(state, recorder, scale)` converts a `PICTrackRecorder` into a lab
`ChargedTrajectory` with per-lane lab times, the input of the trajectory-radiation route
(`TrajectoryRadiationPlan`). Recorders that stream radiation in the boosted frame are refused.

`lab_far_field(state, far_field, *, emission)` relabels a boosted Huygens far field into the lab
through `phydrax.transform_spectral_energy`. It requires a standard grid with Huygens observers,
`J = 0` on the surfaces at every attempted step (`BoostedFrameEvidence.surface_current`), closed
acquisition windows, vacuum, and complete emission; otherwise it refuses. `checkpoint`/`restore`
add one `"boosted-frame"` restart component (snapshot ring and evidence) owned by the prepared run
to the PIC components; `restart_component`/`restore_component` implement `PICRestartState`.

```python
frame = phx.solver.BoostedFramePlan(
    phx.LorentzFrame(phx.boost_matrix(jnp.asarray([0.0, 0.0, beta_b]))),
    phx.geometry.Box(center, size),
)
start, _ = frame.from_lab(0.0, plasma_entrance)
plasma = frame.boost_particles(lab_plasma, jnp.zeros_like(lab_plasma), boosted_time=start)
solver = spectral.SpectralMaxwellPlan(
    bridge,
    variant="galilean",
    galilean_velocity=frame.galilean_velocity,
    charge_conservation="update-with-rho",
).prepare(transfers, currents)
boosted = frame.prepare(
    phx.solver.ElectromagneticPICPlan(solver, species=species, recorders=(tracks,)),
    snapshots=phx.solver.BoostedSnapshotPlan((30.0,), species=(0,)),
)
state = boosted.initialize(
    (plasma.positions, plasma.positions), (plasma.velocities, plasma.velocities), dt,
    time=start, masses=masses, vacuum_fields=(laser,),
)
result = boosted.step_detailed(state, dt)
trajectory = boosted.lab_trajectory(result.accepted_state, 0, scale)
```

### Measured evidence

`tests/unit/solver/test_boosted_frame.py` compares boosted runs against lab references:

- One-dimensional LWFA stage (`a₀ = 1` cos² pulse, `n = 0.01 n_c`, 16-long plasma, 21 test
  electrons at `γ = 20` across one plasma period) at `γ_b = 2` against the lab run, both with 16
  cells per (Doppler-stretched) wavelength: the energy-gain amplitude over the witness phases
  agrees to `3e-4` relative, the phase-averaged gain to `2%` of the amplitude, and every witness to
  `9%` (a `3%`-of-`λ_p` wake phase difference). The back-transformed wake snapshot matches the lab
  field at the same lab time within `11%` in peak and `22%` in L2 (PIC noise and the phase
  difference), and the back-transformed witnesses sit on the lab witnesses within `3e-4`. The
  boosted grid has 425 cells and 825 steps against 768 cells and 1511 steps in the lab.
- A vacuum laser back-transformed from a `γ_b = 2` Galilean run equals the analytic lab pulse
  within `5e-3` of its amplitude, the `(ωΔt′)²/8` error of the linear time interpolation.
- An electron at `γ = 10` through a `K = 0.5` planar undulator boosted at `γ_b = 5`: its lab-frame
  tracks radiate the lab spectrum through the segment-exact trajectory route within `0.6%` (L2
  over three directions and 36 frequencies), with the on-axis peak at `2γ²ω_u/(1 + K²/2)`; the grid
  energy stays at the `10⁻²⁶` self-field level of the `10⁻¹²` test charge while the boosted
  undulator would carry `166` on the grid axis.
- The same LWFA stage seeded by `boost_antenna` of a lab antenna at rest at `z = 22` (sampling the
  laser's trace) instead of a loaded vacuum field: every step accepted with no NCI rejection, the
  gain amplitude within `0.3%` and the phase-averaged gain within `2%` of the amplitude of the lab
  run.

`examples/boosted_frame_lwfa.py` runs the boosted stage alone;
`benchmarks/boosted_frame.py` records the phase-separated cost of plain and guarded boosted steps
against `γ_b`.

## Quasi-cylindrical spectral PIC

`QuasiCylindricalMaxwellPlan(grid, *, variant, charge_conservation, absorber, damping,
galilean_velocity, antennas, observers, permittivity, permeability)` declares a PSATD solver on a
`phydrax.discretization.pic.QuasiCylindricalGrid(radius, radial_count, lower, upper, axial_count,
mode_count)`, and `prepare(transfers)` binds one `PreparedAzimuthalTransfer` per species. The
result, `PreparedQuasiCylindricalMaxwell`, is an `AbstractPreparedPICFieldSolver` driven by the
same `ElectromagneticPICPlan` runtime; it implements `PICSpectralSymbol`, `PICHuygensSampling`,
`PICWindowShift` (axis 2 only), `PICGalileanGrid`, `PICRestartState`, and `PICGaussProjection`.

### Model

Every field is a truncated azimuthal Fourier series on cell-centered radii and a periodic axial
grid:

```text
F(r, θ, z) = Re Σ_{m=0}^{M} F_m(r, z) e^{imθ},   r_j = (j + ½)R/N_r,   z_k = lower + kΔz,
```

with `M + 1 = mode_count`. Transverse vectors are stored in circular components
`F_± = (F_r ∓ iF_θ)/2`, which in mode `m` carry the angular orders `m ∓ 1`; `F_z` and scalars carry
order `m`. Field arrays are `[M+1, N_r, N_z, (F_+, F_−, F_z)]`.

`phydrax.discretization.SharedGridHankelPlan(radius, radial_count, mode_count)` prepares the
radial transforms. Every mode `m` uses one k-grid `k_{m,n} = α_{m,n}/R` from the zeros of `J_m`
(the `N_r` positive zeros for `m = 0`; `k = 0` followed by the first `N_r − 1` positive zeros for
`m ≥ 1`) shared by the three orders `m − 1, m, m + 1`. Synthesis is
`f(r_j) = Σ_n c_n J_p(k_{m,n} r_j)`; the `k = 0` column of order `m − 1` is its limit shape
`(r/R)^{m−1}` and vanishes for orders `m` and `m + 1`. Analysis is the Moore–Penrose pseudoinverse
of each synthesis matrix, prepared once through `phydrax.linalg.pseudoinverse`.
`SharedGridHankelEvidence` reports the rank, the expected rank (full for `m = 0` and order
`m − 1`, one less for orders `m` and `m + 1` at `m ≥ 1`), the condition estimate, and the Penrose
residual `‖AA⁺A − A‖/‖A‖` of every mode and order; solver preparation refuses a transform whose
evidence is unsuccessful, and `PreparedQuasiCylindricalMaxwell.hankel_evidence` exposes it. Radial
content outside the synthesis range (one radial direction per mode `m ≥ 1`) is projected out by
the analysis.

Because `J_{m−1}(α) = −J_{m+1}(α)` at the zeros of `J_m`, the combinations

```text
F̃_x = i(F̂_+ − F̂_−),   F̃_y = −(F̂_+ + F̂_−),   F̃_z = F̂_z
```

of the Hankel–Fourier coefficients are the Cartesian spectrum at the wavevector
`k̃ = (k_{m,n}, 0, k_z)` (Lehe et al., CPC 2016). Every mode `(m, k_⊥, k_z)` is therefore advanced by
the Cartesian propagators above (standard, Galilean, and averaged-Galilean PSATD with
`κ = k_z v_gal`, constant-J) with the derivative symbol `ik̃`, and vacuum dispersion `ω = c|k̃|` is
exact. `stable_step = π/(c max|k̃|)`. The Hankel basis vanishes at `r = R`, a grounded wall, so a
non-neutral charge is admissible; `initialize_field(charge)` returns the Coulomb field
`Ẽ = −ik̃ρ̂/(ε|k̃|²)` and refuses content in the unresolved modes (`m ≥ 1`, `k_⊥ = k_z = 0`). The
unpaired Nyquist entry of an even axial count is zeroed so that mode 0 stays real.
`field_energy` integrates `½∫(ε|E|² + |B|²/μ)dV` with the azimuthal weight `2π` for `m = 0` and
`π` otherwise. `circular_field(electric, magnetic)` builds a state from circular real-space modes,
and `add_propagating_field(field, electric, direction=...)` superposes a vacuum wave whose
transverse `E` is given, completing `E_z` from `ik̃·E = 0` and `B = k̃ × E/ω`.

### Deposition and gathering

`AzimuthalTransferPlan(grid, shape_order=1)` with `shape_order` 1, 2, or 3 prepares the transfer
of one particle support; particles carry Cartesian 3-D positions. A particle of amplitude `a` at
`(r, θ, z)` deposits into mode `m` with the weight `c_m a e^{−ipθ}` (`c₀ = 1`, `c_m = 2` for
`m > 0`, `p` the channel's angular order) and the tensor cardinal B-spline in `(r, z)`; nodes
below the axis fold onto their mirror nodes with parity `(−1)^p`. Node content is divided by the
near-axis-corrected volume

```text
V_j^{(q)} = 2πΔz ∫ G_j(r) (r/r_j)^q r dr,   q = |p|,
```

of the folded shape `G_j`, so the leading regular behavior `r^{|p|}` of every angular order
deposits exactly at every node. A uniform density deposits uniformly including the axis cell,
which the plain volume `2π r_j Δr Δz` overestimates by `13/12` for the linear shape (Ruyten 1993;
Verboncoeur 2001). Gathering uses the same folded shape and synthesizes `Σ_m F_m e^{ipθ}` at the
particle angle. A step fails when an active particle's stencil leaves the radial grid.

The solver deposits the endpoint modal charges and the step-mean current `q v` at the path
midpoint. Its Gauss charge is the spectral charge `ρ̂[m, k_⊥, k_z]`; the deposited velocity is the
step-mean lab velocity, so the current is the lab current in Galilean coordinates too.

### Charge conservation and compatibility

`charge_conservation` uses the Cartesian vocabulary, but only the spectral modes apply:
`"update-with-rho"` (default) and `"spectral-correction"` use only the transverse part of the
deposited current, so Gauss's law holds to roundoff for every variant and shape order.
Construction refuses:

| Configuration | Rule |
|---|---|
| `"vay-deposition"` | refused: it is a Cartesian-stencil deposition |
| `galilean_velocity` | a nonzero subluminal axial speed, required by the Galilean variants and refused by `"standard"` |
| `absorber` / `damping` | `"radial-damping"` requires a `RadialDampingPlan` thinner than the radial grid; a damping plan requires `"radial-damping"` |
| `"averaged-galilean"` | refuses radial damping and antennas (it gathers time-averaged fields) |
| Huygens observers | standard variant only; refused with antennas (a spectral sheet has current at every axial node); the side wall satisfies `radial_face ≤ N_r − thickness − 2` with the damping thickness (zero without damping), caps lie on axial nodes below `N_z`, and the exterior is the solver's vacuum |
| Antennas | medium equal to the solver's vacuum |
| Transfers | every `PreparedAzimuthalTransfer` must use the solver's grid |

### Radial damping and moving window

`absorber="radial-damping"` with `RadialDampingPlan(thickness, *, attenuation=1e-4,
profile_power=3.0)` grades `σ(r) = σ_max((r − r₀)/L)^p` over the outer `thickness` cells, with
`σ_max = −(p + 1)c ln(attenuation)/(2L)` so that a wave crossing the layer and back is attenuated
by `attenuation` in amplitude. Each step multiplies the transverse (solenoidal) `E` and `B` by
`exp(−σΔt)` in real space and re-projects the change onto solenoidal fields, so both Gauss laws
are unchanged; `QuasiCylindricalMaxwellDiagnostics.absorbed_energy` reports the removed energy.
This is a damping layer, not a radial PML.

`PICMovingWindowPlan(pic, 2, shift_cells=...)` translates the window along `z` only.
`shift_window` moves the spectra through the mixed `(k_⊥, z)` representation by whole cells,
zero-fills the leading cells, and then applies a curl-free correction restoring
`ik̃·E = (ρ + ρ_a)/ε` (the spectral divergence is nonlocal, so the truncation would otherwise
break it); the longitudinal `B` is reset to the declared magnetic-charge field.
`QuasiCylindricalMaxwellState.window_offset` accumulates the translation.

### Antennas and Huygens cylinders

`QuasiCylindricalAntennaPlan(plane_coordinate, first_coordinates, second_coordinates, times,
electric, *, magnetic=None, carrier_angular_frequency=0.0, direction="positive", medium=None,
provenance_id=None)` is a stationary one-way sheet on the plane `z = plane_coordinate` with the
conventions of the Cartesian `SampledPlaneCurrentAntennaPlan`: `electric[x, y, t, (E_x, E_y)]` is
the complex envelope of the forward-wave tangential field on a Cartesian tensor sample grid, and
`magnetic` defaults to the plane-wave relation `H' = s ẑ × E'/η`. Preparation interpolates the
samples onto `(r_j, θ_q)`, analyzes them into azimuthal modes, and injects per-mode sheet currents
`K = s ẑ × H'` and `K_m = −s ẑ × E'` as a band-limited axial delta at the antenna's window
coordinate. The declared electric and magnetic sheet charges are tracked in
`QuasiCylindricalMaxwellState.antenna_charge` and `antenna_magnetic_charge`, and the reported
constraints include them. `phydrax.optics.wave.pulse_envelope_quasi_cylindrical_antenna(field,
medium=..., provenance_id=...)` builds the plan from a `PulseEnvelopeField` whose plane normal is
the grid `z` axis.

`QuasiCylindricalHuygensPlan(radial_face, lower, upper, acquisition, exterior, *,
azimuthal_samples=32)` declares a closed cylinder: the side wall at `r = radial_face·Δr` between
the axial nodes `lower` and `upper`, and end caps on those node planes. Each patch is sampled on
`azimuthal_samples` uniform angles from the azimuthal syntheses of the modes, interpolated to the
side-wall cell centers by the four-point midpoint rule in `r` and `z`. `huygens_phasors(field)`
returns the phasors that `MaxwellFarFieldPlan` evaluates, as for the Cartesian boxes;
`QuasiCylindricalMaxwellDiagnostics.surface_current` must stay zero inside the acquisition window.

### Usage

```python
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.solver.maxwell import spectral

D = phx.discretization
grid = D.pic.QuasiCylindricalGrid(2.0, 24, 0.0, 3.0, 24, 3)  # R, N_r, z range, N_z, M + 1
transfer = D.pic.AzimuthalTransferPlan(grid, shape_order=1)

count = 32
rng = np.random.default_rng(3)
radius = 1.2 * np.sqrt(rng.uniform(0.0, 1.0, count))
angle = rng.uniform(0.0, 2.0 * np.pi, count)
position = np.stack(
    (radius * np.cos(angle), radius * np.sin(angle), rng.uniform(0.0, 3.0, count)), -1
)

species, transfers = [], []
for offset, specific, name, mass in ((0, -1.0, "electrons", 1.0), (10**7, 0.01, "ions", 100.0)):
    support = D.ParticleSetPlan(
        jnp.arange(offset, offset + count), mass * jnp.ones((count,)), ambient_dimension=3
    ).prepare()
    species.append(
        D.pic.PICSpeciesPlan(
            D.ParticlePopulationPlan(support),
            D.pic.PICChargeModelPlan(
                specific,
                name,
                minimum_charge_number=1,
                maximum_charge_number=1,
                initial_charge_number=1,
            ),
        )
    )
    transfers.append(transfer.prepare(support))

solver = spectral.QuasiCylindricalMaxwellPlan(
    grid, charge_conservation="spectral-correction"
).prepare(tuple(transfers))
pic = phx.solver.ElectromagneticPICPlan(solver, species=tuple(species))
dt = 0.5 * float(solver.stable_step)
velocity = rng.normal(0.0, 0.2, (count, 3))
state = pic.initialize((position, position), (velocity, np.zeros((count, 3))), dt)
result = pic.step_detailed(state, dt)
```

`examples/quasi_cylindrical_lwfa.py` runs a laser-wakefield configuration, and
`benchmarks/quasi_cylindrical_pic.py` records the phase-separated cost of the Hankel preparation,
one vacuum advance, and one PIC step over a sweep of radial and mode counts (`--small` for smoke
sizes).

### FBPIC oracle

`fbpic_laser_wakefield(provider, grid, *, density, a0, wavelength, waist, length, center,
plasma_lower, plasma_upper, plasma_radius, particles_per_cell, time_step, steps, timeout=3600.0)`
runs FBPIC (Lehe et al., CPC 2016) as an external oracle through a caller-pinned Python interpreter
(`FBPICProvider(executable)` with a `PinnedExecutable`, run by `run_pinned_command`); nothing is
imported into the Phydrax process. The configuration is declared in plasma units (lengths in
`c/ω_p`, times in `1/ω_p`, fields in `m_e c ω_p/e`) and mapped onto FBPIC's SI inputs at the
declared electron `density`: infinite-order PSATD with the curl-free current correction, linear
shapes, no current filter, periodic `z`, a reflective radial wall, an immobile neutralizing ion
background, and FBPIC's `GaussianLaser` at focus. The grid's axial nodes are FBPIC's cell
centers. `FBPICWakefieldResult` holds FBPIC's mode-0 `E_z[z, r]` in plasma units with its axial
and radial coordinates. The cross-code test skips unless `PHYDRAX_FBPIC_PYTHON` names the
interpreter and `PHYDRAX_FBPIC_PYTHON_VERSION` its release.

### Measured evidence

- Vacuum TM modes `m = 0, 1, 2` match the exact traveling cylindrical modes to `5e-15` after 20
  steps at `0.9 stable_step`.
- An `m = 1` Gaussian antenna beam (`w₀ = 1.5λ`) follows the paraxial waist `w(z)` within 0.5%,
  the on-axis amplitude within 3% (the `O(θ²)` nonparaxial correction), and the Gouy phase within
  `0.005` rad; the field behind the one-way sheet is 0.5% of the launched amplitude.
- PIC Gauss residuals are of order `1e-15` for `"update-with-rho"`, `"spectral-correction"`,
  Galilean, averaged-Galilean, and shape orders 1–3.
- A uniform density deposits within 0.2% at the axis node (plain volumes give `13/12`).
- An axisymmetric pulse equals the 3-D Cartesian PSATD solution to `2.6e-7`.
- The Hertzian-dipole far field through a Huygens cylinder is within 0.7%, 1.2%, and 1.9% at 9,
  7.5, and 6 cells per wavelength.
- Radial damping leaves 1.9% of an outgoing pulse's energy, with the energy ledger closing to
  `3e-5`.
- A linear laser wakefield (`a₀ = 0.5`, `k₀ = 5k_p`) matches FBPIC 0.27.0's on-axis wake to 5%
  relative `L²` over the wake.

### Validity and nonclaims

- No radial PML: the outer boundary is a grounded wall, optionally with radial damping.
- The axial direction is periodic; the moving window zero-fills the cells it exposes.
- Antennas are stationary in the lab frame; moving antennas are not provided.
- Constant-J only: no multi-J or linear-J time dependencies.
- Infinite-order spectral derivatives only: no finite-order stencils or local-guarded transforms.
- Single device: no distributed execution.

## Validity domains and limits

- PSATD integrates vacuum Maxwell exactly for any step. `stable_step = π/(c max|[k]|)` is the
  largest step whose highest grid modes do not alias in time, not a CFL limit; the PIC runtime
  gates each step against it.
- A vacuum plane wave keeps roundoff phase error (about `3e-15` over 20 steps) at infinite order
  on collocated and staggered grids; finite orders 2 and 4 follow the modified wavenumber
  `ω = c|[k]|` (`dispersion_frequency`).
- Collocated grids cannot resolve the checkerboard charge modes whose every axis index is `0` or
  Nyquist (`[k]² = 0` there). The solver's charge layout projects them out and reports their
  content as `SpectralMaxwellDiagnostics.unresolved_charge`; staggered grids resolve every nonzero
  mode.
- `"local-guarded"` with `"update-with-rho"` is local only for currents consistent with the
  finite stencil; use `"vay-deposition"`.
- Huygens observers and the PML are standard-only, and the PML region must stay current-free.
  PML layers carry declared absorber charge; the Gauss law is exact outside
  `absorber_support`.
- Spectral antennas need at least eight cells per emitted wavelength along the normal and refuse
  Huygens observers in the same run.
- `"averaged-galilean"` refuses `"linear-j"`.
- Creation-stage PIC processes (strong-field QED) run on Galilean grids in grid coordinates:
  process-owned photons drift at `c k̂ − v_grid` (`PICProcessContext.grid_velocity`), created
  pairs drift like every species, and all quantum parameters use lab-frame fields and momenta.
- Quasi-cylindrical PSATD limits are listed under
  [Validity and nonclaims](#validity-and-nonclaims).
- Boosted-frame PIC is exact Lorentz kinematics on a periodic box: the boosted grid holds the
  Doppler-stretched laser on the initial slice, the contracted lab domain, and the path of every
  tracked particle (no moving window or continuous injection; the Galilean grid keeps the lab
  plasma boundary stationary). Beams are loaded ballistically (free streaming to the slice),
  vacuum fields must not overlap particles, and back-transformed fields carry the second-order
  error of linear time interpolation between steps. Boosted Huygens far fields need a standard
  grid, `J = 0` on the surface, vacuum, and complete emission.

## References

- I. Haber, R. Lee, H. H. Klein, J. P. Boris, "Advances in electromagnetic simulation
  techniques," Proc. Sixth Conf. Numerical Simulation of Plasmas, 46 (1973).
- W. M. Ruyten, "Density-conserving shape factors for particle simulations in cylindrical and
  spherical coordinates," J. Comput. Phys. 105, 224 (1993).
- J. P. Verboncoeur, "Symmetric spline weighting for charge and current density in particle
  simulation," J. Comput. Phys. 174, 421 (2001).
- J.-L. Vay, I. Haber, B. B. Godfrey, "A domain decomposition method for pseudo-spectral
  electromagnetic simulations of plasmas," J. Comput. Phys. 243, 260 (2013).
- B. B. Godfrey, J.-L. Vay, I. Haber, "Numerical stability analysis of the pseudo-spectral
  analytical time-domain PIC algorithm," J. Comput. Phys. 258, 689 (2014).
- R. Lehe, M. Kirchen, I. A. Andriyash, B. B. Godfrey, J.-L. Vay, "A spectral, quasi-cylindrical
  and dispersion-free Particle-In-Cell algorithm," Comput. Phys. Commun. 203, 66 (2016).
- R. Lehe, M. Kirchen, B. B. Godfrey, A. R. Maier, J.-L. Vay, "Elimination of numerical Cherenkov
  instability in flowing-plasma particle-in-cell simulations by using Galilean coordinates,"
  Phys. Rev. E 94, 053305 (2016).
- M. Kirchen, R. Lehe, B. B. Godfrey, I. Dornmair, S. Jalas, K. Peters, J.-L. Vay, A. R. Maier,
  "Stable discrete representation of relativistically drifting plasmas," Phys. Plasmas 23,
  100704 (2016).
- O. Shapoval, J.-L. Vay, H. Vincenti, "Two-step perfectly matched layer for arbitrary-order
  pseudo-spectral analytical time-domain methods," Comput. Phys. Commun. 235, 102 (2019).
- O. Shapoval, R. Lehe, M. Thévenet, E. Zoni, Y. Zhao, J.-L. Vay, "Overcoming timestep
  limitations in boosted-frame particle-in-cell simulations of plasma-based acceleration,"
  Phys. Rev. E 104, 055311 (2021).
- O. Shapoval, E. Zoni, R. Lehe, M. Thévenet, J.-L. Vay, "Pseudospectral particle-in-cell
  formulation with arbitrary charge and current-density time dependencies for the modeling of
  relativistic plasmas," Phys. Rev. E 110, 025206 (2024).
- J.-L. Vay, "Noninvariance of space- and time-scale ranges under a Lorentz transformation and
  the implications for the study of relativistic interactions," Phys. Rev. Lett. 98, 130405
  (2007).
