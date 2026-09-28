# Charged-particle radiation

`phydrax.electromagnetics` computes the radiation of prescribed charged-particle
trajectories in vacuum. A trajectory is a set of sampled lanes; the far-field
spectrum is evaluated per observer direction and angular frequency, the
near-zone Liénard–Wiechert field per observer event, and every result carries
evidence of whether the sampling resolves it. Fields in media and
self-consistent particle–field runs belong to other owners
(`phydrax.solver.maxwell`, particle-in-cell).

## Conventions

Every radiation route in Phydrax uses the same Fourier convention:

- phasors `exp(−iωt)`;
- transient spectra `F(ω) = ∫ f(t) exp(+iωt) dt` over observer time;
- one-sided spectral energy `d²W/(dω dΩ) = ε₀ c |r Ẽ|² / π` for `ω > 0`.

Observer time is `τ = t − n·r(t)/c` (the constant `R/c` is dropped), and the far
field is the acceleration form of the Liénard–Wiechert field,

```text
r Ẽ(ω) = q / (4π ε₀ c) ∫ d/dt [ n × (n × β) / κ ] exp(iωτ(t)) dt ,   κ = 1 − n·β .
```

`q`, `ε₀`, and `c` come from the `ElectromagneticScaleContract` bound to the
plan, so SI and declared code units follow the same code path. `κ` is evaluated
as `(1/γ² + |n × β|²)/(1 + n·β)`, which keeps full relative precision at
`γ = 10⁴` where `1 − n·β` would lose half of the float64 digits. Radiation
phases are float64 only: float32 trajectories are refused.

## Trajectories and observers

```python
import numpy as np
import phydrax as phx
from phydrax.electromagnetics import (
    ChargedTrajectory,
    RadiationObserverPlan,
    TrajectoryRadiationPlan,
)

scale = phx.ElectromagneticScaleContract.si()
trajectory = ChargedTrajectory(
    times,  # [T] shared or [T, P] per lane, float64 seconds
    positions,  # [T, P, 3] metres
    proper_velocities,  # [T, P, 3] u = γ v, metres per second
    charges,  # [P] coulombs per particle
    multiplicities,  # [P] identical particles per lane
    active,  # [T, P] bool
    (id_hi, id_lo),  # [P] uint32 persistent identities
)
observers = RadiationObserverPlan(directions, np.array([0.0, 0.0, 1.0]))
plan = TrajectoryRadiationPlan(
    scale, observers, angular_frequencies, coherence="coherent", route="segment-exact"
)
result = plan.prepare().evaluate(trajectory)
```

The polarization basis of direction `n` is `e1 = normalize(a − (a·n) n)` and
`e2 = n × e1` for the reference axis `a`; directions parallel to `a` are refused.
`field_spectrum[F, D, 2]` holds `r Ẽ` in `(e1, e2)`, `coherency[F, D, 2, 2]` the
matrix `⟨R_i R_j*⟩`, and `stokes[F, D, 4]` holds `I`, `Q = |R1|² − |R2|²`,
`U = 2 Re(R1 R2*)`, `V = −2 Im(R1 R2*)`. A charge circling counterclockwise about
`+z` and seen from `+z` gives `V = +I`.

Detector tracks propagated by `phydrax.applications.detector.propagate_charged_tracks`
become lanes through `phydrax.applications.detector.charged_trajectory`, which
prepends the initial sample, keeps `(event_id, track_id)` identities and the
per-step activity, and refuses a pusher whose units or speed of light differ from
`scale` (see [Detector and calorimeter production](guides_detector_calorimetry.md)).

Phydrax PIC runs record lanes in place with
`phydrax.discretization.pic.PICTrackRecorder`, which follows persistent particle
identities through slot reuse and migration, samples time-centered proper
velocities, converts its ring with `to_charged_trajectory(state, scale)`, and can
stream the same samples into `initialize`/`accumulate` inside the run without
stored tracks (see [Particle-in-cell methods](guides_particle_in_cell.md)).

## openPMD particle tracks

Particle output of external PIC codes and beam simulations enters as openPMD
1.1.0 HDF5 species tracks, and trajectories leave the same way:

```python
from phydrax._external_resource import ResourceLimits, read_bounded_resource
from phydrax.interchange import (
    OpenPMDParticleTrackImportPolicy,
    OpenPMDParticleTrackSelection,
    read_openpmd_particle_tracks_hdf5,
    write_openpmd_particle_tracks_hdf5,
)

limits = ResourceLimits(256 * 2**20, 16, 10_000_000, 100_000, 1)
resource = read_bounded_resource("tracks.h5", trusted_root=run_directory, limits=limits)
imported = read_openpmd_particle_tracks_hdf5(
    resource,
    OpenPMDParticleTrackImportPolicy(
        OpenPMDParticleTrackSelection("electrons", iterations=range(0, 4000, 10))
    ),
    scale=scale,
)
result = plan.prepare().evaluate(imported.trajectory)

write_openpmd_particle_tracks_hdf5(
    "lanes.h5", trajectory, masses, scale=scale, species="electrons", limits=limits
)
```

Particles are followed by their openPMD `id` alone and become lanes in ascending
identity order; `weighting` becomes the lane multiplicity, and `momentum / mass`
(per particle, through the ED-PIC `macroWeighted`/`weightingPower` attributes)
the proper velocity. Record units are converted through `unitSI` and checked
through `unitDimension` against the plan's `ElectromagneticScaleContract`, so a
code-unit file radiates identically in SI or in any declared code units. Missing
or repeated identities, nonmonotonic times, truncated records, unit mismatches,
and resource overflow are refused with an `AdapterReport` before a spectrum is
computed (see [API → Particle-track HDF5 profile](api/interchange.md#particle-track-hdf5-profile)).

## Routes

The trajectory samples are nodes; between nodes the routes differ.

| Route | Between nodes | Accuracy | Cost |
|---|---|---|---|
| `"segment-exact"` | constant `β` (average of the node proper velocities) | second order in `ωΔτ` | `T·F·D` exponentials |
| `"segment-hermite"` | quintic Hermite position, cubic Hermite `u` | fourth order; requires `proper_accelerations` | `T·Q·F·D` |
| `"node-gridded"` | constant `β`, as `"segment-exact"` | same model, plus a reported gridding floor | one Type-3 NUFFT per direction |

With constant `β` per segment, the acceleration is a set of jumps `Δa_j` of
`a = n × (n × β)/κ` at interior nodes and the spectrum is exactly
`Σ_j Δa_j exp(iωτ_j)`. `"segment-exact"` evaluates it in the velocity form
`−iω Σ_j a_j Δτ_j exp(iωτ_mid) sinc(ωΔτ_j/2)` plus the boundary terms of each
active run; `"node-gridded"` evaluates the jump sum with the Type-3 nonuniform
Fourier transform of `phydrax._spectral` over the declared
`observer_time_window`, reporting the absolute floor `ε · |q m| Σ|Δa_j|` for the
requested `TrajectoryRadiationResources.gridded_tolerance`. Its difference from
`"segment-exact"` stays inside that floor at every frequency, including the
exponential synchrotron tail.

Window edges and activity changes carry no jump: a lane radiates as if it moved
uniformly before its first and after its last active sample. When the window
does not hold the complete emission, declare `emission="truncated"`; with the
default `"complete"`, acceleration at the window edge is reported as
`WINDOW_EDGE_ACCELERATION` and the result is not `resolved`. Only complete
emission may be moved between Lorentz frames with
`phx.transform_spectral_energy`.

## Periodic motion and harmonics

For one period of periodic motion sampled so that the interior jumps cover
exactly one period (nodes at half steps, `t_j = (j − 1/2) dt`, `j = 0 … n + 1`),
the spectrum at `m ω₀` is the Fourier coefficient of the received field and the
power per solid angle in harmonic `m` is

```text
dP_m/dΩ = 2π S(m ω₀) / T₀² .
```

`examples/trajectory_cyclotron_synchrotron_radiation.py` uses this to show the
transition from the cyclotron line (`β = 0.01`: Larmor power, second harmonic
`(12/5) β²` of the first, circular polarization on the axis) through Schott
harmonics (`γ = 3`) to the synchrotron comb (`γ = 30`), whose harmonic powers
follow `√3 q² γ F(ω/ω_c) ω₀/(8π² ε₀ R)` with `ω_c = (3/2) γ³ ω₀` and `F` from
`phx.special.synchrotron_f`.

## Coherence

| `coherence` | Coherency |
|---|---|
| `"coherent"` | `|Σ_p m_p R_p|²` |
| `"incoherent"` | `Σ_p m_p |R_p|²` |
| `"gaussian-form-factor"` | `(1 − |F|²) Σ_p m_p |R_p|² + |F|² |Σ_p m_p R_p|²`, `|F|² = exp(−ω² Σᵢ nᵢ² σᵢ² / c²)` from `bunch_sigma` |
| `"tabulated-form-factor"` | the same with a declared `form_factor[F]` magnitude |

`m_p` is the lane multiplicity. The coherent model keeps no per-lane spectra;
the others store them in the streaming state, which the resource estimate
accounts for. The observer-time waveform, `prepared.waveform(result, times)`,
exists only for coherent results.

## Evidence

`result.evidence` reports:

- `status` (`TrajectoryRadiationStatus`) bits `UNRESOLVED_PHASE` (`ω Δτ` of a
  segment above one radian, or above the Hermite quadrature order),
  `UNRESOLVED_AMPLITUDE` (one jump above half of the peak amplitude: the pulse
  itself is not sampled), `WINDOW_EDGE_ACCELERATION`, `ACTIVITY_TRANSITION`,
  `UNSUPPORTED_NODE` (a node outside the gridded window), `NONMONOTONE_TIME`,
  `LANE_MISMATCH` (streaming chunks with different identities or charges), and
  `NONFINITE`;
- `resolved`, `finite`, and `derivative_valid[F, D]`, which is false at
  unresolved frequencies and everywhere after an activity transition, a
  discrete event;
- `minimum_retardation_factor`, `maximum_phase_increment`,
  `maximum_relative_amplitude_increment`, `window_edge_rate`, `segments_used`,
  `gridded_error_floor[D]`, the gridded transform evidence, and the static
  `resource_estimate`.

Periodic lanes are integrated spectrally accurately even when the receding half
of an orbit exceeds one radian per step; the flag still reports it, because a
transient lane sampled that coarsely would alias.

## Streaming and execution

`prepared.initialize(trajectory)`, `prepared.accumulate(state, chunk)`, and
`prepared.finalize(state)` fold consecutive time chunks of the same lanes. The
first sample of a chunk continues the lane from the carried last sample of the
previous chunk, so the streamed result equals `evaluate` on the whole
trajectory, for every route and coherence model. Lanes execute in `lax.scan`
chunks of `particle_chunk` vectorized lanes and segments in blocks of
`segment_block`; the working set and streaming state are estimated before
execution and refused above `maximum_working_bytes` and `maximum_state_bytes`
with `TrajectoryRadiationResourceError`. Each call is traceable: wrap
`prepared.evaluate` or `prepared.accumulate` in `jax.jit` or
`equinox.filter_jit` at the call site, and differentiate spectra with respect to
trajectory samples with `jax.jvp` or `jax.vjp`.

## Validation

`tests/unit/electromagnetics/test_trajectory_radiation.py` checks uniform motion
(zero field), the Liénard–Larmor power at `β = 0.01` (`10⁻⁴`), Schott harmonics
at `γ = 10` (`10⁻³`, `phx.special.jv`), the harmonic sum against the Liénard
total at `γ = 10` with the synchrotron tail (`2·10⁻⁴`), polarization, coherence
scalings, 50-digit retardation factors at `γ = 10⁴`, every evidence flag,
streaming = offline, JVP/VJP against finite differences, refusals, per-lane
times, gridded = exact within the reported floor, fourth-order Hermite
convergence, boost-then-radiate = radiate-then-transform, and the observer-time
waveform against the Liénard–Wiechert acceleration field.
`tools/electromagnetic_radiation_qualification.py [--smoke]` runs the
closed-form gates of the `electromagnetics.vacuum-trajectory-radiation`
candidate profile, and `benchmarks/trajectory_radiation.py` records
phase-separated lowering, compilation, execution, and memory for both segment
routes as the frequency count grows.

## Near-zone Liénard–Wiechert fields

`LienardWiechertFieldPlan` evaluates the complete retarded field of point
charges — velocity (Coulomb) and acceleration (radiation) parts — at observer
events `(t, x, y, z)`, at any distance:

```python
from phydrax.electromagnetics import LienardWiechertFieldPlan

plan = LienardWiechertFieldPlan(
    scale,
    history="refuse",
    exclusion_radius=1.0e-6,
    interpolation="hermite-quintic",
)
result = plan.prepare().evaluate(trajectory, observer_events)  # events [O, 4]
result.electric_field, result.magnetic_field  # [O, 3]
result.velocity_field, result.acceleration_field  # E = velocity + acceleration
result.retarded_times  # [O, P]
```

```text
E = q/(4π ε₀) [ (n − β)(1 − β²)/(κ³ R²) + n × ((n − β) × β̇)/(c κ³ R) ],   B = n × E / c
```

with `R`, `n`, `β`, `β̇` at the retarded time and `κ` in the same
cancellation-free form as the far-field routes. The retarded condition
`t − t_r − |x − r(t_r)|/c = 0` is strictly decreasing along a subluminal
lane, so the root is unique: a fixed `⌈log₂(T − 1)⌉`-step bisection over sample
indices finds its segment, and the native bracketed `scalar_root` (TOMS748)
solves the residual on the Hermite segment to `10⁻¹³` of the step in observer
time. The residual is written relative to the segment's first sample, so far
observers keep full precision. Derivatives of the root come from the implicit
function theorem (`lax.custom_root`), so fields are differentiable with
respect to observer events and trajectory samples.

`interpolation="hermite-cubic"` needs only positions and proper velocities: the
cubic position gives the retarded point, `β`, and `β̇` (second order in the
acceleration field) and the proper-velocity chord gives `1/γ²`. Because `β̇`
differentiates rounded positions twice, its noise grows like `ε N² γ²` for `N`
samples; use it for moderate `γ`. `"hermite-quintic"` also uses proper
accelerations: the quintic position gives the retarded point and the cubic
Hermite proper velocity gives `β`, `β̇`, and `1/γ²` exactly consistently
(fourth order, well conditioned at large `γ`).

Retarded times before the first sample are `RETARDED_BEFORE_WINDOW`:
unsupported under `history="refuse"`, and continued with the first sample's
uniform motion (closed-form root) under `"inertial-extrapolation"`. Retarded
times after the last sample (`RETARDED_AFTER_WINDOW`) are always unsupported.
Unsupported observers have NaN fields and NaN retarded times, never zeros.
Lane samples must advance in time and stay causal (`|Δr| < c Δt`) across
inactive samples too, otherwise the lane is `NONMONOTONE_TIME` or
`SUPERLUMINAL_SAMPLES` and unsupported. A retarded point in an inactive segment
(`INACTIVE_RETARDED`) contributes no field and its charge is reported as
`absent_charge`, which sums to zero for charge-conserving creation or
annihilation. Pairs whose retarded distance is below `exclusion_radius` are
excluded (`EXCLUDED_CHARGE`) and accounted in `excluded_charge` and
`excluded_count`: the point-charge field is not claimed inside that radius.

`result.evidence` (`LienardWiechertEvidence`) holds per-observer `status`
(`LienardWiechertStatus`), per-pair `pair_status[O, P]`, `supported`, `finite`,
`resolved` (no `UNRESOLVED_KINEMATICS`: the position interpolant's `1 − β²`
agrees with the proper velocity's `1/γ²` to 1%), `derivative_valid` (false
where an exclusion or activity boundary decided a contribution),
`minimum_retardation_factor`, `maximum_root_residual` (seconds of observer
time), `maximum_lorentz_mismatch`, `extrapolated_count`, and the static
`resource_estimate`. Observers run in `lax.map` blocks of
`LienardWiechertResources.observer_chunk` and lanes in `lax.scan` chunks of
`particle_chunk`; working set and outputs above the declared limits raise
`LienardWiechertResourceError` before execution.

`examples/lienard_wiechert_near_field.py` compares a uniformly moving charge
with Heaviside's field and a hyperbolic worldline with Born's closed form, and
`benchmarks/lienard_wiechert_fields.py` records phase-separated lowering,
compilation, execution, and memory as the observer count grows.
`tests/unit/electromagnetics/test_lienard_wiechert.py` checks Heaviside's
field (including `γ = 10⁴`), Born's hyperbolic-motion field, second- and
fourth-order convergence, the far-zone limit against the observer-time
waveform, Gauss's law on a sphere, history, causality, exclusion and activity
evidence, chunk invariance, JVP/VJP, and refusals.

## Not claimed

No medium (`n ≠ 1`) or self-consistent radiation reaction: these are the
Maxwell far-field and particle-in-cell owners. Near-zone fields are point-charge
fields outside the declared exclusion radius only. Quantum recoil and photon
statistics are outside the classical spectrum; `ħω` against the electron
energy is the caller's validity check.
