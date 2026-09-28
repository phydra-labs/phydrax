# Plasma waves and emission

This guide covers wave propagation in cold magnetized plasmas with
`phydrax.electromagnetics.ColdPlasmaDielectric`: the Stix dielectric, the two
refractive-index roots with explicit branch identity, cutoffs and resonances,
mode polarization, and Faraday rotation and conversion. The runnable version is
`examples/cold_plasma_waves.py`; the API reference is
[Electromagnetics](api/electromagnetics.md). Hot plasmas — the kinetic
dielectric, complex dispersion roots and cyclotron-maser growth — follow in
[Hot plasmas: kinetic dielectric](#hot-plasmas-kinetic-dielectric), with the
runnable `examples/kinetic_plasma_dispersion.py`.

## Conventions

Phasors carry `exp(−iωt)`. For species `s` with number density `n_s`, charge
`Z_s e`, and mass `m_s`, the plasma frequency is `ω_ps² = n_s Z_s² e²/(ε₀ m_s)`
and the signed cyclotron frequency is `Ω_s = Z_s e |B₀|/m_s` (negative for
electrons). The Stix frame has `B₀ ∥ ẑ`; the wave normal
`κ̂ = (sin θ, 0, cos θ)` lies in the `x–z` plane. The Stix parameters are

- `R = 1 − Σ ω_ps² / (ω (ω + iν_s + Ω_s))`,
- `L = 1 − Σ ω_ps² / (ω (ω + iν_s − Ω_s))`,
- `P = 1 − Σ ω_ps² / (ω (ω + iν_s))`,
- `S = (R + L)/2`, `D = (R − L)/2`.

A collision frequency `ν_s` enters each species' momentum equation as
`ω → ω + iν_s`, so collisions make the Stix parameters complex with
`Im n² > 0` meaning absorption. All physical constants come from the bound
`ElectromagneticScaleContract`: densities are per cubic length unit of the
scale, masses are in electron masses, charges in elementary charges, `B₀` in
the scale's field unit, and `ω`, `ν_s` per time unit.

```python
import jax.numpy as jnp

import phydrax as phx
from phydrax.electromagnetics import ColdPlasmaDielectric, PlasmaWaveMode

plasma = ColdPlasmaDielectric(
    phx.ElectromagneticScaleContract.si(),
    densities=[1.0e11, 1.0e11],
    charge_numbers=[-1.0, 1.0],
    mass_ratios=[1.0, 1836.152673426],
    magnetic_field=[0.0, 0.0, 5.0e-5],
    collision_frequencies=[1.0e3, 10.0],
)
omega = 3.0e6  # rad/s, below the electron cyclotron frequency (whistler band)
stix = plasma.stix_parameters(omega)
epsilon = plasma.dielectric_tensor(omega)  # [..., 3, 3] in the Stix frame
```

Angular frequencies and angles must be float64; float32 or complex arguments
are refused rather than silently promoted.

## Refractive indices and branch identity

The wave operator is `n² (κ̂κ̂ − I) + ε`; its determinant is the Stix
biquadratic `A n⁴ − B n² + C = 0` with `A = S sin²θ + P cos²θ`,
`B = RL sin²θ + PS (1 + cos²θ)`, `C = PRL`. `refractive_indices(ω, θ)` returns
both roots, broadcast over `ω` and `θ`, along a trailing root axis of length 2.
Roots are carried internally as homogeneous pairs so that resonances
(`n² = ∞`) and cutoffs (`n² = 0`) are ordinary points.

Branch names are not assigned from the sign of the discriminant at one angle.
Each root is continued in `θ` along two paths:

- from `θ = 0`, where the roots are exactly `n² = R` and `n² = L`, giving
  `parallel_mode` ∈ {`RIGHT`, `LEFT`};
- from `θ = π/2`, where the roots are exactly `n² = P` (ordinary) and
  `n² = RL/S` (extraordinary), giving `perpendicular_mode` ∈
  {`ORDINARY`, `EXTRAORDINARY`}.

The discriminant is evaluated as
`F² = (RL − PS)² sin⁴θ + 4P²D² cos²θ`, which keeps the splitting of nearly
coincident roots accurate. The branch `(B + F)/(2A)` is continuous in `θ`
whenever `F` is (as a homogeneous pair it passes through resonances), so the
continuation follows the pole-free scalar `F` from its signed closed forms,
`F = 2PD` at `θ = 0` (branch `R`) and `F = RL − PS` at `θ = π/2` (branch
`RL/S`), across `continuation_steps` samples. The result reports, per path,
the minimum relative root separation `|n₊² − n₋²| / (|n₊²| + |n₋²|)`
(`parallel_branch_separation`, `perpendicular_branch_separation`) and the
largest per-step turn `|Δ arg F²|/π` (`parallel_branch_turn`,
`perpendicular_branch_turn`). A label is certified when the turn is below
`1/2`; otherwise `PARALLEL_LABEL_AMBIGUOUS` or `PERPENDICULAR_LABEL_AMBIGUOUS`
is set and the continuation should be refined. In a collisionless plasma
`F²` is real and nonnegative, so labels are certified unless the roots
coincide; with collisions `F²` is complex and a near-crossing of the two modes
appears as a large turn. Labels come from both endpoints, so one root carries
two names: above every cutoff the `RIGHT` root is `EXTRAORDINARY` and the
`LEFT` root is `ORDINARY`.

```python
theta = jnp.linspace(0.0, 1.2, 7)
wave = plasma.refractive_indices(omega, theta)
whistler_n2 = wave.select(PlasmaWaveMode.RIGHT, wave.n_squared)
ordinary_e = wave.select(PlasmaWaveMode.ORDINARY, wave.polarization)
```

`refractive_index` is the principal square root of `n²`. Each root carries a
`ColdPlasmaWaveStatus` word:

| Bit | Meaning |
|---|---|
| `EVANESCENT` | `Re n² < 0`; the root does not propagate |
| `RESONANT` | `n² = ∞` (for example on the resonance cone) |
| `ROOT_DEGENERATE` | the biquadratic has no defined root in this slot |
| `PARALLEL_LABEL_AMBIGUOUS`, `PERPENDICULAR_LABEL_AMBIGUOUS` | continuation evidence fails |
| `POLARIZATION_UNDEFINED` | the wave operator has more than a one-dimensional null space |
| `NONFINITE` | non-finite root data |

In an unmagnetized plasma both roots equal `1 − Σ ω_ps²/ω²`; the result
reports them as coincident with undefined polarization and ambiguous labels
instead of inventing a split.

## Polarization

`polarization` is the unit null vector of the wave operator in the Stix frame,
with its largest component real and positive. `transverse_ratio` is Stix's
`K = i E_x / E_y` (`+1` for the right-circular mode along `B₀`, `−1` for the
left-circular one), and `longitudinal_component` is `κ̂ · E`, nonzero for the
extraordinary mode across `B₀` and near resonances. `polarization_margin`
reports how well separated the null space is from the rest of the operator.

## Cutoffs and resonances

`characteristic_frequencies()` returns every zero of `R`, `L`, `P` (cutoffs)
and `S` (hybrid resonances), including negative and complex frequencies,
together with the magnitude of the Stix parameter re-evaluated at each zero.
For an electron plasma the positive zeros are
`ω_R,L = (±|Ω_e| + √(Ω_e² + 4ω_pe²))/2`, `ω_pe`, and the upper hybrid
frequency `√(ω_pe² + Ω_e²)`; a multi-species plasma adds lower hybrid and
ion-cyclotron structure. When the plasma is charge neutral and
collisionless, the `ω = 0` pole of `R` and `L` cancels across species and the
corresponding spurious zero is removed. Collisions move the zeros off the real
axis.

`resonance_cone(ω)` returns `tan²θ_res = −P/S`; `exists` holds where this is
real and positive, for example for whistlers below the electron cyclotron
frequency, and beyond the cone angle the whistler branch is evanescent.

## QL/QT regimes

The discriminant of the biquadratic splits as
`(RL − PS)² sin⁴θ + 4P²D² cos²θ`. The result reports both parts as
`quasi_transverse_term` and `quasi_longitudinal_term`. When the
quasi-longitudinal part dominates, the roots approach
`n² ≈ 1 − X/(1 ± Y cos θ)` for an electron plasma with `X = ω_pe²/ω²`,
`Y = |Ω_e|/ω`; when the quasi-transverse part dominates they approach the
ordinary and extraordinary forms `1 − X` and `1 − X(1 − X)/(1 − X − Y² sin²θ)`.

## Faraday rotation and conversion

`faraday_coefficients(ω, θ)` builds the Poincaré-sphere rotation generator
`ρ = (ρ_Q, ρ_U, ρ_V)` per unit length from the two mode indices and the
transverse Stokes vectors of their polarizations, in the basis
`ê₁ = (cos θ, 0, −sin θ)` (in the `k–B₀` plane) and `ê₂ = ŷ`. The Stokes vector
obeys `d(Q, U, V)/ds = 2 Re ρ × (Q, U, V)`.

- Along `B₀`, `ρ_V = (ω/2c)(n_L − n_R)` is the rotation rate of the plane of
  linear polarization. In the high-frequency limit it tends to
  `e³ n_e B/(2ε₀ m_e² c ω²)`.
- Across `B₀`, `ρ_Q = (ω/2c)(n_X − n_O)` is Faraday conversion between
  linear and circular polarization, `≈ −(ω/4c) X Y²` at high frequency.

With collisions the modes are no longer orthogonal; `mode_overlap` reports the
departure and the imaginary parts of `ρ` are half the differential attenuation
between the modes.

## Hot plasmas: kinetic dielectric

`KineticPlasmaDielectric` replaces the cold Stix tensor by the linear Vlasov
response of drifting bi-Maxwellian species. Temperatures are `k_B T` in the
scale's energy unit and drifts are along `B₀`; everything else matches
`ColdPlasmaDielectric`, and as `T → 0` the tensor reduces to the cold one.

```python
from phydrax.electromagnetics import KineticDispersionProblem, KineticPlasmaDielectric

charge = float(phx.ElectromagneticScaleContract.si().elementary_charge)
hot = KineticPlasmaDielectric(
    phx.ElectromagneticScaleContract.si(),
    densities=[1.0e18],
    charge_numbers=[-1.0],
    mass_ratios=[1.0],
    magnetic_field=[0.0, 0.0, 0.5],
    parallel_temperatures=[100.0 * charge],
    perpendicular_temperatures=[100.0 * charge],
    harmonic_count=8,
)
# (ω, k∥, k⊥): complex ω is allowed; Im ω < 0 is the analytic continuation.
chi = hot.susceptibility(2.0e11 - 1.0e9j, 1.0e3, 5.0e2)
```

For each harmonic `n` the parallel velocity integral is a moment of the plasma
dispersion function `Z(ζ_n)`, `ζ_n = (ω − k∥V − nΩ)/(k∥ w∥)`, with
`w∥ = √(2T∥/m)`, and the Larmor-radius dependence is `e^{−λ}I_n(λ)`,
`λ = k⊥²T⊥/(mΩ²)`. `Z = i√π w` with the Faddeeva function `w`, which is
entire: damped modes (`Im ω < 0`) are evaluated on Landau's contour without
any special casing. When `k∥ < 0` the causal integral is `sgn(k∥) Z(sgn(k∥) ζ)`;
for a drift-free plasma this makes `χ(−k∥)` the mirror image of `χ(k∥)` under
`z → −z`. Exactly perpendicular propagation (`k∥ = 0`, Bernstein waves) and
parallel propagation (`k⊥ = 0`) are exact limits, not special inputs.

Harmonics `|n| ≤ N` are retained; the pair `±(N + 1)` is evaluated as well and
its size relative to the retained susceptibility is reported as
`truncation_ratio`, with `HARMONIC_TRUNCATION` set above
`truncation_tolerance`. Raise `harmonic_count` until the flag clears; large
`λ` (hot, strongly perpendicular waves) needs `N ≳ λ + a few √λ`.

### Weakly relativistic (Shkarofsky) option

Near electron-cyclotron harmonics at nearly perpendicular incidence the
relativistic mass shift, not Doppler broadening, sets the resonance width.
`model="weakly-relativistic"` expands `γ ≈ 1 + u²/2` in the resonance for
isotropic, drift-free Maxwellians and keeps the lowest Larmor-radius order of
each harmonic. The momentum integrals are Shkarofsky functions
`F_q(z, a)` with `z = μ(1 − nΩ/ω)`, `a = μN∥²/2`, `μ = mc²/T`, evaluated in
closed form from `Z`: its branch point `z = a` is the tangency of the
resonance circle, and the branch cut is placed in the lower half-plane so that
damped roots continue the `Im ω > 0` response. At `N∥ = 0` the absorption is
the classical `Im F_q(z, 0) = −π(−z)^{q−1}e^z/Γ(q)`. `larmor_parameter` above
`larmor_tolerance` sets `LARMOR_RADIUS_LIMIT`.

## Kinetic dispersion roots

`KineticDispersionProblem` finds complex `ω(k)` of the full electromagnetic
dispersion relation (or the electrostatic one) with
`phydrax.nonlinear.VectorLocalRootPlan`, continuing one branch along a
wavenumber path:

```python
import math

scale = phx.ElectromagneticScaleContract.si()
debye = math.sqrt(float(scale.vacuum_permittivity) * 100.0 / (1.0e18 * charge))
omega_p = float(jnp.sqrt(hot.plasma_frequency_squared[0]))
landau = KineticDispersionProblem(hot, model="electrostatic").solve(
    jnp.linspace(0.3, 0.5, 5) / debye, 0.0, 1.16 * omega_p + 0j
)
# landau.frequencies[-1] / omega_p ≈ 1.4157 − 0.1534i (Landau damping)
```

Each point starts from the secant extrapolation of the last two accepted
roots. Every iterate is returned with its residual norm, Jacobian condition
estimate and `KineticDispersionStatus`; a point that does not converge, is
ill-conditioned, or lands farther than `branch_tolerance` from its prediction
(`BRANCH_JUMP`) is reported and not used to continue the branch. Typical uses
are Landau-damped Langmuir waves, electron Bernstein waves between cyclotron
harmonics, and the whistler anisotropy instability (`T⊥ > T∥` electrons give
`Im ω > 0` below `|Ω_e|`); `examples/kinetic_plasma_dispersion.py` runs all
three.

## Cyclotron maser growth

`RelativisticWeakGrowthPlan` computes the weak growth rate of a cold-plasma
mode driven by a dilute energetic population with an arbitrary gyrotropic
distribution `f(u⊥, u∥)` (`u = p/(mc)`), such as a loss cone, ring or
horseshoe. Only the anti-Hermitian part of the energetic species' fully
relativistic susceptibility enters (Wu & Lee 1979; Melrose & Dulk 1982). It
is an integral over the resonance ellipse `γ − N∥u∥ − sΩ/ω = 0` of
`[sY ∂f/∂u⊥ / u⊥ + N∥ ∂f/∂u∥]`, so `∂f/∂u⊥ > 0` on the ellipse means growth.
The growth rate is

`γ = −ω² e*·χᴬ·e / e*·∂_ω(ω² Kᴴ)·e`,

with the cold mode's polarization `e` and Hermitian tensor `Kᴴ`. For
electrons (`Ω < 0`) the fundamental is `s = −1`, and at perpendicular
propagation it resonates only below `|Ω|`.

```python
from phydrax.electromagnetics import LossConeDistribution, RelativisticWeakGrowthPlan

background = ColdPlasmaDielectric(
    phx.ElectromagneticScaleContract.si(),
    densities=[1.0e15],
    charge_numbers=[-1.0],
    mass_ratios=[1.0],
    magnetic_field=[0.0, 0.0, 0.1],
)
maser = RelativisticWeakGrowthPlan(
    background, LossConeDistribution(0.05, index=1), density=1.0e13
)
omega_c = abs(float(maser.cyclotron_frequency))
growth = maser.evaluate(0.999 * omega_c, 0.5 * math.pi, PlasmaWaveMode.ORDINARY)
```

The result reports per-harmonic rates, which harmonics resonate, and the
difference from a half-order quadrature. Modes that do not propagate and
resonance curves that are not ellipses (`N∥² ≥ 1`, e.g. whistlers) are
refused with NaN rates and status bits (`MODE_NOT_PROPAGATING`,
`RESONANCE_NOT_ELLIPTIC`). `WEAK_GROWTH_VIOLATED` is set when `|γ|/ω` exceeds
`weak_growth_tolerance`; the perturbative rate is then outside its domain.

## Magnetobremsstrahlung

`MagnetobremsstrahlungPlan` computes cyclotron, gyrosynchrotron and synchrotron
emission of a gyrotropic population into the two modes of a
`ColdPlasmaDielectric`. The runnable version is
`examples/magnetobremsstrahlung_emission.py`.

```python
from phydrax.electromagnetics import (
    MagnetobremsstrahlungPlan,
    PlasmaWaveMode,
    ThermalJuttnerDistribution,
)

background = ColdPlasmaDielectric(
    phx.ElectromagneticScaleContract.si(),
    densities=[1.0e16],
    charge_numbers=[-1.0],
    mass_ratios=[1.0],
    magnetic_field=[0.0, 0.0, 1.0],
)
plan = MagnetobremsstrahlungPlan(
    background,
    ThermalJuttnerDistribution(0.02),  # θ = kT/(m c²)
    emitter_density=1.0e15,
    maximum_harmonics=32,
)
gyrofrequency = float(plan.gyrofrequency)
emission = plan.evaluate(jnp.linspace(1.5, 4.5, 64) * gyrofrequency, 1.0)
extraordinary = emission.select(PlasmaWaveMode.EXTRAORDINARY, emission.emission)
```

For a mode with index `n_σ`, unit polarization `e_σ` and transverse power
`|e_T|²`, the emission per volume, unit angular frequency and wave-normal
steradian is

`j_σ = q² ω² n_σ / (8π² ε₀ c³ |e_T|²) Σ_s ∫ d³p N f |e_σ*·V_s|² δ(ω − sΩ/γ − k∥v∥)`,

with `V_s = (v⊥(J_{s−1}+J_{s+1})/2, iε v⊥(J_{s−1}−J_{s+1})/2, v∥ J_s)` at
`x = k⊥v⊥γ/Ω`, `Ω = |q|B/m` and `ε = −sign q`. The absorption along the wave
normal uses the momentum gradient of `f`; a thermal population obeys
`j_σ = n_σ² (kTω²/(8π³c²)) α_σ`.

The harmonic set follows from the resonance curve `γ = sY + N∥u∥`
(`Y = Ω/ω`, `N∥ = n cos θ`): an ellipse with `s ≥ 1` when `N∥² < 1`, a
hyperbola admitting `s = 0` and anomalous-Doppler `s < 0` when `N∥² ≥ 1`. Every
harmonic whose curve meets the distribution's momentum support is summed; the
thermal support is chosen so that the omitted mass is below a rigorous bound
reported as `tail_mass_bound`. Evanescent roots, the resonance cone
(`|n| > maximum_refractive_index`), undefined polarization and harmonic sets
beyond `maximum_harmonics` are reported with `MagnetobremsstrahlungStatus` bits
and NaN, never as zero emission.

The `"continuous-harmonic"` route replaces `Σ_s` by `∫ ds` with exact
real-order Bessel functions and places its pitch nodes on the emission cone of
width `√(1 − n²β²)`. For ultra-relativistic particles it reproduces the
single-particle spectrum in a medium, the vacuum form with `1/γ² → 2(1 − nβ)`:
`x = (ω/ω_c) R^{3/2}` and amplitude `n R^{−1/2}` with `R = 2γ²(1 − nβ)`, which is
the Razin suppression `x = (ω/ω_c)(1 + γ²ω_p²/ω²)^{3/2}` when
`n² = 1 − ω_p²/ω²`. Its error against the harmonic
sum falls as the emission-weighted harmonic number
(`effective_harmonic`) grows; results below `minimum_continuous_harmonic` are
flagged `LOW_HARMONIC`.

`stokes_emission` and `propagation_matrix` combine both modes in the Faraday
basis above, with the cold-plasma rotation and conversion `2 Re ρ` from
`faraday_coefficients`.

### External oracles: UFGC and Symphony

Two GPL-3.0 gyrosynchrotron codes serve as pinned external oracles for a
`MagnetobremsstrahlungPlan`; neither is imported into Phydrax nor copied. Each
runs in a caller-pinned Python interpreter that loads the digest-checked
provider library staged into the run directory, and each result carries the
provider version, library digest, license, output digest and an
`AdapterReport` whose losses list everything the provider cannot represent.

- `run_ufgc(UFGCProvider(python, library), plan, directory, angular_frequencies=ω,
  angles=θ)` calls the Ultimate Fast Gyrosynchrotron Codes (Kuznetsov &
  Fleishman 2021) `MWTransferArr` library. UFGC is mode-resolved in the cold
  magnetoionic plasma, so its ordinary/extraordinary `j_σ`, `κ_σ` compare
  directly with `emission`/`absorption`: the plan route selects UFGC's exact
  harmonic code or its continuous code. Thermal and kappa populations must be
  the plasma electrons; a power law maps to UFGC's momentum power law above a
  background of `n_plasma − N`. The coefficients are recovered from an
  optically thick and an optically negligible single-voxel transfer.
- `run_symphony(SymphonyProvider(python, module), plan, directory, ...)` calls
  Symphony (Pandya et al. 2016) for vacuum Stokes `(j, α)` of `I`, `Q`, `V`
  in the `FaradayCoefficients` basis, comparable with `stokes_emission` and
  the first row of `propagation_matrix` when `n_σ ≈ 1`. The pinned revision
  integrates its power law over `γ ∈ [1, ∞)` whatever its `γ_min`, `γ_max`; the
  adapter matches the plan's population for `u ≫ 1` and declares the rest.

`ufgc_input`/`symphony_input` and `read_ufgc_output`/`read_symphony_output`
are the provider-free deck generators and parsers; unsupported plans
(tabulated distributions, non-electron emitters, multi-species or collisional
plasmas for UFGC, angles at or beyond `π/2` for Symphony) are refused before
running.

## Thermal synchrotron and free–free closures

`ThermalSynchrotronModel` (Mahadevan, Narayan & Yi 1996) is the fast
angle-averaged thermal route for `T ≥ 3.2 × 10¹⁰ K` in SI, and
`ThermalFreeFreeModel` gives thermal bremsstrahlung with the Born thermal Gaunt
factor. Both expose `gray_means` (Planck and Rosseland means with support
evidence), from which the compact-object gray closures are computed.

## Rays and polarized transfer

`phydrax.optics.geometric.ColdPlasmaHamiltonian` traces rays of one cold-plasma
mode through a `ColdPlasmaProfile` whose density and magnetic field vary in
space. The Hamiltonian is the Stix biquadratic `D = A n⁴ − B n² + C` in
Cartesian refractive-index components, normalized by `G = −ω ∂D/∂ω|_k` so the
ray parameter is the group light path `c t`; `DispersionRayPlan` integrates it
with the implicit-midpoint symplectic scheme (see
[Graded-index and dispersion rays](guides_graded_index_optics.md)). The mode is
the launch root with the requested `PlasmaWaveMode` label; vacuum launches,
where both modes coincide, are refused.

```python
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.electromagnetics import PlasmaWaveMode
from phydrax.optics.geometric import (
    ColdPlasmaHamiltonian,
    ColdPlasmaProfile,
    DispersionRayPlan,
)

scale = phx.ElectromagneticScaleContract.si()
coordinates = phx.SpatialCoordinateContract(phx.units.METER)
omega = 2.0 * np.pi * 1.0e9
critical = float(scale.vacuum_permittivity * scale.electron_mass) * omega**2 / float(
    scale.elementary_charge
) ** 2
profile = ColdPlasmaProfile(
    scale,
    coordinates,
    charge_numbers=[-1.0],
    mass_ratios=[1.0],
    density=lambda x: jnp.stack([critical * (0.1 + x[0])]),
    magnetic_field=lambda x: jnp.stack([0.0 * x[0], 0.0 * x[0], 0.02 + 0.0 * x[0]]),
    profile_id="ramp",
)
hamiltonian = ColdPlasmaHamiltonian(
    profile, angular_frequency=omega, mode=PlasmaWaveMode.ORDINARY
)
rays = (
    DispersionRayPlan(hamiltonian, 0.01, 260, hamiltonian_tolerance=1.0e-6)
    .prepare()
    .integrate(np.zeros((1, 3)), np.asarray(((np.cos(0.5), np.sin(0.5), 0.0),)))
)
path = hamiltonian.sample_path(rays, (0.0, 0.0, 1.0))
```

`ColdPlasmaRayPath` carries, on every segment, the ray refractive index
`n_r² = n² |sin θ / (sin Θ dΘ/dθ)| / cos α` (Bekefi 1966; `Θ` the group
direction and `α` the angle between wave normal and group velocity), both mode
indices and Stokes vectors, the Faraday rotation vector in a
parallel-transported polarization basis, the QL/QT flag, and the mode-coupling
parameter `|dŝ/ds| / (k |n₀ − n₁|)`.

`phydrax.applications.radiation_transport.PlasmaRayTransferPlan` transports
`S/n_r²`:

    d(S/n_r²)/dℓ = Σ_σ (j_σ/n_σ²)(1, ŝ_σ) − K S/n_r²,   dℓ = cos α ds,

with `j_σ`, `α_σ` in the `MagnetobremsstrahlungPlan` convention (per wave-normal
steradian and per length along the wave normal), so an optically thick thermal
plasma reaches `I_σ/n_r² = kTω²/(8π³c²)` per mode regardless of refraction, and
a lossless source-free path conserves `I/n_r²`. Two limits are explicit:

- `coupling="weak"`: independent modes; the ray carries its own mode, whose
  polarization follows `ŝ_σ`. `WEAK_COUPLING_VIOLATED` is raised when the
  coupling parameter exceeds `coupling_tolerance`.
- `coupling="strong"`: the full Stokes vector with the coupled propagation
  matrix (mode absorption plus Faraday rotation and conversion), exact for any
  coupling parameter when both modes share the ray; `ANISOTROPY_VIOLATED` is
  raised when the relative index splitting exceeds `anisotropy_tolerance`.

`faraday_rotation = ∫ ρ_V dℓ` is the plane rotation along the ray (the rotation
measure times `λ²` at high frequency), and `QUASI_TRANSVERSE` qualifies paths
that cross the region where mode coupling is decided.
`magnetobremsstrahlung_path_coefficients(hamiltonian, path, plan_factory)`
evaluates a `MagnetobremsstrahlungPlan` on each segment's local homogeneous
plasma (host preparation) and orders the coefficients as (ray mode,
companion). The runnable version is `examples/plasma_ray_transfer.py`.
