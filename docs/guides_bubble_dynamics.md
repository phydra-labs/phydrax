# Bubble dynamics

`phydrax.bubble_dynamics` owns spherical (radial) bubble models: acoustically
driven and cavitating micro-bubbles, lipid- and polymer-coated contrast agents,
viscoelastic surroundings, diffusive dissolution, pinned surface nanobubbles,
interacting bubble clouds with Bjerknes forces and far-field emission. Bubble
population kernels (`phydrax.population_balance`) and bubbly-medium dispersion
(`phydrax.acoustics`) compose these models and are documented here as well.
Interface-resolved bubbly flow belongs to `phydrax.applications.two_phase_flow`;
there is no automatic hand-off between the reduced radial models documented here
and resolved flow.

## Model composition

A `RadialBubbleModel` composes one static radial-equation selector with four
laws and a far-field environment:

- a gas law (`AbstractBubbleGasLaw`), evaluated from the bubble **volume**,
  volume rate and an extensive gas state (amount in mol, internal energy where
  defined, law-specific internal state);
- a liquid law (`AbstractBubbleLiquidLaw`) returning the wall stress integral
  `S = 2 ∫_R^∞ (τ_rr − τ_θθ)/r dr` of the radial flow;
- an interface law (`AbstractBubbleInterfaceLaw`) returning capillary, shell
  elastic and shell viscous pressures, and a closed regime set for piecewise
  shells;
- a `BubbleEnvironment` (ambient pressure `p0`, liquid temperature, vapor
  pressure);
- for `"gilmore"`, a barotropic `liquid_material`
  (`phydrax.equations.TaitBarotropicMaterial`), otherwise `liquid_density` and
  `liquid_sound_speed`.

The wall liquid pressure is
`p_L = p_g + p_v − p_capillary − p_elastic − p_viscous + S`. A drive supplies the
excess far-field pressure `p_d(t)`; `p_∞(t) = p0 + p_d(t)`. Physical
coefficients are trainable parameter leaves; selectors, node counts and
capacities are static. Homogeneous laws stacked along a leading axis form a
batched species group for `eqx.filter_vmap`.

### Radial equations

| Selector | Equation |
|---|---|
| `rayleigh_plesset` | `ρ(R R̈ + 3/2 Ṙ²) = p_L − p_∞` |
| `rayleigh_plesset_radiation` | adds `(R/c) d(p_L − p_∞)/dt` |
| `rayleigh_plesset_gas_radiation` | adds `(R/c) dp_g/dt` only (Marmottant et al. 2005, Eq. 3) |
| `keller_miksis` | `(1 − Ṙ/c)RR̈ + 3/2(1 − Ṙ/3c)Ṙ² = (1 + Ṙ/c)(p_L − p_∞)/ρ + R/(ρc) d(p_L − p_∞)/dt` (Keller & Miksis 1980) |
| `gilmore` | `(1 − Ṙ/C)RR̈ + 3/2(1 − Ṙ/3C)Ṙ² = (1 + Ṙ/C)H + (R/C)(1 − Ṙ/C) dH/dt`, `H = h(p_L) − h(p_∞)`, `C = c(p_L)` (Gilmore 1952) |

Viscous and shell terms make `p_L` depend on `Ṙ`, so `dp_L/dt` hides an `R̈`
term. The model evaluates one batched `jax.jvp` of the composed wall quantity
along `(Ṙ, 0, ż, 1)` and `(0, 1, 0, 0)` and solves the resulting scalar linear
equation for `R̈`. No law hand-codes this chain rule. For Gilmore the enthalpy
is `h = e(ρ) + p/ρ` and the local sound speed is `c(ρ(p_L))`, both from the
material owner.

The two radiation-corrected Rayleigh–Plesset selectors differ only in which
pressure radiates. Expanding Keller–Miksis to first order in `Ṙ/c` gives the
correction `(R/c) d(p_L − p_∞)/dt`, which `rayleigh_plesset_radiation` keeps.
`rayleigh_plesset_gas_radiation` is the coated-bubble model of Marmottant et
al. (2005, Eq. 3): `ρ(RR̈ + 3/2 Ṙ²) = p_L − p_∞ + (R/c) dp_g/dt`. It keeps only
the gas-pressure part of that correction, which varies fastest near
compression, and drops the rates of the capillary, viscous, shell, vapor and
drive pressures, so the viscous terms carry no hidden `R̈`. For a polytropic
gas `(R/c) dp_g/dt = −3κ p_g Ṙ/c`, which gives the paper's factor
`p_g(1 − 3κṘ/c)`. The two forms coincide exactly when only the gas pressure
varies: a clean interface without tension, an inviscid liquid, a constant
drive and a constant vapor pressure. Both reduce to Rayleigh–Plesset as
`c → ∞`. Hilgenfeldt, Lohse & Brenner (1996, Eq. 1.2) radiate `p_g − p_d(t)`,
a third variant that is not offered. The paper's inline statement of the
modified equation prints `−(R/c) dP_g/dt`. Its Eq. (3) implies `+`, and `+` is
the sign implemented.

## Law catalogue

**Gas interiors.**

- `PolytropicBubbleGasLaw` (`pV^κ` constant; zero reference pressure is the
  empty cavity) and `HardCorePolytropicBubbleGasLaw` (van der Waals hard core,
  `h = R/8.86` for air, Löfstedt et al. 1993).
- `BoundaryLayerThermalBubbleGasLaw`: uniform-temperature gas with the thermal
  boundary-layer flux `λ(T∞ − T)/ℓ`, `ℓ = min(√(Rχ/|Ṙ|), R/π)` (Toegel et al.
  2000; Stricker, Prosperetti & Lohse 2011).
- `ReducedTransferBubbleGasLaw`: constant transfer coefficient
  `β_T = Re Ψ(Pe)` of Preston, Colonius & Brennen (2007); `β_T → 5` at low
  Péclet number. The imaginary part of `Ψ` is neglected as in the reference
  model.
- `SpectralThermalBubbleGasLaw`: Prosperetti's (1991) homobaric energy equation
  resolved in `s = (r/R)²` at a fixed number of Chebyshev–Lobatto nodes, with
  temperature-dependent conductivity. Its linearization reproduces the exact
  Prosperetti transfer function (`prosperetti_polytropic_index`) to rounding at
  16 nodes. The route is stiff and runs with Kvaerno5.
- `MaterialBubbleGasLaw`: adiabatic gas described by any
  `equations.AbstractThermodynamicMaterial`, including the Noble–Abel
  stiffened gas `equations.NobleAbelStiffenedGasMaterial` (Le Métayer & Saurel
  2016).
- Compartment laws `IsothermalIdealBubbleGasLaw` and `CaloricIdealBubbleGasLaw`
  implement `AbstractBubbleCompartmentGasLaw`: `merge` conserves gas amount and
  internal energy and reports the mixing entropy production; `split`
  partitions both under the uniform-intensive policy with zero entropy
  production. Only the isothermal ideal gas has an additive `pV` invariant; a
  merged polytropic `pV^γ` is not conserved and is not offered.

**Liquids.** `NewtonianBubbleLiquidLaw`; `PowerLawBubbleLiquidLaw` (truncated
Ostwald–de Waele with a declared minimum shear rate and an exact stress
integral); `KelvinVoigtBubbleLiquidLaw` (Yang & Church 2005);
`ZenerBubbleLiquidLaw` (small-strain standard linear solid, relaxation stress as
ODE state, admissibility `μ ≥ Gλ`); `OldroydBBubbleLiquidLaw` (upper-convected
Maxwell polymer advected exactly on Lagrangian shells at Gauss–Legendre
quadrature nodes of the volume coordinate, after Warnez & Johnsen 2015; the
conformation tensor must stay positive definite).

**Interfaces and shells.** `CleanBubbleInterfaceLaw` with an optional
`TolmanCorrectionPolicy` (off by default because real Tolman lengths are
disputed); `MarmottantShell` (Marmottant et al. 2005) with buckled, elastic and
ruptured regimes and optional irreversible break-up; `GompertzMarmottantShell`
(smooth Marmottant–Gompertz law, arXiv:2106.12004) for gradient-based
inference; `HoffShell` (Hoff, Sontum & Hovem 2000); `ChurchShell` (Church 1995,
thick incompressible shell with the liquid law acting at the outer radius);
`SarkarShell` (Sarkar et al. 2005, with the exponential strain-softening
elasticity of Paul et al. 2010); `DoinikovShearThinningShell` (Cross-law shell
viscosity, Doinikov, Haac & Dayton 2009); `MaxwellShell` (thin-shell reduction
of the Maxwell shell of Doinikov & Dayton 2007).

**Drives.** `ConstantPressureDrive`, `HarmonicPressureDrive`,
`PulsedPressureDrive` (Hann-windowed tone burst, continuous rate) and
`SampledPressureDrive` (native `series.SampledSeriesReconstruction`; leaving
the sampled interval ends the solve with `SUPPORT_EXIT`).

## Single-bubble solves

```python
import numpy as np
import phydrax.bubble_dynamics as bd

model = bd.RadialBubbleModel(
    "keller_miksis",
    bd.PolytropicBubbleGasLaw(1.07),
    bd.NewtonianBubbleLiquidLaw(1.0e-3),
    bd.MarmottantShell(0.55, 0.02, 0.072, 1.5e-8),
    bd.BubbleEnvironment(101325.0, 293.15),
    liquid_density=1000.0,
    liquid_sound_speed=1500.0,
)
plan = bd.SingleBubblePlan(
    model,
    bd.HarmonicPressureDrive(5.0e4, 2.0 * np.pi * 2.9e6),
    np.linspace(0.0, 2.0e-6, 101),
)
result = bd.solve_single_bubble(plan.prepare(2.0e-6))
```

`SingleBubblePlan.prepare` fixes the gas content from the Laplace balance at the
equilibrium radius (a negative gas pressure is `INVALID_EQUILIBRIUM`) and
nondimensionalizes the state with inertial scales. `solve_single_bubble`
integrates with `solver.solve_diffrax`. Terminal events (minimum radius,
hard-core reach, wall Mach limit, invalid law state, drive support exit) are
native Diffrax events localized by Newton root finding. A piecewise shell is
integrated as a fixed-capacity sequence of smooth segments: every segment ends
at a localized regime guard, the regime changes, and the next segment restarts
from the exact event state. Each regime guard carries a declared hysteresis
(`1e-7` of the buckling radius by default) so that a restart on the guard is
strictly inside the new regime and so that a bubble starting at rest exactly on
a guard (for example an initially buckled shell) crosses it at a resolvable
wall speed; the switching radius is therefore accurate to that relative offset.
The transitions are recorded in `BubbleRegimeTape`; exceeding the declared
capacity returns `REGIME_CAPACITY`.

`SingleBubbleResult` holds the requested trajectory (NaN and `valid = False`
after a terminal event), the exact terminal state, the terminal regime, a
`BubbleDynamicsStatus` and `BubbleDynamicsEvidence`: accepted/rejected steps,
event kind and time, regime tape, the wall-work ledger, dissipated energy, gas
heat, the minimum effective inertia fraction and `BubbleValidityEvidence`
(maximum wall Mach number, wall pressure, radius and temperature extremes,
hard-core margin, Laplace ratio, optional Knudsen and Tolman ratios, and the
neglected physics of the chosen equation). Failures return the last accepted
state. For Rayleigh–Plesset the ledger identity `ΔK = ∫ 4πR²Ṙ(p_L − p_∞)dt` is
exact and `work_residual` measures the time-integration error; for the
compressible equations `−work_residual` is the acoustically radiated energy.

Derivatives flow through the solve. `differentiation="reverse"` (recursive
checkpointing) supports gradients and VJPs; `"forward"` supports JVPs. Regime
transitions are differentiated through the localized event times;
`derivative_available` is false when a transition is grazing
(`|Ṙ|` below the declared transversality tolerance) or the solve failed.

### Compression-only reference case

The `compression-only-reference` campaign of
`tools/bubble_dynamics_qualification.py` reproduces Fig. 5(b) of Marmottant et
al. (2005).

**Model and parameters.** The campaign uses `rayleigh_plesset_gas_radiation`, a
polytropic gas, a Newtonian liquid and a `MarmottantShell`. Every parameter
stated in the caption is used as printed:

- `R_buckling = R₀ = 0.975 µm`, `χ = 1 N/m` and `κ_s = 15 × 10⁻⁹ kg/s` (the
  caption prints the unit as "N");
- break-up tension `≥ 1 N/m` (a resistant shell, never reached);
- `ρ_l = 10³ kg/m³`, `μ = 10⁻³ Pa s`, `c = 1480 m/s` and `κ = 1.095`;
- a 2.9 MHz, 130 kPa drive.

The water tension, 73 mN/m, comes from Sec. I of the paper and does not enter
below break-up.

**Unstated inputs.** The paper does not give the ambient pressure, so
`p0 = 101325 Pa` is assumed. It also does not give the simulated burst. The
figure shows expansion first and five compressions, so the campaign uses an
unwindowed, rarefaction-first sine burst of five cycles starting at the
figure's `t = 0`, supplied through `SampledPressureDrive`.

**Digitization.** Radii and times were digitized from the published raster.
The axis ticks calibrate the scales at 0.48 nm/px and 3.83 ns/px. The
digitized rest level is 0.97492 µm against the stated 0.975 µm. Standard
uncertainties are 0.5 nm in radius and 4 ns in time. The source URL and hashes
are recorded in the campaign.

**Results.** Inside the burst the simulation matches the figure to within
digitization accuracy:

- The five minima agree to 0.16 nm.
- Maxima 2–5 agree to 1.2 nm.
- The nine rest-level crossings, which the regime tape localizes as
  buckling and unbuckling events, agree to 2 ns.
- The four complete buckled intervals, about 0.254 µs each, agree to 0.9 ns.
- The asymmetry `ΔR⁺/ΔR⁻` is 0.264 in the simulation and 0.261 ± 0.004 in the
  figure, so the oscillation is compression-only.

The figure's burst edges are not reproduced. Its first expansion reaches
0.9955 µm, against 1.0056 µm here. After the burst it recovers to its rest
level within about 0.15 µs, whereas the rectangular burst leaves the bubble
buckled and relaxing slowly. Both differences point to tapered burst edges
that the paper does not specify, so the campaign gates only the burst
interior and reports the edges.

**The paper's quantitative statements.**

- The period-averaged pressure `⟨p_g R⁴⟩/⟨R⁴⟩` of its Eq. (7) is 1.164 `p0`,
  against "1.2 `P₀`" in the text.
- The text says that varying the frequency from 1 to 4 MHz changes
  `ΔR⁺/ΔR⁻` by about 10%. This is **not** reproduced at `R₀ = R_buckling`: with
  the same five-cycle burst the ratio is 0.160 at 1 MHz, 0.264 at 2.9 MHz and
  0.289 at 4 MHz, a spread of 49%. The statement refers to the Fig. 7
  response curve, and the paper does not define the pulse used at other
  frequencies.

**Sensitivity.** Changing the assumed ambient pressure to 100 kPa moves the
interior extrema by at most 0.5 nm. Replacing the gas-only radiation term with
`rayleigh_plesset_radiation` moves them by at most 0.3 nm. Every gate still
passes in both variants, so this case does not discriminate between the two
radiation forms.

## Linear response

`linear_bubble_response(model, equilibrium_radius, angular_frequencies)`
linearizes the composed equation at rest with the bounded dense Jacobian and
prescribes a harmonic wall displacement: the native dense solve returns the
drive that sustains it and the internal-state response, which is regular even at
an undamped resonance. The laws' linearized pressures split the equivalent
oscillator into gas, interface and liquid stiffness and into thermal, viscous,
shell and radiation damping; the effective polytropic index and the
self-consistent resonance `ω = ω₀(ω)` (native local root) are reported. Checks
include Minnaert's frequency with capillarity, the coated Marmottant resonance,
the Keller–Miksis radiation damping `ω²R₀/(2c)` and Prosperetti's thermal
theory.

## Dissolution and surface nanobubbles

`EpsteinPlessetPlan` integrates isothermal diffusive dissolution or growth of a
free spherical bubble (Epstein & Plesset 1950). Laplace pressure raises both
the gas amount and the Henry's-law interface concentration. The static route is
explicit provenance:

- `"quasi_static"`: flux `D Δc/R`; with zero surface tension its lifetime is
  `ρ_g R₀²/(2 D c_s (1 − f))` (`quasi_static_dissolution_time`);
- `"full_history"`: fixed-sphere diffusion with the complete Duhamel history
  of the Laplace-dependent interface concentration. The `1/√(πt)` kernel is a
  sum of exponentials whose relative error on the declared lag range is
  reported; time is integrated in `√t`, which regularizes the start-up flux.

`PinnedSurfaceBubblePlan` follows a spherical-cap bubble on a wall (Lohse &
Zhang 2015). Geometry is explicit: footprint **diameter** `L` and the
**gas-side** cap angle `θ`; the reported liquid-side contact angle is `π − θ`.
The quasi-static flux is Popov's (2005) evaporating-cap solution
`ṅ = −π a D Δc f(θ)`. Pinned caps have the equilibrium
`sin θ = ζ L/L_c`, `L_c = 4σ/p0`, which is stable on the flat branch; a cap
reaching a hemisphere stops with `SUPPORT_EXIT`. Unpinned caps keep `θ` and
dissolve under undersaturation.

Evidence reports the lifetime (infinite when not reached), a gas-amount ledger
residual, initial and final saturation margins, the equilibrium angle and its
stability derivative, the Laplace-pressure ratio and, when a molecular diameter
or Tolman length is declared in `BubbleValidityPolicy`, Knudsen and Tolman
ratios with a continuum-support decision. For Laplace-dominated nanobubbles the
Knudsen number is nearly size independent (about 0.05 for air in water),
because the mean free path shrinks with the Laplace pressure.

## Bubble clouds

A cloud is a static tuple of `BubbleSpeciesGroup`s. A group shares one static
law structure; its dynamic coefficients are batched along the member axis, so a
heterogeneous cloud never selects laws per bubble at run time. Pass one
`RadialBubbleModel` shared by every member, or one model per member with an
identical structure (same `model_id`, tree structure and leaf shapes). Each
member has an equilibrium radius, an absolute position (m), an optional
displaced initial radius, wall velocity and translational velocity, and a
stable, cloud-unique integer id. Clouds accept smooth interface laws only
(`guard_count == 0`); use `GompertzMarmottantShell` rather than the piecewise
`MarmottantShell`.

```python
import numpy as np
import phydrax.bubble_dynamics as bd

model = bd.RadialBubbleModel(
    "keller_miksis",
    bd.PolytropicBubbleGasLaw(1.4),
    bd.NewtonianBubbleLiquidLaw(1.0e-3),
    bd.CleanBubbleInterfaceLaw(0.072),
    bd.BubbleEnvironment(101325.0, 293.15),
    liquid_density=998.0,
    liquid_sound_speed=1481.0,
)
group = bd.BubbleSpeciesGroup(
    model, np.array([3.0e-6, 4.0e-6]), np.array([[0.0, 0.0, 0.0], [40.0e-6, 0.0, 0.0]]),
    bubble_ids=(0, 1),
)
plan = bd.BubbleCloudPlan(
    (group,), bd.HarmonicPressureDrive(2.0e4, 2.0 * np.pi * 2.0e5),
    np.linspace(0.0, 2.0e-5, 401)[1:],
)
result = bd.solve_bubble_cloud(plan.prepare())
forces = bd.mean_bjerknes_forces(result, 1.0e-5, 2.0e-5)
```

### Coupled radial equation

Every bubble is evaluated by its own composed model. With
`φ_i = RadialBubbleRates.inertia_fraction` the per-unit-density inertia
`a_i = φ_i R_i` carries every hidden `R̈` coefficient of the selected radial
equation (Keller–Miksis, Gilmore and the radiation variants included). The
uncoupled equation is `a_i R̈_i = f_i`. The incompressible near field of the
neighbour monopoles adds (Mettin et al. 1997, generalized to all four radial
equations)

    a_i R̈_i + Σ_{j≠i} (R_j² R̈_j + 2 R_j Ṙ_j²)/d_ij = f_i.

The accelerations are solved implicitly and no neighbour `R̈` is lagged. With
`y = R²R̈`, `D = diag(a/R²)` and `G_ij = 1/d_ij` the system is
`(D + G) y = f − G(2RṘ²)`. It is solved in the symmetrically scaled form
`(I + W) z = D^{-1/2}(…)`, `W = D^{-1/2} G D^{-1/2}`. For Rayleigh–Plesset
(`a = R`) and non-overlapping spheres, `D + G` is the Coulomb energy matrix of
uniformly charged spherical shells and is positive definite. The same system
is the Euler–Lagrange equation of the potential-flow Lagrangian, so the
liquid kinetic energy `K = 2πρ Σ_i q_i (R_i Ṙ_i + Σ_{j≠i} q_j/d_ij)`
(`q = R²Ṙ`) satisfies `ΔK = W` exactly for a Rayleigh–Plesset cloud at fixed
positions (`BubbleCloudEvidence.work_identity_exact`). Compressible inertia
fractions can make `I + W` indefinite; that is reported and never repaired.

Two routes solve the coupled system, and `route="auto"` picks between them
with `BubbleCloudResourcePolicy`, not a hard-coded bubble count:

- `"dense"`: pair geometry and a native dense Cholesky solve, allowed only when
  `N² ≤ maximum_dense_entries` and `N³/3 ≤ maximum_dense_flops`. An explicit
  `route="dense"` beyond that bound is refused when the plan is built. One
  native eigendecomposition of `W` per saved row (and in the conditioning
  event) reports the 2-norm condition number `(1 + λ_max)/(1 + λ_min)` of
  `I + W` and its definiteness. The solve stops with `ILL_CONDITIONED` above
  `maximum_condition_number`.
- `"fmm"`: matrix-free conjugate gradients on `I + W` (native
  `linalg.ConjugateGradient`, prepared once and refreshed with the current
  radii and inertias). The pair action is the signed Laplace monopole FMM
  (`solver.UniformFMMPlan.evaluate_monopole`) on a structure prepared once for
  the fixed positions. Softening is `10⁻⁹` of the box, so its relative bias
  `ε²/(2d²)` is negligible for non-overlapping bubbles. FMM capacity evidence
  reaches `BubbleCloudEvidence.fmm`. A CG solve that does not converge stops
  the run with `COUPLING_FAILURE`. Overlap is certified over candidate pairs
  from a native Morton radius relation within
  `overlap_growth_bound · 2R_max`. A bubble that outgrows this certificate
  stops the run with `VALIDITY_EXCEEDED`, and an exhausted candidate capacity
  refuses the run with `COUPLING_FAILURE`.

Overlap (`d_ij ≤ minimum_contact_ratio (R_i + R_j)`) is a terminal `OVERLAP`
status, both initially and as a localized event. No merged or resolved bubble
is synthesized.

### Drive field, translation and Bjerknes forces

`BubbleCloudPressureField` sets the spatial shape of the drive. The default is
`"uniform"`. `"standing_wave"` uses `s(x) = cos(k·x + φ)`, so bubble `i` feels
`s(x_i) p_d(t)`. The primary Bjerknes force is `F₁ = −V ∇p_ac = −V p_d ∇s`.
The secondary force is `F₂ = −V ∇p_nb`, where `p_nb = ρ Σ_j Q̇_j/|x − x_j|` is
the neighbour monopole pressure and `Q̇ = R²R̈ + 2RṘ²`. Both are saved
on every row. `mean_bjerknes_forces` time-averages them by the trapezoidal
rule, which gives Bjerknes's classical sign rule. Bubbles on the same side of
resonance oscillate in phase and attract. Bubbles on opposite sides oscillate
in antiphase and repel. With `translation=BubbleTranslation(μ)`, bubbles
move with the declared added mass: `d(C_a ρ V v)/dt = F₁ + F₂ − C_d π μ R v`.
The defaults are `C_a = ½`, and `C_d = 12` for Levich drag on a clean bubble
at high Reynolds number; use `C_d = 4` for Hadamard–Rybczynski creeping flow.
Translation requires the dense route.

### Retarded coupling

`coupling="retarded"` replaces the instantaneous near field by
`Σ_j Q̇_j(t − d_ij/c)/d_ij`. The delayed accelerations make this a neutral
delay equation. It is solved with the native delay substrate in its
derivative-delay form: one `DerivativeDelay` per unordered pair,
`solve_diffrax_delay` with the full dense delay history (bounded by the plan's
`maximum_steps`), which also serves the saved-row fields and emission.
`NeutralDelayProblem`'s
transformed-state form needs a state-independent neutral coefficient. Here
the coefficient of the delayed accelerations is `1/a_i`, which depends on the
current state. The neutral part is stable for every delay iff `ρ(W) < 1`. An
initial `ρ(W) ≥ 1` is refused with `NEUTRAL_UNSTABLE`, and a native event stops
the solve if the run reaches it. A densely packed cloud can have a positive
definite instantaneous system (`λ_min(W) > −1`) and still be retarded-unstable
(`λ_max(W) ≥ 1`). The route requires fixed positions, the dense route and a
derivative-compatible start: bubbles at rest in equilibrium with `p_d(0) = 0`.
That way no neutral derivative discontinuity is propagated along the
combinatorial set of pair-lag sums. The step size is bounded by the shortest
pair delay, so the cost grows as `c` increases. Evidence reports the
retardation ratio (largest pair delay over the fastest inertial time) and the
occupancy of the delay history. `HISTORY_CAPACITY` reports an exhausted
history. As `c → ∞` the retarded trajectory converges to the incompressible
one.

### Far-field emission

`FarFieldEmissionPlan(observers, times)` on a fixed-position cloud superposes
the linear monopole field `p(x, t) = Σ_b ρ V̈_b(t − r_b/c)/(4π r_b)`. `V̈` is
the exact forward-mode time derivative of the volume flow rate `4πR²Ṙ` along
the solver's dense interpolant (incompressible and retarded routes alike); no
finite difference of saved samples enters. Evaluations run as a bounded
`lax.map` (`working_set`). A sample whose retarded time lies outside the solved
interval is uncovered and set to NaN; it is never extrapolated. The evidence
reports coverage and the far-field ratio `min r/R_max`. The formula assumes
`r ≫ R` and acoustically compact bubbles.

### Cloud evidence and nonclaims

`BubbleCloudEvidence` reports solver work and the terminal event, the energy
ledger (wall work, dissipation, gas heat, liquid kinetic-energy change and
`work_residual`), the minimum contact ratio, the condition number and spectral
radius of the coupling, coupled-solve success, iterations and residuals, the
retardation ratio, the history occupancy, the FMM resource and
overlap-certificate evidence, and the validity extremes of all bubbles.
Cloud-level neglected terms are listed with the per-equation ones.

- Only monopole interaction is modeled. Dipole (translational) near fields,
  the liquid velocity induced by neighbours at a bubble, the `u²/4` slip term
  and the history force are neglected.
- The incompressible coupling enters the compressible equations in Mettin's
  form. The neighbour pressure's `(1 + Ṙ/c)` factor and radiation-derivative
  corrections are not included.
- Piecewise (event-switched) shells are not supported in clouds.
- The FMM route and the retarded route require fixed positions. Emission
  requires fixed positions.
- No derivative claim is made for cloud solves.

## Bubble population balance: coalescence and breakage kernels

`phydrax.population_balance` evaluates three classical turbulent bubble kernels
on a sectional grid. It passes them to `ConservativeSectionalSolver` (fixed-pivot
aggregation plus conservative daughter redistribution); there is no second
solver.

**Grid.** `BubbleSectionalPlan(d)` takes strictly increasing pivot diameters
`d_i` (m). The solver's additive pivots are the gas volumes `v_i = π d_i³/6`
(`plan.volumes`), so the conserved first moment is the total gas volume per
unit mixture volume, including the solver's overflow reservoir.

**Liquid state.** `TurbulentBubblyLiquid(ρ, σ, ν, ε, integral_length_scale=L)`
takes the liquid density, surface tension, kinematic viscosity and turbulent
dissipation rate. The Kolmogorov length is `η = (ν³/ε)^{1/4}` and the turbulent
Weber number is `We = ρ ε^{2/3} d^{5/3}/σ`.

- **Prince & Blanch (1990).** The kernel is `K_ij = (θ^T_ij + θ^LS_ij) λ_ij`:
  - turbulent collisions `θ^T = 0.089 π (d_i + d_j)² ε^{1/3} (d_i^{2/3} + d_j^{2/3})^{1/2}`;
  - optional laminar-shear collisions `(4/3)(r_i + r_j)³ γ̇`;
  - film-drainage efficiency `λ = exp(−t/τ)`, with drainage time `t = (r_ij³ ρ/(16σ))^{1/2} ln(h₀/h_f)`, contact time `τ = r_ij^{2/3} ε^{−1/3}`, `h₀ = 10⁻⁴ m`, `h_f = 10⁻⁸ m`;
  - equivalent radius `r_ij = d_i d_j/(d_i + d_j)`.
- **Lehr, Millies & Mewes (2002).**
  `K_ij = (π/4)(d_i + d_j)² min(u′, u_crit) exp(−((α_max/α)^{1/3} − 1)²)`, with
  approach velocity `u′ = max(√2 ε^{1/3}(d_i^{2/3} + d_j^{2/3})^{1/2}, |u_i − u_j|)`
  and `u_crit = 0.08 m/s`.
- **Luo & Svendsen (1996) binary breakage.** The partial rate is
  `Ω(f; d) = 0.923 (1 − α)(ε/d²)^{1/3} ∫_{ξmin}^{1} (1 + ξ)² ξ^{−11/3} exp(−b ξ^{−11/3}) dξ`,
  with `b = 12 c_f σ/(β ρ ε^{2/3} d^{5/3})`,
  `c_f = f^{2/3} + (1 − f)^{2/3} − 1`, `β = 2.045` and `ξmin = 11.4 η/d`.
  `luo_svendsen_eddy_integral` evaluates the `ξ` integral in closed form as
  three regularized incomplete-gamma terms (shapes 8/11, 5/11, 2/11).
  The frequency `g = ½∫₀¹ Ω df` and the daughter distribution use a
  Gauss–Legendre rule in `t` with `f = t³/2`. Fixed-pivot placement keeps
  `Σ_i v_i D_ij = v_j` to rounding.

Every result carries `BubbleKernelEvidence`:

- the dissipation rate;
- the Kolmogorov length;
- the Weber range against an optional declared support;
- inertial-subrange applicability (`η < d < L`);
- the largest exponent of the `exp(−x)` factors, with an out-of-float-range
  flag (reported, never clipped);
- finiteness.

Coalescence results also report their symmetry residual, and breakage results
their daughter-volume residual.

Nonclaims:

- The models assume locally isotropic turbulence with bubbles in the inertial
  subrange; isotropy cannot be checked.
- Prince–Blanch buoyancy collisions are not provided: the printed and commonly
  reproduced cross-sections differ by a factor of four.
- The Prince–Blanch electrolyte suppression is not provided.
- The Lehr void-fraction factor and `α_max = 0.6` were checked only against a
  secondary reproduction, not the primary text.

## Bubbly-medium acoustics

`phydrax.acoustics.solve_bubbly_medium_dispersion` computes the linear
dispersion of a dilute bubbly liquid from a `BubblyMediumDispersionPlan`. The
plan holds one composed `RadialBubbleModel`, bin radii `R_b` with number
densities `n_b`, and angular frequencies. The mixture follows Commander &
Prosperetti (1989). With the bin response `R̂_b(ω)` per unit excess pressure
from `linear_bubble_response`,

    k² = ω²/c² − 4πρω² Σ_b n_b R_b² R̂_b(ω),

which is Commander & Prosperetti's eq. (41) summed over bins. The natural
frequency and the thermal, viscous, shell and radiation damping all come from
the linearized composed model; no second damping formula exists. The result
reports the decaying branch of `k`, the phase speed, the attenuation in Np/m
and dB/m, the void fraction and the bin resonances. The status is `SUCCESS`,
`RESPONSE_FAILURE`, `NONFINITE` or `OUTSIDE_DILUTE_LIMIT` (default bound
`β ≤ 10⁻²`). At `ω ≪ ω0` the dispersion reduces to the dilute limit
`1/c_m² = 1/c² + 3βρ/(3κp_g0 − 2σ/R)`. Wood's (1930) law is
`1/(ρ_m c_m²) = β/(ρ_g c_g²) + (1 − β)/(ρ_l c_l²)` (`wood_sound_speed`). With
`ρ_g c_g² = κp0` the two agree to `c_CP/c_Wood ≈ √(1 − β)`. At 1 % air in water
(1000 kg/m³, 1500 m/s; air 1.2 kg/m³, 343 m/s) Wood's law gives 119.05 m/s.
Near-resonance accuracy against experiment is not claimed, and neither are
multiple scattering beyond the effective medium, nonlinear propagation or
dense mixtures.

## Nonclaims

- **Bulk nanobubbles.** No stabilization mechanism (charge, skin, contamination)
  for free bulk nanobubbles is represented; free bubbles with Laplace pressure
  above the oversaturation always dissolve in this package.
- **Church density contrast.** `ChurchShell` assumes equal shell and liquid
  densities; the inertial terms of a denser shell are not modeled.
- **Compression-only burst edges.** Agreement with Marmottant et al. (2005,
  Fig. 5b) is claimed only inside the burst. The unstated burst envelope sets
  the first expansion and the post-burst recovery, and these are not
  reproduced. The ambient pressure of that case is assumed. The
  `(χ, σ₀)` parametrization cannot represent a shell buckled at rest
  (`R₀ < R_buckling`), so the left half of the paper's Fig. 7 is out of reach.
- **Tolman correction.** Off by default; the Tolman length of water is disputed.
- **Mass transfer during oscillation**, nonspherical modes, shock formation in
  the liquid (Kirkwood–Bethe emission) and sonochemistry are not modeled by the
  radial solvers; single-bubble solves do not translate (clouds may, see above).

## References

Rayleigh (1917); Minnaert (1933); Gilmore (1952); Epstein & Plesset (1950);
Keller & Miksis (1980); Prosperetti (1977, 1991); Löfstedt, Barber & Putterman
(1993); Church (1995); Hilgenfeldt, Lohse & Brenner (1996); Hoff, Sontum & Hovem
(2000); Toegel, Gompf, Pecha & Lohse
(2000); Marmottant et al. (2005); Sarkar et al. (2005); Yang & Church (2005);
Popov (2005); Doinikov & Dayton (2007); Preston, Colonius & Brennen (2007);
Doinikov, Haac & Dayton (2009); Paul et al. (2010); Stricker, Prosperetti &
Lohse (2011); Warnez & Johnsen (2015); Lohse & Zhang (2015); Le Métayer & Saurel
(2016); Marmottant–Gompertz shell, arXiv:2106.12004.

Clouds, emission, kernels and bubbly acoustics: V. F. K. Bjerknes, *Fields of
Force* (1906); L. A. Crum, J. Acoust. Soc. Am. 57, 1363 (1975); V. G. Levich,
*Physicochemical Hydrodynamics* (1962); L. Greengard & V. Rokhlin, J. Comput.
Phys. 73, 325 (1987); R. Mettin, I. Akhatov, U. Parlitz, C. D. Ohl &
W. Lauterborn, Phys. Rev. E 56, 2924 (1997), doi:10.1103/PhysRevE.56.2924;
D. Fuster & T. Colonius, J. Fluid Mech. 688, 352 (2011),
doi:10.1017/jfm.2011.380; M. J. Prince & H. W. Blanch, AIChE J. 36, 1485
(1990), doi:10.1002/aic.690361004; F. Lehr, M. Millies & D. Mewes, AIChE J. 48,
2426 (2002), doi:10.1002/aic.690481103; H. Luo & H. F. Svendsen, AIChE J. 42,
1225 (1996), doi:10.1002/aic.690420505; J. C. Lasheras et al., Int. J.
Multiphase Flow 28, 247 (2002), doi:10.1016/S0301-9322(01)00046-5; S. Kumar &
D. Ramkrishna, Chem. Eng. Sci. 51, 1311 (1996); K. W. Commander &
A. Prosperetti, J. Acoust. Soc. Am. 85, 732 (1989), doi:10.1121/1.397599;
A. B. Wood, *A Textbook of Sound* (1930).

Examples: `examples/advanced_acoustic_bubble.py`,
`examples/advanced_contrast_agent_microbubble.py`,
`examples/advanced_surface_nanobubble.py` and
`examples/advanced_bubble_cloud.py`.
