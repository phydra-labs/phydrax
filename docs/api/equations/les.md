# Large-eddy simulation equations

This page is the normative convention for Phydrax large-eddy simulation (LES).
The workflow and route matrix are in the
[large-eddy simulation guide](../../guides_large_eddy_simulation.md). Filter identity,
coefficient provenance, the prepared action, the numerical backend, and qualification
status are separate contracts; none may stand in for another.

## Filtered equations and sign convention

For constant-density incompressible flow, Phydrax uses

\[
\partial_i \bar u_i = 0,
\qquad
\partial_t \bar u_i + \bar u_j\partial_j\bar u_i
= -\partial_i\bar p + \nu\partial_{jj}\bar u_i
-\partial_j\tau_{ij}^{\mathrm{sgs}} + f_i .
\]

The exact Reynolds SGS stress is

\[
\tau_{ij}^{\mathrm{sgs}}
= \overline{u_i u_j}-\bar u_i\bar u_j.
\]

`les_reynolds_stress_target(..., convention="full")` returns that full tensor.
`convention="deviatoric"` subtracts one third of its trace. The algebraic runtime
models only the deviatoric specific stress

\[
\tau_{ij}^{d}=-2\nu_t S_{ij}^{d},
\quad
S_{ij}=\tfrac12(A_{ij}+A_{ji}),
\quad
A_{ij}=\partial_j\bar u_i,
\quad
S_{ij}^{d}=S_{ij}-\tfrac13 S_{kk}\delta_{ij}.
\]

`AlgebraicLESResult.specific_deviatoric_stress` is this \(\tau^d\), not its
negative and not a dynamic stress with density units. Every periodic, channel, and
MAC momentum action consumes the negative divergence, \(-\partial_j\tau^d_{ij}\).
The pointwise forward-transfer convention is

\[
\Pi=-\tau_{ij}^{d}S_{ij}.
\]

`AlgebraicLESResult.energy_transfer`, `les_energy_transfer_target`, and learned-stress
`local_transfer` therefore report positive values for resolved-to-subgrid transfer;
negative values are local backscatter. Isotropic SGS stress is not silently absorbed
into the algebraic pressure. It is either explicitly neglected or supplied through a
prognostic SGS-energy contract on routes that carry one.

## Resolved filter identity

`ResolvedLESFilter` is semantic metadata, not an executable convolution. It always
names three ordered axes and exactly one supported family:

| Family | Required scale rule | Meaning |
| --- | --- | --- |
| `sharp-fourier-projection` | `cutoff-equivalent` | Exact retained-mode Fourier projection |
| `implicit-grid-volume` | `volume-equivalent` | Grid/control-volume filter represented by the discretization |
| `explicit-filter` | `kernel-equivalent` | A separately specified executable kernel |

Topology is `tensor-product` or `unstructured`; boundary class is `periodic`,
`wall-bounded`, `open`, or `mixed`; commutation status is `commuting`, `modeled`, or
`unmodeled`; repeated-filter semantics is `idempotent`, `composed`, or `unmodeled`.
Sharp Fourier projection additionally requires periodic tensor-product topology,
commutation, and idempotence. The filter digest is namespaced and cannot be replaced by
a dealiasing-plan ID.

`LESFilterScale` retains positive physical directional widths with final axis
`(x, y, z)`. Its scalar equivalent is

\[
\Delta=(\Delta_x\Delta_y\Delta_z)^{1/3}.
\]

Directional models use each \(\Delta_j\) on the matching derivative axis. Structured
backends may retain one-dimensional width factors and materialize the broadcast local
scale only during evaluation.

`LESParameterProvenance` binds the resolved filter to one `discretization_id`, one
regime, a source kind (`user`, `literature`, `a-priori`, or `a-posteriori`), and unique
evidence IDs. Non-user sources require evidence. `model_id` identifies only a formula;
`prepared_id` additionally includes the concrete scalar coefficient and the complete
parameter provenance. Preparation rejects traced, negative, or non-finite
coefficients. A prepared model is nontrainable static state; use the unprepared plan
when differentiating with respect to its scalar coefficient.

Dealiasing, closure-data filters, resolved filters, and dynamic test filters remain
four different objects. In particular, the oversampling used to evaluate a nonlinear
stress does not alter the retained LES filter.

## Algebraic eddy-viscosity formulas

Let \(A_{ij}=\partial_j\bar u_i\), \(S=(A+A^T)/2\), \(S^d=S-\operatorname{tr}(S)I/3\),
and

\[
\beta_{ij}=\sum_k \Delta_k^2 A_{ik}A_{jk}.
\]

The public algebraic plans implement the following exact discrete formulas:

### Smagorinsky

\[
\nu_t=(C_s\Delta)^2\sqrt{2S:S}.
\]

`SmagorinskyLESPlan(coefficient)` stores \(C_s\), so the supplied coefficient is
squared by this formula.

### WALE

Define

\[
G^2=AA,
\qquad
S^{d,2}=\tfrac12(G^2+(G^2)^T)-\tfrac13\operatorname{tr}(G^2)I,
\]

\[
I_S=S:S,
\qquad
I_D=S^{d,2}:S^{d,2}.
\]

Then

\[
\nu_t=(C_w\Delta)^2
\frac{I_D^{3/2}}{I_S^{5/2}+I_D^{5/4}}.
\]

`WALELESPlan` returns exactly zero when the positive numerator/denominator branch is
inactive; it does not add an epsilon.

### Vreman

With \(B_\beta=((\operatorname{tr}\beta)^2-\beta:\beta)/2\),

\[
\nu_t=C_v\sqrt{\frac{B_\beta}{A:A}}.
\]

`VremanLESPlan` returns zero unless both \(B_\beta\) and \(A:A\) are positive.

### Anisotropic minimum dissipation

\[
\nu_t=C_a\frac{-\beta:S^d}{A:A}.
\]

`AMDLESPlan` returns zero unless both the numerator and denominator are positive. It
therefore encodes the dissipative branch directly and never produces negative eddy
viscosity.

All four plans return \(\tau^d=-2\nu_tS^d\) and \(\Pi=-\tau^d:S\). Their zero
branches and ratio activation boundaries are branchwise differentiable, not globally
smooth.

## Dynamic Smagorinsky contract

The backend, not `PreparedDynamicSmagorinskyPlan`, performs filtering. It supplies
already formed `DynamicLESInputs` with a Leonard tensor \(L\), a coefficient-free
modeled tensor \(M\), local algebraic inputs, exact dynamic provenance, and an explicit
accepted-update mask. Both tensors and the velocity gradient end in `(3, 3)` and have
identical leading shape. The procedure removes both traces and forms

\[
n=L^d:M^d,
\qquad
d=M^d:M^d,
\qquad
C_d=\frac{\langle n\rangle}{\langle d\rangle}.
\]

The resulting stress uses

\[
\nu_t=C_d\Delta^2\sqrt{2S:S},
\qquad
\tau^d=-2\nu_tS^d.
\]

Thus dynamic `coefficient` is \(C_d\), not \(C_s\); it is not squared. The
`GermanoLeastSquaresEvidence` retains pointwise and averaged contractions, the
effective denominator, unconstrained coefficient, policy activity, accepted/rejected
history counts, finiteness, filter-pair provenance, and differentiation semantics.

Averaging policies are explicit:

- `GlobalDynamicLESAveraging` averages all leading axes and is smooth.
- `HomogeneousPlaneDynamicLESAveraging(axis_names)` requires exactly three spatial
  leading axes, retains singleton dimensions, and is smooth.
- `LocalKernelDynamicLESAveraging(weights)` accepts an odd, normalized nonnegative 3-D
  periodic kernel and is smooth.
- `LagrangianDynamicLESAveraging(relaxation)` uses accepted-update exponential history,
  is branchwise, requires an explicit `LagrangianDynamicLESState`, and preserves
  rejected entries exactly. `initial_state()` binds the state to prepared ID and field
  shape.

The denominator policy is either the exact quotient with a zero-denominator zero branch
or a caller-supplied positive dimensional Tikhonov shift. The backscatter policy either
preserves signed \(C_d\), clips it to nonnegative values, or limits it below by a
specified fraction of a positive reference coefficient. No hidden clipping,
regularization, or history initialization occurs.

`DynamicLESProvenance` requires distinct resolved and test filter identities with the
same axes, dimension, topology, boundary class, and commutation status. Every
directional physical test/resolved width ratio must be concrete and greater than one.

The executable adapters are deliberately route-specific:

- `PeriodicDynamicLESPlan` uses exact coarse retained Fourier projection on a strictly
  coarser 3-D periodic Fourier discretization. Germano products use an
  `OversamplingDealiasingPlan` of at least 1.5, separately from both filters.
- `MACDynamicLESPlan` uses the separable `(1/4, 1/2, 1/4)` kernel, a directional width
  ratio of two, and only a periodic, uniform 3-D MAC grid with at least three cells per
  axis and no physical boundary sides.

Both adapters are compiler-integrated alternatives to static algebraic LES.
`compile_periodic_incompressible_flow(..., dynamic_les=...,
dynamic_test_discretization=...)` owns dynamic stages/restrictions; periodic
production wraps prepared ETDRK in `PreparedPeriodicDynamicETDRKMethod` and commits
Lagrangian history transactionally. `compile_mac_incompressible_flow(...,
dynamic_les=...)` owns the periodic-uniform MAC stage, while
`PreparedMACDynamicExplicitMethod` supplies projected, stability-gated explicit
stepping with transactional history. Static frozen MAC IMEX/SBDF2 remains specific
to stateless algebraic LES.

## Prognostic SGS kinetic energy

`KSGSState.kinetic_energy` is specific SGS kinetic energy \(k\) in m²/s². The other
state fields are restart-complete dynamic histories. Every coefficient in
`KSGSCoefficients` is required and positive; Phydrax supplies no literature defaults.
For \(\Delta=(\Delta_x\Delta_y\Delta_z)^{1/3}\), molecular kinematic viscosity
\(\nu\), and deviatoric strain \(S^d\), the static equation is

\[
\nu_t=C_\nu\Delta\sqrt{k},
\qquad
P_{raw}=2\nu_tS^d:S^d,
\]

\[
\epsilon=C_\epsilon\frac{k^{3/2}}{\Delta},
\qquad
D=\nu+C_D\nu_t,
\qquad
P=\min(P_{raw},C_{lim}\epsilon),
\]

\[
\partial_t k+u\cdot\nabla k=P-\epsilon+\nabla\cdot(D\nabla k)+B-\epsilon_{low-Re}.
\]

The backend must first call `transport`, apply its own conservative diffusion operator
with returned \(D\), and pass that evaluated diffusion rate into `KSGSInputs`.
KSGS does not invent a stencil, boundary condition, advection method, or time
integrator.

`BuoyancyKSGSPlan` adds

\[
B=-C_B\nu_tN^2,
\]

so positive stable \(N^2\) is a sink and negative unstable \(N^2\) is a source.
`LowReKSGSPlan` applies

\[
f_\nu=1-\exp(-C_f\Delta\sqrt{k}/\nu),
\qquad
\epsilon_{low-Re}=C_L\nu\lvert\nabla\sqrt{k}\rvert^2,
\]

and requires positive molecular viscosity. `DynamicKSGSPlan` requires a test filter
distinct from the resolved filter with matching axes, topology, boundary, and
commuting or modeled derivative semantics. The test family is explicit or sharp;
unmodeled repetition is refused and an explicit test filter must declare composed
repetition. Its concrete scale ratio exceeds one. It receives explicitly filtered
specific stress tensors whose contraction ratio is dimensionless \(C_\nu\), updates
exponential numerator/denominator histories only where `accept_update` is true, and
rejects negative dynamic coefficients. Physics uses the coefficient already in the
incoming state; an accepted sample changes the returned continuation for the next
evaluation.

All plans reject negative \(k\); no floor is enabled. The production limiter and
dynamic update masks are hard branches, and the square-root derivative is not regular
at \(k=0\). Differentiability claims therefore require positive interior states and a
fixed executed branch. `replace_ksgs_kinetic_energy` changes only \(k\) and preserves
the continuation fields.

The structured MAC backend integrates all four KSGS families in 3-D.
Static/buoyant plans admit periodic, free-slip, or symmetry impermeable boundaries.
Dynamic KSGS is periodic-uniform only and prepares the exact binomial test filter
with ratio two; requested updates commit only when the contraction numerator is
nonnegative and denominator positive. Low-Re KSGS requires at least one true
no-slip wall, computes cell-center distance only to no-slip sides, and does not
treat free-slip or symmetry sides as walls. The unstructured low-Mach constitutive
backend still integrates static KSGS; its pressure-stepped solver preserves that
continuation.

## Favre effective transport

For variable-density flow,

\[
\widetilde\phi=\frac{\overline{\rho\phi}}{\bar\rho}.
\]

`FavreLESFieldContract` fixes species order and accepts only the canonical SI units used
by the homogeneous gas system. `FavreLESInputs` requires positive density, temperature,
and pressure heat capacity; nonnegative mass fractions summing to one; finite velocity,
gradients, and partial specific enthalpies; and, for transported SGS energy,
nonnegative specific \(k\) plus its three-component gradient.

`PreparedFavreLESModel` wraps a prepared algebraic model and exact directional widths.
It requires an explicit positive turbulent Prandtl number, one ordered positive
turbulent Schmidt number for every named species, a finite nonnegative eddy-viscosity
upper bound, a nonnegative SGS-energy dissipation coefficient, a positive SGS-energy
turbulent Schmidt number, and one trace policy:

- `neglected`: SGS kinetic energy and its gradient are forbidden; isotropic stress is zero;
- `provided-sgs-kinetic-energy`: \(k\) and its gradient are required and
  \(\tau^{iso}=2\bar\rho kI/3\).

With algebraic specific \(\tau^d\), the returned conventional SGS covariances and
fluxes are

\[
\tau^{sgs}=\bar\rho\tau^d+\tfrac23\bar\rho kI,
\qquad
q^{sgs}=-\frac{\mu_t c_p}{Pr_t}\nabla\widetilde T,
\]

\[
J_s^{raw}=-\frac{\bar\rho\nu_t}{Sc_{t,s}}\nabla\widetilde Y_s,
\qquad
J_s=J_s^{raw}-\widetilde Y_s\sum_r J_r^{raw},
\]

so \(\sum_sJ_s=0\). The enthalpy flux is
\(q^{sgs}+\sum_s h_sJ_s\). The conservative flux image returned to the gas equation is

\[
F_s^{cons}=-J_s,
\quad
F_{mom}^{cons}=-\tau^{sgs},
\quad
F_E^{cons}=-\widetilde u\cdot\tau^{sgs}
-q^{sgs}-\sum_sh_sJ_s.
\]

With transported SGS energy, the model also returns
\(\rho k\), diffusion flux \(\mu_t\nabla k/Sc_k\), production
\(P_k=\Pi_d+\Pi_{iso}\), dissipation
\(\epsilon_k=C_\epsilon\rho k^{3/2}/\Delta\), and source
\(P_k-\epsilon_k\). That diffusion flux enters both the SGS-energy and total-energy
flux. Production/dissipation changes only the SGS-energy component; total-energy
source is exactly zero, making the exchange auditable.

Stress work therefore appears exactly once. The closure is physical SGS transport,
not a shock sensor, Riemann dissipation, limiter, bulk viscosity, or artificial
viscosity.

`HomogeneousMixtureCompressibleNavierStokesSystem(..., favre_les=model)` is the current
3-D compressible integration. The `provided-sgs-kinetic-energy` policy appends
`rho*k_sgs` after total energy in conserved state and `k_sgs` after temperature in
primitive state; total energy includes SGS energy, isotropic SGS pressure is
hyperbolic, and `FavreLESCoupledRate` separates its source and positivity step.
The neglected policy retains the smaller state. Both require conserved gradients;
the primitive-gradient-only convenience remains refused. Binding does not by itself
qualify every FV/DG application route.

## Backend support and refusals

| Route | Implemented surface | Exact boundary/filter limits |
| --- | --- | --- |
| Periodic spectral algebraic | Compiled equation, diagnostics, guarded ETDRK, statistics, production | Single-device, constant density, 3-D, full-complex periodic Fourier; sharp retained projection; oversampled stress |
| Periodic spectral dynamic | Compiled Germano equation, restriction, transactional ETDRK production | 3-D periodic Fourier; strictly coarser test grid; OU forcing is not composable with dynamic continuation |
| Distributed periodic algebraic | Full rotational flow, ETDRK/SSPRK, statistics, device-resident production/restart | Real JAX slab/pencil mesh; no host gather; backend qualification never inherited |
| Spectral channel algebraic | Compiled channel, enforced complete SBDF2 restriction, optional mixed normal-essential/equilibrium-traction owner | 3-D Fourier–Chebyshev–Fourier; unmodeled wall-normal filter noncommutation; equilibrium owner requires stationary walls and zero prescribed pressure gradient |
| MAC algebraic | Compiled momentum action and frozen implicit IMEX/SBDF2 | 3-D; periodic or free-slip/symmetry impermeable boundaries only; no no-slip/open/inflow |
| Fixed immersed MAC algebraic | Compiled masked SGS, pressure/marker IMEX/SBDF2, optional equilibrium wall traction, balance ledger | Single device, unit density, 3-D, stationary fixed marker route and cell fractions; periodic/free-slip/symmetry outer boundaries |
| MAC dynamic | Compiled Germano equation and projected explicit method | Periodic uniform 3-D only; no physical boundary sides; wall-plane production statistics remain incompatible |
| MAC scalar/ocean KSGS | Named scalar SGS plus static, buoyant, dynamic, or low-Re KSGS | Dynamic requires periodic-uniform binomial filtering; low-Re requires a true no-slip wall; scalar declarations remain complete |
| Learned stress | Periodic Fourier and periodic-uniform MAC divergence/projection backends | Bound 3-D unit-density feature ABI; MAC requires certified transform projection; no generic boundary fallback |
| Unstructured low-Mach Favre | Conservative tetrahedral transport plus pressure-corrected fixed-step continuation | Single device, fixed conforming 3-D tetrahedra, closed boundaries, exact fixed step; static KSGS only |
| Homogeneous compressible Favre | Conservative transport with optional appended `rho*k_sgs` state and coupled source | 3-D canonical SI gas fields, neglected or transported trace; numerical stabilization remains separate |
| LBM Smagorinsky | Athermal collision-local SRT relaxation correction | Unit lattice filter width, positive density, base relaxation rate in `(0, 2)`; separate from the filtered-equation closures above |

For MAC scalars, `MACScalarSGSField` requires exactly one of a positive turbulent
Prandtl number, a positive turbulent Schmidt number, or `no_sgs=True`. The prepared
field set must exactly match the required transported names. Supported scalar
boundaries are periodic, impermeable zero flux, or prescribed total flux.

`FixedImmersedMACLESPlan` composes the static MAC action with one fixed
`MACImmersedBoundaryProjectionPlan`. Caller-owned fluid fractions lie in `[0, 1]`,
contain active fluid, scale directional widths by `fraction**(1/3)` on active
cells, and weight eddy viscosity, deviatoric stress, and energy transfer; solid
cells receive zero SGS stress. Active markers must be stationary and their fixed
transfer relation must be successful. Moving, deforming, distributed, open-boundary,
or truncated-support requests are refused.

Without a wall model the marker solve enforces full no-slip. With a prepared
`VectorEquilibriumWallStressPlan`, it enforces the normal constraint and applies
the model's tangential wall-on-fluid traction. This additionally requires valid
marker normals/distances/roughness, positive molecular viscosity, wall-law
convergence, and dissipation. The filter, model, geometry, marker, boundary, and
immersed solver identities remain distinct.

`FixedImmersedMACLESPlan.imex_euler_method` and `.sbdf2_method` are both
implemented. `PreparedFixedImmersedMACLES.balance_ledger` accepts either result;
non-startup SBDF2 requires the exact input history so extrapolated SGS and wall
actions can be reconstructed. Qualification remains exact to the temporal support
named by the final campaign artifact.

`UnstructuredLowMachLESPlan` remains the constitutive transport owner and refuses
2-D/polyhedral, periodic/open, moving/coupled meshes, dynamic or low-Re KSGS, and
nonzero molecular bulk viscosity. `UnstructuredLowMachLESFixedStepMethod` adds a
gauged matrix-free pressure projection, fixed-step forward-Euler predictor/correction,
complete pressure/flux/restart continuation, advection/diffusion/source/positivity
restrictions, and atomic rollback. It retains shared-flux, conservation, pressure,
energy, and admissibility evidence.

With static KSGS, raw production is the negative work of the shared deviatoric
SGS face momentum flux against the owner-neighbour velocity jump. Each interior
face transfer is split equally between its two cells and divided by cell volume.
Negative raw production fails the route. The production limiter retains
\(P=\min(P_{raw},P_{ceiling})\); the reduction
\(P_{raw}-P\) is added exactly as a modeled enthalpy-density source rather than
discarded. Step evidence gates both
\(P_{raw}=P+P_{thermalized}\) and the total resolved-plus-KSGS-plus-enthalpy
energy balance.

`DistributedPeriodicLESPlan` wraps one prepared scientific action and refuses a
simulated topology, channel schedule, non-oversampled stress, or resource excess.
`compile_distributed_periodic_les` adds the complete rotational equation;
`DistributedPeriodicLESMethodPlan` supplies ETDRK2/4 or SSPRK33/54 with global
current-state admission; and `DistributedPeriodicLESProductionPlan` keeps segments,
statistics, checkpoints, and returned states device-resident. Qualification remains
backend-specific and is never inherited from one-device parity.

## Automatic differentiation

Phydrax differentiates the executed finite-dimensional JAX program:

- velocity gradients, smooth invariant formulas, fixed Fourier/MAC/filter actions,
  Favre interior transport, and distributed collectives are differentiable;
- plan metadata, filter/discretization/prepared IDs, topology, sharding, species order,
  policies, capacities, and prepared scalar coefficients are nontrainable;
- positivity checks, zero branches, production limiting, upwind selection, coefficient
  clipping/bounding, accepted-update masks, and solver acceptance are branchwise;
- Lagrangian/dynamic history must be explicit and is advanced only on its declared
  accepted branch;
- unstructured topology, physical-boundary selection, and test-filter resolution are
  fixed; no derivative through topology or route selection is claimed;
- a failed admissibility, finite, conservation, solver, or evidence gate has no valid
  derivative claim.

A learned stress with `energy_policy="signed"` reports smooth-discrete semantics for
the bound predictor/normalizer. Dissipative and bounded-backscatter policies are
branchwise. A model-error assimilation correction is a different object: it is an
additive divergence-free momentum rate and is explicitly not identifiable as SGS
stress.

## Public API

### Core algebraic and dynamic models

::: phydrax.equations.ResolvedLESFilter

---

::: phydrax.equations.LESFilterScale

---

::: phydrax.equations.LESParameterProvenance

---

::: phydrax.equations.AlgebraicLESInputs

---

::: phydrax.equations.AlgebraicLESResult

---

::: phydrax.equations.SmagorinskyLESPlan

---

::: phydrax.equations.WALELESPlan

---

::: phydrax.equations.VremanLESPlan

---

::: phydrax.equations.AMDLESPlan

---

::: phydrax.equations.PreparedAlgebraicLESModel

---

::: phydrax.equations.DynamicLESProvenance

---

::: phydrax.equations.DynamicLESInputs

---

::: phydrax.equations.DynamicSmagorinskyPlan

---

::: phydrax.equations.PreparedDynamicSmagorinskyPlan

---

::: phydrax.equations.GermanoLeastSquaresEvidence

---

::: phydrax.equations.LagrangianDynamicLESState

### Periodic, MAC, channel, and distributed adapters

::: phydrax.equations.PeriodicAlgebraicLESPlan

---

::: phydrax.equations.PeriodicFourierGridFilterPlan

---

::: phydrax.equations.PeriodicDynamicLESPlan

---

::: phydrax.equations.PeriodicFourierTestFilterPlan

---

::: phydrax.equations.MACAlgebraicLESPlan

---

::: phydrax.applications.incompressible_flow.FixedImmersedMACLESPlan

---

::: phydrax.applications.incompressible_flow.PreparedFixedImmersedMACLES

---

::: phydrax.applications.incompressible_flow.compile_fixed_immersed_mac_les_flow

---

::: phydrax.applications.incompressible_flow.FixedImmersedMarkerMotion

---

::: phydrax.applications.incompressible_flow.ImmersedMACLESStageResult

---

::: phydrax.applications.incompressible_flow.ImmersedLESBalanceLedger

---

::: phydrax.equations.MACDynamicLESPlan

---

::: phydrax.equations.MACExplicitTestFilterPlan

---

::: phydrax.equations.CompiledChannelLESDynamics

---

::: phydrax.equations.channel_les_filter

---

::: phydrax.equations.compile_channel_les

---

::: phydrax.discretization.DistributedPeriodicLESPlan

---

::: phydrax.discretization.PreparedDistributedPeriodicLES

### KSGS, Favre, and unstructured transport

::: phydrax.equations.KSGSCoefficients

---

::: phydrax.equations.KSGSState

---

::: phydrax.equations.StaticKSGSPlan

---

::: phydrax.equations.BuoyancyKSGSPlan

---

::: phydrax.equations.DynamicKSGSPlan

---

::: phydrax.equations.LowReKSGSPlan

---

::: phydrax.equations.FavreLESFieldContract

---

::: phydrax.equations.FavreLESInputs

---

::: phydrax.equations.FavreLESResult

---

::: phydrax.equations.PreparedFavreLESModel

---

::: phydrax.equations.UnstructuredLowMachLESPlan

---

::: phydrax.equations.PreparedUnstructuredLowMachLES

---

::: phydrax.equations.UnstructuredLowMachLESConservationEvidence

### Scalar, learned, and analysis contracts

::: phydrax.discretization.MACScalarSGSField

---

::: phydrax.discretization.MACScalarSGSPlan

---

::: phydrax.closure_data.LESFilterPair

---

::: phydrax.closure_data.FilterSpec

---

::: phydrax.closure_data.PeriodicLESAnalysisContext

---

::: phydrax.closure_data.prepare_periodic_les_analysis

---

::: phydrax.closure_data.LearnedStressBindingPlan

---

::: phydrax.closure_data.PreparedLearnedStressBinding

### Integrated runtime adapters

::: phydrax.applications.incompressible_flow.PreparedPeriodicDynamicETDRKMethod

---

::: phydrax.applications.incompressible_flow.PeriodicDynamicLESProductionState

---

::: phydrax.applications.incompressible_flow.PreparedMACDynamicExplicitMethod

---

::: phydrax.applications.incompressible_flow.MACDynamicLESProductionState

---

::: phydrax.applications.incompressible_flow.CompiledDistributedPeriodicLESDynamics

---

::: phydrax.applications.incompressible_flow.DistributedPeriodicLESMethodPlan

---

::: phydrax.applications.incompressible_flow.DistributedPeriodicLESStatisticsPlan

---

::: phydrax.applications.incompressible_flow.DistributedPeriodicLESProductionPlan

---

::: phydrax.applications.incompressible_flow.PreparedDistributedPeriodicLESProduction

---

::: phydrax.applications.incompressible_flow.PreparedVectorEquilibriumWallStressChannel

---

::: phydrax.applications.incompressible_flow.PreparedStochasticTurbulentInflowMACBoundary

---

::: phydrax.equations.PeriodicLearnedStressPlan

---

::: phydrax.equations.PreparedPeriodicLearnedStress

---

::: phydrax.equations.MACLearnedStressPlan

---

::: phydrax.equations.PreparedMACLearnedStress

---

::: phydrax.equations.FavreLESCoupledRate

---

::: phydrax.solver.UnstructuredLowMachLESFixedStepMethod

---

::: phydrax.solver.UnstructuredLowMachLESRestartState

---

::: phydrax.solver.UnstructuredLowMachLESStepEvidence
