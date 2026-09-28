# Resolved bubbly flow

This guide covers resolved gas bubbles in the structured two-phase VOF
application (`phydrax.applications.two_phase_flow`). The capabilities are:

- bubble identity from device connected components, with an explicit
  atmosphere and a deterministic lineage journal;
- a compartment registry that stores gas amount and internal energy for each
  bubble;
- a mixed MAC projection in which each closed compartment carries its own
  thermodynamic pressure;
- multi-marker VOF, so that nearby bubbles keep their identity;
- film-drainage-gated coalescence;
- surface tension that varies with a transported material scalar
  (thermocapillarity).

The liquid volume content `alpha V` stays the single geometric authority. Every
capability here is layered on the geometric PLIC transport, the balanced
capillary force and the variable-density projection of the two-phase guide.

## Bubble identity and the lineage journal

`BubbleComponentPlan` labels the gas support of `alpha` on the device every
step. It uses `phydrax.topology.ConnectedComponentPlan`, a FastSV-style
hook/shortcut iteration (Zhang, Azad & Hu 2020) over the face-neighbour
relation of the grid.

- **Bounded execution.** The iteration runs inside a bounded
  `lax.while_loop`. Roots are compacted by an exclusive prefix sum, with no
  sort and no `unique`. More components than `component_capacity` is a
  `CAPACITY_EXCEEDED` refusal; slots are never aliased.
- **Mixed-cell policy.** A cell belongs to the gas support when
  `1 - alpha >= gas_threshold` (default `1e-6`). This keeps every gas-bearing
  mixed cell inside its compartment.
- **Atmosphere.** Components touching a declared vent side are the atmosphere.
  They share the identity `ATMOSPHERE_ID = 0` and are never compartments.
- **Stable identities.** `phydrax.topology.ComponentTransitionPlan` groups the
  observed `(old, new)` label pairs with `phydrax.sparse.KeyGroupPlan`,
  weighted by gas volume. No `K x K` overlap matrix is formed. Marker-aware
  bubble tracking keeps pairs that are maximum-overlap for either endpoint;
  a minor cross-overlap is discarded only when both endpoints have a stronger
  partner. Genuine merge and split branches therefore remain explicit, while
  a transported touching pair does not spuriously reconnect when its dominant
  marker partition shifts by a mixed cell. A component that continues a single
  bubble keeps its identity. Every merge, split, reconnect, creation or
  entrainment receives fresh identities in root order.
- **Journal.** Any change of the set of bubble identities is a topology
  event. `transition_records` builds the canonical journal records at the host
  epoch boundary, one per bipartite-connected event group: `merge`, `split`,
  `reconnect`, `create`, `vanish`, `vent`, `entrain`. Each record lists
  parents, children, volumes and weighted overlaps.
  `BubbleTransitionJournal.lineage` returns every ancestor of an identity.
- **Derivatives.** Topology events set `derivative_available = False`.

## Gas compartments

`BubbleCompartmentPlan` keeps one slot per bubble identity. A slot stores the
gas amount, the internal energy, the law-internal state, the geometric volume,
the pressure and the centroid. The only thermodynamic authority is an
`AbstractBubbleCompartmentGasLaw` from `phydrax.bubble_dynamics`; the first
supported routes are `IsothermalIdealBubbleGasLaw` and
`CaloricIdealBubbleGasLaw`.

- **Compliance.** `C = -dV/dp` is the JVP of the law pressure along the law's
  own process direction. That is `V/p` for the isothermal law and
  `V/(gamma p)` for the adiabatic law.
- **Projection commit.** Adiabatic compartments take the exact backward-Euler
  work `dU = -p^{n+1} Q dt`. Isothermal compartments keep `U` and book the heat
  `p^{n+1} Q dt`. `first_law_residual` is zero to rounding, and `eos_residual`
  reports the linearization error of the implicit closure.
- **Topology events (host).** `transact` applies them through the law:
  - merge: `law.merge`, which conserves amount and energy and reports the
    mixing entropy;
  - split: `law.split`, with the uniform-intensive policy and zero entropy
    production;
  - reconnect: merge, then split;
  - create: initialize at the gas-volume mean of the local absolute pressure;
  - entrain: initialize at `p_atm`;
  - vanish and vent: move the amount and energy to the ledger.

  A refusal leaves the registry unchanged. Refusal causes are capacity, a
  missing parent and an inadmissible law state.
- **Tiny bubbles** are never frozen. They keep their compartment until the
  labeling loses them; the removal is then a journaled `vanish`.

## Multi-marker VOF without numerical coalescence

`MultiMarkerPlan` partitions the authoritative gas content of each cell among
`marker_capacity` colors.

- **Transport.** Markers replay the step's `TwoPhaseFluxBundle`, the per-sweep
  liquid and total face rates and dilation sources of the geometric transport.
  Each sweep's gas flux is split by the donor's color shares, and the
  non-flux gas source by the cell's own shares. The color sum therefore
  equals the authoritative gas content to rounding (`sum_residual`).
- **Identity.** Gas cells connect only to face neighbors of the same dominant
  color. Two touching bubbles of different colors stay two identities and
  two compartments.
- **Close-interface graph.** `proximity` finds, inside a static stencil, the
  nearest foreign component of the same color and of a different color for
  every gas cell. The observed component pairs are grouped with
  `KeyGroupPlan`, with bounded capacity and overflow evidence.
- **Recoloring** is a host transaction. For a same-color conflict, the
  larger identity takes the smallest color its close neighbors do not use,
  and its content moves cell by cell. Every move is conservative. The
  transaction refuses rather than aliasing when no color is free
  (`MARKER_CAPACITY_EXCEEDED`).

## Film-drainage-gated coalescence

The unresolved film between two bubbles of different colors is modeled by
one ledger per pair (`FilmContactLedger`).

- **Near-contact potential.** In every gas cell whose nearest foreign-color
  bubble lies closer than the proximity reach `r`, the plan adds the
  interfacial potential `phi_nc = -Pi_0 max(0, 1 - d/r)`. The balanced
  capillary operator's face average turns it into the face force
  `phi_nc (G alpha)`. It acts like a disjoining pressure and keeps the
  resolved film from draining numerically below the grid scale.
- **Load and work.** Half the potential pressure times the PLIC facet measure
  of both interfaces is the pair load `F`. The same `F`, together with the
  potential's work, enters the pair's ledger slot. Its drainage law uses that
  `F`, so the load is counted once.

The drainage laws are closed-form in time for piecewise-constant load. The
ledger integrates them exactly over each step.

| regime (axisymmetric, 3D) | law | source |
|---|---|---|
| immobile | `dh/dt = -8 pi sigma^2 h^3 / (3 mu_c F R_eq^2)` | Reynolds squeeze film with `F = pi a^2 (2 sigma / R_eq)` |
| partially mobile | `dh/dt = -(4 sqrt(3) k sigma / (3 mu_d R_eq a)) h^2`, `k = 0.66` | plane-film model of Chesters (1988, 1991), as stated by Abid & Chesters (1994), Int. J. Multiphase Flow 20:613, eqs. [21], [23]; `k` fitted to Yiantsios & Davis (1990) |
| fully mobile | `dh/dt = -2 sigma h / (3 mu_c R_eq)` | extensional plug flow of the film |

Further definitions and planar variants:

- The equivalent radius is `1/R_eq = (1/R_1 + 1/R_2)/2`
  (Abid & Chesters 1994, eq. [1]). A bubble facing the atmosphere uses
  `R_2 = inf`.
- The rupture thickness is `h_c = (A R_eq / (8 pi sigma))^{1/3}`
  (Abid & Chesters 1994, eq. [29a], from Chesters 1991), or a declared value.
- **Planar films** (2D grids, per unit depth) derive from the same lubrication
  arguments. Immobile: `dh/dt = -sigma^3 h^3 / (mu_c F'^2 R_eq^3)`. Fully
  mobile: `dh/dt = -sigma h / (4 mu_c R_eq)`.
- The partially mobile planar case has no primary source here and is
  refused.
- The validity ratios of Abid & Chesters (1994) §5 (plug flow, small slope)
  are reported as evidence.

The constants were checked against the rendered pages of the TU/e
open-access copy of Abid & Chesters (1994). The 1991 review (Trans. IChemE
69A:259) could not be accessed. Only the 1991 constants restated in the 1994
paper are used.

**Outcome.**

- A draining pair whose film reaches `h_c` is `MERGED`, at the closed-form
  crossing time.
- A pair whose contact support is lost first is `RELEASED` (bounce).
- A merge is a host transaction: the absorbed bubble takes the kept bubble's
  color (the smaller identity, or the atmosphere), the pair becomes exempt
  from recoloring, and the potential between them vanishes. When the
  resolved gas joins, the identity merge and `law.merge` of the compartments
  follow.

## Integrated VOF step and transactions

`IncompressibleTwoPhaseVOFMethod(..., bubbles=plan)` carries the identity,
marker, compartment and contact state in the continuation. Each attempted
step transports the geometric VOF and marker flux bundle, applies viscosity,
adds capillary and near-contact face actions, performs the incompressible or
mixed compartment projection, then advances the bubble records. The
near-contact action is deliberately after viscosity and has its own
`contact_work` ledger term; mixed projection work is separately
`pressure_work`.

The fluid candidate, flux bundle and complete bubble state commit under one
acceptance predicate. A failed geometric, projection, capacity, marker,
registry or contact decision returns the previous accepted continuation
unchanged. Host topology and recoloring transactions are then performed by
`run_bubbly_flow`; a refused transaction also leaves the prior epoch intact.

### Checkpoint parameter realization

`BubblyFlowPlan.plan_id` and
`IncompressibleTwoPhaseVOFMethod.method_id` identify immutable structure; they
deliberately exclude trainable values. `BubblyFlowPlan.parameter_realization_id()`
is the host-side content fingerprint of the current declared parameter leaves,
including nested compartment gas, environment, variable-surface-tension and
film-drainage parameters. It is recomputed when called, so an accepted training
update changes the realization without rebuilding the plan.

`write_two_phase_checkpoint` evaluates and stores that realization separately
from the structural IDs. `read_two_phase_checkpoint` evaluates it again and
requires an exact match before restoring arrays. The checkpoint therefore
refuses a restart under changed near-contact, atmosphere, gas-law or drainage
physics even when the method structure is unchanged. Only the fingerprint is
persisted; parameter values are neither copied into the continuation state nor
added to device runtime data.

## Variable surface tension from a material scalar

`IncompressibleTwoPhaseVOFPlan(surface_tension_law=..., surface_tension_scalar=...)`
declares a `VariableSurfaceTensionPolicy` of the canonical capillary owner.
Its evaluator returns `sigma(q)` and `d sigma/dq`, for example
`LinearSurfaceTensionLaw`, `sigma = sigma_0 + sigma_q (q - q_0)`, with dynamic
leaves. The declared material scalar `q` (for example temperature) is a
mixture scalar content `q V`.

- **Transport.** It is transported conservatively by replaying the step's
  total face-flux bundle. Its global content closes on periodic or impermeable
  boundaries; a uniform value remains uniform to the accepted projection's
  divergence residual.
- **Normal force.** The balanced potential becomes `phi = sigma(q) kappa` on
  usable curvature cells.
- **Tangential force.** `MACBalancedCapillaryOperator` adds the Marangoni face
  force `(grad_s sigma)_d delta_f` (Brackbill, Kothe & Zemach 1992; Seric,
  Afkhami & Kondic 2018). The scalar `delta = |grad alpha|` comes from the
  canonical PLIC facet measure divided by cell volume, with the consistent
  cell-gradient norm used only when primary facet geometry is unavailable,
  and is conservatively averaged to each MAC face. Every component uses this
  scalar face delta rather than its own directional alpha derivative. Missing
  delta geometry is explicit unsupported-face evidence and refuses the step.
  The PLIC normal used to project `grad_s sigma` is the same geometry used by
  the balanced normal force.
- **Capillary step limit.** It uses the largest interfacial `sigma`.

The variable-sigma route requires zero constant `surface_tension` in the
material, so the law is the single authority. A volume scalar is never called
a surfactant concentration: VOF surfactant `Gamma` needs a dedicated
interfacial-transport state and is not claimed.

### Thermocapillary migration reference (Young, Goldstein & Block 1959)

A drop of radius `a` and viscosity `mu'` in a fluid of viscosity `mu`, in the
Stokes and zero-Marangoni-number limit, migrates in an imposed gradient `G`
of the scalar `T`. With `sigma_T = d sigma/dT`, conductivities `k` (outer) and
`k'` (drop), the axisymmetric (3D) velocity is

```text
U_3D = -2 sigma_T G a / ((2 mu + 3 mu') (2 + k'/k)).
```

The 2D planar analogue (a circular cylinder) follows from the same
stream-function construction. Take the outer and inner stream functions
`psi = (-U r + U a^2 / r) sin(theta)` and `psi' = (-U r^3/a^2 + U r) sin(theta)`
(force free, so no `r ln r` term). The surface temperature is
`T_s = 2 G a cos(theta) / (1 + k'/k)`. The tangential stress balance
`tau_out - tau_in + (1/a) d sigma/d theta = 0` then gives

```text
U_2D = -sigma_T G a / (2 (mu + mu') (1 + k'/k)).
```

Applied in 3D, the same calculation reproduces the YGB formula exactly. This
was verified symbolically. The qualification scenario uses `k' = k`, hence
`U_2D = -sigma_T G a / (4 (mu + mu'))`. It independently integrates the
initial discrete Marangoni traction against the face dual measures; the
continuum value is `F_x = sigma_T G pi a`. Applying the analytic Stokes
force-balance ratio `F_x^h / F_x` to `U_2D` gives the discrete YGB migration
prediction used for refinement. A separate short transient checks only
migration direction and material-scalar conservation, not steady-state speed.

Primary references are Young, Goldstein & Block, “The motion of bubbles in a
vertical temperature gradient,” *Journal of Fluid Mechanics* 6 (1959)
350–356, doi:10.1017/S0022112059000684, and Brackbill, Kothe & Zemach,
“A continuum method for modeling surface tension,” *Journal of Computational
Physics* 100 (1992) 335–354, doi:10.1016/0021-9991(92)90240-Y.

## Examples and bounded qualification

- `examples/advanced_bubble_coalescence.py` runs the integrated multi-marker,
  near-contact and drainage workflow and reports the separate contact and
  pressure work ledgers.
- `examples/advanced_thermocapillary_droplet.py` reports a short transient
  migration velocity beside the two-dimensional Young–Goldstein–Block
  reference and the material-scalar conservation residual.
- `tools/bubbly_flow_qualification.py` owns the longer closed/vented breathing,
  Boyle-rise and thermocapillary campaigns. Its bounded thermocapillary gate
  runs two grid resolutions, requires supported interface-delta geometry and
  conservative scalar transport at both resolutions, and requires the
  force-balance YGB migration prediction to improve on refinement. It also
  records the short transient velocity without treating it as a converged
  Stokes solution. Smoke examples are not qualification evidence.

## Mixed MAC compartment projection: derivation (research gate D3)

This derivation was written before the implementation. The route
`solver.MACCompartmentProjectionPlan` exists only because each step below holds.

### Discrete setting

- **Grid.** Cells `c` have volume `V_c`. MAC faces `f` have area `A_f`, centre
  distance `d_f` and dual measure `M_f = A_f d_f`.
- **Operators.** `D` is the per-unit-volume MAC divergence and `G` the face
  gradient of `PreparedMACOperators`. On wall and periodic grids they satisfy
  the discrete adjoint identity

  ```text
  <phi, D u>_V = - <G phi, u>_M,   <a, b>_V = sum_c V_c a_c b_c,   <a, b>_M = sum_f M_f a_f b_f.
  ```

- **Projection operator.** For the face coefficient `c_f = dt / rho_f`, let
  `L phi = -D(c G phi)`. `L` is `V`-self-adjoint and positive semidefinite. On a
  connected wall or periodic grid its kernel is the constant field.
- **Compartments.** There are `K` closed gas compartments `b` with disjoint
  cell supports `chi_b` (from bubble identity). The gas volume content of a
  cell is `g_c = V_c (1 - alpha_c)`, and the compartment volume is
  `V_b = sum_c chi_b(c) g_c > 0`. The weight field is

  ```text
  w_b(c) = chi_b(c) g_c / (V_c V_b),   so   <w_b, 1>_V = 1,   <w_b, x>_V = gas-volume mean of x over b.
  ```

  An optional atmosphere component `a` (gas connected to a declared vent side)
  has weight `w_a` built the same way.

### Continuous model and its discrete counterpart

The gas density is negligible, so each compartment is homobaric: its pressure
`p_b(t)` is uniform. The low-Mach gas continuity equation then gives a uniform
relative dilatation of the gas:

```text
div u = -(1 / (Gamma_b p_b)) dp_b/dt   in compartment b;   integrated: dV_b/dt = Q_b.
```

`Gamma_b = 1` for an isothermal compartment and `gamma` for an adiabatic ideal
compartment. Distributing `Q_b` over the gas volume gives the discrete
kinematic constraint

```text
D u = sum_b w_b Q_b + w_a Q_a.                                   (K1)
```

The thermodynamic closure is a backward-Euler step, linearized about the
current compartment state:

```text
p_b^{n+1} = p_b^n + pi_b,   Q_b = -(C_b / dt) pi_b,   C_b = -dV/dp |process = V_b / (Gamma_b p_b) > 0.   (T1)
```

`C_b` is evaluated from the compartment gas law `A` by a JVP of its pressure
along the reversible-work direction `(dV, dU) = (1, -p)`. For an isothermal
ideal law the energy has no effect on pressure, so this reduces to `V/p`.

### Variational origin: symmetry, coupling and gauge

The projected face velocity `u`, the compartment volume rates `Q_b` and the
atmosphere rate `Q_a` are the stationary point of the discrete Lagrangian

```text
Lagr(u, Q, p) = (1 / (2 dt)) ||u - u*||^2_{rho M}
              + (1 / dt) sum_b U_b(V_b^n + dt Q_b) - p_atm Q_a
              - <p, D u - sum_b w_b Q_b - w_a Q_a>_V.
```

Here `U_b(V)` is the internal energy along the compartment process, so
`dU_b/dV = -p_b`. `p` is the absolute pressure multiplier.

- **Momentum.** Stationarity in `u` gives the momentum update
  `u = u* - (dt/rho_f) G p`. Only the dynamic part `phi = p - h` enters the
  momentum, because the reference and hydrostatic offset `h` is balanced by the
  reduced-gravity interfacial potential of the two-phase step.
- **Compartments.** Stationarity in `Q_b` gives the mechanical coupling
  `<w_b, p>_V = p_b^{n+1}` (C1). The gas-volume mean of the pressure equals the
  compartment pressure.
- **Atmosphere.** Stationarity in `Q_a` gives `<w_a, p>_V = p_atm` (C2).

After linearizing `U_b` as in (T1), the stationarity system in the unknowns
`(phi, Q, Q_a)` is the symmetric KKT operator

```text
[ L          -W         -w_a ] [ phi ]   [ f   ]        f   = -D u*
[ -W^*    -diag(dt/C)     0  ] [ Q   ] = [ -r  ]        r_b = p_b^n - <w_b, h>_V
[ -w_a^*      0           0  ] [ Q_a ]   [ -r_a]        r_a = p_atm - <w_a, h>_V
```

The pairing is `V (+) R^K (+) R`, with `W Q = sum_b w_b Q_b` and
`W^* phi = (<w_b, phi>_V)_b`. The operator is self-adjoint in that pairing
because the same weights `w_b` distribute the dilatation (K1) and average the
pressure (C1). The adjoint of `Q -> W Q` is `phi -> W^* phi`. This pairing is
what makes the block symmetric and gives the work identity below.

**Dimensional check.**

- `phi` [Pa], `c` [s m^3 kg^-1], `L phi` [s^-1].
- `w_b` [m^-3], `Q_b` [m^3 s^-1], `C_b` [m^3 Pa^-1], so `dt/C_b` [Pa s m^-3].
- Row 1 is in [s^-1]; rows 2 and 3 are in [Pa].

**Independent constraint basis and gauge.**

- The columns `w_b` (and `w_a`) have disjoint supports and unit `V`-mass, so
  they are linearly independent. Every active compartment needs `V_b > 0`; an
  empty active compartment is refused as an invalid constraint.
- **At least one closed compartment.** The eliminated operator
  `L + sum_b (C_b/dt) w_b <w_b, .>_V` is positive definite. For
  `x^*Ax = 0`, `x` must be constant and `<w_b, x> = 0`, so `x = 0`. The
  absolute pressure level is then determined by the gas; no gauge is imposed.
- **Atmosphere present.** Its row pins the constant mode, because
  `<w_a, 1>_V = 1 != 0`.
- **Neither present.** The continuity rows are dependent: their sum is the
  net boundary flux. The dependent row is removed and the mean-zero gauge is
  imposed, which recovers `MACVariableDensityProjectionPlan` exactly.
- Incompressible compartments (`C_b = 0`) are outside this route.

### Tiny compartment Schur route

Write `phi = psi + mu 1` with `<1, psi>_V = 0`, and let `L_g` be the gauged
operator already prepared by the variable-density projection. Define

```text
z_0 = L_g^{-1} P0 f,   z_b = L_g^{-1} P0 w_b,   z_a = L_g^{-1} P0 w_a,   Z_ij = <w_i, z_j>_V,
```

where `P0` removes the volume mean. The three solves share one prepared
operator and run as one batch. They use unpreconditioned PCG: for these
localized mean-free basis right-hand sides, diagonal scaling across a sharp
density jump can increase the gauged operator condition number enough to make
a strict true-residual tolerance unattainable in floating-point arithmetic.
The unscaled system retains the same equation and tolerance and converges
within the declared iteration budget. `Z` is symmetric positive semidefinite,
since it is the Gram matrix of the `P0 w_i` in the `L^+` metric. The remaining
unknowns
`y = (Q, Q_a, mu)` solve the symmetric bordered system

```text
[ Z_cc + diag(dt/C)   Z_ca   1 ] [ Q   ]   [ r   - W^* z_0   ]
[ Z_ac                Z_aa   1 ] [ Q_a ] = [ r_a - <w_a,z_0> ]
[ 1^T                 1      0 ] [ mu  ]   [ <1, D u*>_V     ]
```

The last row is the volume compatibility `sum_b Q_b + Q_a = <1, D u*>_V`.

- **Nonsingularity.** The bordered matrix is nonsingular whenever
  `K + n_atm >= 1`. On the subspace `sum y = 0`, `Z + diag(dt/C)` is positive
  definite, and the border vector is nonzero.
- **Recovery.** `psi = z_0 + sum_b Q_b z_b + Q_a z_a`, `pi_b = -(dt/C_b) Q_b`,
  and `u = u* - c G psi`.
- **Consistency check.** Then
  `D u = D u* + L psi = sum_b w_b Q_b + w_a Q_a`, which is exactly (K1).
- **Cost.** One pressure operator and `K + n_atm + 1` right-hand sides. The
  dense work is limited to a `(K + n_atm + 1)^2` matrix with a static bound on
  `K`.

### Discrete work identity

The discrete work identity follows from the adjoint identity and (K1):

```text
<rho (u - u*), u>_M = -dt <G phi, u>_M = dt <phi, D u>_V
                    = dt sum_b Q_b <w_b, phi>_V + dt Q_a <w_a, phi>_V.
```

With (C1) and (C2), `<w_b, phi> = p_b^{n+1} - <w_b, h>`. Using
`<rho(u - u*), u> = KE(u) - KE(u*) + (1/2)||u - u*||^2_rho`:

```text
KE^{n+1} - KE* + (1/2)||u - u*||^2_{rho M}
   = dt sum_b p_b^{n+1} Q_b + dt p_atm Q_a - dt (sum_b Q_b <w_b, h>_V + Q_a <w_a, h>_V).
```

The compartment update closes the identity:

- **Adiabatic compartment.** The energy update is
  `U_b^{n+1} = U_b^n - dt p_b^{n+1} Q_b`, which is exact backward-Euler
  `-p dV` work. The pressure work done on the liquid by the gas therefore
  equals the loss of compartment internal energy, to rounding.
- **Isothermal compartment.** The same work is balanced by the heat
  `dt p_b^{n+1} Q_b` exchanged with the liquid bath.

The `h` term is the work of the reference and hydrostatic pressure on the
displaced liquid. The projection result reports every term separately, together
with the projection dissipation `(1/2)||u - u*||^2`, which is never negative.

### Adjoint

- The solution map `(f, r, r_a) -> (phi, Q, Q_a)` is the inverse of a
  self-adjoint operator, so it is self-adjoint in the same pairing.
- For fixed compartment supports and fixed `alpha`, the projection is linear
  in `(u*, p^n, p_atm)` and affine in `h`. Its fixed-topology JVP is the same
  solve applied to the tangent right-hand side.
- The `block_operator` of the plan exposes the KKT action and an independently
  coded transpose action. Duality `<y, A x> = <A^T y, x>` and self-adjointness
  of the solve are tested.
- A topology event (component merge, split, creation or vanishing) changes
  `chi_b`. No derivative is claimed across it.

### Result

Each gate item holds:

- dimensional consistency;
- declared symmetry, from the variational origin;
- an independent constraint basis;
- an explicit gauge, with dependent-row removal only in the compartment-free
  case;
- an exact discrete work identity;
- adjoint self-duality.

The route is therefore implemented. It is limited to low-Mach, homobaric gas
compartments with finite compliance. Acoustic waves inside the gas and
non-uniform gas pressure are not represented. For those, use the compressible
two-material route `TwoMaterialVOFSystem`.

## Evidence exchange with reduced bubble dynamics

`resolved_bubble_evidence_records` is a host-side, read-only conversion from a
committed `BubbleTransitionRecord` and `BubbleCompartmentState`. It emits one
`ResolvedBubbleEvidenceRecord` per event-side compartment in stable bubble-ID
order. Each record preserves the gas amount, internal energy, law state and
support, equivalent-volume radius, centroid, lineage, epoch, time, ambient
state and source-realization identity.

Translational momentum and liquid impulse are optional evidence keyed by stable
bubble ID. Omission produces `None` and sets the corresponding
`BubbleExchangeInvariant` bit; the converter never substitutes a zero. A
literal zero vector counts as available only when the caller explicitly
supplies it. Merge parent records and split child records retain the complete
parent/child lineage, while vanish and entrain records can become an
unaccepted `ReducedBubbleRequestRecord`:

```python
evidence = resolved_bubble_evidence_records(
    transition,
    compartments,
    time=time,
    source_realization_id=run_id,
    law_id=law.law_id,
    evaluation=evaluation,
    environment=environment,
    translational_momentum=momentum_by_id,
    liquid_impulse=impulse_by_id,
)
request = reduced_bubble_request(
    evidence[0], "single-bubble", "keller-miksis"
)
```

Conversely, `reduced_bubble_failure_from_result` accepts only
`BubbleCloudResult` or `SingleBubbleResult` values that ended with `OVERLAP`,
`SUPPORT_EXIT` or `VALIDITY_EXCEEDED`. It creates a
`ReducedBubbleFailureRecord` requesting a named resolved model. It neither
constructs that model nor modifies the reduced result.

These are evidence and request records, **not automatic solver handoffs**.
`accepted` defaults to false, and `handoff_ready` is always false. Even when
all recordable scalar/vector evidence is present, the missing-invariant mask
retains flow-field removal for resolved-to-reduced requests or flow-field
initialization for reduced-to-resolved requests. A real coupled transaction
must conserve gas amount, internal energy, translational momentum and liquid
impulse while consistently initializing or removing the surrounding resolved
flow field. The module does not claim or perform that transaction. For a 2-D
unit-depth resolved calculation, the stored compartment measure is not a
physical 3-D bubble volume unless the application supplies that interpretation;
the converter does not infer one.
