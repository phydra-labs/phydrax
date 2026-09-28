# Incompressible two-phase VOF hydrodynamics

`IncompressibleTwoPhaseVOFPlan` is a separate fixed-grid one-fluid product for
interface topology changes, breaking, air cavities, contact, and impact on structured
two- and three-dimensional grids. It does not reuse graph eta as interface state.

Status: **candidate**. The product is not a qualified capability until the Hysing
rising-bubble campaign (`tools/two_phase_hysing_qualification.py`) passes its
reference gates.

## Authoritative state

`TwoPhaseVOFState` stores:

- liquid volume content `alpha * cell_volume`;
- density-weighted extensive face momentum;
- phase-confined scalar contents;
- an auxiliary signed interface indicator `(alpha - 1/2) h_min` in length units
  (the level set).

VOF alpha is the interface authority. PLIC, height functions, and the level set are
derived geometry; the level set never overwrites alpha.

Initial alpha must lie in `[0, 1]`. Curved interfaces need area-accurate fractional
alpha, from exact geometry or at least 8 x 8 sub-cell samples per cell, and several
cells per radius of curvature. A sharp 0/1 disk is a staircase: its height-function
curvature is refused as under-resolved.

## Material, walls, and gravity

`TwoPhaseMaterialPlan` declares liquid and gas density and viscosity, surface
tension, and contact angle.

Periodic axes wrap. Every nonperiodic axis is an impermeable wall; `wall_sides`
selects `no-slip` or `free-slip` `MACBoundarySide`s, optionally with a prescribed
tangential wall velocity. The default is no-slip walls.

`gravity=(g_x, g_y[, g_z])` requires `hydrostatic_reference`, the point `Z` at which
the absolute pressure equals `reference_pressure`. Gravity components along periodic
axes are refused because the hydrostatic pressure is not periodic along them.

## Geometric phase transport

Liquid volume is transported by exact PLIC geometry with directional splitting
(Weymouth & Yue 2010; Aulisa et al. 2007 for three-dimensional split advection):

- each sweep moves the exact swept PLIC volume through every face;
- the dilation term `c (div u)` uses the step-constant indicator `c = [alpha^n > 1/2]`;
- the sweep order is the cyclic rotation `(k, k + 1, ...)` with `k = step_index mod D`.

Liquid volume is conserved to the projection's divergence residual. It stays in
`[0, 1]` up to a declared rounding excursion when every directional Courant number is
at most 1/2. There is no limiter and no clipping.

Momentum is transported with the same face liquid and gas mass fluxes, so density and
momentum stay consistent with the phase transport.

The face density is the arithmetic mean, i.e. the mass of the staggered face volume.
Momentum, kinetic energy, the viscous stage, and the projection's `1/rho_f` all use
this one density.

The splitting is first order in time. The step is: geometric transport with the
consistent momentum flux, implicit viscosity, balanced interfacial force, then
variable-density projection. The balanced force is added after the viscous stage
so its gradient part reaches the projection unchanged. If the viscous solve came
first, it would diffuse part of that gradient into a velocity that is not a
gradient, which the projection cannot remove. At density ratio 1000 (Hysing
test case 2) that ordering produced spurious gas velocities of about 0.1 per step.

## PLIC reconstruction

`StructuredPLICPlan` reconstructs the plane `m . xi = beta` in the scaled cell
coordinates `xi` in `[0, 1]^D`:

- the normal points out of the liquid;
- volume ↔ offset uses the closed form of Scardovelli & Zaleski (2000), evaluated
  without cancellation;
- the cubic branches of the inverse are polished by `phydrax.nonlinear.LocalRootPlan`
  and report a residual;
- normals are centered-column or Youngs estimates;
- facet measures, facet centroids, and directional swept fractions are exact.

Wall-adjacent mixed cells take the material contact angle as their PLIC normal.

## Curvature: height functions and status

`HeightFunctionCurvaturePlan` estimates curvature (Cummins, Francois & Kothe 2005;
Popinet 2009). For each interface cell:

1. Column axes are ranked by the PLIC normal.
2. Heights are summed over 7-cell columns for the `3^(D-1)` neighbouring columns.
3. Complete columns give a second-order curvature and the interface crossing.
4. Cells without complete columns fit a quadratic graph in the target-normal
   tangent frame to primary valid PLIC facet centroids inside a fixed
   `(2 r + 1)^D` stencil. The default `r = 3` capacity is 49 facets in two
   dimensions and 343 in three dimensions. Facet measure, distance, and normal
   alignment determine the weights.
5. The local native solve must have enough support, full quadratic rank,
   condition number at most `min(10^6, sqrt(0.1 / eps))`, and weighted RMS
   residual at most `0.35` cell widths. The dtype-safety rule, limits, radius,
   capacity, alignment threshold, distance weight, and rank cutoff are fingerprinted.
   Only primary valid PLIC facets are samples; fallback curvature never feeds another fallback.

The active set contains mixed PLIC cells and cells adjacent to a represented
phase jump. A pure active cell has no target facet of its own: its fitted graph
uses only neighbouring primary valid PLIC facets and supplies the target
crossing as interface-position evidence.

Every interface cell gets one status:

| Status | Meaning | Use |
|---|---|---|
| `VALID` | complete height-function columns | used |
| `FALLBACK` | supported, full-rank, well-conditioned bounded PLIC-centroid quadratic fit | used, counted, and accompanied by support/rank/condition/residual evidence |
| `UNDERRESOLVED` | bounded stencil exhausted, insufficient support or rank, excessive condition/residual, or curvature beyond the resolution bound | refused |

Any `UNDERRESOLVED` cell makes the step unsuccessful.

Height functions are prepared only when surface tension or gravity is present. They
then need uniform spacing and at least 7 cells along every axis; other grids are
refused at preparation.

## Balanced interfacial force

The sign convention is shared by every capillary action in
`phydrax.discretization.finite_volume`:

- `alpha` is the liquid fraction;
- the normal `n = -grad(alpha) / |grad(alpha)|` points out of the liquid;
- the curvature is `kappa = div(n)`, which is `+1/R` for a circular drop and `-1/R`
  for a circular bubble;
- the Laplace jump is `p_liquid - p_gas = sigma * kappa`.

`MACBalancedCapillaryOperator` applies the interfacial potential `phi` as the face
force `phi_f (G alpha)_f`. Here `G` is the exact MAC projection gradient, and `phi_f`
averages the usable cell potentials adjacent to the face (Brackbill, Kothe & Zemach
1992; Francois et al. 2006; Popinet 2009).

A spatially constant potential is then exactly the projection gradient of
`phi * alpha`: a static interface carries the Laplace jump in pressure instead of
driving a parasitic current. A face with an alpha jump but no usable adjacent
curvature is counted as unsupported, and the step is refused.

The force is evaluated at `alpha^{n+1}`.

### Reduced gravity and absolute pressure

Gravity enters the same balanced potential:

`phi = sigma kappa - (rho_l - rho_g) g . (x_I - Z)`

Here `x_I` is the local interface position: the height-function crossing, the exact
PLIC facet centroid on a mixed fallback, or the fitted crossing on a pure fallback.

The stored pressure is the projection pressure, which is the dynamic pressure of this
reduced-gravity formulation. The absolute pressure is

`p_abs = p_ref + (p - p(c_Z)) + rho(alpha) g . (x - Z)`

where `c_Z` is the cell containing `Z`. It is available from
`PreparedIncompressibleTwoPhaseVOF.absolute_pressure(alpha, pressure)` and
`TwoPhaseVOFView.absolute_pressure`. A flat interface at rest remains at rest to the
solver tolerance and reproduces the hydrostatic absolute pressure.

## Viscosity

Before the projection, the native `phydrax.solver.MACVariationalViscosityPlan` solves

`(rho_f + dt A_mu) u = rho_f u_star - dt b_mu`

where:

- `A_mu` is the variable-viscosity action `-div(2 mu S_d)` of
  `PreparedMACVariationalViscosityAction`;
- `b_mu` is its prescribed-wall offset, with wall kinds taken from the plan's
  `wall_sides`;
- `rho_f` is the arithmetic face density.

The system is assembled in measure-weighted SPD form. It is prepared once, with one
native conjugate-gradient solve, and refreshed with the runtime density and viscosity
each step. The solve stops and is accepted on the same true-residual threshold. Its
residual and convergence are step evidence.

The interface viscosity is a first-order interpolation.

## Step acceptance

`IncompressibleTwoPhaseVOFMethod.step` returns a `FixedStepResult`. The candidate is
committed atomically only when all of the following hold; otherwise
`successful=False` and the previous state is kept:

- every value is finite;
- the projection converged;
- the topology and PLIC reconstruction are valid;
- alpha is within its rounding bound;
- the directional advective Courant number is at most 1/2;
- `dt` is at most the Brackbill capillary limit `sqrt(rho_mean h^3 / (2 pi sigma))`;
- the interfacial force is valid: no unsupported faces and no `UNDERRESOLVED` cells;
- the viscous stage converged;
- any solid geometry is accepted, and no cell is cut by both solid and PLIC.

Callers choose `dt` to satisfy the limits; the method does not subcycle.

`TwoPhaseStepEvidence` reports:

| Field | Meaning |
|---|---|
| `capillary_pressure_jump` | mean usable `sigma * kappa` |
| `capillary_balance_residual` | relative dual-measure norm of the force not balanced by the new pressure gradient |
| `parasitic_velocity` | maximum face speed after projection |
| `advective_courant` | directional advective Courant number |
| `capillary_step_limit` | Brackbill capillary step limit |
| `curvature_{valid,fallback,underresolved}_count` | curvature status counts |
| `unsupported_face_count` | faces with an alpha jump but no usable curvature |
| `plic_residual` | PLIC reconstruction residual |
| `viscous_residual`, `viscous_converged` | viscous-stage solve evidence |
| `sweep_offset` | first sweep axis of the step |

It also reports the volume, flux, pressure, divergence, and topology residuals.

## Ledger

`TwoPhaseVOFLedger` accumulates over accepted steps:

- phase volume and momentum changes;
- kinetic, gravitational, and surface-energy changes;
- viscous dissipation;
- wall, capillary, gravity, and body work;
- reinitialization dissipation;
- pressure and divergence residuals;
- topology events.

No single total-energy balance is claimed. The ledger reports three separate
residuals:

- `work_energy_residual`: the discrete kinetic-energy-theorem defect
  `dKE - (W_capillary + W_gravity + W_wall + W_body) + D_viscous`, with force work
  paired to the midpoint velocity. It measures the numerical dissipation of momentum
  transport, projection, and splitting.
- `gravitational_energy_residual`: `dE_gravity + W_gravity`, with the exact discrete
  potential energy of the transported liquid.
- `surface_energy_residual`: `dE_surface + W_capillary`. `dE_surface` is estimated
  from PLIC facet measures, so this residual is evidence of that estimate's
  discretization error. It is not a conservation residual.

## Qualified static solid geometry

An optional qualified sharp-geometry realization replaces full Cartesian cell volumes
and face measures with accepted fluid volumes and open apertures. The replacement
applies throughout content, phase flux, momentum flux, inverse momentum, and pressure
projection.

Geometry, projection, PLIC, and transfer IDs must agree. Failed geometry or pressure
evidence rolls back the complete candidate state.

The contract is static and fixed topology. It currently also requires zero viscosity,
zero gravity, and zero surface tension:

- cut-solid viscous stresses are not resolved;
- the balanced interfacial potential is not carried through cut cells.

A cell cut simultaneously by the solid boundary and the gas-liquid PLIC interface is
rejected, because PLIC is defined on the full Cartesian cell and not on the clipped
fluid polytope.

## Moving bodies, contact, and surface piercing

`TwoPhaseMovingBodyPlan` provides a fixed-radius moving immersed target with an
identified center, velocity, and penalty. Body work is carried in the two-phase
ledger.

`TwoPhaseCapabilityEventPlan` evaluates the canonical VOF/PLIC state for body contact,
surface piercing, boundary wetting/drying, moving contact lines, overturning, and
breaking/topology-change routes. It returns:

- per-cell masks;
- a deterministic event bitset;
- the contact-angle residual;
- derivative availability.

The event product does not pretend that a topology change is smooth. Callers either
begin a new fixed-topology epoch or hand off to the mapped rigid/hydroelastic contact
product.

## Topology and remeshing

`TwoPhaseTopologyEvidence` reports liquid and gas volume, mixed cells, the exact PLIC
interface measure, the changed-cell mask, and the event proxy.

`ConservativeTwoPhaseRemeshPlan` accepts preflighted cell-overlap volumes and
face-transfer matrices between two prepared VOF products. It:

- transfers extensive liquid/scalar content and face momentum atomically;
- reconstructs the auxiliary level set from the transferred alpha;
- reports source/target coverage and conservation defects.

Connectivity selection and transfer construction are host-static. Topology changes
return `derivative_available=False`.

## Examples

`python examples/advanced_two_phase_vof.py` runs a resolved capillary drop with walls,
viscosity, and gravity.

`python examples/advanced_rising_bubble.py` runs a coarse, short Hysing test case 1.
It reports circularity, centroid, and rise velocity through
`phydrax.applications.free_boundary.hysing_bubble_benchmark`.

`python tools/two_phase_hysing_qualification.py` runs TC1 and TC2 at every
requested spatial resolution and time-refinement factor and records failed rows
instead of omitting them. Its base step is the smaller of the capillary limit
and a directional Courant step formed from the domain gravity-wave speed
`sqrt(g L_y)`; it does not infer a stable step from a measured benchmark
velocity. A case passes only when every requested row completes and the
finest-grid observables lie in the pinned reference windows.

## Phase-separated step benchmark

`benchmarks/two_phase_vof_step.py` measures host preparation, lowering, executable
compilation, and synchronized warmed execution separately; no phase is folded into
another. The retained JSON binds those measurements to the current package
source/build inputs, the benchmark driver and shared runtime harness, and the exact
serialized `TwoPhaseStepEvidence` field set. Validate the retained record before using
its measurements:

```console
PYTHONPATH=. .venv/bin/python benchmarks/two_phase_vof_step.py \
  --validate-stored benchmarks/two_phase_vof_step.json
```

The committed reduced smoke retains one real 8×8 case with one warmup and one measured
execution:

```console
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1' \
PYTHONPATH=. .venv/bin/python benchmarks/two_phase_vof_step.py \
  --case 8,0.25,0.1,0.001 --steps 1 --warmup 1 --repeats 1 \
  --maximum-iterations 200 --output benchmarks/two_phase_vof_step.json
```

Any source, driver, or evidence-schema mismatch is rejected rather than treating stale
measurements as current.

## Limits and nonclaims

- **Walls:** height-function columns near walls use zero-gradient ghost cells, which
  amounts to a 90° contact angle. The PLIC normal honours the material contact angle,
  but contact-angle-consistent curvature is not claimed.
- **Solids:** qualified sharp solid geometry requires zero viscosity, gravity, and
  surface tension. It is static, and cells cut by both solid and PLIC fail closed.
- **Surface energy:** the PLIC surface-energy estimate is not conserved in capillary
  flows. It drifts with the estimator's discretization error even while the dynamics
  are correct, so ledger closure is not claimed there.
- **Accuracy:** time splitting is first order, and the interface viscosity
  interpolation is first order.
- **Grid:** uniform spacing is required wherever height functions are prepared.
  Fixed grid only: no AMR/reflux, no distributed phase transport, no subgrid
  air-entrainment model.
- **Differentiability:** interface, wet/dry, moving-contact, piercing, breaking,
  contact, and remesh events are nondifferentiable topology boundaries.
- **Bodies:** penalty bodies do not replace the mapped monolithic rigid/hydroelastic
  contact product.
- **Qualification:** calibrated breaking and impact envelopes require their own
  qualification. The product remains a candidate until the Hysing benchmark passes.

## References

- S. Hysing, S. Turek, D. Kuzmin, N. Parolini, E. Burman, S. Ganesan, L. Tobiska,
  "Quantitative benchmark computations of two-dimensional bubble dynamics",
  *Int. J. Numer. Meth. Fluids* 60 (2009) 1259–1288, doi:10.1002/fld.1934.
- G. D. Weymouth, D. K.-P. Yue, "Conservative volume-of-fluid method for
  free-surface simulations on Cartesian grids", *J. Comput. Phys.* 229 (2010)
  2853–2865.
- R. Scardovelli, S. Zaleski, "Analytical relations connecting linear interfaces and
  volume fractions in rectangular grids", *J. Comput. Phys.* 164 (2000) 228–237.
- S. Popinet, "An accurate adaptive solver for surface-tension-driven interfacial
  flows", *J. Comput. Phys.* 228 (2009) 5838–5866.
- M. M. Francois, S. J. Cummins, E. D. Dendy, D. B. Kothe, J. M. Sicilian,
  M. W. Williams, "A balanced-force algorithm for continuous and sharp interfacial
  surface tension models within a volume tracking framework", *J. Comput. Phys.* 213
  (2006) 141–173.
- J. U. Brackbill, D. B. Kothe, C. Zemach, "A continuum method for modeling surface
  tension", *J. Comput. Phys.* 100 (1992) 335–354.
- S. J. Cummins, M. M. Francois, D. B. Kothe, "Estimating curvature from volume
  fractions", *Comput. Struct.* 83 (2005) 425–434.
- E. Aulisa, S. Manservisi, R. Scardovelli, S. Zaleski, "Interface reconstruction with
  least-squares fit and split advection in three-dimensional Cartesian geometry",
  *J. Comput. Phys.* 225 (2007) 2301–2319.
