# Free-surface FLIP

Phydrax FLIP is a fixed-population particle–MAC method over the existing structured tensor-grid,
particle-splat, MAC-boundary, and linear-solver substrates. It is distinct from MPM/APIC, SPH
free-surface detection, unstructured VOF/PLIC, and resolved marker-force coupling.

## Transfer

`FLIPParticleTransferPlan` prepares one existing particle splat on cell centers and one on every
MAC face orientation. It does not create a new router.

For reference density `rho`, cell parcel volume is deposited as `mass / rho`. Every face transfer
deposits a fused extensive payload:

```text
face_mass     = sum_p N_fp mass_p
face_momentum = sum_p N_fp mass_p velocity_p
face_velocity = face_momentum / face_mass
```

The extensive numerator balances are reported before normalization. Unsupported face velocity is
zero with a false support mask. P2G and G2P use the same prepared B-spline relation.

`SparseFLIPParticleTransferPlan` is the virtual-grid realization of the same
transfer. It binds one `PreparedTensorIndexSpace`, one cell
`SparseBlockTopologyPlan`, and one topology for each staggered face layout.
Route construction remains the existing B-spline authority; only logical
targets are lowered to compact cell/face storage. P2G returns compact cell
volume and face mass/momentum arrays. G2P consumes those compact face arrays
through the same mapped stencils. Topology refresh is fixed-capacity and
transactional.

`SparseMACFreeSurfaceProjectionPlan` lowers compact cell-to-face and
face-to-cell relations for atmospheric pressure projection. Liquid pressure is
solved only on compact storage cells through `phydrax.linalg`; air pressure is
zero and an all-liquid periodic component receives the zero-mean gauge.
Every liquid cell must resolve both adjacent face slots on every axis.

`compile_sparse_flip_problem` composes compact P2G, pressure projection, G2P,
midpoint advection, topology refresh, pressure remapping by logical cell ID,
CFL evidence, and atomic state commit. The current compact problem supports
periodic/atmospheric geometry without a sharp solid bundle. It never falls
back to a dense MAC grid.

## Atmospheric pressure projection

`MACFreeSurfaceProjectionPlan` adds the one pressure closure absent from the full-grid MAC stack: a
runtime liquid mask with atmospheric pressure exactly zero in air. It reuses
`PreparedMACOperators`, `PreparedMACBoundaryPlan`, and `phydrax.linalg`.

Air rows are identity rows; liquid rows use the compatible weighted pressure action. When every
cell is liquid, the same action activates a zero-mean pressure gauge. The result reports liquid/air
counts, divergence, pressure residual, air-pressure defect, kinetic-energy change, nested linear
solve evidence, and fail-closed status.

This is not a particle SDF or ghost-fluid fraction. The initial liquid mask is a first-order
threshold on raw deposited parcel volume divided by cell volume.

## Step ordering

`compile_flip_problem` combines a `FLIPProblemIR`, prepared transfer,
`MACFreeSurfaceProjectionPlan`, and `FLIPMethodPlan` into `CompiledFLIPProblem`.

One step:

1. deposit particle volume and face mass/momentum;
2. classify liquid cells;
3. extrapolate supported pre-grid velocity for a fixed number of layers;
4. save the exact pre-operation grid velocity;
5. apply body acceleration and physical MAC boundaries;
6. for a viscous law, apply the accepted pressure force, then solve backward-Euler MAC
   stress with bounded Picard iteration;
7. project the pressure increment and add it to accepted physical pressure for viscous
   steps; inviscid steps retain the original absolute-pressure projection;
8. extrapolate post-projection velocity;
9. gather the post velocity and exact grid increment at unchanged particle positions;
10. update particle velocity;
11. midpoint-advect positions;
12. commit the complete candidate only when every transfer, viscous closure/solve, projection, CFL,
    support, and finite check passes.

`FLIPMethodPlan.pic_fraction` uses the convention

```text
v_new = (1 - beta) [v_old + G(u_new - u_old)] + beta G(u_new)
```

where beta zero is pure FLIP and beta one is pure PIC. The value is required explicitly; no 95/5
default is inferred from graphics demonstrations.

Grid extrapolation is reconstruction for G2P/advection only. It never enters mass or momentum
balance evidence.

## Implicit viscous stress

`FLIPProblemIR(..., viscous_law=None)` keeps the inviscid numerical path unchanged.
The dense route accepts `GeneralizedNewtonianLaw` or `MuILaw`, not a separate
Newtonian class: use a power law with `power_index=1` and the desired
`consistency_pa_s_n` for constant dynamic viscosity.

The viscous path uses a **first-order backward-Euler incremental
pressure-correction scheme**. After body acceleration and physical boundaries,
the predictor subtracts `K G p^n`, where `K` is the projection owner's
face interpolation of `dt / rho` and `G` its pressure gradient:

```text
u_predictor = B(u_forced - K G p^n)
(rho_f + dt A_eta) u_star = rho_f u_predictor - dt b_eta
u_new = B(u_star - K G delta_p)
p_new = p^n + delta_p
```

`B` retains the same physical MAC velocity boundary handling as projection.
Embedded sharp solids are not supported by the viscous stage. Accepted pressure
is restricted to current liquid/fluid cells and
uses the projection's zero-mean gauge for an all-liquid domain. The correction
has zero atmospheric Dirichlet data in air and homogeneous pressure data on
physical pressure boundaries. The native pressure solve reports `delta_p` in
`pressure_increment` and total physical pressure in `pressure`.

This is the incremental structure introduced by Goda (1979) and van Kan (1986),
not a claim to implement van Kan's second-order temporal discretization.
The lagged-gradient predictor and additive gradient-pressure update are also
explicit in Bell, Solomon & Szymczak (1989), §2, Eqs. (2.3)–(2.4).
Omitting `G p^n` would let implicit viscous stress absorb part of gravity and
make steady pressure depend on viscosity and time step. The two Newtonian
hydrostatic-rest regressions (`eta=1e-3` and `1e3` Pa s) instead preserve the
independent discrete datum `p_j = rho g dy (N-j)` to relative error ≤`1e-10`,
with zero liquid/particle velocity over successive accepted steps. The datum
is the first **air cell**, not a fitted geometric free-surface height.

Cell liquid classification weights stress; adjacent-cell liquid fractions
weight face mass `rho_f`. Air has no cell stress. Massless stress-coupled faces
remain unknowns, giving the existing variational action's natural traction-free
surface rather than a prescribed zero surface velocity.

References:

- Goda (1979), *A multistep technique with implicit difference schemes for
  calculating two- or three-dimensional cavity flows*, JCP 30:76–95,
  [doi:10.1016/0021-9991(79)90088-3](https://doi.org/10.1016/0021-9991(79)90088-3).
- van Kan (1986), *A Second-Order Accurate Pressure-Correction Scheme for Viscous
  Incompressible Flow*, SIAM J. Sci. Stat. Comput. 7:870–891,
  [doi:10.1137/0907059](https://doi.org/10.1137/0907059).
- Bell, Solomon & Szymczak (1989), *A Second-Order Projection Method for the
  Incompressible Navier Stokes Equations on Quadrilateral Grids*, UCRL-100879,
  [primary author manuscript](https://www.osti.gov/servlets/purl/6198366), §2.

Viscosity is recomputed from `sqrt(2 S_d:S_d)` during **bounded Picard iteration**.
`FLIPMethodPlan` exposes `viscous_tolerance`, `maximum_picard_iterations`,
`maximum_viscous_linear_iterations`, `picard_damping`, and `resource_policy`.
The implementation normalizes fixed-point coordinates and independently
certifies the true nonlinear momentum residual relative to the momentum scale.
This backward-error scale includes the norms of the **uncombined** body/velocity
predictor and old-pressure impulse, with the same face mass/measure weights.
Balanced gravity and pressure must not normalize floating-point roundoff by
their nearly-zero difference. The true residual numerator and equations are
unchanged; no pressure/rate floor or equilibrium special case is introduced.
The dimensionless inner Picard update tolerance is three orders tighter than
the final viscous tolerance; its Krylov tolerance is one further order tighter.
The true momentum residual remains authoritative even if the correction criterion passes.
Both convergence certificates, closure status, and native linear evidence must
pass. Undefined zero-rate viscosity, Krylov failure, or Picard nonconvergence
rejects the whole step; there is no inviscid fallback or internal retry.
`FLIPStepResult.viscous` contains the rate, coefficient, lagged pressure,
Hadamard diagnostic/applicability, residuals, iteration count, convergence,
closure admission, nonlinear work, and linear energy/dissipation evidence.
It is `None` for inviscid steps.

For pressure-dependent `MuILaw`, supply dynamic `mu_i_parameters` to the problem
and `initial_pressure` to `initialize_state`. This cell field must be finite,
must be exactly atmospheric-gauge zero in initial air cells, and must accompany
a nonempty liquid region with a free surface. It is not initialized implicitly
to zero. The projection density must match the problem's reference density, so
pressure is in the law's physical units rather than a rescaled projection variable.
Every subsequent viscous stage uses the **previous accepted** atmospheric projection
pressure, fixed through Picard iteration. Configurations
without air/free-surface support are refused initially or rejected at runtime.
Nonpositive physical pressure retains the law's explicit separated branch.
Hadamard diagnostics are not a guarantee of multidimensional well-posedness.

Compact compilation explicitly refuses any non-`None` viscous law; it never
allocates a dense fallback. Dense viscous preparation admits a conservative
matrix-free workspace bound `(64 * faces + 32 * cells) * dtype.itemsize` and
additional state bound `(faces + cells) * dtype.itemsize` against the resource
policy.

The small scientific checks exercise the same grid stress stage independently
of particle-advection error: Newtonian Poiseuille uses the half-cell wall
stencil's quadratic error band `1 / (8 eta N_y^2)`; regularized Bingham uses
two cells of plug-width uncertainty against the analytic **edge-count** oracle
(whose continuum-width quantization is separate), plus a 0.025 velocity band
at 48 cells with `regularization_s=1000`. The 24-cell μ(I) inclined-layer check uses the
GDR MiDi (2004) Bagnold profile and a candidate 12% profile band for binary
free-surface/edge interpolation. These are small grid-stage regression checks,
not particle runout or resolution-independent continuum qualification.

## Qualified static solids

An optional sharp-solid bundle binds one accepted fluid-volume/open-aperture
realization, solid-aware P2G/G2P relations, sharp pressure projection, wall velocity,
and particle collision under the same geometry identity. Particle stencil weights are
restricted to accepted fluid support and renormalized; the matched relation is used in
both transfer directions and reports partition-of-unity and support-loss evidence.

This bundle is supported only for inviscid FLIP (`viscous_law=None`).
Compilation rejects any viscous law combined with solid geometry: the
embedded-wall viscous stage does not implement the exact cut-cell essential
stress affine offsets. Closed-face wall velocities and open-aperture weights
alone do not provide that stress boundary condition. Physical MAC domain
boundaries remain supported for viscous FLIP.

The liquid mask is intersected with active fluid cells. Pressure, divergence,
extrapolation, and grid updates exclude closed support. Geometry, transfer, projection,
collision, and the particle/grid candidate commit or roll back atomically.

Initial sharp-solid support is static and fixed topology. Moving geometry requires the
accepted sharp-epoch refresh/GCL lifecycle; reseeding remains separate.

## Existing MAC and MPM boundaries

FLIP does not call `CompiledMACIncompressibleDynamics.convection`; particle motion already carries
advection. Calling both would double-advect momentum.

The existing `MACHelmholtzSolvePlan`, IMEX/SBDF2 methods, `MACVariableDensityPlan`,
distributed MAC projection, ALE/remesh, and `MACMarkerTransferPlan` remain independent
qualified capabilities. FLIP reuses only the explicitly prepared stress/projection owners.

Current MPM already owns collocated nodal APIC mechanics, deformation, stress and constitutive
history. FLIP uses staggered face transfer and a grid-delta update; MPM runtime state is never used.

## Differentiation and limits

Gradients are branchwise for fixed particle routes, liquid mask, extrapolation wavefront, solver
route, and accepted step. Classification or route transitions invalidate the local derivative
program.

The baseline scope is constant-density, fixed particles, periodic or stationary
closed boundaries, first-order liquid classification, and single-device execution.
Dense implicit generalized-Newtonian and pressure-lagged μ(I) stress are supported;
compact viscous FLIP is explicitly refused. Qualified fixed-topology static solids
are optional for inviscid FLIP only. Moving sharp solids, reseed integration, variable-density FLIP,
staggered APIC–FLIP, and distributed particle ownership remain unsupported.
