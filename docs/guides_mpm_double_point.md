# Saturated double-point MPM with a mixed pressure projection

`DoublePointMPMPlan` binds one deformable MPM skeleton to a separate set of
FLIP water particles on a MAC grid. The phases retain distinct velocity degrees
of freedom and share one physical pore-pressure field. This is not a sequence
of independent solid and fluid pressure projections, nor a MAC projection of a
sampled mixture followed by an approximate solid correction.

The model is saturated with incompressible constituents: Biot coefficient
`alpha = 1` and constituent storage `S = 0`. Material stress is **effective
skeleton stress only**. With tension-positive Cauchy stress and compression-positive
pore pressure, total stress is

\[
\sigma = \sigma' - p I.
\]

Do not add a pore-pressure stress term to the MPM material or its internal-force
predictor: the mixed adjoint applies that pressure force exactly once.

## Preparing the consumer

The application-side owners are
`phx.applications.solid_mechanics.MPMToMACFaceTransferPlan` and
`MixedMPMPressureOperator`. The solver-side owners are
`phx.solver.DoublePointMPMPlan`, `DoublePointMPMState`,
`DoublePointMPMStepResult`, `DoublePointMPMStatus`,
`MixedMPMProjectionResult`, `project_double_point_velocity`, and
`advance_double_point_mpm_step`.

Prepare the solid MPM dynamics, the MAC operators, and a
`phx.discretization.flip.FLIPParticleTransferPlan(mac).prepare(fluid_particles)`.
Then bind the geometry-only nodal-to-face map:

```python
solid_to_mac = phx.applications.solid_mechanics.MPMToMACFaceTransferPlan(
    solid_dynamics, mac, maximum_routes=maximum_routes
).prepare()
plan = phx.solver.DoublePointMPMPlan(
    solid_dynamics,
    fluid_transfer,
    solid_to_mac,
    phx.equations.InterphaseDragLawPlan("stokes"),
    solid_density=rho_s,
    fluid_density=rho_f,
    fluid_viscosity=mu_f,
    grain_diameter=diameter,
    pressure_mask=reservoir_cells,
    pressure_values=reservoir_pressure_pa,
    solid_free=solid_component_mask,
    fluid_free=fluid_face_masks,
    tolerance=1.0e-8,
    maximum_iterations=256,
)
state = plan.initialize(solid_state, fluid_state)
result = phx.solver.advance_double_point_mpm_step(
    plan, state, dt, arguments=material_arguments
)
state = result.accepted_state
```

Densities are intrinsic phase densities, not already volume-fraction-weighted
mixture densities. Positive finite densities, viscosity, and grain diameter
are required. The viscosity here parameterizes interphase drag; it does not by
itself enable the separate viscous-FLIP momentum operator.

The map uses multilinear interpolation of the **actual background nodal grid**,
not a particle-dependent transfer. Its fixed-capacity route evidence is exposed
as `solid_to_mac.evidence`: support, partition-of-unity defect, required routes,
capacity admission, and success. Unsupported face locations or insufficient
route capacity reject use of the map; a truncated interpolation is not accepted.

### Supported geometry and schedule

This coupled path supports **Cartesian, dense, single-field MPM with the USL
stress-update schedule**. The MPM particle domain must retain the assignment's
complete background-grid halo. The MAC grid must bind the same spatial axis
identity/order and compatible periodic-axis semantics. The solid and fluid maps
must bind the exact supplied prepared dynamics and MAC operators.

Axisymmetric geometry, block-sparse nodal storage, multifield MPM, and non-USL
schedules are refused rather than routed through an approximate alternative.
Do not infer double-point support from the broader standalone MPM capabilities.
The same-attempt `predict_grid` / `finalize_grid_velocity` split owns the solid
transition: deformation, stress, material history, and particle advection are
finalized from the corrected velocity in that attempt, not recomputed by taking
an independent completed MPM step.

## Exact mixed operator, physical masses, and pressure pairing

Let `T` sample solid nodal velocity onto MAC faces, `P_s` and `P_f` select
admitted phase velocity components, and `F_s`, `F_f` multiply by the face
fractions `1 - n` and `n`. Using the volumetric MAC divergence,

\[
D_s = \operatorname{div} F_s T P_s,\qquad
D_f = \operatorname{div} F_f P_f,\qquad
D = [D_s,D_f].
\]

`MixedMPMPressureOperator` exposes `solid_divergence`, `fluid_divergence`,
`divergence`, `solid_transpose`, `fluid_transpose`, `inverse_mass_s`,
`inverse_mass_f`, and `pressure_action`. The transpose is the **exact ordinary
algebraic transpose of each divergence action**, including the same masks and
fraction placement. For every admitted nodal vector `v_s`, face vector `v_f`,
and cell multiplier `lambda`,

\[
\lambda^\mathsf T(D_s v_s+D_f v_f)
= v_s^\mathsf T D_s^\mathsf T\lambda
 + v_f^\mathsf T D_f^\mathsf T\lambda.
\]

The map also satisfies ordinary force duality
`<T v_s, f_faces> = <v_s, T^T f_faces>`. Do not substitute the MAC Hilbert-space
pressure gradient for these algebraic transposes.

The block mass is

\[
M = \operatorname{diag}(M_s,M_f).
\]

`M_s` contains the physical deposited nodal solid masses; `M_f` contains the
**actual deposited FLIP face masses**, masked where the fluid fraction is zero.
The fluid quadrature fraction is `theta_a = (M_f)_a / (rho_f V_face_dual)`;
it need not coincide exactly with the solid-derived porosity `n_a` used in
`D` and drag. Replacing deposited masses with synthesized `rho_f n_a V_face_dual`
would discard the FLIP particle/grid momentum ledger. Preserve and inspect the
fluid transfer's quadrature and momentum evidence. Do not multiply an already
physical mass by density or phase fraction again. Positive masses are used
as supplied. Zero-mass and inactive components are excluded from the
inverse-mass action, not repaired
with mass floors. `solid_active` may be a nodal mask or a full nodal-component
mask; `fluid_active` supplies one Boolean mask per face layout. The step's
`solid_free` / `fluid_free` select free components; false components impose
stationary walls. Masks participate in both the divergence and its transpose,
not only in a final velocity overwrite.

Physical pressure requires the **cell-volume pairing**. Write
`V_c = diag(cell_volumes)`; `p` is in Pa and `V_c p` is in Pa m³ (or the
consistent unit-depth measure for a two-dimensional model). Because `D` is a
volumetric divergence, its ordinary transpose has the negative-gradient sign.
Thus the physical pressure acceleration is

\[
\dot v\big|_p = M^{-1}D^\mathsf T V_c p.
\]

The matrix-free symmetric Schur action is

\[
A\lambda = D M^{-1} D^\mathsf T\lambda.
\]

For a pressure increment, the Euclidean solve variable is
`lambda = -V_c delta_p`, **not** pressure in Pa itself. Away from reservoir
rows, one solve and correction give

\[
A\lambda = \frac{D v^*}{\Delta t},\qquad
v^{n+1}=v^*-\Delta t M^{-1}D^\mathsf T\lambda,\qquad
p^{n+1}=p^n-V_c^{-1}\lambda.
\]

The minus sign in the algebraic velocity correction is consistent with
compression-positive physical pressure through that multiplier definition.
Cell volumes must not be dropped even if a uniform-grid calculation makes
them look like a single harmless scale factor. The volume pairing also preserves
the symmetric matrix-free Euclidean solve on nonuniform MAC cells.

### Physical reservoir datum, not an arbitrary gauge

`pressure_mask` selects physical pressure-reservoir cells. Their continuity
rows are replaced by the supplied pressure condition; mixture continuity is
required only on the remaining cells. A drained boundary at atmospheric
**excess** pressure can use `pressure_values=0.0`; another physical pressure
convention must supply its corresponding reservoir values and initial field.
`initialize` requires the initial fluid pressure to satisfy those values.

A datum-free mask is refused. A mean-zero gauge, automatic reference cell,
pressure clipping, or pressure floor is not a substitute for a physical datum.
Every connected component of the active pressure Schur graph must contain a
datum cell. Bounded device-only label propagation checks this each attempt,
including phase-mass and wall-mask changes; an unanchored component rejects
even when its zero right-hand side admits a converged singular solve.
The native pressure solve uses MINRES with certified symmetry, not an
unconditional positive-definiteness claim. Reservoir cells may exchange mixture
volume, so their raw divergence is not an unconstrained continuity error.

## Atomic attempt and lagged implicit drag

Predictor stability, domain, route, and grid-constraint refusals report
`DoublePointMPMStatus.PREDICTOR_FAILED`; constitutive refusals report
`MATERIAL_FAILED`. Split MPM stress and nodal-force increments contribute
midpoint grid work to the MPM external-work ledger without committing stress.

An attempt has the following order:

1. Predict the solid grid with effective skeleton stress only, and deposit
   current solid content `m_s / rho_s` onto MAC cells with the solid MPM
   assignment and its current domain metadata, not the water assignment.
   Compute `n = 1 - phi_s`. Fractions must be finite and lie in `[0, 1]`;
   they are **never clipped**. Incomplete cell-assignment support rejects
   the transfer rather than dropping or renormalizing solid volume.
2. Transfer the separate water particles to MAC velocities and construct
   physical phase masses and masks for the mixed operator.
3. Apply accepted physical pressure once, before drag, to both phases using
   `v += dt M^-1 D^T(V_c p_old)`. No second old-pressure stress force is added.
4. Evaluate the selected native interphase drag law at the lagged slip and
   porosity, then solve the coupled backward-Euler momentum block using native
   `phydrax.linalg` Krylov execution. This is a lagged-coefficient implicit drag
   solve, not a converged nonlinear drag iteration.
5. Perform **one** mixed Schur solve for the pressure increment after drag,
   apply its exact mixed inverse-mass adjoint correction, and store physical
   `p_new = p_old + delta_p`.
6. Finalize the solid's same-attempt USL transition with the corrected nodal
   velocity. Update water particle velocity with the FLIP grid-velocity
   increment; use the corrected PIC velocity for water position advection.
7. Commit both phases together only if every required admission and solve
   succeeds. Otherwise retain both committed phases unchanged.

For a free two-phase system, with slip operator `C v = T v_s - v_f` and
`B = diag(beta V_face_dual)`, the drag block is

\[
(M+\Delta t C^\mathsf T B C)v^*=M\widetilde v.
\]

The same face force is pulled back with `T^T` to the skeleton and applied with
the opposite sign to water. With constant reproduction, this construction
preserves action–reaction and total phase momentum up to the reported solve
residual. Stationary components restrict the block consistently; wall reactions
are external momentum exchange and must not be mistaken for internal imbalance.
Absent phases are masked, not assigned artificial porosity or mass. The drag
law's branch, Reynolds-number, admissibility, derivative, and success evidence
remains available in `result.drag`.

`DoublePointMPMStepResult` exposes `candidate_state` for diagnostics and
`accepted_state` for continuation, plus `successful` and `status`. Status covers
invalid fractions/state, failed predictor, transfer, drag, projection, and material update.
Use `accepted_state` even on rejection; do not partially commit the solid,
water, or pressure from `candidate_state`. Inspect `projection` native linear
status, iteration count, residual, unconstrained divergence norm, and
`pressure_datum_supported` (false for an unanchored active pressure component),
`drag_residual_norm`, the wall-reaction-accounted `drag_momentum_defect`,
`solid_momentum_defect`, `fluid_momentum_defect`, and `solid_step` material
evidence. `effective_stress` and `total_stress` are separate diagnostics; only
effective skeleton stress belongs in the material history. `drag_phase_present`
identifies faces where both phases participate. The reported mixture divergence
and expanded `D_s v_s + D_f v_f` have a separate `divergence_identity_defect`;
this identity is distinct from convergence of the continuity solve.

## Darcy and Terzaghi limits

These references follow from this model's equations under their stated limits;
they are not general finite-storage Biot benchmarks.

Let `beta` be force per mixture volume per intrinsic relative velocity, in
kg m⁻³ s⁻¹. In the low-inertia fluid momentum limit,

\[
0=-n\nabla p+n\rho_f\mathbf g-\beta(v_f-v_s),\qquad
q=n(v_f-v_s)=\frac{n^2}{\beta}(-\nabla p+\rho_f\mathbf g).
\]

Here `q` is Darcy discharge relative to the skeleton, not intrinsic fluid
velocity. With hydraulic head `h = p/(rho_f g) + z` and downward gravity,

\[
q=-k\nabla h,\qquad
k=\frac{n^2\rho_f g}{\beta}.
\]

This `k` is **hydraulic conductivity** (m/s). If intrinsic permeability is
needed, `kappa = mu_f n^2 / beta` has units m². Neither relation equates `beta`
to an already porosity-weighted discharge resistance without the `n²` factor.
Nonlinear slip-dependent drag does not in general produce constant conductivity.

For one-dimensional small-strain consolidation, impose zero lateral strain,
constant total vertical stress after the load increment, low inertia, and
incompressible constituents (`alpha = 1`, `S = 0`). Linearize about constant
porosity and constant drag coefficient, and subtract the hydrostatic pressure.
For constrained skeleton modulus `H` and tension-positive strain `epsilon`,

\[
\sigma_{zz}=H\epsilon-p,\qquad
\dot\epsilon=\dot p/H,\qquad
\dot\epsilon+\partial_z q=0,\qquad
q=-\frac{n^2}{\beta}\partial_z p.
\]

Therefore

\[
\boxed{\partial_t p=C_v\partial_{zz}p},\qquad
\boxed{C_v=H n^2/\beta}.
\]

For isotropic three-dimensional linear elasticity under lateral constraint,
`H = K + 4 G / 3 = E(1 - nu)/((1 + nu)(1 - 2 nu))`. This is the constrained
modulus, not Young's modulus. Equivalently `C_v = H k/(rho_f g)`.
A drained end prescribes excess `p=0`; an impermeable end prescribes `q=0`.
A single-drained layer of thickness `L`, initially at uniform excess `p0`, has

\[
\frac{p(z,t)}{p_0}=
\sum_{m=0}^{\infty}\frac{4}{(2m+1)\pi}
\sin\!\left(\frac{(2m+1)\pi z}{2L}\right)
\exp\!\left[-C_v\left(\frac{(2m+1)\pi}{2L}\right)^2t\right],
\]

with drainage at `z=0` and no discharge at `z=L`. Comparing a discrete reservoir
row to this continuum solution requires matching the physical drainage
location and effective layer length. Constituent compressibility, nonzero
storage, large strain, inertia, variable porosity, or nonlinear drag changes
this reference problem; no Mandel–Cryer qualification is implied.

## Example and reference scope

`examples/double_point_mpm.py`
constructs real solid MPM dynamics, fluid FLIP transfer, the nodal-to-face map,
a physical top-row pressure reservoir, componentwise wall masks, and a mixed
plan. It advances one atomic step with a supplied nonzero initial pressure
profile and prints status, phase acceptance counters, porosity, transfer
capacity, drag and projection evidence, and accepted pressure. A rejection is
printed and then raised; there is no independent projection or fabricated
fallback state. Run from the repository with
`python -m examples.double_point_mpm`. This guide does not report an observed
example run, a resolution study, or a qualification result.

Primary structural reference: Mian Xie, Pedro Navas, and Susana López-Querol
(2026), **A semi-implicit double-point Material Point Method for both
free-surface flow and seepage in deformable porous media**,
[arXiv:2608.00578](https://arxiv.org/abs/2608.00578),
[DOI:10.48550/arXiv.2608.00578](https://doi.org/10.48550/arXiv.2608.00578),
submitted August 1, 2026, CC BY 4.0.
The reference motivates distinct soil/water point sets and semi-implicit
interphase coupling. Its slightly compressible viscous-water treatment,
stabilization, smooth porosity/permeability transitions, and reported laboratory
validation are not claims about this incompressible-constituent Phydrax path.
The exact mixed adjoints, physical-volume pressure pairing, and atomic status
contract above define this implementation's consumer boundary.
