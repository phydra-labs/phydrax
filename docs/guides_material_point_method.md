# Material point method

Phydrax Material Point Method is a fixed-capacity updated-Lagrangian continuum method.
Material points carry mass, kinematics, and constitutive history; a prepared nodal tensor
grid is temporary scratch space for momentum and force balance. The qualified baseline is
explicit Update Stress Last with quadratic B-splines and APIC transfer.

## Prepare particles, grid, and transfer

`ParticleSetPlan` owns stable IDs, masses, and activity. Current positions are temporal
state. `ParticleGridSplatPlan` owns the nodal routes, weights, physical gradients,
node-minus-particle offsets, moments, precision, and accumulation order.

```python
particles = phx.discretization.ParticleSetPlan(
    particle_ids,
    masses,
    ambient_dimension=2,
).prepare()

grid = phx.discretization.TensorGridPlan(
    (
        phx.discretization.UniformAxisSpec(nx, periodic=True, endpoint=False),
        phx.discretization.UniformAxisSpec(ny, periodic=True, endpoint=False),
    ),
    axis_names=("x", "y"),
).prepare(bounds)

splat = phx.discretization.ParticleGridSplatPlan(
    grid,
    assignment=phx.discretization.TensorBSplineSplatAssignment(2),
    boundary="reject",
).prepare(particles)
```

The target must be nodal and uniform. The assignment must provide conservative
first-order reproduction, physical gradients, and APIC-compatible particle
moments; quadratic point B-splines are the baseline choice. Closed-domain MPM
rejects truncated support; it never renormalizes or drops an unreported fraction.

## Physical domain and halo

Periodic axes wrap route construction but keep particle positions unwrapped in runtime
state. This preserves continuous trajectories through periodic seams.

Nonperiodic axes require an explicit `MPMParticleDomainPlan`. The computational grid
must extend beyond the admissible particle box by at least the complete quadratic
B-spline support radius. `support_margin` records that promise, and compilation checks
it against the actual grid bounds and spacing. Runtime positions outside the admissible
box reject the complete step without changing accepted particle state or time.

Physical walls are independent of support halos. The initial implementation exposes
`PrescribedGridVelocityPlan`, a static per-node, per-component velocity mask. Boundary
impulse and work remain explicit diagnostics. It does not claim collision/contact.

## Constitutive response

`AbstractMPMConstitutivePlan` returns first-Piola stress, reference-volume energy,
candidate history, admissibility, and a conservative current-state wave-speed bound.
It is a neutral `MODEL`-authority slot: built-in plans are fixed analytic
implementations, and a learned plan keeps its model trainable.
The baseline `NeoHookeanMPMConstitutivePlan` is stateless and supports:

- one-dimensional mechanics;
- two-dimensional plane strain;
- three-dimensional mechanics.

`NeoHookeanParameters` stores shear modulus and Lamé lambda. Use
`NeoHookeanParameters.from_shear_bulk(mu, K)` when physical shear and bulk moduli are
available. The logarithmic compressible reference energy and first-Piola response are:

```text
psi0(F) = mu/2 (F:F - 3) - mu log(J) + lambda/2 log(J)^2

P(F) = mu (F - F^-T) + lambda log(J) F^-T
```

For one-dimensional mechanics and plane strain, the active deformation is embedded
with inactive stretches fixed to one. Plane-stress models use either the scalar
`PlaneStressMPMConstitutivePlan` or the anisotropic,
director-coupled `GeneralPlaneStressMPMConstitutivePlan`. Nonpositive `J` is a
rejected material trial, not a clamped state.

The field-valued energy and stress operators use the same $\mu$, $\lambda$, $K$,
plane-strain, reference-volume, and nonpositive-$J$ conventions.


Material parameters remain rollout arguments. Place state initialization inside a
parameterized objective when the initial constitutive response depends on trainable
parameters.

## APIC and USL ordering

For route offset `r_ip = x_i - x_p`, APIC uses:

```text
D_p = sum_i N_ip r_ip outer r_ip

p_i = sum_p m_p N_ip [v_p + C_p r_ip]
```

Internal force uses current nodal gradients with a reference-volume first-Piola form:

```text
f_i_internal = -sum_p V0_p [P_p F_p^T] grad(N_ip)
```

After the grid force and prescribed-boundary update:

```text
v_p_new = sum_i N_ip v_i_new
L_p_new = sum_i v_i_new outer grad(N_ip)
B_p_new = sum_i N_ip v_i_new outer r_ip
C_p_new = B_p_new inverse(D_p)
F_p_new = [I + dt L_p_new] F_p
```

`C` is APIC transfer state. `L` is the physical velocity gradient; they are not
interchanged. The material update occurs last, so force uses accepted `(F, P)` and the
candidate material state is committed only when the whole step succeeds.

The APIC moment solve uses `phydrax.linalg.solve_small_linear` and reports conditioning.
The qualified baseline rejects a singular or over-conditioned particle moment.

## Axisymmetric full-ring mechanics

Choose geometry explicitly; a two-dimensional grid does **not** by itself imply
axisymmetry:

```python
method = phx.discretization.ExplicitMPMMethodPlan(
    geometry=phx.discretization.MPMGeometryPlan(
        "axisymmetric", spatial_dimension=2
    ),
    volumetric_stabilization=phx.discretization.MPMVolumetricStabilizationPlan(
        "f_bar"
    ),
)
material = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(3)
```

Particles, positions, velocities, grid coordinates, and APIC affine state use
the spatial order `(r, z)` and two components. The material must have
`"three_dimensional"` kinematics. Its deformation and first-Piola stress are
three-dimensional in the order **`(r, z, theta)`**, with

```text
F = [[Frr, Frz, 0],
     [Fzr, Fzz, 0],
     [0,   0,   Ftt]]
Ltt = sum_routes H_ip * parity_ip * vr_i
F_new = (I_3 + dt * L) F
```

In particular, uniform radial expansion `vr = a*r` gives
`Frr_new = Ftt_new = 1 + dt*a` from the identity; this is not Cartesian
plane strain with a frozen hoop stretch. `PreparedMPMDynamics.dimension` stays
spatial (two); `constitutive_dimension` is three, and `geometry` exposes the
resolved geometry owner. Omitting geometry keeps Cartesian behavior; an explicit
Cartesian plan uses the corresponding spatial dimension.

All supplied masses, reference volumes, momenta, and forces represent a **complete
ring**, not a unit-radian slice or meridional area. For a small rectangular
meridional particle at positive radius, supply
`V0 = 2*pi*r0*dr*dz` and `mass = density0*V0`. The kernel adds **no hidden
`2*pi` or radius multiplier**. Reference volumes remain unchanged during
deformation. The cylindrical weak form, using `tau = P F^T`, gives

```text
delta W_internal =
    sum_p V0_p [tau_ab * d_b(delta v_a) + tau_tt * delta Ltt]

f_ir_internal =
    -sum_routes V0_p parity_ip [tau_rb * G_ipb + tau_tt * H_ip]
f_iz_internal = -sum_routes V0_p tau_zb * G_ipb
```

Here `a,b` run over `(r,z)`. The hoop force is the exact virtual-work adjoint
of the route velocity gradient. For point routes, `G = grad N` and
`H = N/r_p`; fixed particle domains use the ring averages below. The equations
shown are unstabilized; B-bar adds the adjoint of its trace projection.
Using current volume and Cauchy stress is equivalent because
`V_current/J = V0`. These equations follow directly from cylindrical virtual
work. The primary method-origin reference is
[Sulsky and Schreyer (1996), *Axisymmetric form of the material point method
with applications to upsetting and Taylor impact problems*, CMAME 139:409–429](https://doi.org/10.1016/S0045-7825(96)01091-2).
This citation establishes the method's origin, not an equation-by-equation
verification of the original article.

### Axis and supported routes

Active particles require **`r > 0`**, including at initialization; nonpositive
radius is a clear refusal, not a division-floor repair. Axis nodes enforce
`vr = 0`. Grid nodes at negative radius and radial periodicity are refused
during preparation. Linear, quadratic, and cubic point B-splines admit support
crossing the axis: a ghost radial integer node `k < 0` maps to physical node
`abs(k)`. Scalar basis weights retain the full positive stencil. Axial velocity
is even and radial velocity is odd, so reflected routes multiply radial
velocity, momentum, and force by `-1`. Distinct ghost routes remain distinct
even when their physical node indices coincide; APIC offsets are measured from
the **ghost** node, not its reflected physical location. This preserves affine
reproduction and exact force/gradient virtual-work adjointness.

Fixed **uGIMP** uses the physical meridional domain
`[max(0,r_p-l_r),r_p+l_r] × [z_p-l_z,z_p+l_z]` and the ring measure `r dr dz`:

```text
M = integral_D r dr dz
Nbar_i = integral_D N_i r dr dz / M
G_i = integral_D grad(N_i) r dr dz / M
H_i = integral_D (N_i/r) r dr dz / M = integral_D N_i dr dz / M
```

The hoop operator is **not** `Nbar_i/r_p`. Axis clipping is an admitted
physical-domain operation, not stencil truncation or weight renormalization.
Ring averaging generally gives a nonzero weighted first moment about the
particle point. APIC therefore fits the centered weighted covariance and an
intercept: with offsets `d_i`, weighted means `dbar` and `vbar`,
`C = [sum_i w_i (v_i-vbar)(d_i-dbar)^T] [sum_i w_i (d_i-dbar)(d_i-dbar)^T]^-1`
and `v_p = vbar-C*dbar`. The runtime uses a solve, not an explicit inverse.
This recovers both particle-point velocity and affine state for uniform
expansion; assuming a zero first moment would introduce a radial drift.
Evolving cpGIMP/AABB, CPDI, and CPDI2 remain explicitly unsupported for this
geometry because their cylindrical moving-domain operators are not supplied.

A radial physical-domain lower bound of zero needs no negative radial halo;
axial and outer-radial full-support promises still apply. Axisymmetric **B-bar**
replaces the full three-dimensional trace `Lrr+Lzz+Ltt` by its background-cell
average with ring measure `r dr dz`, retaining deviatoric kinematics. Its
internal force is the exact matching adjoint, including the projected hoop
stress. F-bar uses the full three-dimensional trial determinant, including the
hoop increment, with the physical-volume projection convention.
Dense/block-sparse storage, independent nodal fields, and fixed-route implicit
residuals support these reflected and stabilized operators while retaining
two-component spatial layouts. Cartesian angular-momentum diagnostics are
not applicable and must not be interpreted as a cylindrical conservation proof.
Mass remains a full-ring ledger; transfer momentum compares the parity-aware
route deposition rather than incorrectly equating odd-reflected nodal radial
momentum with an unreflected particle sum.

### Pressurized-cylinder relaxation

Run `python -m examples.material_point_axisymmetric` for an initially unstressed
thick cylinder with inner/outer radii 1 and 2, 3D Neo-Hookean material, zero axial
grid velocity, and physical linear body drag. Equivalent full-ring pressure
forces act on the innermost particle layer. Accepted explicit steps build stress
and deformation dynamically; this is not an analytically prestressed force
cancellation.

For the independent small-strain, plane-strain reference, radial equilibrium
`d(sigma_r)/dr + (sigma_r-sigma_theta)/r = 0` gives
`sigma_r = A-B/r^2`, `sigma_theta = A+B/r^2`, where
`A=(pi*a^2-po*b^2)/(b^2-a^2)` and
`B=(pi-po)*a^2*b^2/(b^2-a^2)`. With shear modulus `mu` and Lamé parameter
`lambda`, displacement is
`u=A*r/[2*(mu+lambda)] + B/[2*mu*r]`. Here `pi` and `po` denote the inner
and outer pressures, not the circle constant.

The integration test checks relaxation, full-ring volume, displacement over all
particles, and recovered radial/hoop stresses away from two boundary cells.
Its 10% relative weighted-L2 spatial tolerance is specific to a 24-cell wall
and pressure smeared onto the innermost particle layer; `pressure/mu=1e-3`
keeps finite-strain corrections subordinate. This is a bounded regression
criterion, not an established general accuracy or convergence guarantee.

## Volumetric stabilization

Nearly incompressible materials may lock even when ordinary affine patch tests
pass. Select the treatment explicitly:

```python
stabilization = phx.discretization.MPMVolumetricStabilizationPlan(
    "f_bar", f_bar_blend=1.0
)
method = phx.discretization.ExplicitMPMMethodPlan(
    volumetric_stabilization=stabilization
)
```

`"none"` is the unchanged default. `"b_bar"` replaces only the particle
velocity-gradient trace by a background-cell quadrature average and uses its exact
adjoint in internal force. It is a cell adaptation of
[Hughes' mean-dilatation construction](https://doi.org/10.1002/nme.1620150914);
the [Coombs et al. locking paper](https://doi.org/10.1016/j.cma.2018.01.010)
instead develops a cell F-bar formulation.

`"f_bar"` projects the **total assumed trial Jacobian** from particles to nodes
and back using current physical particle volumes, following
[Zhao–Jiang–Choo Eq. (27) and Algorithm 1](https://arxiv.org/html/2209.02466v3).
It maintains both the assumed deformation used by the material and the raw
physical deformation used to update the projection volume. The optional blend
interpolates local and projected determinants; zero blend is the local increment,
not a reset to the physical deformation. Internal-force gradients remain
unmodified. Neither mode is pressure smoothing. See the
[API formulas and support boundaries](api/discretization/material_point.md#volumetric-stabilization).

Dense/block-sparse nodal storage and per-field projection are supported locally.
Nontrivial distributed/AMR stabilization is explicitly refused. B-bar requires
the entire quadrature stencil to fit the admitted grid support; boundary
truncation is not silently repaired.

Run the dynamic example with
`python -m examples.material_point_volumetric_stabilization --mode f_bar`.
The phase-separated benchmark is
`python -m benchmarks.mpm_volumetric_stabilization --particles 64 1024 16384`.
The independent plane-strain test samples a divergence-free bending displacement
at `nu=0.499` and compares restoring-work modal compliance with its analytic
elastic value. It is **not** a converged cantilever equilibrium calculation.
The coarse Q1 grid has noticeable F-bar boundary-projection error (20–45% in
compliance), while still resolving the much larger unstabilized locking error.
This bounded diagnostic does not establish universal accuracy tolerances.

## Stability and transactional rejection

Every prescribed step reports acoustic, advective, and force restrictions:

```text
acoustic = cfl_acoustic h_min / max(c_wave)
advective = cfl_advective h_min / max(|v|)
force = cfl_force sqrt(h_min / max(|a|))
selected = min(acoustic, advective, force)
```

The constitutive wave-speed bound is mandatory and evaluated on the accepted current
deformation. A scheduled rollout never silently clamps or retries an oversized step.
It returns `STABILITY_LIMIT_EXCEEDED`, preserves all particle/history arrays, and leaves
physical time and accepted-step count unchanged.

The same transaction covers route/domain failure, APIC moment failure, invalid material
state, nonpositive `J`, and nonfinite candidate state.

## Fixed-temporal rollout

```python
mesh = phx.discretization.TemporalMesh.uniform(
    0.0, final_time, steps, role="internal"
)
rollout = phx.solver.ScheduledMPMRolloutPlan(
    compiled.dynamics,
    mesh,
    retention="trajectory",
    replay=phx.solver.MPMReplayPolicy("block", block_size=32),
)
result = rollout.rollout(initial, arguments)
```

Replay and retention are independent:

- replay `full`: ordinary scan;
- replay `step`: rematerialize each step;
- replay `block`: retain block carries and rematerialize inner scans;
- retention `final`: emit only the final particle state;
- retention `checkpoints`: emit fixed-stride post-step states;
- retention `trajectory`: emit every post-step state.

Step/block replay requires deterministic or compensated splat accumulation. Fast
accumulation is available only with full replay and carries no exact replay certificate.
A rejected interval latches the remaining fixed schedule inactive.

## Differentiation

`geometry_ad="piecewise"` differentiates weights inside the executed routing and branch
program. Route changes, support boundaries, prescribed-boundary activation, material
failure, and step acceptance remain discrete boundaries. `gradient_report` compares JVP,
VJP, and centered finite differences only when perturbed rollouts retain identical route
digests and statuses.

`geometry_ad="frozen"` stops assignment geometry derivatives. This is a surrogate
sensitivity, not the finite difference of an ordinarily rebuilt rollout. Its report
therefore certifies JVP/VJP consistency and deliberately omits ordinary finite-difference
agreement.

## Evidence

Each step reports:

- source and grid mass;
- linear and APIC angular momentum transfer defects;
- net internal force;
- partition, gradient-sum, and first-moment defects;
- active nodes and routes;
- APIC moment condition;
- material admissibility and min/max `J`;
- APIC particle and grid kinetic energies;
- reference material energy;
- external and prescribed-boundary work;
- energy-balance defect;
- stability limits, status, rejection reasons, and route digest.

Global angular momentum is certified only on nonperiodic domains. A periodic
torus has no unique global position origin, so periodic runs mark that evidence
invalid rather than comparing unwrapped particles with canonical grid nodes.

A rollout retains scalar evidence independently of particle-state retention.

## Compatibility matrix

| Capability | Qualified composition |
|---|---|
| USF, USL-minus, MUSL variants | Every prepared nodal field; accepted constraints are reapplied simultaneously |
| Plane stress | Scalar isotropic and vector director-coupled closures with algorithmic tangents |
| Finite-strain J2 and geomechanics | Explicit schedules and tangent-bearing implicit operators |
| uGIMP | Explicit schedules and fixed-domain dense implicit routes |
| cpGIMP, CPDI, CPDI2 | Explicit schedules plus qualified moving-domain JVP/VJP route operators |
| Rigid and nodal-field contact | Explicit projection plus sharp/smooth implicit complementarity operators |
| Multiple material fields | All explicit schedules; simultaneous contact for up to three occupied fields per node |
| Adaptive time | Explicit methods; scheduled replay carries derivative contract |
| Active and compact blocks | Explicit storage plus compact residual, contact, fracture, and transpose operators |
| Distributed execution | Static block ownership, owner migration, halo reduction, transactional checkpoints |
| Particle lifecycle and ratio-two AMR | Fixed-capacity activation/split/merge and conservative multilevel transfer |
| Diffuse and sharp fracture | Explicit mechanics plus bounded damage/topology epochs |

## Advanced capabilities and limits

The baseline remains the semantic reference. Qualified extensions are documented
separately:

- [USF/MUSL schedules and adaptive time](guides_mpm_schedules.md);
- [plane stress and finite-strain J2](guides_mpm_constitutive_extensions.md);
- [rigid friction and multiple nodal fields](guides_mpm_contact_fields.md);
- [uGIMP, cpGIMP, CPDI, and CPDI2](guides_mpm_particle_domains.md);
- [adaptive replay and dense implicit MPM](guides_mpm_adaptive_implicit.md);
- [diffuse/sharp fracture and block storage](guides_mpm_fracture_sparse.md).

Commercial profiles extend these reference paths with transfer/advection families,
K-way contact, general plane stress, pressure-dependent materials, coupled
poro-thermal fields, sparse implicit actions, deterministic/distributed execution,
particle lifecycle, ratio-two AMR, event-aware derivatives, and executable release
gates. See the [commercial architecture](guides_mpm_commercial_architecture.md) and
[derivative/VVUQ](guides_mpm_commercial_derivatives_vvuq.md) guides. Every supported
composition is an exact `MPMClaimTuple`; absent or rejected tuples fail closed rather
than inheriting support from a nearby case.

Electrostatic/electromagnetic [PIC](guides_particle_in_cell.md) and
free-surface [FLIP](guides_flip.md) are separate method families over the same
prepared splat data plane; they do not reuse MPM constitutive state or explicit
stress-update dynamics.
