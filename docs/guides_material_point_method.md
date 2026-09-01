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

The target must be nodal, uniform, and degree-two. Closed-domain MPM rejects truncated
support; it never renormalizes or drops an unreported fraction.

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
