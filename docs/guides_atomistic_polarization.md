# Advanced atomistic polarization

PhydraX represents induced-dipole polarization as immutable plans and prepared,
fixed-capacity runtimes. Preparation binds permanent multipoles, d/p/u pair scales, active
sites, the matrix-free operator, its local preconditioner, and (for periodic routes) the
reciprocal resources of the declared cell. Runtime solves never change array shapes and are
suitable for `jax.jit`.

## Site data and multipole conventions

`PermanentMultipoleSiteData(charges, dipoles, quadrupoles, polarizabilities, damping)`
takes positional fixed-capacity arrays of shapes `(N,)`, `(N, 3)`, `(N, 3, 3)`, `(N,)`, and
`(N,)`. The moments are site-centred *primitive* Cartesian coefficients in the system
Cartesian frame, in the `1/2 Theta : grad grad` convention: a point site carries
`(q - mu . grad + 1/2 Theta : grad grad) delta(r - r_i)`, so `Theta` is the raw, traceful,
symmetric second moment `int rho r r`, with no Buckingham `3/2` factor and no implied trace
removal. For point sites the trace only produces a contact term; for Gaussian sites it is a
real radial density and changes energies, so trace handling is never implicit.
Polarizabilities are isotropic, nonnegative, and in system units
(charge^2 length^2 / energy); a polarizability volume converts as `alpha = alpha_v / k`.
Sites with `alpha = 0` or outside the active mask are not polarizable. `damping` holds the
positive per-site parameters `a_i` used only by `damping="exponential-cubic"`.

Two site kinds are distinct definitions:

- `site_kind="point"` uses `1/R` kernels. The default `damping="exponential-cubic"`
  multiplies every bare pair tensor by `1 - exp(-sqrt(a_i a_j) R^3)`. This is a uniform
  exponential damping of every Cartesian tensor order; it is not AMOEBA Thole damping.
  `damping="none"` uses the bare kernels.
- `site_kind="gaussian"` uses unit-normalised isotropic Gaussians
  `(2 pi sigma^2)^(-3/2) exp(-r^2 / (2 sigma^2))` with the pair kernel
  `erf(R / sqrt(2 (sigma_i^2 + sigma_j^2))) / R`, requires `damping="none"`, and requires
  per-site widths at preparation (`widths=`, a `GaussianMultipoleWidths` class table of
  standard deviations). Point sites refuse widths.

Predicted moments enter the potential-program term through `MultipoleMomentFields(basis,
*, charges=..., dipoles=..., second_moments=..., trace=...)`. Each reference must be an
atom-level field with an explicit unit dimensionally equal to charge x length^degree, and
the dipole and second-moment ports' `normalization_id` must name the declared basis
exactly; shapes never select a basis. `"cartesian-primitive"` packs the six traceful
second-moment components `[T_xx, T_xy, T_xz, T_yy, T_yz, T_zz]`;
`"native-real-orthonormal"` and `"nvalchemi-ops-e3nn-component"` are traceless
five-component real-spherical layouts that differ in ordering, polar axis, and
normalization, and a degree-two field in either requires `trace="zero"` declared
explicitly. The polarization term requires every degree (charges, dipoles, and second
moments) to be supplied. Octupoles and higher degrees are not part of the public
definition.

## Operator and field semantics

For induced dipoles `u` on polarizable sites, the prepared operator applies

```
A(u) = alpha^-1 u - T_u(u)
A(u) = E_p
```

without materialising the `3N x 3N` matrix; nonpolarizable rows are the identity. The
induced dipoles minimise the Hylleraas functional of the native multipole bilinear forms,

```
E(u) = 1/2 sum_i u_i . u_i / alpha_i + 1/2 B_u(u, u) + B_p(u, M),
```

whose stationarity condition is this linear system (`B_p(u, M) = -u . E_p`).
`PolarizationScaleData(pairs, direct, polarization, mutual, *, site_capacity)` gives
three sparse pair-scale columns distinct meanings:

- **d field** (`direct`): scaled permanent field used only by the direct-response
  predictor guess;
- **p field** (`polarization`): scaled permanent field on the linear-system right-hand
  side;
- **u field** (`mutual`): scaled induced-dipole field in the mutual response operator.

Each unordered pair is listed once with scales in `[0, 1]`; every unlisted pair scales by
one and no dense `N x N` table exists. `PolarizationScaleData.unscaled(site_capacity)` is
the empty table. Conflating the three fields changes both the predicted initial state and
the stationary energy, so `PolarizationOperatorResult` reports `d_field`, `p_field`, and
`u_field` separately.

```python
multipoles = phx.atomistic.PermanentMultipoleSiteData(
    charges,
    permanent_dipoles,
    quadrupoles,
    polarizabilities,
    damping,
)
scaling = phx.atomistic.PolarizationScaleData(
    scaled_pairs, d_scale, p_scale, u_scale, site_capacity=site_capacity
)
plan = phx.atomistic.PolarizationPlan(
    coulomb_constant=coulomb_constant,
    maximum_iterations=80,
    tolerance=1e-8,
    force_tolerance=1e-9,
    solver_kind="pcg",
)
prepared = plan.prepare(multipoles, scaling=scaling)
result = prepared.solve(positions)
```

`coulomb_constant` is required in the system units. The default `route` is
`FiniteMultipoleRoute()`, the complete nonperiodic pair sum over a bounded all-pairs
reference relation (`O(N^2)` work in `row_block x N` blocks, admitted against
`maximum_workspace_bytes`). `prepare` also accepts `active_mask=` (a boolean `(N,)` array),
`widths=`, `cell=`, and `particles=`. `PolarizationOperatorPlan`,
`PolarizationPreconditionerPlan`, `PolarizationSolverPlan`, and `PolarizationPredictorPlan`
may also be composed directly. The default `"polarizability"` preconditioner applies the
inverse of the local diagonal (Jacobi `alpha`) block and never assembles the mutual
interaction matrix; `"identity"` is the unpreconditioned alternative.

## PCG, TCG, and warm starts

`solver_kind="pcg"` (the default) runs the native preconditioned conjugate-gradient solve
with an absolute residual tolerance in a fixed-capacity loop of `maximum_iterations`
(default 100). Its energy is the functional at the solution, the stationary functional.

`solver_kind="tcg"` executes exactly `tcg_order` CG steps (default 3) from a deterministic
guess, which is useful when fixed cost matters more than iterative convergence. Its energy
is the TCG functional of that order, a different functional from the stationary one. A TCG
state is never reported `converged`; `stationary` is set only when its measured residual
meets `tolerance`.

A prepared solver creates a fixed-shape `PolarizationPredictorState` holding two
induced-dipole history frames. With the default `PolarizationPredictorPlan()`, the first
PCG solve starts from the direct d-field response `alpha E_d` (`direct_fallback=True`;
otherwise zero), the second from the previous solution, and later solves from the
two-frame extrapolation `u_1 + c (u_1 - u_2)` with `c = history_coefficient` in `[0, 2]`
(default 1). The guess is detached from differentiation. The returned
`PolarizationSolveResult.predictor_state` is the immutable state for the next step; a
history frame is appended only when the solve succeeds:

```python
warm = prepared.initial_predictor_state()
first = prepared.solve(positions_0, predictor_state=warm)
second = prepared.solve(
    positions_1,
    predictor_state=first.predictor_state,
)
```

Warm starts apply to PCG only: TCG always starts from its deterministic direct-field or
zero guess and refuses a predictor state, which would make its functional history
dependent. A predictor state is preparation-specific: a history of the wrong fixed shape
raises, and a state produced by another preparation (different multipoles, scaling,
active mask, widths, route resources, or solver policy) makes the solve unsuccessful
rather than silently reusing an incompatible history.

## Residual and force validity

Solver convergence and force validity are separate gates. `tolerance` decides whether the
linear solve is stationary; `force_tolerance` (default: `tolerance`) decides whether PCG
forces are qualified, and `derivative_tolerance` (default: `tolerance`) is the gate reported
for induced-dipole tangents. A PCG state can therefore be converged while `force_valid` is
false. For TCG, `force_valid` follows the method status, because its forces are exact
derivatives of the TCG functional rather than of the stationary one.

Energy and force evaluation is fail-closed:
`evaluate_prepared_polarization(prepared, positions, *, predictor_state=None,
cell_vectors=None, pairs=None)` returns a `PolarizationEvaluation` whose energy, forces,
and (periodic routes only) `cell_gradient` are NaN unless the solve, stability evidence,
force validity, route status, and derivative finiteness all pass.
`prepared_polarization_energy` returns the differentiable scalar with its solve result.

Positive definiteness of `A` is never inferred from a converged solve.
`PolarizationStabilityPolicy(certificate, *, margin=1e-8, maximum_dense_bytes=64 MiB)`
declares how it is established for the congruent `S = alpha^1/2 A alpha^1/2`:

- `"scaled-block-gershgorin"` (the default for nonperiodic routes): a matrix-free proof
  from the Schur test on the 3x3 block norms of the actual damped or Gaussian kernel and
  mutual scales, `lambda_min(S) >= 1 - max_i sum_j ||M_ij||_2`; a bound below `margin`
  means unproved, not unstable. It covers nonperiodic routes only;
- `"dense-spectrum"`: the bounded reference for any route, including periodic Ewald and
  PME, a native dense eigensolve of the materialised `S` admitted against
  `maximum_dense_bytes` at preparation, proven when `lambda_min` exceeds
  `max(margin, 64 * 3N * eps) * max|lambda|`;
- `"assumed-positive-definite"`: an explicit user assumption, reported as
  `stability_assumed=True` with `stability_proven=False`, never as proven.

Without a proof or a declared assumption no state, energy, or force is published. A
nonpositive stationary `u . E_p`, or a negative smallest Ritz value from the optional
bounded Lanczos `PolarizationCurvatureProbe(subspace_dimension=12, maximum_steps=40,
seed=0)`, is a detected indefiniteness (polarization catastrophe) and refuses under every
certificate. Positive values prove nothing: a Ritz value only bounds `lambda_min` from
above, and an unstable mode orthogonal to `E_p` leaves `u . E_p > 0`.

The polarization energy is the scalar variational functional

```
U(u, R) = 1/2 u . A(R)u - u . E_p(R).
```

The induced dipoles are not detached from the energy. PCG differentiates them implicitly
(at stationarity this equals the envelope derivative with `u` held fixed); TCG
differentiates the fixed-order iteration algorithmically. `PolarizationDifferentiationEvidence`
records the derivative mode (`"implicit-stationary"` or `"algorithmic-fixed-order"`), the
functional, the residual against its gate tolerance, stationarity, stability, route status,
and finiteness. `evaluate_implicit_polarization_jvp(prepared, positions, tangent, *,
cell_vectors=None, pairs=None)` returns a `PolarizationJVPResult` with the induced-dipole
primal, its tangent, the state, and this evidence. For PCG the tangent solves

```
A du = -d_R(Au - E_p)
```

and for TCG it is the derivative of the fixed-order iteration. Both derivative paths are
conditional on fixed site capacity, fixed d/p/u scaling, and fixed periodic resources.

## Periodic multipoles

Periodic polarization selects a periodic multipole route at plan construction:
`EwaldMultipoleRoute(alpha, real_cutoff, modes, *, real_space=..., ...)` with an explicit
`ReciprocalModeSetPlan` of reciprocal modes (direct k), or
`ParticleMeshMultipoleRoute(alpha, real_cutoff, mesh, *, real_space=..., ...)` with a
`ReciprocalMeshPlan` (smooth PME). Periodic routes require an explicit `stability=` policy,
because the matrix-free Gershgorin proof covers nonperiodic routes only.

```python
cell = phx.discretization.PeriodicCell(cell_vectors)
route = phx.atomistic.EwaldMultipoleRoute(
    alpha, real_cutoff, phx.atomistic.ReciprocalModeSetPlan((6, 6, 6))
)
plan = phx.atomistic.PolarizationPlan(
    coulomb_constant=coulomb_constant,
    route=route,
    stability=phx.atomistic.PolarizationStabilityPolicy("dense-spectrum"),
)
prepared = plan.prepare(multipoles, scaling=scaling, cell=cell)
evaluation = phx.atomistic.evaluate_prepared_polarization(
    prepared, positions, cell_vectors=cell_vectors
)
```

A particle-mesh route additionally needs the material particle support at preparation,
`plan.prepare(multipoles, cell=system.cell, particles=system.particle_discretization())`
for a prepared atomistic system; explicit-mode Ewald refuses particle support.

Admission is exactly:

- **Cells.** A periodic route requires a fully periodic rank-3 `PeriodicCell`; general
  (triclinic) lattice vectors are admitted, and the potential-program term declares
  orthorhombic and triclinic periodic capabilities for periodic routes. Slab and other
  partially periodic cells are refused: only the tin-foil boundary is defined. A finite
  route refuses a cell, particle support, and `cell_vectors`; a periodic route requires
  `cell_vectors` at every solve or evaluation. A singular runtime cell fails closed.
- **k = 0 and boundary.** Periodic sums exclude `k = 0` (tin-foil, conducting
  macroscopic boundary); an Ewald mode set that includes the zero mode is refused. No
  dipole or second-moment `k = 0` (surface/depolarization) term exists. The polarization
  operator uses the uniform-background neutrality convention, whose
  `-pi k Q^2 / (2 alpha^2 V)` term is a charge-charge quantity and so vanishes for the
  charge-free induced-dipole sites [inference from the formula]. This exclusion is
  distinct from the Fourier D3 dispersion route, whose `k = 0` mode is part of that method
  (see the [dispersion guide](guides_atomistic_dispersion.md)).
- **Real space.** Either `real_space="explicit-image-reference"` (the default), every
  lattice translation of the prepared image stencil bounded by `maximum_image_count` and
  `maximum_workspace_bytes` with a traced completeness margin, or
  `real_space="neighborhood"`, which consumes a bound neighborhood view through
  `NeighborhoodPairs(view, site_capacity)`. That view must be the complete `IMAGE_PAIR_SUM`
  role (or an unselected directed image graph) of the system atoms, with a cutoff not
  below `real_cutoff`; its own stale, capacity, and scientific evidence gates the result.
  The explicit-image route refuses `pairs=`. `alpha` and `real_cutoff` must make omitted
  real-space images negligible for the chosen cell; the mode extents or mesh control
  reciprocal resolution, and an optional `reciprocal_cutoff` is checked against the
  current cell's support.

Reciprocal modes are unscaled, so damping and d/p/u pair scales act on the real-space
(respectively current minimum-image) member only, through explicit correction routes. This
prevents reciprocal modes from silently restoring a centrally excluded or attenuated
interaction. Co-centred (intrasite) interactions are excluded: periodic routes subtract
the smooth Ewald self-interaction of each site. Coincident required pairs, an exhausted
image or workspace bound, and reciprocal nonfiniteness are runtime evidence and fail
closed. For Gaussian sites, the width factors enter reciprocal space exactly, per width
class.

## Polarization in a potential program

`InducedDipolePolarizationPotential(plan, multipoles, *, scaling=None, widths=None,
moment_fields=None, polarizability_field=None, name="induced-dipole-polarization",
force_group=0)` runs the same solve inside the scalar program energy from the
deterministic guess (no predictor history). Permanent moments and polarizabilities are
explicit site data or typed predicted fields; a predicted polarizability must be a
`POLARIZABILITY` field with an explicit physical unit (dimensionless
`ATOMIC_POLARIZABILITY` ratios are refused). Program derivatives therefore include the
dependence on predicted charges, moments, and polarizabilities. Preparation requires the
plan's `coulomb_constant` to equal the system unit constant and the site capacity to equal
the system capacity; the system's active mask, cell, and (for particle-mesh routes)
particle support are bound automatically. The term is successful only when the state is
successful and force-valid; its per-term evidence is a `PolarizationTermEvidence`.

With `real_space="neighborhood"`, the term declares one `IMAGE_PAIR_SUM` requirement
owned by the term name at `real_cutoff` with no pair exclusions, and the program must be
prepared with an explicit bundle that serves it; no resource is inferred. Finite and
explicit-image routes declare no neighborhood requirement and take the empty bundle
`phx.atomistic.AtomisticNeighborhoodBundlePlan((), (), ())`.

```python
term = phx.atomistic.InducedDipolePolarizationPotential(
    phx.atomistic.PolarizationPlan(
        coulomb_constant=system.plan.units.coulomb_constant,
        route=phx.atomistic.ParticleMeshMultipoleRoute(
            1.0,
            3.0,
            phx.atomistic.ReciprocalMeshPlan((12, 12, 12), spline_degree=4),
            real_space="neighborhood",
        ),
        tolerance=1e-12,
        maximum_iterations=60,
        stability=phx.atomistic.PolarizationStabilityPolicy("dense-spectrum"),
    ),
    multipoles,
)
program = phx.atomistic.AtomisticPotentialProgram([term])
(pairs,) = program.requirements.neighborhoods
bundle = phx.atomistic.AtomisticNeighborhoodBundlePlan(
    (pairs,),
    (
        phx.atomistic.AtomisticNeighborhoodSource(
            "polarization-image-pairs",
            pairs.entity,
            (pairs,),
            phx.atomistic.AtomisticCellListImageSearch(maximum_candidate_slots=1 << 16),
            phx.discretization.ParticleImageCapacityLadder(
                (
                    phx.discretization.ParticleImageCapacity(
                        maximum_particles_per_cell=32,
                        maximum_edges=4096,
                        maximum_degree=4096,
                        maximum_images=343,
                        maximum_case_edges=4096,
                    ),
                )
            ),
            skin=0.3,
        ),
    ),
    (
        phx.atomistic.AtomisticRoleCapacityLadder(
            pairs,
            (phx.atomistic.AtomisticRoleCapacity(maximum_routes=4096, maximum_degree=256),),
        ),
    ),
)
potential = program.prepare(system, neighborhood_bundle=bundle)
neighborhoods = potential.neighborhood_bundle.initialize(
    phx.atomistic.bind_atomistic_bundle_geometry(
        potential.neighborhood_bundle, system, positions
    )
)
evaluation = potential.evaluate(positions, neighborhoods, compute_stress=True)
```

Here `system` is a prepared periodic atomistic system and the capacities are explicit
declarations for its size; an exhausted capacity is reported by the bundle evidence, never
silently truncated.

## Stress, virial, and pressure

These quantities are distinct and must not be substituted for one another:

- **Strain stress.** For a periodic route the polarization term declares a cell
  derivative, so a program evaluation with `compute_stress=True` includes the
  polarization functional in the program's tension-positive strain stress
  `sym(dE/dstrain) / V` at fixed fractional coordinates and integer images (see the
  [dynamics guide](guides_atomistic_dynamics.md)). The solve and reciprocal coefficients
  are differentiated through the same scalar; no current-cell coefficient is cached.
  This is the polarization contribution to the potential-energy strain derivative, not
  a separately named pressure.
- **Cell gradient.** `PolarizationEvaluation.cell_gradient` is `dE/d(cell_vectors)` with
  Cartesian positions held fixed. It is not a stress by itself; forming the strain stress
  from it also needs the position gradient.
- **Position-moment virial.** `PolarizableForceFieldEvaluation.virial` is the
  affine-coordinate contraction `sum_i r_i ⊗ F_i` of the supplied positions and forces.
  It is a fixed-frame diagnostic and not a periodic stress.
- **Instantaneous pressure.** The internal pressure of a dynamics state also includes the
  kinetic momentum flux and is reported by the dynamics stack, not by the polarization
  evaluators.

Overlap-type scalar terms below (charge penetration, Pauli repulsion, charge transfer) are
nonperiodic and contribute no periodic stress of their own.

## Advanced scalar force-field terms

`PolarizableForceFieldPlan(terms, *, polarization=None, force_balance_tolerance=1e-5)`
composes optional polarization with original scalar-energy terms:

- `Buffered147Potential(radii, epsilon)` for buffered 14-7 van der Waals interactions with
  arithmetic radius mixing (`delta=0.07`, `gamma=0.12`);
- `ChargePenetrationPotential(core_charges, valence_charges, exponents)` for damped
  core/valence electrostatics;
- `ChargeTransferPotential(amplitudes, exponents)` for short-range transfer attraction;
- `ChargeFluxPotential(reference_charges, bond_routes, bond_coefficients,
  equilibrium_lengths, angle_routes, angle_coefficients, equilibrium_angles)` for
  charge-conserving bond and angle flux coupled to Coulomb energy;
- `DampedDispersionPotential(c6, damping, *, c8=None, c10=None)` for Tang--Toennies-damped
  C6/C8/C10 dispersion. This is a pairwise force-field term, distinct from the D3,
  Fourier D3, and Tkatchenko--Scheffler dispersion terms described in the
  [dispersion guide](guides_atomistic_dispersion.md);
- `PauliRepulsionPotential(amplitudes, exponents)` for Born--Mayer exchange repulsion;
- `StretchBendPotential(site_capacity, routes, stiffness, equilibrium_lengths,
  equilibrium_angles)`, `AngleAnglePotential(site_capacity, routes, stiffness,
  equilibrium_angles)`, and `OutOfPlaneBendPotential(site_capacity, routes, stiffness)`
  for selected valence cross couplings.

Each term owns immutable, fixed-capacity parameters and a stable `term_id`; every term must
have the same site capacity as the multipoles. Optional `pair_scale` tables are symmetric
and zero-diagonal. Route constructors validate index capacity and distinctness before
compilation.

```python
terms = (
    phx.atomistic.Buffered147Potential(radii, epsilon),
    phx.atomistic.DampedDispersionPotential(c6, dispersion_damping),
    phx.atomistic.PauliRepulsionPotential(pauli_amplitude, pauli_exponent),
)
force_field = phx.atomistic.PolarizableForceFieldPlan(
    terms,
    polarization=plan,
).prepare(multipoles=multipoles, scaling=scaling)
evaluation = force_field.evaluate(positions)
```

`prepare` also takes `cell=` and `particles=` for a periodic polarization plan, and
`evaluate(positions, *, predictor_state=None, cell_vectors=None)` threads the PCG warm
start and the periodic cell vectors. All forces are gradients of the reported total scalar
energy; `term_energies` and `polarization_energy` are reported separately.
`PolarizableForceQualification` reports energy, force, virial, and per-term finiteness, the
net-force balance residual against `force_balance_tolerance`, polarization force validity,
the `"energy-gradient"` derivative mode, and the `"affine-coordinate"` virial mode. Scalar
pair and route terms are nonperiodic; only the multipole polarization component consumes
the explicit periodic contract described above.

## Related electrostatics

Induced-dipole polarization is distinct from the latent Ewald summation of
`LatentEwaldPotential`, whose reported polarization and Born effective charges are
properties of learned latent charges rather than solutions of a polarizability response.
Permanent multipoles without induction use `MultipoleElectrostaticPotential` over the same
point/Gaussian definitions and routes.

`PreparedMultipoleElectrostatics.features` returns receiver-smeared potentials,
fields and field gradients. Its eager Gaussian entry point intentionally
synchronizes concrete `receiver_widths` for finite, positive-value validation and
raises `ValueError` for invalid widths. Under JIT or differentiation, width
admission stays in numerical evidence: invalid widths produce
`successful=False`. Use the compiled path inside device-resident numerical
recurrences; the eager validation boundary is not an asynchronous query API.
