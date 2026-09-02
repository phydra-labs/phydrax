# Advanced atomistic polarization

PhydraX represents induced-dipole polarization as immutable plans and prepared,
fixed-capacity runtimes. Preparation binds permanent multipoles, d/p/u scaling, active
sites, the matrix-free operator, and its local preconditioner. Runtime solves never change
array shapes and are suitable for `jax.jit`.

## Operator and field semantics

For induced dipoles `u`, the prepared operator applies

```
A(u) = alpha^-1 u - T_u(u)
A(u) = E_p
```

without materializing the `3N x 3N` matrix. `PolarizationScaleData` gives the three scale
matrices distinct meanings:

- **d field**: scaled permanent field used by the direct-response predictor;
- **p field**: scaled permanent field on the linear-system right hand side;
- **u field**: scaled induced-dipole field in the mutual response operator.

All three matrices are symmetric, zero-diagonal, and fixed at preparation. Conflating the
three fields changes both the predicted initial state and the stationary energy, so the
operator result reports `d_field`, `p_field`, and `u_field` separately.

```python
multipoles = phx.atomistic.PermanentMultipoleSiteData(
    charges,
    permanent_dipoles,
    quadrupoles,
    polarizabilities,
    damping,
)
scaling = phx.atomistic.PolarizationScaleData(d_scale, p_scale, u_scale)
plan = phx.atomistic.PolarizationPlan(
    maximum_iterations=80,
    tolerance=1e-8,
    force_tolerance=1e-9,
    solver_kind="pcg",
)
prepared = plan.prepare(multipoles, scaling=scaling)
result = prepared.solve(positions)
```

`PolarizationOperatorPlan`, `PolarizationPreconditionerPlan`, and
`PolarizationSolverPlan` may also be composed directly. The polarizability
preconditioner applies the inverse local `alpha^-1` block and never assembles the mutual
interaction matrix.

## PCG, TCG, and warm starts

PCG executes a fixed-capacity loop of `maximum_iterations`; converged lanes retain their
state. TCG executes exactly `tcg_order` operator applications in its compiled loop and is
therefore useful when fixed cost is more important than iterative convergence. TCG still
reports the measured residual rather than claiming convergence from its order.

A prepared solver creates a fixed-shape `PolarizationPredictorState` with two induced-
dipole history frames. The first solve uses the direct d-field response. Subsequent solves
use the previous solution and then a two-frame linear predictor. The returned
`PolarizationSolveResult.predictor_state` is the immutable state for the next step:

```python
warm = prepared.initial_predictor_state()
first = prepared.solve(positions_0, predictor_state=warm)
second = prepared.solve(
    positions_1,
    predictor_state=first.predictor_state,
)
```

A predictor state is preparation-specific. Passing it to a solver with different
multipoles, scaling, capacity, or solver policy raises rather than silently reusing an
incompatible history.

## Residual and force validity

Solver convergence and force validity are separate gates. `tolerance` controls whether the
linear solve is accepted; `force_tolerance` controls whether stationary-envelope forces
are qualified. A state can therefore be converged while `force_valid` is false. Energy and
force evaluation is fail-closed: `evaluate_polarization` and
`evaluate_prepared_polarization` return NaN energy/forces unless geometry, periodic data,
solve convergence, force residual, and derivative finiteness all pass.

The polarization energy is the scalar variational functional

```
U(u, R) = 1/2 u . A(R)u - u . E_p(R).
```

Forces use the envelope derivative with the converged `u` held stationary. The associated
`PolarizationDifferentiationEvidence` records the residual gate, fixed-topology contract,
periodic contract, and finite derivative status. `implicit_polarization_jvp` preserves the
compact `(primal, tangent)` interface, while `evaluate_implicit_polarization_jvp` returns a
`PolarizationJVPResult` carrying the same calculation and its explicit implicit-mode
evidence. Both solve

```
A du = -d_R(Au - E_p)
```

for induced-dipole sensitivities. Both derivative paths are conditional on fixed site
capacity, fixed d/p/u scaling, and fixed periodic modes.

## Periodic multipoles

Pass a `MultipolePMEPlan` to `PolarizationPlan(periodic_plan=...)` at plan construction.
Every solve or evaluation using that prepared operator must then provide a finite,
nonsingular `(3, 3)` cell whose row lattice vectors are mutually orthogonal. A cell passed
to a nonperiodic plan is also rejected. The orthogonality gate is deliberate: componentwise
fractional wrapping is a certified nearest-image map for this contract, whereas accepting
an arbitrary skew cell without a prepared lattice-image search would silently select wrong
pair images. This bidirectional contract prevents an accidental mixture of real-space
fields and missing or geometrically inconsistent reciprocal fields.

The periodic operator combines an `erfc(alpha r)` minimum-image real-space kernel with
the reciprocal mesh kernel, removes the reciprocal dipole self field, and uses the
Cartesian convention whose structure factor contains `-i k . mu` and
`-1/2 k . Q . k`. `alpha` must make omitted real-space images negligible for the chosen
cell; `grid_shape` controls reciprocal resolution.

Reciprocal modes are unscaled, so the real-space path applies a separate
`(damping * scale - 1)` pair correction for each of d, p, and u. This prevents reciprocal
modes from silently restoring a centrally excluded or attenuated interaction. Coincident
fully excluded sites are supported by nonperiodic operators; periodic Ewald exclusion
corrections still require distinct active positions. Cell singularity, coincident required
pairs, and reciprocal nonfiniteness are runtime evidence and fail closed.

## Advanced scalar force-field terms

`PolarizableForceFieldPlan` composes polarization with original scalar-energy terms:

- `Buffered147Potential` for buffered 14-7 van der Waals interactions;
- `ChargePenetrationPotential` for damped core/valence electrostatics;
- `ChargeTransferPotential` for short-range transfer attraction;
- `ChargeFluxPotential` for charge-conserving bond and angle flux coupled to Coulomb
  energy;
- `DampedDispersionPotential` for Tang--Toennies C6/C8/C10 dispersion;
- `PauliRepulsionPotential` for Born--Mayer exchange repulsion;
- `StretchBendPotential`, `AngleAnglePotential`, and `OutOfPlaneBendPotential` for selected
  valence cross couplings.

Each term owns immutable, fixed-capacity parameters and a stable `term_id`. Pair scales are
symmetric and zero-diagonal. Route constructors validate index capacity and distinctness
before compilation.

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

All forces are gradients of the reported total scalar energy. The virial is the matching
affine-coordinate contraction of positions and forces. `PolarizableForceQualification`
reports energy, force, virial, per-term finiteness, net-force balance, polarization force
validity, and the derivative modes. Scalar pair and route terms are nonperiodic; only the
multipole polarization component consumes the explicit periodic contract described above.
