# Force-density form-finding

`phydrax.applications.solid_mechanics` provides differentiable equilibrium and
inverse design for pin-jointed axial-member systems. The force-density method
finds geometry from signed member force densities, applied loads, and prescribed
coordinate degrees of freedom. It is a discrete algebraic form-finding method,
not a finite-element constitutive analysis.

## Mathematical contract

For an oriented member-node incidence matrix `D`, nodal positions `X`, applied
nodal loads `p`, and signed member force densities `q`, equilibrium is

```text
A(q) X = p
A(q) = Dᵀ diag(q) D
```

A positive force density denotes tension and a negative force density denotes
compression. If prescribed coordinates define the affine representation
`X = P z + g`, the reduced system is

```text
Pᵀ A(q) P z = Pᵀ [p - A(q) g].
```

The resulting member vector, length, and axial force are

```text
vₑ = Xᵥ - Xᵤ
ℓₑ = ||vₑ||₂
tₑ = qₑ ℓₑ.
```

The equilibrium matrix is a weighted graph Laplacian, not a material stiffness
matrix. Force density does not specify Young's modulus, area, rest length,
buckling resistance, or bending behavior.

## Forward form-finding

A structure owns fixed topology and coordinate constraints. Numeric force
densities, prescribed coordinate values, and load parameters remain separate.

```python
import jax.numpy as jnp
import phydrax as phx

fd = phx.applications.solid_mechanics

structure = fd.ForceDensityStructure.from_edges(
    jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
    3,
    2,
    fixed_nodes=(0, 2),
)
reference = jnp.asarray(((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)))
loads = jnp.asarray(((0.0, 0.0), (0.0, -1.0), (0.0, 0.0)))
inputs = fd.ForceDensityInputs(
    jnp.ones((2,)),
    structure.prescribed_values(reference),
    loads,
)
problem = fd.ForceDensityProblem(structure, sign_mode="tension")
result = fd.force_density_equilibrium(problem, inputs)

assert result.successful
assert jnp.allclose(result.state.positions[1], jnp.asarray((0.0, -0.5)))
```

`ForceDensityState` carries positions, member vectors and lengths, signed axial
forces, applied and internal nodal forces, equilibrium residuals, and support
reactions. Phydrax uses distinct signs:

```text
equilibrium residual = applied - internal
support reaction     = internal - applied, on constrained coordinates.
```

A result is successful only when the selected numerical solver succeeds and the
recomputed physical residual, force balance, geometry, and load model pass their
contracts.

## Coordinate constraints

`fixed_nodes` constrains every coordinate of the selected nodes. More general
rollers and partial restraints use an explicit Boolean coordinate mask:

```python
constraints = jnp.asarray(
    (
        (True, True),
        (False, True),
    )
)
structure = fd.ForceDensityStructure.from_edges(
    jnp.asarray(((0, 1),), dtype=jnp.int32),
    2,
    2,
    constrained_dofs=constraints,
)
```

Every active connected component must constrain its translational mode in every
coordinate. An unconstrained component is rejected while constructing the
structure rather than entering a singular solve. All-constrained systems are
valid and bypass the linear solver while still reporting loads and reactions.

Member orientation is algebraic only: reversing an endpoint pair leaves
positions, lengths, and forces unchanged. Parallel physical members remain
separate members. Active self-loops are invalid.

## Sign modes and solver evidence

`ForceDensityProblem.sign_mode` declares what PhydraX may certify:

- `"tension"`: active force densities are strictly positive;
- `"compression"`: active force densities are strictly negative, and the
  equilibrium system is multiplied by `-1` to expose a positive-definite solve;
- `"fixed-mixed"`: each active member follows a supplied static sign;
- `"unrestricted"`: active values may use either sign.

Anchored sign-definite systems carry construction evidence for self-adjoint
positive definiteness. Mixed-sign systems carry self-adjoint evidence only and
may be indefinite or singular. PhydraX does not add diagonal jitter or infer
stability from initial values.

`plan_force_density` freezes sparse contribution routes and a native linear
solve template. `prepare_force_density` binds coefficients, and
`refresh_force_density` changes only numeric values:

```python
plan = fd.plan_force_density(problem, inputs)
prepared = fd.prepare_force_density(plan, inputs)
first = fd.solve_force_density(prepared)

changed = fd.ForceDensityInputs(
    2.0 * inputs.force_densities,
    inputs.prescribed_values,
    inputs.load_parameters,
)
second = fd.solve_force_density(fd.refresh_force_density(prepared, changed))
```

The default linear policy requests a device-bindable mathematically
differentiable solve. Explicit caller policies may change the provider,
tolerances, resources, preconditioning, or differentiation contract. The
symbolic topology and operator identity cannot drift during numeric refresh.

## Loads

A load model maps positions and a parameter PyTree to full nodal loads.

### Fixed nodal loads

`FixedNodalLoadModel` is the default. Its load parameter is an array with shape
`(nodes, dimension)`, and the equilibrium remains linear.

### Edge line loads

`EdgeLineLoadModel` accepts global vector load per unit length for every member.

- `measure="reference"` uses fixed positive reference lengths and remains
  linear in position;
- `measure="current"` uses current member lengths and makes equilibrium
  nonlinear.

Each integrated member load is split equally between its endpoints. The model
uses global vectors only. A spatial line has no unique local transverse frame
without an additional material director, so Phydrax does not invent one.

### Follower surface pressure

`SurfacePressureLoadModel` requires three-dimensional positions and oriented
`PolygonalConnectivity`. Signed scalar pressure follows the current surface:

- T3 cells use constant triangle shape functions;
- Q4 cells use bilinear shape functions and fixed 2 × 2 Gauss integration;
- orientation reversal flips the pressure force;
- degenerate current cells are invalid nonlinear states.

### Composition

`CompositeForceDensityLoadModel` sums a static tuple of models. Its parameter
value is a matching tuple. If any child depends on position, the complete
problem uses the nonlinear route.

## Position-dependent equilibrium

For a position-dependent load `p(X, θ)`, PhydraX solves the physical root

```text
Pᵀ [A(q) X - p(X, θ)] = 0.
```

An explicit initial position is required. It must satisfy the prescribed
coordinates and the load model's geometric domain. The selected native
nonlinear method—Newton--Krylov by default—owns iteration and globalization.
Final success is certified against the physical force residual, not coordinate
change between iterates.

`implicit_root_result` supplies the solution-map derivative. Tangent and adjoint
systems use the exact final Jacobian

```text
Pᵀ [A(q) - ∂p/∂X] P,
```

not the weighted Laplacian alone and not the finite nonlinear iteration history.
Status, iteration counts, and solver decisions remain nondifferentiable evidence.

## Inverse form-finding

`ForceDensityDesignProblem` adapts the physical solve to the ordinary PhydraX
state/design contract. A decoder maps an arbitrary design PyTree to
`ForceDensityInputs`, while the objective receives the reconstructed physical
state.

```python
plan = fd.plan_force_density(problem, inputs)


def decode(magnitude, _):
    return fd.ForceDensityInputs(
        jnp.repeat(magnitude.reshape(()), 2),
        inputs.prescribed_values,
        inputs.load_parameters,
    )


design_problem = fd.ForceDensityDesignProblem(
    plan,
    decode,
    lambda state, design, _: (state.positions[1, 1] + 0.25) ** 2,
    design_bounds=phx.optim.Bounds(0.2, 5.0),
)
design = fd.solve_force_density_design(
    design_problem,
    jnp.asarray(1.0),
)
```

The decoder may expose force densities, prescribed support coordinates, load
parameters, or any combination. Support identity and topology remain static.
Tension or compression design should use bounds or positive parameter
transforms that keep force densities away from zero.

`force_density_load_path(state)` returns
`sum(abs(axial_force) * member_length)` over active or explicitly selected
members.

## Hard constraints

`ForceDensityDesignConstraint` is one generic adapter from a callback on
`ForceDensityState` to `StateDesignConstraint`. There is no class hierarchy per
node, member, coordinate, or observable.

- `ReducedAdjoint` handles unconstrained and box-bounded soft objectives;
- `ReducedMMA` handles scalar inequalities when selected explicitly;
- vector constraints, equalities, and large elementwise constraint sets use
  `compile_structured_force_density_design` followed by
  `solve_structured_force_density_design`.

The force-density wrapper lowers equilibrium and every physical constraint,
retains the structured KKT result, reconstructs final physical inputs, and
recertifies the accepted equilibrium. A constrained problem never silently
selects an optimization method.

## Differentiable layer use

A prepared solve is an ordinary JAX PyTree. Numeric binding may occur inside a
compiled scan or differentiated objective while preserving the linear template.
The mathematical solve derivative includes operator-coefficient, support, and
load dependence. A neural model can therefore emit force-density magnitudes or
load parameters and call the same prepared solve; no separate neural-layer API
is required.

## Prepared nonlinear refresh and preconditioning

Position-dependent problems prepare one native nonlinear solve and retain its
symbolic Jacobian linear template. `refresh_force_density` calls
`refresh_nonlinear`, preserves derivative structure, and increments numeric
versions. Sign-definite problems use a Jacobi approximate inverse assembled from
the signed weighted-Laplacian equilibrium operator as the default Newton right
preconditioner. Newton still applies the exact root Jacobian
`Pᵀ[A(q) - ∂p/∂X]P`; the weighted operator changes conditioning, not the solved
problem.

`ForceDensityPlan.input_signature` freezes the complete load-parameter PyTree,
leaf paths, shapes, and dtypes. Refresh rejects structural drift before numerical
execution. Plan identity also includes nonlinear termination, precision,
implicit-derivative policy, preconditioner structure, and initial-state dtype.

## Load evidence and physical load laws

`ForceDensityState.load_state` retains aggregate nodal load, individual named
components, validity, and the minimum geometric regularity margin.

- `ReferenceMemberSelfWeightModel` converts reference line mass and gravity to
  endpoint loads.
- `SurfaceTractionLoadModel` integrates global vector traction over current or
  reference T3/Q4 area.
- `SurfacePressureLoadModel` integrates scalar follower pressure and rejects
  degenerate or locally folded Q4 geometry.
- `PneumaticPressureLoadModel` supports fixed pressure or the volume law
  `p Vᵏ = constant` on a closed oriented T3/Q4 surface.
- `CompositeForceDensityLoadModel` preserves every child component rather than
  exposing only an anonymous sum.

`enclosed_surface_volume` returns signed volume and therefore makes surface
orientation part of the physical contract.

## Structural observables

The application exports pure functions rather than one goal subclass per
observable:

- scaled target and uniformity residuals;
- member directions and angles;
- point, line, plane, and segment geometry;
- reaction direction;
- collinearity and graph fairness;
- cell area, Q4 planarity, and rectangularity;
- signed-distance target geometry.

Every dimensional residual requires an explicit physical scale.

## Mechanisms and stability

`analyze_force_density_mechanisms` constructs the restrained rigidity operator
and returns both infinitesimal mechanism and axial self-stress eigenspaces. This
is algebraic evidence; it does not assert stability.

`analyze_force_density_tangent_stability` requires positive member axial
rigidities and assembles separate material and prestress contributions before
certifying the constrained tangent spectrum. Stability is never inferred from
force density alone.

`force_density_continuation_problem` exposes a scalar force/load/support path to
the native continuation runtime for branch tracking and singularity detection.

## Batching, graph evidence, and affine restraints

`solve_force_density_batch` vmaps numeric cases over one prepared topology and
returns one status and diagnostic record per case. Disjoint GraphIR structures
also report residual and balance norms per stored graph.

`ForceDensityStructure.from_affine_constraints` accepts an orthonormal free and
prescribed coordinate basis. This covers oblique rollers, symmetry relations,
and multipoint positional constraints while preserving support-coordinate design.
The coordinate-mask path remains the sparse fast path; general affine constraints
assemble a reduced dense operator.

`from_graph(..., edge_semantics=\"reciprocal-pairs\")` canonicalizes paired
directed graph routes into physical members. Optional stable node and member IDs
remain attached to the structure and do not become numeric JAX leaves.

## Scope and limitations

A successful force-density result establishes equilibrium of a pin-jointed
axial-force network under the declared model. It does not establish:

- material compatibility or member sizing;
- elastic displacement under a changed load case;
- cable slackness or compression buckling;
- bending, joint stiffness, or construction sequence;
- prestress realizability;
- stability of a mixed tension/compression equilibrium.

Mixed-sign systems may contain mechanisms or self-stress null directions.
Shape-only self-stressed objectives may also have a uniform force-density scale
null direction. Use explicit bounds, regularization, spectral analysis, or
continuation evidence rather than numerical perturbations that change the
problem.

Runnable workflows:

- `examples/force_density_form_finding.py`;
- `examples/force_density_inverse_design.py`;
- `examples/force_density_equal_force_truss.py`;
- `examples/force_density_gridshell_planarization.py`;
- `examples/force_density_vault_shape_matching.py`;
- `examples/force_density_support_load_finding.py`;
- `examples/force_density_structured_design.py`;
- `examples/force_density_pneumatic_membrane.py`.
