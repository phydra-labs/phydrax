# Domain-decomposed PINNs

## Overlapping partition-of-unity Poisson PINN

This solves the one-dimensional manufactured problem `u'' + pi² sin(pi x) = 0`
with two overlapping local networks and one globally assembled residual.

```python
import jax.numpy as jnp
import jax.random as jr
import optax
import phydrax as phx

space = phx.domain.Interval1d(0.0, 1.0)
cover = phx.domain.cartesian_subdomain_cover(
    space,
    "x",
    2,
    overlap_fraction=0.2,
)

local = {}
for patch, key in zip(cover.patches, jr.split(jr.key(0), 2), strict=True):
    model = phx.nn.models.MLP(
        in_size=1,
        out_size="scalar",
        width_size=16,
        depth=2,
        key=key,
    )
    local[patch.patch_id] = patch.domain.Model("x")(model)
family = phx.domain.LocalFieldFamily("u", cover, local)

interior = space.component()
boundary = space.component({"x": phx.domain.Boundary()})
forcing = space.Function("x")(
    lambda x: jnp.pi**2 * jnp.sin(jnp.pi * x[0])
)
target = space.Function("x")(lambda x: jnp.sin(jnp.pi * x[0]))

pde_condition = phx.conditions.Residual(
    "u",
    interior,
    lambda u: phx.operators.laplacian(u, var="x") + forcing,
)
boundary_condition = phx.conditions.Dirichlet(
    "u",
    boundary,
    target=target,
)
layout = phx.domain.SampleLayout((("x",),))
pde = phx.terms.ResidualPenalty(
    pde_condition,
    phx.integration.per_step(
        phx.integration.mean_over(interior),
        phx.domain.PointSampling(64, layout=layout),
    ),
)
bc = phx.terms.ResidualPenalty(
    boundary_condition,
    phx.integration.per_step(
        phx.integration.mean_over(boundary),
        phx.domain.PointSampling(16, layout=layout),
    ),
    scale=10.0,
)

problem = phx.solver.FunctionalDecompositionProblem.partition_of_unity(
    family,
    terms=(pde, bc),
)
prepared = phx.solver.prepare_functional_decomposition(
    problem,
    phx.solver.FunctionalDecompositionPlan(
        phx.solver.JointDecompositionTraining(2000),
        required_window_regularity=2,
    ),
)
result = phx.solver.solve_functional_decomposition(
    prepared,
    optax.adam(1.0e-3),
)
u = result.global_field
```

The PDE residual is evaluated after partition-of-unity assembly, so automatic
differentiation includes derivatives of the compact windows.

## Broken fields with conservative interfaces

For a non-overlapping cover, bind local residual terms to stable local names and add
pair-scoped conditions.

```python
cover = phx.domain.cartesian_subdomain_cover(space, "x", 2)
family = phx.domain.LocalFieldFamily("u", cover, local_fields)
pairing = cover.pairings[0]

value_jump = phx.conditions.SubdomainValueJump(
    family.field_name(pairing.left_patch_id),
    family.field_name(pairing.right_patch_id),
    pairing,
)
flux_jump = phx.conditions.SubdomainFluxJump(
    family.field_name(pairing.left_patch_id),
    family.field_name(pairing.right_patch_id),
    pairing,
    lambda u: phx.operators.grad(u, var="x"),
    lambda u: phx.operators.grad(u, var="x"),
)

value_term = phx.solver.ScopedFunctionalTerm(
    phx.terms.ResidualPenalty(value_jump, interface_source),
    phx.solver.PairScope(pairing.pairing_id),
)
flux_term = phx.solver.ScopedFunctionalTerm(
    phx.terms.ResidualPenalty(flux_jump, interface_source),
    phx.solver.PairScope(pairing.pairing_id),
)
problem = phx.solver.FunctionalDecompositionProblem.broken(
    family,
    terms=(*local_pde_terms, value_term, flux_term),
)
```

Use `JointDecompositionTraining` for cPINN/XPINN-style monolithic optimization,
`BlockDecompositionTraining` for block-coordinate optimization, or
`SchwarzDecompositionTraining` for frozen-neighbor local sweeps.

The returned `BrokenField` keeps left and right traces explicit. Calling
`as_domain_function(ownership="first")` chooses a pointwise owner but does not certify
interface continuity.

## Multidimensional and nonuniform covers

```python
axes = (
    phx.domain.AxisPartition([0.0, 0.2, 0.6, 1.0], overlap_fraction=0.15),
    phx.domain.AxisPartition([-1.0, 0.0, 1.0], overlap_fraction=0.2),
)
cover = phx.domain.cartesian_subdomain_cover(
    phx.domain.HyperRectangle([0.0, -1.0], [1.0, 1.0]),
    "x",
    axis_partitions=axes,
)
```

Each internal face has an exact parameter domain, surface measure, and canonical
normal. `validate_mapped_cover` audits the coordinate realizations used by both
traces.

## Colored POU block training

```python
plan = phx.solver.FunctionalDecompositionPlan(
    phx.solver.BlockDecompositionTraining(
        sweeps=100,
        inner_iterations=10,
        sweep="colored",
    )
)
prepared = phx.solver.prepare_functional_decomposition(problem, plan)
result = phx.solver.solve_functional_decomposition(
    prepared,
    optax.adam(1.0e-3),
)
```

Preparation derives a parameter-conflict coloring and one exact local parameter
subspace per patch. `IntegrationOwnership` is available when global residual
integration is localized into patch terms.

## Relaxed Schwarz traces

```python
strategy = phx.solver.SchwarzDecompositionTraining(
    sweeps=50,
    inner_iterations=20,
    sweep="jacobi",
    relaxation=0.7,
    interface_tolerance=1.0e-6,
)
prepared = phx.solver.prepare_functional_decomposition(
    broken_problem,
    phx.solver.FunctionalDecompositionPlan(strategy, trace_points=128),
)
result = phx.solver.solve_functional_decomposition(
    prepared,
    optax.adam(1.0e-3),
)
interface_defect = result.state.trace_state.maximum_defect
```

`TraceExchangeState` keeps outgoing values and relaxed incoming targets on one fixed
paired design. This makes restart and interface stopping independent of randomly
resampled training points.

## Field-preserving h refinement

```python
refined = phx.solver.refine_axis_partition(axis_partition, cell_index=2)
candidate = phx.domain.cartesian_subdomain_cover(
    domain,
    "x",
    axis_partitions=(refined,),
)
transaction = phx.solver.prepare_adaptive_topology_transaction(
    family,
    candidate,
    fixed_audit_points,
)
refined_family = transaction.commit()
```

The transfer freezes the old assembled field and restricts it onto the new patches.
Commit requires both coverage and transfer-error evidence.
