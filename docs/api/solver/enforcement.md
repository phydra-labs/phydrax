# Exact enforcement

Enforcement turns declarative conditions into exact field transforms. The
compiler validates field dependencies and derivative requirements, orders
specifications by boundary/initial/interior stage, and produces one
`EnforcementProgram` applied before every scalar term.

For the low-level transforms, see [Enforcement API](../enforcement.md). For the
mathematical construction, see
[Physics-Constrained Interpolation](../../appendix/physics_constrained_interpolation.md).

## Compile once

```python
import jax.random as jr
import jax.numpy as jnp
import phydrax as phx

space = phx.domain.Interval1d(-1.0, 1.0)
u_free = space.Function("x")(lambda x: x[0])
functions = {"u": u_free}

interior = space.component()
interior_condition = phx.conditions.Residual(
    "u", interior, lambda value: value
)
interior_penalty = phx.terms.ResidualPenalty(
    interior_condition,
    phx.integration.per_step(
        phx.integration.mean_over(interior),
        phx.domain.PointSampling(16),
    ),
)

boundary = space.component({"x": phx.domain.Boundary()})
condition = phx.conditions.Dirichlet("u", boundary, target=0.0)
spec = phx.enforcement.EnforcementSpec(condition, options={"var": "x"})
options = phx.enforcement.EnforcementOptions(num_reference=256)
program = phx.enforcement.compile(
    functions,
    (spec,),
    options=options,
    key=jr.key(0),
)

solver = phx.solver.FunctionalSolver(
    functions=functions,
    terms=(interior_penalty,),
    enforcement=program,
)
u = solver.ansatz_functions()["u"]
```

The interior penalty sees the transformed field. No soft boundary penalty is
needed because the Dirichlet condition is satisfied by construction.

Interior exact data is compiled through the same boundary-preserving program:

```python
anchor_points = jnp.asarray([[-0.5], [0.5]])
anchor_values = jnp.asarray([0.0, 0.0])

anchors = phx.enforcement.InteriorAnchors(
    "u",
    points={"x": anchor_points},
    values=anchor_values,
)
program = phx.enforcement.compile(
    functions,
    (spec,),
    interior=(anchors,),
    options=options,
    key=jr.key(1),
)
solver = phx.solver.FunctionalSolver(
    functions=functions,
    terms=(interior_penalty,),
    enforcement=program,
)
```

Multi-field dependencies are declared on each specification and topologically
ordered by the compiler. Geometry gates are dimensionless; `gate_method="auto"`
selects the global CAD R-equivalence gate, while `"compact"` selects the compact
fallback. `gate_saturation_fraction` and `gate_linear_fraction` configure that
fallback on `EnforcementOptions`.

Compile the program once and pass only that program as `enforcement=`. When
there are no specifications or interior anchors, pass `enforcement=None`
instead of compiling an empty program.

::: phydrax.enforcement.EnforcementProgram
    options:
        members:
            - apply

---

::: phydrax.enforcement.EnforcementSpec

---

::: phydrax.enforcement.EnforcementOptions

---

::: phydrax.enforcement.InteriorAnchors

---

::: phydrax.enforcement.compile
