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
interior_condition = phx.conditions.Residual("u", interior, lambda value: value)
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

## Periodic seams

Periodic declarations enter the same program as one prepared typed realization.
Prepare all `phydrax.conditions.Periodic` declarations of a field together and
select walls with `phydrax.domain.physical_boundary`, which omits identified
faces:

```python
box = phx.domain.HyperRectangle([0.0, 0.0], [1.0, 1.0])
periodic_x = phx.domain.PeriodicIdentification(box, "x", component=0)
seam = periodic_x.pairing()
v = box.Function("x")(lambda x: x[0] * x[1] + jnp.sin(x[0]))
v_functions = {"v": v}
prepared = phx.enforcement.prepare_periodic_projection(
    v_functions,
    (
        phx.conditions.Periodic("v", seam),
        phx.conditions.Periodic("v", seam, order=1),
    ),
    route="analytic",
)
walls = phx.domain.physical_boundary(box, (periodic_x,))
periodic_program = phx.enforcement.compile(
    v_functions,
    (
        *(
            phx.enforcement.EnforcementSpec(
                phx.conditions.Dirichlet("v", wall, target=0.0)
            )
            for wall in walls
        ),
        phx.enforcement.EnforcementSpec(prepared.condition, realization=prepared),
    ),
)
v_exact = periodic_program.apply(v_functions)["v"]
(periodic_spec,) = periodic_program.realization_specs
preserved = periodic_spec.realization.evidence.preserved
```

Local wall and initial ansätze run first; the seam projection then adds a
polynomial in the identified coordinate times the seam residual of the walled
field, which keeps seam-compatible wall and initial data satisfied. The compiled
realization carries one `PeriodicPreservationRecord` per earlier contract. A wall
on an identified face, interior anchors on the periodic field, or another typed
realization that may write the field without preserving its seams is refused at
compile time. See
[Conditions → Periodic seams](../../guides_conditions.md#periodic-seams).

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
