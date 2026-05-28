# Functional solver

`FunctionalSolver` is the main entry point for turning a set of fields and constraints
into a differentiable objective.

For a conceptual overview (loss evaluation, enforced pipelines, training loop behavior), see
[Guides → Solvers and training](../../guides_solver.md).

!!! note
    Key notes:

    - `loss(...)` evaluates the total objective at the current parameters.
    - `ansatz_functions()` returns fields after applying enforced pipelines (if configured).
    - `partition_functions()` exposes the trainable/non-trainable state split used by `solve(...)`.
    - `solve(...)` updates parameters inside `functions` using Optax or evosax optimizers.
    - `solve(..., tensorboard_log_dir=...)` writes TensorBoard scalar logs.
    - `save_onnx("u", ...)` exports one named ansatz function for deployment.
    - Discrete data constraints report data-fit diagnostics alongside their loss.

## Typical usage

```python
import jax.random as jr
import optax
import phydrax as phx

geom = phx.domain.Interval1d(0.0, 1.0)

model = phx.nn.MLP(in_size=1, out_size="scalar", width_size=16, depth=2, key=jr.key(0))
u = geom.Model("x")(model)

structure = phx.domain.ProductStructure((("x",),))
constraint = phx.constraints.ContinuousPointwiseInteriorConstraint(
    "u",
    geom,
    operator=lambda f: f,
    num_points=128,
    structure=structure,
    reduction="mean",
)

solver = phx.solver.FunctionalSolver(functions={"u": u}, constraints=[constraint])
loss0 = solver.loss(key=jr.key(0))
solver = solver.solve(num_iter=20, optim=optax.adam(1e-3), seed=0)
loss1 = solver.loss(key=jr.key(1))
```

::: phydrax.solver.FunctionalSolver
    options:
        members:
            - __init__
            - ansatz_functions
            - __getitem__
            - partition_functions
            - trainable_functions
            - loss
            - solve
            - save_onnx
