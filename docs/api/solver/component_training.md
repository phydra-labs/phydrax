# Solver objectives and component training

Public symbols are re-exported from `phydrax.solver`.

A solver objective turns one FIXED prepared solve into a training signal for a
learned component that lives outside it. The component (a component slot such
as a face closure, preconditioner, or feedback policy, or a `ComponentBinding`)
is held in the trained tree; the objective binds it into the solve through the
owner's own refresh or binding path on every evaluation, so the prepared solve
is never trained and never carries the component's parameters.

| Objective | Kind | Route | Trains |
|---|---|---|---|
| `SolverObjective` | `SOLUTION_MAP` | `IMPLICIT` | models and discretization components whose accepted solution is scored |
| `RolloutObjective` | `ROLLOUT` | `UNROLLED` | closures, surrogates, models, and decisions scored along an unrolled rollout |
| `AlgorithmicWorkObjective` | `ALGORITHMIC_WORK` | `UNROLLED` | accelerators scored by the work of a fixed-length native iteration |

Authority comes only from component slots and `ComponentBinding`s: every
PARAMETER leaf must have an owner, and `authority_admits(authority, route, kind)`
decides which objectives may train it. An accelerator changes how fast a solve
converges, never its answer, so a solution-map objective refuses it before
anything is traced; it trains through `AlgorithmicWorkObjective`. A tree that
mixes authorities needs one compatible objective per authority group, and each
objective holds the groups it does not admit fixed.

```python
import jax.numpy as jnp
import jax.random as jr
import optax
import phydrax as phx

work = phx.solver.AlgorithmicWorkObjective(
    solve,                              # FIXED prepared solve
    lambda solve, preconditioner: ...,  # owner binding path
    measure,                            # runs exactly `work` iterations
    work=8,
    objective_id="fgmres-fixed-work",
    cases=right_hand_sides,
)
result = phx.solver.train_components(
    preconditioner, (work,), optimizer=optax.adam(1e-2), steps=100, key=jr.key(0)
)
```

`measure(owner, case)` returns a `SolverCaseResult` (a scalar `value`, or a
`residual` whose half squared norm is the loss) for solution-map and rollout
objectives, and an `AlgorithmicWorkResult` (original-problem residuals before
and after the fixed work, the iterations performed, and primal validity) for
fixed-work objectives. The fixed-work loss is
`log((||r_k|| + floor) / (||r_0|| + floor))` with the stopped precision-aware
floor `eps * ||r_0|| + tiny`: it rewards only the final residual, saturates at
the working precision, and has an exact zero derivative at an exactly zero
residual. A fixed-work case whose reported iterations differ from `work` is an
early exit and fails.

Admission runs before tracing. It also requires every component model to be
deterministic or bound to one `FrozenRealization`, and, for differentiating
optimizers, each trained model's derivative contract to admit the route
(classical `C^1` for implicit solution maps). A component without a JAX
derivative is refused by gradient optimizers with the derivative-free
alternatives named; distribution-evolution optimizers and
`phydrax.uq.posterior_problem_from_solver_objective` with `fit_eki` consume it
explicitly, and nothing switches between the two silently.

Failed cases never contribute a plausible value. With
`accepted_results="reject-attempt"` (the default) any failed case makes the
attempt nonfinite and the training kernel rolls it back; with
`"reduce-support"` failed cases leave the value and the support, and their
derivatives are gated to exact zeros even when their pullbacks are not finite.

`train_components` accepts the `FunctionalSolver.solve` optimizer union: Optax
transformations (line searches re-evaluate the objective), mirror and
Riemannian optimizers, and distribution-evolution methods. KFAC is refused
because its curvature needs FunctionalSolver residual terms. With
`checkpoint`, the committed state is published after every attempt and a
matching checkpoint resumes exactly; identity mismatches fail closed.

::: phydrax.solver.AbstractSolverObjective
    options:
        members:
            - select
            - admit
            - evaluate

---

::: phydrax.solver.SolverObjective

---

::: phydrax.solver.RolloutObjective

---

::: phydrax.solver.AlgorithmicWorkObjective

---

::: phydrax.solver.SolverCaseResult

---

::: phydrax.solver.AlgorithmicWorkResult

---

::: phydrax.solver.algorithmic_work_loss

---

::: phydrax.solver.SolverObjectiveAdmission

---

::: phydrax.solver.SolverObjectiveEvaluation

---

::: phydrax.solver.AcceptedResultPolicy

---

::: phydrax.solver.train_components

---

::: phydrax.solver.ComponentTrainingResult

---

::: phydrax.solver.ComponentOptimizer
