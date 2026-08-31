# Score-based diffusion transport

Score transport converts a trained time-dependent score field into either a stochastic
reverse-time diffusion or a deterministic probability-flow system. It reuses the
canonical differential solver and continuous-flow density contracts; it does not
introduce a second integration stack.

## Reverse-time SDE

For forward diffusion

```text
dX_t = f(X_t, t) dt + g(t) dW_t
```

and score `s(x, t)`, Phydrax integrates the increasing reverse coordinate `r = T - t`:

```text
dY_r = [-f(Y_r, T-r) + g(T-r)^2 s(Y_r, T-r)] dr
       + g(T-r) dW_r.
```

`ReverseDiffusion` separates random-object materialization from numerical solving:

1. `realize(key, sample_shape)` draws prefix-stable terminal states, one fixed score
   evaluation key, and a global `WienerRealization`;
2. `solve(realization, save_times=...)` advances those exact terminal states;
3. `sample_with_diagnostics` composes both operations;
4. `sample` returns final states only and rejects any failed path.

The diagonal diffusion coefficient lowers to `lineax.DiagonalLinearOperator`; no dense
identity factor is allocated. Solver status, temporal precision, Wiener identity, and
backend result remain on the returned `DifferentialSolution`.

::: phydrax.transport.ReverseDiffusion

---

::: phydrax.transport.ReverseDiffusionRealization

---

::: phydrax.transport.ReverseDiffusionResult

## Probability flow

The reverse-coordinate probability-flow field is

```text
dY_r/dr = -f(Y_r, T-r) + 0.5 g(T-r)^2 s(Y_r, T-r).
```

`probability_flow_system` returns an ordinary autonomous `ContinuousSystem` with no
external input layout. Compose it explicitly with `DiffraxEvolution`,
`ContinuousTransport`, and `ContinuousFlowLaw`:

```python
system = phx.transport.probability_flow_system(
    process,
    score,
    state_layout=phx.dynamics.StateLayout(process.state_shape),
    score_id="trained-score",
)
evolution = phx.solver.DiffraxEvolution(system)
transport = phx.transport.ContinuousTransport(
    terminal_reference.law,
    evolution,
    source_coordinate=0.0,
    target_coordinate=process.terminal_time,
)
law = phx.transport.ContinuousFlowLaw(transport)
```

Exact finite-dimensional traces and uncertainty-bearing Hutchinson estimates retain
the existing continuous-flow restrictions and diagnostics.

::: phydrax.transport.ProbabilityFlowVectorField

---

::: phydrax.transport.probability_flow_system

## General Itô reversal and guidance

`general_reverse_diffusion_problem` lowers matrix or state-dependent diffusion factors
as operator-valued Wiener terms. `general_probability_flow_system` includes the exact
covariance-divergence correction. Named contexts are filtered to each score or
guidance dependency; classifier-free composition requires its unconditional field to
be the base score.

::: phydrax.transport.ScoreContext

---

::: phydrax.transport.GuidedScoreField

---

::: phydrax.transport.TimeConditionedLikelihoodGuidance

---

::: phydrax.transport.PotentialGuidance

---

::: phydrax.transport.ClassifierFreeGuidance

---

::: phydrax.transport.general_reverse_diffusion_problem

---

::: phydrax.transport.general_probability_flow_system

The original `ReverseDiffusion` path remains the memory-linear diagonal, real-vector
specialization. It rejects nontrivial state geometry and complex packing. Every path
retains explicit terminal-reference semantics, and no reverse step is projected or
clipped implicitly.

See [Advanced generative transport](generative_expansion.md) for the distinct
discrete, subspace, field, manifold, complex, path, and scientific-composition
contracts.
