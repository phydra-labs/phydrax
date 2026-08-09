# Semidiscrete transport and quantization

Semidiscrete transport couples a continuous `DensityTarget` to a finite
`DiscreteMeasureTarget` or `WeightedSampleTarget`. It is a separate mathematical
family from finite Sinkhorn transport: the source stays continuous in the declared
problem, and every source reduction uses the caller-supplied
`IntegrationRealization`.

```py
interval = phx.domain.ScalarInterval(0.0, 1.0, label="x")
source = phx.integration.normalized_density(
    phx.integration.over(interval.component()),
    interval.Function("x")(lambda x: jnp.zeros_like(x)),
)
realization = phx.integration.materialize(
    source,
    phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(32)),
)
target = phx.integration.discrete(
    jnp.array([0.25, 0.75]),
    cx.Field(jnp.array([0.5, 0.5]), dims=("atom",)),
    axes="atom",
    normalized=True,
)
problem = phx.transport.semidiscrete_problem(
    source,
    realization,
    target,
    cost=phx.transport.SquaredEuclideanCost(),
)
result = phx.transport.SemidiscreteSinkhorn(0.05)(problem)
```

The target dual potential determines a soft c-transform on the continuous side.
`result.soft_c_transform(points)` evaluates it without pretending that quadrature or
sampling nodes are source atoms. Reusing the same realization gives deterministic
replay and common-random-number semantics, including when the realization was
materialized from a randomized plan.

A converged discrete dual iteration does **not** make the continuous integral exact.
Every result has `approximate == True`, provenance
`approximation="fixed-integration-realization"`, and separate transport and
integration diagnostics. `result.converged` requires both contracts to succeed.
Normalized densities have physical mass one; unnormalized densities retain the mass
estimated by the realization and must match the finite target's declared physical
mass.

## Problem and solver

::: phydrax.transport.SemidiscreteTransportProblem

---

::: phydrax.transport.semidiscrete_problem

---

::: phydrax.transport.SemidiscreteSinkhorn

## Results and diagnostics

::: phydrax.transport.SemidiscreteTransportResult

---

::: phydrax.transport.SemidiscreteTransportDiagnostics

---

::: phydrax.transport.SemidiscreteIntegrationDiagnostics

---

::: phydrax.transport.SemidiscreteProblemProvenance

---

::: phydrax.transport.SemidiscreteTransportProvenance

## Support optimization

`SemidiscreteQuantizer` composes the differentiable fixed-realization objective with
an ordinary Optax transformation. It does not introduce another optimizer framework.
An optional `support_transform` maps optimizer coordinates into physical sensor,
particle, or collocation locations. Use a smooth domain parameterization rather than
clipping an update:

```py
quantizer = phx.transport.SemidiscreteQuantizer(
    phx.transport.SemidiscreteSinkhorn(0.05, max_iterations=200),
    optax.adam(1e-2),
    num_steps=100,
    support_transform=jax.nn.sigmoid,  # unconstrained -> open unit interval
)
design = quantizer(problem, initial_parameters=jnp.array([-1.0, 1.0]))
```

The quantizer rejects an integration or transport failure before using it as a
scientific training objective. Its result retains the final transport result,
optimizer state, physical support, unconstrained parameters, objective history, and
both terminal statuses.

::: phydrax.transport.SemidiscreteQuantizer

---

::: phydrax.transport.SemidiscreteQuantizationResult

---

::: phydrax.transport.SemidiscreteQuantizationDiagnostics
