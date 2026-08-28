# Particle method qualification

Particle methods carry one of four maturity levels:

```text
experimental
qualified
production
certified
```

Execution success, numerical constraints, evidence claims, and production status
are separate. A finite trajectory does not imply that density, divergence,
pressure complementarity, walls, or free-surface constraints are satisfied.

`ParticleConstraintResiduals` uses dimensionless original-equation metrics,
including relative density and divergence L∞/L₂ errors, pressure
complementarity, wall constraints, and free-surface Dirichlet pressure.
`ParticleQualificationProfile` defines fixed thresholds. IISPH and DFSPH report
these residuals independently of their internal projected-iteration residuals.

`ParticleQualificationResult` can satisfy the production gate only when:

- execution succeeded;
- original constraints passed;
- every requested claim has satisfied evidence;
- maturity is production or certified.

The current advanced IISPH/DFSPH reference artifact remains experimental: its
steps execute successfully, but approximately 1.24% density residual does not
meet the default production profile. This is an intentional false-success guard,
not a benchmark failure.

Qualification artifacts retain method, configuration, source, precision,
backend, and evidence identities. Thresholds must be declared before evaluating
the benchmark.
