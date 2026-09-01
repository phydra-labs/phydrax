# Variational substrate

`phydrax.variational` declares representation-independent real scalar functionals.
It owns physical fields, local jet requirements, semantic integration regions, signed
terms, and ordered evaluation evidence. Integration, finite-element compilation,
optimization, and specialized estimators remain in their owning packages.

## Core objects

::: phydrax.variational.FieldJetSpec

---

::: phydrax.variational.LocalFieldJet

---

::: phydrax.variational.LocalGeometry

---

::: phydrax.variational.FunctionalContext

---

::: phydrax.variational.LocalIntegralTerm

---

::: phydrax.variational.Functional

---

::: phydrax.variational.FunctionalEvaluation

## Local conventions

A local density has the signature:

```python
density(fields, geometry, context)
```

`fields` maps semantic names to `LocalFieldJet`. Gradients use trailing layout
`value_shape + (spatial_dimension,)`; unrequested jets are absent rather than
computed and filled with zeros. `geometry.points` uses the integration leading
axes. Exterior terms receive `geometry.normal` only when the
`LocalIntegralTerm` declares `normal=True`. The density returns one real scalar
per point.

`Functional.variable_fields` names physical variables. Other declared fields are
coefficients. A first variation is a covector; converting it to a gradient requires
an explicit pairing, Riesz map, or metric.

## DomainFunction binding

`phydrax.terms.bind_functional` binds semantic fields to `DomainFunction`s and
regions to typed `IntegrationSource` values:

```python
source = phx.integration.fixed(
    phx.integration.materialize(target, plan, key=key)
)
terms = phx.terms.bind_functional(
    functional,
    {"u": displacement},
    {"body": source},
    geometry_variables={"body": "x"},
)
solver = phx.solver.FunctionalSolver(functions={"u": displacement}, terms=terms)
```

The solver derivative is the pullback through the parameterized trial field. It is
not the ambient physical-space functional derivative. `pullback_fields` controls
which bound field parameters receive derivatives.

## Prepared-local discretization binding

Use `finite_element_form_from_functional` when a form object is needed, or
compile directly on finite-element, hp, or another
`AbstractPreparedLocalDiscretization` support:

```python
compiled = phx.equations.compile_finite_element_functional(
    functional,
    discretization,
    fields={"u": "u"},
    regions={"body": None},
)
```

For a compiled functional:

- `compiled.potential(state)` evaluates the realized discrete scalar;
- `compiled.residual(state)` is its dual-valued discrete first variation;
- `compiled.linearization_operator(state)` is the matrix-free Hessian action;
- `compiled.potential_evaluation(state)` retains ordered term values;
- `compiled.as_minimization_problem()` selects minimization explicitly;
- `compiled.as_nonlinear_problem()` selects stationarity explicitly.

The portable compiler consumes method-neutral local interpolation, gradient,
geometry, and transpose actions. Isogeometric bindings therefore use the same
physical density and discrete-variation contract; select their supported
`local_kernel="sum_factorized"` execution policy. `CellEnergyAction` and
`FiniteElementFunctional` remain explicit representation-bound adapters for
specialized callbacks that are not portable `Functional` declarations.

Essential constraints use `u = Pz + g`; the reduced residual is the dual pullback
`P* δJ(Pz + g)`. Value, residual, and Hessian use the same quadrature and density.
This is discretize-then-vary semantics. The API does not claim equivalence to a
separately derived continuous Euler–Lagrange equation.

Cell terms support field values and first gradients, including coupled
multi-field potentials. Prepared-local exterior regions support value jets,
points, and provider-supplied outward normals; the legacy FE fallback supports
two-dimensional polygonal boundaries. Exterior gradients and interior-facet
potentials are rejected during compilation.

## Scope boundaries

The shared name “variational” does not imply one algorithm:

- VMC and TDVP remain solver-owned stochastic/projection methods;
- reverse-KL inference and SING remain in `phydrax.uq`;
- general variational inequalities remain in `phydrax.nonlinear` because an
  operator inequality need not have a scalar potential;
- nonsmooth subdifferentials and proximal maps remain in `phydrax.optim`;
- arbitrary weak, Petrov–Galerkin, flux, and stabilization forms remain in
  `phydrax.equations`.
