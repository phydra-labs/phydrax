# Optimization

`phydrax.optim` owns domain-neutral optimization configurations shared by semantic
workflow adapters. The configuration is intentionally separate from geometry,
posterior, or solver state: each adapter supplies its own finite search space,
objective, reconstruction, validity, and result contract.

## Bounded differential evolution

`DifferentialEvolutionSearch` configures a fixed-dimensional, bounded population
search. It supports `"best1bin"` and `"rand1bin"`, requires a population of at least
four, and uses a typed `phydrax.sampling` reference design for initialization. Latin
hypercube is the default; scrambled Sobol and the other supported designs are
available when their design guarantees fit the requested population size.

The root PRNG key determines initialization and every generation. The current point
is inserted as population member zero. Mutation overshoot is repeatedly reflected
into the closed box, the whole population is evaluated with `jax.vmap`, and
non-finite objective values are counted and treated as positive infinity for
selection. Population convergence means finite objective dispersion satisfies

`standard_deviation <= absolute_tolerance + relative_tolerance * abs(mean)`.

It is not evidence that every basin was covered or that the global optimum was
proved. Exact generation, objective-evaluation, and invalid-evaluation counts are
reported by the consuming adapter.

Use the domain adapter matching the objective contract:

- `DesignConstraintSystem.search(...)` for compiled geometry design states;
- `phydrax.uq.search_map(...)` for full-data posterior densities in unconstrained
  posterior-position coordinates.

The configuration is not a generic callback-based public optimizer. New workflows
should expose explicit bounds, reconstruction, validity, and result semantics rather
than bypassing their domain contract.

::: phydrax.optim.DifferentialEvolutionSearch
    options:
        members:
            - __init__
