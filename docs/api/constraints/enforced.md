# Enforced constraint ansätze

These helpers construct ansätze that satisfy constraints by construction.

For composition/ordering (multi-field dependencies, applying several enforced constraints, etc.),
see [Enforced constraint pipelines](../solver/enforced_constraints.md).

!!! warning
    Compatibility with coord-separable grids:

    - `enforce_neumann`, `enforce_robin`, `enforce_traction`, and `enforce_sommerfeld` rely on geometry
      boundary normals \(n(x)\) (via \(\partial/\partial n\)) and therefore do **not**
      support coord-separable (tuple-of-axes) evaluation. Phydrax raises a `ValueError`
      if you try to evaluate these ansätze on a `CoordSeparableBatch`.
    - `enforce_dirichlet`, `enforce_initial`, `enforce_blend`,
      `enforce_graph_values`, and `enforce_cochain_values` do not require
      boundary normals and can be used in spectral/FNO-style or graph-batch
      evaluations.

!!! warning
    A `Boundary()` component restricted by `where` or `where_all` is not a
    valid input to a direct geometry hard enforcer. Its signed-distance gate
    vanishes on the full boundary and would enforce outside the requested
    subset. Phydrax rejects this case. Construct each piece's ansatz against an
    unfiltered `Boundary()` component, then associate those ansätze with the
    filtered components passed to `enforce_blend`.

::: phydrax.constraints.enforce_dirichlet

---

::: phydrax.constraints.enforce_neumann

---

::: phydrax.constraints.enforce_robin

---

::: phydrax.constraints.enforce_sommerfeld

---

::: phydrax.constraints.enforce_traction

---

::: phydrax.constraints.enforce_initial

---

::: phydrax.constraints.enforce_blend

---

::: phydrax.constraints.enforce_graph_values

---

::: phydrax.constraints.enforce_cochain_values
