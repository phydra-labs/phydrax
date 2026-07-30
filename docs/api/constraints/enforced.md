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
