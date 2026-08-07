# Enforced constraint ansätze

These helpers construct ansätze that satisfy constraints by construction.

For composition/ordering (multi-field dependencies, applying several enforced constraints, etc.),
see [Enforced constraint pipelines](../solver/enforced_constraints.md).

!!! warning
    Compatibility with axis-based grids:

    - `enforce_neumann`, `enforce_robin`, `enforce_traction`, and `enforce_sommerfeld` rely on geometry
      boundary normals \(n(x)\) (via \(\partial/\partial n\)) and therefore do **not**
      support axis-based grid evaluation. PhydraX raises a `ValueError` if you
      try to evaluate these ansätze on a `GridBatch`.
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

!!! info
    Geometry value and derivative ansätze intentionally use different units.
    `enforce_dirichlet` uses a dimensionless enforcement gate that is zero on the
    boundary and order one in the interior. Neumann, Robin, traction, and Sommerfeld
    use the dimensional `boundary_ansatz_factor`, whose outward boundary derivative
    is one, and its gradient as a smooth off-boundary normal extension. This gradient
    agrees with the outward unit normal on regular boundary points but need not remain
    unit length in the interior. Public normal calculations continue to use the
    geometry ADF and normal provider. The compact gate remains an explicit option for
    Dirichlet and preservation overlays.

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
