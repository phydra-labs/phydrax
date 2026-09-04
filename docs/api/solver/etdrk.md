# Exponential time-differencing Runge--Kutta

::: phydrax.solver.ETDRKMethod

---

::: phydrax.solver.PreparedETDRKMethod

---

::: phydrax.solver.LESStabilityGuardedETDRKMethod

---

::: phydrax.solver.PreparedLESStabilityGuardedETDRKMethod

The LES guard accepts only compiled periodic static algebraic LES, reuses the first
complete equation stage, and rejects a requested step above its safety-scaled
current-state restriction. It is not adaptive. See
[Large-eddy simulation](../../guides_large_eddy_simulation.md#static-periodic-time-integration).

Dynamic periodic LES instead uses
`PreparedPeriodicDynamicETDRKMethod`, which evaluates the compiled dynamic closure,
enforces its current-state restriction, and commits Lagrangian continuation only
with the accepted velocity.

::: phydrax.applications.incompressible_flow.PreparedPeriodicDynamicETDRKMethod

---

::: phydrax.solver.solve_etdrk

---

::: phydrax.linalg.matrix_phi1_action

---

::: phydrax.linalg.matrix_phi2_action

---

::: phydrax.linalg.matrix_phi3_action
