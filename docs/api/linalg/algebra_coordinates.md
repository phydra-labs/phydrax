# Real and algebra coordinate maps

::: phydrax.linalg.AbstractRealCoordinateMap

---

::: phydrax.linalg.RealCoordinateEvidence

---

::: phydrax.linalg.ComplexCartesianCoordinates

---

::: phydrax.linalg.AlgebraArraySpace

---

::: phydrax.linalg.AlgebraCoordinatePlan

---

::: phydrax.linalg.PreparedAlgebraCoordinates

---

::: phydrax.linalg.AlgebraCoefficientPairing

---

::: phydrax.linalg.lift_real_operator_to_algebra

---

::: phydrax.linalg.complexify_real_operator

---

::: phydrax.solver.DiffraxAlgebraStatePolicy

---

::: phydrax.solver.AlgebraStatePackingEvidence

## Continuation

`phydrax.continuation.ContinuationRepresentationPolicy` consumes these same prepared
maps. Public complex or finite-algebra branch states are reconstructed at every
accepted point, while nonlinear correction, arclength geometry, stability, and
nullspace actions operate on the map's real coordinate space. An optional execution
space with the same coordinate structure may supply a problem-specific pairing.
