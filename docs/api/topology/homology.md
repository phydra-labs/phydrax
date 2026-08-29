# Exact homology

Homology calculations use compact active coordinates and exact host arithmetic.
Coefficient fields are never inferred from floating dtypes.

::: phydrax.topology.compute_homology

---

::: phydrax.topology.compute_betti_dimensions

---

::: phydrax.topology.HomologyResult

---

::: phydrax.topology.HomologyDegreeResult

---

::: phydrax.topology.BettiDimensionResult

---

::: phydrax.topology.FiniteFieldBasis

Prime-field results can retain cycle and cocycle representatives. Exact rational
analysis currently returns dimensions only; it is used when comparison with a real
Hodge kernel requires characteristic-zero ranks.
