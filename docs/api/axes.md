# Native axes

`phydrax.axes` carries semantic array identity without embedding coordinate values in
JAX PyTree metadata. Numerical support remains owned by domains, discretizations,
graphs, and integration plans.

The core types are:

- `AxisKey`: namespaced identity;
- `Axis`: finite size, labels, and optional support identity;
- `AxisRef`: one source/target or primal/dual use of an axis;
- `UnboundAxis`: a positional axis introduced without scientific identity;
- `AxisLayout`: ordered array schema;
- `AxisArray`: dynamic JAX data with a static layout;
- alignment, reduction, and contraction plans.

`AxisArray` is deliberately not a universal NumPy replacement. Kernels unwrap arrays
against an expected layout, execute ordinary JAX, and rewrap explicit output layouts.
Axis mapping delegates to JAX vectorization and the existing Phydrax execution
substrates rather than owning a second runtime.

Arithmetic, `broadcast_like`, prepared alignment, and `cmap` align exact `AxisRef`
identities, including the complete `AxisKey` scope and the reference role. Equal
display names do not make axes interchangeable. Subset layouts broadcast directly.
`order_as` accepts an `AxisRef` for exact selection; a bare string is accepted only
when exactly one bound axis has that name and otherwise fails as absent or ambiguous.
Disjoint non-sampling semantic axes require the explicit `phydrax.axes.outer`
operation; sampling and coordinate-grid axes retain their declared Cartesian-product
behavior. `AxisContractionPlan` is the canonical factorized-field contraction plan,
while `PairwiseAxisContractionPlan` represents one direct two-array contraction.

::: phydrax.axes.AxisKey

---

::: phydrax.axes.Axis

---

::: phydrax.axes.AxisRef

---

::: phydrax.axes.AxisLayout

---

::: phydrax.axes.AxisArray

---

::: phydrax.axes.AxisAlignmentPlan

---

::: phydrax.axes.AxisReductionPlan

---

::: phydrax.axes.AxisContractionPlan

---

::: phydrax.axes.PairwiseAxisContractionPlan
