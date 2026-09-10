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
