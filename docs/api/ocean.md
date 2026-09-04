# Ocean process modeling

The Cartesian rigid-lid plan binds algebraic MAC LES or prognostic KSGS, never
both, with complete named temperature/salinity scalar SGS. KSGS supports static
and buoyant routes directly; low-Re is admitted only with caller-supplied true
no-slip momentum walls and resolved cell-center wall distance. Dynamic KSGS
requires a periodic-uniform grid and is incompatible with the bounded vertical
ocean geometry. See the
[LES guide](../guides_large_eddy_simulation.md#ocean-and-prognostic-ksgs).

::: phydrax.applications.ocean.OceanAxisConvention

---

::: phydrax.applications.ocean.LinearSeawaterReference

---

::: phydrax.applications.ocean.CartesianBoussinesqOceanPlan

---

::: phydrax.applications.ocean.PreparedCartesianBoussinesqOcean

---

::: phydrax.applications.ocean.OceanStateView

---

::: phydrax.equations.MACAlgebraicLESPlan

---

::: phydrax.discretization.MACScalarSGSPlan

---

::: phydrax.equations.StaticKSGSPlan

---

::: phydrax.equations.BuoyancyKSGSPlan

---

::: phydrax.applications.ocean.OceanBoussinesqContinuationState

---

::: phydrax.applications.ocean.OceanBoussinesqSSPRK33Method

---

::: phydrax.applications.ocean.OceanDiagnosticView

---

::: phydrax.applications.ocean.ocean_diagnostic_view

---

::: phydrax.applications.ocean.write_ocean_checkpoint

---

::: phydrax.applications.ocean.read_ocean_checkpoint

---

::: phydrax.applications.ocean.write_ocean_output

---

::: phydrax.discretization.PreparedMACOceanForcing

---

::: phydrax.discretization.MACOceanForcingEvidence
