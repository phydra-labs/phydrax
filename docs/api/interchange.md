# Host interchange and inspection

Phydrax inspection records are explicit host-side projections of accepted or candidate
scientific products. They are not numerical fields, JAX transformation inputs,
checkpoints, or persistence formats. Product-owned adapters must preserve support,
layout, representation, validity, status, and provenance identities and must report
semantic loss through `AdapterReport`.

`HostInspectionField` stores read-only host values and an explicit validity mask at one
cell, face, vertex, particle, marker, or point location. It never infers units,
resamples data, or converts between point values, averages, moments, and coefficients.
`HostInspectionFrame` groups uniquely named fields for exactly one candidate or
accepted state. `HostInspectionConversion` couples a frame to its interchange report.

Candidate and accepted frames are separate objects. A renderer or asynchronous
publisher may consume them, but renderer handles and serialization methods are outside
the inspection contract.

::: phydrax.interchange.AdapterReport

---

::: phydrax.interchange.HostInspectionField

---

::: phydrax.interchange.HostInspectionFrame

---

::: phydrax.interchange.HostInspectionConversion

## Optional OpticStudio boundary

The OpticStudio adapter is host-only, lazily imports the optional ZOSPy package,
opens only owned standalone sessions, and rejects unsupported sequential features
before mutating the vendor system.

::: phydrax.interchange.opticstudio.OpticStudioBackend

---

::: phydrax.interchange.opticstudio.OpticStudioAnalysisRequest

---

::: phydrax.interchange.opticstudio.export_sequential_to_opticstudio

---

::: phydrax.interchange.opticstudio.run_opticstudio_analysis

## Optional energy execution

These host-only boundaries require explicit runtime/version/license provenance.
They do not execute inside JAX transformations or provide surrogate derivatives.
FMI support is synchronous FMI 2.0 Co-Simulation; HELICS support is value
federation. See [the energy interchange guide](../guides_energy_interchange.md)
for lifecycle, capabilities, unsupported features, and ownership.

::: phydrax.interchange.energy_runtime
    options:
      members: true

::: phydrax.interchange.fmi
    options:
      members: true

::: phydrax.interchange.helics
    options:
      members: true
