# Reduced-order models

## Empirical interpolation

`prepare_empirical_interpolation` derives deterministic interpolation nodes from an
accepted real linear POD artifact. Preparation records conditioning and maximum basis
reproduction error before exposing the node-to-state reconstruction operator.

::: phydrax.rom.EmpiricalInterpolationPlan

::: phydrax.rom.EmpiricalInterpolationArtifact

::: phydrax.rom.PreparedEmpiricalInterpolation

::: phydrax.rom.prepare_empirical_interpolation

See [Multi-fidelity workflows](fidelity.md) for using executable reduced models as
explicit fidelity levels and the
[gravitational-wave inference guide](../guides_gravitational_wave_inference.md#qualified-acceleration)
for reduced-order quadrature.
