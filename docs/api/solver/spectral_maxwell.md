# Spectral Maxwell (PSATD)

`phydrax.solver.maxwell.spectral` is the Cartesian pseudo-spectral analytical
time-domain (PSATD) field solver for explicit electromagnetic PIC: standard,
Galilean, and averaged Galilean propagators with constant, linear, or multi-J
currents, three charge-conservation modes, infinite- or finite-order stencils on
collocated or staggered grids, global or guard-cell local transforms, a
split-field PSATD PML, spectral Huygens boxes, and an NCI monitor. The
quasi-cylindrical azimuthal-mode solver advances every `(m, k_⊥, k_z)` mode with
the same propagators on a shared-grid Hankel spectrum, with radial damping, sheet
antennas, Huygens cylinders, and a pinned external FBPIC oracle. See
[Spectral PIC field solvers](../../guides_spectral_pic.md) for the physics,
compatibility rules, and validity domains.

## Plans and prepared solver

::: phydrax.solver.maxwell.spectral.SpectralMaxwellPlan

---

::: phydrax.solver.maxwell.spectral.PreparedSpectralMaxwell

---

::: phydrax.solver.maxwell.spectral.SpectralMaxwellState

---

::: phydrax.solver.maxwell.spectral.SpectralMaxwellSource

---

::: phydrax.solver.maxwell.spectral.SpectralMaxwellDiagnostics

---

::: phydrax.solver.maxwell.spectral.SpectralLocalUpdate

---

::: phydrax.solver.maxwell.spectral.SpectralOperators

---

::: phydrax.solver.PICGalileanGrid

## Selectors

::: phydrax.solver.maxwell.spectral.SpectralMaxwellVariant

---

::: phydrax.solver.maxwell.spectral.SpectralTimeDependency

---

::: phydrax.solver.maxwell.spectral.SpectralChargeConservation

---

::: phydrax.solver.maxwell.spectral.SpectralStencil

---

::: phydrax.solver.maxwell.spectral.SpectralDecomposition

---

::: phydrax.solver.maxwell.spectral.SpectralGrid

---

::: phydrax.solver.maxwell.spectral.SpectralAbsorber

## Stencils

::: phydrax.solver.maxwell.spectral.stencil_coefficients

---

::: phydrax.solver.maxwell.spectral.modified_wavenumber

## Absorber, antennas, and Huygens surfaces

::: phydrax.solver.maxwell.spectral.SpectralPMLPlan

---

::: phydrax.solver.maxwell.spectral.PreparedSpectralPlaneAntenna

---

::: phydrax.solver.maxwell.spectral.SpectralHuygensBoxPlan

---

::: phydrax.solver.maxwell.spectral.PreparedSpectralHuygensBox

## Numerical Cherenkov instability

::: phydrax.solver.maxwell.spectral.SpectralNCIMonitorPlan

---

::: phydrax.solver.maxwell.spectral.SpectralNCISample

---

::: phydrax.solver.maxwell.spectral.NCIGrowthFit

---

::: phydrax.solver.maxwell.spectral.godfrey_vay_growth_rate

---

::: phydrax.solver.maxwell.spectral.GodfreyVayReference

## Quasi-cylindrical solver

::: phydrax.solver.maxwell.spectral.QuasiCylindricalMaxwellPlan

---

::: phydrax.solver.maxwell.spectral.PreparedQuasiCylindricalMaxwell

---

::: phydrax.solver.maxwell.spectral.QuasiCylindricalMaxwellState

---

::: phydrax.solver.maxwell.spectral.QuasiCylindricalSource

---

::: phydrax.solver.maxwell.spectral.QuasiCylindricalMaxwellDiagnostics

---

::: phydrax.solver.maxwell.spectral.QuasiCylindricalAbsorber

---

::: phydrax.solver.maxwell.spectral.RadialDampingPlan

---

::: phydrax.solver.maxwell.spectral.QuasiCylindricalAntennaPlan

---

::: phydrax.solver.maxwell.spectral.QuasiCylindricalHuygensPlan

## FBPIC oracle

::: phydrax.solver.maxwell.spectral.FBPICProvider

---

::: phydrax.solver.maxwell.spectral.FBPICWakefieldResult

---

::: phydrax.solver.maxwell.spectral.fbpic_laser_wakefield
