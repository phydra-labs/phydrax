# Boosted-frame PIC

`phydrax.solver.BoostedFramePlan` runs a lab-frame laser–plasma or undulator
interaction in a Lorentz-boosted frame on Galilean PSATD
(`phydrax.solver.maxwell.spectral`): exact particle, field, and antenna transforms,
gather-only boosted external fields, a mandatory numerical-Cherenkov guard,
back-transformed lab snapshots in a bounded ring, lab-frame tracks for trajectory
radiation, and restart. See
[Boosted-frame PIC](../../guides_spectral_pic.md#boosted-frame-pic) for the physics,
the validity domain, and measured evidence.

## Frame and transforms

::: phydrax.solver.BoostedFramePlan

---

::: phydrax.solver.BoostedParticles

---

::: phydrax.solver.BoostedExternalField

## Prepared run

::: phydrax.solver.PreparedBoostedFrame

---

::: phydrax.solver.BoostedFrameState

---

::: phydrax.solver.BoostedFrameStepResult

---

::: phydrax.solver.BoostedFrameEvidence

## Back-transformed snapshots

::: phydrax.solver.BoostedSnapshotPlan

---

::: phydrax.solver.BoostedSnapshotState

---

::: phydrax.solver.BoostedParticleSnapshotBuffer

---

::: phydrax.solver.BoostedLabFieldSnapshot

---

::: phydrax.solver.BoostedLabParticleSnapshot
