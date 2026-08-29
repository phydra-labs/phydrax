# Exact topology and numerical Hodge kernels

The bridge computes exact rational Betti dimensions on the same compact active complex
used for numerical Hodge validation. Rank agreement is necessary but not sufficient:
metric orthonormality, kernel residuals, and next-mode evidence are checked separately.

::: phydrax.graph.HodgeHomologyReport

---

::: phydrax.graph.validate_hodge_homology

---

::: phydrax.graph.cochain_harmonic_kernel_certificate

The returned `KernelCertificate` certifies the compact numerical Hodge operator. It
does not choose a `NullspacePolicy`; compatibility projection and gauge behavior remain
physics-specific solver decisions.
