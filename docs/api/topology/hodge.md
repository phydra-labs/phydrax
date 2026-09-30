# Exact topology and numerical Hodge kernels

The bridge computes exact rational Betti dimensions on the same compact active complex
used for numerical Hodge validation. Rank agreement is necessary but not sufficient:
metric orthonormality, kernel residuals, and next-mode evidence are checked separately.

::: phydrax.exterior.HodgeCohomologyReport

---

::: phydrax.exterior.validate_harmonic_cohomology

---

::: phydrax.exterior.harmonic_kernel_certificate

The returned `KernelCertificate` certifies the compact numerical Hodge operator. It
does not choose a `NullspacePolicy`; compatibility projection and gauge behavior remain
physics-specific solver decisions.

`prepare_harmonic_class_frame` measures periods against exact homology generators
and retains a numerical period defect and owning solve evidence. A valid dual
period identity is not replaced by a fabricated matrix or an explicit inverse.
Relative frames use the exact `CellComplexPair` basis and exclude boundary DOFs.

::: phydrax.exterior.HarmonicClassFrame

::: phydrax.exterior.prepare_harmonic_class_frame

