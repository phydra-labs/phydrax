# Exact PDE trial spaces

The Trefftz API provides fixed finite bases whose represented fields satisfy one
homogeneous PDE, trainable real linear coefficient fields, construction certificates,
and sampled residual audits. See [Exact PDE trial spaces](../../guides_exact_trial_spaces.md)
for scientific assumptions and workflows.
Holomorphic polynomial potentials and their coefficient-space constraints are
documented under [Holomorphic potential fields](holomorphic.md).

`TrialSpaceCertificate` is a `phydrax.AbstractConstructionCertificate` with
capability ID `exact-pde-trial-space`. `Domain.Model` attaches it under
`"trial_space_certificate"`; transforms that do not preserve the trial space
drop it, and hard enforcement refuses fields that carry it.

::: phydrax.equations.trefftz.SimilarityNormalization

---

::: phydrax.equations.trefftz.TrefftzResourceBudget

---

::: phydrax.equations.trefftz.TrefftzResourceEvidence

---

::: phydrax.equations.trefftz.TrialSpaceCertificate

---

::: phydrax.equations.trefftz.TrialSpaceAuditReport

---

::: phydrax.equations.trefftz.AbstractTrefftzBasis

---

::: phydrax.equations.trefftz.HarmonicPolynomialBasis

---

::: phydrax.equations.trefftz.PolyharmonicAlmansiBasis

---

::: phydrax.equations.trefftz.HelmholtzPlaneWaveBasis

---

::: phydrax.equations.trefftz.sample_unit_directions

---

::: phydrax.equations.trefftz.LinearTrefftzField

---

::: phydrax.equations.trefftz.trial_space_certificate

---

::: phydrax.equations.trefftz.audit_trial_space

---

::: phydrax.solver.LinearTrialSpaceResult

---

::: phydrax.solver.solve_linear_trial_space
