# Maxwell solvers

Phydrax provides complementary Maxwell substrates.

- `phydrax.solver.CompatibleMaxwellPlan` advances compatible cochain D/B state and
  owns the general time-domain Maxwell lifecycle.
- `phydrax.solver.FrequencyMaxwellOperator` solves a prepared cochain curl-curl
  problem.
- `phydrax.solver.maxwell.fourier_modal` solves transversely periodic layered
  frequency-domain problems with boundary-field propagation.

## Compatible time-domain lifecycle

::: phydrax.solver.maxwell.MaxwellCochainLayout

---

::: phydrax.solver.maxwell.MaxwellResourcePolicy

---

::: phydrax.solver.maxwell.MaxwellMagneticConstraintPolicy

---

::: phydrax.solver.maxwell.CompatibleMaxwellPlan

---

::: phydrax.solver.maxwell.PreparedCompatibleMaxwell

---

::: phydrax.solver.maxwell.CompatibleMaxwellRefreshSpec

---

::: phydrax.solver.maxwell.refresh_compatible_maxwell

---

::: phydrax.solver.maxwell.solve_compatible_maxwell

## Sources, observers, and ports

::: phydrax.solver.maxwell.MaxwellElectricCurrentSourcePlan

---

::: phydrax.solver.maxwell.MaxwellPairedCurrentSourcePlan

---

::: phydrax.solver.maxwell.MaxwellHuygensSourcePlan

---


::: phydrax.solver.maxwell.MaxwellModePortPlan

---

::: phydrax.solver.maxwell.DFTObserverPlan

## Harmonic and material evidence

::: phydrax.solver.maxwell.MaxwellHarmonicDefectReport

---

::: phydrax.solver.maxwell.compatible_maxwell_harmonic_defect

---

::: phydrax.solver.maxwell.MaxwellScalarMaterialAssemblyPolicy

---

::: phydrax.solver.maxwell.assemble_scalar_maxwell_material

## Independent case batching

::: phydrax.solver.maxwell.PreparedCompatibleMaxwellCaseBatch

---

::: phydrax.solver.maxwell.prepare_compatible_maxwell_case_batch

---

::: phydrax.solver.maxwell.solve_compatible_maxwell_case_batch

## Reduced-dimensional compatible Maxwell

::: phydrax.solver.CompatibleMaxwell1DPlan

---

::: phydrax.solver.CompatibleMaxwell1DState

---

::: phydrax.solver.CompatibleMaxwell2DPlan

---

::: phydrax.solver.CompatibleMaxwell2DState

---

::: phydrax.solver.PreparedReducedMaxwellCPML

## Fourier-modal lifecycle

::: phydrax.solver.maxwell.fourier_modal.FourierModalMaxwellProblem

::: phydrax.solver.maxwell.fourier_modal.FourierModalSolvePolicy

::: phydrax.solver.maxwell.fourier_modal.FourierModalSolvePlan

::: phydrax.solver.maxwell.fourier_modal.PreparedFourierModalMaxwell

::: phydrax.solver.maxwell.fourier_modal.plan_fourier_modal_maxwell

::: phydrax.solver.maxwell.fourier_modal.prepare_fourier_modal_maxwell

::: phydrax.solver.maxwell.fourier_modal.refresh_fourier_modal_maxwell

::: phydrax.solver.maxwell.fourier_modal.solve_fourier_modal_maxwell

---

::: phydrax.solver.maxwell.fourier_modal.fourier_modal_numeric_revision

---

::: phydrax.solver.maxwell.fourier_modal.fourier_modal_physical_state_digest

---

::: phydrax.solver.maxwell.fourier_modal.fourier_modal_physical_stack_digest

---

::: phydrax.solver.maxwell.fourier_modal.require_fourier_modal_numeric_revision

::: phydrax.solver.maxwell.fourier_modal.PreparedFourierModalCaseBatch

::: phydrax.solver.maxwell.fourier_modal.FourierModalCaseBatchResult

::: phydrax.solver.maxwell.fourier_modal.prepare_brillouin_zone_maxwell

::: phydrax.solver.maxwell.fourier_modal.solve_fourier_modal_case_batch

## Materials, layers, and ports

::: phydrax.solver.maxwell.fourier_modal.FrequencyMaxwellMaterial

::: phydrax.solver.maxwell.fourier_modal.HomogeneousMaxwellPort
::: phydrax.solver.maxwell.fourier_modal.PeriodicMaxwellPort


::: phydrax.solver.maxwell.fourier_modal.FourierModalLayer
::: phydrax.solver.maxwell.fourier_modal.ContinuousFourierModalLayer

::: phydrax.solver.maxwell.fourier_modal.ContinuousZIntegrationPolicy

::: phydrax.solver.maxwell.fourier_modal.LateralTransformationOpticsPMLPlan

::: phydrax.solver.maxwell.fourier_modal.transform_fourier_modal_material


::: phydrax.solver.maxwell.fourier_modal.FourierModalSourcePlane

## Geometry rasterization

::: phydrax.solver.maxwell.fourier_modal.FourierModalRasterizationPolicy

::: phydrax.solver.maxwell.fourier_modal.FourierModalRasterizationPlan

::: phydrax.solver.maxwell.fourier_modal.FourierModalRasterizationResult

::: phydrax.solver.maxwell.fourier_modal.FourierModalRasterizationEvidence

::: phydrax.solver.maxwell.fourier_modal.rasterize_fourier_modal_material

## Factorization and propagation

::: phydrax.solver.maxwell.fourier_modal.DirectFourierFactorizationPlan

::: phydrax.solver.maxwell.fourier_modal.InverseFourierFactorizationPlan

::: phydrax.solver.maxwell.fourier_modal.VectorFourierFactorizationPlan

::: phydrax.solver.maxwell.fourier_modal.AnalyticInterfaceFramePlan

::: phydrax.solver.maxwell.fourier_modal.JonesDirectFramePlan

::: phydrax.solver.maxwell.fourier_modal.BoundaryCascadePolicy

::: phydrax.solver.maxwell.fourier_modal.ModalPropagationPolicy

## Excitations and observables

::: phydrax.solver.maxwell.fourier_modal.FourierModalExcitation

::: phydrax.solver.maxwell.fourier_modal.plane_wave_excitation
::: phydrax.solver.maxwell.fourier_modal.port_mode_excitation


::: phydrax.solver.maxwell.fourier_modal.fields_in_layer

::: phydrax.solver.maxwell.fourier_modal.diffraction_order_far_field
::: phydrax.solver.maxwell.fourier_modal.FiniteApertureFarFieldPlan

::: phydrax.solver.maxwell.fourier_modal.finite_aperture_far_field

::: phydrax.solver.maxwell.fourier_modal.FourierModalHarmonicAdaptationPolicy

::: phydrax.solver.maxwell.fourier_modal.solve_adaptive_fourier_modal_case


::: phydrax.solver.maxwell.fourier_modal.FourierModalSolveResult

::: phydrax.solver.maxwell.fourier_modal.FourierModalDiagnostics

## Directional power and independent physical loss

::: phydrax.solver.maxwell.fourier_modal.FourierModalLossPolicy

---

::: phydrax.solver.maxwell.fourier_modal.FourierModalLossEvidence

---

::: phydrax.solver.maxwell.fourier_modal.FourierModalLossStatus

---

::: phydrax.solver.maxwell.fourier_modal.evaluate_fourier_modal_loss

---

::: phydrax.solver.maxwell.fourier_modal.FourierModalLossConvergenceEvidence

---

::: phydrax.solver.maxwell.fourier_modal.assess_fourier_modal_loss_convergence

## Equivalent-slab retrieval and local-isotropic qualification

::: phydrax.solver.maxwell.fourier_modal.MaxwellModalSweep

---

::: phydrax.solver.maxwell.fourier_modal.prepare_maxwell_modal_sweep

---

::: phydrax.solver.maxwell.fourier_modal.EquivalentSlabRetrievalPlan

---

::: phydrax.solver.maxwell.fourier_modal.EquivalentSlabRetrieval

---

::: phydrax.solver.maxwell.fourier_modal.EquivalentSlabRetrievalStatus

---

::: phydrax.solver.maxwell.fourier_modal.retrieve_equivalent_slab

---

::: phydrax.solver.maxwell.fourier_modal.LocalIsotropicQualificationPolicy

---

::: phydrax.solver.maxwell.fourier_modal.LocalIsotropicMediumQualification

---

::: phydrax.solver.maxwell.fourier_modal.LocalIsotropicQualificationStatus

---

::: phydrax.solver.maxwell.fourier_modal.qualify_local_isotropic_medium
