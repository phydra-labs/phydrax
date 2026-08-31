# Maxwell solvers

Phydrax provides complementary Maxwell substrates.

- `phydrax.solver.CompatibleMaxwellPlan` advances compatible cochain D/B state and
  owns the general time-domain Maxwell lifecycle.
- `phydrax.solver.FrequencyMaxwellOperator` solves a prepared cochain curl-curl
  problem.
- `phydrax.solver.maxwell.fourier_modal` solves transversely periodic layered
  frequency-domain problems with boundary-field propagation.

## Fourier-modal lifecycle

::: phydrax.solver.maxwell.fourier_modal.FourierModalMaxwellProblem

::: phydrax.solver.maxwell.fourier_modal.FourierModalSolvePolicy

::: phydrax.solver.maxwell.fourier_modal.FourierModalSolvePlan

::: phydrax.solver.maxwell.fourier_modal.PreparedFourierModalMaxwell

::: phydrax.solver.maxwell.fourier_modal.plan_fourier_modal_maxwell

::: phydrax.solver.maxwell.fourier_modal.prepare_fourier_modal_maxwell

::: phydrax.solver.maxwell.fourier_modal.refresh_fourier_modal_maxwell

::: phydrax.solver.maxwell.fourier_modal.solve_fourier_modal_maxwell

::: phydrax.solver.maxwell.fourier_modal.PreparedFourierModalCaseBatch

::: phydrax.solver.maxwell.fourier_modal.FourierModalCaseBatchResult

::: phydrax.solver.maxwell.fourier_modal.prepare_brillouin_zone_maxwell

::: phydrax.solver.maxwell.fourier_modal.solve_fourier_modal_case_batch

## Materials, layers, and ports

::: phydrax.solver.maxwell.fourier_modal.FrequencyMaxwellMaterial

::: phydrax.solver.maxwell.fourier_modal.HomogeneousMaxwellPort

::: phydrax.solver.maxwell.fourier_modal.FourierModalLayer

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

::: phydrax.solver.maxwell.fourier_modal.JonesDirectFramePlan

::: phydrax.solver.maxwell.fourier_modal.BoundaryCascadePolicy

::: phydrax.solver.maxwell.fourier_modal.ModalPropagationPolicy

## Excitations and observables

::: phydrax.solver.maxwell.fourier_modal.FourierModalExcitation

::: phydrax.solver.maxwell.fourier_modal.plane_wave_excitation

::: phydrax.solver.maxwell.fourier_modal.fields_in_layer

::: phydrax.solver.maxwell.fourier_modal.diffraction_order_far_field

::: phydrax.solver.maxwell.fourier_modal.FourierModalSolveResult

::: phydrax.solver.maxwell.fourier_modal.FourierModalDiagnostics
