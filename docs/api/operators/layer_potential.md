# Boundary layer potentials

See [Boundary layer potentials](../../guides_boundary_layer_potentials.md) for the
PDE-certificate and operator-approximation guarantee split.

::: phydrax.operators.AbstractLayerKernel

---

::: phydrax.operators.BoundaryPanelization2D

---

::: phydrax.operators.LayerPotentialTargetReport

---

::: phydrax.operators.LayerDiscretizationReport

---

::: phydrax.operators.LayerEvaluationPlan2D

---

::: phydrax.operators.LayerEvaluationReport

---

::: phydrax.operators.LayerEvaluationResult

---

::: phydrax.operators.evaluate_layer_potential

::: phydrax.operators.LaplaceLayerKernel2D

---

::: phydrax.operators.LaplaceLayerPotential2D

---

::: phydrax.operators.double_layer_principal_value_matrix

---

::: phydrax.solver.InteriorLaplaceDirichletResult

---

::: phydrax.solver.solve_interior_laplace_dirichlet_2d

::: phydrax.operators.BoundaryCornerTopology2D

---

::: phydrax.operators.BoundaryPanelPartition2D

---

::: phydrax.operators.PanelInteractionReport2D

---

::: phydrax.operators.BoundaryOperatorAssemblyReport

---

::: phydrax.operators.HelmholtzLayerKernel2D

---

::: phydrax.operators.HelmholtzLayerPotential2D

---

::: phydrax.operators.HelmholtzCombinedField2D

The following planar kernels provide bounded off-surface reconstruction. They
do not silently claim singular self-quadrature or an exterior compatibility
condition that has not been prepared.

::: phydrax.operators.ModifiedHelmholtzLayerKernel2D

---

::: phydrax.operators.ModifiedHelmholtzLayerPotential2D

---

::: phydrax.operators.ElasticityLayerKernel2D

---

::: phydrax.operators.ElasticityLayerPotential2D

---

::: phydrax.operators.StokesLayerKernel2D

---

::: phydrax.operators.StokesLayerPotential2D

---

`PeriodicScalarSpectralKernel2D` is explicitly a finite reciprocal-lattice
provider. Its cutoff, mode count, resonance margin, and outer-shell indicator
remain visible evidence; it is not labeled an exact Ewald sum.

::: phydrax.operators.PeriodicScalarSpectralKernel2D

---

::: phydrax.operators.PeriodicScalarLayerPotential2D

---

::: phydrax.solver.ExteriorHelmholtzDirichletResult2D

---

::: phydrax.solver.solve_exterior_helmholtz_dirichlet_2d

---

::: phydrax.operators.QBXEvaluation2D

---

::: phydrax.operators.SurfacePanelization3D

---

::: phydrax.operators.SurfaceTargetReport3D

---

::: phydrax.operators.LaplaceLayerKernel3D

---

::: phydrax.operators.LaplaceLayerPotential3D

---

::: phydrax.operators.evaluate_laplace_layer_3d

---

::: phydrax.operators.LaplaceSingleLayerDP0GalerkinPolicy3D

---

::: phydrax.operators.LaplaceSingleLayerDP0AssemblyReport3D

---

::: phydrax.operators.LaplaceSingleLayerDP0Galerkin3D

---

::: phydrax.operators.prepare_laplace_single_layer_dp0_3d

---

The two-dimensional Galerkin owner publishes fixed-geometry derivative
capabilities: `ScalarLaplaceGalerkin2D.derivative_capability` admits the
`dirichlet`, `conormal`, and `far_field_constant` inputs of its linear actions,
and `PreparedExteriorLaplaceDirichlet2D.derivative_capability` admits the
Dirichlet data of the bordered exterior solve under `rhs-only`.
`ExteriorLaplaceDirichletResult2D.derivative_valid` is reported separately from
`accepted`. Geometry, kernel, quadrature, and field-target derivatives are
refused.

::: phydrax.operators.ClosedPolygonalCurve2D

---

::: phydrax.operators.CurveTraversal2D

---

::: phydrax.operators.ScalarTraceConvention2D

---

::: phydrax.operators.ScalarTraceSide2D

---

::: phydrax.operators.BoundaryTraceSpace2D

---

::: phydrax.operators.BoundaryTraceRepresentation2D

---

::: phydrax.operators.ScalarBoundarySpaces2D

---

::: phydrax.operators.ScalarLaplaceGalerkinPolicy2D

---

::: phydrax.operators.ScalarLaplaceGalerkinReport2D

---

::: phydrax.operators.ScalarLaplaceGalerkin2D

---

::: phydrax.operators.prepare_scalar_laplace_galerkin_2d

---

::: phydrax.operators.ScalarLaplaceFieldEvaluation2D

---

::: phydrax.operators.PreparedExteriorLaplaceDirichlet2D

---

::: phydrax.operators.ExteriorFarField2D

---

::: phydrax.operators.ExteriorLaplaceDirichletResult2D

---

::: phydrax.operators.prepare_exterior_laplace_dirichlet_2d

---

::: phydrax.operators.solve_exterior_laplace_dirichlet_2d

---

::: phydrax.operators.BoundaryTraceProjection2D

---

::: phydrax.operators.TraceProjectionResult2D

---

::: phydrax.operators.prepare_boundary_trace_projection_2d

---

::: phydrax.operators.AbstractLayerBackend

---

::: phydrax.operators.DirectNearFarReferenceBackend2D

::: phydrax.operators.QBXEvaluation3D

---

::: phydrax.operators.evaluate_qbx_3d

---

::: phydrax.operators.HelmholtzLayerKernel3D

---

::: phydrax.operators.HelmholtzLayerPotential3D

---

::: phydrax.operators.HelmholtzCombinedField3D

---

::: phydrax.solver.ExteriorHelmholtzDirichletResult3D

---

::: phydrax.solver.solve_exterior_helmholtz_dirichlet_3d

---

::: phydrax.operators.RCIPPreconditioner2D

::: phydrax.operators.LaplaceTreecodeBackend2D

---

::: phydrax.operators.LaplaceTreecodeEvaluation2D

::: phydrax.operators.LaplaceFMMBackend2D

---

::: phydrax.operators.LaplaceFMMEvaluation2D

---

::: phydrax.operators.GlobalQBXFMMEvaluation2D

---

::: phydrax.operators.evaluate_global_qbx_fmm_2d
