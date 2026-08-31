# Force-density structural design API

The public surface is exported from `phydrax.applications.solid_mechanics`. See
[Force-density form-finding](../guides_force_density.md) for equations, workflows,
and scientific limitations.

## Topology and equilibrium

::: phydrax.applications.solid_mechanics.ForceDensityStructure

::: phydrax.applications.solid_mechanics.ForceDensityProblem

::: phydrax.applications.solid_mechanics.ForceDensityInputs

::: phydrax.applications.solid_mechanics.ForceDensityState

::: phydrax.applications.solid_mechanics.ForceDensityResult

::: phydrax.applications.solid_mechanics.plan_force_density

::: phydrax.applications.solid_mechanics.prepare_force_density

::: phydrax.applications.solid_mechanics.refresh_force_density

::: phydrax.applications.solid_mechanics.solve_force_density

::: phydrax.applications.solid_mechanics.solve_force_density_batch

## Loads

::: phydrax.applications.solid_mechanics.ForceDensityLoadState

::: phydrax.applications.solid_mechanics.FixedNodalLoadModel

::: phydrax.applications.solid_mechanics.EdgeLineLoadModel

::: phydrax.applications.solid_mechanics.ReferenceMemberSelfWeightModel

::: phydrax.applications.solid_mechanics.SurfaceTractionLoadModel

::: phydrax.applications.solid_mechanics.SurfacePressureLoadModel

::: phydrax.applications.solid_mechanics.PneumaticPressureLoadModel

::: phydrax.applications.solid_mechanics.CompositeForceDensityLoadModel

## Inverse and constrained design

::: phydrax.applications.solid_mechanics.ForceDensityDesignProblem

::: phydrax.applications.solid_mechanics.ForceDensityDesignConstraint

::: phydrax.applications.solid_mechanics.solve_force_density_design

::: phydrax.applications.solid_mechanics.compile_structured_force_density_design

::: phydrax.applications.solid_mechanics.solve_structured_force_density_design

## Mechanisms, stability, and continuation

::: phydrax.applications.solid_mechanics.analyze_force_density_mechanisms

::: phydrax.applications.solid_mechanics.analyze_force_density_tangent_stability

::: phydrax.applications.solid_mechanics.force_density_continuation_problem

## Observables

Pure observables include member direction and angle residuals, scaled target and
uniformity residuals, point/line/plane/segment geometry, reaction direction,
collinearity, graph fairness, T3/Q4 areas, Q4 planarity and rectangularity, and
signed-distance target geometry. They return arrays and compose directly in
ordinary PhydraX objectives and constraints.
