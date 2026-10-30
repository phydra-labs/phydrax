# Neurofluid modeling

::: phydrax.applications.neurofluid

## Mixed-dimensional discretization

::: phydrax.discretization.MetricNetworkPlan

::: phydrax.discretization.EmbeddedMeasureTransferPlan

::: phydrax.discretization.CircleAverageKernel

::: phydrax.discretization.BallAverageKernel

## Mixed-dimensional equations

::: phydrax.equations.BulkDGTransportPlan

::: phydrax.equations.NetworkTransportPlan

::: phydrax.equations.PermeabilityExchangePlan

::: phydrax.equations.ReservoirCouplingPlan

::: phydrax.equations.MixedDimensionalTransportPlan

## Compartment meshing

::: phydrax.geometry.CompartmentComplex

::: phydrax.geometry.CompartmentInterfaceDefinition

::: phydrax.geometry.CompartmentMeshingSource

::: phydrax.meshing.CellMeshingResult

::: phydrax.meshing.NativeMeshingProvider

::: phydrax.meshing.VolumeMeshingSpec

::: phydrax.meshing.RegionSeed

::: phydrax.meshing.MeshZone

::: phydrax.meshing.RegionRole

## H(div) finite elements

H(div) references use `form_element("tetrahedron", 2, order, family=...,
twist="twisted", proxy="flux")`; trimmed and full select RT and BDM.

::: phydrax.discretization.fem.form_element

::: phydrax.equations.fem.HDivNormalBoundaryCondition


::: phydrax.equations.fem.HDivStokesPlan
