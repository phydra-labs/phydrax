# CFD–DEM coupling

## Conservative transfer

`ConservativeParticleGridTransferPlan` builds a fixed-capacity normalized particle-to-cell relation. Gather and deposit use the same weights. Particle volume, momentum, force, impulse, and heat are deposited as extensive content; division by cell volume occurs only in the receiving fluid discretization.

A relation fails when support is empty, cannot be normalized, or exceeds cells-per-particle capacity. Inactive particles contribute exactly zero.

## Unresolved coupling

`UnresolvedCFDEMCouplingPlan` combines prepared DEM, conservative transfer, porosity bounds, and one typed closure. `StokesDragPlan` is valid only below its declared Reynolds limit. It returns particle force and the exact opposite fluid momentum source. Porosity or regime violation fails rather than extrapolating.

`advance_cfd_dem_window` requires an integer number of DEM substeps per fluid macro-step. Hydrodynamic impulse is accumulated over the accepted window; fluid and DEM candidates commit atomically or both roll back.

## Resolved immersed boundary

`ResolvedIBCFDEMCouplingPlan` uses fixed markers, shared interpolation/spreading weights, and a no-slip penalty constraint. Marker forces reduce to body force and torque while the exact opposite source is spread to fluid cells. The discrete interpolation/spreading work-adjoint residual is an acceptance criterion.

## Thermal coupling

`ThermalCFDEMCouplingPlan` samples fluid temperature and deposits the opposite heat source through the same conservative relation. Particle and cell heat capacities define an explicit stability restriction. Mechanical drag work is not automatically converted to heat.

Distributed ownership, turbulence modulation, lubrication, added mass, phase change, and radiation remain unsupported.
