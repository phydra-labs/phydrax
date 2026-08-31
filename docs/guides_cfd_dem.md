# CFD–DEM coupling

## Conservative transfer

`ConservativeParticleGridTransferPlan` builds a fixed-capacity normalized particle-to-cell relation. Gather and deposit use the same weights. Particle volume, momentum, force, impulse, and heat are deposited as extensive content; division by cell volume occurs only in the receiving fluid discretization.

A relation fails when support is empty, cannot be normalized, or exceeds cells-per-particle capacity. Inactive particles contribute exactly zero.

## Unresolved coupling

`UnresolvedCFDEMCouplingPlan` combines prepared DEM, conservative transfer, porosity bounds, and one typed closure. `StokesDragPlan` is valid only below its declared Reynolds limit. It returns particle force and the exact opposite fluid momentum source. Porosity or regime violation fails rather than extrapolating.

`advance_cfd_dem_window` requires an integer number of DEM substeps per fluid macro-step. Hydrodynamic impulse is accumulated over the accepted window; fluid and DEM candidates commit atomically or both roll back.

## Resolved immersed boundary

`ResolvedIBCFDEMCouplingPlan` uses fixed markers, shared interpolation/spreading weights, and a no-slip penalty constraint. Marker forces reduce to body force and torque while the exact opposite source is spread to fluid cells. The discrete interpolation/spreading work-adjoint residual is an acceptance criterion.

## Reactive heat and species coupling

`ParticleContinuumExchangePlan` samples fluid temperature and species concentration and deposits the exact opposite extensive heat and species sources through the same conservative transfer. `ParticleContactExchangePlan` handles reciprocal contact heat independently. `ReciprocalPairRadiationPlan` adds reciprocal particle and optional wall radiation.

`ReactiveCFDDEMCouplingPlan` combines prepared DEM, radial particle conversion, continuum exchange, optional contact exchange, morphology, and radiation. `advance_reactive_cfd_dem_window` uses Strang or fixed-iteration strong coupling and commits fluid, DEM, conversion, and morphology candidates atomically.

Distributed ownership, turbulence modulation, added mass, and monolithic fluid–particle Newton solves remain unsupported. Contact-scale near-gap lubrication is available through the DEM cohesion channel; it is not a bulk hydrodynamic closure.
