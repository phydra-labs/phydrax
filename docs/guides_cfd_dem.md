# CFD–DEM coupling

## Conservative transfer

`MeshSplatTarget` plus `MeshCompactKernelSplatAssignment` builds the canonical
fixed-capacity normalized particle-to-cell route on a declared `CellMesh`.
Gather and extensive deposition use one adjoint-paired stencil. Particle volume,
momentum, force, impulse, and heat remain extensive until division by the
receiving cell measure. Empty support, route overflow, or failed normalization
is explicit evidence; inactive runtime population slots contribute exactly zero.

## Unresolved coupling

`UnresolvedCFDEMCouplingPlan` combines prepared DEM, conservative transfer, porosity bounds, and one typed closure. `StokesDragPlan` is valid only below its declared Reynolds limit. It returns particle force and the exact opposite fluid momentum source. Porosity or regime violation fails rather than extrapolating.

`advance_cfd_dem_window` requires an integer number of DEM substeps per fluid macro-step. Hydrodynamic impulse is accumulated over the accepted window; fluid and DEM candidates commit atomically or both roll back.

## Resolved immersed-boundary penalty coupling

`MACPenaltyIBCFDEMCouplingPlan` binds fixed material marker quadrature to
rigid-sphere DEM ownership. `MACMarkerTransferPlan` uses local cubic tensor
B-spline routes on the actual staggered MAC face layouts. Gather and spread are
material/face-measure Hilbert adjoints; force, torque, virtual work, support,
and fixed-body reaction are explicit evidence.

`advance_mac_penalty_ib_cfd_dem_window` subcycles DEM contact dynamics,
accumulates trapezoidal hydrodynamic impulses, inserts the penalty source before
pressure projection, and commits fluid and particle states atomically.
`IBPenaltyPlan` reports numerical validity separately from slip qualification;
strict slip qualification remains the default acceptance policy. This path is
not the exact pressure-plus-marker multiplier method described in the dedicated
[immersed-boundary guide](guides_immersed_boundary.md).

## Reactive heat and species coupling

`ParticleContinuumExchangePlan` samples fluid temperature and species concentration and deposits the exact opposite extensive heat and species sources through the same conservative transfer. `ParticleContactExchangePlan` handles reciprocal contact heat independently. `ReciprocalPairRadiationPlan` adds reciprocal particle and optional wall radiation.

`ReactiveCFDDEMCouplingPlan` combines prepared DEM, radial particle conversion, continuum exchange, optional contact exchange, morphology, and radiation. `advance_reactive_cfd_dem_window` uses Strang or fixed-iteration strong coupling and commits fluid, DEM, conversion, and morphology candidates atomically.

Distributed ownership, turbulence modulation, added mass, and monolithic fluid–particle Newton solves remain unsupported. Contact-scale near-gap lubrication is available through the DEM cohesion channel; it is not a bulk hydrodynamic closure.
