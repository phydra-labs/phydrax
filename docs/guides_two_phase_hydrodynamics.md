# Incompressible two-phase VOF hydrodynamics

`IncompressibleTwoPhaseVOFPlan` is a separate fixed-grid one-fluid product for
interface topology changes, breaking, air cavities, contact, and impact. It does not
reuse graph eta as interface state.

## Authoritative state

`TwoPhaseVOFState` stores:

- liquid volume content `alpha * cell_volume`;
- density-weighted extensive face momentum;
- phase-confined scalar contents;
- an optional level-set auxiliary used only for geometry.

VOF alpha remains interface authority. Level set and PLIC are derived/method geometry.

## Material and transport

`TwoPhaseMaterialPlan` declares liquid/gas density and viscosity, surface tension, and
contact angle.

One face flux bundle provides:

- total volume flux;
- liquid volume flux;
- gas volume flux;
- phase-consistent density flux;
- momentum flux using the same mass flux.

The update limits donor liquid content before transport. Alpha is not independently
clipped after the update.

## PLIC and CLSVOF

`PreparedIncompressibleTwoPhaseVOF.plic` builds normals, plane offsets, mixed-cell mask,
and reconstruction residual. The current implementation supports two- and
three-dimensional structured grids with fixed topology.

Level set is advected/reconciled toward alpha-derived geometry. Reinitialization and
volume-correction evidence are ledgered; level set never overwrites alpha authority.

## Variable-density projection and capillarity

The pressure projection uses updated mixture density and the identical coefficient in
pressure action and velocity correction.

Capillary pressure derives from the discrete interface-area functional and is mapped
through the compatible pressure-gradient path. Surface energy and capillary work are
reported separately.

## Qualified static solid geometry

An optional qualified sharp-geometry realization replaces full Cartesian cell volumes
and face measures with accepted fluid volumes and open apertures throughout content,
phase flux, momentum flux, inverse momentum, and pressure projection. Geometry,
projection, PLIC, and transfer IDs must agree; failed geometry or pressure evidence
rolls back the complete candidate state.

The initial contract is static and fixed topology. A cell cut simultaneously by the
solid boundary and the gas-liquid PLIC interface is rejected because the current PLIC
reconstruction is defined on the full Cartesian cell, not the clipped fluid polytope.
No contact angle is inferred in that case.

## Moving bodies, contact, and surface piercing

`TwoPhaseMovingBodyPlan` provides a fixed-radius moving immersed target with identified
center, velocity, and penalty; body work is carried in the two-phase ledger.
`TwoPhaseCapabilityEventPlan` evaluates the canonical VOF/PLIC state for body contact,
surface piercing, boundary wetting/drying, moving contact lines, overturning, and
breaking/topology-change routes. It returns per-cell masks, a deterministic event
bitset, contact-angle residual, and derivative availability. The event product does
not pretend that a topology change is smooth; callers either begin a new fixed-topology
epoch or hand off to the mapped rigid/hydroelastic contact product.

## Topology and ledger

`TwoPhaseTopologyEvidence` reports liquid/gas volume, mixed cells, interface measure,
changed-cell mask, and event proxy.

`ConservativeTwoPhaseRemeshPlan` accepts preflighted cell-overlap volumes and
face-transfer matrices between two prepared VOF products. It transfers extensive
liquid/scalar content and face momentum atomically, reconstructs the auxiliary level
set from transferred alpha, and reports source/target coverage and conservation
defects. Connectivity selection and transfer construction are host-static; topology
changes return `derivative_available=False`.

`TwoPhaseVOFLedger` records phase volumes, momentum, kinetic/gravitational/surface
energy, viscosity, capillary/body work, limiter, CLSVOF repair, reinitialization,
pressure/divergence, topology events, and total residual.

## Limits

- PLIC reconstruction is fixed-grid, with branches and contact-angle routes fixed inside
  an execution epoch;
- qualified sharp solids are initially static and fixed topology;
- cells cut by both solid and gas-liquid interfaces fail closed;
- interface, wet/dry, moving-contact, piercing, breaking, contact, and remesh events are
  nondifferentiable topology boundaries;
- no AMR/reflux or distributed phase transport;
- no subgrid air-entrainment model;
- penalty bodies do not replace the mapped monolithic rigid/hydroelastic contact
  product;
- calibrated breaking/impact envelopes require their own qualification.
