# Reactive particle process operations

Process operations use fixed-capacity arrays and explicit event commits. They never resize a JAX state or reuse an active slot.

## Complete particle templates

`ReactiveParticleTemplatePlan` defines a complete insertion state: mass, radius, material ID, translational and angular kinematics, and internal conversion fields. Spherical inertia is derived from mass and radius. A template must initialize every coupled subsystem needed by the inserted particle.

`ReactiveParticleTemplateDistributionPlan` selects templates from a declared probability vector using a JAX key. Sampling is deterministic for the same key, pool state, and event index.

## Fixed-pool insertion

`ParticleInsertionPlan` and `insert_reactive_particles` select statically allocated particle slots whose dynamic body activity is false. A successful insertion atomically writes:

- dynamic body properties;
- rigid kinematics;
- internal energy and species inventory;
- porosity, surface area, scale, and reaction fronts;
- activity and event counters.

The insertion rejects when capacity is exhausted, a sampled template is inadmissible, or any conservation residual exceeds tolerance. Pool slots intended for future insertion must be present in the static particle discretization even though their runtime body properties start inactive.

## Regions, residence, and flow

`ParticleRegionPlan` evaluates a fixed geometric inclusion mask. `ParticleResidenceState` accumulates time only while a dynamic particle is active and inside the region.

`MassFlowSurfacePlan` detects directed segment crossings of a plane and sums the masses of active crossing particles. Crossing direction follows the declared surface normal. The result is an extensive mass increment, not a rate; divide by the observation interval only at the reporting layer.

## Removal and deactivation

Removal plans select active slots from explicit geometric or state guards. Accepted removal zeros dynamic contributions and updates activity without changing static owner maps or IDs. Inventory leaving the computational system is recorded in the process ledger.

## Fragmentation

`ThermochemicalFragmentationPlan` activates preallocated children from one parent. The commit checks mass, linear momentum, angular momentum, energy, and per-schema species residuals. Insufficient free children or invalid split fractions reject the event atomically.

## Neighbor caches

Insertion, removal, radius change, and fragmentation are topology events for particle neighborhoods. Runtime body activity and radius are passed to DEM geometry. A cached relation may only be reused while displacement and radius-growth margins remain inside its declared skin certificate.

## Current scope

The process layer is nondistributed. Dynamic memory allocation, cross-rank migration, event queues with unbounded cardinality, and implicit population balances are intentionally absent.
