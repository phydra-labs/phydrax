# Multiphysics and patch contact

Mechanical contact changes transport, surface evolution, and unresolved
microscale response. Phydrax models those effects as explicit interface closure;
it does not hide them inside a normal-force law.

## Hydroelastic patches

`HydroelasticMaterialPlan` defines pressure modulus, compliant slab thickness,
normal dissipation, friction, and velocity regularization. A compliant pressure
trace may be generated from local compression or supplied by a volumetric
pressure field. `evaluate_hydroelastic_contact` integrates normal and frictional
traction over `ContactInterfacePlan` quadrature and reports patch area,
resultant force/moment, pressure positivity, dissipation, and action-reaction.

`HydroelasticPressureFieldPlan` binds a canonical affine tetrahedral
`CellMesh`; pressure is supplied separately by
`HydroelasticPressureFieldState`. `HydroelasticPatchExtractionPlan.prepare`
uses certified host tetrahedron intersections to fix nonmatching overlap
polytopes. Pure-JAX evaluation cuts the affine pressure difference, returns
plus/minus cell IDs and interpolation weights, and reports predicate, tie, area,
pressure-balance, capacity, and derivative evidence.

Rigid-rigid hydroelastic contact is unsupported rather than replaced by point
contact. At least one side must provide a compliant pressure field.

## Device broad phase

`LBVHContactSearchPlan` builds Morton ordering from swept point/edge/face AABBs
and deterministically packs the existing candidate batches. Node, depth, visit,
duplicate-code, stack, visit, and output budgets are explicit evidence.
Exhaustion fails closed; dense search remains the candidate-equivalence
authority. “LBVH” is a bounded algorithm/resource contract, not a universal
device-optimality claim.

## Rough contact

`HomogenizedRoughContactPlan` provides a differentiable pressure-separation and
real-contact-area closure for macroscale interfaces. `PeriodicRoughContactPlan`
solves a periodic nonnegative-pressure half-space problem using FFT compliance
and projected iterations. The result reports pressure, elastic displacement,
gap, complementarity, load, and contact area.

`hertz_sphere_half_space` supplies the analytical Hertz reference used to
qualify smooth patch pressure and force.

## Thermal, electrical, and mass transfer

`CoupledContactTransportLaw` consumes three oriented interface jumps:

```text
temperature
+ electric potential
+ chemical or hydraulic potential
```

Pressure and gap control the effective conductance. Frictional dissipation is
returned separately as generated heat. `assemble_contact_fluxes` partitions
that heat between participants while enforcing equal/opposite conductive heat,
electrical current, and mass flux.

## Wear and cohesive evolution

`FrictionWearEvolutionLaw` advances accumulated slip, Archard-style wear depth,
cohesive damage, rate-state history, and film thickness. Damage and wear are
irreversible. Remeshing uses `ContactStateTransferPlan`, which applies affine
parent interpolation while preserving the maximum inherited damage and wear.

## Lubricated contact

`LubricationContactPlan` remains the local squeeze/shear/asperity closure.
`ReynoldsFilmPlan` adds a connected P1 surface-film owner with declared pressure
references and a fixed-iteration projected Reynolds cavitation solve. It
assembles variable `h³/(12 μ)` diffusion, squeeze and tangential transport,
interpolates pressure to `ContactInterfacePlan`, and uses the canonical
equal/opposite traction assembly.

`ReynoldsFilmEvidence` reports PDE residual, complementarity, flux balance,
load, dissipation, minimum film, active-set margin, and solver status. A
nonpositive film, unreferenced component, active-set tie, or failed solve is
never replaced by pointwise squeeze pressure. This is a Reynolds variational
inequality, not an Elrod–Adams mass-fraction model.
## Distributed ownership

`DistributedContactPartitionPlan` assigns stable vertex and route ownership,
counts cross-rank halo routes, and fails closed on halo-capacity overflow.
`ContactGraphPlan` forms route components for local block and additive coarse
preconditioning. Distribution never changes physical route keys or accepted
interface history.
