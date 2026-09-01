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

`HydroelasticPressureFieldPlan` represents nodal pressure on a common
tetrahedral partition; `extract_hydroelastic_pressure_patch` deterministically
marches the pressure-difference zero set into fixed-capacity patch quadrature
and fails closed on overflow.

Rigid-rigid hydroelastic contact is unsupported rather than replaced by point
contact. At least one side must provide a compliant pressure field.

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

`LubricationContactPlan` combines squeeze-film pressure, cavitation, viscous
shear, and an asperity-contact blend across a minimum-film and transition
thickness. `evaluate_lubrication_contact` reports film, fluid pressure, asperity
fraction, normal/tangential traction, dissipation, and cavitation state.

This local closure is not a global Reynolds PDE solver. It is the resolved-route
constitutive layer that a future film-pressure discretization can drive.

## Distributed ownership

`DistributedContactPartitionPlan` assigns stable vertex and route ownership,
counts cross-rank halo routes, and fails closed on halo-capacity overflow.
`ContactGraphPlan` forms route components for local block and additive coarse
preconditioning. Distribution never changes physical route keys or accepted
interface history.
