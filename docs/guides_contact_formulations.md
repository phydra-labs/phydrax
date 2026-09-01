# Contact formulations and closure

Phydrax separates contact geometry, local interface kinematics, constitutive
closure, mathematical enforcement, and nonlinear solution. These layers are not
interchangeable contact modes.

## Participants and kinematics

`AbstractContactParticipant` maps one mechanics state to world-space collision
vertices and maps contact forces back to generalized forces. Linear finite
element, rod, shell, static, rigid, articulated, and meshfree participants can
share one `ContactParticipantScene` while retaining independent state spaces.

`ContactKinematicsEpoch` lowers candidate stencils to law-independent local
quantities: gap, relative normal velocity, tangential velocity/slip, normal and
tangent frame, quadrature weight, material pair, stable route key, and branch
margins.

## Contact material pairs

`ContactMaterialPairTable` stores explicit pair parameters. No hidden material
mixing rule or fallback is applied. Each pair declares normal stiffness,
static/dynamic friction, restitution, adhesion energy, thermal/electrical
conductance, wear coefficient, hardness, roughness, and availability masks.

## Closure components

`ContactClosurePlan` composes:

- one `AbstractNormalContactLaw`;
- one `AbstractTangentialContactLaw`;
- one `AbstractInterfaceEvolutionLaw`;
- one `AbstractContactTransportLaw`.

The closure returns local tractions, potential density, dissipation, consistent
normal tangent, candidate history, and transport fluxes. A formulation checks
closure capabilities before use.

Available normal laws include physical clamped barriers, prefiltered geometric
inverse-power barriers, compliant Hunt-Crossley-style response, and cohesive
adhesive barriers. Tangential laws include frictionless, regularized
static/dynamic Coulomb, anisotropic elliptic Coulomb, and rate-state friction.
`FrictionWearEvolutionLaw` advances slip, adhesion damage, wear, state-variable,
and film-thickness histories transactionally.

## Smooth primal formulation

`assemble_smooth_contact` converts a potential-capable closure to balanced
surface forces. The stencil coefficients are the exact local Jacobian transpose,
so participant pullbacks preserve force and moment. Smooth formulations are
appropriate for incremental-potential equilibrium and dynamics.

## Cone contact and impact

`ContactConeProgram` represents normal and tangential impulses in local contact
coordinates with a matrix-free-compatible effective mass and compliance. The
solver projects each route onto its Coulomb cone and reports projected residual,
complementarity, cone, impulse-sign, dissipation, and convergence evidence.

`assemble_contact_impulses` maps local impulses to participant surface impulses.
`RollingSpinningResistancePlan` adds bounded rolling and spinning impulses with
an explicit dissipation certificate.

## Mortar and Nitsche

`ContactInterfacePlan` is a fixed quadrature map between nonmatching plus/minus
nodal traces. Its affine trace maps and transpose traction assembly conserve
interface action exactly.

- `MortarContactPlan` updates normal and tangential multipliers through an
  augmented-Lagrangian projection.
- `OneSidedNitscheContactPlan` uses one consistency stress trace.
- `UnbiasedNitscheContactPlan` averages both traces and is invariant to the
  declared participant order when the traces are exchanged.
- `MeshTiePlan` enforces vector continuity and can release routes beyond a
  declared tension limit.

## Guarantee semantics

Every collision backend declares one ordered guarantee:

```text
UNAVAILABLE
HEURISTIC
PRACTICAL_CONSERVATIVE
ENCLOSURE_CONSERVATIVE
ANALYTIC_CONSERVATIVE
ROUNDING_CERTIFIED
```

The ordinary inclusion backend provides an enclosure-conservative guarantee.
`CertifiedAABBCCDPlan` uses outward-rounded swept primitive boxes and reports
unresolved intervals at their lower endpoint, yielding a deliberately
conservative certified backend. A requested guarantee stronger than a backend
can provide rejects during preparation.

## State ownership

`ContactRouteState` is keyed by physical routes and stores stick/slip mode,
accumulated slip, adhesion damage, wear, rate-state variable, and film thickness.
Route remapping, proxy refinement, repartition, and remeshing produce candidate
state with explicit transition evidence. Only an accepted physical solve commits
candidate interface state.
