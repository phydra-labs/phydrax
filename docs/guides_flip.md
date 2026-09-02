# Free-surface FLIP

Phydrax FLIP is a fixed-population particle–MAC method over the existing structured tensor-grid,
particle-splat, MAC-boundary, and linear-solver substrates. It is distinct from MPM/APIC, SPH
free-surface detection, unstructured VOF/PLIC, and resolved marker-force coupling.

## Transfer

`FLIPParticleTransferPlan` prepares one existing particle splat on cell centers and one on every
MAC face orientation. It does not create a new router.

For reference density `rho`, cell parcel volume is deposited as `mass / rho`. Every face transfer
deposits a fused extensive payload:

```text
face_mass     = sum_p N_fp mass_p
face_momentum = sum_p N_fp mass_p velocity_p
face_velocity = face_momentum / face_mass
```

The extensive numerator balances are reported before normalization. Unsupported face velocity is
zero with a false support mask. P2G and G2P use the same prepared B-spline relation.

## Atmospheric pressure projection

`MACFreeSurfaceProjectionPlan` adds the one pressure closure absent from the full-grid MAC stack: a
runtime liquid mask with atmospheric pressure exactly zero in air. It reuses
`PreparedMACOperators`, `PreparedMACBoundaryPlan`, and `phydrax.linalg`.

Air rows are identity rows; liquid rows use the compatible weighted pressure action. When every
cell is liquid, the same action activates a zero-mean pressure gauge. The result reports liquid/air
counts, divergence, pressure residual, air-pressure defect, kinetic-energy change, nested linear
solve evidence, and fail-closed status.

This is not a particle SDF or ghost-fluid fraction. The initial liquid mask is a first-order
threshold on raw deposited parcel volume divided by cell volume.

## Step ordering

`compile_flip_problem` combines a `FLIPProblemIR`, prepared transfer,
`MACFreeSurfaceProjectionPlan`, and `FLIPMethodPlan` into `CompiledFLIPProblem`.

One step:

1. deposit particle volume and face mass/momentum;
2. classify liquid cells;
3. extrapolate supported pre-grid velocity for a fixed number of layers;
4. save the exact pre-operation grid velocity;
5. apply body acceleration and physical MAC boundaries;
6. project with atmospheric pressure;
7. extrapolate post-projection velocity;
8. gather the post velocity and exact grid increment at unchanged particle positions;
9. update particle velocity;
10. midpoint-advect positions;
11. commit the complete candidate only when every transfer, projection, CFL, support, and finite
    check passes.

`FLIPMethodPlan.pic_fraction` uses the convention

```text
v_new = (1 - beta) [v_old + G(u_new - u_old)] + beta G(u_new)
```

where beta zero is pure FLIP and beta one is pure PIC. The value is required explicitly; no 95/5
default is inferred from graphics demonstrations.

Grid extrapolation is reconstruction for G2P/advection only. It never enters mass or momentum
balance evidence.

## Qualified static solids

An optional sharp-solid bundle binds one accepted fluid-volume/open-aperture
realization, solid-aware P2G/G2P relations, sharp pressure projection, wall velocity,
and particle collision under the same geometry identity. Particle stencil weights are
restricted to accepted fluid support and renormalized; the matched relation is used in
both transfer directions and reports partition-of-unity and support-loss evidence.

The liquid mask is intersected with active fluid cells. Pressure, divergence,
extrapolation, and grid updates exclude closed support. Geometry, transfer, projection,
collision, and the particle/grid candidate commit or roll back atomically.

Initial sharp-solid support is static and fixed topology. Moving geometry requires the
accepted sharp-epoch refresh/GCL lifecycle; reseeding remains separate.

## Existing MAC and MPM boundaries

FLIP does not call `CompiledMACIncompressibleDynamics.convection`; particle motion already carries
advection. Calling both would double-advect momentum.

Initial FLIP is inviscid. The existing `MACHelmholtzSolvePlan`, IMEX/SBDF2 methods,
`MACVariableDensityPlan`, distributed MAC projection, ALE/remesh, and `MACMarkerTransferPlan` remain
independent qualified capabilities. They are not duplicated or silently inherited.

Current MPM already owns collocated nodal APIC mechanics, deformation, stress and constitutive
history. FLIP uses staggered face transfer and a grid-delta update; MPM runtime state is never used.

## Differentiation and limits

Gradients are branchwise for fixed particle routes, liquid mask, extrapolation wavefront, solver
route, and accepted step. Classification or route transitions invalidate the local derivative
program.

The baseline scope is constant-density, inviscid, fixed particles, periodic or
stationary closed boundaries, first-order liquid classification, and single-device
execution. Qualified fixed-topology static solids are optional. Moving sharp solids,
reseed integration, variable-density FLIP, staggered APIC–FLIP, free-surface
viscosity, and distributed particle ownership remain unsupported.
