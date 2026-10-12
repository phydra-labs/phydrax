# Superquadric DEM

Phydrax resolves three-dimensional rigid superquadrics through the shared compliant DEM compiler and stepper. Particle neighborhoods, stable contact keys, contact laws, deterministic reduction, and rigid-body integration are common to spheres, clumps, superquadrics, and convex bodies; only contact geometry is shape-specific.

## Shape contract

`phydrax.geometry.Superquadric` is an analytic geometry source. Semi-axes are positive. The two exponents describe the meridional and equatorial shape powers. The compiled geometry provides boundary field, outward normal, support point, contact curvature, exact volume, and analytic principal inertia moments.

`SuperquadricSetPlan` stores one body-frame shape and material ID per particle slot. `PreparedSuperquadricSet` derives bounding radii and rigid-body mass properties once from immutable shape data and particle mass.

## Contact geometry

`SuperquadricContactPlan` uses fixed-iteration support-map optimization for each candidate pair. `superquadric_pair_contact` returns:

- common normal;
- gap and witness points;
- left and right lever arms;
- effective contact radius from local curvature;
- primal/contact residual;
- validity certificate.

Unsupported, nonconvex, nonfinite, or unconverged contacts reject. The solver does not replace a failed shape query with a bounding-sphere force.

The sphere limit recovers gap equal to center distance minus both radii and effective radius equal to the harmonic radius. Axis-aligned ellipsoids recover their analytic support points.

## Dynamics

Compile the shape plan through the common equation owner:

```text
shapes = phx.discretization.SuperquadricSetPlan(
    semi_axes, epsilon1, epsilon2, material_ids,
    contact_geometry=phx.discretization.SuperquadricContactPlan(iterations=24),
)
compiled = phx.equations.compile_discrete_element_problem(
    phx.equations.DiscreteElementProblemIR(
        "superquadric-collision", materials, gravity=(0.0, 0.0, 0.0),
    ),
    particles,
    shapes,
    phx.discretization.CompliantDEMMethodPlan(contact_model),
    neighborhood=neighborhood,
)
state = compiled.initialize_state(
    0.0, position, velocity, angular_velocity, orientation=orientation,
)
```

`DEMRuntimeState` contains `RigidBodyKinematics` (position, velocity, scalar-first quaternion orientation, and world angular velocity), shared body properties, and keyed contact histories. `PreparedCompliantDEMDynamics.step_detailed` uses the shared rigid-body kick–drift–kick integration and recomputes contact at the end configuration. Its candidate and accepted states expose transactional rollback and energy evidence. Pair forces and moments use one common contact point. Geometry is available at `evaluation.geometry.geometry`, contact response at `evaluation.particle_contact`, and total body load at `evaluation.loads.total`.

The common engine takes contact velocity from the half-kick state at the new
configuration. The former superquadric engine instead published a second force
sample at full-step velocity and omitted the normal-force lever moment. Those
are intentionally not preserved: every nonspherical contact now applies the
full force at the common contact point. Recorded pre-unification trajectories
remain regression references. Their tests explicitly correct the force-sampling
time and the missing normal impulse, and bound the remaining translational
time-discretization error using the dashpot coefficient and timestep. This is
physics parity, not a claim of bit-identical obsolete force bookkeeping.

## Timestep and neighborhood sizing

Neighborhoods use immutable bounding radii plus the contact model's finite interaction range. Bounding spheres only select candidates; they do not decide contact. Contact stiffness, curvature, angular speed, and shape aspect ratio all affect practical timestep resolution. Curvature validity and route capacity remain explicit acceptance criteria; a spherical Rayleigh estimate is not a general anisotropic-body stability guarantee.

Periodic neighborhoods are currently supported only for spheres. Clump,
superquadric, and convex populations are refused at preparation rather than
using unshifted world witnesses for minimum-image body pairs.

## Scope

Supported: smooth convex superquadrics, pair contact, translational and rotational rigid-body state, compositional contact laws, dense or fixed-capacity neighborhoods, branchwise JAX differentiation through the realized contact iteration.

Unsupported: concave exponents, distributed ownership, clumped superquadrics, and simultaneous nonsmooth multi-contact complementarity. Triangle-wall contact uses `SuperquadricTriangleContactPlan` with the shared compiler; see the wall-contact guide.

Run `examples/superquadric_collision.py` for a complete collision and `tools/extended_dem_qualification_campaign.py` for sphere, ellipsoid, and exponent sweeps.
