# Superquadric DEM

Phydrax provides a separate three-dimensional rigid-superquadric path. It shares particle neighborhoods, stable pair keys, compositional contact laws, deterministic reduction, and rigid-body integration with spherical DEM while retaining shape-specific contact geometry.

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

Prepare a `SuperquadricDEMPlan` directly:

```text
dynamics = phx.discretization.SuperquadricDEMPlan(
    shapes,
    phx.discretization.SuperquadricContactPlan(iterations=24),
    contact_model,
).prepare(particles, materials, neighborhood)
```

`SuperquadricDEMState` carries position, velocity, scalar-first quaternion orientation, world-frame angular velocity, and contact history. `PreparedSuperquadricDEMDynamics.step` uses rigid-body kick–drift–kick integration and recomputes contact at the end configuration. Pair forces and moments remain equal and opposite under the common contact point convention.

## Timestep and neighborhood sizing

Neighborhoods use immutable bounding radii plus the contact model's finite interaction range. Bounding spheres only select candidates; they do not decide contact. Contact stiffness, curvature, angular speed, and shape aspect ratio all affect practical timestep resolution. The superquadric path currently reports contact validity but does not expose the spherical DEM Rayleigh restriction helper.

## Scope

Supported: smooth convex superquadrics, pair contact, translational and rotational rigid-body state, compositional contact laws, dense or fixed-capacity neighborhoods, branchwise JAX differentiation through the realized contact iteration.

Unsupported: concave exponents, distributed ownership, superquadric–triangle wall contact, clumped superquadrics, and simultaneous nonsmooth multi-contact complementarity.

Run `examples/superquadric_collision.py` for a complete collision and `tools/extended_dem_qualification_campaign.py` for sphere, ellipsoid, and exponent sweeps.
