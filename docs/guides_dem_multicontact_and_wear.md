# DEM multicontact, wall observables, and wear

## Elastic half-space correction

Pairwise soft-sphere overlap can overpredict deformation when one particle carries several simultaneous contacts. `ElasticHalfSpaceMulticontactPlan` adds a bounded fixed-point correction to `SoftSphereDEMMethodPlan`:

```text
method = phx.discretization.SoftSphereDEMMethodPlan(
    contact,
    multicontact=phx.discretization.ElasticHalfSpaceMulticontactPlan(
        iterations=8,
        convergence_tolerance=1.0e-4,
    ),
)
```

Each iteration scatters neighboring contact loads back to particle endpoints, estimates nonlocal elastic deflection, and updates effective pair gaps. The iteration count is static and the final residual is an acceptance criterion. Omitting `multicontact` preserves the pairwise path without allocating a second history model.

This correction is a compliant multicontact approximation, not a rigid complementarity solver.

## Wall facet observables

`evaluate_wall_facet_observables` reduces sphere–triangle contact response to fixed facet arrays:

- force and torque;
- traction;
- mechanical work rate;
- heat rate;
- force and torque balance residuals.

Triangle feature ownership remains deterministic at shared edges and vertices. Facet observables use the same contact response as particle loads; they are not reconstructed from a separate pressure approximation.

## Finnie wear

`FinnieWearPlan` maps tangential impact work and incidence angle to nonnegative wear rate using material-pair coefficients and hardness. `FinnieWearState` accumulates facet volume loss. Candidate wear commits only when the contact response, material lookup, geometry, and volume accounting succeed.

`commit_triangle_wall_wear` moves vertices along area-weighted facet normals from the accepted volume loss. Geometry mutation is explicit and separated from rate evaluation. The committed wall receives a new fingerprint so replay cannot confuse pre-wear and post-wear geometry.

Wear parameters must be calibrated for the selected units and material. The implementation does not claim a universal Finnie coefficient.

## Force and torque servo walls

`ServoDEMBarrierMotion` supports `FORCE` and `TORQUE` control modes. The PID update includes:

- output and velocity saturation;
- conditional integration antiwindup;
- gain scheduling by particle-contact stiffness;
- maximum displacement per neighborhood skin;
- explicit state for displacement, velocity, integral error, and previous error.

A servo update cannot move farther than the configured skin allowance within one rebuild interval. Saturation and invalid target axes reject rather than silently changing controller semantics.

## Qualification

Use wall force/torque residuals, wear volume residual, multicontact fixed-point residual, and neighborhood status together. A small force residual alone does not qualify contact geometry or wear evolution.
