# Commercial MPM geomechanics and coupled fields

## General plane stress

`GeneralPlaneStressMPMConstitutivePlan` augments a two-dimensional in-plane
deformation with a three-component transverse director and solves:

```text
P[:,3] = 0.
```

It stores the director in material history and returns the full implicit Schur
condensation of the three-dimensional algorithmic tangent. The scalar isotropic
`P33 = 0` closure remains the cheaper specialization.

`MPMMaterialOrientation` and `OrientedMPMConstitutivePlan` rotate deformation,
stress, and tangent actions through one validated proper-orthogonal material frame.

## Pressure-dependent materials

Commercial geomechanical plans are independent models:

- `DruckerPragerMPMConstitutivePlan`: smooth pressure-dependent yield with separate
  friction and dilation angles and a nonsymmetric non-associated tangent.
- `MohrCoulombMPMConstitutivePlan`: principal-stress edge/apex branch codes and
  semismooth return evidence.
- `ModifiedCamClayMPMConstitutivePlan`: cap consistency, preconsolidation pressure,
  void ratio, and hardening state.

Each response reports yield residual, plastic multiplier, branch/corner state,
dissipation, admissibility, material step recommendation, and algorithmic tangent.
No incremental-potential or symmetric-positive-definite claim is made for
non-associated flow.

`NonlocalSofteningPlan` requires a positive characteristic length. It combines local
and nonlocal history with optional rate regularization. Local negative hardening is
not presented as mesh objective.

## Typed coupled fields

`MPMPhysicalFieldPlan` distinguishes mechanical velocity, pore pressure, saturation,
temperature, damage, and species. Contact velocity slots are not reused for scalar
physics.

`PreparedMPMCoupledFieldOperator` implements:

```text
S p_dot + alpha div(u_dot) - div(k/mu grad p) = source

rho c T_dot - div(k_T grad T) = chi plastic_dissipation + source
```

It returns Darcy and heat fluxes, pore-pressure and temperature residuals, and the
Biot/thermal effective-stress correction. Dirichlet and flux boundary conditions are
one prepared `MPMCoupledBoundaryPlan`. JVP and transpose actions include pressure and
temperature cross-dependence.

Application validation remains material, drainage, thermal, geometry, loading, and
observable specific.
