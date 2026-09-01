# MPM constitutive extensions

Advanced material plans preserve the MPM first-Piola/reference-volume contract.
`MPMConstitutiveResponse` additionally reports dissipation, branch code, and a
material step recommendation. `AbstractImplicitMPMConstitutivePlan` supplies the
algorithmic tangent `dP/dF` required by implicit mechanics.

## Local constitutive roots

`LocalConstitutiveRootPlan` is a bounded scalar Newton root with implicit-function
derivatives and explicit convergence, derivative, residual, and finiteness evidence.
A failed local root rejects the complete MPM attempt and supplies an outer adaptive
cutback; materials never retry internally.

## Plane stress

`PlaneStressMPMConstitutivePlan` wraps one three-dimensional
implicit-capable material and a `BlockDiagonalPlaneStressReductionPlan`. It
embeds the in-plane deformation as a block diagonal tensor and solves the
safeguarded local equation

```text
P33(Fbar, exp(eta3)) = 0.
```

The response records root/failure evidence, positive thickness stretch, the
implicit sensitivity, and the Schur-condensed tangent. Reference thickness
scales areal energy and resultants but not the local closure root.

The scalar closure is valid only for declared block-diagonal membrane
kinematics. Mixed incompressibility uses the separately typed coupled
thickness/pressure reducer; transverse-shear-coupled laws are rejected.

## Multiplicative finite-strain J2

`FiniteStrainJ2MPMConstitutivePlan` uses:

```text
F = Fe Fp
```

with Hencky elasticity, associated J2 flow, linear isotropic hardening, and committed
state `(Fp, equivalent plastic strain)`. It reports:

- elastic/plastic branch;
- yield residual and plastic multiplier;
- `det(Fp)`;
- plastic dissipation;
- current wave-speed bound;
- algorithmic tangent and tangent validity.

The zero-deviatoric trial is an explicit elastic branch. No clipped normalization is
used to hide an undefined plastic direction.

`finite_strain_j2_plane_stress_plan()` composes the same J2 return inside the
plane-stress root, so plastic consistency and `P33 = 0` are solved together rather
than corrected sequentially.

Drucker-Prager, Cam-Clay, non-associated flow, softening, poromechanics, and thermal
plasticity remain separate material families.
