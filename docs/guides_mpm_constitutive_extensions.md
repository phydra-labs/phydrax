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

## Isotropic plane stress

`IsotropicPlaneStressMPMConstitutivePlan` wraps one three-dimensional implicit-capable
material. It embeds the in-plane deformation as a block diagonal tensor and solves
for `eta3 = log(lambda3)` from:

```text
P33(Fbar, exp(eta3)) = 0.
```

The converged out-of-plane stretch is committed in material history. The in-plane
algorithmic tangent is the implicit Schur complement of the three-dimensional
tangent. The conservative three-dimensional acoustic bound remains valid in plane
stress.

This scalar adapter remains the minimal isotropic path. For anisotropy and coupled
out-of-plane shear, `GeneralPlaneStressMPMConstitutivePlan` solves the three director
components simultaneously and returns the full implicit Schur-complement tangent;
`OrientedMPMConstitutivePlan` carries an explicit proper-orthogonal material frame.

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
