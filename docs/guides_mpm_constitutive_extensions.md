# MPM constitutive extensions

Advanced material plans preserve the MPM first-Piola/reference-volume contract.
`MPMConstitutiveResponse` additionally reports dissipation, branch code, and a
material step recommendation. `AbstractImplicitMPMConstitutivePlan` supplies the
algorithmic tangent `dP/dF` required by implicit mechanics.

## Local constitutive roots

`phydrax.nonlinear.LocalRootPlan` is a bounded scalar Newton root with implicit-function
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

This scalar adapter remains the minimal isotropic path and is valid only for declared
block-diagonal membrane kinematics. Mixed incompressibility uses the separately typed
coupled thickness/pressure reducer; transverse-shear-coupled laws are rejected by the
scalar adapter. For anisotropy and coupled out-of-plane shear,
`GeneralPlaneStressMPMConstitutivePlan` solves the three director components
simultaneously and returns the full implicit Schur-complement tangent;
`OrientedMPMConstitutivePlan` carries an explicit proper-orthogonal material frame.

## Multiplicative finite-strain J2

`FiniteStrainJ2MPMConstitutivePlan` uses:

```text
F = Fe Fp
```

with Hencky elasticity, associated J2 flow, and linear isotropic hardening.
Committed history has seven entries: the plastic metric `Cp_inverse = Fp_inverse
Fp_inverse.T`, packed as `(11, 22, 33, 23, 13, 12)`, followed by equivalent plastic
strain. Off-diagonal entries appear once, without Mandel scaling. Initialize with
`material.initialize_state(batch_shape, dtype)`, not zeros: the stress-free metric
is `(1, 1, 1, 0, 0, 0)`. It reports:

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

## Shared finite-strain kernel

J2 and Drucker–Prager share a private principal-Hencky kernel, not a generic
multisurface solver. It forms `Be = F Cp_inverse F.T`, evaluates principal
Hencky elasticity and the material-specific return through
`phydrax.linalg.hermitian_spectral_map`, and reconstructs Kirchhoff stress and
the corrected elastic metric. Native small solves recover `P = tau F_inverse.T`
and the updated plastic metric. Invalid determinants, non-SPD metrics, or failed
returns retain committed history and expose failure evidence.
The kernel's `evaluate_principal_point` receives a required keyword
`wave_modulus` from the material. Wave-speed evidence therefore uses the owning
elastic law's modulus rather than assuming every parameter object carries
constant shear and bulk moduli.

The material map is permutation-equivariant; scalar invariants are
permutation-invariant. Spectral diagnostics are nondifferentiable, while
first-order stress derivatives remain defined at repeated elastic eigenvalues.
`evaluate_linearized` reports validity of the discrete-update tangent; second-order
derivatives at degenerate spectra are not guaranteed.

J2 uses tension-positive stress, `p = -trace(tau)/3`, and
`q = sqrt(3/2 * s:s)`. Its yield is `q - (yield_stress + H alpha)` and its
plastic multiplier is `f_trial / (3 G + H)` on the plastic branch.

## Multiplicative Mohr–Coulomb

`MohrCoulombMPMConstitutivePlan` uses the shared principal-Hencky kernel
with its own ordered face → left/right edge → apex closed-form return.
It sorts trial stresses and unsorts corrected stress/elastic metric, preserving
permutation equivariance even at repeated principal stresses. This is not an
additive nine-component plastic-strain update or a semismooth local root.
History is `Cp_inverse(6), alpha`, where `alpha` accumulates the sum of plane
multipliers and `c(alpha) = c + H alpha`. Branch codes are elastic `0`,
face `1`, edge `2`, apex `3`, failure `-1`; `edge_side` distinguishes both edges.
See the [Mohr–Coulomb return and initialization contract](guides_mpm_commercial_geomechanics.md#mohrcoulomb-ordered-return)
for parameter conventions, all-plane evidence, prestress, and tangent validity.


## Finite-strain Modified Cam-Clay

`ModifiedCamClayMPMConstitutivePlan()` implements the associated two-invariant
Borja–Tamagnini model, not the scalar cap return of a constant-modulus law.
See the [energy and prestress initialization contract](guides_mpm_commercial_geomechanics.md#modified-cam-clay-energy-and-initialization)
and Borja and Tamagnini (1998),
[DOI: 10.1016/S0045-7825(97)00141-2](https://doi.org/10.1016/S0045-7825(97)00141-2).
Its parameters are `ModifiedCamClayParameters(mu0, alpha0, M, lambda_raw, kappa_raw)`;
the raw slopes are normalized by the fixed history `1 + e0`, not current volume.
History has nine entries: `Cp_inverse(6), pc, p0, e0`. The initial mean pressure
`p0` is independent of `pc`, and `epsilon_v0 = 0` is fixed. Identity-metric
`initialize_state` leaves zero reference scalars and is inadmissible; callers
must use `initialize_state_from_stress` and check its admissibility.

The hyperelastic potential is
`psi = B R + (3/2) (mu0 + alpha0 B) b^2`,
where `B = p0 exp(-a/R)`, `R = kappa_raw/(1 + e0)`,
`a = trace(epsilon_e)`, and `b = sqrt(2/3 * dev(epsilon_e):dev(epsilon_e))`.
Thus `mu0` is additive, whereas the zero-shear reference modulus is
`mu0 + alpha0 p0`; differentiating the same potential supplies both shear
stress and the shear contribution to pressure. The material has free energy
and refuses `p <= 0` rather than clipping pressure.

On the plastic branch, `phydrax.nonlinear.VectorLocalRootPlan` solves exactly
three unknowns `(a, b, gamma)`: corrected elastic volume strain, corrected
elastic shear strain, and the multiplier of the unscaled stress-squared yield
function. With trial invariants `(a_trial, b_trial)` and
`D = (lambda_raw - kappa_raw)/(1 + e0)`, hardening is eliminated analytically:

```text
pc(a) = pc_old exp((a - a_trial)/D)
p(a,b) = B [1 + 3 alpha0 b^2/(2 R)]
q(a,b) = 3 (mu0 + alpha0 B) b
r1 = a - a_trial - gamma (2p - pc)
r2 = b - b_trial + 2 gamma q/M^2
r3 = q^2/M^2 + p(p - pc)
```

The converged vector root differentiates implicitly through its residual
Jacobian, not through Newton iteration history. The full `evaluate_linearized`
algorithmic tangent `dP/dF` is obtained by AD of the discrete stress update,
including changing trial directions, spectral reconstruction, plastic-metric
kinematics, and first-Piola conversion. The local 3×3 Newton Jacobian alone is
not the full tangent; it must not be substituted for it or symmetrized.
Repeated elastic eigenvalues use the spectral substrate's confluent limits,
and zero trial shear uses the isotropic limit without an undefined direction.
`root_successful` and `tangent_successful` expose solve and derivative validity;
a failed return retains committed history and requests outer rejection rather
than performing internal retries.

## Granular μ(I) material

`MuIMPMConstitutivePlan(shear_modulus, bulk_modulus)` composes the same
principal-Hencky substrate with `phydrax.rheology.MuIParameters`. Its 15-entry
history stores six packed spatial elastic-metric entries `(11,22,33,23,13,12)`
and nine row-major previous-total-deformation entries. The relative update
`f = F_next F_previous^-1`, formed by a native solve, advances
`Be_trial = f Be_previous f.T`; the corrected elastic metric is stored directly.
The elastic/viscoplastic/disconnected branches use codes `0/1/2`; unsuccessful
returns use `-1` and retain history.
Positive plastic flow uses the paper's shear norm `t = q / sqrt(3)`, solves
`t_trial - G dt plastic_shear_rate - p mu(I) = 0` on a bracketed logistic
variable with `LocalRootPlan`, and preserves plastic volume. Inertial number
uses Cauchy pressure `p / det(F)`, not Kirchhoff pressure.

Disconnection at nonpositive trial pressure resets the elastic metric to identity
and commits the current total deformation, reporting released energy as dissipation.
Reconnection cannot restore the erased shear stress. The positive-flow tangent
relaxation time supplies `suggested_step`; it does not replace the elastic CFL
bound. Native spectral and implicit-root derivatives supply the full `dP/dF`
and dynamic friction-parameter sensitivities. Zero pressure and exact yield
have finite branch-selected tangents but unsuccessful unique-tangent evidence.
See the [equations, units, and initialization contract](guides_mu_i_granular_rheology.md#elasticviscoplastic-mpm-material).

## Fracture spectral split

The AT2 split uses native Hermitian functions for Hencky strain and its
positive/negative principal parts. A finite primal does not certify a unique
split derivative: tangent status is false at a zero active Hencky eigenvalue
or the volumetric trace kink. An inactive plane-strain zero eigenvalue is
excluded from this kink test.



Drucker-Prager, Cam-Clay, non-associated flow, softening, poromechanics, and thermal
plasticity remain separate material families.
