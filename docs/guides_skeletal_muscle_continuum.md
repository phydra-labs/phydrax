# Skeletal-muscle continuum mechanics

`phydrax.applications.skeletal_muscle.continuum` owns one terminal, prescribed-activation continuum fidelity: the GASAM law introduced by Engelhardt, Sachse, Burgkart, and Wall (2025). It is a coupled active/passive material. It is not a passive substrate for a cellular tension model, and it has no external-tension input.

## Source and claim boundary

The implementation follows Eqs. (15), (16), (20), and (25)--(27), and the GASAM column of Table 5 in:

- Engelhardt L, Sachse R, Burgkart R, Wall WA. *Constitutive Models for Active Skeletal Muscle: Review, Comparison, and Application in a Novel Continuum Shoulder Model*. Int J Numer Method Biomed Eng. 2025;41:e70036. [doi:10.1002/cnm.70036](https://doi.org/10.1002/cnm.70036).

The passive generalized-invariant energy originates in:

- Ehret AE, Böl M, Itskov M. *A continuum constitutive model for the active behaviour of skeletal muscle*. J Mech Phys Solids. 2011;59:625--636. [doi:10.1016/j.jmps.2010.12.008](https://doi.org/10.1016/j.jmps.2010.12.008).

The original 2011 active route is not exposed separately. Engelhardt et al. document that the earlier GASA stress omits the deformation derivative of its activation parameter. GASAM includes that derivative and provides the explicit activation weight used here, making stress the derivative of the complete potential. This is the exact reason the implementation selects the later source-named GASAM fidelity rather than silently correcting or partially reproducing the original route.

For reference fiber structural tensor $M=m\otimes m$ and

$$
\widetilde L=\frac{\omega_0}{3}I+(1-\omega_0)M,
\quad
\widetilde I_p=C:\widetilde L,
\quad
\widetilde J=\operatorname{cof}(C):\widetilde L,
$$

GASAM modifies the first invariant using $\widetilde I=\widetilde I_p+\omega_a\lambda^2$, where $\lambda=\sqrt{C:M}$. The activation weight is

$$
\omega_a=\frac{\log\phi}{\alpha\lambda^2},\qquad
\phi=1+\frac{4\alpha}{\gamma}e^{\alpha(1-\widetilde I_p)}
P_{\mathrm{opt}}a\int_{\lambda_{\min}}^\lambda f_\xi(s)\,ds.
$$

Here $a\in[0,1]$ is the prescribed normalized activation (the source's normalized time-activation factor). The implementation analytically integrates Eq. (20), rather than introducing numerical quadrature:

$$
\int_{\lambda_{\min}}^\lambda f_\xi(s)\,ds
=(\lambda_{\mathrm{opt}}-\lambda_{\min})
\left[e^{1/2}-e^{1/2-(\lambda-\lambda_{\min})^2/[2(\lambda_{\mathrm{opt}}-\lambda_{\min})^2]}\right]
$$

for $\lambda>\lambda_{\min}$, and zero otherwise. The complete exact-incompressible energy is

$$
\Psi=\frac{\gamma}{4}\left[
\frac{e^{\alpha(\widetilde I-1)}-1}{\alpha}
+\frac{e^{\beta(\widetilde J-1)}-1}{\beta}
\right].
$$

Substituting Eq. (26) into Eq. (16) makes the active part of this same potential exactly $P_{\mathrm{opt}}a\int f_\xi\,d\lambda$. The kernel uses that algebraically identical form to avoid cancellation; it does not expose the term as a second force owner or as a separately replaceable active stress.

Exact incompressibility is owned by Phydrax's existing mixed $u$--$p$ material and Taylor--Hood/Q2--Q1 FEM route; no penalty or duplicate FEM implementation is introduced.

## Units, axes, and signs

- $F$ is dimensionless with axes `(..., spatial_i, reference_J)` and must satisfy $\det F>0$.
- $m$ is dimensionless with axis `(reference_J,)`; it is normalized during preparation and $m$ and $-m$ are equivalent.
- `stiffness_pa` ($\gamma$), `peak_active_nominal_stress_pa` ($P_{opt}$), pressure, and returned stresses use Pa. Positive fiber normal stress is tension. The core mixed potential is $\Psi_{\mathrm{iso}}+p(J-1)$, so a positive multiplier `pressure_pa` contributes tensile hydrostatic stress; it is not compression-positive fluid pressure.
- Energy density uses J/m³, numerically equal to Pa.
- `prescribed_activation` is dimensionless on the closed support $[0,1]$.
- $\alpha$, $\beta$, $\omega_0$, $\lambda_{min}$, and $\lambda_{opt}$ are dimensionless.

`published_multiload_fit()` converts Table 5's $\gamma=27.1072$ kPa and $P_{opt}=64.6809$ kPa to SI Pa. The source fitted passive response across six load modes but active response only in fiber-direction uniaxial tension. Do not interpret that fit as validated patient-specific anatomy.

## Transaction and use

```python
import jax.numpy as jnp
from phydrax.applications.skeletal_muscle import continuum

fibers = continuum.UniformFiberArchitecturePlan("specimen-x").prepare(
    jnp.array([1.0, 0.0, 0.0])
)
plan = continuum.EngelhardtGasam2025Plan("specimen")
material = plan.prepare(
    continuum.EngelhardtGasam2025Parameters.published_multiload_fit(),
    fibers,
    0.0,
)
commit = material.propose_activation(0.7).commit()
material = material.with_commit(commit)
response = material.evaluate(jnp.eye(3), 0.0)
```

Preparation requires supported fiber evidence. Activation changes follow Plan → Prepared → Candidate → commit. A rejected candidate selects the complete previous `GasamMaterialState`; it never leaves a partially updated activation or evidence object.

Use `prepare_qualified_mixed()` with a core `MixedFiniteElementConstraintPlan` and explicit pressure gauge. Preparation fails closed unless the assembled pair has finite residuals, a valid gauge, an LBB-conforming P2/P1 or Q2/Q1 pair, positive assembled inf-sup evidence, and locking-safety evidence. `solve_manufactured_rest()` exercises the compiled mixed residual and core Newton--Krylov owner for the manufactured $u=0,p=0$ zero-load solution.

## Qualification and differentiation boundary

Run:

```console
python tools/skeletal_muscle_continuum_qualification.py
python examples/skeletal_muscle_continuum.py
python benchmarks/skeletal_muscle_continuum.py --smoke
```

Qualification checks objectivity, stress as the complete energy gradient, tangent/JVP agreement, continuum power, passive and active limits, a fixed-capacity affine mesh sequence, exact mixed-FEM preparation, manufactured rest, and rollback. The EBI passive energy is reported in its source as polyconvex and coercive. This implementation does **not** extend that theorem to the active GASAM energy. It reports only a finite directional Legendre--Hadamard scan at the qualified state. Eq. (20) has a hard branch at $\lambda=\lambda_{min}$; AD and tangent claims are local to either open branch, never global across that point.

## Tendon/aponeurosis omission

No tendon or aponeurosis type is provided. The selected source models tendon attachments with boundary constraints; it does not provide an independently identified tendon/aponeurosis parameter set in the same reference configuration together with a validated muscle-interface traction/power law. Adding only a generic elastic tendon or an empty interface would not satisfy source identity or interface-power validation, so the capability is intentionally absent rather than stubbed.
