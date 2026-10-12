# μ(I) granular rheology

`phydrax.rheology.MuILaw` is a pointwise friction law, not a new MPM material
or a compressible granular-flow solver. Parameters are dynamic scalar JAX leaves;
the law contains only static curve structure. They can be passed explicitly through
`jit`, `vmap`, and differentiated constitutive updates.

```python
from phydrax.rheology import MuILaw, MuIParameters

parameters = MuIParameters(0.383, 0.643, 0.279, 0.001, 2500.0)
law = MuILaw()
result = law.evaluate(20.0, 100.0, parameters)
# I = 0.1; mu = 0.383 + 0.260 * 0.1 / 0.379
# eta = 100 * mu / 20 Pa s
```

## Definitions and physical domain

Compression-positive physical pressure $p$ is in Pa. The shear-rate magnitude is
$\dot\gamma=\sqrt{2D:D}$, with $D$ the symmetric velocity gradient. Grain
diameter $d$ is in m and intrinsic grain density $\rho_s$ in kg/m³. The Jop law is

$$I=\dot\gamma d\sqrt{\rho_s/p},\qquad
\mu_J(I)=\mu_s+(\mu_2-\mu_s)\frac{I}{I_0+I},\qquad
\eta=\frac{p\mu(I)}{\dot\gamma}.$$

At positive pressure and zero rate, apparent viscosity is undefined/infinite:
`viscosity_defined=False`, `finite=False`, `successful=False`. The base law's
friction limit is exactly $\mu_s$; this is a yield-stress magnitude, not a
well-defined zero-rate stress direction. At finite $p\le0$, the material separates:
zero stress, zero viscosity, `separated=True`. There is no pressure or rate floor.
Finite nonnegative rates are required; inadmissible runtime inputs carry unsuccessful
evidence. Do not multiply infinite viscosity by zero rate to obtain stress; use the
reported `shear_stress_magnitude_pa` and the relevant branch status.

## Barker–Gray partial regularization

Select `MuILaw(regularization="barker-gray-2017")` and supply both dynamic
`barker_alpha` ($0<\alpha<2$) and `barker_mu_infinity` ($\mu_\infty\ge0$).
The final published curve is **two branches**, Eqs. (6.3)–(6.4), not a three-way
splice of the low curve, the unmodified Jop law, and a high-rate continuation:

$$\mu(I)=\begin{cases}
\sqrt{\alpha/\ln(A_-/I)},&0<I\le I_{N1},\\
\dfrac{\mu_s I_0+\mu_2 I+\mu_\infty I^2}{I_0+I},&I>I_{N1},
\end{cases}\qquad \mu(0)=0,$$

$$\ln A_-=\ln I_{N1}+\frac{\alpha}{\mu_+(I_{N1})^2}.$$

$I_{N1}$ is the **lower neutral root of the original Jop curve**. It is computed
from the criterion below, not supplied as a fitted constant. The implementation
brackets in $x=I/(I_0+I)$ using 256 fixed intervals, bisects, and uses the native
implicit scalar-root derivative. Absent or unresolved narrow neutral intervals
return `neutral_point_valid=False` and an unsuccessful evaluation. The logarithmic
matching expression avoids overflow in $A_-$. The upper branch tends to
$\mu_\infty I$ when $\mu_\infty>0$. Setting that coefficient to zero recovers
Eqs. (5.2)–(5.3). The singular apparent viscosity at zero rate is **not** removed.
First-order AD is valid on strict, flowing branches; zero-rate and branch-transition
derivatives are not guaranteed.

## Well-posedness and packing are diagnostics

For $r=I\mu'(I)/\mu(I)$, the reported `hadamard_margin` is

$$-C=-\left[4r^2-4r+\mu^2(1-r/2)^2\right].$$

Positive margin means well-posedness in the cited incompressible **2-D**
friction-only principal-symbol analysis; negative margin indicates ill-posedness.
`hadamard_applicable` marks flowing friction-law states for which that interpretation
is available. It does not certify a 3-D FLIP or elastic–viscoplastic MPM model, nor
is it local admission evidence. A converged discretization does not remove the
underlying PDE's high-rate ill-posedness.

`MuIPackingFractionLaw(Phi_max, b)` optionally reports the empirical steady target
$\Phi=\Phi_\max-bI$. It is not clipped; values outside physical packing bounds warn
that this extrapolated diagnostic is outside its useful range. It does not change
incompressible FLIP volume evolution.

## Generalized Newtonian zero-rate cutover

`GeneralizedNewtonianLaw.viscosity(rate)` now returns
`GeneralizedNewtonianEvaluation`, **not an array**. Consume `viscosity_pa_s` only
along with `viscosity_defined`, `finite`, and `successful`. A power law with $n<1$
returns infinity and unsuccessful status at zero; $n=1$ is Newtonian and finite,
while $n>1$ has zero apparent viscosity there. Bingham/Herschel–Bulkley use
Papanastasiou regularization when `regularization_s>0`, with the exact limit
$\eta_\mathrm{yield}(0)=\tau_y m$ and derivative $-\tau_y m^2/2$.
`regularization_s=0` selects the unsmoothed yield law, whose zero-rate apparent
viscosity is infinite when $\tau_y>0$. A regularized HB law with $n<1$ retains its
singular power-law contribution. Tiny positive rates always use the actual equation.

For pressure admission, Picard evidence, and the dense/compact viscous-FLIP support
boundary, see [FLIP](guides_flip.md).

## Elastic–viscoplastic MPM material

`phydrax.applications.solid_mechanics.MuIMPMConstitutivePlan(G, K)` uses the
canonical dynamic `MuIParameters` above, with the unregularized Jop curve.
The elastic moduli are dynamic plan leaves. This is a multiplicative Hencky
realization of the flow and stress invariants in Dunatunga & Kamrin (2015),
*JFM* **779**, 483–513, Eqs. (2.6), (2.10)–(2.13), (3.12)–(3.13)
([author paper](https://arxiv.org/abs/1411.5447),
[DOI](https://doi.org/10.1017/jfm.2015.383)), not the paper's Jaumann
hypoelasticity or density equation of state.

History has **15 entries**: the spatial elastic metric $B_{e,n}$ packed as
`(11,22,33,23,13,12)` (off-diagonals once), followed by the nine row-major
entries of the previous total deformation gradient $F_n$. There are no scalar
history entries. `initialize_state` initializes both tensors to identity;
old six-entry plastic-metric histories are not accepted.

The classical relative-deformation update forms
$f=F_{n+1}F_n^{-1}$ by solving $F_n^T f^T=F_{n+1}^T$ with the native small
linear solver, then $B_{e,\mathrm{tr}}=f B_{e,n}f^T$. The returned principal
elastic stretches reconstruct $B_{e,n+1}$ in the trial eigenbasis, and
$F_n$ is replaced by $F_{n+1}$. There is no reference-frame plastic-metric
pullback. Under steady simple shear, the elastic metric stays near identity
instead of developing the obsolete plastic metric's $O(\gamma^4)$ condition;
the relative solve still sees total-$F$ conditioning $O(\gamma^2)$.
Diagnostics report `previous_deformation_condition`, `deformation_condition`,
`trial_elastic_metric_condition`, and `relative_deformation_residual`.
Native solve status, dtype-dependent singularity floor and condition limit
remain authoritative; unbounded total deformation is not a numerical guarantee.

The MPM material contract returns first Piola stress $P=\tau F^{-T}$ for the
particle's total $F$, so every step also solves with the current total
deformation. In steady shear the scaled determinant of $F$ decays like
$\gamma^{-3}$, and the native small solve refuses it once it falls below its
singularity floor (about $\gamma\approx10^5$ accumulated shear in float64). The
step is then rejected explicitly with the committed history retained; it is
not repaired. Long steady flows beyond that accumulated strain are outside the
current total-deformation MPM representation.

`initialize_state_from_stress(kirchhoff_stress, parameters)` inverts Hencky
elasticity for $B_e$ at $F_n=I$, where initial Kirchhoff and Cauchy stress agree.
Stress is tension-positive; hydrostatic compression is $-pI$. The returned
`PrestressedMaterialState` reports static-friction admissibility without
projecting an overstress. Use its `state` only after checking `admissible`.
With Kirchhoff deviator $s$, $t=\sqrt{s:s/2}=q/\sqrt{3}$ and compression-positive
$p=-\operatorname{tr}(\tau)/3$, the connected return solves

$$t_{\mathrm{tr}}-G\Delta t\,\dot\gamma_p-p\mu(I)=0,\qquad
I=\dot\gamma_p d\sqrt{\rho_s/(p/J)},\quad J=\det F.$$

The paper's plastic shear rate is $\sqrt{2D^p:D^p}$, not a von Mises equivalent
strain rate: $\Delta\gamma=\Delta t\,\dot\gamma_p/\sqrt{3}$. The inertial number
uses **physical Cauchy pressure** $p/J$. The return preserves plastic volume.
At $t_{\rm tr}\le p\mu_s$, plastic rate is zero and the update is elastic;
undefined zero-rate apparent viscosity does not reject this branch.

Positive flow is bracketed between zero and
$(t_{\rm tr}-p\mu_s)/(G\Delta t)$, then solved by native `LocalRootPlan`
on a logistic variable. No pressure, rate, eigenvalue, or multiplier clipping
is used. Stress and coefficient derivatives use the root's implicit derivative.
Branch codes are `0` elastic, `1` viscoplastic, `2` disconnected, `-1` failed.

For $p_{\rm tr}\le0$, stress and stored energy are zero, and the stored elastic
metric resets to $B_e=I$ while the previous deformation becomes the current $F$.
Trial elastic energy is reported as released dissipation.
Reconnection starts from this stress-free state, without resurrecting prior
shear stress. On a flowing branch dissipation is $t\Delta t\dot\gamma_p$;
`suggested_step` is the tangent relaxation time
$p\,d\mu/d\dot\gamma_p\,/G$. Elastic/disconnected branches suggest infinity,
and failures suggest half the attempted step and retain committed history.
This recommendation is not a replacement for the grid's elastic wave CFL bound.

`evaluate_linearized` supplies the discrete first-Piola tangent. It is finite at
identity but reports `tangent_successful=False` at zero pressure and exactly at
yield: the connectivity/yield switch has no unique tangent there. No PDE
well-posedness guarantee is implied by local convergence.

## Primary sources

- Jop, Forterre & Pouliquen (2006), *A constitutive law for dense granular flows*,
  Nature **441**, 727–730, Eqs. (1)–(3),
  [doi:10.1038/nature04801](https://doi.org/10.1038/nature04801),
  [author manuscript](https://arxiv.org/abs/cond-mat/0612110).
- Barker & Gray (2017), *Partial regularisation of the incompressible μ(I)-rheology
  for granular flow*, JFM **828**, 5–32, Eqs. (3.9), (5.2)–(5.3), (6.3)–(6.4),
  [doi:10.1017/jfm.2017.428](https://doi.org/10.1017/jfm.2017.428),
  [published open-access PDF](https://orca.cardiff.ac.uk/id/eprint/144010/1/partial-regularisation-of-the-incompressible-i-rheology-for-granular-flow.pdf).
- Barker, Schaeffer, Bohórquez & Gray (2015), *Well-posed and ill-posed behaviour
  of the μ(I)-rheology for granular flow*, JFM **779**, 794–818, Eq. (2.43),
  [doi:10.1017/jfm.2015.412](https://doi.org/10.1017/jfm.2015.412),
  [published open-access PDF](https://orca.cardiff.ac.uk/id/eprint/144012/1/well-posed-and-ill-posed-behaviour-of-the-itmui-rheology-for-granular-flow.pdf).
- Simo (1992), *Algorithms for static and dynamic multiplicative plasticity
  that preserve the classical return mapping schemes of the infinitesimal
  theory*, CMAME **99**, 61–112,
  [doi:10.1016/0045-7825(92)90123-2](https://doi.org/10.1016/0045-7825(92)90123-2);
  Simo & Hughes (1998), *Computational Inelasticity*, §9,
  [doi:10.1007/b98904](https://doi.org/10.1007/b98904).
