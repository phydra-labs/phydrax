# Geomechanics candidate qualification

Phydrax enables x64 by default; the launch recipes explicitly select it.

`tools/geomechanics_qualification.py` executes bounded, deterministic scientific
comparisons through public Phydrax APIs. It composes the existing `ScientificCase`,
`SupportTuple`, `PublicationReference`, `QualificationCriterion`,
`CampaignStartRecord`, `CampaignObservationRecord`, and `QualificationEvidence`
records. There is no public geomechanics qualification class or catalog.

**Every tolerance is a proposed candidate criterion, not an approved acceptance
limit.** A numerical `passed` outcome means only that the recorded run met that
candidate bound on its exact support. It does not authorize release, validate
field behavior, certify engineering design, qualify other materials or geometries,
or establish transient-flow/large-deformation/general-contact accuracy.
The generic criterion record requires an `approval_id`; this tool deliberately
uses `proposed-candidate-NOT-approved`, never fabricated human approval metadata.
Evidence is labeled automated and not human reviewed.

## Reproduction and evidence

From the geomechanics worktree, with its environment installed:

```sh
JAX_ENABLE_X64=1 PYTHONPATH=. .venv/bin/python tools/geomechanics_qualification.py \
  --case ergun-packed-bed --case schiller-naumann-terminal-velocity \
  --output /tmp/geomechanics-drag.json
JAX_ENABLE_X64=1 PYTHONPATH=. .venv/bin/python tools/geomechanics_qualification.py \
  --all --output /tmp/geomechanics-qualification.json
.venv/bin/python -m pytest tests/unit/test_geomechanics_qualification_tool.py -q -n 2
```

`--case` is repeatable; `--all` and `--case` are mutually exclusive. Unknown,
empty, and duplicate selections are refused. The full campaign runs each selected
case once at its registered resolution sequence; there is no hidden retry,
capacity growth, parameter fitting, or alternate solver on failure. The CLI
requires float64 and records Python/JAX/NumPy versions, backend and actual devices.
Measurements in the program worktree are under shared-machine contention.
Exit status is zero only when every selected case passes its candidate metrics;
failed or inconclusive cases produce exit status one **after writing the report**.
Including the omitted oblique case therefore makes `--all` inconclusive at best.

For each metric, criteria are issued before the actual campaign start. If
`--output` is supplied, the report containing start records is persisted before
the numerical runner executes. Without an output path, start records are retained
in memory and the completed report is printed to stdout. An interrupted run can
therefore leave an honest `started` checkpoint rather than fabricated observations.
The subsequent raw observation, its SHA-256 content identity, observations and
final evidence are linked and checked by `validate_qualification_causality`.
The embedded raw record is the content-addressed observation artifact; no fake
external reference arrays or calibration/locked data split are constructed.

Reports contain all resolution-level errors, finest-resolution metrics, numerical
success/inconclusive flags, resource limits/use, publications, support tuples,
criterion/start/observation/evidence IDs, and outcome. Missing metrics remain
`null` and cannot pass. Numerical rejection or overflow is failed; an incomplete
bounded scientific regime is inconclusive. An explicitly diagnosed declared
total-deformation range refusal before steady admission is also scientifically
inconclusive, while its unsuccessful numerical status remains visible and its
native gate is not relaxed. A completed, applicable comparison
outside its candidate tolerance is failed. The aggregate preserves failure before
inconclusive before passed. JSON rejects NaN/infinity and is bounded to nine
cases and 2 MB. Scientific arrays are not dumped unboundedly.

## Cases, formulas, and candidate metrics

Stress is tension-positive. Compression-positive pore pressure is denoted $p$.
All errors below are dimensionless. Resolution sequences and numerical/material
parameters are part of the pre-run exact support record and also reported at each
level. No claim is extrapolated beyond the measured sequence; in particular,
roundoff-level direct-correlation agreement is not evidence of spatial convergence.

### Lamé thick cylinder

For inner/outer radii $a,b$ and positive pressures $p_i,p_o$,

$$
A=\frac{a^2p_i-b^2p_o}{b^2-a^2},\qquad
B=\frac{a^2b^2(p_i-p_o)}{b^2-a^2},\qquad
\sigma_r=A-\frac{B}{r^2},\quad \sigma_\theta=A+\frac{B}{r^2}.
$$

The candidate is relative $L_2$ radial/hoop stress error **≤ 0.01**.
The numerical route is full-ring axisymmetric MPM with three-dimensional material
kinematics, plane-strain axial constraint, canonical pressure boundary quadrature,
and bounded explicit relaxation. The host oracle evaluates Lamé independently;
it does not prescribe the analytical displacement as the numerical solution.
This is a low-strain elastic model comparison, not finite-strain cylinder validation.
Support: $a=1$, $b=2$, axial length 0.25, $p_i=0.001$, $p_o=0$,
$G=1$, $K=2$, density one (consistent SI-scaled units); body-drag rate 2 s⁻¹.
The radial-cell sequence is **8, 16, 32, 64**, with two particles per radial cell
and four axial particles, **1,200, 2,400, 4,800, 9,600 steps**, respectively;
step sizes 0.01, 0.005, 0.0025, 0.00125 s give the same 12 s horizon.
The maximum particle count is 512. Late kinetic decay, displacement drift and
force equilibrium are applicability diagnostics, not stress exclusions.
Source: Lamé solution, Timoshenko & Goodier, *Theory of Elasticity*, third edition
(1970), thick-walled cylinders.

### Rough strip footings

The ideal is a weightless half-space with a rough surface strip, plane strain,
cohesion $c$, and associated flow. For Tresca,

$$q_\mathrm{ult}=cN_c,\qquad N_c=\pi+2.$$

For friction angle $\phi>0$ and dilation $\psi=\phi$,

$$
N_q=e^{\pi\tan\phi}\tan^2(\pi/4+\phi/2),\qquad
N_c=(N_q-1)\cot\phi,\qquad q_\mathrm{ult}=cN_c
$$

with zero surcharge. The candidate for each is relative measured $N_c$ error
**≤ 0.02**. The Tresca limit uses Mohr–Coulomb at $\phi=0$ (J2 is not itself the
Tresca yield surface). The delivered three-dimensional material is used on a thin
Cartesian extrusion with every out-of-plane velocity constrained to zero, giving
plane-strain deformation. B-bar is enabled. The strip is imposed by rough nodal
indentation, and reaction is measured from constraint impulse, not assigned from
Prandtl's formula. Finite-domain effects, slow-loading/plateau sufficiency and
resolution behavior are retained as applicability diagnostics; failure to reach
an ultimate plateau is inconclusive, not a bearing-capacity pass. No $N_\gamma$
(self-weight factor) is compared.
Support: width $B=1$, box $6B$ wide by $3B$ deep, $c=1$, $G=5000$, $K=50000$,
density one and zero hardening. The rigidity index is **$G/c=5000$**; $K/G=10$
preserves the original $\nu=0.4516129$. The frictional case has
$\phi=\psi=20^\circ$. In the rigid-perfectly-plastic limit the Prandtl collapse
load is independent of $G$; increased rigidity mobilizes it at much smaller
settlement, but an actual load–settlement plateau is still required.

The corrected runner uses **8 and 16 cells across the full footing width**,
equivalently **4 and 8 cells across the half-width**. Both indentation speeds
**0.02 and 0.01** are unchanged. Target settlement is now **$0.006B$**;
at the finest resolution it is only 0.192 of the $h/2=0.03125B$ static-platen
support limit. Each case is bounded to 9,216 particles and **24,002 total step
attempts**; the slow finest solve has 10,667 steps. The explicit time bound is
$0.0009/n$ s, with the kernel's undeformed three-dimensional wave factor 1.5
included (initial acoustic CFL about 0.321, below the unchanged 0.4 limit).
No stability or qualification admission rule is relaxed.

The pre-run work model targets approximately 30 minutes/case and refuses
forecasts above 2,100 s. It scales earlier measured cost by actual new step
count and particle count; it is not a runtime guarantee or silent truncation.
Capacity remains the mean reaction over the last 20% of settlement. Bounded
33-point accepted load–settlement curves, plateau intervals, and their position
relative to $h/2$ are reported. The original rate/plateau/mesh outcome gates
remain unchanged.

The earlier $G/c=100$, settlement $0.05B$ configuration and its failed finest
attempts are preserved as **superseded configurations**, not combined with
the corrected run to manufacture a convergence rate. Current 8→16 comparisons
use identical corrected physical parameters. The finite box and reference-node
platen remain explicitly limited support, not infinite-domain or finite-
penetration moving-contact qualification.

The zero-surcharge problem independently measures **only $N_c$**.
$N_{q,\mathrm{inferred}}=N_{c,\mathrm{measured}}\tan\phi+1$ is reported as an
explicitly algebraic diagnostic, never independently measured or qualified.
At $\phi=0$ it is identically one and supplies no capacity sensitivity.
Independent $N_q$ identification would require equilibrated prestress,
maintained uniform surface surcharge, and matched additional capacity solves.
That extra campaign is not substituted by an apparent algebraic agreement.

Sources: Prandtl (1920), *Über die Härte plastischer Körper*,
*Nachrichten der Gesellschaft der Wissenschaften zu Göttingen* 74–85;
Reissner (1924), *Zum Erddruckproblem*, ZAMM 4,
[DOI:10.1002/zamm.19240040605](https://doi.org/10.1002/zamm.19240040605).

### Hertz normal impact

For elastic reduced mass $m^*$, radius $R^*$, modulus $E^*$ and incoming normal
speed $v_0$, with zero damping and friction,

$$
k=\frac43E^*\sqrt{R^*},\quad F=k\delta^{3/2},\quad
\delta_\max=\left(\frac{5m^*v_0^2}{4k}\right)^{2/5},\quad
F_\max=k\delta_\max^{3/2},
$$

$$
T_c=\frac{2\delta_\max}{v_0}
\int_0^1\frac{dx}{\sqrt{1-x^{5/2}}}.
$$

The candidates are contact-duration and peak-force relative errors each
**≤ 0.005**. The numerical solution is actual compliant DEM integration of an
elastic sphere pair, with step refinement; host energy/quadrature determines the
reference. Other energy/restitution/overlap observations are diagnostics, not
silently added acceptance claims.
Support: equal masses 1 kg, radii 0.5 m, $E=2\times10^6$ Pa, $\nu=0.25$,
relative incident speed 1 m/s, two bodies and one contact route.
The sequence is **256, 512, 1,024 steps** over 1.25 reference contact durations
(1,792 total steps), with a linearly interpolated separation crossing.
Source: Hertz (1881),
*Über die Berührung fester elastischer Körper*; Johnson (1985), *Contact Mechanics*,
[DOI:10.1017/CBO9781139171731](https://doi.org/10.1017/CBO9781139171731).

### Ergun packed bed

For void fraction $\epsilon$, superficial speed $U$, sphere diameter $d$,
viscosity $\mu$ and fluid density $\rho$,

$$
\frac{\Delta P}{L}=
\frac{150\mu(1-\epsilon)^2U}{\epsilon^3d^2}+
\frac{1.75\rho(1-\epsilon)U^2}{\epsilon^3d}.
$$

The public `InterphaseDragLawPlan("ergun")` receives intrinsic slip $U/\epsilon$;
the fluid-phase balance gives pressure gradient $\beta U/\epsilon^2$.
The candidate is maximum relative pressure-gradient error **≤ $10^{-6}$**.
Support: $\epsilon=0.4$, $d=0.002$ m, $\mu=0.001$ Pa s,
$\rho=1000$ kg/m³, geometric sampling of $U=10^{-6}$–$10^{-2}$ m/s at
**4 then 8 flow points**. This is direct law comparison, not resolved packed-bed
CFD or permeability validation. Source: Ergun (1952), *Fluid flow through packed
columns*, *Chemical Engineering Progress* 48:89–94.

### Schiller–Naumann terminal velocity

For $0<Re\le1000$,

$$
Re=\rho_fdv/\mu,\quad C_D=\frac{24}{Re}(1+0.15Re^{0.687}),\quad
\frac\pi6d^3(\rho_p-\rho_f)g=\frac12\rho_f C_D\frac\pi4d^2v^2.
$$

An independent host Brent root of the force balance is compared with a JAX
bisection using the public interphase law at **20, 30, 40 iterations**.
The candidate relative speed error is **≤ $10^{-7}$** and normalized force
residual **≤ $10^{-8}$**. Support: $d=0.001$ m, $\rho_p=2500$ kg/m³,
$\rho_f=1000$ kg/m³, $\mu=0.001$ Pa s, $g=9.81$ m/s², isolated-particle
porosity one, speed bracket [0, 0.5] m/s, host root bounded to 100 iterations.
At zero speed the finite Stokes coefficient is used, not division by $Re=0$.

Source: Schiller & Naumann, *Über die grundlegenden Berechnungen bei der
Schwerkraftaufbereitung*, VDI-Zeitschrift **77 (1933)**:318–320.
The primary-journal holdings in the
[Bayerische Staatsbibliothek catalog](https://opacplus.bsb-muenchen.de/discovery/fulldisplay/alma991054944389707356/49BVB_BSB:VU1)
and [HathiTrust record](https://catalog.hathitrust.org/Record/000637388)
both explicitly assign volume 77 to 1933 and volume 79 to 1935. These independently
resolve the year/volume contradiction in secondary English-title references
that say 1935. The original article scan was not obtained; the year is verified
against the original journal's library records, not inferred from those
secondary references.

### Bagnold inclined plane

For thick steady flow with constant packing $\Phi$, rough no-slip base and free
surface at depth $h$,

$$
P(y)=\rho_s\Phi g\cos\theta(h-y),\quad
\tau(y)=\rho_s\Phi g\sin\theta(h-y),\quad
I=\dot\gamma d\sqrt{\rho_s/P},
$$

$$
\mu(I)=\mu_s+(\mu_2-\mu_s)\frac{I}{I_0+I},\quad
I_\theta=I_0\frac{\tan\theta-\mu_s}{\mu_2-\tan\theta},
$$

$$
u(y)=\frac{2I_\theta}{3d}\sqrt{\Phi g\cos\theta}
\left[h^{3/2}-(h-y)^{3/2}\right].
$$

Equivalently, with $\rho_\mathrm{bulk}=\Phi\rho_s$ and
$B_\mathrm{bulk}=\tan\theta/(\Phi I_\theta^2)$,
$\tau=B_\mathrm{bulk}\rho_\mathrm{bulk}d^2\dot\gamma^2$. This gives exactly
$u=(2/3)\sqrt{g\sin\theta/(B_\mathrm{bulk}d^2)}
[h^{3/2}-(h-y)^{3/2}]$, without dropping the packing factor.
The candidate relative profile $L_2$ error remains **≤ 0.02**. Primary output
comes from **full coupled public implicit μ(I) MPM**:
`MuIMPMConstitutivePlan` with `PreparedImplicitMPMDynamics.step_detailed`,
backward-Euler momentum, native Newton–Krylov/GMRES, APIC transfers and moving
particles. Pressure is measured from the accepted Cauchy stress, not prescribed
after initialization or obtained from a law-only reduction. The pointwise
public-law inversion remains **diagnostic only** and cannot provide the primary
profile, initialization, or outcome.

The original parameters are unchanged: $\mu_s=0.383$, $\mu_2=0.643$,
$I_0=0.279$, $d=0.001$ m, $\rho_s=2500$ kg/m³, $\Phi=0.6$,
$\rho_\mathrm{bulk}=1500$ kg/m³, $h=0.08$ m, $g=9.81$ m/s²,
$\tan\theta=0.5$. The depth sequence is **8, 16, 32 cells**. The 3-D extruded
plane-strain layer has three periodic x/z nodes per 0.08 m period (the complete
quadratic B-spline stencil requirement), two y samples/cell, and three y halo
cells per end. Maximum work geometry is 576 particles and 351 grid nodes.
All velocities vanish on y≤0 nodes; v_z=0 everywhere. The particle free surface
is naturally unloaded, without a height-fitted pressure or velocity boundary.

Elastic support is explicitly finite: **G=200,000 Pa, K=1,000,000 Pa**,
maximum initial p/G≈0.00526. At F=I, hydrostatic initial compression
$p_0=\rho_\mathrm{bulk}g\cos\theta(h-y_p)$ is initialized with
`MuIMPMConstitutivePlan.initialize_state_from_stress(-p0*I, parameters)`.
The history has **15 entries**: packed elastic $B_e$ (six components) and the
previous total F (nine row-major components), with no scalars. Initially
$B_e=\exp[-2p_0/(3K)]I$ and $F_\mathrm{previous}=I$.
Initial deviatoric stress is zero. Initial velocity is the genuinely transient
linear field $u=\sqrt{g/h}\,y$, with its consistent affine gradient—not the
analytical steady profile. This finite elastic–viscoplastic Hencky realization
is compared with the ideal continuum profile; incompressibility and agreement
with the paper's Jaumann implementation are not claimed.

The material's shear norm is $t=\sqrt{s:s/2}=q/\sqrt3$, and plastic rate
$r=\sqrt{2D^p:D^p}$; I uses Cauchy pressure. No q-as-shear substitution is made.
Each level is bounded to 131,072 attempts, 768 gravity times (about 69.4 s),
and 1,800 wall seconds checked between 32-step observation blocks; one compilation
or block overrun is reported honestly. The refined maximum dt is 0.001 s, with predicted
shear increment ≤0.08 and actual candidate increment required ≤0.1.
There is no retry, stress reinjection, metric remapping, or rate/pressure floor.

Numerical convergence uses a force-scaled absolute Newton threshold
$10^{-6}\|\{m_pg\}\|_2$, relative Newton/Krylov tolerance $10^{-6}$, at most
25 Newton steps and 1,024 cumulative Krylov iterations. These are recorded solver
controls, not altered scientific criteria. Total-F and elastic-metric conditioning,
material/tangent admission, stress/pressure/flux errors, and actual r/I are
reported. Three consecutive observation windows must meet the unchanged
$10^{-4}$ profile, acceleration/g and kinetic-change gates, negligible transverse
energy, and declared domain/temporal support. Ordinary numerical rejection fails;
admitted nonsteady/resource-limited attempts and diagnosed total-deformation range
refusals before steady admission remain inconclusive. A sub-ULP observation-time
remainder is resolved at floating-point time precision rather than evolved
with a synthetic minimum timestep.

The original n8 numerical configuration (dt≤0.004 s, cumulative Krylov cap256)
reached the documented total-F range before a registered steady window:
particle133 at t=10.06877 s had positive candidate detF=1.03519 but scaled
determinant 9.99431e-16 below the native 1e-15 floor. Relative-deformation solve
status was success; the current-F kinematic solve returned status1. This was not
a condition-limit or local-root refusal. Its total-F shear component was
F_xy≈101058; the full deformation is not pure simple shear. Numerical rejection
is retained, with scientific disposition **inconclusive: range limit**.
See [precise forensic evidence](local://bagnold-failure.md) and the
[material's documented range](guides_mu_i_granular_rheology.md).
The n16/n32 numerical refinement does not shrink the physical layer, reset
history, relax the native gate, or change the proposed profile/steadiness bounds.

The separate full viscous-FLIP attempt is preserved as diagnostic history,
including its rerun after incremental-pressure repair. Neither it nor a close
law-only profile is substituted for a steady coupled-MPM observation.
Sources: GDR MiDi (2004), *On dense granular flows*, EPJE 14:341–365,
[DOI:10.1140/epje/i2003-10153-0](https://doi.org/10.1140/epje/i2003-10153-0);
Jop, Forterre & Pouliquen (2006), *A constitutive law for dense granular flows*,
Nature 441:727–730, [DOI:10.1038/nature04801](https://doi.org/10.1038/nature04801).
The MPM realization and normalization follow Dunatunga & Kamrin (2015),
*Simulating the dynamics of dense granular flows with a continuum model*,
JFM 779:483–513, [DOI:10.1017/jfm.2015.383](https://doi.org/10.1017/jfm.2015.383).

### Terzaghi one-dimensional consolidation

For drained $z=0$ and no-flow $z=H$, initial excess pressure $p_0$, and
$T_v=c_vt/H^2$,

$$
\frac{p(z,t)}{p_0}=\frac4\pi\sum_{n=0}^{159}
\frac{\sin[(2n+1)\pi z/(2H)]}{2n+1}
\exp[-(2n+1)^2\pi^2T_v/4],
$$

$$
U(t)=1-\frac8{\pi^2}\sum_{n=0}^{159}
\frac{\exp[-(2n+1)^2\pi^2T_v/4]}{(2n+1)^2}.
$$

The tool exercises the **full** `advance_double_point_mpm_step`, not an
operator-only pressure diffusion reduction. The reference is the model's
incompressible-constituent, quasi-static limit with unit Biot coefficient.
The pre-registered physical resolutions are **8 and 16 cells**, with time factors
**0.05 and 0.2**, a 100:1 consolidation-to-wave-time ratio, load/modulus $10^{-5}$,
and at most **12,000 attempts per level**. The 160-term series uses depth measured
from the drained surface; the solver coordinate runs upward from the no-flow base.
One computational MAC halo cell at each end preserves complete moving assignment
support. Drainage is imposed at the last physical cell center, giving
$H=L-h/2$; the independently computed settlement includes the drained half-cell
cap between that center and the actual loaded top marker. No halo cells enter the
physical pressure norm. Pressure and settlement errors are reported against both
the prescribed target time and the actual accepted elapsed time, and the
quasi-static stress-equilibrium gap is measured. The final registered metrics use
the target-time reference, so early rejection cannot masquerade as target
attainment. The candidate pressure relative $L_2$ error is **≤ 0.01**;
settlement relative error is **≤ 0.02**.
If the bounded solver cannot reach the requested regime, it records inconclusive
with measured reference errors/time/equilibrium gaps; rejected steps are failed.
No pressure-only reduction is substituted and no $c_v$ is fitted to the observed
curve. This does not establish compressible-Biot, Mandel–Cryer, dynamic pore-wave,
axisymmetric or contact-coupled accuracy. Source: Terzaghi (1925),
*Erdbaumechanik auf bodenphysikalischer Grundlage*, Franz Deuticke.

### Optional Wu–Thornton–Li oblique impact

Wu, Thornton & Li (2009), *A semi-analytical model for oblique impacts of
elastoplastic spheres*, Proc. R. Soc. A 465:937–960,
[DOI:10.1098/rspa.2008.0221](https://doi.org/10.1098/rspa.2008.0221), is the
correct oblique reference, not the 2003 normal-impact paper.
The [publisher record](https://royalsocietypublishing.org/rspa/article/465/2103/937/82796/A-semi-analytical-model-for-oblique-impacts-of)
and [author-institution bibliography](https://research.birmingham.ac.uk/en/publications/a-semi-analytical-model-for-oblique-impacts-of-elastoplastic-sphe/)
were located, but no explicitly rights-cleared quantitative dataset was
identified. This case is recorded **omitted**, with inconclusive evidence and
`null` velocity error, not skipped-as-passed. No curves are digitized, no figures
or tables redistributed, and no elastic Hertz model substituted for the published
elastoplastic oblique-impact reference. Its 0.03 velocity-error criterion remains
an unused candidate proposal.

## Data rights and interpretation

All executed references are independently implemented mathematical equations.
No external numerical datasets, publication images, digitized curves, or upstream
code/output are bundled. Publication references are bibliographic metadata, not
claims that their data are reusable. No external simulator code or outputs are
inspected, copied, translated, or treated as an oracle. The oblique rights gap is explicit.

A finest-resolution candidate pass is not a mesh-independence certificate.
Review every level, domain/step bounds and applicability flags before interpreting
an error. Incomplete plastic capacity plateaus and failure of quasi-static scale
separation cannot be concealed by a close individual observable. References,
metrics, parameter changes, precision changes, or resource limits change support
and campaign identities and require a new candidate run. Numeric thresholds
require subsequent scientific review and real approval outside this tool before
any approved qualification claim.

## Measured program-worktree example

The following CPU/float64 measurements were collected on October 11, 2026 under
shared-machine contention. They are observations, **not approvals**. Each value
is the finest registered resolution's dimensionless error. Full records retain
every level and causal start/observation/evidence chain.

| Case | Resolution sequence | Metric | Measured error | Candidate bound | Outcome |
| --- | --- | --- | ---: | ---: | --- |
| Lamé thick cylinder | 8, 16, 32, 64 radial cells | stress relative L2 | 0.00832145 | 0.01 | inconclusive |
| Prandtl Tresca footing, G/c=5000 | 8, 16 cells/B; both rates | relative Nc | 0.0759058 | 0.02 | inconclusive |
| Prandtl frictional footing, G/c=5000 | 8, 16 cells/B; both rates | relative Nc | 0.0545413 | 0.02 | inconclusive |
| Hertz normal impact | 256, 512, 1024 steps | relative duration | 7.65464e-7 | 0.005 | passed candidate |
| Hertz normal impact | same | relative peak force | 9.01270e-7 | 0.005 | passed candidate |
| Ergun packed bed | 4, 8 flow samples | relative pressure gradient | 6.66134e-16 | 1e-6 | passed candidate |
| Schiller–Naumann terminal velocity | 20, 30, 40 root iterations | relative terminal speed | 8.26450e-13 | 1e-7 | passed candidate |
| Schiller–Naumann terminal velocity | same | normalized force residual | 1.29319e-12 | 1e-8 | passed candidate |
| Bagnold full coupled μ(I) MPM | 8 (range diagnostic), 16, 32; controls recorded per part | nonsteady profile relative L2 | 0.210095 (finest) | 0.02 | inconclusive: range/work limit |
| Bagnold law-only diagnostic (superseded primary) | 16, 32, 64 depth samples | profile relative L2 | 3.54452e-5 | not a primary criterion | diagnostic only |
| Terzaghi full double-point MPM | 8, 16 physical cells | pressure relative L2 | 0.00347490 | 0.01 | passed candidate |
| Terzaghi full double-point MPM | same | relative settlement | 0.00114013 | 0.02 | passed candidate |
| Wu–Thornton–Li oblique impact | omitted | relative velocity | unavailable | 0.03 | inconclusive: rights gap |

Lamé stress errors decreased **0.0559617 → 0.0282097 → 0.0145260 →
0.00832145**. Reference-error slopes for successive doublings are approximately
0.988, 0.958 and 0.804: observed error reduction, not an asserted asymptotic order.
Equilibrium residuals were **0.0111846, 0.0140486, 0.0105986, 0.0109300**;
they do **not** show monotonic convergence below the unchanged 0.01 admission
bound. The finest stress metric meets its candidate tolerance, but equilibrium
remains inconclusive; close displacement is not substituted for that gate.

With the corrected **G/c=5000** setup, both footing cases complete every
registered step without rejection. Slow-rate Tresca Nc decreases **5.91218 →
5.53187**, versus π+2 = 5.14159; relative errors decrease **0.149873 →
0.0759058**, giving a measured reference-error slope **0.98146**.
Slow-rate frictional Nc decreases **16.45617 → 15.64382**, versus 14.83471;
errors decrease **0.109302 → 0.0545413**, slope **1.00290**.
The latter algebraically implies Nq **6.98956 → 6.69388**, versus 6.39939,
with the same diagnostic slope; it is **not independent Nq measurement**.

Accepted 33-point load–settlement curves and late-window drift/variation are
retained. The slower Tresca curves establish plateaus over **[0.0048,0.006]B**,
well before h/2=0.0625/0.03125B. Slower frictional load curves have late-window
drift 0.00983/0.01783 and variation 0.00803/0.01479; the finer full plateau gate
still fails its strict physical plane-strain threshold (1.279e-10 versus 1e-10).
Fast-rate plateaus are unresolved in both cases, and the finest rate gaps are
0.01929 (Tresca) and 0.06963 (frictional). Mesh gaps 0.07397/0.05476 exceed
0.02. Thus the observed slower-load error reductions are **diagnostic trends**,
not approved mesh-independent ultimate-capacity qualification. No outcome
threshold was relaxed.

The earlier G/c=100, settlement0.05B coarse observations and rejected n16
attempts are kept as superseded configurations. They are not mixed with this
corrected sequence to manufacture a convergence rate. B-bar's assumed
constitutive deformation is not mistaken for physical out-of-plane motion.

The full coupled Bagnold campaign did **not** establish a registered steady
window. The n8 forensic part (dt≤0.004 s, Krylov≤256) reached its documented
current-F scaled-determinant range at **10.06877 s**, with nonsteady profile error
**0.275602**. The n16/n32 numerical refinement kept the original physical support,
using dt≤0.001 s and Krylov≤1024. It measured errors **0.507885/0.210095** at
**16.78106/10.24586 s**, respectively. n16 accepted **21,778 steps**, then reached
the same native range with positive detF≈1.2438 and scaled determinant≈9.982e-16.
n32 accepted **12,600 steps** without numerical rejection, then exhausted the
pre-registered wall bound (1,800.44 s including the last observation block).
Elastic B_e conditions remained about **1.01075/1.00566**. No native gate,
pressure/rate floor, history reset or physical-layer shrink was used.

These nonsteady errors at unequal accepted times are **not a spatial-convergence
or steady-profile claim**; the case is scientifically inconclusive. The original
numerical refusals remain visible alongside that declared-range disposition.
The repaired full-FLIP diagnostic also remained nonsteady: after incremental
pressure repair it lost fixed-layer support at 7.21995/9.68847/15.40781 s for
depth8/16/32, errors 0.613739/0.441238/0.270002. It is preserved as diagnostic
history, not substituted for solver qualification. Neither close local-law
agreement nor an admitted intermediate transaction upgrades these outcomes.

The full Terzaghi solver accepted **795 and 1,697 steps**, respectively, reaching
Tv = 0.2 after 17.58 and 18.77 mixture-wave transits without rejection.
Pressure errors decreased 0.00704595 → 0.00347490, and the finest
load-scaled quasi-static pressure-equilibrium gap was **0.000923237**.
The final settlement errors were 0.000128500 → 0.00114013: they are within the
candidate bound but are **not monotonic**, so these two levels do not establish
settlement-order convergence or a zero-inertia limit. At Tv ≈ 0.05 both pressure
and settlement errors decreased with refinement. No operator-only reduction
was used for these observations.
