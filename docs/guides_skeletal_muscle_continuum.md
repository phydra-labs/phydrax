# Skeletal-muscle continuum mechanics

`phydrax.applications.skeletal_muscle.continuum` provides three distinct source-named continuum owners. The GASAM law introduced by Engelhardt, Sachse, Burgkart, and Wall (2025) remains a terminal, prescribed-activation coupled active/passive material: it is not a passive substrate for a cellular tension model and has no external-tension input. The separate Heidlauf–Röhrle (2014) route below accepts prescribed, already normalized active stress; it is not an integrated cellular model. The Almonacid et al. (2024) route reproduces one pinned idealized muscle–aponeurosis executable case; it is neither an anatomical nor a full musculotendon model.

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

## Heidlauf–Röhrle 2014 prescribed-stress continuum

`HeidlaufRoehrle2014Plan` independently implements Eqs. (1)–(2) and the continuum parameters in Table 2 of Heidlauf T, Röhrle O. *A multiscale chemo-electro-mechanical skeletal muscle model to analyze muscle contraction and force generation for different muscle fiber arrangements*. Front Physiol. 2014;5:498. [doi:10.3389/fphys.2014.00498](https://doi.org/10.3389/fphys.2014.00498). The [publisher XML](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2014.00498/xml) is pinned under `tests/fixtures/heidlauf_roehrle_2014/continuum_source.xml`, with its SHA-256 and CC BY 4.0 attribution in `continuum_manifest.json`.

For the reference unit fiber direction $a_0$, $C=F^TF$, $\lambda=\lVert Fa_0\rVert$, and $I_1=\operatorname{tr} C$, the exact source stress is

$$
S=2c_{10}I+2c_{01}(I_1I-C)
+b_1(\lambda^{d_1-2}-\lambda^{-2})a_0\otimes a_0
+\lambda^{-1}P^{\max}\bar\gamma\,a_0\otimes a_0-pC^{-1},
\qquad P=FS.
$$

`published_table_2()` uses the core unit conversion to obtain
$c_{10}=6.352\times10^{-7}$ Pa, $c_{01}=3627$ Pa,
$b_1=0.02756$ Pa, $d_1=43.373$, and $P^{\max}=73000$ Pa.
These parameters remain trainable numeric leaves, separate from source, topology and executable identities.

**No compression cutoff is present in the published Eq. (1).** The anisotropic term is retained for every positive stretch, including its negative value below $\lambda=1$; the word “stretches” in the source discussion does not introduce a tension-only branch. The implementation uses unprojected $F$, not a GASAM-style isochoric energy argument. Its independently integrated passive reference energy is

$$
W_{\rm passive}=c_{10}(I_1-3)+c_{01}(I_2-3)
+b_1\left[\frac{\lambda^{d_1}-1}{d_1}-\log\lambda\right].
$$

This energy satisfies $\partial W_{\rm passive}/\partial F=P_{\rm passive}$.
The active contribution is imposed as
$P_{\rm active}=P^{\max}\bar\gamma(Fa_0/\lambda)\otimes a_0$;
no additional force–length/velocity factor or clipping is applied.
The prescribed $\bar\gamma$ is finite and signed, not a bounded activation:
Eq. (6) neither bounds it by one nor supplies a nonnegative clamp.
Its work is $P_{\rm active}:\dot F=P^{\max}\bar\gamma\dot\lambda$.
The active source can depend on velocity/history and is not included in stored passive energy or differentiated as a conservative coupled potential.

### Pressure, input transaction and native FE interface

- Pointwise `evaluate(F, pressure_pa, normalized_active_stress=None)` returns first/second Piola and Cauchy stresses, passive energy, active nominal stress, and domain evidence. **Unlike GASAM, positive source pressure is compressive.** Its contribution is exactly $-pF^{-T}$; off the incompressibility surface the Cauchy pressure term is $-(p/J)I$.
- The native mixed residual uses $\int P:\nabla_Xv\,dV$ and $\int-\log J\,q\,dV$. Thus its pressure and constraint derivative blocks are exactly adjoint even away from $J=1$. This is an exact-incompressibility numerical realization, not a bulk penalty; it has the same constraint $J=1$ and preserves the source pressure term at trial states.
- `form()` returns core `FiniteElementForm`/`CellResidualAction` objects, and `prepare_qualified_mixed()` consumes a core `MixedFiniteElementConstraintPlan`, returning `PreparedMixedFiniteElementConstraint` with Taylor–Hood/Q2–Q1, gauge and assembled inf-sup evidence. There is no parallel assembler or nonlinear solver.
- FE pressure is an increment with the explicitly bound `pressure_origin_pa`: $p_{\rm source}=p_{\rm FE}+p_{\rm origin}$. A core mean-zero/pinned increment gauge must be chosen consistently with the boundary/preload problem. At passive stress-free rest, use $p_{\rm origin}=2c_{10}+4c_{01}$, not zero. This offset is part of the form identity; it is not a hidden source-law change.
- `HeidlaufRoehrle2014StressInput(gamma, source_state_token, source_id)` carries prescribed $\bar\gamma$, eight `uint32` words of the supplying state/protocol sample's content digest, and a `source_successful` gate. `propose_active_stress(input).commit(successful=outer_gate)` selects the whole input, source receipt, evidence and accepted-update count, or rolls all of them back. `with_commit()` rejects stale/foreign material states and changed parameter revisions.
- `first_piola_points(F, p, gamma)` accepts shapes `(..., 3, 3)`, `(...)`, `(...)` with exact coverage. `block_tangent()` returns $dP/dF$, $dP/dp$, $dP/d\bar\gamma$ and the constraint deformation derivative. Gamma remains an independent constitutive input, suitable for a separately qualified coupling root.
- Nonuniform prescribed stress uses `HeidlaufRoehrle2014ActiveStressField(callback, provenance, numeric_revision, source_id=...)`. Its callback receives `(reference_quadrature_points, execution_context)`, reads dynamic `execution_context.user_args`, and returns one gamma value per quadrature point with exactly the points' leading shape. Source identity must match the material plan; the fixed mapping, support, reference/current frames and algorithm belong to its core `SemanticProvenance` and `NumericRevision`. The callback owns source/coverage/state-receipt checks, never mechanical force. Failed source evaluation must not produce an admissible finite replacement stress.

The point interface and native FE residual are usable independently with caller-prescribed source stress. They do **not** close the 2014 non-isometric cell source gate, moving-fiber diffusion, conservative transfer, source-staggered oracle or implicit electromechanical transaction. The acquired cellular artifacts disagree with the 2014 paper; no integrated or physiological validation claim follows from material consistency.

Run the separate numerical drivers:

```console
python examples/heidlauf_roehrle_2014_continuum.py
python tools/heidlauf_roehrle_2014_continuum_qualification.py
python benchmarks/heidlauf_roehrle_2014_continuum.py --smoke
```

They report passive energy/stress consistency, objectivity, finite-difference tangent agreement, active virtual power, native mixed manufactured rest and constitutive rollback. The integration regression also exercises a spatially varying prescribed gamma field and its FE source derivative against exact affine virtual work. None is held-out biological validation or a source-model contraction replay.

## Almonacid 2024 idealized muscle–aponeurosis continuum

`Almonacid2024MuscleAponeurosisPlan` independently implements the pinned repository equations and procedural idealized medial-gastrocnemius geometry associated with Almonacid et al., *A Three-Dimensional Model of Skeletal Muscle Tissues*, SIAM J. Appl. Math. 84 (2024), DOI [`10.1137/22M1506985`](https://doi.org/10.1137/22M1506985). Executable identity is commit `0698e3d87d7261c81437d410dda161fe3b8efcbf`. The LGPL-2.1 source license, immutable inputs, hashes, build/runtime records, three 101-step external numerical campaigns, and material-point oracle inventory are under `tests/fixtures/flexodeal_0698e3d` and `benchmarks/skeletal_muscle_source_evidence.json`.

The native force owner is one three-field weak residual:

- continuous vector Q2 displacement;
- four cell-discontinuous total-degree-one monomial pressure modes;
- four matching dilation modes;
- fifth-order Gauss volume and face quadrature;
- source backward-Euler inertia and deformation-rate update;
- perfectly tied conforming muscle/aponeurosis interfaces;
- face 0 fixed, source pulling-face x displacement prescribed, and all other exterior faces traction-free.

The repository material constants are used exactly. They differ from the coefficients printed in the publication table and are never silently exchanged. Muscle and aponeurosis branches, non-unit deformed fiber orientation, pressure sign, active force–length/velocity branches, and mixed residuals remain those of the executable. `Almonacid2024MaterialParameters` and `almonacid_2024_material_response()` are also public for independently qualified point evaluations.

`almonacid_2024_repository_case()` verifies the pinned input manifest during host preparation and returns the Plan, trainable physical parameters, content-identified control history, time step, and terminal time. It requires JAX float64. The source activation table reaches `1.000004`; finite values are retained exactly rather than clipped to an invented `[0, 1]` support. Control amplitudes remain differentiable; time grids, source identity, topology, and accepted state are nontrainable. A candidate can be installed only with `candidate.commit(current_prepared)`, which checks the complete originating state, parameters, geometry, quadrature, topology, and numerical policy. Rejection selects the entire prior preparation. Accepted trajectories lock the physical law; changing density or material parameters requires replay or re-preparation.

```python
import equinox as eqx
import jax
from phydrax.applications.skeletal_muscle import continuum

with jax.enable_x64(True):
    plan, parameters, history, dt, _ = continuum.almonacid_2024_repository_case(
        "tests/fixtures/flexodeal_0698e3d"
    )
    prepared = plan.prepare(parameters)
    candidate = eqx.filter_jit(lambda model, control: model.propose(control))(
        prepared, history.sample(dt)
    )
    prepared = candidate.commit(prepared)
```

The true Newton Jacobian stays matrix-free. Setup differentiates only bounded material-point and element blocks and exactly condenses each eight-mode pressure/dilation block. On CPU, `almonacid_2024_repository_case()` binds the native JIT-compatible `SparseLU(provider="jax-cpu")` action to the condensed sparse mechanical Schur system; a directly constructed Plan defaults to the more robust full mixed sparse action. Non-CPU dynamic Plans retain bounded element-Schwarz mechanics, while quasistatic fallback uses fixed-pattern sparse triangular corrections. Implicit forward and reverse derivatives bind setup to the accepted root; each reverse action is the exact coordinate transpose of its forward preconditioner, not a stale Newton setup.

Run:

```console
python examples/almonacid_2024_continuum.py --refinement 0 --steps 2
python tools/almonacid_2024_material_qualification.py
python tools/almonacid_2024_continuum_qualification.py --output qualification.json
python tools/almonacid_2024_refinement_qualification.py --output refinement.json
python benchmarks/almonacid_2024_continuum.py --smoke --output benchmark.json
```

The material qualifier compares 95 values generated by the unmodified pinned C++ material implementation. The continuum qualifier compares native states, quadrature fields, stress components, source nearest-quadrature-point boundary forces, and source diagnostic contractions against executed external outputs. These are cross-implementation numerical checks. They do not establish biological validation, an inf-sup lower-bound theorem, anatomical fidelity, or a full musculotendon model.

## Tendon and anatomical MTA boundary

The Almonacid route contains two idealized aponeurosis sheets and no tendon domain. No coherent same-anatomy mesh/imaging, preload, tendon properties, loads, and held-out deformation bundle is available. Consequently Phydrax does not label this route anatomical or full MTA, and does not attach a generic tendon or synthetic interface to manufacture that claim.
