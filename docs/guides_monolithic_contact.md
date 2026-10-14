# Monolithic implicit contact and affine bodies

Phydrax enables x64 by default; the launch recipes explicitly select it.

Monolithic contact solves one shared physical residual over an ordered tuple of
mechanics blocks. It is neither partitioned force exchange nor an impulse or
penalty method. Geometry, mechanics, search, safety, and accepted history remain
separate owners. The repository example is a bounded affine-body/wall case, not
a release qualification or evidence for arbitrary scenes.

## Shared residual and exact linearization

Let $q=(q_1,\ldots,q_n)$ and let $h_n$ be accepted histories. A geometry surface
has positions $x_i(q_i)$, tangent $J_i(q_i)$, and algebraic transpose pullback.
Using contact and friction residual covectors $g_c$ and $g_f$,

$$R(q)=R_m(q;q_n,h_n,t,\Delta t)+J_c(q)^Tg_c(x(q))+J_s(q)^Tg_f(q)=0.$$

Physical surface forces have the opposite sign to these residual covectors.
For any direction $v$, the contact action includes

$$DR_c(q)[v]=(DJ_c(q)[v])^Tg_c+J_c(q)^TDg_c(x(q))[J_c(q)v].$$

Rebuild the kinematic linear operator at every Newton base state. Omitting its
curvature term for nonlinear geometry changes the Jacobian. Contact also creates
mixed blocks $K_{ij}$ between different mechanics participants; solving each
participant separately omits these blocks. The product space preserves each
participant's native PyTree without dense global Hessian assembly.

`AbstractImplicitContactParticipant` owns `evaluate`, `residual_jvp`, `potential`,
`potential_hvp`, and `step_limit`. Evaluation derives trial history from the same
accepted history on every call; it must not commit material state or solve a
private inner mechanics root. `AbstractContactParticipant` is instead a geometry
map. In particular, `LinearContactParticipant(PreparedCollisionSurface)` supplies
geometry only, not mass or constitutive residuals.

A `ContactParticipantScene` may contain several surfaces owned by one mechanics
block. `geometry_owner_indices` binds each scene surface to an integer block, or
`None` for a fixed wall. Every `None` entry requires an explicit state in
`fixed_geometry_states`; owned entries use `None` there. Coordinates and histories
are ordered tuples of mechanics blocks, not tuples of surfaces.

## Residual merit and the potential gate

`ContactSolvePolicy(merit=..., method=..., termination=...,
maximum_lag_iterations=..., lag_tolerance=...)` selects native `NewtonKrylov`
with GMRES and `RootLineSearch` by default. `NonlinearTermination` owns nonlinear,
step, and linear-work tolerances. No symmetry projection is applied.

- `"automatic"` selects potential merit only when every active contribution is
  potential-compatible; otherwise it uses physical residual merit.
- `"potential"` requires verified consistency for every active contribution.
- `"residual"` uses the scaled physical residual norm.

Potential availability is not proof that its derivative equals the physical
residual. Participant evidence retains `potential_consistency_checked` and
`potential_consistency_defect`. Non-associated Drucker–Prager/Mohr–Coulomb updates
can have nonsymmetric tangents and must not be relabeled as energy gradients.
Frozen-frame friction does not make such a system conservative.

## Affine-body derivation

A 3-D affine body has four control vectors $y_a$, hence exactly 12 DOFs. The
unknown is control displacement $q=y-y_{\rm reference}$. For local material
coordinates $X$, define

$$N(X)=(1-X_1-X_2-X_3,X_1,X_2,X_3),\qquad x(X)=\sum_{a=0}^3N_a(X)y_a.$$

Caller-supplied reference-volume quadrature has positive weights $w_r$ and shape
rows reproducing constants. With $B_r=N(X_r)\otimes I_3$,

$$T=\frac12\sum_r\rho w_r\|B_r\dot y\|^2
 =\frac12\dot y^T M\dot y,\quad
 M=B^T\operatorname{diag}(\rho w_r I_3)B=M_{\rm control}\otimes I_3,$$

$$[M_{\rm control}]_{ac}=\sum_r\rho w_rN_{ra}N_{rc}.$$

Thus the consistent mass includes cross-control terms; a guessed diagonal mass
is not the same model. This is the control-basis change of Lan et al.'s constant
mass integral $\int_\Omega\rho J(X)^TJ(X)\,dX$ (equations 3–4).

Let $D(y)=[y_1-y_0,y_2-y_0,y_3-y_0]$ and
$F=D(y)D(y_{\rm reference})^{-1}$, implemented by a small solve. With reference
volume $V=\sum_r w_r$ and rigidity modulus $E$,

$$C=F^TF-I,\qquad E_r=\frac{VE}{8}\|C\|_F^2,\qquad
\frac{\partial E_r}{\partial F}=\frac{VE}{2}FC.$$

Lan et al. equation 6 uses $\kappa V\|AA^T-I\|_F^2$. For square matrices,
$\|AA^T-I\|_F^2=\|A^TA-I\|_F^2$; Phydrax's normalization corresponds to
$\kappa=E/8$ with $A=F$. The polynomial energy and tangent require no eigenvalue
or singular-value decomposition and remain finite at $F=I$. Proper rotations
have zero rigidity energy, but reflections do too: the separate strict
$\det F>0$ domain bound is essential, not an energy penalty or determinant repair.

`AffineBodyPlan` takes topology, reference controls `(B,4,3)`, local material
coordinates `(vertices,3)`, local vertex body indices, quadrature shape values `(B,Q,4)`,
weights `(B,Q)`, density `(B,)`, and rigidity modulus `(B,)`; gravity is a keyword.
`prepare_affine_body_participant` binds Newmark dynamics and geometry.
`AffineBodyHistory(velocity, acceleration)` contains arrays `(B,4,3)`.

`AffineBodyPlan.vertex_body_ids` indexes **local control rows** in `[0, B)`.
By contrast, `topology.feature_policy.body_ids` contains **global scientific
contact identities**, which need not equal those local indices numerically.
The explicit per-vertex relation must assign one homogeneous global body ID to
each local row, with distinct global IDs for different rows. Do not infer
scientific ownership from matching numeric labels or array shapes.

For Newmark parameters $\beta,\gamma$, define
$y_{\rm pred}=y_n+\Delta t\,v_n+(1/2-\beta)\Delta t^2a_n$,
$a_{n+1}=(y-y_{\rm pred})/(\beta\Delta t^2)$, and
$v_{n+1}=v_n+\Delta t[(1-\gamma)a_n+\gamma a_{n+1}]$.
The body residual combines $Ma_{n+1}$, rigidity gradient, gravity load, and
shared contact. Static equilibrium omits inertia.

## Barrier, fixed envelope, and continuous domain safety

The canonical physical clamped-log barrier uses squared clearance coordinate
$s=d^2-d_{\min}^2$ and activation threshold
$\hat s=(2d_{\min}+\hat d)\hat d$. It vanishes outside activation and diverges
as $s\downarrow0$; reference-space measures and stencil corrections remain
owned by `ConvergentContactPotentialPlan`. `stiffness` has pressure units.
Participant-scene preparation requires explicit `rest_positions`.

For $0<s<\hat s$, $b(s,\hat s)=-(s-\hat s)^2\log(s/\hat s)$;
it is zero for $s\ge\hat s$ and infinite for $s\le0$. The physical length-valued
barrier is $\bar b=\hat d\,b/\hat s^2$, so pressure times area times $\bar b$
has energy units. Its interior derivatives are

$$b'=(\hat s-s)[2\log(s/\hat s)-\hat s/s+1],\qquad
b''=(\hat s/s+2)\hat s/s-2\log(s/\hat s)-3.$$

The smoothly clamped barrier originates in Li et al. §4.2; squared clearance,
physical scaling, and reference-measure weighting above state the actual
Phydrax convention rather than identifying it with the paper's notation.

Candidate search is an explicit host preparation boundary. A finite-capacity
`ContactCandidateEpoch` is frozen through Newton and friction refresh, including
route order. Its conservative motion envelope must cover every possible route
for the whole step. Bounds prevent leaving that envelope. Overflow, incomplete
coverage, memory/work exhaustion, or unsafe geometry refuses the complete step;
prepare a larger fixed plan at the next accepted boundary, never grow capacity
inside a compiled solve.
Empty primitive sets (including isolated-vertex surfaces without edges) produce
complete empty candidate batches at the declared fixed capacities.

A barrier or positive endpoint gap does not certify the traversed path. Every
line search starts at

$$\alpha\le\min(1,\alpha_{\rm CCD},\alpha_1,\ldots,\alpha_n),$$

where participant limits protect material domains, simplex orientation, affine
orientation, and the envelope. A zero, nonfinite, negative, or greater-than-one
bound refuses the direction. CCD work exhaustion does not permit a fallback
endpoint test. Linear affine/FEM/frozen-MPM paths use the existing continuous
bounds. Nonlinear rod/IGA maps are admitted when geometry declares
`NONLINEAR_TRAJECTORY` with a conservative full-path bounds provider.

Degenerate CCD primitives fail closed rather than using masked distance
sentinels to certify separation. Affine-body inversion margins scale with the
reference control-frame measure, so changing length units does not invalidate
an otherwise admissible undeformed body.

`collision_free_step_limit` can receive dynamic
`trajectory_bounds(lower_time, upper_time)` returning scene lower/upper sample
enclosures and success evidence. Its bounded depth-first interval subdivision
checks entire intervals, not interpolated endpoints. The root also enforces the
search envelope by bisection on certified full-path enclosures. Missing providers
refuse the solve.

`CertifiedAABBCCDPlan` with nonlinear geometry is refused: these enclosures do not
carry its stronger rounding certificate. Use the admitted enclosure guarantee,
and inspect the returned evidence rather than claiming stronger safety.
`prepare_implicit_reduced_rod_contact_geometry(existing_reduced_capsule_participant,
bounds_action=provider)` binds an existing rod geometry to that explicit provider.
IGA geometry uses its existing proxy plan and source space with a bounded
provider; a static proxy-error certificate alone is not a trajectory certificate.

### Native globalization hook transport

`NonlinearSystemProblem` accepts optional `potential`/`potential_id` and
`step_limit`/`step_limit_id` callback pairs. Both members of each pair are
required together; leaving them absent preserves residual Armijo behavior.
`RootLineSearch` reports `potential-armijo` or `residual-armijo` provenance.
Potential Armijo uses the physical potential and its directional JVP. A finite
directional derivative must be strictly negative; an ascent or stationary
direction is refused rather than permitting an energy-increasing Armijo step.
Domain-invalid trials retain recoverable-domain failure evidence and are not
counted as nonfinite potential evaluations.

Diagonal scaling transports both state and direction to the physical hook;
left preconditioning preserves the physical potential and safety hooks. An
arbitrary nonlinear right reconstruction is refused when a declared bound
certifies a linear physical trajectory. Mixed nonlinear execution likewise
refuses state-dtype conversion with such a bound: rounding can change the
certified path. These refusals preserve the domain certificate rather than
silently applying it to a different trajectory.

Geometry callback numerical state is intentionally dynamic PyTree data rather
than static plan metadata. This changes callback/plan partition and compilation
identity; it is not an identity-preserving compatibility layer.

## Lagged friction and atomic acceptance

Each inner solve freezes route keys, closest coefficients, tangent bases, and
compressive normal-force magnitudes. For frozen frame $T_k$, normal force
$\lambda_k$, coefficient $\mu_k$, and regularized slip norm $\phi$,

$$D_{\rm lag}=\sum_k\mu_k\lambda_k\phi(\|T_k^Tv_{\rm rel}\|).$$

With `velocity_threshold` $\epsilon>0$, the canonical regularization is

$$\phi(r)=\begin{cases}\epsilon/3+r^2/\epsilon-r^3/(3\epsilon^2),
&0\le r<\epsilon,\\r,&r\ge\epsilon.\end{cases}$$

The force is the negative velocity gradient of $D_{\rm lag}$, pulled back through
the current slip map. Its derivative is zero at zero slip; above threshold its
magnitude is $\mu\lambda$. The additive constant in $\phi$ does not create a
nonzero force at rest.

Its position scaling follows the incremental integrator. Refresh frames only
after a complete root solve, then test the lag fixed-point residual. Li et al.
explicitly do not guarantee lag convergence; finite iteration exhaustion here
rejects the transaction rather than silently accepting frictionless motion.
Inspect force, dissipation, frame/route, and lag evidence. Friction is rejected
for static equilibrium preparation.
Nonlinear geometry disables friction potential merit: `"automatic"` resolves to
residual merit even when the mechanics otherwise has a potential. This avoids
claiming a frozen-frame energy derivative reproduces the nonlinear pulled-back
friction residual.

The public lifecycle is:

```python
accepted = initialize_monolithic_contact_state(plan, coordinates, histories)
prepared = prepare_monolithic_contact_step(plan, accepted, dt, args=arguments)
result = solve_monolithic_contact_step(prepared)
accepted = result.commit()
```

`MonolithicContactPlan(participants, scene, contact, search, ccd,
geometry_owner_indices=..., fixed_geometry_states=..., friction=..., policy=...)`
owns static preparation. `result` retains previous, candidate, accepted state,
the frozen candidate epoch, participant/contact/friction/safety evidence,
diagnostics, and rejection reasons. `commit()` accepts all blocks and histories
only if every gate passes; otherwise it returns the previous state.
`rollback()` always returns that exact previous state. Time, step, state version,
and friction history are never partially advanced.
`accepted_state` retains the device-selected, staged FEM trial transaction.
Use `result.commit()` before preparing the next material-bearing FEM step:
this host lifecycle boundary promotes accepted material trials, including static
material revisions, while participant counters advance with acceptance.
Rejected solves retain the previous FEM material history and counters.

`prepare_monolithic_contact_equilibrium` and `solve_monolithic_contact_equilibrium`
use the same assembly and transaction without inertia; acceptance leaves time
and step unchanged while advancing state version.

## Local derivatives and runnable evidence

`monolithic_contact_solution_jvp(plan, result, arguments, tangent_arguments)` and
`monolithic_contact_solution_vjp(plan, result, arguments, cotangent_state)` take
argument PyTrees and differentiate the accepted root on the retained epoch, not
search, capacity changes, CCD decisions, line-search iterations, or lag refresh.
Successful primal and linear solves plus route/feature/domain margins are
required; a branch change invalidates a local derivative certificate.

Run `JAX_ENABLE_X64=1 python -m examples.monolithic_contact`. The example uses one
bounded tetrahedral affine body and a fixed triangular wall, supplied degree-two
volume quadrature, and explicit acceptance evidence. It is not a stack, free-spin,
mixed MPM/FEM, or friction qualification. `benchmarks/monolithic_contact.py`
separates preparation/search, participant residual, contact/pullback, CCD/domain
bounds, JVP, native Newton, friction refresh, and full-step timing. Capacity bytes
and compiler memory estimates are distinguished from measured peak device memory.

## Primary references

- Lei Lan, Danny M. Kaufman, Minchen Li, Chenfanfu Jiang, and Yin Yang (2022),
  [Affine Body Dynamics: Fast, Stable and Intersection-free Simulation of Stiff Materials](https://arxiv.org/pdf/2201.10022),
  §§3.1–3.2, equations 2–6; [ACM DOI](https://doi.org/10.1145/3528223.3530064).
- Minchen Li, Zachary Ferguson, Teseo Schneider, Timothy R. Langlois, Denis Zorin,
  Daniele Panozzo, Chenfanfu Jiang, and Danny M. Kaufman (2020),
  [Incremental Potential Contact: Intersection- and Inversion-free, Large-Deformation Dynamics](http://web.uvic.ca/~teseo/profile/publications/ipc/2020-IPC.pdf),
  §§4.2, 4.4, 5.2; [ACM DOI](https://doi.org/10.1145/3386569.3392425).
