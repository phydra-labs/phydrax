# Physics-Constrained Interpolation (overlay pipeline)

This appendix formalizes the mathematics behind Phydrax’s **Physics-Constrained Interpolation (PCI)** enforced overlay pipeline:

1. boundary enforced constraints (possibly piecewise, via blending),
2. initial constraints (possibly higher order in the evolution variable, and gated to preserve boundary constraints),
3. interior *exact* data satisfaction via an anchor/data overlay stage while preserving boundary and initial constraints.

Here “PCI” refers to the *entire* staged enforcement map $u\mapsto \tilde u$; the final stage is the
interior anchor/data overlay.

The implementation corresponds to the staging performed by `EnforcedConstraintPipeline` / `EnforcedConstraintPipelines`,
the enforced ansätze in `phydrax.constraints` (e.g. Dirichlet/Neumann/Robin), the BVH-accelerated weight construction
used for boundary blending, and the IDW-based interior anchor overlay.

## A.0. Setting and notation

Let the computational domain be a product:

$$
\mathcal D \;=\; \prod_{\ell\in\mathcal L}\mathcal D_\ell,
$$

where $\mathcal L$ is a finite set of *labels* (e.g. $\mathcal L=\lbrace x,t\rbrace$ for space–time).
We write a point as $z=(z_\ell)_{\ell\in\mathcal L}$.

A field is a map $u:\mathcal D\to \mathbb R^C$. Phydrax represents a parameterized base field
$u_\theta$ (a `DomainFunction`) and produces an *enforced* field $\tilde u_\theta$ by applying
a staged transformation:

$$
\mathcal P:\ u \mapsto \tilde u.
$$

We consider three classes of enforced requirements:

- **Boundary constraints**: conditions on a boundary subset $S_B\subset \mathcal D$, typically
  $S_B=\partial\Omega\times \prod_{\ell\neq x}\mathcal D_\ell$ for a geometry factor $\Omega$.
- **Initial constraints**: conditions on a fixed slice $S_I=\lbrace t=t_0\rbrace\times \prod_{\ell\neq t}\mathcal D_\ell$.
  Higher-order initial constraints fix $\partial_t^k u(\cdot,t_0)$ for $k\le K$.
- **Interior data**: anchor requirements $\tilde u(z_i)=y_i$ for prescribed interior points $z_i\in\mathcal D$,
  optionally including time-dependent tracks $\tilde u(x_m,t)=y_m(t)$ for sensors $x_m$.

The central design goal is **constraint preservation by construction**:
later stages must not re-violate earlier enforced constraints.

## A.1. Enforced constraints as constraint-preserving operators

Let $\mathcal F$ be a function space over $\mathcal D$. An enforced constraint defines a subset
$\mathcal C\subset\mathcal F$ (e.g. functions satisfying Dirichlet boundary conditions).

An operator $\mathcal T:\mathcal F\to\mathcal F$ is **$\mathcal C$-preserving** if:

$$
u\in\mathcal C\quad\Rightarrow\quad \mathcal T(u)\in\mathcal C.
$$

The overlay pipeline is a composition:

$$
\tilde u \;=\; \mathcal T_3\bigl(\mathcal T_2(\mathcal T_1(u))\bigr)
$$

where:

- $\mathcal T_1$ enforces boundary constraints (possibly piecewise-blended),
- $\mathcal T_2$ enforces initial constraints but is designed to be boundary-preserving,
- $\mathcal T_3$ enforces interior data while preserving boundary and initial constraints
  (including derivative constraints up to specified orders).

The remaining sections specify concrete constructions and the invariance proofs.

## A.2. Boundary enforced ansätze

Let $\Omega\subset\mathbb R^d$ be a geometry factor with boundary $\partial\Omega$.
Assume a signed distance–like function $\phi:\Omega\to\mathbb R$ satisfying $\phi=0$ on $\partial\Omega$,
and an outward unit normal $n$ on $\partial\Omega$.

The statements below are written in an ideal smooth setting; in practice Phydrax uses an
approximate distance function and numerically defined boundary normals. Exactness claims
should be interpreted as exact for the idealized $\phi,n$, and “up to numerical tolerance”
in typical discretized settings.

### A.2.1. Dirichlet (value) constraints

Given a target $g:\partial\Omega\to\mathbb R^C$, define:

$$
u^\star(x)\;=\; g(x) + \phi(x)\,\bigl(u(x)-g(x)\bigr).
$$

**Proposition A.1 (Dirichlet exactness).** If $\phi=0$ on $\partial\Omega$, then $u^\star=g$ on $\partial\Omega$.

*Proof.* For $x\in\partial\Omega$, $\phi(x)=0$ implies $u^\star(x)=g(x)$. $\square$

### A.2.2. Neumann (normal derivative) constraints

Given a target $g:\partial\Omega\to\mathbb R^C$ for $\partial_n u=g$, define:

$$
u^\star \;=\; u + \frac{\phi}{\partial_n\phi}\,\bigl(g-\partial_n u\bigr).
$$

**Proposition A.2 (Neumann exactness).** Assume $u,\phi$ are differentiable and $\partial_n\phi\neq 0$ on $\partial\Omega$.
Then $\partial_n u^\star=g$ on $\partial\Omega$.

*Proof.* Differentiate along $n$:

$$
\partial_n u^\star
=\partial_n u + \partial_n\left(\frac{\phi}{\partial_n\phi}\right)\bigl(g-\partial_n u\bigr)
+ \frac{\phi}{\partial_n\phi}\bigl(\partial_n g - \partial_n\partial_n u\bigr).
$$

On $\partial\Omega$, $\phi=0$ annihilates the last term. Moreover,

$$
\partial_n\left(\frac{\phi}{\partial_n\phi}\right)
=\frac{\partial_n\phi}{\partial_n\phi}+\phi\cdot(\cdots)=1
\quad\text{on }\partial\Omega.
$$

Hence $\partial_n u^\star=\partial_n u + (g-\partial_n u)=g$ on $\partial\Omega$. $\square$

### A.2.3. Robin (mixed) constraints

For $a\,u + b\,\partial_n u = g$ on $\partial\Omega$, one may use the analogous correction:

$$
u^\star \;=\; u + \frac{\phi}{b\,\partial_n\phi}\,\bigl(g-a\,u-b\,\partial_n u\bigr),
$$

under the nondegeneracy assumption $b\,\partial_n\phi\neq 0$ on $\partial\Omega$.
The proof is the same as Proposition A.2: the correction term vanishes on the boundary but has
the correct first normal derivative to cancel the residual of the Robin operator.

### A.2.4. Approximate distance field (ADF) and blur

PCI relies on a signed distance–like function $\phi$ to build boundary factors and normals.
Phydrax constructs a smooth **approximate distance field (ADF)** from a mesh and, by default,
applies a local Gaussian blur to improve stability of downstream derivatives.

#### A.2.4.1. Base smooth distance (mesh ADF)

Let $\mathcal M$ be a triangulated surface defining $\partial\Omega$. For each query point $x$,
Phydrax computes a smooth proxy distance by combining candidate triangle distances with a soft-min:

$$
d(x)\;\approx\;\operatorname{softmin}_\beta\{\,d_T(x)\,\}_{T\in\mathcal C(x)},
$$

where $d_T(x)$ is the point-to-triangle distance and $\mathcal C(x)$ is a BVH-selected candidate
set (beam traversal of a packed AABB tree). The softness parameter $\beta>0$ controls the
sharpness of the soft-min; smaller $\beta$ yields a smoother blend of nearby triangle distances.

For 2D geometries built from CAD, Phydrax extrudes the mesh to 3D and evaluates the 3D ADF away
from end caps so that the resulting distance coincides with the 2D boundary distance on the
side walls.

An optional **squash** transform can be applied:

$$
\rho(s)=\delta\,\operatorname{tanh}\!\left(\frac{s}{\delta}\right),
$$

which is linear near $s=0$ but saturates in the interior to limit curvature. The *unsquashed*
field (pre-$\operatorname{tanh}$) is exposed as `adf_orig`.

#### A.2.4.2. Local ADF blur

Given a base field $\phi_{\text{orig}}$ (typically `adf_orig`), the blurred ADF is defined by
sampling in a local neighborhood:

$$
\phi_{\text{blur}}(x)
=
\frac{\phi_{\text{orig}}(x)+\sum_{i=1}^m w_i(x)\,\phi_{\text{orig}}(x+r(x)u_i)}
{1+\sum_{i=1}^m w_i(x)}.
$$

Here $(u_i)$ are deterministic offsets on the unit disk (2D Fibonacci disk) or unit ball
(3D Fibonacci sphere with radial stratification). The radius is adaptive:

$$
r(x)=\max\!\bigl(\sqrt{\phi_{\text{orig}}(x)^2+\varepsilon^2}-\varepsilon,\ r_{\min}\bigr),
$$

and the Gaussian weights are

$$
w_i(x)=\exp\!\left(-\frac{\|r(x)\,u_i\|^2}{2(\sigma\,r(x))^2+\varepsilon}\right),
$$

with $\sigma=\texttt{sigma_scale}$. The parameters $(\varepsilon,r_{\min})$ prevent numerical
instabilities and collapse of the kernel near the boundary.

**Implementation note.** By default, `geom.adf` is set to the blurred field
$\phi_{\text{blur}}=\texttt{adf_blur}(\texttt{adf_orig})$, while `adf_orig` provides the
unsquashed base field. Thus enforced boundary ansätze use the blurred ADF by default.

## A.3. Piecewise boundary constraints and blending

Boundary conditions are commonly specified piecewise on disjoint boundary subsets
$\Gamma_1,\dots,\Gamma_m\subset\partial\Omega$. For each $\Gamma_i$ one can build an ansatz
$u_i^\star=\mathcal H_i(u)$ that satisfies the desired condition on $\Gamma_i$.
The pipeline then combines them into a single enforced field by weighted blending:

$$
u_B(x)
=
\frac{\sum_{i=1}^m w_i(x)\,u_i^\star(x)\;+\;w_{\text{rem}}(x)\,u(x)}
{\sum_{i=1}^m w_i(x)\;+\;w_{\text{rem}}(x)}.
$$

The optional remainder weight $w_{\text{rem}}$ is supported on the complement of the union
of the boundary subsets and prevents subset constraints from “leaking” to other segments.

### A.3.1. Exactness under weight dominance

To make the blend exact on each piece, it suffices that the weights *dominate* near the corresponding subset.

**Assumption A.1 (dominant weights).** For each $k$, $w_k(x)\to +\infty$ as $x\to\Gamma_k$, while every
$w_j$ with $j\neq k$ and $w_{\text{rem}}$ remain bounded in a neighborhood of $\Gamma_k$.

**Proposition A.3 (piecewise exactness).** Under Assumption A.1 and continuity of $u_i^\star$, the blended field has
the same boundary trace as the dominant piece on $\Gamma_k$ (i.e. $u_B\to u_k^\star$ as $x\to\Gamma_k$ away from
junction sets where multiple pieces meet):

$$
u_B|_{\Gamma_k} = u_k^\star|_{\Gamma_k}.
$$

Consequently, if $u_k^\star$ satisfies the desired enforced boundary condition on $\Gamma_k$, so does $u_B$.

*Proof.* Write $u_B=(w_k u_k^\star + R)/(w_k+r)$ where $R=\sum_{j\neq k}w_j u_j^\star+w_{\text{rem}}u$
and $r=\sum_{j\neq k}w_j+w_{\text{rem}}$. By Assumption A.1, $R/w_k\to 0$ and $r/w_k\to 0$ as $x\to\Gamma_k$.
Thus $u_B\to u_k^\star$ as $x\to\Gamma_k$ (and hence their boundary traces agree). $\square$

### A.3.2. How the weights are constructed (MLS + BVH)

In Phydrax, each $w_i$ is derived from a distance-to-subset proxy $\rho_i(x)\ge 0$ with $\rho_i=0$ on $\Gamma_i$,
typically via an inverse-square law $w_i(x)\propto (\rho_i(x)+\varepsilon)^{-2}$.

The proxy $\rho_i$ is computed from a dense reference sample $P=\lbrace p_j\rbrace\subset\Gamma_i$ and associated outward normals
$\lbrace n_j\rbrace$. For a query point $x$, the implementation computes an oriented MLS projection distance

$$
f(x)=\sum_{j\in\mathcal N(x)} \alpha_j(x)\,\langle n_j,\,x-p_j\rangle,
$$

where $\alpha_j(x)$ are nonnegative weights concentrating on nearby points and penalizing normal mismatch.
Then $\rho(x)$ is obtained from $f(x)$ by a smooth nonnegative transformation (a softplus-based absolute distance).

A naive MLS evaluation would require scanning all reference points. Instead, neighbor candidates $\mathcal N(x)$
are chosen using a static AABB BVH (bounding volume hierarchy) built over $P$. The BVH provides a fast approximate
nearest-neighbor primitive: it restricts the MLS sum to a candidate set. If that candidate set contains all points
with non-negligible kernel weight (in a chosen tolerance sense), the BVH-accelerated estimate approximates the full
MLS distance; in the limit of exhaustive search (beam width $\to\infty$), it matches the full neighborhood evaluation.

### A.3.3. Junction sets and compatibility

The “dominant weight” argument above is valid away from points where multiple pieces touch. Define the junction set:

$$
J \;=\; \bigcup_{i\neq j}\left(\overline{\Gamma_i}\cap\overline{\Gamma_j}\right).
$$

At points of $J$, distances (and thus weights) for two or more pieces may vanish simultaneously, so no single piece
need dominate. Rigorous *everywhere* exactness therefore requires additional compatibility or a priority rule:

- If the boundary data are compatible on $J$ (e.g. Dirichlet targets agree on overlaps), then any limiting blend yields
  the same boundary trace there.
- If the data are incompatible on $J$, then no single-valued field can satisfy all piecewise constraints simultaneously
  on $J$; any construction must either (i) define a priority convention on $J$, or (ii) accept that exactness is claimed
  only on each $\Gamma_k\setminus J$ (a standard relaxation since $J$ is lower-dimensional).

## A.4. The BVH structure and beam traversal (packed AABB tree)

Phydrax’s BVH is a packed binary AABB tree with:

- node arrays storing bounding boxes and child links,
- fixed-size leaf payload arrays storing item indices (with padding),
- a beam traversal that keeps the best $B$ nodes according to AABB lower bounds.

Mathematically, for a node with AABB $[b_{\text{min}},b_{\text{max}}]\subset\mathbb R^d$, the squared distance lower bound
from a query $x$ is:

$$
d^2_{\text{AABB}}(x,[b_{\text{min}},b_{\text{max}}])
=
\sum_{k=1}^d \bigl(\mathop{\text{max}}\left\lbrace 0,\,b_{\text{min},k}-x_k,\,x_k-b_{\text{max},k}\right\rbrace\bigr)^2.
$$

This is a lower bound on the squared distance to any point inside the node’s subtree.

Beam traversal uses this bound to choose a small set of candidate leaves, then returns their payload items. In the
limit of infinite beam width (or in settings where the relevant neighbors always lie within the selected leaves),
the BVH-accelerated method is identical to an exact neighborhood search. For finite beam width it is an approximation,
and its accuracy depends on whether the candidate set captures the effective support of the MLS kernel.

## A.5. Higher-order initial constraints via a Hermite/Taylor ansatz

Let $t$ be the evolution variable with an initial slice $t=t_0$. Suppose we want to enforce, for a given
integer $K\ge 0$:

$$
\partial_t^k u(\cdot,t_0)=g_k(\cdot),\qquad k=0,1,\dots,K.
$$

Define the truncated Taylor polynomial:

$$
P_K(t)=\sum_{k=0}^{K}\frac{(t-t_0)^k}{k!}\,g_k,
$$

and define the enforced initial ansatz:

$$
u_I(t)=P_K(t) + (t-t_0)^{K+1}\bigl(u(t)-P_K(t)\bigr).
$$

**Proposition A.4 (exact initial derivatives).** Assume $u$ is $K+1$ times differentiable in $t$.
Then $\partial_t^k u_I(\cdot,t_0)=g_k(\cdot)$ for $k=0,\dots,K$.

*Proof.* The correction term $(t-t_0)^{K+1}(u-P_K)$ has a factor $(t-t_0)^{K+1}$, so all of its derivatives
up to order $K$ vanish at $t=t_0$. Therefore $\partial_t^k u_I(t_0)=\partial_t^k P_K(t_0)=g_k$. $\square$

This construction is used as the “initial target overlay” when multiple initial derivative targets are specified.

## A.6. Mixed boundary/initial constraints and the boundary gate

Boundary and initial sets intersect (e.g. $\partial\Omega\times\lbrace t_0\rbrace$). In PDE theory, exact satisfaction of
both requires *compatibility conditions* on the intersection; incompatible data cannot be enforced simultaneously.

The pipeline resolves the interaction by a staged priority:

1. Enforce boundary constraints first to obtain $u_B$.
2. Compute an initial-enforced candidate $u_{\text{init}}$ from $u_B$.
3. Blend the update through a **boundary gate** $\gamma$ that vanishes on the constrained boundary.

$$
u_{BI} \;=\; u_B + \gamma\,(u_{\text{init}}-u_B).
$$

This guarantees boundary constraints remain satisfied by construction because the update is identically zero on the
boundary. The gate is chosen to vanish to sufficiently high order to preserve boundary constraints involving spatial
derivatives up to a prescribed order.

**Remark (initial exactness tradeoff).** If the initial-enforcement map produces a candidate with
$u_{\text{init}}(\cdot,t_0)=g_0(\cdot)$, then on the initial slice:

$$
u_{BI}(\cdot,t_0)=u_B(\cdot,t_0) + \gamma(\cdot)\,\bigl(g_0-u_B(\cdot,t_0)\bigr).
$$

Unless $\gamma\equiv 1$ away from the boundary (or $u_B(\cdot,t_0)=g_0$ already), this blend relaxes *exact* initial
enforcement near the boundary in order to preserve boundary constraints. In Phydrax’s implementation, $\gamma$ is smooth,
vanishes to high order on the boundary, and tends to $1$ with increasing distance, so initial constraints are typically
satisfied approximately away from the boundary (subject to compatibility at $\partial\Omega\times\lbrace t_0\rbrace$).

### A.6.1. Vanishing-order lemma (derivative preservation)

Let $s$ be a local normal coordinate to $\partial\Omega$ (e.g. $s=\phi(x)$). Consider an update:

$$
u_{\text{new}} = u + \gamma(s)\,(v-u),
$$

where $\gamma(0)=0$.

**Lemma A.5 (preservation by high-order vanishing).** If $\gamma(s)=\mathcal O(|s|^{m})$ as $s\to 0$, then
for every integer $0\le k\le m-1$:

$$
\partial_s^k u_{\text{new}}(0) = \partial_s^k u(0),
$$

provided the derivatives exist.

*Proof.* Write $u_{\text{new}}-u=\gamma(s)\,w(s)$ with $w=v-u$. By Leibniz’ rule, each $\partial_s^k\bigl(\gamma w\bigr)$
is a sum of terms $(\partial_s^j\gamma)(\partial_s^{k-j}w)$. If $\gamma=\mathcal O(|s|^m)$, then $\partial_s^j\gamma(0)=0$
for all $j\le m-1$, implying all such terms vanish at $s=0$ when $k\le m-1$. $\square$

Thus, by choosing $\gamma$ to vanish to order $m\ge K+1$, the blend preserves boundary operators involving derivatives
up to order $K$ in the boundary-normal direction.

## A.7. Interior exact data satisfaction (anchor/data overlay stage)

After boundary and initial staging in the PCI pipeline, the interior anchor/data overlay enforces interior data exactly
at specified anchors/tracks while preserving boundary and initial constraints (including derivative constraints up to
specified orders).

### A.7.1. Interior anchors and (optional) tracks

**Anchor mode.** Given points $z_i\in\mathcal D$ and targets $y_i\in\mathbb R^C$, we require:

$$
\tilde u(z_i)=y_i.
$$

**Sensor-track mode.** Given fixed sensors $x_m$ and time-dependent observations $y_m(t)$, we require:

$$
\tilde u(x_m,t)=y_m(t).
$$

where $y_m$ is represented by an interpolant (e.g. a cubic Hermite spline in time).

### A.7.2. The protecting gate M(z)

Let boundary constraints for a geometry label $\ell=x$ involve derivatives up to order $K_x$ (e.g. Dirichlet: $K_x=0$,
Neumann/Robin: $K_x=1$). Let initial constraints fix time derivatives up to order $K_t$.

Define a gate:

$$
M(z)
=
\Bigl(\prod_{\ell\in\mathcal B} |\phi_\ell(z_\ell)|^{K_\ell+1}\Bigr)
\cdot (\mathop{\text{max}}(t-t_0,0))^{K_t+1},
$$

with $\mathcal B\subseteq \mathcal L$ the set of geometry labels with boundary constraints and $\phi_\ell$ a signed
distance–like function for that geometry factor.

On domains where $t\ge t_0$ identically, $\mathop{\text{max}}(t-t_0,0)=t-t_0$, so this reduces to $(t-t_0)^{K_t+1}$. The $\text{max}$ form
is convenient to keep $M$ nonnegative and real-valued.

Key properties:

- $M(z)=0$ on constrained boundary sets (where $\phi_\ell=0$),
- $M(z)=0$ on the initial slice $t=t_0$,
- $M$ vanishes to high enough order so that derivatives of $M(\cdot)h(\cdot)$ up to the constrained orders also vanish
  on those sets (formalized below).

Anchors/tracks are required to satisfy $M(z_i)>0$; placing an interior anchor on a constrained set is incompatible with
the goal of preserving those enforced constraints.

### A.7.3. IDW interpolation of the *scaled residual*

Let $u$ be the field after boundary/initial stages. For anchors $z_i$ with targets $y_i$, define scaled residuals:

$$
r_i = \frac{y_i - u(z_i)}{M(z_i)}.
$$

Define a (possibly anisotropic) squared distance on $\mathcal D$ with per-label lengthscales $\ell_\alpha>0$:

$$
d(z,z_i)^2=\sum_{\alpha\in\mathcal L}\left\lVert\frac{z_\alpha-z_{i,\alpha}}{\ell_\alpha}\right\rVert^2.
$$

For an IDW exponent $p>0$, define weights:

$$
w_i(z)\propto \bigl(d(z,z_i)^2+\varepsilon\bigr)^{-p/2},
\qquad \sum_i w_i(z)=1.
$$

The interior overlay defines:

$$
\tilde u(z)
=
u(z) + M(z)\sum_i w_i(z)\,r_i.
$$

This is an interpolation of the scaled residual field $r$, multiplied by the protecting gate $M$.

### A.7.4. Exact anchor satisfaction (snap rule)

In exact arithmetic, the IDW interpolant need not satisfy \(w_i(z_i)=1\) (it typically satisfies this only in the
limit \(\varepsilon\to 0\) with exact evaluation). The implementation therefore uses a *snap rule*: if a query is closer
than a prescribed threshold \(\varepsilon_{\text{snap}}\) to an anchor, the weight becomes one-hot.

**Proposition A.6 (exactness at anchors under snapping).** Suppose \(M(z_k)>0\) and \(w_k(z_k)=1\). Then \(\tilde u(z_k)=y_k\).

*Proof.* Evaluate the overlay at \(z=z_k\):

\[
\begin{aligned}
\tilde u(z_k) &= u(z_k) + M(z_k)\,r_k \\
&= u(z_k) + M(z_k)\frac{y_k-u(z_k)}{M(z_k)} \\
&= y_k.
\end{aligned}
\]
\(\square\)

### A.7.5. Preservation of boundary and initial constraints

The interior correction has the form $\Delta u(z)=M(z)\,h(z)$ where $h(z)=\sum_i w_i(z)r_i$.

**Lemma A.7 (vanishing factor kills derivatives).** Let $s\mapsto \eta(s)=|s|^{m}$ for an integer $m\ge 1$.
If $h$ is $C^{m-1}$ near $s=0$, then for every integer $0\le k\le m-1$,

$$
\frac{d^k}{ds^k}\bigl[\eta(s)\,h(s)\bigr]\Big|_{s=0}=0.
$$

*Proof.* By Leibniz’ rule, every term in the $k$-th derivative contains a factor $|s|^{m-j}$ with $m-j\ge 1$ when $k\le m-1$,
and therefore vanishes at $s=0$. $\square$

**Remark (regularity of the gate).** For integer $m\ge 1$, $\eta(s)=|s|^m$ is $C^{m-1}$ at $s=0$ (and smooth if $m$ is even),
which is exactly the regularity needed to conclude preservation of derivatives through order $m-1$.

**Remark (mixed derivatives).** In local coordinates $(s,y)$ near the boundary (normal $s$ and tangential $y$), any differential
operator involving at most $m-1$ derivatives in $s$ annihilates $\eta(s)\,h(s,y)$ at $s=0$; tangential derivatives act on $h$
and do not reduce the vanishing order in $s$. The product structure of $M(z)$ yields the analogous preservation on boundary–initial
intersections for mixed normal/time derivatives up to the prescribed orders, assuming the corresponding derivatives exist.

Applying this lemma with $m=K_\ell+1$ for each boundary label $\ell$ and $m=K_t+1$ for time implies:

- the interior overlay does not change the value of $u$ on boundary/initial sets where $M=0$,
- it also does not change derivatives up to order $K_\ell$ (resp. $K_t$) on those sets, provided the relevant derivatives exist.

Consequently, if the boundary and initial stages have produced $u$ satisfying the enforced constraints up to those orders,
then $\tilde u=u+\Delta u$ satisfies them as well.

### A.7.6. Multiple sources, envelopes, and coincidence handling

When multiple data sources are present, Phydrax optionally multiplies IDW weights by a source-local envelope
$\psi_s(z)=\exp(-d_s(z)^2/s^2)$, where $d_s(z)$ is the distance to the nearest anchor from source $s$.
This allows localized influence while preserving exactness at snapped anchors.

If anchors from different sources are coincident (according to the same snap metric used at runtime), they are deduplicated.
Conflicting coincident targets are rejected: exact enforcement of incompatible pointwise data is not possible.

## A.8. Multi-field pipelines, co-variables, and topological ordering

For a vector of fields $(u^{(1)},\dots,u^{(M)})$, an enforced constraint for one field may require access to other fields
(co-variables). This defines a directed dependency graph on field names. If the graph is acyclic, there exists a
topological order $u^{(i_1)},\dots,u^{(i_M)}$ such that every co-variable needed for enforcing $u^{(i_k)}$ has been
enforced earlier in the order.

Applying per-field pipelines in this order yields a well-defined global enforcement map:

$$
\mathcal P:\ (u^{(1)},\dots,u^{(M)})\mapsto (\tilde u^{(1)},\dots,\tilde u^{(M)}).
$$

If the dependency graph contains a cycle, a global deterministic enforcement map cannot be defined without additional
fixed-point structure; the implementation rejects such cycles.

## A.9. Compatibility and scope conditions

For the strongest “exactness” statements, the following conditions are essential:

1. **Compatibility on junctions and intersections.** Piecewise boundary data must be compatible on junction sets
   $J=\bigcup_{i\neq j}(\overline{\Gamma_i}\cap\overline{\Gamma_j})$ if exact satisfaction is desired there, and boundary
   and initial data must be compatible on their intersection (e.g. $\partial\Omega\times\lbrace t_0\rbrace$); otherwise,
   no construction can satisfy all requirements simultaneously. The staging priority enforces boundary constraints exactly
   and uses gates to avoid re-violating them.
2. **Nondegeneracy.** Neumann/Robin constructions require $\partial_n\phi\neq 0$ on $\partial\Omega$.
3. **Anchor placement.** Interior anchors/tracks must lie strictly away from constrained sets so that $M(z_i)>0$.
4. **Regularity.** To preserve derivative constraints through order $K$, the gate factors and the field must admit
   the corresponding derivatives (at least $C^K$ in the relevant coordinates).
5. **Gating tradeoff.** If the boundary gate $\gamma$ is not identically $1$ away from the boundary, then the gated blend
   cannot preserve boundary constraints and enforce initial constraints exactly everywhere simultaneously; instead, initial
   constraints are typically satisfied approximately away from the boundary.
6. **Approximation layers.** BVH selection, MLS distance proxies, and approximate distance fields introduce numerical
   approximation. The mathematical statements above should be interpreted either in the idealized continuous limit or as
   “up to numerical tolerance” for practical computations.

## A.10. Correspondence to the implementation (terminology)

The following implementation concepts align with the mathematics above:

- **Boundary stage**: piecewise constraints combined via weighted blending $u_B$.
- **Initial stage**: higher-order `enforce_initial` overlay and/or other initial enforced constraints, blended via a boundary gate $\gamma$.
- **Interior stage**: anchor/data correction $u\mapsto u + M\cdot(\text{IDW interpolant of scaled residuals})$ (with snapping for exact anchors).
- **BVH**: packed AABB tree used to accelerate boundary-subset weight evaluation for blending.
