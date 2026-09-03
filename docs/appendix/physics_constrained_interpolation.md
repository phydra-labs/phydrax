# Physics-Constrained Interpolation (overlay pipeline)

This appendix formalizes the mathematics behind Phydrax’s **Physics-Constrained Interpolation (PCI)** enforced overlay pipeline:

1. boundary enforced constraints (possibly piecewise, via blending),
2. initial constraints (possibly higher order in the evolution variable, and gated to preserve boundary constraints),
3. interior *exact* data satisfaction via an anchor/data overlay stage while preserving boundary and initial constraints.

Here “PCI” refers to the *entire* staged enforcement map $u\mapsto \tilde u$; the final stage is the
interior anchor/data overlay.

The implementation corresponds to the compiled enforcement constructed by `EnforcementProgram` in
`phydrax.enforcement`, including its Dirichlet/Neumann/Robin enforced ansätze, the BVH-accelerated weight construction
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

Let $\Omega\subset\mathbb R^d$ be a geometry factor with boundary
$\partial\Omega$. Phydrax keeps three zero-set-preserving fields distinct:

- a certified negative-inside boundary field $\phi$ supplied by the compiled
  geometry;
- a dimensionless enforcement gate $\beta$, with $\beta=0$ on
  $\partial\Omega$ and $\beta=\mathcal O(1)$ in the interior, for value ansätze
  and constraint-preserving overlays; and
- a dimensional boundary ansatz factor $\psi$, with $\psi=0$ and
  $\partial_n\psi=1$ on $\partial\Omega$, for derivative hard constraints.

The outward unit normal is denoted by $n$ and comes from the compiled normal
provider. Derivative ansätze use the canonical off-boundary extension
$\nu=\nabla\psi$. At every regular boundary point, $\nu=n$; elsewhere it
inherits the boundary field's certified regularity. Analytic, simplicial, and
B-Rep representations preserve the same boundary zero set as their compiled
field. Mesh edges and vertices use the selected outward pseudonormal because a
unique classical normal does not exist.

### A.2.1. Dirichlet (value) constraints

Given a target $g:\partial\Omega\to\mathbb R^C$, define:

$$
u^\star(x)\;=\; g(x) + \beta(x)\,\bigl(u(x)-g(x)\bigr).
$$

**Proposition A.1 (Dirichlet exactness).** If $\beta=0$ on
$\partial\Omega$, then $u^\star=g$ on $\partial\Omega$.

*Proof.* For $x\in\partial\Omega$, $\beta(x)=0$ implies
$u^\star(x)=g(x)$. $\square$

### A.2.2. Neumann (normal derivative) constraints

Given a target $g:\partial\Omega\to\mathbb R^C$ for $\partial_n u=g$, define
$\nu=\nabla\psi$ and:

$$
u^\star \;=\; u + \psi\,\bigl(g-D_\nu u\bigr).
$$

**Proposition A.2 (Neumann exactness).** Assume $u,\psi$ are differentiable,
$\psi=0$, and $\partial_n\psi=1$ on $\partial\Omega$. Then
$\partial_n u^\star=g$ on every regular boundary point.

*Proof.* Because $\psi$ is constant on the boundary, its tangential derivatives
vanish there. Hence $\nu=\nabla\psi=n$ and $D_\nu u=\partial_nu$ on the boundary.
Differentiating the ansatz along $n$ gives:

$$
\partial_n u^\star
=\partial_n u + \partial_n\psi\bigl(g-D_\nu u\bigr)
+ \psi\,\partial_n\bigl(g-D_\nu u\bigr).
$$

On $\partial\Omega$, $\psi=0$ annihilates the last term and
$\partial_n\psi=1$, hence
$\partial_n u^\star=\partial_n u+(g-\partial_nu)=g$. $\square$

Using $\nu$ rather than normalizing a nearest-boundary field changes no boundary
operator. It only chooses a differentiable interior extension for the residual,
avoiding artificial normal jumps where nearest boundary points are nonunique.

### A.2.3. Robin (mixed) constraints

For $a\,u + b\,\partial_n u = g$ on $\partial\Omega$, use

$$
u^\star \;=\; u + \frac{\psi}{b}\,
\bigl(g-a\,u-b\,D_\nu u\bigr),
$$

under the nondegeneracy assumption $b\neq0$ on $\partial\Omega$. The proof is
the same as Proposition A.2 because $D_\nu u=\partial_nu$ on the boundary.

### A.2.4. Certified boundary fields and scale-normalized gates

Hard constraints need a reliable zero set and an explicit derivative contract.
Every compiled geometry therefore exposes three distinct objects:

- the negative-inside boundary field $\phi$ and its `FieldCertificate`;
- the outward normal provider $n$; and
- region membership, which remains a separate kernel query.

The certificate records zero-set accuracy, sign reliability, distance semantics,
regularity, validity region, safe-step information, and parameter
differentiability. Analytic signed distances, mesh distances, B-Rep
approximations, smooth blends, and general level sets therefore do not make the
same numerical claim.

The domain adapter exposes $\phi$ as `adf`. Define a unit-jet source $\chi$.
If $\phi$ has signed-distance semantics, set $\chi=\phi$. Its regular boundary
jet already satisfies

$$
\chi|_{\partial\Omega}=0,
\qquad
\partial_n\chi|_{\partial\Omega}=1.
$$

For a certified level set, use

$$
\chi(x)
=
\frac{\phi(x)}{\|\nabla\phi(x)\|},
$$

with a finite denominator fallback away from regular points. On a regular
boundary point, the term produced by differentiating the denominator is
multiplied by $\phi=0$, so

$$
\chi|_{\partial\Omega}=0,
\qquad
\nabla\chi|_{\partial\Omega}
=
\frac{\nabla\phi}{\|\nabla\phi\|}
=n,
\qquad
\partial_n\chi|_{\partial\Omega}=1.
$$

The final `boundary_ansatz_factor` is $\psi=q(\chi)$, where $q$ is the compact
transform below with its default fractions. Because $q$ is exactly the identity
in an open zero-set collar, $\psi$ retains the same boundary value and unit
normal derivative while its interior amplitude is bounded. At a sharp edge,
corner, CSG seam, or medial set, the field certificate is only piecewise smooth
and a unique classical normal need not exist. Simplicial and B-Rep kernels use
the selected outward face pseudonormal at such points.

#### A.2.4.1. Compact dimensionless gate

Dirichlet ansätze need only a dimensionless zero-set-preserving multiplier. Let
$L>0$ be the shortest span of the geometry bounds,
$\delta=sL$ for `saturation_fraction` $s$, and
$a$ be `linear_fraction`. With

$$
r=\frac{|\phi|}{\delta},
\qquad
t=\frac{r-a}{1-a},
$$

define

$$
G(t)
=
t-66t^7+\frac{495}{2}t^8-385t^9
+308t^{10}-126t^{11}+21t^{12}.
$$

The compact dimensional profile is

$$
q(\phi)=
\begin{cases}
\phi,
&r\le a,\\[1mm]
\operatorname{sign}(\phi)\,\delta
\left[a+(1-a)G(t)\right],
&a<r<1,\\[1mm]
\operatorname{sign}(\phi)\,\delta
\left[a+\tfrac12(1-a)\right],
&r\ge1.
\end{cases}
$$

The default general-geometry gate is

$$
\beta
=
-\frac{q(\phi)}
{\delta\left[a+\tfrac12(1-a)\right]}.
$$

It is zero on the boundary, positive in the interior, scale invariant under
uniform scaling, exactly linear in an open boundary collar, and constant past
the saturation radius. The transition polynomial agrees with the identity
through sixth order at its inner join and is flat through sixth order at its
outer join. `gate_method="auto"` uses this transform when no domain-specific
exact builder exists; `gate_method="compact"` selects it explicitly.

#### A.2.4.2. Broad generic gate

`gate_method="global_r_equivalence"` explicitly selects the nonsaturating generic
profile. With

$$
z=1.15\frac{-\phi}{L/2},
$$

the interior branch is

$$
\beta=z(2-z).
$$

An odd bounded exterior branch is used for $z<0$. This profile can be broader
than the compact transform but inherits every nonsmoothness of the source field;
the name is retained as the public method selector, not as a claim that a
separate mesh R-equivalence field is constructed.

For an interval $[a,b]$, Phydrax uses the exact analytic gate
$4(x-a)(b-x)/(b-a)^2$. An axis-aligned hyperrectangle uses the product of its
per-axis gates. These exact profiles are dimensionless, smooth, equal to one at
the box center, and ignore compact-profile controls.

All gate and ansatz constructions operate on the compiled geometry contract.
They do not inspect whether the source was analytic, simplicial, reconstructed,
or B-Rep. Representation-specific approximation and differentiability limits
remain visible through `FieldCertificate`.

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

## A.5. Higher-order initial constraints via a gated Taylor ansatz

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
u_I(t)=P_K(t) + g(t)\,\bigl(u(t)-P_K(t)\bigr).
$$

Assume $g$ satisfies:

$$
g^{(j)}(t_0)=0,\qquad j=0,1,\dots,K.
$$

Then $u_I$ matches all prescribed initial derivatives.

**Proposition A.4 (exact initial derivatives).** Assume $u$ is $K+1$ times differentiable in $t$.
Then $\partial_t^k u_I(\cdot,t_0)=g_k(\cdot)$ for $k=0,\dots,K$.

*Proof.* Write $u_I=P_K+g(u-P_K)$. By Leibniz, every derivative of $g(u-P_K)$ of order $\le K$ contains a factor
$g^{(j)}(t_0)$ with $j\le K$, hence vanishes at $t_0$. Therefore
$\partial_t^k u_I(t_0)=\partial_t^k P_K(t_0)=g_k$. $\square$

In implementation, two gate families are supported:

- **Polynomial gate** (legacy):
  $g_{\text{poly}}(t)=(t-t_0)^{K+1}$.
- **Rational gate** (default):
  $q=K+1$, $\tau=\frac{\Delta t}{L\,\varepsilon^{1/q}}$, and
  $g_{\text{rat}}(t)=\frac{\tau^q}{1+\tau^q}$, where $\varepsilon=\texttt{gate\_eps}>0$, and $\Delta t$ is the oriented (or absolute) distance from the initial slice
  depending on whether the initial component is `FixedStart`, `FixedEnd`, or `Fixed`.

Near $t_0$, $g_{\text{rat}}(t)=\mathcal O((t-t_0)^{K+1})$, so it preserves the same exact-derivative property while
remaining bounded away from unbounded polynomial growth far from the initial slice.

This gated Taylor construction is used as the “initial target overlay” when multiple initial derivative targets are specified.

## A.6. Mixed boundary/initial constraints and the boundary gate

Boundary and initial sets intersect (e.g. $\partial\Omega\times\lbrace t_0\rbrace$). In PDE theory, exact satisfaction of
both requires *compatibility conditions* on the intersection; incompatible data cannot be enforced simultaneously.

The pipeline resolves the interaction by a staged priority:

1. Enforce boundary constraints first to obtain $u_B$.
2. Compute an initial-enforced candidate $u_{\text{init}}$ from $u_B$.
3. If boundary/initial interaction is incompatible, blend through a **boundary gate** $\gamma$ that vanishes on the constrained boundary.
   If they are compatible, accept the initial-enforced update directly.

$$
u_{BI} \;=\;
\begin{cases}
u_{\text{init}}, & \text{(boundary-compatible initial overlay)},\\[4pt]
u_B + \gamma\,(u_{\text{init}}-u_B), & \text{(gated compatibility branch)}.
\end{cases}
$$

In the gated branch, boundary constraints remain satisfied by construction because the update is identically zero on the
boundary. The gate is chosen to vanish to sufficiently high order to preserve boundary constraints involving spatial
derivatives up to a prescribed order.

For boundary labels $\ell$ with constrained derivative order $K_\ell$, the
implementation uses

$$
\gamma(x)=\prod_\ell |\beta_\ell(x_\ell)|^{K_\ell+1}.
$$

The dimensionless base gate $\beta_\ell$ controls conditioning; the exponent
controls vanishing order and therefore which boundary derivatives are preserved.

**Remark (initial exactness tradeoff).** If the initial-enforcement map produces a candidate with
$u_{\text{init}}(\cdot,t_0)=g_0(\cdot)$, then on the initial slice:

$$
u_{BI}(\cdot,t_0)=u_B(\cdot,t_0) + \gamma(\cdot)\,\bigl(g_0-u_B(\cdot,t_0)\bigr).
$$

This tradeoff applies only in the gated branch. In the compatibility branch, the pipeline keeps
$u_{BI}=u_{\text{init}}$, so exact initial matching from A.5 is preserved globally. In the gated branch,
unless $\gamma\equiv 1$ away from the boundary (or $u_B(\cdot,t_0)=g_0$ already), the blend relaxes *exact* initial
enforcement near the boundary in order to preserve boundary constraints. In Phydrax’s implementation, $\gamma$ is smooth,
vanishes to high order on the boundary, and tends to $1$ with increasing distance, so initial constraints are typically
satisfied approximately away from the boundary (subject to compatibility at $\partial\Omega\times\lbrace t_0\rbrace$).

### A.6.1. Vanishing-order lemma (derivative preservation)

Let $s$ be a local normal coordinate to $\partial\Omega$ (for example,
$s=\phi(x)$). The base enforcement gate satisfies
$\beta(x)=\mathcal O(|s|)$ near a regular boundary point. Consider an update:

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

Let boundary constraints for a geometry label $\ell=x$ involve derivatives up
to order $K_x$ (e.g. Dirichlet: $K_x=0$, Neumann/Robin: $K_x=1$). Let initial
constraints fix time derivatives up to order $K_t$.

Define a gate:

$$
M(z)
=
\Bigl(\prod_{\ell\in\mathcal B}
|\beta_\ell(z_\ell)|^{K_\ell+1}\Bigr)
\cdot (\mathop{\text{max}}(t-t_0,0))^{K_t+1},
$$

with $\mathcal B\subseteq\mathcal L$ the set of geometry labels with boundary
constraints and $\beta_\ell$ the dimensionless enforcement gate for that
geometry factor.

On domains where $t\ge t_0$ identically,
$\mathop{\text{max}}(t-t_0,0)=t-t_0$, so this reduces to
$(t-t_0)^{K_t+1}$. The $\text{max}$ form is convenient to keep $M$
nonnegative and real-valued.

Key properties:

- $M(z)=0$ on constrained boundary sets (where $\beta_\ell=0$),
- $M(z)=0$ on the initial slice $t=t_0$,
- $M$ vanishes to high enough order so that derivatives of
  $M(\cdot)h(\cdot)$ up to the constrained orders also vanish on those sets
  (formalized below).

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

In exact arithmetic, the IDW interpolant need not satisfy $w_i(z_i)=1$ (it typically satisfies this only in the
limit $\varepsilon\to 0$ with exact evaluation). The implementation therefore uses a *snap rule*: if a query is closer
than a prescribed threshold $\varepsilon_{\text{snap}}$ to an anchor, the weight becomes one-hot.

**Proposition A.6 (exactness at anchors under snapping).** Suppose $M(z_k)>0$ and $w_k(z_k)=1$. Then $\tilde u(z_k)=y_k$.

*Proof.* Evaluate the overlay at $z=z_k$:

$$
\tilde u(z_k)
= u(z_k) + M(z_k)\,r_k
= u(z_k) + M(z_k)\frac{y_k-u(z_k)}{M(z_k)}
= y_k.
$$
$\square$

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

## A.8. Joint multi-field affine projection

For a product of fields $U=(u^{(1)},\ldots,u^{(M)})$, collect every certified
linear condition into one operator

$$
C: X_1\oplus\cdots\oplus X_M\longrightarrow Y.
$$

Given a right inverse $R$ on the declared target range, the joint correction is

$$
P_b(U)=U+R\bigl(b-CU\bigr).
$$

Thus $CP_b(U)=b$, $P_b(P_b(U))=P_b(U)$, and every already-feasible field tuple is
fixed. Off-diagonal blocks of $C$ encode coupled fields directly; cyclic equations
therefore require no pivot or per-field topological ordering.

If the rows of $C$ are dependent, the compiler retains the generalized-inverse
identity $CRC=C$ and separately checks that $b$ lies in the range of $C$. Conflicting
targets fail. Coefficient-space, kernel/RKHS, graph, geometry, and interface
providers differ only in how they construct the lift $R$ and its certificate.

Local boundary/initial ansatz maps remain useful inexpensive feasible
parameterizations, but they are not automatically idempotent projectors. A later
joint realization must either operate in their preservation kernel or include
their conditions in the joint operator.

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
5. **Gating tradeoff (conditional).** If the pipeline takes the gated branch and the boundary gate $\gamma$ is not
   identically $1$ away from the boundary, then the gated blend cannot preserve boundary constraints and enforce initial
   constraints exactly everywhere simultaneously; initial constraints are then typically satisfied approximately near the
   boundary. If the compatibility branch is taken, this tradeoff is avoided.
6. **Approximation layers.** BVH selection, MLS distance proxies, and approximate distance fields introduce numerical
   approximation. The mathematical statements above should be interpreted either in the idealized continuous limit or as
   “up to numerical tolerance” for practical computations.

## A.10. Correspondence to the implementation (terminology)

The following implementation concepts align with the mathematics above:

- **Boundary stage**: piecewise constraints combined via weighted blending $u_B$.
- **Initial stage**: higher-order `enforce_initial` gated Taylor overlay and/or other initial enforced constraints, with
  boundary-gated blending only when boundary compatibility requires it.
- **Interior stage**: anchor/data correction $u\mapsto u + M\cdot(\text{IDW interpolant of scaled residuals})$ (with snapping for exact anchors).
- **Joint affine realization**: `prepare_affine_projector` assembles all coupled
  rows and prepares one certified right-inverse action.
- **Coefficient realization**: `CoefficientElimination` produces a
  `ConstraintMap` and target-dependent lift inside a certified linear
  representation.
- **Fiber realization**: analytic, realized, or separable fiber units retain
  unconstrained coordinates as an explicit residual domain.
- **Dynamic realization**: `PreparedEnforcementStep` freezes one accepted-step
  realization; refresh is transactional and checkpointable.
- **Nonlinear/feasibility realization**: local retractions and closed/open
  feasible maps carry relation-specific certificates rather than affine
  projection claims.
- **BVH**: packed AABB tree used to accelerate boundary-subset weight evaluation for blending.
