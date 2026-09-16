# Black-hole geometry and stationary horizons

Phydrax keeps spacetime geometry in `phydrax.metrix` and stationary compact-object
physics in `phydrax.applications.compact_objects`. A metric is not a black-hole
simulation object, and a stationary Killing horizon is not used as a generic record
for a marginal, apparent, dynamical, trapping, or event horizon.

## Convention identity

`RelativityConvention` binds every calculation to the metric signature, Riemann sign,
extrinsic-curvature sign, spacetime orientation, future-time orientation, azimuthal
orientation, and Fourier sign. The canonical preset is

- signature $(-,+,+,+)$ (`mostly_plus`);
- $R^\rho{}_{\sigma\mu\nu}=\partial_\mu\Gamma^\rho{}_{\nu\sigma}-\partial_\nu\Gamma^\rho{}_{\mu\sigma}+\cdots$;
- $K_{ij}=-\tfrac12\mathcal L_n\gamma_{ij}$; and
- future, spacetime, and azimuthal orientations positive, with Fourier time factor
  $e^{-i\omega t}$.

The immutable `convention_id` crosses metric, tetrad, ADM, matter, radiation,
perturbation, extraction, and interchange boundaries. A sign change creates a
different contract; it is never repaired during evaluation.

## Exact metrics, charts, and domains

The public exact-metric surface contains:

| Constructor | Coordinates | Exact admitted chart domain |
| --- | --- | --- |
| `ingoing_schwarzschild_metric` | future-ingoing $(v,r,\theta,\phi)$ | finite $M>0$, $r>0$, and $0<\theta<\pi$; regular at $r=2M$ |
| `kerr_boyer_lindquist_metric` | $(t,r,\theta,\phi)$ | finite $M>0$, finite signed $a$, $r>0$, $0<\theta<\pi$, $\Sigma>0$, and $\Delta\ne0$ |
| `ingoing_kerr_metric` | future-ingoing $(v,r,\theta,\tilde\phi)$ | finite $M>0$, finite signed $a$, $r>0$, $0<\theta<\pi$, and $\Sigma>0$; the real Kerr horizons are chart-regular |

Here $\Sigma=r^2+a^2\cos^2\theta$ and $\Delta=r^2-2Mr+a^2$.
`boyer_lindquist_to_ingoing_kerr_transition` is available only for $|a|\le M$ and
excludes $\Delta=0$, where that coordinate transition is singular. The metric itself
may represent finite overextremal Kerr away from its ring; horizon-producing utilities
do not reinterpret an overextremal input as a black hole.

Domain predicates return `MetricDomainEvidence`, not clipped coordinates. Its signed
`margin` is positive inside, zero at a provider boundary, and negative outside.
`MetricDomainStatus` distinguishes `VALID`, `NEAR_BOUNDARY`, `OUTSIDE`, `NONFINITE`,
and `REJECTED`; `ExactMetricDomainStatus` additionally distinguishes an axis, ring,
Boyer--Lindquist horizon, and invalid parameters. `derivative_valid` requires a
qualified smooth interior point and is false in the declared boundary band.

The metric utilities also provide cancellation-safe Kerr inner/outer radii,
stationary-limit radii, the Kretschmann scalar, and the right-handed Pontryagin
scalar. They are exact formula evaluators, not numerical-relativity horizon finders.

## Killing fields and tetrads

`stationary_killing_vector` and `axial_killing_vector` follow the declared time and
azimuthal orientations. `killing_equation_residual` and
`maximum_killing_equation_residual` retain independent Killing evidence.
`stationary_axial_inner_product_evidence` exposes the Gram determinant and causal
classification rather than inferring an observer where the stationary generator is
not timelike.

`orthonormal_tetrad`, `zamo_observer_tetrad`, and
`kerr_principal_null_tetrad` produce evaluated frames with Gram, orientation, and
domain evidence. The Kerr principal null frame uses Kinnersley normalization.
Projection/reconstruction and dual-frame operations preserve whether an input is a
vector or covector. Parallel-transport evidence evaluates $u^\nu\nabla_\nu e_A{}^\mu$
on a differentiable tetrad field; a locally orthonormal frame is not thereby claimed
to be parallel transported.

## ADM exchange boundary

`ADMGridGeometry` is the immutable geometry-to-matter snapshot: lapse, contravariant
shift, spatial metric and inverse, $\sqrt\gamma$, extrinsic curvature, active/valid
masks, chart, scale, convention, and topology. `geometry_lineage_id` identifies the
static provider/grid family; scalar integer `snapshot_token` identifies the exact
dynamic stage realization. It deliberately contains no spatial derivatives.
`ValenciaGeometrySource` carries derivative data needed by every Valencia equation
source and binds it to the exact `ADMGridGeometry` snapshot.

`StressEnergyProjection` is the reverse matter-to-spacetime boundary. It carries
Eulerian energy density, covariant momentum density, spatial stress, source and
conservation defects, and the same scale/convention/topology plus exact
`geometry_lineage_id`/`snapshot_token`. Compatibility requires both static lineage and
the dynamic token. This prevents a valid projection from one SSPRK stage being reused
at another stage with the same grid.

## Stationary Killing-horizon thermodynamics

`KerrInput` stores geometric mass $M$ (length) and signed angular momentum $J$
(length squared). `classify_kerr` reports indeterminate, subextremal, extremal, or
overextremal branches. `evaluate_stationary_kerr_horizon` evaluates only the
stationary Kerr Killing horizon:

$$
a=J/M,\qquad r_\pm=M\pm\sqrt{M^2-a^2},\qquad
A=4\pi(r_+^2+a^2),
$$

$$
M_{\rm irr}=\sqrt{A/(16\pi)},\qquad
\kappa=\frac{r_+-r_-}{2(r_+^2+a^2)},\qquad
\Omega_H=\frac{a}{r_+^2+a^2}.
$$

`evaluate_kerr_entropy_temperature` is the only conversion to declared physical
entropy and temperature and requires `RelativityScaleContract` with explicit
constants. The first-law evaluator checks the directional identity
$dM=\kappa\,dA/(8\pi)+\Omega_H\,dJ$ and the algebraic Smarr relation independently.
Its derivative is admitted only on a smooth subextremal branch. Fixed-$J$ and
fixed-$\Omega_H$ response records are distinct ensembles and expose the Davies or
extremal conditioning singularity; neither is a generic heat-capacity type.

Extended stationary plans cover Einstein--Hilbert Wald entropy (exactly the area
entropy), asymptotically flat Kerr--Newman equilibrium thermodynamics,
four-dimensional Kerr--Newman--AdS extended thermodynamics with mass as enthalpy,
and charged Reissner--Nordström thermodynamics in a finite Dirichlet cavity. These
plans do not claim higher-curvature Wald entropy, dynamical entropy production, or a
universal ensemble.

## Differentiation and status discipline

A result's `finite`, `converged`, `physically_valid`, `qualified`, and
`derivative_valid` fields are independent predicates. `qualified` never follows from
finiteness alone. Coordinate axes, chart boundaries, Kerr branch changes, extremality,
ill-conditioned response ensembles, frame-construction failures, and caller-selected
chart transitions are explicit derivative boundaries. Consume values only under the
predicate required by the intended claim.