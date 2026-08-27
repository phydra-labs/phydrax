# Holomorphic potential fields

Phydrax separates complex analyticity evidence from physical PDE evidence. A
holomorphic-map certificate records the operations that preserve analyticity; harmonic,
biharmonic, elasticity, pluriharmonic, trace, and domain-local wrappers state their
additional assumptions independently.

## Pointwise providers

`HolomorphicPolynomialPotential` is the deterministic scalar-input baseline.
`HolomorphicMLP` uses complex-affine layers and the entire complex exponential while
storing every trainable leaf in real Cartesian form. Dense and explicitly low-rank
complex layers share the same certificate semantics.

Scalar `jet` remains available for one complex input. `multi_jet` accepts a canonical
`HolomorphicMultiIndexSet` for several complex inputs. Public multijets contain raw
partial derivatives; the implementation propagates normalized multivariate Taylor
coefficients through affine maps, exponentials, and product convolutions.

## Linear holomorphic frames

`HolomorphicPolynomialFrame` separates a fixed holomorphic basis from its real
coefficients. Its basis matrix has shape `(complex_outputs, real_coefficients)` and
represents

```text
F(z) = B(z) q,   q real.
```

The frame supports multiple complex outputs and multivariate monomial index sets.
`HolomorphicJetFunctionalTerm` represents one real part of a weighted output derivative,
and `HolomorphicPointFunctional` combines explicit terms at one complex point. Real and
imaginary values, supplied-normal derivatives, Robin data, coupled outputs, and
several-variable first derivatives all use this one data-bearing contract.

## Reusable coefficient constraints

Constraint geometry and targets are independent:

```python
frame = phx.equations.HolomorphicPolynomialFrame.one_variable(3)
functionals = (
    phx.equations.HolomorphicPointFunctional.value(-1.0),
    phx.equations.HolomorphicPointFunctional.value(1.0),
)
operator = phx.equations.HolomorphicConstraintOperatorPlan(
    frame,
    functionals,
).prepare()
coefficient_map = operator.affine_map(jnp.asarray([0.0, 0.0]))
potential = phx.equations.ConstrainedHolomorphicPotential(coefficient_map)
```

Preparation factorizes the real functional matrix once and stores its right inverse and
canonical right nullspace. Different target vectors reuse that factorization. For
constraint matrix `A`, target `b`, minimum-norm lift `q0`, nullspace `N`, and trainable
free coordinates `eta`, evaluation uses

```text
A q0 = b,   A N = 0,   q = q0 + N eta.
```

Homogeneous targets retain finite-subspace and linear-parameter evidence. Nonzero
targets form an affine finite-parametric family. Exactness applies to the prepared
finite functionals, not to unsampled boundary points.

## Nonlinear cardinal projection

`HolomorphicConstraintProjector` corrects any compatible certified nonlinear provider
with a fixed holomorphic cardinal lift. It requires full row rank so the lift can
represent arbitrary functional residuals. The correction is additive and holomorphic;
no trainable parameter projection or optimizer-state rotation occurs. A
`HolomorphicProjectionState` computes child functional values once and reuses the
correction across a query batch.

## Coupled physical functionals

Named Goursat and plane-elasticity compilers construct real rows on multi-output
holomorphic frames:

- biharmonic value, supplied-normal derivative, and Robin functionals;
- plane stress components;
- traction components on a supplied normal;
- plane displacement components with an explicit `PlaneIsotropicMaterial` identity.

Pure-traction and potential gauges remain visible through rank and nullity. They are not
silently regularized away.

## Conditional holomorphic operators

`HolomorphicBasisTrunk` adapts a continuous frame to the shared DeepONet decoder.
Supported modes are:

- full unconstrained coefficients;
- a fixed target lift plus learned nullspace coordinates;
- source targets concatenated with learned nullspace coordinates.

`ConditionalHolomorphicDeepONet` certifies holomorphy only in query coordinates. Source
encoders may be arbitrary. Constrained decoders reject a free output bias because that
bias can leave the prepared affine set. `TargetAugmentedBranchEncoder` extracts declared
real target entries deterministically and concatenates them with learned free
coordinates.

## Continuous trace evidence

`HolomorphicTraceCertificate` distinguishes four claims:

- finite-functional exact;
- continuous finite-subspace exact;
- continuous validated bound;
- sampled audit.

`DiskHolomorphicTracePlan` maps a finite real Fourier trace on a circle exactly to a
Taylor polynomial. Its continuous claim follows from coefficient identity, not dense
collocation. `HolomorphicContourFunctional` and `holomorphic_period_functional` bind
explicit nodes and complex quadrature weights; quadrature evidence remains distinct
from an exact analytic contour identity.

## Meromorphic frames and topology

`MeromorphicLinearFrame` combines a regular polynomial part with fixed principal parts.
It issues meromorphic rather than global holomorphic evidence. `PoleClearanceReport`
checks pole separation from a closed disk and from other poles;
`DomainHolomorphicCertificate` binds a valid clearance report to the meromorphic
construction.

`MeromorphicVariableProjectionPlan` treats pole locations as nonlinear real Cartesian
variables and residues/regular coefficients as one linear block through the existing
variable-projection solver. `fit` executes the reduced solve directly.
`continuation_problem` exposes reduced-objective stationarity as a
`ParameterContinuationProblem` along a linear observation path, so the existing
continuation, tangent, and bifurcation substrates can track pole branches without a
second nonlinear runtime.

Several-variable meromorphic functions are not modeled by vector-valued isolated
poles; their divisor geometry remains a separate capability.

On multiply connected domains, Laurent frames may omit logarithmic harmonic modes and
single-valued conjugates may require period conditions. Frame and trace documentation
must state those completeness boundaries explicitly.

## Several complex variables

`HolomorphicMultiIndexSet` gives deterministic total-degree or explicit derivative
ordering. `HolomorphicPolynomialFrame` accepts several complex coordinates and a full
invertible complex affine normalization. Point normals use real coordinate order
`(x0, ..., xm-1, y0, ..., ym-1)`.

The real part of a holomorphic scalar map on several complex variables is
pluriharmonic. It is harmonic in the paired real coordinates but is not a complete
harmonic family when the complex dimension exceeds one. `PluriharmonicPotential`
exposes analytic real gradients, Hessians, and Laplacians while retaining that narrower
claim.

## Kähler gauges

`KahlerHolomorphicGauge` adds the real part of a compatible holomorphic provider to a
`KahlerPotentialGeometry`. The mixed complex Hessian is unchanged by construction, so
the Kähler metric is invariant. `KahlerGaugeInvarianceReport` is a numerical validation
of that construction, not the source of the proof.

## Complex parameter interchange

`phx.export.export_complex_parameters` converts supported real Cartesian model leaves
into a canonical `ComplexInterchangeState`. Dense and low-rank complex layers,
`HolomorphicMLP`, polynomial potentials, constrained frame coefficients,
meromorphic coefficients, and trainable pole locations retain explicit provider and
architecture identities.

```python
state = phx.export.export_complex_parameters(model)
restored = phx.export.import_complex_parameters(model_template, state)
```

Import requires exact architecture and array shapes. Precision narrowing is rejected
unless enabled by `ComplexImportPolicy`; destination leaf dtypes and sharding remain
authoritative. Constrained coefficient import converts the complex matrix back to real
frame coordinates, recovers nullspace coordinates, and rejects values outside the
destination affine set.

This is a mathematical interchange surface, not native complex training. The restored
model still has real trainable leaves and uses the existing real optimizer,
continuation, checkpoint, and constraint semantics.

## Evidence boundaries

The following remain intentionally uncertified:

- arbitrary real-coordinate separable products;
- arbitrary Python boundary callables as continuous evidence;
- continuous exactness inferred from collocation density;
- generic finite-algebra holomorphic networks;
- native-complex optimizer conventions;
- automatic transport of diagonal adaptive optimizer state through moving nullspace
  bases.
