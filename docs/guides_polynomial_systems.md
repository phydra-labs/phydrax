# Sparse polynomial systems

Phydrax stores a polynomial system as one canonical sparse support and one aligned
coefficient vector. The support is static; coefficients may refresh without changing
term ordering, equation meaning, or provider plans.

## Canonical representation

For term `t`, `equation_indices[t]` selects the output equation and
`exponents[t, j]` is the nonnegative integer exponent of variable `j`. Coefficients
are aligned with this canonical COO order. `SparsePolynomialSystem.from_coo`
canonicalizes support and coefficients together; use it when the input term order is
not already canonical.

Exact zeros remain in a declared fixed support. This matters for coefficient families:
a coefficient may vanish at one parameter value without changing compilation or path
semantics.

`SparsePolynomialSystem.evaluate` accepts points with shape
`batch_shape + (variable_count,)`. `jacobian` evaluates derivatives directly from
integer powers and never divides a monomial by a coordinate, so zero coordinates do
not create artificial `0/0` values.

## Scaling

`PolynomialScaling` applies diagonal variable and equation scaling only. Multiplicative
scaling preserves support and is reversible. Additive shifts are intentionally absent
because they expand monomial support.

Provider coordinates and physical coordinates remain distinct in every result.

## Variable groups and degree forecasts

`PolynomialVariableGroup` declares an affine or projective coordinate block. Groups
are semantic input, not an inferred optimization. Structural analysis reports:

- total degree per equation;
- multidegree per declared group;
- projective homogeneity;
- the total-degree Bézout bound;
- the multihomogeneous Bézout bound.

These are path-count bounds, not runtime guarantees. A provider may report a smaller
polyhedral mixed-volume count. Plans retain every number separately.

## Scaling symmetries

Exponent differences within each equation define an integer lattice. Exact normal-form
analysis reports free diagonal scaling weights and finite root-of-unity factors. The
analysis is evidence only: Phydrax does not automatically choose a gauge, quotient a
solution set, or alter multiplicities.

A caller that reduces a symmetry must supply the section and reconstruction map and
must replay the original system after reconstruction.

## Isolated roots

An isolated-root plan requires a square affine system and an explicit provider.
Preparation pins provider identity and resources. Refresh may change coefficients but
not support, labels, dtype, scaling structure, or provider policy.

Every provider path remains visible. Path status, numerical clustering, near-real
classification, original residual replay, and application acceptance are independent
fields. Failed, singular, infinite, and duplicate endpoints are never silently removed.

The optional HomotopyContinuation process provider is host-only. It uses a pinned Julia
executable and project, a fixed packaged worker, bounded data-only JSON, a governed
random seed, and the existing no-shell external runtime. It never installs software or
falls back to a native solver.

## Selected-root polishing

A selected complex candidate can lower to the native nonlinear substrate using real
Cartesian coordinates. For a complex Jacobian `J`, the real block operator is

```text
[ Re(J)  -Im(J) ]
[ Im(J)   Re(J) ]
```

This supports local polishing and regular-root implicit derivatives. It does not make
the unordered root set differentiable through collisions, clustering, or cardinality
changes.

## Positive-dimensional results

A `WitnessSet` stores a polynomial system, a declared affine slice, and the finite
intersection points. `MultigradedWitnessCollection`, regeneration, monodromy, trace,
and pseudo-witness results preserve their full stage and path evidence. Irreducibility
is `undecided` unless a provider operation and its trace evidence establish otherwise.

## Exact symbolic operations

Exact systems use integer, rational, or prime-field coefficients outside JAX arrays.
The optional Macaulay2 boundary admits only a closed operation set and a fixed worker.
There is no arbitrary source evaluator, package loader, serialized-object evaluator,
or numerical-provider routing.

## Quotient-algebra roots

The bounded native quotient method is intended for small affine zero-dimensional
systems. A support-only plan forecasts the Macaulay layout before allocation.
Coefficient preparation performs two-threshold rank selection, conditioned
quotient-basis selection, multiplication solves, commutator audits, and joint
spectral recovery. Repeated roots, rank ambiguity, closure failure, poor
conditioning, and original-residual rejection remain terminal statuses. It is
not a fallback for the external path provider.

## Polynomial actions

Declared finite tables, reductive generator matrices, and diagonal weight
actions can produce verified invariant or equivariant coefficient subspaces.
Relation residuals, ambiguity bands, metric weights, and candidate isotypic
blocks are explicit. A supplied reductive action remains caller-declared, and a
Casimir block is not labelled irreducible without separate proof. Verified
bases lower to the existing enforcement representation contracts.

## Symmetric tensor decomposition

`phydrax.tensor_decomposition` supplies a resource-planned symmetric CP/Waring
path: catalecticant rank evidence, multiplication operators, joint spectral
recovery, deterministic scale/phase/permutation representatives, original
tensor replay, and optional native least-squares refinement. Nongeneric,
positive-dimensional, repeated-factor, and over-ranked requests fail explicitly.

## Multi-patch G1 constraints

The geometry surface package prepares immutable half-edge topology and sparse
G0/parametric-C1 constraints for supported tensor-product biquintic patches.
Prepared nullspaces and projections use native sparse/linalg owners. General
polynomial gluing data and extraordinary vertices are classified but remain
unsupported rather than receiving an approximate continuity claim.
