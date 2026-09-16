# Polynomial images and implicit equations

`phydrax.geometry.polynomial_image` discovers numerical structure in the image of an
explicit sparse polynomial map. Discovery is deliberately separate from exact ideal
equality, real geometry, and certified topology.

## Supported source data

The native workflow supports either:

- an affine-space source sampled from an explicit PRNG key; or
- caller-supplied source samples with retained qualification and identity.

It does not silently sample an arbitrary source ideal. Positive-dimensional source
varieties require a witness-set provider and explicit witness provenance.

## Image dimension

At each source sample, Phydrax evaluates the map Jacobian and records its complete
singular spectrum, selected numerical rank, gap, and condition evidence. A dimension
result succeeds only when independently sampled ranks agree under the declared rank
policy. Near-rank-deficient cases return numerical ambiguity instead of a confident
integer.

## Fixed-support interpolation

For a declared target monomial support, the workflow evaluates those monomials at
image samples, forms an interpolation matrix, and extracts its numerical nullspace.
Each null vector is normalized deterministically and interpreted as a candidate
polynomial relation.

Training residual and held-out residual are separate. A relation that vanishes only on
the interpolation samples is rejected.

Dense all-monomial support is optional and resource-planned. Prefer physically or
symmetry motivated support when available.

## Exact containment

When a candidate relation has exact rational coefficients, sparse exact composition
can prove that it vanishes identically after substitution through the polynomial map.
This establishes

```text
image(map) is contained in the candidate zero set.
```

It does not establish that the candidate equations generate the full image ideal or
that their zero set has no extra components. Exact elimination or radical evidence is
needed for that stronger claim.

## Geometry promotion

A numerical candidate never becomes a `CertifiedImplicitCover` directly. Promotion
requires an independent qualification appropriate to the downstream claim, such as:

- exact polynomial containment or ideal evidence;
- value and gradient bounds on the real domain;
- regularity evidence;
- the local topology conditions required by certified implicit meshing.

The existing implicit-surface discovery package remains responsible for turning an
admitted scalar field into a frozen mesh. Polynomial-image discovery only proposes
that field.
