# Holomorphic potential fields

Phydrax represents selected two-dimensional PDE solutions through complex holomorphic
potentials. The complex model is certified separately from the physical transformation:
a holomorphic-map certificate proves the Cauchy–Riemann structure, while the wrapper
issues the Laplace, biharmonic, or plane-elasticity trial-space certificate.

## Linear polynomial potentials

`HolomorphicPolynomialPotential` is the deterministic baseline. It stores real and
imaginary coefficient leaves, evaluates branches with Horner recurrences, and computes
physical-coordinate derivatives analytically.

```python
import equinox as eqx
import jax.numpy as jnp
import phydrax as phx

potential = phx.equations.HolomorphicPolynomialPotential(1, 2)
potential = eqx.tree_at(
    lambda value: value.coefficient_real,
    potential,
    jnp.asarray([[0.0, 0.0, 1.0]]),
)
harmonic_model = phx.equations.HarmonicPotential2D(potential)
```

The resulting field is `Re(z²)`. `Domain.Model` attaches an algebraic Laplace
certificate, so boundary conditions compose through ordinary residual penalties while
generic hard enforcement is rejected.

## Holomorphic MLP

`phx.nn.models.HolomorphicMLP` uses dense or explicitly low-rank complex-affine
layers and the entire complex exponential. Every trainable leaf is real Cartesian
state; complex values are assembled only during evaluation. Split activations,
clipping, modulus operations, conjugation, dropout, and batch normalization are not
part of this contract.

```python
model = phx.nn.models.HolomorphicMLP(
    in_size=1,
    out_size=2,
    hidden_sizes=(16, 16),
)
```

The model accepts complex coordinates directly. Physical wrappers convert real `(x,y)`
coordinates to `z=x+iy` and request the required holomorphic jet.

Low-rank affine plans are declared per layer:

```python
factorized = phx.nn.models.HolomorphicMLP(
    in_size=1,
    out_size=2,
    hidden_sizes=(16, 16),
    linear_ranks=(1, 4, 2),
)
```

`LowRankComplexLinear` realizes one complex-affine map as two complex-linear
contractions without materializing the dense effective weight. Construction begins
from the existing dense initializer and records retained spectral energy and
truncation residual. This changes parameterization, not holomorphicity.

## Certified branch and product compositions

`HolomorphicBranchBundle` concatenates independent certified providers. It is useful
when the Goursat or elasticity branches need different architectures, normalizations,
or singular structure:

```python
potentials = phx.equations.HolomorphicBranchBundle((phi, psi))
```

The bundle concatenates child jets order by order. It remains a linear finite
subspace only when every child has that property.

`HolomorphicProductPotential` represents a finite sum of products evaluated at the
same scalar complex coordinate:

```text
F_b(z) = Σ_r Π_j f_{j,r,b}(z).
```

Every factor outputs `latent_rank * branches` values. Product jets are assembled by
convolving the factors' normalized Taylor coefficients, so higher derivatives use
the exact generalized Leibniz rule rather than nested differentiation.

```python
potential = phx.equations.HolomorphicProductPotential(
    (factor_a, factor_b),
    latent_rank=4,
    branches=2,
)
```

Multiplying two trainable factor spaces is nonlinear in the combined parameters even
when each factor is individually linear. The certificate therefore reports a finite
parametric family. `gauge_report` exposes the multiplicative factor-scale imbalance;
the implementation does not silently renormalize factors or optimizer state.

The current generic `Separable` and `LatentContractionModel` wrappers remain
uncertified. They accept arbitrary factors and activations and may project complex
outputs to real values.

## What separability does not mean

Arbitrary real-coordinate products such as `a(x)b(y)` are not generally
holomorphic in `z=x+iy`. Cauchy--Riemann compatibility ties such factors to a narrow
exponential-type family. Split `x`/`y` MLPs therefore cannot mint a holomorphic-map
certificate.

Parameter/query separation is safe when coefficients are independent of `z`:

```text
φ(z; μ) = Σ_r c_r(μ) h_r(z).
```

A conditional operator based on this form remains a separate follow-up because its
source-schema and query-holomorphic evidence differ from a pointwise potential
provider.

`HolomorphicMapCertificate` records parameter coverage and linearity independently.
Physical wrappers inherit both values without widening the claim. Polynomial potentials
declare a `finite-subspace` with linear coefficients; `HolomorphicMLP` declares a
`finite-parametric-family` with nonlinear parameter dependence.

## Physical representations

- `HarmonicPotential2D`: one potential branch and real scalar output.
- `BiharmonicPotential2D`: two Goursat branches and real scalar output.
- `PlaneElasticityPotential2D`: Kolosov–Muskhelishvili stress or mixed
  stress/displacement output.
- `PlaneIsotropicMaterial`: explicit plane-strain or plane-stress hypothesis.

Plane elasticity assumes a homogeneous isotropic material and zero body force. The
simply connected assumption concerns completeness of the potential representation, not
pointwise equilibrium of a represented state.

## Several complex variables

The neural architecture permits complex vector inputs, but real parts of holomorphic
maps on `C^m`, `m>1`, are pluriharmonic rather than a complete harmonic family. Phydrax
does not present this as a general nD Laplace solver. Use the nD harmonic polynomial
basis for general scalar harmonic fields.

## Optimizers and gauges

Complex trainable arrays are intentionally absent, so standard real Optax semantics
remain valid. `solve_linear_trial_space` separately checks certificate linearity:
linear polynomial physical fields are eligible; nonlinear HMLP fields are rejected.

Potential representations may have gauge/null modes. The direct solver reports rank and
nullity rather than hiding them. Pure-traction elasticity additionally requires the
usual force, moment, and rigid-motion treatment at the problem level.
