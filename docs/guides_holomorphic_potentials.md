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

`phx.nn.models.HolomorphicMLP` uses only `ComplexLinear` layers and the entire complex
exponential. Every trainable leaf is real Cartesian state; complex values are assembled
only during evaluation. Split activations, clipping, modulus operations, conjugation,
dropout, and batch normalization are not part of this contract.

```python
model = phx.nn.models.HolomorphicMLP(
    in_size=1,
    out_size=2,
    hidden_sizes=(16, 16),
)
```

The model accepts complex coordinates directly. Physical wrappers convert real `(x,y)`
coordinates to `z=x+iy` and request the required holomorphic jet.

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
