# Operators

Operators build PDE terms such as gradients, divergences, Laplacians, and integrals.

## Families

- **Differential**: \(\nabla u\), \(\nabla\cdot v\), \(\Delta u\), stochastic
  generators/adjoints, surface operators, and fractional operators.
- **Integral**: $\int_\Omega u\,d\Omega$, means, quadrature helpers, and convolution.
- **Path integral**: Brownian bridges, Euclidean kernels, Feynman–Kac expectations,
  diffusion paths, and discrete first-passage observables.
- **Interpolation**: reusable anisotropic Smolyak surrogates returned as
  `DomainFunction` objects.
- **Functional**: norms, inner products, and averages.
- **Linear algebra**: determinants, traces, norms, and `einsum`-style contractions.
- **Delay**: delay operators for time-dependent fields.
- **Mechanics**: labeled pullbacks, Euler–Lagrange equations, canonical Hamiltonian
  dynamics, Poisson brackets, and Hamilton–Jacobi residuals.
- **Quantum**: complex matrix algebra, composite Hilbert spaces, state observables,
  information measures, physical densities, and closed- or open-system residuals.

!!! note
    Rich mathematical guides:

    - [Differential operators](../../guides_differential.md)
    - [Integrals and measures](../../guides_integrals.md)
    - [Interpolation and Smolyak surrogates](interpolation.md)
    - [Euclidean path integrals and Feynman–Kac expectations](../../guides_path_integrals.md)
    - [Lagrangian and Hamiltonian mechanics](../../guides_mechanics.md)
    - [Quantum operators and dynamics](../../guides_quantum.md)
