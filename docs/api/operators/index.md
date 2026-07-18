# Operators

Operators build PDE terms such as gradients, divergences, Laplacians, and integrals.

## Families

- **Differential**: $\nabla u$, $\nabla\cdot v$, $\Delta u$, surface and fractional operators.
- **Integral**: $\int_\Omega u\,d\Omega$, means, quadrature helpers, and convolution.
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
    - [Lagrangian and Hamiltonian mechanics](../../guides_mechanics.md)
    - [Quantum operators and dynamics](../../guides_quantum.md)
