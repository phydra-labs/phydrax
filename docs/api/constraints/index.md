# Constraints

Constraints define objective terms for training or evaluation. They operate on domain
functions and typically return a scalar loss term.

For the mathematical conventions used by sampled constraints (residual norms, reduction modes,
measures, and filtering), see [Guides → Constraints and objectives](../../guides_constraints.md).

Stochastic continuous constraints provide stationary/evolution
backward-Kolmogorov and Fokker--Planck residuals. Probability-density positivity,
normalization, initial data, and boundary behavior remain explicit separate
constraints or ansätze.
