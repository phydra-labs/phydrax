# Particle local solves and adaptive roots

`SmallLinearSolvePlan` provides explicit batched 1×1, 2×2, and 3×3 solves with
determinant, rank, condition, residual, refinement work, status, and success.
SPH first-order correction uses this substrate instead of forming a batch-global
dense problem.

`AdaptiveHRootPlan` solves the coupled density--smoothing-length relation in a
declared bracket with safeguarded Newton proposals and bisection fallback. The
result records h, density, residual, derivative, bracket width, evaluations,
bound activation, convergence, and success. `adaptive_h_implicit_tangent`
provides the converged frozen-bound implicit derivative.

Production qualification requires local residual and condition bounds, constant
and linear reproduction, grad-h energy balance, and relation support prepared
for the declared maximum h.
