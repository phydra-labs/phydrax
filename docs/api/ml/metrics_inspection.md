# Metrics and inspection

## Metrics and scorers

Every metric returns explicit value/status evidence through its documented result
contract. Exact label, order, rank, and cluster metrics are distinct from the
`smooth_*` probability, soft-order, and soft-assignment metrics. Output reduction,
averaging, gains, calibration norms, and empty/undefined policies are explicit.

::: phydrax.ml.metrics
    options:
        filters: ["!^_"]

## Model inspection

Gradient/Jacobian/Hessian sensitivity use the callable model's actual JAX program.
Partial dependence and permutation importance preserve case/sample geometry and
weights. Influence functions require the listed regularity of the fitted objective;
linear leverage and Cook's distance use exact model structure.

`inspect_spectral_neuron` reports the selected spectrum through invariant
cluster projectors. With eigenvalues `λ`, the selected numerical cluster uses
`τ = absolute_tolerance + relative_tolerance × max(1, maxⱼ |λⱼ|)` and includes
exactly the modes within `τ` of the model's selected eigenvalue. Exterior gaps
are measured from the cluster boundary; a missing endpoint neighbour is
reported as `+∞`.

A singleton tolerance cluster with exterior gaps greater than `τ` is reported
as numerically simple. This is conservative numerical evidence, not a claim
that two distinct eigenvalues inside `τ` are mathematically nondifferentiable.
Signed local sensitivities are returned only for a numerically simple selected
mode. Repeated or unresolved clusters instead return the basis-independent
bound `‖PAᵢP‖₂`, where `P` is the full cluster projector.

Global feature bounds `‖Aᵢ‖₂` and their perturbation enclosure are expressed in
the layer's current input units. The report never exposes a solver-selected
eigenvector or basis-dependent tie subgradient.

::: phydrax.ml.inspection.inspect_spectral_neuron

::: phydrax.ml.inspection.SpectralNeuronInspection

::: phydrax.ml.inspection
    options:
        filters: ["!^_"]
