# Unbalanced entropic transport

Unbalanced transport compares finite nonnegative measures whose physical masses need
not agree. PhydraX uses the explicit generalized-KL convention

\[
\min_{\pi\geq 0}
\langle C,\pi\rangle
+\varepsilon\,\mathrm{KL}(\pi\mid\alpha\otimes\beta)
+\rho_s\,\mathrm{KL}(\pi\mathbf 1\mid\alpha)
+\rho_t\,\mathrm{KL}(\pi^\mathsf T\mathbf 1\mid\beta),
\]

where
\(\mathrm{KL}(p\mid q)=\sum_i[p_i\log(p_i/q_i)-p_i+q_i]\).
The source and target penalties \(\rho_s\) and \(\rho_t\) are independent and
strictly positive. Inputs retain their declared physical mass, mask, event shape, and
provenance; the implementation neither clips weights nor silently normalizes them.

## Problem and solver

::: phydrax.transport.UnbalancedTransportProblem

::: phydrax.transport.unbalanced_problem

::: phydrax.transport.UnbalancedSinkhorn

::: phydrax.transport.UnbalancedSinkhornDiagnostics

::: phydrax.transport.UnbalancedSinkhornResult

The solver is stabilized in the log domain. Dense and blockwise execution solve the
same finite problem. Results expose transported mass, both relaxed marginals, the
physical plan and matrix-free plan actions, and the complete objective decomposition:
transport cost, entropy KL, source marginal KL, and target marginal KL. A coupling
at or below `mass_collapse_tolerance` receives the explicit
`TRANSPORT_MASS_COLLAPSED` status and is not reported as converged.

## Debiased divergence

For the product-reference entropy convention above, the unbalanced Sinkhorn
divergence is

\[
S_\varepsilon(\alpha,\beta)=
\mathrm{OT}_\varepsilon(\alpha,\beta)
-\tfrac12\mathrm{OT}_\varepsilon(\alpha,\alpha)
-\tfrac12\mathrm{OT}_\varepsilon(\beta,\beta)
+\tfrac{\varepsilon}{2}\bigl(m(\alpha)-m(\beta)\bigr)^2.
\]

The mass correction is part of the returned result, not an optional repair. All three
native solves are retained for status and convergence inspection.

::: phydrax.transport.UnbalancedSinkhornDivergenceResult

::: phydrax.transport.unbalanced_sinkhorn_divergence

::: phydrax.transport.PreparedUnbalancedSinkhornReference

::: phydrax.transport.prepare_unbalanced_sinkhorn_reference

::: phydrax.transport.unbalanced_sinkhorn_divergence_against

::: phydrax.transport.require_unbalanced_converged

## Scientific integrations

`phydrax.uq.spatial_unbalanced_sinkhorn_divergence` is restricted to physical spatial
or intensity measures. `phydrax.terms.SpatialUnbalancedSinkhornDivergenceTerm` uses a
prepared fixed reference and rejects every nonconverged training solve. Ordinary
normalized empirical predictive metrics continue to use balanced transport; unequal
mass should not be introduced there without a scientific mass interpretation.
