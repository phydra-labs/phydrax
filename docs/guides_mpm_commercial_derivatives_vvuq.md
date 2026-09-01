# Commercial MPM derivatives and VVUQ

## Derivative taxonomy

Every commercial derivative returns `MPMGradientResult` with one
`MPMDerivativeEvidence` tag:

```text
SMOOTH_DISCRETE
BRANCHWISE
EVENT_AWARE
GENERALIZED_SET
SURROGATE
STOCHASTIC_ESTIMATOR
NONDIFFERENTIABLE
```

No bare gradient is returned across routing, contact, adaptive, activation, or fracture
topology decisions.

- `branchwise_gradient` certifies JVP/VJP consistency and a positive branch margin.
- `smooth_surrogate_gradient` identifies the smoothed model and its bias bound.
- `locate_event` requires a scalar sign-changing event bracket and nonzero
  transversality.
- `saltation_action` applies a specified reset Jacobian to an isolated transversal
  event.
- `generalized_contact_derivative` returns the complete candidate set and selected
  rule rather than pretending it is one classical gradient.
- `stochastic_derivative_estimate` records samples and estimator variance.
- `nondifferentiable_result` returns a reason and journal digest.

Adaptive controller choices remain stopped. Scheduled replay differentiates the
accepted journal. Grazing/simultaneous events, chatter, route cardinality changes,
block-slot changes, crack branch/merge, remeshing, and particle lifecycle edits default
to `NONDIFFERENTIABLE` unless an explicit event/generalized/estimator contract applies.

## VVUQ gates

`MPMReleaseGate` implements G0–G7:

```text
G0 intended use
G1 code verification
G2 solution verification
G3 validation and UQ
G4 derivative product
G5 reproducibility, provenance, and SBOM
G6 quality, reliability, and security
G7 independent release decision
```

`MPMStandardsTraceabilityMatrix` records standard, exact edition, applicability,
requirement, evidence IDs, and status. It is traceability evidence, not certification.

The primary solid-mechanics framework is ASME V&V 10-2019 (R2025). V&V 20 applies to
CFD/heat portions only. V&V 40 applies to medical-device contexts. NASA-STD-7009B,
NAFEMS ESQMS/P18, ISO 9001/25010/12207, and NIST SSDF are profile-specific guidance.

Validation claims are always context-of-use specific. A code/solution-verified kernel
without independent physical data must not claim application validation.
