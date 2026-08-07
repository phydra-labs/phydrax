# Domain functions

`DomainFunction` carries a domain, an ordered dependency subset, an evaluator,
typed derivative rules, and immutable metadata. Construct fields through
`Domain.Function`, `Domain.Model`, or `Domain.Parameter` so the execution
protocol is explicit.

Ordinary callables are wrapped by `PointwiseEvaluator`. `FunctionBinding`
declares whether the callable consumes the keyword-only randomness key.
Grid-native, graph-native, or otherwise structured execution implements
`BatchEvaluator.__call_batch__`. Model execution uses the corresponding
`phx.nn.ModelBinding`; no hidden call keyword changes evaluator mode.

::: phydrax.domain.DomainFunction
    options:
        members:
            - __init__
            - depends_on
            - promote
            - with_metadata
            - T
            - __call__

---

::: phydrax.domain.FunctionBinding

---

::: phydrax.domain.PointwiseEvaluator

---

::: phydrax.domain.BatchEvaluator

---

::: phydrax.domain.DerivativeRule

---

::: phydrax.domain.CallbackDerivativeRule
