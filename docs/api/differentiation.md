# Derivative contracts and ports

The `phydrax` root owns one derivative vocabulary shared by native solvers,
learned components, fitted ML models, and artifacts, together with the explicit
scientific ports through which models bind to their owners.

## Derivative vocabulary

A `DerivativeSurface` names the quantity a derivative is taken with respect to.
Capability surfaces must propagate through every participant of a combined
map; owned surfaces (`MODEL_PARAMETER`, `MODEL_STATE`, `EVENT`,
`STOCHASTIC_REALIZATION`) belong only to the components that hold them.
`GradientLevel` orders claims as smooth, almost-everywhere, conditional, and
none, but conditions are resolved before comparison: a claim whose conditions
are not known to hold is at most conditional.

::: phydrax.DerivativeSurface
    options:
      show_root_heading: true
      show_source: false

::: phydrax.is_owned_surface
    options:
      show_root_heading: true
      show_source: false

::: phydrax.GradientLevel
    options:
      show_root_heading: true
      show_source: false

::: phydrax.weakest_level
    options:
      show_root_heading: true
      show_source: false

::: phydrax.resolve_gradient_level
    options:
      show_root_heading: true
      show_source: false

::: phydrax.gradient_level_at_least
    options:
      show_root_heading: true
      show_source: false

::: phydrax.DerivativeRoute
    options:
      show_root_heading: true
      show_source: false

## Regularity

`DerivativeRegularity` declares the smoothness of a component map in its value
arguments (`INPUT`, `PRIMAL_STATE`) with an upper bound on piecewise-polynomial
degree. Sums take the maximum degree bound, products add bounds, composition
multiplies them, any smooth non-polynomial stage yields smooth pieces, and
continuity takes the minimum. Admission rejects only proven degeneracy: a ReLU
network with a linear output has a vanishing Laplacian, while a ReLU network
with a `tanh` output admits it almost everywhere with the
`singular-part-ignored` condition.

::: phydrax.DerivativeRegularity
    options:
      show_root_heading: true
      show_source: false

::: phydrax.RegularityPieces
    options:
      show_root_heading: true
      show_source: false

::: phydrax.RegularityPolicy
    options:
      show_root_heading: true
      show_source: false

::: phydrax.admit_regularity
    options:
      show_root_heading: true
      show_source: false

## Contracts and admission

A `DerivativeContract` is the canonical static declaration of supported
surfaces, route, regularity, conditions, and nondifferentiable outputs. Requests
are admitted before tracing; `require` raises a `ValueError` beginning with
`DERIVATIVE_UNSUPPORTED` that names the unsupported surfaces and reasons.
`meet` combines parallel parts, and `compose` combines sequential pipeline
stages through the downstream input. `DerivativeContract.smooth` declares a smooth
map on the given surfaces, and `supported_surfaces` lists the surfaces whose level
is not `NONE`; a contract without supported surfaces is constant.

::: phydrax.SurfaceDerivative
    options:
      show_root_heading: true
      show_source: false

::: phydrax.DerivativeContract
    options:
      show_root_heading: true
      show_source: false

::: phydrax.DifferentiationRequest
    options:
      show_root_heading: true
      show_source: false

::: phydrax.DerivativeAdmission
    options:
      show_root_heading: true
      show_source: false

::: phydrax.DERIVATIVE_SUPPORTED
    options:
      show_root_heading: true
      show_source: false

::: phydrax.DERIVATIVE_UNSUPPORTED
    options:
      show_root_heading: true
      show_source: false

## Branch differentiation

::: phydrax.BranchDifferentiationPolicy
    options:
      show_root_heading: true
      show_source: false

::: phydrax.branch_policy_contract
    options:
      show_root_heading: true
      show_source: false

## Authority and objectives

A component's authority determines which (route, objective kind) pairs may
train it. An accelerator never trains through an implicit solution map, because
the certified answer does not depend on its parameters.

::: phydrax.ComponentAuthority
    options:
      show_root_heading: true
      show_source: false

::: phydrax.ObjectiveKind
    options:
      show_root_heading: true
      show_source: false

::: phydrax.authority_admits
    options:
      show_root_heading: true
      show_source: false

## Capability evidence

Evidence kinds are not a total order. A `CapabilityRequirement` lists
acceptable combinations of evidence kinds, and declaration alone never
satisfies a safety-critical requirement.

::: phydrax.CapabilityEvidenceKind
    options:
      show_root_heading: true
      show_source: false

::: phydrax.CapabilityRequirement
    options:
      show_root_heading: true
      show_source: false

::: phydrax.AbstractConstructionCertificate
    options:
      show_root_heading: true
      show_source: false

## Ports

A `ValuePort` carries explicit scientific identity: semantic ID, support
space, event shape, component IDs, per-component dimensions, representation,
frame, normalization, semantic event axes, and variance. Ports bind only
through an explicit `PortMapping`; names, shapes, and order never establish
identity. `resolve_port_mapping` compares declared semantics strictly and
records every aspect a side left undeclared in `PortBindingEvidence`.

::: phydrax.ValuePort
    options:
      show_root_heading: true
      show_source: false

::: phydrax.PortVariance
    options:
      show_root_heading: true
      show_source: false

::: phydrax.ModelPorts
    options:
      show_root_heading: true
      show_source: false

::: phydrax.PortMapping
    options:
      show_root_heading: true
      show_source: false

::: phydrax.PortBindingEvidence
    options:
      show_root_heading: true
      show_source: false

::: phydrax.resolve_port_mapping
    options:
      show_root_heading: true
      show_source: false

::: phydrax.PortProvider
    options:
      show_root_heading: true
      show_source: false
