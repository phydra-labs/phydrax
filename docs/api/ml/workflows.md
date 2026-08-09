# Preprocessing and composition

## Preprocessing

Scalers, imputers, categorical encoders, basis expansions, projections, and
feature hashing fit immutable transform models. Exact category discovery,
ordinal ordering, quantile knots, and hash collisions remain discrete fit events.

::: phydrax.ml.preprocessing
    options:
        filters: ["!^_"]

## Composition

Pipelines, unions, column transforms, and transformed-target regression fit every
child inside the current training batch or fold. Fitted compositions retain child
schemas, diagnostics, provenance, and derivative contracts.

::: phydrax.ml.compose
    options:
        filters: ["!^_"]

## Model selection

Split plans make ordinary, stratified, grouped, blocked, rolling, and nested fold
geometry explicit. Search plans keep candidate status, metrics, fold evidence,
and selection nondifferentiability rather than treating the selected index as a
continuous parameter.

::: phydrax.ml.model_selection
    options:
        filters: ["!^_"]
