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

## Pure out-of-fold assembly

`assemble_out_of_fold_predictions` consumes an existing
`CrossValidationResult` and its authoritative `MLBatch`. It performs no fitting
or prediction. The cross-validation result remains the sole owner of fold fits,
keys, scores, and split evidence; the assembler returns only dense scalar or
vector predictions in original sample order, a `sample_mask`, `fold_ids`, and
combined validity/status.
The prediction shape is
`batch.case_shape + (batch.sample_count,) + trailing_output_shape`;
`sample_mask` has `batch.case_shape + (batch.sample_count,)` and also respects
the batch's existing sample availability. `fold_ids` has shape
`(batch.sample_count,)`, while assembly `valid` and `status` are scalar.

Assembly requires the validation indices to cover every
`split_result.sample_indices` member exactly once and no other member. Locations
outside that selected universe contain nonsemantic zeros, `False`, and fold id
`-1`; consumers must use `sample_mask`, not inspect the fill value. PyTree
predictions, incompatible case/sample prefixes or trailing shapes, repeated or
missing validation rows, and nonpartitioning rolling holdouts are rejected.

When `MLBatch.groups` is present, group labels must agree across case lanes. A
group cannot be cut by the selected universe, divided among validation folds, or
shared between a fold's training and validation indices. These checks are local
to OOF assembly and do not change the broader cross-validation split contract.

::: phydrax.ml.model_selection.assemble_out_of_fold_predictions

---

::: phydrax.ml.model_selection.OutOfFoldPredictionResult

## Fixed conditional-loss workflow

Conditional-risk fitting uses ordinary, caller-chosen native recipes; there is no
automatic base/risk selector or bundled workflow model. Freeze both recipes and
their hyperparameters before constructing development OOF predictions. Searching
either recipe against one reused OOF table is leaky: targets from a would-be
risk-validation fold have already influenced many base fits that generated that
table. A future nested stack would have to regenerate base OOF features inside
every outer training partition.

The fixed lifecycle is:

1. make base predictions by fold-local development fits and assemble them once;
2. reduce each OOF prediction/target pair to one explicitly defined additive loss
   and fit the fixed risk recipe to those losses;
3. refit the fixed base recipe on the permitted full development role;
4. freeze both fits, optionally calibrate scalar intervals on an independent
   calibration role, and evaluate once on a locked test role.

Every data-dependent base or risk transform belongs inside its respective
`Pipeline`. Groups must remain whole across folds and lifecycle roles. The APIs
enforce in-memory fold geometry, but positional indices and group labels are not
audit-grade data lineage and no combined deployment artifact is implied.
