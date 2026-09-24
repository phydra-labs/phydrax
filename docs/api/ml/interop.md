# Artifacts, conversion, and export

## Native artifacts

Native ML artifacts are pickle-free, checksum-validated archives of registered
array-model structure and provenance. The schemas and derived ports recorded in
the manifest are those the fitted executable carries (`phydrax.ml.fit` binds
them), the derivative contract comes from the archived `FitResult`, and the
identity triplet (`SemanticProvenance`, `NumericRevision`,
`ExecutableSignature`) returned by `executable_identity` is recomputed from the
restored executable and verified on load. `load_ml_model` returns that schema-
and port-bound executable; records written in the previous format fail closed.
Loading never imports an external estimator implementation.

::: phydrax.ml.artifacts
    options:
        filters: ["!^_"]

## External fitted-model conversion

`from_sklearn` accepts only audited exact fitted classes and configurations.
`from_xgboost_artifact` parses saved JSON/UBJSON without importing XGBoost. Both
validate fitted state, copy prediction-affecting arrays and metadata once, and
fail closed on unsupported semantics. `ConversionProvenance` content-addresses
the source model, configuration, feature names, class labels, and license as a
`SemanticProvenance` whose `"source"` resource names the library and version.

`save_ml_onnx` delegates native callable export through the existing Phydrax ONNX
boundary and may validate representative inputs numerically.

::: phydrax.ml.interop
    options:
        filters: ["!^_"]
