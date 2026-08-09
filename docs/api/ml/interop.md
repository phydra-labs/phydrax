# Artifacts, conversion, and export

## Native artifacts

Native ML artifacts are pickle-free, checksum-validated archives of registered
array-model structure, schemas, fit metadata, and provenance. Loading reconstructs
a native model; it never imports an external estimator implementation.

::: phydrax.ml.artifacts
    options:
        filters: ["!^_"]

## External fitted-model conversion

`from_sklearn` accepts only audited exact fitted classes and configurations.
`from_xgboost_artifact` parses saved JSON/UBJSON without importing XGBoost. Both
validate fitted state, copy prediction-affecting arrays and metadata once, record
source/license provenance, and fail closed on unsupported semantics.

`save_ml_onnx` delegates native callable export through the existing Phydrax ONNX
boundary and may validate representative inputs numerically.

::: phydrax.ml.interop
    options:
        filters: ["!^_"]
