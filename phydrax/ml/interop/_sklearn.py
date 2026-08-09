#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._model import AbstractArrayModel
from ...kernels import SquaredExponentialKernel
from .._schema import FeatureSchema, TargetSchema
from ..clustering._common import HardClusterModel
from ..decomposition._subspace import SubspaceModel
from ..kernel_methods._estimators import (
    KernelRidgeModel,
    SupportVectorClassifierModel,
    SupportVectorRegressorModel,
)
from ..linear._base import (
    LinearRegressorModel,
    LogisticClassifierModel,
    MultinomialLogisticModel,
)
from ..linear._glm import GammaModel, PoissonModel, TweedieModel
from ..linear._least_squares import OLSModel, RidgeModel
from ..linear._sparse import ElasticNetModel, LassoModel
from ..mixture._gaussian import GaussianMixtureModel
from ..multiclass._models import OneVsOneModel
from ..preprocessing._categorical import (
    FittedOneHotEncoder,
    FittedOrdinalEncoder,
    FittedSimpleImputer,
)
from ..preprocessing._scalers import (
    FittedMaxAbsScaler,
    FittedMinMaxScaler,
    FittedRobustScaler,
    FittedStandardScaler,
)
from ..tree._representation import TreeEnsemble
from ._contracts import (
    ConversionError,
    ConversionProvenance,
    ConversionResult,
    UnsupportedConversionError,
)


_CONVERTER_SCHEMA = "phydrax.sklearn.fitted.v1"
_LICENSE = "BSD-3-Clause"
_NUMERIC_KINDS = frozenset("fiu")


@dataclass(frozen=True)
class _Converted:
    model: AbstractArrayModel
    configuration: dict[str, object]
    feature_names: tuple[str, ...] = ()
    class_labels: tuple[object, ...] = ()
    semantic_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class _SklearnAPI:
    version: str
    version_major: int
    version_minor: int
    check_is_fitted: Callable[..., None]
    not_fitted_error: type[Exception]
    registry: dict[type[object], Callable[[Any, "_Snapshot", "_SklearnAPI"], _Converted]]


class _Snapshot:
    """Validate and hash each source array at the one-time copy boundary."""

    def __init__(self) -> None:
        self._hash = hashlib.sha256()

    def _record(self, name: str, value: np.ndarray) -> None:
        contiguous = np.ascontiguousarray(value)
        self._hash.update(name.encode("utf-8"))
        self._hash.update(b"\0")
        self._hash.update(contiguous.dtype.str.encode("ascii"))
        self._hash.update(b"\0")
        self._hash.update(
            repr(tuple(int(size) for size in contiguous.shape)).encode("ascii")
        )
        self._hash.update(b"\0")
        self._hash.update(contiguous.tobytes(order="C"))

    def audit(
        self,
        name: str,
        value: Any,
        /,
        *,
        ndim: int | None = None,
        shape: tuple[int | None, ...] | None = None,
        finite: bool = True,
        integer: bool = False,
        boolean: bool = False,
    ) -> np.ndarray:
        array = np.asarray(value)
        if array.dtype.kind not in _NUMERIC_KINDS and not boolean:
            raise ConversionError(f"{name} must be a dense real numeric array.")
        if boolean and array.dtype.kind != "b":
            if array.dtype.kind not in "iu" or np.any((array != 0) & (array != 1)):
                raise ConversionError(f"{name} must contain only boolean values.")
            array = array.astype(bool, copy=False)
        if integer and array.dtype.kind not in "iu":
            raise ConversionError(f"{name} must be an integer array.")
        if ndim is not None and array.ndim != ndim:
            raise ConversionError(
                f"{name} must have rank {ndim}; got shape {array.shape}."
            )
        if shape is not None:
            if array.ndim != len(shape) or any(
                expected is not None and int(actual) != int(expected)
                for actual, expected in zip(array.shape, shape, strict=True)
            ):
                raise ConversionError(
                    f"{name} must have shape {shape}; got {array.shape}."
                )
        if finite and np.any(~np.isfinite(array)):
            raise ConversionError(f"{name} must contain only finite values.")
        self._record(name, array)
        return array

    def array(self, name: str, value: Any, /, **kwargs: Any) -> Array:
        source = self.audit(name, value, **kwargs)
        copied = jnp.asarray(source)
        if np.dtype(copied.dtype) != source.dtype:
            raise UnsupportedConversionError(
                f"{name} has dtype {source.dtype}, which the active JAX runtime cannot preserve."
            )
        return copied

    def configuration(self, configuration: dict[str, object]) -> None:
        for name, value in sorted(configuration.items()):
            self._hash.update(str(name).encode("utf-8"))
            self._hash.update(b"=")
            self._hash.update(repr(value).encode("utf-8"))
            self._hash.update(b"\0")

    def hexdigest(self) -> str:
        return self._hash.hexdigest()


def _parse_version(version: str, /) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        raise UnsupportedConversionError(
            f"Unrecognized scikit-learn version {version!r}; conversion is fail-closed."
        )
    major, minor = int(match.group(1)), int(match.group(2))
    if major != 1 or minor > 10:
        raise UnsupportedConversionError(
            f"scikit-learn {version} is outside the audited converter schema."
        )
    return major, minor


@cache
def _sklearn_api() -> _SklearnAPI:
    import sklearn
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.ensemble import (
        AdaBoostClassifier,
        AdaBoostRegressor,
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.exceptions import NotFittedError
    from sklearn.impute import SimpleImputer
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import (
        ElasticNet,
        GammaRegressor,
        Lasso,
        LinearRegression,
        LogisticRegression,
        MultiTaskElasticNet,
        MultiTaskLasso,
        PoissonRegressor,
        Ridge,
        TweedieRegressor,
    )
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import (
        MaxAbsScaler,
        MinMaxScaler,
        OneHotEncoder,
        OrdinalEncoder,
        RobustScaler,
        StandardScaler,
    )
    from sklearn.svm import NuSVC, NuSVR, SVC, SVR
    from sklearn.tree import (
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        ExtraTreeClassifier,
        ExtraTreeRegressor,
    )
    from sklearn.utils.validation import check_is_fitted

    version = str(sklearn.__version__)
    major, minor = _parse_version(version)
    registry: dict[type[object], Callable[[Any, _Snapshot, _SklearnAPI], _Converted]] = {
        StandardScaler: _convert_standard_scaler,
        MinMaxScaler: _convert_minmax_scaler,
        MaxAbsScaler: _convert_maxabs_scaler,
        RobustScaler: _convert_robust_scaler,
        SimpleImputer: _convert_simple_imputer,
        OneHotEncoder: _convert_one_hot_encoder,
        OrdinalEncoder: _convert_ordinal_encoder,
        LinearRegression: _convert_linear_regression,
        Ridge: _convert_ridge,
        Lasso: _convert_lasso,
        ElasticNet: _convert_elastic_net,
        MultiTaskLasso: _convert_lasso,
        MultiTaskElasticNet: _convert_elastic_net,
        LogisticRegression: _convert_logistic_regression,
        PoissonRegressor: _convert_poisson,
        GammaRegressor: _convert_gamma,
        TweedieRegressor: _convert_tweedie,
        PCA: _convert_pca,
        TruncatedSVD: _convert_truncated_svd,
        KMeans: _convert_kmeans,
        GaussianMixture: _convert_gaussian_mixture,
        KernelRidge: _convert_kernel_ridge,
        SVC: _convert_svc,
        NuSVC: _convert_svc,
        SVR: _convert_svr,
        NuSVR: _convert_svr,
        DecisionTreeClassifier: _convert_tree_classifier,
        DecisionTreeRegressor: _convert_tree_regressor,
        ExtraTreeClassifier: _convert_tree_classifier,
        ExtraTreeRegressor: _convert_tree_regressor,
        RandomForestClassifier: _convert_forest_classifier,
        RandomForestRegressor: _convert_forest_regressor,
        ExtraTreesClassifier: _convert_forest_classifier,
        ExtraTreesRegressor: _convert_forest_regressor,
        AdaBoostClassifier: _convert_adaboost_classifier,
        AdaBoostRegressor: _convert_adaboost_regressor,
        GradientBoostingClassifier: _convert_gradient_boosting_classifier,
        GradientBoostingRegressor: _convert_gradient_boosting_regressor,
    }
    return _SklearnAPI(
        version=version,
        version_major=major,
        version_minor=minor,
        check_is_fitted=check_is_fitted,
        not_fitted_error=NotFittedError,
        registry=registry,
    )


def _require_fitted(api: _SklearnAPI, estimator: Any, attributes: Sequence[str]) -> None:
    try:
        api.check_is_fitted(estimator, attributes=list(attributes))
    except api.not_fitted_error as error:
        raise ConversionError(
            f"The supported {type(estimator).__qualname__} instance is not fitted."
        ) from error


def _feature_schema(
    estimator: Any, feature_count: int, /
) -> tuple[FeatureSchema, tuple[str, ...]]:
    if feature_count <= 0:
        raise ConversionError("n_features_in_ must be positive.")
    source_names = estimator.__dict__.get("feature_names_in_")
    if source_names is None:
        return FeatureSchema.anonymous(feature_count), ()
    names_array = np.asarray(source_names)
    if names_array.shape != (feature_count,):
        raise ConversionError("feature_names_in_ does not match n_features_in_.")
    names = tuple(names_array.tolist())
    if any(type(name) is not str or not name for name in names):
        raise UnsupportedConversionError(
            "Only nonempty string feature_names_in_ values are representable."
        )
    if len(set(names)) != feature_count:
        raise ConversionError("feature_names_in_ must be unique.")
    return FeatureSchema(names), names


def _feature_count(estimator: Any, /) -> int:
    value = estimator.n_features_in_
    if isinstance(value, (bool, np.bool_)):
        raise ConversionError("n_features_in_ must be an integer.")
    count = int(value)
    if count != value or count <= 0:
        raise ConversionError("n_features_in_ must be a positive integer.")
    return count


def _numeric_scalar(value: Any, name: str, /, *, positive: bool = False) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in _NUMERIC_KINDS:
        raise ConversionError(f"{name} must be a real numeric scalar.")
    result = float(array)
    if not np.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ConversionError(f"{name} must be {qualifier}.")
    return result


def _vector(
    snapshot: _Snapshot,
    name: str,
    value: Any,
    size: int,
    /,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> Array:
    result = snapshot.array(name, value, shape=(size,))
    source = np.asarray(value)
    if positive and np.any(source <= 0):
        raise ConversionError(f"{name} must be strictly positive.")
    if nonnegative and np.any(source < 0):
        raise ConversionError(f"{name} must be nonnegative.")
    return result


def _labels(
    snapshot: _Snapshot, value: Any, /, *, minimum: int = 2
) -> tuple[Array, tuple[object, ...]]:
    source = np.asarray(value)
    if source.ndim != 1 or source.shape[0] < minimum:
        raise ConversionError(f"classes_ must contain at least {minimum} labels.")
    if source.dtype.kind not in "biuf" or np.any(~np.isfinite(source)):
        raise UnsupportedConversionError(
            "Only finite numeric class labels have an immutable JAX representation."
        )
    python_labels = tuple(
        item.item() if isinstance(item, np.generic) else item for item in source
    )
    if len(set(python_labels)) != len(python_labels):
        raise ConversionError("classes_ must contain unique labels in source order.")
    labels = snapshot.array(
        "classes",
        source,
        shape=(source.shape[0],),
        boolean=source.dtype.kind == "b",
    )
    return labels, python_labels


def _affine_configuration(estimator: Any, family: str) -> dict[str, object]:
    return {
        "family": family,
        "with_centering": bool(estimator.__dict__.get("with_centering", False)),
        "with_mean": bool(estimator.__dict__.get("with_mean", False)),
        "with_scaling": bool(estimator.__dict__.get("with_scaling", False)),
        "with_std": bool(estimator.__dict__.get("with_std", False)),
    }


def _convert_standard_scaler(
    estimator: Any, snapshot: _Snapshot, api: _SklearnAPI
) -> _Converted:
    _require_fitted(api, estimator, ("n_features_in_", "mean_", "var_", "scale_"))
    count = _feature_count(estimator)
    schema, names = _feature_schema(estimator, count)
    with_mean = bool(estimator.with_mean)
    with_std = bool(estimator.with_std)
    if with_mean:
        if estimator.mean_ is None:
            raise ConversionError("StandardScaler with_mean=True requires mean_.")
        center = _vector(snapshot, "mean", estimator.mean_, count)
    else:
        if estimator.mean_ is not None:
            snapshot.audit("mean_audit", estimator.mean_, shape=(count,))
        center = jnp.zeros(
            (count,), dtype=jnp.asarray(estimator.scale_ if with_std else 0.0).dtype
        )
    if with_std:
        if estimator.var_ is None or estimator.scale_ is None:
            raise ConversionError(
                "StandardScaler with_std=True requires var_ and scale_."
            )
        snapshot.audit("variance", estimator.var_, shape=(count,))
        if np.any(np.asarray(estimator.var_) < 0.0):
            raise ConversionError("StandardScaler var_ must be nonnegative.")
        scale = _vector(snapshot, "scale", estimator.scale_, count, positive=True)
    else:
        if estimator.scale_ is not None:
            raise ConversionError("StandardScaler with_std=False must have scale_=None.")
        if estimator.var_ is not None:
            snapshot.audit("variance_audit", estimator.var_, shape=(count,))
        scale = jnp.ones_like(center)
    model = FittedStandardScaler(center, scale, schema=schema, case_shape=())
    return _Converted(
        model,
        _affine_configuration(estimator, "standard"),
        names,
        semantic_notes=(
            "dense affine transform; learned zero-variance scale=1 preserved",
        ),
    )


def _convert_minmax_scaler(
    estimator: Any, snapshot: _Snapshot, api: _SklearnAPI
) -> _Converted:
    _require_fitted(
        api,
        estimator,
        ("n_features_in_", "scale_", "min_", "data_min_", "data_max_", "data_range_"),
    )
    count = _feature_count(estimator)
    schema, names = _feature_schema(estimator, count)
    scale = _vector(snapshot, "scale", estimator.scale_, count, positive=True)
    offset = _vector(snapshot, "min", estimator.min_, count)
    data_min = snapshot.audit("data_min", estimator.data_min_, shape=(count,))
    data_max = snapshot.audit("data_max", estimator.data_max_, shape=(count,))
    data_range = snapshot.audit("data_range", estimator.data_range_, shape=(count,))
    if np.any(data_max < data_min) or not np.allclose(
        data_max - data_min, data_range, rtol=1e-12, atol=1e-15
    ):
        raise ConversionError("MinMaxScaler learned extrema are inconsistent.")
    feature_range = tuple(float(item) for item in estimator.feature_range)
    if (
        len(feature_range) != 2
        or not np.all(np.isfinite(feature_range))
        or feature_range[0] >= feature_range[1]
    ):
        raise ConversionError("MinMaxScaler feature_range must be finite and increasing.")
    model = FittedMinMaxScaler(
        jnp.zeros_like(scale),
        jnp.reciprocal(scale),
        offset,
        schema=schema,
        case_shape=(),
        feature_range=feature_range,
        clip=bool(estimator.clip),
    )
    return _Converted(
        model,
        {
            "family": "min_max",
            "feature_range": feature_range,
            "clip": bool(estimator.clip),
        },
        names,
        semantic_notes=("dense x*scale_+min_ affine transform",),
    )


def _convert_maxabs_scaler(
    estimator: Any, snapshot: _Snapshot, api: _SklearnAPI
) -> _Converted:
    _require_fitted(api, estimator, ("n_features_in_", "max_abs_", "scale_"))
    count = _feature_count(estimator)
    schema, names = _feature_schema(estimator, count)
    max_abs = snapshot.audit("max_abs", estimator.max_abs_, shape=(count,))
    if np.any(max_abs < 0.0):
        raise ConversionError("MaxAbsScaler max_abs_ must be nonnegative.")
    scale = _vector(snapshot, "scale", estimator.scale_, count, positive=True)
    if max_abs.dtype.kind != "f":
        raise ConversionError("MaxAbsScaler learned scales must have floating dtype.")
    constant = max_abs < 10.0 * np.finfo(max_abs.dtype).eps
    expected = np.where(constant, 1.0, max_abs)
    if not np.array_equal(np.asarray(scale), expected):
        raise ConversionError("MaxAbsScaler scale_ is inconsistent with max_abs_.")
    model = FittedMaxAbsScaler(scale, schema=schema, case_shape=())
    return _Converted(
        model,
        {"family": "max_abs"},
        names,
        semantic_notes=("dense division by learned max-absolute scale",),
    )


def _convert_robust_scaler(
    estimator: Any, snapshot: _Snapshot, api: _SklearnAPI
) -> _Converted:
    _require_fitted(api, estimator, ("n_features_in_", "center_", "scale_"))
    count = _feature_count(estimator)
    schema, names = _feature_schema(estimator, count)
    with_centering = bool(estimator.with_centering)
    with_scaling = bool(estimator.with_scaling)
    if with_centering:
        if estimator.center_ is None:
            raise ConversionError("RobustScaler with_centering=True requires center_.")
        center = _vector(snapshot, "center", estimator.center_, count)
    else:
        if estimator.center_ is not None:
            raise ConversionError(
                "RobustScaler with_centering=False must have center_=None."
            )
        center = jnp.zeros((count,), dtype=float)
    if with_scaling:
        if estimator.scale_ is None:
            raise ConversionError("RobustScaler with_scaling=True requires scale_.")
        scale = _vector(snapshot, "scale", estimator.scale_, count, positive=True)
    else:
        if estimator.scale_ is not None:
            raise ConversionError(
                "RobustScaler with_scaling=False must have scale_=None."
            )
        scale = jnp.ones_like(center)
    quantile_range = tuple(float(item) for item in estimator.quantile_range)
    if len(quantile_range) != 2 or not (
        0.0 <= quantile_range[0] < quantile_range[1] <= 100.0
    ):
        raise ConversionError("RobustScaler quantile_range is invalid.")
    model = FittedRobustScaler(
        center,
        scale,
        schema=schema,
        case_shape=(),
        quantile_range=quantile_range,
    )
    return _Converted(
        model,
        {
            "family": "robust",
            "with_centering": with_centering,
            "with_scaling": with_scaling,
            "quantile_range": quantile_range,
            "unit_variance": bool(estimator.unit_variance),
        },
        names,
        semantic_notes=("dense median/quantile affine transform",),
    )


def _convert_simple_imputer(
    estimator: Any, snapshot: _Snapshot, api: _SklearnAPI
) -> _Converted:
    _require_fitted(api, estimator, ("n_features_in_", "statistics_"))
    count = _feature_count(estimator)
    schema, names = _feature_schema(estimator, count)
    if bool(estimator.add_indicator):
        raise UnsupportedConversionError(
            "SimpleImputer add_indicator=True selects fit-dependent indicator columns not represented natively."
        )
    if not isinstance(estimator.strategy, str) or estimator.strategy not in {
        "mean",
        "median",
        "most_frequent",
        "constant",
    }:
        raise UnsupportedConversionError(
            "Callable or unknown SimpleImputer strategies are unsupported."
        )
    fill = _vector(snapshot, "statistics", estimator.statistics_, count)
    missing = np.asarray(estimator.missing_values)
    if missing.ndim != 0 or missing.dtype.kind not in _NUMERIC_KINDS:
        raise UnsupportedConversionError(
            "SimpleImputer missing_values must be a numeric scalar or NaN."
        )
    missing_value = missing.item()
    missing_is_nan = bool(
        np.issubdtype(missing.dtype, np.floating) and np.isnan(missing_value)
    )
    if not missing_is_nan and not np.isfinite(missing_value):
        raise UnsupportedConversionError(
            "Only finite numeric or NaN missing sentinels are supported."
        )
    model = FittedSimpleImputer(
        fill,
        missing_values=missing_value,
        missing_is_nan=missing_is_nan,
        add_indicator=False,
        input_schema=schema,
        output_schema=schema,
        case_shape=(),
    )
    return _Converted(
        model,
        {
            "family": "simple_imputer",
            "strategy": estimator.strategy,
            "missing_values": missing_value,
            "add_indicator": False,
            "keep_empty_features": bool(
                estimator.__dict__.get("keep_empty_features", False)
            ),
        },
        names,
        semantic_notes=(
            "all learned statistics are finite, so no fit-dependent feature deletion occurs",
        ),
    )


def _categorical_bank(
    snapshot: _Snapshot, categories: Any, count: int, /
) -> tuple[Array, Array, tuple[int, ...], tuple[tuple[object, ...], ...]]:
    if not isinstance(categories, (list, tuple)) or len(categories) != count:
        raise ConversionError(
            "categories_ must contain one vocabulary per input feature."
        )
    arrays: list[np.ndarray] = []
    vocabularies: list[tuple[object, ...]] = []
    dtype: np.dtype[Any] | None = None
    for feature, values in enumerate(categories):
        array = np.asarray(values)
        if array.ndim != 1 or array.shape[0] == 0:
            raise ConversionError(f"categories_[{feature}] must be a nonempty vector.")
        if array.dtype.kind not in "biuf" or np.any(~np.isfinite(array)):
            raise UnsupportedConversionError(
                "Categorical vocabularies must contain finite numeric scalars only."
            )
        if dtype is None:
            dtype = array.dtype
        elif array.dtype != dtype:
            raise UnsupportedConversionError(
                "All categorical vocabularies must share one exact numeric dtype."
            )
        vocabulary = tuple(
            item.item() if isinstance(item, np.generic) else item for item in array
        )
        if len(set(vocabulary)) != len(vocabulary):
            raise ConversionError(f"categories_[{feature}] contains duplicate values.")
        arrays.append(array)
        vocabularies.append(vocabulary)
    capacity = max(array.shape[0] for array in arrays)
    assert dtype is not None
    bank = np.zeros((count, capacity), dtype=dtype)
    valid = np.zeros((count, capacity), dtype=bool)
    offsets = [0]
    for feature, array in enumerate(arrays):
        size = int(array.shape[0])
        bank[feature, :size] = array
        valid[feature, :size] = True
        offsets.append(offsets[-1] + size)
    return (
        snapshot.array(
            "categories",
            bank,
            shape=(count, capacity),
            boolean=bank.dtype.kind == "b",
        ),
        snapshot.array("category_valid", valid, shape=(count, capacity), boolean=True),
        tuple(offsets),
        tuple(vocabularies),
    )


def _categorical_output_schema(
    input_schema: FeatureSchema,
    vocabularies: tuple[tuple[object, ...], ...],
    /,
) -> FeatureSchema:
    names = tuple(
        f"{input_schema.names[feature]}__category_{category}"
        for feature, values in enumerate(vocabularies)
        for category in range(len(values))
    )
    return FeatureSchema(names, kinds=("boolean",) * len(names))


def _reject_infrequent_encoder(estimator: Any) -> None:
    if (
        estimator.__dict__.get("min_frequency") is not None
        or estimator.__dict__.get("max_categories") is not None
    ):
        raise UnsupportedConversionError(
            "Infrequent-category grouping uses version-dependent private mappings and is unsupported."
        )


def _convert_one_hot_encoder(
    estimator: Any, snapshot: _Snapshot, api: _SklearnAPI
) -> _Converted:
    _require_fitted(api, estimator, ("n_features_in_", "categories_", "drop_idx_"))
    count = _feature_count(estimator)
    input_schema, names = _feature_schema(estimator, count)
    _reject_infrequent_encoder(estimator)
    if estimator.handle_unknown != "error":
        raise UnsupportedConversionError(
            "Only OneHotEncoder(handle_unknown='error') has the native fail-on-unknown semantics."
        )
    if estimator.drop is not None or estimator.drop_idx_ is not None:
        raise UnsupportedConversionError(
            "Dropped one-hot categories are not represented by the native encoder."
        )
    sparse_output = estimator.__dict__.get(
        "sparse_output", estimator.__dict__.get("sparse", True)
    )
    if bool(sparse_output):
        raise UnsupportedConversionError(
            "Sparse OneHotEncoder output is unsupported by the dense native encoder."
        )
    feature_name_combiner = estimator.__dict__.get("feature_name_combiner", "concat")
    if type(feature_name_combiner) is not str or feature_name_combiner != "concat":
        raise UnsupportedConversionError(
            "Custom OneHotEncoder feature-name combiners are unsupported."
        )
    categories, valid, offsets, vocabularies = _categorical_bank(
        snapshot, estimator.categories_, count
    )
    output_schema = _categorical_output_schema(input_schema, vocabularies)
    model = FittedOneHotEncoder(
        categories,
        valid,
        offsets=offsets,
        unknown_policy="fail",
        input_schema=input_schema,
        output_schema=output_schema,
    )
    return _Converted(
        model,
        {
            "family": "one_hot",
            "handle_unknown": "error",
            "drop": None,
            "sparse_output": False,
            "dtype": np.dtype(estimator.dtype).str,
        },
        names,
        semantic_notes=(
            "numeric categories, full dense blocks, source category order preserved",
        ),
    )


def _convert_ordinal_encoder(
    estimator: Any, snapshot: _Snapshot, api: _SklearnAPI
) -> _Converted:
    _require_fitted(api, estimator, ("n_features_in_", "categories_"))
    count = _feature_count(estimator)
    input_schema, names = _feature_schema(estimator, count)
    _reject_infrequent_encoder(estimator)
    if estimator.handle_unknown != "error":
        raise UnsupportedConversionError(
            "Only OrdinalEncoder(handle_unknown='error') is exactly represented natively."
        )
    categories, valid, _, vocabularies = _categorical_bank(
        snapshot, estimator.categories_, count
    )
    output_schema = FeatureSchema(input_schema.names, kinds=("ordinal",) * count)
    model = FittedOrdinalEncoder(
        categories,
        valid,
        unknown_policy="fail",
        unknown_value=-1,
        input_schema=input_schema,
        output_schema=output_schema,
    )
    return _Converted(
        model,
        {
            "family": "ordinal",
            "handle_unknown": "error",
            "dtype": np.dtype(estimator.dtype).str,
            "category_counts": tuple(len(values) for values in vocabularies),
        },
        names,
        semantic_notes=("numeric categories and source category-index order preserved",),
    )


def _linear_state(
    estimator: Any,
    snapshot: _Snapshot,
    api: _SklearnAPI,
    /,
) -> tuple[Array, Array, int, tuple[int, ...], tuple[str, ...]]:
    _require_fitted(api, estimator, ("n_features_in_", "coef_", "intercept_"))
    features = _feature_count(estimator)
    _, names = _feature_schema(estimator, features)
    coefficients = np.asarray(estimator.coef_)
    intercept = np.asarray(estimator.intercept_)
    if coefficients.dtype.kind != "f" or np.any(~np.isfinite(coefficients)):
        raise ConversionError("coef_ must be a finite dense real floating array.")
    if coefficients.ndim == 1:
        if coefficients.shape != (features,) or intercept.ndim != 0:
            raise ConversionError(
                "Scalar-output linear coefficient/intercept shapes are inconsistent."
            )
        beta = snapshot.array("coefficients", coefficients[:, None], shape=(features, 1))
        bias = snapshot.array("intercept", intercept.reshape(1), shape=(1,))
        target_shape: tuple[int, ...] = ()
    elif coefficients.ndim == 2:
        outputs = int(coefficients.shape[0])
        if (
            outputs <= 0
            or coefficients.shape[1] != features
            or intercept.shape != (outputs,)
        ):
            raise ConversionError(
                "Multi-output linear coefficient/intercept shapes are inconsistent."
            )
        beta = snapshot.array("coefficients", coefficients.T, shape=(features, outputs))
        bias = snapshot.array("intercept", intercept, shape=(outputs,))
        target_shape = (outputs,)
    else:
        raise ConversionError("coef_ must have rank one or two.")
    return beta, bias, features, target_shape, names


def _linear_regression_result(
    estimator: Any,
    snapshot: _Snapshot,
    api: _SklearnAPI,
    model_type: type[LinearRegressorModel],
    family: str,
    configuration: dict[str, object],
) -> _Converted:
    beta, bias, _, target_shape, names = _linear_state(estimator, snapshot, api)
    model = model_type(beta, bias, case_shape=(), target_shape=target_shape)
    config = {"family": family, "fit_intercept": bool(estimator.fit_intercept)}
    config.update(configuration)
    return _Converted(
        model,
        config,
        names,
        semantic_notes=(
            "dense affine prediction with source multi-output orientation preserved",
        ),
    )


def _convert_linear_regression(
    estimator: Any, snapshot: _Snapshot, api: _SklearnAPI
) -> _Converted:
    return _linear_regression_result(
        estimator,
        snapshot,
        api,
        OLSModel,
        "linear_regression",
        {"positive": bool(estimator.positive)},
    )


def _convert_ridge(estimator: Any, snapshot: _Snapshot, api: _SklearnAPI) -> _Converted:
    alpha = np.asarray(estimator.alpha)
    if (
        alpha.ndim > 1
        or alpha.dtype.kind not in _NUMERIC_KINDS
        or np.any(~np.isfinite(alpha))
        or np.any(alpha < 0)
    ):
        raise ConversionError("Ridge alpha must be finite and nonnegative.")
    return _linear_regression_result(
        estimator,
        snapshot,
        api,
        RidgeModel,
        "ridge",
        {"alpha": tuple(alpha.reshape(-1).tolist()), "solver": estimator.solver},
    )


def _convert_lasso(estimator: Any, snapshot: _Snapshot, api: _SklearnAPI) -> _Converted:
    alpha = _numeric_scalar(estimator.alpha, "alpha")
    if alpha < 0.0:
        raise ConversionError("Lasso alpha must be nonnegative.")
    return _linear_regression_result(
        estimator,
        snapshot,
        api,
        LassoModel,
        "lasso",
        {"alpha": alpha},
    )


def _convert_elastic_net(
    estimator: Any, snapshot: _Snapshot, api: _SklearnAPI
) -> _Converted:
    alpha = _numeric_scalar(estimator.alpha, "alpha")
    ratio = _numeric_scalar(estimator.l1_ratio, "l1_ratio")
    if alpha < 0.0 or not 0.0 <= ratio <= 1.0:
        raise ConversionError("ElasticNet requires alpha >= 0 and l1_ratio in [0, 1].")
    return _linear_regression_result(
        estimator,
        snapshot,
        api,
        ElasticNetModel,
        "elastic_net",
        {"alpha": alpha, "l1_ratio": ratio},
    )


def _convert_logistic_regression(
    estimator: Any, snapshot: _Snapshot, api: _SklearnAPI
) -> _Converted:
    _require_fitted(api, estimator, ("n_features_in_", "coef_", "intercept_", "classes_"))
    features = _feature_count(estimator)
    _, names = _feature_schema(estimator, features)
    labels, python_labels = _labels(snapshot, estimator.classes_)
    classes = len(python_labels)
    coefficients = np.asarray(estimator.coef_)
    intercept = np.asarray(estimator.intercept_)
    if coefficients.dtype.kind != "f" or intercept.dtype.kind != "f":
        raise ConversionError(
            "LogisticRegression coefficients must be dense floating arrays."
        )
    if np.any(~np.isfinite(coefficients)) or np.any(~np.isfinite(intercept)):
        raise ConversionError("LogisticRegression coefficients must be finite.")
    solver = str(estimator.solver)
    if classes == 2:
        if coefficients.shape != (1, features) or intercept.shape != (1,):
            raise ConversionError(
                "Binary LogisticRegression state has inconsistent shapes."
            )
        beta = snapshot.array("coefficients", coefficients.T, shape=(features, 1))
        bias = snapshot.array("intercept", intercept, shape=(1,))
        model: AbstractArrayModel = LogisticClassifierModel(
            beta,
            bias,
            labels,
            case_shape=(),
            target_shape=(),
        )
        convention = "binary_sigmoid_classes_1_score"
    else:
        multi_class = estimator.__dict__.get("multi_class")
        multinomial = solver != "liblinear" and multi_class in {
            None,
            "auto",
            "deprecated",
            "multinomial",
        }
        if not multinomial:
            raise UnsupportedConversionError(
                "Multiclass one-vs-rest LogisticRegression probabilities are not "
                "represented by the native softmax model."
            )
        if coefficients.shape != (classes, features) or intercept.shape != (classes,):
            raise ConversionError(
                "Multinomial LogisticRegression state has inconsistent shapes."
            )
        beta = snapshot.array("coefficients", coefficients.T, shape=(features, classes))
        bias = snapshot.array("intercept", intercept, shape=(classes,))
        model = MultinomialLogisticModel(beta, bias, labels, case_shape=())
        convention = "multinomial_softmax"
    return _Converted(
        model,
        {
            "family": "logistic_regression",
            "solver": solver,
            "fit_intercept": bool(estimator.fit_intercept),
            "class_convention": convention,
        },
        names,
        python_labels,
        ("classes_ order and binary positive-class coefficient convention preserved",),
    )


def _glm_state(
    estimator: Any,
    snapshot: _Snapshot,
    api: _SklearnAPI,
) -> tuple[Array, Array, tuple[str, ...]]:
    beta, bias, _, target_shape, names = _linear_state(estimator, snapshot, api)
    if target_shape:
        raise ConversionError(
            "scikit-learn generalized linear regressors must be scalar-output."
        )
    return beta, bias, names


def _convert_poisson(estimator: Any, snapshot: _Snapshot, api: _SklearnAPI) -> _Converted:
    beta, bias, names = _glm_state(estimator, snapshot, api)
    model = PoissonModel(beta, bias, case_shape=(), target_shape=(), inverse_link="exp")
    return _Converted(
        model,
        {
            "family": "poisson",
            "inverse_link": "exp",
            "fit_intercept": bool(estimator.fit_intercept),
        },
        names,
        semantic_notes=(
            "stored affine predictor followed by the exact exponential mean link",
        ),
    )


def _convert_gamma(estimator: Any, snapshot: _Snapshot, api: _SklearnAPI) -> _Converted:
    beta, bias, names = _glm_state(estimator, snapshot, api)
    model = GammaModel(beta, bias, case_shape=(), target_shape=(), inverse_link="exp")
    return _Converted(
        model,
        {
            "family": "gamma",
            "inverse_link": "exp",
            "fit_intercept": bool(estimator.fit_intercept),
        },
        names,
        semantic_notes=(
            "stored affine predictor followed by the exact exponential mean link",
        ),
    )


def _convert_tweedie(estimator: Any, snapshot: _Snapshot, api: _SklearnAPI) -> _Converted:
    beta, bias, names = _glm_state(estimator, snapshot, api)
    power = _numeric_scalar(estimator.power, "power")
    link = str(estimator.link)
    resolved_link = (
        "exp" if link == "log" or (link == "auto" and power > 0.0) else "identity"
    )
    if link not in {"auto", "log", "identity"}:
        raise UnsupportedConversionError(f"Unsupported Tweedie link {link!r}.")
    model = TweedieModel(
        beta,
        bias,
        case_shape=(),
        target_shape=(),
        inverse_link=resolved_link,
        power=power,
    )
    return _Converted(
        model,
        {
            "family": "tweedie",
            "power": power,
            "link": link,
            "resolved_inverse_link": resolved_link,
            "fit_intercept": bool(estimator.fit_intercept),
        },
        names,
        semantic_notes=(
            "stored affine predictor and fitted Tweedie link resolved explicitly",
        ),
    )


def _convert_pca(estimator: Any, snapshot: _Snapshot, api: _SklearnAPI) -> _Converted:
    _require_fitted(
        api,
        estimator,
        (
            "n_features_in_",
            "components_",
            "mean_",
            "explained_variance_",
            "explained_variance_ratio_",
            "singular_values_",
            "n_components_",
        ),
    )
    if bool(estimator.whiten):
        raise UnsupportedConversionError(
            "Whitened PCA transform and inverse_transform cannot both be represented by SubspaceModel."
        )
    features = _feature_count(estimator)
    _, names = _feature_schema(estimator, features)
    components_source = np.asarray(estimator.components_)
    if (
        components_source.ndim != 2
        or components_source.shape[1] != features
        or components_source.shape[0] <= 0
    ):
        raise ConversionError("PCA components_ has an invalid shape.")
    components = snapshot.array(
        "components", components_source, shape=components_source.shape
    )
    count = int(components_source.shape[0])
    if int(estimator.n_components_) != count:
        raise ConversionError("PCA n_components_ does not match components_.")
    mean = _vector(snapshot, "mean", estimator.mean_, features)
    explained_variance = snapshot.audit(
        "explained_variance", estimator.explained_variance_, shape=(count,)
    )
    if np.any(explained_variance < 0.0):
        raise ConversionError("PCA explained_variance_ must be nonnegative.")
    snapshot.audit(
        "explained_variance_ratio", estimator.explained_variance_ratio_, shape=(count,)
    )
    singular = _vector(
        snapshot, "singular_values", estimator.singular_values_, count, nonnegative=True
    )
    model = SubspaceModel(
        mean,
        components,
        jnp.ones_like(mean),
        jnp.ones((features,), dtype=bool),
        singular,
        centered=True,
        weighting_provenance="sklearn-unweighted-euclidean",
        centering_provenance="copied-mean_",
        mask_provenance="all-features-supported",
    )
    return _Converted(
        model,
        {
            "family": "pca",
            "n_components": count,
            "whiten": False,
            "svd_solver": estimator.svd_solver,
        },
        names,
        semantic_notes=(
            "transform and inverse_transform preserve copied unwhitened components and mean",
        ),
    )


def _convert_truncated_svd(
    estimator: Any, snapshot: _Snapshot, api: _SklearnAPI
) -> _Converted:
    _require_fitted(
        api,
        estimator,
        (
            "n_features_in_",
            "components_",
            "explained_variance_",
            "explained_variance_ratio_",
            "singular_values_",
        ),
    )
    features = _feature_count(estimator)
    _, names = _feature_schema(estimator, features)
    components_source = np.asarray(estimator.components_)
    if (
        components_source.ndim != 2
        or components_source.shape[1] != features
        or components_source.shape[0] <= 0
    ):
        raise ConversionError("TruncatedSVD components_ has an invalid shape.")
    count = int(components_source.shape[0])
    components = snapshot.array("components", components_source, shape=(count, features))
    snapshot.audit("explained_variance", estimator.explained_variance_, shape=(count,))
    snapshot.audit(
        "explained_variance_ratio", estimator.explained_variance_ratio_, shape=(count,)
    )
    singular = _vector(
        snapshot, "singular_values", estimator.singular_values_, count, nonnegative=True
    )
    offset = jnp.zeros((features,), dtype=components.dtype)
    model = SubspaceModel(
        offset,
        components,
        jnp.ones_like(offset),
        jnp.ones((features,), dtype=bool),
        singular,
        centered=False,
        weighting_provenance="sklearn-unweighted-euclidean",
        centering_provenance="origin-anchored",
        mask_provenance="all-features-supported",
    )
    return _Converted(
        model,
        {
            "family": "truncated_svd",
            "n_components": count,
            "algorithm": estimator.algorithm,
        },
        names,
        semantic_notes=(
            "origin-anchored transform and inverse_transform use copied components",
        ),
    )


def _convert_kmeans(estimator: Any, snapshot: _Snapshot, api: _SklearnAPI) -> _Converted:
    _require_fitted(
        api,
        estimator,
        ("n_features_in_", "cluster_centers_", "labels_", "inertia_", "n_iter_"),
    )
    features = _feature_count(estimator)
    _, names = _feature_schema(estimator, features)
    centers_source = np.asarray(estimator.cluster_centers_)
    if (
        centers_source.ndim != 2
        or centers_source.shape[1] != features
        or centers_source.shape[0] <= 0
    ):
        raise ConversionError("KMeans cluster_centers_ has an invalid shape.")
    clusters = int(centers_source.shape[0])
    if int(estimator.n_clusters) != clusters:
        raise ConversionError("KMeans n_clusters does not match cluster_centers_.")
    centers = snapshot.array(
        "cluster_centers", centers_source, shape=(clusters, features)
    )
    labels = snapshot.audit("fit_labels", estimator.labels_, ndim=1, integer=True)
    if np.any(labels < 0) or np.any(labels >= clusters):
        raise ConversionError("KMeans labels_ contains an out-of-range cluster index.")
    inertia = _numeric_scalar(estimator.inertia_, "inertia_")
    if inertia < 0.0 or int(estimator.n_iter_) <= 0:
        raise ConversionError("KMeans inertia_ or n_iter_ is invalid.")
    model = HardClusterModel(
        centers,
        jnp.ones((clusters,), dtype=bool),
        metric="squared-euclidean",
        method="sklearn-k-means",
    )
    return _Converted(
        model,
        {
            "family": "kmeans",
            "cluster_count": clusters,
            "metric": "squared_euclidean",
            "algorithm": estimator.algorithm,
        },
        names,
        semantic_notes=(
            "predict uses source-order nearest-center argmin; early-stop labels_ are provenance only",
        ),
    )


def _expand_gaussian_geometry(
    covariance_type: str,
    covariance: np.ndarray,
    precision_cholesky: np.ndarray,
    components: int,
    features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if covariance_type == "full":
        if (
            covariance.shape != (components, features, features)
            or precision_cholesky.shape != covariance.shape
        ):
            raise ConversionError(
                "Full GaussianMixture covariance state has inconsistent shapes."
            )
        covariance_full = covariance
        precision_full = precision_cholesky @ np.swapaxes(precision_cholesky, -1, -2)
        diagonal = np.diagonal(precision_cholesky, axis1=-2, axis2=-1)
        native_type = "full"
    elif covariance_type == "tied":
        if (
            covariance.shape != (features, features)
            or precision_cholesky.shape != covariance.shape
        ):
            raise ConversionError(
                "Tied GaussianMixture covariance state has inconsistent shapes."
            )
        covariance_full = np.broadcast_to(
            covariance, (components, features, features)
        ).copy()
        precision_one = precision_cholesky @ precision_cholesky.T
        precision_full = np.broadcast_to(precision_one, covariance_full.shape).copy()
        diagonal = np.broadcast_to(np.diag(precision_cholesky), (components, features))
        native_type = "tied"
    elif covariance_type == "diag":
        if (
            covariance.shape != (components, features)
            or precision_cholesky.shape != covariance.shape
        ):
            raise ConversionError(
                "Diagonal GaussianMixture covariance state has inconsistent shapes."
            )
        covariance_full = np.zeros(
            (components, features, features), dtype=covariance.dtype
        )
        precision_full = np.zeros_like(covariance_full)
        index = np.arange(features)
        covariance_full[:, index, index] = covariance
        precision_full[:, index, index] = precision_cholesky * precision_cholesky
        diagonal = precision_cholesky
        native_type = "diagonal"
    elif covariance_type == "spherical":
        if (
            covariance.shape != (components,)
            or precision_cholesky.shape != covariance.shape
        ):
            raise ConversionError(
                "Spherical GaussianMixture covariance state has inconsistent shapes."
            )
        identity = np.eye(features, dtype=covariance.dtype)
        covariance_full = covariance[:, None, None] * identity
        precision_full = (precision_cholesky * precision_cholesky)[
            :, None, None
        ] * identity
        diagonal = np.broadcast_to(precision_cholesky[:, None], (components, features))
        native_type = "spherical"
    else:
        raise UnsupportedConversionError(
            f"Unsupported GaussianMixture covariance_type {covariance_type!r}."
        )
    if np.any(diagonal <= 0.0):
        raise ConversionError(
            "GaussianMixture precision Cholesky diagonals must be positive."
        )
    log_determinant = -2.0 * np.sum(np.log(diagonal), axis=-1)
    return covariance_full, precision_full, log_determinant, native_type


def _convert_gaussian_mixture(
    estimator: Any, snapshot: _Snapshot, api: _SklearnAPI
) -> _Converted:
    _require_fitted(
        api,
        estimator,
        (
            "n_features_in_",
            "weights_",
            "means_",
            "covariances_",
            "precisions_",
            "precisions_cholesky_",
            "converged_",
            "n_iter_",
        ),
    )
    features = _feature_count(estimator)
    _, names = _feature_schema(estimator, features)
    means_source = np.asarray(estimator.means_)
    if (
        means_source.ndim != 2
        or means_source.shape[1] != features
        or means_source.shape[0] <= 0
    ):
        raise ConversionError("GaussianMixture means_ has an invalid shape.")
    components = int(means_source.shape[0])
    if int(estimator.n_components) != components:
        raise ConversionError("GaussianMixture n_components does not match means_.")
    weights_source = snapshot.audit(
        "mixing_weights", estimator.weights_, shape=(components,)
    )
    if np.any(weights_source <= 0.0) or not np.isclose(
        np.sum(weights_source), 1.0, rtol=1e-10, atol=1e-12
    ):
        raise ConversionError("GaussianMixture weights_ must be positive and sum to one.")
    covariance = snapshot.audit("source_covariances", estimator.covariances_)
    precision = snapshot.audit("source_precisions", estimator.precisions_)
    precision_cholesky = snapshot.audit(
        "precision_cholesky", estimator.precisions_cholesky_
    )
    covariance_full, precision_full, log_determinant, native_type = (
        _expand_gaussian_geometry(
            str(estimator.covariance_type),
            covariance,
            precision_cholesky,
            components,
            features,
        )
    )
    expected_precision_shape = {
        "full": (components, features, features),
        "tied": (features, features),
        "diag": (components, features),
        "spherical": (components,),
    }[str(estimator.covariance_type)]
    if precision.shape != expected_precision_shape:
        raise ConversionError("GaussianMixture precisions_ has an inconsistent shape.")
    weights = snapshot.array("native_mixing_weights", weights_source, shape=(components,))
    means = snapshot.array("means", means_source, shape=(components, features))
    native_covariance = snapshot.array(
        "native_covariance", covariance_full, shape=(components, features, features)
    )
    native_precision = snapshot.array(
        "native_precision", precision_full, shape=(components, features, features)
    )
    native_log_det = snapshot.array(
        "native_log_determinant", log_determinant, shape=(components,)
    )
    model = GaussianMixtureModel(
        weights,
        means,
        native_covariance,
        native_precision,
        native_log_det,
        covariance_type=native_type,
    )
    return _Converted(
        model,
        {
            "family": "gaussian_mixture",
            "components": components,
            "covariance_type": estimator.covariance_type,
        },
        names,
        semantic_notes=(
            "precision-Cholesky Gaussian log-density convention expanded to full native geometry",
        ),
    )


def _linear_kernel(left: Array, right: Array, /) -> Array:
    return jnp.dot(left, right)


def _polynomial_kernel(
    left: Array,
    right: Array,
    /,
    *,
    gamma: float,
    coef0: float,
    degree: int,
) -> Array:
    return (gamma * jnp.dot(left, right) + coef0) ** degree


def _sigmoid_kernel(
    left: Array,
    right: Array,
    /,
    *,
    gamma: float,
    coef0: float,
) -> Array:
    return jnp.tanh(gamma * jnp.dot(left, right) + coef0)


def _cosine_kernel(left: Array, right: Array, /) -> Array:
    denominator = jnp.linalg.norm(left) * jnp.linalg.norm(right)
    return jnp.where(denominator > 0.0, jnp.dot(left, right) / denominator, 0.0)


def _kernel(
    name: Any,
    /,
    *,
    features: int,
    gamma: Any,
    degree: Any,
    coef0: Any,
    allow_cosine: bool,
) -> tuple[Any, dict[str, object]]:
    if type(name) is not str:
        raise UnsupportedConversionError(
            "Callable kernels cannot be snapshotted into native kernel state."
        )
    if name == "linear":
        return _linear_kernel, {"kernel": "linear"}
    resolved_gamma = (
        1.0 / features
        if gamma is None
        else _numeric_scalar(gamma, "gamma", positive=True)
    )
    if name == "rbf":
        length_scale = 1.0 / np.sqrt(2.0 * resolved_gamma)
        return SquaredExponentialKernel(length_scale=length_scale), {
            "kernel": "rbf",
            "gamma": resolved_gamma,
        }
    resolved_coef0 = _numeric_scalar(coef0, "coef0")
    if name in {"poly", "polynomial"}:
        resolved_degree = int(degree)
        if (
            isinstance(degree, (bool, np.bool_))
            or resolved_degree != degree
            or resolved_degree < 0
        ):
            raise ConversionError(
                "Polynomial kernel degree must be a nonnegative integer."
            )

        def polynomial(left: Array, right: Array, /) -> Array:
            return _polynomial_kernel(
                left,
                right,
                gamma=resolved_gamma,
                coef0=resolved_coef0,
                degree=resolved_degree,
            )

        return polynomial, {
            "kernel": "polynomial",
            "gamma": resolved_gamma,
            "coef0": resolved_coef0,
            "degree": resolved_degree,
        }
    if name == "sigmoid":

        def sigmoid(left: Array, right: Array, /) -> Array:
            return _sigmoid_kernel(
                left, right, gamma=resolved_gamma, coef0=resolved_coef0
            )

        return sigmoid, {
            "kernel": "sigmoid",
            "gamma": resolved_gamma,
            "coef0": resolved_coef0,
        }
    if name == "cosine" and allow_cosine:
        return _cosine_kernel, {"kernel": "cosine"}
    raise UnsupportedConversionError(f"Kernel {name!r} is not exactly supported.")


def _convert_kernel_ridge(
    estimator: Any, snapshot: _Snapshot, api: _SklearnAPI
) -> _Converted:
    _require_fitted(api, estimator, ("n_features_in_", "X_fit_", "dual_coef_"))
    features = _feature_count(estimator)
    _, names = _feature_schema(estimator, features)
    support_source = np.asarray(estimator.X_fit_)
    if (
        support_source.ndim != 2
        or support_source.shape[1] != features
        or support_source.shape[0] <= 0
    ):
        raise UnsupportedConversionError(
            "KernelRidge requires dense two-dimensional X_fit_."
        )
    samples = int(support_source.shape[0])
    dual_source = np.asarray(estimator.dual_coef_)
    if dual_source.ndim == 1:
        if dual_source.shape != (samples,):
            raise ConversionError("KernelRidge dual_coef_ does not match X_fit_.")
        coefficients_source = dual_source[:, None]
        output_shape: tuple[int, ...] = ()
    elif (
        dual_source.ndim == 2
        and dual_source.shape[0] == samples
        and dual_source.shape[1] > 0
    ):
        coefficients_source = dual_source
        output_shape = (int(dual_source.shape[1]),)
    else:
        raise ConversionError("KernelRidge dual_coef_ has an invalid shape.")
    support = snapshot.array("support", support_source, shape=(samples, features))
    coefficients = snapshot.array(
        "dual_coefficients", coefficients_source, shape=coefficients_source.shape
    )
    kernel, kernel_config = _kernel(
        estimator.kernel,
        features=features,
        gamma=estimator.gamma,
        degree=estimator.degree,
        coef0=estimator.coef0,
        allow_cosine=True,
    )
    model = KernelRidgeModel(
        support=support,
        coefficients=coefficients,
        intercept=jnp.zeros((coefficients_source.shape[1],), dtype=coefficients.dtype),
        support_mask=jnp.ones((samples,), dtype=bool),
        kernel=kernel,
        feature_count=features,
        output_shape=output_shape,
        case_shape=(),
        method="sklearn-kernel-ridge",
    )
    configuration = {"family": "kernel_ridge"}
    configuration.update(kernel_config)
    return _Converted(
        model,
        configuration,
        names,
        semantic_notes=(
            "copied training support and dual expansion; precomputed/callable kernels rejected",
        ),
    )


def _svm_state(
    estimator: Any,
    snapshot: _Snapshot,
    api: _SklearnAPI,
    /,
) -> tuple[Array, Array, Array, int, tuple[str, ...], dict[str, object]]:
    _require_fitted(
        api,
        estimator,
        (
            "n_features_in_",
            "support_",
            "support_vectors_",
            "dual_coef_",
            "intercept_",
            "n_support_",
            "_gamma",
        ),
    )
    features = _feature_count(estimator)
    _, names = _feature_schema(estimator, features)
    support_source = np.asarray(estimator.support_vectors_)
    if (
        support_source.ndim != 2
        or support_source.shape[1] != features
        or support_source.shape[0] <= 0
    ):
        raise UnsupportedConversionError(
            "SVM conversion requires dense support_vectors_."
        )
    supports = int(support_source.shape[0])
    support_indices = snapshot.audit(
        "support_indices", estimator.support_, shape=(supports,), integer=True
    )
    if np.any(support_indices < 0) or len(set(support_indices.tolist())) != supports:
        raise ConversionError("SVM support_ must contain unique nonnegative indices.")
    dual_source = np.asarray(estimator.dual_coef_)
    intercept_source = np.asarray(estimator.intercept_)
    if dual_source.shape != (1, supports) or intercept_source.shape != (1,):
        raise ConversionError(
            "Binary/regression SVM dual_coef_ or intercept_ shape is inconsistent."
        )
    support = snapshot.array(
        "support_vectors", support_source, shape=(supports, features)
    )
    coefficients = snapshot.array("dual_coefficients", dual_source.T, shape=(supports, 1))
    intercept = snapshot.array("intercept", intercept_source, shape=(1,))
    resolved_gamma = _numeric_scalar(estimator._gamma, "_gamma", positive=True)
    kernel, kernel_config = _kernel(
        estimator.kernel,
        features=features,
        gamma=resolved_gamma,
        degree=estimator.degree,
        coef0=estimator.coef0,
        allow_cosine=False,
    )
    kernel_config["resolved_gamma"] = resolved_gamma
    kernel_config["kernel_object"] = kernel
    return support, coefficients, intercept, features, names, kernel_config


def _convert_svc(estimator: Any, snapshot: _Snapshot, api: _SklearnAPI) -> _Converted:
    _require_fitted(api, estimator, ("classes_",))
    labels, python_labels = _labels(snapshot, estimator.classes_)
    if len(python_labels) != 2:
        raise UnsupportedConversionError(
            "Multiclass libsvm OVO coefficient reconstruction is deliberately rejected; "
            "only binary SVC/NuSVC is supported."
        )
    if bool(estimator.probability):
        raise UnsupportedConversionError(
            "SVC probability calibration requires probA_/probB_ semantics not carried by the native decision model."
        )
    support, coefficients, intercept, features, names, kernel_config = _svm_state(
        estimator, snapshot, api
    )
    n_support = snapshot.audit(
        "n_support", estimator.n_support_, shape=(2,), integer=True
    )
    if np.any(n_support < 0) or int(np.sum(n_support)) != support.shape[0]:
        raise ConversionError("SVC n_support_ is inconsistent with support_vectors_.")
    kernel = kernel_config.pop("kernel_object")
    binary = SupportVectorClassifierModel(
        support=support,
        coefficients=coefficients,
        intercept=intercept,
        support_mask=jnp.ones((support.shape[0],), dtype=bool),
        kernel=kernel,
        feature_count=features,
        output_shape=(),
        case_shape=(),
        method="sklearn-binary-svc",
    )
    target_schema = TargetSchema("binary", class_labels=python_labels)
    model = OneVsOneModel((binary,), ((0, 1),), labels, target_schema)
    configuration = {
        "family": "binary_svc",
        "probability": False,
        "public_binary_sign_convention": "classes_[1]-positive",
    }
    configuration.update(kernel_config)
    return _Converted(
        model,
        configuration,
        names,
        python_labels,
        (
            "public sign-flipped binary dual/intercept arrays and source label order preserved",
        ),
    )


def _convert_svr(estimator: Any, snapshot: _Snapshot, api: _SklearnAPI) -> _Converted:
    support, coefficients, intercept, features, names, kernel_config = _svm_state(
        estimator, snapshot, api
    )
    n_support = snapshot.audit("n_support", estimator.n_support_, ndim=1, integer=True)
    if np.any(n_support < 0) or int(np.sum(n_support)) != support.shape[0]:
        raise ConversionError("SVR n_support_ is inconsistent with support_vectors_.")
    kernel = kernel_config.pop("kernel_object")
    model = SupportVectorRegressorModel(
        support=support,
        coefficients=coefficients,
        intercept=intercept,
        support_mask=jnp.ones((support.shape[0],), dtype=bool),
        kernel=kernel,
        feature_count=features,
        output_shape=(),
        case_shape=(),
        method="sklearn-svr",
    )
    configuration = {"family": "svr"}
    configuration.update(kernel_config)
    return _Converted(
        model,
        configuration,
        names,
        semantic_notes=(
            "copied dense support expansion with resolved kernel parameters",
        ),
    )


@dataclass(frozen=True)
class _SourceTree:
    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    default_left: np.ndarray
    leaf: np.ndarray
    value: np.ndarray


def _source_tree(estimator: Any, features: int, api: _SklearnAPI) -> _SourceTree:
    _require_fitted(api, estimator, ("tree_",))
    tree = estimator.tree_
    nodes = int(tree.node_count)
    if nodes <= 0:
        raise ConversionError("A fitted tree must contain at least one node.")
    feature = np.asarray(tree.feature)
    threshold = np.asarray(tree.threshold)
    left = np.asarray(tree.children_left)
    right = np.asarray(tree.children_right)
    default_left = np.asarray(tree.missing_go_to_left)
    value = np.asarray(tree.value)
    for name, array in (
        ("feature", feature),
        ("threshold", threshold),
        ("children_left", left),
        ("children_right", right),
        ("missing_go_to_left", default_left),
    ):
        if array.shape != (nodes,):
            raise ConversionError(f"tree_.{name} must have shape ({nodes},).")
    if (
        feature.dtype.kind not in "iu"
        or left.dtype.kind not in "iu"
        or right.dtype.kind not in "iu"
    ):
        raise ConversionError("Tree feature and child arrays must be integer arrays.")
    if threshold.dtype.kind != "f" or default_left.dtype.kind not in "biu":
        raise ConversionError("Tree thresholds or missing routing have invalid dtypes.")
    leaf = (left == -1) & (right == -1)
    split = ~leaf
    if np.any((left == -1) != (right == -1)):
        raise ConversionError("Tree leaves must use -1 for both children.")
    if np.any(feature[split] < 0) or np.any(feature[split] >= features):
        raise ConversionError("Tree split feature index is out of range.")
    if np.any(np.isnan(threshold[split])):
        raise UnsupportedConversionError("NaN tree split thresholds are unsupported.")
    if (
        np.any(left[split] < 0)
        or np.any(left[split] >= nodes)
        or np.any(right[split] < 0)
        or np.any(right[split] >= nodes)
    ):
        raise ConversionError("Tree child index is out of range.")
    if api.version_minor >= 10:
        categories = np.asarray(tree._n_categories)
        if categories.shape != (features,) or np.any(categories >= 0):
            raise UnsupportedConversionError(
                "scikit-learn categorical tree bitsets are version-dependent and are not converted."
            )
    seen: set[int] = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in seen:
            raise ConversionError("Tree child graph is cyclic or has multiple parents.")
        seen.add(node)
        if not leaf[node]:
            stack.append(int(right[node]))
            stack.append(int(left[node]))
    if len(seen) != nodes:
        raise ConversionError("Tree contains nodes unreachable from root zero.")
    if value.ndim != 3 or value.shape[0] != nodes or value.dtype.kind != "f":
        raise ConversionError(
            "tree_.value must have shape (node, output, class) and floating dtype."
        )
    if np.any(~np.isfinite(value[leaf])):
        raise ConversionError("Tree leaf values must be finite.")
    return _SourceTree(
        feature, threshold, left, right, default_left.astype(bool), leaf, value
    )


def _assemble_tree_ensemble(
    snapshot: _Snapshot,
    api: _SklearnAPI,
    estimators: Sequence[Any],
    features: int,
    output_count: int,
    value_builder: Callable[[_SourceTree, int], np.ndarray],
    /,
    *,
    tree_weights: np.ndarray,
    base_score: np.ndarray,
    feature_schema: FeatureSchema,
    target_schema: TargetSchema,
    objective_transform: str,
    aggregation: str = "sum",
) -> TreeEnsemble:
    if not estimators:
        raise ConversionError("Tree ensemble must contain at least one fitted estimator.")
    source_trees = tuple(
        _source_tree(estimator, features, api) for estimator in estimators
    )
    capacity = max(tree.feature.shape[0] for tree in source_trees)
    count = len(source_trees)
    feature = np.full((count, capacity), -2, dtype=np.int64)
    threshold = np.zeros((count, capacity), dtype=np.float64)
    left = np.full((count, capacity), -1, dtype=np.int64)
    right = np.full((count, capacity), -1, dtype=np.int64)
    default_left = np.zeros((count, capacity), dtype=bool)
    leaf_value = np.zeros((count, capacity, output_count), dtype=np.float64)
    node_mask = np.zeros((count, capacity), dtype=bool)
    leaf_mask = np.zeros((count, capacity), dtype=bool)
    for tree_index, tree in enumerate(source_trees):
        nodes = int(tree.feature.shape[0])
        values = np.asarray(value_builder(tree, tree_index))
        if (
            values.shape != (nodes, output_count)
            or values.dtype.kind != "f"
            or np.any(~np.isfinite(values[tree.leaf]))
        ):
            raise ConversionError(
                "Converted tree leaf values have an invalid shape or value."
            )
        feature[tree_index, :nodes] = tree.feature
        threshold[tree_index, :nodes] = tree.threshold
        left[tree_index, :nodes] = tree.left
        right[tree_index, :nodes] = tree.right
        default_left[tree_index, :nodes] = tree.default_left
        leaf_value[tree_index, :nodes] = values
        node_mask[tree_index, :nodes] = True
        leaf_mask[tree_index, :nodes] = tree.leaf
    if tree_weights.shape != (count,) or np.any(~np.isfinite(tree_weights)):
        raise ConversionError("Tree ensemble weights have an invalid shape or value.")
    if base_score.shape != (output_count,) or np.any(~np.isfinite(base_score)):
        raise ConversionError("Tree ensemble base score has an invalid shape or value.")
    return TreeEnsemble(
        feature_index=snapshot.array("tree_feature_index", feature, integer=True),
        threshold=snapshot.array("tree_threshold", threshold, finite=False),
        left_child=snapshot.array("tree_left_child", left, integer=True),
        right_child=snapshot.array("tree_right_child", right, integer=True),
        default_left=snapshot.array("tree_default_left", default_left, boolean=True),
        leaf_value=snapshot.array("tree_leaf_value", leaf_value),
        node_mask=snapshot.array("tree_node_mask", node_mask, boolean=True),
        leaf_mask=snapshot.array("tree_leaf_mask", leaf_mask, boolean=True),
        tree_mask=snapshot.array(
            "tree_mask", np.ones((count,), dtype=bool), boolean=True
        ),
        tree_weight=snapshot.array("tree_weight", tree_weights),
        base_score=snapshot.array("tree_base_score", base_score),
        feature_schema=feature_schema,
        target_schema=target_schema,
        objective_transform=objective_transform,
        aggregation=aggregation,
        input_dtype="float32",
        max_steps=capacity,
    )


def _regression_tree_values(tree: _SourceTree, outputs: int) -> np.ndarray:
    if tree.value.shape[1:] != (outputs, 1):
        raise ConversionError("Regression tree value shape does not match n_outputs_.")
    return tree.value[:, :, 0]


def _classification_tree_values(tree: _SourceTree, classes: int) -> np.ndarray:
    if tree.value.shape[1:] != (1, classes):
        raise ConversionError("Classification tree value shape does not match classes_.")
    counts = tree.value[:, 0, :]
    if np.any(counts[tree.leaf] < 0.0):
        raise ConversionError("Classification tree leaf weights must be nonnegative.")
    totals = np.sum(counts, axis=-1, keepdims=True)
    if np.any(totals[tree.leaf] <= 0.0):
        raise ConversionError("Classification tree leaves must have positive class mass.")
    return np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0.0)


def _tree_feature_guard(estimator: Any) -> None:
    categorical = estimator.__dict__.get("categorical_features")
    if categorical is not None:
        raise UnsupportedConversionError(
            "Categorical sklearn tree preprocessing/bitsets are unsupported."
        )


def _convert_tree_regressor(
    estimator: Any, snapshot: _Snapshot, api: _SklearnAPI
) -> _Converted:
    _require_fitted(api, estimator, ("n_features_in_", "n_outputs_", "tree_"))
    if api.version_minor < 4:
        raise UnsupportedConversionError(
            "Trees predating explicit missing_go_to_left state are unsupported."
        )
    _tree_feature_guard(estimator)
    features = _feature_count(estimator)
    schema, names = _feature_schema(estimator, features)
    outputs = int(estimator.n_outputs_)
    if outputs <= 0:
        raise ConversionError("DecisionTreeRegressor n_outputs_ must be positive.")
    target = TargetSchema("continuous")
    model = _assemble_tree_ensemble(
        snapshot,
        api,
        (estimator,),
        features,
        outputs,
        lambda tree, _: _regression_tree_values(tree, outputs),
        tree_weights=np.ones((1,), dtype=float),
        base_score=np.zeros((outputs,), dtype=float),
        feature_schema=schema,
        target_schema=target,
        objective_transform="identity",
    )
    return _Converted(
        model,
        {
            "family": "decision_tree_regressor",
            "outputs": outputs,
            "missing_routing": "copied",
        },
        names,
        semantic_notes=(
            "<= threshold routing and missing_go_to_left copied for every node",
        ),
    )


def _convert_tree_classifier(
    estimator: Any, snapshot: _Snapshot, api: _SklearnAPI
) -> _Converted:
    _require_fitted(
        api,
        estimator,
        ("n_features_in_", "n_outputs_", "classes_", "n_classes_", "tree_"),
    )
    if api.version_minor < 4:
        raise UnsupportedConversionError(
            "Trees predating explicit missing_go_to_left state are unsupported."
        )
    _tree_feature_guard(estimator)
    if int(estimator.n_outputs_) != 1:
        raise UnsupportedConversionError(
            "Multi-output sklearn tree classification is unsupported."
        )
    features = _feature_count(estimator)
    schema, names = _feature_schema(estimator, features)
    _, python_labels = _labels(snapshot, estimator.classes_)
    classes = len(python_labels)
    if int(estimator.n_classes_) != classes:
        raise ConversionError(
            "DecisionTreeClassifier n_classes_ is inconsistent with classes_."
        )
    target = TargetSchema(
        "binary" if classes == 2 else "multiclass", class_labels=python_labels
    )
    model = _assemble_tree_ensemble(
        snapshot,
        api,
        (estimator,),
        features,
        classes,
        lambda tree, _: _classification_tree_values(tree, classes),
        tree_weights=np.ones((1,), dtype=float),
        base_score=np.zeros((classes,), dtype=float),
        feature_schema=schema,
        target_schema=target,
        objective_transform="identity",
    )
    return _Converted(
        model,
        {
            "family": "decision_tree_classifier",
            "classes": classes,
            "missing_routing": "copied",
        },
        names,
        python_labels,
        ("leaf class masses normalized in source class order",),
    )


def _forest_estimators(
    estimator: Any, api: _SklearnAPI, classifier: bool
) -> tuple[Any, ...]:
    _require_fitted(api, estimator, ("estimators_",))
    values = tuple(estimator.estimators_)
    if not values:
        raise ConversionError("A fitted forest must contain at least one tree.")
    from sklearn.tree import (
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        ExtraTreeClassifier,
        ExtraTreeRegressor,
    )

    allowed = (
        (DecisionTreeClassifier, ExtraTreeClassifier)
        if classifier
        else (DecisionTreeRegressor, ExtraTreeRegressor)
    )
    if any(type(value) not in allowed for value in values):
        raise UnsupportedConversionError(
            "Forest contains a nonstandard base estimator type."
        )
    return values


def _convert_forest_regressor(
    estimator: Any, snapshot: _Snapshot, api: _SklearnAPI
) -> _Converted:
    _require_fitted(api, estimator, ("n_features_in_", "n_outputs_", "estimators_"))
    if api.version_minor < 4:
        raise UnsupportedConversionError(
            "Forests predating explicit missing routing are unsupported."
        )
    _tree_feature_guard(estimator)
    features = _feature_count(estimator)
    schema, names = _feature_schema(estimator, features)
    outputs = int(estimator.n_outputs_)
    trees = _forest_estimators(estimator, api, False)
    model = _assemble_tree_ensemble(
        snapshot,
        api,
        trees,
        features,
        outputs,
        lambda tree, _: _regression_tree_values(tree, outputs),
        tree_weights=np.full((len(trees),), 1.0 / len(trees)),
        base_score=np.zeros((outputs,), dtype=float),
        feature_schema=schema,
        target_schema=TargetSchema("continuous"),
        objective_transform="identity",
    )
    return _Converted(
        model,
        {
            "family": "forest_regressor",
            "trees": len(trees),
            "outputs": outputs,
            "aggregation": "mean",
        },
        names,
        semantic_notes=("per-tree predictions aggregate by the exact arithmetic mean",),
    )


def _convert_forest_classifier(
    estimator: Any, snapshot: _Snapshot, api: _SklearnAPI
) -> _Converted:
    _require_fitted(
        api,
        estimator,
        ("n_features_in_", "n_outputs_", "classes_", "n_classes_", "estimators_"),
    )
    if api.version_minor < 4:
        raise UnsupportedConversionError(
            "Forests predating explicit missing routing are unsupported."
        )
    _tree_feature_guard(estimator)
    if int(estimator.n_outputs_) != 1:
        raise UnsupportedConversionError(
            "Multi-output sklearn forest classification is unsupported."
        )
    features = _feature_count(estimator)
    schema, names = _feature_schema(estimator, features)
    _, python_labels = _labels(snapshot, estimator.classes_)
    classes = len(python_labels)
    if int(estimator.n_classes_) != classes:
        raise ConversionError("Forest n_classes_ is inconsistent with classes_.")
    trees = _forest_estimators(estimator, api, True)
    target = TargetSchema(
        "binary" if classes == 2 else "multiclass", class_labels=python_labels
    )
    model = _assemble_tree_ensemble(
        snapshot,
        api,
        trees,
        features,
        classes,
        lambda tree, _: _classification_tree_values(tree, classes),
        tree_weights=np.full((len(trees),), 1.0 / len(trees)),
        base_score=np.zeros((classes,), dtype=float),
        feature_schema=schema,
        target_schema=target,
        objective_transform="identity",
    )
    return _Converted(
        model,
        {
            "family": "forest_classifier",
            "trees": len(trees),
            "classes": classes,
            "aggregation": "mean_probability",
        },
        names,
        python_labels,
        ("per-tree normalized class probabilities average in source class order",),
    )


def _adaboost_estimators(
    estimator: Any, api: _SklearnAPI, classifier: bool
) -> tuple[Any, ...]:
    _require_fitted(
        api, estimator, ("estimators_", "estimator_weights_", "estimator_errors_")
    )
    values = tuple(estimator.estimators_)
    if not values:
        raise ConversionError(
            "A fitted AdaBoost ensemble must contain at least one estimator."
        )
    from sklearn.tree import (
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        ExtraTreeClassifier,
        ExtraTreeRegressor,
    )

    allowed = (
        (DecisionTreeClassifier, ExtraTreeClassifier)
        if classifier
        else (DecisionTreeRegressor, ExtraTreeRegressor)
    )
    if any(type(value) not in allowed for value in values):
        raise UnsupportedConversionError(
            "AdaBoost base estimators must be exact fitted sklearn hard trees."
        )
    return values


def _convert_adaboost_regressor(
    estimator: Any, snapshot: _Snapshot, api: _SklearnAPI
) -> _Converted:
    _require_fitted(
        api, estimator, ("n_features_in_", "estimators_", "estimator_weights_")
    )
    if api.version_minor < 4:
        raise UnsupportedConversionError(
            "AdaBoost trees predating explicit missing routing are unsupported."
        )
    features = _feature_count(estimator)
    schema, names = _feature_schema(estimator, features)
    trees = _adaboost_estimators(estimator, api, False)
    weights_source = np.asarray(estimator.estimator_weights_)
    if weights_source.ndim != 1 or weights_source.shape[0] < len(trees):
        raise ConversionError(
            "AdaBoostRegressor estimator_weights_ is inconsistent with estimators_."
        )
    weights = weights_source[: len(trees)]
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0) or np.sum(weights) <= 0.0:
        raise ConversionError("AdaBoostRegressor active estimator weights are invalid.")
    model = _assemble_tree_ensemble(
        snapshot,
        api,
        trees,
        features,
        1,
        lambda tree, _: _regression_tree_values(tree, 1),
        tree_weights=weights,
        base_score=np.zeros((1,), dtype=float),
        feature_schema=schema,
        target_schema=TargetSchema("continuous"),
        objective_transform="identity",
        aggregation="weighted_median",
    )
    return _Converted(
        model,
        {
            "family": "adaboost_regressor",
            "trees": len(trees),
            "aggregation": "weighted_median",
            "loss": estimator.loss,
        },
        names,
        semantic_notes=("AdaBoost.R2 weighted-median prediction rule preserved exactly",),
    )


def _convert_adaboost_classifier(
    estimator: Any, snapshot: _Snapshot, api: _SklearnAPI
) -> _Converted:
    _require_fitted(
        api,
        estimator,
        ("n_features_in_", "classes_", "n_classes_", "estimators_", "estimator_weights_"),
    )
    if api.version_minor < 4:
        raise UnsupportedConversionError(
            "AdaBoost trees predating explicit missing routing are unsupported."
        )
    features = _feature_count(estimator)
    schema, names = _feature_schema(estimator, features)
    _, python_labels = _labels(snapshot, estimator.classes_)
    classes = len(python_labels)
    if int(estimator.n_classes_) != classes:
        raise ConversionError(
            "AdaBoostClassifier n_classes_ is inconsistent with classes_."
        )
    trees = _adaboost_estimators(estimator, api, True)
    for tree_estimator in trees:
        nested_labels = np.asarray(tree_estimator.classes_)
        if nested_labels.shape != (classes,) or not np.array_equal(
            nested_labels, np.asarray(estimator.classes_)
        ):
            raise ConversionError(
                "AdaBoost base-tree class order differs from the ensemble class order."
            )
    weights_source = np.asarray(estimator.estimator_weights_)
    if weights_source.ndim != 1 or weights_source.shape[0] < len(trees):
        raise ConversionError(
            "AdaBoostClassifier estimator_weights_ is inconsistent with estimators_."
        )
    weights = weights_source[: len(trees)]
    total = float(np.sum(weights))
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0) or total <= 0.0:
        raise ConversionError("AdaBoostClassifier active estimator weights are invalid.")

    def values(tree: _SourceTree, _: int) -> np.ndarray:
        probabilities = _classification_tree_values(tree, classes)
        predictions = np.argmax(probabilities, axis=-1)
        one_hot = np.eye(classes, dtype=float)[predictions]
        return (one_hot - (1.0 - one_hot) / (classes - 1.0)) / (classes - 1.0)

    target = TargetSchema(
        "binary" if classes == 2 else "multiclass", class_labels=python_labels
    )
    model = _assemble_tree_ensemble(
        snapshot,
        api,
        trees,
        features,
        classes,
        values,
        tree_weights=weights / total,
        base_score=np.zeros((classes,), dtype=float),
        feature_schema=schema,
        target_schema=target,
        objective_transform="softmax",
    )
    return _Converted(
        model,
        {
            "family": "adaboost_classifier",
            "trees": len(trees),
            "classes": classes,
            "algorithm": "SAMME",
        },
        names,
        python_labels,
        ("SAMME weighted votes and probability softmax scaling preserved",),
    )


def _gradient_estimators(
    estimator: Any, api: _SklearnAPI, columns: int
) -> tuple[Any, ...]:
    _require_fitted(api, estimator, ("estimators_",))
    matrix = np.asarray(estimator.estimators_, dtype=object)
    if matrix.ndim != 2 or matrix.shape[0] <= 0 or matrix.shape[1] != columns:
        raise ConversionError(
            "GradientBoosting estimators_ has an inconsistent stage/class shape."
        )
    values = tuple(matrix.reshape(-1).tolist())
    from sklearn.tree import DecisionTreeRegressor

    if any(type(value) is not DecisionTreeRegressor for value in values):
        raise UnsupportedConversionError(
            "GradientBoosting stages must be exact sklearn DecisionTreeRegressor instances."
        )
    return values


def _gradient_regression_base(
    estimator: Any, snapshot: _Snapshot, api: _SklearnAPI
) -> np.ndarray:
    init = estimator.init_
    if type(init) is str:
        if init != "zero":
            raise UnsupportedConversionError(
                "Unknown GradientBoosting init string is unsupported."
            )
        return np.zeros((1,), dtype=float)
    from sklearn.dummy import DummyRegressor

    if type(init) is not DummyRegressor:
        raise UnsupportedConversionError(
            "Custom GradientBoosting init estimators are unsupported."
        )
    _require_fitted(api, init, ("constant_",))
    constant = snapshot.audit("gradient_init_constant", init.constant_)
    if constant.size != 1:
        raise ConversionError("GradientBoostingRegressor init constant must be scalar.")
    return constant.reshape(1).astype(float, copy=False)


def _gradient_classification_base(
    estimator: Any,
    snapshot: _Snapshot,
    api: _SklearnAPI,
    classes: int,
) -> np.ndarray:
    init = estimator.init_
    if type(init) is str:
        if init != "zero":
            raise UnsupportedConversionError(
                "Unknown GradientBoosting init string is unsupported."
            )
        return np.zeros((1 if classes == 2 else classes,), dtype=float)
    from sklearn.dummy import DummyClassifier

    if type(init) is not DummyClassifier or init.strategy != "prior":
        raise UnsupportedConversionError(
            "Only the fitted default DummyClassifier(strategy='prior') GradientBoosting initializer is supported."
        )
    _require_fitted(api, init, ("classes_", "class_prior_"))
    if not np.array_equal(np.asarray(init.classes_), np.asarray(estimator.classes_)):
        raise ConversionError("GradientBoosting init class order differs from classes_.")
    prior = snapshot.audit(
        "gradient_init_class_prior", init.class_prior_, shape=(classes,)
    )
    if np.any(prior < 0.0) or not np.isclose(np.sum(prior), 1.0, rtol=1e-12, atol=1e-15):
        raise ConversionError("GradientBoosting init class_prior_ is invalid.")
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(prior.astype(np.float64), epsilon, 1.0 - epsilon)
    if classes == 2:
        return np.asarray([np.log(clipped[1] / (1.0 - clipped[1]))])
    log_prior = np.log(clipped)
    return log_prior - np.mean(log_prior)


def _convert_gradient_boosting_regressor(
    estimator: Any, snapshot: _Snapshot, api: _SklearnAPI
) -> _Converted:
    _require_fitted(api, estimator, ("n_features_in_", "estimators_", "init_"))
    if estimator.loss != "squared_error":
        raise UnsupportedConversionError(
            "Only squared-error GradientBoostingRegressor is exactly supported."
        )
    if api.version_minor < 4:
        raise UnsupportedConversionError(
            "Gradient trees predating explicit missing routing are unsupported."
        )
    features = _feature_count(estimator)
    schema, names = _feature_schema(estimator, features)
    trees = _gradient_estimators(estimator, api, 1)
    learning_rate = _numeric_scalar(
        estimator.learning_rate, "learning_rate", positive=True
    )
    base = _gradient_regression_base(estimator, snapshot, api)
    model = _assemble_tree_ensemble(
        snapshot,
        api,
        trees,
        features,
        1,
        lambda tree, _: _regression_tree_values(tree, 1),
        tree_weights=np.full((len(trees),), learning_rate),
        base_score=base,
        feature_schema=schema,
        target_schema=TargetSchema("continuous"),
        objective_transform="identity",
    )
    return _Converted(
        model,
        {
            "family": "gradient_boosting_regressor",
            "loss": "squared_error",
            "stages": len(trees),
            "learning_rate": learning_rate,
        },
        names,
        semantic_notes=(
            "constant init plus learning-rate-scaled regression-tree leaves",
        ),
    )


def _convert_gradient_boosting_classifier(
    estimator: Any, snapshot: _Snapshot, api: _SklearnAPI
) -> _Converted:
    _require_fitted(
        api,
        estimator,
        ("n_features_in_", "estimators_", "init_", "classes_", "n_classes_"),
    )
    if estimator.loss != "log_loss":
        raise UnsupportedConversionError(
            "Only log-loss GradientBoostingClassifier is exactly supported."
        )
    if api.version_minor < 4:
        raise UnsupportedConversionError(
            "Gradient trees predating explicit missing routing are unsupported."
        )
    features = _feature_count(estimator)
    schema, names = _feature_schema(estimator, features)
    _, python_labels = _labels(snapshot, estimator.classes_)
    classes = len(python_labels)
    if int(estimator.n_classes_) != classes:
        raise ConversionError(
            "GradientBoostingClassifier n_classes_ is inconsistent with classes_."
        )
    columns = 1 if classes == 2 else classes
    trees = _gradient_estimators(estimator, api, columns)
    learning_rate = _numeric_scalar(
        estimator.learning_rate, "learning_rate", positive=True
    )
    base = _gradient_classification_base(estimator, snapshot, api, classes)

    def values(tree: _SourceTree, index: int) -> np.ndarray:
        scalar = _regression_tree_values(tree, 1)[:, 0]
        if columns == 1:
            return scalar[:, None]
        output = np.zeros((scalar.shape[0], classes), dtype=scalar.dtype)
        output[:, index % columns] = scalar
        return output

    target = TargetSchema(
        "binary" if classes == 2 else "multiclass", class_labels=python_labels
    )
    model = _assemble_tree_ensemble(
        snapshot,
        api,
        trees,
        features,
        columns,
        values,
        tree_weights=np.full((len(trees),), learning_rate),
        base_score=base,
        feature_schema=schema,
        target_schema=target,
        objective_transform="sigmoid" if classes == 2 else "softmax",
    )
    return _Converted(
        model,
        {
            "family": "gradient_boosting_classifier",
            "loss": "log_loss",
            "classes": classes,
            "stages": len(trees) // columns,
            "trees_per_stage": columns,
            "learning_rate": learning_rate,
        },
        names,
        python_labels,
        (
            "constant prior init, stage/class tree layout, and sigmoid/softmax loss link preserved",
        ),
    )


def from_sklearn(estimator: object, /) -> ConversionResult:
    """Convert one explicitly supported fitted sklearn estimator into frozen native state.

    The dispatch is by exact public estimator class. Conversion is a one-time,
    fail-closed boundary: prediction on the returned model never calls sklearn and
    retains no reference to the source estimator.
    """
    api = _sklearn_api()
    source_type = type(estimator)
    converter = api.registry.get(source_type)
    if converter is None:
        qualified = f"{source_type.__module__}.{source_type.__qualname__}"
        raise UnsupportedConversionError(
            f"Exact estimator class {qualified} is not in the sklearn conversion registry."
        )
    snapshot = _Snapshot()
    try:
        converted = converter(estimator, snapshot, api)
    except (ConversionError, UnsupportedConversionError):
        raise
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise ConversionError(
            f"Malformed fitted state for {source_type.__module__}.{source_type.__qualname__}: {error}"
        ) from error
    configuration = dict(converted.configuration)
    configuration["converter_schema"] = _CONVERTER_SCHEMA
    configuration["semantic_notes"] = converted.semantic_notes
    snapshot.configuration(configuration)
    configuration["sha256"] = snapshot.hexdigest()
    provenance = ConversionProvenance(
        source="scikit-learn",
        source_version=api.version,
        source_model=f"{source_type.__module__}.{source_type.__qualname__}",
        configuration=configuration,
        feature_names=converted.feature_names,
        class_labels=converted.class_labels,
        license_id=_LICENSE,
    )
    return ConversionResult(converted.model, provenance)


__all__ = ["from_sklearn"]
