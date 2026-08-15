#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import ast
import builtins
import copy
import importlib
from types import SimpleNamespace
from typing import Any, cast, Protocol

import jax
import numpy as np
import pytest

from phydrax.ml.interop import (
    ConversionError,
    from_sklearn,
    UnsupportedConversionError,
)


class _InvertibleModel(Protocol):
    def inverse_transform(self, values: Any, /) -> Any: ...


class _ArrayPredictor(Protocol):
    def predict(self, values: Any, /) -> Any: ...


class _OffsetModel(Protocol):
    offsets: tuple[int, ...]


class _CenterModel(Protocol):
    centers: jax.Array


class _MixtureModel(Protocol):
    def log_prob(self, values: Any, /) -> Any: ...

    def predict(self, values: Any, /) -> Any: ...


class _SupportClassifierModel(Protocol):
    def pairwise_decision_function(self, values: Any, /) -> Any: ...

    def predict(self, values: Any, /) -> Any: ...


class _LabelPredictor(Protocol):
    def predict_labels(self, values: Any, /) -> Any: ...


class _TreeModel(Protocol):
    default_left: jax.Array


class _LabelArrayModel(Protocol):
    labels: jax.Array


def _configuration(result):
    return {
        name: ast.literal_eval(value) for name, value in result.provenance.configuration
    }


@pytest.fixture(scope="module")
def sk():
    sklearn = pytest.importorskip("sklearn")
    from sklearn import (
        cluster,
        decomposition,
        ensemble,
        impute,
        kernel_ridge,
        linear_model,
        mixture,
        preprocessing,
        svm,
        tree,
    )

    return SimpleNamespace(
        sklearn=sklearn,
        cluster=cluster,
        decomposition=decomposition,
        ensemble=ensemble,
        impute=impute,
        kernel_ridge=kernel_ridge,
        linear_model=linear_model,
        mixture=mixture,
        preprocessing=preprocessing,
        svm=svm,
        tree=tree,
    )


def _guard_sklearn_import(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sklearn" or name.startswith("sklearn."):
            raise AssertionError(
                "Importing the converter module must not import sklearn."
            )
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_converter_module_import_is_lazy(monkeypatch):
    from phydrax.ml.interop import _sklearn

    _guard_sklearn_import(monkeypatch)
    importlib.reload(_sklearn)


@pytest.mark.parametrize(
    ("class_name", "kwargs"),
    [
        ("StandardScaler", {}),
        ("MinMaxScaler", {"feature_range": (-2.0, 3.0), "clip": True}),
        ("MaxAbsScaler", {}),
        (
            "RobustScaler",
            {
                "with_centering": True,
                "with_scaling": True,
                "quantile_range": (20.0, 80.0),
                "unit_variance": True,
            },
        ),
    ],
)
def test_supported_scalers_match_transform(sk, class_name, kwargs):
    values = np.array(
        [[-4.0, 1.0, 10.0], [-1.0, 3.0, 10.0], [2.0, 8.0, 10.0], [7.0, 20.0, 10.0]]
    )
    query = np.array([[-9.0, 2.0, 10.0], [1.0, 11.0, 10.0], [12.0, 30.0, 10.0]])
    estimator = getattr(sk.preprocessing, class_name)(**kwargs).fit(values)
    result = from_sklearn(estimator)

    np.testing.assert_allclose(
        np.asarray(result.model(query)),
        estimator.transform(query),
        rtol=1e-12,
        atol=1e-12,
    )
    if not kwargs.get("clip", False):
        transformed = estimator.transform(query)
        np.testing.assert_allclose(
            np.asarray(
                cast(_InvertibleModel, result.model.model).inverse_transform(transformed)
            ),
            estimator.inverse_transform(transformed),
            rtol=1e-12,
            atol=1e-12,
        )


def test_simple_imputer_matches_numeric_nan_and_finite_sentinels(sk):
    cases = [
        (np.nan, np.array([[1.0, np.nan], [3.0, 5.0], [np.nan, 7.0]])),
        (-999.0, np.array([[1.0, -999.0], [3.0, 5.0], [-999.0, 7.0]])),
    ]
    for missing, values in cases:
        estimator = sk.impute.SimpleImputer(
            strategy="median", missing_values=missing, add_indicator=False
        ).fit(values)
        result = from_sklearn(estimator)
        np.testing.assert_allclose(
            np.asarray(result.model(values)),
            estimator.transform(values),
            rtol=0.0,
            atol=0.0,
        )


def test_dense_numeric_one_hot_encoder_matches_source_blocks(sk):
    values = np.array([[0, 10], [1, 20], [2, 10], [1, 30]], dtype=np.int64)
    estimator = sk.preprocessing.OneHotEncoder(
        handle_unknown="error", drop=None, sparse_output=False, dtype=np.float64
    ).fit(values)
    result = from_sklearn(estimator)

    np.testing.assert_array_equal(
        np.asarray(result.model(values)), estimator.transform(values)
    )
    assert cast(_OffsetModel, result.model.model).offsets == (0, 3, 6)


def test_numeric_ordinal_encoder_matches_source_category_order(sk):
    values = np.array([[3, 10], [1, 30], [2, 20], [3, 20]], dtype=np.int64)
    estimator = sk.preprocessing.OrdinalEncoder(
        handle_unknown="error", dtype=np.int32
    ).fit(values)
    result = from_sklearn(estimator)

    np.testing.assert_array_equal(
        np.asarray(result.model(values)), estimator.transform(values)
    )


@pytest.mark.parametrize(
    "class_name",
    [
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "MultiTaskLasso",
        "MultiTaskElasticNet",
    ],
)
def test_supported_linear_regressors_match_predict(sk, class_name):
    features = np.array(
        [
            [-2.0, 0.0, 1.0],
            [-1.0, 1.0, 0.0],
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 1.0],
            [2.0, 3.0, 4.0],
            [3.0, 5.0, 2.0],
        ]
    )
    scalar_target = 1.5 * features[:, 0] - 0.75 * features[:, 1] + 0.2
    multi_target = np.stack(
        (scalar_target, -0.5 * scalar_target + features[:, 2]), axis=-1
    )
    query = np.array([[-0.5, 2.0, 1.0], [2.5, 4.0, 3.0]])
    estimator_type = getattr(sk.linear_model, class_name)
    if class_name.startswith("MultiTask"):
        estimator = estimator_type(alpha=0.03, max_iter=20_000).fit(
            features, multi_target
        )
    elif class_name in {"Lasso", "ElasticNet"}:
        kwargs = {"alpha": 0.03, "max_iter": 20_000}
        if class_name == "ElasticNet":
            kwargs["l1_ratio"] = 0.35
        estimator = estimator_type(**kwargs).fit(features, scalar_target)
    elif class_name == "Ridge":
        estimator = estimator_type(alpha=0.2).fit(features, multi_target)
    else:
        estimator = estimator_type().fit(features, multi_target)
    result = from_sklearn(estimator)

    np.testing.assert_allclose(
        np.asarray(result.model(query)), estimator.predict(query), rtol=2e-7, atol=2e-7
    )


@pytest.mark.parametrize("classes", [2, 3])
def test_logistic_regression_preserves_binary_and_multinomial_conventions(sk, classes):
    features = np.array(
        [
            [-3.0, -1.0],
            [-2.0, 0.0],
            [-1.0, 2.0],
            [0.0, -2.0],
            [1.0, 0.0],
            [2.0, 2.0],
            [3.0, -1.0],
            [4.0, 1.0],
            [0.5, 3.0],
        ]
    )
    target = (
        np.array([10, 10, 10, 20, 20, 20, 20, 20, 20])
        if classes == 2
        else np.array([10, 10, 30, 20, 20, 30, 20, 30, 30])
    )
    estimator = sk.linear_model.LogisticRegression(solver="lbfgs", max_iter=2_000).fit(
        features, target
    )
    result = from_sklearn(estimator)
    expected = estimator.predict_proba(features)
    actual = np.asarray(result.model(features))

    np.testing.assert_allclose(
        actual,
        expected[:, 1] if classes == 2 else expected,
        rtol=2e-7,
        atol=2e-7,
    )
    np.testing.assert_array_equal(
        np.asarray(cast(_ArrayPredictor, result.model.model).predict(features)),
        estimator.predict(features),
    )


@pytest.mark.parametrize(
    ("class_name", "kwargs"),
    [
        ("PoissonRegressor", {"alpha": 0.1}),
        ("GammaRegressor", {"alpha": 0.1}),
        ("TweedieRegressor", {"power": 1.5, "link": "log", "alpha": 0.1}),
        ("TweedieRegressor", {"power": 0.0, "link": "identity", "alpha": 0.1}),
    ],
)
def test_supported_glm_links_match_predict(sk, class_name, kwargs):
    features = np.linspace(0.0, 2.0, 12)[:, None]
    if kwargs.get("link") == "identity":
        target = 1.0 + 0.3 * features[:, 0]
    else:
        target = np.exp(0.2 + 0.35 * features[:, 0])
    estimator = getattr(sk.linear_model, class_name)(max_iter=2_000, **kwargs).fit(
        features, target
    )
    result = from_sklearn(estimator)

    np.testing.assert_allclose(
        np.asarray(result.model(features)),
        estimator.predict(features),
        rtol=2e-7,
        atol=2e-7,
    )


@pytest.mark.parametrize("class_name", ["PCA", "TruncatedSVD"])
def test_decomposition_transform_and_inverse_match(sk, class_name):
    values = np.array(
        [
            [1.0, 0.0, 2.0, 1.0],
            [2.0, 1.0, 0.0, 3.0],
            [3.0, 1.0, 1.0, 2.0],
            [4.0, 2.0, 3.0, 0.0],
            [5.0, 3.0, 2.0, 1.0],
        ]
    )
    estimator_type = getattr(sk.decomposition, class_name)
    estimator = estimator_type(n_components=2, random_state=0).fit(values)
    result = from_sklearn(estimator)
    scores = estimator.transform(values)

    np.testing.assert_allclose(
        np.asarray(result.model(values)), scores, rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(
        np.asarray(cast(_InvertibleModel, result.model.model).inverse_transform(scores)),
        estimator.inverse_transform(scores),
        rtol=1e-11,
        atol=1e-11,
    )


def test_kmeans_preserves_center_order_ties_and_predictions(sk):
    values = np.array([[-3.0, 0.0], [-2.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    estimator = sk.cluster.KMeans(
        n_clusters=2, n_init=1, random_state=3, algorithm="lloyd"
    ).fit(values)
    result = from_sklearn(estimator)
    query = np.array([[-4.0, 0.0], [0.0, 0.0], [4.0, 0.0]])

    np.testing.assert_array_equal(
        np.asarray(result.model(query)), estimator.predict(query)
    )
    np.testing.assert_allclose(
        np.asarray(cast(_CenterModel, result.model.model).centers),
        estimator.cluster_centers_,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("covariance_type", ["full", "tied", "diag", "spherical"])
def test_gaussian_mixture_geometry_probabilities_and_scores_match(sk, covariance_type):
    values = np.array(
        [
            [-3.0, -1.0],
            [-2.5, -0.5],
            [-2.0, -1.5],
            [2.0, 1.0],
            [2.5, 0.5],
            [3.0, 1.5],
        ]
    )
    estimator = sk.mixture.GaussianMixture(
        n_components=2,
        covariance_type=covariance_type,
        reg_covar=1e-5,
        random_state=0,
        n_init=2,
    ).fit(values)
    result = from_sklearn(estimator)
    native = cast(_MixtureModel, result.model.model)

    np.testing.assert_allclose(
        np.asarray(result.model(values)),
        estimator.predict_proba(values),
        rtol=2e-7,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        np.asarray(native.log_prob(values)),
        estimator.score_samples(values),
        rtol=2e-7,
        atol=2e-7,
    )
    np.testing.assert_array_equal(
        np.asarray(native.predict(values)), estimator.predict(values)
    )


@pytest.mark.parametrize(
    ("kernel", "kwargs"),
    [
        ("linear", {}),
        ("rbf", {"gamma": 0.7}),
        ("polynomial", {"gamma": 0.4, "degree": 2, "coef0": 0.3}),
        ("sigmoid", {"gamma": 0.2, "coef0": 0.1}),
        ("cosine", {}),
    ],
)
def test_kernel_ridge_fixed_kernels_match(sk, kernel, kwargs):
    features = np.array([[-2.0, 1.0], [-1.0, 0.5], [0.0, 1.0], [1.0, 2.0], [2.0, 1.5]])
    target = np.stack((features[:, 0] ** 2, features[:, 1] - features[:, 0]), axis=-1)
    estimator = sk.kernel_ridge.KernelRidge(alpha=0.2, kernel=kernel, **kwargs).fit(
        features, target
    )
    result = from_sklearn(estimator)

    np.testing.assert_allclose(
        np.asarray(result.model(features)),
        estimator.predict(features),
        rtol=2e-7,
        atol=2e-7,
    )


@pytest.mark.parametrize("class_name", ["SVC", "NuSVC"])
def test_binary_svc_variants_preserve_decision_and_labels(sk, class_name):
    features = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    target = np.array([10, 10, 10, 20, 20, 20])
    kwargs = {"kernel": "rbf", "gamma": 0.6, "probability": False}
    if class_name == "NuSVC":
        kwargs["nu"] = 0.4
    estimator = getattr(sk.svm, class_name)(**kwargs).fit(features, target)
    result = from_sklearn(estimator)
    native = cast(_SupportClassifierModel, result.model.model)

    np.testing.assert_allclose(
        np.asarray(native.pairwise_decision_function(features))[:, 0],
        estimator.decision_function(features),
        rtol=2e-7,
        atol=2e-7,
    )
    np.testing.assert_array_equal(
        np.asarray(native.predict(features)), estimator.predict(features)
    )


@pytest.mark.parametrize("class_name", ["SVR", "NuSVR"])
def test_svr_variants_preserve_dense_support_expansion(sk, class_name):
    features = np.linspace(-2.0, 2.0, 10)[:, None]
    target = np.sin(features[:, 0])
    kwargs = {"kernel": "poly", "gamma": 0.5, "degree": 3, "coef0": 0.2}
    if class_name == "NuSVR":
        kwargs["nu"] = 0.4
    else:
        kwargs["epsilon"] = 0.05
    estimator = getattr(sk.svm, class_name)(**kwargs).fit(features, target)
    result = from_sklearn(estimator)

    np.testing.assert_allclose(
        np.asarray(result.model(features)),
        estimator.predict(features),
        rtol=2e-7,
        atol=2e-7,
    )


@pytest.mark.parametrize(
    ("class_name", "classifier"),
    [
        ("DecisionTreeRegressor", False),
        ("ExtraTreeRegressor", False),
        ("DecisionTreeClassifier", True),
        ("ExtraTreeClassifier", True),
    ],
)
def test_single_hard_tree_classes_match(sk, class_name, classifier):
    features = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    target = (
        np.array([0, 0, 0, 1, 1, 1])
        if classifier
        else np.array([-2.0, -1.0, -0.5, 1.0, 2.0, 2.5])
    )
    estimator = getattr(sk.tree, class_name)(max_depth=2, random_state=0).fit(
        features, target
    )
    result = from_sklearn(estimator)
    native = cast(_LabelPredictor, result.model.model)

    if classifier:
        np.testing.assert_allclose(
            np.asarray(result.model(features)),
            estimator.predict_proba(features),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            np.asarray(native.predict_labels(features)), estimator.predict(features)
        )
    else:
        np.testing.assert_allclose(
            np.asarray(result.model(features)),
            estimator.predict(features),
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize(
    ("class_name", "classifier"),
    [
        ("RandomForestRegressor", False),
        ("ExtraTreesRegressor", False),
        ("RandomForestClassifier", True),
        ("ExtraTreesClassifier", True),
    ],
)
def test_random_and_extra_forests_match_mean_aggregation(sk, class_name, classifier):
    features = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0], [4.0]])
    target = (
        np.array([0, 0, 0, 1, 1, 1, 1])
        if classifier
        else np.array([-2.0, -1.0, -0.5, 1.0, 2.0, 2.5, 4.0])
    )
    estimator = getattr(sk.ensemble, class_name)(
        n_estimators=5, max_depth=3, random_state=0
    ).fit(features, target)
    result = from_sklearn(estimator)
    native = cast(_LabelPredictor, result.model.model)

    if classifier:
        np.testing.assert_allclose(
            np.asarray(result.model(features)),
            estimator.predict_proba(features),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            np.asarray(native.predict_labels(features)), estimator.predict(features)
        )
    else:
        np.testing.assert_allclose(
            np.asarray(result.model(features)),
            estimator.predict(features),
            rtol=1e-12,
            atol=1e-12,
        )


def test_tree_missing_routing_is_copied(sk):
    features = np.array([[-2.0], [-1.0], [1.0], [2.0], [np.nan]])
    target = np.array([-2.0, -1.0, 1.0, 2.0, 7.0])
    estimator = sk.tree.DecisionTreeRegressor(max_depth=2, random_state=0).fit(
        features, target
    )
    result = from_sklearn(estimator)
    query = np.array([[np.nan], [-1.5], [1.5]])

    np.testing.assert_allclose(
        np.asarray(result.model(query)), estimator.predict(query), rtol=0.0, atol=0.0
    )
    np.testing.assert_array_equal(
        np.asarray(
            cast(_TreeModel, result.model.model).default_left[
                0, : estimator.tree_.node_count
            ]
        ),
        estimator.tree_.missing_go_to_left.astype(bool),
    )


def test_adaboost_classifier_samme_probabilities_and_votes_match(sk):
    features = np.array([[-3.0], [-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0], [4.0]])
    target = np.array([0, 0, 1, 1, 2, 2, 2, 1])
    estimator = sk.ensemble.AdaBoostClassifier(
        n_estimators=6, learning_rate=0.7, random_state=0
    ).fit(features, target)
    result = from_sklearn(estimator)

    np.testing.assert_allclose(
        np.asarray(result.model(features)),
        estimator.predict_proba(features),
        rtol=2e-7,
        atol=2e-7,
    )
    np.testing.assert_array_equal(
        np.asarray(cast(_LabelPredictor, result.model.model).predict_labels(features)),
        estimator.predict(features),
    )


def test_adaboost_regressor_weighted_median_matches(sk):
    features = np.linspace(-3.0, 3.0, 16)[:, None]
    target = features[:, 0] ** 2 + 0.2 * features[:, 0]
    estimator = sk.ensemble.AdaBoostRegressor(
        n_estimators=8, learning_rate=0.8, random_state=0, loss="linear"
    ).fit(features, target)
    result = from_sklearn(estimator)

    np.testing.assert_allclose(
        np.asarray(result.model(features)),
        estimator.predict(features),
        rtol=2e-7,
        atol=2e-7,
    )


def test_gradient_boosting_squared_error_matches_constant_init_and_stage_sum(sk):
    features = np.linspace(-2.0, 2.0, 14)[:, None]
    target = features[:, 0] ** 2 - 0.3 * features[:, 0]
    estimator = sk.ensemble.GradientBoostingRegressor(
        loss="squared_error",
        n_estimators=7,
        learning_rate=0.15,
        max_depth=2,
        random_state=0,
    ).fit(features, target)
    result = from_sklearn(estimator)

    np.testing.assert_allclose(
        np.asarray(result.model(features)),
        estimator.predict(features),
        rtol=2e-7,
        atol=2e-7,
    )


@pytest.mark.parametrize("classes", [2, 3])
def test_gradient_boosting_log_loss_matches_binary_and_multiclass_links(sk, classes):
    features = np.array(
        [[-3.0], [-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    )
    target = (
        np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
        if classes == 2
        else np.array([0, 0, 1, 1, 2, 2, 2, 1, 0])
    )
    estimator = sk.ensemble.GradientBoostingClassifier(
        loss="log_loss", n_estimators=6, learning_rate=0.15, max_depth=2, random_state=0
    ).fit(features, target)
    result = from_sklearn(estimator)
    expected = estimator.predict_proba(features)
    actual = np.asarray(result.model(features))

    np.testing.assert_allclose(
        actual, expected[:, 1] if classes == 2 else expected, rtol=2e-7, atol=2e-7
    )
    np.testing.assert_array_equal(
        np.asarray(cast(_LabelPredictor, result.model.model).predict_labels(features)),
        estimator.predict(features),
    )


def test_exact_class_dispatch_rejects_subclasses(sk):
    class DerivedLinearRegression(sk.linear_model.LinearRegression):
        pass

    estimator = DerivedLinearRegression().fit(
        np.array([[0.0], [1.0]]), np.array([0.0, 1.0])
    )
    with pytest.raises(UnsupportedConversionError, match="Exact estimator class"):
        from_sklearn(estimator)


@pytest.mark.parametrize(
    ("module_name", "class_name", "kwargs"),
    [
        ("preprocessing", "StandardScaler", {}),
        ("impute", "SimpleImputer", {}),
        ("linear_model", "LinearRegression", {}),
        ("linear_model", "LogisticRegression", {}),
        ("decomposition", "PCA", {}),
        ("cluster", "KMeans", {"n_clusters": 2}),
        ("mixture", "GaussianMixture", {"n_components": 2}),
        ("kernel_ridge", "KernelRidge", {}),
        ("svm", "SVC", {}),
        ("tree", "DecisionTreeRegressor", {}),
        ("ensemble", "RandomForestRegressor", {}),
        ("ensemble", "AdaBoostRegressor", {}),
        ("ensemble", "GradientBoostingRegressor", {}),
    ],
)
def test_supported_unfitted_classes_fail_closed(sk, module_name, class_name, kwargs):
    module = getattr(sk, module_name)
    estimator = getattr(module, class_name)(**kwargs)
    with pytest.raises(ConversionError, match="not fitted"):
        from_sklearn(estimator)


@pytest.mark.parametrize(
    "case",
    [
        "imputer_indicator",
        "one_hot_ignore",
        "one_hot_sparse",
        "one_hot_drop",
        "one_hot_infrequent",
        "ordinal_unknown_value",
        "ordinal_string_categories",
        "logistic_multiclass_ovr",
        "logistic_string_labels",
        "tweedie_unknown_link",
        "pca_whiten",
        "kernel_callable",
        "kernel_precomputed",
        "svc_probability",
        "svc_multiclass",
        "svc_precomputed",
        "tree_multioutput_classifier",
        "adaboost_custom_base",
        "gradient_regression_loss",
        "gradient_classifier_loss",
        "hist_gradient_boosting",
    ],
)
def test_unsupported_prediction_semantics_are_rejected(sk, case):
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y_reg = np.array([-2.0, -1.0, 1.0, 2.0])
    y_binary = np.array([0, 0, 1, 1])
    if case == "imputer_indicator":
        estimator = sk.impute.SimpleImputer(add_indicator=True).fit(
            np.array([[1.0], [np.nan], [2.0]])
        )
    elif case == "one_hot_ignore":
        estimator = sk.preprocessing.OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        ).fit(x)
    elif case == "one_hot_sparse":
        estimator = sk.preprocessing.OneHotEncoder(
            handle_unknown="error", sparse_output=True
        ).fit(x)
    elif case == "one_hot_drop":
        estimator = sk.preprocessing.OneHotEncoder(
            drop="first", handle_unknown="error", sparse_output=False
        ).fit(x)
    elif case == "one_hot_infrequent":
        estimator = sk.preprocessing.OneHotEncoder(
            min_frequency=2,
            handle_unknown="infrequent_if_exist",
            sparse_output=False,
        ).fit(np.array([[0.0], [0.0], [1.0], [2.0]]))
    elif case == "ordinal_unknown_value":
        estimator = sk.preprocessing.OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int32
        ).fit(x)
    elif case == "ordinal_string_categories":
        estimator = sk.preprocessing.OrdinalEncoder(
            handle_unknown="error", dtype=np.int32
        ).fit(np.array([["low"], ["medium"], ["high"]]))
    elif case == "logistic_multiclass_ovr":
        estimator = sk.linear_model.LogisticRegression(
            solver="liblinear", max_iter=500
        ).fit(x, y_binary)
        estimator.classes_ = np.array([0, 1, 2])
        estimator.coef_ = np.broadcast_to(estimator.coef_, (3, x.shape[1])).copy()
        estimator.intercept_ = np.broadcast_to(estimator.intercept_, (3,)).copy()
    elif case == "logistic_string_labels":
        estimator = sk.linear_model.LogisticRegression(max_iter=500).fit(
            x, np.array(["negative", "negative", "positive", "positive"])
        )
    elif case == "tweedie_unknown_link":
        estimator = sk.linear_model.TweedieRegressor(
            power=1.5, link="log", max_iter=500
        ).fit(x, np.exp(y_reg))
        estimator.link = "sqrt"
    elif case == "pca_whiten":
        estimator = sk.decomposition.PCA(n_components=1, whiten=True).fit(x)
    elif case == "kernel_callable":
        estimator = sk.kernel_ridge.KernelRidge(
            kernel=lambda left, right: float(np.dot(left, right))
        ).fit(x, y_reg)
    elif case == "kernel_precomputed":
        estimator = sk.kernel_ridge.KernelRidge(kernel="precomputed").fit(x @ x.T, y_reg)
    elif case == "svc_probability":
        estimator = sk.svm.SVC(probability=True).fit(x, y_binary)
    elif case == "svc_multiclass":
        estimator = sk.svm.SVC().fit(
            np.array([[-3.0], [-2.0], [0.0], [1.0], [3.0], [4.0]]),
            np.array([0, 0, 1, 1, 2, 2]),
        )
    elif case == "svc_precomputed":
        training = np.array([[-3.0], [-2.0], [0.0], [1.0], [3.0], [4.0]])
        estimator = sk.svm.SVC(kernel="precomputed").fit(
            training @ training.T, np.array([0, 0, 1, 1, 0, 1])
        )
    elif case == "tree_multioutput_classifier":
        estimator = sk.tree.DecisionTreeClassifier().fit(
            x, np.stack((y_binary, 1 - y_binary), axis=-1)
        )
    elif case == "adaboost_custom_base":
        estimator = sk.ensemble.AdaBoostClassifier(
            estimator=sk.linear_model.LogisticRegression(), n_estimators=2, random_state=0
        ).fit(x, y_binary)
    elif case == "gradient_regression_loss":
        estimator = sk.ensemble.GradientBoostingRegressor(
            loss="absolute_error", n_estimators=2, random_state=0
        ).fit(x, y_reg)
    elif case == "gradient_classifier_loss":
        estimator = sk.ensemble.GradientBoostingClassifier(
            loss="exponential", n_estimators=2, random_state=0
        ).fit(x, y_binary)
    else:
        estimator = sk.ensemble.HistGradientBoostingRegressor(max_iter=2).fit(x, y_reg)
    with pytest.raises(UnsupportedConversionError):
        from_sklearn(estimator)


@pytest.mark.parametrize(
    "mutation",
    ["nonfinite_coefficient", "wrong_coefficient_shape", "zero_scale", "center_count"],
)
def test_malformed_supported_state_raises_conversion_error(sk, mutation):
    if mutation in {"nonfinite_coefficient", "wrong_coefficient_shape"}:
        estimator = sk.linear_model.LinearRegression().fit(
            np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]]),
            np.array([0.0, 1.0, 2.0]),
        )
        if mutation == "nonfinite_coefficient":
            estimator.coef_[0] = np.nan
        else:
            estimator.coef_ = np.zeros((3,))
    elif mutation == "zero_scale":
        estimator = sk.preprocessing.StandardScaler().fit(
            np.array([[0.0, 1.0], [1.0, 2.0]])
        )
        estimator.scale_[0] = 0.0
    else:
        estimator = sk.cluster.KMeans(n_clusters=2, n_init=1, random_state=0).fit(
            np.array([[-1.0], [0.0], [2.0], [3.0]])
        )
        estimator.cluster_centers_ = np.zeros((3, 1))
    with pytest.raises(ConversionError):
        from_sklearn(estimator)


def test_version_schema_rejects_unknown_and_future_versions():
    from phydrax.ml.interop import _sklearn

    with pytest.raises(UnsupportedConversionError, match="Unrecognized"):
        _sklearn._parse_version("development")
    with pytest.raises(UnsupportedConversionError, match="outside"):
        _sklearn._parse_version("2.0.0")
    with pytest.raises(UnsupportedConversionError, match="outside"):
        _sklearn._parse_version("1.11.dev0")


def test_conversion_is_immutable_source_free_and_provenance_is_complete(sk):
    features = np.array([[-2.0, 1.0], [-1.0, 0.0], [1.0, 2.0], [2.0, 3.0]])
    target = np.array([-1.0, -0.5, 2.0, 3.0])
    estimator = sk.linear_model.LinearRegression().fit(features, target)
    duplicate = copy.deepcopy(estimator)
    result = from_sklearn(estimator)
    duplicate_result = from_sklearn(duplicate)
    before = np.asarray(result.model(features))
    configuration = _configuration(result)

    estimator.coef_[:] = 1_000.0
    estimator.intercept_ = -1_000.0
    estimator.predict = lambda _: (_ for _ in ()).throw(
        AssertionError("source predict called")
    )
    after = np.asarray(result.model(features))
    changed_result = from_sklearn(estimator)

    np.testing.assert_array_equal(after, before)
    assert result.provenance.source == "scikit-learn"
    assert result.provenance.source_version == sk.sklearn.__version__
    assert result.provenance.source_model.endswith(".LinearRegression")
    assert result.provenance.license_id == "BSD-3-Clause"
    assert configuration["converter_schema"] == "phydrax.sklearn.fitted.v1"
    assert configuration["semantic_notes"]
    assert len(configuration["sha256"]) == 64
    assert configuration["sha256"] == _configuration(duplicate_result)["sha256"]
    assert configuration["sha256"] != _configuration(changed_result)["sha256"]
    leaves = jax.tree_util.tree_leaves(result.model)
    assert all(type(leaf) is not type(estimator) for leaf in leaves)
    assert any(isinstance(leaf, jax.Array) for leaf in leaves)


def test_feature_names_and_numeric_class_order_are_preserved(sk):
    pandas = pytest.importorskip("pandas")
    features = pandas.DataFrame(
        {"temperature": [-2.0, -1.0, 1.0, 2.0], "pressure": [0.0, 1.0, 1.0, 2.0]}
    )
    target = np.array([30, 30, 10, 10])
    estimator = sk.linear_model.LogisticRegression(max_iter=500).fit(features, target)
    result = from_sklearn(estimator)

    assert result.provenance.feature_names == ("temperature", "pressure")
    assert result.provenance.class_labels == tuple(
        str(label) for label in estimator.classes_
    )
    np.testing.assert_array_equal(
        np.asarray(cast(_LabelArrayModel, result.model.model).labels), estimator.classes_
    )
