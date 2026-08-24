# Native machine learning

`phydrax.ml` provides finite-dimensional statistical learning models that remain
native JAX programs and compose with the rest of Phydrax. It is not a mutable
estimator compatibility layer. A fitting recipe is an immutable configuration;
`phydrax.ml.fit` is a pure map from a recipe, data, masks, weights, and an explicit
key to a frozen executable model, diagnostics, and a declared gradient contract.

## The fitting lifecycle

```python
import jax
import jax.numpy as jnp
import phydrax as phx

features = jnp.array([[-1.0, 0.2], [-0.4, -0.7], [0.1, 0.3], [0.8, -0.2], [1.0, 0.9]])
targets = 0.7 * features[:, 0] - 0.25 * features[:, 1] + 1.2
weights = jnp.array([0.8, 1.0, 1.2, 1.0, 0.7])
query = jnp.array([0.25, -0.1])
key = jax.random.key(0)
recipe = phx.ml.linear.RidgeRecipe(alpha=1e-3)
result = phx.ml.fit(
    recipe,
    jnp.asarray(features),
    jnp.asarray(targets),
    sample_weight=weights,
)
prediction = result.model(query)
```

The recipe is unchanged by fitting. The returned `FitResult` contains:

- `model`: a `FrozenModel` that is excluded from Phydrax solver parameter
  partitions but remains an ordinary differentiable JAX PyTree when called;
- `diagnostics`: family-specific numerical evidence;
- `valid` and `status`: scalar or case-shaped fit validity;
- `method`: the resolved numerical method, not an ambiguous `"auto"` policy;
- `gradient_contract`: the precise derivative surface supported by the fit and
  prediction operations.

Call `result.as_trainable()` to make the fitted arrays an explicit warm start for a
later optimization problem. This unwraps the same arrays; it does not copy them or
silently change the fitting gradient.

## Canonical data semantics

Dense features have shape

```text
case_shape + (sample, feature)
```

Targets begin with `case_shape + (sample,)` and retain every remaining output
axis. A fit therefore handles scalar, vector, tensor, and multi-output targets
without flattening their scientific meaning. `MLBatch` stores these axes together
with:

- a feature mask and target mask;
- a sample mask;
- statistical sample weights;
- physical quadrature or empirical-measure weights;
- optional integer group identities;
- explicit feature and target schemas.

A recipe declares whether it uses statistical weights, measure weights, their
product, or no weights. Masking is applied before arithmetic: a masked or
zero-weight `NaN` cannot leak through a multiplication by zero.

`SparseFeatures` is a fixed-width sparse-row representation backed by
`phydrax.sparse.RowRelation`. Algorithms either implement a genuine sparse path or
reject sparse input. They never densify silently.

## Schemas and categorical values

`FeatureSchema` assigns a stable name and semantic kind to each feature.
`TargetSchema` records target kind, output names, and the external class
vocabulary. Classification kernels operate on contiguous internal integer class
indices. The fitted model preserves the external label ordering and converts only
at its explicit input/output boundary.

Category discovery, hard bin construction, rank selection, and class vocabulary
construction are discrete fit operations. They are never advertised as
continuously differentiable. Use an explicit schema when reproducibility across
fits or artifacts matters.

## Exact and relaxed algorithms

Phydrax gives different mathematical objects different types:

| Operation | Exact surface | Differentiable surface |
| --- | --- | --- |
| Classification | labels from `argmax` | logits and probabilities |
| Clustering | cluster indices | responsibilities or soft assignments |
| Feature selection | selected integer indices | continuous feature gates |
| Neighbors | hard top-k indices | kernel or attention weights |
| Trees | hard feature/threshold routing | soft routing gates |
| Quantiles and bins | sorted order and hard bins | separately named smooth surrogate |

There is no default straight-through estimator. Hard outputs terminate the
continuous gradient path. A relaxed model can be hardened into a new exact model;
the hardening operation itself remains nondifferentiable.

## Differentiating through fitting

Fitting gradients are separate from prediction gradients. Each family records one
of these fit modes:

- **direct**: closed-form array operations are differentiated directly;
- **implicit**: a root or KKT system is differentiated under its regularity
  conditions;
- **unrolled**: a fixed-capacity iterative program is differentiated through
  `jax.lax.scan`;
- **spectral**: projector or basis derivatives require eigengap conditions;
- **relaxed**: the fit differentiates through an explicitly smooth replacement for
  a discrete algorithm;
- **stopped**: the fit contains a declared discrete choice and supplies no gradient
  through that choice.

The contract separately records gradients with respect to training features,
targets, weights, continuous hyperparameters, prediction inputs, and fitted
parameters. Conditions such as full rank, positive regularization, active-set
stability, a nonzero eigengap, or finite temperature are part of the result.

See [ML differentiation contracts](../appendix/ml_differentiability.md) for the
failure boundaries and numerical interpretation.

## Composition without leakage

`Pipeline`, `FeatureUnion`, `ColumnTransformer`, and
`TransformedTargetRegressor` are immutable fit recipes. A pipeline fits each stage
using only the `MLBatch` supplied to that fit. Model-selection utilities construct
the training fold first and fit the whole composed recipe inside the fold. A
scaler, imputer, encoder, target transform, or feature selector therefore cannot
consume validation observations accidentally.

Randomness is split deterministically from the explicit fit key. Stage names,
resolved schemas, child diagnostics, and child gradient contracts remain available
on the fitted composition.

## Scientific integration

Fitted pointwise models implement the shared `AbstractArrayModel` and
`ModelBinding` contracts used by neural, uncertainty, and domain models. They can
therefore be:

- called under `jax.jit`, `jax.vmap`, and JAX transformations;
- bound into a `DomainFunction` when their input schema matches a domain layout;
- used as fixed reduced-order closures or explicitly converted to trainable warm
  starts;
- composed with `phydrax.kernels`, `phydrax.optim`, operator models, and UQ;
- stored in checksum-validated native ML artifacts;
- exported through the existing ONNX callable boundary when their JAX primitives
  are representable.

Batch-dependent transforms declare blockwise execution. Pointwise predictors do
not pretend to accept a scientific batch when their mathematical contract is one
sample at a time.

## Interoperability boundary

Optional converters accept selected already-fitted sklearn objects and XGBoost
JSON/UBJSON artifacts. Conversion is one-time and fail-closed:

1. the converter checks the exact source class and supported configuration;
2. every required fitted attribute is validated;
3. arrays, feature/class ordering, objective transforms, and missing-value
   semantics are copied into a native immutable model;
4. source version, configuration, and license provenance are recorded;
5. native prediction never imports or calls the source package.

Unsupported options raise `UnsupportedConversionError`; there is no fallback to
source prediction and no per-call NumPy conversion.

## Numerical policies

Recipes make the following choices explicit:

- QR/SVD versus regularized augmented solves;
- rank and conditioning thresholds;
- direct, implicit, or unrolled differentiation;
- fixed iteration and capacity limits;
- dense versus blockwise pairwise execution;
- exact versus histogram tree splitting;
- hard versus soft decision policies;
- float32 or float64 data placement.

A fit that is rank-deficient, nonfinite, infeasible, nonconverged, or capacity
exhausted returns that status. Numerical fallback is algorithm-specific and
recorded; no catch-all fallback changes device, dtype, solver, or semantics.

## Public family catalog

The [ML API reference](../api/ml/index.md) renders every public recipe, fitted
model, result, diagnostic, and function. The catalog below separates the
algorithmic surfaces; names ending in `Model` are the immutable fitted
executables returned by the corresponding recipe.

### Data transforms and workflows

- `preprocessing`: `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`,
  `RobustScaler`, `NormScaler`, `SimpleImputer`, `OneHotEncoder`,
  `OrdinalEncoder`, `TargetEncoder`, `PolynomialFeatures`, `SplineTransformer`,
  `PowerTransformer`, `QuantileTransformer`, `FourierFeatures`,
  `RandomFourierFeatures`, `GaussianRandomProjection`,
  `SparseRandomProjection`, and `FeatureHasher`.
- `compose`: `Pipeline`, `FeatureUnion`, `ColumnTransformer`, `ColumnSelector`,
  and `TransformedTargetRegressor`. `BatchTransformModel`,
  `SchemaTransformModel`, and `ReversibleTransformModel` define the child
  capabilities accepted by these compositions.
- `model_selection`: `KFoldPlan`, `StratifiedKFoldPlan`, `GroupKFoldPlan`,
  `BlockSplitPlan`, `TimeSeriesSplitPlan`, `RollingWindowSplitPlan`,
  `NestedSplitPlan`, `CrossValidator`, `GridSearch`, `RandomSearch`,
  `SuccessiveHalvingSearch`, `ParameterGrid`, `cross_validate`,
  `nested_cross_validate`, and `select_metric`.

Scaling and affine basis expansion are direct smooth fits on regular data.
Category/vocabulary discovery, empirical quantile knots, hashing, fold
construction, candidate selection, and halving are discrete. A transform that
needs dense arithmetic rejects `SparseFeatures`; sparse projections and hashing
retain genuine sparse paths where declared.

### Linear, generalized-linear, robust, sparse, and online learning

`linear` provides:

- `OLSRecipe`, `RidgeRecipe`, and `TikhonovRecipe`;
- `LassoRecipe`, `ElasticNetRecipe`, `GroupLassoRecipe`, and
  `SparseGroupLassoRecipe`;
- `LogisticRegressionRecipe` and `MultinomialLogisticRegressionRecipe`;
- `PoissonRegressorRecipe`, `GammaRegressorRecipe`, and
  `TweedieRegressorRecipe`;
- `HuberRegressorRecipe`, `QuantileRegressorRecipe`,
  `RANSACRegressorRecipe`, and `TheilSenRegressorRecipe`;
- `SGDRegressorRecipe`, `SGDClassifierRecipe`, `PerceptronRecipe`,
  `PassiveAggressiveRegressorRecipe`, and
  `PassiveAggressiveClassifierRecipe`.

Regularized dense solves use direct derivatives. Proximal and online methods
differentiate a fixed unrolled program. Quantile/KKT derivatives require stable
active constraints. RANSAC sampling, Theil--Sen order statistics, support
selection, and hard class labels are discrete. Rank, infeasibility,
nonconvergence, and configured QP/iteration capacity are reported rather than
repaired by changing solver.

### Probabilistic classification and calibration

- `discriminant`: `LinearDiscriminantRecipe`,
  `QuadraticDiscriminantRecipe`, `ShrinkageDiscriminantRecipe`, and
  `RegularizedDiscriminantRecipe`.
- `naive_bayes`: `GaussianNaiveBayesRecipe`,
  `BernoulliNaiveBayesRecipe`, `MultinomialNaiveBayesRecipe`,
  `ComplementNaiveBayesRecipe`, and `CategoricalNaiveBayesRecipe`.
- `multiclass`: `OneVsRestRecipe`, `OneVsOneRecipe`, `OutputCodeRecipe`,
  `MultilabelRecipe`, `ClassifierChainRecipe`, and
  `SmoothClassifierChainRecipe`.
- `calibration`: `PlattCalibrationRecipe`, `TemperatureCalibrationRecipe`,
  `VectorCalibrationRecipe`, `MatrixCalibrationRecipe`,
  `IsotonicCalibrationRecipe`, `SmoothIsotonicCalibrationRecipe`,
  `MulticlassCalibrationRecipe`, and `CalibratedClassifierRecipe`.

Logits, log-probabilities, and probabilities are the differentiable prediction
surfaces. Label decoding is terminal. Covariance-based fits require positive
class mass and the reported rank/regularization conditions. Categorical
vocabularies, one-vs-one voting, output-code construction, classifier-chain
ordering, and exact isotonic blocks are discrete; smooth chains and smooth
isotonic calibration are separately named alternatives.

These recipes fit complete classical estimators from `MLBatch`. To train an
arbitrary neural or physics-composed `DomainFunction`, use the classification terms
under `phydrax.terms`. Binary outputs are one Bernoulli logit; mutually exclusive
multiclass outputs are full categorical logits; independent multilabel outputs are
one sigmoid logit per named label. `TargetSchema("ordinal", class_labels=...)`
declares an ordered vocabulary for the cumulative-link term and is distinct from
query/item ranking.

`SupervisedSoftClassificationTerm` consumes explicit probability-valued targets;
label smoothing is performed when constructing those targets, not through a hidden
term option. Focal and overlap objectives are explicit non-likelihood risks and may
damage probability calibration. NLL, Brier, and calibration diagnostics remain the
probabilistic reference surfaces.

### Decomposition and latent representations

`decomposition` provides `PCA`, `IncrementalPCA`, `POD`, `TruncatedSVD`,
`FactorAnalysis`, `ICA`, `NMF`, `DictionaryLearning`, `SparseCoding`, `PLS`,
and `CCA`, together with the corresponding fitted model and diagnostic types.

Fixed-rank linear projections are smooth in their fitted arrays. Fit derivatives
through PCA/POD/SVD/factor/CCA subspaces require the recorded rank and spectral
gap. Basis derivatives additionally require a stable canonical sign or complex
phase. NMF, dictionary learning, sparse coding, and ICA expose the finite
unrolled fit actually executed. Energy-based rank selection and component
permutations are discrete.

### Kernels, neighbors, density, and metric learning

- `kernel_methods`: `KernelRidgeRecipe`, `LeastSquaresSVMRecipe`,
  `SupportVectorClassifierRecipe`, `SupportVectorRegressorRecipe`,
  `OneClassSVMRecipe`, `KernelPCARecipe`, `NystromRecipe`,
  `RandomFourierFeaturesRecipe`, `GaussianProcessClassifierRecipe`,
  `BernoulliGaussianProcessClassifierRecipe`, and
  `CategoricalGaussianProcessClassifierRecipe`.
- `neighbors`: `KNeighborsRegressorRecipe`, `KNeighborsClassifierRecipe`,
  `RadiusNeighborsRegressorRecipe`, `RadiusNeighborsClassifierRecipe`,
  `KernelNeighborsRegressorRecipe`, `KernelNeighborsClassifierRecipe`,
  `KernelDensityRecipe`, `LocalOutlierFactorRecipe`, `NearestCentroidRecipe`,
  `MahalanobisMetricRecipe`, and `NeighborhoodComponentsAnalysisRecipe`.

Kernel solves and fixed-support predictions are smooth under conditioning
requirements. Kernel PCA and Nyström fits add spectral-gap conditions. Random
features require an explicit key; the sampled basis is fixed for a fitting
derivative. Exact support-vector active sets, Nyström landmark choices, k-neighbor
indices, radius membership, LOF neighborhoods, and nearest-centroid labels are
hard. Kernel neighbors, density scores, GP probabilities, and learned linear
metrics provide the smooth alternatives stated by their contracts.

### Covariance and mixture models

- `covariance`: `EmpiricalCovariance`, `WeightedCovariance`,
  `DiagonalCovariance`, `FactorCovariance`, `LedoitWolfCovariance`,
  `OASCovariance`, `RobustCovariance`, `GraphicalLasso`, and
  `StreamingGaussianMoments`.
- `mixture`: `GaussianMixture` and `BayesianGaussianMixture`, with explicit
  covariance type, initialization, and empty-component policy.

Moment fits are direct when effective sample mass and regularized covariance are
valid. Factor fits require a retained spectral gap. Robust covariance,
graphical lasso, and mixture fits expose fixed unrolled iterations and
nonconvergence. Responsibilities and log densities are smooth; component labels,
initialization choices, pruning, and hard assignments are discrete. Complex
covariance uses Hermitian products; real probability/order policies reject
unsupported complex inputs.

### Clustering, manifolds, and graph learning

- `clustering`: `KMeans`, `MiniBatchKMeans`, `StreamingKMeans`, `KMedoids`,
  `SoftKMeans`, `MeanShift`, `DBSCAN`, `AffinityPropagation`,
  `AgglomerativeClustering`, `ConnectivityClustering`,
  `SpectralClustering`, `SpectralBiclustering`, and
  `SpectralCoclustering`.
- `manifold`: `build_neighbor_graph`, `LocallyLinearEmbeddingRecipe`,
  `IsomapRecipe`, `SpectralEmbeddingRecipe`,
  `MultidimensionalScalingRecipe`, `TSNERecipe`, and
  `FuzzyGraphEmbeddingRecipe`.
- `semi_supervised`: `LabelPropagationRecipe`, `LabelSpreadingRecipe`,
  `HardLabelPropagationRecipe`, `SoftSelfTrainingRecipe`,
  `HardSelfTrainingRecipe`, `SoftOneClassCompositionRecipe`, and
  `HardOneClassCompositionRecipe`.

`SoftKMeans`, continuous MDS objectives, t-SNE/fuzzy embeddings, soft graph
probabilities, label propagation probabilities, and soft compositions expose
finite-program derivatives under their listed conditions. Cluster indices,
medoids, density connectivity, graph edges/components, hierarchical merges,
spectral rank/order, pseudo-label acceptance, and external hard labels are
discrete. Spectral methods additionally report gap and disconnected-graph
evidence.

### Trees and ensembles

- `tree`: `DecisionTreeRegressor`, `DecisionTreeClassifier`,
  `ExtraTreeRegressor`, `ExtraTreeClassifier`, `RandomTreeRegressor`,
  `RandomTreeClassifier`, `RandomForestRegressor`,
  `RandomForestClassifier`, `ExtraTreesRegressor`, `ExtraTreesClassifier`,
  `AdaBoostRegressor`, `AdaBoostClassifier`, `GradientBoostingRegressor`,
  `GradientBoostingClassifier`, `HistGradientBoostingRegressor`,
  `HistGradientBoostingClassifier`, `XGBoostRegressor`,
  `XGBoostClassifier`, `XGBoostRanker`, `SoftDecisionTreeRecipe`,
  `SoftRandomForestRecipe`, and `SoftGradientBoostedTreesRecipe`.
- Tree inspection functions are `capacity_diagnostics`,
  `convergence_diagnostics`, `export_tree`, `feature_importance`,
  `partial_dependence`, `soft_tree_gradient_attribution`, and `tree_shap`.
- `ensemble`: `BaggingRecipe`, `RandomSubspaceRecipe`, `HardVotingRecipe`,
  `SoftVotingRecipe`, `StackingRecipe`, and `MixtureOfExpertsRecipe`, plus
  homogeneous, heterogeneous, and feature-subset fitted containers.

Hard trees are differentiable only in continuous leaf values and post-fit
objective transforms; split feature, threshold, missing route, topology,
bootstrap sample, and hard vote are discrete. Exact, histogram, first-order, and
XGBoost-style second-order splitting are distinct recorded methods. Soft tree
gates and soft voting/experts are genuinely relaxed array programs. Hardening
creates a new hard model and carries no straight-through derivative. Fixed
tree/node/category capacities and convergence limits are observable diagnostics.

### Feature selection, inspection, and metrics

- `feature_selection`: `VarianceFilterRecipe`, `ScoreFilterRecipe`,
  `MutualInformationFilterRecipe`, `RecursiveFeatureEliminationRecipe`,
  `SequentialFeatureSelectionRecipe`, `ModelBasedSelectionRecipe`, and
  `ContinuousSparseGateRecipe`.
- `inspection`: `gradient_sensitivity`, `jacobian_sensitivity`,
  `hessian_sensitivity`, `partial_dependence`,
  `individual_conditional_expectation`, `permutation_importance`,
  `influence_functions`, and `leverage_and_cooks_distance`.
- `metrics` includes regression losses/scores; exact and smooth
  classification scores; calibration errors and proper probabilistic scores;
  ranking/retrieval scores; clustering agreement and geometry scores; and
  `AbstractScorer`/`FunctionScorer` wrappers. Exact functions and their
  `smooth_*` alternatives are separate public names in the
  [metrics reference](../api/ml/metrics_inspection.md).

Selected indices, mutual-information bins, recursive/sequential choices,
permutations, exact labels/ranks, confusion counts, and cluster assignments are
discrete. Continuous feature gates, callable sensitivity, smooth metrics, and
regular influence systems retain their documented derivatives. Metrics return
explicit empty, invalid, single-class, undefined, and zero-denominator status
instead of a plausible sentinel.

### Outliers, artifacts, and conversion

- `outliers`: `CovarianceOutlierRecipe`, `EllipticEnvelopeRecipe`,
  `KernelDensityOutlierRecipe`, `RobustNoveltyRecipe`, `OneClassSVMRecipe`,
  and `IsolationForestRecipe`; `SmoothIsolationForestModel` is the relaxed
  forest surface.
- `artifacts`: `save_ml_artifact`, `read_ml_artifact`, and `load_ml_model`.
- `interop`: `from_sklearn`, `from_xgboost_artifact`, and `save_ml_onnx`.

Continuous scores may be smooth even when the reported inlier/outlier decision is
hard. Isolation topology and support selection are discrete; smooth isolation is
a separate fitted model. Artifacts and converters are terminal serialization or
copy boundaries and do not carry gradients through source formats.

## Keys, transforms, and failure handling

A recipe whose algorithm samples anything requires an explicit JAX key. The key
is split by named stage and fixed-capacity iteration; repeating a fit with the same
key and arrays repeats the result. Phydrax never reads a global NumPy or Python
random state.

Use the returned executable exactly as declared:

```python
result = phx.ml.fit(recipe, features, targets, key=key)
if not bool(result.valid):
    raise ValueError((int(result.status), result.diagnostics))

fixed_prediction = result.model(query)
warm_start = result.as_trainable()
```

`result.model` and `warm_start` contain the same fitted arrays. The first is a
`NonTrainableState` for solver partitioning; the second deliberately exposes
those leaves to later optimization. Neither object is mutable.

Invalid dynamic data is represented by `valid`, `status`, and diagnostic arrays so
case-batched and transformed fits remain JAX programs. Invalid static
configuration, schema contradictions, unsupported dtypes/storage, and impossible
shape contracts raise eager `TypeError` or `ValueError`. No exception handler
turns an unsupported algorithm into another algorithm.

## Reproducing the performance benchmark

`tools/ml_benchmarks.py` measures complete fit-and-inference workflows rather than
isolated array primitives. Its default small and medium matrices cover ridge,
PCA, fixed-iteration k-means, kernel ridge, and fixed-capacity histogram trees:

```text
python tools/ml_benchmarks.py \
    --output benchmarks/native_ml.json \
    --warmup 1 \
    --repeat 3 \
    --scales small medium
```

Every timed call synchronizes inputs before dispatch and all result leaves before
stopping the timer. The first warmup records compilation where the whole workflow
is JIT-compatible; later repetitions record steady fit plus inference. Tree fitting
is intentionally reported as eager fitting plus JIT inference because its exact
structure-building boundary is discrete.

The JSON records the software/device environment, exact shapes and recipe
configuration, validity/status, per-repetition time, throughput, output
shape/dtype/statistics, and a SHA-256 checksum. Results are machine-specific
evidence, not a cross-device speed guarantee. A benchmark aborts on an invalid fit,
non-success status, empty prediction, or nonfinite output so a fast no-op cannot be
mistaken for a successful implementation.
