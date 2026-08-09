# ML differentiation contracts

This appendix defines what a derivative claim in `phydrax.ml` means. It is a
mathematical contract, not merely a statement that JAX can trace an implementation.

## Two derivative surfaces

A fitted predictor creates two different maps:

```text
prediction: (fitted parameters, query) -> output
fit:        (training data, weights, hyperparameters) -> fitted parameters
```

A model may have smooth prediction while its fit is nondifferentiable. Hard trees
are the simplest example: leaf values are differentiable parameters, but split
feature and threshold selection are discrete. Conversely, a fitted spectral basis
may be callable through smooth matrix multiplication while the basis returned by
fitting is not uniquely differentiable at repeated singular values.

`GradientContract` records prediction gradients and fit gradients separately. Its
levels are:

- `smooth`: continuously differentiable under the listed domain conditions;
- `almost-everywhere`: nondifferentiable only on declared boundaries;
- `conditional`: valid only while listed regularity or active-set conditions hold;
- `none`: no mathematical derivative is provided.

## Direct differentiation

Closed-form and fixed array programs use ordinary JAX differentiation. Examples
include strictly regularized ridge regression, affine scaling, Gaussian moment
estimation, and smooth prediction kernels.

For weighted ridge with design `X`, target `Y`, nonnegative diagonal weights `W`,
and regularization `R`, the fitted coefficients solve

```text
(Xᴴ W X + R) B = Xᴴ W Y.
```

The direct derivative is valid when the augmented system has the reported full
rank. A zero-weight sample is excluded before arithmetic, so a nonfinite value in
that sample cannot contaminate the solve. Negative or nonfinite weights make the
fit invalid rather than being silently clipped.

Ordinary unregularized least squares is differentiable only while rank remains
constant. At a rank change, the pseudoinverse map is not continuously
differentiable; the result records rank deficiency instead of hiding it behind a
fallback.

## Implicit differentiation

A fitted state `z*` defined by a residual equation

```text
F(z*, data, hyperparameters) = 0
```

can be differentiated by solving the linearized system

```text
(∂F/∂z) dz = -(∂F/∂input) dinput.
```

This is used only when the relevant Jacobian or KKT system is nonsingular and the
active constraints are stable. The result lists these conditions. At an active-set
change, hinge, quantile, constrained sparse, or QP fit is generally only
piecewise-smooth.

An implicit custom derivative must preserve the primal solver's tolerance and
regularization semantics. It may not substitute a different backward objective.

## Unrolled differentiation

Fixed-iteration algorithms execute a shape-stable `jax.lax.scan`. The gradient is
that of the finite program actually executed, including its initialization and
number of rounds. Typical uses include proximal sparse regression, EM, soft
clustering, iterative reweighting, and differentiable boosting.

Convergence masking freezes a converged state while retaining a fixed output
structure. Diagnostics report both the fixed capacity and the iteration at which
the tolerance was first met. Nonconvergence does not silently increase the number
of iterations under `jit`.

Unrolled gradients can have different numerical behavior from the derivative of an
ideal infinite-iteration fixed point. The fit mode makes this distinction visible.

## Spectral differentiation

PCA, POD, CCA, spectral clustering, and related methods involve eigenspaces or
singular subspaces.

### Projector derivatives

A subspace projector is invariant to sign, complex phase, and rotations inside the
subspace. Its derivative is well-defined while the retained subspace is separated
from the discarded subspace by a nonzero spectral gap. Projector-valued operations
are therefore the strongest default differentiation surface.

### Basis derivatives

An individual basis is not unique:

- every real singular vector admits a sign change;
- every complex singular vector admits a unit phase;
- a repeated singular block admits an arbitrary unitary rotation.

Phydrax applies a deterministic sign/phase convention away from ties and reports
the minimum retained gap. This removes incidental sign or phase flips but cannot
make a repeated eigenspace basis uniquely differentiable. Basis-gradient claims
are conditional on the reported gap and canonical pivot remaining nondegenerate.

### Rank selection

Selecting a rank from an energy threshold is discrete. A fixed-rank fit can be
spectrally differentiated under its gap conditions; the selected integer rank
cannot. Exact rank and retained-energy diagnostics remain available as terminal
outputs.

## Discrete algorithms

The following operations have no continuous derivative through their exact choice:

- sorting and hard quantiles;
- `argmax`, hard class labels, and hard cluster assignments;
- nearest-neighbor top-k indices;
- selected feature indices;
- category discovery and vocabulary construction;
- graph connectivity and connected components;
- tree split feature, threshold, and topology;
- bootstrap or subset samples;
- early stopping that changes program length.

JAX may return a zero or branch-local derivative for part of such a program. That
does not make the discrete choice differentiable. These outputs appear in
`nondifferentiable_outputs`, and the exact fit uses `fit_mode="stopped"` where the
choice controls learned structure.

## Relaxed alternatives

A relaxed model is a separate mathematical model:

- softmax probabilities instead of labels;
- kernel attention instead of top-k neighbors;
- responsibilities instead of cluster indices;
- continuous feature gates instead of selected indices;
- sigmoid or sparse-continuous tree gates instead of threshold routing;
- smooth sorting or quantile approximations instead of exact order statistics.

Temperature and regularization remain array-valued continuous hyperparameters when
possible. A temperature approaching zero can make derivatives singular or
ill-conditioned. Hardening produces a new exact model and terminates the gradient
path; Phydrax does not install a straight-through gradient by default.

## Weights and masks

The fit contract distinguishes:

- statistical sample weight, which changes empirical importance;
- measure weight, which represents quadrature or a physical empirical measure;
- their explicit product;
- structural masks, which remove observations or entries before arithmetic.

Gradients with respect to a positive weight are meaningful for smooth weighted
objectives. At zero weight, inclusion can change and the derivative is generally
one-sided or conditional. Mask booleans and group identifiers are structural and
nondifferentiable.

## Complex values

Linear, spectral, covariance, kernel, and readout families preserve complex values
where their mathematics is defined. Conjugate products are used for Hermitian
inner products. Algorithms based on ordering, categories, tree thresholds, or
real-valued class probabilities reject complex inputs explicitly.

JAX's complex differentiation convention applies to the executable array program.
A family must still state whether its objective is real-valued and whether the
reported gradient is with respect to real parameters, complex parameters, or both.

## Diagnosing invalid derivatives

Before using a fitting gradient, inspect:

1. `result.valid` and `result.status`;
2. rank, condition number, convergence, and capacity diagnostics;
3. `result.gradient_contract.fit_mode`;
4. the per-input gradient levels;
5. every listed condition and nondifferentiable output;
6. family-specific eigengap, active-set, temperature, or topology evidence.

A result with `ML_UNSUPPORTED_GRADIENT` has not produced the requested gradient
contract. Changing the solver, adding regularization, fixing rank/capacity, or using
an explicit relaxed model may create a valid contract; suppressing the status does
not.

## Family contract matrix

The exact `GradientContract` is stored on each `FitResult`; constructor options
can change a row below from direct to implicit/unrolled/stopped. This matrix is the
family-level rule, not a substitute for reading the returned conditions.

| Family | Smooth or conditional surface | Terminal or nondifferentiable surface | Principal validity conditions |
| --- | --- | --- | --- |
| Affine scaling and imputation | transform values; fitted moments for fixed masks | missingness mask and strategy choice | positive effective mass, finite observed values |
| Categorical encoding | transform for a fixed vocabulary | vocabulary discovery, category order, unknown-category policy | schema consistency and fixed vocabulary |
| Polynomial, spline, and Fourier bases | values, knots/frequencies when declared continuous | degree, knot count, sampled frequency identity | finite domain; explicit key for sampled bases |
| Random projections and hashing | values for a fixed projection/hash layout | sampled matrix, hash bucket, collisions | explicit key/capacity and fixed layout |
| OLS/ridge/Tikhonov | direct solve and prediction | rank/status diagnostics | constant rank or positive regularization |
| Lasso/elastic/group sparse fits | finite proximal program or stable KKT solution | active support and exact zero pattern | fixed iterations or stable active set |
| GLMs and logistic models | logits/means and finite iterative fit | decoded labels | finite link domain, positive curvature, convergence |
| Huber/quantile fits | piecewise-smooth objective or stable KKT solution | kink/active constraint identity | no residual on a changing kink; nonsingular KKT system |
| RANSAC and Theil--Sen | prediction conditional on fitted coefficients | random subset, inlier set, median/order statistic | fixed selected structure only |
| SGD/perceptron/passive-aggressive | fixed unrolled update program | shuffling/violation branch identity | explicit key where randomized, fixed capacity |
| Discriminant analysis | scores/probabilities and regularized covariance fit | labels and class vocabulary | positive class mass and nonsingular regularized covariance |
| Naive Bayes | log probabilities and probabilities | class/category vocabulary and labels | positive smoothed mass, valid event domain |
| Multiclass compositions | child scores/probabilities | one-vs-one vote, code/chain order, labels | every child fit valid; fixed composition |
| Platt/temperature/vector/matrix calibration | calibrated probabilities and smooth fit | labels | positive scale/temperature, rank and convergence |
| Exact isotonic calibration | piecewise-linear prediction for fixed blocks | pool-adjacent-violator block construction | fixed block partition |
| Smooth isotonic calibration | relaxed weights and calibrated prediction | none beyond optional reporting labels | positive finite bandwidth |
| PCA/POD/truncated SVD | projector and fixed-rank reconstruction | selected rank | retained/discarded eigengap and fixed rank |
| Spectral bases, ICA, CCA, factor analysis | basis conditional on canonical phase and gap | component permutation and selected rank | nonzero gap, nondegenerate pivot, required rank |
| NMF/dictionary/sparse coding | fixed unrolled factorization | exact support/component identity | finite iterations and positive/regularized updates |
| Kernel ridge/least-squares SVM | kernel parameters, support values, solve, prediction | none beyond status for the dense solve | positive regularization and conditioned kernel system |
| Support-vector models | prediction and stable KKT solution | support/active-set identity, hard labels | stable active set and nonsingular KKT system |
| Kernel PCA/Nyström | fixed-landmark spectral map | landmark choice and retained rank | fixed landmarks, eigengap, conditioned normalization |
| Random Fourier features | values/continuous kernel scales for fixed draws | sampled frequencies/phases | explicit fixed key and finite positive scales |
| GP classification | latent/probability prediction and fixed iterative approximation | class labels | positive covariance/noise and converged finite approximation |
| Exact k/radius neighbors | weighted target values inside a fixed neighborhood | top-k order, ties, radius membership, indices | no distance tie/boundary crossing for local piecewise derivatives |
| Kernel neighbors/KDE | kernel weights, density, targets, continuous metric | optional hard labels | positive bandwidth, positive effective mass |
| Mahalanobis/NCA metric learning | linear metric and finite unrolled objective | neighbor target identity/labels | positive regularization and fixed iterations |
| Empirical/weighted covariance | moments, log density, regularization | rank/status diagnostics | positive effective mass and positive regularized covariance |
| Shrinkage/factor covariance | shrinkage and fixed-rank covariance | selected factor rank | finite shrinkage denominator and eigengap |
| Robust covariance/graphical lasso | finite unrolled reweighting/proximal program | support pattern and convergence event | fixed capacity, finite positive covariance |
| Gaussian mixtures | responsibilities and finite EM program | component identity, pruning, hard assignments | positive component mass/covariance and fixed initialization |
| Soft k-means | centroids, responsibilities, temperature | optional hardened indices | positive temperature/mass and fixed iterations |
| Hard centroid/medoid/density clustering | values conditional on fixed assignment | assignments, medoids, connectivity, cluster count | no assignment/tie/connectivity change |
| Hierarchical/affinity clustering | values conditional on fixed graph/merge sequence | merge pair/order and exemplars | no tie or topology change |
| Spectral clustering/biclustering | relaxed embedding/projector | graph edges, rank/order, hard labels | connected graph as required and nonzero eigengap |
| LLE/Isomap/spectral embedding | regular reconstruction/geodesic/spectral coordinates | neighbor graph, shortest-path predecessor, basis rank | fixed connected graph, local rank, eigengap |
| MDS/t-SNE/fuzzy embeddings | finite objective/update program | initialization identity and graph construction | explicit key when stochastic, fixed graph/capacity |
| Label propagation/spreading | soft class probabilities | graph topology, external hard labels | fixed graph and converged/fixed iterations |
| Self-training/one-class compositions | soft child scores | acceptance/pseudo-label set and hard output | fixed acceptance structure for conditional derivatives |
| Hard trees and forests | leaf values, base score, smooth post-transform | split feature/threshold/route/topology/bootstrap/label | fixed valid structure and traversal capacity |
| Soft trees and soft boosting | gates, leaves, temperatures, finite boosting fit | hardening and any sampled structure | positive temperature and fixed capacity |
| AdaBoost/gradient/histogram/XGBoost fits | leaf/score updates conditional on structure | split/bin/tree choice and early-stop length | fixed chosen structure, positive Hessian/mass, capacity |
| Bagging/random subspaces | child predictions conditional on sampled subsets | sample/feature subset identity | explicit fixed key and valid child fits |
| Soft voting/mixture of experts | probabilities, expert weights, gate | optional hard expert/vote | normalized finite weights and valid children |
| Hard voting/stacking selection | meta-model values conditional on child outputs | hard vote and fold/candidate identity | fixed folds/children and valid meta-design |
| Exact feature selection | transformed retained values | selected indices, bins, recursive/sequential path | fixed selected set |
| Continuous sparse gates | gate values and relaxed fit | optional hard threshold | positive temperature/regularization and fixed iterations |
| Sensitivity and partial dependence | derivative of the actual callable and weighted reduction | hard model branches inherited from the model | model's own prediction contract and valid domain |
| Permutation importance | score conditional on a fixed permutation | permutation draw and ranking | explicit fixed key and valid scorer |
| Influence functions | implicit parameter/data influence | active set inherited from objective | nonsingular regularized Hessian/KKT system |
| Exact metrics | smooth value only where their exact formula is smooth | labels, counts, sorts, ranks, cluster assignments | nonempty domain and nonzero denominators |
| `smooth_*` metrics | probabilities, soft ranks/assignments and continuous reductions | no claim for later hardening | positive temperature and valid denominators |
| Model selection/search | fold-local continuous candidate scores | split, candidate index, ranking, halving survival | fixed split/candidate set; valid fold fits |
| Artifacts/converters/export | no derivative through serialization/conversion | byte layout, source schema, conversion decision | checksum/schema/version/configuration support |

## Status and derivative precedence

Status is evidence about the primal fit. It is not inferred from whether `jax.grad`
returns an array. Implementations apply the following conceptual precedence:

1. incompatible static configuration or unsupported storage/dtype raises;
2. empty or underfull effective data reports `ML_INSUFFICIENT_DATA`;
3. active nonfinite values report `ML_NONFINITE`;
4. primal infeasibility reports `ML_INFEASIBLE`;
5. numerical rank failure reports `ML_RANK_DEFICIENT`;
6. an unfinished finite iteration reports `ML_NONCONVERGED`;
7. exhausted structural storage reports `ML_CAPACITY_EXHAUSTED`;
8. a requested derivative outside the mathematical contract reports
   `ML_UNSUPPORTED_GRADIENT`;
9. otherwise the fit reports `ML_SUCCESS`.

A family can refine precedence when one condition makes another undefined, but its
diagnostics must retain the underlying evidence. Regularization can resolve a
rank-deficient raw problem while the diagnostics still report the raw singularity.

## JAX transformation boundary

The fitted array model is the unit passed to `jax.jit`, `jax.vmap`,
`jax.jacfwd`, `jax.jacrev`, or solver composition. Python estimator mutation,
source-package callbacks, data-dependent allocation, and exception-driven
algorithm replacement are outside the contract. Fixed-capacity loops keep one
shape and dtype in every carry. Case axes remain structural batch axes; they are
never inferred from a trailing output or feature dimension.
