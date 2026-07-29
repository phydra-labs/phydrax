# Neural-operator uncertainty

Operator uncertainty uses the same `PredictiveField` statistics as pointwise models,
but retains the complete `OperatorBatch` output contract: physical case axes, query
geometry, query masks, quadrature, and `OperatorOutputSpec` channels.

Tensor-grid outputs keep their `OperatorAxis` names. Point-cloud and channel axes use
reserved internal names so they cannot collide with physical case axes. Every
stochastic source remains an explicit `SampleAxis`; no reduction infers uncertainty
semantics from axis position.

Masked query padding is replaced by zero only in predictive storage. The attached
query mask remains authoritative, and statistics, proper scores, and calibration
exclude the padding. A non-finite value at a valid query location either records an
invalid whole-function draw or raises immediately according to `valid_policy`.

## Predictive results

::: phydrax.uq.OperatorPredictiveField
    options:
        members:
            - __init__
            - from_predictive
            - output_mask
            - output_weights
            - mean
            - variance
            - std
            - quantile
            - interval
            - epistemic_variance
            - input_variance
            - observation_variance
            - total_variance
            - decompose_variance

---

::: phydrax.uq.OperatorPredictionInterval
    options:
        members:
            - __init__

---

::: phydrax.uq.operator_prediction_field

---

::: phydrax.uq.operator_predictive_from_samples

---

::: phydrax.uq.operator_input_predictive

---

::: phydrax.uq.sample_operator_predictive

## Geometry-aware operator uncertainty

`OperatorPredictiveField` is architecture-independent: predictions from `GINO`,
`RIGNO`, and `GAOT` retain case-dependent point-cloud query coordinates, masks,
and quadrature exactly as predictions from grid operators do. The stochastic
contract is stricter than shape equality:

- every ensemble or posterior draw must use the same `OperatorOutputSpec`, case
  axes, and query geometry for the batch being summarized;
- source-coordinate, source-value, source-mask, and source-quadrature draws may
  vary together under declared input-sample axes;
- pointwise means, quantiles, CRPS, and coverage require common output query
  locations across draws; remesh first when queries differ;
- energy scores and functional conformal calibration treat one complete valid
  output field as one event and use output quadrature when available.

Deep ensembles are the default epistemic baseline for every geometry
architecture. MC dropout is meaningful only if the stochastic layer was active
during fitting: `RIGNO` exposes `processor_edge_dropout`; `GAOT` exposes
attention and feed-forward dropout. One root PRNG key denotes one coherent
whole-function draw. Use `phydrax.nn.inference_mode` before deterministic
deployment or posterior-density construction.

Weight-space inference should target an explicit `ParameterSubspace`. Useful
small choices include a GINO output projection, a RIGNO decoder/readout, or a
GAOT output projection. Never use global leaf order as a proxy for “last layer”
in an encode–process–decode model. A geometry ensemble captures parameter
uncertainty conditional on its observed mesh; uncertain domain shapes belong in
the input-sample axes instead.

## Fixed observation likelihood

`FixedOperatorObservationLikelihood` defines a normalized finite-dimensional sensor
likelihood, not a continuum norm or a training objective. It fixes the complete batch,
combines the query and observation masks, sum-reduces likelihood elements within each
physical case, and then sums cases in `log_prob`. Quadrature weights are deliberately
absent unless a user derives a different stochastic observation model explicitly.

::: phydrax.uq.FixedOperatorObservationLikelihood
    options:
        members:
            - __init__
            - per_case_log_prob
            - log_prob
            - standardized_residual

## Whole-function conformal calibration

`OperatorFunctionalConformal` treats one complete output field or trajectory as one
exchangeable calibration case. The maximum score produces simultaneous bands. The
quadrature-weighted L2 score produces a norm ball; it does not define pointwise lower
and upper bounds.

::: phydrax.uq.OperatorFunctionalConformal
    options:
        members:
            - __init__
            - calibrate
            - interval

## Operator-aware proper scores

::: phydrax.uq.operator_ensemble_crps

---

::: phydrax.uq.operator_energy_score

---

::: phydrax.uq.operator_interval_coverage

---

::: phydrax.uq.operator_interval_width
