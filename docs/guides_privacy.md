# Differential privacy

`phydrax.privacy` is a research control plane for differential-privacy
semantics, accounting, private operator training, and release evidence. It does
not implement new privacy mathematics. The initial mechanism is assembled from
[JAX Privacy](https://github.com/google-deepmind/jax_privacy) and accounted by
[Google DP Accounting](https://github.com/google/differential-privacy/tree/main/python/dp_accounting).

Install the exact private-training provider with:

```bash
uv sync --extra privacy-jax
```

The `privacy-jax` and `fmi` extras currently conflict because their upstream
packages require incompatible `attrs` ranges. Use separate environments until
that upstream metadata conflict is resolved.

## Guarantee boundary

A differential-privacy statement is meaningful only with all of these fixed:

- the privacy unit;
- the neighboring-dataset relation;
- the trusted party and release boundary;
- contribution grouping and clipping;
- batch selection;
- the randomized mechanism;
- the complete composed event;
- the accountant and target δ;
- every released metric, selected model, and preprocessing statistic;
- the randomness and finite-precision implementation.

`PrivacyDefinition` binds the first three assumptions. `DPSGDPlan` binds
Poisson sampling, per-unit ℓ₂ clipping, the fixed iteration ceiling, numerical
dtype, normalization divisor, accountant, and pinned provider. The prepared
provider returns one inseparable sampler, clipped query, Gaussian noiser, and
accounting event.

The initial qualified coordinate is deliberately narrow:

| Coordinate | Support |
| --- | --- |
| Trust model | central curator |
| Privacy unit | one operator case |
| Adjacency | add/remove one |
| Sampling | independent Poisson |
| Mechanism | Gaussian DP-SGD |
| Accounting | PLD or default-order RDP |
| Parameters | dense real float32 or float64 arrays |
| Batch structure | fixed static case schema; dynamic array values |
| Data source | in-memory case source |
| Execution | one process and one device |
| Randomness | research JAX PRNG |
| Disposition | research, never public-release authorized |

Local, shuffle, distributed, user-level, complex, Riemannian, mixed-precision,
adaptive-clipping, and private-selection profiles are not implied.

## Define a private training plan

```python
import phydrax as phx


definition = phx.privacy.PrivacyDefinition(
    phx.privacy.PrivacyUnit("operator-case"),
    phx.privacy.NeighboringRelation.ADD_OR_REMOVE_ONE,
    trust_model=phx.privacy.TrustModel.CENTRAL,
)
scope = phx.privacy.PrivateDataScope("curator-issued-study-id", definition)
budget = phx.privacy.PrivacyBudget(epsilon=3.0, delta=1e-6)
privacy = phx.privacy.PrivateTrainingPlan(
    phx.privacy.DPSGDPlan(
        scope,
        budget,
        sampling_probability=0.01,
        iterations=1000,
        clipping_norm=1.0,
        normalize_by=32.0,
        microbatch_size=1,
        dtype="float32",
    )
)
```

`scope_id` is an opaque curator identity. Do not use a raw patient, user,
experiment, or dataset identifier. `sampling_probability` is public and fixed;
the certificate does not publish dataset size. `normalize_by` is also public
and must not be the realized random batch size.

There is no default ε or δ. Their interpretation depends on the privacy unit,
adjacency, threat model, population, and use case.

## Train an operator

```python
result = phx.nn.operator.training.fit_operator(
    model,
    private_dataset,
    privacy=privacy,
    steps=1000,
    include_model_losses=False,
    loss_terms=(
        phx.nn.operator.training.SupervisedOperatorLoss(),
    ),
    normalization=public_physical_normalization,
    key=research_key,
)
```

The provider samples operator cases. For each selected case, Phydrax evaluates
one isolated scalar objective, differentiates it, clips the complete case
gradient once, sums clipped gradients, and adds noise before the optimizer sees
the result. Spatial points, mesh entities, time samples, and target values in
one case are therefore one protected contribution rather than falsely treated
as independent records.

A sampled empty batch still executes the Gaussian mechanism. An internal dummy
lane is evaluated and its complete gradient is replaced by exact zero by the
upstream padding contract before noise is applied. No dummy value or selected
index enters public evidence.

### Fail-closed restrictions

Private operator training rejects:

- automatic normalization fitted from private data;
- attached model losses without an explicit public/private objective partition;
- ordinary gradient accumulation;
- conflict-free gradient composition and update alignment;
- dynamic loss scaling;
- non-public validation;
- TensorBoard output;
- caller-owned mini-batch sizes;
- complex parameters;
- distributed execution;
- a step request beyond the mechanism iteration ceiling.

Provider microbatching is controlled by `DPSGDPlan.microbatch_size`. Variable
Poisson batches currently support only `None` or `1`; larger fixed microbatches
require a padding and accounting profile that is not yet implemented. A shorter
`steps` value may stop at an earlier mechanism boundary and yields a certificate
for the actual number of completed mechanisms. Resume may extend the same run
up to the original fixed iteration ceiling.

Validation is permitted only when `PrivateTrainingPlan.validation_is_public`
is explicit. Such validation is outside the private data scope. Private
validation and model selection need a separately accounted mechanism and are
not implemented by this profile.

## Metrics and logs

Raw train loss, component losses, gradient norms, clipping scales, selected
indices, and sampling keys are private runtime state. Private fits therefore
return empty train metric mappings and do not evaluate initial or terminal
training loss. Accessing `OperatorFitResult.initial_loss` or `final_loss` raises
instead of returning a fabricated value.

Iteration numbers, public configuration, and the accounted privacy bound are
safe control metadata. Returned wall time remains inside the trusted process
because timing side channels are not qualified, and inference artifacts omit
it. No private metric release mechanism is currently implemented.

## Checkpoints and artifacts

A training checkpoint contains optimizer state, sampler state, raw JAX noise
state, and exact progress. It is a restricted continuation artifact, not a DP
release. The sampler and typed noise key are restored exactly; a caller's new
resume key cannot change continuation.

A task-bound `TrainedOperator` carries an optional `PrivacyCertificate`.
Operator inference artifacts serialize that public-safe certificate, but a
private inference artifact refuses colocated training state. The certificate
contains the scope and definition identities, provider/version, exact event,
mechanism-plan and qualification-profile identities, query ℓ₂ sensitivity,
planned iteration ceiling, accounted ε and δ, randomness assurance, and content
addresses. It contains no raw data identity, selected index, metric, or key.

`save_operator_artifact` stores private artifacts as `restricted` by default.
Passing `public_release=True` invokes the certificate release gate and therefore
fails for the current research profile.

The current JAX provider records
`RandomnessAssurance.RESEARCH_PRNG`. Consequently:

```python
result.privacy_certificate.require_public_release()
```

raises `PermissionError`. Passing tests or empirical attacks cannot upgrade
this disposition. A public-release profile requires a compatible reviewed
secure-noise provider and independent release evidence.

## Release composition

`PrivacyReleaseLedger` composes distinct randomized release roots over one
`PrivateDataScope`:

```python
ledger = phx.privacy.PrivacyReleaseLedger(scope, budget)
ledger = ledger.register("model-root", result.privacy_certificate)
```

Registering the same root and certificate again is idempotent. Registering an
independent root composes its event and fails atomically if the combined ε
exceeds the budget. Deterministic post-processing should retain the original
release-root identity; assigning it a new root intentionally charges it as an
independent release.

## What differential privacy does not replace

This substrate does not provide encryption, raw-data access control, secure
aggregation, local DP, legal or regulatory compliance, checkpoint protection,
or protection for an incorrectly declared privacy unit. Differential privacy
bounds the influence of the declared neighboring change on the declared
released outputs. Everything outside that boundary remains a separate security,
rights, and governance responsibility.
