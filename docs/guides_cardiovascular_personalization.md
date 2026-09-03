# Cardiovascular personalization

The cardiovascular personalization layer turns accepted model states and normalized observations into auditable, subsystem-scoped inverse problems. It does not provide a single “fit everything” entry point. Electrophysiology, mechanics, loading/circulation, and unloaded-reference geometry use distinct route types so that parameters with different state equations, gauges, and identifiability limits cannot be silently mixed.

This functionality is for numerical research and manufactured qualification. It is not clinical validation, a diagnostic device, or evidence of patient-specific predictive accuracy.

## Declare physical parameters

Each `CardiacParameterSpec` records:

- a stable name and `CardiovascularQuantitySpec`, including kernel/SI units, axes, sign convention, support association, and reference configuration;
- a native UQ bijector such as `IdentityBijector`, `ExpBijector`, or `SigmoidIntervalBijector`;
- a closed `CardiacParameterSupport`;
- a native normalized probability law used as the physical-space prior;
- the owning `CardiacSubsystem`; and
- a declared `ParameterIdentifiability` role (`PRIMARY`, `NUISANCE`, `FIXED`, or `CONDITIONAL`).

The declared role is metadata, not a claim that a parameter is identifiable. Local rank must still be checked at the accepted solution. `CardiacParameterSchema.parameter_space(...)` lowers only non-`FIXED` coordinates to the existing `phydrax.uq.ParameterSpace`; fixed values are injected unchanged into every state, observable, and evidence call. It does not introduce a second transform or prior implementation.

```python
from phydrax.applications.cardiovascular import (
    CardiacParameterSchema,
    CardiacParameterSpec,
    CardiacParameterSupport,
    CardiacSubsystem,
    cardiovascular_quantity,
)
from phydrax.uq import LogNormal, ExpBijector

conductivity = CardiacParameterSpec(
    "intracellular_conductivity",
    cardiovascular_quantity("electrical_conductivity"),
    ExpBijector(),
    CardiacParameterSupport(1.0e-4, 10.0),
    LogNormal(-0.7, 0.35),
    CardiacSubsystem.ELECTROPHYSIOLOGY,
)
schema = CardiacParameterSchema((conductivity,), schema_id="study.ep.conductivity")
```

Use kernel units at the solver boundary. Convert only through the quantity specification’s exact `to_si` and `from_si` operations.

## Prepare modality likelihoods

Convert an ingest-qualified `ObservationRecord` with `ModalityObservation.from_record`. A `ModalityLikelihoodChannel` fixes the observation mask before execution and can additionally declare:

- a `ReferenceGauge`, which differences valid samples against one valid reference and drops the redundant coordinate;
- a `LinearNuisanceModel` with an explicit basis and native prior;
- a `GaussianModelDiscrepancy` with a deterministic bias and optional low-rank covariance factor; and
- either an existing elementwise `AbstractLikelihood` or a native covariance action.

`ModalityLikelihoodChannel.correlated_gaussian(...)` combines measurement and model-discrepancy covariance, applies the fixed mask and gauge to the complete covariance, and prepares its precision through the PhydraX linear-algebra substrate. It rejects a singular reduced covariance rather than repairing it. A gauged Gaussian must use this constructor; an elementwise Gaussian cannot represent the correlation introduced by a shared reference. Likewise, an elementwise Gaussian accepts low-rank discrepancy only when its covariance is diagonal on the retained samples, so off-diagonal covariance is never silently discarded.

Compose stable-order channels with `MultimodalLikelihoodPlan(...).prepare()`. Predictions and nuisance values use the same tuple order as `plan.modalities`. Every evaluation retains per-channel residuals, likelihood contributions, nuisance-prior contributions, and fail-closed finite/successful evidence.

To evaluate generalization to a modality not used by the inverse solve, create a distinct plan identity:

```python
training_plan = full_plan.held_out(("tagged_mri_strain",))
training = training_plan.prepare()
```

Do not mask a held-out channel while retaining it in the objective: `held_out` removes it from the training likelihood and produces a new content identity. Evaluate the excluded channel separately at the accepted parameters.

## Build a subsystem inverse

Choose exactly one route adapter:

- `ElectrophysiologyInverseProblem` for electrophysiology parameters;
- `MechanicsInverseProblem` for passive and active mechanics parameters;
- `LoadingInverseProblem` for loading and circulation parameters; or
- `UnloadedGeometryInverseProblem` for reference-configuration parameters.

Each adapter requires a state residual, an observable projection, and a `fixed_topology` evidence function. The schema is rejected if it contains a subsystem not owned by that route. This is intentional: alternating or staged calibration should pass accepted outputs between route-specific problems rather than assemble one monolithic inverse.

`as_state_design_problem` lowers the adapter to the existing `StateDesignProblem`. `solve` and `solve_multistart` retain the native state and adjoint acceptance evidence. A cardiovascular inverse result is successful only when all of the following are accepted:

1. the state residual and state-solver status;
2. the adjoint transpose residual and adjoint-solver status;
3. every modality likelihood and nuisance support;
4. the parameter prior/support; and
5. the fixed-topology boundary used for differentiation.

`solve_multistart` takes explicit physical start points in stable order. It does not hide a random generator or promote a failed start merely because its reported objective is smaller.

## Diagnose local information

Run diagnostics after selecting an accepted route result:

- `SensitivitySVDPlan` differentiates a fixed-topology observable vector, scales parameter and observation coordinates, and uses the native SVD/factorization substrate. Inspect `rank`, `nullity`, `condition_number`, and `nullspace_basis`.
- `fisher_local_diagnostics` forms local Fisher information from a declared observation precision and optional prior information. Both matrices require symmetric positive-semidefinite evidence from the native self-adjoint eigensolver. Its pseudocovariance is descriptive local curvature, not a posterior or confidence guarantee.
- `ProfileLikelihoodPlan` fixes one coordinate on a declared grid and optimizes the other coordinates with a native minimization method, retaining every grid status.
- `check_directional_derivative` compares AD with a centered finite difference at the same fixed-topology point.

A rank-deficient sensitivity or Fisher matrix is explicit confounding evidence. Do not report separate fitted values for a confounded direction without adding genuinely informative data or an externally justified constraint.

## Design informative experiments

`ExperimentDesignCandidate` stores a sensitivity, noise precision, cost, and `ForwardAdjointEvidence`. A candidate is ineligible unless its forward state, adjoint, fixed topology, and derivative finiteness are all accepted. `ExperimentDesignPlan` supports D-, A-, and E-optimal greedy utilities relative to positive-definite prior information and a hard resource budget. The result records the fixed-capacity selected indices, selected mask, score history, total cost, final information, and candidate evidence mask.

Experiment-design scores are local numerical utilities. They do not establish feasibility, safety, or clinical utility of a pacing, imaging, or intervention protocol.

## Record investigational research validation

The host-only validation records govern research evaluation; they do not run models and do not issue a clinical or regulatory certificate.

`ClinicalResearchContext` binds a research question, study context, investigational-use statement, protocol, exactly one IRB approval or waiver, de-identification identities, and data-rights identities. Construction refuses protected health information, clinical-decision use, and regulated claims.

`ClinicalResearchValidationPlan` prospectively separates training, calibration, and validation cohorts and requires distinct site and temporal holdouts. It also binds the endpoint, comparator, subgroup analyses, out-of-distribution definition, failure-analysis plan, and prespecified acceptance criteria.

`ClinicalResearchValidationRecord` stores immutable, finite, JSON-compatible results under the plan identity. `record.evaluate()` returns `ClinicalResearchValidationEvidence` describing whether every required research-analysis section is present. `record_complete` means only that the governed record is complete; it does not mean that a model is safe, effective, clinically valid, approved, or certified.

## Manufactured qualification

Run the dedicated qualification surface:

```console
python tools/cardiovascular_personalization_qualification.py
```

It exercises deterministic synthetic recovery from multiple explicit starts, held-out modality composition, rank-deficient confounding detection, an AD/finite-difference check, and rejection of an experiment candidate without accepted adjoint evidence. These manufactured cases qualify software behavior only.
