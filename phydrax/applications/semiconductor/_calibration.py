#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded semiconductor inference, not a foundry parameter database.

Each case is one joint correlated measurement event. Independent cases form
native Gaussian likelihood blocks; correlated sweeps must remain in one case.
All physical arrays are SI. Explicit scales only condition inference, never
change the physical forward interface. Control errors are linearized locally;
measurement, de-embedding and control errors are declared mutually independent.
Numerical error is a qualification bound, not another fitted noise source.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._array_archive import array_collection_digest, write_array_archive
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...artifacts import ScientificArtifactEnvelope
from ...ein import contract
from ...linalg import DenseLinearOperator
from ...linalg.svd import svd, SVDProblem, SVDSolvePolicy
from ...qualification import (
    QualificationCoverageReport,
    QualificationEvidence,
    QualificationMatrix,
    ReferenceArtifactManifest,
    SupportTuple,
)
from ...uq import (
    DenseCovariance,
    find_map,
    fit_laplace,
    LaplaceResult,
    LinearizedGaussianMeasurementLikelihood,
    MAPResult,
    ParameterSpace,
    PosteriorProblem,
    propagate_linearized,
)
from ._quantities import _text, SemiconductorQuantitySpec


def _names(values: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values or len(set(values)) != len(values):
        raise ValueError("Require a nonempty immutable tuple of unique identities.")
    return tuple(_text(value, "calibration identity") for value in values)


def _array(value, shape=None, *, positive=False):
    result = jnp.asarray(value, dtype=float)
    if (shape is not None and result.shape != shape) or not bool(
        jnp.all(jnp.isfinite(result) & ((result > 0) if positive else True))
    ):
        raise ValueError(
            "Physical arrays require the declared shape and finite admissible values."
        )
    return result


def _quantities(values):
    if not isinstance(values, tuple) or not values:
        raise ValueError("Declare immutable native quantity specifications.")
    _names(tuple(value.name for value in values))
    if any(
        float(value.to_si(1.0)) != 1.0 or float(value.to_si(0.0)) != 0.0
        for value in values
    ):
        raise ValueError(
            "Calibration arrays and their native quantity specifications must be SI."
        )
    return values


def _quantity_record(quantity):
    return dict(
        name=quantity.name,
        quantity_kind=quantity.quantity_kind,
        quantity_id=quantity.quantity_id,
        unit_id=quantity.unit.unit_id,
        unit_symbol=quantity.unit.symbol,
        unit_reference=quantity.unit.reference_system_id,
        axes=quantity.axes,
        sign_convention=quantity.sign_convention,
        support_association=quantity.support_association,
        reference_configuration=quantity.reference_configuration,
    )


class SemiconductorCalibrationDomain(StrictModule):
    """Exact process revision, population, geometry set and control envelope."""

    process_revision: str = eqx.field(static=True)
    population: str = eqx.field(static=True)
    geometry_ids: tuple[str, ...] = eqx.field(static=True)
    controls: tuple[SemiconductorQuantitySpec, ...] = eqx.field(static=True)
    lower: Array
    upper: Array
    domain_id: str = eqx.field(static=True)

    def __init__(
        self, process_revision, population, geometry_ids, controls, lower, upper
    ):
        self.process_revision = _text(process_revision, "calibration identity")
        self.population = _text(population, "calibration identity")
        self.geometry_ids = _names(geometry_ids)
        self.controls = _quantities(controls)
        self.lower = _array(lower, (len(controls),))
        self.upper = _array(upper, self.lower.shape)
        if not bool(jnp.all(self.upper >= self.lower)):
            raise ValueError("Control-domain upper bounds must not precede lower bounds.")
        self.domain_id = canonical_fingerprint(self.to_record())

    def to_record(self):
        return dict(
            process_revision=self.process_revision,
            population=self.population,
            geometry_ids=self.geometry_ids,
            controls=tuple(_quantity_record(q) for q in self.controls),
            lower=self.lower.tolist(),
            upper=self.upper.tolist(),
        )


class SemiconductorMeasurementCase(StrictModule):
    """One independent correlated event, including its complete control history.

    ``controls`` has shape (history, control); ``observed`` is a flattened event
    with one quantity specification per entry. Covariances use flattened order
    and SI product units. De-embedding corrections have already been applied to
    observed values: its record and residual covariance are still mandatory.
    ``correlation_group`` identifies all events sharing unmodeled random errors;
    two cases cannot share it. Joint cross-error covariances must be reduced to
    this independent-source contract externally, with traceable provenance.
    """

    case_id: str = eqx.field(static=True)
    observation_ids: tuple[str, ...] = eqx.field(static=True)
    process_revision: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    lot: str = eqx.field(static=True)
    wafer: str = eqx.field(static=True)
    die: str = eqx.field(static=True)
    structure: str = eqx.field(static=True)
    condition_group: str = eqx.field(static=True)
    correlation_group: str = eqx.field(static=True)
    source_kind: str = eqx.field(static=True)
    source_uri: str = eqx.field(static=True)
    instrument_record: str = eqx.field(static=True)
    deembedding_record: str = eqx.field(static=True)
    uncertainty_assumptions: str = eqx.field(static=True)
    terminal_definitions: tuple[str, ...] = eqx.field(static=True)
    observables: tuple[SemiconductorQuantitySpec, ...] = eqx.field(static=True)
    reference: ReferenceArtifactManifest
    controls: Array
    observed: Array
    control_scale: Array
    observation_scale: Array
    control_covariance: Array
    measurement_covariance: Array
    deembedding_covariance: Array

    def __init__(
        self,
        *,
        case_id,
        observation_ids,
        process_revision,
        geometry_id,
        lot,
        wafer,
        die,
        structure,
        condition_group,
        correlation_group,
        source_kind,
        source_uri,
        instrument_record,
        deembedding_record,
        uncertainty_assumptions,
        terminal_definitions,
        observables,
        reference,
        controls,
        observed,
        control_scale,
        observation_scale,
        control_covariance,
        measurement_covariance,
        deembedding_covariance,
    ):
        self.case_id = _text(case_id, "calibration identity")
        self.observation_ids = _names(observation_ids)
        self.process_revision, self.geometry_id = (
            _text(process_revision, "calibration identity"),
            _text(geometry_id, "calibration identity"),
        )
        self.lot, self.wafer, self.die = (
            _text(lot, "calibration identity"),
            _text(wafer, "calibration identity"),
            _text(die, "calibration identity"),
        )
        self.structure, self.condition_group = (
            _text(structure, "calibration identity"),
            _text(condition_group, "calibration identity"),
        )
        self.correlation_group = _text(correlation_group, "calibration identity")
        if source_kind not in ("experimental", "synthetic", "external-reference"):
            raise ValueError("Measurement source kind must be explicit.")
        self.source_kind, self.source_uri = (
            source_kind,
            _text(source_uri, "calibration identity"),
        )
        self.instrument_record, self.deembedding_record = (
            _text(instrument_record, "calibration identity"),
            _text(deembedding_record, "calibration identity"),
        )
        self.uncertainty_assumptions = _text(
            uncertainty_assumptions, "calibration identity"
        )
        self.terminal_definitions = _names(terminal_definitions)
        self.observables = _quantities(observables)
        self.reference = reference
        reference.require_uncertainty()
        self.controls = _array(controls)
        if self.controls.ndim != 2 or not all(self.controls.shape):
            raise ValueError("Controls require a nonempty history by control array.")
        self.observed = _array(observed, (len(observation_ids),))
        if len(observables) != self.observed.size or self.observed.size > 256:
            raise ValueError(
                "Each joint event needs aligned quantities and at most 256 outputs; do not split correlations."
            )
        self.control_scale = _array(
            control_scale, (self.controls.shape[1],), positive=True
        )
        self.observation_scale = _array(
            observation_scale, self.observed.shape, positive=True
        )
        self.control_covariance = _array(
            control_covariance, (self.controls.size, self.controls.size)
        )
        shape = (self.observed.size, self.observed.size)
        self.measurement_covariance = _array(measurement_covariance, shape)
        self.deembedding_covariance = _array(deembedding_covariance, shape)
        # Native covariance admission preserves PSD sources; the likelihood below
        # separately requires positive definite total observation uncertainty.
        xscale = jnp.broadcast_to(self.control_scale, self.controls.shape).reshape(-1)
        DenseCovariance(self.control_covariance / (xscale[:, None] * xscale[None, :]))
        for covariance in (self.measurement_covariance, self.deembedding_covariance):
            DenseCovariance(
                covariance
                / (self.observation_scale[:, None] * self.observation_scale[None, :])
            )

    def group_key(self, axis):
        if axis == "lot":
            return (self.lot,)
        if axis == "wafer":
            return (self.lot, self.wafer)
        if axis == "die":
            return (self.lot, self.wafer, self.die)
        if axis == "structure":
            return (self.structure,)
        if axis == "condition":
            return (self.condition_group,)
        raise ValueError(f"Unknown held-out grouping axis {axis!r}.")

    def to_record(self):
        return dict(
            case_id=self.case_id,
            observation_ids=self.observation_ids,
            process_revision=self.process_revision,
            geometry_id=self.geometry_id,
            lot=self.lot,
            wafer=self.wafer,
            die=self.die,
            structure=self.structure,
            condition_group=self.condition_group,
            correlation_group=self.correlation_group,
            source_kind=self.source_kind,
            source_uri=self.source_uri,
            instrument_record=self.instrument_record,
            deembedding_record=self.deembedding_record,
            uncertainty_assumptions=self.uncertainty_assumptions,
            terminal_definitions=self.terminal_definitions,
            observables=tuple(_quantity_record(q) for q in self.observables),
            reference=self.reference.to_record(),
            arrays_digest=array_collection_digest(self.arrays()),
        )

    def arrays(self):
        return dict(
            controls=self.controls,
            observed=self.observed,
            control_scale=self.control_scale,
            observation_scale=self.observation_scale,
            control_covariance=self.control_covariance,
            measurement_covariance=self.measurement_covariance,
            deembedding_covariance=self.deembedding_covariance,
        )


class SemiconductorCalibrationCriteria(StrictModule):
    maximum_predictive_rms: float = eqx.field(static=True)
    minimum_marginal_coverage: float = eqx.field(static=True)
    coverage_standard_deviations: float = eqx.field(static=True)
    maximum_numerical_error_fraction: float = eqx.field(static=True)
    identifiability_relative_tolerance: float = eqx.field(static=True)
    minimum_heldout_groups: int = eqx.field(static=True)

    def __post_init__(self):
        positives = (
            self.maximum_predictive_rms,
            self.coverage_standard_deviations,
            self.maximum_numerical_error_fraction,
            self.identifiability_relative_tolerance,
        )
        if any(not math.isfinite(x) or x <= 0 for x in positives):
            raise ValueError(
                "Acceptance tolerances must be finite, positive and predeclared."
            )
        if not 0 < self.minimum_marginal_coverage <= 1 or self.minimum_heldout_groups < 1:
            raise ValueError(
                "Require positive coverage and independent held-out group count."
            )

    def to_record(self):
        return dict(
            maximum_predictive_rms=self.maximum_predictive_rms,
            minimum_marginal_coverage=self.minimum_marginal_coverage,
            coverage_standard_deviations=self.coverage_standard_deviations,
            maximum_numerical_error_fraction=self.maximum_numerical_error_fraction,
            identifiability_relative_tolerance=self.identifiability_relative_tolerance,
            minimum_heldout_groups=self.minimum_heldout_groups,
        )


class SemiconductorCalibrationCampaign(StrictModule):
    campaign_name: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    numerical_configuration: str = eqx.field(static=True)
    split_lock_record: str = eqx.field(static=True)
    training_ids: tuple[str, ...] = eqx.field(static=True)
    heldout_ids: tuple[str, ...] = eqx.field(static=True)
    holdout_axes: tuple[str, ...] = eqx.field(static=True)
    domain: SemiconductorCalibrationDomain
    cases: tuple[SemiconductorMeasurementCase, ...]
    criteria: SemiconductorCalibrationCriteria
    physical_references: tuple[tuple[str, ReferenceArtifactManifest], ...]
    campaign_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        campaign_name,
        model_id,
        source_revision,
        numerical_configuration,
        split_lock_record,
        training_ids,
        heldout_ids,
        holdout_axes,
        domain,
        cases,
        criteria,
        physical_references=(),
    ):
        self.campaign_name, self.model_id = (
            _text(campaign_name, "calibration identity"),
            _text(model_id, "calibration identity"),
        )
        self.source_revision = _text(source_revision, "calibration identity")
        self.numerical_configuration = _text(
            numerical_configuration, "calibration identity"
        )
        self.split_lock_record = _text(split_lock_record, "calibration identity")
        self.training_ids, self.heldout_ids = _names(training_ids), _names(heldout_ids)
        self.holdout_axes = _names(holdout_axes)
        if not set(holdout_axes) <= {"lot", "wafer", "die", "structure", "condition"}:
            raise ValueError("Unknown held-out grouping axis.")
        if not isinstance(cases, tuple) or not isinstance(physical_references, tuple):
            raise TypeError("Campaign inputs must be immutable tuples.")
        self.domain, self.cases, self.criteria = domain, cases, criteria
        self.physical_references = physical_references
        roles = tuple(role for role, _ in physical_references)
        if len(set(roles)) != len(roles) or not set(roles) <= {
            "geometry",
            "dopants",
            "materials",
            "contacts",
        }:
            raise ValueError(
                "Physical metrology roles must be unique geometry/dopants/materials/contacts."
            )
        ids = _names(tuple(case.case_id for case in cases))
        if set(training_ids) & set(heldout_ids) or set(ids) != set(training_ids) | set(
            heldout_ids
        ):
            raise ValueError(
                "Locked train/held-out split must partition all cases without leakage."
            )
        observations = tuple(o for case in cases for o in case.observation_ids)
        if len(set(observations)) != len(observations):
            raise ValueError("Observation reuse leaks information across campaign cases.")
        if len({c.correlation_group for c in cases}) != len(cases):
            raise ValueError(
                "Correlated events must be one joint case, not independent likelihood blocks."
            )
        for case in cases:
            if (
                case.process_revision != domain.process_revision
                or case.geometry_id not in domain.geometry_ids
            ):
                raise ValueError(
                    "Measurement process/geometry is outside the declared domain."
                )
            if case.controls.shape[1] != len(domain.controls) or not bool(
                jnp.all((case.controls >= domain.lower) & (case.controls <= domain.upper))
            ):
                raise ValueError(
                    "Measurement control history is outside the declared domain."
                )
        for axis in holdout_axes:
            train_keys = {c.group_key(axis) for c in self.training}
            heldout_keys = {c.group_key(axis) for c in self.heldout}
            if train_keys & heldout_keys:
                raise ValueError(
                    f"Held-out {axis} leakage: complete physical groups must be withheld."
                )
            if len(heldout_keys) < criteria.minimum_heldout_groups:
                raise ValueError(f"Insufficient independent held-out {axis} groups.")
        self.campaign_id = canonical_fingerprint(self.to_record())

    @property
    def training(self):
        by_id = {case.case_id: case for case in self.cases}
        return tuple(by_id[name] for name in self.training_ids)

    @property
    def heldout(self):
        by_id = {case.case_id: case for case in self.cases}
        return tuple(by_id[name] for name in self.heldout_ids)

    def to_record(self):
        return dict(
            campaign_name=self.campaign_name,
            model_id=self.model_id,
            source_revision=self.source_revision,
            numerical_configuration=self.numerical_configuration,
            split_lock_record=self.split_lock_record,
            training_ids=self.training_ids,
            heldout_ids=self.heldout_ids,
            holdout_axes=self.holdout_axes,
            domain=self.domain.to_record(),
            cases=tuple(c.to_record() for c in self.cases),
            criteria=self.criteria.to_record(),
            physical_references=tuple(
                (role, r.to_record()) for role, r in self.physical_references
            ),
        )


class SemiconductorParameterBinding(StrictModule):
    """Native prior/bijectors plus ordered physical names, SI units and scales.

    ``admissible`` is physical prior support, not a solver convergence test.
    ``prior_record`` identifies the actual prior and independent metrology;
    callable code is bound by the campaign's source revision, not introspection.
    """

    space: ParameterSpace
    quantities: tuple[SemiconductorQuantitySpec, ...] = eqx.field(static=True)
    sensitivity_scale: Array
    prior_record: str = eqx.field(static=True)
    admissible: Callable = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(self, space, quantities, sensitivity_scale, *, prior_record, admissible):
        self.space, self.quantities = space, _quantities(quantities)
        self.sensitivity_scale = _array(
            sensitivity_scale, (len(quantities),), positive=True
        )
        self.prior_record, self.admissible = (
            _text(prior_record, "calibration identity"),
            admissible,
        )
        physical = space.constrain(space.initial)
        if (
            not isinstance(physical, Array)
            or physical.shape != self.sensitivity_scale.shape
        ):
            raise ValueError(
                "The native ParameterSpace must bind one physical parameter vector."
            )
        if not bool(self.physically_admissible(physical)):
            raise ValueError(
                "Initial parameters are outside declared physical prior support."
            )
        self.binding_id = canonical_fingerprint(
            dict(
                quantities=tuple(q.quantity_id for q in quantities),
                sensitivity_scale=self.sensitivity_scale.tolist(),
                prior_record=prior_record,
                initial_physical=physical.tolist(),
            )
        )

    def physically_admissible(self, physical):
        value = jnp.asarray(physical)
        if value.shape != self.sensitivity_scale.shape:
            raise ValueError("Physical parameter shape changed.")
        admitted = jnp.asarray(self.admissible(value), dtype=bool)
        if admitted.shape:
            raise ValueError("Physical admissibility must return one scalar boolean.")
        return (
            jnp.all(jnp.isfinite(value))
            & admitted
            & jnp.isfinite(self.space.log_prior(value))
        )


class SemiconductorForwardResult(StrictModule):
    """Raw solver evidence. Invalid predictions are candidates, never observations."""

    prediction: Array
    numerical_valid: Array
    numerical_error: Array
    solver_status: Array
    evidence_id: str = eqx.field(static=True)


class SemiconductorForwardEvaluation(StrictModule):
    physical_parameters: Array
    cases: tuple[SemiconductorForwardResult, ...]
    case_ids: tuple[str, ...] = eqx.field(static=True)
    status: str = eqx.field(static=True)

    @property
    def predictions(self):
        if self.status != "valid":
            raise RuntimeError(f"No qualified predictions: {self.status}.")
        return tuple(case.prediction for case in self.cases)


class SemiconductorForwardUnresolved(RuntimeError):
    """Host failure retaining the actual physical point and rejected solver evidence."""

    def __init__(self, evaluation):
        self.evaluation = evaluation
        super().__init__(
            f"Semiconductor forward evaluation is {evaluation.status}; inference must stop."
        )


def _forward_valid(result, case, criteria):
    if (
        result.prediction.shape != case.observed.shape
        or result.numerical_error.shape != case.observed.shape
    ):
        raise ValueError(
            "Forward predictions and numerical error bounds must match the measured event."
        )
    if jnp.shape(result.numerical_valid) or jnp.shape(result.solver_status):
        raise ValueError("Forward validity/status must be scalar.")
    _text(result.evidence_id, "calibration identity")
    sigma = jnp.sqrt(jnp.diag(case.measurement_covariance + case.deembedding_covariance))
    return (
        result.numerical_valid
        & jnp.all(jnp.isfinite(result.prediction))
        & jnp.all(jnp.isfinite(result.numerical_error) & (result.numerical_error >= 0))
        & jnp.all(
            result.numerical_error <= criteria.maximum_numerical_error_fraction * sigma
        )
    )


def evaluate_semiconductor_forward(prepared, physical, *, heldout=False):
    """Host audit: physical exclusion and unresolved numerics have different statuses."""
    cases = prepared.campaign.heldout if heldout else prepared.campaign.training
    physical = jnp.asarray(physical)
    if not bool(prepared.binding.physically_admissible(physical)):
        return SemiconductorForwardEvaluation(
            physical, (), tuple(c.case_id for c in cases), "physical-inadmissible"
        )
    results = tuple(prepared.forward(physical, case) for case in cases)
    valid = all(
        bool(_forward_valid(r, c, prepared.campaign.criteria))
        for r, c in zip(results, cases, strict=True)
    )
    return SemiconductorForwardEvaluation(
        physical,
        results,
        tuple(c.case_id for c in cases),
        "valid" if valid else "numerically-unresolved",
    )


class PreparedSemiconductorCalibration(StrictModule):
    campaign: SemiconductorCalibrationCampaign
    binding: SemiconductorParameterBinding
    forward: Callable = eqx.field(static=True)
    likelihoods: tuple[LinearizedGaussianMeasurementLikelihood, ...]
    posterior: PosteriorProblem


def _prediction(forward, campaign, physical, case):
    result = forward(physical, case)
    return eqx.error_if(
        result.prediction,
        ~_forward_valid(result, case, campaign.criteria),
        f"Numerically unresolved semiconductor forward: {case.case_id}; not zero physical likelihood.",
    )


def _normalized_prediction(forward, campaign, physical, case, controls):
    physical_case = eqx.tree_at(lambda c: c.controls, case, controls * case.control_scale)
    return (
        _prediction(forward, campaign, physical, physical_case) / case.observation_scale
    )


def prepare_semiconductor_calibration(
    campaign, binding, forward, *, commercial_use=False
):
    """Prepare normalized native likelihoods using training data only.

    The complete event preserves bias/frequency/time correlations. Invalid native
    likelihood arithmetic raises; neither a failed solver nor covariance failure
    is interpreted as zero physical likelihood. No rejection-based solver bias is
    admitted. An inadmissible physical point is excluded before forward execution.
    """
    for case in campaign.cases:
        case.reference.require_rights(
            training_use=case.case_id in campaign.training_ids,
            commercial_use=commercial_use,
        )
    for _, reference in campaign.physical_references:
        reference.require_rights(training_use=True, commercial_use=commercial_use)
        reference.require_uncertainty()
    terms = []
    for case in campaign.cases:
        xscale = jnp.broadcast_to(case.control_scale, case.controls.shape).reshape(-1)
        yscale = case.observation_scale
        predict = lambda p, x, case=case: _normalized_prediction(
            forward, campaign, p, case, x
        )
        term = LinearizedGaussianMeasurementLikelihood(
            predict,
            (case.controls / case.control_scale)[None, ...],
            (case.observed / yscale)[None, ...],
            input_covariance=case.control_covariance
            / (xscale[:, None] * xscale[None, :]),
            observation_covariance=(
                case.measurement_covariance + case.deembedding_covariance
            )
            / (yscale[:, None] * yscale[None, :]),
            label=case.case_id,
        )
        if case.case_id in campaign.training_ids:
            terms.append(term)
    likelihoods = tuple(terms)

    def log_likelihood(physical):
        def admitted(p):
            value = sum(
                (jnp.sum(term.per_case_log_prob(p)) for term in likelihoods),
                jnp.zeros(()),
            )
            return eqx.error_if(
                value,
                ~jnp.isfinite(value),
                "Unresolved semiconductor measurement likelihood; not physical prior exclusion.",
            )

        return jax.lax.cond(
            binding.physically_admissible(physical),
            admitted,
            lambda p: jnp.asarray(-jnp.inf, dtype=p.dtype),
            physical,
        )

    def predict(physical, case):
        physical = eqx.error_if(
            physical,
            ~binding.physically_admissible(physical),
            "Prediction parameters are outside physical prior support.",
        )
        return _prediction(forward, campaign, physical, case)

    posterior = PosteriorProblem(binding.space, log_likelihood, predict=predict)
    prepared = PreparedSemiconductorCalibration(
        campaign, binding, forward, likelihoods, posterior
    )
    evaluation = evaluate_semiconductor_forward(
        prepared, binding.space.constrain(binding.space.initial)
    )
    if evaluation.status != "valid":
        raise SemiconductorForwardUnresolved(evaluation)
    return prepared


def _effective_covariance(prepared, physical, case):
    """SI covariance, including the same linearized measured-control errors as UQ."""
    derivative = jax.jacfwd(
        lambda x: _prediction(
            prepared.forward,
            prepared.campaign,
            physical,
            eqx.tree_at(lambda c: c.controls, case, x.reshape(case.controls.shape)),
        )
    )(case.controls.reshape(-1))
    pushed = contract(
        "ai,ij,bj->ab", derivative, case.control_covariance, derivative, backend="jax"
    )
    return case.measurement_covariance + case.deembedding_covariance + pushed


def _spectrum(matrix):
    result = svd(
        SVDProblem(DenseLinearOperator(matrix)),
        policy=SVDSolvePolicy(count=min(matrix.shape)),
    )
    if not bool(result.successful):
        raise RuntimeError(
            "Native semiconductor sensitivity/covariance SVD is unresolved."
        )
    return result


def _whiten(covariance, values):
    spectrum = _spectrum(covariance)
    if not bool(jnp.all(spectrum.singular_values > 0)):
        raise ValueError("Qualification covariance must be positive definite.")
    # Native singular vectors have a trailing mode axis.
    projected = contract("ji,j...->i...", spectrum.left_vectors, values, backend="jax")
    shape = (-1,) + (1,) * (values.ndim - 1)
    return projected / jnp.sqrt(spectrum.singular_values).reshape(shape)


class SemiconductorIdentifiability(StrictModule):
    singular_values: Array
    physical_sensitivity_scale: Array
    rank: int = eqx.field(static=True)
    parameter_count: int = eqx.field(static=True)

    @property
    def locally_identifiable(self):
        return self.rank == self.parameter_count


def semiconductor_identifiability(prepared, physical):
    """Likelihood-only local mean sensitivity, not global identifiability.

    Whitening uses the full covariance frozen at the reported point. Prior
    curvature never creates identifiable directions. Covariance-parameter-only
    information is deliberately not counted as mean-response identification.
    """
    evaluation = evaluate_semiconductor_forward(prepared, physical)
    if evaluation.status != "valid":
        raise SemiconductorForwardUnresolved(evaluation)
    rows = []
    for case in prepared.campaign.training:
        jacobian = jax.jacfwd(
            lambda p, case=case: _prediction(prepared.forward, prepared.campaign, p, case)
        )(physical)
        scaled = (
            jacobian
            * prepared.binding.sensitivity_scale
            / case.observation_scale[:, None]
        )
        covariance = _effective_covariance(prepared, physical, case)
        covariance = covariance / (
            case.observation_scale[:, None] * case.observation_scale[None, :]
        )
        rows.append(_whiten(covariance, scaled))
    spectrum = _spectrum(jnp.concatenate(rows, axis=0))
    tolerance = prepared.campaign.criteria.identifiability_relative_tolerance
    rank = int(
        jnp.sum(spectrum.singular_values > tolerance * spectrum.singular_values[0])
    )
    return SemiconductorIdentifiability(
        spectrum.singular_values,
        prepared.binding.sensitivity_scale,
        rank,
        len(prepared.binding.quantities),
    )


class SemiconductorPredictiveCheck(StrictModule):
    case_id: str = eqx.field(static=True)
    prediction: Array
    parameter_covariance: Array
    observation_covariance: Array
    standardized_residual: Array
    correlated_rms: Array
    marginal_coverage: Array
    numerical_result: SemiconductorForwardResult


class SemiconductorCalibrationResult(StrictModule):
    prepared: PreparedSemiconductorCalibration
    optimization: MAPResult
    posterior: LaplaceResult
    identifiability: SemiconductorIdentifiability
    training_evaluation: SemiconductorForwardEvaluation
    heldout: tuple[SemiconductorPredictiveCheck, ...]


def calibrate_semiconductor(prepared, *, max_steps=500, gradient_tolerance=1e-6):
    """Native MAP, undamped exact Laplace and untouched held-out prediction.

    This is a local Gaussian posterior approximation, not a global or MCMC
    claim. Native curvature/stationarity failures stop this route rather than
    manufacturing uncertainty. The caller can use ``prepared.posterior`` with
    native sampling after establishing forward solvability and diagnostics.
    """
    optimization = find_map(
        prepared.posterior, max_steps=max_steps, gradient_tolerance=gradient_tolerance
    )
    physical = optimization.parameters
    training = evaluate_semiconductor_forward(prepared, physical)
    heldout = evaluate_semiconductor_forward(prepared, physical, heldout=True)
    for evaluation in (training, heldout):
        if evaluation.status != "valid":
            raise SemiconductorForwardUnresolved(evaluation)
    identifiability = semiconductor_identifiability(prepared, physical)
    posterior = fit_laplace(
        prepared.posterior,
        optimization.position,
        stationarity_tolerance=gradient_tolerance,
    )
    checks = []
    for case, numerical in zip(prepared.campaign.heldout, heldout.cases, strict=True):
        propagation = propagate_linearized(
            lambda z, case=case: prepared.posterior.predict(z, case),
            optimization.position,
            DenseCovariance(posterior.covariance),
            source="epistemic",
        )
        parameter_covariance = propagation.materialize_covariance(
            max_dimension=256
        ).matrix
        observation_covariance = _effective_covariance(prepared, physical, case)
        total = parameter_covariance + observation_covariance
        residual = propagation.mean - case.observed
        scale = case.observation_scale
        standardized = _whiten(
            total / (scale[:, None] * scale[None, :]), residual / scale
        )
        coverage = jnp.mean(
            jnp.abs(residual)
            <= prepared.campaign.criteria.coverage_standard_deviations
            * jnp.sqrt(jnp.diag(total))
        )
        checks.append(
            SemiconductorPredictiveCheck(
                case.case_id,
                propagation.mean,
                parameter_covariance,
                observation_covariance,
                standardized,
                jnp.sqrt(jnp.mean(standardized**2)),
                coverage,
                numerical,
            )
        )
    return SemiconductorCalibrationResult(
        prepared, optimization, posterior, identifiability, training, tuple(checks)
    )


class SemiconductorCalibrationQualification(StrictModule):
    support: SupportTuple
    numerical: QualificationCoverageReport
    parameter_evidence: QualificationCoverageReport
    heldout_evidence: QualificationCoverageReport
    evidence: tuple[QualificationEvidence, ...]
    empirical_gates: tuple[str, ...] = eqx.field(static=True)
    campaign_id: str = eqx.field(static=True)

    @property
    def numerically_qualified(self):
        return self.numerical.passed

    @property
    def empirically_calibrated(self):
        return self.numerically_qualified and not self.empirical_gates


def qualify_semiconductor_calibration(result, evidence, *, at_time):
    """Fail-closed, exact-domain numerical and named-process empirical claims.

    Numerical proof requires independently reviewed conservation, refinement and
    solver evidence for this model/build. Parameter claims additionally require
    campaign-bound global-identifiability/metrology evidence; full local rank
    alone is insufficient. Experimental data rights were checked before fitting.
    """
    campaign = result.prepared.campaign
    numerical = QualificationMatrix(
        {
            name: dict(
                evidence_kind="scientific",
                subject_id=campaign.model_id,
                build_id=campaign.source_revision,
                criterion_id=name,
                replay_id=campaign.numerical_configuration,
            )
            for name in (
                "semiconductor-conservation",
                "semiconductor-refinement",
                "semiconductor-solver",
            )
        }
    ).evaluate(evidence, at_time=at_time)
    parameter = QualificationMatrix(
        {
            "physical-identification": dict(
                evidence_kind="scientific",
                subject_id=campaign.campaign_id,
                build_id=campaign.source_revision,
                criterion_id="semiconductor-global-identifiability",
                raw_artifact_id=result.prepared.binding.binding_id,
            )
        }
    ).evaluate(evidence, at_time=at_time)
    heldout = QualificationMatrix(
        {
            "independent-heldout": dict(
                evidence_kind="scientific",
                subject_id=campaign.campaign_id,
                build_id=campaign.source_revision,
                criterion_id="semiconductor-independent-heldout",
                raw_artifact_id=campaign.split_lock_record,
            )
        }
    ).evaluate(evidence, at_time=at_time)
    gates = []
    if not result.identifiability.locally_identifiable:
        gates.append(
            "Likelihood mean response is locally rank deficient; prior-constrained directions are not identified."
        )
    if not parameter.passed:
        gates.append(
            "Independent global identifiability or separating physical metrology evidence is required."
        )
    if not heldout.passed:
        gates.append(
            "Independent review of the locked held-out acquisition groups and untouched validation is required."
        )
    if any(c.source_kind != "experimental" for c in campaign.cases):
        gates.append(
            "Authorized experimental training and independent held-out measurements "
            "are required; synthetic/reference recovery is not process calibration."
        )
    for role in ("geometry", "dopants", "materials", "contacts"):
        if role not in {name for name, _ in campaign.physical_references}:
            gates.append(
                f"Independent {role} metrology with provenance and uncertainty is required."
            )
    measured_controls = jnp.concatenate(tuple(c.controls for c in campaign.cases), axis=0)
    if bool(
        jnp.any(jnp.min(measured_controls, axis=0) > campaign.domain.lower)
        | jnp.any(jnp.max(measured_controls, axis=0) < campaign.domain.upper)
    ):
        gates.append(
            "Measurements do not span the declared control envelope; extrapolated "
            "ranges are not empirically calibrated."
        )
    if {c.geometry_id for c in campaign.cases} != set(campaign.domain.geometry_ids):
        gates.append(
            "The declared geometry domain includes structures without measurement evidence."
        )
    for check in result.heldout:
        if (
            not bool(jnp.isfinite(check.correlated_rms))
            or float(check.correlated_rms) > campaign.criteria.maximum_predictive_rms
        ):
            gates.append(f"Held-out predictive correlated RMS failed: {check.case_id}.")
        if (
            not bool(jnp.isfinite(check.marginal_coverage))
            or float(check.marginal_coverage)
            < campaign.criteria.minimum_marginal_coverage
        ):
            gates.append(
                f"Held-out marginal predictive coverage failed: {check.case_id}."
            )
    support = SupportTuple(
        "semiconductor.device-calibration",
        dict(
            process_revision=campaign.domain.process_revision,
            domain_id=campaign.domain.domain_id,
            model_id=campaign.model_id,
            campaign_id=campaign.campaign_id,
            parameter_binding=result.prepared.binding.binding_id,
            posterior="local-laplace",
        ),
    )
    return SemiconductorCalibrationQualification(
        support,
        numerical,
        parameter,
        heldout,
        tuple(evidence),
        tuple(gates),
        campaign.campaign_id,
    )


def archive_semiconductor_calibration(
    path,
    result,
    qualification,
    *,
    producer_version,
    license_id,
    redistribution=False,
    export=False,
):
    """Write native checksum-verified array archive and artifact provenance.

    Raw numerical candidates/statuses, observations, covariance sources, prior,
    locked split and empirical gaps survive archival. Publication rights are
    checked separately from local inference rights.
    """
    campaign = result.prepared.campaign
    if qualification.campaign_id != campaign.campaign_id:
        raise ValueError("Qualification belongs to a different campaign.")
    for reference in (
        *(r for _, r in campaign.physical_references),
        *(c.reference for c in campaign.cases),
    ):
        reference.require_rights(redistribution=redistribution, export=export)
    arrays = dict(
        physical_parameters=result.optimization.parameters,
        unconstrained_parameters=result.optimization.position,
        unconstrained_posterior_covariance=result.posterior.covariance,
        physical_posterior_covariance=result.posterior.physical_covariance(),
        likelihood_singular_values=result.identifiability.singular_values,
        objective_history=result.optimization.objective_history,
    )
    for case in campaign.cases:
        arrays.update(
            {
                f"measurement/{case.case_id}/{name}": value
                for name, value in case.arrays().items()
            }
        )
    for case_id, numerical in zip(
        result.training_evaluation.case_ids, result.training_evaluation.cases, strict=True
    ):
        arrays.update(
            {
                f"training/{case_id}/prediction": numerical.prediction,
                f"training/{case_id}/numerical_error": numerical.numerical_error,
                f"training/{case_id}/numerical_valid": numerical.numerical_valid,
                f"training/{case_id}/solver_status": numerical.solver_status,
            }
        )
    for check in result.heldout:
        prefix = f"heldout/{check.case_id}"
        arrays.update(
            {
                f"{prefix}/prediction": check.prediction,
                f"{prefix}/parameter_covariance": check.parameter_covariance,
                f"{prefix}/observation_covariance": check.observation_covariance,
                f"{prefix}/standardized_residual": check.standardized_residual,
                f"{prefix}/numerical_error": check.numerical_result.numerical_error,
                f"{prefix}/numerical_valid": check.numerical_result.numerical_valid,
                f"{prefix}/solver_status": check.numerical_result.solver_status,
            }
        )
    record = dict(
        campaign=campaign.to_record(),
        campaign_id=campaign.campaign_id,
        parameter_binding=result.prepared.binding.binding_id,
        prior_record=result.prepared.binding.prior_record,
        parameter_quantities=tuple(
            _quantity_record(q) for q in result.prepared.binding.quantities
        ),
        support=qualification.support.to_record(),
        numerical=qualification.numerical.to_record(),
        parameter_evidence=qualification.parameter_evidence.to_record(),
        heldout_evidence=qualification.heldout_evidence.to_record(),
        empirical_gates=qualification.empirical_gates,
        empirically_calibrated=qualification.empirically_calibrated,
        evidence=tuple(e.to_record() for e in qualification.evidence),
        training_forward_evidence=tuple(
            r.evidence_id for r in result.training_evaluation.cases
        ),
        heldout_forward_evidence=tuple(
            c.numerical_result.evidence_id for c in result.heldout
        ),
        array_digest=array_collection_digest(arrays),
    )
    envelope = ScientificArtifactEnvelope(
        artifact_kind="semiconductor-calibration",
        content_digest=canonical_fingerprint(record),
        producer="phydrax",
        producer_version=producer_version,
        build_id=campaign.source_revision,
        license_id=license_id,
        resource_id=campaign.domain.domain_id,
        status="complete",
        parent_artifact_ids=tuple(c.reference.manifest_id for c in campaign.cases),
    )
    record["artifact_id"] = envelope.artifact_id
    write_array_archive(path, manifest=record, arrays=arrays)
    return Path(path), envelope


def archive_semiconductor_forward_evaluation(
    path, campaign, evaluation, *, producer_version, license_id
):
    """Retain a rejected native solve without assigning it a physical likelihood.

    This local audit archive does not grant redistribution/export rights and
    does not represent the raw candidates as qualified predictions.
    """
    if set(evaluation.case_ids) not in (
        set(campaign.training_ids),
        set(campaign.heldout_ids),
    ):
        raise ValueError("Forward evaluation does not belong to a campaign split.")
    expected = (
        0 if evaluation.status == "physical-inadmissible" else len(evaluation.case_ids)
    )
    if (
        evaluation.status
        not in ("valid", "physical-inadmissible", "numerically-unresolved")
        or len(evaluation.cases) != expected
    ):
        raise ValueError(
            "Forward audit status and retained case evidence are inconsistent."
        )
    arrays = {"physical_parameters": evaluation.physical_parameters}
    for case_id, result in zip(evaluation.case_ids, evaluation.cases):
        arrays.update(
            {
                f"{case_id}/raw_candidate": result.prediction,
                f"{case_id}/numerical_error": result.numerical_error,
                f"{case_id}/numerical_valid": result.numerical_valid,
                f"{case_id}/solver_status": result.solver_status,
            }
        )
    record = dict(
        campaign_id=campaign.campaign_id,
        campaign=campaign.to_record(),
        status=evaluation.status,
        case_ids=evaluation.case_ids,
        forward_evidence=tuple(r.evidence_id for r in evaluation.cases),
        array_digest=array_collection_digest(arrays),
    )
    valid = evaluation.status == "valid"
    envelope = ScientificArtifactEnvelope(
        artifact_kind="semiconductor-forward-evaluation",
        content_digest=canonical_fingerprint(record),
        producer="phydrax",
        producer_version=producer_version,
        build_id=campaign.source_revision,
        license_id=license_id,
        resource_id=campaign.domain.domain_id,
        status="complete" if valid else "failed",
        failure_reason="none" if valid else evaluation.status,
        parent_artifact_ids=tuple(c.reference.manifest_id for c in campaign.cases),
    )
    record["artifact_id"] = envelope.artifact_id
    write_array_archive(path, manifest=record, arrays=arrays)
    return Path(path), envelope
