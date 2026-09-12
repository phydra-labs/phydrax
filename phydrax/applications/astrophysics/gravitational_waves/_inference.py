#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ....uq import (
    AbstractPosteriorTerm,
    NestedProposalPlan,
    NestedSamplingCapacity,
    NestedSamplingPlan,
    PosteriorProblem,
)
from ....uq._result_context import UQResultContext
from ._likelihood import (
    AbstractGravitationalWaveLikelihood,
    GravitationalWavePosteriorTerm,
)
from ._marginalization import (
    GravitationalWaveMarginalizationPlan,
    GravitationalWaveMarginalizedPosteriorTerm,
)
from ._parameters import GravitationalWaveParameterPlan


class PreparedGravitationalWaveInference(StrictModule):
    """Native posterior and nested topology bound to one exact likelihood."""

    parameters: GravitationalWaveParameterPlan
    likelihood: AbstractGravitationalWaveLikelihood
    term: AbstractPosteriorTerm
    posterior: PosteriorProblem
    nested_sampling_plan: NestedSamplingPlan
    result_context: UQResultContext
    preparation_id: str = eqx.field(static=True)


def prepare_gravitational_wave_inference(
    parameters: GravitationalWaveParameterPlan,
    likelihood: AbstractGravitationalWaveLikelihood,
    /,
    *,
    analysis_id: str,
    nested_capacity: NestedSamplingCapacity,
    initial_live: int,
    nested_proposals: NestedProposalPlan | None = None,
    marginalization: GravitationalWaveMarginalizationPlan | None = None,
) -> PreparedGravitationalWaveInference:
    if not isinstance(parameters, GravitationalWaveParameterPlan):
        raise TypeError("parameters must be GravitationalWaveParameterPlan.")
    if not isinstance(likelihood, AbstractGravitationalWaveLikelihood):
        raise TypeError("likelihood must implement AbstractGravitationalWaveLikelihood.")
    if not isinstance(nested_capacity, NestedSamplingCapacity):
        raise TypeError("nested_capacity must be NestedSamplingCapacity.")
    if marginalization is not None and not isinstance(
        marginalization, GravitationalWaveMarginalizationPlan
    ):
        raise TypeError(
            "marginalization must be GravitationalWaveMarginalizationPlan or None."
        )
    if (
        parameters.parameterization_id
        != likelihood.waveform.capabilities.parameterization_id
    ):
        raise ValueError("Parameter and waveform parameterization identities disagree.")
    if marginalization is not None and marginalization.likelihood is not likelihood:
        raise ValueError("Marginalization belongs to a different likelihood.")
    term: AbstractPosteriorTerm = (
        GravitationalWavePosteriorTerm(likelihood)
        if marginalization is None
        else GravitationalWaveMarginalizedPosteriorTerm(marginalization)
    )
    prediction = (
        (lambda physical: likelihood.detector_signal(physical)[0])
        if marginalization is None
        else None
    )
    posterior = PosteriorProblem.from_terms(
        parameters.parameter_space,
        (term,),
        predict=prediction,
    )
    proposal = (
        NestedProposalPlan(
            discrete_gibbs=bool(parameters.nested_prior.finite_supports),
            periodic_slice=bool(parameters.nested_prior.periodic),
        )
        if nested_proposals is None
        else nested_proposals
    )
    nested = NestedSamplingPlan(
        nested_capacity,
        parameters.nested_prior,
        proposal,
        initial_live=initial_live,
    )
    problem_id = canonical_fingerprint(
        {
            "kind": "gravitational-wave-posterior-problem",
            "parameters": parameters.plan_id,
            "likelihood": likelihood.likelihood_id,
            "marginalization": (
                None if marginalization is None else marginalization.plan_id
            ),
        }
    )
    context = UQResultContext(
        analysis_id=analysis_id,
        problem_id=problem_id,
        likelihood_id=likelihood.likelihood_id,
        parameterization_id=parameters.parameterization_id,
        data_ids=likelihood.network.data_ids,
        provider_ids=(likelihood.waveform.waveform_id,),
        approximation_id=likelihood.approximation_id,
        normalization="absolute",
    )
    preparation_id = canonical_fingerprint(
        {
            "kind": "prepared-gravitational-wave-inference",
            "parameters": parameters.plan_id,
            "likelihood": likelihood.likelihood_id,
            "nested": nested.plan_id,
            "marginalization": (
                None if marginalization is None else marginalization.plan_id
            ),
        }
    )
    return PreparedGravitationalWaveInference(
        parameters,
        likelihood,
        term,
        posterior,
        nested,
        context,
        preparation_id,
    )


__all__ = ["PreparedGravitationalWaveInference", "prepare_gravitational_wave_inference"]
