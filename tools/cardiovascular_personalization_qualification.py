#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Manufactured qualification cases for cardiovascular personalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp

from phydrax._likelihoods import GaussianLikelihood
from phydrax.applications.cardiovascular._quantities import CardiovascularQuantitySpec
from phydrax.applications.cardiovascular.personalization._design import (
    check_directional_derivative,
    ExperimentDesignCandidate,
    ExperimentDesignPlan,
    ForwardAdjointEvidence,
    SensitivitySVDPlan,
)
from phydrax.applications.cardiovascular.personalization._inverse import (
    ElectrophysiologyInverseProblem,
)
from phydrax.applications.cardiovascular.personalization._likelihood import (
    ModalityLikelihoodChannel,
    ModalityObservation,
    MultimodalLikelihoodPlan,
)
from phydrax.applications.cardiovascular.personalization._parameters import (
    CardiacParameterSchema,
    CardiacParameterSpec,
    CardiacParameterSupport,
    CardiacSubsystem,
    ParameterIdentifiability,
)
from phydrax.optim import OptimizationTermination, ReducedAdjoint
from phydrax.units import ONE
from phydrax.uq import IdentityBijector, Normal, SigmoidIntervalBijector, Uniform


def _schema() -> CardiacParameterSchema:
    quantity = CardiovascularQuantitySpec("activation_scale", "strain", ONE)
    return CardiacParameterSchema(
        (
            CardiacParameterSpec(
                "activation_scale",
                quantity,
                SigmoidIntervalBijector(0.1, 4.0),
                CardiacParameterSupport(0.1, 4.0),
                Uniform(0.1, 4.0),
                CardiacSubsystem.ELECTROPHYSIOLOGY,
            ),
            CardiacParameterSpec(
                "fixed_reference",
                CardiovascularQuantitySpec("fixed_reference", "strain", ONE),
                IdentityBijector(),
                CardiacParameterSupport(-10.0, 10.0),
                Normal(0.0, 1.0),
                CardiacSubsystem.ELECTROPHYSIOLOGY,
                identifiability=ParameterIdentifiability.FIXED,
            ),
        ),
        schema_id="qualification.ep.activation-scale",
    )


def _activation_likelihood(target: float, record_id: str):
    observation = ModalityObservation(
        record_id,
        "activation_time",
        jnp.asarray([target]),
        jnp.asarray([True]),
        "time",
        "ms",
    )
    return ModalityLikelihoodChannel(observation, likelihood=GaussianLikelihood(0.1))


def qualify_synthetic_recovery() -> dict[str, object]:
    channel = _activation_likelihood(2.0, "qualification-activation")
    likelihood = MultimodalLikelihoodPlan((channel,)).prepare()
    inverse = ElectrophysiologyInverseProblem(
        _schema(),
        likelihood,
        lambda state, physical, args: state - physical[0],
        lambda state, physical, args: (state.reshape((1,)),),
        fixed_topology=lambda state, physical, args: jnp.asarray(True),
        problem_id="qualification-ep-recovery",
    )
    result = inverse.solve_multistart(
        jnp.asarray([1.0]),
        (
            (jnp.asarray(0.4), jnp.asarray(1.25)),
            (jnp.asarray(3.6), jnp.asarray(1.25)),
        ),
        method=ReducedAdjoint(),
        termination=OptimizationTermination(
            absolute_optimality=2.0e-5,
            relative_optimality=0.0,
            maximum_steps=80,
        ),
    )
    estimate = float(result.best.physical_parameters[0])
    error = abs(estimate - 2.0)
    fixed_reference = float(result.best.physical_parameters[1])
    passed = bool(
        result.successful
        and result.best.evidence.state_accepted
        and result.best.evidence.adjoint_accepted
        and error <= 3.0e-3
        and fixed_reference == 1.25
        and len(result.best.state_design.design) == 1
    )
    return {
        "case": "manufactured-activation-scale-recovery",
        "problem_id": inverse.problem_id,
        "start_count": len(result.results),
        "accepted_starts": int(jnp.sum(result.accepted)),
        "best_index": int(result.best_index),
        "estimate": estimate,
        "absolute_error": error,
        "fixed_reference": fixed_reference,
        "optimizer_coordinate_count": len(result.best.state_design.design),
        "state_accepted": bool(result.best.evidence.state_accepted),
        "adjoint_accepted": bool(result.best.evidence.adjoint_accepted),
        "passed": passed,
    }


def qualify_held_out_modality() -> dict[str, object]:
    activation = _activation_likelihood(2.0, "qualification-activation-heldout")
    strain_observation = ModalityObservation(
        "qualification-strain-heldout",
        "strain",
        jnp.asarray([0.3, 0.5]),
        jnp.asarray([True, True]),
        "strain",
        "1",
    )
    strain = ModalityLikelihoodChannel(
        strain_observation,
        likelihood=GaussianLikelihood(0.05),
    )
    plan = MultimodalLikelihoodPlan((activation, strain))
    full = plan.prepare().evaluate((jnp.asarray([2.0]), jnp.asarray([0.3, 0.5])))
    held_plan = plan.held_out(("strain",))
    held = held_plan.prepare().evaluate((jnp.asarray([2.0]),))
    expected = full.channel_results[0].log_likelihood
    difference = float(jnp.abs(held.log_likelihood - expected))
    passed = bool(
        full.successful
        and held.successful
        and difference <= 1.0e-7
        and held_plan.plan_id != plan.plan_id
    )
    return {
        "case": "held-out-strain-modality",
        "full_plan_id": plan.plan_id,
        "held_out_plan_id": held_plan.plan_id,
        "retained_log_likelihood_error": difference,
        "passed": passed,
    }


def qualify_identifiability_and_design() -> dict[str, object]:
    forward = lambda parameters, args: jnp.asarray(
        [parameters[0] + parameters[1], 2.0 * (parameters[0] + parameters[1])]
    )
    sensitivity = SensitivitySVDPlan(
        forward,
        jnp.ones(2),
        jnp.ones(2),
        relative_rank_tolerance=1.0e-7,
    ).evaluate(jnp.asarray([1.0, 1.0]))
    null_image = sensitivity.scaled_jacobian @ sensitivity.nullspace_basis[:, 0]
    null_residual = float(jnp.sqrt(jnp.sum(null_image * null_image)))
    derivative = check_directional_derivative(
        lambda values: jnp.sum(jnp.sin(values) ** 2),
        jnp.asarray([0.2, -0.4]),
        jnp.asarray([1.0, 0.5]),
        step=2.0e-3,
        relative_tolerance=2.0e-4,
    )
    accepted = ForwardAdjointEvidence(True, True, True, True)
    rejected = ForwardAdjointEvidence(True, False, True, True)
    candidates = (
        ExperimentDesignCandidate(
            "orthogonal-x", jnp.asarray([[1.0, 0.0]]), jnp.eye(1), accepted
        ),
        ExperimentDesignCandidate(
            "orthogonal-y", jnp.asarray([[0.0, 1.0]]), jnp.eye(1), accepted
        ),
        ExperimentDesignCandidate(
            "uncertified-high-gain",
            jnp.asarray([[20.0, 20.0]]),
            jnp.eye(1),
            rejected,
        ),
    )
    design = (
        ExperimentDesignPlan(
            candidates,
            0.1 * jnp.eye(2),
            maximum_experiments=2,
            budget=2.0,
        )
        .prepare()
        .select()
    )
    passed = bool(
        sensitivity.successful
        and sensitivity.confounded
        and int(sensitivity.rank) == 1
        and null_residual <= 2.0e-6
        and derivative.accepted
        and design.successful
        and int(design.selected_count) == 2
        and not bool(design.selected_mask[2])
    )
    return {
        "case": "local-confounding-derivative-and-design",
        "sensitivity_rank": int(sensitivity.rank),
        "sensitivity_successful": bool(sensitivity.successful),
        "sensitivity_confounded": bool(sensitivity.confounded),
        "sensitivity_nullity": int(sensitivity.nullity),
        "nullspace_residual_norm": null_residual,
        "directional_derivative_relative_error": float(derivative.relative_error),
        "directional_derivative_accepted": bool(derivative.accepted),
        "experiment_design_successful": bool(design.successful),
        "selected_candidate_count": int(design.selected_count),
        "selected_candidate_indices": [
            int(value)
            for value in design.selected_indices[: int(design.selected_count)].tolist()
        ],
        "uncertified_candidate_rejected": not bool(design.selected_mask[2]),
        "passed": passed,
    }


def qualification() -> dict[str, object]:
    cases = (
        qualify_synthetic_recovery(),
        qualify_held_out_modality(),
        qualify_identifiability_and_design(),
    )
    return {
        "qualification": "cardiovascular-personalization",
        "cases": cases,
        "passed": all(bool(case["passed"]) for case in cases),
        "scope": "manufactured numerical qualification; not clinical validation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run manufactured cardiovascular personalization qualification."
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = qualification()
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")
    if not bool(report["passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
