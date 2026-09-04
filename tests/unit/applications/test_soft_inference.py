#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.robotics._soft_inference import (
    BoundedParameterMap,
    build_soft_plant_mpc,
    calibrate_reduced_rod,
    CalibrationAcceptance,
    CalibrationExperiment,
    CoDesignHeldOutScenario,
    FixedModeDerivativeEvidence,
    PositiveParameterMap,
    ReducedRodCalibrationProblem,
    ReducedRodCalibrationStatus,
    ReducedRodParameterization,
    SoftCoDesignConstraint,
    SoftRobotCoDesignProblem,
    SPDParameterMap,
)
from phydrax.control import ControlProblem, PiecewiseConstantControlParameterization
from phydrax.control._sampling_mpc import SamplingMPCRealizations
from phydrax.dynamics import TimeGrid
from phydrax.optim import SQP
from tests._control_systems import make_discrete_control_dynamics


def test_reduced_rod_physical_maps_round_trip_and_enforce_spd():
    parameterization = ReducedRodParameterization(
        (
            PositiveParameterMap("density", minimum=1e-6),
            BoundedParameterMap("friction", 0.0, 1.0),
            SPDParameterMap("constitutive", 2, diagonal_floor=1e-6),
        ),
        parameterization_id="rod-physical-maps",
    )
    latent = {
        "density": jnp.asarray(0.4),
        "friction": jnp.asarray(-0.7),
        "constitutive": jnp.asarray([0.1, -0.2, 0.5]),
    }

    physical = parameterization.to_physical(latent)
    recovered = parameterization.to_latent(physical)

    np.testing.assert_allclose(recovered["density"], latent["density"], rtol=1e-6)
    np.testing.assert_allclose(recovered["friction"], latent["friction"], rtol=1e-6)
    np.testing.assert_allclose(
        recovered["constitutive"], latent["constitutive"], rtol=1e-5
    )
    assert float(physical["density"]) > 0.0
    assert 0.0 < float(physical["friction"]) < 1.0
    assert np.min(np.linalg.eigvalsh(np.asarray(physical["constitutive"]))) > 0.0


def _calibration_problem(*, held_out_observation: float = 6.0):
    parameterization = ReducedRodParameterization(
        (PositiveParameterMap("stiffness", minimum=0.0, maximum=10.0),),
        parameterization_id="calibration:stiffness-map",
    )

    def experiment(load, observation, split, identifier):
        return CalibrationExperiment(
            lambda physical, args: jnp.asarray(
                [load * physical["stiffness"] - observation]
            ),
            split=split,
            experiment_id=identifier,
            route_id="fixed-base/contact-free/elastic",
            weight=jnp.asarray(2.0),
            route_valid=lambda physical, args: physical["stiffness"] > 0.0,
        )

    source_realization = {"revision": "source", "stiffness": jnp.asarray(1.0)}
    problem = ReducedRodCalibrationProblem(
        parameterization,
        {"stiffness": jnp.asarray(1.0)},
        source_realization,
        (
            experiment(1.0, 2.0, "train", "train:load-1"),
            experiment(2.0, 4.0, "train", "train:load-2"),
            experiment(2.5, 5.0, "validation", "validation:load-2.5"),
            experiment(
                3.0,
                held_out_observation,
                "held_out",
                "held-out:load-3",
            ),
        ),
        acceptance=CalibrationAcceptance(
            maximum_training_rmse=1e-5,
            maximum_validation_rmse=1e-5,
            maximum_held_out_rmse=1e-5,
            maximum_held_out_absolute=1e-5,
            maximum_condition_number=1e5,
            require_validation=True,
        ),
        realize=lambda physical, source: {
            "revision": "candidate",
            "stiffness": physical["stiffness"],
        },
        admissible=lambda physical, realization, args: realization["stiffness"] > 0.0,
        source_realization_id="rod-realization:source",
        rod_id="rod:fixed-base",
        reduction_id="reduction:first-mode",
        actuator_id="actuator:ideal-tendon",
        plant_id="plant:reduced-rod",
        problem_id="calibration:synthetic-stiffness",
    )
    return problem, source_realization


def test_calibration_recovers_synthetic_parameter_with_disjoint_held_out_evidence():
    problem, _ = _calibration_problem()
    initial = problem.parameterization.to_latent({"stiffness": jnp.asarray(0.5)})

    result = calibrate_reduced_rod(problem, initial)

    assert bool(result.successful)
    assert int(result.status) == ReducedRodCalibrationStatus.SUCCESS
    np.testing.assert_allclose(
        result.candidate_physical_parameters["stiffness"], 2.0, atol=1e-5
    )
    assert bool(result.identifiability.full_rank)
    assert int(result.identifiability.numerical_rank) == 1
    assert result.training.experiment_ids == ("train:load-1", "train:load-2")
    assert result.validation.experiment_ids == ("validation:load-2.5",)
    assert result.held_out.experiment_ids == ("held-out:load-3",)
    assert np.all(np.isfinite(np.asarray(result.identifiability.latent_covariance)))


def test_rank_deficient_training_data_is_not_made_identifiable_by_acceptance():
    parameterization = ReducedRodParameterization(
        (
            PositiveParameterMap("density", minimum=0.0, maximum=5.0),
            PositiveParameterMap("stiffness", minimum=0.0, maximum=5.0),
        )
    )
    experiment = lambda split, identifier: CalibrationExperiment(
        lambda physical, args: jnp.asarray(
            [physical["density"] + physical["stiffness"] - 3.0]
        ),
        split=split,
        experiment_id=identifier,
        route_id="fixed-mode:sum-only",
    )
    source = {"density": jnp.asarray(1.0), "stiffness": jnp.asarray(2.0)}
    problem = ReducedRodCalibrationProblem(
        parameterization,
        source,
        source,
        (
            experiment("train", "train:sum"),
            experiment("held_out", "held-out:sum"),
        ),
        acceptance=CalibrationAcceptance(maximum_held_out_rmse=1e-8),
        realize=lambda physical, source_realization: physical,
        source_realization_id="rank-deficient:source",
        rod_id="rod:rank-deficient",
        reduction_id="reduction:rank-deficient",
        actuator_id="actuator:none",
        plant_id="plant:rank-deficient",
        problem_id="calibration:rank-deficient",
    )
    latent = parameterization.to_latent(source)

    evidence = problem.identifiability(latent)

    assert int(evidence.numerical_rank) == 1
    assert evidence.parameter_count == 2
    assert not bool(evidence.full_rank)
    assert not bool(evidence.accepted)
    assert np.all(np.isnan(np.asarray(evidence.latent_covariance)))
    assert np.linalg.norm(np.asarray(evidence.null_projection)) > 0.0


def test_held_out_failure_rejects_candidate_and_retains_exact_source_realization():
    problem, source_realization = _calibration_problem(held_out_observation=100.0)
    initial = problem.parameterization.to_latent({"stiffness": jnp.asarray(0.5)})

    result = calibrate_reduced_rod(problem, initial)

    assert not bool(result.accepted)
    assert int(result.status) == ReducedRodCalibrationStatus.HELD_OUT_FAILED
    assert result.accepted_realization is source_realization
    assert result.accepted_realization_id == problem.source_realization_id
    assert bool(result.training.accepted)
    assert not bool(result.held_out.accepted)


def _fixed_mode_evidence(*, contact_margin=1.0, primal_result_id="primal:co-design"):
    return FixedModeDerivativeEvidence(
        material_margin=1.0,
        kinematic_margin=1.0,
        actuator_margin=1.0,
        contact_margin=contact_margin,
        active_set_margin=1.0,
        condition_number=2.0,
        jvp_residual=1e-9,
        vjp_residual=1e-9,
        finite=True,
        primal_accepted=True,
        route_fixed=True,
        morphology_id="morphology:rod-a",
        actuator_id="actuator:tendon-a",
        control_id="control:piecewise-constant-a",
        fixed_mode_id="mode:contact-free-taut",
        primal_result_id=primal_result_id,
        maximum_condition_number=100.0,
        maximum_derivative_residual=1e-6,
    )


def _co_design_problem(*, held_out=True, contact_margin=1.0):
    parameterization = ReducedRodParameterization(
        (
            BoundedParameterMap("length", 0.5, 2.0),
            BoundedParameterMap("tendon_gain", 0.1, 3.0),
        ),
        parameterization_id="co-design:physical-map",
    )
    source_design = {"length": jnp.asarray(0.0), "tendon_gain": jnp.asarray(0.0)}
    source_realization = {"revision": "source"}
    problem = SoftRobotCoDesignProblem(
        parameterization,
        source_design,
        source_realization,
        lambda state, physical, args: (
            state - jnp.asarray([physical["length"] + physical["tendon_gain"]])
        ),
        lambda state, physical, args: (
            jnp.sum((state - 2.0) ** 2) + 0.1 * jnp.square(physical["length"])
        ),
        realize=lambda physical, source: {"revision": "candidate", **physical},
        derivative_evidence=lambda state, physical, realization, primal_id, args: (
            _fixed_mode_evidence(
                contact_margin=contact_margin,
                primal_result_id=primal_id,
            )
        ),
        held_out_scenarios=(
            CoDesignHeldOutScenario(
                lambda state, physical, realization, args: jnp.asarray(held_out),
                scenario_id="held-out:load-envelope",
            ),
        ),
        constraints=(
            SoftCoDesignConstraint(
                lambda state, physical, args: state[0],
                upper=3.0,
                constraint_id="stress-proxy",
            ),
        ),
        morphology_id="morphology:rod-a",
        actuator_id="actuator:tendon-a",
        control_id="control:piecewise-constant-a",
        fixed_mode_id="mode:contact-free-taut",
        source_realization_id="co-design:source",
        problem_id="co-design:fixed-mode",
    )
    return problem, source_design, source_realization


def test_co_design_lowers_to_existing_state_design_and_sqp_with_bound_ids():
    problem, source_design, _ = _co_design_problem()

    state_design = problem.as_state_design_problem()
    compilation = problem.compile_sqp(jnp.asarray([1.0]), source_design, sample_args=None)

    assert state_design.problem_id == "co-design:fixed-mode:state-design"
    assert isinstance(compilation.method, SQP)
    assert compilation.problem is problem
    assert len(compilation.minimization.constraints) == 2
    assert problem.morphology_id == "morphology:rod-a"
    assert problem.actuator_id == "actuator:tendon-a"
    assert problem.control_id == "control:piecewise-constant-a"
    assert problem.fixed_mode_id == "mode:contact-free-taut"


class _CoDesignOptimization(eqx.Module):
    state: jax.Array
    design: dict[str, jax.Array]
    successful: jax.Array


def test_co_design_rejects_mode_boundary_and_retains_source_design():
    problem, source_design, source_realization = _co_design_problem(contact_margin=0.0)
    optimization = _CoDesignOptimization(
        jnp.asarray([2.0]),
        {
            "length": jnp.asarray(0.5),
            "tendon_gain": jnp.asarray(-0.5),
        },
        jnp.asarray(True),
    )

    result = problem.accept_result(optimization)

    assert not bool(result.accepted)
    assert not bool(result.derivative_evidence.accepted)
    assert result.accepted_realization is source_realization
    np.testing.assert_allclose(
        result.accepted_design["length"],
        problem.parameterization.to_physical(source_design)["length"],
    )


def test_soft_plant_mpc_retains_selected_replay_and_fixed_mode_evidence():
    grid = TimeGrid(jnp.asarray([0.0, 1.0]), time_id="soft-mpc:grid")
    dynamics = make_discrete_control_dynamics(
        lambda context, state, control, args: state + args["stiffness"] * control,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="soft-mpc:dynamics",
    )
    parameterization = PiecewiseConstantControlParameterization(
        grid,
        (1,),
        parameterization_id="control:piecewise-constant-a",
    )
    problem = ControlProblem(
        dynamics,
        grid,
        jnp.asarray([0.0]),
        terminal_cost=lambda time, state, args: jnp.square(state[0]),
        args={"stiffness": jnp.asarray(1.0)},
        problem_id="soft-mpc:problem",
    )
    realizations = SamplingMPCRealizations(
        {"stiffness": jnp.asarray([1.0, 2.0])},
        ("rod:soft", "rod:stiff"),
        weights=jnp.asarray([0.5, 0.5]),
        support_mask=jnp.asarray([True, True]),
        posterior_id="posterior:rod-calibration",
        campaign_id="campaign:rod-calibration",
    )
    plan = build_soft_plant_mpc(
        problem,
        parameterization,
        realizations,
        realization_binding=lambda base, physical: physical,
        realization_binding_id="bind:rod-parameters",
        derivative_evidence=lambda replay, result: _fixed_mode_evidence(
            primal_result_id=result.result_id
        ),
        morphology_id="morphology:rod-a",
        actuator_id="actuator:tendon-a",
        control_id="control:piecewise-constant-a",
        fixed_mode_id="mode:contact-free-taut",
        candidate_count=1,
        iteration_count=1,
        update="predictive",
    )

    result = plan.solve(
        plan.sampling.initialize(jnp.asarray([[0.5]]), 0.0),
        jax.random.key(12),
    )

    assert bool(result.successful)
    assert result.selected_accepted_replay is result.sampling.replay
    np.testing.assert_allclose(
        result.selected_accepted_replay.states[:, -1, 0], [0.5, 1.0]
    )
    assert result.derivative_evidence.fixed_mode_id == "mode:contact-free-taut"
