#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from phydrax._sampling import (
    CallableProposal,
    derive_key,
    MetropolisHastings,
    SampleAddress,
)
from phydrax.metrix import BosonicGaussianState
from phydrax.operators.quantum import (
    ApproximationAxis,
    ApproximationQuantity,
    CallableDiscreteQuantumOperator,
    ConnectedConfigurations,
    drude_lorentz_pade,
    LogAmplitude,
    OpenSystemApproximationEvidence,
    OpenSystemPhysicalityEvidence,
)
from phydrax.solver._gaussian_lindblad import (
    GaussianLindbladProblem,
    solve_gaussian_lindblad,
)
from phydrax.solver._heom_implicit import solve_heom_adaptive_bdf
from phydrax.solver._lindblad import LindbladProblem, solve_lindblad
from phydrax.solver._memory_kernel import (
    certify_memory_kernel_map,
    exponential_memory_qubit_problem,
    solve_memory_kernel,
)
from phydrax.solver._mps_quantum_jump import (
    LocalMPSJump,
    MPSQuantumJumpProblem,
    solve_mps_quantum_jump,
)
from phydrax.solver._neural_sampled_trajectory import (
    audit_connected_vmc_jump_projection,
    ConnectedVMCNeuralTrajectoryPolicy,
    ConnectedVMCNeuralTrajectoryProblem,
    solve_connected_vmc_neural_trajectory,
)
from phydrax.solver._process_tomography import (
    informationally_complete_process_experiments,
    tomography_designs_disjoint,
)
from phydrax.solver._purified_tebd import solve_purified_strang
from phydrax.solver._quantum_jump import QuantumJumpProblem, StateVectorOperator
from phydrax.solver._quantum_jump_generic import solve_quantum_jump_generic
from phydrax.solver._stinespring_tomography import (
    fit_causal_process_memory,
    fit_stinespring_process,
    StinespringTomographyProblem,
)
from phydrax.solver._variational_monte_carlo import VariationalMonteCarloProblem
from phydrax.solver._xxz_open import boundary_driven_xxz_problem
from phydrax.tensor_network import (
    CausalProcessTensor,
    CombLegSpec,
    NearestNeighborHamiltonian,
    product_mps,
    SequentialStinespringProcess,
)

from .contracts import (
    CampaignCapacityEvidence,
    CampaignPrecisionBundle,
    OpenSystemCampaignRecord,
    SemanticReplayEvidence,
)


class _CampaignTableAmplitude(eqx.Module):
    parameters: jax.Array

    def __call__(self, configuration):
        index = (configuration[0] > 0).astype(jnp.int32)
        value = self.parameters[index]
        return LogAmplitude(jnp.real(value), jnp.exp(1j * jnp.imag(value)))


def _precision(dtype, domain, children=None):
    name = jnp.dtype(dtype).name
    real = "float64" if name in ("float64", "complex128") else "float32"
    return CampaignPrecisionBundle(
        domain,
        f"phydrax-{domain}",
        {
            "storage": name,
            "compute": name,
            "accumulation": name,
            "certification": real,
            "output": name,
        },
        children=children,
    )


def _replay(*, event=0.0, disagreement=0.0, observable=0.0):
    return SemanticReplayEvidence(
        variates_equal=True,
        address_schema_equal=True,
        event_time_difference=event,
        channel_disagreement_probability=disagreement,
        observable_difference=observable,
        event_time_tolerance=1e-6,
        disagreement_tolerance=0.05,
        observable_tolerance=1e-5,
    )


def _record(
    campaign_id,
    representation_id,
    axes,
    quantities,
    physicality,
    precision,
    replay,
    execution,
    *,
    artifact_arrays,
    work,
    capacity_evidence,
    precision_policy_ids,
    unsupported=(),
):
    approximation = OpenSystemApproximationEvidence(
        representation_id,
        tuple(axes),
        tuple(quantities),
        execution_valid=execution,
        precision_evidence=precision.evidence,
        precision_policy_ids=tuple(precision_policy_ids),
    )
    return OpenSystemCampaignRecord(
        campaign_id,
        representation_id,
        approximation,
        physicality,
        precision,
        replay,
        execution_success=execution,
        capacity_evidence=tuple(capacity_evidence),
        artifact_arrays=artifact_arrays,
        work=work,
        unsupported_claims=unsupported,
    )


def gaussian_campaign():
    gamma = 0.4
    hbar = 2.0
    occupation = 1.0
    initial = BosonicGaussianState(
        jnp.zeros(2), 0.5 * hbar * jnp.eye(2), hbar=hbar
    )
    problem = GaussianLindbladProblem(
        -0.5 * gamma * jnp.eye(2),
        gamma * (occupation + 0.5) * hbar * jnp.eye(2),
        jnp.zeros(2),
        initial,
        problem_id="nonunit-hbar-thermal-oscillator",
    )
    coarse = solve_gaussian_lindblad(problem, step_size=0.04, steps=10)
    fine = solve_gaussian_lindblad(problem, step_size=0.02, steps=20)
    stationary = problem.stationary_state()
    final_time = 0.4
    analytic_covariance = stationary.covariance + jnp.exp(
        -gamma * final_time
    ) * (initial.covariance - stationary.covariance)
    error = jnp.linalg.norm(fine.covariances[-1] - analytic_covariance)
    precision = _precision(
        fine.covariances.dtype,
        "gaussian-campaign",
        children={"solver": fine.precision_evidence},
    )
    closure = jnp.maximum(-jnp.min(fine.uncertainty_margins), 0.0)
    physicality = OpenSystemPhysicalityEvidence(
        closure_residual=closure,
        certified_properties=("representation-closure",),
        precision_evidence=precision.evidence,
    )
    return _record(
        "gaussian-affine-v1",
        "bosonic-gaussian",
        (
            ApproximationAxis("time-step", 0.02, units="time"),
            ApproximationAxis("hbar", hbar),
        ),
        (
            ApproximationQuantity(
                "analytic-covariance-error",
                error,
                1e-6,
                units="covariance",
                norm_id="frobenius",
                estimate_kind="estimate",
            ),
        ),
        physicality,
        precision,
        _replay(),
        fine.valid,
        artifact_arrays={
            "coarse-covariances": coarse.covariances,
            "fine-covariances": fine.covariances,
            "analytic-covariance": analytic_covariance,
            "fine-means": fine.means,
        },
        work={"coarse-steps": 10, "fine-steps": 20},
        capacity_evidence=(
            CampaignCapacityEvidence(
                "state-dimension",
                fine.means.shape[-1],
                fine.means.shape[-1],
                saturated=False,
            ),
        ),
        precision_policy_ids=(fine.precision.policy_id,),
        unsupported=("arbitrary-non-gaussian-dynamics",),
    )

def dense_trajectory_campaign():
    lowering = jnp.asarray([[0, 1], [0, 0]], dtype=complex)
    raising = jnp.conj(lowering.T)
    down = StateVectorOperator.from_matrix(
        lowering, operator_id="thermal-down"
    )
    up = StateVectorOperator.from_matrix(
        jnp.sqrt(0.3) * raising, operator_id="thermal-up"
    )
    initial_state = jnp.asarray([1.0 + 0.0j, 0.0j])
    problem = QuantumJumpProblem(
        StateVectorOperator.from_matrix(
            jnp.zeros((2, 2), dtype=complex), operator_id="zero-hamiltonian"
        ),
        (down, up),
        initial_state,
        problem_id="thermal-two-channel-trajectories",
    )
    save_times = jnp.asarray([0.0, 0.4])
    coarse = solve_quantum_jump_generic(
        problem,
        jax.random.PRNGKey(3),
        t0=0.0,
        t1=0.4,
        save_times=save_times,
        trajectory_count=64,
        maximum_events_per_channel=8,
        dt0=0.01,
        rtol=1e-5,
        atol=1e-7,
    )
    fine = solve_quantum_jump_generic(
        problem,
        jax.random.PRNGKey(3),
        t0=0.0,
        t1=0.4,
        save_times=save_times,
        trajectory_count=64,
        maximum_events_per_channel=8,
        dt0=0.005,
        rtol=1e-7,
        atol=1e-9,
    )

    def empirical(solution):
        final = solution.states[:, -1, :]
        state = final[:, :2] + 1j * final[:, 2:]
        return jnp.mean(
            jax.vmap(
                lambda value: value[:, None] * jnp.conj(value[None, :])
            )(state),
            axis=0,
        )

    coarse_density = empirical(coarse)
    fine_density = empirical(fine)
    dense_problem = LindbladProblem(
        jnp.zeros((2, 2), dtype=complex),
        jnp.stack((lowering, jnp.sqrt(0.3) * raising)),
        jnp.outer(initial_state, jnp.conj(initial_state)),
        problem_id="thermal-two-channel-dense-reference",
    )
    dense = solve_lindblad(dense_problem, step_size=0.01, steps=40)
    dense_density = dense.states[-1]
    coupled_difference = jnp.linalg.norm(coarse_density - fine_density)
    reference_difference = jnp.linalg.norm(fine_density - dense_density)
    execution_valid = jnp.all(fine.valid) & dense.valid
    precision = _precision(fine.states.dtype, "trajectory-campaign")
    trace_residual = jnp.abs(jnp.trace(fine_density) - 1.0)
    hermiticity_residual = jnp.max(
        jnp.abs(fine_density - jnp.conj(fine_density.T))
    )
    positivity_margin = jnp.min(jnp.linalg.eigvalsh(fine_density))
    physicality = OpenSystemPhysicalityEvidence(
        trace_residual=trace_residual,
        hermiticity_residual=hermiticity_residual,
        positivity_margin=positivity_margin,
        certified_properties=("trace", "hermiticity", "positivity"),
        precision_evidence=precision.evidence,
    )
    return _record(
        "dense-trajectories-v1",
        "generic-event-vector-trajectories",
        (
            ApproximationAxis("relative-tolerance", 1e-7),
            ApproximationAxis("trajectory-count", 64),
        ),
        (
            ApproximationQuantity(
                "coupled-observable-difference",
                coupled_difference,
                0.15,
                units="density",
                norm_id="frobenius",
                estimate_kind="statistical",
                confidence=0.95,
            ),
            ApproximationQuantity(
                "dense-reference-difference",
                reference_difference,
                0.15,
                units="density",
                norm_id="frobenius",
                estimate_kind="statistical",
                confidence=0.95,
            ),
        ),
        physicality,
        precision,
        _replay(
            event=coupled_difference,
            disagreement=0.0,
            observable=coupled_difference,
        ),
        execution_valid,
        artifact_arrays={
            "coarse-final-density": coarse_density,
            "fine-final-density": fine_density,
            "dense-reference-density": dense_density,
            "fine-states": fine.states,
        },
        work={"trajectories": 64, "dense-steps": 40},
        capacity_evidence=(
            CampaignCapacityEvidence(
                "maximum-events-per-path",
                int(jnp.max(fine.events.counts)),
                fine.events.max_events,
                saturated=jnp.any(fine.events.overflow),
            ),
        ),
        precision_policy_ids=(precision.evidence.evidence_id,),
        unsupported=("pathwise-event-identity-under-refinement",),
    )


def mps_campaign():
    state = product_mps(
        jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    )
    hamiltonian = NearestNeighborHamiltonian(
        (jnp.zeros((4, 4), dtype=complex),),
        (2, 2),
        hamiltonian_id="campaign-zero",
    )
    lowering = jnp.asarray([[0, 1], [0, 0]], dtype=complex)
    problem = MPSQuantumJumpProblem(
        hamiltonian,
        (LocalMPSJump(0, lowering, jump_id="loss"),),
        state,
        problem_id="eventful-mps-amplitude-damping",
    )
    root_key = jax.random.PRNGKey(5)
    result = solve_mps_quantum_jump(
        problem,
        root_key,
        step_size=0.1,
        steps=20,
        maximum_bond_dimension=4,
        maximum_events=8,
    )
    discarded = jnp.max(result.discarded_weight_history, initial=0.0)
    threshold_address = SampleAddress(
        "quantum-trajectory",
        "mps-jump-threshold",
        target=problem.problem_id,
        role="threshold",
    )
    threshold = jax.random.uniform(derive_key(root_key, threshold_address, 0))
    exact_event_time = -jnp.log(threshold)
    event_time_error = jnp.where(
        result.active_events[0],
        jnp.abs(result.jump_times[0] - exact_event_time),
        jnp.asarray(jnp.inf),
    )
    root_residual = jnp.max(
        jnp.where(result.active_events, result.root_residuals, 0.0)
    )
    execution = result.valid & jnp.any(result.active_events)
    precision = _precision(
        result.final_state.tensors[0].dtype,
        "mps-campaign",
        children={"solver": result.final_state.precision_evidence},
    )
    norm_residual = jnp.abs(result.final_state.norm() - 1.0)
    closure_residual = jnp.maximum(
        root_residual,
        jnp.where(result.event_capacity_saturated, jnp.inf, 0.0),
    )
    physicality = OpenSystemPhysicalityEvidence(
        trace_residual=norm_residual,
        closure_residual=closure_residual,
        certified_properties=("trace", "representation-closure"),
        precision_evidence=precision.evidence,
    )
    return _record(
        "mps-trajectories-v1",
        "mps-trajectories",
        (
            ApproximationAxis("time-step", 0.1, units="time"),
            ApproximationAxis("bond-dimension", 4),
        ),
        (
            ApproximationQuantity(
                "maximum-discarded-weight",
                discarded,
                1e-6,
                units="norm-squared",
                norm_id="maximum",
                estimate_kind="estimate",
            ),
            ApproximationQuantity(
                "event-time-reference-error",
                event_time_error,
                1e-6,
                units="time",
                norm_id="absolute",
                estimate_kind="estimate",
            ),
            ApproximationQuantity(
                "maximum-root-residual",
                root_residual,
                1e-8,
                units="probability",
                norm_id="maximum",
                estimate_kind="bound",
            ),
        ),
        physicality,
        precision,
        _replay(event=event_time_error, observable=norm_residual),
        execution,
        artifact_arrays={
            "final-state": result.final_state.to_dense(maximum_elements=16),
            "event-times": result.jump_times,
            "event-channels": result.jump_channels,
            "active-events": result.active_events,
            "root-residuals": result.root_residuals,
            "discarded-weight": result.discarded_weight_history,
        },
        work={"steps": 20, "events": result.event_count},
        capacity_evidence=(
            CampaignCapacityEvidence(
                "maximum-events",
                result.event_count,
                result.maximum_events,
                saturated=result.event_capacity_saturated,
            ),
            CampaignCapacityEvidence(
                "bond-dimension",
                max(result.final_state.bond_dimensions, default=1),
                4,
                saturated=discarded > 1e-6,
            ),
        ),
        precision_policy_ids=(result.final_state.precision.policy_id,),
        unsupported=("exact-mps-unravelling",),
    )


def lpdo_campaign():
    coarse_problem = boundary_driven_xxz_problem(
        2, half_step=0.01, boundary_rate=0.2
    )
    fine_problem = boundary_driven_xxz_problem(
        2, half_step=0.005, boundary_rate=0.2
    )
    coarse = solve_purified_strang(
        coarse_problem,
        step_size=0.02,
        steps=2,
        maximum_bond_dimension=4,
        maximum_purification_dimension=8,
    )
    result = solve_purified_strang(
        fine_problem,
        step_size=0.01,
        steps=4,
        maximum_bond_dimension=4,
        maximum_purification_dimension=8,
    )
    coarse_density = coarse.final_state.to_dense_density(normalize=True)
    fine_density = result.final_state.to_dense_density(normalize=True)
    refinement_error = jnp.linalg.norm(fine_density - coarse_density)
    trace_error = jnp.max(jnp.abs(result.raw_trace_history - 1.0))
    bond_discarded = jnp.max(result.bond_discarded_history, initial=0.0)
    kraus_discarded = jnp.max(result.kraus_discarded_history, initial=0.0)
    canonical_residual = jnp.max(result.canonical_residual_history, initial=0.0)
    positivity_margin = jnp.min(
        jnp.linalg.eigvalsh(0.5 * (fine_density + jnp.conj(fine_density.T)))
    )
    precision = _precision(
        result.final_state.tensors[0].dtype,
        "lpdo-campaign",
        children={"solver": result.final_state.precision_evidence},
    )
    closure = jnp.max(
        jnp.stack((bond_discarded, kraus_discarded, canonical_residual))
    )
    physicality = OpenSystemPhysicalityEvidence(
        trace_residual=trace_error,
        positivity_margin=positivity_margin,
        closure_residual=closure,
        certified_properties=("trace", "positivity", "representation-closure"),
        precision_evidence=precision.evidence,
    )
    truncation_saturated = (bond_discarded > 1e-6) | (kraus_discarded > 1e-6)
    return _record(
        "lpdo-xxz-v1",
        "locally-purified-density",
        (
            ApproximationAxis("time-step", 0.01, parent_value=0.02, units="time"),
            ApproximationAxis("physical-bond", 4),
            ApproximationAxis("purification-rank", 8),
        ),
        (
            ApproximationQuantity(
                "time-refinement-error",
                refinement_error,
                1e-3,
                units="density",
                norm_id="frobenius",
                estimate_kind="estimate",
            ),
            ApproximationQuantity(
                "maximum-trace-residual",
                trace_error,
                1e-6,
                units="trace",
                norm_id="maximum",
                estimate_kind="bound",
            ),
            ApproximationQuantity(
                "maximum-bond-discarded-weight",
                bond_discarded,
                1e-6,
                units="norm-squared",
                norm_id="maximum",
                estimate_kind="bound",
            ),
            ApproximationQuantity(
                "maximum-kraus-discarded-weight",
                kraus_discarded,
                1e-6,
                units="norm-squared",
                norm_id="maximum",
                estimate_kind="bound",
            ),
            ApproximationQuantity(
                "maximum-canonical-residual",
                canonical_residual,
                1e-6,
                units="norm",
                norm_id="maximum",
                estimate_kind="bound",
            ),
        ),
        physicality,
        precision,
        _replay(observable=refinement_error),
        result.valid & coarse.valid,
        artifact_arrays={
            "coarse-density": coarse_density,
            "fine-density": fine_density,
            "trace-history": result.raw_trace_history,
            "bond-discarded": result.bond_discarded_history,
            "kraus-discarded": result.kraus_discarded_history,
            "canonical-residual": result.canonical_residual_history,
        },
        work={"coarse-steps": 2, "fine-steps": 4},
        capacity_evidence=(
            CampaignCapacityEvidence(
                "physical-bond",
                max(
                    (tensor.shape[-1] for tensor in result.final_state.tensors[:-1]),
                    default=1,
                ),
                4,
                saturated=truncation_saturated,
            ),
            CampaignCapacityEvidence(
                "purification-rank",
                max(result.final_state.purification_dimensions),
                8,
                saturated=truncation_saturated,
            ),
        ),
        precision_policy_ids=(result.final_state.precision.policy_id,),
        unsupported=("global-steady-state-uniqueness",),
    )


def heom_campaign():
    density = jnp.asarray([[0.6 + 0j, 0j], [0j, 0.4 + 0j]])
    expansion1 = drude_lorentz_pade(0.01, 1.0, 2.0, 1)
    expansion2 = drude_lorentz_pade(0.01, 1.0, 2.0, 2)
    from phydrax.solver._heom import HEOMHierarchy, HEOMProblem
    from phydrax.solver._heom_production import solve_heom_continuation_grid

    base = HEOMProblem(
        jnp.zeros((2, 2), dtype=complex),
        jnp.asarray([[1, 0], [0, -1]], dtype=complex),
        expansion1,
        HEOMHierarchy(expansion1.rank, 1),
        density,
        problem_id="adaptive-heom-spin-boson",
    )
    grid = solve_heom_continuation_grid(
        base,
        (expansion1, expansion2),
        (1, 2),
        step_size=0.002,
        steps=2,
    )
    high_problem = HEOMProblem(
        base.hamiltonian,
        base.coupling_operator,
        expansion2,
        HEOMHierarchy(expansion2.rank, 2),
        density,
        problem_id="adaptive-heom-spin-boson:high",
    )
    loose = solve_heom_adaptive_bdf(
        high_problem,
        final_time=0.004,
        initial_step=0.002,
        relative_tolerance=1e-4,
        absolute_tolerance=1e-7,
        maximum_step=0.002,
        maximum_attempts=64,
    )
    tight = solve_heom_adaptive_bdf(
        high_problem,
        final_time=0.004,
        initial_step=0.001,
        relative_tolerance=1e-5,
        absolute_tolerance=1e-8,
        maximum_step=0.001,
        maximum_attempts=128,
    )
    depth = jnp.max(grid.depth_differences, initial=0.0)
    bath = jnp.max(grid.bath_differences, initial=0.0)
    adaptive_difference = jnp.linalg.norm(
        loose.solution.root_states[-1] - tight.solution.root_states[-1]
    )
    accepted_errors = jnp.where(
        tight.evidence.accepted_steps, tight.evidence.error_ratios, 0.0
    )
    maximum_error_ratio = jnp.max(accepted_errors, initial=0.0)
    top_tier_norm = tight.solution.maximum_auxiliary_norm_by_level[-1]
    root_states = jnp.concatenate(
        (
            grid.final_roots.reshape((-1, 2, 2)),
            tight.solution.root_states,
        ),
        axis=0,
    )
    trace_residual = jnp.max(
        jnp.abs(jnp.trace(root_states, axis1=-2, axis2=-1) - 1.0)
    )
    hermiticity_residual = jnp.max(
        jnp.abs(root_states - jnp.swapaxes(jnp.conj(root_states), -1, -2))
    )
    positivity_margin = jnp.min(
        jnp.linalg.eigvalsh(
            0.5 * (root_states + jnp.swapaxes(jnp.conj(root_states), -1, -2))
        )
    )
    execution = grid.valid & loose.valid & tight.valid
    precision = _precision(
        tight.solution.root_states.dtype,
        "heom-campaign",
        children={"solver": tight.solution.precision_evidence},
    )
    physicality = OpenSystemPhysicalityEvidence(
        trace_residual=trace_residual,
        hermiticity_residual=hermiticity_residual,
        positivity_margin=positivity_margin,
        certified_properties=("trace", "hermiticity", "positivity"),
        precision_evidence=precision.evidence,
    )
    hierarchy_saturated = top_tier_norm > 0.1
    return _record(
        "heom-spin-boson-v1",
        "adaptive-heom",
        (
            ApproximationAxis("hierarchy-depth", 2, parent_value=1),
            ApproximationAxis("bath-pole-order", 2, parent_value=1),
            ApproximationAxis("relative-tolerance", 1e-5, parent_value=1e-4),
        ),
        (
            ApproximationQuantity(
                "depth-difference",
                depth,
                0.1,
                units="density",
                norm_id="frobenius",
                estimate_kind="estimate",
            ),
            ApproximationQuantity(
                "bath-difference",
                bath,
                0.1,
                units="density",
                norm_id="frobenius",
                estimate_kind="estimate",
            ),
            ApproximationQuantity(
                "adaptive-tolerance-difference",
                adaptive_difference,
                1e-3,
                units="density",
                norm_id="frobenius",
                estimate_kind="estimate",
            ),
            ApproximationQuantity(
                "maximum-local-error-ratio",
                maximum_error_ratio,
                1.0,
                units="ratio",
                norm_id="maximum",
                estimate_kind="bound",
            ),
            ApproximationQuantity(
                "maximum-top-tier-norm",
                top_tier_norm,
                0.1,
                units="density",
                norm_id="frobenius",
                estimate_kind="estimate",
            ),
        ),
        physicality,
        precision,
        _replay(observable=adaptive_difference),
        execution,
        artifact_arrays={
            "grid-final-roots": grid.final_roots,
            "depth-differences": grid.depth_differences,
            "bath-differences": grid.bath_differences,
            "loose-root-states": loose.solution.root_states,
            "tight-root-states": tight.solution.root_states,
            "adaptive-step-sizes": tight.evidence.attempted_step_sizes,
            "adaptive-accepted": tight.evidence.accepted_steps,
            "adaptive-error-ratios": tight.evidence.error_ratios,
        },
        work={
            "grid-points": 4,
            "loose-attempts": loose.evidence.attempted_step_sizes.shape[0],
            "tight-attempts": tight.evidence.attempted_step_sizes.shape[0],
        },
        capacity_evidence=(
            CampaignCapacityEvidence(
                "adaptive-attempts",
                tight.evidence.attempted_step_sizes.shape[0],
                tight.maximum_attempts,
                saturated=tight.evidence.capacity_saturated,
            ),
            CampaignCapacityEvidence(
                "hierarchy-depth",
                2,
                2,
                saturated=hierarchy_saturated,
            ),
        ),
        precision_policy_ids=(tight.solution.temporal_precision.policy_id,),
        unsupported=("infinite-depth-heom-proof",),
    )


def memory_campaign():
    density = jnp.asarray([[0.6 + 0j, 0j], [0j, 0.4 + 0j]])
    coarse_problem = exponential_memory_qubit_problem(0.01, 1.0, density)
    fine_problem = exponential_memory_qubit_problem(0.01, 1.0, density)
    coarse = solve_memory_kernel(coarse_problem, step_size=0.002, steps=10)
    result = solve_memory_kernel(fine_problem, step_size=0.001, steps=20)
    certification = certify_memory_kernel_map(
        fine_problem,
        step_size=0.001,
        steps=20,
    )
    refinement = jnp.linalg.norm(coarse.states[-1] - result.states[-1])
    cp_margin = jnp.min(certification.cp_margins)
    tp_residual = jnp.max(certification.trace_preservation_residuals)
    trace_residual = jnp.max(result.trace_residuals)
    hermiticity_residual = jnp.max(result.hermiticity_residuals)
    positivity_margin = jnp.min(result.minimum_eigenvalues)
    precision = _precision(
        result.states.dtype,
        "memory-campaign",
        children={"solver": result.precision_evidence},
    )
    physicality = OpenSystemPhysicalityEvidence(
        trace_residual=trace_residual,
        hermiticity_residual=hermiticity_residual,
        positivity_margin=positivity_margin,
        channel_cp_margin=cp_margin,
        trace_preservation_residual=tp_residual,
        certified_properties=(
            "trace",
            "hermiticity",
            "positivity",
            "complete-positivity",
            "trace-preservation",
        ),
        precision_evidence=precision.evidence,
    )
    return _record(
        "constructive-memory-v1",
        "direct-memory-map",
        (
            ApproximationAxis(
                "memory-step", 0.001, parent_value=0.002, units="time"
            ),
            ApproximationAxis("memory-horizon", fine_problem.kernel.memory_horizon),
        ),
        (
            ApproximationQuantity(
                "time-refinement-error",
                refinement,
                1e-3,
                units="density",
                norm_id="frobenius",
                estimate_kind="estimate",
            ),
            ApproximationQuantity(
                "maximum-trace-preservation-residual",
                tp_residual,
                1e-8,
                units="matrix",
                norm_id="maximum",
                estimate_kind="bound",
            ),
            ApproximationQuantity(
                "maximum-complete-positivity-violation",
                jnp.maximum(-cp_margin, 0.0),
                1e-8,
                units="eigenvalue",
                norm_id="maximum",
                estimate_kind="bound",
            ),
        ),
        physicality,
        precision,
        _replay(observable=refinement),
        result.execution_valid & coarse.execution_valid,
        artifact_arrays={
            "coarse-states": coarse.states,
            "fine-states": result.states,
            "superoperators": certification.superoperators,
            "choi-matrices": certification.choi_matrices,
            "cp-margins": certification.cp_margins,
            "trace-preservation-residuals": (
                certification.trace_preservation_residuals
            ),
        },
        work={"coarse-steps": 10, "fine-steps": 20, "basis-solves": 4},
        capacity_evidence=(
            CampaignCapacityEvidence(
                "memory-steps",
                20,
                20,
                saturated=False,
            ),
        ),
        precision_policy_ids=(
            result.temporal_precision.policy_id,
            result.geometry_precision.policy_id,
            result.hermitian_precision.policy_id,
            result.integration_precision.policy_id,
        ),
        unsupported=("universal-direct-memory-kernel-complete-positivity",),
    )


def process_recovery_campaign():
    spec = CombLegSpec(2, 1, 1)
    angle = 0.15
    source_isometry = jnp.asarray(
        (
            (jnp.cos(angle), -jnp.sin(angle)),
            (jnp.sin(angle), jnp.cos(angle)),
        ),
        dtype=complex,
    )
    source_model = SequentialStinespringProcess(
        spec,
        jnp.diag(jnp.sqrt(jnp.asarray([0.7, 0.3]))),
        (source_isometry,),
        (1,),
        process_id="nontrivial-process-source",
    )
    model = SequentialStinespringProcess(
        spec,
        jnp.eye(2, dtype=complex),
        (jnp.eye(2, dtype=complex),),
        (1,),
        process_id="nontrivial-process-campaign",
    )
    source = source_model.materialize()
    training = informationally_complete_process_experiments(
        source,
        shots=200.0,
        design_seed=0,
        experiment_id="process-recovery-training",
    )
    held_out = informationally_complete_process_experiments(
        source,
        shots=160.0,
        design_seed=1,
        experiment_id="process-recovery-held-out",
    )
    if not tomography_designs_disjoint(training, held_out):
        raise ValueError("Process recovery designs must be setting-disjoint.")
    result = fit_stinespring_process(
        StinespringTomographyProblem(model, training),
        iterations=5,
        learning_rate=1e-4,
        held_out_experiments=held_out,
    )
    fitted = result.model.materialize()
    observed_probabilities = jnp.stack(
        [experiment.count / experiment.trials for experiment in held_out]
    )
    initial_probabilities = jnp.stack(
        [experiment.probability(model.materialize()) for experiment in held_out]
    )
    source_probabilities = jnp.stack(
        [experiment.probability(source) for experiment in held_out]
    )
    fitted_probabilities = jnp.stack(
        [experiment.probability(fitted) for experiment in held_out]
    )
    initial_error = jnp.max(
        jnp.abs(initial_probabilities - observed_probabilities)
    )
    held_out_error = jnp.max(
        jnp.abs(fitted_probabilities - observed_probabilities)
    )
    improvement_ratio = held_out_error / jnp.maximum(initial_error, 1e-12)
    recovery_execution = (
        result.valid
        & (initial_error > 1e-8)
        & (held_out_error < initial_error)
    )
    precision = _precision(source_isometry.dtype, "process-campaign")
    fitted_trace_residual = jnp.abs(jnp.trace(fitted.initial_state) - 1.0)
    fitted_positivity = jnp.min(jnp.linalg.eigvalsh(fitted.initial_state))
    fitted_tp_residual = jnp.max(fitted.channel_completeness_residuals)
    physicality = OpenSystemPhysicalityEvidence(
        trace_residual=fitted_trace_residual,
        positivity_margin=fitted_positivity,
        channel_cp_margin=0.0,
        trace_preservation_residual=fitted_tp_residual,
        certified_properties=(
            "trace",
            "positivity",
            "complete-positivity",
            "trace-preservation",
        ),
        precision_evidence=precision.evidence,
    )
    return _record(
        "process-recovery-v1",
        "sequential-stinespring",
        (
            ApproximationAxis("memory-dimension", 1),
            ApproximationAxis("intervention-settings", len(training)),
        ),
        (
            ApproximationQuantity(
                "held-out-probability-error",
                held_out_error,
                1e-2,
                units="probability",
                norm_id="maximum",
                estimate_kind="estimate",
            ),
            ApproximationQuantity(
                "post-fit-to-pre-fit-error-ratio",
                improvement_ratio,
                0.5,
                units="ratio",
                norm_id="scalar",
                estimate_kind="estimate",
            ),
        ),
        physicality,
        precision,
        _replay(),
        recovery_execution,
        artifact_arrays={
            "source-initial-density": source.initial_state,
            "initial-guess-initial-density": model.materialize().initial_state,
            "fitted-initial-density": fitted.initial_state,
            "source-isometry": source_isometry,
            "initial-guess-isometry": model.isometries[0],
            "fitted-isometry": result.model.isometries[0],
            "loss-history": result.loss_history,
            "held-out-observed-probabilities": observed_probabilities,
            "held-out-source-probabilities": source_probabilities,
            "held-out-initial-probabilities": initial_probabilities,
            "held-out-fitted-probabilities": fitted_probabilities,
            "pre-fit-probability-error": initial_error,
            "post-fit-probability-error": held_out_error,
            "post-fit-to-pre-fit-error-ratio": improvement_ratio,
            "identifiability-rank": result.identifiability_rank,
            "physical-parameter-count": result.physical_parameter_count,
            "identifiability-singular-values": result.singular_values,
        },
        work={
            "optimization-iterations": 5,
            "training-experiments": len(training),
            "held-out-experiments": len(held_out),
        },
        capacity_evidence=(
            CampaignCapacityEvidence(
                "intervention-settings",
                len(training),
                100_000,
                saturated=False,
            ),
            CampaignCapacityEvidence(
                "memory-dimension",
                1,
                1,
                saturated=False,
            ),
        ),
        precision_policy_ids=(precision.evidence.evidence_id,),
        unsupported=("unique-process-recovery-outside-gauge-and-design",),
    )


def distillation_campaign():
    def swap_with_first_memory(memory_dimension):
        matrix = jnp.zeros(
            (2 * memory_dimension, 2 * memory_dimension), dtype=complex
        )
        spectator_dimension = memory_dimension // 2
        for system in range(2):
            for memory in range(2):
                for spectator in range(spectator_dimension):
                    source_index = (
                        system * memory_dimension
                        + memory * spectator_dimension
                        + spectator
                    )
                    target_index = (
                        memory * memory_dimension
                        + system * spectator_dimension
                        + spectator
                    )
                    matrix = matrix.at[target_index, source_index].set(1.0)
        return matrix

    def controlled_phase(memory_dimension):
        diagonal = []
        spectator_dimension = memory_dimension // 2
        for system in range(2):
            for memory in range(2):
                for _ in range(spectator_dimension):
                    diagonal.append(-1.0 if system == memory == 1 else 1.0)
        return jnp.diag(jnp.asarray(diagonal, dtype=complex))

    source_spec = CombLegSpec(2, 4, 2)
    source_memory = jnp.zeros((4, 4), dtype=complex).at[0, 0].set(1.0)
    source_density = jnp.kron(
        jnp.diag(jnp.asarray([0.65, 0.35], dtype=complex)),
        source_memory,
    )
    source = CausalProcessTensor(
        source_spec,
        source_density,
        (
            swap_with_first_memory(4)[None, ...],
            controlled_phase(4)[None, ...],
        ),
        process_id="active-compressible-memory-4",
    )
    target = SequentialStinespringProcess(
        CombLegSpec(2, 2, 2),
        jnp.diag(
            jnp.asarray([jnp.sqrt(0.6), 1e-3, jnp.sqrt(0.4), 1e-3], dtype=complex)
        ),
        (swap_with_first_memory(2), controlled_phase(2)),
        (1, 1),
        process_id="active-compressible-memory-2-guess",
    )
    training = informationally_complete_process_experiments(
        source,
        shots=200.0,
        design_seed=0,
        experiment_id="distillation-training",
    )
    held_out = informationally_complete_process_experiments(
        source,
        shots=160.0,
        design_seed=1,
        experiment_id="distillation-held-out",
    )
    result = fit_causal_process_memory(
        source,
        target,
        training,
        held_out,
        iterations=5,
        learning_rate=1e-6,
        probability_tolerance=5e-2,
        identifiability_tolerance=1e-7,
        optimize_isometries=False,
    )
    precision = _precision(source.initial_state.dtype, "distillation-campaign")
    trace_residual = jnp.abs(jnp.trace(result.process.initial_state) - 1.0)
    positivity_margin = jnp.min(
        jnp.linalg.eigvalsh(result.process.initial_state)
    )
    tp_residual = jnp.max(result.process.channel_completeness_residuals)
    physicality = OpenSystemPhysicalityEvidence(
        trace_residual=trace_residual,
        positivity_margin=positivity_margin,
        channel_cp_margin=0.0,
        trace_preservation_residual=tp_residual,
        certified_properties=(
            "trace",
            "positivity",
            "complete-positivity",
            "trace-preservation",
        ),
        precision_evidence=precision.evidence,
    )
    return _record(
        "causal-distillation-v1",
        "causal-memory-refit",
        (
            ApproximationAxis(
                "memory-dimension", 2, parent_value=4
            ),
            ApproximationAxis("slot-count", 2),
        ),
        (
            ApproximationQuantity(
                "held-out-probability-error",
                result.maximum_held_out_probability_error,
                5e-2,
                units="probability",
                norm_id="maximum",
                estimate_kind="estimate",
            ),
            ApproximationQuantity(
                "post-fit-to-pre-fit-error-ratio",
                result.post_fit_to_pre_fit_error_ratio,
                0.75,
                units="ratio",
                norm_id="scalar",
                estimate_kind="estimate",
            ),
        ),
        physicality,
        precision,
        _replay(),
        result.valid,
        artifact_arrays={
            "source-initial-state": source.initial_state,
            "source-channel-0": source.channel_kraus[0],
            "source-channel-1": source.channel_kraus[1],
            "initial-guess-initial-state": target.materialize().initial_state,
            "refitted-initial-state": result.process.initial_state,
            "refitted-channel-completeness": (
                result.process.channel_completeness_residuals
            ),
            "training-observed-probabilities": (
                result.training_observed_probabilities
            ),
            "training-refitted-probabilities": (
                result.training_fitted_probabilities
            ),
            "held-out-observed-probabilities": (
                result.held_out_observed_probabilities
            ),
            "held-out-initial-probabilities": (
                result.held_out_initial_probabilities
            ),
            "held-out-refitted-probabilities": (
                result.held_out_fitted_probabilities
            ),
            "pre-fit-probability-error": (
                result.maximum_held_out_initial_probability_error
            ),
            "post-fit-probability-error": (
                result.maximum_held_out_probability_error
            ),
            "post-fit-to-pre-fit-error-ratio": (
                result.post_fit_to_pre_fit_error_ratio
            ),
            "identifiability-singular-values": (
                result.tomography.singular_values
            ),
        },
        work={
            "optimization-iterations": 8,
            "training-experiments": len(training),
            "held-out-experiments": len(held_out),
        },
        capacity_evidence=(
            CampaignCapacityEvidence(
                "source-memory",
                4,
                4,
                saturated=False,
            ),
            CampaignCapacityEvidence(
                "retained-memory",
                2,
                2,
                saturated=False,
            ),
            CampaignCapacityEvidence(
                "intervention-settings",
                len(training),
                100_000,
                saturated=False,
            ),
        ),
        precision_policy_ids=(precision.evidence.evidence_id,),
        unsupported=("physical-arbitrary-mpo-compression",),
    )


def neural_campaign():
    gamma = 1.0

    def connected(configurations, matrix_elements, valid):
        return ConnectedConfigurations(
            (-configurations)[..., None, :],
            matrix_elements[..., None],
            valid[..., None],
            configuration_shape=(1,),
        )

    no_jump = CallableDiscreteQuantumOperator(
        lambda configurations: (
            -0.5j * gamma * (configurations[..., 0] > 0)
        ),
        lambda configurations: connected(
            configurations,
            jnp.zeros(configurations.shape[:-1], dtype=complex),
            jnp.zeros(configurations.shape[:-1], dtype=bool),
        ),
        configuration_shape=(1,),
        operator_id="amplitude-damping-no-jump",
    )
    collapse = CallableDiscreteQuantumOperator(
        lambda configurations: jnp.zeros(
            configurations.shape[:-1], dtype=complex
        ),
        lambda configurations: connected(
            configurations,
            jnp.sqrt(gamma)
            * jnp.ones(configurations.shape[:-1], dtype=complex),
            configurations[..., 0] < 0,
        ),
        configuration_shape=(1,),
        operator_id="amplitude-damping-collapse",
    )
    proposal = CallableProposal(
        lambda key, configuration: -configuration,
        lambda proposed, current: jnp.asarray(0.0),
        proposal_id="two-level-flip",
    )
    initial_populations = jnp.asarray([0.2, 0.8])
    model = _CampaignTableAmplitude(
        jnp.log(jnp.sqrt(initial_populations)).astype(complex)
    )
    vmc = VariationalMonteCarloProblem(
        model,
        no_jump,
        MetropolisHastings(proposal),
        jnp.asarray([[-1], [1], [-1], [1]], dtype=jnp.int32),
        complex_parameter_mode="nonholomorphic",
        problem_id="enumerable-neural-amplitude-damping",
    )

    def project_jump(channel, current_model, coordinates):
        del channel, current_model
        return jnp.asarray(
            [0.0, -20.0, 0.0, 0.0], dtype=coordinates.dtype
        )

    def projection_residual(channel, source_model, projected_model):
        del channel, source_model
        weights = jnp.exp(2.0 * jnp.real(projected_model.parameters))
        return weights[1] / jnp.sum(weights)

    problem = ConnectedVMCNeuralTrajectoryProblem(
        vmc,
        (collapse,),
        project_jump,
        projection_residual,
    )
    audit = audit_connected_vmc_jump_projection(
        problem,
        0,
        tolerance=1e-12,
    )
    policy = ConnectedVMCNeuralTrajectoryPolicy(
        step_size=0.1,
        steps=20,
        draws_per_step=64,
        transitions_per_draw=1,
        damping=0.1,
        rate_relative_error_tolerance=2.0,
        projection_residual_tolerance=1e-12,
        require_projected_jump=True,
    )
    result = solve_connected_vmc_neural_trajectory(
        problem, policy, jr.key(0)
    )
    rate_error = jnp.max(result.rate_standard_error_history, initial=0.0)
    rate_bias = jnp.abs(
        result.rate_history[0, 0] - gamma * initial_populations[1]
    )
    projection_error = audit.residual
    precision = _precision(
        result.final_state.parameter_coordinates.dtype, "neural-campaign"
    )
    physicality = OpenSystemPhysicalityEvidence(
        trace_residual=0.0,
        closure_residual=projection_error,
        certified_properties=("trace", "representation-closure"),
        precision_evidence=precision.evidence,
    )
    return _record(
        "enumerable-neural-v1",
        "connected-vmc-neural-trajectory",
        (
            ApproximationAxis("sample-count", 256),
            ApproximationAxis("time-step", 0.1, units="time"),
            ApproximationAxis("parameter-dimension", vmc.initial_coordinates.size),
        ),
        (
            ApproximationQuantity(
                "rate-standard-error",
                rate_error,
                1.0,
                units="rate",
                norm_id="maximum",
                estimate_kind="statistical",
                confidence=0.95,
            ),
            ApproximationQuantity(
                "initial-rate-reference-error",
                rate_bias,
                0.25,
                units="rate",
                norm_id="absolute",
                estimate_kind="estimate",
            ),
            ApproximationQuantity(
                "jump-projection-residual",
                projection_error,
                1e-12,
                units="probability",
                norm_id="absolute",
                estimate_kind="bound",
            ),
        ),
        physicality,
        precision,
        _replay(),
        result.valid & audit.valid,
        artifact_arrays={
            "parameter-history": result.parameter_history,
            "rates": result.rate_history,
            "rate-standard-errors": result.rate_standard_error_history,
            "effective-sample-size": result.effective_sample_size_history,
            "jump-decisions": result.jump_history,
            "projected-jump-observed": result.projected_jump_observed,
            "jump-channels": result.channel_history,
            "decision-uniforms": result.decision_uniform_history,
            "channel-uniforms": result.channel_uniform_history,
            "projection-residual": result.projection_residual_history,
            "audit-projected-coordinates": audit.projected_coordinates,
            "audit-projection-residual": audit.residual,
        },
        work={"steps": 20, "sample-count": 256},
        capacity_evidence=(
            CampaignCapacityEvidence(
                "parameter-dimension",
                result.final_state.parameter_coordinates.size,
                result.final_state.parameter_coordinates.size,
                saturated=False,
            ),
            CampaignCapacityEvidence(
                "sample-count",
                256,
                256,
                saturated=False,
            ),
        ),
        precision_policy_ids=(precision.evidence.evidence_id,),
        unsupported=("exact-neural-unravelling-without-closure",),
    )


__all__ = [
    "dense_trajectory_campaign",
    "distillation_campaign",
    "gaussian_campaign",
    "heom_campaign",
    "lpdo_campaign",
    "memory_campaign",
    "mps_campaign",
    "neural_campaign",
    "process_recovery_campaign",
]
