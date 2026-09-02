#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.operators.quantum._operations import LocalUnitaryOperation, QuantumProgram
from phydrax.operators.quantum._register import HilbertRegisterLayout
from phydrax.solver._process_learning import (
    fit_stinespring_process_model,
    ProcessExperimentPlan,
    QuantumDigitalTwinState,
    StinespringProcessModel,
)
from phydrax.solver._purified_lindblad import LocalKrausChannel
from phydrax.solver._quantum_compilation import (
    compile_quantum_program,
    ControlHamiltonianTerm,
    discretize_fixed_grid_control,
    FixedGridQuantumControl,
    HardwareTopology,
    QuantumCompilationPolicy,
)
from phydrax.solver._quantum_experiment import (
    ClassicalRegisterLayout,
    estimate_quantum_experiment_gradient,
    execute_quantum_experiment_exact,
    prepare_quantum_experiment,
    QuantumExperimentProgram,
    sample_quantum_experiment,
)
from phydrax.solver._quantum_measurement import (
    apply_mps_quantum_instrument,
    QuantumInstrument,
)
from phydrax.solver._quantum_program import DenseQuantumProgramPolicy
from phydrax.solver._quantum_service import (
    admit_quantum_service_request,
    QuantumProgramInterchange,
    QuantumServicePolicy,
    QuantumServiceRequest,
    record_quantum_service_run,
)
from phydrax.solver._tensor_open_quantum import (
    evolve_lpdo_local_channels,
    LPDOChannelEvolutionPlan,
    MPOHamiltonian,
    MPOLindbladian,
)
from phydrax.tensor_network._core import (
    LocallyPurifiedDensity,
    MatrixProductOperator,
    MatrixProductState,
)


_C64 = jnp.complex64
_I = jnp.eye(2, dtype=_C64)
_X = jnp.asarray([[0, 1], [1, 0]], dtype=_C64)
_H = jnp.asarray([[1, 1], [1, -1]], dtype=_C64) / jnp.sqrt(2.0)
_P0 = jnp.asarray([[1, 0], [0, 0]], dtype=_C64)
_P1 = jnp.asarray([[0, 0], [0, 1]], dtype=_C64)


def _experiment():
    layout = HilbertRegisterLayout(("q0",), (2,))
    prefix = QuantumProgram(
        layout,
        (LocalUnitaryOperation(_H, ("q0",)),),
        state_kind="state-vector",
    )
    branches = (
        QuantumProgram(
            layout,
            (LocalUnitaryOperation(_X, ("q0",)),),
            state_kind="density-matrix",
        ),
        QuantumProgram(
            layout,
            (LocalUnitaryOperation(_I, ("q0",)),),
            state_kind="density-matrix",
        ),
    )
    instrument = QuantumInstrument(
        jnp.stack((_P0, _P1))[:, None, :, :],
        jnp.ones((2, 1), dtype=bool),
        tolerance=1e-5,
    )
    experiment = QuantumExperimentProgram(
        prefix,
        instrument,
        branches,
        ClassicalRegisterLayout(("readout",), (1,), maximum_total_bits=1),
        (0, 1),
        ((0,), (1,)),
        branch_capacity=2,
    )
    prepared = prepare_quantum_experiment(
        experiment, DenseQuantumProgramPolicy(compute_dtype="complex64")
    )
    return experiment, prepared


def test_dense_branch_oracle_feed_forward_shots_and_replay():
    experiment, prepared = _experiment()
    exact = execute_quantum_experiment_exact(prepared, jnp.asarray([1, 0], dtype=_C64))
    assert bool(exact.valid)
    assert jnp.allclose(exact.instrument_result.probabilities, 0.5, atol=2e-5)
    expected = jnp.asarray([[0, 0], [0, 1]], dtype=_C64)
    assert jnp.allclose(exact.branch_densities[0], expected, atol=2e-5)
    assert jnp.allclose(exact.branch_densities[1], expected, atol=2e-5)
    assert exact.register_values_by_outcome.tolist() == [[0], [1]]

    key = jr.key(91)
    whole = sample_quantum_experiment(exact, shots=4000, key=key)
    first = sample_quantum_experiment(exact, shots=1300, key=key)
    second = sample_quantum_experiment(
        exact, shots=2700, first_shot_address=1300, key=key
    )
    assert jnp.array_equal(
        whole.outcomes, jnp.concatenate((first.outcomes, second.outcomes))
    )
    assert jnp.all(jnp.abs(whole.counts / 4000.0 - 0.5) < 0.04)
    evidence = estimate_quantum_experiment_gradient(
        exact,
        whole,
        jnp.asarray([[-0.5], [0.5]]),
        jnp.asarray([0.0, 1.0]),
    )
    assert bool(evidence.valid)
    assert jnp.allclose(evidence.exact_gradient, jnp.asarray([0.5]))
    assert evidence.standard_error.shape == (1,)

    payload = QuantumProgramInterchange(experiment.prefix)
    assert payload.materialize().program_id == experiment.prefix.program_id


def test_zero_probability_evidence_and_mps_mixed_outcome_refusal():
    _, prepared = _experiment()
    identity_prefix = QuantumProgram(
        prepared.program.prefix.layout,
        (LocalUnitaryOperation(_I, ("q0",)),),
        state_kind="state-vector",
    )
    zero_experiment = QuantumExperimentProgram(
        identity_prefix,
        prepared.program.instrument,
        prepared.program.branch_programs,
        prepared.program.classical_layout,
        (0, 1),
        ((0,), (1,)),
        branch_capacity=2,
    )
    zero_prepared = prepare_quantum_experiment(
        zero_experiment, DenseQuantumProgramPolicy(compute_dtype="complex64")
    )
    exact = execute_quantum_experiment_exact(
        zero_prepared, jnp.asarray([1, 0], dtype=_C64)
    )
    assert bool(exact.valid)
    assert exact.zero_probability.tolist() == [False, True]
    assert not bool(exact.normalization_applied[1])

    mixed = QuantumInstrument(
        jnp.stack((jnp.sqrt(0.5) * _I, jnp.sqrt(0.5) * _X))[None, ...],
        jnp.ones((1, 2), dtype=bool),
        tolerance=1e-5,
    )
    mps = MatrixProductState((jnp.asarray([[[1], [0]]], dtype=_C64),))
    with pytest.raises(ValueError, match="multi-Kraus mixed outcomes"):
        apply_mps_quantum_instrument(mixed, mps, 0)


def test_explicit_route_ledger_and_fixed_grid_controls():
    layout = HilbertRegisterLayout(("q0", "q1", "q2"), (2, 2, 2))
    cnot = jnp.asarray(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=_C64,
    )
    program = QuantumProgram(
        layout,
        (LocalUnitaryOperation(cnot, ("q0", "q2")),),
        state_kind="state-vector",
    )
    topology = HardwareTopology(
        ("p0", "p1", "p2"),
        (("p0", "p1"), ("p1", "p2")),
        ("swap", "unitary-1q", "unitary-2q"),
    )
    compiled = compile_quantum_program(
        program,
        topology,
        QuantumCompilationPolicy(route_strategy="swap", maximum_swaps=2),
    )
    assert bool(compiled.valid)
    assert compiled.swap_count == 1
    assert compiled.ledger[0].swap_edges == (("p0", "p1"),)
    assert compiled.ledger[0].emitted_native_gates == ("swap", "unitary-2q")
    assert len(compiled.compiled_program.operations) == 2

    single = HilbertRegisterLayout(("q",), (2,))
    control = FixedGridQuantumControl(
        single,
        (ControlHamiltonianTerm(_X, ("q",), tolerance=1e-5),),
        jnp.asarray([0.0, 0.2, 0.5]),
        jnp.asarray([[1.0], [2.0]]),
    )
    discretized = discretize_fixed_grid_control(control)
    assert bool(discretized.valid)
    assert jnp.allclose(discretized.grid_intervals, jnp.asarray([0.2, 0.3]))
    assert len(discretized.program.operations) == 2
    assert jnp.all(discretized.step_unitarity_residuals < 1e-5)


def test_lpdo_channel_reports_cp_tp_psd_trace_and_truncation_separately():
    lpdo = LocallyPurifiedDensity((jnp.asarray([[[[1]], [[0]]]], dtype=_C64),))
    kraus = jnp.stack((jnp.sqrt(0.7) * _I, jnp.sqrt(0.3) * _X))
    channel = LocalKrausChannel(0, kraus, channel_id="bit-flip")
    result = evolve_lpdo_local_channels(
        lpdo,
        LPDOChannelEvolutionPlan(
            (channel,),
            steps=2,
            maximum_purification_dimension=4,
            trace_preservation_tolerance=1e-5,
            trace_tolerance=1e-5,
            maximum_discarded_weight=1e-5,
        ),
    )
    evidence = result.evidence
    assert jnp.all(evidence.completely_positive_by_construction)
    assert jnp.all(evidence.trace_preserving_channels)
    assert jnp.all(evidence.positive_semidefinite_by_construction)
    assert bool(evidence.trace_within_tolerance)
    assert bool(evidence.truncation_within_budget)
    assert bool(evidence.valid)
    hamiltonian = MPOHamiltonian(
        MatrixProductOperator((_I.reshape((1, 2, 2, 1)),)),
        tolerance=1e-5,
    )
    lindbladian = MPOLindbladian(
        MatrixProductOperator((jnp.zeros((1, 4, 4, 1), dtype=_C64),)),
        tolerance=1e-5,
    )
    assert bool(hamiltonian.valid)
    assert bool(lindbladian.trace_preserving_generator)
    assert bool(lindbladian.valid)


def test_process_fit_holdout_checkpoint_and_service_refusal():
    identity_model = StinespringProcessModel(
        _I, dimension=2, environment_dimension=1, tolerance=1e-5
    )
    inputs = jnp.stack((_P0, _P1))
    effects = jnp.broadcast_to(jnp.stack((_P0, _P1)), (2, 2, 2, 2))
    counts = jnp.asarray([[20, 0], [0, 20]], dtype=jnp.int32)
    experiments = ProcessExperimentPlan(inputs, effects, counts, tolerance=1e-5)
    fit = fit_stinespring_process_model(
        identity_model,
        experiments,
        experiments,
        iterations=2,
        learning_rate=0.05,
    )
    checkpoint = QuantumDigitalTwinState(fit)
    assert bool(fit.valid)
    assert jnp.isfinite(fit.held_out_negative_log_likelihood)
    assert fit.stiefel_residual < 1e-5
    assert int(checkpoint.completed_iterations) == 2

    experiment, _ = _experiment()
    topology = HardwareTopology(("p0",), (), ("unitary-1q",))
    request = QuantumServiceRequest(experiment, topology, requested_shots=101)
    admission = admit_quantum_service_request(
        request,
        QuantumServicePolicy(
            maximum_wires=1,
            maximum_operations=3,
            maximum_branches=2,
            maximum_shots=100,
            maximum_classical_bits=1,
            allowed_topology_ids=(topology.topology_id,),
        ),
    )
    assert not bool(admission.accepted)
    assert admission.refusal_codes == ("shot-capacity-exceeded",)
    record = record_quantum_service_run(
        admission, None, None, logical_start_tick=7, logical_finish_tick=7
    )
    assert record.status == "refused"
    assert not bool(record.executed)
