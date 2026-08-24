#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def test_flow_matching_widens_before_the_event_reduction():
    precision = phx.metrix.GeometryPrecisionPolicy(
        coordinate_dtype="float32",
        compute_dtype="float32",
        accumulation_dtype="float64",
        decision_dtype="float64",
    )
    residual = jnp.concatenate(
        (
            jnp.asarray([1e8], dtype=jnp.float32),
            jnp.ones((4096,), dtype=jnp.float32),
        )
    )
    state = jnp.zeros_like(residual)
    metric = phx.terms.EuclideanFlowMatchingMetric(precision=precision)

    value = metric(state, residual, jnp.zeros_like(residual))
    reference = jnp.sum(residual.astype(jnp.float64) ** 2)
    late_cast = jnp.sum(residual**2).astype(jnp.float64)

    assert value.dtype == jnp.float64
    assert value == reference
    assert value != late_cast
    assert dict(precision.evidence_for(state).observed)["accumulation"] == "float64"


def test_newton_mixed_precision_preserves_state_and_certifies_in_float64():
    space = phx.linalg.ArraySpace((2,), dtype=jnp.float32)
    problem = phx.nonlinear.NonlinearSystemProblem(
        lambda state, _: state**2 - 2.0,
        state_space=space,
        residual_space=space,
        problem_id="mixed-precision-square-root",
    )
    precision = phx.nonlinear.NonlinearPrecisionPolicy(
        state_dtype="float32",
        residual_dtype="float32",
        accumulation_dtype="float64",
        decision_dtype="float64",
        output_dtype="float32",
    )
    result = phx.nonlinear.root(
        problem,
        jnp.full((2,), 1.5, dtype=jnp.float32),
        termination=phx.nonlinear.NonlinearTermination(
            absolute_residual=1e-5,
            relative_residual=0.0,
            maximum_steps=8,
        ),
        precision=precision,
    )

    assert bool(result.successful)
    assert result.state.dtype == jnp.float32
    assert result.diagnostics.final_residual_norm.dtype == jnp.float64
    assert result.provenance.precision_policy_id == precision.policy_id
    assert result.precision_evidence is not None
    assert dict(result.precision_evidence.observed)["certification"] == "float64"
    assert jnp.allclose(result.state, jnp.sqrt(2.0), rtol=1e-5)


def test_ssprk_precision_survives_diffrax_time_promotion_and_dense_output():
    precision = phx.solver.TemporalPrecisionPolicy(
        state_dtype="float32",
        stage_dtype="float32",
        accumulation_dtype="float64",
        decision_dtype="float64",
        checkpoint_dtype="float32",
        output_dtype="float32",
    )
    problem = phx.solver.DifferentialProblem(
        lambda time, state, _: -state,
        jnp.ones((1,), dtype=jnp.float32),
        t0=0.0,
        t1=1.0,
        problem_id="mixed-precision-decay",
    )
    solution = phx.solver.solve_diffrax(
        problem,
        save_times=jnp.asarray([0.0, 1.0]),
        solver=phx.solver.SSPRK33(precision=precision),
        dt0=0.125,
        dense=True,
        max_steps=9,
    )

    assert solution.states.dtype == jnp.float32
    assert solution.evaluate(jnp.asarray([0.25, 0.75])).dtype == jnp.float32
    assert solution.temporal_evidence is not None
    evidence = solution.temporal_evidence.precision_evidence
    assert evidence is not None
    observed = dict(evidence.observed)
    assert observed["accumulation"] == "float64"
    assert observed["checkpoint"] == "float32"
    assert observed["output"] == "float32"

    unsupported = phx.solver.SSPRK33(
        precision=phx.solver.TemporalPrecisionPolicy(
            state_dtype="float32",
            coefficient_dtype="float64",
        )
    )
    with pytest.raises(ValueError, match="Diffrax requires coefficient precision"):
        phx.solver.solve_diffrax(
            problem,
            save_times=jnp.asarray([0.0, 1.0]),
            solver=unsupported,
            dt0=0.125,
            max_steps=9,
        )


def test_geometry_results_retain_precision_evidence_and_output_dtype():
    precision = phx.metrix.GeometryPrecisionPolicy(
        coordinate_dtype="float32",
        compute_dtype="float64",
        accumulation_dtype="float64",
        decision_dtype="float64",
        output_dtype="float32",
    )
    chart = phx.metrix.CoordinateChart("plane", ("x", "y"))
    metric = phx.metrix.euclidean_metric(chart)
    point = jnp.asarray([0.3, -0.2], dtype=jnp.float32)
    tangent = jnp.asarray([0.4, -0.1], dtype=jnp.float32)

    geodesic = phx.metrix.integrate_metric_geodesic(
        metric,
        point,
        tangent,
        steps=4,
        precision=precision,
    )
    validation = phx.metrix.validate_metric(
        metric,
        point,
        precision=precision,
    )
    mean = phx.metrix.frechet_mean(
        phx.metrix.SphereManifold(3),
        jnp.asarray(
            [[1.0, 0.0, 0.0], [jnp.cos(0.2), jnp.sin(0.2), 0.0]],
            dtype=jnp.float32,
        ),
        iterations=8,
        precision=precision,
    )

    assert geodesic.endpoint.dtype == jnp.float32
    assert jnp.allclose(geodesic.endpoint, point + tangent, atol=1e-6)
    assert dict(geodesic.precision_evidence.observed)["compute"] == "float64"
    assert validation.minimum_eigenvalue.dtype == jnp.float64
    assert dict(validation.precision_evidence.observed)["certification"] == "float64"
    assert mean.point.dtype == jnp.float32
    assert dict(mean.precision_evidence.observed)["accumulation"] == "float64"


def test_cochain_and_information_geometry_use_explicit_precision_policies():
    precision = phx.metrix.GeometryPrecisionPolicy(
        coordinate_dtype="float32",
        compute_dtype="float64",
        accumulation_dtype="float64",
        decision_dtype="float64",
        output_dtype="float32",
    )
    reduced = phx.graph.cochain_metric_reduce(
        jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32),
        jnp.ones((3,), dtype=jnp.float32),
        jnp.asarray([0, 0, 1]),
        n_graph=2,
        precision=precision,
    )

    family = phx.uq.BernoulliFamily()
    information = phx.uq.ExponentialFamilyInformationGeometry(
        family,
        precision=precision,
    )
    natural = family.natural(jnp.asarray([0.0], dtype=jnp.float32))
    gradient = information.natural_gradient(
        natural,
        jnp.asarray([1.0], dtype=jnp.float32),
        policy=phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU()),
    )

    assert reduced.dtype == jnp.float64
    assert reduced == pytest.approx(2.25)
    assert gradient.dtype == jnp.float32
    assert jnp.allclose(gradient, jnp.asarray([4.0], dtype=jnp.float32))


def test_finite_volume_precision_controls_runtime_reductions_and_restart(tmp_path):
    precision = phx.discretization.FiniteVolumePrecisionPolicy(
        "float32",
        reconstruction_dtype="float32",
        flux_dtype="float32",
        reduction_dtype="float64",
        output_dtype="float32",
        checkpoint_dtype="float32",
    )
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    system = phx.equations.EulerSystem()
    discretization = phx.discretization.FiniteVolumePlan(
        grid,
        component_names=system.component_names,
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "mixed-precision-fv",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.HLLCFluxPlan(),
    )
    compiled = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        method,
        precision=precision,
    )
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        compiled.dynamics,
        phx.discretization.FluxPositivityPlan(),
    )
    primitive = jnp.broadcast_to(
        jnp.asarray([1.0, 0.1, 1.0], dtype=jnp.float32),
        (8, 3),
    )
    state = precision.storage(system.primitive_to_conserved(primitive))
    initial = phx.solver.FiniteVolumeRuntimeState(
        state,
        precision.decision(0.0),
        precision.decision(1e-4),
    )

    advanced = runtime.advance(initial)
    _, diagnostics = compiled.dynamics.residual_with_diagnostics(
        precision.decision(0.0),
        state,
    )
    case = phx.solver.FiniteVolumeCaseSpec(
        "mixed-precision-fv",
        runtime,
        phx.solver.FiniteVolumeExecutionSpec(1.0, 10),
    )
    checkpoint_plan = phx.solver.FiniteVolumeCheckpointPlan(case)
    path = tmp_path / "mixed.fvckpt"
    phx.solver.write_finite_volume_checkpoint(path, checkpoint_plan, initial)
    restored = phx.solver.read_finite_volume_checkpoint(path, checkpoint_plan)

    assert advanced.runtime_state.conservative_state.dtype == jnp.float32
    assert advanced.accepted_integrated_fluxes[0].dtype == jnp.float64
    assert diagnostics.conservation_defect.dtype == jnp.float64
    assert advanced.precision_evidence.evidence_id == precision.evidence().evidence_id
    assert dict(advanced.precision_evidence.observed)["accumulation"] == "float64"
    assert dict(advanced.precision_evidence.children)["reconstruction"].domain == (
        "finite-volume-reconstruction"
    )
    assert restored.runtime_state.conservative_state.dtype == jnp.float32
    assert restored.precision_evidence.evidence_id == precision.evidence().evidence_id


def test_hermitian_and_metric_measure_precision_are_explicit():
    hermitian_precision = phx.linalg.HermitianPrecisionPolicy(
        compute_dtype="float64",
        factorization_dtype="float64",
        accumulation_dtype="float64",
        decision_dtype="float64",
        output_dtype="float32",
    )
    matrix = jnp.asarray(
        [[2.0, 0.2 + 0.1j], [0.2 - 0.1j, 1.5]],
        dtype=jnp.complex128,
    )
    root = phx.linalg.hermitian_sqrt(
        matrix,
        precision=hermitian_precision,
    )

    integration_precision = phx.integration.IntegrationPrecisionPolicy(
        evaluation_dtype="float32",
        accumulation_dtype="float64",
        decision_dtype="float64",
        output_dtype="float32",
    )
    chart = phx.metrix.CoordinateChart("line", ("x",))
    metric = phx.metrix.euclidean_metric(chart)
    measure = phx.metrix.WeightedRiemannianMeasure(
        metric,
        lambda point: jnp.zeros(point.shape[:-1], dtype=point.dtype),
    )
    normalization = phx.integration.normalize_metric_measure(
        measure,
        jnp.asarray([[0.0], [1.0]], dtype=jnp.float32),
        jnp.asarray([0.5, 0.5], dtype=jnp.float32),
        precision=integration_precision,
    )

    assert root.value.dtype == jnp.complex64
    assert root.spectrum.eigenvectors.dtype == jnp.complex128
    assert dict(root.spectrum.precision_evidence.observed)["factorization"] == (
        "complex128"
    )
    assert normalization.mass.dtype == jnp.float32
    assert normalization.log_mass.dtype == jnp.float64
    assert dict(normalization.precision_evidence.observed)["accumulation"] == "float64"


def test_optimization_models_certificates_and_sensitivities_retain_precision():
    precision = phx.nonlinear.NonlinearPrecisionPolicy(
        state_dtype="float32",
        residual_dtype="float32",
        direction_dtype="float32",
        accumulation_dtype="float64",
        decision_dtype="float64",
        output_dtype="float32",
    )
    problem = phx.optim.NonlinearLeastSquaresProblem(
        lambda parameters, target: parameters - target,
        problem_id="precision-pounding",
    )
    termination = phx.optim.OptimizationTermination(
        absolute_optimality=1e-5,
        relative_optimality=0.0,
        maximum_steps=8,
    )
    result = phx.optim.POUNDERS(
        initial_radius=0.5,
        precision=precision,
    ).solve(
        problem,
        jnp.asarray([2.0], dtype=jnp.float32),
        termination=termination,
        args=jnp.asarray([1.0], dtype=jnp.float32),
    )

    constraint = phx.optim.NonlinearConstraint(
        lambda parameters, _: jnp.asarray(
            [jnp.sum(parameters)],
            dtype=parameters.dtype,
        ),
        lower=1.0,
        upper=1.0,
        constraint_id="sum",
    )
    constrained = phx.optim.MinimizationProblem(
        lambda parameters, target: jnp.sum((parameters - target) ** 2),
        constraints=(constraint,),
        problem_id="precision-constrained",
    )
    sensitivity = phx.optim.constrained_solution_jvp(
        constrained,
        jnp.asarray([0.2, 0.8], dtype=jnp.float32),
        jnp.asarray([0.2, 0.8], dtype=jnp.float32),
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        precision=precision,
    )
    kkt = phx.optim.solve_kkt(
        jnp.asarray([[2.0]], dtype=jnp.float32),
        jnp.asarray([[1.0]], dtype=jnp.float32),
        jnp.asarray([1.0], dtype=jnp.float32),
        jnp.asarray([0.0], dtype=jnp.float32),
        phx.optim.plan_kkt(1, 1),
        precision=precision,
    )

    assert result.parameters.dtype == jnp.float32
    assert result.diagnostics.final_optimality_norm.dtype == jnp.float64
    assert result.precision_evidence is not None
    assert set(dict(result.precision_evidence.children)) == {
        "certificate",
        "interpolation-model",
    }
    assert result.optimality_certificate.precision_evidence is not None
    assert sensitivity.linear_plan_id
    assert sensitivity.precision_evidence is not None
    assert kkt.residual_norm.dtype == jnp.float64
    assert kkt.precision_evidence is not None


def test_open_system_hierarchy_and_memory_archive_nested_precision():
    geometry = phx.metrix.GeometryPrecisionPolicy(
        coordinate_dtype="complex64",
        compute_dtype="complex64",
        accumulation_dtype="complex128",
        decision_dtype="float64",
        output_dtype="complex64",
    )
    hermitian = phx.linalg.HermitianPrecisionPolicy(
        compute_dtype="float32",
        factorization_dtype="float64",
        accumulation_dtype="float64",
        decision_dtype="float64",
        output_dtype="float32",
    )
    temporal = phx.solver.TemporalPrecisionPolicy(
        state_dtype="complex64",
        stage_dtype="complex64",
        accumulation_dtype="complex128",
        residual_dtype="complex64",
        decision_dtype="float64",
        output_dtype="complex64",
    )
    integration = phx.integration.IntegrationPrecisionPolicy(
        evaluation_dtype="float32",
        accumulation_dtype="float64",
        decision_dtype="float64",
        output_dtype="complex64",
    )
    density = jnp.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.complex64)

    base_heom = phx.solver.drude_lorentz_qubit_heom(
        0.1,
        0.5,
        density,
        depth=1,
    )
    heom_problem = phx.solver.HEOMProblem(
        base_heom.hamiltonian,
        base_heom.coupling_operator,
        base_heom.expansion,
        base_heom.hierarchy,
        base_heom.initial_state[0],
        geometry_precision=geometry,
        hermitian_precision=hermitian,
    )
    heom = phx.solver.solve_heom(
        heom_problem,
        step_size=jnp.asarray(0.01, dtype=jnp.float32),
        steps=1,
        temporal_precision=temporal,
        geometry_precision=geometry,
        hermitian_precision=hermitian,
    )

    base_memory = phx.solver.exponential_memory_qubit_problem(
        0.1,
        0.5,
        density,
    )
    memory_problem = phx.solver.MemoryKernelMasterEquation(
        base_memory.local_generator,
        base_memory.kernel,
        density,
        geometry_precision=geometry,
        hermitian_precision=hermitian,
    )
    memory = phx.solver.solve_memory_kernel(
        memory_problem,
        step_size=jnp.asarray(0.01, dtype=jnp.float32),
        steps=1,
        temporal_precision=temporal,
        integration_precision=integration,
        geometry_precision=geometry,
        hermitian_precision=hermitian,
    )

    assert heom.root_states.dtype == jnp.complex64
    assert heom.maximum_auxiliary_norm_by_level.dtype == jnp.float64
    assert heom.approximation.precision_evidence is heom.precision_evidence
    assert len(heom.approximation.precision_policy_ids) == 3
    assert "root-spectrum" in dict(heom.precision_evidence.children)
    assert memory.states.dtype == jnp.complex64
    assert memory.minimum_eigenvalues.dtype == jnp.float64
    assert "memory-quadrature" in dict(memory.precision_evidence.children)
    assert len(memory.approximation.precision_policy_ids) == 4


def test_quantum_trajectory_ensemble_uses_widened_reductions():
    geometry = phx.metrix.GeometryPrecisionPolicy(
        coordinate_dtype="complex64",
        compute_dtype="complex64",
        accumulation_dtype="complex128",
        decision_dtype="float64",
        output_dtype="complex64",
    )
    temporal = phx.solver.TemporalPrecisionPolicy(
        state_dtype="complex64",
        stage_dtype="complex64",
        accumulation_dtype="complex128",
        decision_dtype="float64",
        output_dtype="complex64",
    )
    base = phx.solver.amplitude_damping_trajectory_problem(
        0.2,
        jnp.asarray([0.0, 1.0], dtype=jnp.complex64),
    )
    problem = phx.solver.QuantumJumpProblem(
        base.hamiltonian,
        base.collapse_operators,
        base.initial_state,
        geometry_precision=geometry,
    )
    ensemble = phx.solver.solve_quantum_jump_ensemble(
        problem,
        jax.random.key(0),
        step_size=jnp.asarray(0.01, dtype=jnp.float32),
        steps=2,
        trajectory_count=4,
        temporal_precision=temporal,
        geometry_precision=geometry,
    )
    mean, error = ensemble.observable(problem.hamiltonian)

    assert ensemble.states.dtype == jnp.complex64
    assert mean.dtype == jnp.float64
    assert error.dtype == jnp.float64
    assert ensemble.approximation.precision_evidence is ensemble.precision_evidence
    assert ensemble.approximation.precision_policy_ids == (
        temporal.policy_id,
        geometry.policy_id,
    )


def test_tensor_network_factorization_precedes_storage_cast():
    precision = phx.tensor_network.TensorNetworkPrecisionPolicy(
        storage_dtype="complex64",
        contraction_dtype="complex64",
        factorization_dtype="complex128",
        accumulation_dtype="complex128",
        decision_dtype="float64",
        output_dtype="complex64",
    )
    state = phx.tensor_network.product_mps(
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.complex64),
        precision=precision,
    )
    gate = jnp.eye(4, dtype=jnp.complex64).reshape((2, 2, 2, 2))
    updated, truncation = phx.tensor_network.apply_two_site_gate(
        state,
        0,
        gate,
        maximum_bond_dimension=2,
    )

    observed = dict(truncation.precision_evidence.observed)
    assert all(tensor.dtype == jnp.complex64 for tensor in updated.tensors)
    assert updated.to_dense().dtype == jnp.complex64
    assert truncation.discarded_weight.dtype == jnp.float64
    assert observed["storage"] == "complex64"
    assert observed["factorization"] == "complex128"
    assert observed["accumulation"] == "complex128"
