import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _basis(space, matrix, name):
    return phx.rom.ReducedBasisArtifact(
        phx.linalg.LinearSubspace(
            space,
            jnp.asarray(matrix, dtype=space.dtype),
            orthonormal=True,
            subspace_id=f"{name}-subspace",
        ),
        role="state",
        state_contract_id=f"{name}-state",
        support_id="closure-support",
        measure_id="closure-measure",
        geometry_id="closure-geometry",
        source_artifact_ids=(f"{name}-source",),
    )


def test_physical_pod_resources_and_capability_governance():
    space = phx.linalg.ArraySpace((3,), dtype=jnp.float64, space_id="pod-space")
    result = phx.ml.decomposition.PhysicalPODPlan(
        2, retained_energy=1.0, centered=False
    ).fit(
        space,
        jnp.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=jnp.float64),
        source_artifact_ids=("pod-snapshots",),
    )
    assert result.achieved_rank == 2
    assert result.target_met
    np.testing.assert_allclose(result.orthogonality_defect, 0.0, atol=1e-12)

    policy = phx.rom.ROMResourcePolicy(maximum_reduced_dimension=2)
    assert policy.admit(full_dimension=3, reduced_dimension=2)
    assert not policy.admit(full_dimension=3, reduced_dimension=3)
    declaration = phx.rom.ROMCapabilityDeclaration(
        "rom.affine-steady",
        phx.rom.ROMMaturity.EXPERIMENTAL,
        {"geometry": "fixed", "certified": False},
        required_gates=("algebra", "scientific"),
    )
    profile = declaration.profile(provider="phydrax", version="current")
    assert not profile.released


def test_rectangular_projection_affine_evolution_and_trace_lift():
    full = phx.linalg.ArraySpace((2,), dtype=jnp.float64, space_id="system-full")
    trial = _basis(full, [[1.0], [0.0]], "trial")
    test = _basis(full, [[1.0, 0.0], [0.0, 1.0]], "test")
    reduction = phx.rom.trial_test_reduction_from_bases(trial, test)
    operator = phx.linalg.DenseLinearOperator(
        jnp.eye(2, dtype=jnp.float64),
        source=full,
        target=phx.linalg.DualSpace(full),
        operator_id="rectangular-identity",
    )
    rectangular = phx.rom.RectangularLinearROMProblem(
        reduction, operator, problem_id="rectangular-problem"
    )
    result = rectangular.solve(jnp.asarray([2.0, 0.0], dtype=jnp.float64))
    np.testing.assert_allclose(result.value, np.asarray([2.0]), atol=1e-10)

    trace_space = phx.linalg.ArraySpace((1,), dtype=jnp.float64, space_id="trace")
    trace = phx.linalg.DenseLinearOperator(
        jnp.asarray([[0.0, 1.0]], dtype=jnp.float64),
        source=full,
        target=trace_space,
        operator_id="trace-operator",
    )
    lift = phx.rom.ReducedLiftArtifact(
        reduction,
        trace,
        (jnp.asarray([0.0, 1.0], dtype=jnp.float64),),
        jnp.asarray([[1.0]], dtype=jnp.float64),
        term_ids=("boundary",),
        evidence_ids=("trace-proof",),
    )
    np.testing.assert_allclose(
        lift.boundary_values(jnp.asarray([3.0])), np.asarray([3.0])
    )

    square = phx.rom.trial_test_reduction_from_bases(test)
    evolution = phx.rom.AffineEvolutionROMProblem(
        square,
        (operator,),
        (operator,),
        (jnp.zeros((2,), dtype=jnp.float64),),
        mass_term_ids=("mass",),
        operator_term_ids=("operator",),
        right_hand_side_term_ids=("rhs",),
        source_artifact_ids=("evolution-family",),
    )
    system = phx.rom.prepare_affine_evolution_rom(evolution).bind(
        jnp.asarray([1.0]), jnp.asarray([2.0]), jnp.asarray([1.0])
    )
    residual = system.residual(
        0.0,
        jnp.asarray([1.0, 2.0]),
        jnp.asarray([-2.0, -4.0]),
        None,
    )
    np.testing.assert_allclose(residual, 0.0, atol=1e-12)


def test_scm_primal_dual_bundle_and_immutable_generation(tmp_path):
    scm = phx.rom.SuccessiveConstraintArtifact(
        jnp.asarray([0.5, 0.25]),
        jnp.asarray([2.0, 2.0]),
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        jnp.asarray([0.5, 0.25]),
        family_id="scm-family",
        support_id="scm-support",
    )
    assert float(scm.lower_bound(jnp.asarray([1.0, 1.0]))) >= 0.75 - 1e-5
    output = phx.rom.PrimalDualOutputBound(
        1.0,
        0.1,
        0.2,
        0.3,
        0.5,
        output_id="heat-flux",
        evidence_ids=("primal", "dual", "stability"),
    )
    np.testing.assert_allclose(output.corrected_output, 1.1)
    np.testing.assert_allclose(output.absolute_error_bound, 0.12)

    reference = phx.rom.ROMArtifactReference(
        "model", "artifact", "recipe", "artifact.phx"
    )
    bundle = phx.rom.ROMDeploymentBundle(
        "model",
        (reference,),
        capability_profile_ids=("capability",),
        build_provenance_id="build",
        execution_requirements_id="execution",
        resource_policy_id="resources",
        qualification_ids=("qualification",),
    )
    path = phx.rom.write_rom_deployment_bundle(tmp_path / "bundle.phx", bundle)
    assert phx.rom.read_rom_deployment_bundle(path).bundle_id == bundle.bundle_id

    parent = phx.rom.ROMGeneration(
        0,
        parent_generation_id=None,
        representation_id="representation-0",
        dynamics_id="dynamics-0",
        partition_id="partition-0",
        support_id="support-0",
        qualification_ids=("qualification-0",),
    )
    child = phx.rom.ROMGeneration(
        1,
        parent_generation_id=parent.generation_id,
        representation_id="representation-1",
        dynamics_id="dynamics-1",
        partition_id="partition-1",
        support_id="support-1",
        qualification_ids=("qualification-1",),
    )
    transaction = phx.rom.EnrichmentTransaction(
        parent,
        child,
        truth_artifact_ids=("truth-1",),
        replay_artifact_id="replay-1",
    )
    assert transaction.child.generation == 1


def test_quadratic_neural_atlas_sensor_and_assimilation_contracts():
    latent = phx.linalg.ArraySpace((2,), dtype=jnp.float64, space_id="latent")
    full = phx.linalg.ArraySpace((3,), dtype=jnp.float64, space_id="chart-full")
    quadratic = phx.rom.QuadraticStateChart(
        latent,
        full,
        jnp.zeros((3,)),
        jnp.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        jnp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]),
        support_id="chart-support",
        geometry_id="chart-geometry",
    )
    decoded = quadratic.decode(jnp.asarray([2.0, 3.0]))
    np.testing.assert_allclose(decoded, np.asarray([2.0, 3.0, 13.0]))
    tangent = quadratic.jvp(jnp.asarray([2.0, 3.0]), jnp.asarray([1.0, 0.0]))
    np.testing.assert_allclose(tangent, np.asarray([1.0, 0.0, 4.0]))

    decoder = lambda code, points: jnp.stack(
        (code[0] + points[:, 0], code[1] - points[:, 0]), axis=-1
    )
    neural = phx.rom.CoordinateConditionedStateChart(
        decoder,
        jnp.asarray([[0.0], [1.0]]),
        latent,
        phx.linalg.ArraySpace((4,), dtype=jnp.float64, space_id="field"),
        decoder_id="decoder",
        support_id="query-support",
        geometry_id="query-geometry",
    )
    assert neural.decode(jnp.asarray([1.0, 2.0])).shape == (4,)

    configuration = phx.rom.SensorConfiguration(
        jnp.asarray([[0.0], [1.0]]),
        jnp.eye(2),
        channel_names=("left", "right"),
        unit_contract_id="temperature",
        frame_id="reference",
        geometry_id="geometry",
        cadence_id="unit-time",
    )
    history = phx.rom.ObservationHistory(
        jnp.asarray([[1.0, 2.0], [2.0, 3.0]]),
        jnp.asarray([0.0, 1.0]),
        jnp.ones((2, 2), dtype=bool),
        jnp.asarray([True, False]),
        configuration,
    )
    estimator = phx.rom.LinearSensorHistoryEstimator(
        jnp.eye(4),
        jnp.zeros((4,)),
        jnp.eye(4),
        history_length=2,
        configuration_id=configuration.configuration_id,
        partition_id="sensor-partition",
    )
    mean, _, admitted = estimator.estimate(history)
    assert bool(admitted)
    np.testing.assert_allclose(mean, np.asarray([1.0, 2.0, 2.0, 3.0]))

    filter_ = phx.rom.ReducedKalmanAssimilator(
        jnp.eye(2), 0.1 * jnp.eye(2), jnp.eye(2), 0.2 * jnp.eye(2)
    )
    predicted, covariance = filter_.predict(jnp.zeros((2,)), jnp.eye(2))
    updated, updated_covariance, nis = filter_.update(
        predicted, covariance, jnp.ones((2,))
    )
    assert updated.shape == (2,) and updated_covariance.shape == (2, 2)
    assert float(nis) > 0.0


def test_ssm_balancing_interpolatory_and_structure_preserving_reductions():
    evidence = phx.dynamics.identification.SpectralSubmanifoldEvidence(
        jnp.asarray([-1.0, -2.0]),
        jnp.asarray(1.0),
        spectral_quotient=2,
        minimum_resonance_detuning=0.1,
    )
    coordinates = jnp.asarray([[-1.0], [-0.5], [0.5], [1.0]], dtype=jnp.float64)
    states = jnp.concatenate((coordinates, coordinates**2), axis=1)
    rates = -coordinates
    ssm = phx.dynamics.identification.fit_spectral_submanifold(
        states,
        coordinates,
        rates,
        jnp.asarray([[1], [2]]),
        evidence,
        validity_radius=2.0,
        observation_contract_id="embedded-state",
        partition_id="ssm-partition",
    )
    np.testing.assert_allclose(
        ssm.decode(jnp.asarray([0.25])), np.asarray([0.25, 0.0625]), atol=1e-8
    )

    matrix = jnp.diag(jnp.asarray([-1.0, -2.0, -3.0]))
    input_matrix = jnp.ones((3, 1))
    output_matrix = jnp.ones((1, 3))
    balanced = phx.control.balanced_truncation(matrix, input_matrix, output_matrix, 2)
    assert balanced.matrix.shape == (2, 2)
    assert bool(balanced.stable)
    krylov = phx.control.rational_krylov_reduction(
        matrix, input_matrix, output_matrix, (1.0, 2.0)
    )
    assert krylov.matrix.shape[0] <= 2

    symplectic = phx.rom.SymplecticReduction(
        jnp.eye(2), jnp.asarray([[0.0, 1.0], [-1.0, 0.0]])
    )
    np.testing.assert_allclose(symplectic.defect, 0.0)
    port = phx.rom.PortHamiltonianReduction(
        jnp.asarray([[0.0, 1.0], [-1.0, 0.0]]),
        0.1 * jnp.eye(2),
        jnp.eye(2),
        jnp.ones((2, 1)),
        jnp.eye(2),
    )
    assert port.system_matrix().shape == (2, 2)
