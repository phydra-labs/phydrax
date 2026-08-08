import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx


def _state_space_problem():
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([0.5, 1.0]),
        jnp.asarray([[1.0], [2.0]]),
        case_ids=("only",),
        sequence_id="export-sequence",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.asarray([0.0]),
        jnp.asarray([[1.0]]),
        state_shape=(1,),
        prior_id="export-prior",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.1]]),
        state_shape=(1,),
        process_id="export-process",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.2]]),
        state_shape=(1,),
        observation_shape=(1,),
    )
    model = phx.stochastic.StateSpaceModel(
        prior, transition, observation, model_id="export-model"
    )
    return phx.stochastic.StateSpaceProblem(
        model, observations, initial_time=0.0, problem_id="export-problem"
    )


def test_filter_and_smoother_results_have_portable_archives(tmp_path):
    problem = _state_space_problem()
    kalman = phx.uq.kalman_filter(problem)
    particle = phx.uq.bootstrap_particle_filter(jr.key(70), problem, num_particles=8)
    ensemble = phx.uq.ensemble_transform_kalman_filter(
        jr.key(71), problem, ensemble_size=8
    )
    results = (
        (kalman, "kalman_filter"),
        (phx.uq.rts_smoother(kalman), "kalman_smoother"),
        (particle, "particle_filter"),
        (ensemble, "ensemble_filter"),
        (phx.uq.ensemble_kalman_smoother(ensemble), "ensemble_smoother"),
    )

    for index, (result, kind) in enumerate(results):
        destination = phx.uq.export_result(result, tmp_path / f"{index}.phxresult")
        archive = phx.uq.read_result_archive(destination)
        assert archive.kind == kind
        assert archive.metadata["problem_id"] == problem.problem_id
        assert archive.arrays
        assert all(not value.flags.writeable for value in archive.arrays.values())

    kalman_archive = phx.uq.read_result_archive(tmp_path / "0.phxresult")
    assert np.array_equal(
        kalman_archive.array("filtered_means"), np.asarray(kalman.filtered_means)
    )
    particle_archive = phx.uq.read_result_archive(tmp_path / "2.phxresult")
    assert np.array_equal(
        particle_archive.array("ancestor_indices"),
        np.asarray(particle.ancestor_indices),
    )


def _bsde_evaluation():
    times = jnp.linspace(0.0, 1.0, 3)
    increments = jnp.asarray([[[0.2], [-0.1]], [[-0.3], [0.4]]])
    states = jnp.concatenate(
        (jnp.zeros((2, 1, 1)), jnp.cumsum(increments, axis=1)), axis=1
    )
    paths = phx.stochastic.BSDEPathBatch(
        times,
        states,
        increments,
        sample_shape=(2,),
        state_shape=(1,),
        noise_shape=(1,),
        path_id="export-bsde-paths",
        process_id="export-wiener",
    )
    problem = phx.stochastic.BSDEProblem(
        lambda key: paths,
        lambda time, state, args: jnp.zeros((1,)),
        lambda time, state, args: jnp.ones((1, 1)),
        lambda time, state, value, control, args: jnp.zeros((1,)),
        lambda state, args: state,
        state_shape=(1,),
        noise_shape=(1,),
        output_shape=(1,),
        problem_id="export-bsde",
        process_id="export-wiener",
    )
    return phx.stochastic.evaluate_bsde(
        problem,
        paths,
        lambda time, state: state,
        control_predictor=lambda time, state: jnp.ones((1, 1)),
    )


def test_bsde_evaluation_has_portable_residual_and_provenance_archive(tmp_path):
    evaluation = _bsde_evaluation()
    destination = phx.uq.export_result(evaluation, tmp_path / "bsde.phxresult")
    archive = phx.uq.read_result_archive(destination)

    assert archive.kind == "bsde_evaluation"
    assert archive.metadata["path_id"] == evaluation.paths.path_id
    assert archive.metadata["control_mode"] == "explicit"
    assert np.array_equal(
        archive.array("local_residuals"), np.asarray(evaluation.local_residuals)
    )
    assert np.array_equal(
        archive.array("paths.wiener_increments"),
        np.asarray(evaluation.paths.wiener_increments),
    )


def _physical_case_state_space_problem():
    case_shape = (2, 2)
    observations = phx.stochastic.ObservationSequence(
        jnp.broadcast_to(jnp.asarray([0.5, 1.0]), case_shape + (2,)),
        jnp.linspace(-0.5, 1.0, 8).reshape(case_shape + (2, 1)),
        case_axes=("site", "replicate"),
        case_shape=case_shape,
        case_ids=(
            "site-a/replicate-1",
            "site-a/replicate-2",
            "site-b/replicate-1",
            "site-b/replicate-2",
        ),
        sequence_id="physical-export-sequence",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.zeros(case_shape + (1,)),
        jnp.asarray([[0.7]]),
        state_shape=(1,),
        prior_id="physical-export-prior",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.asarray([[0.8]]),
        jnp.asarray([[0.3]]),
        state_shape=(1,),
        offset=jnp.asarray([0.1]),
        process_id="physical-export-process",
        approximation_id="physical-export-transition",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.2]]),
        state_shape=(1,),
        observation_shape=(1,),
    )
    return phx.stochastic.StateSpaceProblem(
        phx.stochastic.StateSpaceModel(
            prior, transition, observation, model_id="physical-export-model"
        ),
        observations,
        initial_time=0.0,
        problem_id="physical-export-problem",
    )


def test_particle_fisher_archives_map_physical_cases_and_score_coordinates(
    tmp_path,
):
    problem = _physical_case_state_space_problem()
    filtered = phx.uq.bootstrap_particle_filter(
        jr.key(72),
        problem,
        num_particles=2,
        resampling_policy="never",
    )
    smoother = phx.uq.particle_backward_smoother(filtered)
    score = phx.uq.particle_fisher_score(smoother)
    fisher = phx.uq.particle_fisher_information(smoother)

    for index, (result, source_score, kind) in enumerate(
        (
            (score, score, "particle_fisher_score"),
            (fisher, fisher.score_result, "particle_fisher_information"),
        )
    ):
        destination = phx.uq.export_result(
            result, tmp_path / f"particle-fisher-{index}.phxresult"
        )
        archive = phx.uq.read_result_archive(destination)

        assert archive.kind == kind
        assert archive.metadata["case_shape"] == [2, 2]
        assert archive.metadata["case_axes"] == ["site", "replicate"]
        assert archive.metadata["case_ids"] == [
            "site-a/replicate-1",
            "site-a/replicate-2",
            "site-b/replicate-1",
            "site-b/replicate-2",
        ]
        case_mapping = tuple(
            zip(
                np.ndindex(tuple(archive.metadata["case_shape"])),
                archive.metadata["case_ids"],
                strict=True,
            )
        )
        assert case_mapping[0] == ((0, 0), "site-a/replicate-1")
        assert case_mapping[-1] == ((1, 1), "site-b/replicate-2")
        assert archive.array("case_scores").shape == (
            2,
            2,
            source_score.parameter_size,
        )

        specification = archive.trees["transition_score"]
        paths = specification["paths"]
        leaf_shapes = specification["leaf_shapes"]
        flat_slices = specification["flat_slices"]
        tree = archive.tree("transition_score")
        assert list(tree) == paths
        assert len(paths) == len(leaf_shapes) == len(flat_slices)

        reconstructed_leaves = []
        expected_start = 0
        for path, shape, flat_slice in zip(paths, leaf_shapes, flat_slices, strict=True):
            start, stop = flat_slice
            leaf = tree[path]
            assert start == expected_start
            assert list(leaf.shape) == shape
            assert stop - start == leaf.size
            reconstructed_leaves.append(leaf.reshape(-1))
            expected_start = stop
        reconstructed_score = np.concatenate(reconstructed_leaves)
        assert expected_start == source_score.parameter_size
        assert np.allclose(reconstructed_score, np.asarray(source_score.flat_score))
