#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""End-to-end smoke coverage for advanced generative transport contracts."""

from __future__ import annotations

import json

import jax.numpy as jnp
import jax.random as jr
import lineax as lx

import phydrax as phx


def _case(name, condition, **values):
    passed = bool(condition)
    return {"case": name, "passed": passed, **values}


def run_smoke():
    cases = []

    array_layout = phx.stochastic.ArrayEventLayout((2, 2))
    array = jnp.arange(4.0).reshape((2, 2))
    tree_layout = phx.stochastic.PyTreeEventLayout(
        {"a": jnp.zeros((2,)), "b": jnp.zeros((1,))}
    )
    tree = {"a": jnp.asarray([1.0, 2.0]), "b": jnp.asarray([3.0])}
    array_roundtrip = array_layout.from_real_coordinates(
        array_layout.to_real_coordinates(array)
    )
    tree_roundtrip = tree_layout.from_real_coordinates(
        tree_layout.to_real_coordinates(tree)
    )
    cases.append(
        _case(
            "event-layouts",
            jnp.array_equal(array_roundtrip, array)
            and all(jnp.array_equal(tree_roundtrip[key], tree[key]) for key in tree),
        )
    )

    factor = phx.uq.GaussianFactor(jnp.asarray([[1.0, 0.0], [0.3, 0.7]]))
    factor_law = phx.uq.GaussianFactorLaw(
        jnp.zeros((2,)), factor, event_shape=(2,)
    )
    factor_sample = factor_law.sample(jr.key(0), (16,))
    cases.append(
        _case(
            "gaussian-factor-law",
            jnp.all(factor_law.contains(factor_sample))
            and jnp.all(jnp.isfinite(factor_law.log_prob(factor_sample))),
        )
    )

    matrix_process = phx.stochastic.MatrixGaussianDiffusion(
        -0.2 * jnp.eye(2), 0.4 * jnp.asarray([[1.0, 0.0], [0.2, 1.0]])
    )
    marginal = matrix_process.marginal_transition(
        jnp.asarray([0.5, -0.2]), t0=0.0, t1=0.3
    )
    score = matrix_process.conditional_score(
        marginal.mean, jnp.asarray([0.5, -0.2]), t0=0.0, t1=0.3
    )
    state_dependent = phx.stochastic.StateDependentItoDiffusion(
        lambda t, x: -0.1 * x,
        lambda t, x: jnp.diag(0.5 + 0.1 * x**2),
        dimension=2,
        noise_dimension=2,
        process_id="state-dependent-smoke",
    )
    divergence = state_dependent.covariance_divergence(0.2, jnp.asarray([0.3, -0.4]))
    cases.append(
        _case(
            "matrix-state-dependent-diffusion",
            jnp.allclose(score, 0.0) and jnp.all(jnp.isfinite(divergence)),
        )
    )

    operator_term = phx.solver.WienerTerm(
        "operator-noise",
        lambda t, state, args: lx.MatrixLinearOperator(0.2 * jnp.eye(2)),
        (2,),
        structure="additive",
        representation="operator",
    )
    diagonal_term = phx.solver.WienerTerm(
        "diagonal-noise",
        lambda t, state, args: jnp.full((2,), 0.1),
        (2,),
        structure="additive",
        representation="diagonal",
    )
    problem = phx.solver.DifferentialProblem(
        lambda t, state, args: jnp.zeros_like(state),
        jnp.zeros((2,)),
        t0=0.0,
        t1=0.05,
        wiener_terms=(operator_term, diagonal_term),
    )
    realization = phx.stochastic.WienerRealization.independent(
        jr.key(1),
        problem.noise_shape,
        support=(0.0, 0.05),
        sample_shape=(4,),
        tolerance=1e-4,
    )
    solution = phx.solver.solve_diffrax_ensemble(
        problem,
        save_times=jnp.asarray([0.05]),
        realization=realization,
        dt0=0.01,
    )
    cases.append(_case("structured-wiener-blocks", jnp.all(solution.successful)))

    state_domain = phx.domain.HyperRectangle(
        jnp.full((2,), -10.0), jnp.full((2,), 10.0), label="x"
    )
    context_domain = phx.domain.HyperRectangle(
        jnp.full((2,), -10.0), jnp.full((2,), 10.0), label="observation"
    )
    domain = state_domain @ phx.domain.TimeInterval(0.0, 1.0) @ context_domain
    base_field = domain.Function("x", "t", "observation")(
        lambda x, t, observation: -x
    )
    likelihood = domain.Function("x", "t", "observation")(
        lambda x, t, observation: -0.5 * jnp.sum((x - observation) ** 2)
    )
    base = phx._score_field.StateTimeScoreField(
        base_field,
        state_label="x",
        time_label="t",
        context_labels=("observation",),
    )
    context = phx.transport.ScoreContext({"observation": jnp.asarray([1.0, -1.0])})
    guidance = phx.transport.TimeConditionedLikelihoodGuidance(
        likelihood,
        context_labels=("observation",),
    )
    guided = phx.transport.GuidedScoreField(base, (guidance,))
    guided_score, evaluations, valid = guided.evaluate(
        jnp.asarray([0.2, 0.1]), 0.5, context, key=jr.key(2)
    )
    cases.append(
        _case(
            "conditional-guidance",
            valid and len(evaluations) == 1 and jnp.all(jnp.isfinite(guided_score)),
        )
    )

    schedule = phx.stochastic.DiscreteGaussianDiffusionSchedule.linear(8)
    clean = jnp.asarray([[0.2, -0.1], [0.4, 0.7]])
    noise = jr.normal(jr.key(3), clean.shape)
    noisy = schedule.corrupt(clean, noise, jnp.asarray([2, 5]))
    recovered = schedule.clean_from_epsilon(noisy, noise, jnp.asarray([2, 5]))

    def zero_predictor(state, timestep, *, key=None):
        del timestep, key
        return jnp.zeros_like(state)

    ancestral = phx.stochastic.AncestralGaussianDiffusion(
        schedule, zero_predictor, (2,)
    ).sample(jr.key(4), (4,))
    ddim = phx.stochastic.DDIMTransport(
        schedule, zero_predictor, (2,), num_inference_steps=4
    ).sample(jr.key(5), (4,))
    cases.append(
        _case(
            "discrete-gaussian-diffusion",
            jnp.allclose(recovered, clean)
            and jnp.all(ancestral.valid)
            and ancestral.terminal_relationship == "approximate"
            and jnp.all(ddim.valid)
            and ddim.terminal_reference_id == "standard-normal",
        )
    )

    categorical = phx.stochastic.CategoricalDiffusionSchedule.uniform(5, 3)
    categories = jnp.asarray([[0, 1], [2, 1]], dtype=jnp.int32)
    corrupted = categorical.corrupt(categories, jnp.asarray([1, 3]), jr.key(6))
    posterior = categorical.posterior_probabilities(
        categories, corrupted, jnp.asarray([1, 3])
    )

    def categorical_predictor(state, timestep, *, key=None):
        del timestep, key
        return jnp.zeros(state.shape + (3,))

    categorical_sample = phx.stochastic.CategoricalReverseDiffusion(
        categorical, categorical_predictor, (2,)
    ).sample(jr.key(7), (4,))
    cases.append(
        _case(
            "categorical-diffusion",
            jnp.allclose(jnp.sum(posterior, axis=-1), 1.0)
            and jnp.all(categorical_sample.valid)
            and categorical_sample.terminal_relationship == "assumed",
        )
    )

    subspace = phx.stochastic.AffineSubspaceLayout(
        jnp.zeros((2,)), jnp.asarray([[1.0], [0.0]]), event_shape=(2,)
    )
    coefficient_law = phx.uq.DiagonalNormalLaw(
        jnp.zeros((1,)), jnp.ones((1,)), event_shape=(1,)
    )
    subspace_law = phx.stochastic.SubspaceGaussianLaw(subspace, coefficient_law)
    subspace_sample = subspace_law.sample(jr.key(8), (8,))
    cases.append(
        _case(
            "subspace-hausdorff-law",
            subspace_law.density_measure_kind == "hausdorff"
            and jnp.all(subspace_law.contains(subspace_sample)),
        )
    )

    spatial_basis = phx.stochastic.SpatialNoiseBasis(
        jnp.eye(2),
        jnp.asarray([1.0, 0.5]),
        quadrature_weights=jnp.ones((2,)),
        field_space_id="smoke-field-space",
    )
    field_geometry = phx.stochastic.FieldNoiseGeometry(spatial_basis)
    field_diffusion = phx.stochastic.FieldGaussianDiffusion(
        field_geometry, phx.stochastic.VariancePreservingDiffusion(2)
    )
    field = jnp.asarray([0.2, -0.3])
    field_sample = field_diffusion.perturb(jr.key(9), field, time=0.3)
    cases.append(_case("field-diffusion", jnp.all(jnp.isfinite(field_sample))))

    manifold = phx.metrix.EuclideanManifold((2,))
    manifold_law = phx.stochastic.ManifoldProbabilityLaw(
        manifold,
        lambda key, shape: jr.normal(key, shape + (2,)),
        lambda value: -0.5 * jnp.sum(value**2) - jnp.log(2.0 * jnp.pi),
        law_id="euclidean-manifold-normal",
    )
    manifold_process = phx.stochastic.IsotropicRiemannianDiffusion(
        manifold,
        lambda t, x: jnp.zeros_like(x),
        lambda t: jnp.asarray(1.0),
        lambda key, point: jr.normal(key, point.shape),
    )
    manifold_score = phx.stochastic.RiemannianScoreField(
        manifold, lambda x, t, key=None: -x, score_id="euclidean-score"
    )
    manifold_result = phx.stochastic.sample_manifold_reverse_diffusion(
        manifold_process,
        manifold_score,
        manifold_law.sample(jr.key(10)),
        jr.key(11),
        num_steps=8,
    )
    cases.append(_case("manifold-diffusion", manifold_result.valid))

    complex_law = phx.stochastic.ComplexNormalLaw(
        jnp.asarray([0.0 + 0.0j]), 1.0, event_shape=(1,)
    )
    complex_process = phx.stochastic.ComplexVariancePreservingDiffusion((1,))
    complex_sample = complex_law.sample(jr.key(12), (8,))
    complex_perturbed = complex_process.perturb(
        jr.key(13), complex_sample[0], time=0.4
    )
    cases.append(
        _case(
            "complex-diffusion",
            jnp.all(jnp.isfinite(complex_law.log_prob(complex_sample)))
            and jnp.all(jnp.isfinite(complex_perturbed)),
        )
    )

    graph = phx.graph.GraphIR(
        nodes={"features": jnp.asarray([[0.2], [-0.4]])},
        senders=jnp.asarray([0, 1]),
        receivers=jnp.asarray([1, 0]),
        n_node=jnp.asarray([2]),
        n_edge=jnp.asarray([2]),
    )
    graph_diffusion = phx.graph.FixedTopologyGraphDiffusion(
        graph,
        phx.stochastic.VariancePreservingDiffusion(2),
        payload_kind="nodes",
        payload_key="features",
    )
    graph_sample = graph_diffusion.perturb(graph, jr.key(14), time=0.2)
    cases.append(
        _case(
            "graph-diffusion",
            jnp.all(jnp.isfinite(graph_sample.nodes["features"])),
        )
    )

    scale = phx.atomistic.AtomisticScaleContract("angstrom", "electronvolt")
    atomistic = phx.atomistic.AtomisticBatch(
        jnp.asarray([[1, 1]], dtype=jnp.int32),
        jnp.asarray([[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]]),
        jnp.asarray([[1.0, 1.0]]),
        scale,
    )
    coordinate_diffusion = phx.atomistic.AtomisticCoordinateDiffusion(
        atomistic, phx.stochastic.VariancePreservingDiffusion(6)
    )
    hybrid = phx.atomistic.AtomisticHybridDiffusion(
        coordinate_diffusion,
        phx.stochastic.CategoricalDiffusionSchedule.uniform(4, 2),
        (1, 6),
    )
    atomistic_sample, species_sample = hybrid.perturb(
        atomistic,
        jr.key(15),
        jr.key(16),
        continuous_time=0.2,
        discrete_timestep=1,
    )
    centered = jnp.sum(atomistic_sample.positions, axis=1)
    cases.append(
        _case(
            "atomistic-hybrid-diffusion",
            jnp.allclose(centered, 0.0, atol=1e-10)
            and jnp.all(species_sample >= 0),
        )
    )

    path_layout = phx.stochastic.TrajectoryEventLayout.from_increments(
        jnp.asarray([0.0, 0.5, 1.0]), (1,)
    )
    path_diffusion = phx.stochastic.PathCoefficientDiffusion(
        path_layout,
        phx.stochastic.VariancePreservingDiffusion(path_layout.coefficient_layout.rank),
    )
    path = jnp.zeros((3, 1))
    path_sample = path_diffusion.perturb(jr.key(17), path, time=0.2)
    cases.append(
        _case(
            "path-diffusion",
            jnp.allclose(path_sample[0], path[0]) and jnp.all(jnp.isfinite(path_sample)),
        )
    )

    def encoder(value, *, key):
        del key
        location = jnp.asarray(value)
        return phx.nn.latent.LatentPosterior(
            phx.uq.DiagonalNormalLaw(
                location,
                jnp.ones_like(location) * 0.1,
                event_shape=(2,),
            ),
            jnp.asarray(True),
            "identity-encoder",
        )

    def decoder(latent, *, key):
        del key
        return phx.nn.latent.DecodedDistribution(
            None,
            latent,
            jnp.asarray(True),
            "identity-decoder",
            "sample-only",
        )

    representation = phx.nn.latent.CallableLatentRepresentation(
        encoder,
        decoder,
        data_event_shape=(2,),
        latent_event_shape=(2,),
        representation_id="identity-latent",
        density_capability="sample-only",
    )
    latent_model = phx.nn.latent.LatentDiffusion(
        representation,
        lambda key, shape: jr.normal(key, shape + (2,)),
        latent_sampler_id="normal-latent",
    )
    latent_sample = latent_model.sample(jr.key(18), (4,))
    cases.append(_case("latent-diffusion", jnp.all(latent_sample.valid)))

    target = phx.terms.EnergyTarget(
        lambda value: 0.5 * jnp.sum(value**2, axis=-1),
        (1,),
        target_id="gaussian-energy",
        normalizer_status="exact",
    )
    pcd = phx.terms.PersistentContrastiveDivergence(
        target,
        lambda key, shape: jr.normal(key, shape + (1,)),
        step_size=0.01,
        num_steps=2,
    )
    energy_state = pcd.initialize(jr.key(19), 16)
    energy_state = pcd.advance(energy_state)
    energy_loss = pcd.contrastive_loss(jnp.zeros((16, 1)), energy_state)
    cases.append(
        _case(
            "energy-model-training",
            jnp.all(energy_state.valid) and jnp.isfinite(energy_loss),
        )
    )

    autoregressive = phx.uq.AutoregressiveLaw(
        lambda prefix, index: phx.uq.Normal(
            jnp.asarray(0.0) if index == 0 else prefix[-1], 1.0
        ),
        3,
        order_id="left-to-right",
    )
    autoregressive_sample = autoregressive.sample(jr.key(20), (8,))
    cases.append(
        _case(
            "autoregressive-law",
            jnp.all(autoregressive.contains(autoregressive_sample))
            and jnp.all(jnp.isfinite(autoregressive.log_prob(autoregressive_sample))),
        )
    )

    implicit = phx.terms.ImplicitGenerator(
        lambda key, shape: jr.normal(key, shape + (2,)),
        (2,),
        generator_id="normal-implicit-generator",
    )
    fake = implicit.sample(jr.key(21), (16,))
    real = jr.normal(jr.key(22), (16, 2))
    adversarial = phx.terms.wasserstein_adversarial_evaluation(
        lambda value, key=None: jnp.sum(value),
        real,
        fake,
        jr.key(23),
        gradient_penalty_weight=1.0,
    )
    cases.append(_case("adversarial-generator", adversarial.finite))

    embedding = phx.nn.layers.SinusoidalTimeEmbedding(8)
    cases.append(
        _case(
            "diffusion-conditioning",
            embedding(jnp.asarray([0.0, 1.0])).shape == (2, 8),
        )
    )

    return {
        "capability": "advanced-generative-transport",
        "passed": all(case["passed"] for case in cases),
        "cases": cases,
    }


def main() -> int:
    report = run_smoke()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
