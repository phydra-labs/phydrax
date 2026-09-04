import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def test_fixed_graph_and_atomistic_diffusion_preserve_structural_invariants():
    graph = phx.graph.GraphIR(
        nodes={"features": jnp.asarray([[0.2, 0.1], [-0.4, 0.3]])},
        senders=jnp.asarray([0, 1]),
        receivers=jnp.asarray([1, 0]),
        n_node=jnp.asarray([2]),
        n_edge=jnp.asarray([2]),
        node_mask=jnp.asarray([True, False]),
    )
    graph_diffusion = phx.graph.FixedTopologyGraphDiffusion(
        graph,
        phx.stochastic.VariancePreservingDiffusion(4),
        payload_kind="nodes",
        payload_key="features",
    )
    perturbed_graph = graph_diffusion.perturb(graph, jr.key(0), time=0.2)
    assert jnp.array_equal(perturbed_graph.senders, graph.senders)
    assert jnp.array_equal(perturbed_graph.receivers, graph.receivers)
    assert jnp.array_equal(perturbed_graph.node_mask, graph.node_mask)
    graph_score = graph_diffusion.conditional_score(perturbed_graph, graph, time=0.2)
    assert jnp.all(graph_score[1] == 0.0)
    with pytest.raises(ValueError, match="exactly the template topology"):
        graph_diffusion.perturb(
            graph.replace(node_mask=jnp.asarray([True, True])),
            jr.key(9),
            time=0.2,
        )

    loss = phx.graph.graph_denoising_loss(
        graph_diffusion,
        lambda noisy, time, key=None: (
            graph_diffusion.conditional_score(noisy, graph, time=time) + 1.0
        ),
        graph,
        jr.key(11),
        time=0.2,
    )
    assert jnp.allclose(loss, 1.0)
    scale = phx.atomistic.AtomisticScaleContract(
        phx.units.ANGSTROM, phx.units.ELECTRONVOLT
    )
    batch = phx.atomistic.AtomisticBatch(
        jnp.asarray([[1, 1]], dtype=jnp.int32),
        jnp.asarray([[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]]),
        jnp.asarray([[1.0, 1.0]]),
        scale,
    )
    coordinate = phx.atomistic.AtomisticCoordinateDiffusion(
        batch, phx.stochastic.VariancePreservingDiffusion(6)
    )
    hybrid = phx.atomistic.AtomisticHybridDiffusion(
        coordinate,
        phx.stochastic.CategoricalDiffusionSchedule.uniform(4, 2),
        (1, 6),
    )
    perturbed_batch, species = hybrid.perturb(
        batch,
        jr.key(1),
        jr.key(2),
        continuous_time=0.2,
        discrete_timestep=1,
    )
    assert jnp.allclose(jnp.sum(perturbed_batch.positions, axis=1), 0.0, atol=1e-10)
    assert jnp.all(species >= 0)


def test_atomistic_equivariance_report_permutes_every_atom_aligned_field():
    batch = phx.atomistic.AtomisticBatch(
        jnp.asarray([[1, 6]], dtype=jnp.int32),
        jnp.asarray([[[-0.7, 0.2, 0.0], [0.4, -0.1, 0.0]]]),
        jnp.asarray([[1.0, 12.0]]),
        phx.atomistic.AtomisticScaleContract(phx.units.ANGSTROM, phx.units.ELECTRONVOLT),
    )

    def centered_score(value, time):
        del time
        mass = jnp.where(value.atom_mask, value.masses, 0.0)
        center = jnp.sum(mass[..., None] * value.positions, axis=1, keepdims=True)
        center = center / jnp.sum(mass, axis=1, keepdims=True)[..., None]
        return value.positions - center

    rotation = jnp.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    report = phx.atomistic.atomistic_score_equivariance(
        centered_score,
        batch,
        0.3,
        rotation,
        jnp.asarray([1, 0]),
    )

    assert report.valid
    assert report.rotation_defect < 1e-12
    assert report.permutation_defect < 1e-12
    assert report.translation_residual < 1e-12


def test_atomistic_conditional_score_uses_centered_noise_pseudoinverse():
    masses = jnp.asarray([[1.0, 12.0]])
    batch = phx.atomistic.AtomisticBatch(
        jnp.asarray([[1, 6]], dtype=jnp.int32),
        jnp.asarray([[[-0.7, 0.2, 0.0], [0.4, -0.1, 0.0]]]),
        masses,
        phx.atomistic.AtomisticScaleContract(phx.units.ANGSTROM, phx.units.ELECTRONVOLT),
    )
    process = phx.stochastic.VariancePreservingDiffusion(6)
    diffusion = phx.atomistic.AtomisticCoordinateDiffusion(batch, process)
    perturbed = diffusion.perturb(batch, jr.key(10), time=0.25)
    score = diffusion.conditional_score(perturbed, batch, time=0.25)

    weights = masses[0] / jnp.sum(masses[0])
    projection = jnp.eye(2) - jnp.ones((2, 1)) * weights[None, :]
    covariance = projection @ projection.T
    clean_centered = batch.positions - jnp.sum(
        weights[None, :, None] * batch.positions, axis=1, keepdims=True
    )
    unconstrained = process.conditional_score(
        perturbed.positions.reshape((-1,)),
        clean_centered.reshape((-1,)),
        t1=0.25,
    ).reshape((2, 3))
    expected = jnp.linalg.pinv(covariance) @ unconstrained

    assert jnp.allclose(score[0], expected, rtol=1e-9, atol=1e-10)
    assert jnp.allclose(jnp.sum(masses[..., None] * score, axis=1), 0.0, atol=1e-10)


def test_latent_diffusion_composes_sample_only_decoder_without_density_claim():
    def encoder(value, *, key):
        del key
        location = jnp.asarray(value)
        return phx.nn.latent.LatentPosterior(
            phx.uq.DiagonalNormalLaw(
                location,
                jnp.full(location.shape, 0.1),
                event_shape=(2,),
            ),
            jnp.asarray(True),
            "encoder",
        )

    def decoder(latent, *, key):
        del key
        return phx.nn.latent.DecodedDistribution(
            None,
            latent,
            jnp.asarray(True),
            "decoder",
            "sample-only",
        )

    representation = phx.nn.latent.CallableLatentRepresentation(
        encoder,
        decoder,
        data_event_shape=(2,),
        latent_event_shape=(2,),
        representation_id="identity-representation",
        density_capability="sample-only",
    )
    model = phx.nn.latent.LatentDiffusion(
        representation,
        lambda key, shape: jr.normal(key, shape + (2,)),
        latent_sampler_id="normal-latent",
    )
    sample = model.sample(jr.key(3), (4,))
    assert jnp.all(sample.valid)
    assert sample.decoded.law is None
    assert sample.latent.shape == (4, 2)


def test_energy_autoregressive_and_adversarial_contracts_remain_distinct():
    target = phx.terms.EnergyTarget(
        lambda value: 0.5 * jnp.sum(value**2, axis=-1),
        (1,),
        target_id="normal-energy",
        normalizer_status="unknown",
    )
    pcd = phx.terms.PersistentContrastiveDivergence(
        target,
        lambda key, shape: jr.normal(key, shape + (1,)),
        step_size=0.01,
        num_steps=2,
    )
    state = pcd.advance(pcd.initialize(jr.key(4), 16))
    assert jnp.all(state.valid)
    assert jnp.isfinite(pcd.contrastive_loss(jnp.zeros((16, 1)), state))

    autoregressive = phx.uq.AutoregressiveLaw(
        lambda prefix, index: phx.uq.Normal(
            jnp.asarray(0.0) if index == 0 else prefix[-1], 1.0
        ),
        3,
        order_id="left-to-right",
    )
    sequence = autoregressive.sample(jr.key(5), (8,))
    assert jnp.all(autoregressive.contains(sequence))
    assert jnp.all(jnp.isfinite(autoregressive.log_prob(sequence)))

    implicit = phx.terms.ImplicitGenerator(
        lambda key, shape: jr.normal(key, shape + (2,)),
        (2,),
        generator_id="sample-only-normal",
    )
    fake = implicit.sample(jr.key(6), (8,))
    real = jr.normal(jr.key(7), (8, 2))
    evaluation = phx.terms.wasserstein_adversarial_evaluation(
        lambda value, key=None: jnp.sum(value),
        real,
        fake,
        jr.key(8),
        gradient_penalty_weight=1.0,
    )
    assert evaluation.finite


def test_time_embedding_has_fixed_dimension_and_endpoint_values():
    embedding = phx.nn.layers.SinusoidalTimeEmbedding(8)
    values = embedding(jnp.asarray([0.0, 1.0]))
    assert values.shape == (2, 8)
    assert jnp.allclose(values[0, :4], 0.0)
    assert jnp.allclose(values[0, 4:], 1.0)
