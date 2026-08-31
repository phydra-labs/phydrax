import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def test_subspace_law_has_hausdorff_density_and_basis_invariant_samples():
    layout = phx.stochastic.AffineSubspaceLayout(
        jnp.asarray([1.0, -1.0]),
        jnp.asarray([[1.0], [2.0]]),
        event_shape=(2,),
    )
    coefficient = phx.uq.DiagonalNormalLaw(
        jnp.zeros((1,)), jnp.ones((1,)), event_shape=(1,)
    )
    law = phx.stochastic.SubspaceGaussianLaw(layout, coefficient)
    samples = law.sample(jr.key(0), (32,))

    assert law.density_measure_kind == "hausdorff"
    assert jnp.all(law.contains(samples))
    assert jnp.all(jnp.isfinite(law.log_prob(samples)))
    assert not law.contains(jnp.asarray([1.0, 0.0]))


def test_field_diffusion_preserves_mode_coordinates_across_mesh_transfer():
    source_basis = phx.stochastic.SpatialNoiseBasis(
        jnp.eye(2),
        jnp.asarray([1.0, 0.5]),
        quadrature_weights=jnp.ones((2,)),
        field_space_id="source-space",
        mode_ids=("constant", "contrast"),
    )
    target_basis = phx.stochastic.SpatialNoiseBasis(
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        jnp.asarray([1.0, 0.5]),
        quadrature_weights=jnp.ones((2,)),
        field_space_id="target-space",
        mode_ids=("constant", "contrast"),
    )
    source = phx.stochastic.FieldNoiseGeometry(source_basis)
    target = phx.stochastic.FieldNoiseGeometry(target_basis)
    coefficients = jnp.asarray([0.3, -0.2])
    field = source.synthesize(coefficients)
    transferred = source.transfer(field, target)
    recovered, residual = target.coefficients(transferred)

    assert jnp.allclose(recovered, coefficients)
    assert residual < 1e-12

    diffusion = phx.stochastic.FieldGaussianDiffusion(
        source, phx.stochastic.VariancePreservingDiffusion(2)
    )
    perturbed = diffusion.perturb(jr.key(1), field, time=0.2)
    assert jnp.all(jnp.isfinite(perturbed))


def test_sphere_reverse_diffusion_preserves_membership_and_tangent_score():
    manifold = phx.metrix.SphereManifold(3)
    process = phx.stochastic.IsotropicRiemannianDiffusion(
        manifold,
        lambda t, point: jnp.zeros_like(point),
        lambda t: jnp.asarray(0.3),
        lambda key, point: jr.normal(key, point.shape),
    )
    score = phx.stochastic.RiemannianScoreField(
        manifold,
        lambda point, time, key=None: jnp.zeros_like(point),
        score_id="uniform-sphere-score",
    )
    result = phx.stochastic.sample_manifold_reverse_diffusion(
        process,
        score,
        jnp.asarray([1.0, 0.0, 0.0]),
        jr.key(2),
        num_steps=16,
    )

    assert result.valid
    assert jnp.allclose(jnp.linalg.vector_norm(result.final_state), 1.0, atol=1e-10)


def test_complex_normal_and_vp_diffusion_use_explicit_real_coordinate_score():
    law = phx.stochastic.ComplexNormalLaw(
        jnp.asarray([0.2 + 0.1j]),
        jnp.asarray([1.3]),
        event_shape=(1,),
    )
    samples = law.sample(jr.key(3), (16,))
    packed_score = law.score(samples, convention="real-packed")
    complex_score = law.score(samples, convention="wirtinger")

    assert packed_score.shape == (16, 2)
    assert complex_score.shape == samples.shape
    assert jnp.all(jnp.isfinite(law.log_prob(samples)))

    process = phx.stochastic.ComplexVariancePreservingDiffusion((1,))
    perturbed = process.perturb(jr.key(4), samples[0], time=0.4)
    assert jnp.iscomplexobj(perturbed)
    assert perturbed.shape == (1,)


def test_path_increment_diffusion_keeps_initial_state_and_enforces_causality():
    layout = phx.stochastic.TrajectoryEventLayout.from_increments(
        jnp.asarray([0.0, 0.5, 1.0]), (1,)
    )
    process = phx.stochastic.PathCoefficientDiffusion(
        layout,
        phx.stochastic.VariancePreservingDiffusion(layout.coefficient_layout.rank),
        score_dependency="causal",
    )
    path = jnp.asarray([[0.0], [0.2], [-0.2]])
    perturbed = process.perturb(jr.key(5), path, time=0.3)

    assert jnp.allclose(perturbed[0], path[0])
    assert jnp.all(jnp.isfinite(perturbed))
    process.require_causal_mask(jnp.tril(jnp.ones((3, 3), dtype=bool)))


def test_path_layout_keeps_padded_nodes_fixed_and_rejects_active_padding_modes():
    times = jnp.asarray([0.0, 0.5, 1.0])
    valid = jnp.asarray([True, True, False])
    layout = phx.stochastic.TrajectoryEventLayout(
        times,
        (1,),
        jnp.asarray([[0.0], [1.0], [0.0]]),
        origin=jnp.asarray([[0.0], [0.0], [7.0]]),
        valid_time=valid,
    )
    process = phx.stochastic.PathCoefficientDiffusion(
        layout,
        phx.stochastic.VariancePreservingDiffusion(1),
    )
    trajectory = jnp.asarray([[0.0], [0.3], [7.0]])
    perturbed = process.perturb(jr.key(6), trajectory, time=0.4)
    assert perturbed[-1, 0] == 7.0

    with pytest.raises(ValueError, match="vanish at padded"):
        phx.stochastic.TrajectoryEventLayout(
            times,
            (1,),
            jnp.asarray([[0.0], [1.0], [1.0]]),
            valid_time=valid,
        )
