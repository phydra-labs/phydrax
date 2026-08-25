import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _periodic_basis(size, mode_names):
    axis = phx.discretization.FourierAxisSpec(size).materialize(0.0, 1.0)
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
    available = {
        "constant": jnp.ones((size,)),
        "cosine": jnp.sqrt(2.0) * jnp.cos(2.0 * jnp.pi * axis.nodes),
        "sine": jnp.sqrt(2.0) * jnp.sin(2.0 * jnp.pi * axis.nodes),
    }
    modes = jnp.stack(tuple(available[name] for name in mode_names), axis=-1)
    eigenvalues = jnp.asarray([0.3, 0.12, 0.08])[: len(mode_names)]
    return phx.stochastic.SpatialNoiseBasis.from_modes(
        modes,
        eigenvalues,
        quadrature_weights=spatial.quadrature_weights,
        state_shape=spatial.physical_shape,
        mode_ids=mode_names,
        field_space_id=spatial.physical_space.field_space_id,
    )


def test_modal_spectral_noise_requires_point_value_random_field_basis():
    axis = phx.discretization.FourierAxisSpec(8).materialize(0.0, 1.0)
    spatial = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
    modal_basis = phx.stochastic.SpatialNoiseBasis.from_spectrum(
        spatial,
        0.1,
        rank=3,
    )

    with pytest.raises(ValueError, match="real point-value basis"):
        phx.stochastic.SpatialBasisSynthesis.from_spatial_noise_basis(modal_basis)


def test_gaussian_field_replays_and_matches_declared_weighted_covariance():
    basis = _periodic_basis(12, ("constant", "cosine", "sine"))
    synthesis = phx.stochastic.SpatialBasisSynthesis.from_spatial_noise_basis(
        basis,
        mean=0.4,
    )
    field = phx.stochastic.StaticGaussianRandomField(
        synthesis,
        role="coefficient",
        source="material-log-conductivity",
    )
    realization = field.realize(
        jr.key(0),
        sample_shape=(4096,),
        label="covariance-check",
    )
    first = field.sample(realization)
    replay = field.sample(realization)
    modal = synthesis.modal_coefficients(first.values)
    diagnostics = phx.stochastic.gaussian_field_diagnostics(field, realization)

    assert jnp.array_equal(first.values, replay.values)
    assert first.role == "coefficient"
    assert first.case_values.shape == (4096, 12)
    assert len(first.operator_case_provenance()) == 4096
    assert jnp.allclose(
        modal,
        jnp.sqrt(basis.eigenvalues) * realization.coefficients,
        atol=1e-12,
    )
    assert diagnostics.coefficient_covariance_relative_error < 0.06
    assert diagnostics.pointwise_variance_relative_error < 0.06
    assert diagnostics.replay_exact


def test_transformed_random_field_is_explicit_and_preserves_latent_identity():
    basis = _periodic_basis(8, ("constant", "cosine"))
    gaussian = phx.stochastic.StaticGaussianRandomField(
        phx.stochastic.SpatialBasisSynthesis.from_spatial_noise_basis(basis),
        role="initial_condition",
        source="initial-profile",
    )
    lognormal = gaussian.transform(jnp.exp, transform_id="exp-lognormal-v1")
    realization = lognormal.realize(jr.key(1), sample_shape=(32,))
    base_sample = gaussian.sample(realization)
    transformed = lognormal.sample(realization)

    assert transformed.transformed
    assert transformed.transform_id == "exp-lognormal-v1"
    assert transformed.role == "initial_condition"
    assert (
        transformed.coefficient_realization_id == base_sample.coefficient_realization_id
    )
    assert transformed.coupling_id == base_sample.coupling_id
    assert transformed.field_id != base_sample.field_id
    assert jnp.all(transformed.values > 0.0)
    assert jnp.allclose(transformed.values, jnp.exp(base_sample.values))

    malformed = gaussian.transform(
        lambda values: values[..., 0],
        transform_id="drops-space-axis",
    )
    with pytest.raises(ValueError, match="preserve the field shape"):
        malformed.sample(realization)


def test_cross_resolution_coupling_aligns_modes_not_merely_keys():
    coarse_basis = _periodic_basis(8, ("constant", "cosine"))
    fine_basis = _periodic_basis(16, ("constant", "cosine", "sine"))
    coarse = phx.stochastic.StaticGaussianRandomField(
        phx.stochastic.SpatialBasisSynthesis.from_spatial_noise_basis(coarse_basis),
        role="input",
        source="coarse-input",
    )
    fine = phx.stochastic.StaticGaussianRandomField(
        phx.stochastic.SpatialBasisSynthesis.from_spatial_noise_basis(fine_basis),
        role="input",
        source="fine-input",
    )
    coupling = phx.stochastic.GaussianFieldCoupling(
        (coarse, fine),
        label="resolution-pair",
    )
    realization = coupling.realize(jr.key(2), sample_shape=(64,))
    coarse_sample, fine_sample = coupling.sample(realization)
    coarse_coefficients = realization.select(coarse.mode_ids).coefficients
    fine_coefficients = realization.select(fine.mode_ids).coefficients

    assert coupling.common_mode_ids == ("constant", "cosine")
    assert coupling.mode_ids == ("constant", "cosine", "sine")
    assert jnp.array_equal(coarse_coefficients, fine_coefficients[..., :2])
    assert coarse_sample.values.shape == (64, 8)
    assert fine_sample.values.shape == (64, 16)
    assert coarse_sample.coupling_id == fine_sample.coupling_id == coupling.coupling_id

    uncoupled_coarse = coarse.realize(jr.key(3), sample_shape=(4,))
    uncoupled_fine = fine.realize(jr.key(3), sample_shape=(4,))
    assert uncoupled_coarse.coupling_id != uncoupled_fine.coupling_id
    with pytest.raises(ValueError, match="construct an explicit coupling"):
        fine.sample(uncoupled_coarse)


@pytest.mark.parametrize(
    "role",
    (
        "input",
        "initial_condition",
        "coefficient",
        "boundary_data",
        "forcing",
        "observation",
    ),
)
def test_static_random_field_roles_are_explicit(role):
    basis = _periodic_basis(8, ("constant",))
    field = phx.stochastic.StaticGaussianRandomField(
        phx.stochastic.SpatialBasisSynthesis.from_spatial_noise_basis(basis),
        role=role,
    )
    sample = field.sample(field.realize(jr.key(4), sample_shape=(2,)))
    assert sample.role == role


def test_static_random_field_rejects_implicit_or_unknown_semantics():
    basis = _periodic_basis(8, ("constant",))
    synthesis = phx.stochastic.SpatialBasisSynthesis.from_spatial_noise_basis(basis)
    with pytest.raises(ValueError, match="role must be"):
        phx.stochastic.StaticGaussianRandomField(synthesis, role="process")
    field = phx.stochastic.StaticGaussianRandomField(synthesis)
    with pytest.raises(ValueError, match="transform_id"):
        field.transform(jnp.exp, transform_id="")
