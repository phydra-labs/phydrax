import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.closure_data._binding import LearnedClosureBindingPlan
from phydrax.closure_data._state import FlowStateSchema
from phydrax.discretization import spectral


def _fourier(shape=(6, 8)):
    return phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(size) for size in shape),
        axis_names=tuple(f"x{axis}" for axis in range(len(shape))),
        field_name="u",
    ).prepare(
        tuple(phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi) for _ in shape)
    )


@pytest.mark.parametrize("factor", (1.0, 0.5, jnp.inf, jnp.nan))
def test_oversampling_dealiasing_requires_finite_factor_above_one(factor):
    with pytest.raises(ValueError, match="finite and greater than one"):
        phx.discretization.OversamplingDealiasingPlan(factor)


def test_oversampling_dealiasing_validates_resource_and_spectral_family():
    with pytest.raises(TypeError, match="real number"):
        phx.discretization.OversamplingDealiasingPlan("1.5")
    with pytest.raises(TypeError, match="must be an integer"):
        phx.discretization.OversamplingDealiasingPlan(
            1.5,
            maximum_evaluation_modes=12.5,
        )
    with pytest.raises(ValueError, match="must be positive"):
        phx.discretization.OversamplingDealiasingPlan(
            1.5,
            maximum_evaluation_modes=0,
        )

    fourier = _fourier((8,))
    with pytest.raises(ValueError, match="maximum_evaluation_modes"):
        phx.discretization.OversamplingDealiasingPlan(
            1.5,
            maximum_evaluation_modes=11,
        ).prepare(fourier, required_polynomial_degree=None)

    bounded = phx.discretization.TensorSpectralPlan(
        (phx.discretization.ChebyshevBasisPlan(8),),
        axis_names=("x",),
    ).prepare((phx.discretization.AxisDomain.interval(-1.0, 1.0),))
    with pytest.raises(ValueError, match="tensor Fourier bases only"):
        phx.discretization.OversamplingDealiasingPlan(
            1.5,
            maximum_evaluation_modes=64,
        ).prepare(bounded, required_polynomial_degree=None)
    with pytest.raises(TypeError, match="tensor spectral discretization"):
        phx.discretization.OversamplingDealiasingPlan(
            1.5,
            maximum_evaluation_modes=64,
        ).prepare(object(), required_polynomial_degree=None)


def test_prepared_oversampling_transfers_shapes_constants_and_retained_modes():
    retained = _fourier()
    prepared = phx.discretization.OversamplingDealiasingPlan(
        1.5,
        maximum_evaluation_modes=108,
    ).prepare(retained, required_polynomial_degree=None)
    coefficients = jnp.arange(96, dtype=jnp.float64).reshape((6, 8, 2))
    coefficients = coefficients + 1j * jnp.flip(coefficients, axis=(0, 1))

    embedded = prepared.embed(coefficients)
    restricted = prepared.restrict(embedded)
    retained_constant = retained.project(jnp.ones(retained.physical_shape))
    evaluation_values = prepared.reconstruct(retained_constant)
    projected_constant = prepared.project(jnp.ones(prepared.evaluation.physical_shape))

    assert prepared.retained is retained
    assert prepared.evaluation.modal_shape == (9, 12)
    assert embedded.shape == (9, 12, 2)
    assert jnp.allclose(prepared.filter(coefficients), coefficients)
    assert jnp.allclose(restricted, coefficients)
    assert jnp.allclose(evaluation_values, 1.0)
    assert jnp.allclose(retained.reconstruct(projected_constant), 1.0)
    assert prepared.report.kind == "oversampling"
    assert prepared.report.retained_shape == (6, 8)
    assert prepared.report.evaluation_shape == (9, 12)
    assert not prepared.report.exact
    assert "nonpolynomial" in prepared.report.reason
    assert (
        phx.discretization.OversamplingDealiasingPlan
        is spectral.OversamplingDealiasingPlan
    )


def test_learned_spectral_drift_rejects_identity_oversampling_filter():
    space = _fourier((6, 6))
    projector = phx.discretization.PeriodicLerayProjector(space)
    coordinates = phx.discretization.HermitianSpectralCoordinates(
        space,
        component_shape=(2,),
    )
    dealiasing = phx.discretization.OversamplingDealiasingPlan(
        1.5,
        maximum_evaluation_modes=81,
    ).prepare(space, required_polynomial_degree=None)
    schema = FlowStateSchema(
        ("u", "v"),
        ("m/s", "m/s"),
        (1.0, 1.0),
        velocity_names=("u", "v"),
    )
    binding = LearnedClosureBindingPlan(
        lambda value, args: value,
        deployment_kind="spectral_drift",
        schema_id=schema.schema_id,
        input_component_names=("u", "v"),
        output_component_names=("u", "v"),
        model_artifact_id="model",
        normalizer_provenance_id="normalizer",
    )

    with pytest.raises(ValueError, match="filter action is identity"):
        binding.bind_spectral_drift(
            schema,
            projector,
            coordinates,
            dealiasing,
        )
