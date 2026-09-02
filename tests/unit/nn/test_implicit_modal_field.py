import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.discretization.spectral._modal_discovery import PreparedModalSupport


class _UnitCoefficient(eqx.Module):
    def __call__(self, query, *, key=None):
        del query, key
        return jnp.asarray(1.0 + 0.0j)


class _ConstantModulation(eqx.Module):
    value: float

    def __call__(self, features, *, key=None):
        del features, key
        return jnp.asarray(self.value)


def _space(count=4):
    return phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(count),),
        axis_names=("x",),
        field_name="u",
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 1.0),))


def test_exponential_spectral_envelope_preserves_zero_and_declares_aggregation():
    modes = jnp.asarray([[0.0, 0.0], [1.0, 2.0]])
    summed = phx.nn.models.ExponentialSpectralEnvelope(
        jnp.asarray([0.5, 1.0]),
        trainable=False,
        aggregation="sum",
    )
    averaged = phx.nn.models.ExponentialSpectralEnvelope(
        jnp.asarray([0.5, 1.0]),
        trainable=False,
        aggregation="mean",
    )

    assert jnp.allclose(summed(modes), jnp.asarray([1.0, jnp.exp(-2.5)]))
    assert jnp.allclose(averaged(modes), jnp.asarray([1.0, jnp.exp(-1.25)]))
    assert jnp.allclose(summed.rates, jnp.asarray([0.5, 1.0]))


def test_implicit_modal_field_composes_basis_modulation_decay_and_reality():
    space = _space()
    modulation = phx.nn.models.SpectralBasisModulation(
        _ConstantModulation(2.0),
        space,
        coarse_counts=(2,),
    )
    envelope = phx.nn.models.ExponentialSpectralEnvelope(
        jnp.asarray([0.25]),
        trainable=False,
    )
    field = phx.nn.models.ImplicitModalField(
        _UnitCoefficient(),
        space,
        envelope=envelope,
        basis_modulation=modulation,
        real_field=True,
    )

    coefficients = field(0.3, key=jr.key(1))
    expected = 2.0 * jnp.exp(-0.25 * jnp.abs(field.mode_numbers[:, 0]))
    inputs = field.model_inputs(0.3)
    state, tangent = field.time_tangent(0.3, key=jr.key(1))

    assert modulation.feature_size == 4
    assert coefficients.shape == space.modal_shape
    assert jnp.allclose(coefficients, expected)
    assert inputs.shape == (space.modal_shape[0], 2)
    assert jnp.allclose(inputs[:, 0], field.mode_numbers[:, 0])
    assert jnp.allclose(inputs[:, 1], 0.3)
    assert jnp.allclose(state, coefficients)
    assert jnp.allclose(tangent, 0.0)
    assert field.hermitian_coordinates.reality_defect(coefficients) == 0.0
    assert field.query(0.3, jnp.asarray([0, 2])).shape == (2,)
    assert field.physical_values(0.3).shape == space.physical_shape


def test_implicit_modal_field_binds_one_time_domain_and_guards_resources():
    space = _space()
    field = phx.nn.models.ImplicitModalField(_UnitCoefficient(), space)
    time = phx.domain.ScalarInterval(0.0, 1.0, label="time")
    function = field.as_domain_function(time)

    assert function.deps == ("time",)
    assert function.metadata["representation"] == "modal_coefficient"
    assert function.metadata["spectral_discretization_id"] == space.prepared_id

    with pytest.raises(ValueError, match="exceeding maximum_query_points"):
        phx.nn.models.ImplicitModalField(
            _UnitCoefficient(),
            space,
            maximum_query_points=3,
        )


def test_sparse_modal_field_ignores_padded_duplicate_indices_and_gradients():
    coefficients = jnp.asarray([[2.5], [17.0], [-9.0]])
    support = PreparedModalSupport(
        jnp.asarray([[0], [0], [1]], dtype=jnp.int32),
        coefficients,
        jnp.asarray([True, False, False]),
        jnp.asarray([6.25, 0.0, 0.0]),
        jnp.asarray([0.0, 0.0, 0.0]),
        "padded-support",
        "padded-plan",
    )
    field = phx.nn.models.SparseImplicitModalField(support, (2,))

    assert jnp.array_equal(field(), jnp.asarray([[2.5], [0.0]]))

    def zero_mode(values):
        updated = eqx.tree_at(lambda candidate: candidate.coefficients, support, values)
        return phx.nn.models.SparseImplicitModalField(updated, (2,))()[0, 0]

    gradient = jax.grad(zero_mode)(coefficients)
    assert jnp.array_equal(gradient, jnp.asarray([[1.0], [0.0], [0.0]]))
