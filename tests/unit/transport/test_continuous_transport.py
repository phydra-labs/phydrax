import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


class _TranslationField(eqx.Module):
    velocity: jnp.ndarray

    def __call__(self, time, state, args):
        del time, args
        return jnp.broadcast_to(self.velocity, state.shape)


class _NonfiniteField(eqx.Module):
    def __call__(self, time, state, args):
        del time, args
        return jnp.full_like(state, jnp.nan)


def _normal(dimension):
    family = phx.uq.MultivariateNormalFamily(dimension)
    return family.law_from_location_covariance(
        jnp.zeros((dimension,)), jnp.eye(dimension)
    )


def _transport(field, *, dimension=2):
    layout = phx.dynamics.StateLayout((dimension,))
    system = phx.dynamics.ContinuousSystem(
        field,
        state_layout=layout,
        system_id="continuous-transport-test-system",
    )
    evolution = phx.solver.DiffraxEvolution(
        system,
        rtol=1e-8,
        atol=1e-10,
        max_steps=1024,
    )
    return phx.transport.ContinuousTransport(_normal(dimension), evolution)


def test_continuous_transport_preserves_sample_shape_and_solver_evidence():
    offset = jnp.asarray([1.25, -0.75])
    transport = _transport(_TranslationField(offset))
    result = eqx.filter_jit(transport.sample_with_diagnostics)(jr.key(2), (2, 3))

    assert result.source_states.shape == (2, 3, 2)
    assert result.final_states.shape == (2, 3, 2)
    assert result.valid.shape == (2, 3)
    assert jnp.allclose(result.final_states - result.source_states, offset, atol=2e-7)
    assert result.successful
    assert result.num_samples == 6
    assert result.evolution_id == transport.evolution.evolution_id


def test_continuous_transport_rejects_source_event_mismatch():
    layout = phx.dynamics.StateLayout((2,))
    system = phx.dynamics.ContinuousSystem(
        _TranslationField(jnp.zeros((2,))),
        state_layout=layout,
        system_id="mismatch-system",
    )
    evolution = phx.solver.DiffraxEvolution(system)

    with pytest.raises(ValueError, match="event shape"):
        phx.transport.ContinuousTransport(phx.uq.Normal(0.0, 1.0), evolution)


def test_continuous_transport_exposes_failure_and_strict_sample_rejects_it():
    transport = _transport(_NonfiniteField())
    result = transport.sample_with_diagnostics(jr.key(5), (2,))

    assert not result.successful
    assert not jnp.any(result.valid)
    with pytest.raises(eqx.EquinoxRuntimeError, match="evolution failed"):
        eqx.filter_jit(transport.sample)(jr.key(5), (2,))
