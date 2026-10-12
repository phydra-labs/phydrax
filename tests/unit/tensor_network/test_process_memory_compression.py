from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.tensor_network import (
    CausalProcessStatus,
    CausalProcessTensor,
    CombLegSpec,
    project_process_memory_subspace,
    QuantumInstrument,
)


def _process(*, leakage: bool = False) -> CausalProcessTensor:
    rng = np.random.default_rng(51)
    factor = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    state = factor @ factor.conj().T
    state /= np.trace(state)
    unitary = np.diag(np.exp(1j * np.arange(6, dtype=np.float64) * 0.3))
    if leakage:
        unitary[[0, 2]] = unitary[[2, 0]]
    kraus = np.stack(
        (np.sqrt(0.7) * np.eye(6, dtype=np.complex128), np.sqrt(0.3) * unitary)
    )
    return CausalProcessTensor(
        CombLegSpec(2, 3, 2, spec_id="test-axes"),
        jnp.asarray(state),
        (jnp.asarray(kraus), jnp.asarray(kraus)),
        process_id="test-process",
    )


@pytest.mark.parametrize(
    "leakage", [False, True], ids=["invariant-subspace", "leaking-subspace"]
)
def test_memory_projection_matches_explicit_selector_evidence(leakage: bool) -> None:
    process = _process(leakage=leakage)
    selector = np.kron(np.eye(2, dtype=np.complex128), np.eye(3, dtype=np.complex128)[:2])
    projector = selector.conj().T @ selector
    discarded = np.eye(6, dtype=np.complex128) - projector
    initial = selector @ np.asarray(process.initial_state) @ selector.conj().T
    weight = np.trace(initial).real
    expected_channels = tuple(
        np.stack(
            [selector @ np.asarray(operator) @ selector.conj().T for operator in values]
        )
        for values in process.channel_kraus
    )
    expected_leakage = max(
        np.linalg.norm(discarded @ np.asarray(operator) @ projector)
        for values in process.channel_kraus
        for operator in values
    )
    result = project_process_memory_subspace(process, 2)
    np.testing.assert_allclose(
        result.process.initial_state, initial / weight, rtol=1e-13, atol=1e-13
    )
    for actual, expected in zip(
        result.process.channel_kraus, expected_channels, strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(result.retained_initial_weight, weight, atol=1e-13)
    np.testing.assert_allclose(result.discarded_initial_weight, 1.0 - weight, atol=1e-13)
    np.testing.assert_allclose(
        result.maximum_invariant_subspace_leakage, expected_leakage, atol=1e-13
    )
    assert bool(result.valid) is (not leakage)
    assert result.process.process_id == "test-process:memory-2"
    assert result.process.spec.spec_id == "test-axes:memory-2"
    if leakage:
        assert result.maximum_channel_completeness_residual > 0.1


def test_local_kraus_contractions_match_full_kronecker_process() -> None:
    process = _process()
    operators = np.asarray(
        [[[0.8, 0.0], [0.0, 0.6]], [[0.0, 0.8j], [0.6, 0.0]], [[5.0, 0.0], [0.0, 5.0]]],
        dtype=np.complex128,
    )
    instrument = QuantumInstrument(
        jnp.asarray(operators[None]),
        jnp.asarray([True]),
        jnp.asarray([[True, True, False]]),
        instrument_id="complex-local",
    )
    state = np.asarray(process.initial_state)
    for channel in process.channel_kraus:
        updated = np.zeros_like(state)
        for operator in operators[:2]:
            local = np.kron(operator, np.eye(3, dtype=np.complex128))
            updated += local @ state @ local.conj().T
        state = sum(
            np.asarray(operator) @ updated @ np.asarray(operator).conj().T
            for operator in channel
        )
    probability = np.trace(state).real
    reduced = np.trace(state.reshape(2, 3, 2, 3), axis1=1, axis2=3) / probability
    result = process.contract((instrument, instrument), (0, 0))
    np.testing.assert_allclose(result.final_system_state, reduced, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.probability, probability, rtol=1e-12, atol=1e-12)
    assert bool(result.valid)
    assert result.status == CausalProcessStatus.SUCCESS


def test_memory_projection_full_dimension_preserves_process() -> None:
    process = _process()
    result = project_process_memory_subspace(process, 3)
    assert result.process is process
    assert result.retained_initial_weight == 1.0
    assert result.discarded_initial_weight == 0.0
    assert bool(result.valid)


@pytest.mark.parametrize("retained", [0, 4], ids=["empty-memory", "outside-memory"])
def test_memory_projection_refuses_invalid_dimensions(retained: int) -> None:
    with pytest.raises(ValueError, match="outside the process memory"):
        project_process_memory_subspace(_process(), retained)


def test_local_kraus_zero_probability_keeps_failure_status() -> None:
    process = _process()
    instrument = QuantumInstrument(
        jnp.stack(
            (
                jnp.zeros((1, 2, 2), dtype=jnp.complex128),
                jnp.eye(2, dtype=jnp.complex128)[None],
            )
        ),
        jnp.asarray([True, True]),
        jnp.asarray([[True], [True]]),
        instrument_id="zero-outcome",
    )
    result = process.contract((instrument, instrument), (0, 0))
    assert result.probability == 0.0
    assert result.status == CausalProcessStatus.ZERO_PROBABILITY
    assert not bool(result.valid)
    assert jnp.all(result.final_system_state == 0.0)


@pytest.mark.parametrize(
    "invalid_initial", [True, False], ids=["discarded-state", "discarded-channel"]
)
def test_projection_preserves_nonfinite_discarded_evidence(invalid_initial: bool) -> None:
    process = _process()
    initial = (
        process.initial_state.at[2, 2].set(jnp.nan)
        if invalid_initial
        else process.initial_state
    )
    channels = (
        process.channel_kraus
        if invalid_initial
        else (
            process.channel_kraus[0].at[0, 2, 2].set(jnp.nan),
            process.channel_kraus[1],
        )
    )
    invalid = CausalProcessTensor(
        process.spec, initial, channels, process_id=process.process_id
    )
    result = project_process_memory_subspace(invalid, 2)
    assert not bool(result.valid)
    assert not bool(result.process.valid)
    if invalid_initial:
        assert jnp.isnan(result.retained_initial_weight)
    else:
        assert jnp.isnan(result.maximum_invariant_subspace_leakage)
