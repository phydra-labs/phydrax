#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable, Sequence
from typing import NoReturn

import jax.numpy as jnp
import pytest
from jax import Array
from jax.typing import ArrayLike

from phydrax.tensor_network import _peps
from phydrax.tensor_network._contraction import (
    ContractionPlan,
    ContractionResourcePolicy,
    PreparedContraction,
)
from phydrax.tensor_network._contraction_provider import (
    ContractionExecutionPolicy,
    ContractionProvider,
)
from phydrax.tensor_network._precision import TensorNetworkPrecisionPolicy


def _exact_case(
    kind: str,
) -> tuple[
    Callable[..., _peps.PEPSContractionResult],
    tuple[_peps.PEPS | _peps.PEPO | tuple[int, ...], ...],
    Array,
    TensorNetworkPrecisionPolicy,
]:
    precision = TensorNetworkPrecisionPolicy(
        storage_dtype="complex64",
        contraction_dtype="complex64",
        accumulation_dtype="complex64",
        output_dtype="complex64",
    )
    left = jnp.asarray([[1 + 2j, -0.5j], [0.75, 2 - 1j]], dtype=jnp.complex64)
    right = jnp.asarray([[0.5, 1 + 0.25j], [-1j, 1.5]], dtype=jnp.complex64)
    state = _peps.PEPS(
        (left.reshape(1, 2, 1, 1, 2), right.reshape(1, 1, 1, 2, 2)),
        1,
        2,
        precision=precision,
    )
    dense = left.T @ right
    if kind == "norm":
        return _peps.contract_peps_exact, (state,), jnp.vdot(dense, dense), precision
    if kind == "overlap":
        other_left = left + jnp.asarray(0.25 - 0.5j, dtype=jnp.complex64)
        other = _peps.PEPS(
            (other_left.reshape(1, 2, 1, 1, 2), right.reshape(1, 1, 1, 2, 2)),
            1,
            2,
            precision=precision,
        )
        expected = jnp.vdot(dense, other_left.T @ right)
        return _peps.contract_peps_exact, (state, other), expected, precision
    if kind == "amplitude":
        return _peps.peps_amplitude, (state, (1, 0)), dense[1, 0], precision
    first = jnp.stack((left, right))
    second = jnp.stack((right, left * jnp.asarray(0.5j, dtype=jnp.complex64)))
    operator = _peps.PEPO(
        (first.reshape(1, 2, 1, 1, 2, 2), second.reshape(1, 1, 1, 2, 2, 2)),
        1,
        2,
        precision=precision,
    )
    dense_operator = jnp.tensordot(first, second, axes=(0, 0))
    dense_operator = dense_operator.transpose(0, 2, 1, 3).reshape(4, 4)
    return (
        _peps.contract_pepo_trace_exact,
        (operator,),
        jnp.trace(dense_operator),
        precision,
    )


@pytest.mark.parametrize("kind", ("norm", "overlap", "amplitude", "trace"))
def test_exact_peps_native_execution_oracle_and_identity(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, arguments, expected, precision = _exact_case(kind)
    resources = ContractionResourcePolicy(maximum_workspace_bytes=2**20)
    execution = ContractionExecutionPolicy()
    changed_execution = ContractionExecutionPolicy(maximum_workspace_bytes=2**18)
    prepare = _peps.prepare_contraction
    seen: list[ContractionExecutionPolicy | None] = []

    def record_preparation(
        plan: ContractionPlan,
        operands: Sequence[ArrayLike],
        *,
        execution: ContractionExecutionPolicy | None = None,
    ) -> PreparedContraction:
        assert plan.resources is resources
        assert plan.precision is precision
        seen.append(execution)
        return prepare(plan, operands, execution=execution)

    monkeypatch.setattr(_peps, "prepare_contraction", record_preparation)
    default = contract(*arguments, resources=resources)
    explicit = contract(*arguments, resources=resources, execution=execution)
    changed = contract(*arguments, resources=resources, execution=changed_execution)
    repeated = contract(*arguments, resources=resources, execution=changed_execution)
    assert seen[0] is None
    assert seen[1] is execution
    assert seen[2] is changed_execution and seen[3] is changed_execution
    for result in (default, explicit, changed, repeated):
        assert jnp.allclose(result.value, expected, rtol=2e-5, atol=2e-5)
        assert result.value.dtype == jnp.dtype("complex64")
        assert result.evidence.exact
        assert not result.evidence.global_error_bound_claimed
        assert result.evidence.contraction.provider == "native"
        assert result.evidence.contraction.external_workspace_bound_bytes == 0
        assert result.evidence.contraction.external_cache_bound_bytes == 0
    assert default.evidence.replay_id == explicit.evidence.replay_id
    assert (
        default.evidence.contraction.execution_id
        == explicit.evidence.contraction.execution_id
    )
    assert changed.evidence.contraction.plan_id == explicit.evidence.contraction.plan_id
    assert (
        changed.evidence.contraction.execution_id
        != explicit.evidence.contraction.execution_id
    )
    assert changed.evidence.replay_id != explicit.evidence.replay_id
    assert repeated.evidence.replay_id == changed.evidence.replay_id


@pytest.mark.parametrize("kind", ("norm", "overlap", "amplitude", "trace"))
@pytest.mark.parametrize("provider", ("cutensor", "cutensornet"))
def test_exact_peps_forwards_selected_provider_refusal(
    kind: str, provider: ContractionProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, arguments, _, precision = _exact_case(kind)
    execution = ContractionExecutionPolicy(provider)
    resources = ContractionResourcePolicy(maximum_workspace_bytes=2**20)
    seen: list[ContractionExecutionPolicy | None] = []

    class ProviderUnavailable(RuntimeError):
        pass

    def refuse_preparation(
        plan: ContractionPlan,
        operands: Sequence[ArrayLike],
        *,
        execution: ContractionExecutionPolicy | None = None,
    ) -> NoReturn:
        assert plan.resources is resources
        assert plan.precision is precision
        seen.append(execution)
        raise ProviderUnavailable("selected provider unavailable")

    monkeypatch.setattr(_peps, "prepare_contraction", refuse_preparation)
    with pytest.raises(ProviderUnavailable, match="selected provider unavailable"):
        contract(*arguments, resources=resources, execution=execution)
    assert len(seen) == 1 and seen[0] is execution
