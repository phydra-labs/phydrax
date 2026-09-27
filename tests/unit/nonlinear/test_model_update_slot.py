#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""A learned model inside the canonical `FunctionNonlinearUpdate` accelerator slot."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from tests._ported_models import full_port, in_order, PortedAffine


nl = phx.nonlinear


class _AbstractGain(phx.AbstractArrayModel):
    gain: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, gain: Any) -> None:
        self.gain = jnp.asarray([gain])
        self.in_size = 1
        self.out_size = 1

    def __call__(self, x: Any, /, *, key: Any = None) -> Any:
        return self.gain * x


class _Gain(_AbstractGain):
    pass


class _HostGain(_AbstractGain):
    def model_execution_contract(self) -> Any:
        return phx.ModelExecutionContract(
            derivative=phx.DerivativeContract(route=phx.DerivativeRoute.STOPPED),
            execution=phx.ExecutionCapabilities("host-inference", host_only=True),
        )


class _Correction(phx.StrictModule):
    """Learned correction ``x + model(target - x)`` of the defect."""

    model: object

    def __call__(self, state: Any, target: Any) -> Any:
        model = (
            self.model.model
            if isinstance(self.model, phx.ComponentBinding)
            else self.model
        )
        # ty: ignore[call-non-callable]
        return state + model(target - state)


def _problem() -> Any:
    return nl.NonlinearSystemProblem(
        lambda state, target: state - target,
        trial_validity=lambda state, target: jnp.all(state > 0.0),
        trial_validity_id="positive-state",
        problem_id="positive-root",
    )


def _apply(update: Any, state: Any, target: Any) -> Any:
    prepared = nl.prepare_nonlinear_update(_problem(), state, update, args=target)
    return nl.apply_prepared_nonlinear_update(prepared, state, args=target)[0]


def test_model_update_slot_scenario_1() -> None:
    state, target = jnp.asarray([1.0]), jnp.asarray([2.0])

    # ty: ignore[invalid-argument-type, missing-argument]
    outside = _apply(nl.FunctionNonlinearUpdate(_Correction(_Gain(-5.0))), state, target)
    nonfinite = _apply(
        # ty: ignore[invalid-argument-type, missing-argument]
        nl.FunctionNonlinearUpdate(_Correction(_Gain(jnp.nan))),
        state,
        target,
    )

    assert not bool(outside.applied)
    assert outside.status == int(nl.NonlinearUpdateStatus.DOMAIN_REJECTED)
    assert int(outside.diagnostics.domain_failures) == 1
    assert not bool(nonfinite.applied)
    assert nonfinite.status == int(nl.NonlinearUpdateStatus.NONFINITE_EVALUATION)

    result = nl.NonlinearRichardson(
        # ty: ignore[invalid-argument-type, missing-argument]
        nl.FunctionNonlinearUpdate(_Correction(_Gain(-5.0)))
    ).solve(
        _problem(),
        state,
        termination=nl.NonlinearTermination(maximum_steps=5),
        args=target,
    )
    assert not bool(result.successful)
    assert jnp.array_equal(result.state, state)
    state, target = jnp.asarray([1.0]), jnp.asarray([2.0])
    # ty: ignore[invalid-argument-type, missing-argument]
    update = nl.FunctionNonlinearUpdate(_Correction(_Gain(3.0)), update_id="learned")

    proposal = _apply(update, state, target)
    assert bool(proposal.applied)
    assert float(proposal.diagnostics.final_residual_norm) == pytest.approx(2.0)
    assert float(proposal.diagnostics.initial_residual_norm) == pytest.approx(1.0)

    result = nl.NonlinearRichardson(update).solve(
        _problem(),
        state,
        termination=nl.NonlinearTermination(
            absolute_residual=1e-10, relative_residual=0.0, maximum_steps=100
        ),
        args=target,
    )
    iterations = int(result.diagnostics.iterations)
    assert bool(result.successful)
    assert jnp.allclose(result.state, target, atol=1e-9)
    # Initial, per-iteration update pair, and final evaluation account for
    # 2 * iterations + 2; every backtracking trial is an extra evaluation of the
    # original residual that the proposal alone never paid for.
    assert int(result.diagnostics.residual_evaluations) > 2 * iterations + 2
    # ty: ignore[invalid-argument-type, missing-argument]
    model = _Gain(0.5)
    update = nl.FunctionNonlinearUpdate(_Correction(model))

    ((location, contract),) = update.component_contracts()
    assert location == "function.model"
    assert contract.authority is phx.ComponentAuthority.ACCELERATOR
    assert contract.slot_semantic_id == nl.AbstractNonlinearUpdate.slot_semantic_id

    parameters, _, _ = phx.partition_parameters(nl.NonlinearRichardson(update))
    assert [id(leaf) for leaf in jax.tree.leaves(parameters)] == [id(model.gain)]

    bound = phx.bind_component(model, nl.AbstractNonlinearUpdate)
    assert nl.FunctionNonlinearUpdate(_Correction(bound)).component_contracts()
    with pytest.raises(ValueError, match="model authority"):
        nl.FunctionNonlinearUpdate(
            _Correction(phx.bind_component(model, phx.ComponentAuthority.MODEL))
        )
    with pytest.raises(TypeError, match="callable module"):
        # ty: ignore[invalid-argument-type]
        nl.FunctionNonlinearUpdate(model)


def test_model_update_slot_scenario_2() -> None:
    owner = phx.ModelPorts(
        inputs=(full_port("root.defect", (1,)),),
        outputs=(full_port("root.correction", (1,)),),
    )
    model = PortedAffine(owner, out_size=1, weight=jnp.asarray([[0.5]]))
    with pytest.raises(ValueError, match="nonlinear.update'.*owner_ports"):
        nl.FunctionNonlinearUpdate(_Correction(model))

    bound = phx.bind_component(
        model,
        nl.AbstractNonlinearUpdate,
        owner_ports=owner,
        port_mapping=in_order(owner, owner),
    )
    update = nl.FunctionNonlinearUpdate(_Correction(bound))
    ((location, contract),) = update.component_contracts()
    assert location == "function.model"
    # ty: ignore[unresolved-attribute]
    assert contract.port_binding.outputs == ((owner.outputs[0].port_id,) * 2,)
    # ty: ignore[unresolved-attribute]
    assert contract.port_binding.unverified == ()
    assert jnp.allclose(
        _apply(update, jnp.asarray([1.0]), jnp.asarray([2.0])).state, jnp.asarray([1.5])
    )
    # ty: ignore[invalid-argument-type, missing-argument]
    native = nl.FunctionNonlinearUpdate(_Correction(_Gain(0.5)))
    # ty: ignore[invalid-argument-type, missing-argument]
    host = nl.FunctionNonlinearUpdate(_Correction(_HostGain(0.5)))

    assert native.capabilities.jit and native.capabilities.differentiable_action
    assert not host.capabilities.jit
    assert not host.capabilities.differentiable_action
    assert not nl.NonlinearRichardson(host).capabilities.jit
    assert nl.FunctionNonlinearUpdate(lambda state, args: state).capabilities.jit
