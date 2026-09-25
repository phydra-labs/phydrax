#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from tests._ported_models import full_port, in_order, PortedAffine


Common = phx.AdmissibilityReason
Reason = phx.solver.LearnedStepCorrectionReason
STEP = 0.1
INITIAL = jnp.asarray([1.0, 2.0])


class _AffineIncrement(phx.AbstractArrayModel):
    """Increment `weight @ [accepted, candidate] + bias` of a learned correction."""

    weight: jax.Array
    bias: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, weight, bias):
        self.weight = jnp.asarray(weight, dtype=jnp.float64)
        self.bias = jnp.asarray(bias, dtype=jnp.float64)
        self.out_size, self.in_size = self.weight.shape

    def __call__(self, x, /, *, key=None):
        del key
        return self.weight @ x + self.bias


def _decay(time, state, args):
    del time, args
    return -state


def _correction(bias, weight=None, **checks):
    weight = jnp.zeros((2, 4)) if weight is None else weight
    return phx.solver.LearnedStepCorrection(
        _AffineIncrement(weight, bias),
        state_shape=(2,),
        maximum_relative_correction=0.5,
        **checks,
    )


def _problem(transform=None, *, steps=3):
    return phx.solver.FixedStepProblem(
        phx.solver.SSPRK33FixedStepMethod(_decay, transform=transform),
        INITIAL,
        t0=0.0,
        t1=STEP * steps,
        step_size=STEP,
    )


def _trajectory(problem):
    return phx.solver.FixedStepRolloutPlan(retention="trajectory").rollout(problem)


def test_admitted_correction_changes_the_rollout():
    bias = jnp.asarray([0.01, -0.01])
    corrected = _trajectory(
        _problem(_correction(bias, conserved=[[1.0, 1.0]], lower_bounds=0.0))
    )
    native = _trajectory(_problem())
    method = phx.solver.SSPRK33FixedStepMethod(_decay)
    expected = [INITIAL]
    for index in range(3):
        step = method.step(
            jnp.asarray(index),
            jnp.asarray(STEP * index),
            expected[-1],
            jnp.asarray(STEP),
            None,
        )
        expected.append(step.accepted_state + bias)

    assert corrected.successful
    assert jnp.all(corrected.transform_applied)
    assert jnp.all(corrected.transform_admissibility.eligible)
    assert jnp.all(corrected.transform_admissibility.reason_bits == 0)
    np.testing.assert_allclose(corrected.states, jnp.stack(expected), atol=1e-14)
    np.testing.assert_allclose(
        corrected.transform_correction_norm,
        jnp.full((3,), jnp.linalg.norm(bias)),
        rtol=1e-12,
    )
    assert not jnp.allclose(corrected.final_state, native.final_state)


@pytest.mark.parametrize(
    ("bias", "checks", "reasons"),
    (
        (
            (jnp.nan, 0.0),
            {},
            Common.NONFINITE | Common.OUTSIDE_SUPPORT | Reason.STABILITY_BOUND,
        ),
        ((0.01, 0.01), {"conserved": [[1.0, 1.0]]}, Reason.CONSERVATION),
        ((-0.01, 0.01), {"lower_bounds": (0.9, -jnp.inf)}, Reason.LOWER_BOUND),
        ((0.5, -0.5), {}, Reason.STABILITY_BOUND),
    ),
)
def test_rejected_correction_keeps_the_native_rollout_with_reason_bits(
    bias, checks, reasons
):
    corrected = _trajectory(_problem(_correction(jnp.asarray(bias), **checks)))
    native = _trajectory(_problem())

    assert corrected.successful
    assert jnp.array_equal(corrected.states, native.states)
    assert not jnp.any(corrected.transform_applied)
    assert jnp.all(corrected.transform_correction_norm == 0.0)
    assert not jnp.any(corrected.transform_admissibility.eligible)
    assert jnp.all(corrected.transform_admissibility.reason_bits == int(reasons))


def test_rejected_transaction_commits_the_unchanged_native_candidate():
    correction = _correction(jnp.asarray([0.5, -0.5]))
    native = 0.9 * INITIAL

    transaction = correction.propose(jnp.asarray(0), INITIAL, native)
    committed = phx.lifecycle.commit_candidate(transaction)

    assert not committed.committed
    assert jnp.array_equal(transaction.source, native)
    assert jnp.array_equal(transaction.proposed, native + jnp.asarray([0.5, -0.5]))
    assert jnp.array_equal(committed.state, native)
    assert int(committed.evidence.reason_bits) == int(Reason.STABILITY_BOUND)


def test_composite_transforms_keep_every_correction_reason():
    admitted = _correction(jnp.asarray([0.01, -0.01]))
    rejected = _correction(jnp.asarray([0.01, 0.01]), conserved=[[1.0, 1.0]])
    composite = phx.solver.CompositeAcceptedStepTransform((admitted, rejected))
    native = 0.9 * INITIAL

    result = composite.apply(jnp.asarray(0), jnp.asarray(STEP), INITIAL, native, None)

    assert result.successful
    assert result.applied
    assert jnp.array_equal(result.transformed_state, native + jnp.asarray([0.01, -0.01]))
    assert int(result.admissibility.reason_bits) == int(Reason.CONSERVATION)
    assert not result.admissibility.eligible


@pytest.mark.parametrize(
    "replay",
    (
        phx.solver.FixedStepReplayPolicy("step"),
        phx.solver.FixedStepReplayPolicy("block", block_size=2),
    ),
)
def test_checkpointed_rollout_gradients_match_finite_differences(replay):
    weight_key, bias_key, direction_key = jax.random.split(jax.random.key(0), 3)
    correction = _correction(
        2e-3 * jax.random.normal(bias_key, (2,)),
        2e-3 * jax.random.normal(weight_key, (2, 4)),
        conserved=[[1.0, 1.0]],
        conservation_tolerance=1.0,
    )
    problem = _problem(correction, steps=4)
    plan = phx.solver.FixedStepRolloutPlan(replay=replay)
    parameters, model_state, fixed = phx.partition_parameters(problem)

    def objective(values):
        rollout = plan.rollout(phx.combine_parameters(values, model_state, fixed))
        return jnp.sum(rollout.final_state**2)

    gradient = jax.grad(objective)(parameters)
    leaves, treedef = jax.tree_util.tree_flatten(parameters)
    keys = jax.random.split(direction_key, len(leaves))
    direction = jax.tree_util.tree_unflatten(
        treedef,
        [jax.random.normal(key, leaf.shape) for key, leaf in zip(keys, leaves)],
    )
    epsilon = 1e-6

    def shifted(scale):
        return objective(
            jax.tree.map(lambda leaf, step: leaf + scale * step, parameters, direction)
        )

    finite_difference = (shifted(epsilon) - shifted(-epsilon)) / (2.0 * epsilon)
    directional = sum(
        jnp.vdot(value, step)
        for value, step in zip(
            jax.tree_util.tree_leaves(gradient), jax.tree_util.tree_leaves(direction)
        )
    )

    assert jax.tree_util.tree_leaves(parameters) == [
        correction.model.weight,
        correction.model.bias,
    ]
    assert jnp.all(plan.rollout(problem).transform_applied)
    assert all(
        jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(gradient)
    )
    np.testing.assert_allclose(directional, finite_difference, rtol=1e-6)


def test_correction_slot_confers_discretization_authority():
    contract = _correction(jnp.zeros(2)).component_contract()

    assert contract.authority is phx.ComponentAuthority.DISCRETIZATION
    assert contract.slot_semantic_id == "solver.accepted-step-transform"
    assert (
        phx.dynamics.AbstractDiscreteModelRolloutTransition.slot_contract().authority
        is phx.ComponentAuthority.MODEL
    )


def test_port_declaring_correction_binds_in_owner_order():
    accepted, candidate = full_port("y:accepted", (2,)), full_port("y:candidate", (2,))
    owner = phx.ModelPorts(
        inputs=(accepted, candidate), outputs=(full_port("y:increment", (2,)),)
    )
    weight = jnp.asarray([[0.0, 0.0, 0.1, 0.0], [0.0, 0.0, 0.0, 0.1]])
    arguments = dict(state_shape=(2,), maximum_relative_correction=0.5)
    model = PortedAffine(owner, out_size=2, weight=weight)
    with pytest.raises(ValueError, match="accepted-step-transform'.*owner_ports"):
        phx.solver.LearnedStepCorrection(model, **arguments)
    swapped = phx.ModelPorts(inputs=(candidate, accepted), outputs=owner.outputs)
    with pytest.raises(ValueError, match="never repacked"):
        phx.solver.LearnedStepCorrection(
            PortedAffine(swapped, out_size=2),
            **arguments,
            ports=owner,
            port_mapping=in_order(swapped, swapped),
        )

    correction = phx.solver.LearnedStepCorrection(
        model, **arguments, ports=owner, port_mapping=in_order(owner, owner)
    )
    evidence = correction.component_contract().port_binding
    assert evidence.inputs == ((accepted.port_id,) * 2, (candidate.port_id,) * 2)
    assert evidence.unverified == ()
    transaction = correction.propose(
        jnp.asarray(0), jnp.asarray([1.0, 2.0]), jnp.asarray([0.9, 1.8])
    )
    np.testing.assert_allclose(transaction.proposed, [0.99, 1.98])


@pytest.mark.parametrize(
    ("keywords", "message"),
    (
        ({"state_shape": (3,)}, "2 \\* state size features"),
        ({"conservation_tolerance": 1e-8}, "requires conserved invariants"),
        ({"conserved": [[1.0, 1.0, 1.0]]}, "invariant_count, state_size"),
        ({"maximum_relative_correction": 0.0}, "finite and positive"),
        ({"lower_bounds": (jnp.nan, 0.0)}, "finite or -inf"),
    ),
)
def test_correction_refuses_inconsistent_declarations(keywords, message):
    arguments = {"state_shape": (2,), "maximum_relative_correction": 0.5, **keywords}
    with pytest.raises(ValueError, match=message):
        phx.solver.LearnedStepCorrection(
            _AffineIncrement(jnp.zeros((2, 4)), jnp.zeros(2)), **arguments
        )


class _ProposingTransform(phx.solver.AbstractAcceptedStepTransform):
    """Direct transform proposing `candidate + offset` with fixed evidence."""

    offset: float = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    transform_id: str = "test:proposing-transform"

    def __init__(self, offset, successful, dtype="float64"):
        self.offset = offset
        self.successful = successful
        self.dtype = dtype

    def apply(self, step_index, time, previous_state, candidate_state, args, /):
        del step_index, time, previous_state, args
        return phx.solver.AcceptedStepTransformResult(
            (candidate_state + self.offset).astype(self.dtype),
            jnp.asarray(True),
            jnp.asarray(self.successful),
            jnp.asarray(abs(self.offset)),
        )


def test_failed_direct_transform_cannot_change_the_source_state():
    method = phx.solver.SSPRK33FixedStepMethod(
        _decay, transform=_ProposingTransform(5.0, False)
    )
    native = phx.solver.SSPRK33FixedStepMethod(_decay)
    arguments = (jnp.asarray(0), jnp.asarray(0.0), INITIAL, jnp.asarray(STEP), None)

    result = method.step(*arguments)

    assert not result.successful
    assert jnp.array_equal(result.accepted_state, INITIAL)
    assert jnp.array_equal(
        result.candidate_state, native.step(*arguments).candidate_state
    )


def test_direct_transform_results_are_validated_centrally():
    method = phx.solver.SSPRK33FixedStepMethod(
        _decay, transform=_ProposingTransform(0.0, True, "float32")
    )
    with pytest.raises(ValueError, match="shape and dtype"):
        method.step(jnp.asarray(0), jnp.asarray(0.0), INITIAL, jnp.asarray(STEP), None)
