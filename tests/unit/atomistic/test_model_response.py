"""Native higher-order response on explicitly admitted source-model pieces."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax._differentiation import RegularityPolicy
from phydrax._trainable import combine_parameters, partition_parameters
from phydrax.atomistic import (
    AbstractAtomisticPotential,
    atomistic_energy_derivatives,
    AtomisticBatch,
    AtomisticConditioning,
    AtomisticGraphExecutionPlan,
    AtomisticGraphLayout,
    AtomisticGraphPayload,
    AtomisticObservable,
    AtomisticObservableRequest,
)
from phydrax.atomistic._distillation import ATOMISTIC_HVP_TRAINING_REQUEST
from phydrax.atomistic._hessian import (
    ATOMISTIC_HESSIAN_REQUEST,
    bind_atomistic_hessian,
    prepare_atomistic_hessian,
)
from phydrax.nn.atomistic import PreparedUMAMergedModel, UMAPotential
from tests._support.atomistic_model_families import (
    family_scenario,
    finite_execution,
    uma_binding,
    uma_model,
    water_batch,
)


type ResponseFixture = tuple[
    str,
    AbstractAtomisticPotential,
    AtomisticBatch,
    AtomisticGraphExecutionPlan,
    AtomisticGraphPayload,
    AtomisticConditioning,
]

AE = RegularityPolicy(allow_almost_everywhere=True)
MIXED = AtomisticObservableRequest(
    (AtomisticObservable.HESSIAN_ACTION,),
    derivatives=(ATOMISTIC_HESSIAN_REQUEST, ATOMISTIC_HVP_TRAINING_REQUEST),
)
DIRECTION = jnp.asarray(
    [[[0.2, -0.3, 0.1], [-0.1, 0.4, -0.2], [0.05, 0.15, -0.25]]], dtype=jnp.float64
)


def _fixture(family: str) -> ResponseFixture:
    coordinates = np.asarray(
        [[[0.0, 0.0, 0.0], [0.91, 0.17, 0.23], [-0.24, 0.82, 0.57]]],
        dtype=np.float64,
    )
    batch = water_batch(coordinates)
    override = None
    if family == "uma":
        override = uma_model(
            3,
            lmax=1,
            mmax=1,
            experts=2,
            ff_type="spectral",
            sphere_channels=2,
            hidden_channels=2,
            edge_channels=2,
            num_distance_basis=3,
            num_layers=1,
        )
    scenario = family_scenario(family, periodic=False, atom_capacity=3, model=override)
    model = scenario.model
    assert isinstance(model, AbstractAtomisticPotential)
    conditioning = scenario.conditioning(batch, ((1, 2),))
    assert conditioning is not None
    execution = finite_execution(3)
    payload = AtomisticGraphLayout.for_batch(
        batch, execution, cutoff=model.configuration.cutoff
    ).prepare(batch)
    return family, model, batch, execution, payload, conditioning


@pytest.fixture(scope="module", params=("aimnet2-nse", "uma"))
def response(request: pytest.FixtureRequest) -> ResponseFixture:
    return _fixture(request.param)


def test_explicit_piecewise_hessian_matches_force_directional_difference(
    response: ResponseFixture,
) -> None:
    family, model, batch, execution, payload, conditioning = response
    with pytest.raises(ValueError, match="almost-everywhere-not-allowed"):
        prepare_atomistic_hessian(
            model, batch, execution, payload=payload, conditioning=conditioning
        )
    operator = prepare_atomistic_hessian(
        model, batch, execution, payload=payload, conditioning=conditioning, policy=AE
    ).operator
    actual = operator.mv(DIRECTION)

    @eqx.filter_jit
    def forces(position: Array) -> Array:
        result = atomistic_energy_derivatives(
            model, payload, position, conditioning=conditioning
        )
        assert result.forces is not None
        return result.forces

    step = 1.0e-5
    reference = -(
        forces(batch.positions + step * DIRECTION)
        - forces(batch.positions - step * DIRECTION)
    ) / (2.0 * step)
    assert bool(jnp.all(operator.action_valid))
    # This reference rejects a plausible zero-action implementation.
    assert float(jnp.linalg.norm(reference)) > 1.0e-6
    np.testing.assert_allclose(actual, reference, rtol=2.0e-5, atol=2.0e-8)
    if family == "uma":
        assert (
            "atomistic-nonsmooth-locus:uma-neighbor-selection-membership"
            in operator.unresolved_conditions
        )
        assert "singular-part-ignored" in operator.unresolved_conditions
    else:
        assert operator.unresolved_conditions == ()


def test_mixed_parameter_hvp_loss_gradient_matches_parameter_difference(
    response: ResponseFixture,
) -> None:
    _, model, batch, _, payload, conditioning = response
    parameters, state, fixed = partition_parameters(model)
    target = 0.03 * DIRECTION

    def loss(values: Any) -> Array:
        operator = bind_atomistic_hessian(
            combine_parameters(values, state, fixed),
            payload,
            batch.positions,
            conditioning=conditioning,
            request=MIXED,
            policy=AE,
        )
        return jnp.sum((operator.mv(DIRECTION) - target) ** 2)

    gradient = jax.jit(jax.grad(loss))(parameters)
    leaves, structure = jax.tree.flatten(parameters)
    rng = np.random.default_rng(41)
    tangent = jax.tree.unflatten(
        structure,
        [
            jnp.asarray(0.01 * rng.normal(size=leaf.shape), dtype=leaf.dtype)
            for leaf in leaves
        ],
    )
    derivative = sum(
        jnp.vdot(left, right)
        for left, right in zip(
            jax.tree.leaves(gradient), jax.tree.leaves(tangent), strict=True
        )
    )
    step = 1.0e-5
    objective = jax.jit(loss)
    upper = jax.tree.map(lambda value, delta: value + step * delta, parameters, tangent)
    lower = jax.tree.map(lambda value, delta: value - step * delta, parameters, tangent)
    reference = (objective(upper) - objective(lower)) / (2.0 * step)
    assert abs(float(reference)) > 1.0e-6
    np.testing.assert_allclose(derivative, reference, rtol=2.0e-5, atol=1.0e-8)


def test_known_cutoff_join_invalidates_hessian_point(response: ResponseFixture) -> None:
    _, model, batch, execution, payload, conditioning = response
    position = batch.positions.at[0, 1].set(
        jnp.asarray([model.configuration.cutoff, 0.0, 0.0], dtype=jnp.float64)
    )
    operator = prepare_atomistic_hessian(
        model,
        batch,
        execution,
        positions=position,
        payload=payload,
        conditioning=conditioning,
        policy=AE,
    ).operator
    assert not bool(operator.action_valid[0])


def test_frozen_merged_hessian_is_traceable_and_has_no_trainable_surface() -> None:
    _, raw, batch, execution, payload, conditioning = _fixture("uma")
    assert isinstance(raw, UMAPotential)
    merged = raw.merged(uma_binding(raw, batch, periodic=False, charge=1, multiplicity=2))

    @eqx.filter_jit
    def action(model: PreparedUMAMergedModel, position: Array) -> Array:
        return bind_atomistic_hessian(
            model, payload, position, conditioning=conditioning, policy=AE
        ).mv(DIRECTION)

    reference = prepare_atomistic_hessian(
        raw, batch, execution, payload=payload, conditioning=conditioning, policy=AE
    ).operator.mv(DIRECTION)
    np.testing.assert_allclose(
        action(merged, batch.positions), reference, rtol=1.0e-10, atol=1.0e-12
    )
    with pytest.raises(ValueError, match="derivative-unsupported"):
        prepare_atomistic_hessian(
            merged,
            batch,
            execution,
            payload=payload,
            conditioning=conditioning,
            request=MIXED,
            policy=AE,
        )
