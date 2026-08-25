#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _burgers_pair():
    system = phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: 0.5 * state**2,
        lambda left, right, axis, args: jnp.maximum(
            jnp.abs(left[..., 0]),
            jnp.abs(right[..., 0]),
        ),
        system_id="entropy-test-burgers",
    )
    return phx.equations.ConvexEntropyPair(
        system,
        lambda state: 0.5 * state[..., 0] ** 2,
        lambda state: state,
        lambda state, axis, args: state[..., 0] ** 3 / 3.0,
        lambda state: jnp.all(jnp.isfinite(state), axis=-1),
        entropy_id="quadratic-burgers",
    )


def test_scalar_convex_entropy_pair_matches_burgers_identities():
    pair = _burgers_pair()
    states = jnp.asarray([[-0.7], [0.2], [1.1]])
    reference = jnp.asarray([[-0.3], [0.4], [0.9]])
    report = phx.equations.validate_convex_entropy_pair(
        pair,
        states,
        comparison_states=reference,
        variable_tolerance=1e-8,
        flux_tolerance=1e-8,
        symmetry_tolerance=1e-8,
        relative_entropy_tolerance=1e-8,
    )

    assert bool(report.valid)
    assert report.axes == (0,)
    assert jnp.allclose(pair.entropy_variables(states), states)
    assert jnp.allclose(pair.entropy_flux(states, 0), states[..., 0] ** 3 / 3.0)
    assert jnp.allclose(
        pair.relative_entropy(states, reference),
        0.5 * (states[..., 0] - reference[..., 0]) ** 2,
    )
    assert jnp.allclose(pair.entropy_potential(states, 0), states[..., 0] ** 3 / 6.0)
    assert jnp.allclose(pair.symmetrizer_action(states, jnp.ones_like(states)), 1.0)


def test_entropy_pair_interface_residual_matches_conservative_and_dissipative_fluxes():
    pair = _burgers_pair()
    left = jnp.asarray([[-0.5], [0.2]])
    right = jnp.asarray([[0.7], [-0.4]])
    conservative_flux = (
        pair.entropy_potential(right, 0) - pair.entropy_potential(left, 0)
    ) / (pair.entropy_variables(right)[..., 0] - pair.entropy_variables(left)[..., 0])
    conservative_flux = conservative_flux[..., None]
    residual = pair.interface_entropy_residual(left, right, conservative_flux, 0)
    assert jnp.allclose(residual, 0.0)
    dissipative_flux = conservative_flux - 0.5 * (right - left)
    assert jnp.all(pair.interface_entropy_residual(left, right, dissipative_flux, 0) <= 1e-8)


def test_euler_entropy_pair_matches_existing_variables_and_all_axes():
    system = phx.equations.EulerSystem(2)
    pair = phx.equations.ideal_gas_euler_entropy_pair(system)
    primitive = jnp.asarray(
        [
            [1.0, 0.4, -0.2, 1.0],
            [0.8, -0.1, 0.3, 0.7],
        ]
    )
    state = system.primitive_to_conserved(primitive)
    report = phx.equations.validate_convex_entropy_pair(
        pair,
        state,
        axes=(0, 1),
        variable_tolerance=1e-7,
        flux_tolerance=1e-7,
        symmetry_tolerance=1e-7,
        relative_entropy_tolerance=1e-7,
    )

    assert bool(report.valid)
    assert jnp.allclose(pair.entropy_variables(state), system.entropy_variables(state))
    assert jnp.allclose(
        pair.entropy_flux(state, 0),
        primitive[..., 1] * pair.entropy(state),
    )
    assert jnp.allclose(
        pair.entropy_flux(state, 1),
        primitive[..., 2] * pair.entropy(state),
    )
    assert jnp.allclose(
        pair.entropy_potential(state, 0),
        state[..., 1],
    )
    assert jnp.allclose(
        pair.entropy_potential(state, 1),
        state[..., 2],
    )


def test_entropy_pair_methods_are_jittable_and_batch_local():
    pair = _burgers_pair()
    states = jnp.asarray([[-0.4], [0.6], [1.2]])
    directions = jnp.asarray([[0.3], [-0.1], [0.8]])
    relative = jax.jit(pair.relative_entropy)(states, states + directions)
    action = jax.jit(pair.symmetrizer_action)(states, directions)
    assert relative.shape == (3,)
    assert action.shape == states.shape
    assert jnp.allclose(relative, 0.5 * directions[..., 0] ** 2)
    assert jnp.allclose(action, directions)


def test_entropy_validation_reports_invalid_pair_without_raising_when_requested():
    pair = _burgers_pair()
    invalid = phx.equations.ConvexEntropyPair(
        pair.system,
        lambda state: 0.5 * state[..., 0] ** 2,
        lambda state: 2.0 * state,
        lambda state, axis, args: state[..., 0] ** 3 / 3.0,
        pair.admissible_function,
        entropy_id="wrong-variables",
    )
    report = phx.equations.validate_convex_entropy_pair(
        invalid,
        jnp.asarray([[-0.2], [0.4]]),
        raise_on_error=False,
    )
    assert not bool(report.valid)
    assert report.maximum_entropy_variable_residual > 0.0
    assert report.maximum_flux_compatibility_residual > 0.0

    with pytest.raises(ValueError, match="Convex entropy pair validation failed"):
        phx.equations.validate_convex_entropy_pair(
            invalid,
            jnp.asarray([[-0.2], [0.4]]),
        )


def test_entropy_pair_rejects_wrong_shapes_axes_and_domains():
    pair = _burgers_pair()
    with pytest.raises(ValueError, match="trailing component dimension"):
        pair.entropy(jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="Entropy flux axis"):
        pair.entropy_flux(jnp.ones((2, 1)), 1)
    with pytest.raises(Exception, match="outside entropy pair"):
        pair.entropy(jnp.asarray([[jnp.nan]]))
    with pytest.raises(ValueError, match="comparison states must match"):
        phx.equations.validate_convex_entropy_pair(
            pair,
            jnp.ones((2, 1)),
            comparison_states=jnp.ones((3, 1)),
        )


def test_public_entropy_methods_reject_nonfinite_states_with_permissive_predicate():
    base = _burgers_pair()
    pair = phx.equations.ConvexEntropyPair(
        base.system,
        base.entropy_function,
        base.entropy_variables_function,
        base.entropy_flux_function,
        lambda state: jnp.ones(state.shape[:-1], dtype=bool),
        entropy_id="permissive-domain",
    )
    invalid = jnp.asarray([[jnp.nan]])

    assert not bool(jnp.all(pair.admissible(invalid)))
    with pytest.raises(Exception, match="outside entropy pair"):
        pair.entropy(invalid)
    with pytest.raises(Exception, match="outside entropy pair"):
        pair.entropy_variables(invalid)
    with pytest.raises(Exception, match="outside entropy pair"):
        pair.entropy_flux(invalid, 0)
    with pytest.raises(Exception, match="outside entropy pair"):
        jax.jit(pair.entropy)(invalid)


def test_entropy_pair_rejects_nonfloating_callable_outputs():
    base = _burgers_pair()
    state = jnp.asarray([[0.2]])
    integer_entropy = phx.equations.ConvexEntropyPair(
        base.system,
        lambda value: jnp.ones(value.shape[:-1], dtype=jnp.int32),
        base.entropy_variables_function,
        base.entropy_flux_function,
        base.admissible_function,
        entropy_id="integer-entropy",
    )
    complex_variables = phx.equations.ConvexEntropyPair(
        base.system,
        base.entropy_function,
        lambda value: value.astype(jnp.complex64),
        base.entropy_flux_function,
        base.admissible_function,
        entropy_id="complex-variables",
    )
    complex_flux = phx.equations.ConvexEntropyPair(
        base.system,
        base.entropy_function,
        base.entropy_variables_function,
        lambda value, axis, args: value[..., 0].astype(jnp.complex64),
        base.admissible_function,
        entropy_id="complex-flux",
    )

    with pytest.raises(TypeError, match="real floating-point"):
        integer_entropy.entropy(state)
    with pytest.raises(TypeError, match="real floating-point"):
        complex_variables.entropy_variables(state)
    with pytest.raises(TypeError, match="real floating-point"):
        complex_flux.entropy_flux(state, 0)


def test_entropy_flux_and_validation_propagate_runtime_args():
    system = phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: 0.5 * args["scale"] * state**2,
        lambda left, right, axis, args: args["scale"]
        * jnp.maximum(jnp.abs(left[..., 0]), jnp.abs(right[..., 0])),
        system_id="scaled-burgers",
    )
    pair = phx.equations.ConvexEntropyPair(
        system,
        lambda state: 0.5 * state[..., 0] ** 2,
        lambda state: state,
        lambda state, axis, args: args["scale"] * state[..., 0] ** 3 / 3.0,
        system.admissible,
        entropy_id="scaled-quadratic-burgers",
    )
    state = jnp.asarray([[-0.4], [0.7]])
    args = {"scale": 2.5}
    report = phx.equations.validate_convex_entropy_pair(
        pair,
        state,
        args=args,
    )

    assert bool(report.valid)
    assert jnp.allclose(
        pair.entropy_flux(state, 0, args),
        args["scale"] * state[..., 0] ** 3 / 3.0,
    )
    assert jnp.allclose(
        pair.entropy_potential(state, 0, args),
        args["scale"] * state[..., 0] ** 3 / 6.0,
    )


def test_entropy_validation_rejects_empty_axes_and_reports_nonfinite_evidence():
    base = _burgers_pair()
    with pytest.raises(ValueError, match="must be non-empty"):
        phx.equations.validate_convex_entropy_pair(
            base,
            jnp.asarray([[0.2]]),
            axes=(),
        )

    singular_variables = phx.equations.ConvexEntropyPair(
        base.system,
        base.entropy_function,
        lambda state: jnp.sqrt(state**2),
        base.entropy_flux_function,
        base.admissible_function,
        entropy_id="singular-variable-jacobian",
    )
    report = phx.equations.validate_convex_entropy_pair(
        singular_variables,
        jnp.asarray([[0.0]]),
        raise_on_error=False,
    )
    assert not bool(report.finite)
    assert not bool(report.valid)
