#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.discretization.finite_volume import (
    advance_driven_euler_banks,
    driven_euler_cell_keys,
    DrivenEulerSource,
    DrivenEulerSourceArgs,
)


def _banks() -> DrivenEulerSourceArgs:
    keys = driven_euler_cell_keys(
        jr.key(42),
        jnp.array([0, 1], dtype=jnp.uint32),
        jnp.array([7, 7], dtype=jnp.uint32),
    )
    return DrivenEulerSourceArgs(
        jnp.array([[2.0, 0.7, 0.3], [3.0, 0.8, 0.2]]),
        jnp.array([0.4, 0.5]),
        keys,
        jnp.array([0.1, 0.2]),
        jnp.array([True, False]),
    )


def test_keys_use_both_global_id_words_and_follow_permutation() -> None:
    root = jr.key(3)
    hi = jnp.array([1, 2, 1], dtype=jnp.uint32)
    lo = jnp.array([4, 4, 5], dtype=jnp.uint32)
    data = jr.key_data(driven_euler_cell_keys(root, hi, lo))
    assert not np.array_equal(data[0], data[1])
    assert not np.array_equal(data[0], data[2])
    order = jnp.array([2, 0, 1])
    np.testing.assert_array_equal(
        jr.key_data(driven_euler_cell_keys(root, hi[order], lo[order])), data[order]
    )
    with pytest.raises(TypeError):
        driven_euler_cell_keys(root, hi.astype(jnp.int32), lo)


def test_source_consumes_all_banks_and_is_pure() -> None:
    source = DrivenEulerSource(1.7)
    banks = _banks()
    state = jnp.array([[2.0, 1.0, 0.5, 5.0], [1.0, 0.5, 0.0, 3.0]])
    coordinates = jnp.zeros((2, 2))
    time = jnp.array(0.6)
    before = jax.tree.map(
        lambda x: (
            jr.key_data(x) if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key) else x
        ),
        banks,
    )
    value = source(time, state, coordinates, banks)
    np.testing.assert_array_equal(value, source(time, state, coordinates, banks))
    np.testing.assert_array_equal(value[1], jnp.zeros(4))
    assert value[0, 0] == 0 and value[0, 2] == 0 and value[0, 1] >= 0
    # Energy is body-force work plus specific heat input times density.
    acceleration = value[0, 1] / state[0, 0]
    heat = banks.material[0, 1] * jnp.exp(-banks.material[0, 2] * (time - banks.clock[0]))
    np.testing.assert_allclose(
        value[0, -1], state[0, 1] * acceleration + state[0, 0] * heat
    )
    variants = (
        banks._replace(material=banks.material.at[0, 0].multiply(1.5)),
        banks._replace(material=banks.material.at[0, 1].multiply(1.5)),
        banks._replace(material=banks.material.at[0, 2].multiply(1.5)),
        banks._replace(history=banks.history.at[0].add(0.3)),
        banks._replace(clock=banks.clock.at[0].add(0.3)),
        banks._replace(keys=banks.keys.at[0].set(jr.key(918))),
    )
    for variant in variants:
        assert not np.allclose(value[0], source(time, state, coordinates, variant)[0])
    after = jax.tree.map(
        lambda x: (
            jr.key_data(x) if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key) else x
        ),
        banks,
    )
    for old, new in zip(before, after, strict=True):
        np.testing.assert_array_equal(old, new)


def test_accepted_bank_update_is_masked_and_advances_all_banks() -> None:
    banks = _banks()
    dt = jnp.array(0.25)
    updated = advance_driven_euler_banks(
        None, SimpleNamespace(accepted_step_size=dt), banks
    )
    np.testing.assert_allclose(
        updated.material[0, 1], banks.material[0, 1] * jnp.exp(-banks.material[0, 2] * dt)
    )
    np.testing.assert_allclose(updated.history[0], banks.history[0] + dt)
    np.testing.assert_allclose(updated.clock[0], banks.clock[0] + dt)
    assert not np.array_equal(jr.key_data(updated.keys[0]), jr.key_data(banks.keys[0]))
    for old, new in zip(banks, updated, strict=True):
        if jax.dtypes.issubdtype(old.dtype, jax.dtypes.prng_key):
            old, new = jr.key_data(old), jr.key_data(new)
        np.testing.assert_array_equal(old[1], new[1])
    np.testing.assert_array_equal(updated.active, banks.active)


def test_select_rows_preserves_every_bank_and_typed_keys() -> None:
    banks = _banks()
    indices = jnp.asarray((1, 0, 1), dtype=jnp.int32)
    selected = jax.jit(lambda value, rows: value.select_rows(rows))(banks, indices)
    for old, new in zip(banks, selected, strict=True):
        if jax.dtypes.issubdtype(old.dtype, jax.dtypes.prng_key):
            old, new = jr.key_data(old), jr.key_data(new)
        np.testing.assert_array_equal(new, old[indices])


def test_mixed_precision_source_and_conditional_bank_update() -> None:
    banks = _banks()
    banks = banks._replace(
        material=banks.material.astype(jnp.float32),
        history=banks.history.astype(jnp.float32),
        clock=banks.clock.astype(jnp.float32),
    )
    state = jnp.asarray(((2, 1, 0, 5), (1, 0, 0, 3)), dtype=jnp.float32)
    output = DrivenEulerSource(1.7)(
        jnp.asarray(0.6, dtype=jnp.float64), state, jnp.zeros((2, 2)), banks
    )
    assert output.dtype == jnp.float32
    dt = jnp.asarray(0.25, dtype=jnp.float64)
    update = jax.jit(
        lambda accept, value: jax.lax.cond(
            accept,
            lambda b: advance_driven_euler_banks(
                None, SimpleNamespace(accepted_step_size=dt), b
            ),
            lambda b: b,
            value,
        )
    )
    for accepted in (False, True):
        result = update(jnp.asarray(accepted), banks)
        assert (
            result.material.dtype
            == result.history.dtype
            == result.clock.dtype
            == jnp.float32
        )
        if not accepted:
            np.testing.assert_array_equal(result.history, banks.history)
            np.testing.assert_array_equal(
                jr.key_data(result.keys), jr.key_data(banks.keys)
            )
