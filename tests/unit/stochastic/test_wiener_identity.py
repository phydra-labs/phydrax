#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Static Wiener law identity, dynamic path data and host identity materialization."""

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.stochastic import WienerRealization
from phydrax.stochastic._realization import CompositeStochasticRealization


def _independent(key: Any, *, paths: int = 3) -> WienerRealization:
    return WienerRealization.independent(
        key,
        (2,),
        support=(0.0, 1.0),
        sample_shape=(paths,),
        tolerance=1e-3,
        noise_id="test-noise",
        label="identity-test",
    )


def test_traced_construction_keeps_the_law_static_and_the_digest_host_side() -> None:
    traces: list[None] = []

    @eqx.filter_jit
    def build(key: Any) -> WienerRealization:
        traces.append(None)
        return _independent(key)

    eager = _independent(jr.key(4))
    staged = build(jr.key(4))
    other = build(jr.key(5))

    # A new key is dynamic path data: no retrace and no host hash in the trace.
    assert len(traces) == 1
    assert staged.law_id == eager.law_id == other.law_id
    assert len(eager.realization_id) == 64
    assert staged.realization_id == eager.realization_id
    assert staged.coupling_id == eager.coupling_id
    assert other.realization_id != eager.realization_id
    assert other.coupling_id != eager.coupling_id
    starts = jnp.asarray([0.0, 0.25])
    ends = jnp.asarray([0.25, 1.0])
    np.testing.assert_array_equal(
        staged.increments(starts, ends), eager.increments(starts, ends)
    )


def test_identity_is_never_read_from_tracers_while_declared_coupling_is_static() -> None:
    @eqx.filter_jit
    def read(key: Any) -> Any:
        return WienerRealization.independent(key, (1,), support=(0.0, 1.0)).realization_id

    with pytest.raises(ValueError, match="law_id"):
        read(jr.key(0))

    @eqx.filter_jit
    def declared(key: Any) -> Any:
        realization = WienerRealization.antithetic(
            key, (1,), support=(0.0, 1.0), num_pairs=2, coupling_id="paired"
        )
        if realization.coupling_id != "paired":
            raise AssertionError("A declared coupling label must stay static.")
        return realization.path_signs

    np.testing.assert_array_equal(declared(jr.key(1)), [1.0, -1.0, 1.0, -1.0])


def test_antithetic_coupling_keeps_its_prefix_stable_key_algorithm() -> None:
    paired = WienerRealization.antithetic(
        jr.key(2), (1,), support=(0.0, 1.0), num_pairs=2
    )
    keys = jr.key_data(paired.path_keys)

    np.testing.assert_array_equal(keys[0], keys[1])
    np.testing.assert_array_equal(keys[0], jr.key_data(jr.fold_in(jr.key(2), 0)))
    np.testing.assert_array_equal(keys[2], jr.key_data(jr.fold_in(jr.key(2), 1)))
    increments = paired.increments(jnp.asarray([0.0]), jnp.asarray([1.0]))
    np.testing.assert_array_equal(increments[0], -increments[1])
    independent = WienerRealization.independent(
        jr.key(2), (1,), support=(0.0, 1.0), sample_shape=(4,)
    )
    # Same root, different path layout: one law family, distinct realizations.
    assert paired.law_id == independent.law_id
    assert paired.realization_id != independent.realization_id
    assert paired.coupling_id == independent.coupling_id
    with pytest.raises(eqx.EquinoxRuntimeError, match="signs"):
        WienerRealization(
            jr.key(0),
            (1,),
            support=(0.0, 1.0),
            sample_shape=(2,),
            _path_signs=jnp.asarray([1.0, 0.5]),
        )


def test_composite_identity_is_materialized_from_its_components() -> None:
    first = _independent(jr.key(3), paths=2)
    second = _independent(jr.key(4), paths=2)
    composite = CompositeStochasticRealization({"a": first, "b": second})

    @eqx.filter_jit
    def build(key: Any) -> CompositeStochasticRealization:
        return CompositeStochasticRealization(
            {"a": _independent(key, paths=2), "b": second}
        )

    staged = build(jr.key(3))

    assert len(composite.realization_id) == 64
    assert staged.realization_id == composite.realization_id
    assert staged.coupling_id == composite.coupling_id
    assert (
        CompositeStochasticRealization({"a": second, "b": first}).realization_id
        != composite.realization_id
    )
