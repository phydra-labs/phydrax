#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.discretization._cell_complex import interval_cell_complex
from phydrax.discretization._cochain import CochainDiscretization
from phydrax.discretization._cochain_hodge import DiagonalHodge


def test_metric_refresh_inside_scan_changes_adjoint_without_changing_binding() -> None:
    topology = interval_cell_complex(np.asarray([[0, 1], [1, 2]]), 3)
    original = CochainDiscretization(
        topology,
        (DiagonalHodge(jnp.ones(3)), DiagonalHodge(jnp.ones(2))),
        numeric_revision="moving-metric-binding",
    )
    values = jnp.asarray([0.4, -0.7])

    def step(
        state: CochainDiscretization, scale: jax.Array
    ) -> tuple[CochainDiscretization, jax.Array]:
        refreshed = state.with_metric(
            (DiagonalHodge(jnp.ones(3)), DiagonalHodge(scale * jnp.ones(2))),
            numeric_revision=state.numeric_revision,
        )
        refreshed = eqx.tree_at(lambda item: item.time, refreshed, scale)
        return refreshed, refreshed.codifferential(1, values)

    final, outputs = eqx.filter_jit(
        lambda: jax.lax.scan(step, original, jnp.asarray([1.0, 2.0, 3.0]))
    )()
    expected = np.asarray([-0.4, 1.1, -0.7])
    np.testing.assert_allclose(outputs, np.asarray([1.0, 2.0, 3.0])[:, None] * expected)
    assert final.numeric_revision == original.numeric_revision
    assert final.realization_id == original.realization_id
    np.testing.assert_allclose(final.time, 3.0)


def test_dynamic_hodge_evidence_rejects_nonpositive_metric() -> None:
    valid = jax.jit(lambda scale: DiagonalHodge(scale * jnp.ones(3)).valid)
    assert bool(valid(jnp.asarray(1.0)))
    assert not bool(valid(jnp.asarray(-1.0)))
    assert not bool(valid(jnp.asarray(0.0)))
