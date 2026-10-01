# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.meshfree._surface_transfer import SurfaceTransferPlan
from phydrax.sparse import EdgeRelation, RowRelation


def test_equal_area_does_not_imply_constant_preservation() -> None:
    relation = RowRelation(np.asarray([[0, 1], [0, 1]], dtype=np.int32), source_size=2)
    route = SurfaceTransferPlan(
        relation,
        np.asarray([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64),
        np.asarray([1.0, 1.0], dtype=np.float64),
        np.asarray([1.0, 1.0], dtype=np.float64),
        source_id="old",
        target_id="new",
        mode="positive",
    ).prepare()
    assert route.transfer.properties.conservative
    assert route.transfer.properties.positivity_preserving
    assert not route.transfer.properties.constant_preserving
    np.testing.assert_allclose(
        route.apply(np.asarray([1.0, 1.0], dtype=np.float64)), [1.5, 0.5], atol=1e-12
    )
    np.testing.assert_allclose(route.evidence.column_residual, 0.0, atol=1e-12)


def test_signed_correction_conserves_content_and_has_correct_reverse_maps() -> None:
    relation = RowRelation(
        np.asarray([[0, 1], [0, 1], [0, 1]], dtype=np.int32), source_size=2
    )
    route = SurfaceTransferPlan(
        relation,
        np.asarray([[1.1, -0.1], [0.3, 0.7], [-0.2, 1.2]], dtype=np.float64),
        np.asarray([2.0, 3.0], dtype=np.float64),
        np.asarray([1.0, 2.0, 4.0], dtype=np.float64),
        source_id="old",
        target_id="new",
    ).prepare()
    c, y = jnp.array([1.3, -0.4]), jnp.array([0.2, -0.3, 0.7])
    new = route.apply(c)
    np.testing.assert_allclose(
        jnp.vdot(route.target_measures, new),
        jnp.vdot(route.source_measures, c),
        atol=1e-12,
    )
    dual = route.transfer.dual_pullback_operator
    hilbert = route.transfer.hilbert_adjoint_operator
    assert dual is not None and hilbert is not None
    np.testing.assert_allclose(jnp.vdot(new, y), jnp.vdot(c, dual.mv(y)), atol=1e-12)
    np.testing.assert_allclose(
        jnp.vdot(route.target_measures * new, y),
        jnp.vdot(route.source_measures * c, hilbert.mv(y)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        route.transfer.target.vector_space.inner(new, y),
        route.transfer.source.vector_space.inner(c, hilbert.mv(y)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        route.apply_content(route.source_measures * c),
        route.target_measures * new,
        atol=1e-12,
    )


def test_transfer_refuses_uncovered_and_negative_positive_allocations() -> None:
    uncovered = EdgeRelation(
        np.asarray([0], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        source_size=2,
        target_size=1,
    )
    with pytest.raises(ValueError, match="no cross-target route"):
        SurfaceTransferPlan(
            uncovered,
            np.asarray([1.0], dtype=np.float64),
            np.asarray([1.0, 1.0], dtype=np.float64),
            np.asarray([2.0], dtype=np.float64),
            source_id="old",
            target_id="new",
        ).prepare()
    negative = RowRelation(np.asarray([[0, 1]], dtype=np.int32), source_size=2)
    with pytest.raises(ValueError, match="nonnegative base"):
        SurfaceTransferPlan(
            negative,
            np.asarray([[1.0, -1.0]], dtype=np.float64),
            np.asarray([1.0, 1.0], dtype=np.float64),
            np.asarray([2.0], dtype=np.float64),
            source_id="old",
            target_id="new",
            mode="positive",
        ).prepare()


def test_transfer_refuses_case_local_routes_without_global_column_identity() -> None:
    relation = RowRelation(
        np.zeros((2, 1, 1), dtype=np.int32), source_size=1, case_shape=(2,)
    )
    with pytest.raises(ValueError, match="unbatched compact target"):
        SurfaceTransferPlan(
            relation,
            np.ones((2, 1, 1)),
            np.ones(1),
            np.ones(2),
            source_id="old",
            target_id="new",
        )
