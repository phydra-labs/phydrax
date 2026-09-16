#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.conformal_bootstrap import (
    ConformalDataPlan,
    CrossingChannel,
    CrossingSectorPlan,
    ExchangedOperatorSector,
    ExternalScalarOperator,
    GlobalScalarBlockPlan,
    prepare_crossing_sectors,
    prepare_global_scalar_blocks,
)
from phydrax.tensor_network import Irrep, RepresentationCategory


def _z2_category():
    unit = Irrep("even", 1, dual_label="even")
    odd = Irrep("odd", 1, dual_label="odd")
    return RepresentationCategory(
        (unit, odd),
        (
            ("even", "even", (("even", 1),)),
            ("even", "odd", (("odd", 1),)),
            ("odd", "even", (("odd", 1),)),
            ("odd", "odd", (("even", 1),)),
        ),
        unit_label="even",
    )


def _data(dimension, *, category=None):
    representation = "odd" if category is not None else "scalar"
    external = tuple(
        ExternalScalarOperator(f"phi-{index}", 0.6, representation) for index in range(4)
    )
    sector_representation = "even" if category is not None else "scalar"
    sectors = (
        ExchangedOperatorSector(
            "even-scalars",
            sector_representation,
            (0, 2),
            parity="even",
            minimum_dimension=max(0.1, (dimension - 2.0) / 2.0),
        ),
    )
    channels = (CrossingChannel("s-t", (2, 1, 0, 3), involutive=True),)
    return ConformalDataPlan(
        dimension,
        external,
        sectors,
        channels,
        category=category,
    )


def test_conformal_data_validates_category_fusion_and_crossing_gauge():
    data = _data(3.0, category=_z2_category())
    plan = CrossingSectorPlan(
        data,
        ("symmetric", "antisymmetric"),
        (((0.0, 1.0), (1.0, 0.0)),),
        basis_gauge_id="z2-explicit-gauge",
    )
    prepared = prepare_crossing_sectors(plan)
    vector = jnp.asarray((2.0, -1.0))
    np.testing.assert_allclose(prepared.apply("s-t", vector), (-1.0, 2.0))
    np.testing.assert_allclose(
        prepared.inverse_apply("s-t", prepared.apply("s-t", vector)), vector
    )
    assert bool(prepared.evidence.accepted)
    np.testing.assert_allclose(prepared.evidence.involution_residuals, 0.0)


def test_crossing_compiler_rejects_false_involution():
    data = _data(3.0)
    plan = CrossingSectorPlan(
        data,
        ("a", "b"),
        (((1.0, 1.0), (0.0, 1.0)),),
        basis_gauge_id="invalid-involution",
    )
    with pytest.raises(ValueError, match="involution"):
        prepare_crossing_sectors(plan)


def test_two_dimensional_global_block_matches_factorized_closed_reference():
    data = _data(2.0)
    prepared = prepare_global_scalar_blocks(
        GlobalScalarBlockPlan(
            data,
            ((0.2, 0.2),),
            (0,),
            ((0, 0), (1, 0)),
            recursion_order=4,
            hypergeometric_order=384,
        )
    )
    values = prepared.values(2.0, 0)
    expected = np.log(0.8) ** 2
    np.testing.assert_allclose(values, (expected,), rtol=1e-12, atol=1e-13)
    evidence = prepared.evidence(2.0, 0)
    assert bool(evidence.finite)
    assert float(jnp.max(evidence.truncation_proxy)) < 1e-12
    assert float(jnp.max(evidence.casimir_residuals)) < 1e-8
    assert "finite" in evidence.claim


def test_three_dimensional_radial_recursion_is_symmetric_and_finite():
    data = _data(3.0)
    prepared = prepare_global_scalar_blocks(
        GlobalScalarBlockPlan(
            data,
            ((0.2, 0.3), (0.3, 0.2)),
            (0, 2),
            ((0, 0),),
            recursion_order=4,
            hypergeometric_order=128,
        )
    )
    evidence = prepared.evidence(1.2, 0)
    assert bool(evidence.finite)
    np.testing.assert_allclose(evidence.values[0], evidence.values[1], rtol=1e-11)
    assert float(jnp.max(evidence.truncation_proxy)) < 1.0
    assert np.all(np.isfinite(np.asarray(evidence.casimir_residuals)))


def test_four_dimensional_diagonal_limit_and_derivative_are_finite():
    prepared = prepare_global_scalar_blocks(
        GlobalScalarBlockPlan(
            _data(4.0),
            ((0.3, 0.3),),
            (0,),
            ((0, 0), (1, 0)),
            recursion_order=2,
            hypergeometric_order=256,
        )
    )
    evidence = prepared.evidence(3.0, 0)
    assert bool(evidence.finite)
    assert np.all(np.isfinite(np.asarray(evidence.values)))
    assert np.all(np.isfinite(np.asarray(evidence.derivatives)))
