#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from fractions import Fraction
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.conformal_bootstrap import (
    compile_global_blocks_to_pmp,
    ConformalDataPlan,
    CrossingChannel,
    CrossingSectorPlan,
    ExchangedOperatorSector,
    ExternalScalarOperator,
    GlobalBlockPMPPlan,
    GlobalScalarBlockPlan,
    prepare_crossing_sectors,
    prepare_global_scalar_blocks,
)
from phydrax.tensor_network import Irrep, RepresentationCategory


def _z2_category() -> Any:
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


def _data(dimension: Any, *, category: Any = None) -> Any:
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


def test_conformal_crossing_ir_scenario_1() -> None:
    data = _data(3.0, category=_z2_category())
    plan = CrossingSectorPlan(
        data,
        ("symmetric", "antisymmetric"),
        # ty: ignore[invalid-argument-type]
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
    data = _data(3.0)
    plan = CrossingSectorPlan(
        data,
        ("a", "b"),
        # ty: ignore[invalid-argument-type]
        (((1.0, 1.0), (0.0, 1.0)),),
        basis_gauge_id="invalid-involution",
    )
    with pytest.raises(ValueError, match="involution"):
        prepare_crossing_sectors(plan)
    data = _data(2.0)
    prepared = prepare_global_scalar_blocks(
        GlobalScalarBlockPlan(
            data,
            # ty: ignore[invalid-argument-type]
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


def test_conformal_crossing_ir_scenario_2() -> None:
    data = _data(3.0)
    prepared = prepare_global_scalar_blocks(
        GlobalScalarBlockPlan(
            data,
            # ty: ignore[invalid-argument-type]
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
    prepared = prepare_global_scalar_blocks(
        GlobalScalarBlockPlan(
            _data(4.0),
            # ty: ignore[invalid-argument-type]
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


def test_global_block_fit_evidence_measures_the_published_polynomials() -> None:
    prepared = prepare_global_scalar_blocks(
        GlobalScalarBlockPlan(
            _data(2.0),
            # ty: ignore[invalid-argument-type]
            ((0.2, 0.2),),
            (0,),
            ((0, 0), (1, 0)),
            recursion_order=4,
            hypergeometric_order=384,
        )
    )
    plan = GlobalBlockPMPPlan(
        prepared,
        (1.0,),
        (4.0,),
        polynomial_degree=14,
        sample_count=24,
        significant_digits=8,
        maximum_fit_error=1e-3,
    )
    compiled = compile_global_blocks_to_pmp(plan)
    dimensions = np.asarray(compiled.evidence.sample_dimensions)[0]
    published = compiled.program.blocks[0].polynomials[0][0]
    table = [
        np.asarray(prepared.derivative_table(value, 0)).reshape(-1)
        for value in dimensions
    ]

    def evaluate(coefficients: tuple[Fraction, ...], point: Fraction) -> Fraction:
        value = Fraction(0)
        for coefficient in reversed(coefficients):
            value = value * point + coefficient
        return value

    residual = Fraction(0)
    deviation = Fraction(0)
    grid = [Fraction(index, 64) * 3 for index in range(65)]
    for functional, strings in enumerate(published):
        exact = tuple(Fraction(value) for value in strings)
        for dimension, row in zip(dimensions.tolist(), table, strict=True):
            target = Fraction(float(row[functional]))
            value = evaluate(exact, Fraction(dimension) - 1)
            residual = max(residual, abs(target - value) / max(1, abs(target)))
        fitted = tuple(
            Fraction(value)
            for value in np.polynomial.polynomial.polyfit(
                dimensions - 1.0, [row[functional] for row in table], 14
            ).tolist()
        )
        for point in grid:
            deviation = max(
                deviation, abs(evaluate(exact, point) - evaluate(fitted, point))
            )
    # Rounding to eight digits dominates the unrounded least-squares residual.
    assert residual > 1e-9
    assert Fraction(float(compiled.evidence.maximum_relative_errors[0])) >= residual
    assert (
        Fraction(float(compiled.evidence.maximum_publication_deviations[0])) >= deviation
    )
