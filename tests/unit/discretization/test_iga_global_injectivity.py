# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Global IGA certificate decisions on bounded affine control nets."""

from __future__ import annotations

import numpy as np
import pytest

from phydrax.discretization.iga import BSplineGrid, NURBSGeometryState
from phydrax.discretization.iga._basis import TensorSplineBasisSpec
from phydrax.discretization.iga._certificate import (
    CertificateDisposition,
    certify_global_injectivity,
)
from phydrax.discretization.iga._volume import BlockComplex, TensorNURBSVolume


def _affine_volume(block_id: str, *, reversed_x: bool = False) -> TensorNURBSVolume:
    grid = BSplineGrid(np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float64), 1)
    basis = TensorSplineBasisSpec((grid, grid), axis_names=("xi", "eta"))
    x, y = np.meshgrid(np.asarray((0.0, 1.0)), np.asarray((0.0, 1.0)), indexing="ij")
    if reversed_x:
        x = 1.0 - x
    controls = np.stack((x, y), axis=-1)
    return TensorNURBSVolume(
        block_id, basis, NURBSGeometryState(controls, np.ones((2, 2), dtype=np.float64))
    )


@pytest.mark.parametrize(
    ("reversed_x", "disposition", "degree", "has_counterexample"),
    (
        pytest.param(False, CertificateDisposition.PASS, 1, False, id="positive-square"),
        pytest.param(
            True, CertificateDisposition.COUNTEREXAMPLE, None, True, id="reflected-square"
        ),
    ),
)
def test_global_certificate_preserves_local_orientation_decisions(
    reversed_x: bool,
    disposition: CertificateDisposition,
    degree: int | None,
    has_counterexample: bool,
) -> None:
    complex_ = BlockComplex((_affine_volume("square", reversed_x=reversed_x),))
    result = certify_global_injectivity(complex_)

    assert result.disposition is disposition
    assert result.boundary.disposition is disposition
    assert result.mapping_degree.disposition is disposition
    assert result.mapping_degree.degree == degree
    assert (result.counterexample is not None) is has_counterexample
    assert result.complex_id == complex_.complex_id
    assert result.boundary.diagnostics == result.diagnostics
    assert result.boundary.incidence_ids == ()
    assert result.boundary.separation_pairs == ()


def test_global_certificate_keeps_connectivity_before_separation_diagnostics() -> None:
    complex_ = BlockComplex((_affine_volume("first"), _affine_volume("second")))
    result = certify_global_injectivity(complex_)

    assert result.disposition is CertificateDisposition.INCONCLUSIVE
    assert result.counterexample is None
    assert result.mapping_degree.degree is None
    assert tuple(value.code for value in result.diagnostics) == (
        "block-complex-disconnected",
        "nonadjacent-separation-unproved",
    )
    assert result.boundary.diagnostics == result.diagnostics


def test_global_certificate_refuses_an_untyped_complex_at_the_boundary() -> None:
    with pytest.raises(TypeError, match="complex_"):
        certify_global_injectivity(None)  # ty: ignore[invalid-argument-type]
