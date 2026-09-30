#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from tests.unit.topology._fixtures import annulus_complex


@pytest.mark.parametrize(
    ("degree", "boundary", "betti"),
    [
        (0, "absolute", 1),
        (1, "absolute", 1),
        (2, "absolute", 0),
        (0, "relative", 0),
        (1, "relative", 1),
        (2, "relative", 1),
    ],
)
def test_annulus_harmonic_dimension_matches_exact_topology(
    degree: int, boundary: phx.exterior.ComplexBoundary, betti: int
) -> None:
    realization = annulus_complex()
    harmonic, report = phx.exterior.validate_harmonic_cohomology(
        realization, degree, boundary=boundary
    )
    assert harmonic.dimension == betti
    assert report.exact_dimension == betti
    assert bool(report.complete)
    assert float(jnp.max(report.kernel_residuals, initial=0.0)) < 1e-8
    assert (
        harmonic.basis.shape[0]
        == realization.hilbert_complex(boundary=boundary).space(degree).size
    )


def test_relative_harmonic_certificate_uses_only_active_coordinates() -> None:
    realization = annulus_complex()
    subspace, certificate, report = phx.exterior.harmonic_kernel_certificate(
        realization, 1, boundary="relative"
    )
    assert subspace.capacity == 1
    assert subspace.space.size == int(
        np.count_nonzero(~np.asarray(realization.boundary_masks[1]))
    )
    assert certificate.complete
    assert bool(certificate.valid)
    assert bool(report.complete)
    assert np.max(np.asarray(certificate.right_residual_norms)) < 1e-8


def test_harmonic_evidence_refuses_different_metric_or_boundary() -> None:
    realization = annulus_complex()
    relative, _ = phx.exterior.validate_harmonic_cohomology(
        realization, 1, boundary="relative"
    )
    with pytest.raises(ValueError):
        phx.exterior.validate_harmonic_cohomology(realization, 1, harmonic=relative)
    altered = realization.with_metric(
        tuple(
            phx.discretization.DiagonalHodge(2 * realization.hodge_diagonal(k))
            for k in range(realization.dimension + 1)
        ),
        numeric_revision="annulus-doubled-metric",
    )
    harmonic, _ = phx.exterior.validate_harmonic_cohomology(altered, 1)
    with pytest.raises(ValueError):
        phx.exterior.validate_harmonic_cohomology(realization, 1, harmonic=harmonic)
