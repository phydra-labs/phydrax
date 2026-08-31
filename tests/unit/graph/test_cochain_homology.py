#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from tests.unit.topology._fixtures import annulus_complex


def test_annulus_exact_betti_dimensions_match_absolute_harmonics():
    complex_ir = annulus_complex()
    harmonics, report = phx.graph.validate_hodge_homology(complex_ir, 1)

    assert harmonics.ranks == (1, 1, 0)
    assert report.exact_dimension == 1
    assert report.harmonic_rank == 1
    assert bool(report.ranks_match)
    assert bool(report.complete)
    assert float(jnp.max(report.kernel_residuals)) < 1e-8


def test_annulus_relative_homology_uses_boundary_pair():
    complex_ir = annulus_complex()
    harmonics, report = phx.graph.validate_hodge_homology(
        complex_ir,
        2,
        boundary_policy="relative",
    )

    assert harmonics.ranks == (0, 1, 1)
    assert report.exact_dimension == 1
    assert report.harmonic_rank == 1
    assert bool(report.complete)


def test_harmonic_kernel_certificate_is_compact_and_verified():
    complex_ir = annulus_complex()
    subspace, certificate, report = phx.graph.cochain_harmonic_kernel_certificate(
        complex_ir,
        1,
    )

    assert subspace.capacity == 1
    assert int(subspace.dimension) == 1
    assert certificate.complete
    assert bool(certificate.valid)
    assert bool(report.complete)
    assert np.max(np.asarray(certificate.right_residual_norms)) < 1e-8


def test_hodge_bridge_rejects_harmonics_from_other_boundary_policy():
    complex_ir = annulus_complex()
    relative = phx.graph.compute_harmonic_subspace(
        complex_ir,
        boundary_policy="relative",
        max_modes=3,
    )
    with pytest.raises(ValueError, match="different boundary policy"):
        phx.graph.validate_hodge_homology(
            complex_ir,
            1,
            boundary_policy="absolute",
            harmonic_subspace=relative,
        )


def test_hodge_bridge_rejects_harmonics_from_other_complex():
    left = annulus_complex()
    right = phx.graph.triangle_mesh_to_cochain_complex(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[0, 1, 2]], dtype=np.int32),
    )
    harmonics = phx.graph.compute_harmonic_subspace(right, max_modes=3)
    with pytest.raises(ValueError, match="different metric complex"):
        phx.graph.validate_hodge_homology(left, 1, harmonic_subspace=harmonics)
