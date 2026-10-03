# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Higher-form meshfree workflow: holed, 3-D, curved sheets and Maxwell consumer."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from examples.meshfree_higher_forms import (
    audit_complex,
    holed_square,
    maxwell_flux_evolution,
    research_clique_circle,
    sampled_commutation_errors,
    solid_torus,
    unit_sphere,
)
from phydrax.discretization.meshfree import PreparedMeshfreeCellComplex


@pytest.fixture(scope="module")
def torus() -> PreparedMeshfreeCellComplex:
    return solid_torus(1)


@pytest.mark.parametrize(
    "build",
    [holed_square, unit_sphere],
    ids=["square-with-hole", "closed-sphere"],
)
def test_planar_and_curved_audit(
    build: Callable[[], PreparedMeshfreeCellComplex],
) -> None:
    complex_ = build()
    report = audit_complex(complex_)
    assert report.admitted
    assert report.harmonic_ranks == report.betti
    assert max(report.commutation) < 1e-9
    assert report.stokes_defect < 1e-10


def test_solid_torus_audit(torus: PreparedMeshfreeCellComplex) -> None:
    report = audit_complex(torus)
    assert report.admitted
    assert report.betti == (1, 1, 0, 0)
    assert report.harmonic_ranks == report.betti
    assert max(report.commutation) < 1e-9
    assert report.stokes_defect < 1e-10


def test_maxwell_evolves_faces_under_the_volume_constraint(
    torus: PreparedMeshfreeCellComplex,
) -> None:
    report = maxwell_flux_evolution(torus, steps=12)
    assert report.stable_dt > 0.0
    assert report.electric_activity > 1e-6
    assert report.magnetic_divergence < 1e-11
    assert report.energy_drift < 5e-2


def test_sampled_gmls_moments_converge_under_refinement() -> None:
    errors = sampled_commutation_errors((1, 2))
    assert errors[1] < 0.5 * errors[0], errors


def test_abstract_clique_records_topology_only() -> None:
    betti, work = research_clique_circle()
    assert betti[:2] == (1, 1)
    assert work > 0
    assert np.all(np.asarray(betti) >= 0)
