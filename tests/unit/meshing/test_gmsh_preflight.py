"""Gmsh support preflight through the public provider, independent of Gmsh installs."""

import pytest

from phydrax.meshing import (
    CellFamilyPolicy,
    CellMeshingTarget,
    SizeCombinationPolicy,
    SizeControlStrength,
    SurfaceMeshingSpec,
    UniformSizeControl,
)
from phydrax.meshing.providers import GmshProvider
from tests._support.cad_models import native_solid


@pytest.mark.parametrize(
    ("combination", "strength", "conflicting"),
    [
        pytest.param(
            SizeCombinationPolicy.REJECT_HARD_CONFLICTS,
            SizeControlStrength.HARD,
            True,
            id="hard-target-conflict",
        ),
        pytest.param(
            SizeCombinationPolicy.EXPLICIT_PRIORITY,
            SizeControlStrength.HARD,
            True,
            id="equal-priority-hard-conflict",
        ),
        pytest.param(
            SizeCombinationPolicy.REJECT_HARD_CONFLICTS,
            SizeControlStrength.SOFT,
            False,
            id="soft-minimum-selection",
        ),
    ],
)
def test_gmsh_preflight_reports_overlapping_uniform_target_contract(
    combination: SizeCombinationPolicy,
    strength: SizeControlStrength,
    conflicting: bool,
) -> None:
    source = native_solid("box")
    provider = GmshProvider()
    scope = provider.whole_scope(source, 2)
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            UniformSizeControl(scope, 0.5, strength=strength),
            UniformSizeControl(scope, 0.75, strength=strength),
        ),
        size_combination=combination,
    )
    report = provider.validate(source, specification)
    issue = "Overlapping UniformSizeControl values have incompatible hard intervals or targets"
    assert (issue in report.unsupported) is conflicting


def test_gmsh_preflight_reports_disjoint_hard_size_intervals() -> None:
    source = native_solid("box")
    provider = GmshProvider()
    scope = provider.whole_scope(source, 2)
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            UniformSizeControl(scope, 0.5, minimum_size=0.25, maximum_size=0.75),
            UniformSizeControl(scope, 1.5, minimum_size=1.0, maximum_size=2.0),
        ),
    )
    report = provider.validate(source, specification)
    assert "Whole-source hard size intervals conflict" in report.unsupported
    assert (
        "Overlapping UniformSizeControl values have incompatible hard intervals or targets"
        in report.unsupported
    )
