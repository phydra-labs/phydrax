"""Support evidence and refusal ordering for native image and PLC providers."""

from dataclasses import replace

import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import meshcore_available
from phydrax.meshing.providers._native_compartment import (
    compartment_support_issues,
    PreparedCompartmentVolume,
)
from phydrax.meshing.providers._native_support import format_support_issues
from tests._support.image_sources import _source, image_specification
from tests._support.meshing_fixtures import box_plc


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="native volume providers require meshcore"
)
M = phx.meshing


def test_image_support_checks_keep_target_then_fill_validation_order() -> None:
    source = _source(np.ones((1, 1, 1), dtype=np.int16))
    specification = replace(
        image_specification(source),
        target=M.CellMeshingTarget(
            3, 3, M.CellFamilyPolicy(required=("hexahedron",)), geometry_order=2
        ),
        fill_strategy=M.VolumeFillStrategy.POLYHEDRAL,
    )
    issues = compartment_support_issues(source, specification)
    assert tuple(issue.code for issue in issues) == (
        "geometry",
        "cell-families",
        "fill-strategy",
    )
    assert format_support_issues(issues) == (
        "affine three-dimensional tetrahedral geometry; "
        "a pure tetrahedral cell-family policy; the simplex volume-fill strategy"
    )
    report = M.NativeMeshingProvider(
        M.NativeMeshingOptions("image_material_tetrahedral")
    ).validate(source, specification)
    assert not report.supported
    assert report.unsupported == tuple(issue.detail for issue in issues)


def test_image_preparation_formats_typed_support_findings_at_refusal() -> None:
    from phydrax._meshcore import NativeExecutionScope

    source = _source(np.ones((1, 1, 1), dtype=np.int16))
    specification = replace(
        image_specification(source), fill_strategy=M.VolumeFillStrategy.POLYHEDRAL
    )
    with (
        NativeExecutionScope(specification.limits.native_limits()) as scope,
        scope.host_workspace() as workspace,
        pytest.raises(ValueError, match="PreparedCompartmentVolume.specification"),
    ):
        PreparedCompartmentVolume(
            source, specification, M.NativeVolumeSchedule(), scope=scope, workspace=workspace
        )


def test_plc_provider_preserves_geometry_family_then_fill_issue_order() -> None:
    complex_ = box_plc()
    source = M.NativePlcSource(complex_, "cube", "reference")
    boundary = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "cube:facets",
        np.arange(complex_.facet_count, dtype=np.int64),
    )
    specification = M.VolumeMeshingSpec(
        M.CellMeshingTarget(
            3, 3, M.CellFamilyPolicy(required=("hexahedron",)), geometry_order=2
        ),
        boundary,
        M.VolumeFillStrategy.POLYHEDRAL,
        size_controls=(M.UniformSizeControl(boundary, 3.0),),
    )
    report = M.NativeMeshingProvider(M.NativeMeshingOptions("plc_tetrahedral")).validate(
        source, specification
    )
    assert not report.supported
    assert report.unsupported == (
        "a tetrahedral volume target",
        "affine tetrahedra in ambient dimension three",
        "the simplex fill strategy",
    )
