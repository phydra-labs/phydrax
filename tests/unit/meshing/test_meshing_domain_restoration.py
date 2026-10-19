#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np
import pytest

from phydrax._fingerprint import tree_content_equal
from phydrax._model._structure import (
    model_from_logical_array_recipe,
    model_recipe_array_inventory,
    model_recipe_array_values,
    model_structure_recipe,
)
from phydrax.geometry import LineCurve, MeshingDomain, MeshingDomainCurve
from phydrax.geometry._meshing_domain import PhysicalCurveMap
from phydrax.lifecycle import register_meshing_source_artifacts
from tests._support.cad_models import native_solid
from tests._support.meshing_domain import square_patch
from tests._support.recipes import recipe_field_position


def test_stored_parametric_domain_rebuild_preserves_declared_metadata() -> None:
    curves = tuple(MeshingDomainCurve(*ends) for ends in ((0, 1), (1, 2), (2, 3), (3, 0)))
    stored = MeshingDomain(
        (square_patch((0, 1, 2, 3)),),
        curves,
        4,
        source_id="stored-square",
        source_revision="stored-r1",
        tolerance=2.0e-8,
        accuracy="bounded_approximation",
        source_indices=((10, 11, 12, 13), (20, 21, 22, 23), (30,)),
        source_occurrences=((("part",),) * 4, (("part",),) * 4, (("part",),)),
        authority_id="declared-square-source",
    )
    rebuilt = MeshingDomain.from_stored_fields(stored)
    assert rebuilt is not stored
    assert tree_content_equal(rebuilt, stored)
    assert rebuilt.content_domain_id() == stored.domain_id


def test_stored_parametric_domain_rebuild_rejects_corrupted_tolerance() -> None:
    curves = tuple(MeshingDomainCurve(*ends) for ends in ((0, 1), (1, 2), (2, 3), (3, 0)))
    stored = MeshingDomain(
        (square_patch((0, 1, 2, 3)),),
        curves,
        4,
        source_id="stored-square",
        source_revision="stored-r1",
    )
    register_meshing_source_artifacts()
    recipe = model_structure_recipe(stored)
    arrays = model_recipe_array_values(stored, recipe, prefix="stored-domain")
    recipe["items"][recipe_field_position(recipe, "tolerance")] = model_structure_recipe(
        0.0
    )
    corrupted = model_from_logical_array_recipe(recipe, arrays, prefix="stored-domain")
    assert isinstance(corrupted, MeshingDomain)
    with pytest.raises(ValueError, match="MeshingDomain.tolerance"):
        MeshingDomain.from_stored_fields(corrupted)


def test_stored_brep_model_domain_rebuild_preserves_physical_carriers() -> None:
    model = native_solid("box")
    stored = MeshingDomain.from_brep(model)
    assert stored.brep_authority is model
    rebuilt = MeshingDomain.from_stored_fields(stored)
    assert tree_content_equal(rebuilt, stored)
    assert rebuilt.brep_authority is model
    assert rebuilt.content_domain_id() == stored.domain_id


def test_stored_brep_geometry_domain_rebuild_preserves_source_metadata() -> None:
    model = native_solid("box")
    geometry = model.geometry
    assert geometry is not None
    stored = MeshingDomain.from_brep_geometry(
        geometry,
        model.patches,
        np.asarray(model.orientation, dtype=np.float64),
        source_id=model.source_id,
        source_revision=model.source_revision,
        authority_id=model.model_id,
    )
    assert stored.brep_authority is geometry
    rebuilt = MeshingDomain.from_stored_fields(stored)
    assert tree_content_equal(rebuilt, stored)
    assert rebuilt.brep_authority is geometry
    assert rebuilt.content_domain_id() == stored.domain_id


def test_stored_physical_curve_map_retains_complete_endpoint_error_bank() -> None:
    stored = PhysicalCurveMap(
        (LineCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),),
        np.asarray(((0.0, 1.0),), dtype=np.float64),
        vertex_realization_errors=np.asarray(((0.125, 0.25),), dtype=np.float64),
    )
    errors = stored.endpoint_errors.copy()
    rebuilt = PhysicalCurveMap.from_stored_fields(stored)
    assert rebuilt is stored
    assert tree_content_equal(rebuilt, stored)
    np.testing.assert_array_equal(rebuilt.endpoint_errors, errors)
    assert not rebuilt.endpoint_errors.flags.writeable


@pytest.mark.parametrize("invalid", (-0.125, np.nan), ids=("negative", "nonfinite"))
def test_stored_physical_curve_map_refuses_forged_endpoint_error_bank(
    invalid: float,
) -> None:
    stored = PhysicalCurveMap(
        (LineCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),),
        np.asarray(((0.0, 1.0),), dtype=np.float64),
        vertex_realization_errors=np.asarray(((0.125, 0.25),), dtype=np.float64),
    )
    register_meshing_source_artifacts()
    recipe = model_structure_recipe(stored)
    arrays = model_recipe_array_values(stored, recipe, prefix="physical-map")
    error_field = next(
        entry
        for entry in model_recipe_array_inventory(recipe, prefix="physical-map")
        if entry.path.endswith(".endpoint_errors")
    )
    forged = np.array(arrays[error_field.name], dtype=np.float64, copy=True)
    forged[0, 0] = invalid
    arrays[error_field.name] = forged
    with pytest.raises(ValueError, match="PhysicalCurveMap.endpoint_errors"):
        corrupted = model_from_logical_array_recipe(recipe, arrays, prefix="physical-map")
        assert isinstance(corrupted, PhysicalCurveMap)
        PhysicalCurveMap.from_stored_fields(corrupted)
