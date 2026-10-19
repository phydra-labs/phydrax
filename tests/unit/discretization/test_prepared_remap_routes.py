#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from phydrax.discretization import (
    prepare_unstructured_conservative_remap,
    PreparedUnstructuredConservativeRemap,
    UnstructuredFiniteVolumePlan,
)
from phydrax.geometry import CommonRefinementStatus, PreparedCommonRefinement
from tests._support.cell_meshes import unit_square


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="phydrax-meshcore unavailable"
)


@pytest.fixture(scope="module")
def prepared_remap() -> PreparedUnstructuredConservativeRemap:
    geometry = UnstructuredFiniteVolumePlan.from_cell_mesh(unit_square()).prepare()
    prepared = prepare_unstructured_conservative_remap(
        geometry, geometry, provenance="prepared-route-contract"
    )
    assert prepared.succeeded, prepared.reason
    return prepared


def test_prepared_common_route_retains_geometry_evidence(
    prepared_remap: PreparedUnstructuredConservativeRemap,
) -> None:
    match prepared_remap.route:
        case PreparedCommonRefinement() as refinement:
            assert prepared_remap.evidence is refinement.evidence
            assert refinement.succeeded
        case _:
            pytest.fail("The corner-cell remap must retain common-refinement geometry.")


def test_restored_remap_preserves_route_plan_and_content_identity(
    prepared_remap: PreparedUnstructuredConservativeRemap,
) -> None:
    restored = PreparedUnstructuredConservativeRemap.from_stored_fields(prepared_remap)
    assert restored.route is prepared_remap.route
    assert restored.plan is prepared_remap.plan
    assert restored.remap_id == prepared_remap.remap_id
    assert restored.status is prepared_remap.status
    assert restored.succeeded == prepared_remap.succeeded
    assert jax.tree_util.tree_structure(restored) == jax.tree_util.tree_structure(
        prepared_remap
    )
    for restored_leaf, prepared_leaf in zip(
        jax.tree_util.tree_leaves(restored),
        jax.tree_util.tree_leaves(prepared_remap),
        strict=True,
    ):
        np.testing.assert_array_equal(restored_leaf, prepared_leaf)


def test_prepared_route_refuses_absent_geometry_evidence(
    prepared_remap: PreparedUnstructuredConservativeRemap,
) -> None:
    with pytest.raises(TypeError, match="route"):
        PreparedUnstructuredConservativeRemap(
            None,  # ty: ignore[invalid-argument-type]
            prepared_remap.plan,
            status=CommonRefinementStatus.SUCCESS,
            reason="absent geometry evidence",
        )


def test_prepared_route_requires_plan_for_success(
    prepared_remap: PreparedUnstructuredConservativeRemap,
) -> None:
    with pytest.raises(ValueError, match=r"remap\.plan"):
        PreparedUnstructuredConservativeRemap(
            prepared_remap.route,
            None,
            status=CommonRefinementStatus.SUCCESS,
            reason="missing content plan",
        )


def test_prepared_route_refuses_plan_for_failed_status(
    prepared_remap: PreparedUnstructuredConservativeRemap,
) -> None:
    with pytest.raises(ValueError, match=r"remap\.plan"):
        PreparedUnstructuredConservativeRemap(
            prepared_remap.route,
            prepared_remap.plan,
            status=CommonRefinementStatus.COVERAGE_GAP,
            reason="incompatible content-plan acceptance",
        )


def test_restored_remap_refuses_lost_success_plan(
    prepared_remap: PreparedUnstructuredConservativeRemap,
) -> None:
    corrupted = eqx.tree_at(lambda value: value.plan, prepared_remap, None)
    with pytest.raises(ValueError, match=r"remap\.plan"):
        PreparedUnstructuredConservativeRemap.from_stored_fields(corrupted)
