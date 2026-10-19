#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax._model._structure import (
    model_from_array_recipe,
    model_recipe_array_values,
    model_structure_recipe,
)
from phydrax.geometry import (
    CircleCurve,
    intersect_surface_regions,
    LineCurve,
    MeshingDomain,
    MeshingDomainBoundarySource,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
    PlanePatch,
    SurfaceRegion,
)
from phydrax.geometry._meshing_domain import PhysicalCurveMap
from phydrax.geometry.brep import NativePeriodEndpoint
from phydrax.geometry.brep._intersection import (
    BranchRootEndpoint,
    CurveSurfaceIntersectionRoot,
)
from phydrax.lifecycle._meshing_sources import (
    register_meshing_source_artifacts,
    validate_meshing_source_closure,
)
from tests._support.meshing_domain import (
    ROOT_POSITION_ATOL,
    SOURCE_ROUNDOFF_ATOL,
    TRIM_RADIUS_ATOL,
)


def test_source_branch_root_survives_curve_use_and_physical_atlas_lowering() -> None:

    first = SurfaceRegion(
        PlanePatch(
            np.asarray((-1.0, 0.0, 0.0)),
            np.asarray((2.0, 0.0, 0.0)),
            np.asarray((0.0, 1.0, 0.0)),
        ),
        np.asarray(((0.0, 0.0), (1.0, 1.0))),
    )
    second = SurfaceRegion(
        PlanePatch(
            np.asarray((0.0, -1.0, -1.0)),
            np.asarray((0.0, 3.0, 0.0)),
            np.asarray((0.0, 0.0, 2.0)),
        ),
        np.asarray(((0.0, 0.0), (1.0, 1.0))),
    )
    intersection = intersect_surface_regions(first, second)
    assert intersection.complete
    (branch,) = intersection.curves
    edge = LineCurve(np.asarray((-1.0, 0.37, 0.0)), np.asarray((2.0, 0.0, 0.0)))
    root = CurveSurfaceIntersectionRoot(
        edge,
        second,
        parameter_lower=np.asarray((0.49, 0.45, 0.49)),
        parameter_upper=np.asarray((0.51, 0.46, 0.51)),
    )
    image = np.asarray((0.5, 0.37, 1.37 / 3.0, 0.5))
    charts = np.flatnonzero(
        np.all(
            (np.asarray(branch.box_lower) <= image)
            & (image <= np.asarray(branch.box_upper)),
            axis=1,
        )
    )
    endpoint = BranchRootEndpoint(
        root,
        branch,
        int(charts[0]),
        source_pcurve=LineCurve(np.asarray((0.0, 0.37)), np.asarray((1.0, 0.0))),
        source_side="first",
        source_first=0.0,
        source_last=1.0,
    )
    lower, upper = endpoint.parameter_enclosure()
    nominal = 0.5 * (lower + upper)
    use = PatchCurveUse(
        0,
        branch.p_curve("first"),
        nominal,
        float(branch.num_charts),
        first_root=endpoint,
    )
    mapping = PhysicalCurveMap(
        (branch,),
        np.asarray(((nominal, float(branch.num_charts)),)),
        ((endpoint, None),),
    )
    point = np.asarray(
        mapping.map(jnp.asarray((0,), dtype=jnp.int32), jnp.asarray(((0.0,),)))
    )
    # Independent source-plane/edge intersection, including transported scalar
    # root uncertainty rather than a replacement nominal junction definition.
    np.testing.assert_allclose(
        point[0], (0.0, 0.37, 0.0), rtol=0.0, atol=ROOT_POSITION_ATOL
    )
    assert use.first_root is endpoint
    assert np.all(np.isfinite(mapping.endpoint_errors))
    assert mapping.endpoint_errors[0, 0] > 0.0


def test_curved_trim_queries_survive_original_source_recipe_restoration() -> None:

    outer = CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 1.0)
    hole = CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 0.2)
    patch = MeshingSurfacePatch(
        PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        (
            (
                PatchCurveUse(
                    0,
                    outer,
                    0.0,
                    2 * np.pi,
                    first_root=NativePeriodEndpoint(outer, turns=0),
                    last_root=NativePeriodEndpoint(outer, turns=1),
                ),
            ),
            (
                PatchCurveUse(
                    1,
                    hole,
                    2 * np.pi,
                    0.0,
                    first_root=NativePeriodEndpoint(hole, turns=1),
                    last_root=NativePeriodEndpoint(hole, turns=0),
                ),
            ),
        ),
    )
    domain = MeshingDomain(
        (patch,),
        (MeshingDomainCurve(0, 0), MeshingDomainCurve(1, 1)),
        2,
        source_id="authored-annulus",
        source_revision="circle-r1",
    )
    source = MeshingDomainBoundarySource(domain, (0,), resolution=4)
    register_meshing_source_artifacts()
    validate_meshing_source_closure(source)
    recipe = model_structure_recipe(source)
    restored = model_from_array_recipe(
        recipe,
        model_recipe_array_values(source, recipe, prefix="trim-source"),
        prefix="trim-source",
    )
    assert isinstance(restored, MeshingDomainBoundarySource)
    validate_meshing_source_closure(restored)
    assert restored.source_id == source.source_id
    assert restored.source_revision == source.source_revision
    assert restored.source_scope_id == source.source_scope_id
    assert isinstance(restored.domain.patches[0].loops[0][0].pcurve, CircleCurve)
    queries = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.99 * np.cos(np.pi / 4), 0.99 * np.sin(np.pi / 4), 0.5),
            (0.15, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        restored.boundary_distance(queries).upper,
        (0.2, 0.5, 0.05),
        rtol=SOURCE_ROUNDOFF_ATOL,
        atol=SOURCE_ROUNDOFF_ATOL,
    )
    radii = np.linalg.norm(restored.boundary_samples(100).points[:, :2], axis=1)
    assert np.all(radii >= 0.2 - TRIM_RADIUS_ATOL)
    assert np.all(radii <= 1.0 + TRIM_RADIUS_ATOL)
