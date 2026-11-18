#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from dataclasses import replace

import numpy as np

from phydrax._meshcore import meshcore_identity
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._mesh_certificates import _same_tree_content
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.geometry.brep import brep_box
from phydrax.meshing._canonical import _accept_native_construction, _NATIVE_PROVIDER
from phydrax.meshing._contracts import CellFamilyPolicy, CellMeshingTarget, SurfaceMeshingSpec
from phydrax.meshing._scope import MeshingScope
from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_surface import prepare_native_surface


def test_deferred_surface_request_accepts_the_same_mesh() -> None:
    contract = SpatialCoordinateContract.si()
    model = brep_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), coordinate_contract=contract)
    scope = MeshingScope.whole(model, 2)
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(UniformSizeControl(scope, 0.8, strength=SizeControlStrength.SOFT),),
    )
    prepared = prepare_native_surface(
        MeshingDomain.from_brep(model), specification,
        NativeMeshingOptions("parametric_surface"), contract
    )
    certified = prepared.execute(steering="certified")
    screened = prepared.execute(steering="screened")
    assert screened.certified_steps == ("curve_parameters", "pole_sampling", "refinement")
    assert screened.certification.fidelity_witness is None
    deferred = screened.deferred_certification
    assert deferred is not None
    request = deferred.resolve()
    expected = certified.certification
    assert _same_tree_content(
        (
            request.schedule, request.source_id, request.source_revision,
            request.domain, request.cell_regions, request.fidelity_source,
            request.fidelity_tolerance, request.junction_vertices,
            request.scoped_fidelity, request.protected_features,
            request.maximum_normal_deviation, request.fidelity_witness,
            request.feature_source, request.source_coverage_witness,
            request.validity_parent,
        ),
        (
            expected.schedule, expected.source_id, expected.source_revision,
            expected.domain, expected.cell_regions, expected.fidelity_source,
            expected.fidelity_tolerance, expected.junction_vertices,
            expected.scoped_fidelity, expected.protected_features,
            expected.maximum_normal_deviation, expected.fidelity_witness,
            expected.feature_source, expected.source_coverage_witness,
            expected.validity_parent,
        ),
    )
    assert request.source_id == certified.certification.source_id
    assert request.source_revision == certified.certification.source_revision
    assert request.fidelity_tolerance == certified.certification.fidelity_tolerance
    assert request.fidelity_witness is not None
    assert certified.certification.fidelity_witness is not None
    assert request.fidelity_witness.witness_id == certified.certification.fidelity_witness.witness_id
    args = (specification, _NATIVE_PROVIDER, meshcore_identity(), "surface-test", "parametric_surface")
    accepted = _accept_native_construction(certified, *args)
    replayed = _accept_native_construction(
        replace(screened, certification=request, deferred_certification=None), *args
    )
    np.testing.assert_array_equal(accepted.mesh.coordinates, replayed.mesh.coordinates)
    np.testing.assert_array_equal(accepted.mesh.blocks[0].vertices, replayed.mesh.blocks[0].vertices)
    assert tuple(value.association_id for value in accepted.associations) == tuple(
        value.association_id for value in replayed.associations
    )
