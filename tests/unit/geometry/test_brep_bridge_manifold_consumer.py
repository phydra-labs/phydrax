"""The rational bridge feeds a second-geometry-jet manifold basis consumer."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from phydrax.discretization.iga._brep_surface_binding import BRepSplineManifoldBinding
from phydrax.discretization.iga._manifold import ParametricBasisPayload
from phydrax.geometry.brep import brep_surface_bridge
from tests.unit.geometry.test_brep_hermite_bridge_acceptance import (
    _curved_source_models,
    _embedding_args,
)


def test_c2_bridge_seam_transports_covariant_basis_hessian() -> None:
    left_model, right_model = _curved_source_models()
    result = brep_surface_bridge(
        left_model,
        right_model,
        left_model.face_ids[0],
        left_model.edge_ids[2],
        right_model.face_ids[0],
        right_model.edge_ids[0],
        left_side="v1",
        right_side="v0",
        left_along=(Fraction(0), Fraction(1)),
        right_along=(Fraction(0), Fraction(1)),
        left_collar=Fraction(1),
        right_collar=Fraction(1),
        continuity=2,
        **_embedding_args(),
    )
    assert result.status == "certified", result.reason
    assert result.model is not None
    assert result.evidence is not None
    bridge_binding = BRepSplineManifoldBinding(result.model, 2, full_patch_domain=True)
    along = np.array([0.2, 0.4, 0.7, 0.8])
    # A single scalar trial function phi(u,v)=u, authored in the common seam
    # coordinates on both sides. Its value, gradient and Hessian therefore
    # correspond exactly; the covariant Hessian is purely the geometry term.
    payload = ParametricBasisPayload(
        along[:, None],
        np.broadcast_to([[[1.0, 0.0]]], (len(along), 1, 2)),
        np.zeros((len(along), 1, 2, 2)),
        provider_revision="shared-coordinate-u",
    )
    for model, source_v, bridge_v in (
        (left_model, 1.0, 0.0),
        (right_model, 0.0, 1.0),
    ):
        binding = BRepSplineManifoldBinding(model, 0, full_patch_domain=True)
        source_points = np.stack((along, np.full_like(along, source_v)), axis=-1)
        bridge_points = np.stack((along, np.full_like(along, bridge_v)), axis=-1)
        source_result = binding.realize_basis(source_points, payload)
        bridge_result = bridge_binding.realize_basis(bridge_points, payload)
        assert source_result.geometry.source_id == binding.binding_id
        assert bridge_result.geometry.source_id == bridge_binding.binding_id
        np.testing.assert_allclose(
            source_result.geometry.frame.origin,
            bridge_result.geometry.frame.origin,
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            source_result.geometry.mapping_hessian,
            bridge_result.geometry.mapping_hessian,
            rtol=1e-9,
            atol=1e-9,
        )
        assert np.max(np.abs(source_result.covariant_hessians)) > 1e-4
        np.testing.assert_allclose(
            source_result.covariant_hessians,
            bridge_result.covariant_hessians,
            rtol=1e-9,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            source_result.ambient_covariant_hessians,
            bridge_result.ambient_covariant_hessians,
            rtol=1e-9,
            atol=1e-9,
        )
