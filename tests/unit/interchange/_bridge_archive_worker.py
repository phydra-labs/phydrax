from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from phydrax.geometry.brep._bridge_embedding_proof import (
    BridgeEmbeddingEvidence,
    certify_bridge_embedding,
)
from phydrax.geometry.brep._intersection import ConicParameterEndpoint
from phydrax.geometry.brep._model import BRepModel
from phydrax.geometry.brep._patches import AbstractSurfacePatch, BSplineSurfacePatch
from phydrax.geometry.brep._surface_bridge import brep_surface_bridge
from phydrax.interchange._cad_archive import load_brep_archive, save_brep_archive
from phydrax.interchange._cad_archive import _reconstruct_brep_graph
from phydrax.meshing._model_codec import read_model_artifact, write_model_artifact
from tests.unit.geometry.test_brep_hermite_bridge_acceptance import (
    _along_derivative,
    _bridge,
    _curved_source_models,
    _curved_sources,
    _embedding_args,
    _endpoint_jet,
    _rooted_source_models,
    _homogeneous,
)


def _state(patch: AbstractSurfacePatch) -> dict[str, object]:
    if not isinstance(patch, BSplineSurfacePatch):
        raise TypeError("Bridge archive must contain B-spline surface patches.")
    return {
        "degrees": [patch.u_degree, patch.v_degree],
        "exact_controls": repr(patch.exact_control_point_bank),
        "exact_weights": repr(patch.exact_weight_bank),
        "exact_u_knots": repr(patch.exact_u_knot_bank),
        "exact_v_knots": repr(patch.exact_v_knot_bank),
        "controls": np.asarray(patch.control_points).tolist(),
        "weights": np.asarray(patch.weights).tolist(),
        "u_knots": np.asarray(patch.u_knots).tolist(),
        "v_knots": np.asarray(patch.v_knots).tolist(),
        "control_point_ids": patch.control_point_ids,
    }


def _replay(
    left: BSplineSurfacePatch,
    right: BSplineSurfacePatch,
    bridge: BSplineSurfacePatch,
    continuity: int,
) -> None:
    # Full Bernstein coefficient comparison, including mixed along derivatives.
    for source, source_high, bridge_high in ((left, True, False), (right, False, True)):
        for order in range(continuity + 1):
            source_jets = tuple(
                _endpoint_jet(row, order, source_high) for row in _homogeneous(source)
            )
            bridge_jets = tuple(
                _endpoint_jet(row, order, bridge_high) for row in _homogeneous(bridge)
            )
            for derivative in range(3):
                if source_jets != bridge_jets:
                    raise ValueError(
                        f"Cold bridge seam jet mismatch at order {order}, along {derivative}."
                    )
                source_jets = _along_derivative(source_jets)
                bridge_jets = _along_derivative(bridge_jets)
    evidence = certify_bridge_embedding(bridge, left, right, **_embedding_args())
    if not isinstance(evidence, BridgeEmbeddingEvidence):
        raise ValueError(f"Cold bridge embedding replay refused: {evidence}")


def _published_patches(
    model: BRepModel,
) -> tuple[BSplineSurfacePatch, BSplineSurfacePatch, BSplineSurfacePatch]:
    if len(model.patches) != 3:
        raise ValueError("Published bridge must contain exactly three spline patches.")
    left, right, bridge = model.patches
    if not (
        isinstance(left, BSplineSurfacePatch)
        and isinstance(right, BSplineSurfacePatch)
        and isinstance(bridge, BSplineSurfacePatch)
    ):
        raise TypeError("Published bridge must contain exactly three spline patches.")
    return left, right, bridge


def _model_state(model: BRepModel) -> dict[str, object]:
    return {
        "source_revision": model.source_revision,
        "model_id": model.model_id,
        "geometry_id": model.geometry.geometry_id,
        "faces": [repr(value) for value in model.face_ids],
        "edges": [repr(value) for value in model.edge_ids],
        "vertex_roots": [
            None if root is None else root.root_id for root in model.geometry.vertex_roots
        ],
        "edge_endpoints": [
            [
                endpoint.root_id if isinstance(endpoint, ConicParameterEndpoint) else None
                for endpoint in pair
            ]
            for pair in model.geometry.edge_endpoint_roots
        ],
        "coedge_endpoints": [
            [
                endpoint.root_id if isinstance(endpoint, ConicParameterEndpoint) else None
                for endpoint in pair
            ]
            for pair in model.geometry.coedge_endpoint_roots
        ],
        "patches": [_state(patch) for patch in model.patches],
    }


def _rooted_replay(model: BRepModel, source_edge_count: int) -> dict[str, object]:
    geometry = model.geometry
    root = geometry.vertex_roots[3]
    if root is None or len(root.source_edge_lifts) < 4:
        raise ValueError("Published bridge lost the rooted corner and its lifts.")
    bridge = _published_patches(model)[2]
    uses = []
    for coedge in geometry.face_loops[2][0]:
        edge = geometry.coedge_edges[coedge]
        for endpoint, vertex in enumerate(geometry.edge_vertices[edge]):
            if vertex == 3:
                endpoint_root = geometry.coedge_endpoint_roots[coedge][endpoint]
                if not isinstance(endpoint_root, ConicParameterEndpoint):
                    raise TypeError("Bridge corner lost a conic p-curve endpoint.")
                pcurve = geometry.pcurves[coedge]
                if root.source_endpoint_uv_box(bridge, pcurve, endpoint_root) is None:
                    raise ValueError("Bridge corner lift does not replay on its p-curve.")
                uses.append((pcurve, endpoint_root, edge, endpoint))
    if len(uses) != 2 or not root.same_uv_endpoint(
        bridge, uses[0][0], uses[0][1], uses[1][0], uses[1][1]
    ):
        raise ValueError("Bridge corner p-curves do not share their rooted endpoint.")
    cross = [use for use in uses if use[2] >= source_edge_count]
    if len(cross) != 1:
        raise ValueError("Bridge corner must have exactly one cross-edge use.")
    edge_root = geometry.edge_endpoint_roots[cross[0][2]][cross[0][3]]
    if not isinstance(edge_root, ConicParameterEndpoint):
        raise TypeError("Bridge cross-edge lost its conic endpoint.")
    return {
        "vertex_root_id": root.root_id,
        "source_lift_count": len(root.source_edge_lifts),
        "coedge_root_ids": [use[1].root_id for use in uses],
        "cross_edge_root_id": edge_root.root_id,
    }


def main() -> None:
    mode, directory = sys.argv[1], Path(sys.argv[2])
    rooted = mode in ("write-rooted", "read-rooted")
    continuity = 1 if rooted else 2
    if mode in ("write", "write-rooted"):
        left, right = _curved_sources()
        bridge = _bridge(left, right, continuity)
        _replay(left, right, bridge, continuity)
        for name, patch in zip(
            ("left", "right", "bridge"), (left, right, bridge), strict=True
        ):
            write_model_artifact(directory / f"{name}.phx", patch, limits=None)
        models = _rooted_source_models() if rooted else _curved_source_models()
        left_model, right_model = models
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
            continuity=continuity,
            **_embedding_args(),
        )
        if (
            result.model is None
            or result.evidence is None
            or result.status != "certified"
        ):
            raise ValueError(f"Public bridge refused: {result.reason}")
        published = result.model
        published_patches = _published_patches(published)
        if [_state(p) for p in published_patches] != [
            _state(p) for p in (left, right, bridge)
        ]:
            raise ValueError(
                "Published bridge changed the exact input or bridge carriers."
            )
        _replay(
            published_patches[0], published_patches[1], published_patches[2], continuity
        )
        root_state = (
            _rooted_replay(
                published, len(left_model.edge_ids) + len(right_model.edge_ids)
            )
            if rooted
            else None
        )
        for name, model in zip(
            ("left-model", "right-model", "published-model"),
            (*models, published),
            strict=True,
        ):
            save_brep_archive(model, directory / f"{name}.phx")
        print(
            json.dumps(
                {
                    "patches": [_state(p) for p in (left, right, bridge)],
                    "models": [_model_state(m) for m in (*models, published)],
                    "root": root_state,
                },
                sort_keys=True,
            )
        )
    elif mode in ("read", "read-rooted"):
        patches = tuple(
            read_model_artifact(
                directory / f"{name}.phx",
                expected_type=BSplineSurfacePatch,
                restore=_reconstruct_brep_graph,
                limits=None,
            )
            for name in ("left", "right", "bridge")
        )
        expected = json.loads(sys.argv[3])
        if [_state(p) for p in patches] != expected["patches"]:
            raise ValueError("Cold bridge source or coefficient identity changed.")
        models = tuple(
            load_brep_archive(directory / f"{name}.phx")
            for name in ("left-model", "right-model", "published-model")
        )
        if [_model_state(m) for m in models] != expected["models"]:
            raise ValueError("Cold bridge source or published model identity changed.")
        if [_state(m.patches[0]) for m in models[:2]] != expected["patches"][:2]:
            raise ValueError("Cold source models lost their exact spline carriers.")
        published_patches = _published_patches(models[2])
        if [_state(p) for p in published_patches] != expected["patches"]:
            raise ValueError("Cold published model lost its exact spline carriers.")
        _replay(
            published_patches[0], published_patches[1], published_patches[2], continuity
        )
        root_state = (
            _rooted_replay(models[2], len(models[0].edge_ids) + len(models[1].edge_ids))
            if rooted
            else None
        )
        if root_state != expected["root"]:
            raise ValueError("Cold bridge rooted endpoint identities changed.")
        print("bridge-cold-restored")
    else:
        raise ValueError(f"Unknown mode {mode!r}.")


if __name__ == "__main__":
    main()
