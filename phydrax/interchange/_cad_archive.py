#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Lossless native B-Rep persistence through registered model recipes."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple, TYPE_CHECKING

import numpy as np

from .._array_archive import ArrayArchiveLimits
from .._interpolation import (
    ASTSCertificate,
    ASTSConsumerRequirement,
    ASTSRefinement,
    ASTSTransferPlan,
    ExtractedBernstein,
    LocalExtractedBernsteinRealization,
    LocalKnotVector2D,
    TAnchor2D,
    TCell2D,
    TEdge2D,
    TJunctionExtension2D,
    TMesh2D,
    TSplineGeometryCertificate,
    TSplineRationalJet,
    TVertex2D,
)
from .._model._artifacts import artifact_value_id, register_artifact_value
from .._physical import SpatialCoordinateContract
from ..discretization._exact_algebraic_geometry import RadicalElement
from ..geometry._atlas import TrimDomain
from ..geometry.brep._assembly import (
    BRepAffinePlacement,
    BRepAssemblyDefinition,
    BRepAssemblyGraph,
    BRepAssemblyOccurrence,
    BRepAssemblyUse,
    BRepPlacementEvidence,
)
from ..geometry.brep._model import (
    BRepAssemblyContainer,
    BRepBoundaryMap,
    BRepEntityId,
    BRepGeometry,
    BRepImportReport,
    BRepModel,
    BRepQualifiedIncidence,
    BRepTessellation,
    BRepTopology,
)
from ..geometry.brep._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    ConePatch,
    ConicParameterRoot,
    ConicRootLinePCurve,
    CylinderPatch,
    EllipseCurve,
    ExactLineCurve,
    ExtrusionSurface,
    HyperbolaCurve,
    LineCurve,
    OffsetCurve,
    OffsetSurface,
    ParabolaCurve,
    PlaneCylinderSectionPCurve,
    PlanePatch,
    RevolutionSurface,
    RuledSurface,
    SpherePatch,
    SurfaceIsoparametricCurve,
    SurfaceMappedCurve,
    TorusPatch,
)
from ..geometry.brep._tspline import TSplineSurfacePatch
from ..meshing._model_codec import read_model_artifact, write_model_artifact
from ..units._dimension import DimensionSignature
from ..units._unit import UnitDefinition


if TYPE_CHECKING:
    from ..geometry.brep._intersection_curve import SurfaceRegion


class ConicRootLineEndpointWitnessArchiveRecipe(NamedTuple):
    """Authoritative sources needed to replay one exact conic trim endpoint."""

    source_curve: AbstractCurve
    source_pcurve: ConicRootLinePCurve
    source_region: SurfaceRegion
    parameter_exact: Fraction
    source_interval: tuple[Fraction, Fraction]
    trim_curve: AbstractCurve
    trim_interval: tuple[Fraction, Fraction]


_BREP_TYPES = (
    BRepAffinePlacement,
    BRepPlacementEvidence,
    BRepAssemblyDefinition,
    BRepAssemblyGraph,
    BRepAssemblyOccurrence,
    BRepAssemblyUse,
    BRepBoundaryMap,
    BRepEntityId,
    BRepGeometry,
    BRepImportReport,
    BRepModel,
    BRepTessellation,
    BRepTopology,
    AbstractCurve,
    AbstractSurfacePatch,
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    ConePatch,
    ConicParameterRoot,
    ConicRootLinePCurve,
    CylinderPatch,
    EllipseCurve,
    ExtrusionSurface,
    HyperbolaCurve,
    LineCurve,
    ParabolaCurve,
    PlanePatch,
    PlaneCylinderSectionPCurve,
    RevolutionSurface,
    OffsetCurve,
    RuledSurface,
    OffsetSurface,
    ExactLineCurve,
    SpherePatch,
    SurfaceIsoparametricCurve,
    SurfaceMappedCurve,
    TorusPatch,
    TSplineSurfacePatch,
)
_INTERPOLATION_TYPES = (
    ASTSCertificate,
    ASTSRefinement,
    ASTSTransferPlan,
    ASTSConsumerRequirement,
    ExtractedBernstein,
    LocalExtractedBernsteinRealization,
    LocalKnotVector2D,
    TAnchor2D,
    TCell2D,
    TEdge2D,
    TMesh2D,
    TSplineGeometryCertificate,
    TSplineRationalJet,
    TVertex2D,
    TJunctionExtension2D,
)
_TSPLINE_RECONSTRUCTION_FIELDS = (
    "mesh",
    "source_mesh",
    "transfer_plans",
    "transfer_meshes",
    "control_points",
    "weights",
    "exact_source_control_points",
    "exact_source_weights",
)


def _validate_tspline_patch(patch: TSplineSurfacePatch, /) -> None:
    patch.validate_source_bank()
    stored_source = patch.source_basis_certificate
    stored_target = patch.basis_certificate
    source_certificate = patch.source_mesh.certify(tolerance=stored_source.tolerance)
    target_certificate = patch.mesh.certify(tolerance=stored_target.tolerance)
    transfer_certificates = tuple(
        mesh.certify(tolerance=certificate.tolerance)
        for mesh, certificate in zip(
            patch.transfer_meshes, patch.transfer_basis_certificates, strict=True
        )
    )
    if (
        not source_certificate.passed
        or not target_certificate.passed
        or source_certificate.certificate_id != stored_source.certificate_id
        or target_certificate.certificate_id != stored_target.certificate_id
        or any(
            not recomputed.passed or recomputed.certificate_id != stored.certificate_id
            for recomputed, stored in zip(
                transfer_certificates, patch.transfer_basis_certificates, strict=True
            )
        )
    ):
        raise ValueError(
            "T-spline basis certificates do not match recomputed source meshes."
        )
    patch.validate_exact_transfer_chain()


def save_brep_archive(
    model: BRepModel,
    path: str | Path,
    /,
    *,
    ingress_limits: ArrayArchiveLimits | None = None,
) -> Path:
    """Write the complete registered B-Rep source, identity, and realization state."""
    if type(model) is not BRepModel:
        raise TypeError("model must be the canonical BRepModel type.")
    for patch in model.patches:
        if isinstance(patch, TSplineSurfacePatch):
            _validate_tspline_patch(patch)
    _register_brep_types()
    return write_model_artifact(path, model, limits=ingress_limits)


def load_brep_archive(
    path: str | Path,
    /,
    *,
    ingress_limits: ArrayArchiveLimits | None = None,
) -> BRepModel:
    """Cold-restore a B-Rep by re-running every registered native constructor."""
    _register_brep_types()
    return read_model_artifact(
        path,
        expected_type=BRepModel,
        restore=_reconstruct_brep_graph,
        limits=ingress_limits,
    )


def _register_type(namespace: str, value: type, /) -> None:
    try:
        artifact_value_id(value)
    except TypeError:
        register_artifact_value(f"{namespace}:{value.__name__}", value)


def _register_brep_types() -> None:
    for value in _BREP_TYPES:
        _register_type("phydrax.geometry.brep", value)
    for value in _INTERPOLATION_TYPES:
        _register_type("phydrax.discretization.iga", value)
    for namespace, value in (
        ("phydrax.geometry", TrimDomain),
        ("phydrax", SpatialCoordinateContract),
        ("phydrax.units", UnitDefinition),
        ("phydrax.units", DimensionSignature),
        ("phydrax.discretization", RadicalElement),
    ):
        _register_type(namespace, value)
    from ..geometry.brep._boolean_core import AffinePCurve, PeriodicPCurve
    from ..geometry.brep._intersection import (
        BranchRootEndpoint,
        ConicParameterEndpoint,
        ConicRootLineEndpointWitness,
        CurveSurfaceIntersectionRoot,
        IntersectionCurvePointRoot,
        NativePeriodEndpoint,
        TrimIntersectionRoot,
        TrimRootEndpoint,
        TripleSurfaceIntersectionRoot,
    )
    from ..geometry.brep._intersection_curve import (
        CurveTrimSegment,
        IntersectionCurve,
        IntersectionPCurve,
        SurfaceRegion,
    )
    from ..geometry.brep._model import (
        BRepAssemblyContainer,
    )
    from ..geometry.brep._placed import PlacedCurve, PlacedSurface
    from ..geometry.brep._root_bindings import (
        BRepBranchNodeRoots,
        BRepConicVertex,
        BRepCurveSurfaceLift,
        BRepPlacedVertex,
        BRepRootSupport,
        BRepVertexRoot,
    )

    for value in (
        BranchRootEndpoint,
        CurveSurfaceIntersectionRoot,
        IntersectionCurve,
        IntersectionCurvePointRoot,
        IntersectionPCurve,
        CurveTrimSegment,
        NativePeriodEndpoint,
        PlacedCurve,
        PlacedSurface,
        SurfaceRegion,
        TripleSurfaceIntersectionRoot,
        TrimIntersectionRoot,
        TrimRootEndpoint,
        ConicParameterEndpoint,
        ConicRootLineEndpointWitness,
        BRepCurveSurfaceLift,
        BRepPlacedVertex,
        BRepRootSupport,
        BRepVertexRoot,
        BRepBranchNodeRoots,
        BRepConicVertex,
        BRepAssemblyContainer,
        AffinePCurve,
        PeriodicPCurve,
    ):
        _register_type("phydrax.geometry.brep", value)
    _register_type("phydrax.interchange", ConicRootLineEndpointWitnessArchiveRecipe)


def _reconstruct_brep_graph(root: Any, /) -> Any:
    pending: list[tuple[Any, bool]] = [(root, False)]
    state: dict[int, bool] = {}
    postorder: list[Any] = []
    while pending:
        value, expanded = pending.pop()
        identity = id(value)
        if expanded:
            state[identity] = True
            postorder.append(value)
            continue
        if identity in state:
            if not state[identity]:
                raise ValueError("B-Rep source graph contains a cycle.")
            continue
        state[identity] = False
        pending.append((value, True))
        for child in reversed(_children(value)):
            child_identity = id(child)
            if child_identity in state:
                if not state[child_identity]:
                    raise ValueError("B-Rep source graph contains a cycle.")
                continue
            pending.append((child, False))

    restored: dict[int, Any] = {}
    for value in postorder:
        identity = id(value)
        if type(value) is TSplineSurfacePatch:
            fields = {
                name: restored[id(object.__getattribute__(value, name))]
                for name in _TSPLINE_RECONSTRUCTION_FIELDS
            }
            fields["source_basis_certificate"] = object.__getattribute__(
                value, "source_basis_certificate"
            )
            realization = object.__getattribute__(value, "realization")
            fields["basis_certificate"] = object.__getattribute__(
                realization, "basis_certificate"
            )
            fields["transfer_basis_certificates"] = object.__getattribute__(
                value, "transfer_basis_certificates"
            )
            result = _construct_dataclass(type(value), fields)
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            fields = {
                field.name: restored[id(object.__getattribute__(value, field.name))]
                for field in dataclasses.fields(value)
            }
            result = _construct_dataclass(type(value), fields)
        elif isinstance(value, tuple) and hasattr(type(value), "_fields"):
            if type(value) is ConicRootLineEndpointWitnessArchiveRecipe:
                result = _construct_dataclass(
                    type(value),
                    {
                        name: restored[id(object.__getattribute__(value, name))]
                        for name in type(value)._fields
                    },
                )
            else:
                result = type(value)(*(restored[id(child)] for child in value))
        elif isinstance(value, tuple):
            result = tuple(restored[id(child)] for child in value)
        elif isinstance(value, list):
            result = [restored[id(child)] for child in value]
        elif isinstance(value, set):
            result = {restored[id(child)] for child in value}
        elif isinstance(value, frozenset):
            result = frozenset(restored[id(child)] for child in value)
        elif isinstance(value, Mapping):
            entries = [
                (restored[id(key)], restored[id(item)]) for key, item in value.items()
            ]
            if type(value).__name__ == "frozendict":
                from .._frozendict import frozendict

                result = frozendict(entries)
            elif type(value).__name__ == "mappingproxy":
                result = MappingProxyType(dict(entries))
            else:
                result = dict(entries)
        else:
            result = value
        restored[identity] = result
    _validate_brep_entity_owners(tuple(restored.values()))
    return restored[id(root)]


def _validate_brep_entity_owners(values: tuple[Any, ...], /) -> None:
    models = {
        (value.source_id, value.source_revision): value
        for value in values
        if isinstance(value, BRepModel)
    }
    qualified_entities_by_owner: dict[tuple[str, str], set[BRepEntityId]] = {}
    for entity in (value for value in values if isinstance(value, BRepEntityId)):
        owner_key = (entity.source_id, entity.source_revision)
        owner = models.get(owner_key)
        if owner is None:
            raise ValueError("BRep entity identity has no retained owning source model.")
        topology = owner.geometry.topology
        entity_count = {
            "solid": topology.num_solids,
            "face": topology.num_faces,
            "edge": topology.num_edges,
            "vertex": topology.num_vertices,
        }[entity.kind]
        if entity.index >= entity_count:
            raise ValueError("BRep entity index is outside its owning source topology.")
        if entity.occurrence_path:
            qualified_entities = qualified_entities_by_owner.get(owner_key)
            if qualified_entities is None:
                incidence = owner.geometry.qualified_entity_incidence(*owner_key)
                qualified_entities = {
                    entity_id
                    for relation in incidence
                    for entity_id in (relation.container, relation.member)
                }
                qualified_entities_by_owner[owner_key] = qualified_entities
            if entity not in qualified_entities:
                raise ValueError(
                    "BRep entity is absent from its retained occurrence-qualified source topology."
                )


def _children(value: Any, /) -> tuple[Any, ...]:
    if type(value) is TSplineSurfacePatch:
        return tuple(
            object.__getattribute__(value, name)
            for name in _TSPLINE_RECONSTRUCTION_FIELDS
        )
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return tuple(
            object.__getattribute__(value, field.name)
            for field in dataclasses.fields(value)
        )
    if isinstance(value, tuple) and hasattr(type(value), "_fields"):
        return tuple(value)
    if isinstance(value, (tuple, list, set, frozenset)):
        return tuple(value)
    if isinstance(value, Mapping):
        return tuple(child for pair in value.items() for child in pair)
    return ()


def _construct_intersection_dataclass(
    model_type: type, fields: Mapping[str, Any], /
) -> Any:
    from ..geometry.brep._intersection import (
        BranchRootEndpoint,
        ConicParameterEndpoint,
        ConicRootLineEndpointWitness,
        CurveSurfaceIntersectionRoot,
        IntersectionCurvePointRoot,
        NativePeriodEndpoint,
        TrimIntersectionRoot,
        TrimRootEndpoint,
        TripleSurfaceIntersectionRoot,
    )
    from ..geometry.brep._intersection_curve import (
        CurveTrimSegment,
        IntersectionCurve,
        IntersectionPCurve,
        SurfaceRegion,
    )
    from ..geometry.brep._root_bindings import (
        BRepBranchNodeRoots,
        BRepConicVertex,
        BRepCurveSurfaceLift,
        BRepPlacedVertex,
        BRepRootSupport,
        BRepVertexRoot,
    )

    if model_type is SurfaceRegion:
        return SurfaceRegion(fields["patch"], fields["parameter_box"])
    if model_type is IntersectionCurve:
        return IntersectionCurve(
            fields["first"],
            fields["second"],
            chart_axes=fields["chart_axes"],
            chart_start=fields["chart_start"],
            chart_end=fields["chart_end"],
            chart_lower=fields["chart_lower"],
            chart_upper=fields["chart_upper"],
            node_axes=fields["node_axes"],
            node_values=fields["node_values"],
            node_lower=fields["node_lower"],
            node_upper=fields["node_upper"],
            node_period_shifts=fields["node_period_shifts"],
            closed=fields["closed"],
            start_kind=fields["start_kind"],
            end_kind=fields["end_kind"],
        )
    if model_type is IntersectionPCurve:
        return IntersectionPCurve(
            fields["curve"],
            fields["side"],
            first=fields["first"],
            last=fields["last"],
            reversed=fields["reversed"],
        )
    if model_type is CurveTrimSegment:
        return CurveTrimSegment(
            fields["curve"],
            fields["first"],
            fields["last"],
            reversed=fields["reversed"],
            first_root=fields["first_root"],
            last_root=fields["last_root"],
        )
    if model_type is TrimIntersectionRoot:
        expected_fields = {
            "first",
            "second",
            "parameter_lower",
            "parameter_upper",
            "parameters",
            "enclosure_lower",
            "enclosure_upper",
            "preconditioner",
            "endpoint_witness",
            "contraction",
            "root_id",
        }
        if set(fields) != expected_fields:
            raise ValueError("Stored trim root field set is not canonical.")
        endpoint_witness = fields["endpoint_witness"]
        if (
            endpoint_witness is not None
            and type(endpoint_witness) is not ConicRootLineEndpointWitness
        ):
            raise TypeError("Stored trim root lost its rebuilt conic endpoint witness.")
        root = TrimIntersectionRoot(
            fields["first"],
            fields["second"],
            parameter_lower=fields["parameter_lower"],
            parameter_upper=fields["parameter_upper"],
            endpoint_witness=endpoint_witness,
        )
        if (
            root.root_id != fields["root_id"]
            or not np.array_equal(
                np.asarray(root.parameters), np.asarray(fields["parameters"])
            )
            or not np.array_equal(
                np.asarray(root.enclosure_lower), np.asarray(fields["enclosure_lower"])
            )
            or not np.array_equal(
                np.asarray(root.enclosure_upper), np.asarray(fields["enclosure_upper"])
            )
            or not np.array_equal(
                np.asarray(root.preconditioner), np.asarray(fields["preconditioner"])
            )
            or root.contraction != fields["contraction"]
        ):
            raise ValueError(
                "Stored trim root identity or Krawczyk evidence differs from replay."
            )
        return root
    if model_type is CurveSurfaceIntersectionRoot:
        expected_fields = {
            "curve",
            "surface",
            "parameter_lower",
            "parameter_upper",
            "parameters",
            "preconditioner",
            "endpoint_witness",
            "contraction",
            "root_id",
        }
        if set(fields) != expected_fields:
            raise ValueError("Stored curve-surface root field set is not canonical.")
        endpoint_witness = fields["endpoint_witness"]
        if (
            endpoint_witness is not None
            and type(endpoint_witness) is not ConicRootLineEndpointWitness
        ):
            raise TypeError(
                "Stored curve-surface root lost its rebuilt endpoint witness."
            )
        root = CurveSurfaceIntersectionRoot(
            fields["curve"],
            fields["surface"],
            parameter_lower=fields["parameter_lower"],
            parameter_upper=fields["parameter_upper"],
            endpoint_witness=endpoint_witness,
        )
        if (
            root.root_id != fields["root_id"]
            or not np.array_equal(
                np.asarray(root.parameters), np.asarray(fields["parameters"])
            )
            or not np.array_equal(
                np.asarray(root.preconditioner), np.asarray(fields["preconditioner"])
            )
            or root.contraction != fields["contraction"]
        ):
            raise ValueError(
                "Stored curve-surface root identity or Krawczyk evidence differs from replay."
            )
        return root
    if model_type is IntersectionCurvePointRoot:
        return IntersectionCurvePointRoot(fields["curve"], fields["parameter"])
    if model_type is TripleSurfaceIntersectionRoot:
        return TripleSurfaceIntersectionRoot(
            fields["first"],
            fields["second"],
            fields["third"],
            parameter_lower=fields["parameter_lower"],
            parameter_upper=fields["parameter_upper"],
        )
    if model_type is TrimRootEndpoint:
        return TrimRootEndpoint(
            fields["root"],
            fields["operand"],
            affine_parameter_offset=fields["affine_parameter_offset"],
            affine_parameter_scale=fields["affine_parameter_scale"],
            affine_transforms=fields["affine_transforms"],
            source_pcurve=fields["source_pcurve"],
            source_surface=fields["source_surface"],
        )
    if model_type is BranchRootEndpoint:
        return BranchRootEndpoint(
            fields["root"],
            fields["curve"],
            fields["chart"],
            source_pcurve=fields["source_pcurve"],
            source_side=fields["source_side"],
            source_first=fields["source_first"],
            source_last=fields["source_last"],
            periodic_shifts=fields["periodic_shifts"],
            affine_transforms=fields["affine_transforms"],
        )
    if model_type is NativePeriodEndpoint:
        return NativePeriodEndpoint(
            fields["curve"],
            fields["patch"],
            fields["axis"],
            rational=fields["rational"],
            turns=fields["turns"],
        )

    if model_type is ConicParameterEndpoint:
        return ConicParameterEndpoint(
            fields["root"],
            fields["carrier"],
            fields["parameter_exact"],
            fields["point_exact"],
        )
    if model_type is BRepRootSupport:
        return BRepRootSupport(fields["patch"], fields["root"])
    if model_type is BRepCurveSurfaceLift:
        expected_fields = {
            "curve",
            "surface",
            "pcurve",
            "first",
            "last",
            "parameter_box",
            "root_id",
        }
        if set(fields) != expected_fields:
            raise ValueError(
                "Stored B-Rep curve-surface lift field set is not canonical."
            )
        lift = BRepCurveSurfaceLift(
            fields["curve"],
            fields["surface"],
            fields["pcurve"],
            fields["first"],
            fields["last"],
            source_region=SurfaceRegion(fields["surface"], fields["parameter_box"]),
        )
        if lift.root_id != fields["root_id"]:
            raise ValueError(
                "Stored B-Rep curve-surface lift differs from its source chart."
            )
        return lift
    if model_type is BRepConicVertex:
        return BRepConicVertex(fields["root"], offset_exact=fields["offset_exact"])

    if model_type is BRepBranchNodeRoots:
        branch_node = BRepBranchNodeRoots(fields["node"], fields["endpoints"])
        if branch_node.root_id != fields["root_id"]:
            raise ValueError("Stored B-Rep branch-node roots differ from their proof.")
        return branch_node
    if model_type is BRepVertexRoot:
        return BRepVertexRoot(
            fields["primary"],
            aliases=fields["aliases"],
            spatial_root=fields["spatial_root"],
            joint_root=fields["joint_root"],
            source_edge_lifts=fields["source_edge_lifts"],
            branch_node=fields["branch_node"],
        )
    if model_type is BRepPlacedVertex:
        return BRepPlacedVertex(
            fields["source_point"],
            fields["rotation"],
            fields["translation"],
            fields["source_entity_id"],
            source_root=fields["source_root"],
            affine=fields["affine"],
        )
    raise TypeError(
        f"No explicit intersection constructor for {model_type.__qualname__}."
    )


def _construct_dataclass(model_type: type, fields: Mapping[str, Any], /) -> Any:
    if model_type is DimensionSignature:
        return DimensionSignature.from_dict(
            {
                "terms": [
                    {
                        "axis": axis,
                        "numerator": numerator,
                        "denominator": denominator,
                    }
                    for axis, numerator, denominator in fields["terms"]
                ],
                "dimension_id": fields["dimension_id"],
            }
        )
    if model_type is UnitDefinition:
        if not isinstance(fields["dimension"], DimensionSignature):
            raise TypeError("Stored unit lost its exact dimension signature.")
        return UnitDefinition.from_dict(
            {
                "symbol": fields["symbol"],
                "dimension": fields["dimension"].to_dict(),
                "reference_system_id": fields["reference_system_id"],
                "scale_numerator": fields["scale_numerator"],
                "scale_denominator": fields["scale_denominator"],
                "unit_id": fields["unit_id"],
            }
        )
    if model_type is SpatialCoordinateContract:
        if not isinstance(fields["length_unit"], UnitDefinition):
            raise TypeError("Stored spatial contract lost its exact length unit.")
        return SpatialCoordinateContract.from_dict(
            {
                "length_unit": fields["length_unit"].to_dict(),
                "length_coordinate_kind": fields["length_coordinate_kind"],
                "coordinate_system": fields["coordinate_system"],
                "reference_frame": fields["reference_frame"],
                "spatial_id": fields["spatial_id"],
            }
        )
    if model_type is ConicRootLineEndpointWitnessArchiveRecipe:
        expected_fields = {
            "source_curve",
            "source_pcurve",
            "source_region",
            "parameter_exact",
            "source_interval",
            "trim_curve",
            "trim_interval",
        }
        if set(fields) != expected_fields:
            raise ValueError(
                "Stored conic endpoint witness source fields are not canonical."
            )
        from ..geometry.brep._intersection import ConicRootLineEndpointWitness

        return ConicRootLineEndpointWitness(
            fields["source_curve"],
            fields["source_pcurve"],
            fields["source_region"],
            fields["parameter_exact"],
            fields["source_interval"],
            fields["trim_curve"],
            fields["trim_interval"],
        )
    if model_type.__module__ == "phydrax.discretization._surface_chart_records":
        from ..discretization._surface_chart_records import (
            SurfaceChartWitness,
            SurfaceRootMeshCellBinding,
        )

        if model_type is not SurfaceChartWitness:
            raise TypeError(
                f"No explicit surface-chart constructor for {model_type.__qualname__}."
            )
        root_mesh_cell_bindings = fields["root_mesh_cell_bindings"]
        if not isinstance(root_mesh_cell_bindings, tuple) or any(
            binding is not None and type(binding) is not SurfaceRootMeshCellBinding
            for binding in root_mesh_cell_bindings
        ):
            raise TypeError("Stored surface chart has malformed root-mesh bindings.")
        witness = SurfaceChartWitness(
            geometry_id=fields["geometry_id"],
            topology_id=fields["topology_id"],
            domain_id=fields["domain_id"],
            cell_global_ids=fields["cell_global_ids"],
            patches=fields["patches"],
            charts=fields["charts"],
            source_root_cell_ids=fields["source_root_cell_ids"],
            source_piece_ids=fields["source_piece_ids"],
            geometry_entity_ids=fields["geometry_entity_ids"],
            occurrence_paths=fields["occurrence_paths"],
            chart_rational_tokens=fields["chart_rational_tokens"],
            root_mesh_cell_bindings=root_mesh_cell_bindings,
        )
        if witness.witness_id != fields["witness_id"]:
            raise ValueError(
                "Stored source chart witness differs from its exact chart bank."
            )
        return witness
    if model_type.__module__ in (
        "phydrax.geometry.brep._intersection",
        "phydrax.geometry.brep._intersection_curve",
        "phydrax.geometry.brep._root_bindings",
    ):
        return _construct_intersection_dataclass(model_type, fields)
    if model_type.__module__ == "phydrax.geometry.brep._placed":
        from ..geometry.brep._placed import PlacedCurve, PlacedSurface

        if model_type is PlacedSurface:
            return PlacedSurface(
                fields["definition"],
                fields["rotation"],
                fields["translation"],
                affine=fields["affine"],
            )
        if model_type is PlacedCurve:
            return PlacedCurve(
                fields["definition"],
                fields["rotation"],
                fields["translation"],
                affine=fields["affine"],
            )
        raise TypeError(f"No explicit placed constructor for {model_type.__qualname__}.")
    if model_type is TVertex2D:
        return TVertex2D(fields["vertex_id"], fields["parameter"])
    if model_type is TEdge2D:
        return TEdge2D(
            fields["edge_id"],
            fields["vertex_ids"],
            fields["cell_ids"],
            fields["axis"],
        )
    if model_type is TCell2D:
        return TCell2D(
            fields["cell_id"],
            fields["parameter_bounds"],
            vertex_ids=fields["vertex_ids"],
            edge_ids=fields["edge_ids"],
            level=fields["level"],
            parent_id=fields["parent_id"],
            refinement_address=fields["refinement_address"],
        )
    if model_type is LocalKnotVector2D:
        return LocalKnotVector2D(fields["u"], fields["v"])
    if model_type is TAnchor2D:
        return TAnchor2D(fields["anchor_id"], fields["parameter"], fields["local_knots"])
    if model_type is TJunctionExtension2D:
        return TJunctionExtension2D(
            fields["junction_vertex_id"],
            fields["axis"],
            fields["missing_direction"],
            fields["segment"],
            backward_crossings=fields["backward_crossings"],
            forward_crossings=fields["forward_crossings"],
        )
    if model_type is ASTSTransferPlan:
        plan = ASTSTransferPlan(
            fields["source_mesh_id"],
            fields["target_mesh_id"],
            fields["source_count"],
            fields["target_count"],
            fields["target_indices"],
            fields["source_indices"],
            fields["exact_coefficients"],
        )
        if plan.plan_id != fields["plan_id"]:
            raise ValueError(
                "Stored ASTS transfer identity differs from its exact coefficients."
            )
        return plan
    if model_type is ASTSConsumerRequirement:
        return ASTSConsumerRequirement(
            fields["consumer"],
            fields["intrinsic_dimension"],
            fields["ambient_dimensions"],
            fields["required_parameter_derivative_order"],
            fields["requires_positive_rational_weights"],
            fields["requires_regular_geometry_map"],
            fields["requires_consistent_orientation"],
            fields["requires_cross_cell_c1"],
            fields["basis_profile"],
        )
    if model_type is TMesh2D:
        mesh = TMesh2D(
            fields["cells"],
            fields["anchors"],
            degree=fields["degree"],
            cell_lineage=fields["cell_lineage"],
            reserved_cell_ids=fields["reserved_cell_ids"],
            reserved_anchor_ids=fields["reserved_anchor_ids"],
        )
        if mesh.mesh_id != fields["mesh_id"]:
            raise ValueError(
                "Stored T-mesh identity differs from its exact source fields."
            )
        return mesh
    if model_type is ASTSRefinement:
        stored_certificate = fields["certificate"]
        certificate = fields["refined_mesh"].certify(
            tolerance=stored_certificate.tolerance
        )
        if (
            not certificate.passed
            or certificate.certificate_id != stored_certificate.certificate_id
        ):
            raise ValueError(
                "Stored T-mesh refinement certificate differs from recomputation."
            )
        return ASTSRefinement(
            fields["source_mesh"],
            fields["refined_mesh"],
            fields["marked_cell_ids"],
            fields["closure_cell_ids"],
            fields["cell_lineage"],
            fields["transfer_plan"],
            certificate,
        )
    if model_type is BRepEntityId:
        return BRepEntityId(
            fields["source_id"],
            fields["source_revision"],
            fields["kind"],
            fields["index"],
            fields["occurrence_path"],
        )
    if model_type is BRepAffinePlacement:
        return BRepAffinePlacement(fields["matrix"])
    if model_type is BRepPlacementEvidence:
        expected_fields = {
            "rotation_deviation_bound",
            "model_extent",
            "point_deviation_bound",
            "translation_deviation_bound",
            "entrywise_deviation_bounds",
        }
        if set(fields) != expected_fields:
            raise ValueError("Stored BRep placement evidence field set is not canonical.")
        return BRepPlacementEvidence(
            fields["rotation_deviation_bound"],
            fields["model_extent"],
            fields["point_deviation_bound"],
            translation_deviation_bound=fields["translation_deviation_bound"],
            entrywise_deviation_bounds=fields["entrywise_deviation_bounds"],
        )
    if model_type is BRepAssemblyUse:
        return BRepAssemblyUse(
            fields["use_id"],
            fields["placement"],
            fields["solid_index"],
            fields["definition_id"],
            placement_evidence=fields["placement_evidence"],
        )
    if model_type is BRepAssemblyDefinition:
        return BRepAssemblyDefinition(fields["definition_id"], fields["uses"])
    if model_type is BRepAssemblyOccurrence:
        return BRepAssemblyOccurrence(
            fields["path"],
            fields["solid_index"],
            fields["placement"],
            placement_evidence=fields["placement_evidence"],
        )
    if model_type is BRepAssemblyGraph:
        return BRepAssemblyGraph(
            fields["roots"],
            fields["definitions"],
            fields["solid_count"],
            occurrence_evidence=fields["occurrence_evidence"],
        )
    if model_type is BRepAssemblyContainer:
        return BRepAssemblyContainer(
            fields["path"], fields["member_paths"], fields["child_paths"]
        )
    if model_type is BRepQualifiedIncidence:
        return BRepQualifiedIncidence(
            fields["container"],
            fields["member"],
            fields["orientation"],
            fields["use_index"],
        )
    if model_type is BRepImportReport:
        report = BRepImportReport(
            source_id=fields["source_id"],
            source_digest=fields["source_digest"],
            source_format=fields["source_format"],
            coordinate_contract=fields["coordinate_contract"],
            import_policy_id=fields["import_policy_id"],
            num_solids=fields["num_solids"],
            num_faces=fields["num_faces"],
            num_edges=fields["num_edges"],
            num_vertices=fields["num_vertices"],
            converted_surface_count=fields["converted_surface_count"],
            curve_surface_tolerance=fields["curve_surface_tolerance"],
            curve_surface_scale=fields["curve_surface_scale"],
        )
        if report.origin_revision != fields["origin_revision"]:
            raise ValueError("Stored B-Rep import origin differs from its source fields.")
        return report
    if model_type is BRepTopology:
        topology = BRepTopology(
            face_edges=fields["face_edges"],
            edge_faces=fields["edge_faces"],
            face_wires=fields["face_wires"],
            solid_faces=fields["solid_faces"],
            solid_face_orientations=fields["solid_face_orientations"],
            num_vertices=fields["num_vertices"],
        )
        if topology.face_solids != fields["face_solids"]:
            raise ValueError(
                "Stored B-Rep face-to-solid incidence differs from recomputation."
            )
        return topology
    if model_type is BRepTessellation:
        tessellation = BRepTessellation(
            linear_deflection=fields["linear_deflection"],
            angular_deflection=fields["angular_deflection"],
            source_revision=fields["source_revision"],
            policy_id=fields["policy_id"],
            vertices=fields["vertices"],
            triangles=fields["triangles"],
            face_ids=fields["face_ids"],
            parameters=fields["parameters"],
            deviation_bounds=fields["deviation_bounds"],
            normal_bounds=fields["normal_bounds"],
            vertex_source_dimensions=fields["vertex_source_dimensions"],
            vertex_source_indices=fields["vertex_source_indices"],
            vertex_parameters=fields["vertex_parameters"],
            triangle_occurrence_ids=fields["triangle_occurrence_ids"],
            vertex_occurrence_ids=fields["vertex_occurrence_ids"],
            vertex_global_ids=fields["vertex_global_ids"],
            triangle_global_ids=fields["triangle_global_ids"],
            source_chart_witness=fields["source_chart_witness"],
        )
        if tessellation.tessellation_id != fields["tessellation_id"]:
            raise ValueError(
                "Stored B-Rep tessellation identity differs from its source arrays."
            )
        return tessellation
    if model_type is BRepBoundaryMap:
        return BRepBoundaryMap(fields["patches"], fields["parameter_bounds"])
    if model_type is BRepModel:
        model = BRepModel(
            geometry=fields["geometry"],
            patches=fields["patches"],
            parameter_bounds=fields["parameter_bounds"],
            orientation=fields["orientation"],
            physical_tags=fields["physical_tags"],
            coordinate_contract=fields["coordinate_contract"],
            report=fields["report"],
            trim_domains=fields["trim_domains"],
            tessellation=fields["tessellation"],
        )
        if (
            model.source_revision != fields["source_revision"]
            or model.model_id != fields["model_id"]
        ):
            raise ValueError(
                "Stored B-Rep model identity differs from its source geometry."
            )
        return model
    if model_type is BRepGeometry:
        geometry_fields = {
            name: fields[name]
            for name in (
                "vertex_points",
                "curves",
                "edge_curves",
                "edge_ranges",
                "edge_vertices",
                "pcurves",
                "coedge_edges",
                "coedge_senses",
                "face_loops",
                "shell_faces",
                "shell_orientations",
                "solid_shells",
            )
        }
        for name in (
            "vertex_exact_points",
            "vertex_roots",
            "edge_endpoint_roots",
            "coedge_endpoint_roots",
            "edge_vertex_gap_bounds",
            "edge_face_gap_bounds",
            "assembly_graph",
            "occurrences",
            "assembly_containers",
        ):
            if name in fields:
                geometry_fields[name] = fields[name]
        return BRepGeometry(**geometry_fields)
    if model_type is LineCurve:
        return LineCurve(
            fields["origin"],
            fields["direction"],
            origin_exact=fields["origin_exact"],
            direction_exact=fields["direction_exact"],
        )
    if model_type is CircleCurve:
        return CircleCurve(
            fields["center"],
            fields["first_axis"],
            fields["second_axis"],
            fields["radius"],
            center_exact=fields["center_exact"],
            first_axis_exact=fields["first_axis_exact"],
            second_axis_exact=fields["second_axis_exact"],
            radius_exact=fields["radius_exact"],
        )
    if model_type is EllipseCurve:
        return EllipseCurve(
            fields["center"],
            fields["first_axis"],
            fields["second_axis"],
            fields["first_radius"],
            fields["second_radius"],
            center_exact=fields["center_exact"],
            first_axis_exact=fields["first_axis_exact"],
            second_axis_exact=fields["second_axis_exact"],
            first_radius_exact=fields["first_radius_exact"],
            second_radius_exact=fields["second_radius_exact"],
        )
    if model_type is ParabolaCurve:
        return ParabolaCurve(
            fields["vertex"],
            fields["first_axis"],
            fields["second_axis"],
            fields["focal_length"],
        )
    if model_type is HyperbolaCurve:
        return HyperbolaCurve(
            fields["center"],
            fields["first_axis"],
            fields["second_axis"],
            fields["first_radius"],
            fields["second_radius"],
        )
    if model_type is OffsetCurve:
        curve = OffsetCurve(
            fields["base"],
            fields["distance_exact"],
            fields["direction"],
        )
        if not np.array_equal(np.asarray(curve.distance), np.asarray(fields["distance"])):
            raise ValueError(
                "Stored offset distance differs from its exact source coefficient."
            )
        return curve
    if model_type is ExtrusionSurface:
        return ExtrusionSurface(fields["curve"], fields["direction"])
    if model_type is RevolutionSurface:
        return RevolutionSurface(
            fields["curve"], fields["axis_origin"], fields["axis_direction"]
        )
    if model_type is RuledSurface:
        return RuledSurface(fields["first"], fields["second"])
    if model_type is SurfaceIsoparametricCurve:
        expected_fields = {
            "surface",
            "fixed_axis",
            "fixed_value",
            "fixed_value_exact",
            "parameter_range",
        }
        if set(fields) != expected_fields:
            raise ValueError("Stored isoparametric curve field set is not canonical.")
        return SurfaceIsoparametricCurve(
            fields["surface"],
            fields["fixed_axis"],
            fields["fixed_value"],
            fixed_value_exact=fields["fixed_value_exact"],
            parameter_range=fields["parameter_range"],
        )
    if model_type is PlanePatch:
        return PlanePatch(fields["origin"], fields["first_axis"], fields["second_axis"])
    if model_type is CylinderPatch:
        return CylinderPatch(
            fields["origin"],
            fields["first_axis"],
            fields["second_axis"],
            fields["axis"],
            fields["radius"],
            origin_exact=fields["origin_exact"],
            first_axis_exact=fields["first_axis_exact"],
            second_axis_exact=fields["second_axis_exact"],
            axis_exact=fields["axis_exact"],
            radius_exact=fields["radius_exact"],
        )
    if model_type is ConePatch:
        return ConePatch(
            fields["origin"],
            fields["first_axis"],
            fields["second_axis"],
            fields["axis"],
            fields["reference_radius"],
            fields["semi_angle"],
        )
    if model_type is SpherePatch:
        return SpherePatch(
            fields["center"],
            fields["first_axis"],
            fields["second_axis"],
            fields["axis"],
            fields["radius"],
        )
    if model_type is TorusPatch:
        return TorusPatch(
            fields["center"],
            fields["first_axis"],
            fields["second_axis"],
            fields["axis"],
            fields["major_radius"],
            fields["minor_radius"],
            center_exact=fields["center_exact"],
            first_axis_exact=fields["first_axis_exact"],
            second_axis_exact=fields["second_axis_exact"],
            axis_exact=fields["axis_exact"],
            major_radius_exact=fields["major_radius_exact"],
            minor_radius_exact=fields["minor_radius_exact"],
        )
    if model_type is BSplineSurfacePatch:
        return BSplineSurfacePatch(
            fields["control_points"],
            fields["weights"],
            fields["u_knots"],
            fields["v_knots"],
            fields["u_degree"],
            fields["v_degree"],
            exact_control_points=fields["exact_control_point_bank"],
            exact_weights=fields["exact_weight_bank"],
            exact_u_knots=fields["exact_u_knot_bank"],
            exact_v_knots=fields["exact_v_knot_bank"],
            control_point_ids=fields["control_point_ids"],
        )
    if model_type is BSplineCurve:
        return BSplineCurve(
            fields["control_points"],
            fields["weights"],
            fields["knots"],
            fields["degree"],
            exact_control_points=fields["exact_control_point_bank"],
            exact_weights=fields["exact_weight_bank"],
            exact_knots=fields["exact_knot_bank"],
        )
    if model_type is ConicParameterRoot:
        return ConicParameterRoot(
            fields["conic"],
            fields["cosine_exact"],
            fields["sine_exact"],
            turns=fields["turns"],
        )
    if model_type is ConicRootLinePCurve:
        expected_fields = {
            "root",
            "phase_shift_exact",
            "height_origin_exact",
            "height_direction_exact",
            "angle",
            "height_origin",
            "height_direction",
            "parameter_range",
        }
        if set(fields) != expected_fields:
            raise ValueError("Stored conic p-curve fields are not canonical.")
        angle = np.asarray(fields["angle"])
        if (
            angle.shape != ()
            or angle.dtype != np.dtype(np.float64)
            or not bool(np.isfinite(angle))
        ):
            raise ValueError(
                "Stored conic p-curve angle must be a finite float64 scalar."
            )
        curve = ConicRootLinePCurve(
            fields["root"],
            fields["height_origin_exact"],
            fields["height_direction_exact"],
            angle_realization=float(angle),
            phase_shift_exact=fields["phase_shift_exact"],
            parameter_range=fields["parameter_range"],
        )
        if (
            not np.array_equal(np.asarray(curve.angle), np.asarray(fields["angle"]))
            or not np.array_equal(
                np.asarray(curve.height_origin), np.asarray(fields["height_origin"])
            )
            or not np.array_equal(
                np.asarray(curve.height_direction),
                np.asarray(fields["height_direction"]),
            )
        ):
            raise ValueError(
                "Stored conic p-curve shadows differ from exact source coefficients."
            )
        return curve
    if model_type is PlaneCylinderSectionPCurve:
        expected_fields = {
            "offset_exact",
            "linear_exact",
            "cosine_exact",
            "sine_exact",
            "angle_offset_root",
            "parameter_range",
            "offset",
            "linear",
            "cosine",
            "sine",
            "angle_offset",
        }
        if set(fields) != expected_fields:
            raise ValueError(
                "Stored plane-cylinder section p-curve field set is not canonical."
            )
        curve = PlaneCylinderSectionPCurve(
            fields["offset_exact"],
            fields["linear_exact"],
            fields["cosine_exact"],
            fields["sine_exact"],
            parameter_range=fields["parameter_range"],
            angle_offset_root=fields["angle_offset_root"],
        )
        if (
            not np.array_equal(np.asarray(curve.offset), np.asarray(fields["offset"]))
            or not np.array_equal(np.asarray(curve.linear), np.asarray(fields["linear"]))
            or not np.array_equal(np.asarray(curve.cosine), np.asarray(fields["cosine"]))
            or not np.array_equal(np.asarray(curve.sine), np.asarray(fields["sine"]))
            or not np.array_equal(
                np.asarray(curve.angle_offset), np.asarray(fields["angle_offset"])
            )
        ):
            raise ValueError(
                "Stored harmonic p-curve shadows differ from the exact source phase "
                "and coefficients."
            )
        return curve
    if model_type is TrimDomain:
        curve_loops = fields["curve_loops"]
        if curve_loops:
            if fields["outer"] is not None or fields["holes"]:
                raise ValueError(
                    "Stored TrimDomain mixes polygon and source-curve loops."
                )
            return TrimDomain.from_curve_loops(curve_loops)
        return TrimDomain(fields["outer"], fields["holes"], curve_loops=())
    if model_type.__module__ == "phydrax.geometry.brep._boolean_core":
        from ..geometry.brep._boolean_core import AffinePCurve, PeriodicPCurve

        if model_type is AffinePCurve:
            return AffinePCurve(fields["curve"], fields["matrix"], fields["offset"])
        if model_type is PeriodicPCurve:
            return PeriodicPCurve(
                fields["source_curve"], fields["patch"], fields["period_shifts"]
            )
        raise TypeError(
            f"No explicit Boolean curve constructor for {model_type.__qualname__}."
        )
    if model_type is SurfaceMappedCurve:
        return SurfaceMappedCurve(fields["surface"], fields["pcurve"])
    if model_type is ExactLineCurve:
        return ExactLineCurve(fields["first_exact"], fields["last_exact"])
    if model_type is OffsetSurface:
        expected_fields = {
            "base",
            "distance",
            "distance_exact",
            "frame_sign",
            "_sphere_revolution_witness",
        }
        if set(fields) != expected_fields:
            raise ValueError("Stored OffsetSurface field set is not canonical.")
        stored_witness = fields["_sphere_revolution_witness"]
        patch = OffsetSurface(
            fields["base"],
            fields["distance_exact"],
            frame_sign=fields["frame_sign"],
        )
        if patch.exact_sphere_revolution_witness() != stored_witness:
            raise ValueError(
                "Stored offset-surface revolution witness differs from source recomputation."
            )
        if not np.array_equal(np.asarray(patch.distance), np.asarray(fields["distance"])):
            raise ValueError(
                "Stored offset-surface distance differs from its exact source coefficient."
            )
        return patch
    if model_type is TSplineSurfacePatch:
        source_basis = fields["source_basis_certificate"]
        target_basis = fields["basis_certificate"]
        transfer_meshes = tuple(fields["transfer_meshes"])
        stored_transfer_certificates = tuple(fields["transfer_basis_certificates"])
        source_mesh = fields["source_mesh"]
        target_mesh = fields["mesh"]
        source_certificate = source_mesh.certify(tolerance=source_basis.tolerance)
        target_certificate = target_mesh.certify(tolerance=target_basis.tolerance)
        transfer_certificates = tuple(
            mesh.certify(tolerance=certificate.tolerance)
            for mesh, certificate in zip(
                transfer_meshes, stored_transfer_certificates, strict=True
            )
        )
        if (
            not source_certificate.passed
            or not target_certificate.passed
            or source_certificate.certificate_id != source_basis.certificate_id
            or target_certificate.certificate_id != target_basis.certificate_id
            or any(
                not recomputed.passed
                or recomputed.certificate_id != stored.certificate_id
                for recomputed, stored in zip(
                    transfer_certificates, stored_transfer_certificates, strict=True
                )
            )
        ):
            raise ValueError("Stored T-spline basis evidence differs from recomputation.")
        patch = TSplineSurfacePatch(
            target_mesh,
            fields["control_points"],
            fields["weights"],
            exact_control_points=fields["exact_source_control_points"],
            exact_weights=fields["exact_source_weights"],
            source_mesh=source_mesh,
            transfer_plans=fields["transfer_plans"],
            source_basis_certificate=source_certificate,
            basis_certificate=target_certificate,
            transfer_meshes=transfer_meshes,
            transfer_basis_certificates=transfer_certificates,
        )
        patch.validate_exact_transfer_chain()
        return patch
    from ..geometry._meshing_domain import _PlacedSurfacePatch

    if model_type is _PlacedSurfacePatch:
        if set(fields) != {
            "source",
            "placement",
            "linear",
            "offset",
            "orientation_sign",
        }:
            raise ValueError("Stored placed surface field set is not canonical.")
        placed = _PlacedSurfacePatch(fields["source"], fields["placement"])
        if (
            placed.orientation_sign != fields["orientation_sign"]
            or not np.array_equal(np.asarray(placed.linear), np.asarray(fields["linear"]))
            or not np.array_equal(np.asarray(placed.offset), np.asarray(fields["offset"]))
        ):
            raise ValueError(
                "Stored placed surface realization differs from its exact placement."
            )
        return placed
    raise TypeError(f"No explicit CAD constructor for {model_type.__qualname__}.")


__all__ = ["load_brep_archive", "save_brep_archive"]
