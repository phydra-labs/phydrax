# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._records import _quality_summary, _result_identity
from tools.meshing_qualification._runtime import (
    _measured,
    _missing_dependency,
    _runtime_identity,
    qualification_failure,
)


_CAD_CURVING_ASSOCIATION = phx.meshing.AssociationPropagationPolicy(
    classification_tolerance=1.0e-8
)


def _cad_curving_volume(mesh: Any, geometry: Any) -> float:
    """Volume of mapped P1/P2/P3 tetrahedra.

    The degree-aware finite-element rule integrates their polynomial Jacobian
    determinant exactly.
    """
    field = phx.discretization.FiniteElementFieldSpec(
        "x", phx.discretization.lagrange_element("tetrahedron", 1)
    )
    discretization = phx.discretization.FiniteElementPlan(
        mesh, field, coordinate_spec=geometry
    ).prepare()
    blocks = discretization.evaluate_geometry("x", geometry.coordinates)
    return float(sum(jnp.sum(block.measure) for block in blocks))


def _cad_curving_certificate(certificate: Any) -> dict[str, object]:
    lower = np.asarray(certificate.determinant_lower)
    upper = np.asarray(certificate.determinant_upper)
    return {
        "certificate_id": certificate.certificate_id,
        "certified_valid": certificate.certified_valid_count,
        "invalid": certificate.invalid_count,
        "unresolved": certificate.unresolved_count,
        "minimum_determinant_lower_bound": float(np.min(lower)),
        "minimum_lower_to_upper_ratio": float(np.min(lower / upper)),
        "depth_histogram": np.bincount(np.asarray(certificate.depth)).tolist(),
        "retained_bytes": sum(
            array.nbytes
            for array in (
                certificate.status,
                certificate.determinant_lower,
                certificate.determinant_upper,
                certificate.depth,
            )
        ),
    }


def _cad_curving_association(mesh: Any, projection: Any) -> Any:
    """Vertex association plus completeness and ambiguity of every entity level."""

    vertices, vertex_record = _measured(
        lambda: phx.meshing.associate_mesh_vertices(
            mesh, projection, policy=_CAD_CURVING_ASSOCIATION
        )
    )
    entities, entity_record = _measured(
        lambda: tuple(
            phx.meshing.associate_mesh_entities(
                mesh,
                vertices,
                projection,
                dimension,
                policy=_CAD_CURVING_ASSOCIATION,
            )
            for dimension in (1, 2, 3)
        )
    )
    counts = {
        str(dimension): {
            "association_id": association.association_id,
            "rows": association.target_global_ids.shape[0],
            "resolved": int(np.count_nonzero(np.asarray(association.resolved))),
            "ambiguous": int(np.count_nonzero(np.asarray(association.ambiguous))),
            "maximum_residual": float(np.max(np.asarray(association.residuals))),
        }
        for dimension, association in enumerate((vertices, *entities))
    }
    if any(
        entry["resolved"] != entry["rows"] or entry["ambiguous"]
        for entry in counts.values()
    ):
        raise qualification_failure(
            "_cad_curving_association",
            ("cad-curving: a mesh entity has no unique B-Rep class."),
            {"counts": counts},
        )
    tolerance = _CAD_CURVING_ASSOCIATION.classification_tolerance
    maximum_residual = counts["0"]["maximum_residual"]
    if not isinstance(maximum_residual, float):
        raise TypeError(
            f"_cad_curving_association.maximum_residual must be a measured float; got {type(maximum_residual).__name__!r}."
        )
    if maximum_residual > tolerance:
        raise qualification_failure(
            "_cad_curving_association",
            ("cad-curving: a mesh vertex lies off its B-Rep class."),
            {"maximum_residual": maximum_residual, "tolerance": tolerance},
        )
    return vertices, counts, {"vertices": vertex_record, "entities": entity_record}


def _cad_curving_degree(
    mesh: Any,
    association: Any,
    projection: Any,
    degree: Any,
    exact_volume: Any,
    fidelity_tolerance: float,
) -> Any:
    """Curve, independently certify, and measure one geometry degree."""

    policy = phx.meshing.HighOrderCurvingPolicy(
        degree=degree, fidelity_tolerance=fidelity_tolerance
    )
    curved, curving_record = _measured(
        lambda: phx.meshing.curve_cell_mesh(mesh, association, projection, policy=policy)
    )

    if curved.status is not phx.meshing.HighOrderCurvingStatus.CURVED:
        raise qualification_failure(
            "_cad_curving_degree",
            (f"cad-curving: P{degree} ended with {curved.status}."),
            {"curved": curved},
        )
    certificate, certification_record = _measured(
        lambda: phx.discretization.certify_cell_geometry_validity(
            curved.geometry,
            mesh=mesh,
        )
    )
    valid = phx.discretization.CellValidityStatus.CERTIFIED_VALID

    if not np.all(np.asarray(certificate.status) == valid):
        raise qualification_failure(
            "_cad_curving_degree",
            (f"cad-curving: a P{degree} cell is not certified valid."),
            {"certificate": certificate, "valid": valid},
        )

    if certificate.certificate_id != curved.evidence.certificate.certificate_id:
        raise qualification_failure(
            "_cad_curving_degree",
            ("cad-curving: recertification differs from curving evidence."),
            {"certificate": certificate, "curved": curved},
        )

    if curved.evidence.maximum_residual > policy.residual_tolerance:
        raise qualification_failure(
            "_cad_curving_degree",
            (f"cad-curving: P{degree} nodes lie off the B-Rep."),
            {"curved": curved, "policy": policy},
        )
    straight = phx.meshing.verify_curved_geometry(
        curved.straight,
        mesh,
        association,
        projection,
        policy=policy,
    )
    if straight.accepted:
        raise qualification_failure(
            "_cad_curving_degree",
            ("cad-curving: straight nodes lie on the curved B-Rep."),
            {"straight": straight},
        )
    volumes, volume_record = _measured(
        lambda: (
            _cad_curving_volume(mesh, curved.straight),
            _cad_curving_volume(mesh, curved.geometry),
        )
    )
    errors = tuple(abs(volume - exact_volume) / exact_volume for volume in volumes)
    if not errors[1] < 0.25 * errors[0]:
        raise qualification_failure(
            "_cad_curving_degree",
            (f"cad-curving: P{degree} did not reduce the volume error."),
            {"errors": errors},
        )

    geometry = curved.geometry
    return {
        "target": {
            "result_id": curved.result_id,
            "geometry_layout_id": geometry.geometry_layout_id,
            "evidence_id": curved.evidence.evidence_id,
            "geometry_nodes": geometry.coordinates.shape[0],
        },
        "quality": {
            "before": {
                "minimum_scaled_jacobian": straight.minimum_scaled_jacobian,
                "maximum_distortion": straight.maximum_distortion,
                "certificate": _cad_curving_certificate(straight.certificate),
            },
            "after": {
                "minimum_scaled_jacobian": curved.evidence.minimum_scaled_jacobian,
                "maximum_distortion": curved.evidence.maximum_distortion,
                "certificate": _cad_curving_certificate(certificate),
            },
        },
        "conservation": {
            "exact_volume": exact_volume,
            "straight_volume": volumes[0],
            "curved_volume": volumes[1],
            "straight_relative_error": errors[0],
            "curved_relative_error": errors[1],
            "straight_node_residual": straight.maximum_residual,
            "curved_node_residual": curved.evidence.maximum_residual,
            "residual_tolerance": policy.residual_tolerance,
        },
        "curving": {
            "status": curved.status.value,
            "policy_id": policy.policy_id,
            "relaxation_rounds": len(curved.minimizations),
            "constrained_nodes": int(np.count_nonzero(curved.evidence.constrained)),
        },
        "resources": {
            "curving": curving_record,
            "certification": certification_record,
            "volume": volume_record,
            "geometry_retained_bytes": geometry.coordinates.nbytes
            + sum(routes.nbytes for routes in geometry.geometry_dofs),
        },
    }


def _cad_curving_solid(
    name: Any,
    shape: Any,
    exact_volume: Any,
    size: Any,
    directory: Path,
    fidelity_tolerance: float,
) -> Any:
    """Gmsh P1 tetrahedra of one OCCT solid curved to P2 and P3 on its B-Rep."""

    contract = phx.SpatialCoordinateContract.si()
    path = directory / f"{name}.brep"
    phx.interchange.persist_occt_shape(shape, path, coordinate_contract=contract)
    model = phx.interchange.import_occt_brep(
        path, coordinate_contract=contract, linear_deflection=0.05, angular_deflection=0.2
    )
    source = phx.geometry.BRepSource(model)
    provider = phx.meshing.GmshProvider()
    scope = provider.whole_scope(source, 3)
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, size, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
    )
    result, meshing_record = _measured(
        lambda: provider.plan(source, specification).execute()
    )

    if not result.audit.passed:
        raise qualification_failure(
            "_cad_curving_solid",
            (f"cad-curving: the Gmsh {name} mesh failed its audit."),
            {"result": result},
        )
    projection, projection_record = _measured(
        lambda: phx.geometry.prepare_brep_projection(model)
    )
    association, counts, association_records = _cad_curving_association(
        result.mesh,
        projection,
    )

    straight_certificate = phx.discretization.certify_cell_geometry_validity(result.mesh)
    degrees = {
        f"P{degree}": _cad_curving_degree(
            result.mesh,
            association,
            projection,
            degree,
            exact_volume,
            fidelity_tolerance,
        )
        for degree in (2, 3)
    }
    return {
        "source": {
            "source_id": model.source_id,
            "source_revision": model.source_revision,
            "source_digest": model.source_digest,
            "model_id": model.model_id,
            "projection_id": projection.projection_id,
            "brep_entity_counts": list(projection.entity_counts),
            "straight": _result_identity(result),
        },
        "target": {key: record["target"] for key, record in degrees.items()},
        "quality": {
            "before": {
                **_quality_summary(result.quality),
                "certificate": _cad_curving_certificate(straight_certificate),
            },
            "after": {key: record["quality"] for key, record in degrees.items()},
        },
        "association": counts,
        "conservation": {key: record["conservation"] for key, record in degrees.items()},
        "curving": {key: record["curving"] for key, record in degrees.items()},
        "resources": {
            "meshing": meshing_record,
            "projection": projection_record,
            "association": association_records,
            **{key: record["resources"] for key, record in degrees.items()},
        },
        "provider": {
            "provider": result.provider.name,
            "version": result.runtime.actual_version,
        },
    }


def qualify_cad_curving(fidelity_tolerance: float | None) -> dict[str, object]:
    """Certified P2/P3 curving of Gmsh tetrahedra onto OCCT cylinder and sphere B-Reps."""

    # Optional provider imports are admitted only for this fixture workflow.
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere

    if (
        fidelity_tolerance is None
        or not np.isfinite(fidelity_tolerance)
        or fidelity_tolerance <= 0.0
    ):
        raise ValueError(
            "cad_curving.fidelity_tolerance must be explicitly supplied, finite, "
            f"and positive; got {fidelity_tolerance}"
        )
    if find_spec("OCP") is None:
        return _missing_dependency(
            "OCP",
            "This explicit CAD comparison requires phydrax[cad-occt-interop].",
        )
    if find_spec("gmsh") is None:
        return _missing_dependency(
            "gmsh",
            "Install the optional Gmsh Python package: "
            "pip install 'phydrax[meshing-gmsh]'.",
        )

    with TemporaryDirectory(prefix="phydrax-cad-curving-") as temporary:
        directory = Path(temporary)
        solids = {
            "cylinder": _cad_curving_solid(
                "cylinder",
                BRepPrimAPI_MakeCylinder(1.0, 2.0).Shape(),
                2.0 * np.pi,
                0.7,
                directory,
                fidelity_tolerance,
            ),
            "sphere": _cad_curving_solid(
                "sphere",
                BRepPrimAPI_MakeSphere(1.0).Shape(),
                4.0 / 3.0 * np.pi,
                0.6,
                directory,
                fidelity_tolerance,
            ),
        }
    fields = ("source", "target", "quality", "association", "conservation", "curving")
    return {
        "status": "passed",
        "fidelity_tolerance": fidelity_tolerance,
        **{
            field: {name: solid[field] for name, solid in solids.items()}
            for field in fields
        },
        "resources": {name: solid["resources"] for name, solid in solids.items()},
        "runtime": {
            **_runtime_identity(),
            "providers": {name: solid["provider"] for name, solid in solids.items()},
        },
    }


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_cad_curving(args.cad_curving_fidelity)
