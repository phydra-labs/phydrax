#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Run portable synthetic qualification of imaging, compartments, and H(div)."""

from __future__ import annotations

import json
from hashlib import sha256

import numpy as np

import phydrax as phx


def manifest(payload: bytes):
    return phx.qualification.ReferenceArtifactManifest(
        "qualification-image",
        checksum_algorithm="sha256",
        checksum=sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="synthetic",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="public",
        nondimensionalization={"length": 1.0},
        uncertainty={"value": 0.0},
        lineage_ids=("synthetic",),
    )


def qualify() -> dict[str, object]:
    contract = phx.SpatialCoordinateContract(
        phx.units.MILLIMETER,
        coordinate_system="cartesian-lps",
        reference_frame="qualification",
    )
    affine = phx.imaging.ImageIndexAffine(
        np.eye(4), "voxels", contract, phx.imaging.ImageAxisConvention.LPS
    )
    values = np.ones((2, 2, 2), dtype=np.int16)
    values[1] = 2
    asset = phx.imaging.MedicalImageAsset(
        "qualification-labels",
        "segmentation",
        values,
        affine,
        phx.imaging.ImageFieldSpec.named(
            "segmentation", phx.units.ONE, phx.measurement.ValueKind.CATEGORICAL
        ),
        phx.imaging.DeidentificationEvidence(
            "qualification-deid", "subject-0", "synthetic", True, True, True
        ),
        manifest(values.tobytes()),
        phx.measurement.DerivationRecord(
            phx.measurement.DataOrigin.SYNTHETIC,
            phx.measurement.DataStage.RECONSTRUCTED,
            transformation_id="synthetic-qualification-generator",
        ),
    )
    labels = phx.imaging.LabelVolume(
        asset,
        phx.imaging.LabelOntology(
            "qualification-labels",
            "synthetic",
            "1",
            (
                phx.imaging.LabelDefinition(1, "first-label", "First"),
                phx.imaging.LabelDefinition(2, "second-label", "Second"),
            ),
        ),
    )
    definitions = (
        phx.geometry.CompartmentDefinition(
            "first", ("first-label",), "material", allowed_neighbor_ids=("second",)
        ),
        phx.geometry.CompartmentDefinition(
            "second", ("second-label",), "material", allowed_neighbor_ids=("first",)
        ),
    )
    interface = phx.geometry.CompartmentInterfaceDefinition(
        "first-second", "first", "second", "exchange"
    )
    complex_ = phx.imaging.build_compartment_complex(labels, definitions, (interface,))
    surfaces = phx.imaging.extract_compartment_surfaces(labels, complex_)
    element = phx.discretization.tetrahedral_rt_element()
    centers = np.asarray(
        (
            (1 / 3, 1 / 3, 0.0),
            (1 / 3, 0.0, 1 / 3),
            (1 / 3, 1 / 3, 1 / 3),
            (0.0, 1 / 3, 1 / 3),
        )
    )
    values_rt, _ = element.tabulate(centers)
    normals = np.asarray(
        ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0), (1.0, 1.0, 1.0), (-1.0, 0.0, 0.0))
    )
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    areas = np.asarray((0.5, 0.5, np.sqrt(3.0) / 2.0, 0.5))
    flux = np.asarray(
        [
            [
                areas[face] * np.dot(values_rt[face, basis], normals[face])
                for basis in range(4)
            ]
            for face in range(4)
        ]
    )
    flux_error = float(np.max(np.abs(flux - np.eye(4))))
    tetrahedron = phx.discretization.CellMesh.from_tetrahedra(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        ),
        np.asarray(((0, 1, 2, 3),)),
    )
    boundary_face_id = int(np.asarray(tetrahedron.entity_set(2).entity_ids)[0])
    normal_boundary = phx.equations.fem.HDivNormalBoundaryCondition(
        np.asarray((boundary_face_id,)),
        resistance=2.0,
        prescribed_flux=1.0,
    )
    hdiv = phx.equations.fem.HDivStokesPlan(
        tetrahedron,
        phx.discretization.PressureGaugePolicy("mean-zero"),
        normal_boundaries=(normal_boundary,),
    ).prepare()
    zero_velocity, pressure, multiplier = hdiv.state_space.zeros()
    velocity = hdiv.normal_flux_operator.adjoint_mv(np.asarray((1.0 / 6.0,)))
    normal_flow_error = float(
        np.max(np.abs(np.asarray(hdiv.normal_flux(velocity)) - 1.0))
    )
    resistance_power_error = abs(
        float(
            np.vdot(
                np.asarray(velocity),
                np.asarray(hdiv.normal_resistance_operator.mv(velocity)),
            )
        )
        - 2.0
    )
    constraint_error = float(
        np.max(np.abs(np.asarray(hdiv.residual((velocity, pressure, multiplier))[2])))
    )
    if (
        not complex_.adjacency.successful
        or flux_error > 1.0e-12
        or not bool(hdiv.evidence.successful)
        or normal_flow_error > 1.0e-12
        or resistance_power_error > 1.0e-12
        or constraint_error > 1.0e-12
    ):
        raise RuntimeError("Neurofluid synthetic qualification failed.")
    return {
        "compartments": len(complex_.compartments),
        "interfaces": len(surfaces.surfaces),
        "interface_triangles": surfaces.surfaces[0].surface.mesh.entity_set(2).count,
        "rt0_flux_error": flux_error,
        "bdm2_velocity_dofs": int(zero_velocity.size),
        "dg1_pressure_dofs": int(pressure.size),
        "normal_flow_error": normal_flow_error,
        "resistance_power_error": resistance_power_error,
        "normal_constraint_error": constraint_error,
        "successful": True,
    }


def main() -> None:
    print(json.dumps(qualify(), indent=2))


if __name__ == "__main__":
    main()
