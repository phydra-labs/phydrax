"""Independent image-cell sources and native requests for meshing scenarios."""

from itertools import combinations, product

import numpy as np

import phydrax as phx
from phydrax.geometry._compartments import CompartmentMeshingSource
from phydrax.geometry.multiregion_surface._label_extraction import LabelFieldVolumeBinding


def _source(
    values: np.ndarray, matrix: np.ndarray | None = None
) -> CompartmentMeshingSource:
    contract = phx.SpatialCoordinateContract(
        phx.units.MILLIMETER, coordinate_system="cartesian-lps", reference_frame="patient"
    )
    affine = phx.imaging.ImageIndexAffine(
        np.eye(4) if matrix is None else matrix,
        "voxels",
        contract,
        phx.imaging.ImageAxisConvention.LPS,
    )
    manifest = phx.qualification.ReferenceArtifactManifest(
        "synthetic-labels",
        checksum_algorithm="sha256",
        checksum="2" * 64,
        size_bytes=1,
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
    asset = phx.imaging.MedicalImageAsset(
        "labels",
        "segmentation",
        values,
        affine,
        phx.imaging.ImageFieldSpec.named(
            "segmentation", phx.units.ONE, phx.measurement.ValueKind.CATEGORICAL
        ),
        phx.imaging.DeidentificationEvidence(
            "deid", "subject", "protocol", True, True, True
        ),
        (manifest,),
        phx.measurement.DerivationRecord(
            phx.measurement.DataOrigin.SYNTHETIC,
            phx.measurement.DataStage.RECONSTRUCTED,
            transformation_id="synthetic-native-compartment",
        ),
    )
    label_values = tuple(np.unique(values).tolist())
    ontology = phx.imaging.LabelOntology(
        "ontology",
        "synthetic",
        "1",
        tuple(
            phx.imaging.LabelDefinition(value, f"label:{value}", f"Material {value}")
            for value in label_values
        ),
    )
    labels = phx.imaging.LabelVolume(asset, ontology)
    regions = tuple(f"region:{value}" for value in label_values)
    definitions = tuple(
        phx.geometry.CompartmentDefinition(
            region,
            (f"label:{value}",),
            "material",
            allowed_neighbor_ids=tuple(other for other in regions if other != region),
        )
        for value, region in zip(label_values, regions, strict=True)
    )
    interfaces = tuple(
        phx.geometry.CompartmentInterfaceDefinition(
            f"interface:{first}:{second}", first, second, "exchange", required=False
        )
        for first, second in combinations(regions, 2)
    )
    complex_ = phx.imaging.build_compartment_complex(labels, definitions, interfaces)
    # Independent coarse box: outward faces are not copied from volume output.
    corners = np.asarray(tuple(product((0, 1), repeat=3)), dtype=np.float64)
    corners = corners * np.asarray(values.shape, dtype=np.float64) - 0.5
    faces = np.asarray(
        (
            (0, 1, 3),
            (0, 3, 2),
            (4, 7, 5),
            (4, 6, 7),
            (0, 5, 1),
            (0, 4, 5),
            (2, 7, 6),
            (2, 3, 7),
            (0, 6, 4),
            (0, 2, 6),
            (1, 7, 3),
            (1, 5, 7),
        ),
        dtype=np.int64,
    )
    if np.linalg.det(affine.matrix[:3, :3]) < 0.0:
        faces = faces[:, ::-1]
    outer = phx.geometry.SurfaceModel.from_triangles(
        affine.index_to_world(corners),
        faces,
        phx.geometry.SurfaceMetadata(
            source_id="outer-box",
            source_revision=labels.label_volume_id,
            coordinate_contract=contract,
            provenance=("synthetic",),
        ),
    )
    return CompartmentMeshingSource(
        labels,
        complex_,
        outer,
        phx.imaging.extract_compartment_surfaces(labels, complex_),
    )


def _execute(
    source: CompartmentMeshingSource | LabelFieldVolumeBinding,
) -> phx.meshing.CellMeshingResult:
    scope = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        f"{source.source_id}:boundary",
        np.asarray((0,), dtype=np.int64),
    )
    request = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, 3.0, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
    )
    return (
        phx.meshing.NativeMeshingProvider(
            phx.meshing.NativeMeshingOptions("image_material_tetrahedral")
        )
        .plan(source, request, coordinate_contract=source.coordinate_contract)
        .execute()
    )


def _region_measure_array(evidence: phx.meshing.RegionMeshingEvidence) -> np.ndarray:
    """Require actual affine material measures before comparing their values."""
    measures: list[float] = []
    for value in evidence.coverage.achieved_region_measures:
        if value is None:
            raise ValueError("The affine image fixture has no achieved region measure.")
        measures.append(value)
    return np.asarray(measures, dtype=np.float64)
