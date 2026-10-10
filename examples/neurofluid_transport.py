#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Generate a native compartment mesh and conserve closed 3D--1D--0D inventory.

Coordinates and transport coefficients use millimeters, seconds, and millimolar.
The occupied-voxel source has a shared first/second interface at x = 0.5 mm.
"""

import json
from hashlib import sha256

import jax.numpy as jnp
import numpy as np

import phydrax as phx


contract = phx.SpatialCoordinateContract(
    phx.units.MILLIMETER,
    coordinate_system="cartesian-lps",
    reference_frame="synthetic-neurofluid",
)
values = np.ones((2, 2, 2), dtype=np.int16)
values[1] = 2
manifest = phx.qualification.ReferenceArtifactManifest(
    "synthetic-neurofluid-labels",
    checksum_algorithm="sha256",
    checksum=sha256(values.tobytes()).hexdigest(),
    size_bytes=values.nbytes,
    license_id="synthetic",
    commercial_use_permitted=True,
    redistribution_permitted=True,
    training_use_permitted=True,
    export_permitted=True,
    export_classification="public",
    nondimensionalization={"length": 1.0},
    uncertainty={"value": 0.0},
    lineage_ids=("synthetic-neurofluid",),
)
asset = phx.imaging.MedicalImageAsset(
    "synthetic-neurofluid-labels",
    "segmentation",
    values,
    phx.imaging.ImageIndexAffine(
        np.eye(4), "voxels", contract, phx.imaging.ImageAxisConvention.LPS
    ),
    phx.imaging.ImageFieldSpec.named(
        "segmentation", phx.units.ONE, phx.measurement.ValueKind.CATEGORICAL
    ),
    phx.imaging.DeidentificationEvidence(
        "synthetic-deid", "synthetic-subject", "synthetic", True, True, True
    ),
    (manifest,),
    phx.measurement.DerivationRecord(
        phx.measurement.DataOrigin.SYNTHETIC,
        phx.measurement.DataStage.RECONSTRUCTED,
        transformation_id="synthetic-neurofluid-generator",
    ),
)
labels = phx.imaging.LabelVolume(
    asset,
    phx.imaging.LabelOntology(
        "synthetic-neurofluid",
        "synthetic",
        "1",
        (
            phx.imaging.LabelDefinition(1, "first-label", "First"),
            phx.imaging.LabelDefinition(2, "second-label", "Second"),
        ),
    ),
)
compartments = phx.imaging.build_compartment_complex(
    labels,
    (
        phx.geometry.CompartmentDefinition(
            "first", ("first-label",), "material", allowed_neighbor_ids=("second",)
        ),
        phx.geometry.CompartmentDefinition(
            "second", ("second-label",), "material", allowed_neighbor_ids=("first",)
        ),
    ),
    (
        phx.geometry.CompartmentInterfaceDefinition(
            "first-second", "first", "second", "exchange"
        ),
    ),
)
outer = phx.geometry.SurfaceModel.from_triangles(
    np.asarray(
        (
            (-0.5, -0.5, -0.5),
            (1.5, -0.5, -0.5),
            (1.5, 1.5, -0.5),
            (-0.5, 1.5, -0.5),
            (-0.5, -0.5, 1.5),
            (1.5, -0.5, 1.5),
            (1.5, 1.5, 1.5),
            (-0.5, 1.5, 1.5),
        ),
        dtype=np.float64,
    ),
    np.asarray(
        (
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (3, 7, 6),
            (3, 6, 2),
            (0, 4, 7),
            (0, 7, 3),
            (1, 2, 6),
            (1, 6, 5),
        ),
        dtype=np.int64,
    ),
    phx.geometry.SurfaceMetadata(
        source_id="synthetic-outer",
        source_revision=labels.label_volume_id,
        coordinate_contract=contract,
        provenance=("synthetic",),
    ),
)
source = phx.geometry.CompartmentMeshingSource(
    labels,
    compartments,
    outer,
    phx.imaging.extract_compartment_surfaces(labels, compartments),
)
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
            scope, 1.0, strength=phx.meshing.SizeControlStrength.SOFT
        ),
    ),
)
bulk_mesh = (
    phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions("image_material_tetrahedral")
    )
    .plan(source, request, coordinate_contract=contract)
    .execute()
)
network = phx.discretization.MetricNetworkPlan.from_arrays(
    np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), dtype=np.float64),
    np.asarray(((0, 1),), dtype=np.int64),
    contract,
    areas=np.asarray((0.01,), dtype=np.float64),
    perimeters=np.asarray((0.2,), dtype=np.float64),
    root_vertex_ids=np.asarray((0,), dtype=np.int64),
    tip_vertex_ids=np.asarray((1,), dtype=np.int64),
).prepare()
case = phx.applications.neurofluid.NeurofluidCase(
    "synthetic-native-case", labels, compartments, bulk_mesh, network
)
cell_count = bulk_mesh.mesh.entity_set(3).count
boundary_count = np.count_nonzero(
    np.asarray(bulk_mesh.mesh.entity_set(2).subset("boundary").mask)
)
parameters = phx.applications.neurofluid.NeurofluidTransportParameters(
    porosity=np.full(cell_count, 0.8, dtype=np.float64),
    bulk_diffusivity=np.full(cell_count, 0.01, dtype=np.float64),
    bulk_velocity=np.zeros((cell_count, 3), dtype=np.float64),
    bulk_boundary_volume_flux=np.zeros(boundary_count, dtype=np.float64),
    bulk_boundary_inflow_concentration=np.zeros(boundary_count, dtype=np.float64),
    bulk_removal_rate=np.zeros(cell_count, dtype=np.float64),
    network_diffusivity=np.asarray((0.01,), dtype=np.float64),
    units=phx.applications.neurofluid.NeurofluidTransportUnits(
        phx.units.MILLIMETER, phx.units.SECOND, phx.units.MILLIMOLAR
    ),
    network_volume_flow=np.zeros(1, dtype=np.float64),
    exchange_coefficients=np.full(2, 0.5, dtype=np.float64),
    averaging_radius=0.01,
    reservoir_volumes=np.asarray((0.003,), dtype=np.float64),
    reservoir_coefficients=np.asarray((0.2,), dtype=np.float64),
)
runtime = phx.applications.neurofluid.NeurofluidTransportPlan(case, parameters).prepare()
state = phx.equations.MixedDimensionalTransportState(
    jnp.full(cell_count, 3.0, dtype=jnp.float64),
    jnp.asarray((1.0, 1.5), dtype=jnp.float64),
    jnp.asarray((0.5,), dtype=jnp.float64),
)
initial_mass = float(runtime.ledger(state).total_mass)
solve_policy = phx.linalg.LinearSolvePolicy(
    tolerance=phx.linalg.TolerancePolicy(
        relative=1.0e-12, absolute=1.0e-14, max_steps=2048
    )
)
for _ in range(5):
    step = runtime.step_backward_euler(state, 0.05, policy=solve_policy)
    if not bool(step.accepted):
        raise RuntimeError("Mixed-dimensional transport step was rejected.")
    state = step.state
final = runtime.ledger(state)
mass_error = abs(float(final.total_mass) - initial_mass)
if mass_error > 1.0e-9 or not bool(final.successful):
    raise RuntimeError(
        "Closed mixed-dimensional transport did not conserve mass: "
        f"initial={initial_mass:.17g}, final={float(final.total_mass):.17g}, "
        f"mass_error={mass_error:.17g}, ledger_successful={bool(final.successful)}, "
        f"exchange_defect={float(final.exchange_defect):.17g}."
    )
evidence = bulk_mesh.region_evidence
if evidence is None:
    raise RuntimeError("Native compartment generation omitted region evidence.")
compartment_ids = tuple(sorted(set(evidence.cell_region_ids)))
region_indices = {region: index for index, region in enumerate(compartment_ids)}
diagnostics = phx.applications.neurofluid.neurofluid_diagnostics(
    runtime,
    state,
    jnp.asarray(
        tuple(region_indices[region] for region in evidence.cell_region_ids),
        dtype=jnp.int32,
    ),
    compartment_ids,
)
print(
    json.dumps(
        {
            "provider": bulk_mesh.provider.name,
            "source_revision": evidence.source_revision,
            "compartments": compartment_ids,
            "adjacency": evidence.adjacency_pairs,
            "compartment_inventory": np.asarray(diagnostics.compartment_mass).tolist(),
            "network": np.asarray(state.network).tolist(),
            "reservoirs": np.asarray(state.reservoirs).tolist(),
            "mass_error": mass_error,
            "exchange_defect": float(final.exchange_defect),
        },
        indent=2,
    )
)
