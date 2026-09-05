# Cardiovascular geometry interchange and fixed attachments

Cardiovascular geometry enters Phydrax through explicit, immutable boundary
records. Image coordinates, field-transfer semantics, and Purkinje–myocardial
junction (PMJ) routes are prepared on the host. Compiled execution consumes only
fixed-shape arrays and checks evidence against the prepared configuration and
geometry epochs.

The kernel geometry scale is millimetres. Do not infer a coordinate frame or a
unit from filenames, array orientation, scanner conventions, or magnitude.

## Medical-image boundary metadata

`MedicalImageAffine` stores the complete 4 × 4 voxel-index-to-world affine. The
frame and translation unit are part of its identity:

```python
import jax.numpy as jnp
import phydrax as phx

cv = phx.applications.cardiovascular

voxel_to_lps = jnp.asarray(
    (
        (1.25, 0.00, 0.00, 15.0),
        (0.00, 1.25, 0.00, -20.0),
        (0.00, 0.00, 2.00, 6.0),
        (0.00, 0.00, 0.00, 1.0),
    )
)
affine = cv.anatomy.MedicalImageAffine(
    voxel_to_lps,
    cv.anatomy.ImageCoordinateFrame.LPS,
    cv.anatomy.ImageLengthUnit.MILLIMETER,
)
```

Only `LPS` and `RAS` are admitted. `reframe` performs the explicit patient-frame
change by negating the first two world axes; applying LPS → RAS → LPS preserves
the original affine. `in_millimeters` converts metre, centimetre, or micrometre
world coordinates to the kernel scale with exact factors of 1000, 10, or 0.001.
It scales the three world-coordinate rows, including translation, and preserves
the homogeneous row.

The affine is combined with separate identities for acquisition,
de-identification, and data rights:

```python
acquisition = cv.anatomy.ImageAcquisitionIdentity(
    "acq-deid-17",
    "series-deid-4",
    "MR",
    "cine-short-axis",
)
deidentification = cv.anatomy.ImageDeidentificationIdentity(
    "dicom-basic-profile",
    "deid-run-22",
    "attestation-22",
)
rights = cv.anatomy.ImageDataRightsIdentity(
    "rights-17",
    "license-clinical-research",
    "controller-site-a",
    permitted_use_ids=("geometry-reconstruction", "model-validation"),
)
image_boundary = cv.anatomy.CardiacImageBoundaryMetadata(
    affine,
    acquisition,
    deidentification,
    rights,
    coordinate_frame=cv.anatomy.ImageCoordinateFrame.LPS,
    host_fields={
        "field_strength_t": 3.0,
        "sequence_id": "cine-bSSFP",
        "slice_thickness_mm": 2.0,
    },
)
```

`coordinate_frame` deliberately repeats the affine frame at the host boundary.
A disagreement is rejected rather than guessed or silently converted.
Conversion is only performed by an explicit `reframe` call.

### PHI policy

`CardiacImageBoundaryMetadata` is not a general DICOM metadata container. Its
`host_fields` mapping accepts only this non-PHI allowlist:

- `acquisition_plane`, `body_part`, `contrast_agent_class`;
- `echo_time_ms`, `repetition_time_ms`, `temporal_resolution_ms`;
- `field_strength_t`, `flip_angle_degree`;
- `reconstruction_kernel`, `sequence_id`;
- `slice_thickness_mm`, `spatial_resolution_mm`.

Names are normalized to lowercase and sorted for stable identity. Patient name,
patient identifier, birth date, acquisition date, accession number, address,
physician, institution, free-form notes, and every other unlisted field are
refused. Keep source-system audit data in the controlled host archive; pass only
the de-identification and rights identities across this boundary.

## Cardiac field transfers

Phydrax's generic `FieldTransfer` remains the numerical authority. The cardiac
wrapper does not create another interpolation or projection system. It binds an
existing transfer to:

- a quantity, kernel unit, and component-axis convention;
- exact source and target material/reference identities;
- exact source and target `DiscreteFieldSpace` identities;
- source and target geometry epochs;
- source and target reference-configuration epochs;
- fixed source and target coverage masks;
- constant-reproduction and adjoint tolerances.

Given an existing `field_transfer`, prepare the semantic guard once:

```python
configuration = cv.anatomy.CardiacTransferConfiguration.for_transfer(
    field_transfer,
    "transmembrane-voltage",
    "mV",
    "source-material-reference",
    "target-material-reference",
)
prepared_epoch = cv.anatomy.CardiacTransferEpoch(
    4,  # source geometry
    7,  # target geometry
    2,  # source reference configuration
    3,  # target reference configuration
)
cardiac_transfer = cv.anatomy.CardiacFieldTransfer(
    field_transfer,
    configuration,
    prepared_epoch,
    constant_tolerance=1.0e-7,
    adjoint_tolerance=1.0e-7,
)
```

Every action returns a candidate value and `CardiacTransferEvidence`:

```python
result = cardiac_transfer.apply(
    source_voltage,
    current_epoch,
    configuration_id=configuration.configuration_id,
)
if bool(result.evidence.accepted):
    accepted_target_voltage = result.value
```

The evidence contains the per-coordinate coverage masks and their fractions,
constant-field error, normalized sampled pairing-adjoint defect, configuration
match, epoch match, finite status, and final fail-closed `accepted` flag. The
adjoint defect is `abs(<Tx,y> - <x,T*y>) / max(1, abs(<Tx,y>), abs(<x,T*y>))`;
normalization keeps the tolerance meaningful across field magnitudes while
retaining absolute behavior below unit scale. A transfer that claims
`constant_preserving` or `adjoint_paired` is accepted only when the corresponding
numerical evidence meets its tolerance. An unclaimed property is reported but is
not silently promoted into a structural claim.

Changing any one of the four epoch values invalidates the result. Changing the
quantity/reference configuration identity or losing any covered coordinate also
invalidates it. The numerical candidate remains available for diagnosis; it is
never substituted with zeros or an old state. Only a caller's explicit
candidate/evidence/commit policy may install it. Rebuild the generic
`FieldTransfer` and cardiac wrapper at a geometry or reference-configuration
epoch boundary, then checkpoint the new transfer and configuration identities.

## Fixed-capacity PMJ attachment

`PurkinjeAttachmentPlan` prepares one myocardial support index for each active
Purkinje graph candidate. Discovery is a host operation. For each graph point it
chooses the active myocardial support with minimum Euclidean distance; an exact
tie is resolved by the lowest myocardial support index. The resulting integer
routes never change during compiled execution.

```python
graph_points_mm = jnp.asarray(
    ((0.0, 0.25, 0.0), (9.0, 0.0, 0.0), (4.0, 5.0, 0.0))
)
myocardial_support_points_mm = jnp.asarray(
    ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (10.0, 0.0, 0.0))
)
pmj_mask = jnp.asarray((True, True, False))
epoch = cv.anatomy.PMJAttachmentEpoch(5, 8)

attachment = cv.anatomy.PurkinjeAttachmentPlan(
    4,
    2.0,
).prepare(
    graph_points_mm,
    myocardial_support_points_mm,
    pmj_candidate_mask=pmj_mask,
    graph_geometry_id="purkinje-graph-v1",
    myocardial_geometry_id="myocardial-support-v3",
    epoch=epoch,
)
candidate = attachment.evaluate(
    graph_points_mm,
    myocardial_support_points_mm,
    epoch,
)
```

The route arrays, active mask, graph/support pairs, and distance evidence always
have shape `pmj_capacity`. Inactive route slots contain zero output values and a
false mask. Preparation rejects candidate overflow, an empty candidate set, an
empty active myocardial support set, nonfinite coordinates, or inconsistent
masks. It never truncates candidates. Raise the declared capacity and prepare a
new topology epoch instead.

`PMJAttachmentEvidence` reports the distance of every active route in
millimetres, the distance-coverage mask, attached and uncovered counts, coverage
fraction, remaining capacity, epoch match, finite status, and final acceptance.
A route beyond `maximum_distance_mm` is retained for diagnosis but makes the
candidate unacceptable. Runtime evaluation does not search for a closer support
and has no fallback route. Moving another support closer therefore does not
change an attachment; refresh requires an explicit host preparation and epoch
change.

### Fixed-route coupling

The prepared object supplies shape-stable coupling primitives:

- `gather_graph` collects graph values into PMJ-capacity order;
- `gather_myocardium` collects values from the fixed myocardial indices;
- `scatter_to_myocardium` sums PMJ values at their fixed support indices.

Inactive slots are masked before scatter, and repeated support indices
accumulate. These maps and runtime route distances are differentiable with
respect to values and coordinates for the fixed integer routes. Nearest-support
discovery, candidate masks, capacity, graph connectivity, and route changes are
not differentiated. If a geometry epoch changes, evidence becomes invalid even
when array shapes happen to remain equal.

## Qualified curved volume geometry

`HighOrderCardiacGeometryPlan` qualifies curved volume coordinates without
introducing another mesh representation. It consumes the existing
`CellMesh` topology and `CellGeometrySpec` geometry routes. The
admitted envelope is deliberately narrow:

- an H1, point-value, identity-mapped degree-two `SimplexLagrange` coordinate
  element with 10 local DOFs for each tetrahedral block (P2);
- an H1, point-value, identity-mapped isotropic degree-two
  `TensorProductLagrange` coordinate element with 27 local DOFs for each
  hexahedral block (Q2);
- three-dimensional coordinates in millimetres and fixed cell/DOF routes.

Linear geometry, anisotropic tensor elements, prisms, pyramids, and generic
polyhedra are not qualified by this route. Their existence in generic
finite-element infrastructure does not imply cardiovascular qualification.

```python
geometry_epoch = cv.anatomy.HighOrderGeometryEpoch(3, 2)
high_order_plan = cv.anatomy.HighOrderCardiacGeometryPlan(
    cell_mesh,
    finite_element_coordinate_spec,
    boundary_role_id=\"ventricular-boundary-roles-v1\",
    boundary_profile=cardiac_boundary_profile,
    prepared_epoch=geometry_epoch,
    minimum_jacobian_determinant=1.0e-10,
    minimum_cell_measure_mm3=1.0e-10,
)
high_order_geometry = high_order_plan.prepare()
candidate = high_order_geometry.evaluate(
    finite_element_coordinate_spec.coordinates,
    geometry_epoch,
    boundary_role_id=high_order_plan.boundary_role_id,
    boundary_profile_id=high_order_plan.boundary_profile.profile_id,
)
```

Preparation tabulates the existing coordinate element once. P2 tetrahedra use a
five-point-per-Duffy-axis Gauss rule; Q2 hexahedra use a
four-point-per-tensor-axis Gauss rule. Jacobian qualification includes all
quadrature points and all coordinate reference nodes. Cell measures use the
positive quadrature weights and the absolute Jacobian determinant, while
orientation requires a positive signed determinant at every qualification
probe. The default coordinates must pass both configured margins before
preparation succeeds. This is deterministic sampled qualification, not a global
injectivity theorem.

Runtime evaluation contracts new coordinates through those fixed DOF routes and
pretabulated gradients. The candidate preserves the supplied coordinates even
when evidence fails; there is no affine fallback. Coordinate and cell-measure
derivatives therefore cover only the fixed topology and routes. Connectivity,
element family, polynomial degree, boundary roles, and route discovery are
outside the differentiation boundary.

`HighOrderCardiacGeometryEvidence` reports per-block minimum signed Jacobian
determinants, per-block minimum cell measures, finite/orientation/measure
qualification, exact boundary-role and profile matches, and separate geometry
and reference epoch matches. Its lifecycle flags have distinct meanings:

- `transfer_required` is true after a geometry epoch, reference epoch, role, or
  profile binding change, so fields cannot be reused without explicit transfer;
- `rebuild_required` is true after a geometry epoch, role, or profile binding
  change, so the curved route itself must be prepared again;
- a same-epoch invalid coordinate proposal sets `accepted` false but does not
  request transfer or rebuild, allowing the outer candidate policy to reject a
  line-search proposal without changing topology.

## Interchange checklist

Before committing an imported geometry state:

1. preserve the source affine exactly and declare its LPS or RAS frame;
2. preserve its length unit, then explicitly convert to millimetres;
3. attach acquisition, de-identification, and data-rights identities;
4. reject rather than forward PHI or unapproved free-form metadata;
5. bind every field transfer to exact field spaces, references, and four epochs;
6. inspect transfer coverage, constant, adjoint, configuration, and epoch evidence;
7. prepare PMJ candidates within the declared fixed capacity;
8. inspect PMJ distance, coverage, capacity, finite, and epoch evidence;
9. qualify P2 tetrahedral or Q2 hexahedral curved coordinates against their
   fixed FE routes, quadrature, roles, profile, and epochs;
10. inspect Jacobian, measure, transfer, and rebuild evidence;
11. commit and checkpoint only accepted candidates and their stable identities;
12. end the current epoch and re-prepare on any topology, route, frame, reference,
    or support-identity change.
