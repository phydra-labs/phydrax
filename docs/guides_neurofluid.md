# Neurofluid modeling

`phydrax.applications.neurofluid` composes shared imaging, compartment meshing,
metric networks, mixed-dimensional transport, finite-element flow, observations,
and PDE-constrained inversion. Generic numerical machinery remains outside the
application package.

## Case construction

`NeurofluidCase` binds one pseudonymous case revision to:

- a `LabelVolume` and valid `CompartmentComplex`;
- a certified tetrahedral `CellMeshingResult` with source-region evidence;
- a prepared PVS metric network;
- optional concentration and diffusion-tensor images.

Every carrier must use one `SpatialCoordinateContract`. The case revision changes
when segmentation, mesh, network, concentration, or tensor evidence changes.

`CompartmentMeshingSource` binds the label volume, exact `CompartmentComplex`,
outer surface, and extracted oriented interfaces to one source revision and
coordinate contract. Its `occupied-voxel-cells` interpretation treats each image
sample as a cell centered at its voxel index, with faces at index offsets of
±0.5; the image affine maps those cells into physical coordinates.

Use an ordinary `VolumeMeshingSpec` for physical target sizes and controls, then
prepare and execute `NativeMeshingProvider` with
`NativeMeshingOptions("image_material_tetrahedral")`. The canonical
`CellMeshingResult.region_evidence` binds cell global IDs to authoritative source
region IDs, canonical mesh zones, oriented interface facets and patches,
adjacency, and current domain coverage. Cell assignments follow the conforming
material complex, not centroid samples or zone display names.

`NeurofluidCase` requires this evidence to match the exact segmentation revision
and `CompartmentComplex`, and validates its bindings against the current mesh,
geometry, zones, and patches. Transport consumes `bulk_mesh.mesh` and
`bulk_mesh.coordinate_contract` directly. Adapted or moved results must renew
their region evidence before case admission.

## Tracer concentration

`TracerRelaxivityCalibration` evaluates

```text
C = (1/T₁ − 1/T₁,baseline) / r₁
```

The relaxivity carries an explicit reciprocal concentration-time unit and is
converted to the calibration's chosen concentration/time system. Baseline and
contrast maps must share one affine and shape. Negative finite measurements are
retained as observations; `nonnegative_state_candidate` reports whether they can
directly initialize a physical nonnegative state.

## Forward transport

`NeurofluidTransportUnits` declares length, time, and concentration units and
derives diffusivity, velocity, volume-flow, and exchange-rate units. Parameter
arrays are numerical values in that exact system.

`NeurofluidTransportParameters` supplies cell porosity/diffusivity/velocity,
signed bulk boundary fluxes, inflow concentrations, removal rates, network
diffusion and flow, exchange coefficients, averaging radius, and terminal-
reservoir parameters. `NeurofluidTransportPlan` builds the circle-average
bulk/network transfer and generic mixed-dimensional runtime.

`neurofluid_diagnostics` aggregates bulk mass by semantic compartment and reports
total mass, external loss, exchange defect, and minimum concentration.

Derive diagnostic assignment indices from the validated
`bulk_mesh.region_evidence.cell_region_ids`, not zone names; authoritative
material adjacency is `region_evidence.adjacency_pairs`. The
`examples/neurofluid_transport.py` example generates
a native two-compartment mesh and advances a closed bulk/network/reservoir
system, reporting compartment inventory, interface adjacency, total-inventory
error, and exchange defect in an explicit millimeter/second/millimolar system.

## Flow

`TaylorHoodCSFFlowPlan` prepares the existing P2/P1 mixed Stokes operator with an
explicit pressure gauge. Tetrahedral H(div) families use
`form_element("tetrahedron", 2, order, family="trimmed" | "full",
twist="twisted", proxy="flux")`, with trimmed/full selecting RT/BDM spaces.
`HDivStokesPlan` uses the complete BDM₂/DG₁ pair: six degree-two normal-flux
moments per face, six interior moments, contravariant Piola mapping, and
discontinuous linear pressure.

`PVSNetworkFlowPlan` solves a variable-conductance graph pressure problem with
prescribed boundary pressures and returns edge volume flow plus interior branch
balance. `FlowTransportSchedule` retains sampled periodic flow and requires full
period coverage plus matching endpoint flow before accepting a cycle average.

The H(div) route assembles symmetric tangential Nitsche consistency, adjoint-
consistency, and penalty terms on interior and tangential no-slip exterior faces.
`HDivNormalBoundaryCondition` selects exterior faces by persistent global ID.
Non-negative hydraulic resistance contributes a positive-semidefinite face
operator; prescribed signed total flow (positive outward) adds an explicit
Lagrange-multiplier constraint to the residual. Every operator is separately
fingerprinted and stored with sparse routes.

## Image-space inversion

`ImageSpaceObservation` applies a prepared model-to-image operator before forming
a masked heteroscedastic Gaussian likelihood. No image-to-mesh projection error
is hidden from the objective.

`NeurofluidParameterSchema` declares names, bounds, and initial values.
`NeurofluidInverseProblem` builds a native `StateDesignProblem` and reports
weighted forward sensitivity singular values, numerical rank, condition number,
and linearized posterior covariance. Optimization success is not described as
identification when the sensitivity is rank deficient.

Topology discovery, registration, segmentation, and meshing remain outside the
differentiated fixed-route problem.

## External tools

Host providers for dcm2niix, Greedy, ANTs, FreeSurfer, FastSurfer, and SynthSeg
run argument vectors without a shell, verify explicit outputs, enforce source
rights, and create `ScientificArtifactEnvelope` records. Subprocess stdout and
stderr are discarded by default because controlled-tool logs may contain
sensitive paths; `retain_logs=True` is an explicit host-side choice. Provider
availability is resolved only when invoked. A `NeurofluidPipelineManifest`
records a topologically ordered stage DAG but does not replace an external
workflow engine.

External model/license terms remain authoritative. SVMTK is not added because its
GPLv3 boundary requires a separate legal decision.
