# Neurofluid modeling

`phydrax.applications.neurofluid` composes shared imaging, compartment meshing,
metric networks, mixed-dimensional transport, finite-element flow, observations,
and PDE-constrained inversion. Generic numerical machinery remains outside the
application package.

## Case construction

`NeurofluidCase` binds one pseudonymous case revision to:

- a revision-bound `CompartmentMeshingSource` carrying the categorical `LabelVolume`,
  exact `CompartmentComplex`, occupied-voxel interpretation, outer surface, and
  oriented source interfaces;
- an accepted tetrahedral `CellMeshingResult` from the unified native meshing API;
- a prepared PVS metric network;
- optional concentration and diffusion-tensor images.

Create the bulk mesh from `CompartmentMeshingSource` with `NativeMeshingProvider`
using the explicit `image_material_tetrahedral` route. Supply a `VolumeMeshingSpec`
with a complete source-bound facet scope, size control, and a `RegionSeed` for each
declared source region. The provider uses `NativeMeshingOptions` with the selected route
and a native volume schedule; execute the plan with the source's physical coordinate
contract. Region seed material IDs and roles bind to source region IDs. Result cell
associations, material zones, and interface patches retain those bindings. Zone
display names are not used to infer compartment identity.

Construct the case as `NeurofluidCase(case_id, compartment_source, bulk_mesh, network)`.
The case verifies the exact segmentation/source revision, every cell's source-region
and material assignment, interface endpoint orientation, and one physical coordinate
contract for the source, bulk mesh, network, and optional images.
The native `CellMeshingResult` is consumed directly; there is no compartment-
specific result wrapper or mandatory external meshing engine. It must retain passed
volume-coverage certification; the case compares its dense per-cell region table with
source associations and material-zone scopes.

An explicitly extracted label-field surface is a distinct source interpretation; do
not treat its triangles as a signed implicit field. fTetWild remains available through
its explicit optional provider for independent comparison workflows.

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
