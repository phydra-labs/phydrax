# Native geophysics

`phydrax.applications.geophysics` now contains two complementary surfaces. The
atmosphere/ocean data, binding, assimilation, and learned-operator contracts remain
available and are documented in the geophysical data and assimilation guides. This
guide covers the solid-Earth, near-surface, subsurface, and planetary forward and
inverse models.

The modality packages reuse `phydrax.discretization`, `phydrax.linalg`,
`phydrax.observation`, `phydrax.uq`, and `phydrax.nonlinear`. They do not introduce a
second solver or posterior framework. Geometry, coordinate, clock, observation,
resource, and qualification identities stay explicit across composition.

## Capability matrix

“Analytic-qualified” below means that the repository suite exercises a stated
analytic limit, conservation law, adjoint pairing, or manufactured equilibrium. It is
not field validation, a released scientific claim, or evidence for an untested mesh,
material range, backend, precision, or acquisition.

| Family | Implemented model and geometry | Derivative and execution scope | Deliberate boundary |
| --- | --- | --- | --- |
| Electrical DC | `FinitePatchDCPlan`: 3D P1 tetrahedral conductivity with finite boundary-current patches, insulating remainder, and a projected constant gauge | Native projected solve, scalar/tensor conductivity JVP/VJP; analytic-qualified conservation, reciprocity, and gauge invariance | No point-source or contact-impedance interpretation |
| Complete electrodes | `CompleteElectrodeDCPlan`: 3D complete-electrode KKT system with finite positive contact impedance, shunting, electrode currents, gauge, power, and current defects | Native MINRES; fixed mesh, patches, and contact topology | No capacitive electrode interface or frequency dependence |
| Point and invariant-source DC | `PointElectrodeDCPlan`, `LineCurrentDCPlan`, and `TwoPointFiveDDCPlan`: 3D interior point sources with homogeneous-primary subtraction, 2D line currents, and Fourier-integrated 2.5D point currents | Native correction/KKT and complex GMRES solves; fixed source/receiver locations | No automatic infinite-domain truncation certificate or survey-geometry differentiation |
| Induced polarization | Passive Cole–Cole and Debye spectra, Debye memory evolution, and `SpectralIPPlan` | Causal fixed-parameter constitutive derivatives and complex native solves | No arbitrary empirical complex conductivity outside the declared passive sign convention |
| Borehole, marine, casing | Qualified trajectory sampling, borehole arrays, conductive marine layers, and mixed-dimensional formation/casing KKT systems | Fixed trajectory/topology and native block solve | No automatic well-path reconstruction or unresolved casing geometry inference |
| Gravity and magnetics | Free-space point/tetrahedral quadrature potential, acceleration, gravity gradient, induced/remanent magnetic dipoles, terrain, regional trend, and Fourier continuation | JAX field derivatives; downward continuation is explicitly amplification-limited | No singular receiver/source coincidence or unregularized downward continuation |
| Global potential fields | ICGEM gravity and Schmidt geomagnetic spherical-harmonic synthesis | Fixed coefficient normalization, degree, epoch, and reference radius | Static `gfc` gravity profile only; no hidden tide or frame conversion |
| Acoustic waves | Constant- and variable-density staggered Cartesian acoustics with prepared source/receiver stencils | Fixed-step replay, JVP/VJP, FWI, Gauss–Newton action, source variable projection, and RTM | Cartesian grids; 2D means line-source physics per unit thickness |
| Elastic waves | Isotropic stress/half-step velocity and anisotropic stiffness with passive standard-linear-solid memory | Fixed periodic grid and fixed relaxation spectrum; equilibrium and finiteness qualified | No curvilinear free-surface meshing in this backend |
| Wave boundaries/backends | Split Cartesian CPML, acoustic pressure-free surface, traction-free stress projection, generic mass/stiffness spectral-element stepping, and fixed AMR epoch transfer | CPML state is restart-complete; AMR selection is nondifferentiable while each epoch is differentiable | CPML evidence is profile-specific; the generic spectral-element backend requires caller-qualified operators |
| Traveltime and seismology | Graph eikonal relaxation, homogeneous event location, moment-tensor radiation, layered Love waves, homogeneous Rayleigh waves, ambient-noise correlation, and HVSR | Fixed graph paths or fixed mode identity only; path/mode margins expose derivative availability | No claim of a general unstructured fast-marching or full waveform surface-wave solver |
| Conductive EM | Layered TE/TM/MT recursions, digital Hankel transforms, 3D tetrahedral Nédélec H(curl) frequency EM, primary–secondary fields, and implicit quasistatic TDEM | Native GMRES/CG routes with PEC edge elimination and dissipated-power evidence | Fixed lowest-order tetrahedral H(curl); no automatic air-layer/domain construction |
| MT and GPR | Impedance, tipper, phase tensor, galvanic distortion, remote reference; passive Lorentz/Drude full-wave GPR with real transient sources and CPML | Fixed calibration matrices and streaming compatible-Maxwell state | GPR rejects harmonic/complex forcing; MT distortion and remote-reference assumptions remain explicit |
| Poromechanics and faults | Mixed Biot KKT stepping, phase-field damage, regularized rate/state friction, Coulomb contact, Maxwell relaxation, and quasi-dynamic earthquake-cycle roots | Fixed fault network/contact topology; derivatives only on successful smooth branches | No spontaneous topology change hidden inside a differentiated step |
| Deformation and geodynamics | GNSS, InSAR, tilt, and strain linear observations; operator-defined spherical-shell Stokes/thermal stepping | Native block solve with explicit momentum/thermal callable identities | Caller supplies qualified metric operators; this is not a mesh generator |
| Planetary | Reference-body coordinates, point plus centrifugal potential, radial P/S rays, and implicit radial thermal conduction | Fixed body, radial property nodes, phase, and frame | No ephemeris, relativistic correction, or 3D planetary tomography claim |

## Semantic foundation and interchange

Every physical kernel consumes Cartesian SI coordinates. `GeospatialContract` records
horizontal/vertical datum, axis, registration, seam, validity, and reference-frame
semantics. `CoordinateTransformPlan` executes one caller-supplied PROJ pipeline on the
host with network access disabled. The plan pins expected pyproj/PROJ versions and
the SHA-256 of `proj.db` plus every declared grid; observed resources become part of
the transform record. It never asks PROJ to choose an operation from source/target
CRS labels. Version/data pins do not content-pin native shared-library bytes, so the
host environment remains qualification scope.

`TimeReferenceContract` maps TAI, GPS, UTC, instrument, or source-relative samples
through explicit epoch, offset, and drift semantics. UTC requires a checksum-pinned
leap-second table. A TAI instant inside a positive leap second is rejected when the
chosen nominal UTC representation cannot encode it.

`BoreholeTrajectory` and prepared borehole sampling keep measured depth distinct from
Cartesian position. `ReferenceBodyContract` and `PlanetaryCoordinateContract` declare
ellipsoid, rotation, latitude, longitude, and body-fixed/inertial conventions.

Host adapters are bounded and provenance-bearing:

- SEG-Y revision 1 fixed IEEE and revision 2 fixed/variable IEEE profiles;
- genuine miniSEED 3 through `pymseed`, SAC and StationXML through ObsPy;
- GeoTIFF, CF-NetCDF, consolidated ZIP-Zarr, and qualified geospatial grids;
- electrical survey tables, EDI, EMTF XML, ICGEM `gfc`, geomagnetic coefficients,
  SINEX, RINEX, and LAS;
- optional PyGMT rendering.

Unsupported format semantics fail before normalized allocation. Exact source bytes,
resource limits, preserved fields, assumptions, and declared losses remain in adapter
reports. Installing a parser dependency does not qualify a scientific dataset.

Install `phydrax[geodesy]` for executed PROJ pipelines and
`phydrax[geophysics-io]` for the optional waveform, raster, geodetic, EM, and well
format runtimes. These extras are not imported by numerical kernels.

## Electrical models

`ElectricalSurvey` declares finite exterior patches, balanced current patterns,
balanced receiver functionals, and source selection. `FinitePatchDCPlan` solves

```text
-div(sigma grad(phi)) = 0
```

with insulating unelectroded boundary and a certified constant nullspace. Scalar,
cellwise scalar, constant tensor, and cellwise symmetric-positive-definite
conductivity are accepted in S/m.

Use a different truthful plan when the physics differs:

- `CompleteElectrodeDCPlan` for finite contact impedance and electrode shunting;
- `PointElectrodeDCPlan` for interior 3D point currents with an analytic homogeneous
  primary and discrete correction;
- `LineCurrentDCPlan` for translationally invariant line-current physics;
- `TwoPointFiveDDCPlan` for Fourier-integrated 3D point currents on a 2D mesh;
- `SpectralIPPlan` for passive frequency-dependent conductivity.

No alias maps these models onto the narrower finite-patch contract.

## Potential fields

`GravityQuadratureSource.from_tetrahedra` creates a fixed high-order volume quadrature.
`FreeSpaceGravityPlan` returns positive geopotential, its acceleration gradient, and
the acceleration as the spatial gradient of that geopotential, matching the
spherical/planetary convention. `FreeSpaceMagneticPlan` combines induced and remanent
magnetization. Receiver/source separation is checked explicitly.

`TerrainCorrectionPlan`, `RegionalTrendPlan`, and `FourierContinuationPlan` are separate
processing operations. Downward continuation clamps the spectral exponent before
exponentiation and records that regularization. Spherical-harmonic plans consume only
qualified coefficient models with explicit normalization and reference radius.

## Seismic forward, imaging, and source workflows

`AcousticGrid`, `SeismicAcquisition`, and prepared sampling stencils define the fixed
Cartesian support. `ConstantDensityAcousticPlan` and `VariableDensityAcousticPlan`
share physical source-rate semantics and restart-compatible pressure/velocity state.
`PeriodicIsotropicElasticWavePlan` adds isotropic elastic stress/velocity propagation;
`PeriodicAnisotropicViscoelasticPlan` adds full stiffness matrices and causal SLS
memory.

`CartesianCPML` stores all memory coefficients and state. A physical free surface
cannot overlap a CPML width. `TractionFreeBoundary` projects the appropriate stress
components. `SpectralElementWavePlan` accepts already-qualified native mass,
stiffness, and inverse-mass actions rather than inventing an element realization.

`AcousticWaveformInversionPlan` uses observation covariance actions directly. It
provides the objective and gradient, a matrix-free Gauss–Newton action when whitening
exists, and RTM as the negative slowness-squared gradient. `AcousticSourceProjectionPlan`
eliminates linear source-basis coefficients through `LinearNuisancePlan`; source
coefficients are not silently folded into wavespeed.

## Electromagnetics

`TetrahedralNedelecSpace` supplies oriented lowest-order edge degrees of freedom,
covariant Piola mapping, mass and curl–curl actions, and discrete grad–curl–div exact
sequence checks. `FrequencyDomainEMPlan` uses the conductive exp(-i omega t)
convention and removes PEC boundary edges. Primary–secondary forcing is exactly the
background/operator difference applied to a background solution.

`ImplicitTimeDomainEMPlan` advances quasistatic conductive diffusion. Full-wave GPR
instead wraps the compatible-Maxwell displacement-current solver, requires passive
dispersive ADE material, CPML, sources, observers, and explicit real transient
envelopes. The two routes are not interchangeable.

## Porous, reactive, and deformation composition

The porous-media package adds phase/component/energy inventories, Rachford–Rice phase
appearance, compositional K-value callbacks, Brooks–Corey hysteresis, dynamic
capillarity, freeze/thaw enthalpy, vapor equilibrium, atmospheric exchange,
unstructured shallow water, completions, and rate/pressure well switching. See the
[porous-media guide](guides_porous_media.md) for the full conservation contracts.

`MixedBiotPoromechanicsPlan` accepts native elasticity, coupling, storage, and flow
operators. Fault and damage laws expose active-set and derivative availability rather
than differentiating through a hidden branch. Geodetic observations are independent
linear operators that can be combined with those state models.

## Observation, inference, and monitoring

Observation covariance actions now share one contract across `phydrax.observation`
and `phydrax.uq`: diagonal, Cholesky, low-rank-plus-diagonal, precision-operator,
Kronecker, and circulant routes expose solve, quadratic, log-determinant, and whitening
only where mathematically available. Robust contamination and censored Gaussian
likelihoods remain normalized; Huber is explicitly an unnormalized objective.

Spatial/temporal priors include graph/H1 metrics, total variation, temporal
differences, cross-gradient structure, and certified operator-defined SPDE precision.
Stochastic A-, D-, and E-optimal design uses fixed random probes and operator actions
without materializing a parameter-space identity. Its maximization score is negative
inverse trace, log-determinant, or minimum Ritz value. These estimators require a
Euclidean `ArraySpace`; callers transform non-Euclidean metric coordinates explicitly.

`MultimodalJointInferencePlan` combines independently identified modality terms and
explicit structural, petrophysical-discrepancy, or shared-interface couplings.
`MatrixFreeMAPPlan`, `EnsembleKalmanInversionPlan`, and `PCNSampler` reuse native
linear algebra and JAX random keys. `SequentialMonitoringPlan` records every epoch,
observation, geometry, acquisition, dynamics identity, predictor identity, calibration
drift, and key.

## Distribution, topology, resources, and restart

`DistributedHaloPlan` builds padded owned/halo layouts and colored peer permutations.
`DistributedLocalOperator` executes only explicitly supplied local forward and
transpose actions with `lax.ppermute`; it does not imply that every modality plan is
sharded. Serial references and transpose pairing support qualification before a
partitioned run.

`TopologyEpochTransition` consumes a fixed conservative `FieldTransfer` with separate
dual pullback and Hilbert adjoint. Topology choice is nondifferentiable; derivatives
remain available only within a fixed epoch.

`GeophysicalResourceEstimate` and `GeophysicalResourcePolicy` reject declared memory,
source, observation, or step limits before execution. `GeophysicalCheckpointPlan`
content-addresses plan, geometry, observation, partition, physical state, auxiliary
state, inference state, clock, accepted step, source position, random key, and topology
epoch. Identity mismatch fails closed.

## Qualification and benchmarks

Analytic and manufactured tests cover exact sequences, conservation, adjoint pairing,
passivity, homogeneous fields, spherical monopoles, graph traveltimes, waveform
inversion at the generating model, reaction equilibrium, native inference, restart,
and policy refusals.

`GeophysicalReferenceRecipe` governs an external-oracle or field comparison with an
exact `ReferenceArtifactManifest`, coordinate/time identities, masks, tolerances,
decoded-sample bound, and uncertainty. Field evidence additionally requires
campaign-start and campaign-observation IDs. The reference runner verifies the NPZ
size, digest, member names, and uncompressed allocation bounds before loading arrays;
it never downloads the `source_locator`.

Representative commands:

```text
PYTHONPATH=. python examples/geophysics/dc_resistivity.py
PYTHONPATH=. python examples/geophysics/porous_infiltration.py
PYTHONPATH=. python examples/geophysics/hydrogeophysical_inference.py
PYTHONPATH=. python examples/geophysics/acoustic_survey.py
PYTHONPATH=. python benchmarks/dc_resistivity.py --warmup 1 --repeats 5
PYTHONPATH=. python benchmarks/porous_media.py --warmup 1 --repeats 5
PYTHONPATH=. python benchmarks/acoustic_propagation.py --warmup 1 --repeats 5
XLA_FLAGS=--xla_force_host_platform_device_count=4 PYTHONPATH=. \
  python benchmarks/geophysics_production.py --partitions 1 2 4
```

The production benchmark records one-, two-, and four-partition forward/transpose
errors, pairing residuals, communication storage, warm timing, exact restart, and
expected refusal of mismatched checkpoints, resource overruns, and unavailable device
counts. Timing is environment-specific; correctness claims come from the recorded
residuals and identities.

The checked-in `benchmarks/geophysics_production.json` records the exercised CPU
environment and residuals; reruns replace timing rather than defining a portable
performance threshold.
