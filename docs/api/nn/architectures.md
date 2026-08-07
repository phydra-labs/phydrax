# Architectures

Common end-to-end model families (dense, separable, basis-edge, and complex-valued).

!!! note
    Key notes:

    - `MLP` is a standard feed-forward network with optional residual connection.
    - `ModifiedMLP` reuses two persistent input encoders in every hidden-layer
      gate to improve information flow through deep coordinate networks.
    - `KAN` replaces node activations with typed edge functions. It defaults to
      global orthogonal polynomials and supports fixed or trainable B-spline grids.
    - `FeynmaNN` builds complex hidden states with a sum-over-paths block.
    - `MLP`, `ModifiedMLP`, `KAN`, `FeynmaNN`, and `FNO` support `scan=True`
      to use a scan-over-depth execution path when topology is compatible.
    - `scan=True` is primarily a compile-time optimization for deeper repeated blocks.

::: phydrax.nn.MLP
    options:
        members:
            - __init__
            - __call__

---
::: phydrax.nn.ModifiedMLP
    options:
        members:
            - __init__
            - __call__

---


::: phydrax.nn.KAN
    options:
        members:
            - __init__
            - __call__

`edge_basis` accepts one shared basis or one basis per layer. By default the
spline basis uses an open-uniform grid on `[-1, 1]`, Greville-abscissa identity
initialization, compact support, and a basis-specific Sobolev regularizer:

```python
import phydrax as phx

basis = phx.nn.BSplineEdgeBasis(degree=3, num_intervals=8)
model = phx.nn.KAN(
    in_size=2,
    out_size="scalar",
    width_size=32,
    depth=3,
    edge_basis=basis,
    scan=True,
)
```

An explicit nonuniform fixed grid uses the same basis contract:

```python
grid = phx.nn.BSplineGrid(
    [-1.0, -1.0, -1.0, -0.6, -0.1, 0.45, 1.0, 1.0, 1.0],
    2,
)
basis = phx.nn.BSplineEdgeBasis(grid=grid, regularization_order=1)
```

Knot arrays remain fixed under ordinary optimizer partitioning. A supplied grid
determines the degree and coefficient count; `num_intervals` is only valid for
the open-uniform convenience construction. Sobolev quadrature follows every
positive knot span, including nonuniform and repeated-knot grids.

One independent fixed grid per layer input channel is realized without changing
the dense coefficient-array contract:

```python
basis = phx.nn.BSplineEdgeBasis(
    degree=3,
    num_intervals=8,
    per_input=True,
)
```

Every outgoing edge associated with one input channel shares that channel's
grid. `BSplineGridBank` requires a homogeneous degree, knot count, positive-span
count, and active interval, so repeated hidden layers remain scan-compatible.
Use `KANGridAdaptationPlan(per_input=True)` to estimate each input grid from its
own normalized activation marginal. The returned report aligns every transferred
grid with `paths` and `input_indices`; `None` denotes one layer-shared transfer.
Adaptation preserves coefficient count, projects learned edge functions into the
new basis, returns a new model, and never mutates the source model.

Trainable fixed-count knots are an explicit alternative:

```python
grid = phx.nn.TrainableBSplineGrid.open_uniform(
    3,
    8,
    minimum_span=1e-3,
)
basis = phx.nn.BSplineEdgeBasis(
    grid=grid,
    knot_entropy_weight=1e-4,
    knot_neighbor_weight=1e-4,
)
```

`TrainableBSplineGrid` keeps both endpoints and their open-knot multiplicities
fixed. A softmax allocation of live span logits preserves strict ordering,
fixed coefficient count, and the configured physical minimum span. Span
routing is discrete and receives no tangent; basis arithmetic, knot
denominators, dynamic Sobolev quadrature, JVPs, VJPs, and Hessians retain knot
tangents within the selected span. Entropy and neighboring log-span penalties
discourage collapsed or violently alternating grids.

Trainable grid logits are ordinary optimizer parameters. Coefficients and logits
may be updated jointly, but alternating coefficient and lower-rate knot phases
are safer for stiff PDE losses. `adapt_kan_grids` intentionally rejects
trainable grids: quantile regridding is a pure between-phase structural
operation, whereas trainable knots are same-structure optimizer state.
Trainable per-input grid banks are not supported.

Fixed polynomial-spline edges can also change coefficient count between optimizer
phases. Refinement consumes nonnegative per-positive-span indicators keyed by
`(layer, output, input)` and applies one global knot budget:

```python
import jax.numpy as jnp

refined, refinement = phx.nn.refine_kan_edges(
    model,
    {(1, 3, 2): jnp.asarray([0.1, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])},
    budget=1,
)
coarsened, coarsening = phx.nn.coarsen_kan_edges(
    refined,
    {(1, 3, 2): 1e-6},
    budget=1,
)
```

`refine_kan_edges` ranks indicators globally, inserts the midpoint of each
selected positive span, and transfers the represented function exactly by
Boehm knot insertion. `coarsen_kan_edges` tries each removable interior knot
and accepts the best projection only when its coefficient-specific absolute
L2 error certificate is within the requested tolerance. Both operations are
pure and return aligned `KANCapacityAdaptationReport` diagnostics.

Capacity is allocated only to selected scalar edges. Internally,
`KANEdgeBlock` groups edges with identical grids into homogeneous dense arrays;
this avoids padding every edge to the largest coefficient count. Repeated
hidden layers remain scan-compatible when their resulting block layouts match.
Otherwise `scan=True` safely falls back to the ordinary layer loop. Structural
adaptation invalidates shape-dependent optimizer state, so perform it between
training phases and initialize fresh optimizer state for the returned model.
The structural API intentionally rejects trainable-knot, rational-spline, and
non-spline edges.

Positive-weight rational B-spline edges are available when a quotient is
materially more parameter-efficient than an ordinary spline:

```python
basis = phx.nn.RationalBSplineEdgeBasis(
    degree=3,
    num_intervals=5,
    maximum_log_weight=4.0,
)
model = phx.nn.KAN(
    in_size=2,
    out_size="scalar",
    edge_basis=basis,
)
```

Each edge owns `RationalBSplineEdgeParameters(control_values,
raw_log_weights)`. Bounded log-weights are centered along the coefficient axis
before exponentiation, removing the otherwise unidentifiable global weight
scale while guaranteeing positive finite denominators. Zero raw log-weights
reduce exactly to an ordinary polynomial B-spline, including identity
initialization.

The basis regularizes the Sobolev energy of the rational output itself, not the
numerator and denominator separately. Optional log-weight magnitude,
neighbor-variation, and denominator-floor terms control identifiability and
conditioning. Value, parameter-gradient, Jacobian, Hessian, and homogeneous
scan paths accept the structured edge-parameter PyTree. Rational edges support
fixed shared grids and fixed per-input grid banks; trainable knots and explicit
quantile regridding remain separate polynomial-spline policies.

With `use_tanh=False`, values are inclusively clamped to the canonical interval:
exact endpoints retain the one-sided edge derivative, while strictly exterior
inputs have zero derivative.

---

::: phydrax.nn.AbstractEdgeBasis

---

::: phydrax.nn.OrthogonalPolynomialEdgeBasis
    options:
        members:
            - __init__

---

::: phydrax.nn.BSplineGrid
    options:
        members:
            - __init__

---

::: phydrax.nn.BSplineGridBank

---

::: phydrax.nn.TrainableBSplineGrid

---

::: phydrax.nn.KANGridAdaptationPlan

---

::: phydrax.nn.KANGridAdaptationReport

---

::: phydrax.nn.adapt_kan_grids

---
::: phydrax.nn.KANEdgeBlock

---

::: phydrax.nn.KANCapacityAdaptationReport

---

::: phydrax.nn.refine_kan_edges

---

::: phydrax.nn.coarsen_kan_edges

---

::: phydrax.nn.RationalBSplineEdgeParameters

---

::: phydrax.nn.RationalBSplineEdgeBasis

---

::: phydrax.nn.BSplineEdgeBasis
    options:
        members:
            - __init__

---

::: phydrax.nn.FeynmaNN
    options:
        members:
            - __init__
            - __call__

---

## Neural operators

PhydraX neural operators consume a canonical `OperatorBatch`, not an unlabelled
tensor convention. A batch separates:

- one or more source functions (`FunctionSamples`);
- source coordinates, quadrature weights, and masks;
- an independent query discretization;
- named physical-case axes.

Tensor grids use `OperatorAxis`; point clouds use explicit coordinates. This keeps
quadrature, periodicity, basis identity, masks, and source/query separation intact
through model dispatch. `Domain.Model(...)` builds this view automatically for
coord-separable domain batches. Models can also be called directly with an
`OperatorBatch`.

For a batch with `case_shape = C` and one source with `sample_shape = S`,
scalar source values have shape `C + S`; vector fields add one final channel
axis, `C + S + (channels,)`. Tensor grids store shared one-dimensional
`OperatorAxis` objects and have `S = (n_1, ..., n_d)`. Point clouds instead
store coordinates with shape `S + (coord_dim,)` when shared, or
`C + S + (coord_dim,)` when geometry varies by case. Quadrature and masks have
shape `S` or `C + S`. Queries normally set `values=None` and follow the same
geometry rules. `case_axes` names every dimension of `C`; it never includes
sample or channel axes.

Masks are mathematical metadata, not merely a padding optimization: a false
source entry contributes zero measure, and a false query entry is restored as
zero in model output. On irregular meshes, supply physical cell/volume/boundary
quadrature rather than relying on the point-cloud fallback of unit weights.
Measure-aware attention and integral operators otherwise approximate a
counting-measure operator whose result changes with sampling density.
`GreenKernelOperator` is stricter and rejects missing physical quadrature on
both its volume and boundary sources.

---

::: phydrax.nn.FunctionSamples

---

::: phydrax.nn.OperatorBatch

### Reusable state, multi-query batches, and physical branches

`AbstractEncodedOperatorModel` separates source encoding from query decoding.
The encoded state can be reused across independent query sets or streamed
chunks, but concrete models own the state's schema.

`OperatorBatch.queries` associates one source mapping with any number of named
query discretizations. Each query retains its own coordinates, quadrature, mask,
and topology. `OperatorPrediction.fields` binds each named output to its query by
`OperatorFieldBatch.query_name`; no positional multi-query adapter or implicit
model dispatch is involved.

Context strategies make fixed-capacity state construction reusable:

- `LearnedTokenContext` creates an abstract trainable token bank;
- `PooledGeometryContext` performs deterministic measure-weighted pooling;
- `SampledAnchorContext` retains selected physical locations, masks, and
  measures.

`OperatorBranchGraph`, `OperatorBranchSpec`, and `BranchInteractionSpec` define
typed conditioning/prediction branches and a static directed interaction
schedule. `apply_branch_interactions` applies each stage synchronously, so
opposite directed edges read the same pre-stage state.

`DifferentialFieldDecoder` wraps a pointwise coordinate model and derives
gradients, curl, rotated gradients, or symmetric gradients.
`DifferentialNormalization` converts normalized Jacobians back to physical
units; `LinearDifferentialTransform` exposes an explicit validated coefficient
map. This decoder is pointwise and is not an `OperatorBatch` query decoder.

::: phydrax.nn.AbstractEncodedOperatorModel

---

::: phydrax.nn.LearnedTokenContext

---

::: phydrax.nn.PooledGeometryContext

---

::: phydrax.nn.SampledAnchorContext

---

::: phydrax.nn.OperatorBranchSpec

---

::: phydrax.nn.BranchInteractionSpec

---

::: phydrax.nn.OperatorBranchGraph

---

::: phydrax.nn.apply_branch_interactions

---

::: phydrax.nn.DifferentialNormalization

---

::: phydrax.nn.LinearDifferentialTransform

---

::: phydrax.nn.DifferentialFieldDecoder

### Production sampling and streamed inference

Large operator datasets need not be materialized as one `OperatorBatch`.
`CallbackOperatorCaseSource` separates lightweight geometry metadata from
selective case reads. `AnchorQuerySamplingPolicy` chooses fixed, random, or
weighted source anchors and query points, while `read_operator_case_batch`
preserves case-axis names, masks, coordinates, and the selected physical
measure.

For inference on query sets larger than device memory,
`ArrayOperatorQuerySource` and the callback query-source interface expose
fixed-capacity chunks. `decode_query_chunks` encodes an
`AbstractEncodedOperatorModel` once, pads only the final query chunk, compiles a
shape-stable decoder, and writes through an `OperatorPredictionSink`.
`ArrayPredictionSink` collects an in-memory result; `NpyPredictionSink` writes
the same ordered chunks incrementally. These APIs are re-exported from
`phydrax.nn`; their protocol types remain available from
`phydrax.nn.operator_training`.

::: phydrax.nn.AnchorQuerySamplingPolicy

---

::: phydrax.nn.CallbackOperatorCaseSource

---

::: phydrax.nn.read_operator_case_batch

---

::: phydrax.nn.ArrayOperatorQuerySource

---

::: phydrax.nn.ArrayPredictionSink

---

::: phydrax.nn.decode_query_chunks

### Fourier and basis operators

`FNO` is the canonical N-dimensional implementation. Its required `n_modes`
sequence selects the spatial dimensionality and retained modes per axis. It includes:

- learned positive and negative mode blocks in every full-FFT dimension;
- native leading case axes;
- coordinate embeddings, residual blocks, channel normalization, and optional
  nonperiodic padding;
- dense, CP, and Tucker spectral weights;
- active-mode curricula and multiscale spectral branches;
- loop/scan execution parity.

`HOFNO` is an experimental-tier periodic-grid variant for nonlinear maps with
explicit polynomial Fourier-mode interactions. Each block projects the hidden
field into `interaction_order` factors, multiplies them pointwise, applies a
depthwise or dense Fourier multiplier, and then applies a pre-RMSNorm
feed-forward residual update. The default `aliasing="dealiased"` path
Fourier-oversamples every factor before multiplication and handles even-grid
Nyquist coefficients explicitly; `"collocation"` is an intentional
paper-style aliasing ablation. `factor_bias=False` makes the interaction
homogeneous of exactly the configured degree, while `factor_bias=True` permits
lower-degree terms.

HOFNO deliberately has a narrower contract than FNO: one all-valid source,
coincident source/query discretizations, uniformly spaced periodic tensor axes,
and one to three spatial dimensions. Nonperiodic domain padding is rejected
rather than conflated with nonlinear de-aliasing. Coordinate embeddings remain
an explicit opt-in. `interaction_order=1` is the controlled backbone for
order-ablation studies, not an alias for the stable `FNO` architecture.

`BasisSpectralConvND` provides quadrature-projected Fourier, sine, cosine, and
Legendre policies for nonperiodic or nonuniform tensor axes.

Periodic grid reconstruction is shared across spectral architectures and public
array-level evaluation. `spectral_resample` transfers to an aligned or
period-shifted uniform grid. `sample_fourier_grid` evaluates paired arbitrary
coordinates with an exact direct backend or an explicitly tolerance-controlled
NUFFT backend. Both preserve channel-last fields and leading case axes; the
point sampler also accepts physical uniform axis nodes and periods.

::: phydrax.nn.spectral_resample

---

::: phydrax.nn.sample_fourier_grid

---

::: phydrax.nn.FNO
    options:
        members:
            - __init__
            - __call__
            - with_active_modes

---
::: phydrax.nn.HOFNO
    options:
        members:
            - __init__
            - __call__
            - with_active_modes

---



::: phydrax.nn.SpectralConvND
    options:
        members:
            - __init__
            - __call__
            - with_active_modes

---

::: phydrax.nn.BasisSpectralConvND
    options:
        members:
            - __init__
            - __call__


#### Implicit, axial, wavelet, and manifold variants

`IFNO` reuses one learned Fourier update for a statically bounded number of
fixed-point iterations. `evaluate_with_diagnostics` returns an
`IFNOConvergence` record for the final update; the tolerance reports convergence
but does not terminate compiled execution early. `AxialFactorizedFNO` instead
applies learned one-axis spectral transforms sequentially. Both retain FNO's
tensor-grid and coincident source/query requirements. Axial factorization lowers
the multidimensional transform cost but omits simultaneous cross-axis spectral
mixing within one transform.

`WaveletNeuralOperator` (WNO) uses an exactly reconstructing multiresolution
filter bank on one fixed tensor shape. `MultiwaveletOperator` (MWT) is the
one-dimensional Alpert polynomial variant. They require identical source and
query tensor axes and are not arbitrary-point decoders; the constructor's
`spatial_shape` or `num_points` is part of the model contract.

`ManifoldSpectralOperator` projects through a supplied Laplace eigenbasis.
Source and optional target `SpectralDiscretization` plans must represent a
fixed/aligned manifold basis. A target plan permits a pre-aligned
cross-discretization, not arbitrary query coordinates or independently
remeshed manifolds with unresolved eigenbasis alignment.

`SpectralDiscretization.from_stiffness(K, M, ...)` uses the finite-element
convention \(K\succeq0\), \(M\succ0\) and solves \(K v=\lambda M v\).
`from_triangle_mesh(...)` assembles cotangent stiffness and lumped mass as
sparse matrices and computes only the requested low modes for large meshes;
small or nearly full spectra use a dense solve. Repeated eigenvalues are grouped
as one eigenspace for basis-gauge-safe spectral mixing.

::: phydrax.nn.IFNO
    options:
        members:
            - __init__
            - __call__
            - evaluate_with_diagnostics

---

::: phydrax.nn.IFNOConvergence

---

::: phydrax.nn.AxialFactorizedFNO
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.WaveletNeuralOperator
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.MultiwaveletOperator
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.SpectralDiscretization

---

::: phydrax.nn.ManifoldSpectralOperator
    options:
        members:
            - __init__
            - __call__

### Branch–trunk operators

`DeepONet` supports fixed-vector branches, quadrature-aware variable-sensor
branches, independent point/grid queries, query chunking, trainable output bias,
POD bases, and multiple functional inputs. A branch mapping plus
`fusion="product"` is the MIONet configuration; `sum` and learned `concat` reuse
the same substrate.

::: phydrax.nn.DeepONet
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.IntegralBranchEncoder
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.PODBasis
    options:
        members:
            - __init__
            - evaluate

### Local, temporal, and attention operators

- `LocalIntegralOperator` evaluates a quadrature-weighted coordinate kernel over
  physical-radius neighborhoods.
- `LocalDifferentialOperator` uses constant-preserving source differences and a
  physical-radius scale.
- `LocalGlobalOperator` composes local and global paths.
- `LaplaceTemporalOperator` implements stable causal pole–residue dynamics with
  constrained negative-real-part poles and explicit conjugate reconstruction.
- `OperatorAttention`, `SliceAttention`, `CodomainAttention`, and
  `AxialOperatorAttention` cover continuum self/cross attention, Transolver-style
  slices, variable physical fields, and tensor-axis factorization.

::: phydrax.nn.LocalIntegralOperator
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.LocalDifferentialOperator
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.LaplaceTemporalOperator
    options:
        members:
            - __init__
            - __call__
            - poles

---

::: phydrax.nn.OperatorAttention
    options:
        members:
            - __init__
            - __call__
            - cross

### Grid and manifold operators

`CNO` uses oversampled nonlinearities and band-limited resampling to prevent
ordinary CNN aliasing. `UNO` adds a U-shaped multiresolution topology over the
same representation-aware blocks. `SFNO` uses true spherical harmonics, shares
weights by spherical degree, and is not a planar FFT over an equirectangular
image.

::: phydrax.nn.CNO
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.UNO
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.SFNO
    options:
        members:
            - __init__
            - __call__

### Geometry operators

`GINO`, `GeometryInformedFlower`, `RIGNO`, and `GAOT` share one
encode–process–decode contract:

1. each named source `FunctionSamples` is transferred to a case-local latent
   geometry;
2. a processor acts only on that latent representation;
3. a second transfer evaluates the result at the independent query coordinates.

The transfer graph is rebuilt inside JAX for every physical case. Source values,
coordinates, masks, and quadrature stay coupled; no edge or message crosses a
case boundary. Query masks are restored exactly at the output. Multiple source
functions may have unrelated point clouds: pass a mapping to `in_channels`, and
the model gives every source its own encoder before learned latent fusion.

`GraphKernelTransfer` implements a learned graph integral. With
`reduction="integral"`, every message is multiplied by the source quadrature
measure. `GraphAttentionTransfer` instead adds log measure to its neighborhood
softmax. `MultiscaleGraphTransfer` evaluates distinct physical radii, augments
them with geometry moments, and fuses the scales by concatenation or learned
gates. A radius is a dimensional coordinate quantity; `coordinate_scale` only
normalizes features presented to learned kernels.

- `GINO` transfers irregular data to a tensor latent grid, applies an
  N-dimensional `FNO`, and decodes at arbitrary query points. Use
  `bounds_policy="global"` with fixed training-domain bounds for strict
  cross-case comparability; use `"case_bbox"` only when a case-relative latent
  coordinate system is intended.
- `GeometryInformedFlower` uses the same irregular-to-tensor and
  tensor-to-query transfers around a multiscale `Flower` processor. A named
  occupancy or signed-distance source can define hard latent support; named
  case conditions and Flower diagnostics propagate through the wrapper.
  Optional end-to-end conservation uses the physical source and query
  quadrature. The model treats samples as weighted ambient point sets and does
  not consume mesh connectivity.
- `RIGNO` selects a fixed-size regional point set independently for every case,
  rebuilds its regional graph dynamically, and applies measure-normalized
  residual message passing. Farthest-point sampling is deterministic and masks
  padded physical points.
- `GAOT` uses MAGNO-style multiscale graph attention on both transfers and a
  measure-aware patch transformer on a 2D or 3D tensor latent grid. Every latent
  dimension must be divisible by its corresponding patch size.

All four classes require an `OperatorBatch`; they deliberately reject an
unlabelled array whose geometry, source identity, mask, and measure would be
ambiguous. Uniform measure is never inferred unless
`assume_uniform_measure=True` is requested explicitly.

The operator benchmark registry exposes controlled
`GeometryInformedFlower` configurations rather than separate public model
classes:

- `geometry_informed_flower`: resolution-consistent transitions;
- `geometry_informed_flower_learned`: learned stride-two transitions;
- `geometry_informed_flower_support`: resolution-consistent transitions plus
  explicit hard latent support;
- `geometry_informed_flower_support_conservative`: the supported configuration
  plus physical source-to-query integral projection.

They share transfer operators, latent bounds, channel widths, query decoding,
and optimization policy. The learned-transition configuration necessarily has
additional trainable parameters; capacity-matched studies report that
difference explicitly. Sourcewise benchmark normalization leaves occupancy/SDF
support values in physical units. For a conserved scalar, centering is adjusted
by the source and query measures so inverse normalization cannot break the
conservation projection.

Five-seed, capacity-matched decision runs on conservative ring transport and
deformed elliptic geometry support the resolution-consistent transition as the
default. Learned transitions improve one resolution-transfer regime but lose
overall robustness, especially under missing sensors. Hard support improves
mask/dropout behavior at a small nominal-accuracy and runtime cost. Conservation
projection is exact when the observed source mass is trustworthy, but is not a
generic remedy for corrupted or missing source mass. These are opt-in modeling
choices; neither `GINO` nor `GeometryInformedFlower` is promoted beyond research
tier by the current evidence.

::: phydrax.nn.GINO
    options:
        members:
            - __init__
            - __call__

---
::: phydrax.nn.GeometryInformedFlower
    options:
        members:
            - __init__
            - __call__
            - evaluate_with_diagnostics

---


::: phydrax.nn.RIGNO
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.GAOT
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.GraphKernelTransfer

---

::: phydrax.nn.GraphAttentionTransfer

---

::: phydrax.nn.MultiscaleGraphTransfer

---

::: phydrax.nn.TensorGridLatentGeometry

---

::: phydrax.nn.RegionalPointLatentGeometry

---

::: phydrax.nn.GeometryOperatorDiagnostics

---

::: phydrax.nn.RegionalGraphProcessor

---

::: phydrax.nn.OperatorTransformerProcessor

Transolver-style `SliceAttention` remains a separate layer-level configuration:
physical points are assigned to a fixed number of learned slices, global
attention acts between slices, and the result is desliced. It is not an alias
for any of the geometry architectures above.

### Metric cochain operators

`CochainNeuralOperator` acts on typed discrete differential forms over one
`CochainComplexIR`. Each `OperatorFieldSpec` declares its cochain degree,
primal/dual side, orientation law, and sampling semantics through
`phydrax.graph.CochainFieldSpec`. `function_samples_from_cochain` then binds values,
physical Hodge-star measures, boundary masks, coordinates, and the shared
cell-complex topology into canonical `FunctionSamples`.

`TopologicalCochainBlock` communicates only through declared metric DEC routes:
self, exterior derivative, codifferential, lower/upper Hodge-Laplacian, and an
optional exact harmonic projection. Trainable maps act on channels at cells;
incidence, Hodge stars, boundary policy, and harmonic bases remain immutable
runtime data. Signed cochains transform under cell reorientation, while
zero-form point values are orientation-invariant. All source and target fields
must therefore share one complex relation; this is not an arbitrary point-cloud
operator.

The default route set excludes the dense harmonic projection. Enable
`TopologicalRouteConfig(harmonic=True)` only after attaching a
`compute_harmonic_subspace(...)` result to the complex. Absolute and relative
boundary policies select different closed subcomplexes and must match between
sample construction and model execution.

::: phydrax.graph.CochainFieldSpec

---

::: phydrax.nn.function_samples_from_cochain

---

::: phydrax.nn.TopologicalRouteConfig

---

::: phydrax.nn.TopologicalCochainBlock

---

::: phydrax.nn.CochainNeuralOperator

---

### Latent-token, heterogeneous, and equivariant operators

`MeasureAwareAttention` is the common continuum-attention primitive. Source
quadrature enters attention normalization, source masks remove padded samples,
and query masks remove padded outputs. Physical measures are required for
sampling-density invariance; an omitted point-cloud measure becomes unit
counting weights and changes the represented operator.

`UPT` compresses one source function to a fixed learned token set and decodes an
independent query set. `ABUPT` extends that pattern to an `OperatorBranchGraph`
whose typed conditioning/prediction branches have fixed anchor capacities and
declared cross-branch interactions. Both support arbitrary query coordinates,
but the finite token/anchor set remains an information bottleneck.

`Transolver` turns weighted physical points into learned slice tokens and then
decodes arbitrary queries. `slice_top_k=1` is the hard, non-overlapping
Transolver configuration. Values greater than one use normalized overlapping
memberships and are registered as the `TransolverPlusPlus` configuration;
`slice_top_k=num_slices` is fully soft overlap. This configuration distinction
does not imply checkpoint or benchmark parity with an external Transolver++
implementation.

`GNOT` requires a named `in_channels` mapping and gives each heterogeneous
source its own value/coordinate encoder, measure-aware query cross-attention,
and learned fusion gate. The query may carry channel-last covariates. Source and
query masks and physical measures must describe the actual discretizations.

`CoDANO` declares multiphysics roles with `OperatorFieldSpec`, transfers fields
to a common tensor latent grid, alternates spatial spectral mixing with codomain
attention, and decodes each declared target independently. `EqGINO` is the
three-dimensional O(3)-equivariant geometry configuration: inputs and outputs
must use explicit `O3Representation` contracts and source quadrature. Equivariance
does not remove its finite-radius or representation-content assumptions.

::: phydrax.nn.MeasureAwareAttention
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.UPT
    options:
        members:
            - __init__
            - __call__
            - encode_inputs
            - decode_query

---

::: phydrax.nn.ABUPT
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.Transolver
    options:
        members:
            - __init__
            - __call__
            - encode_inputs
            - decode_query

---

::: phydrax.nn.GNOT
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.CoDANO
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.EqGINO

### Nonlinear, prompt, probabilistic, and PDE-IR conditioning

`CoordinateConditionedOperator` is the NOMAD-style path. Branch encoders produce
one function-level latent code; a genuinely nonlinear
`FiLMCoordinateDecoder` combines that code with every independent query
coordinate. It supports arbitrary query sets, but the global latent remains a
finite-dimensional bottleneck.

`InContextOperator` consumes a `PromptedOperatorBatch`: every
`OperatorSupervisedExample` pairs an `OperatorBatch` with its sampled target,
and `OperatorPrompt` supplies a fixed-capacity demonstration axis plus a mask.
Prompt padding and demonstration permutation are represented explicitly. This
is supervised in-context conditioning, not bundled task-distribution
pretraining.

`GaussianFunctionOperator` wraps a base operator whose channel-last output
contains a mean, an optional unconstrained scale, and optional low-rank factors. Its
`GaussianOperatorDistribution` reuses latent factor draws across the complete
query field, rather than treating query points as independent calls. Geometry,
masks, and query constraints are inherited from the wrapped base model; research
status is not a calibration claim.

All distributional operators implement `AbstractProbabilisticOperatorModel` and
return an `AbstractOperatorDistribution` over the complete valid output field.
`GaussianFunctionOperator(scale_mode="fixed")` keeps a declared diagonal noise
floor and learns only the location and optional low-rank factors. The wrapper's
`uncertainty_source` must describe what is sampled: use `process` for stochastic
dynamics and `observation` for a sensor model.

`ConditionalFlowFunctionOperator` uses a FlowJAX coupling flow for a non-Gaussian
residual around any deterministic Phydrax operator. `OperatorBatchConditioner`
concatenates named `FixedBranchEncoder` or `IntegralBranchEncoder` outputs. Its
finite output event, shared query geometry, quadrature, and mask are
constructor-fixed; this is a fixed-discretization transition density, not an
arbitrary-query neural operator. Retain it only when held-out distributional
metrics improve over the Gaussian baseline.

`LatentFlowJAXCoefficientProcess` is the finite-dimensional process counterpart.
It wraps a conditional FlowJAX residual law over a latent coefficient state.
`conditional_coupling_flow_process` supplies the standard current-state and
time conditioner with an identity residual location. The class implements
`AbstractMarginalTransitionLaw` only: it does not invent a pathwise driver or
cocycle from independent transition samples. Use `semigroup_objective` to train
or diagnose Chapman--Kolmogorov consistency.

`OperatorTransitionSpec`, `OperatorMarginalTransition`, and
`OperatorPathwiseTransition` connect these complete-field models to the common
stochastic-process contracts without adding stochastic semantics to the
architecture itself. Marginal rollouts are Markov chains; only a pathwise
adapter driven by one explicit `WienerRealization` carries common-path and
cocycle provenance. See
[Neural-operator uncertainty](../uq/operator.md#process-consistent-operator-transitions).

`PDEConditionEncoder` embeds canonical `PDETokenBatch` trees.
`attach_pde_condition` adds the encoded result as a named, one-anchor
`FunctionSamples` branch. The downstream model must be configured to consume
that branch. This preserves PDE-IR structure as conditioning metadata; it does
not enforce the PDE or provide equation-to-solution pretraining.

The token tree is generated from the canonical expression traversal: addition
and multiplication are recursively flattened and sorted, while argument slots
remain explicit for noncommutative operators. Declaration and attribute tokens
cover coordinate kind, size, bounds, periodicity, field representation and
scales, parameter values and scales, region and condition kinds, derivative
order and axis, component selection, and nondimensionalization. Symbol
conditioning uses declaration/reference equality rather than lexical names, so
a consistent alpha-renaming leaves the encoding unchanged while `u + u` remains
distinct from `u + v`. Arbitrary `PDEProblemIR.metadata` is provenance, not a
neural semantic channel.

::: phydrax.nn.FiLMCoordinateDecoder
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.CoordinateConditionedOperator
    options:
        members:
            - __init__
            - __call__
            - encode_inputs
            - decode_query

---

::: phydrax.nn.OperatorSupervisedExample

---

::: phydrax.nn.OperatorPrompt

---

::: phydrax.nn.PromptedOperatorBatch

---

::: phydrax.nn.InContextOperator
    options:
        members:
            - __init__
            - __call__
            - encode_prompt

---

::: phydrax.nn.AbstractOperatorDistribution

---

::: phydrax.nn.AbstractProbabilisticOperatorModel

---

::: phydrax.nn.GaussianFunctionOperator
    options:
        members:
            - __init__
            - __call__
            - distribution
            - sample

---

::: phydrax.nn.GaussianOperatorDistribution

---


::: phydrax.nn.OperatorBatchConditioner
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.ConditionalFlowFunctionOperator
    options:
        members:
            - __init__
            - distribution
            - sample

---

::: phydrax.nn.FlowJAXOperatorDistribution

---

::: phydrax.nn.conditional_coupling_flow_operator

---

::: phydrax.nn.StateTimeProcessConditioner

---

::: phydrax.nn.IdentityCoefficientTransition

---

::: phydrax.nn.LatentFlowJAXCoefficientProcess
    options:
        members:
            - __init__
            - marginal_transition

---

::: phydrax.nn.FlowJAXProcessDistribution

---

::: phydrax.nn.conditional_coupling_flow_process

---

::: phydrax.nn.PDEConditionEncoder
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.attach_pde_condition

### Pretraining-shaped grid architectures

`Poseidon` implements the native scOT-style two-dimensional patch hierarchy,
shifted-window transformer stages, ConvNeXt skip processing, and optional
continuous-time conditioning. Its `image_shape` is static, source and query
grids coincide, and the shape must be divisible by the patch and multiscale
merges. PhydraX bundles neither pretrained Poseidon weights nor the data and
training recipe needed to claim foundation-model behavior.

`DPOT` consumes a 2D tensor grid plus an explicit history-time axis and emits the
same spatial grid plus a fixed forecast-time axis. The constructor fixes
`history_steps`, `forecast_steps`, image shape, patch shape, and AFNO capacity.
`dpot_corrupt_history` implements scale-relative Gaussian corruption over the
three sample axes while respecting a supplied mask; `DPOT.corrupt_batch`
preserves the remaining `OperatorBatch` metadata. Neither large-scale denoising
pretraining nor pretrained DPOT checkpoints are bundled.

::: phydrax.nn.Poseidon
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.DPOT
    options:
        members:
            - __init__
            - __call__
            - corrupt_batch

---

::: phydrax.nn.dpot_corrupt_history

### Stable temporal evolution and learned Green kernels

`KoopmanTemporalOperator` quadrature-pools a state at elapsed time zero, evolves
global nonlinear observables with a contractive continuous or discrete latent
operator, and decodes an explicit nonnegative time axis. Stability is guaranteed
for the latent transition parameterization, not for prediction error or exact
conservation. Source and query must be tensor-product grids with matching
spatial axis names and order; a different regular query grid is allowed, an
arbitrary point cloud is not.

`GreenKernelOperator` is a learned volume/boundary prototype with distinct
forcing and boundary integral paths and arbitrary query coordinates. Both
sources must provide unnormalized physical quadrature and masks for padding.
Boundary normals or condition descriptors belong in boundary value channels.
The learned kernels do not impose a particular Green function, PDE, or boundary
condition exactly.

::: phydrax.nn.KoopmanTemporalOperator
    options:
        members:
            - __init__
            - __call__
            - decay_rates
            - evolution_matrix

---

::: phydrax.nn.GreenKernelOperator
    options:
        members:
            - __init__
            - __call__

### Flower and geometry-aware warp layers

`Flower` keeps the paper-faithful architecture as its default: learned
stride-two/transposed-convolution transitions, coincident tensor grids,
source-hole rejection, deterministic routes, and no conservation projection.
Every broader behavior is explicit:

| Option | Contract |
| --- | --- |
| `transition_mode="resolution_consistent"` | Measure-aware pair restriction and physical-coordinate prolongation; required for nonuniform multilevel grids and masked multilevel execution |
| `query_mode="interpolate"` | Decode the aligned dense field to a different tensor grid or explicit point coordinates |
| `source_mask_mode="renormalize"` | Normalize interpolation over valid stencil corners |
| `source_mask_mode="strict"` | Reject every stencil that crosses an invalid source point |
| `probabilistic_routing=True` | Diagonal Gaussian displacement fields; an evaluation key samples a route and no key uses its mean |
| `conserve_mass=True` | Channelwise minimum-constant correction matching source and query integrals; equal input/output channel counts and physical query measure are required |

Periodic axes use a half-open node interval. Nonuniform periodic axes require
positive quadrature weights whose sum is the physical period. Conservative
arbitrary-point queries require explicit quadrature weights. Call
`evaluate_with_diagnostics` to obtain the exact evaluated displacement,
coordinates, Jacobian, determinant, interpolation support, and probabilistic
route scale without changing the ordinary model return type.

`MultiheadWarp` is the regular-grid routing layer. `warp_field` and
`conservative_remap` expose scalar, density, vector, covector, and mixed-tensor
pullbacks through `phydrax.metrix.TensorType`. `ManifoldMultiheadWarp` instead
requires a caller-supplied ambient tangent projector and retraction; ambient
Euclidean displacement is not treated as intrinsically valid manifold transport.

---

::: phydrax.nn.Flower
    options:
        members:
            - __init__
            - __call__
            - evaluate_with_diagnostics

---

::: phydrax.nn.MultiheadWarp
    options:
        members:
            - __init__
            - __call__
            - diagnostics
            - diagnostics_from_displacement
            - displacement
            - transport

---

::: phydrax.nn.ProbabilisticMultiheadWarp
    options:
        members:
            - __init__
            - __call__
            - diagnostics
            - distribution

---

::: phydrax.nn.ManifoldMultiheadWarp
    options:
        members:
            - __init__
            - __call__
            - diagnostics
            - displacement

---

::: phydrax.metrix.TensorType

---

::: phydrax.nn.warp_field

---

::: phydrax.nn.conservative_remap

### Public maturity tiers

`operator_architecture_status(name)` is the canonical maturity registry. Tiers
describe the public implementation contract, not a claim that one family wins
every PDE:

| Tier | Exact registry entries | Recommendation eligible |
| --- | --- | --- |
| Stable | `FNO`, `TFNO`, `DeepONet`, `MIONet`, `PODDeepONet` | Yes |
| Experimental | `HOFNO`, `CNO`, `GraphNeuralOperator`, `SFNO`, `LocalDifferentialOperator`, `LocalGlobalOperator`, `LocalIntegralOperator`, `OperatorAttention`, `SliceAttention`, `AxialOperatorAttention`, `CodomainAttention`, `IFNO`, `AxialFactorizedFNO`, `ConditionalFlowFunctionOperator` | No |
| Research | `Flower`, `UNO`, `LaplaceTemporalOperator`, `GINO`, `GeometryInformedFlower`, `RIGNO`, `GAOT`, `WaveletNeuralOperator`, `MultiwaveletOperator`, `ManifoldSpectralOperator`, `CoordinateConditionedOperator`, `UPT`, `CochainNeuralOperator`, `ABUPT`, `CoDANO`, `EqGINO`, `InContextOperator`, `GaussianFunctionOperator`, `Poseidon`, `DPOT`, `Transolver`, `TransolverPlusPlus`, `GNOT`, `KoopmanTemporalOperator`, `GreenKernelOperator` | No |

TFNO is `FNO(factorization="tucker")`. MIONet is a product-fusion `DeepONet`
with a mapping of branch encoders. POD-DeepONet is a `DeepONet` with a fixed
`PODBasis` trunk. The registry normalizes those names and aliases while keeping
their underlying architecture and configuration visible. Unknown names fail
closed rather than inheriting a tier.

Only stable entries are recommendation-eligible. Experimental and research
entries are public so users can select them explicitly; public availability,
focused regression coverage, or a research label is not a recommendation or a
claim of benchmark superiority. In particular, architecture maturity does not
substitute for scenario-specific validation, and bundled constructors do not
imply bundled pretrained weights.

The 2026-07-24 portfolio review keeps the default recommendation surface
deliberately small: exactly the five stable registry entries above. `TFNO` is a
Tucker-factorized configuration of the canonical FNO core; `MIONet` and
`PODDeepONet` are explicit DeepONet configurations rather than parallel
implementations. Attention entries are reusable layers, not standalone roadmap
recommendations.
Every experimental or research family
remains explicit opt-in and is excluded from automatic recommendation.

Non-stable families remain explicit opt-ins only while their distinct execution
contract justifies their maintenance burden. A family may be removed when
matched evidence does not establish enough practical value over a simpler
architecture. No family should enter the stable portfolio without scenario
integrity, matched multi-seed evidence, and the relevant family-parity checks.

---

::: phydrax.nn.operator_architecture_status

---

::: phydrax.nn.OperatorArchitectureStatus

### Architecture applicability

Choose from the geometry and operator contract, not from a single leaderboard.
The benchmark harness enforces the same source/query protocol across families.

| Family | Native geometry | Independent query set | Honest applicability and principal limitation |
| --- | --- | --- | --- |
| `FNO`, `TFNO` | Uniform tensor grid | No; source/query coincide | Periodic or padded translation-dominated fields and zero-shot regular-grid resolution transfer; FFT geometry and retained-mode bias remain |
| `HOFNO` | Uniform periodic tensor grid | No; source/query coincide | Explicit degree-controlled projected products with optional spectral de-aliasing; all-valid scalar/generic-channel fields only, and the experimental tier is not recommendation-eligible |
| `IFNO` / `AxialFactorizedFNO` | Uniform tensor grid | No | Shared-weight fixed-point iteration or sequential axis transforms; diagnostics do not early-stop IFNO, and axial transforms omit simultaneous cross-axis mixing |
| `WaveletNeuralOperator` / `MultiwaveletOperator` | Constructor-fixed tensor grid; MWT is 1D | No | Multiresolution or polynomial subbands; exact shape and aligned nodes are fixed by the transform |
| `ManifoldSpectralOperator` | Fixed/aligned manifold eigenbasis | Target plan only | Intrinsic spectral processing across pre-aligned discretizations; not an arbitrary-query or independently remeshed-manifold model |
| `BasisSpectralConvND` | Separable nonuniform tensor grid | No | Boundary-adapted sine/cosine/Legendre expansions; dense transforms and separable grids only |
| `CNO` / `UNO` | Uniform tensor grid | No | Alias-controlled or multiscale grid fields; convolutional geometry, hierarchy, and divisibility constraints |
| `DeepONet`, `MIONet`, `PODDeepONet` | Fixed/variable sensors or point clouds | Yes; POD uses its fixed output basis | Function-to-point and multiple-input maps; finite branch–trunk rank, while POD cannot leave its fitted span |
| `GraphNeuralOperator` | Graph or point geometry | Yes | GraphIR message passing and source-to-query transfer; graph construction and receptive field are part of the model |
| `CochainNeuralOperator` | Typed primal/dual cochains on one metric cell complex | Resolution transfer on a compatible complex relation | Mixed-degree differential forms with exact sparse DEC routes and optional harmonic projection; requires oriented incidence, valid Hodge stars, and declared field semantics |
| `LocalDifferentialOperator`, `LocalIntegralOperator`, `LocalGlobalOperator` | Coordinates on grids or point clouds | Yes | Local closures and kernels with dimensional radii; finite neighborhoods and sampling quality limit them |
| `OperatorAttention`, `SliceAttention`, `AxialOperatorAttention`, `CodomainAttention` | Weighted point sets or tensor axes | Cross-attention can use separate queries | Layer primitives rather than complete roadmap models; quadratic, slice, axial, or field-factorization bottlenecks remain |
| `LaplaceTemporalOperator` | Monotone nonperiodic time samples | Yes in time | Stable causal transients within a pole–residue model class |
| `SFNO` | Colatitude–longitude sphere | No | Rotation-equivariant spherical fields; spherical tensor grid only |
| `GINO`, `RIGNO`, `GAOT` | Meshes and point clouds | Yes | Geometry-dependent PDEs; graph/latent construction, physical measure, and transfer discretization remain |
| `GeometryInformedFlower` | Weighted point clouds or rasterized irregular domains | Yes | Geometry transfer around a masked multiscale Flower processor; ambient point neighborhoods do not encode mesh connectivity or intrinsic geodesics |
| `EqGINO` | 3D point geometry with O(3) representations | Yes | Equivariant scalar/vector/tensor fields; finite radius and declared representation content constrain the map |
| `UPT` | Weighted source set | Yes | Fixed latent-token compression and arbitrary-query decoding; token count is an information bottleneck |
| `ABUPT` | Typed weighted branches and anchors | Yes, by declared prediction branch | Heterogeneous interacting physics branches; graph roles, anchor capacities, and interactions must be specified |
| `CoDANO` | Heterogeneous named fields on a common tensor latent grid | Yes | Multiphysics field fusion with declared roles; common-grid transfer and latent spectral discretization remain |
| `CoordinateConditionedOperator` | Any branch-encoder source geometry | Yes | NOMAD-style nonlinear coordinate decoding; one global function latent can bottleneck complex fields |
| `InContextOperator` | Weighted demonstrations plus current source/query | Yes | Supervised function demonstrations; static prompt capacity and no bundled task-distribution pretraining |
| `GaussianFunctionOperator` | Inherited from its base operator | Inherited | Coherent diagonal-plus-low-rank function distributions; calibration and geometry validity are not added by the wrapper |
| `ConditionalFlowFunctionOperator` | Constructor-fixed tensor grid or point cloud with one shared mask | No | Conditional non-Gaussian complete-field density; finite FlowJAX event size forbids changed query geometry or resolution transfer |
| `PDEConditionEncoder` | Canonical PDE-IR token tree | Conditioning branch only | Structure-aware global equation conditioning; the downstream operator must consume it and no PDE is enforced automatically |
| `Poseidon` | Constructor-fixed 2D tensor grid | No | Native scOT-style multiscale architecture; divisibility constraints and no bundled pretraining/checkpoint |
| `DPOT` | Fixed 2D grid with history/forecast axes | No in space; fixed forecast axis | AFNO autoregression and denoising corruption; static horizon/shape and no bundled large-scale pretraining/checkpoint |
| `Transolver` / `TransolverPlusPlus` | Weighted physical points | Yes | Hard slices at `slice_top_k=1`, normalized overlap above one; slice count/top-k are information bottlenecks |
| `GNOT` | Named heterogeneous point/grid sources | Yes | Per-source encoders and learned query-local gates; all source/query measures and masks must remain physically meaningful |
| `KoopmanTemporalOperator` | Tensor-product spatial grid plus explicit time | A different regular spatial grid and times only | Contractive latent evolution; stability does not ensure forecast accuracy and point clouds are rejected |
| `GreenKernelOperator` | Volume and boundary point/grid samples | Yes | Separate learned volume/boundary integrals; physical quadrature is mandatory and exact PDE/BC satisfaction is not imposed |

Fixed-grid operators deliberately reject or semantically exclude an unrelated
query cloud. Use branch–trunk, local/graph transfer, UPT/ABUPT, Transolver,
GNOT/CoDANO, NOMAD-style decoding, EqGINO, or Green kernels when sensors and
queries differ. Preserve masks whenever cardinality varies and use physical
quadrature for every continuum integral or measure-aware attention path;
padding without a mask and irregular sampling without a measure change the
mathematical operator.

#### Reliability benchmark

Operator Benchmark v2 writes canonical validation artifacts. It is a validation
and promotion protocol, not a single leaderboard. Its twenty easy/hard physical
ladders cover smooth periodic dynamics, controlled degree-one/degree-two
polynomial Poisson maps, shocks, elliptic coefficient contrast, irregular
geometry, independent source/query sets, multiple functional inputs, long
rollouts, causal transients, spherical fields, geometry/parameter extrapolation,
exact or deliberately broken square-group symmetry, mixed-degree Darcy
cochains, and annular harmonic one-form projection.

The command profiles are deliberately distinct:

| Profile | Cases per generated level | Role | Promotion eligible |
| --- | ---: | --- | --- |
| `smoke` | 8 | API, shape, and artifact smoke test | No |
| `shortlist` | 24 | Eliminate unsuitable families cheaply | No |
| `decision` | 128 | Final five-seed, resumable comparison | Yes |

##### Intended Flower portfolio

Flower coverage is planned as three named, separately reported
configurations: paper-faithful one-level (`levels=1`), paper-faithful
multilevel (`levels>1` with learned stride-two transitions), and multilevel
resolution-consistent Flower
(`transition_mode="resolution_consistent"`). For this portfolio, all use
`boundary="periodic"`, `query_mode="coincident"`, and
`probabilistic_routing=False`. The default `transition_mode="learned"` defines
the paper-faithful behavior; resolution-consistent transitions are explicit
opt-in behavior, not a new default.

The architecture-neutral ladders cover smooth constant-speed and
variable-speed periodic advection, viscous periodic Burgers shock formation
and long rollout, and periodic acoustic waves. Their scenario records expose
grid resolution, time step and horizon; advection speed or speed range;
Burgers viscosity or Reynolds-like range; and acoustic wave speed and
wavenumber content. This makes translation/phase, finite outputs, periodic
mass conservation, shock steepening and viscous smoothing, long-rollout
behavior, and acoustic phase/energy behavior directly checkable.

Each deterministic case keeps its seed and physical-realization ID in the
existing scenario dataclass and checksum pathway. Provenance identifies the
equation, periodic boundary convention, initial-condition population,
physical/discretization ranges, and reference construction. Analytic
translation or phase is the reference where available; numerical references
instead require residual and grid-refinement evidence. A resolution shift
reuses the physical realization rather than generating a different case.
These checks feed the existing runner and the same integrity, hardness,
accuracy, robustness, efficiency, matching, parity, and provenance promotion
gates; there is no Flower-specific exception. Existing quick/smoke profiles
remain small and promotion-ineligible.

Comparator coverage remains honest. FNO and IFNO are eligible only on aligned
uniform coincident grids and are not independent-query decoders. CNO and UNO
also require compatible hierarchy and divisibility. Wavelet operators require
constructor-fixed aligned shapes and a supported spatial dimension. Therefore
no FNO/IFNO/CNO/UNO/Wavelet comparator is implied for a resolution-transfer
level whose shape or grid semantics violate its existing contract.

Every generated population uses a deterministic seed and independent multimode
coefficients. Resolution and geometry shifts reuse the same physical
realizations rather than silently changing both the PDE instance and the
discretization. The registry now admits 34 easy/hard levels to the shortlist
hardness contract, including six transport/acoustic levels and four
square-symmetry/control levels. The checked-in core decision evidence predates
those additions and covers 22 levels; for that historical core only, the
recorded minima are source effective rank 7.22, target effective rank 3.33,
99%-energy target rank 6, and train-to-test realization distance 0.061. Fresh
34-level decision artifacts are required before quoting updated continuous-rank
minima. The live shortlist contract requires every level to pass its rank,
realization-separation, and 0.05 persistence gates.

Scenario checks happen before promotion:

- physical-realization IDs must be disjoint across train, validation, and test;
- provenance, dimensional ranges, and reference-solver evidence must be
  complete;
- identity and persistence baselines must not solve the training map;
- the nearest held-out realization must remain separated from training;
- concatenated source inputs and targets must pass effective-rank,
  99%-energy-rank, and rank-fraction thresholds.

Raw sensor corruption and missing sensors are different evaluations.
`sensor_corruption` zeroes values without changing the observation mask;
`sensor_dropout` zeroes the same kind of values and marks them missing.
`--train-sensor-dropout` creates a separate deterministic mask-augmented
training run while leaving validation and test data untouched.

Three comparison modes are available:

- **capacity matched** selects the closest architecture-specific size only when
  the target lies in the common feasible parameter range;
- **compute matched** compiles a normalized loss-and-gradient step and matches
  measured JAX/XLA FLOPs while recording accessed bytes;
- **Pareto** sweeps every requested size scale and reports validation error,
  worst shifted error, training FLOPs, inference latency, backend peak memory,
  and parameter count.

An unmeasurable objective remains `None`; the corresponding Pareto point is
incomplete and has no dominance label. In particular, CPU runs do not pretend
to provide accelerator peak-memory measurements.

Every learning-rate/seed/size trial persists its full training curve, periodic
validation curve, best checkpoint, optimizer state, PRNG key, elapsed time, and
resume point. Plateau detection accepts absolute and relative improvement
thresholds. The selected best-validation state is restored before shifted test
sets are evaluated exactly once. Test metrics therefore cannot influence
hyperparameter or early-stopping decisions.

Each run writes one JSON document and eight Parquet tables: aggregates, complete
trials, sample-efficiency curves, paired symmetry defects, scenario-difficulty
audits, Pareto fronts, per-scenario promotions, and portfolio promotions. The
schema records source/target rank, realization novelty, convergence rate, size
scale, comparison measurements, statistical confidence intervals, and every
gate reason.

Promotion is conjunctive. A candidate must pass integrity, baseline hardness,
five-seed convergence and variance, nominal accuracy, shift robustness,
runtime/parameter/memory limits, a capacity or measured-FLOP ratio within 10% of
the common target, and pinned family parity. Reference baselines are never
promotable. Quick and shortlist
profiles are explicitly ineligible. Missing evidence means no promotion, not an
inferred recommendation.

The checked-in parity evidence is
`tools/operator_benchmarks/reference/family_parity.json`. Its immutable revisions
pin the official DeepONet specification, NeuralOperator, Laplace-operator,
torch-harmonics, RIGNO, and GAOT sources. Those family checks exercise
branch-trunk contraction, dense versus factorized spectral weights, direct
versus recurrent causal quadrature, spherical longitude equivariance, the GINO
weighted integral transform, RIGNO regional permutation equivariance, and GAOT
measure-aware scaled dot-product attention.

External checkpoint procurement is recorded separately in
`tools/operator_benchmarks/reference/external_candidates.json`.
`ExternalOperatorAdapter` loads a candidate only after code and weight licenses,
input/output schemas, preprocessing, normalization, dataset provenance,
checkpoint URI, immutable revision, and SHA-256 checks all pass. It must then
beat the best native model on every matched nominal, robustness, latency, and
parameter-count regime.

#### Evidence status

Artifacts under `reference/v2-converged/` and `reference/v2-compute/` predate
the multimode population, baseline-hardness, exact-resume, and Pareto changes.
Decision artifacts under `tools/operator_benchmarks/reference/v3-decision/`
add those controls but predate the deformed-domain integrity audit, resolved
architecture configuration, and GINO/RIGNO/GAOT registrations. All checked-in
artifacts are historical diagnostics for the current geometry families, not
promotion evidence.

Current reports retain named per-field metrics and target schemas, post-split symmetry augmentation, per-seed paired
group-action defects, base-realization sample-efficiency curves, selected
architecture size scales, parameter counts, and measured training FLOPs. The
square-symmetry ladder compares ordinary FNO with a rotationally augmented FNO
as a diagnostic; augmentation does not imply exact architecture-level
equivariance or a specialized promotion claim.

The first decision run evaluated the hard independent-query Green
operator with 128 physical cases, seeds 0–4, 300 maximum steps, one pinned
learning rate, and a single size scale. The scenario audit passes with source
effective rank 15.33, target effective rank 10.18, 99%-energy target rank 19,
and nearest held-out realization distance 0.677.

| Candidate | Parameters | Convergence rate | Nominal | Query geometry | Input noise | Raw corruption | Masked dropout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Constant reference | 0 | 1.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Nearest-neighbor reference | 0 | 1.0 | 30.5300 | 31.1174 | 30.5017 | 25.2342 | 25.6279 |
| `DeepONet` | 7,297 | 0.6 | 1.0013 | 1.0020 | 1.0013 | 1.0013 | 1.0010 |
| `LocalIntegralOperator` | 2,641 | 0.2 | 0.9955 | 0.9960 | 0.9955 | 0.9973 | 0.9974 |

Neither learned model beats the constant reference materially, and neither
converges for all five seeds. No candidate is promoted. Peak device memory is
unavailable on the CPU backend, so every Pareto point is explicitly incomplete
and no dominance label is emitted. POD-ROM is structurally inapplicable because
the evaluation changes query geometry; requested architectures that are absent
from every selected scenario now fail before training instead of disappearing
from the report.


A separate five-seed `DeepONet` run applies 20% mask-aware sensor dropout to
training only. Nominal relative L2 changes from 1.00133 ± 0.00151 to
1.00133 ± 0.00153; masked-dropout relative L2 changes from
1.00100 ± 0.00096 to 1.00092 ± 0.00100. The difference is smaller than the
seed dispersion. This augmentation therefore provides no measured benefit on
the hard independent-query case and is not used to justify promotion.


The hard causal-transient decision run also passes the data audit: source
effective rank 18.78, target effective rank 19.54, 99%-energy target rank 30,
and nearest-realization distance 0.772.

| Candidate | Parameters | Convergence rate | Nominal | Resolution transfer | Input noise | Raw corruption | Masked dropout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Constant reference | 0 | 1.0 | 0.9986 | 0.9986 | 0.9986 | 0.9986 | 0.9986 |
| Nearest-neighbor reference | 0 | 1.0 | 6.9951 | 7.0007 | 7.0010 | 5.8298 | 5.8305 |
| `DeepONet` | 7,297 | 0.0 | 1.0012 | 1.0012 | 1.0014 | 1.0057 | 1.0141 |
| `LaplaceTemporalOperator` | 74 | 0.2 | 1.0664 | 1.0191 | 1.0679 | 1.0547 | 1.1450 |

Neither learned candidate beats the constant reference or converges reliably.
The Laplace model also has large seed dispersion: its nominal relative-L2
standard deviation is 0.409. It remains research-tier pending a diagnosed
optimization or expressivity improvement; pole-count efficiency alone is not a
promotion argument.


The hard smooth-periodic decision run passes with source effective rank 29.76,
target effective rank 22.56, 99%-energy target rank 58, and nearest-realization
distance 0.880.

| Candidate | Parameters | Convergence rate | Nominal | Input noise | Raw corruption | Masked dropout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Constant reference | 0 | 1.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `FNO` | 1,185,538 | 0.2 | 0.3656 | 0.3698 | 0.6597 | 0.6587 |
| TFNO configuration | 48,898 | 0.0 | 0.5458 | 0.5508 | 0.8321 | 0.8271 |

`FNO` is materially more accurate than the constant reference, but it does not
meet the accuracy, variance, efficiency, or five-seed convergence gates. TFNO
reduces the parameter count by 24.2× at lower accuracy and without converged
seeds. Neither is promoted by this run.

CNO was intentionally omitted from the final artifact after the original
five-seed command exceeded one hour with only seed 0 at step 125 of 300 and
four CNO seeds unstarted. That interrupted run exposed process-dependent
scenario checksums caused by last-bit FFT roundoff. Reference generation is now
bitwise stable across fresh processes, checksum identity records the reference
verdict instead of incidental residual roundoff, and a fresh-process `--resume`
restored every FNO/TFNO trial at step 250 or 300 without retraining. The
incomplete CNO run supplies no promotion evidence.


### Architecture-aware uncertainty

Start with independently trained ensembles for every family. Add a posterior
subspace only when its parameterization matches the architecture and its cost is
justified. The generic `ParameterSubspace.last_layer()` follows global PyTree order;
it is not an architecture-aware selector.

| Family | Practical first method | Targeted posterior subspace | Architecture-specific caution |
| --- | --- | --- | --- |
| `FNO` / TFNO | Deep ensemble; coherent channel-shared MC dropout | Final projection, then explicitly named spectral blocks | Full spectral-weight posteriors are large; preserve signed-frequency structure |
| `DeepONet` | Deep ensemble | Final branch head plus trunk head, bias, and fusion parameters as intended | Selecting only the globally final branch under-represents uncertainty |
| MIONet configuration | Deep ensemble | Every coefficient-producing branch head, trunk head, fusion mixer, and bias | Product fusion makes the joint head posterior nonlinear |
| POD-DeepONet | Deep ensemble or branch-head posterior | Coefficient-producing branch head and bias | No uncertainty method recovers output modes absent from the fixed POD span |
| `CNO` / `UNO` | Deep ensemble; configured dropout | Final decoder/readout | Grid and multiresolution assumptions still govern shifted predictions |
| Local operators | Deep ensemble | Final local/global decoder or selected kernel parameters | Neighborhood radius and sampling-density shift are model-form risks |
| Operator attention / Transolver | Deep ensemble | Final value/readout projections first | Full attention posteriors are expensive; slice bottlenecks remain |
| `LaplaceTemporalOperator` | Deep ensemble | Stable pole/residue parameter subset with constraints preserved | Do not break conjugate pairing or negative-real-part stability |
| `SFNO` | Deep ensemble | Selected harmonic-degree gains or final projection | Parameterization must preserve rotation equivariance |
| Graph geometry pipelines | Heterogeneous or homogeneous ensemble | Final graph decoder/readout | Query graph and source measures remain part of the predictive contract |
| External adapters | Ensemble only after manifest validation | Framework-specific, explicitly mapped leaves | Schema, normalization, provenance, and checkpoint identity must match |

`OperatorPredictiveField` preserves query masks, quadrature, case axes, and output
channels through these methods. `FixedOperatorObservationLikelihood` is the bridge
to MAP, NUTS/HMC, Pathfinder, SMC, and Laplace inference. It represents finite noisy
observations, not an operator training norm. `OperatorFunctionalConformal` provides
whole-field calibration after a disjoint physical-case split. See the
[operator-UQ guide](../../guides_uncertainty.md#neural-operator-uncertainty) and
[cookbook](../../cookbook/operator_uncertainty.md).

### Losses, training, and external models

Operator metrics reduce over sample coordinates before averaging physical cases:
`operator_l2_loss`, `operator_h1_loss`, `operator_sobolev_loss`,
`operator_spectral_loss`, and `operator_conservation_error`.

`OperatorDatasetConstraint` and `PhysicsInformedOperatorConstraint` attach array
data and residual terms at the same or different resolutions.
`DifferentialPhysicsInformedOperatorConstraint` instead binds a source context as
a coordinate-aware `DomainFunction`, so residual builders can compose ordinary
PhydraX `grad`, `laplacian`, divergence, and integral operators without manual
finite differences. PINO remains a training composition, not a separate model
class.

`OperatorContextModel` and `bind_operator_context` expose independent,
differentiable point queries while keeping all source functions fixed.

The `operator_training` package supplies deterministic dataset splits,
mask-preserving collation, persisted training-only normalization, exact
model/optimizer/RNG checkpoints, explicit parameter/compute/reduction dtypes,
scheduled autoregressive rollouts, and prefetching sharded loaders.

`ExternalOperatorAdapter` requires a version-2
`OperatorCheckpointManifest` with immutable source and checkpoint revisions,
separate code and weight licenses, field schemas, preprocessing, normalization,
dataset provenance, and a mandatory SHA-256 digest. Loading verifies the
checkpoint before framework-specific tokenization or execution.

---

::: phydrax.nn.OperatorContextModel
    options:
        members:
            - __init__
            - __call__
            - domain_function

---

::: phydrax.nn.OperatorNormalizationPolicy
    options:
        members:
            - normalize_batch
            - denormalize_batch
            - normalize_target
            - denormalize_prediction

---

::: phydrax.nn.OperatorBatchLoader
    options:
        members:
            - __init__
            - epoch

### Canonical operator data and task contracts

`OperatorBatch` is the only execution batch. It stores named source functions and
named query geometries; `OperatorTargetBatch` stores named supervised fields.
`OperatorTask` is the immutable scientific contract used by validation,
training, artifacts, and deployed inference. It records source/query structure
separately from fixed-query discretization. Fitting requires both semantics to be
explicit. Fixed queries must share geometry across cases and batches, and their
physical geometry fingerprints are enforced by `TrainedOperator`. A trained
operator combines the task with an execution model, normalization, dtype policy,
training evidence, provenance, physical output pipeline, and explicit
output-field mapping.

---

::: phydrax.nn.OperatorTask

---

::: phydrax.nn.OperatorTrainingEvidence

---

::: phydrax.nn.TrainedOperator
    options:
        members:
            - prepare
            - predict
            - predict_prepared

### Production fitting

`fit_operator` accepts an in-memory dataset, a selective `OperatorCaseSource`, or
an `OperatorBatchLoader`. It owns optimizer state, random keys, best-model
selection, validation scheduling, callbacks, exact checkpoint/resume,
microbatching, mixed precision, distributed case sharding, and the physical
output pipeline. `OperatorFitResult.execution_model` and
`last_execution_model` make the execution-space result explicit. Loss callables
and all callable schedules require stable identities for exact-resume
compatibility.

Optimizers with distinct training and evaluation iterates can supply
`evaluation_parameters(optimizer_state, training_parameters)`. Validation,
best-model selection, `execution_model`, and `last_execution_model` then use that
evaluation view; gradients and updates continue to use raw training parameters.
Checkpointed runs must also supply a stable `evaluation_parameters_id`. The
identifier is part of the exact-resume contract, so changing or omitting it rejects
resume before training continues. With the default `None` lifecycle, existing
checkpoint fingerprints are unchanged.

---

::: phydrax.nn.fit_operator

---

::: phydrax.nn.OperatorFitResult

---

::: phydrax.nn.OperatorValidationPolicy

---

::: phydrax.nn.OperatorMixedPrecisionPolicy

---

::: phydrax.nn.OperatorShardingPolicy

### Losses and physical-space output maps

`SupervisedOperatorLoss` supplies a named, measure-aware L2 objective.
`OperatorLossTerm` adapts a stable custom scalar objective, and
`WeakOperatorLoss` evaluates residual moments against one or more test
functions. `CochainResidualLoss` scatters typed prediction/source samples onto
their canonical cell complex, evaluates a `CochainResidualProgram`, and applies
a segmented Hodge-aware reduction. Every generic term declares
`space="physical"` or `"execution"`; physical is the default.
`OperatorLossContext` carries paired predictions, batches, and targets for both
spaces. `OperatorOutputPipeline` applies exact hard-constraint and conservation
transforms only after dimensionalization, inside both fitting and
`TrainedOperator` inference. Targetless fitting requires explicit physics terms
and explicit or previously fitted output scaling.

---

::: phydrax.nn.SupervisedOperatorLoss

---

::: phydrax.nn.OperatorLossTerm

---

::: phydrax.nn.WeakOperatorLoss


---

::: phydrax.nn.OperatorLossContext

---

::: phydrax.nn.CochainResidualInput

---

::: phydrax.nn.CochainResidualLoss
---

::: phydrax.nn.HardConstraintTransform

---

::: phydrax.nn.ConservationProjection

---

::: phydrax.nn.OperatorOutputPipeline

### Measured metrics and matrix-free derivatives

The Hilbert helpers use the query's physical quadrature and mask and accept an
optional channel metric. `linearize_operator` fixes every source except the named
one and returns a matrix-free derivative. Its Hilbert adjoint uses independent
source/output measures and therefore is distinct from the raw Euclidean
pullback.

---

::: phydrax.nn.operator_hilbert_inner_product

---

::: phydrax.nn.operator_hilbert_norm

---

::: phydrax.nn.operator_hilbert_relative_error

---

::: phydrax.nn.operator_weak_form_loss

---

::: phydrax.nn.project_operator_conservation

---

::: phydrax.nn.linearize_operator

---

::: phydrax.nn.OperatorLinearization
    options:
        members:
            - pushforward
            - pullback
            - adjoint
            - adjoint_identity_error

### Lazy data, streaming inference, and artifacts

`OperatorCaseSource` separates lightweight case metadata from selective source
and target reads. Query sampling is applied before collation, so values,
coordinates, masks, and quadrature remain aligned. Encoded operators can decode
an `OperatorQuerySource` into an `OperatorPredictionSink` without retaining the
full query or prediction on device. The canonical artifact verifies execution-model and
training-state digests, fixed-query geometry fingerprints, physical output-pipeline identity,
normalization, dtype, provenance, and architecture contracts before constructing
a deployed operator. Artifacts are development snapshots rather than a
backward-compatibility surface; regenerate them after the canonical representation changes.

---

::: phydrax.nn.OperatorCaseSource

---

::: phydrax.nn.CallbackOperatorCaseSource

---

::: phydrax.nn.AnchorQuerySamplingPolicy

---

::: phydrax.nn.decode_query_chunks

---

::: phydrax.nn.save_operator_training_checkpoint

---

::: phydrax.nn.load_operator_training_checkpoint

---

::: phydrax.nn.save_operator_artifact

---

::: phydrax.nn.load_trained_operator

