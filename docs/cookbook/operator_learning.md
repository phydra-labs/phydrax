# Operator learning (DatasetDomain × coordinates)

This recipe shows the “operator-learning” decomposition

$$
\Omega \;=\; \Omega_{\text{data}}\times\Omega_x\times\cdots,
$$

where \(\Omega_{\text{data}}\) indexes a dataset of inputs (forcing, coefficients, initial conditions, etc.) and
\(\Omega_x\) is the coordinate domain where you evaluate outputs.

In Phydrax, \(\Omega_{\text{data}}\) is represented by `DatasetDomain`, and operator models are wrapped via
`Domain.Model(...)` so they can be used like any other `DomainFunction`.

## Dataset factor

`DatasetDomain` stores an in-memory PyTree of arrays with a shared leading dataset axis, and samples by indexing.
See [API → Domain → Composition](../api/domain/composition.md).

For row-indexed time series with a shared `dt` and different lengths, use
`TrajectoryDatasetDomain` instead of `DatasetDomain @ TimeInterval`. It keeps the
dataset row and sampled time coupled so physics residuals and ragged supervised data
constraints see only valid `(data, t)` pairs.

## DeepONet skeleton on \(\Omega_{\text{data}}\times\Omega_x\) {: data-toc-label="DeepONet skeleton on Ω_data × Ω_x"}

Assume each dataset sample contains a vector of coefficients \(c\in\mathbb{R}^K\) that parameterizes an input.
For this runnable example, we choose a simple analytic “operator” that maps \(c\) to a 1D field
\(u(x)=\sum_{k=1}^K c_k \sin(k\pi x)\).

!!! example
    ```python
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import optax
    import phydrax as phx

    key = jr.key(0)

    # N dataset samples, each carrying K coefficients.
    N = 32
    K = 8
    coeffs = jr.normal(key, shape=(N, K))

    data_dom = phx.domain.DatasetDomain(coeffs, label="data", measure="probability")
    geom = phx.domain.Interval1d(0.0, 1.0)
    domain = data_dom @ geom

    latent = 32
    branch = phx.nn.models.MLP(in_size=K, out_size=latent, width_size=64, depth=2, key=jr.key(1))
    trunk = phx.nn.models.MLP(in_size=1, out_size=latent, width_size=64, depth=2, key=jr.key(2))
    deeponet = phx.nn.operator.architectures.DeepONet(branch=branch, trunk=trunk, coord_dim=1, latent_size=latent)

    # u_hat(data, x): predicted field on the x-axis for each dataset sample
    u_hat = domain.Model("data", "x")(deeponet)

    # Supervised target u_true(data, x): analytic mapping from coefficients to a function of x
    @domain.Function("data", "x")
    def u_true(c, x):
        ks = jnp.arange(1, K + 1, dtype=float)
        return jnp.sum(c * jnp.sin(jnp.pi * ks * x[0]))

    # Supervised residual on Ω_data × Ω_x.
    def residual(u_f):
        return u_f - u_true

    # Sample empirical rows in one point block and x on an explicit axis.
    nx = 32
    component = domain.component()
    sampling = phx.domain.GridSampling(
        {"x": phx.domain.UniformAxisSpec(nx)},
        dense=phx.domain.PointSampling(
            8,
            layout=phx.domain.SampleLayout((("data",),)),
            design="uniform",
        ),
    )
    condition = phx.conditions.Residual("u", component, residual)
    source = phx.integration.per_step(
        phx.integration.mean_over(component),
        sampling,
    )
    term = phx.terms.ResidualPenalty(condition, source)

    solver = phx.solver.FunctionalSolver(functions={"u": u_hat}, terms=[term])
    solver = solver.solve(num_iter=20, optim=optax.adam(1e-3), seed=0)
    ```

!!! note
    This page focuses on the domain/model wiring. For structured-input conventions and operator architectures (DeepONet/FNO),
    see [API → NN → Architectures](../api/nn/architectures.md). For sampling semantics, see [Guides → Domains and sampling](../guides_domain.md).

## Canonical source/query batches

Use `OperatorBatch` directly when source sensors and query points differ, when
quadrature or masks are part of the problem, or when several functional inputs
must be kept distinct.

!!! example
    ```python
    import jax.numpy as jnp
    import jax.random as jr
    import phydrax as phx

    sensor_x = jnp.array([[0.0], [0.15], [0.4], [0.8], [1.0]])
    sensor_w = jnp.array([0.075, 0.2, 0.325, 0.3, 0.1])
    query_x = jnp.linspace(0.0, 1.0, 64)[:, None]
    values = jnp.sin(jnp.pi * sensor_x[:, 0])

    source = phx.nn.operator.FunctionSamples(
        values=values,
        coordinates=sensor_x,
        quadrature_weights=sensor_w,
    )
    query = phx.nn.operator.FunctionSamples(values=None, coordinates=query_x)
    batch = phx.nn.operator.OperatorBatch(
        inputs={"forcing": source},
        queries={"query": query},
    )

    latent = 32
    feature_model = phx.nn.models.MLP(
        in_size=2,  # value + source coordinate
        out_size=latent,
        width_size=64,
        depth=2,
        key=jr.key(0),
    )
    branch = phx.nn.operator.architectures.IntegralBranchEncoder(
        feature_model=feature_model,
        latent_size=latent,
        coord_dim=1,
    )
    trunk = phx.nn.models.MLP(
        in_size=1,
        out_size=latent,
        width_size=64,
        depth=2,
        key=jr.key(1),
    )
    model = phx.nn.operator.architectures.DeepONet(
        branch={"forcing": branch},
        trunk=trunk,
        coord_dim=1,
        latent_size=latent,
    )
    prediction = model(batch)  # (64,)
    ```

`IntegralBranchEncoder` is permutation-invariant only when values, coordinates,
weights, and masks are permuted together. Masked sensors contribute exactly zero.


### Shape, measure, and mask invariants

Let `C = batch.case_shape` and `S = samples.sample_shape`. A scalar field has
shape `C + S`; a vector field has shape `C + S + (channels,)`. Shared tensor
grids use `axes=(OperatorAxis(...), ...)`, while a point cloud uses coordinates
with shape `S + (coord_dim,)` or per-case shape
`C + S + (coord_dim,)`. Quadrature and masks have shape `S` when shared or
`C + S` when case-specific. Query values are normally `None`.

`case_axes` names only `C`. Do not flatten a physical case into the sample axis:
losses reduce the full sample geometry before averaging cases. A padded source
or query requires a mask, and values, coordinates, quadrature, and mask must be
padded and permuted together. For an irregular discretization, supply physical
cell/volume/boundary weights. Unit weights are a counting measure, not a generic
replacement for missing quadrature.

### Arbitrary-query token, slice, and nonlinear paths

The same measured source/query batch can drive UPT, hard-slice Transolver,
overlapping Transolver++ configuration, heterogeneous GNOT, and a NOMAD-style
nonlinear coordinate decoder. `slice_top_k=1` is the hard partition;
`slice_top_k>1` gives normalized overlapping memberships.

```python
source_mask = jnp.array([True, True, True, True, False])
query_mask = jnp.arange(query_x.shape[0]) < query_x.shape[0] - 1
measured_batch = phx.nn.operator.OperatorBatch(
    inputs={
        "forcing": phx.nn.operator.FunctionSamples(
            values=values,
            coordinates=sensor_x,
            quadrature_weights=sensor_w,
            mask=source_mask,
        )
    },
    queries={"query": phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=query_x,
        quadrature_weights=jnp.full((query_x.shape[0],), 1.0 / query_x.shape[0]),
        mask=query_mask,
    )},
)

upt = phx.nn.operator.architectures.UPT(
    in_channels="scalar",
    out_channels="scalar",
    coord_dim=1,
    width=8,
    num_tokens=4,
    depth=1,
    num_heads=2,
    source_key="forcing",
    key=jr.key(30),
)
transolver = phx.nn.operator.architectures.Transolver(
    coord_dim=1,
    num_slices=4,
    width=8,
    depth=1,
    num_heads=2,
    slice_top_k=1,
    source_key="forcing",
    key=jr.key(31),
)
transolver_pp = phx.nn.operator.architectures.Transolver(
    coord_dim=1,
    num_slices=4,
    width=8,
    depth=1,
    num_heads=2,
    slice_top_k=2,
    source_key="forcing",
    key=jr.key(32),
)
gnot = phx.nn.operator.architectures.GNOT(
    in_channels={"forcing": "scalar"},
    out_channels="scalar",
    coord_dim=1,
    hidden_channels=8,
    encoder_width=8,
    encoder_depth=1,
    fusion_width=8,
    fusion_depth=1,
    transformer_depth=1,
    num_heads=2,
    key=jr.key(33),
)

nomad_decoder = phx.nn.operator.architectures.FiLMCoordinateDecoder(
    latent_size=latent,
    coord_dim=1,
    out_size="scalar",
    width=16,
    depth=2,
    key=jr.key(34),
)
nomad = phx.nn.operator.architectures.CoordinateConditionedOperator(
    branch={"forcing": branch},
    decoder=nomad_decoder,
    coord_dim=1,
    latent_size=latent,
    source_key="forcing",
)

predictions = tuple(
    model(measured_batch)
    for model in (upt, transolver, transolver_pp, gnot, nomad)
)
assert all(prediction.shape == query_x.shape[:-1] for prediction in predictions)
assert all(jnp.allclose(prediction[-1], 0.0) for prediction in predictions)
```

UPT has one fixed token bank. AB-UPT instead requires an
`OperatorBranchGraph` with typed conditioning/prediction branches and fixed
anchor capacities. CoDANO uses `OperatorFieldSpec` roles and a common tensor
latent grid; GNOT keeps separate source encoders and query-local gates. EqGINO
requires 3D coordinates, explicit `O3Representation` input/output contracts,
source quadrature, and a physical interaction radius. These are modeling
contracts, not interchangeable aliases.

For in-context use, build each `OperatorSupervisedExample` from an
`OperatorBatch` and target field, place examples in a fixed-capacity
`OperatorPrompt`, and pair it with the current batch in
`PromptedOperatorBatch`. The prompt mask distinguishes padding from a real
demonstration. No task-distribution pretraining is bundled.

For PDE-IR conditioning, construct `PDEConditionEncoder`, tokenize a canonical
equation as a `PDETokenBatch`, then call `attach_pde_condition`. The result is a
named one-anchor source branch, so the selected multi-input architecture must
declare and consume that branch; attaching it does not enforce the equation.

`tokenize_pde_ir` includes the execution-relevant PDE schema rather than only
the expression operators: coordinate bounds and periodicity, representations,
component and derivative axes, parameter vectors and scales, conditions,
regions, and nondimensionalization all affect the conditioning. Canonically
equivalent associative expressions produce identical tokens. Consistent
renaming of declared coordinates, fields, parameters, equations, conditions,
and regions does not change the encoder result because symbol identity is
represented relationally, not through lexical embeddings. Free-form problem
metadata remains outside the neural input.


```python
pde_encoder = phx.nn.operator.architectures.PDEConditionEncoder(
    width=8,
    depth=1,
    key=jr.key(35),
)
```

`GaussianFunctionOperator` is a distributional wrapper, not an uncertainty
upgrade for an arbitrary deterministic output. For a scalar diagonal-plus-rank-1
field distribution, its base must emit three channel-last parameters at every
query: mean, raw marginal scale, and one factor loading.

```python
gaussian_base = phx.nn.operator.architectures.UPT(
    in_channels="scalar",
    out_channels=3,
    coord_dim=1,
    width=8,
    num_tokens=4,
    depth=1,
    num_heads=2,
    source_key="forcing",
    key=jr.key(36),
)
gaussian_operator = phx.nn.operator.architectures.GaussianFunctionOperator(
    gaussian_base,
    out_channels="scalar",
    factor_rank=1,
)
field_distribution = gaussian_operator.distribution(measured_batch)
assert field_distribution.mean.shape == query_x.shape[:-1]
```

## N-dimensional FNO and resolution transfer

`FNO` accepts native case batches and keeps parameter count independent of grid
resolution. Use periodic axes for unpadded FFT operation; set `domain_padding`
for nonperiodic data.

!!! example
    ```python
    nx, ny = 48, 64
    x = jnp.linspace(0.0, 1.0, nx, endpoint=False)
    y = jnp.linspace(0.0, 1.0, ny, endpoint=False)
    coefficient = jr.normal(jr.key(2), (16, nx, ny, 2))

    fno = phx.nn.operator.architectures.FNO(
        n_modes=(12, 16),
        in_channels=2,
        out_channels=1,
        width=48,
        depth=4,
        factorization="tucker",
        rank=0.5,
        domain_padding=0.05,
        key=jr.key(3),
    )
    prediction = fno((coefficient, x, y))  # (16, nx, ny)

    # Zero-shot evaluation on a different grid uses the same parameters.
    x_fine = jnp.linspace(0.0, 1.0, 72, endpoint=False)
    y_fine = jnp.linspace(0.0, 1.0, 96, endpoint=False)
    coefficient_fine = jr.normal(jr.key(4), (16, 72, 96, 2))
    prediction_fine = fno((coefficient_fine, x_fine, y_fine))
    ```

For nonuniform or boundary-adapted axes, use `BasisSpectralConvND` with
`fourier`, `sine`, `cosine`, or `legendre` policies instead of treating an
arbitrary grid as uniformly periodic.

### Experimental higher-order Fourier mixing

Use `HOFNO` only for all-valid, uniformly spaced periodic tensor grids with
coincident source and query nodes. Its projected-product branch exposes
polynomial Fourier-mode interactions directly. De-aliasing is separate from
boundary padding: HOFNO rejects nonzero `domain_padding`.

!!! example
    ```python
    hofno_nodes = jnp.linspace(0.0, 1.0, 16, endpoint=False)
    hofno_source = jr.normal(jr.key(20), (2, 16, 16))

    hofno = phx.nn.operator.architectures.HOFNO(
        n_modes=(5, 5),
        width=8,
        depth=1,
        interaction_order=2,
        factor_bias=False,
        spectral_channel_mixing="depthwise",
        aliasing="dealiased",
        ffn_expansion=2,
        key=jr.key(21),
    )
    hofno_prediction = hofno((hofno_source, hofno_nodes, hofno_nodes))
    ```

`interaction_order=1` and `interaction_order=2` use the same block topology,
normalization, feed-forward expansion, and execution path, making them the
appropriate controlled pair. Use `aliasing="collocation"` only as an explicit
aliasing ablation. HOFNO is experimental-tier and explicit opt-in: preliminary
controlled results do not establish superiority on general PDE operators.

### Post-split square-symmetry augmentation

The `square_symmetry` benchmark can compare an ordinary FNO with the same FNO
trained using four rotational copies of each selected realization. Augmentation
is applied only after the physical train/validation/test split, so transformed
copies cannot leak across partitions. Validation and test realizations remain
unaugmented. This is a data intervention, not a claim of exact
architecture-level equivariance.

### Fixed-grid roadmap constructors

IFNO and axial-factorized FNO use the same canonical coincident-grid contract as
FNO. IFNO always executes its static iteration count; its convergence record is
diagnostic, not an adaptive stopping mechanism. WNO and MWT additionally fix
the transform shape at construction. MWT is one-dimensional.

```python
grid_nodes = jnp.linspace(0.0, 1.0, 8, endpoint=False)
grid_axis = phx.nn.operator.OperatorAxis(
    "x",
    grid_nodes,
    quadrature_weights=jnp.full((8,), 1.0 / 8.0),
    periodic=True,
)
grid_samples = phx.nn.operator.FunctionSamples(
    values=jnp.sin(2.0 * jnp.pi * grid_nodes),
    axes=(grid_axis,),
    mask=jnp.ones((8,), dtype=bool),
)
grid_batch = phx.nn.operator.OperatorBatch(
    inputs={"state": grid_samples},
    queries={"query": phx.nn.operator.FunctionSamples(
        values=None,
        axes=(grid_axis,),
        mask=jnp.ones((8,), dtype=bool),
    )},
)

ifno = phx.nn.operator.architectures.IFNO(
    n_modes=3,
    width=8,
    iterations=3,
    source_key="state",
    key=jr.key(40),
)
axial_fno = phx.nn.operator.architectures.AxialFactorizedFNO(
    n_modes=(3,),
    width=8,
    depth=1,
    source_key="state",
    key=jr.key(41),
)
wno = phx.nn.operator.architectures.WaveletNeuralOperator(
    1,
    in_channels="scalar",
    out_channels="scalar",
    levels=2,
    wavelet="db2",
    boundary="periodization",
    width=8,
    depth=1,
    source_key="state",
    key=jr.key(42),
)
mwt = phx.nn.operator.architectures.MultiwaveletOperator(
    in_channels="scalar",
    out_channels="scalar",
    order=2,
    levels=2,
    boundary="periodization",
    width=8,
    depth=1,
    source_key="state",
    key=jr.key(43),
)

ifno_prediction, ifno_convergence = ifno.evaluate_with_diagnostics(grid_batch)
fixed_grid_predictions = (
    ifno_prediction,
    axial_fno(grid_batch),
    wno(grid_batch),
    mwt(grid_batch),
)
assert all(prediction.shape == (8,) for prediction in fixed_grid_predictions)
assert ifno_convergence.iterations == 3
```

WNO and MWT own shape-independent numerical transforms. The same trained object
can be called on another compatible uniform coincident grid without rebuilding
its learned layers. New array shapes may trigger ordinary JAX recompilation.

`ManifoldSpectralOperator` takes a precomputed
`phydrax._spectral.SpectralDiscretization`; an optional target plan must use an
aligned Laplace eigenbasis. Build a mesh plan directly with
`phx.graph.spectral_discretization_from_triangle_mesh(mesh, n_modes=...)`.
There is no provider wrapper. A target plan is a declared cross-discretization,
not an arbitrary-coordinate query path.

SFNO similarly receives a prepared exact spherical sampling plan rather than
inferring a transform from arbitrary nodes:

```python
sphere_plan = phx.nn.operator.architectures.SphericalHarmonicPlan(
    16,
    sampling="mw",
    execution="recursive",
)
sfno = phx.nn.operator.architectures.SFNO(
    sphere_plan,
    width=16,
    depth=2,
    source_key="state",
    key=jr.key(44),
)
```

Construct the colatitude and longitude `OperatorAxis` objects from
`sphere_plan.theta`, `sphere_plan.phi`, and their corresponding quadrature
weights. SFNO rejects shifted nodes, missing samples, masks containing invalid
sites, and grids from a different sampling theorem.

For coincident irregular-time sequences, choose the state contract explicitly.
`DiagonalStateSpaceMixer` is input-independent; `SelectiveStateSpaceMixer`
learns input-dependent positive time scaling, injection, and readout while
remaining affine in latent state:

```python
times = jnp.asarray([0.0, 0.1, 0.35, 0.0, 0.8])
signal = jnp.sin(times)
valid = jnp.asarray([True, True, True, True, True])
reset = jnp.asarray([False, False, False, True, False])

selective = phx.nn.operator.architectures.SelectiveStateSpaceMixer(
    state_size=16,
    input_integration="linear",
    execution="associative",
    training_delta_range=(0.05, 0.4),
    key=jr.key(44),
)
prediction, step_diagnostics = selective.evaluate_with_diagnostics(
    signal,
    times,
    mask=valid,
    reset=reset,
)
assert step_diagnostics.segment_count == 2
```

A valid node after padding must declare `reset=True`; padding alone preserves
state and emits zero. The `OperatorBatch` path is coincident and does not infer
resets. Use the reported out-of-range interval fraction as extrapolation
evidence, not as an accuracy guarantee.

CNO consumes observation masks and physical quadrature in every convolution.
Keep missingness separate from a genuinely zero field value. Under full uniform
support the normalized route equals ordinary convolution; under sensor dropout
it renormalizes by observed non-negative measure.

Poseidon/scOT and DPOT are native architecture implementations initialized from
the supplied key. PhydraX does not bundle pretrained weights, large-scale
pretraining data, or evidence of foundation-model performance. DPOT corruption
is a pretraining primitive, not a bundled pretraining pipeline.

```python
poseidon = phx.nn.operator.architectures.Poseidon(
    image_shape=(8, 8),
    patch_size=2,
    embed_dim=4,
    depths=(1, 1),
    num_heads=(1, 2),
    window_size=2,
    skip_depths=(0,),
    key=jr.key(44),
)
dpot = phx.nn.operator.architectures.DPOT(
    image_shape=(8, 8),
    history_steps=3,
    forecast_steps=2,
    patch_size=2,
    embed_dim=8,
    depth=1,
    modes=(2, 2),
    num_blocks=2,
    out_layer_dim=4,
    normalization_groups=2,
    key=jr.key(45),
)
history = jnp.ones((8, 8, 3))
corrupted_history = phx.nn.operator.architectures.dpot_corrupt_history(
    history,
    noise_scale=1e-3,
    key=jr.key(46),
    mask=jnp.ones_like(history, dtype=bool),
    channel_axis=None,
)
assert corrupted_history.shape == history.shape
```

Poseidon requires one fixed, coincident 2D grid divisible by every patch and
multiscale merge. DPOT source values end in
`(height, width, history_time[, channels])`, and queries use the same spatial
grid plus exactly `forecast_steps` time nodes. `DPOT.corrupt_batch` is preferred
when an `OperatorBatch` already carries masks and axes because it preserves
that metadata.

Stable Koopman evolution and Green kernels solve different geometry problems:

```python
koopman = phx.nn.operator.architectures.KoopmanTemporalOperator(
    spatial_ndim=1,
    latent_size=4,
    hidden_size=8,
    depth=1,
    evolution="continuous",
    time_axis="time",
    source_key="state",
    key=jr.key(47),
)
green = phx.nn.operator.architectures.GreenKernelOperator(
    coord_dim=1,
    forcing_channels="scalar",
    boundary_channels="scalar",
    out_channels="scalar",
    width=8,
    depth=1,
    kernel_depth=1,
    forcing_key="forcing",
    boundary_key="boundary",
    key=jr.key(48),
)
```

Koopman source and query geometries must be tensor products with the same
spatial axis names/order and one explicit nonnegative query-time axis. Its
latent transition is contractive by construction; prediction accuracy and
conservation are not guaranteed. `GreenKernelOperator` accepts arbitrary query
coordinates but requires separate forcing and boundary inputs with
unnormalized physical quadrature. Boundary normals or condition descriptors are
value channels, and the learned kernel does not exactly enforce a PDE or
boundary condition.

All of these roadmap entries remain explicit-use only:

```python
assert phx.nn.operator.operator_architecture_status("FNO").recommendation_eligible
for architecture_name in (
    "IFNO",
    "WNO",
    "Poseidon",
    "DPOT",
    "Transolver++",
    "GNOT",
    "KoopmanTemporalOperator",
    "GreenKernelOperator",
):
    assert not phx.nn.operator.operator_architecture_status(
        architecture_name
    ).recommendation_eligible
```


## Irregular geometry with GINO, Geometry-Informed Flower, RIGNO, and GAOT

Use explicit point-cloud coordinates when the physical mesh changes between
cases. Coordinates have shape `case_shape + sample_shape + (coord_dim,)`;
quadrature and masks have `case_shape + sample_shape`. The following batch has
two independently deformed source clouds and two independent query clouds.

This executable survey evaluates several unrelated one-shot programs under
`jax.disable_jit()` so documentation validation does not spend most of its time
compiling throwaway shapes. Remove that scope for repeated training or deployment
calls and compile the stable model/batch signature instead.

```python
base = jnp.stack(
    jnp.meshgrid(
        jnp.linspace(0.0, 1.0, 4),
        jnp.linspace(0.0, 1.0, 3),
        indexing="ij",
    ),
    axis=-1,
).reshape((12, 2))
deformation = 0.04 * jnp.stack(
    (
        jnp.sin(jnp.pi * base[:, 0]) * jnp.sin(jnp.pi * base[:, 1]),
        jnp.sin(2.0 * jnp.pi * base[:, 0]) * jnp.sin(jnp.pi * base[:, 1]),
    ),
    axis=-1,
)
source_coordinates = jnp.stack((base, base + deformation))
source_values = (
    jnp.sin(jnp.pi * source_coordinates[..., 0])
    * jnp.cos(jnp.pi * source_coordinates[..., 1])
)

query_base = jnp.stack(
    (
        jnp.linspace(0.05, 0.95, 10),
        jnp.mod(0.17 + 0.37 * jnp.arange(10), 1.0),
    ),
    axis=-1,
)
query_coordinates = jnp.stack((query_base, query_base + 0.5 * deformation[:10]))
query_mask = jnp.array(
    [[True] * 10, [True] * 8 + [False, False]]
)
geometry_batch = phx.nn.operator.OperatorBatch(
    inputs={
        "forcing": phx.nn.operator.FunctionSamples(
            values=source_values,
            coordinates=source_coordinates,
            quadrature_weights=jnp.full((2, 12), 1.0 / 12.0),
        )
    },
    queries={"query": phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=query_coordinates,
        mask=query_mask,
    )},
    case_axes=("case",),
)

gino = phx.nn.operator.architectures.GINO(
    in_channels="scalar",
    out_channels="scalar",
    coord_dim=2,
    latent_shape=(4, 4),
    bounds_policy="global",
    latent_channels=4,
    modes=(2, 2),
    fno_width=4,
    fno_depth=1,
    encoder_neighbors=8,
    decoder_neighbors=8,
    transfer_width=8,
    transfer_depth=1,
    source_key="forcing",
    key=jr.key(20),
)
geometry_flower = phx.nn.operator.architectures.GeometryInformedFlower(
    in_channels="scalar",
    out_channels="scalar",
    coord_dim=2,
    latent_shape=(4, 4),
    boundary="clamp",
    bounds_policy="global",
    latent_channels=4,
    flower_width=4,
    flower_levels=2,
    flower_num_heads=1,
    flower_groups=1,
    encoder_neighbors=8,
    decoder_neighbors=8,
    transfer_width=8,
    transfer_depth=1,
    source_key="forcing",
    key=jr.key(21),
)

rigno = phx.nn.operator.architectures.RIGNO(
    in_channels="scalar",
    out_channels="scalar",
    coord_dim=2,
    regional_count=8,
    latent_channels=4,
    processor_neighbors=4,
    processor_depth=1,
    processor_width=8,
    processor_mlp_depth=1,
    encoder_neighbors=8,
    decoder_neighbors=8,
    transfer_width=8,
    transfer_depth=1,
    source_key="forcing",
    key=jr.key(22),
)
gaot = phx.nn.operator.architectures.GAOT(
    in_channels="scalar",
    out_channels="scalar",
    coord_dim=2,
    latent_shape=(4, 4),
    patch_shape=2,
    transfer_radius=0.5,
    transfer_scales=(1.0, 2.0),
    latent_channels=4,
    transformer_width=8,
    transformer_depth=1,
    transformer_heads=2,
    transfer_neighbors=8,
    transfer_width=8,
    transfer_heads=2,
    transfer_depth=1,
    source_key="forcing",
    key=jr.key(23),
)

with jax.disable_jit():
    geometry_predictions = {
        "gino": gino(geometry_batch),
        "geometry_flower": geometry_flower(geometry_batch),
        "rigno": rigno(geometry_batch),
        "gaot": gaot(geometry_batch),
    }
assert all(prediction.shape == (2, 10) for prediction in geometry_predictions.values())
assert all(
    jnp.allclose(prediction[1, 8:], 0.0)
    for prediction in geometry_predictions.values()
)
```

`global` tensor-grid bounds are computed jointly across the batch unless fixed
`latent_bounds` are supplied. For deployment, persist bounds fitted from the
training population rather than allowing a held-out case to redefine them.
`case_bbox` is appropriate only when affine case-relative coordinates are part
of the intended operator. RIGNO has no tensor bounds; its farthest-point regional
set is selected separately for every case.

`GeometryInformedFlower` does not infer mesh topology. To mask an embedded
irregular domain, add occupancy or signed-distance `FunctionSamples` to the
batch and name it with `latent_support_key`; this projects a hard support mask
onto the latent tensor grid. Declare case-only inputs with
`conditioning_channels`. `evaluate_with_diagnostics` returns both the geometry
transfer data and nested `FlowerDiagnostics`. End-to-end
`conserve_mass=True` requires explicit physical quadrature on the conserved
source and query.

The support field is geometry metadata, not a post-hoc output mask. Keep its
physical values unnormalized, use `source_mask_mode="renormalize"` when a thin
domain may occupy only part of a multilevel stencil, and name the physical
field whose integral must be conserved:

```python
ring_angle = jnp.linspace(0.0, 2.0 * jnp.pi, 16, endpoint=False)
ring_coordinates = 0.6 * jnp.stack(
    (jnp.cos(ring_angle), jnp.sin(ring_angle)),
    axis=-1,
)
ring_weights = jnp.full((16,), 2.0 * jnp.pi * 0.6 / 16.0)
support_axis = jnp.linspace(-1.0, 1.0, 8)
support_coordinates = jnp.stack(
    jnp.meshgrid(support_axis, support_axis, indexing="ij"),
    axis=-1,
).reshape((-1, 2))
domain_sdf = (
    jnp.abs(jnp.linalg.norm(support_coordinates, axis=-1) - 0.6) - 0.25
)
density = 1.0 + 0.2 * jnp.cos(3.0 * ring_angle)

conservative_batch = phx.nn.operator.OperatorBatch(
    inputs={
        "density": phx.nn.operator.FunctionSamples(
            values=density,
            coordinates=ring_coordinates,
            quadrature_weights=ring_weights,
        ),
        "domain_sdf": phx.nn.operator.FunctionSamples(
            values=domain_sdf,
            coordinates=support_coordinates,
            quadrature_weights=jnp.full((64,), 4.0 / 64.0),
        ),
    },
    queries={"query": phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=ring_coordinates,
        quadrature_weights=ring_weights,
    )},
)
conservative_flower = phx.nn.operator.architectures.GeometryInformedFlower(
    in_channels="scalar",
    out_channels="scalar",
    coord_dim=2,
    latent_shape=(8, 8),
    latent_channels=4,
    flower_width=4,
    flower_levels=1,
    flower_num_heads=1,
    flower_groups=1,
    encoder_neighbors=8,
    decoder_neighbors=8,
    transfer_width=8,
    transfer_depth=1,
    source_key="density",
    latent_support_key="domain_sdf",
    latent_support_kind="sdf",
    source_mask_mode="renormalize",
    conserve_mass=True,
    conservation_source_key="density",
    key=jr.key(24),
)
with jax.disable_jit():
    conservative_prediction, conservative_diagnostics = (
        conservative_flower.evaluate_with_diagnostics(conservative_batch)
    )
assert conservative_diagnostics.latent_mask is not None
assert jnp.allclose(
    jnp.sum(conservative_prediction * ring_weights),
    jnp.sum(density * ring_weights),
)
```

The projection conserves the mass represented by the **observed** source
samples. Sensor corruption or dropout can change that observed mass while the
clean physical target mass stays fixed. In that regime, first reconstruct the
missing mass or provide an independently known total; do not enable
`conserve_mass` merely to force a clean-target metric.

Neighborhood counts are static capacities. A radius applies an additional
physical-distance mask; it does not change the compiled edge-array shape.
Choose transfer radii in dimensional coordinates, include source quadrature,
and preserve query masks whenever cardinality varies. `assume_uniform_measure`
is an explicit approximation for point samples with genuinely equal cell
measure, not a remedy for missing mesh volumes.

## Typed cochains, metric DEC routes, and topological PINO

Use `CochainNeuralOperator` when unknowns live on different cell degrees of one
oriented complex—for example, vertex pressure and edge flux. Do not flatten
these fields into an untyped point cloud: incidence signs, Hodge-star measures,
boundary cells, and degree determine both the neural operator and its physics
loss.

```python
cochain_vertices = jnp.asarray(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
)
cochain_faces = jnp.asarray([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32)
cochain_complex = phx.graph.triangle_mesh_to_cochain_complex(
    cochain_vertices,
    cochain_faces,
)
forcing_values = jnp.arange(8.0).reshape(2, 4)
cochain_batch = phx.nn.operator.OperatorBatch(
    inputs={
        "forcing": phx.nn.operator.function_samples_from_cochain(
            cochain_complex,
            0,
            values=forcing_values,
        )
    },
    queries={
        "vertices": phx.nn.operator.function_samples_from_cochain(
            cochain_complex,
            0,
            values=None,
        ),
        "edges": phx.nn.operator.function_samples_from_cochain(
            cochain_complex,
            1,
            values=None,
        ),
    },
    case_axes=("case",),
    case_shape=(2,),
)
zero_form = phx.graph.CochainFieldSpec(
    0,
    cell_orientation="invariant",
    sampling="point_value",
)
one_form = phx.graph.CochainFieldSpec(
    1,
    cell_orientation="signed",
    sampling="cell_integral",
)
cochain_fields = (
    phx.nn.operator.OperatorFieldSpec(
        "forcing",
        role="source",
        source_name="forcing",
        cochain=zero_form,
    ),
    phx.nn.operator.OperatorFieldSpec(
        "pressure",
        role="target",
        query_name="vertices",
        cochain=zero_form,
    ),
    phx.nn.operator.OperatorFieldSpec(
        "flux",
        role="target",
        query_name="edges",
        cochain=one_form,
    ),
)
cochain_task = phx.nn.operator.OperatorTask(
    "mixed-darcy",
    fields=cochain_fields,
    queries=(
        phx.nn.operator.OperatorQuerySpec(
            "vertices",
            geometry_kind="cell_complex",
            coordinate_components=("x", "y"),
            topology_site="cell",
            quadrature="physical_required",
        ),
        phx.nn.operator.OperatorQuerySpec(
            "edges",
            geometry_kind="cell_complex",
            coordinate_components=("x", "y"),
            topology_site="cell",
            quadrature="physical_required",
        ),
    ),
    problem=phx.nn.operator.OperatorProblemSpec(
        source_query_relation="shared_topology",
        query_is_fixed=False,
        requires_resolution_transfer=True,
    ),
)
cochain_operator = phx.nn.operator.architectures.CochainNeuralOperator(
    cochain_fields,
    width=2,
    depth=1,
    key=jr.key(25),
)
cochain_prediction = cochain_operator.predict(cochain_batch)
assert cochain_prediction.field("pressure").values.shape == (2, 4)
assert cochain_prediction.field("flux").values.shape == (2, 5)
```

The default route configuration uses self, exterior-derivative,
codifferential, and split Hodge-Laplacian paths. Harmonic projection is opt-in
because it requires topology-level preprocessing:
`compute_harmonic_subspace(complex_ir)` followed by
`complex_ir.with_harmonic_subspace(...)` and
`TopologicalRouteConfig(harmonic=True)`. Absolute and relative boundary
policies are distinct operators; use the same policy when constructing samples,
the neural operator, and physics residuals.

### Physics-only topological operator fitting

`CochainResidualProgram` lets a fixed-complex DEC PINN and an operator-level
topological PINO execute one residual definition. Operator inputs bind each
declared program field either to a predicted output or to a sampled source.
Sparse per-degree samples are scattered onto the canonical cell complex,
degree-masked, evaluated, and then gathered through a Hodge-aware segmented
reduction.

```python
def mixed_darcy_residual(graph, fields, *, key):
    del key
    pressure_gradient = phx.graph.cochain_exterior_derivative(
        graph,
        fields["pressure"],
        0,
        boundary_policy="absolute",
    )
    flux_divergence = phx.graph.cochain_codifferential(
        graph,
        fields["flux"],
        1,
        boundary_policy="absolute",
    )
    return {
        "constitutive": fields["flux"] + pressure_gradient,
        "mass": flux_divergence - fields["forcing"],
    }


darcy_program = phx.graph.CochainResidualProgram(
    inputs={
        "pressure": zero_form,
        "flux": one_form,
        "forcing": zero_form,
    },
    outputs={
        "constitutive": one_form,
        "mass": zero_form,
    },
    residual_fn=mixed_darcy_residual,
    identity="cookbook.operator.mixed_darcy.v1",
)
residual_inputs = {
    "pressure": phx.nn.operator.training.CochainResidualInput("prediction", "pressure"),
    "flux": phx.nn.operator.training.CochainResidualInput("prediction", "flux"),
    "forcing": phx.nn.operator.training.CochainResidualInput("source", "forcing"),
}
physics_losses = (
    phx.nn.operator.training.CochainResidualLoss(
        name="constitutive",
        program=darcy_program,
        inputs=residual_inputs,
        output="constitutive",
        reduction="metric_mean",
    ),
    phx.nn.operator.training.CochainResidualLoss(
        name="mass",
        program=darcy_program,
        inputs=residual_inputs,
        output="mass",
        reduction="metric_sum",
    ),
)
targetless = phx.nn.operator.training.OperatorDataset(
    cochain_batch,
    phx.nn.operator.OperatorTargetBatch.from_arrays({}, cochain_batch),
)
fit = phx.nn.operator.training.fit_operator(
    cochain_operator,
    targetless,
    task=cochain_task,
    training_evidence=phx.nn.operator.OperatorTrainingEvidence("task_specific"),
    loss_terms=physics_losses,
    normalization=None,
    batch_size=1,
    epochs=1,
    steps=1,
)
```

Targetless fitting is allowed only with explicit physics loss terms. It cannot
use `normalization="fit"` because no supervised output targets exist from which
to estimate output scale; pass physical nondimensionalization explicitly or a
previously fitted normalization policy. Program identity, field bindings,
reduction, optional topology lock, and every other loss setting enter the exact
checkpoint contract, so changing the residual rejects resume rather than
silently continuing a different optimization problem.

`graph_mean` treats every cell equally within a complex, `metric_mean`
normalizes by Hodge-star mass, and `metric_sum` retains that mass. Every
nonempty complex contributes one segment regardless of mesh cardinality, and
padded cells are excluded. Set `topology_fingerprint` on a
`CochainResidualLoss` when training must remain locked to one exact canonical
complex; omit it for topology-varying batches that still satisfy the declared
field schema.

The benchmark ladders `cochain_mixed_darcy` and
`cochain_annulus_harmonic` exercise, respectively, joint zero-/one-form
prediction under mesh refinement and a rank-one harmonic projection on a
noncontractible annulus. The benchmark emits named pressure/flux/harmonic field
metrics rather than concatenating physically different cochain spaces.

## Multiple inputs and POD

Pass a branch mapping to `DeepONet`. `fusion="product"` gives the MIONet
combiner; `fusion="sum"` and `fusion="concat"` cover additive and learned fusion.
A `PODBasis` replaces the learned trunk when a fixed reduced output basis is
available.

```python
pod_modes = jnp.ones((query_x.shape[0], latent))
pod = phx.nn.operator.architectures.PODBasis(pod_modes, latent_size=latent)
model = phx.nn.operator.architectures.DeepONet(
    branch={
        "initial": branch,
        "forcing": branch,
        "coefficient": branch,
    },
    trunk=pod,
    coord_dim=1,
    latent_size=latent,
    fusion="product",
)
multi_input_batch = phx.nn.operator.OperatorBatch(
    inputs={"initial": source, "forcing": source, "coefficient": source},
    queries={"query": query},
)
prediction = model(multi_input_batch)
```

## Data plus native physics at different resolutions

PINO is expressed by composing terms around one operator-backed
`DomainFunction`. The supervised term may use a coarse query set while the
physics term uses a separate fine query set.

```python
coarse_batch = batch
coarse_target = jnp.zeros(query.sample_shape)
fine_physics_batch = batch

data_term = phx.terms.OperatorDatasetTerm(
    "u",
    coarse_batch,
    coarse_target,
    loss="l2",
    relative=False,
)

# Domain.Model retains the trainable operator in the solver function tree.
operator_function = geom.Model("x")(model)
physics_term = phx.terms.DifferentialPhysicsInformedOperatorTerm(
    "u",
    fine_physics_batch,
    geom,
    "x",
    lambda u: phx.operators.laplacian(u, var="x"),
    loss="l2",
)
terms = phx.terms.operator_term_suite(data_term, physics_term)
```

`DifferentialPhysicsInformedOperatorTerm` fixes every source function in
the `OperatorBatch`, constructs differentiable point queries, and applies the
residual `DomainFunction` at the query coordinates. Ordinary PhydraX
differential operators therefore remain the single derivative implementation.
For direct inspection:

```python
context = phx.nn.operator.adapters.bind_operator_context(model, fine_physics_batch)
u_for_this_source = context.domain_function(geom, "x")
laplacian_u = phx.operators.laplacian(u_for_this_source, var="x")
```

Each loss first reduces coordinates for one physical case and then averages
cases. This avoids changing the objective merely by changing points per case.

## Reproducible production training

Fit normalization on the training partition only, then pass the same persisted
policy to validation, inference, and checkpointing:

```python
training_axis = phx.nn.operator.OperatorAxis("x", jnp.linspace(0.0, 1.0, K))
training_values = coeffs
training_targets = 2.0 * coeffs
training_model = phx.nn.operator.architectures.FNO(
    width=8,
    depth=1,
    n_modes=(3,),
    key=jr.key(10),
)
dataset = phx.nn.operator.training.operator_dataset_from_arrays(
    {"forcing": training_values},
    {"output": training_targets},
    source_axes={"forcing": (training_axis,)},
    query_axes=(training_axis,),
)
split = phx.nn.operator.training.split_operator_dataset(
    dataset,
    policy=phx.nn.operator.training.OperatorSplitPolicy(seed=1729),
)
normalization = phx.nn.operator.training.fit_operator_normalization(
    split.train.batch,
    split.train.targets,
)
normalization_path = phx.nn.operator.training.save_operator_normalization(
    "/tmp/phydrax-operator-normalization.json",
    normalization,
)

dtype_policy = phx.nn.operator.training.OperatorDTypePolicy(
    parameter_dtype="float32",
    compute_dtype="float32",
    reduction_dtype="float64",
)
loader = phx.nn.operator.training.OperatorBatchLoader(
    split.train,
    batch_size=16,
    seed=1729,
    normalization=normalization,
    dtype_policy=dtype_policy,
    prefetch=2,
)
with loader.epoch(0) as batches:
    for training_batch in batches:
        prediction = training_model(training_batch.batch)
```

`fit_operator` is the owner of optimization, dynamic loss terms, validation,
best-model selection, gradient accumulation, mixed precision, sharding, callbacks,
and exact resume. The loader remains useful for inspection, but production code
should not build a second optimizer loop around it.

The loader uses a versioned, stateless position-to-case permutation, so memory
does not scale with the number of cases and `loader.epoch(epoch,
start_batch=...)` begins directly at a saved suffix. `prefetch` bounds an
ordered host-side queue; it changes timing only, not `loader.fingerprint`.
In-memory sources permit one background reader. Callback sources remain
synchronous unless their adapter explicitly declares `background_read_safe=True`.
Use the context-manager form whenever iteration may stop early so its producer
is closed deterministically.

Keep loader timing separate from model-quality benchmarks:

```console
python -m tools.operator_benchmarks.data_plane \
    --cases 4096 --batch-size 64 --resolution 128 --prefetch 2
```

The record separates ordering compilation, cold and cached fingerprints,
first-batch latency, steady-state throughput, host and device memory, and the
pre-I/O resume gate. Controlled reader and consumer latencies can be supplied
to measure overlap without mixing model accuracy into the data-plane result.

`fit_operator` remains the production optimization owner:

```python
fit_result = phx.nn.operator.training.fit_operator(
    training_model,
    split.train,
    validation=split.validation,
    epochs=1,
    steps=1,
    batch_size=16,
    normalization=normalization,
    dtype_policy=dtype_policy,
    validation_policy=phx.nn.operator.training.OperatorValidationPolicy(every=1),
    key=jr.key(11),
)
training_model = fit_result.execution_model
assert fit_result.completed_steps == 1
```

Pass an `OperatorTask` plus matching `OperatorTrainingEvidence` to bind physical
dimensions, source/query roles, output fields, and the admissible deployment
regime. The task contract separates the structural source/query relation
(`coincident`, `shared_topology`, or `independent`) from whether query geometry is
fixed across the dataset. Both values must be explicit for fitting. A fixed-query
task also sets `fixed_geometry=True` on every query; fitting rejects geometry
changes across cases or batches, and `TrainedOperator` persists and enforces the
physical geometry fingerprints at inference.

In task-bound mode `fit_result.trained_operator` is a `TrainedOperator`;
`prepare` performs host-side contract validation once and `predict_prepared` is
the compiled hot path. Its prediction is dimensionalized and retains named
queries, fields, case axes, normalization, dtype, and provenance contracts.
`fit_result.execution_model` and `last_execution_model` are the selected and last
unconstrained execution-space models.

`loss_terms` accepts `SupervisedOperatorLoss`, general `OperatorLossTerm`, and
`WeakOperatorLoss`. Every callable loss needs a stable identity because its
fingerprint participates in checkpoint compatibility. Loss terms default to
physical space; set `space="execution"` only for an objective intentionally
defined on normalized execution values. `OperatorLossContext` exposes paired
execution and physical predictions, batches, and targets. Set
`gradient_accumulation` for exact case-weighted microbatch accumulation; use
`OperatorMixedPrecisionPolicy` for dynamic loss scaling and
`OperatorShardingPolicy` to shard a named case dimension while keeping
parameters and shared geometry replicated.

### Exact output transforms, weak forms, and operator adjoints

`OperatorOutputPipeline` is a physical-space deployment contract owned by
`fit_operator` and `TrainedOperator`, not an execution-model wrapper. Pass it as
`output_pipeline=` during fitting so supervised and physics losses see the same
constrained physical prediction used at inference. A hard transform uses
`lift + envelope * raw`; a conservation transform projects with physical
quadrature and masks. When both invariants are required, place conservation
after the hard transform and supply a correction basis that vanishes wherever
the hard boundary must remain fixed:

```python
def boundary_envelope(coordinates, batch, *, key):
    del batch, key
    x = coordinates[..., 0]
    return x * (1.0 - x)

physics_latent = 8
physics_branch = phx.nn.operator.architectures.IntegralBranchEncoder(
    feature_model=phx.nn.models.MLP(
        in_size=2,
        out_size=physics_latent,
        width_size=16,
        depth=1,
        key=jr.key(12),
    ),
    latent_size=physics_latent,
    coord_dim=1,
)
physics_base = phx.nn.operator.architectures.DeepONet(
    branch={"forcing": physics_branch},
    trunk=phx.nn.models.MLP(
        in_size=1,
        out_size=physics_latent,
        width_size=16,
        depth=1,
        key=jr.key(13),
    ),
    coord_dim=1,
    latent_size=physics_latent,
)

output_pipeline = phx.nn.operator.training.OperatorOutputPipeline(
    phx.nn.operator.training.HardConstraintTransform(
        "output",
        envelope_fn=boundary_envelope,
        identity="homogeneous-dirichlet-v1",
    ),
    phx.nn.operator.training.ConservationProjection(
        "output",
        source_name="forcing",
        correction_fn=boundary_envelope,
        identity="dirichlet-compatible-integral-v1",
    ),
)
physics_task = phx.nn.operator.OperatorTask(
    "constrained-independent-query",
    fields=(
        phx.nn.operator.OperatorFieldSpec(
            "forcing",
            role="source",
            source_name="forcing",
        ),
        phx.nn.operator.OperatorFieldSpec(
            "output",
            role="target",
            query_name="query",
        ),
    ),
    queries=(
        phx.nn.operator.OperatorQuerySpec(
            "query",
            geometry_kind="point_cloud",
            coordinate_components=("x",),
            coordinate_dimensions=((),),
        ),
    ),
    problem=phx.nn.operator.OperatorProblemSpec(
        source_query_relation="independent",
        query_is_fixed=False,
    ),
)
physics_model = phx.nn.operator.training.TrainedOperator(
    physics_base,
    physics_task,
    training_evidence=phx.nn.operator.OperatorTrainingEvidence("task_specific"),
    output_pipeline=output_pipeline,
)
physics_prediction = physics_model.predict(measured_batch, key=jr.key(14))
physics_values = physics_prediction.field("output").values

source_total = phx.nn.operator.training.operator_integral(
    measured_batch.input("forcing").values,
    measured_batch.input("forcing"),
)
predicted_total = phx.nn.operator.training.operator_integral(
    physics_values,
    measured_batch.query("query"),
)
assert jnp.allclose(predicted_total, source_total)
assert jnp.allclose(physics_values[0], 0.0)
```

Transform identities are part of the model contract. Later transforms may
invalidate earlier invariants unless their correction space is compatible;
the explicit vanishing correction basis above preserves the Dirichlet ansatz.
Masked/padded sites remain zero and contribute no conservation measure.

Use `operator_weak_form_loss` for precomputed residuals and test functions, or
`WeakOperatorLoss` for a dynamic term inside `fit_operator`. Test functions have
shape `sample_shape + (num_tests,)` (or the corresponding case-prefixed shape);
normalization makes the objective invariant to rescaling an individual test
function. `space="physical"` selects the physical query measure.

`operator_hilbert_inner_product`, `operator_hilbert_norm`, and
`operator_hilbert_relative_error` implement masked, quadrature-aware real or
complex Hilbert metrics with an optional channel metric. `linearize_operator`
returns matrix-free JVP, Euclidean/Hermitian pullback, and Hilbert-adjoint
actions. For a `TrainedOperator`, this is the derivative from physical source
values to dimensionalized physical outputs:

```python
linearization = phx.nn.operator.training.linearize_operator(
    physics_model,
    measured_batch,
    "forcing",
)
source_perturbation = jnp.ones_like(measured_batch.input("forcing").values)
output_perturbation = linearization.pushforward(source_perturbation)
assert output_perturbation.shape == physics_values.shape
```

### Lazy cases and streamed query decoding

For datasets backed by files, databases, or a simulator, implement
`CallbackOperatorCaseSource` with separate metadata and case readers. Combine it
with `AnchorQuerySamplingPolicy` and `read_operator_case_batch`; only requested
case indices and sampled source/query points are read. Values, coordinates,
quadrature, and masks remain one selection unit.

A callback source must provide `content_fingerprint`, for example a SHA-256
digest over an immutable dataset revision, adapter version, and per-file
checksums. The digest must be available without metadata or case reads.
Provenance alone is not content identity: target values, geometry, measures,
masks, topology, and output specifications can change while provenance stays
constant. Opt in to `background_read_safe=True` only when one dedicated reader
thread may call the adapter safely and every read finishes in finite time.

Encoded operators can evaluate query sets in fixed-capacity chunks without
re-encoding the source:

```python
query_source = phx.nn.operator.training.ArrayOperatorQuerySource(
    measured_batch.query("query"),
    case_shape=measured_batch.case_shape,
    fingerprint="cookbook-query-v1",
)
prediction_sink = phx.nn.operator.training.ArrayPredictionSink()
streamed_prediction = phx.nn.operator.training.decode_query_chunks(
    nomad,
    measured_batch,
    query_source,
    prediction_sink,
    chunk_size=2,
    compile=True,
)
assert streamed_prediction.shape == predictions[-1].shape
assert prediction_sink.metadata.query_fingerprint == "cookbook-query-v1"
```

The final chunk is padded and masked internally, so padding contributes neither
output nor physical measure. Use `NpyPredictionSink` when the assembled output
must remain off device and outside process memory.

`save_operator_training_checkpoint` persists the model, optimizer and
gradient-accumulation state, exact PRNG key, normalization, dtype policy,
semantic fit schema, logical epoch/batch cursor, loader identity, and user
metadata in the current versioned format. Resume validates the manifest,
version, state checksum, source content, ordering algorithm, and fit contract
before case I/O. It then reads the exact next batch once and uses it for the
first resumed update. A successful save publishes the new manifest atomically
and prunes superseded state blobs, so periodic validation retains one resumable
state per trial. Old checkpoint formats are rejected rather than migrated.

`save_operator_artifact` stores the execution model, physical output pipeline and
fingerprint, fixed-query geometry fingerprints, normalization, dtype, evidence,
and optional exact-resume state as one verified contract. Portable recipes use
versioned architecture and value registry identities rather than defining-module
paths. Only the current canonical representation is accepted; regenerate
development artifacts when that representation changes.

## Generalized Flower transport

The default `Flower` remains the paper-faithful aligned-grid model. Opt into
resolution-consistent transitions and independent query interpolation
explicitly:

```python
import jax
import jax.numpy as jnp
import phydrax as phx

source_axis = phx.nn.operator.OperatorAxis(
    "x",
    jnp.arange(16) / 16,
    quadrature_weights=jnp.full((16,), 1 / 16),
    periodic=True,
)
query_axis = phx.nn.operator.OperatorAxis(
    "x",
    jnp.arange(24) / 24,
    quadrature_weights=jnp.full((24,), 1 / 24),
    periodic=True,
)
source_values = jnp.sin(2 * jnp.pi * source_axis.nodes)
batch = phx.nn.operator.OperatorBatch(
    inputs={
        "state": phx.nn.operator.FunctionSamples(
            values=source_values,
            axes=(source_axis,),
        )
    },
    queries={
        "query": phx.nn.operator.FunctionSamples(values=None, axes=(query_axis,))
    },
)
flower = phx.nn.operator.architectures.Flower(
    in_channels="scalar",
    out_channels="scalar",
    spatial_ndim=1,
    boundary="periodic",
    width=8,
    levels=2,
    num_heads=2,
    groups=2,
    source_key="state",
    transition_mode="resolution_consistent",
    query_mode="interpolate",
    conserve_mass=True,
    key=jax.random.key(0),
)
prediction, diagnostics = flower.evaluate_with_diagnostics(batch)

source_mass = jnp.sum(
    source_values * source_axis.quadrature_weights
)
query_mass = jnp.sum(
    prediction * query_axis.quadrature_weights
)
assert prediction.shape == (24,)
assert jnp.allclose(source_mass, query_mass)
assert len(diagnostics.blocks) == 2 * flower.levels - 1
```

For arbitrary point coordinates, attach physical `quadrature_weights` to the
query whenever `conserve_mass=True`. A nonuniform periodic source axis likewise
needs positive axis quadrature weights whose sum is its period. Use
`source_mask_mode="renormalize"` to interpolate around missing samples or
`"strict"` to invalidate any stencil crossing a hole.

Probabilistic routes are sampled only when a key is supplied. Passing the same
key to `evaluate_with_diagnostics` makes its recorded displacement exactly the
route used for the returned prediction; keyless evaluation uses the mean route.

## Audited operator benchmarks

Use Operator Benchmark v2 artifacts for architecture decisions.
The three profiles have different contracts:

| Profile | Purpose | Promotion eligible |
| --- | --- | --- |
| `smoke` | Fast API and artifact check on tiny populations | No |
| `shortlist` | Remove clearly unsuitable families before expensive runs | No |
| `decision` | 128 independently generated physical cases, five seeds, pinned provenance, and resumable training | Yes |

### Intended Flower transport and wave portfolio

The benchmark portfolio is intended to compare three explicit Flower
configurations. This is a benchmark plan, not a claim that result artifacts
have already been generated:

| Configuration | Semantics |
| --- | --- |
| Paper-faithful one-level Flower | `levels=1`, `boundary="periodic"`, `query_mode="coincident"`, `probabilistic_routing=False`, and the default `transition_mode="learned"` |
| Paper-faithful multilevel Flower | `levels>1` with learned stride-two transitions, `boundary="periodic"`, `query_mode="coincident"`, and `probabilistic_routing=False` |
| Resolution-consistent Flower | An explicit multilevel opt-in with `transition_mode="resolution_consistent"` and the same periodic, coincident-grid contract; it does not replace or silently change the paper-faithful default |

The one-level and multilevel paper-faithful variants remain the default Flower
semantics. Resolution-consistent transitions are a separate ablation for
regular-grid resolution transfer. Conservation is evaluated as a scientific
property; it must not be inferred merely from an architecture label or silently
forced by changing the paper-faithful configuration.

The architecture-neutral transport/wave ladders are:

| Ladder | Scenario fields and direct scientific checks |
| --- | --- |
| Smooth constant-speed periodic advection | Grid resolution, time step, horizon, periodic initial modes, speed, and expected translation/phase; check finite targets, analytic translation/phase error, and periodic mass conservation within the declared numerical tolerance |
| Smooth variable-speed periodic advection | Grid resolution, time step, horizon, periodic initial modes, and the speed law or sampled speed range; check finite targets, the expected variable-speed phase/translation, and periodic mass conservation |
| Viscous periodic Burgers shock formation and rollout | Grid resolution, time step, rollout horizon, viscosity or Reynolds-like range, and initial modes; check finite targets, shock steepening followed by viscous smoothing, long-rollout behavior, and periodic mass conservation |
| Periodic acoustic waves | Grid resolution, time step, horizon, acoustic wave speed, and wavenumber content; check finite targets, propagated phase, periodicity, and physically consistent energy behavior |
| Controlled periodic polynomial Poisson | Band-limited source `v`, degree `p`, and zero-mean solution of `-Δu = v**p - mean(v**p)`; targets come from an oversampled Fourier solve followed by orthogonal projection. Compare the order-one control, order-two collocation ablation, and order-two dealiased HOFNO under matched training and artifact protocols |

Every level should retain deterministic seeds and case IDs, the resolved
physical/discretization ranges, and a scenario checksum. Provenance should
identify the governing equation, boundary convention, initial-condition
population, and reference construction. Use exact analytic translation or
phase evidence where it exists; otherwise require passing reference-solver
residual and refinement evidence. Resolution shifts must reuse the same
physical realization so a grid change is not confounded with a new PDE case.

These ladders reuse the existing runner, scenario dataclasses, provenance and
checksum records, difficulty levels, and conjunctive promotion gates. They do
not introduce a Flower-specific promotion path. Existing quick/smoke coverage
stays small and promotion-ineligible, and the shortlist and decision guidance
below is unchanged.

Comparator inclusion is compatibility-gated, not automatic. FNO and IFNO can
participate on aligned uniform coincident grids but do not provide arbitrary
independent-query decoding. CNO and UNO additionally require compatible grid
hierarchies and divisibility. WNO and MWT can reuse a model across compatible
sample counts, but transformed-axis count, uniform coincident geometry, level
depth, filter or polynomial order, and boundary rules remain fixed.
Consequently, FNO/IFNO/CNO/UNO/Wavelet results should appear only for levels
whose grid semantics satisfy those contracts; absence from an incompatible
level is not a failed run.

Run a broad shortlist with Pareto reporting before committing decision-profile
compute:

```console
uv run python -m tools.operator_benchmarks --v2 \
  --benchmark-profile shortlist \
  --ladders smooth_periodic --difficulty hard \
  --architectures constant,nearest_neighbor,fno,tfno,cno,uno \
  --comparison pareto --size-scales 0.75,1.0,1.5 \
  --steps 300 --learning-rates 0.0003,0.001,0.003 \
  --seeds 0,1,2,3,4 --repeats 5 \
  --validation-interval 25 --patience 8 \
  --relative-minimum-delta 0.0001 \
  --parity-evidence tools/operator_benchmarks/reference/family_parity.json \
  --output artifacts/operator-shortlist \
  --commit-identity <immutable-revision>
```

For irregular-domain operators, run the manufactured deformed-elliptic ladder
separately. It audits Jacobian positivity, conservative quadrature, deformation
provenance, independent source/query clouds, physical population splits, and
isolated geometry, sensor, and boundary-condition shifts:

```console
uv run python -m tools.operator_benchmarks --v2 \
  --benchmark-profile shortlist \
  --ladders irregular_geometry --difficulty hard \
  --architectures constant,nearest_neighbor,gino,rigno,gaot \
  --comparison pareto --size-scales 0.75,1.0,1.5 \
  --steps 300 --learning-rates 0.0003,0.001,0.003 \
  --seeds 0,1,2,3,4 --repeats 5 \
  --validation-interval 25 --patience 8 \
  --relative-minimum-delta 0.0001 \
  --parity-evidence tools/operator_benchmarks/reference/family_parity.json \
  --output artifacts/geometry-operator-shortlist \
  --commit-identity <immutable-revision>
```

Every trial stores the resolved latent shape, regional count, bounds policy,
neighbor capacities, physical radii, scale set, patch shape, and processor
depth. Compare artifacts only when those resolved configurations and the
scenario checksum are visible.

Run typed cochain models only on the cell-complex ladders. The pointwise route
is the locality-free control; `cochain_no_harmonic` is included only on the
annulus ladder; the full model enables exact harmonic projection there:

```console
uv run python -m tools.operator_benchmarks --v2 \
  --benchmark-profile shortlist \
  --ladders cochain_mixed_darcy,cochain_annulus_harmonic \
  --difficulty all \
  --architectures cochain_pointwise,cochain_no_harmonic,cochain_neural_operator \
  --comparison pareto --size-scales 0.75,1.0,1.5 \
  --steps 300 --learning-rates 0.0003,0.001,0.003 \
  --seeds 0,1,2,3,4 --repeats 5 \
  --validation-interval 25 --patience 8 \
  --relative-minimum-delta 0.0001 \
  --output artifacts/cochain-operator-shortlist \
  --commit-identity <immutable-revision>
```

Rerun only the surviving families with `--benchmark-profile decision`. A
decision run requires an immutable commit identity and an output or explicit
checkpoint directory. It writes a checkpoint after each validation and at the
final step. Resume the exact model, optimizer, PRNG key, elapsed time, and
learning/validation curves with:

```console
uv run python -m tools.operator_benchmarks --v2 \
  --benchmark-profile decision \
  --ladders independent_query --difficulty hard \
  --architectures constant,nearest_neighbor,deeponet,local_integral \
  --comparison pareto --size-scales 0.75,1.0,1.5 \
  --steps 300 --learning-rates 0.0003,0.001,0.003 \
  --seeds 0,1,2,3,4 --repeats 5 \
  --validation-interval 25 --patience 8 \
  --relative-minimum-delta 0.0001 \
  --parity-evidence tools/operator_benchmarks/reference/family_parity.json \
  --output artifacts/operator-decision \
  --commit-identity <immutable-revision> --resume
```

Do not combine raw corruption and missing-sensor robustness.
`sensor_corruption` zeroes values without changing their observation mask.
The full standard benchmark reports nested 10%, 30%, and 50% mask-aware
`sensor_dropout` evaluations; quick mode retains one 30% smoke point. To measure
whether missingness-aware training is useful, run a separate artifact set with
`--train-sensor-dropout 0.2`; never overwrite the unaugmented run. The
`irregular_causal_relaxation_scenario` separately probes nonuniform physical
steps, step-range extrapolation, and ragged schedules for temporal mixers.

Capacity matching chooses from the architecture-specific size grid. A requested
target outside the common feasible parameter interval fails before training;
the runner never labels an incomparable target as matched. Compute matching
uses compiled JAX/XLA loss-and-gradient FLOPs and records accessed bytes.
Pareto mode sweeps all requested size scales and reports validation error,
worst shifted-test error, training FLOPs, inference latency, parameter count,
and backend peak memory. If a backend cannot measure memory, the point remains
explicitly incomplete and receives no fabricated dominance label.

### Square-symmetry augmentation benchmark

The `square_symmetry` ladder contains an isotropic scalar diffusion operator
with `p4m` symmetry, a chiral `p4` operator that intentionally breaks
reflections while retaining rotations, and anisotropic controls whose declared
symmetry is intentionally absent. Reference actions are checked before
training, and physical realization IDs are split before transformed partners
are generated.

Compare ordinary FNO with the same architecture trained using post-split `p4`
augmentation:

```console
uv run python -m tools.operator_benchmarks --v2 \
  --benchmark-profile decision \
  --ladders square_symmetry --difficulty all \
  --architectures fno,fno_p4_augmented \
  --comparison pareto --size-scales 0.5,0.75,1.0,1.5,2.0 \
  --sample-fractions 0.25,0.5,1.0 \
  --steps 1000 --learning-rates 0.0003,0.001,0.003 \
  --seeds 0,1,2,3,4 --repeats 5 \
  --validation-interval 25 --patience 8 \
  --output artifacts/fno-square-symmetry \
  --commit-identity <immutable-revision>
```

`sample_fractions` counts base physical training realizations. Augmented FNO
receives four transformed versions of each selected realization, and compute
matching measures that larger batch. Paired group-action defects remain
diagnostics: data augmentation does not guarantee an equivariant model.


Hyperparameters and early stopping use validation loss only. Shifted test
evaluations run once after the final validation choice, so test metrics cannot
influence selection. Plateau stopping uses both an absolute and relative
minimum improvement and restores the best validation checkpoint.

Each output directory contains one JSON record and nine Parquet tables:
aggregates, trials, sample-efficiency curves, paired symmetry defects,
symmetry-promotion decisions, scenario difficulty, Pareto fronts, per-scenario
promotions, and portfolio promotions. Trial rows retain the complete training
and validation curves, convergence state, resume point, selected size scale,
and comparison measurements.
Promotion requires scenario integrity, baseline
hardness, source/target rank, realization novelty, five-seed convergence,
accuracy, robustness, efficiency, matching, parity, and provenance gates.

## Uncertainty-aware operator evaluation

Operator predictions can retain epistemic ensemble or dropout draws, uncertain
source-function draws, posterior samples, and conditional observation variance
without flattening case/query/channel geometry. Calibrate and score complete
physical output cases through `OperatorPredictiveField`; do not reinterpret query
points as independent samples.

The dedicated [neural-operator uncertainty recipe](operator_uncertainty.md) covers:

- homogeneous and heterogeneous operator ensembles;
- coherent MC dropout and crossed input/epistemic axes;
- fixed normalized observation likelihoods and selected-parameter posteriors;
- whole-function conformal bands;
- quadrature-aware CRPS, energy score, coverage, and width.

Run the separate reproducible UQ benchmark with:

```console
python -m tools.operator_benchmarks --uq --quick \
  --seeds 0,1,2,3 --steps 5000 --learning-rate 0.003 \
  --validation-interval 50 --patience 20 --minimum-delta 1e-7 \
  --output artifacts/operator-uq-reference
```

The benchmark keeps calibration-case checksums and resolution, rollout, and
source/query geometry shifts visible in JSON and Parquet artifacts.
