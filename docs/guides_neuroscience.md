# Regional neural dynamics and BOLD

`phydrax.applications.neuroscience` composes regional neural laws, physical connectome delays, persistent neurovascular state, labelled sampled series, and native nonlinear least squares. It is separate from cellular electrophysiology: a region is an aggregate modelling coordinate, not an enlarged membrane compartment.

## Connectivity and units

`RegionalConnectivity` fixes stable region identities and the matrix convention
`weights[target, source]`. Weights are dimensionless; delays are stored in seconds. Input delays may be declared in seconds or milliseconds and are converted once during host preparation. `normalization="incoming_abs"` divides each incoming row by its absolute-weight sum and leaves isolated rows zero.

Zero delays use the current integration state. Positive delays use the accepted native delay history with its existing interpolation, causality, discontinuity, adjoint, and segmented-continuation contracts. Positive delays are never rounded to timestep bins. A zero-weight edge with a positive declared delay retains its causal identity, so differentiating its weight through zero does not change the delay topology.

```python
from phydrax.applications import neuroscience as ns

connectivity = ns.RegionalConnectivity(
    ("left", "right"),
    [[0.0, 0.4], [0.7, 0.0]],
    [[0.0, 0.021], [0.037, 0.0]],
)
```

## Regional laws

`WilsonCowan` evolves bounded excitatory and inhibitory fractions with explicit time constants, recruitment functions, and coupling gain. Its variables are fractions unless the caller supplies and documents another interpretation; sigmoid outputs are not silently labelled as firing rates.

`Hopf` evolves two-coordinate normal-form oscillators. Frequency is supplied in hertz and converted to angular frequency in the law. Coupling is directed diffusive coupling: a self-edge cancels against its own weighted source term rather than creating artificial gain.

A history callable supplies the complete neural prehistory with shape `[region, 2]`. It is a scientific initial condition. Zero-filled history is not introduced by the runtime.

```python
import jax.numpy as jnp

history = lambda time_s, args: jnp.zeros((2, 2))
problem = ns.regional_problem(
    connectivity,
    ns.Hopf(a_per_s=-0.2, frequency_hz=[0.08, 0.11], coupling_per_s=0.6),
    history,
    t0=0.0,
    t1=20.0,
)
```

## Persistent Balloon–Windkessel observation

`BalloonWindkessel` evolves vasodilatory signal and logarithms of flow, venous volume, and deoxyhemoglobin. The logarithmic coordinates preserve the representation of positive ratios; they do not hide overflow, invalid parameters, or integration failure. `NeuralBOLDDrive` explicitly maps the two neural coordinates into vasodilatory input.

Use `regional_bold_problem` to augment neural state with all four hemodynamic coordinates. BOLD is then integrated continuously with the neural trajectory; it is not a stateless convolution or a response reset at output samples.

```python
problem = ns.regional_bold_problem(
    connectivity,
    ns.Hopf(a_per_s=-0.2, frequency_hz=[0.08, 0.11], coupling_per_s=0.6),
    history,
    ns.BalloonWindkessel(),
    ns.NeuralBOLDDrive([1.0, 0.0], [0.0, 0.0], gain=0.5),
    t0=0.0,
    t1=20.0,
)
solution = ns.solve_regional(problem, save_times=jnp.linspace(0.0, 20.0, 81))
```

`RegionalSolution.neural` and `.bold` are `SampledSeries` values with time support, validity masks, region order, and the native solver result. Failed or nonphysical active predictions remain invalid; they are not clipped into plausible observations.

For long delayed trajectories, `segmented=True` uses the native bounded delay continuation. Resume with the same original problem and `continuation=prior.continuation`; the continuation retains neural and hemodynamic state plus accepted delay history.

## Observation and fitting

`BOLDObservation.from_samples` converts declared time (`s` or `ms`) and signal (`fraction` or `percent`) units once. It keeps missing data in an explicit mask and requires at least one active finite datum. Residuals validate sample times, region order, shape, prediction masks, and standard deviations.

`BOLDObservation.least_squares_problem(predict)` lowers a prediction callable into the existing `NonlinearLeastSquaresProblem`. It does not introduce a brain-specific optimizer. Fit only a declared identifiable parameter subset; convergence of a BOLD loss does not establish separate identifiability of coupling, drive gain, and hemodynamic parameters.

## Differentiation and validity

Derivatives follow the selected native ODE/DDE solver, interpolation, adjoint, and fixed delay topology. Preparing a different delay pattern, changing region identities, or changing state layout is a new structural problem. Exact physical delays do not imply exact analytic trajectories.

This application does not infer cell-to-region aggregation, convert membrane voltage into regional activity, or synthesize spike trains from rates. Those are explicit model adapters when required, not lossless identities.

## Example and benchmark

Run the delayed two-region pulse example:

```console
python examples/regional_bold.py
```

The scaling benchmark records region/edge count, exact delay diversity, forward
and parameter-gradient compilation/execution, sample validity, and BOLD bounds:

```console
python benchmarks/regional_bold.py --regions 16 --duration 2 --dt 0.01
```

Generated benchmark results are not stored in the repository.
