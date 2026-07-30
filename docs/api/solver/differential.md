# Differential equation integration

The differential backend integrates finite-dimensional initial-value problems through
[Diffrax](https://docs.kidger.site/diffrax/). It is separate from `FunctionalSolver`:
`FunctionalSolver` minimizes a physics/data functional, while `solve_diffrax` numerically
integrates a supplied drift and optional diffusion.

## Problem and driver contract

`DifferentialProblem` represents

$$
dY_t = f(t,Y_t,a)\,dt + g(t,Y_t,a)\,dW_t.
$$

Omit `diffusion` for an ODE. For an SDE, `WienerDriver.noise_shape` declares the
finite-dimensional Wiener increment seen by `g`. For a state vector of size $n$ and
`noise_shape=(m,)`, `g` normally returns an $n\times m$ array. Spatial SPDEs must first
be semidiscretized to a finite state; the driver metadata can record the retained noise
basis and realization identity.

A driver owns its PRNG key. Reusing the same `WienerDriver` replays the same Brownian
path; changing its key changes the path. `basis_id` and `realization_id` are provenance,
not numerical inputs. Use `levy_area="space_time"` or `"space_time_time"` only with a
solver that requires the corresponding Levy-area information.

::: phydrax.solver.DifferentialProblem
    options:
        members:
            - __init__
            - stochastic

---

::: phydrax.solver.WienerDriver
    options:
        members:
            - __init__

## ODE solve

The default deterministic solver is `diffrax.Tsit5` with a PID step-size controller.
The result remains differentiable with respect to array-valued initial states,
parameters, and vector-field leaves.

```python
import jax.numpy as jnp
import phydrax as phx

problem = phx.solver.DifferentialProblem(
    lambda t, y, rate: -rate * y,
    jnp.asarray([1.0]),
    t0=0.0,
    t1=2.0,
    args=jnp.asarray(0.4),
)
solution = phx.solver.solve_diffrax(
    problem,
    save_times=jnp.linspace(0.0, 2.0, 21),
)
```

::: phydrax.solver.solve_diffrax

## Dense vector interpolation

Pass `dense=True` to retain Diffrax's local interpolants and enable
`DifferentialSolution.evaluate`. Query times may be a scalar or an arbitrarily shaped
array:

```python
solution = phx.solver.solve_diffrax(
    problem,
    save_times=jnp.asarray([0.0, 2.0]),
    dense=True,
)
query_times = jnp.asarray([[0.1, 0.4], [1.2, 1.8]])
interpolated = solution.evaluate(query_times)
assert interpolated.shape == (2, 2, 1)
```

For one trajectory, the output shape is `query_times.shape + state_shape`. For an
ensemble it is `sample_shape + query_times.shape + state_shape`: every realization is
evaluated on the same query array without flattening either the process or query axes.
Scalar query times omit the query axis. Dense evaluation remains JAX-transformable and
differentiable through the solve.

Dense output is opt-in because it retains per-step interpolation data. Query times must
be non-empty, finite, and inside the interval available to every realization; this is
the common interval when event termination differs across an ensemble. The `left`
argument selects the left or right limit at a jump. `has_dense_interpolation` reports
whether evaluation is available. The vectorization is implemented internally by
Phydrax and requires no interpolation package beyond Diffrax.

## SDE solve and process ensemble

The default Itô solver is fixed-step Euler--Maruyama (`diffrax.Euler`); the default
Stratonovich solver is `diffrax.EulerHeun`. SDE calls require both a `WienerDriver` and
an explicit `dt0`. Pass an explicit Diffrax solver/controller for higher-order,
adaptive, stiff, or event-aware integration; compatibility between its stochastic
term, interpretation, and Levy-area requirement remains the caller's responsibility.

```python
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

problem = phx.solver.DifferentialProblem(
    lambda t, y, args: -0.2 * y,
    jnp.zeros((2,)),
    t0=0.0,
    t1=1.0,
    diffusion=lambda t, y, args: 0.3 * jnp.eye(2),
    interpretation="ito",
)
driver = phx.solver.WienerDriver(
    jr.key(0),
    (2,),
    tolerance=1e-3,
    basis_id="state-space",
    realization_id="run-0",
)
ensemble = phx.solver.solve_diffrax_ensemble(
    problem,
    save_times=jnp.linspace(0.0, 1.0, 11),
    driver=driver,
    num_paths=128,
    dt0=1e-2,
)
predictive = ensemble.to_predictive(
    sample_dim="path",
    time_dim="time",
    state_dims=("state",),
)
```

`solve_diffrax_ensemble` splits the driver key once per path and returns arrays with
shape `sample_shape + (num_times,) + state_shape`. `to_predictive` labels that leading
axis as `process` uncertainty by default. It does not reinterpret discretization or
solver error as process uncertainty.

::: phydrax.solver.solve_diffrax_ensemble

## Semidiscrete SPDEs

Phydrax's native SPDE path is finite-dimensional method of lines:

\[
dU_t=F_h(t,U_t,a)\,dt+G_h(t,U_t,a)B\,dW_t.
\]

The spatial discretization defines the leading state axes and a matrix-free
Laplacian. A finite-rank `SpatialNoiseBasis` defines

\[
B=\Phi\operatorname{diag}(\sqrt{q_1},\ldots,\sqrt{q_r}),
\qquad
\Phi^\mathsf TM\Phi=I,
\]

where \(M\) is the spatial quadrature mass. `SemidiscreteSPDE.wiener_driver`
then derives `noise_shape=(r,)` and propagates the basis fingerprint. This
prevents a driver and diffusion factor from silently disagreeing about retained
noise modes.

### Spatial discretizations

`TensorGridDiscretization` consumes the existing materialized
`AxisDiscretization` objects:

| Axis basis | Laplacian | Boundary semantics |
| --- | --- | --- |
| `uniform` | second-order centered finite difference | periodic |
| `fourier` | FFT spectral derivative | periodic |
| `sine` | odd-extension spectral derivative | homogeneous Dirichlet |
| `cosine` | even-extension spectral derivative | homogeneous Neumann |

Tensor-grid states begin with the declared spatial shape; trailing channel axes
are preserved by `laplacian`, `flatten`, and `unflatten`.
`laplacian_matrix()` is an explicit diagnostic for small systems. Ordinary
integration should use the matrix-free application.

`SpectralSpatialDiscretization` wraps an existing
`phydrax.nn.SpectralDiscretization`. It reuses that plan's analysis, synthesis,
eigenvalues, quadrature, degeneracy ordering, and `basis_id`; it does not define
a second manifold eigenbasis convention.

::: phydrax.solver.AbstractSpatialDiscretization

---

::: phydrax.solver.TensorGridDiscretization

---

::: phydrax.solver.SpectralSpatialDiscretization

### Finite-rank spatial noise

Construct a basis from explicit weighted-orthonormal modes, a covariance
spectrum evaluated on low Laplacian modes, a nodal covariance matrix, or a
continuous kernel sampled at the discretization points. Full-rank covariance
factorization reconstructs the supplied nodal covariance; a smaller `rank`
is an explicit KL truncation. Negative covariance eigenvalues, non-orthonormal
modes, shape mismatches, and ranks larger than the discrete state are rejected
before integration.

`basis_id` hashes state shape, modes, covariance eigenvalues, quadrature,
mode IDs, and spatial discretization provenance. It changes when the grid,
rank, spectrum, or modes change.

::: phydrax.solver.SpatialNoiseBasis

### Composition and integration

```python
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

axis = phx.domain.FourierAxisSpec(32).materialize(0.0, 1.0)
space = phx.solver.TensorGridDiscretization((axis,))
noise = phx.solver.SpatialNoiseBasis.from_spectrum(
    space,
    lambda eigenvalue: 0.02 * jnp.exp(-0.05 * eigenvalue),
    rank=6,
)
initial = jnp.sin(2.0 * jnp.pi * axis.nodes)

spde = phx.solver.semidiscretize_reaction_diffusion(
    initial,
    space,
    t0=0.0,
    t1=0.2,
    kappa=0.01,
    reaction=lambda t, state, args: state - state**3,
    noise_basis=noise,
    interpretation="ito",
)
driver = spde.wiener_driver(
    jr.key(0),
    tolerance=1e-4,
    realization_id="allen-cahn-0",
)
ensemble = phx.solver.solve_diffrax_ensemble(
    spde.problem,
    save_times=jnp.linspace(0.0, 0.2, 21),
    driver=driver,
    num_paths=128,
    dt0=1e-3,
)
```

`semidiscretize_spde` accepts a general drift and either a
`SpatialNoiseBasis` or an explicit diffusion plus `noise_shape`.
`semidiscretize_reaction_diffusion` supplies
\(\kappa\Delta_hU+R(t,U,a)\) and optionally scales a basis with a scalar,
pointwise, or full diffusion amplitude. Initial state, drift, diffusion, basis,
and noise shapes are checked eagerly.

Both Itô and Stratonovich interpretations pass unchanged into
`DifferentialProblem`; the usual Diffrax solver/Levy-area compatibility rules
still apply. These APIs solve a finite-rank semidiscrete system. They do not
claim direct infinite-dimensional white-noise integration or automatic
discretization-error uncertainty.

::: phydrax.solver.SemidiscreteSPDE

---

::: phydrax.solver.semidiscretize_spde

---

::: phydrax.solver.semidiscretize_reaction_diffusion

## Result contract

`DifferentialSolution.valid` marks finite saved states. `successful` reduces that mask
across saved times for each realization. `backend_result`, `stats`, `event_mask`,
`solver_name`, `interpretation`, the driver, and per-realization keys preserve the
integration and stochastic provenance needed to reproduce a path ensemble.

::: phydrax.solver.DifferentialSolution
    options:
        members:
            - __init__
            - num_times
            - successful
            - has_dense_interpolation
            - evaluate
            - to_predictive
