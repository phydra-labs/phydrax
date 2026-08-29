# Particle-grid splatting

PHYDRAX splatting is a measure-aware transfer between stable material particles and a declared
structured target support. It is a physics data-plane operation, not an appearance renderer.
Multilinear and degree-one through degree-three tensor B-spline assignments share one prepared
transfer, evidence, and differentiation contract.

## Prepare one transfer

A splat plan binds a prepared tensor grid to a particle discretization. Particle identity, mass,
and activity remain structural; current positions remain runtime state.

```python
import jax.numpy as jnp
import phydrax as phx

grid = phx.discretization.TensorGridPlan(
    (
        phx.discretization.UniformCellAxisSpec(16, periodic=True),
        phx.discretization.UniformCellAxisSpec(16, periodic=True),
    ),
    axis_names=("x", "y"),
).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
particles = phx.discretization.ParticleSetPlan(
    jnp.arange(4),
    jnp.ones((4,)),
    ambient_dimension=2,
).prepare()
assignment = phx.discretization.TensorBSplineSplatAssignment(2)
prepared = phx.discretization.ParticleGridSplatPlan(
    grid, assignment=assignment
).prepare(particles)
```

The default assignment is `MultilinearSplatAssignment`. It supports nonuniform nodal layouts and
uniform mixed point/interval layouts. `TensorBSplineSplatAssignment` supports degrees one, two, and
three on uniform structured axes.

## Assignment families

Every assignment declares its support width, nonnegativity, partition-of-unity, reproduction, and
derivative capabilities. For dimension `d`:

- multilinear route width is `2**d`;
- degree-`p` tensor B-spline route width is `(p + 1)**d`.

B-spline assignments support nodal, cell, face, and edge layouts. The target `GridLocation` resolves
one exact `TensorEntityLayout` and physical target measure. Mixed layouts therefore retain their
actual point/interval axis identities instead of being treated as image pixels.

Assignments expose route weights, position gradients, target offsets, captured fractions, and
per-source moments:

```text
first moment   = sum_r weight[r] offset[r]
second moment  = sum_r weight[r] offset[r] offset[r]^T
gradient sum   = sum_r d(weight[r]) / d(position)
```

On a complete periodic assignment, partition defect, first moment, and gradient sum vanish up to the
declared precision. These quantities are evidence and building blocks for later PIC, MPM, APIC, and
immersed-boundary methods; splatting does not itself claim those solver semantics.

## Extensive content and density

For source content `q[i]` and assignment coefficients `A[j, i]`, deposition computes

```text
content[j] = sum_i A[j, i] q[i].
```

The coefficients are dimensionless. Target density is derived afterward using the target measure:

```text
density[j] = content[j] / target_measure[j].
```

```python
position = jnp.asarray(
    [[0.1, 0.2], [0.6, 0.25], [0.35, 0.8], [0.85, 0.7]]
)
mass = jnp.asarray([1.0, 2.0, 1.5, 0.5])
state = prepared.build(position)
result = prepared.deposit_content(state, mass)
```

`result.content` is extensive. `result.density` is a field value. These meanings are never
interchanged implicitly.

The source provenance in `result.balance` is the particle support ID. Arbitrary deposited content
does not silently inherit the material-mass measure: callers may deposit charge, momentum, energy,
or another extensive quantity with their own units. The target measure ID is retained because it is
used explicitly to derive density.

`result.balance` reports active, captured, dropped, and target totals, the floating-point balance
defect, partition defect, route count, tolerance, and the exact policy and support/measure identities
behind the claim. `require_closed_conservation` fails unless the result has complete support and
satisfies the numerical balance contract.

## Intensive reconstruction

An intensive field requires explicit nonnegative sample weights. PHYDRAX returns the unnormalized
numerator, denominator, normalized value, support mask, and effective denominator tolerance:

```python
velocity = jnp.asarray(
    [[1.0, 0.0], [0.5, -1.0], [-0.25, 0.75], [1.5, 0.5]]
)
reconstructed = prepared.reconstruct(state, velocity, mass)
```

The result is

```text
numerator[j]   = sum_i A[j, i] weight[i] value[i]
denominator[j] = sum_i A[j, i] weight[i]
values[j]      = numerator[j] / denominator[j]
```

where `values` is zero and `support` is false when the denominator has no numerical coverage.
All-zero sample weights are a valid unsupported reconstruction. Normalized reconstruction is not an
extensive conservation operation.

## Grid-to-particle gather

The same assignment stencil interpolates target values back to particles:

```python
x, y = grid.cells().coordinates_by_axis
xx, yy = jnp.meshgrid(x, y, indexing="ij")
observed = prepared.gather(state, xx + 2.0 * yy)
```

The algebraic transpose of this gather stencil is the content-deposition operator. Tests certify
the corresponding real and complex inner-product identities.

## Boundary policies

Periodic axes inherit the target grid period and wrap routes across the seam.

A nonperiodic plan selects one policy:

- `boundary="reject"` fails closed when an active source leaves the domain or its nonzero assignment
  support is truncated. This is the default for closed conservative transfer.
- `boundary="drop"` retains every in-domain route and reports the missing fraction of each truncated
  source. Target content plus dropped content equals active source content up to numerical defect.

There is no hidden endpoint clamp or source renormalization. Inactive particles are made numerically
inert before nonfinite inactive storage can enter arithmetic.

## Execution, resources, and precision

`SplatExecutionPolicy` separates three accumulation modes:

- `fast`: accelerator-oriented scatter/segment reduction; no bitwise determinism claim.
- `deterministic`: canonical stable-particle and route order.
- `compensated`: the deterministic order with TwoSum correction.

`ParticlePrecisionPolicy` controls geometry, assignment evaluation, accumulation, certification,
and output dtypes independently. Casting a low-precision result to a wider output does not change
the recorded accumulation precision.

Preparation reports relation storage and scalar-payload workspace. For payload width `c`, the actual
output workspace scales approximately as `scalar_workspace_bytes * c`; arbitrary payload memory is
not hidden behind the scalar count.

## Differentiation

The reference implementation is ordinary JAX and supports `jit`, `vmap`, `scan`, JVP, and VJP.
With `geometry_ad="piecewise"`, route indices, support masks, and periodic branches are frozen while
assignment weights remain differentiable with respect to particle positions. Gradients are valid
inside one routing program; exact knots, support boundaries, and periodic seams are discrete route
transitions.

`geometry_ad="frozen"` stops the geometry path while preserving payload derivatives under `jit`,
`vmap`, and `scan`.

No derivative is claimed through particle creation/deletion, active-topology changes, rejected
boundaries, or capacity changes.

## Current scope

The substrate supports structured multilinear and uniform tensor B-spline transfer. SPH footprints,
Gaussian footprints, anisotropic support, irregular targets, detector projection, and ordered
optical compositing remain separately qualified extensions over this reference contract.
