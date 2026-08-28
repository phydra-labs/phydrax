# Smoothed finite elements

Phydrax implements smoothed finite elements as method-specific reductions over a
shared internal patch/boundary-moment substrate. A smoothing patch is not an
ordinary quadrature domain: edge-, node-, and axisymmetric patches may span
multiple primal cells and gather non-cell-local DOF stencils.

## Exact source scope

The source-backed methods are deliberately distinct:

- Cui et al. (2008): Q4 first-order shear-deformation plates/shells, with
  separately selectable membrane, bending, shear, and nonlinear-membrane
  smoothing-cell partitions.
- Liu, Nguyen-Thoi, and Lam (2009): 2-D T3 edge-based smoothing for static,
  eigen, and forced vibration; stiffness is smoothed while mass/load remain
  ordinary FEM.
- Liu et al. (2009): 2-D node-star T3/Q4/n-gon smoothing; upper-energy and
  locking claims are treated as empirical/conditional.
- Wan et al. (2016): fully smoothed axisymmetric CS/NS/ES with smoothed
  stiffness and consistent mass. This is not the 3-D face-based FS-FEM method.

## Boundary moments

For smoothing patch `S`,

```text
Gbar[S,a] = (1 / measure(S)) * integral_boundary(S) N_a n dGamma
```

`boundary_moment` computes this from an oriented `SmoothingPatchLayout` and
runtime `SmoothingPatchGeometry`. Engineering strain matrices are derived by
`smoothed_symmetric_gradient_matrix`.

Every patch carries:

- an owner entity;
- a gathered field stencil;
- affine patch-vertex construction;
- a closed oriented boundary;
- boundary trace values and rules;
- stable structural identity.

## Edge smoothing

`edge_smoothing_layout(mesh)` constructs one patch per triangular mesh edge.
Interior patches combine one centroid-fan subtriangle from both incident T3
cells; boundary patches use one. The edge-star areas partition each T3 into
three equal subareas.

```python
import jax.numpy as jnp
import phydrax as phx

vertices = jnp.asarray(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]]
)
cells = jnp.asarray(
    [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
    dtype=jnp.int32,
)
mesh = phx.discretization.CellMesh.from_triangles(vertices, cells)
smoothed = phx.discretization.fem.smoothing
constitutive = smoothed.plane_stress_matrix(1.0, 0.3)
edge_plan = smoothed.SmoothedElasticityPlan("ES", mesh, constitutive)
operator = edge_plan.operator(vertices)
residual = operator.mv(jnp.zeros((10,)))
```

## Node smoothing

`node_smoothing_layout(mesh)` constructs each node star by unioning the
centroid/mid-edge subcells of incident T3 cells and extracting the uncancelled
oriented boundary. Boundary-node stars retain physical-boundary pieces.

No universal upper-bound certificate is emitted. Any energy-bound evidence must
state its assumptions and remain empirical unless independently proved.

## Q4 plate/shell smoothing

`Q4FSDTSmoothingPlan` keeps four channels independent:

- membrane;
- bending;
- shear;
- nonlinear membrane.

The source-backed default uses three smoothing cells for membrane/bending/
nonlinear membrane and one for shear. `channels(coordinates)` returns boundary
moments for membrane/bending/nonlinear channels and an explicit shear shape
average. The nonlinear extension remains limited to the displayed transverse-
slope formulation; it is not a generic finite-rotation shell theory.

## Fully smoothed axisymmetry

`FullySmoothedAxisymmetricPlan` supports CS/ES/NS patch factories. Radial,
axial, and shear rows use boundary moments. Hoop strain and smoothed mass require
an analytic radial primitive supplied as boundary values:

```text
b_hoop[a] = (1 / (rbar * area)) * integral_patch N_a dr dz
```

The plan rejects nonpositive central radii. Axis-crossing formulations require a
separate analytic-axis derivation.

## Stabilization and evidence

`SmoothingStabilizationPolicy` exposes only explicit policies:

- compatible-cell blend;
- projected-gradient penalty;
- rank-complement penalty;
- selective volumetric smoothing.

No diagonal shift is inserted silently.

`certify_smoothing_operator` reports patch closure, partition defect, affine
reproduction, rigid/extra near-zero modes, and constrained minimum eigenvalue.
Energy evidence is labeled `none`, empirical-lower-like,
empirical-upper-like, or proved-under-explicit-assumptions.

## Current limits

- ES/NS factories currently require 2-D T3 meshes.
- Cell-smoothed plate partitions currently require Q4 blocks.
- Fully smoothed axisymmetry supports the available CS/ES/NS patch factories and
  caller-supplied analytic primitive values.
- 3-D face-based smoothing is not implemented from the attached axisymmetric
  paper.
- Nonlinear/history-dependent smoothing must use the accepted-step material
  lifecycle and an independently derived objective formulation.
