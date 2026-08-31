# GIMP and CPDI particle domains

Every structured assignment still emits one `SplatAssignmentState` with fixed route
indices, weights, physical gradients, node-minus-particle offsets, validity, and
moments. Advanced assignments add a typed `assignment_input`; they do not introduce
a second transfer engine.

## uGIMP and cpGIMP/AABB

`UniformGIMPSplatAssignment` exactly convolves a linear nodal tent with an
axis-aligned particle box. Prepared reference half-widths give uGIMP. With
`evolving=True`, current cpGIMP/AABB widths are:

```text
half_width[a] = sum_b abs(F[a,b]) reference_half_width[b].
```

This is explicitly an axis-aligned bounding-box approximation; it does not preserve
rotated/sheared particle geometry. Width envelope, route width, and halo are prepared
and overflow fails closed.

## Affine CPDI

`AffineCPDISplatAssignment` owns a reference half-edge matrix and updates:

```text
A = F A0
corner[c] = x + A sign[c].
```

Corner linear-grid routes are deterministically sorted and duplicate nodes are
combined into fixed capacity. CPDI weights are corner averages. Physical gradients
use parent-domain shape gradients transformed by `A^-T`, not averaged point gradients.

## CPDI2

`CPDI2SplatAssignment` carries transactional current corners, accepted center, and
accepted deformation. An incremental deformation convects the accepted arbitrary
quadrilateral/hexahedron. The center parent Jacobian maps parent gradients into
physical gradients. Orientation reversal, singularity, excessive conditioning,
extent overflow, or missing corner support reject the step.

## Differentiation

Weights, gradients, widths, edges, and corners are differentiable inside a fixed
routing/dedup program. Floor/index changes, duplicate-group changes, inversion, and
capacity overflow are structural branch boundaries.

Each family must qualify partition, gradient sum, first moment, APIC second moment,
affine reproduction, complete halo, and domain rollback independently.
