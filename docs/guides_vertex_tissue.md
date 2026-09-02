# Vertex-tissue mechanics

Phydrax represents a vertex tissue as a fixed-capacity, stable-ID cell complex. `VertexTissuePlan` owns topology and constitutive parameters; `PreparedVertexTissue` binds one finite reference geometry; and `VertexTissueState` contains only fixed-shape vertex coordinates, conserved cell-field contents, time, and the prepared-epoch identity. This separation keeps energy, force, coupling, and overdamped stepping JIT-compatible while reserving discrete topology changes for accepted host boundaries.

## Explicit incidence and stable IDs

Use `polygonal_vertex_tissue_plan` for a 2D confluent polygonal complex. It requires stable vertex, edge, and cell IDs together with:

- an `(edge_capacity, 2)` edge-to-vertex array;
- ordered cell-to-edge rows with trailing `-1` padding;
- a matching orientation array containing `+1`, `-1`, or zero on padding;
- an `(edge_capacity, 2)` edge-to-cell array, with `-1` in the second slot on a boundary.

Use `polyhedral_vertex_tissue_plan` for a 3D polyhedral complex. It additionally requires ordered face-to-vertex rows, oriented cell-to-face rows, and face-to-cell incidence. Every active cell boundary is checked as a closed oriented two-manifold. Every face edge must occur in the explicit edge table, the two cells sharing an interior face must use opposite face orientations, and duplicate interfaces are rejected up to endpoint order or cyclic face permutation and reversal.

Stable IDs are nonnegative. `-1` is the only inactive ID or incidence padding value. Capacity is the array shape, not the number of active IDs. Dynamic workflows preallocate inactive slots; a transition may activate or deactivate those slots without changing any compiled state shape.

```python
import jax.numpy as jnp

from phydrax.applications.cellular_mechanics import (
    polygonal_vertex_tissue_plan,
)

positions = jnp.asarray(
    ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
)
plan = polygonal_vertex_tissue_plan(
    jnp.arange(4),
    jnp.arange(4),
    jnp.asarray(((0, 1), (1, 2), (2, 3), (3, 0))),
    jnp.asarray((100,)),
    jnp.asarray(((0, 1, 2, 3),)),
    jnp.ones((1, 4), dtype=jnp.int32),
    jnp.asarray(((0, -1), (0, -1), (0, -1), (0, -1))),
    1.0,
    2.0,
    4.0,
    1.0,
    field_names=("morphogen_content",),
)
tissue = plan.prepare(positions)
state = tissue.initialize_state(jnp.asarray(((3.0,),)))
evaluation = tissue.evaluate(state)
```

Plans reject malformed padding, dangling or duplicate interfaces, nonclosed polygon loops, same-direction shared edges, inconsistent face orientation, inactive incidence, duplicate active IDs, invalid material domains, and nonfinite host inputs. Runtime evaluation reports `finite`, `manifold`, `orientation_valid`, `quality_valid`, inactive-field validity, overall `valid`, and a typed `VertexTissueStatus`. Invalid compiled evaluations return zero forces rather than propagating an inadmissible load.

## Energy, conservative forces, and active loading

For cell measure `M_c` (area in 2D or signed volume in 3D) and boundary measure `B_c` (perimeter or surface area), the conservative cell energy is

```
E_cell = 1/2 K_M,c (M_c - M0,c)^2
       + 1/2 K_B,c (B_c - B0,c)^2
       + 1/2 Gamma_c B_c^2.
```

`active_contractility` supplies `Gamma_c`. Each edge in 2D or face in 3D contributes

```
E_interface = (tension_i - adhesion[type_left, type_right]) measure_i.
```

Adhesion is applied only to two-cell interfaces. Boundary interfaces receive the declared line or surface tension without an adhesion term. The adhesion matrix must be finite, symmetric, and nonnegative.

`vertex_tissue_potential_energy` is the canonical scalar energy. `evaluate_vertex_tissue` obtains conservative vertex forces as its exact negative JAX gradient and separately distributes each cell's declared resultant `cell_traction` uniformly over that cell's distinct incident vertices. `VertexTissueEvaluation` keeps conservative, active, and total forces separate and reports the net conservative-force residual needed to certify translation invariance.

Areas use oriented edge sums. Volumes use oriented face-triangle fluxes, while planar face area uses the norm of the oriented polygon area vector so simple concave faces are not overcounted by an invalid fan. Proper rotations and translations do not change the scalar energy. Positive signed cell measure is an explicit orientation requirement.

## Overdamped dynamics

`VertexTissueDynamicsPlan` prepares a fixed-shape overdamped step against one tissue epoch. For vertex drag `zeta_v`, the update uses

```
velocity_v = (conservative_force_v + active_force_v + external_force_v) / zeta_v.
```

The result contains the candidate and selected states, before/after evaluations, dissipation rate, active input power, energy change, displacement guard, passive energy-descent evidence, inactive-field-rate evidence, and acceptance status. A nonfinite, inverted, poor-quality, overlong, passive energy-increasing, or inactive-field-populating candidate rolls every state leaf back. Cell-field source rates may be supplied without changing state shape.

```python
from phydrax.applications.cellular_mechanics import VertexTissueDynamicsPlan

dynamics = VertexTissueDynamicsPlan(
    1.0e-3,
    maximum_displacement=0.05,
).prepare(tissue)
result = dynamics.step(state)
state = result.state
```

## Cell fields and particle coupling

Cell scalar fields are stored as conserved cell-integrated contents, and inactive capacity rows must remain exactly zero. `cell_field_density` divides active contents by current positive cell measure. `interpolate_cell_fields` maps cell values to vertices by incidence averaging, and `spread_vertex_field_sources` is its exact transpose.

`couple_vertex_tissue_particles` gathers the owning cell's fields to each particle and scatters particle resultants uniformly to the owning cell's vertices. It reports route, nonfinite, and force-conservation evidence. A particle cell index of `-1` is masked; any other inactive or out-of-capacity route fails closed.

## Candidate, evaluation, and commit topology epochs

A topology event carries a complete replacement `VertexTissuePlan`, replacement coordinates, a target-by-source cell transfer matrix, and the source prepared identity. Proposal also fingerprints the exact source coordinates, per-cell contents, time, and prepared epoch. Evaluation and commit both recheck that source-state binding so evidence cannot be replayed against a later state or another epoch. The workflow is deliberately explicit:

```python
from phydrax.applications.cellular_mechanics import (
    VertexTissueEventKind,
    VertexTissueTopologyEvent,
    commit_vertex_tissue_topology,
    evaluate_vertex_tissue_topology,
    propose_vertex_tissue_topology,
)

event = VertexTissueTopologyEvent(
    VertexTissueEventKind.DIVISION,
    tissue.prepared_id,
    daughter_plan,
    daughter_positions,
    daughter_by_parent_transfer,
)
candidate = propose_vertex_tissue_topology(tissue, state, event)
evidence = evaluate_vertex_tissue_topology(tissue, state, candidate)
result = commit_vertex_tissue_topology(tissue, state, candidate, evidence)
```

The same transaction supports `T1`, `T2`, `T3`, `DIVISION`, `EXTRUSION`, `APOPTOSIS`, `FACE_TRANSITION`, and `EDGE_TRANSITION`. Certification checks:

- the source prepared identity and exact source-state fingerprint are current;
- candidate coordinates, copied time, and fields equal the declared event geometry and transfer;
- dimension and vertex/edge/face/cell/field capacities are unchanged;
- event-specific cell-count and incidence changes are consistent;
- target incidence remains manifold and target geometry is finite, positively oriented, and above quality thresholds;
- the nonnegative transfer has unit sum in every active source column and zero rows for inactive targets;
- total content in every scalar field is conserved;
- surviving cells preserve parent and generation metadata and their own identity routes;
- a new division daughter names an active source parent, has parent generation plus one, and receives content only from that parent.

`commit_vertex_tissue_topology` selects the candidate only when every certificate passes; otherwise it returns the exact source prepared object and state. `rollback_vertex_tissue_topology` explicitly rejects even a passing candidate. A committed event always has a newly fingerprinted `prepared_id`, so stale states cannot enter the new compiled epoch.

T2 is restricted to removal of a triangular 2D cell. T1 requires one stable shared edge to exchange two old owners for two disjoint new owners across a four-cell local neighborhood. T3 requires a real same-cell-set vertex/edge incidence change and rejects ID-only relabeling. Extrusion and apoptosis remove exactly one cell, and division adds exactly one cell. A 3D face transition changes face/cell incidence without changing edge incidence; an edge transition changes both edge and dependent face incidence, so the two event labels cannot certify the same candidate.

## Differentiation boundary

Energy, mechanics evaluation, coupling, and overdamped dynamics are differentiable for a fixed prepared topology, positive cell orientation, fixed particle routes, and unchanged acceptance branch. A discrete topology event has no implicit gradient across its incidence, lineage, or transfer decision. Differentiate a committed epoch only conditionally on the event sequence and active slots remaining fixed; use an explicit event estimator when sensitivities to event selection itself are required.
