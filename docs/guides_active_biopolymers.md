# Active biopolymers and dynamic pair relations

PhydraX represents changing biological connectivity with a fixed-capacity relation table. The table keeps compiled array shapes constant while giving every slot a stable relation ID and an incarnation counter. A relation is addressed by the pair `(relation_id, incarnation)`, not by its current array slot alone. Reusing a vacated slot increments its incarnation, so a delayed unbind or move cannot mutate a newer relation accidentally.

## Dynamic relation lifecycle

`DynamicPairRelationPlan` fixes:

- endpoint types and their prepared availability;
- relation, event, and parameter capacities;
- the compatibility tensor for `(relation kind, left type, right type)`;
- a symmetric kind-exclusion matrix for relations that may not share endpoints;
- which relation kinds treat reversed endpoints as duplicates; and
- the maximum relation incarnation.

Preparing the plan produces `PreparedDynamicPairRelations`. Its state contains fixed-width arrays for stable IDs, incarnations, endpoints, kind, occupied and active masks, age, parameters, historical occupancy, and a scalar numeric version. `PairRelationEventBatch` supports bind, unbind, move, activate, and deactivate events. Event IDs determine evaluation order independently of the batch input order.

```python
import jax.numpy as jnp
import numpy as np

from phydrax.discretization.particle import (
    DynamicPairRelationPlan,
    PairRelationEventKind,
    make_pair_relation_events,
)

relations = DynamicPairRelationPlan(
    np.zeros(16, dtype=np.int32),
    8,
    2,
    symmetric_kinds=np.asarray([True]),
    event_capacity=4,
).prepare()
state = relations.initialize()
events = make_pair_relation_events(
    4,
    2,
    event_ids=[7],
    event_kind=[PairRelationEventKind.BIND],
    left=[2],
    right=[9],
    relation_kind=[0],
    parameters=[[4.0, 1.5]],
)
result = relations.apply(state, events)
state = result.accepted_state
```

Evaluation is separate from commit. `evaluate` exposes the exact source state/version, candidate state, and per-event status; `commit` accepts the complete candidate only when every requested event succeeds and the supplied current state still equals that source. `apply` is the combined convenience operation. Evidence distinguishes capacity overflow, invalid endpoints or endpoint types, duplicates, exclusions, stale identities or stale source evaluations, malformed requests, structurally invalid source states, nonfinite parameters, and incarnation overflow. Failure is atomic: `accepted_state` remains the input graph even when earlier ordered events produced a useful diagnostic candidate.

`PairSpringPlan` maps two fixed parameter columns to stiffness and rest length. Its prepared energy evaluates `½ k (length - rest_length)²` over active relations. Forces are computed as the exact negative gradient of that scalar energy. Coincident active endpoints, negative spring controls, or nonfinite values fail closed in `PairSpringEvaluation`.

## Chromatin loops

`ChromatinDynamicsPlan` combines two-foot relation occupancy with diffusion capture, dissociation, and outward loop extrusion. Capture requires two unoccupied, non-roadblock sites within the spatial capture distance. Extrusion attempts move the left foot one genomic site left and the right foot one site right. Boundaries, roadblocks, occupied sites, and simultaneous destination collisions reject the move without corrupting any other loop.

```python
import jax.random as jr
import numpy as np
from phydrax.applications.cellular_mechanics import ChromatinDynamicsPlan

sites = np.arange(64, dtype=float)[:, None]
roadblocks = np.zeros(64, dtype=bool)
roadblocks[[12, 47]] = True
chromatin = ChromatinDynamicsPlan(
    sites,
    16,
    roadblocks=roadblocks,
    binding_rate=0.5,
    unbinding_rate=0.02,
    extrusion_rate=1.0,
    capture_distance=6.0,
    spring_stiffness=2.0,
    spring_rest_length=1.0,
    realization_id=31,
).prepare()
state = chromatin.initialize()
step = chromatin.step(state, jr.key(0), 0.01)
state = step.accepted_state
```

`ChromatinObservables` reports site and roadblock occupancy together with per-loop genomic span, spatial distance, active mask, counts, means, bound fraction, and spring energy. `ChromatinStepEvidence` adds capture, collision, binding, unbinding, extrusion, relation-status, and spring evidence.

## Actin turnover and lineage

`ActinNetworkPlan` prepares a fixed node capacity and a fixed relation capacity. Filament elongation and branching allocate an inactive node and a relation as one transaction, transferring exactly one monomer mass from the soluble pool. Terminal depolymerization returns the same mass. Capping changes endpoint eligibility. Severing removes an identity-addressed edge and assigns the complete daughter subtree a new lineage ID using a bounded fixed-shape closure pass.

```python
import jax.numpy as jnp
import numpy as np
from phydrax.applications.cellular_mechanics import ActinNetworkPlan

actin = ActinNetworkPlan(
    256,
    512,
    ambient_dimension=2,
    initial_monomer_pool=500.0,
    monomer_mass=1.0,
    segment_length=0.5,
).prepare()
state = actin.initialize(np.asarray([[0.0, 0.0]]))
addition = actin.polymerize(state, 0, jnp.asarray([1.0, 0.0]), event_id=1)
state = addition.accepted_state
```

Every `ActinNetworkEvidence` carries mass before and after the candidate, the residual, lineage validity, node overflow, invalid-request and stale-identity flags, spring evidence, and the event code. A topology candidate is accepted only if its mass residual is within floating-point roundoff, all active relations connect active nodes, every node has at most one incoming lineage edge, and the numerical state is finite.

## Crosslinkers and motors

`MotorCrosslinkerPlan` uses relation kind zero for passive symmetric crosslinks and kind one for directed motors. A motor step follows the supplied fixed-size successor table at its right endpoint. The move is suppressed at a filament end or when the spring load reaches the stall force. The left endpoint is unchanged. `MotorCrosslinkerEvidence` reports stepped, stalled, and endpoint-blocked counts alongside relation and spring evidence.

The successor table makes filament direction explicit and avoids rebuilding a dynamic adjacency structure inside compiled code. It may be updated between execution epochs when the actin topology changes.

## Focal adhesions and traction

`FocalAdhesionPlan` creates typed cell and substrate endpoint spaces. Compatibility permits only directed cell-to-substrate relations, while kind exclusion enforces single occupancy at each adhesion site. Capture is distance limited. Rupture follows a force-accelerated rate based on the conservative spring energy.

```python
import numpy as np
from phydrax.applications.cellular_mechanics import FocalAdhesionPlan

adhesions = FocalAdhesionPlan(
    32,
    64,
    48,
    ambient_dimension=2,
    spring_stiffness=5.0,
    rest_length=0.1,
).prepare()
state = adhesions.initialize(
    np.float32,
    cell_endpoints=np.asarray([0] + [-1] * 47),
    substrate_endpoints=np.asarray([3] + [-1] * 47),
)
evidence = adhesions.traction(state, cell_positions, substrate_positions)
```

Traction is the energy-derived force restricted to cell endpoints. Evidence includes per-site cell traction, its vector sum and norm, adhesion turnover counts, relation status, spring validity, and overall fail-closed success.

## Randomness, capacities, and differentiation

Stochastic runtimes derive every draw from the tuple `(root key, realization ID, step index, process address, fixed slot index)`. The same plan, state, key, and step therefore replay exactly. Changing an unrelated event's acceptance does not renumber later random draws.

Capacities are scientific controls rather than hidden resizing hints. Size relation and event capacities for the largest execution epoch, inspect overflow evidence, and prepare a larger epoch explicitly when needed. No compiled transition performs Python mutation or changes an array shape.

Spring energies and continuous forces remain differentiable for a fixed realized topology and event branch. Bind, unbind, extrusion, stepping, capping, branching, severing, and stochastic acceptance are discrete branch decisions. Gradients across a change in those decisions are not represented as pathwise derivatives; freeze the event history or provide an explicit discrete estimator when differentiating a trajectory.
