# Dynamic dark-sector epoch runtime

A physically unbounded shower, cascade or event graph is represented as an unbounded
chain of durable finite-capacity epochs. Device arrays and compiled loops remain bounded.

## Event graph

`EventGraphRepository` stores immutable content-addressed `GlobalEntity`, `GlobalEvent`,
`GlobalEventEdge` and `GlobalWorkItem` records. `EventGraphEpochManifest` binds the exact
run, parent, compile/capacity/species/topology/matrix-element revisions, checkpoint,
conservation status and evidence. The run tip advances through one compare-and-swap
commit.

Entity identity binds both static frame ID and exact frame realization. Device-local
slots and ranks are never physical identities.

## Compiled epoch

`DarkSectorEpochPlan` fixes packet, event, product, radiation, work and frontier
capacities and widths. `DarkSectorEpochState` owns all fixed pools and conservation
ledgers. Overflow returns durable deferred work and backpressure before physical
mutation. `DarkSectorRunCoordinator` resumes the committed frontier, leases work and
publishes only complete epochs.

Crashes before the run-tip commit are invisible; crashes after it resume from the
committed epoch. Worker retries must produce identical digests. Garbage collection is
reachability-based from retained run tips/manifests.

## Matrix-element adaptation

`MatrixElementRevision` binds process/model/provider/rights, normalization, support,
proposal, adaptation/training/optimizer/error state and parameters. Adaptation occurs
only between committed epochs from immutable sufficient statistics and disjoint held-out
evidence. Accepted revisions govern future epochs only; historical event weights are
never rewritten.

## Semantically unbounded continuation

Decay and shower frontiers materialize deterministic `GlobalWorkItem` records. A finite
epoch may defer any amount of eligible future work; subsequent epochs can revise
capacities, species or topology while preserving stable IDs and parent relations. No
work is dropped or silently truncated.
