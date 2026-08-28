# Production particle data plane

`MultiPopulationCellPlan` builds population-specific cell tables without
constructing a synthetic combined particle set. A `ParticleSearchKey` selects a
qualified target/source population pair and search radius. Runtime bipartite
packing traverses only source occupants from neighboring target cells and reports
population overflow, domain violations, pair count, and pair overflow.

`PreparedSearchGroup` records searches sharing the same prepared cell state.
`fused_bipartite_interaction` is the pure-JAX reference for an interaction that
computes and reduces pair payloads without exposing materialized edge features.
Materialized relations remain available for graph export and edge-local state.

Stable logical order is independent of cell storage order. Inside differentiated
execution, any capacity failure rejects the state. Host orchestration may rebuild
and replay only from a previous accepted checkpoint.
