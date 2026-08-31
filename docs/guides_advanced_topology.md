# Advanced computational topology

This guide extends the exact finite-complex substrate with maps, derived algebra,
field topology, metric class frames, temporal diagnostics, and certification-aware
geometry and dynamics.

## Functorial topology

`CellularChainMap` uses column chains and validates
`boundary_target @ F == F @ boundary_source` exactly over the integers. A
`CellularPairMap` additionally proves that the relative subcomplex maps into the
target relative subcomplex and constructs the quotient map. `FilteredCellularChainMap`
adds an explicit nonnegative filtration shift.

Induced homology and cohomology coordinates require named cycle/cocycle bases.
Cohomology is contravariant; its coordinate matrix is not generally the transpose of
the homology matrix. Mapping-cone acyclicity is always qualified by its coefficient
field.

## Field topology and extended persistence

`FieldTopologyPlan` binds one compact complex, explicit cell-vertex support, threshold
axis, coefficient field, and lower/superlevel convention. A snapshot is exact for the
declared finite filtration. A series requires one fixed topology and compact layout;
changing meshes require exact maps rather than a vineyard label.

`compute_extended_persistence` builds one finite sublevel-to-relative module and
decomposes its induced maps. It returns ordinary, relative, extended-positive, and
extended-negative components separately. Components retain raw node and field
coordinates; plotting conventions remain downstream.

## Diagram calculus

Packed diagrams support Betti curves, total persistence, and persistence images in
JAX. A frozen pairing exposes local endpoint derivatives only while the complete order
remains valid.

Persistence-specific Wasserstein assignment augments each diagram with its own
diagonal copies and uses the native certified Hungarian solver. Bottleneck distance
uses thresholded perfect matching; it is not a Hungarian sum-cost objective. Essential
bars only match compatible essential bars.

`FrozenTopologyTerm` evaluates a scalar model field on declared vertex points, pushes
it through a prepared star filtration, and returns ordering validity and margin in its
term diagnostics. Pairing refresh remains an outer host operation.

## Scalable and temporal paths

`compute_persistent_cohomology` returns the field-equivalent interval decomposition
and exact terminal cocycles. `compute_h0_persistence_union_find` is the specialized H0
path. `cancel_unit_pair` performs exact integral elementary cancellation and validates
the resulting chain complex. Structured cubical persistence reuses the canonical
`StructuredCochainBridge`; it does not invent a foreign cube indexing scheme.

A vineyard is a fixed-layout sequence of exact persistence snapshots with stable
creator-cell lineage. `compute_zigzag_topology` validates insert/remove closure and
returns exact homology after every operation. It does not mislabel a Betti history as a
full zigzag interval decomposition.

## PDE topology

Exact rational representatives label the free homology classes. A
`HarmonicClassFrame` realizes those labels as metric harmonic cochains and normalizes
their period matrix. `HarmonicConstraint` declares prescribed, free, gauge, or
deflated period policy; the topology layer never chooses the physical policy.

`HodgeSubspaceTracking` compares common-space metric projectors and principal angles.
Different meshes must first be transferred to one declared common complex and metric.

## Integral and certified claims

Integral homology uses chain-compatible Smith transformations: incoming boundaries
are expressed in an exact kernel basis before torsion invariant factors are computed.
It never infers torsion from selected finite fields.

`CertifiedImplicitCover` consumes externally established interval value and gradient
bounds. `CertifiedImplicitTopology` binds those bounds to a finite topology and an
explicit theorem identifier. Failure to establish regularity leaves certification
false.

`CellMapEnclosure` is an outer multivalued cell map with a declared index pair.
`compute_conley_homology_index` computes relative homology only after isolation
evidence succeeds. For discrete maps this is the homology index; a full Conley index
map additionally requires an induced index endomorphism.

## Deliberate gates

The public API does not expose multiparameter barcodes, generic cup products, cellular
sheaves, or spectral-sequence pages. Those require multigraded presentations, explicit
cell diagonals, stalk/restriction data, or a concrete filtered bicomplex respectively.
