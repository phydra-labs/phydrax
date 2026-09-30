# Advanced topology

## Exact matrices and maps

::: phydrax.topology.ExactIntegerCOO

::: phydrax.topology.ExactChainComplex

::: phydrax.topology.CellularChainMap

::: phydrax.topology.CellularPairMap

::: phydrax.topology.FilteredCellularChainMap

::: phydrax.topology.CellularChainContraction

::: phydrax.topology.compute_induced_topology_map

::: phydrax.topology.compute_mapping_cone_homology

## Field and extended persistence

::: phydrax.topology.FieldTopologyPlan

::: phydrax.topology.FieldTopologySnapshot

::: phydrax.topology.FieldTopologySeries

::: phydrax.topology.compute_extended_persistence

::: phydrax.topology.PersistentCohomologyResult

## Features and distances

::: phydrax.topology.PersistenceFeaturePolicy

::: phydrax.topology.persistence_image

::: phydrax.topology.diagram_wasserstein_distance

::: phydrax.topology.diagram_bottleneck_distance

::: phydrax.terms.FrozenTopologyTerm

::: phydrax.uq.TopologyEnsembleSummary

## Reduction and temporal topology

::: phydrax.topology.cancel_unit_pair

::: phydrax.topology.compute_structured_cubical_persistence

::: phydrax.topology.compute_vineyard

::: phydrax.topology.compute_zigzag_topology

## PDE, geometry, and dynamics

::: phydrax.topology.compute_rational_homology_basis

::: phydrax.exterior.HarmonicClassFrame

::: phydrax.solver.HarmonicConstraint

::: phydrax.exterior.HodgeSubspaceTracking

::: phydrax.topology.MergeTree

::: phydrax.topology.compute_cell_local_homology

::: phydrax.geometry.CertifiedImplicitCover

::: phydrax.dynamics.CellMapEnclosure

::: phydrax.dynamics.compute_conley_homology_index

## Integral homology

::: phydrax.topology.IntegralHomologyResult

::: phydrax.topology.compute_integral_homology

## Exact cochain products

`alexander_whitney_diagonal(topology, support, /)` and
`serre_diagonal(cubical, /)` construct canonical oriented diagonals for simplicial
and tensor-product cell complexes. Repeated periodic attaching occurrences retain
their multiplicity. `cup_product` requires both operand topology identities to
match the diagonal and reduces every product mod p before int64 accumulation.
Sheaf restrictions are keyed by incidence occurrence, not only cell pair.

::: phydrax.topology.alexander_whitney_diagonal

::: phydrax.topology.serre_diagonal
