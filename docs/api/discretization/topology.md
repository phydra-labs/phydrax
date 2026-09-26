# Discrete topology and support

Topology contains combinatorial entities and incidence. Support binds topology to one
embedding identity. Coordinates and numerical measures are not hidden inside entity
IDs.

This module is the canonical finite-topology carrier. Exact invariant analysis,
filtrations, and persistence live in [`phydrax.topology`](../topology/index.md).

::: phydrax.discretization.EntitySubset

---

::: phydrax.discretization.EntitySet

---

::: phydrax.discretization.OrientedIncidence

---

::: phydrax.discretization.TensorTopology

---

::: phydrax.discretization.CellComplexTopology

---

::: phydrax.discretization.PointTopology

---

::: phydrax.discretization.DiscreteSupport

---

::: phydrax.discretization.DiscreteMeasure

---

::: phydrax.discretization.CochainDiscretization

---

::: phydrax.discretization.CochainFieldSpec

---

::: phydrax.discretization.CochainBoundaryPolicy

## Distributed halos and fixed topology epochs

`DistributedHaloPlan` uses fixed padded local capacity and colored point-to-point peer
permutations. `DistributedLocalOperator` requires both the local forward and local
transpose actions and exposes serial references for qualification.

`TopologyEpochTransition` accepts only a conservative `FieldTransfer` with separate
dual pullback and Hilbert adjoint. Epoch selection itself remains nondifferentiable.

::: phydrax.discretization.DistributedHaloPlan

::: phydrax.discretization.DistributedLocalOperator

::: phydrax.discretization.TopologyEpoch

::: phydrax.discretization.TopologyEpochTransition

::: phydrax.discretization.TopologyEpochTransitionResult

## Device adaptive simplex epochs

`MaskedSimplexMesh` is the capacity-bucketed simplex layout (activity masks,
positively oriented cell rows, packed sibling half-facets; slot order equals
global-ID order). `AdaptiveSimplexState` adds the Maubach labels and bisection
forest; `refine_adaptive_simplex`, `coarsen_adaptive_simplex`, and
`refine_adaptive_simplex_parts` are module-level compiled entry points keyed by
the static `AdaptiveSimplexLayout`. Refusals return the input state with
`AdaptiveSimplexStatus` flags in the `AdaptiveSimplexReport`. Preparation and
commit live in `phydrax.meshing` (`prepare_adaptive_simplex`,
`commit_adaptive_simplex`).

::: phydrax.discretization.MaskedSimplexMesh

::: phydrax.discretization.masked_simplex_facet_neighbors

::: phydrax.discretization.masked_simplex_signature

::: phydrax.discretization.AdaptiveSimplexPolicy

::: phydrax.discretization.adaptive_simplex_bucket

::: phydrax.discretization.AdaptiveSimplexLayout

::: phydrax.discretization.AdaptiveSimplexState

::: phydrax.discretization.adaptive_simplex_state

::: phydrax.discretization.AdaptiveSimplexStatus

::: phydrax.discretization.AdaptiveSimplexCounter

::: phydrax.discretization.AdaptiveSimplexReport

::: phydrax.discretization.AdaptiveSimplexUpdate

::: phydrax.discretization.refine_adaptive_simplex

::: phydrax.discretization.coarsen_adaptive_simplex

::: phydrax.discretization.AdaptiveSimplexParts

::: phydrax.discretization.refine_adaptive_simplex_parts
