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

::: phydrax.discretization.AbstractCellDeRhamComplex

::: phydrax.discretization.DiagonalHodge

::: phydrax.discretization.SparseHodge

`CochainDiscretization(topology, hodges, /, *, boundary_masks=None,
coordinates=None, primal_measures=None, dual_measures=None, key=None,
numeric_revision=None, time=0.0, differentials=None)` binds topology to
metric-only Hodges. Use `boundary="absolute"` or `boundary="relative"` on
calculus calls. Relative operators restrict to active coordinates, invert the
restricted pairing, and zero-extend the result; they never mask a full inverse.

Form metadata is declared by `phydrax.exterior.FormType`. Obtain a degree's vector
space with `complex.hilbert_complex(boundary="absolute").space(k)`.
See [exterior complexes](../exterior/complexes.md).

## Distributed halos and fixed topology epochs

`DistributedHaloPlan` uses fixed padded local capacity and colored point-to-point peer
permutations. `DistributedLocalOperator` requires both the local forward and local
transpose actions and exposes serial references for qualification.

`TopologyEpochTransition` accepts only a conservative `FieldTransfer` with separate
dual pullback and Hilbert adjoint. Epoch selection itself remains nondifferentiable
(`differentiation_available`), while values differentiate through the frozen
transfer (`value_derivative_available`): `pullback` is the coordinate dual, the
VJP of `apply`, and `adjoint` the Hilbert adjoint in the field-space pairings.

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
the static `AdaptiveSimplexLayout`. Every call records its `AdaptiveSimplexStatus`
flags in `AdaptiveSimplexState.status_flags`, accumulated over the prepared
epoch; a failed call rolls every other array back but records its terminal
flags, and every later call on that state is refused on device
(`AdaptiveSimplexReport.failed`, zero operation counts). Preparation and commit
live in `phydrax.meshing` (`prepare_adaptive_simplex`,
`commit_adaptive_simplex`, which rejects an epoch holding a terminal flag).

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
