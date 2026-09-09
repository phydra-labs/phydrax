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
