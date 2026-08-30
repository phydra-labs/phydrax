# Atomistic molecular learning

`phydrax.atomistic` is the finite, nonperiodic molecular data, prediction, and
training surface. `phydrax.nn.atomistic` owns drop-in PaiNN and low-degree
NequIP energy models. The public contracts keep atomic identity, units, graph
capacity, validity, and provenance explicit.

## Structures, batches, and graph realization

::: phydrax.atomistic.AtomisticScaleContract

::: phydrax.atomistic.AtomisticPrecisionPolicy

::: phydrax.atomistic.AtomicStructure

::: phydrax.atomistic.AtomisticBatch

::: phydrax.atomistic.AtomisticGraph

::: phydrax.atomistic.realize_atomistic_graph

## PaiNN and low-degree NequIP energy prediction

::: phydrax.nn.atomistic.AbstractAtomisticPotential

::: phydrax.nn.atomistic.checkpoint_atomistic_potential

::: phydrax.nn.atomistic.PaiNNPotential

::: phydrax.nn.atomistic.NequIPPotential

The NequIP implementation is a Cartesian O(3) research model restricted to
degrees zero, one, and two. It uses the same finite-molecule graph, prediction,
and training contracts as PaiNN. It does not claim periodic cells, stress,
long-range interactions, molecular dynamics, high-degree irreps, or MACE-style
symmetric contractions.

::: phydrax.atomistic.AtomisticPrediction

::: phydrax.atomistic.AtomisticProvenance

::: phydrax.atomistic.AtomisticStatus

::: phydrax.atomistic.energy_and_forces

## Low-degree O(3) tensor products

::: phydrax.nn.operator.layers.O3TensorProductPlan

::: phydrax.nn.operator.layers.O3TensorProduct

## Typed training

::: phydrax.atomistic.AtomisticTrainingProblem

::: phydrax.atomistic.AtomisticTrainingPolicy

::: phydrax.atomistic.AtomisticTrainingNormalization

::: phydrax.atomistic.AtomisticTrainingResult

::: phydrax.atomistic.fit_atomistic_potential

## Local rMD17 data

::: phydrax.atomistic.RMD17Dataset

::: phydrax.atomistic.RMD17Split

::: phydrax.atomistic.load_rmd17_npz

::: phydrax.atomistic.split_rmd17
