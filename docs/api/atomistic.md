# Atomistic molecular learning

`phydrax.atomistic` is the finite, nonperiodic molecular data, prediction, and
training surface. `phydrax.nn.atomistic` owns the PaiNN energy model. The public
contracts keep atomic identity, units, graph capacity, validity, and provenance
explicit.

## Structures, batches, and graph realization

::: phydrax.atomistic.AtomisticScaleContract

::: phydrax.atomistic.AtomisticPrecisionPolicy

::: phydrax.atomistic.AtomicStructure

::: phydrax.atomistic.AtomisticBatch

::: phydrax.atomistic.AtomisticGraph

::: phydrax.atomistic.realize_atomistic_graph

## PaiNN energy and conservative-force prediction

::: phydrax.nn.atomistic.PaiNNPotential

::: phydrax.atomistic.AtomisticPrediction

::: phydrax.atomistic.AtomisticProvenance

::: phydrax.atomistic.AtomisticStatus

::: phydrax.atomistic.energy_and_forces

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
