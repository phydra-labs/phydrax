# Lattice-field applications

The application layer composes canonical topology, quantum-register,
local-Hamiltonian, tensor-network, and solver owners. It contains no alternate
execution engine.

## Hamiltonian Z2 gauge theory

::: phydrax.applications.lattice_field.Z2GaugeModel

::: phydrax.applications.lattice_field.prepare_z2_gauss_sector

::: phydrax.applications.lattice_field.z2_gauge_hamiltonian

::: phydrax.applications.lattice_field.z2_gauss_terms

::: phydrax.applications.lattice_field.z2_loop_operator

::: phydrax.applications.lattice_field.z2_homology

The exact Gauss-sector enumeration is resource bounded. `PrimeField(2)` is
used only for host-side topology/homology calculations; runtime states remain
ordinary finite qubit arrays.

## Open Schwinger chains

::: phydrax.applications.lattice_field.SchwingerChainModel

::: phydrax.applications.lattice_field.schwinger_local_hamiltonian

::: phydrax.applications.lattice_field.schwinger_mpo

::: phydrax.applications.lattice_field.reconstruct_schwinger_flux

::: phydrax.applications.lattice_field.schwinger_gauss_residual

::: phydrax.applications.lattice_field.schwinger_observables

::: phydrax.applications.lattice_field.schwinger_background_schedule

::: phydrax.applications.lattice_field.schwinger_local_background_schedule

The model is the explicitly declared open, one-dimensional staggered spin
encoding. The exact local expansion is intended for small references; the
electric prefix-square MPO has constant bond dimension. Periodic and
higher-dimensional Gauss-law elimination are not claimed.

## Production lattice contracts

::: phydrax.applications.lattice_field.LatticeRegulator

::: phydrax.applications.lattice_field.LatticeTheoryPoint

::: phydrax.applications.lattice_field.LatticeEnsemblePlan

::: phydrax.applications.lattice_field.ContinuumExtrapolationPlan

::: phydrax.applications.lattice_field.LandauGaugeFixingPlan

::: phydrax.applications.lattice_field.CoulombGaugeFixingPlan

## Distributed QCD and interchange

::: phydrax.applications.lattice_field.DistributedGaugeTheoryPlan

::: phydrax.applications.lattice_field.DistributedHMCPlan

::: phydrax.applications.lattice_field.DistributedRHMCPlan

::: phydrax.applications.lattice_field.GaugeIOPlan

::: phydrax.applications.lattice_field.EnsembleManifest

::: phydrax.applications.lattice_field.MeasurementSchedule

## QCD observables and recipes

::: phydrax.applications.lattice_field.HypercubicGaugeObservablePlan

::: phydrax.applications.lattice_field.PropagatorSolvePlan

::: phydrax.applications.lattice_field.WilsonFlowPlan

::: phydrax.applications.lattice_field.QuenchedSU3Recipe

::: phydrax.applications.lattice_field.WilsonCloverNf2Recipe

::: phydrax.applications.lattice_field.StaggeredHisqStyleRHMCRecipe

::: phydrax.applications.lattice_field.ContinuumStudyPlan
## Finite-density QCD

::: phydrax.applications.lattice_field.ChemicalChargeConvention

::: phydrax.applications.lattice_field.SusceptibilityEstimate

::: phydrax.applications.lattice_field.prepare_taylor_eos

::: phydrax.applications.lattice_field.evaluate_taylor_eos

::: phydrax.applications.lattice_field.solve_heavy_ion_path

::: phydrax.applications.lattice_field.canonical_sector_transform

::: phydrax.applications.lattice_field.evaluate_qcd_reweighting

::: phydrax.applications.lattice_field.evaluate_ideal_boltzmann_hrg

::: phydrax.applications.lattice_field.build_taylor_eos_table

::: phydrax.applications.lattice_field.qualify_eos_table


## Retained-link Hamiltonian theories

::: phydrax.applications.lattice_field.PeriodicSchwingerModel

::: phydrax.applications.lattice_field.CompactU1GaugeModel2D

::: phydrax.applications.lattice_field.GaugePreservingProductFormulaPlan

::: phydrax.applications.lattice_field.GaugeSimulationResourcePolicy
