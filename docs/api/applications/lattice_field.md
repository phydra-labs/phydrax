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
