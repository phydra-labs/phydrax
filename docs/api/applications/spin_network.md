# Canonical spin networks

::: phydrax.applications.spin_network

This research-only surface represents one finite oriented graph with fixed
doubled SU(2) edge spins and one explicit incident-edge order at every vertex.
`prepare_spin_network` constructs a left-associated total-spin-zero intertwiner
basis at each vertex using the native Condon–Shortley Clebsch–Gordan owner. The
global basis is the product of vertex intertwiner multiplicities.

Evidence retains vertex transform orthonormality, projector idempotence, Gauss
invariance, graph identity, and exact finite resource dimensions. Edge area
values use the caller-declared positive Immirzi parameter and Planck length
squared with the convention `8 pi gamma l_P^2 sqrt(j(j+1))`.

`SpinNetworkState` stores amplitudes only over the admitted fixed-graph
intertwiner basis and audits normalization. It does not implement graph-changing
dynamics, a Hamiltonian constraint, semiclassical states, continuum geometry,
or a quantum-gravity claim.

## Recoupling, geometry, coherent states, and graph moves

`SpinNetworkCouplingTree` supplies multiple ordered intertwiner charts for one
vertex. Canonical product-space embeddings produce unitary recoupling maps.
Projected angle, oriented-volume, length, and flux-norm operators retain
Hermiticity evidence. Livine–Speziale coherent intertwiners report closure and
normalization.

`refine_spin_network_edge` performs an exact bivalent identity refinement.
`RegulatedHamiltonianConstraintPlan` requires finite move matrices, lapse
values, graph-change IDs, regularization, and ordering. Its result remains a
finite graph-dependent regulated action, not a unique continuum Hamiltonian.
