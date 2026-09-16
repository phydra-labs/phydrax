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
