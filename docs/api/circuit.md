# Circuit analysis

Phydrax provides typed power-wave scattering networks and grounded linear
modified-nodal-analysis circuits under `phydrax.circuit`.

Prepared harmonic-balance planning, refresh, resource evidence, and periodic
diagnostics are described in
[Periodic circuit analysis](../guides_circuit_periodic.md).

Frequency-domain MNA power evidence is an opt-in `MNAPowerLedger`, separate from solve
diagnostics and scattering contractivity. Transient and harmonic-balance accounting use
the separate `CircuitEnergyLedger` and `CircuitPeriodicEnergyLedger` contracts. See
[Circuit networks](../guides_circuit_networks.md), [Causal circuit dynamics](../guides_circuit_dynamics.md),
and [Periodic circuit analysis](../guides_circuit_periodic.md) for sign, phasor, support,
and availability semantics.

::: phydrax.circuit
