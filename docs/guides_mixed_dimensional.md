# Mixed-dimensional transport

Phydrax represents a physical embedded network as an interval `CellMesh` plus a
`MetricNetworkPlan`. Preparation computes oriented incidence, edge lengths and
tangents, cross-sectional areas, perimeters, and lumped node measures. Gradient,
divergence, diffusion, upwind flux, and mass actions are JAX-compatible.

## Embedded measure transfer

`EmbeddedMeasureTransferPlan` prepares circle or ball averages from a tetrahedral
bulk mesh to fixed embedded sites. Host preparation constructs quadrature points,
locates them in tetrahedra, and stores fixed sparse routes.
`EmbeddedSourceAssociation.VERTEX` and `.CELL` explicitly distinguish P1 from
DG0 sources; equal vertex/cell counts are never used to infer the field space.

The runtime exposes:

1. the primal average T;
2. the algebraic dual pullback Tᵀ;
3. the measure-weighted Hilbert adjoint.

These roles are distinct in `FieldTransfer` as `primal_operator`,
`dual_pullback_operator`, and `hilbert_adjoint_operator`.

For bulk concentration c₃ and network concentration c₁, a
`PermeabilityExchangePlan` computes exchange density

```text
j = κ (T c₃ − c₁)
```

and contributes `Tᵀ(M₁ j)` to the bulk residual and `−M₁ j` to the network
residual. The contributions cancel exactly in the total mass ledger. Terminal
reservoir coupling uses the same equal-and-opposite construction.

Circle averages require one tangent per target site. Ball averages use a
positive symmetric rule. Every quadrature route must be supported; partial
kernel truncation and compartment mixing are not silently inferred.

`lower_distributed` records target and donor-route ownership without changing
global identities. It does not partition the underlying bulk or network by
itself; it binds a prepared route to existing ownership plans.

## Transport

`BulkDGTransportPlan` implements conservative tetrahedral DG0 transport:

- cellwise porosity and mass;
- scalar or SPD tensor diffusion;
- internal-face two-point diffusion;
- upwind advection from fixed cell velocity;
- signed exterior volume flux and declared inflow concentration;
- cellwise nonnegative implicit removal.

`NetworkTransportPlan` implements conservative graph advection and diffusion.
`MixedDimensionalTransportPlan` combines bulk, network, permeability exchange,
and terminal reservoirs. It provides backward Euler and first-order IMEX Euler:
advection is explicit in the IMEX route while diffusion and exchange remain
implicit.

`MixedDimensionalMassLedger` reports bulk, network, reservoir, and total mass,
external boundary/removal loss, exchange cancellation, balance defect, and
minimum concentration. Rejected linear solves retain the accepted state.

The public state remains a structured `MixedDimensionalTransportState`; flattening
is internal to the block linear solve. `diagonal_preconditioner` supplies a mass
and exchange scale for native Krylov composition.

Run `python examples/neurofluid_transport.py` for a closed 3D–1D–0D exchange
system with total-mass verification.
