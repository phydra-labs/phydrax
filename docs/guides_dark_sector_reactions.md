# Inelastic and radiative dark-sector reactions

The first admitted reaction profile is bounded nonrelativistic reversible `2 <-> 2`
scattering over weighted SIDM packets. There is no string-dispatched channel registry
and no unbounded particle creation.

## Reaction plans

`DarkTwoBodyReactionPlan` binds one typed incoming/outgoing species pair, forward and
reverse differential kernels, energy change, conserved charges, degeneracies, support,
units and identity. `InelasticSIDMPlan` owns a fixed tuple of channels and performs
partial-rate selection with semantic event randomness.

For one channel:

`Q = rest-mass change + internal-energy change - radiated energy`

`E_rel,out = E_rel,in + Q`.

Endothermic channels remain closed below threshold. Exothermic channels use the outgoing
reduced mass to construct recoil. A nonrelativistic-validity gate refuses events needing
a relativistic treatment.

Forward/reverse kernels must satisfy declared microreversibility and detailed balance,
including degeneracy and phase-space factors.

## Atomic dynamic-mass transaction

A successful event atomically updates:

- species marks;
- microscopic and internal masses;
- packet weights;
- gravitational macro masses;
- canonical momenta;
- retained child packet IDs and lineage;
- reaction epoch;
- dark-radiation state;
- dynamic PM mass evidence.

Charge, rest/internal/kinetic/radiation energy and physical/canonical momentum are
reported separately. Any support, threshold, capacity, detailed-balance, finite or
closure failure returns the original complete state.

## Dark-radiation ledger

`DarkRadiationLedgerPlan` appends one fixed-capacity `DarkRadiationPacket` with emitted
energy/momentum, species, source event, parent IDs and exact emission time level. It
reports four-momentum balance and rolls back on capacity exhaustion or invalid ledger
state.

The ledger is an export/local-reservoir profile. It is not resolved `2 -> n` transport.
Transport requires another fixed-capacity product and independent scientific claim.

## Restart

Inelastic checkpoints bind channel/species identities, packet support, radiation
capacity, state arrays, lineage and reaction epoch. Restore rejects changed channel,
species, support or capacity semantics before any continuation step.
