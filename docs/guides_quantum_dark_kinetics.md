# Quantum-statistical dark kinetics

Quantum statistics is a separate occupation-space profile. Packet weights are not
occupancies and never receive ad hoc `(1 +/- weight)` factors.

## State

`QuantumKineticState` stores dimensionless occupation per internal state on fixed
species/spatial/momentum support, Bose/Fermi marks, optional condensate density, exact
unit/frame realization and accepted time. Fermions require eigen/point values in `[0,1]`;
bosons require nonnegative occupation.

## Uehling--Uhlenbeck collisions

`UehlingUhlenbeckPlan` applies one shared `2 <-> 2` event flux:

```text
f3 f4 (1+s1 f1)(1+s2 f2) - f1 f2 (1+s3 f3)(1+s4 f4)
```

with `s=+1` for bosons and `s=-1` for fermions. Stoichiometric updates share one flux,
so charge and four-momentum moments cancel. Bound-preserving subcycling rejects rather
than clips. Evidence includes detailed balance, invariants, Fermi/Bose bounds, entropy
production and rollback.

## Condensate coupling

`CondensateCouplingPlan` is an explicit normal/condensate interaction profile. When a
normal gas reaches its supported condensation boundary without this profile, the state
returns `CONDENSATE_REQUIRED`; no finite bin is overloaded to mimic a delta mode.

## Qualification

Required cases include the dilute classical limit, Fermi blocking, Bose enhancement,
FD/BE equilibrium, pointwise detailed balance, H theorem, conserved moments and restart.
Topology/support changes remain nondifferentiable.
