# Circuit networks

`phydrax.circuit` composes frequency-domain component models without replacing the
field solver, linear solver, optimizer, or uncertainty substrate that produced them.
It has two explicit formulations: power-wave scattering networks and grounded linear
modified-nodal-analysis circuits.

## Conventions

The time convention is `exp(-i ω t)`. A scattering matrix is ordered as
`S[..., output, input]`; incident waves point into a component and outgoing waves point
out. Generic circuit amplitudes are square-root-watt power waves, so the power entering
a port set is `sum(abs(a)**2 - abs(b)**2)`.

Electrical references use RMS voltage/current phasors with current positive into the
component and finite `real(Z0) > 0`. Modal references carry the exact mode, basis,
orientation, normalization, propagation, and reference-plane identities supplied by
the field solver. Equal array shapes do not make ports compatible.

A direct link requires matching references. Impedance renormalization, modal basis
transformation, and reference-plane shifts are explicit operations with diagnostics;
they are never hidden connection behavior.

Wave ports are block-valued. `coordinate_ids` give the stable flattened channel order;
a scalar port is a block of size one. A direct block connection pairs coordinates in
order, while `WaveConnectionMap` permits an explicit lossless unitary coordinate map.
Port count and channel count are therefore distinct.

## Scattering components and networks

A scattering component exposes its ordered ports without numeric evaluation and returns
a frequency-leading `ScatteringResponse`. Planning a network recursively flattens
component hierarchy, assigns stable leaf/port ordering, verifies every pairwise link,
and compiles external-port and internal-probe gathers. Physical feedback loops are
valid; recursive model-definition cycles are not.

For component relations `b = S a` and pairwise connection operator `C`, external
injection `E u` gives

```text
(I - C S) a = E u
b = S a
```

Preparation forms one native `phydrax.linalg` problem and retains the original defects
`b - S a` and `a - C b - E u`. Singular or ill-conditioned feedback is reported by the
native solve; Phydrax does not add epsilon, a pseudoinverse, or a fallback backend.

Ordinary solves evaluate only supplied excitation columns. `scattering_submatrix`
requests selected input/output ports, while `full_scattering_matrix` explicitly pays for
one right-hand side per external input. Internal probes gather solved incident/outgoing
waves and do not insert a physical tap.

`ScatteringNetworkPolicy` separately bounds channel count, retained matrix/factor
storage, and right-hand-side working storage. A solve whose supplied column count
exceeds the declared RHS envelope fails before allocating internal RHS arrays.

`prepare_scattering_action` compiles the same physical equation as sparse block actions
and solves it with native FGMRES without a global dense system matrix. It is currently
one-frequency-per-preparation; explicit case batches and recycled sequential sweeps
cover frequency ensembles. Dense execution remains the bounded reference path.


## Fourier-modal components

The Fourier-modal adapter admits only propagating, nongrazing, finite-unit-flux modes
whose basis and reference planes are known. Evanescent or grazing modes remain in the
Fourier-modal boundary cascade and Redheffer path; discarding them is a separated-device
approximation, not exact near-field composition.

The specialized Maxwell block order differs from canonical circuit order. The adapter
performs the required left/right output reordering and records that provenance instead
of exposing the raw block matrix directly.

## Grounded MNA circuits

An `MNAStamp` stores the local frequency-domain blocks

```text
[Y  B]
[C  D]
```

for terminal voltages and auxiliary variables. Planning canonicalizes electrical nodes,
requires and eliminates an explicit ground, assigns terminal/auxiliary offsets, and
compiles one fixed assembly. Numeric stamp values refresh without changing topology.
The initial production formulation rejects floating or disconnected voltage gauges;
there is no implicit GMIN or arbitrary node pinning.

Dense MNA is explicitly hard-budgeted and supports frequency batches. Sparse MNA
prepares one scalar frequency at a time, emits only the compiled edge relation and
its coefficient vector, and requires the native `FGMRES` action path; it never
materializes a dense matrix.


External nodal ports are ordered positive/negative node pairs with electrical wave
references. Unexcited ports remain terminated in their declared reference impedances.
S, Y, Z, and two-port ABCD conversion is explicit and retains source/target convention,
eligibility, residual, and reference metadata. Under `exp(-i ω t)`, capacitor
admittance is `-i ω C` and inductor impedance is `-i ω L`.

`CircuitInstance` is the analysis-independent nodal instance name. Built-in RLC
elements retain frequency laws and also lower to causal implicit laws. Dynamic-only
sources, behavioral laws, and learned dissipative laws are composed through
`CircuitElement` and fail closed if a missing frequency law is requested by MNA.

## Interchange

The SPICE adapter accepts a strict RLC and independent-source subset with hierarchical
subcircuits and explicit external `NodalPort` definitions. Unsupported statements are
errors. The restricted behavioral compiler accepts a validated scalar expression for
two-terminal current; it does not execute arbitrary source code.

## Scientific evidence

A successful circuit result requires finite values, native linear-solve success, and
absolute and scale-safe relative residuals of the original equations.
`MNADiagnostics` therefore reports solve and equation evidence only. It does not report
a port-power balance: wave power and terminal power reconstructed from the same port
voltage/current pair are an identity, not independent conservation evidence.

`evaluate_mna_power_ledger` is an explicit, opt-in conservation audit. MNA voltages and
currents are RMS phasors. Port current is positive into the circuit, while each element
terminal current is positive into that element; consequently the retained port
contribution is power absorbed by the external port. The `MNAPowerLedger` keeps every
port and supported RLC element real/reactive contribution separately and then assesses
their complex-power closure. With `exp(-i ω t)`, reactive power is
`imag(V conj(I))`; ideal capacitors therefore contribute positive reactive power and
ideal inductors negative reactive power under this convention. Source contribution
axes are also separate, but ordinary MNA has no independent-source forcing stamp, so
it never infers source power from an incident wave.

Controlled, source-like, and black-box admittance/impedance elements do not acquire a
power model merely because their terminal solution is available. Their IDs and reasons
make the ledger unavailable, with closure residuals left undefined. A retained ledger
can be independently reassessed with `assess_mna_power_ledger`; changing one contribution
therefore changes closure rather than being hidden by a second port identity.

Whole-matrix passivity and reciprocity audits remain distinct: they require a
materialized complete matrix and compatible power-normalized bases, and selected
columns cannot certify either property. Scattering contractivity is not MNA power
closure. Active linear components are supported but never inferred passive.

Gradients are real derivatives of real objectives through complex arithmetic and the
native implicit solve. Topology, port order/type, mode selection, frequency-array shape,
file parsing, and rank changes are replan or nondifferentiable boundaries. A singular
primal solve has no valid implicit derivative.
