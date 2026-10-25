# Native atomistic model families

Native architecture execution, source-checkpoint conversion, scientific validation,
weight rights, and device qualification are separate claims. A random native model
or source-layout fixture is not evidence of pretrained chemical accuracy.

## Models and observables

| Family | Native owner | Scalar energy | Conditioning |
|---|---|---|---|
| MACE | `MACEPotential` | Source-defined scalar | Optional declared joint embeddings, including physical charge/multiplicity where configured |
| MACELES | `MACELESPotential` | Local MACE plus global latent interaction | Underlying MACE model's declared inputs |
| AIMNet2 | `AIMNet2Potential` | Source-model energy and configured embedded corrections | Continuous total charge; two-channel NSE adds spin multiplicity |
| Conservative UMA | `UMAPotential` | Force-generating scalar of the admitted source head | Categorical charge, spin index, and task |
| Property-only UMA | `UMAPropertyModel` | Absent | Same categorical source inputs |

The numerical model boundary is
`evaluate_graph(AtomisticGraphInputs, request) -> AtomisticModelOutputs`.
The graph owns current coordinates and routed geometry. Conditions retain their
ports, units, numerical kinds, availability, and explicit default-use evidence.
Request additional properties explicitly: a potential's default evaluation need
not execute its direct-force or other auxiliary heads.

Conservative forces and tensile-positive stress derive from the same scalar.
`atomistic_energy_derivatives(model, payload, positions, /, ...)` is the numerical
boundary over an already prepared `AtomisticGraphPayload`; it does not rediscover
neighbors or hash numerical content. Host `energy_and_forces` accepts structures
or batches and records complete provenance.

### Local, global, and reported energy

MACELES exposes its latent interaction through `GLOBAL_ENERGY_COMPONENT`, with
`LATENT_CHARGES` rather than physical `CHARGES`. Its local atomic contributions
are not a decomposition of the global term; `atom_energy_complete=False` preserves
that distinction. No arbitrary atomic shares are invented.

For periodic MACELES, mode-sphere regularity metadata names the parametric
boundary `|k| * dl = 2 pi`. It does not convert the dynamic reciprocal spacing
into a host scalar while binding compiled Hessian or mixed parameter/HVP
actions. The definition's numerical state and identity retain the exact `dl`;
`reciprocal_cutoff` supplies its numerical cutoff. Reprepare derivative-admission
objects that used the former value-specific condition identifiers.

A UMA mean-reduced source report is not automatically a force-generating
potential. Admitted gradient-force configurations keep their scalar definition
separate from `REPORTED_ENERGY`. A property-only configuration returns absent
energy, not zero, and refuses conservative energy/force/stress training and
Hessian requests. `DIRECT_FORCES` and `DIRECT_STRESS` are source-head reports,
not relabeled energy gradients.

### Conditions and scientific graph selection

AIMNet2's physical charge condition remains continuous. Its NSE charge update
retains the source epsilon in the denominator, so `CHARGE_RESIDUAL` reports the
difference from the target instead of falsely asserting exact conservation.
The same finite epsilon means exact supercell extensivity is not guaranteed:
replicating a case changes the denominator's epsilon relative to its summed
flexibility. Native execution preserves that source definition rather than
renormalizing charges to force extensivity.
Spin multiplicity and UMA's source spin index are distinct ports: UMA index zero
is a null token, not physical multiplicity zero. Task defaults and unmapped or
out-of-range categories remain explicit.

UMA's nearest-neighbor cap and degeneracy band are part of its scientific model.
The graph must carry that selection. Exceeding the scientific cap is not the same
as exhausting candidate storage; capacity failures retain their own evidence.
Conditioned source mixtures and other discrete selections are fixed for a
continuous derivative request.

## Higher-order response

`prepare_atomistic_hessian` is the host identity/preparation boundary;
`bind_atomistic_hessian` is its numerical, fixed-payload counterpart. Both compose
the native force linearization and matrix-free actions. Parameter gradients of an
HVP loss require an explicit mixed third-order request, not merely an ordinary
coordinate-Hessian request.

| Realization | Coordinate response | Mixed parameter-HVP response |
|---|---|---|
| Admitted exact MACE | Classical Hessian under its declared support | Almost-everywhere policy where third-order cutoff joins require it |
| AIMNet2 positive-DispParam/source energy paths | Piecewise; explicit almost-everywhere policy | Same policy plus the declared mixed request |
| Raw conservative UMA | Piecewise; explicit almost-everywhere policy | Same policy plus the declared mixed request |
| Frozen merged conservative UMA | Coordinate response on the same admitted pieces | Refused: frozen merged buffers are not trainable parameters |
| Property-only UMA | Refused | Refused |

For AIMNet2, the source cosine distance floor is not globally smooth; hard TS
truncation and thresholded Softplus may be discontinuous. Only output stages
actually feeding scalar energy contribute to its regularity contract. A TS input
not established as positive DispParam output is not granted that support claim.
Repulsion envelope buffers, internal activations, and dispersion clamp joins keep
explicit conditions where geometry alone cannot certify them.

For UMA, capped-neighbor membership can change inside the cutoff. Finite S2 edge
quadrature does not establish exact roll-gauge invariance. Normalization/frame
joins and parameter-only pnorm routing kinks are kept distinct. A keyed gauge
requires one fixed key and stable route identities throughout differentiation;
it is not an expectation over resampled rolls.

Strict regularity policy refuses these piecewise Hessians. Explicit
`RegularityPolicy(allow_almost_everywhere=True)` admits branchwise actions and
retains `unresolved_conditions`. Successful numerical action status is not proof
that every internal selection, activation, or gauge margin was certified. Known
coincidence and edge-radius violations invalidate the point. Do not infer support
from finite results or advertise a classical self-adjoint Hessian when only an
almost-everywhere derivative was admitted. Inspect the actual realization's
`derivative_contract`; source and accelerated/tabulated variants can differ.

## Prepared models and durable artifacts

MACE preparation and UMA expert merging bind exact source revisions. UMA merging
also binds composition, charge, spin, and task. A changed revision or material
binding requires explicit preparation again. Prepared raw models and merged blocks
are deliberately frozen through `ExplicitFreeze`; fixed arrays remain dynamic
PyTree leaves rather than hidden static numerical data.

`PreparedUMAMergedModel.prepared_identity` and `source_revision_identity` carry
lossless dynamic digest words. Their host properties `prepared_id` and
`source_revision_id` decode the full digests; scientific architecture and method
identity belong to `.model`. Restore validation replays the binding, recomputes
merged operators exactly, and checks the source revision and prepared identity.

Native model artifacts restore registered types and exact array inventories,
validate native structural contracts, rerun family scientific validators, and
recompute identities. Source provenance preserves its MACE, AIMNet2, or UMA
family-specific admission evidence. No Torch object or source runtime is used
for native inference after conversion.

A composed dynamics restart captures every learned term, not one shared model
assumed to stand for them all. `write_atomistic_restart` derives the executed
terms from its plan; `read_atomistic_restart_models` returns each restored model
artifact and constructor-rebuilt term binding. Conditions, defaults, optional
absence, keys, and requested outputs survive with the complete dynamics state.
See [atomistic dynamics](guides_atomistic_dynamics.md) for resource rebinding and
continuation.

## Source and qualification boundaries

Source reconstruction consumes or explicitly refuses each declared source tensor
and configuration entry. Conversion proves an admitted parameter/basis mapping;
independent source-oracle campaigns establish fidelity. Predicted physical charges
can feed native electrostatics through `AtomisticFieldReference`, keeping their
coordinate response connected. Follow the declared calculator composition to avoid
counting an embedded correction twice.

Pretrained AIMNet2 member digests/licenses, gated UMA weights/terms, and applicable
MACE model licenses remain separate prerequisites. Source-framework installations
are isolated extraction/oracle environments, never native product runtimes.

Hardware, precision, derivative order, topology, and capacity must be qualified as
one explicit support tuple. No source TF32 policy is inherited automatically.
The model-family benchmark separates construction, payload preparation, lowering,
compilation, first/warm execution, compiler memory, and retained arrays. Its
`--cutoffs` axis (`native` keeps each family's own radius) rebuilds the local
radius through each family's canonical configuration: the MACE architecture
cutoff (for MACELES the local MACE, never the latent Ewald `dl`), the AIMNet2
source `aev.rc_s`, and the UMA source `cutoff`. AIMNet2's distinct short-range
Coulomb radius stays declared, so an AEV radius below it is refused. Each row
records the realized local radius and every distinct declared radius. Missing
source-build attestation or provider/hardware prerequisites remain explicit rather
than being replaced by an invented identity or a random fixture.

Each measured workload retains the canonical `PhaseMemoryEvidence` payload:
bracketing RSS and device allocator observations, interval-sampled peaks, sample
count and maximum sampling gap. Kernel records bracket warm execution; fit and
merge records bracket their complete workload. Allocator lifetime high-water
marks are separate from interval-sampled peaks, and neither compiler estimates
nor sampled peaks are presented as certified process/device memory bounds.
