# Atomistic uncertainty and acquisition

`CommitteeAtomisticPotential` evaluates several compatible potential programs on the same
coordinate map and neighborhood. The reduction produces mean energy and forces plus energy,
per-atom, and force-disagreement evidence.

## Runtime policy

`OODPolicy` makes out-of-distribution behavior explicit:

- fail closed and reject the segment;
- continue while recording evidence;
- use a conservative smooth uncertainty penalty;
- request a segment-level fallback provider.

A conservative blend is defined as one scalar energy before differentiation. Never blend
forces independently: that generally creates a nonconservative field and invalidates energy
accounting.

Fallback is a segment boundary, not a compiled-step callback. Finish or reject the current
segment, materialize a frame, evaluate the configured provider on the host, and record both
the trigger evidence and fallback provenance.

## Acquisition

`AcquisitionPlan` scores accepted frames from committee evidence, applies threshold
and capacity limits, then performs deterministic farthest-point diversity selection over
provided descriptors. Records retain the source frame identity, score, descriptor, policy,
and committee identity.

## Production checks

1. Qualify every committee member on the same units, topology, coordinate map, and output
   convention.
2. Choose thresholds from held-out calibration data rather than a trajectory under active
   selection.
3. Persist raw evidence; a scalar score alone cannot diagnose which atoms or force
   components triggered the policy.
4. Keep the fallback evaluator and acquisition writer outside JIT-compiled integration.
5. Re-evaluate acquired configurations with the authoritative method before adding labels
   to a training set.
