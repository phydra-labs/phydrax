# Differentiable DEM

## Sharp branchwise mode

A sharp branchwise derivative is the JVP/VJP of the executed fixed-step program conditioned on unchanged pair routes, contact activity, no-tension branch, stick/slip state, material IDs, acceptance bits, barrier feature, and topology events.

`DEMSensitivityPolicy` declares minimum gap, no-tension, friction, frame, overlap, neighborhood, and perturbation margins. `DEMLocalValidityCertificate` marks a gradient usable only when forward physics succeeds and every margin passes. Invalid sensitivities are represented as invalid/NaN payloads, never silent zero gradients.

`DEMTrainableMaterialParameters` separates unconstrained continuous optimization coordinates from static material count/ID topology. Young modulus uses logarithmic coordinates, Poisson ratio a bounded logistic coordinate, restitution a logistic coordinate, and friction a softplus coordinate.

## Replay and checkpointing

`checkpointed_dem_rollout` rematerializes fixed scan blocks in reverse AD and records acceptance, rejection reasons, route digest, active/sliding counts, and cache epoch. A replay mismatch invalidates the VJP.

## Smooth surrogate mode

`SmoothPenaltyNormalPlan` and `SmoothCoulombTangentialPlan` define a separately fingerprinted modified model. Gap, force, direction, and projection smoothing scales are public parameters. `DEMSurrogateBiasCertificate` compares sharp and smooth force, energy, and observables against a declared tolerance.

## Inverse and UQ workflows

`DEMInverseProblem` vmaps independent fixed-capacity worlds. It reports loss, gradient, local certificate fraction, Jacobian singular values, rank, and condition number. An inverse gradient is usable only when every case is locally valid and the selected parameters are identifiable. Ensemble evaluation retains invalid-case rates instead of dropping failed simulations.

## Hybrid event mode

`DEMHybridEventPlan` localizes one bracketed guard, applies a reset, and computes the saltation matrix including event-time sensitivity. Grazing, simultaneous competing guards, unbracketed roots, nonfinite state, or residual above tolerance fail. General simultaneous-impact and topology-event sensitivities remain research limitations.
