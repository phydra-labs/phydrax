# Skeletal-muscle personalization and UQ

`SkeletalObservationChannel` binds one fixed-shape observation to a channel ID,
canonical quantity ID, calibration/asset ID, values, standard uncertainty, and valid
mask. `SkeletalMultimodalLikelihoodPlan` evaluates an unweighted sum of normalized
Gaussian channel likelihoods while retaining channel-resolved residuals and evidence.
It is an application adapter, not a second inference engine.

Missing samples are represented only by the fixed valid mask and contribute exactly
zero. Active samples require finite values and positive uncertainty. Prediction order
must match the immutable channel order; channels are never inferred from array shape.

The resulting scalar likelihood plugs directly into `phydrax.uq.ParameterSpace` and
`PosteriorProblem`, then existing MAP, MCMC, variational, ensemble, profile,
sensitivity, and experiment-design owners. Hard recruitment, spike, wrap, slack,
contact, and rollback boundaries require fixed-regime or derivative-free/hybrid
inference. A finite JAX gradient is not an identifiability claim.

Force-only data cannot independently identify neural drive, thresholds, relative-force
scale, physical-force scale, moment arms, tendon slack, and passive stiffness. EMG,
force, torque, kinematics, and energetics are admitted only through their own physical
observation models and shared timebase. Held-out protocols, subjects, sessions, and
electrodes remain mandatory for physiological claims.

Run `examples/skeletal_muscle_multimodal_uq.py` for a force-plus-surface-EMG likelihood
inside the generic posterior owner.


## Exact replay for surrogate-controlled decisions

`SkeletalSurrogateReplayPlan` owns an identified source `ControlProblem`, its
control parameterization, an exact observation projection, a valid mask, and error
tolerances. Evaluation re-executes the candidate controls through
`ControlProblem.evaluate(...)`, retains the complete `ControlResult`, and requires
successful dynamics, objective, and sampled path/terminal feasibility before
comparing the surrogate prediction. Callers cannot supply arbitrary values labeled
“exact.”

Acceptance requires every active discrepancy to satisfy the declared absolute or
relative tolerance. A numerically exact surrogate prediction is rejected when the
source control is physically infeasible. The surrogate may propose or accelerate;
only the source problem accepts.