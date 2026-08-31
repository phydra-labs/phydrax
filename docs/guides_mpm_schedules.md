# MPM schedules and adaptive time

`ExplicitMPMMethodPlan` owns one static schedule:

```python
method = phx.discretization.ExplicitMPMMethodPlan(
    schedule=phx.discretization.USFMPMSchedule(),
)
```

## USL-minus

`USLMPMSchedule` is the original Phydrax ordering. Accepted stress supplies the
first grid force, then constrained grid velocity supplies particle velocity,
APIC affine state, deformation, and the final material update. It performs no
second momentum extrapolation.

## USF

`USFMPMSchedule` gathers the pre-force grid velocity gradient, advances the
material trial first, and assembles force from the trial first-Piola stress.
Particle velocity and position still use the force-updated constrained grid.
Material failure rejects before committing any state.

## MUSL

`MUSLMPMSchedule` is classical pre-advection translational MUSL:

1. execute the stress-last grid update;
2. gather updated particle velocity and APIC affine state;
3. re-extrapolate mass and translational particle momentum on the original routes;
4. reapply rigid contact and prescribed velocity;
5. use the remapped grid velocity gradient for deformation/material update;
6. advance position with the first gathered velocity.

The second transfer deliberately omits APIC affine momentum. An affine or
post-advection retransfer would be a different method and is not hidden behind
this name.

Every result carries `MPMScheduleEvidence`, including phase code, second-transfer
mass/momentum defects, second constraint work, and a phase digest.

## Adaptive realization

`AdaptiveMPMRolloutPlan` owns retries and produces a fixed-capacity
`RealizedTemporalMesh` plus `MPMAdaptiveAttemptJournal`:

```python
adaptive = phx.solver.AdaptiveMPMRolloutPlan(
    compiled.dynamics,
    phx.solver.MPMAdaptivePolicy(
        maximum_steps=256,
        maximum_retries=6,
    ),
    final_time=1.0,
    initial_step_size=1e-2,
)
realized = adaptive.rollout(initial, arguments)
```

Restrictions include acoustic, advective, force, contact, material, source-domain,
and nonlinear limits. Rejected attempts preserve all accepted particle, material,
field, assignment, block, topology, and time state.

The adaptive controller is stopped with respect to differentiation. Differentiate
the accepted schedule through:

```python
scheduled = phx.solver.ScheduledMPMRolloutPlan.from_realized(
    compiled.dynamics,
    realized.realized_mesh,
    replay=phx.solver.MPMReplayPolicy("block", block_size=32),
)
```

That derivative is conditional on the realized temporal and branch program; it
is not a derivative of controller decisions.
