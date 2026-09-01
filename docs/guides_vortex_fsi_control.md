# Vortex FSI, actuators, and control

## Lifting and wakes

`MultiLiftingSurfacePlan` compiles multiple framed surfaces while retaining
component, body, local panel, flap, and trailing-edge ownership. Horseshoe, ring,
and lifting-line solves use explicit Kutta/Kelvin evidence. Load providers remain
separate: Kutta–Joukowski, unsteady Bernoulli, impulse, moments, added mass, and
Trefftz induced drag.

Shared wake topology uses oriented vertices, edges, rings, and sheet incidence.
Midpoint and RK3 wake steps do not shed during stages. Core diffusion, age,
curvature refinement, downstream coarsening, and reconnection are declared
candidate events.

`MultiAxisAirfoilPolar` resolves angle, Reynolds number, Mach number, and flap
state with explicit endpoint policy. Dynamic stall and low-Mach corrections are
named opt-in models. `BladeElementRotorPlan` and `ActuatorLineFlowPlan` return
section forces, circulation, shed sources, thrust, torque, power, and balance
evidence.

## Native FSI

`VortexRigidCouplingPlan` advances `PreparedRigidBodySet` and
`RigidBodyKinematics` through native kick-drift-kick dynamics. Prescribed, loose,
Aitken, and strong modes report load, velocity, work, and convergence residuals;
failed windows restore prior fluid/body/load state.
`VortexFlexibleCouplingPlan` advances a native `SecondOrderDifferentialSystem`
through generalized-alpha integration.

## Control and acoustics

`VortexTrajectoryControlPlan` provides direct or multiple shooting and direct
collocation objectives through the native optimization API. Event signatures
must remain fixed when differentiated. `VortexMPCPlan` applies a selected prefix
of each optimized horizon.

`AerodynamicLoadHistory` is the portable aerodynamic/acoustic boundary. FW-H
tonal and broadband section models consume accepted histories as postprocessing;
they do not alter aerodynamic steps.
