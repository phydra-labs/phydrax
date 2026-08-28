# Transport velocity and adaptive smoothing length

`PreparedTransportVelocityDynamics` uses physical and transport velocity as
separate state fields. Position follows transport velocity. A positive constant
background pressure regularizes disordered particles, while the transport stress
accounts for momentum carried by the velocity difference. The accepted-step
refresh is implemented by `TransportVelocityFixedStepMethod`. The compiler
rejects free-surface use because background pressure is incompatible with a
physical atmospheric surface.

`AlgebraicSmoothingLengthPlan` evaluates
`hᵢ = η(mᵢ/ρᵢ)^(1/d)` under explicit bounds.
`CoupledSummationSmoothingLengthPlan` iterates the density--h relation and returns
its residual, work, and convergence status. `adaptive_smoothing_state` computes
the grad-h factor Ω from the smoothing-length derivative of the kernel.

`variable_h_pressure_gradient` uses the two directed kernel gradients and Ω
factors required by the variational fixed-mass formulation. Candidate search
must be prepared with the declared maximum support radius. Smoothing-length
bounds, convergence, and pair-support asymmetry are observable branch decisions.
