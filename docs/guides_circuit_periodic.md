# Periodic circuit analysis

`TemporalHarmonicPlan` represents one uniformly sampled period and evaluates state rates by Fourier differentiation. `solve_harmonic_balance` solves the original implicit circuit residual at every time sample through the native nonlinear substrate; it does not maintain a second device equation implementation.

The result retains the time waveform, Fourier coefficients, nonlinear solve evidence, original residual norm, scale-safe relative residual, high-frequency tail estimate, and finiteness. Increasing the sample count is a replan boundary and must be used to establish truncation convergence.

`shoot_periodic_circuit` integrates one native DAE period and reports the exact final-minus-initial mismatch. It is useful for validating a harmonic solution or constructing an external shooting corrector. `floquet_multipliers` differentiates a supplied period map and uses the native general eigensolver for periodic stability.

Nonsmooth events, changing topology, subharmonic selection, and autonomous oscillator phase conditions require explicit problem formulations. They are not hidden inside the basic forced-periodic analysis.
