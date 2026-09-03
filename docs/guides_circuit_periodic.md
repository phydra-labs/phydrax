# Periodic circuit analysis

`TemporalHarmonicPlan` represents one uniformly sampled period and evaluates state
rates by Fourier differentiation. `solve_harmonic_balance` evaluates the original
implicit circuit DAE residual at every time sample through the native nonlinear
substrate; it does not maintain a second device-equation implementation.

## Prepared harmonic balance

`plan_harmonic_balance` fixes sample count, circuit state layout, waveform dtype,
and resource policy while leaving angular frequency numeric. Its cost evidence
reports unknown count and waveform, Fourier-coefficient, and workspace bytes before
nonlinear preparation.

`prepare_harmonic_balance` binds one initial waveform, current prepared circuit DAE,
frequency, runtime arguments, nonlinear method, and termination policy.
`refresh_harmonic_balance` accepts changed circuit coefficients, frequency,
waveform values, and argument values under the exact same structure. It delegates
to the native nonlinear refresh path, preserves `prepared_id`, and increments
`numeric_version`.

`solve_prepared_harmonic_balance` retains the time waveform, Fourier coefficients,
native nonlinear result, original residual norm, scale-safe relative residual,
high-frequency tail estimate, tail qualification, resource plan, and numerical
provenance. The convenience `solve_harmonic_balance` performs plan, prepare, and
prepared solve through the same implementation.

The nonlinear path remains the existing matrix-free Newton–Krylov/JVP solve. No
dense global harmonic Jacobian, hidden regularization, or second circuit solver is
introduced.

Increasing sample count, changing circuit state layout or waveform dtype, or
changing the nonlinear method/termination structure requires replanning or
repreparation as appropriate. Frequency is refreshable because it does not change
the Fourier-collocation shape.

## Periodic energy evidence

`evaluate_harmonic_balance_energy_ledger` applies the transient energy contract to the
existing harmonic-balance waveform and its Fourier derivative. It uses the existing
DAE laws and accepts the same runtime arguments; no second periodic device model is
introduced. Source and external-port powers retain their separate signed axes.

On the endpoint-exclusive uniform grid, periodic quadrature is the sample mean times
the declared period. `CircuitPeriodicEnergyLedger` independently retains integrated
storage rate, the periodic endpoint storage change, their endpoint defect, passive
dissipated energy, external-port energy, source energy, the integral of the pointwise
balance, and the period balance defect. Closure also requires the retained nonlinear
success, residual norms, and harmonic-balance aliasing-tail evidence. Exact periodic
representation alone is not treated as energy closure, and sample-count convergence
remains a separate study.

`shoot_periodic_circuit` integrates one native DAE period and reports the exact
final-minus-initial mismatch. It is useful for validating a harmonic solution or
constructing an external shooting corrector. `floquet_multipliers` differentiates
a supplied period map and uses the native general eigensolver for periodic
stability.

Nonsmooth events, changing topology, subharmonic selection, autonomous oscillator
phase conditions, and sample-count convergence require explicit problem
formulations. They are not hidden inside the basic forced-periodic analysis.
