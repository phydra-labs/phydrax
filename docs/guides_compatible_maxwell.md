# Compatible time-domain Maxwell

`CompatibleMaxwellPlan` evolves conservative electric displacement, magnetic flux, and
charge cochains on one exact complex. Electric and magnetic fields are constitutive
outputs; material, CPML, source, boundary, and observer state remain auxiliary.

## Cochain roles

The plan owns one explicit role layout.

- Full three-dimensional Maxwell stores `D`, `E`, and electric current on degree one,
  `B` and `H` on degree two, and charge on degree zero.
- TEz on a genuine two-dimensional bridge retains in-plane electric degree-one fields,
  an out-of-plane magnetic degree-two field, and degree-zero charge. Magnetic divergence
  is absent because there is no following degree.
- TMz retains out-of-plane electric degree-zero fields and in-plane magnetic degree-one
  fields. It has no retained charge degree; magnetic closedness is nontrivial.

A one-cell-thick three-dimensional grid is neither the implementation nor the
qualification oracle for a reduced model. Material tensors that couple retained and
suppressed components fail during preparation.

## Magnetic closedness

Pure Faraday forcing preserves closed magnetic flux because the next exterior
derivative composed with the electric derivative is zero. Other actions—magnetic
conductivity, PMC masking, CPML memory, or magnetic source forcing—require their own
construction evidence.

The automatic constraint policy elides projection only when the initial state is closed
and every active action carries that evidence. Otherwise Phydrax computes the same
Euclidean minimum-norm correction through a resource-bounded sparse native solve,
restores the declared harmonic periods, and reports original residual and solver work.
No production path materializes the incidence matrix densely.

## Sources and ports

Sources are prepared plans. Static support and spatial profiles are lowered once;
waveform amplitudes and controls remain dynamic. Electric current enters the full D
step at the midpoint. Magnetic current enters the two B half steps at their respective
times. Charge uses the complete electric forcing, so the same discrete continuity law
is audited.

A one-way mode or Huygens launch uses paired electric and magnetic trace forcing from
one oriented surface. Production mode ports initially require propagating, lossless,
nondegenerate modes with finite nonzero signed surface power. The same mode basis and
reference-plane identity drive launch, DFT observation, decomposition, power, and the
circuit scattering adapter.

## CPML

CPML retains one memory for each active directional derivative and cochain support.
Only exact lower/upper boundary slabs are stored; corner degrees of freedom intentionally
appear in several directional terms. The ordinary curl is evaluated on the logical
interior, and packed corrections are scattered back in a deterministic order.

Fixed runs prepare distinct recurrence coefficients for electric full steps and magnetic
half steps. Changing the time step requires explicit refresh or preparation; stale
coefficients cannot be reused. Variable public stepping computes the same recurrence on
packed terms.

## Execution and resources

The public state always uses logical one-dimensional cochains. Orientation tensors,
padding, case axes, and shards are private execution layouts. The resource policy counts
primary and auxiliary state, projection workspace, observers, checkpoints, padding,
case axes, per-device state, and requested acquisition before allocation.

Potentially promoted primary, material, observer, CPML, and projection arrays reserve
complex-128 width even when their zero state is initially real.


`solve_compatible_maxwell` scans the same private step core used by
`PreparedCompatibleMaxwell.leapfrog_step`, returns the final state and streaming
observations, and does not implicitly retain a trajectory. Numeric refresh is allowed
only when topology, role layout, prepared array/state shapes and dtypes, static
execution semantics, source callable identity, PML term layout, dtype, and backend
signature remain unchanged.

## Harmonic defects

The semi-discrete frequency residual evaluates the actual prepared cochain operator and
reports degree-paired absolute and relative norms. A fixed-step harmonic defect is
stronger: it compares the complete affine one-step state map with multiplication by
`exp(-i ω dt)`, including physical and eligible linear auxiliary state and the exact
source phases. Nonlinear, time-varying, or otherwise ineligible systems fail closed
rather than receiving a misleading frequency residual.

## Evidence boundary

Scientific qualification includes chain identities, Gauss continuity, magnetic
closedness, harmonic periods, energy/power balance, TEz/TMz analytic and invariant-3-D
convergence, CPML reflection over frequency/angle/polarization/corners, paired-source
directionality and power, and directional derivatives for every advertised control.
Finite output alone is characterization, not validation.
