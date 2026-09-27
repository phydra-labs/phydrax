# Accelerator beam dynamics

`phydrax.applications.accelerator` provides a fixed-capacity reference beamline profile and provider boundaries for broader accelerator production.

## Convention and bunch

`AcceleratorConvention` fixes the coordinate order `(x, px/p0, y, py/p0, zeta, delta)`, longitudinal sign, and momentum normalization. The reference particle is separate from every macroparticle. `AcceleratorBunch` stores stable IDs, weights, activity, reference rest energy, reference momentum, and reference charge.

## Beamline

`BeamlinePlan` lowers a fixed sequence of drifts, split-map quadrupoles, steerers, thin RF gaps, circular apertures, and sector-bend references into `lax.scan`. `track_beamline` returns every element state, activity history, first loss element, finite-step evidence, and the accepted final bunch. Capacity and aperture losses are explicit; no particle is silently discarded.

`beam_diagnostics` returns weighted six-dimensional moments, transverse geometric emittances, Twiss alpha/beta, and transmission.

## Collective physics

`SpaceChargeKickPlan` applies a kick only when an external or existing-PIC field solve carries exact source, frame transform, boundary condition, and residual evidence. It does not relabel PIC as a universal accelerator collective solver.

`LongitudinalWakePlan` and `apply_longitudinal_wake` are a prescribed, unitless research primitive, not qualified wakefield support. The plan takes strictly increasing `zeta_edges`, one caller-supplied `wake_values` entry per bin, and a scalar `kick_scale`. Application bins active, valid particle weights by the longitudinal coordinate (column 4), forms the causal discrete convolution of that binned density with `wake_values` truncated to the bin count, and adds `kick_scale` times the bin potential to column 5 of in-range particles; out-of-range particles are not kicked. No physical units, charge normalization, wake model, impedance, transverse wake, or multi-bunch/multi-turn memory is implied; the caller owns the meaning of `wake_values` and `kick_scale`.

Qualified wakefields and impedances, coherent synchrotron radiation, radiation damping and excitation, beam-beam effects, collimation scattering, high-order maps, and machine correction remain separately qualified native or external provider capabilities.

## Interchange

`accelerator_bunch_from_openpmd_columns` admits one explicit normalized openPMD column profile. `write_madx_sequence` writes only its stated drift/quadrupole/horizontal-steerer subset and refuses other elements rather than approximating them.
