# Free-electron lasers

`phydrax.applications.accelerator.fel` integrates the time-independent,
period-averaged (Kroll–Morton–Rosenbluth) free-electron laser. Every beam
slice is one radiation wavelength long and evolves independently of the
others; there is no slippage, so the model describes seeded amplifiers,
steady-state gain, harmonic generation, taper, and saturation of a
monochromatic field at the slice frequency. Time-dependent SASE, seeding
schemes with slippage, and full-wave FEL simulation are not part of this
owner.

## Model

With phasors `E_x = Re[Ẽ_h e^{i(hkz − hωt)}]` (the repository's `exp(−iωt)`
convention) and ponderomotive phase `θ = (k + k_u) z − ωt`, each macroparticle
`j` and harmonic `h` obey

- `dθ_j/dz = k_u − k (1 + a_w² + u_x² + u_y²) / (2γ_j²)`,
- `dγ_j/dz = −(e/mₑc²) Σ_h κ_h Re[Ẽ_h(x_j) e^{ihθ_j}] / γ_j`,
- `∂_z Ẽ_h = (i/2hk) ∇⊥² Ẽ_h + (κ_h I / (ε₀ c N ΔA)) Σ_j W_j e^{−ihθ_j} / γ_j`,

with coupling `κ_h = a_w [JJ]_h / √2`, `u = γ(x′, y′)`, and multilinear
deposit weights `W_j`. A planar device has `a_w = K/√2` and
`[JJ]_h = (−1)^((h−1)/2)[J_((h−1)/2)(hξ) − J_((h+1)/2)(hξ)]`,
`ξ = K²/(4 + 2K²)`; a helical device has `a_w = K` and couples only its
fundamental. Even planar harmonics and helical harmonics above one have no
on-axis coupling in this model and are refused.

The Pierce parameter of these equations is
`ρ³ = e²κ²n_e / (8ε₀mₑc²γ³k_u²)` with `n_e = I/(ec·2πσ_xσ_y)`; the cold
one-dimensional field grows as `exp(√3ρk_u z)` (power gain length
`λ_u/(4π√3ρ)`). `fel_scaling_estimate` also evaluates Ming Xie's fitted 3-D
gain length and saturation estimate for every slice.

## Lattice

A lattice is a sequence of `FELUndulatorSegment`s bound to one
`ElectromagneticScaleContract`. Each segment wraps an X1a
`InsertionDeviceField`; its peak field sets `K`, so a stepwise taper is a
sequence of segments with decreasing peak fields. The break after a segment
can carry a thin quadrupole (`quadrupole_integrated_gradient`) and a phase
shifter at its entrance. Focusing inside a module is the device's natural
focusing (`k² = a_w²k_u²/γ²` in `y` for planar, `a_w²k_u²/(2γ²)` per plane for
helical) plus an optional symmetric FODO-averaged `smooth_focusing_gradient`.
The averaged model uses each module's flat-top length `N λ_u`.

```python
import math

import jax
import phydrax as phx
from phydrax.applications.accelerator import InsertionDeviceField, fel

scale = phx.ElectromagneticScaleContract.si()
period, gamma, deflection = 0.03, 1.0e4, 3.5
peak = deflection * 2 * math.pi * float(scale.electron_mass) * float(
    scale.speed_of_light
) / (float(scale.elementary_charge) * period)
device = InsertionDeviceField(
    peak, period, 1200, polarization="planar", center=0.0, aperture=(1e-2, 1e-2)
)
lattice = fel.FELUndulatorLattice(
    scale, (fel.FELUndulatorSegment(device),), step_length=0.05
)
wavelength = lattice.resonant_wavelength(gamma)
```

## Beam slices and loading

`FELBeamSlices` declares per-slice position, current, energy, relative energy
spread, and Gaussian transverse phase space (normalized emittance, Twiss `β`,
`α`) together with a `uint32` identity per slice. `FELLoading` represents a
slice by `B` beamlets of `M` particles at equally spaced phases (quiet
start); `M ≥ 2 max(h)` is enforced. `shot_noise="fawley"` adds Fawley's
phase perturbations so that each slice carries the Poisson bunching
`⟨|b_h|²⟩ = 1/N_λ` of its `N_λ = Iλ/(ec)` electrons. All random draws use
`derive_key(key, address, 0, slice_identity, beamlet, event)`, so a solve is
invariant to slice order and `slice_batch`.

## Solving

```python
slices = fel.FELBeamSlices(
    [0.0], [3000.0], [gamma], [1e-4], [[1e-6, 1e-6]], [[10.0, 10.0]], [[0.0, 0.0]]
)
plan = fel.FELPlan(
    lattice,
    wavelength,
    loading=fel.FELLoading(256, 6, shot_noise="quiet"),
    harmonics=(1, 3),
    seed=fel.FELSeed(1e3, waist=30e-6),
)
result = plan.solve(slices, jax.random.key(0))
```

`transverse="one-dimensional"` couples one field value per harmonic over the
slice area `2πσ_xσ_y`. `transverse="angular-spectrum"` evolves the field on a
uniform `PlaneFieldSpace` and diffracts it with the optics
`AngularSpectrumPlan` (exact Helmholtz propagator with the carrier removed;
finite windows need padding and report cropped power). Each step is the
symmetric Strang sequence `T(Δz/2) D(Δz/2) S(Δz) D(Δz/2) T(Δz/2)`: exact
betatron rotation, diffraction, and a fourth-order Runge–Kutta source kick of
the coupled particle/field system; the deposit is the exact transpose of the
gather.

`FELResult` reports `power[s, z, h]`, complex bunching, beam power, mean
energy and energy spread at every step; the exit field; the harmonic
frequencies `hω` and, on the grid route, the exit far field `dP/dΩ`; initial
and exit particles; `FELGainEvidence` (fitted power gain length over a
declared window of the peak power, saturation power and position, and the
analytic scaling); and the `FELEnergyLedger`
`beam_power_change + field_power_change + diffraction_loss + wake_loss`,
whose relative defect is checked against `ledger_tolerance`. `FELStatus`
flags nonfinite output, diffraction leakage, particles leaving the field
window, ledger defects, unresolved detuning phase steps, and nonlinear
shot-noise loading.

## Collective effects

`FELWakeLoss` applies an X2 longitudinal `WakeFunctionPlan` per slice: the
causal wake potential of the slice charges `q_s = I_s Δζ/c` (half self term)
per declared structure length becomes a constant energy-loss rate along the
lattice, and the removed power appears as `wake_loss` in the ledger. Slices
must be uniformly spaced. The X3 free-space space-charge solver acts on the
bunch scale and has no counterpart inside a one-wavelength periodic slice, so
it is not applied by this owner.

## Validity and non-claims

- Slowly varying envelope, period averaging, and `γ ≫ 1`; steps must resolve
  the detuning phase (`maximum_phase_step`).
- Time-independent slices: no slippage, no SASE spectrum, no
  slice-to-slice coupling other than the declared wake potential.
- Stepwise taper only; module terminations are not resolved.
- Ming Xie's formula is a fit to the exact 3-D eigenmode growth, stated to be
  accurate within about ten percent for a round, matched beam in smooth
  focusing; it is reported as evidence, not used by the solver.

Runnable example: `examples/free_electron_laser_averaged.py`. Benchmark:
`benchmarks/fel_averaged.py`.
