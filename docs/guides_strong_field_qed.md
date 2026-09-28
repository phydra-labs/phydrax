# Strong-field QED in PIC

Phydrax models the two first-order strong-field QED processes of charged
particles in intense fields — nonlinear Compton emission of hard photons by
leptons and nonlinear Breit–Wheeler pair creation by photons — as Monte Carlo
events inside electromagnetic PIC, in the locally constant field approximation
(LCFA) or its infrared-improved form. Cascades (pairs that emit photons that
create pairs) follow from composing both in one PIC process.

| Piece | Owner | Role |
| --- | --- | --- |
| `QEDTable` | `phydrax.discretization.pic` | host-prepared, fingerprinted rate and cumulative spectrum of one process |
| `NonlinearComptonPlan` | `phydrax.discretization.pic` | photon emission with optical-depth Monte Carlo and adaptive subcycling |
| `NonlinearBreitWheelerPlan` | `phydrax.discretization.pic` | pair creation by photons |
| `QEDPhotonSpeciesPlan` | `phydrax.discretization.pic` | photon bank with identities, escape box and escape histograms |
| `QEDCascadeProcess` | `phydrax.discretization.pic` | creation-stage PIC process coupling all of the above |
| `RelativisticPushPlan.precess` | `phydrax.discretization.pic` | Thomas–Bargmann–Michel–Telegdi spin precession consistent with the pusher |
| `tools/qed_tables.py` | repository tool | deterministic table generation, recording and bitwise verification |

## Rates and tables

For a lepton of Lorentz factor `γ` and quantum parameter
`χ = γ √((E + v×B)² − (v·E)²/c²) / E_S` (with `E_S = m²c³/(|q|ħ)`), the photon
spectrum in the energy fraction `ξ = ω/ε` is `dW/dξ = (α mc²/(ħγ)) s(χ, ξ)`
with `s = [F(δ) + ξ²/(1−ξ) G(δ)]/(√3π δ)`, `δ = 2ξ/(3χ(1−ξ))`, and `F`, `G` the
synchrotron kernels of `phydrax.special`. A photon of energy `ε_γ` with
`χ_γ = (ε_γ/mc²) √((E + c k̂×B)² − (k̂·E)²)/E_S` creates a pair whose electron
carries `ξ` at `dW/dξ = (α m²c⁴/(ħ ε_γ)) s(χ_γ, ξ)` with
`s = [G(δ)/(ξ(1−ξ)) − F(δ)]/(√3π δ)`, `δ = 2/(3χ_γ ξ(1−ξ))`; the total is
`R = χ_γ T(χ_γ)` in Erber's normalization.

`QEDTable(process, maximum_chi=...)` integrates these spectra by composite
Gauss–Legendre quadrature in a variable where every spectrum is smooth
(`log δ` for Compton with the exact `δ^{1/3}` head, a normalized `arcosh`
variable for pairs), tabulates `log K(χ)` / `log R(χ) + 8/(3χ)` with exact
forward-mode slopes for cubic Hermite interpolation, and stores normalized
cumulative rows `P(χ, ξ)` for linear inverse-CDF sampling. Construction measures
and reports the quadrature error, the rate interpolation error, the CDF
interpolation errors between rows and between nodes (their sum bounds the
Kolmogorov–Smirnov distance of sampled spectra), the smallest CDF increment
(monotone evidence), and the truncated spectral mass, and refuses tables beyond
the requested tolerances. Below the table the exact small-`χ` asymptotes
continue the rates (`K → 5χ/(2√3)`, `T → (3/16)√(3/2) e^{−8/(3χ)}`).

`QEDTable(..., polarization="positive" | "negative")` tabulates the same
process for a polarized incoming particle (below); `"averaged"` (default) is the
unpolarized table, and the two polarized tables average to it. The tool builds
and verifies all six tables.

```bash
python tools/qed_tables.py --maximum-chi 100 --output qed-tables.npz
python tools/qed_tables.py --maximum-chi 100 --check qed-tables.npz
```

## Emission models

`NonlinearComptonPlan(model, scale, charge, mass, table, ...)` accepts
`"lcfa"` or `"improved-lcfa"` (Di Piazza, Tamburini, Meuren & Keitel 2019).
The improved model computes the local field-variation time
`τ = 2√(F⊥²/(Ḟ⊥² + |F⊥·F̈⊥|))` of the transverse Lorentz force along each
trajectory (from the two previous steps kept in the process state), the photon
energy `ω_LCFA` whose formation time equals `τ`, and holds the spectrum flat
below `0.7 ω_LCFA`, which removes the unphysical `ξ^{−2/3}` infrared divergence.
Without a finite `τ` (constant fields, new particles) the models coincide.

## Polarization and spin

`QEDPolarizationModel` selects what the plans resolve; the Compton and
Breit–Wheeler plans of one cascade must agree:

| Model | Leptons | Photons | Extra tables |
| --- | --- | --- | --- |
| `"unpolarized"` | spin-averaged | polarization-summed | none (Q2 arithmetic and random numbers unchanged) |
| `"photon-polarized"` | spin-averaged | linear Stokes `(Q, U)` sampled at emission, used and evolved by pair creation | Breit–Wheeler `"positive"`/`"negative"` |
| `"spin-and-photon-polarized"` | rest-frame polarization vector `S`, precessed and radiatively evolved | as above | also Compton `"positive"`/`"negative"` |

The rates are the Seipt–King LCFA rates (Phys. Rev. A 102, 052805, 2020) in
the spin-quantization-axis scheme of Li et al. (Phys. Rev. Lett. 122, 154801,
2019), rewritten with the synchrotron kernels `F`, `G` and
`H(x) = x K_{1/3}(x)` (`phydrax.special.synchrotron_h`):

- A lepton's spin enters only through `P = S·ê` with `ê = v̂ × F̂⊥` (the
  rest-frame magnetic field direction for an electron). Its photon spectrum is
  `s − P ξH(δ)/(√3πδ)`, a mixture of the `"positive"` and `"negative"` Compton
  tables with weights `(1 ± P)/2`. At an emission the final spin `P_f = ±1` and
  the photon polarization `τ = ±1` along `e₁ = F̂⊥` are sampled jointly from the
  four channel rates `NonlinearComptonPlan.channel_spectrum`; the spin collapses
  to `P_f ê`, which includes spin flips. Without emission the spin follows the
  no-emission (one-loop mass-operator) evolution, and the optical depth
  decreases by the exact survival exponent. The ensemble then relaxes to the
  Sokolov–Ternov state, antiparallel to `ê` (electron spins antiparallel,
  positron spins parallel to a magnetic field), with equilibrium `8/(5√3)` and
  flip rate `(5√3/8) α χ³ mc²/(ħγ)` as `χ → 0`.
- Spin precession is owned by the pusher: `RelativisticPushPlan.precess`
  integrates `dS/dt = (q/m) S × [(a + 1/γ)B − aγ(β·B)β/(γ + 1) − (a + 1/(γ+1))
  β×E/c]` with the same Cayley rotation and rotation Lorentz factor as the
  Boris, Vay or Higuera–Cary push, so with `a = 0` the spin stays locked to the
  momentum and with `a = (g−2)/2` it turns relative to it at `aγω_c`. The
  cascade precesses every bound lepton species each step with the run's
  pusher (`magnetic_moment_anomaly`, default the CODATA 2022 electron value
  `ELECTRON_MAGNETIC_MOMENT_ANOMALY`).
- A photon's linear Stokes parameters are carried relative to its
  polarization axis; each Breit–Wheeler step rotates them to the local axis
  `(E + c k̂×B)⊥`, where `τ = Q` weights the `"positive"` (along the field) and
  `"negative"` tables (pair creation along the field is half that across it as
  `χ_γ → 0`); surviving photons follow the no-decay evolution (vacuum
  dichroism). Pair spins `P_± = ±1` on each lepton's own `ê` are sampled from
  the four Seipt–King channels, and pairs are created in those states.

Spins are keyed by persistent identity; fresh identities are unpolarized.
`QEDCascadeProcess.polarize(state, species, species_state, spin)` sets initial
polarization vectors. Escaped photons add their `(Q, U)` along the projection
of `polarization_reference` to `QEDPolarizationState.escaped_stokes`, so the
degree of linear polarization of a histogram bin is `|Σ w(Q, U)| / Σ w`.
Emitted photons from unpolarized leptons are polarized along the acceleration
with mean Stokes `G(δ)/(F(δ) + ξ²/(1−ξ) G(δ))`: 3/5 of the photon number and
3/4 of the power in the classical limit. Spin and polarization carry no energy
of their own, so the ledgers are unchanged.

## Monte Carlo, subcycling and randomness

Each lepton and photon carries an optical depth drawn as `−log U`. Emission
decrements it by `W h` per subcycle; the step is split into
`ceil(W Δt / maximum_event_probability)` subcycles, at most
`maximum_subcycles`, and a particle that would need more is a support failure
that rejects the PIC step (a decision that depends only on the state, never on
a random draw). Photon decay uses the exact constant-rate decrement. Every
random number is derived from `(step, id_hi, id_lo, event)` with the
particle's persistent identity (polarized models draw one more uniform per
event for the polarization or spin channel), so results do not depend on
storage slots;
allocation follows canonical (emitter, identity, subcycle) order. The random
stream is keyed by the process fingerprint, which includes the capacities.

## Conservation and ledgers

Each plan declares the quantity it conserves exactly. With
`conservation="momentum"` (default) photons and pairs are collinear with exact
momentum; the energy defect `O(m²c⁴/ε)` per event, computed without
cancellation, is energy the field supplies. With `"energy"` the energy split is
exact and the momentum defect is reported instead. In PIC,
`PICEnergyLedger` closes as

    defect = total + radiated + created_rest_energy − field_exchange − previous_total,

where `radiated` is the net energy handed to photons (emitted minus decayed,
banked or not), `created_rest_energy` the pairs' rest energy, and
`field_exchange` the collinear-kinematics defect.

## Cascades in PIC

`QEDCascadeProcess` is a `"creation"`-stage process, so one PIC step is
gather → push → (momentum processes) → Compton emission → Breit–Wheeler
pair creation → drift and current deposit → field advance → population
processes. Pairs are created at their photon's step-start position — charge
neutral there, which the runtime verifies by redeposition — and drift and
deposit current within the same step. Photons live in the process-owned bank,
move ballistically, and leave through the escape box into polar-angle ×
energy histograms; emitted photons below `minimum_photon_energy` are counted as
untracked radiation. Photon fields are gathered through the transfer route of
`gather_species`. The process claims `"subgrid-reaction"` radiation ownership,
so it replaces radiation reaction in a run.

```python
compton = pic.NonlinearComptonPlan(
    "improved-lcfa", scale, -e, m, pic.QEDTable("nonlinear-compton", maximum_chi=100.0),
    maximum_chi=100.0, minimum_gamma=1.0,
)
pairs = pic.NonlinearBreitWheelerPlan(
    scale, e, m, pic.QEDTable("nonlinear-breit-wheeler", maximum_chi=100.0),
    maximum_chi=100.0,
)
photons = pic.QEDPhotonSpeciesPlan(
    4096, 1, escape_lower=(0.0,), escape_upper=(100.0,), energy_edges=edges,
)
cascade = pic.QEDCascadeProcess(
    compton, photons, emitters=(0, 1), breit_wheeler=pairs,
    electron=0, positron=1, gather_species=0, minimum_photon_energy=2.0 * m,
)
run = phx.solver.ElectromagneticPICPlan(
    solver, species=(electrons, positrons), processes=(cascade,),
    ownership="subgrid-reaction", key=jax.random.key(0),
)
```

The process state (optical depths, trajectory histories, photon bank, escape
record, and in polarized models the spins and photon Stokes vectors) is carried in `ElectromagneticPICState.processes`, checkpointed as the
restart component `process/{index}`, and shifted by moving windows.

Allocation is atomic: when the photon bank or a pair species cannot hold every
event of a step, the step is rejected unchanged with
`PICRejectionReason.PROCESS` and `QEDCascadeEvidence.photon_capacity_refused`
or `pair_capacity_refused`. `QEDCascadeEvidence.occupancy` reports each bound
species' and the bank's fill fraction and `merge_requested` where it reaches
`merge_occupancy`; a `ParticleMergePlan(..., minimum_occupancy=...)` in the
same run merges the lepton species when that trigger is reached.

## Galilean field grids

On a solver implementing `PICGalileanGrid` with a nonzero `grid_velocity`,
positions are grid coordinates `x − v_grid t`: the runtime drifts leptons
(including created pairs) at `v − v_grid` and the cascade drifts photons at
`c k̂ − v_grid`, probes gather the lab-frame fields at the grid positions, and
every quantum parameter, formation time and ledger term uses lab-frame fields
and momenta. A Galilean run therefore reproduces the lab-grid run with
positions shifted by `v_grid t`. The escape box is declared in grid
coordinates.

## Validity evidence

Per event, `formation_ratio` is the formation time over the local
field-variation time; events above one are flagged `OUTSIDE_LCFA_VALIDITY`, and
improved-LCFA events from the flat infrared part `INFRARED_CORRECTED`. The
one-step trident and photon splitting are not modeled: leptons and photons
whose formation is not local (the regime where these channels are not bounded
by the modeled two-step description) are flagged `ONE_STEP_TRIDENT` and
`PHOTON_SPLITTING`. `χ` beyond the table and event probabilities beyond the
subcycling bound are support failures. Emission and pair creation are discrete
events and are not differentiable.

## Qualification

The candidate profile `pic.polarized-qed` gates: polarized tables averaging to
the unpolarized ones; polarized rates and channel spectra against the
Seipt–King Airy-function rates; Sokolov–Ternov spin-flip channels, flip rate
and equilibrium `8/(5√3)`; Monte Carlo relaxation to the LCFA equilibrium;
T-BMT precession at `aγω_c` for all three pushers and in PIC; emitted photon
polarization against the LCFA and the classical 3/5 and 3/4; polarized pair
rates and pair spins; Galilean-grid equivalence of QED emission; and exact
restart of a spin-polarized cascade.

The candidate profile `pic.qed-cascade` gates: Compton and Breit–Wheeler rates
against direct Bessel quadrature and Erber's asymptotics; Kolmogorov–Smirnov
tests of photon and pair spectra at fixed `χ`; the `χ → 0` mean emitted power
against the quantum-corrected Landau–Lifshitz drift; the improved-LCFA
infrared spectrum; declared-quantity conservation with the field-exchange
defect; the rotating-electric-field cascade growth rate against the model of
Grismayer et al. (Phys. Rev. E 95, 023210, 2017) within 25 %; ledger closure;
storage-slot-order invariance; atomic capacity refusal; and exact restart.

## Nonclaims

No one-step trident, photon splitting or vacuum birefringence (the real part
of the polarization operator; only its absorptive part, pair creation, evolves
photon polarization); polarized rates are diagonal in the spin quantization
axis and the linear photon basis, so circular photon polarization (helicity
transfer from longitudinal spin) and spin components transverse to `ê` in the
emission channels are not resolved; the magnetic-moment anomaly is a constant
(no field-dependent correction); no photon merging (photon-bank size is
controlled by `minimum_photon_energy` and the escape box); the LCFA remains an
approximation whose validity is reported, not enforced.
