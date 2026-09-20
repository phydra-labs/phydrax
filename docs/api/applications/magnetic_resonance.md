# Magnetic resonance

::: phydrax.applications.magnetic_resonance

## Exact finite-spin contract

The NMR, EPR, and static-site muSR profiles share one finite-spin engine but are
separate profile types. A spin matrix is dimensionless and obeys `[Ix, Iy] = i
Iz`. Every assembled generator is `H / ħ` in rad/s; `ħ =
1.054571817e-34 J s` is retained by `MagneticResonanceConvention` for conversion
to energy. Signed gyromagnetic ratios are used without taking an absolute value:

- Zeeman: `H_Z / ħ = -γ B · g · I`, where `g` is the site's dimensionless
  molecular-frame Zeeman tensor after orientation into the laboratory frame.
- Chemical shift: `H_CS / ħ = -γ B · (δ × 1e-6) · I`; positive isotropic `δ`
  therefore increases the positive-γ receiver frequency relative to the bare
  Zeeman line.
- Scalar J: `H_J / ħ = 2π J (I_a · I_b)` for `J` in Hz.
- Dipolar: `(μ0 / 4π) ħ γ_a γ_b / r³` multiplies
  `I_a · I_b - 3(I_a · r̂)(I_b · r̂)` with positions in meters.
- Hyperfine and quadrupolar tensors are full Cartesian coefficient tensors in
  Hz and receive one `2π` factor. Quadrupolar tensors must be symmetric,
  traceless, and target spin `I >= 1`.

`SingleCrystalOrientation` is the active ZYZ map `Rz(α) Ry(β) Rz(γ)` from the
molecular frame to the laboratory frame. Vectors transform as `R v` and
second-rank tensors as `R T Rᵀ`.

## Pulses, acquisition, and instrument

`FixedPulseSequence` contains interval-held laboratory magnetic fields in tesla.
`prepare_pulse_sequence` lowers those three field lines through the existing
`QuantumControlSchedule`; it does not introduce a second control engine. Exact
density evolution materializes the bounded total Hamiltonian and applies one
dense matrix exponential per interval. Trace, Hermiticity, positivity, and
unitarity residuals remain attached to the result.

`AcquisitionPlan` samples the complete static Hamiltonian at `t_n = n Δt`. The
receiver is the weighted lowering operator `Σ w_k (Ix_k - i Iy_k)` with an
explicit phase. `fid_spectrum` returns signed Hz and rad/s axes using
`Δt FFT(fid)` with the negative exponential followed by `fftshift`.
`InstrumentPlan` applies only an explicit positive gain, receiver phase, and
frequency-reference offset; the bare FID remains in `InstrumentResult`.

`MagneticResonanceResourcePolicy` rejects both Hilbert dimension `D` and dense
state/operator storage `D²` before preparation. Its defaults are conservative
code limits, not a released-support claim.

## Scope boundaries

The implemented profiles are exact small finite-spin, fixed-schedule,
single-crystal laboratory-frame calculations. There is no weak-coupling
renderer hidden behind the exact API. Powder averaging, magic-angle spinning,
rotating-wave approximations, relaxation, exchange, and open-system line shapes
are not part of these profiles and must be introduced as separately qualified
candidates if implemented.
