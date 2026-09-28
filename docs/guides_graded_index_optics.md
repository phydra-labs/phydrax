# Graded-index and dispersion rays

`DispersionRayPlan(hamiltonian, step_size, step_count)` integrates the Hamiltonian ray system `dx/dτ = ∂H/∂p`, `dp/dτ = −∂H/∂x` of an arbitrary dispersion relation on its level set `H = 0`, with canonical momentum `p = c k / ω` (the refractive-index vector), geometric length, phase (optical) length `∫ p · ∂H/∂p dτ`, and the full canonical tangent map. The Hamiltonian owns its normalization: every `g(x, p) D(x, p)` with `g ≠ 0` on `D = 0` traces the same rays with a different parameter `τ`.

Two symplectic integrators are available (`DispersionRayMethod`):

- `"implicit-midpoint"` for any `AbstractDispersionHamiltonian`: each step solves `m = z + (h/2) J ∇H(m)` with `phydrax.nonlinear.VectorLocalRootPlan` and sets `z' = 2m − z`. The step map is the Cayley transform of `h J ∇²H(m)`, so the tangent map is symplectic to roundoff; a nonquadratic `H` is conserved to `O(h²)`, so declare `hamiltonian_tolerance`.
- `"kick-drift-kick"` (Störmer–Verlet) for separable `AbstractSeparableDispersionHamiltonian` values `½|p|² + V(x)`; a non-separable Hamiltonian is refused.

## Graded-index media

Graded-index rays are the isotropic specialization `RefractiveIndexHamiltonian(field)`, `H = ½(|p|² − n(x)²)`, so `p = n t` for the unit tangent `t` and `ds = n dτ`. `GradedIndexRayPlan(field, step_size, step_count)` binds it to the kick–drift–kick schedule with roundoff Hamiltonian-drift evidence (`hamiltonian_tolerance=None`, i.e. `2·10³ ε step_count`); `prepare()` returns the `PreparedDispersionRay` consumed by `CurvedSchlierenPlan`.

Refractive media are explicit `StructuredRefractiveIndexField`, `TetrahedralRefractiveIndexField`, or `AnalyticRefractiveIndexField` values. Structured interpolation is piecewise trilinear. Tetrahedral P1 gradients are cellwise constant and route derivatives are conditional on unchanged cell membership.

## Evidence

`DispersionRayEvidence` retains medium coverage, maximum Hamiltonian drift, symplectic residual `‖MᵀJM − J‖`, midpoint-root convergence and residual, the extreme refractive indices `|p|`, finite state, plan and Hamiltonian identity, and per-ray `DispersionRayStatus` bits: `LAUNCH_INVALID`, `UNSUPPORTED`, `NONFINITE`, `ROOT_UNCONVERGED`, `HAMILTONIAN_DRIFT`, `SYMPLECTIC_RESIDUAL`, `RESONANCE` (`|p|` above `resonance_index`), and the qualifying `CUTOFF` (`|p|` below `cutoff_index`, a turning point where the geometric amplitude needs an Airy connection). `RayFanPlan` lowers tangent maps to transverse determinants, singular values, and bounded caustic-crossing evidence. A caustic invalidates single-route geometric amplitude; it is not repaired by clipping.

`CurvedSchlierenPlan` projects final-minus-initial wave normal onto the detector basis and returns a typed angular measurement. It remains distinct from straight-ray and coherent-wave Schlieren.

## Magnetized cold-plasma rays

`ColdPlasmaProfile(scale, coordinates, charge_numbers=, mass_ratios=, density=, magnetic_field=, profile_id=)` describes a collisionless multi-species plasma with traceable analytic density and field profiles (coordinates in the scale's length unit). `ColdPlasmaHamiltonian(profile, angular_frequency=, mode=)` is the Stix biquadratic `D = A n⁴ − B n² + C` written in Cartesian components (`n² (S n⊥² + P n∥²) − RL n⊥² − PS (n² + n∥²) + PRL`, smooth at `n = 0` and along `B₀`), normalized by `G = −ω ∂D/∂ω|_k` so that `τ = c t` is the group light path and `|dx/dτ| = v_g/c`. The quartic contains both modes; the ray's mode is fixed at launch by the labeled root of the `ColdPlasmaDielectric` branch identification (`PlasmaWaveMode`) and preserved by continuity. Launches whose root is evanescent, resonant, degenerate (vacuum, where both modes coincide) or ambiguously labeled carry `LAUNCH_INVALID`. Rays reflect at cutoffs as turning points of the canonical flow: on a linear density ramp the ordinary mode follows the parabola `x = y cot θ₀ − y²/(4 L n₀² sin²θ₀)` and turns at `x = L n₀² cos²θ₀`.

`ColdPlasmaHamiltonian.sample_path(rays, transverse_reference)` returns a `ColdPlasmaRayPath`: per segment the chord length `ds` and its wave-normal projection `cos α ds`, the angle to `B₀`, both mode indices (ray mode first), the ray refractive index `n_r² = n² |sin θ/(sin Θ dΘ/dθ)| / cos α` (Bekefi 1966) of the ray mode, mode Stokes vectors and the Faraday rotation vector `2 Re ρ` in a parallel-transported polarization basis, C1 mode labels and status, the mode-coupling parameter (mode-axis turning rate over the beat wavenumber `k|n₀ − n₁|`), the relative index splitting, and quasi-transverse flags. `PlasmaRayTransferPlan` in `phydrax.applications.radiation_transport` transports `S/n_r²` along the path in the weak (independent modes) or strong (coupled Stokes) mode-coupling limit; see [Plasma waves and emission](guides_plasma_waves_and_emission.md#rays-and-polarized-transfer). The runnable example is `examples/plasma_ray_transfer.py`.
