# Convex entropy pairs

A `ConvexEntropyPair` binds one conservation system to a mathematical entropy
`η(u)`, entropy variables `v(u)`, directional entropy fluxes `qᵃ(u)`, and an explicit
admissible-state predicate.

For an admissible state `u`,

`v(u) = ∇η(u)`.

The relative entropy is the ordered Bregman divergence

`η(u ∥ ū) = η(u) - η(ū) - ⟨v(ū), u - ū⟩`.

It is not a geodesic distance and is not generally symmetric. Invalid states are not
clipped, floored, or jittered.

## Euler entropy pair

```python
import phydrax as phx

system = phx.equations.EulerSystem(2)
pair = phx.equations.ideal_gas_euler_entropy_pair(system)
```

The ideal-gas pair uses the normalization already implemented by
`EulerSystem.entropy_variables`:

```text
s = log(p) - γ log(ρ)
η(u) = -ρs / (γ - 1)
```

The directional entropy flux is

`qᵃ(u) = velocity_a η(u)`.

The entropy potential is

`Ψᵃ(u) = ⟨v(u), fᵃ(u)⟩ - qᵃ(u)`.

User-defined entropy-flux callables receive `(state, axis, args)`. The same runtime
`args` are forwarded to physical fluxes during entropy-potential evaluation and pair
validation.
For this normalization, `Ψᵃ(u)` equals the directional momentum. The pair is a
convective Euler entropy pair when attached to `CompressibleNavierStokesSystem`; it
does not certify viscous entropy production.

## Validation

`validate_convex_entropy_pair(...)` checks representative admissible states for:

- explicit entropy variables versus autodiff `∇η`;
- entropy-flux compatibility `∇qᵃ = v · ∂fᵃ/∂u`;
- symmetry of `H(u) ∂fᵃ/∂u`, where `H = ∇²η`;
- finite and nonnegative relative entropy;
- zero diagonal relative entropy;
- Hessian positive definiteness and conditioning;
- precision evidence.

A successful report is local evidence at the supplied states. It is not a global proof
of convexity or entropy stability.

## Interface residuals

For a numerical flux `f*`, the Tadmor interface residual is

`Rᵃ = ⟨v_R - v_L, f*⟩ - (Ψ_Rᵃ - Ψ_Lᵃ)`.

Interpretation:

- `Rᵃ = 0`: entropy-conservative interface;
- `Rᵃ ≤ 0`: entropy-stable dissipative interface.

`EntropyConservativeEulerFluxPlan` and `EntropyStableEulerFluxPlan` retain their
existing flux construction. `ConvexEntropyPair.interface_entropy_residual(...)`
provides the explicit diagnostic using the declared entropy pair and physical axis.

## Finite-volume diagnostics

Pass an entropy pair through the standard compiler:

```python
compiled = phx.equations.compile_conservation_problem(
    problem,
    discretization,
    method,
    entropy_pair=pair,
)
```


Compiler-attached diagnostics currently support structured and mapped structured
finite-volume geometry. Supplying `entropy_pair` with triangle or modern unstructured
geometry fails explicitly; ALE/cut/overset entropy accounting requires stage content
rates and geometric volume terms that are outside this contract. Standalone entropy,
relative-entropy, and interface-residual operations remain geometry-independent.

This is a clean replacement for the former
`FiniteVolumeMethodPlan(..., entropy_diagnostics=True)` and
`diagnostics.entropy_dissipation` surface. Entropy selection now occurs through the
system-bound pair, and runtime values live under `diagnostics.entropy`.

`CompiledConservationProblem.residual_with_diagnostics(...)` then returns nested
`FiniteVolumeEntropyDiagnostics` containing:

- volume-weighted total entropy;
- volume-weighted semidiscrete entropy rate;
- source entropy rate;
- convective entropy rate;
- admissibility;
- finite-volume precision evidence.

The pair is rejected when a viscous flux plan is present. The current diagnostic
surface does not split viscous entropy production, so it refuses to label a
convective remainder that still contains viscous terms.

For bounded domains, the convective rate includes boundary transport unless an
explicit numerical entropy flux balance is supplied. It must not be interpreted as a
closed entropy-production certificate. Periodic, source-free entropy-stable cases can
be expected to have nonpositive semidiscrete entropy rates.

`integrated_finite_volume_relative_entropy(...)` computes the physical cell-volume
weighted relative entropy against a supplied reference state.

## Precision and domain policy

Finite-volume entropy calculations use the prepared finite-volume precision policy.
The pair retains its own state-domain semantics; the finite-volume compiler does not
introduce a second density, pressure, or energy floor. A failed admissibility check
remains visible and no repair is performed.

::: phydrax.equations.ConvexEntropyPair

---

::: phydrax.equations.ConvexEntropyValidationReport

---

::: phydrax.equations.ideal_gas_euler_entropy_pair

---

::: phydrax.equations.validate_convex_entropy_pair

---

::: phydrax.discretization.FiniteVolumeEntropyDiagnostics

---

::: phydrax.discretization.integrated_finite_volume_relative_entropy
