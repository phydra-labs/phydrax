# Cavity quantum electrodynamics

::: phydrax.applications.cavity_quantum

Classical Maxwell modes lower to energy-normalized finite quantum-mode,
participation, coupling, Purcell, Maxwell-Bloch, and Maxwell-Lindblad
contracts. Native adaptive H(curl) uses the general-order trimmed tetrahedral
form complex. Nested refinements use covariant form-moment transfer; non-nested
adaptations use covariant-Piola L2 projection on a certified common refinement,
in one `CompositionRebind` (`PreparedAdaptiveHcurlCapability.adapt`).
Unsupported meshes or resources, a refused common refinement, and failed
transfer evidence remain fail-closed and preserve the accepted epoch.
