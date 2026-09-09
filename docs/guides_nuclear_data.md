# Nuclear data and reactions

`phydrax.nuclear` separates physical identity from evaluated numerical data. A `NuclideKey` identifies proton number, mass number, and isomer state; an update to a mass or decay library therefore changes a data-table identity without changing the nuclide identity.

## Governed data

`NuclearDataProvenance` binds a `ReferenceArtifactManifest` to the authority, library release, evaluation, processing tool, processing release, parameters, and parent data. Import admission is not accuracy evidence. Missing uncertainty remains unquantified rather than exact.

Raw ENDF, GNDS, ACE, EXFOR, ENSDF, and RIPL structures are outside compiled execution. Convert them host-side into one finite, source-bound table before entering JAX.

## Energy groups

`EnergyGroupStructure` stores strictly increasing boundaries in joules. External fast-to-thermal layouts must be reversed explicitly by their adapter. The final interval includes its upper boundary; other intervals are left-closed and right-open.

A group scalar flux is group-integrated with unit `1/(m² s)`. Multiplication by a group-collapsed microscopic cross section in `m²` produces a per-target rate in `s⁻¹`. Group-average and group-integrated quantities are incompatible even when their arrays have equal shapes.

## Composition and material state

`NuclideComposition` declares atom-fraction or mass-fraction basis. Values are never normalized silently. Basis conversion requires an evaluated mass table and returns a closure residual. `NuclearMaterialState` adds density, temperature, and homogenization identity without inventing phase or self-shielding assumptions.

## Reactions

`NuclearReactionChannel` enumerates reactants and products and checks baryon, charge, and lepton ledgers. `ThermalFusionReactionPlan` supports one two-reactant, two-product Maxwellian branch using a source-bound tabulated reactivity. It does not extrapolate beyond the admitted temperature interval.

`ActivationNetworkPlan` freezes its nuclide and transition graph before execution. Flux-driven and decay transitions are distinct. Piecewise-constant irradiation uses native sparse matrix-exponential and phi-function actions. Negative inventories are not clipped.

## Scope

The initial nuclear substrate does not provide a bundled evaluated-data library, ENDF processor, continuous-energy transport, Monte Carlo transport, burnup management, dose conversion, or regulatory qualification. All capability profiles remain unreleased candidates until their declared gates have admitted evidence.
