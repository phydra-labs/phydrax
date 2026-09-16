# Polymer construction and reaction epochs

`phydrax.applications.polymer_construction` is a host-side discrete workflow. It is intentionally separate from field-theory contour graphs, PRISM site mixtures, and realized `MolecularTopologyPlan` connectivity.

## Material recipes

`PolymerMaterialRecipePlan` declares bead types, masses, charges, explicit chain sequences, ring flags, connection ports, units, cell, stable-ID origin, and fixed particle capacity. `lower_polymer_recipe` deterministically emits

- `AtomisticSystemPlan`;
- `MolecularTopologyPlan`;
- `PolymerChainLayoutPlan`;
- realized stable-ID connection ports;
- target-versus-realized ensemble statistics;
- a content-addressed lowering record.

Coordinates never author bonds, types, charges, tacticity, or ports. Inactive capacity is retained explicitly.

Mapping and admitted-JSON adapters return `AdapterReport` evidence and preserve source identity. External bytes must pass `ExternalArtifactPolicy` before parsing. PACKMOL-style coordinates, chemistry typing, or external MD files remain separate adapters and may not silently synthesize required semantics.

## Reaction templates

`PolymerReactionTemplate` matches explicit compatibility classes and declares either a cure or repair rewrite. `apply_polymer_reaction` consumes ports and adds one typed bond for cure; repair requires that exact bond and restores the corresponding port capacity while removing it. Every attempt emits an immutable accepted or refused ledger event, and either commits the complete replacement state or retains the prior state.

Initial nonperiodic reactions preserve particles, masses, types, charges, and units. Cure and repair remain construction protocols, not reaction kinetics.

Periodic reactions require an explicit lattice-image shift. The selected molecule is reimaged atomically before the new bond is evaluated. Missing or dimensionally invalid winding data is refused; it is never inferred from wrapped coordinates. Network observables use edge winding consistency to detect periodic spanning cycles.

## Network evidence

`polymer_network_observables` reports conversion with its explicit initial-port denominator, available ports, connected components, largest-component fraction, cycle rank, accepted/refused events, and periodic spanning. It does not infer gelation kinetics or bulk material properties.
