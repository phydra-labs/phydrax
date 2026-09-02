# Homogeneous thermodynamics

Phydrax represents a homogeneous equilibrium phase through a complete molar Helmholtz model. The ideal reference contribution and residual interaction contribution share one chemical component catalog, phase-specific species schema, reference state, and content-sensitive model identity.

## Component and species identity

`ChemicalComponentCatalog` owns chemical identity, molar masses, elemental composition, charge, and provenance. `ChemicalSpeciesSchema` maps phase-specific species occurrences onto those components. This permits several phase occurrences of one component without duplicating its conserved identity.

Use `ChemicalSpeciesSchema.from_unique_species` when every species is a distinct component. Use the primary constructor when a component occurs in several phase instances.

Gas phases carry an explicit standard pressure. The same value defines species standard Gibbs energies, thermodynamic reverse rates, ideal mixing, and chemical equilibrium.

## Helmholtz composition

`IdealGasReferenceHelmholtzTerm` combines standard-state species thermodynamics with ideal mixing. A residual term supplies nonideal interactions. `HomogeneousHelmholtzPlan` derives pressure, entropy, internal energy, enthalpy, Gibbs energy, heat capacities, response coefficients, and frozen-composition sound speed from their sum.

The valid state uses positive temperature and molar density, nonnegative mole fractions summing to one, positive constant-volume heat capacity, positive mechanical derivative, and positive sound-speed squared. Invalid runtime states return unsuccessful evidence rather than being accepted after clipping.

`evaluate` returns caloric and mechanical properties. `evaluate_chemical` additionally returns chemical potentials and fugacity coefficients. `solve_density_energy` recovers temperature from component mass densities and full internal-energy density with a bracketed fixed-shape solve.

## Differentiation boundaries

A homogeneous state is differentiable only inside one fixed phase and one regular state-solver branch. Phase appearance, critical coalescence, spinodals, support changes, and active-component changes require explicit unsuccessful derivative evidence in the corresponding equilibrium solver.

## Performance

Species and component counts are static. Leading batch dimensions may vary without changing semantics. Pair contractions use native array operations, and composition Hessians are reserved for explicit stability calculations. External property engines are host-side qualification oracles only; they are never called from compiled kernels.
