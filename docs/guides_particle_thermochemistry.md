# Particle thermochemistry

Particle conversion uses explicit species, phase, element, energy, and morphology contracts. It does not infer chemistry from array positions or silently repair a nonconservative mechanism.

## Species and phases

`ParticleSpeciesSchema` declares:

- stable species names;
- one `ParticlePhase` per species;
- molar masses;
- stable element names;
- an element-by-species composition matrix.

Reaction and phase-change plans must carry the same schema fingerprint as their material bundle and prepared batch.

## Thermodynamic material

`ParticleThermodynamicMaterialPlan` accepts polynomial molar heat-capacity coefficients and reference molar internal energies. It integrates the polynomial analytically. Temperature reconstruction uses a bounded, fixed-iteration Newton solve and reports the energy inversion residual and distance to both temperature bounds.

Reference internal energies may be negative. Only finite extensive energy is required; admissibility comes from successful temperature inversion, positive mixture heat capacity, and the declared temperature bounds.

`ParticleThermochemicalMaterialBundle` couples the thermodynamic and transport descriptions while retaining both fingerprints.

## Reaction networks

`ParticleReactionNetworkPlan` uses nonnegative reactant and product stoichiometric matrices, Arrhenius prefactors, and activation energies. Construction rejects any reaction whose stoichiometric change violates the declared element matrix.

At runtime the network returns extent rate, species amount rate, reaction energy rate, element residual, exhaustion margin, and an explicit reactant-depletion restriction. Reaction heat follows the same reference-energy convention as the thermodynamic material.

```text
reaction = phx.equations.ParticleReactionNetworkPlan(
    schema,
    reactant_stoichiometry,
    product_stoichiometry,
    pre_exponential_factors,
    activation_energies,
)
```

## Evaporation

`EvaporationPhaseChangePlan` transfers amount between one liquid species and one gas species. Both species must have equal molar mass and identical element composition. `AntoineSaturationPressurePlan` supplies the temperature-dependent saturation pressure. Latent heat is an explicit internal-energy sink.

The phase evaluation reports saturation and remaining-liquid margins. With condensation disabled, negative evaporation driving force is clipped to zero. Exhaustion remains an event surface rather than a negative-inventory correction.

## Shrinking core

`ShrinkingCoreConversionPlan` evaluates film, ash-layer diffusion, and surface-reaction resistances in series. `ShrinkingCoreState.normalized_core_radius` remains in the closed interval from zero to one. The result exposes conversion, gas and solid consumption, front rate, individual resistances, and an exhaustion step restriction.

The shrinking-core object is intentionally local. A model author decides how its gas and solid rates map into a typed species schema and how its front coordinate is stored in `ParticleInternalBatchState.reaction_front`.

## Morphology

`DensityPorosityMorphologyPlan` derives particle mass, radius, and inertia from conserved species inventory, molar volume, and porosity. The resulting `ParticleDynamicBodyProperties` can update a prepared spherical DEM runtime without recompiling static topology. Radius changes are checked against the declared neighborhood skin.

`ThermochemicalFragmentationPlan` and deactivation plans operate on preallocated slots. Event commits check mass, momentum, angular momentum, species, and energy residuals before changing activity.

## Sensitivity

`ParticleConversionSensitivityPolicy` turns species, porosity, scale, temperature, phase, and reaction distances into a `ParticleConversionValidityCertificate`. Sharp JVP/VJP results are usable only away from branch changes. Exhaustion and other transverse events use the generic `HybridEventPlan`; grazing and simultaneous events fail rather than returning an unqualified derivative.

Run `examples/particle_radial_drying.py` for a phase-change workflow.
