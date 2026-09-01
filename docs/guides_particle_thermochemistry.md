# Particle thermochemistry

Particle conversion uses explicit species, phase, element, energy, and morphology contracts. It does not infer chemistry from array positions or silently repair a nonconservative mechanism.

## Species and phases

`ChemicalSpeciesSchema` declares:

- stable species names;
- one `ChemicalPhaseKind` per species;
- molar masses and integer charge;
- stable element names;
- an element-by-species composition matrix.

Reaction and phase-change plans must carry the same schema fingerprint as their material bundle and prepared batch.

## Thermodynamic material

`ParticleThermodynamicMaterialPlan` composes a common prepared species-thermodynamic
model. Polynomial material data uses `PolynomialSpeciesThermodynamicsPlan`; NASA
thermodynamics is available to gas mechanisms. Temperature reconstruction uses a
bounded bisection solve and reports the energy residual and distance to both
temperature bounds. Reference internal energies may be negative.

`ParticleThermochemicalMaterialBundle` couples the thermodynamic and transport descriptions while retaining both fingerprints.

## Reaction networks

`ParticleReactionProcessPlan` binds a common `PreparedChemicalMechanism` to particle
reaction locations: bulk, internal surface, or outer surface. The mechanism owns
stoichiometry, thermodynamics, and rate laws; the particle adapter owns only geometric
measure. Preparation rejects elemental or charge imbalance.

```text
reaction = phx.equations.ParticleReactionProcessPlan(
    prepared_mechanism,
    locations=reaction_locations,
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
