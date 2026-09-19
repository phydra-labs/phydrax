# Omniphysics closure foundations

PhydraX classifies every mainstream capability through an orthogonal closure matrix spanning physical fields, carrier representations, coupling locations, execution regimes, topology regimes, and workflow classes. A family is not closed while a required cell remains unclassified. The generated [closure matrix](api/capability_closure.md) is an inventory rather than a release index.

## Source absorption

`SourceReference` records the exact external source, revision, licence class, studied concepts, code-inspection status, permitted reuse, provider boundary, notices, data rights, and reviewer. Strong-copyleft and proprietary sources cannot authorize copied core implementation. The checked source ledger is generated at `docs/data/source_absorption.json`.

## Shared owners

- `phydrax.materials`: material identity, phase state, process history, homogenization, and bounded transformations.
- `phydrax.manufacturing`: machine paths, schedules, moving sources, material activation, and process history.
- `phydrax.frequency`: frequency/phasor conventions, ports, and scattering evidence.
- `phydrax.population_balance`: sectional growth, aggregation, breakage, moments, and realizability.
- `phydrax.system_modeling`: across/through connectors and acausal connection sets.
- `phydrax.rheology`: generalized-Newtonian, viscoelastic, thixotropic, and conformation models.
- `phydrax.interfacial_transport`: surfactant, adsorption, dynamic wetting, and thin-film pressure.
- `phydrax.structural_dynamics`: harmonic response, Craig–Bampton constraint modes, and SEA controls.
- `phydrax.correlation`: FRF, modal assurance, and force reconstruction.
- `phydrax.electrohydrodynamics`: Maxwell stress, leaky-dielectric surface charge, and interface traction.
- `phydrax.phoresis`: electric, dielectric, thermal, diffusive, acoustic, and magnetic particle forcing.
- `phydrax.smart_materials`: piezoelectric, dielectric-elastomer, magnetostrictive, and thermoelectric couplings.
- `phydrax.chemo_mechanics`: chemical strain, stress-coupled chemical potential, degradation, and growth.
- `phydrax.tribology`: Reynolds lubrication, Hertz contact, and Archard wear.
- `phydrax.thermal_systems`: enclosure radiation, Stefan phase change, and heat-pipe limits.
- `phydrax.membranes`: solution diffusion, reverse osmosis, and electrodialysis transport.
- `phydrax.surface_chemistry`: Langmuir kinetics and catalyst-pellet effectiveness.
- `phydrax.optomechanics`: Maxwell optical stress, photoelasticity, and thermo-optic transfer.

## Flagship candidate workflows

- `phydrax.applications.additive_manufacturing`: moving-source DED with material activation, thermal ledgers, and constrained thermal stress.
- `phydrax.applications.electroviscoelastic`: phase-dependent polymer stress, Maxwell stress, electric traction jump, and surface charge.
- `phydrax.acoustics`: monopole pressure, impedance transmission, and vibroacoustic power.
- `phydrax.electrochemistry`: Nernst equilibrium, Butler–Volmer kinetics, Nernst–Planck flux, and porous properties.
- `phydrax.process_systems`: material streams, isothermal flash, and recycle convergence.
- `phydrax.applications.industrial_processes`: welding heat sources, casting solid fraction, forming stress, composite cure, machining power, and oxidation.
- `phydrax.applications.engineering_systems`: bounded controls for durability, rotordynamics, turbomachinery, engines, icing, reservoir, hydrology, marine, wind, biomedical, aerospace, autonomy, and named research candidates.

All new profiles remain unreleased candidates. Their bounded equations and controls do not certify parts, materials, safety, clinical outcomes, reserves, structures, airworthiness, or mission survival.
