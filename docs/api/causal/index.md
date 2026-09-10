# Causal inference

`phydrax.causal` owns causal variables, study designs, graph semantics, identification certificates, structural causal models, observed-data effect estimation, causal discovery evidence, and intervention-design adapters.

The package is array/PyTree-native and content-addressed. pandas, NetworkX, sklearn, and PyMC objects are not canonical state. Existing Phydrax PGM, ML, UQ, weighting, artifact, and qualification contracts remain the numerical owners.

## API areas

- [Structures](structures.md): DAG, ADMG, MAG, PDAG, CPDAG, PAG, separation, projection.
- [Identification](identification.md): law recovery, adjustment, general ID, certificates, finite evaluation.
- [Structural causal models](scm.md): mechanisms, interventions, UQ binding, counterfactuals.
- [Estimation](estimation.md): cross-fitting, overlap, g-computation, IPW, AIPW.
- [Discovery](discovery.md): CI tests, PC-Stable, conservative FCI, bounded GES.
- [Diagnostics and qualification](diagnostics.md): diagnostics, sensitivity, claims, archives, experiment selection.

See the [causal inference guide](../../guides_causal_inference.md) before interpreting any numerical output.
