# Full relativistic dark-sector sources and rights ledger

Scientific citation does not grant code/data/tune/table rights. Every external runtime
artifact still requires `ReferenceArtifactManifest` admission.

## Gravity and kinetic matter

- Weak-field relativistic cosmology and stress-energy PM: https://arxiv.org/abs/1604.06065
- Einstein--Vlasov numerical formulation survey: https://arxiv.org/abs/1106.1367
- Existing Phydrax Z4c, GRHD/GRMHD and ADM stress-exchange owners remain authoritative.

## Showers and hadronization

- PYTHIA 8.3 manual: https://arxiv.org/abs/2203.11601
- Herwig 7 overview: https://arxiv.org/abs/1512.01178
- Hidden-valley/dark-shower phenomenology: https://arxiv.org/abs/0712.2039
- Multichannel adaptation: https://arxiv.org/abs/hep-ph/9405257

Full Standard Model shower/hadronization is provider-bound. Native dark shower/string/
cluster/bound-state profiles are independently implemented only for named models.

## Quantum and thermal kinetics

- Quantum kinetic equations with spin/flavor coherence: https://arxiv.org/abs/1309.2628
- Nonequilibrium QFT and Kadanoff--Baym methods: https://arxiv.org/abs/1503.02907
- Off-shell transport: https://arxiv.org/abs/nucl-th/9903070
- HTL/LPM thermal rates require source-specific scheme and table manifests.

## Radiation

- M1 entropy closure: https://doi.org/10.1016/0022-4073(84)90112-2
- Variable-Eddington transport example: https://arxiv.org/abs/1201.2223

## Runtime

Semantically unbounded execution reuses Phydrax immutable repositories, chunk commits,
distributed checkpoints and bounded host-task backpressure. No external workflow engine
is a hidden dependency.
