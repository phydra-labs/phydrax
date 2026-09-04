# Skeletal-muscle external-model interchange and execution worksets

The interchange layer has two deliberately narrow jobs:

1. bind one external skeletal-muscle artifact to a complete, immutable identity; and
2. lower independent items into deterministic, homogeneous, fixed-shape JAX worksets.

It is not a CellML compiler, a co-simulation protocol, a mutable model builder, or a registry of muscle laws. It does not combine force routes. An external descriptor names exactly one `force_owner`; a provider-native raw signed force therefore remains mutually exclusive with D1, De Groote, Shorten, or any other native force owner.

## Strict external identity

`ExternalModelDescriptor` includes every field that can change the meaning of an external execution:

- source package, immutable source revision, source URI, license expression and URI;
- a provenance reference and SHA-256 digest of the provenance record;
- every asset name, URI, media type, byte count, and SHA-256 digest;
- an ordered transformation chain with tool revisions, specification digests, input digests, and output digests;
- exact provider package versions and the final compiled digest;
- the dimensional contract, coordinate map, actuator map, and sensor map; and
- the one enabled force owner.

The constructor rejects a transformation chain with undeclared inputs, an unused declared asset, a stage that omits the immediately preceding output, or a final output different from the compiled digest. `prepare_external_model_descriptor` then compares the descriptor with an `ExternalModelHostInventory`. Source identity, assets, provider versions, compiled digest, and external channel coverage must all match exactly. A mismatch raises `ExternalModelPreparationError`; its `evidence` retains the descriptor ID, inventory ID, and all detected failure reasons.

The host inventory is evidence supplied by the concrete provider adapter after it has inspected the provider. It is not a substitute for hashing assets or interrogating provider versions in that adapter.

SHA-256 terminology and digest size follow NIST FIPS 180-4 [1]. A `license_expression` is stored rather than interpreted; adapters can use the SPDX expression grammar where applicable [2]. Neither fact grants a license or establishes provenance by itself.

## Dimensional and channel contract

`ExternalModelQuantity.si_dimensions` is the seven-exponent tuple

`(mass, length, time, electric current, thermodynamic temperature, amount of substance, luminous intensity)`.

This ordering and the SI unit meanings follow the BIPM SI Brochure [3]. `ExternalModelChannelBinding` always means

`target = scale × source + offset`.

The three maps have explicit directions:

- coordinate: external provider coordinate → Phydrax coordinate;
- actuator: Phydrax actuator value → external provider actuator;
- sensor: external provider sensor → Phydrax observation.

All external channels must be covered exactly during preparation. External axes use the order reported by `ExternalModelHostInventory`; Phydrax axes use bindings sorted by their Phydrax-side name (coordinate and sensor target names, actuator source names). Mapping indices and affine factors become fixed JAX arrays in `PreparedExternalModelDescriptor`; calls operate on the last array axis and support arbitrary leading batch axes.

The runnable example uses this explicit contract:

| Canonical quantity | Role | SI dimensions | External unit | Phydrax/kernel unit | Axis and sign | Support |
|---|---|---:|---|---|---|---|
| `musculotendon_length` | coordinate | `(0, 1, 0, 0, 0, 0, 0)` | cm | m | positive along the declared musculotendon line of action; scale `+0.01` | one lumped musculotendon actuator |
| `independent_excitation` | actuator | `(0, 0, 0, 0, 0, 0, 0)` | 1 | 1 | positive excitation; scale `+1` | one lumped musculotendon actuator |
| `raw_provider_force` | sensor | `(1, 1, -2, 0, 0, 0, 0)` | N | N | provider signed force is opposite the declared Phydrax line-of-action sign in the example; scale `-1` | one lumped musculotendon actuator |

These names and the example sign are an explicit interchange contract, not a universal claim about external providers. A real adapter must replace the example provenance, hashes, revisions, units, axes, and sign with values established from its authoritative source. `raw_provider_force` is not normalized and is not multiplied by another force law.

## Deterministic fixed-shape worksets

The private general owner is `phydrax._execution_workset`. It extends the existing `PoolExecutionSignature` rather than creating a skeletal-specific execution registry.

`ExecutionWorksetPlan` accepts one semantic ID and one complete execution signature per item. Preparation:

1. sorts items by semantic ID;
2. groups them by exact signature ID;
3. chunks each group into capacities from 1 through 64;
4. pads only the final bucket for a signature; and
5. records a mask and a reversible canonical item permutation.

Values supplied to `gather`, evaluation, and checkpoints are always ordered by `plan.semantic_ids`, not by the order passed to the constructor. `gather` and `scatter` work on numeric array pytrees and preserve trailing shapes. Padded lanes gather a safe duplicate from their own homogeneous bucket so a kernel never receives a synthetic zero state; `valid_mask` identifies those lanes, and scatter masks their outputs before returning canonical item order.

Each semantic ID is hashed into a checked 32-bit RNG index. A work item key is obtained by folding that index and an explicit per-item restart counter into the root key. The semantic key therefore does not depend on bucket capacity, padding, or lane placement. JAX JEP 263 specifies the functional PRNG model and `fold_in` semantics [4]; the semantic-ID hashing, collision rejection, and restart-counter policy are Phydrax contracts introduced here.

`evaluate_execution_worksets_serial` and `evaluate_execution_worksets_vmap` invoke the same operation with

`(signature, item, key, semantic_rng_index)`.

Both return `ExecutionWorksetEvaluation` and `ExecutionWorksetEvidence`. The evaluation carries the whole result and the atomically advanced per-item RNG counters; counter overflow fails before a new continuation state is exposed. Evidence reports finite output, exact coverage, active items, and padded lanes. The serial route is an executable reference for the vectorized route, not a different mathematical model.

`ExecutionWorksetCheckpoint` content-addresses the prepared identity, canonical semantic IDs, complete item-state pytree, and RNG counters. Restore rejects a checkpoint from another preparation and rejects any payload whose recomputed content identity differs. There is no partial-state restore or symptom-level fallback.

## Example and qualification

Run the example:

```console
python examples/skeletal_muscle_interchange_worksets.py
```

Run the focused qualification tool:

```console
python tools/skeletal_muscle_interchange_qualification.py
```

The qualification covers strict identity rejection and retained failure evidence, affine channel directions, stable gather/scatter, serial/vmap equality, capacity-independent semantic RNG keys, and checkpoint/restart identity. Benchmark source is in `benchmarks/skeletal_muscle_execution_worksets.py`.

## Distributed execution gate

No distributed skeletal-muscle workset API is released. `ExecutionWorksetPlan` therefore rejects signatures whose `shard_count` is not one. The measured development environment had `jax.local_device_count() == 1` and the sole device was `TFRT_CPU_0`. That cannot establish real-device serial equivalence, deterministic and fast collective reductions, checkpoint/restart across shards, or scaling. Device emulation and a fake multi-device fallback are intentionally absent. The qualification and benchmark record their measured local device count and device names so a future real-device implementation has an explicit release gate.

## References

1. National Institute of Standards and Technology, *Secure Hash Standard (SHS)*, FIPS PUB 180-4, 2015. <https://doi.org/10.6028/NIST.FIPS.180-4>
2. SPDX Workgroup, *SPDX Specification 3.0.1, Annex D: SPDX License Expressions*, 2025. <https://spdx.github.io/spdx-spec/v3.0.1/annexes/spdx-license-expressions/>
3. Bureau International des Poids et Mesures, *The International System of Units (SI)*, 9th edition, 2019 (updated 2026), sections 2.3–2.4. <https://doi.org/10.59161/AUEZ1291>
4. JAX project, *JEP 263: JAX PRNG Design*, 2020. <https://docs.jax.dev/en/latest/jep/263-prng.html>
