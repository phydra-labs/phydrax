# RNA partition function

The built-in Nussinov model is a declared scoring grammar, not an empirical RNA
thermodynamic parameter set.

```python
import jax.numpy as jnp

from phydrax.bioinformatics.rna import nussinov_energy_model, partition_function

# Canonical model code order is A=0, C=1, G=2, U=3.
sequence_codes = jnp.asarray([2, 2, 2, 0, 0, 0, 1, 1, 1], dtype=jnp.int32)
model = nussinov_energy_model(
    pair_energy=-1.0,
    wobble_energy=-0.5,
    unpaired_energy=0.0,
    temperature=310.15,
    minimum_hairpin_length=3,
)
result = partition_function(sequence_codes, model)
assert bool(result.valid)

print(float(result.log_partition))
print(result.pair_marginals)
print(result.unpaired_marginals)
print(float(result.expected_energy), float(result.entropy))
```

The inside/outside recurrence sums every noncrossing partial matching admitted by the
declared additive grammar. The result is exact for that grammar and floating-point
Boltzmann evaluation; it is not an exact physical RNA ensemble. Hard constraints,
sequence codes, and pair decisions are nondifferentiable, while energy-parameter
derivatives of `log_partition` have the documented expected-count meaning. Crossing
pairs require `restricted_pseudoknot_fold`, which is explicitly greedy and heuristic.
