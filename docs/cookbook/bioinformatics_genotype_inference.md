# Genotype inference from PL values

This recipe keeps the finite state space, likelihood, prior, posterior, and thresholded
hard call separate.

```python
from phydrax.bioinformatics.genomics import (
    allele_frequency_genotype_prior,
    enumerate_genotype_states,
    genotype_likelihoods_from_pl,
    infer_genotype,
)

# Diploid biallelic Number=G order: 0/0, 0/1, 1/1.
states = enumerate_genotype_states(
    allele_count=2,
    ploidy=2,
    max_genotypes=3,
)
assert bool(states.valid)

likelihoods = genotype_likelihoods_from_pl(
    [35, 0, 42],
    states,
    depth=18,
)
prior = allele_frequency_genotype_prior([0.7, 0.3], states)
result = infer_genotype(
    likelihoods,
    prior,
    states,
    min_depth=10,
    min_posterior=0.90,
    max_genotype_quality=99.0,
)
assert bool(result.posterior.valid)

if bool(result.hard_call.called):
    print(result.hard_call.alleles, float(result.hard_call.genotype_quality))
else:
    print("no-call", int(result.hard_call.status))
print(result.posterior.probabilities, result.posterior.dosage)
```

PL is converted to natural-log likelihood internally. The allele-frequency prior is the
random-mating multinomial prior and requires normalized frequencies. `max_genotypes`
must cover the complete unordered state space; overflow is observable and cannot be
used for a partial call. Read-based inference additionally requires a complete local
allele/haplotype candidate set. The posterior is still useful when the explicit hard
call threshold produces a no-call.
