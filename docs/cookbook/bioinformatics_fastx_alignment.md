# FASTX to affine alignment

This recipe keeps FASTA identifiers on the host, rejects lowering loss, and runs a
full-domain global alignment.

```python
from phydrax.bioinformatics.interchange import lower_fasta, parse_fasta
from phydrax.bioinformatics.sequence import (
    AffineGapPenalties,
    AlignmentExecutionPlan,
    DNA_IUPAC,
    SequenceLoweringPlan,
    align_affine,
    nucleotide_substitution_table,
)

records = parse_fasta(
    ">query example\nACGTTGCA\n"
    ">target example\nACGTAGCA\n"
)

lowering = SequenceLoweringPlan(
    DNA_IUPAC,
    record_capacity=2,
    sequence_capacity=8,
    invalid_symbol_policy="reject",
    overflow_policy="reject",
)
batch, lowering_report = lower_fasta(records, lowering, numeric_record_ids=[10, 11])
assert not bool(lowering_report.loss_occurred)

execution = AlignmentExecutionPlan.full(
    8,
    8,
    traceback_capacity=16,
)
result = align_affine(
    batch.token_codes[0],
    batch.token_codes[1],
    nucleotide_substitution_table(match_score=2.0, mismatch_score=-3.0),
    AffineGapPenalties(-5.0, -1.0),
    execution,
    mode="global",
    query_mask=batch.valid_mask[0],
    target_mask=batch.valid_mask[1],
)
assert bool(result.valid)
assert bool(result.exact)
print(float(result.score), int(result.alignment_length))
```

`AlignmentExecutionPlan.full` makes the result exact for the declared affine global
model and finite rectangle. A diagonal-band plan is exact only within its band and must
be reported as conditional if `result.truncated` is true. Traceback operations and tie
choices are nondifferentiable. FASTQ input follows the same boundary through
`parse_fastq` and `lower_fastq`, but requires an explicit Phred encoding and does not
implicitly alter alignment scores.
