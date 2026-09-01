# Annotation to translated CDS

GFF parsing is loss-preserving but deliberately does not guess a transcript model. This
recipe performs that semantic mapping explicitly for one single-exon CDS.

```python
from phydrax.bioinformatics.genomics import (
    CDSModel,
    ReferenceGenome,
    Strand,
    TranscriptModel,
    assemble_cds,
    translate_cds,
)
from phydrax.bioinformatics.interchange import GFF3FeatureLine, parse_gff3
from phydrax.bioinformatics.sequence import decode_sequences

reference = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"
genome = ReferenceGenome.from_sequences(
    {"chr1": reference},
    assembly_id="cookbook-reference",
)

lines = parse_gff3(
    "##gff-version 3\n"
    "chr1\tdemo\texon\t1\t39\t.\t+\t.\tID=exon1;Parent=tx1\n"
    "chr1\tdemo\tCDS\t1\t39\t.\t+\t0\tID=cds1;Parent=tx1\n"
)
features = tuple(line for line in lines if isinstance(line, GFF3FeatureLine))
exon = next(line for line in features if line.feature_type == "exon")
cds_line = next(line for line in features if line.feature_type == "CDS")
assert exon.seqid == cds_line.seqid == "chr1"
assert exon.strand == cds_line.strand == "+"
assert cds_line.phase is not None

transcript = TranscriptModel(
    0,
    genome.dictionary.resolve("chr1"),
    [exon.start],
    [exon.end],
    [True],
    strand=Strand.FORWARD,
    reference_length=len(reference),
)
cds = CDSModel(
    transcript,
    [cds_line.start],
    [cds_line.end],
    [True],
    [cds_line.phase],
)
assembly = assemble_cds(genome, cds, capacity=39)
assert bool(assembly.valid)
assert bool(assembly.phase_consistent)

translated = translate_cds(assembly)
assert bool(translated.valid)
protein = decode_sequences(translated.translation.sequences)[0]
print(protein)
```

Coordinates in `GFF3FeatureLine` are already converted to zero-based half-open form.
The reference stays host-resident and is digest-checked by `ReferenceGenome`. For
multi-exon or reverse-strand transcripts, preserve biological 5′→3′ segment order and
the declared GFF phase for every CDS segment. `translate_cds.exact` becomes false when
ambiguity or an incomplete codon is present; validity and exactness are separate.
