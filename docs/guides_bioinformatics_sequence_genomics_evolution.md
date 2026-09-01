# Sequence, genomics, and evolution

This guide covers the native contracts in `sequence`, `genomics`, `phylogenetics`,
`population`, and `metagenomics`. Host file formats are covered separately in the
[interchange guide](guides_bioinformatics_interchange.md).

## Sequence representation

`AlphabetPlan` defines canonical symbols, ambiguity distributions, complements,
missing/gap/mask semantics, and a fingerprint. `DNA_IUPAC`, `RNA_IUPAC`, and
`PROTEIN_IUPAC` are ready-to-use plans. `SequenceBatch` is a fixed
`(record, position)` array with independent record, position-valid, and soft-mask
state. `QualityBatch` keeps Phred scores and encoding separate from sequence tokens.
`SequenceDistribution` represents categorical laws and must not be confused with a
hard sequence call.

Use `encode_sequence`/`encode_sequences` when no padding or truncation is needed. Use
`SequenceLoweringPlan` and `lower_sequences` or the FASTX lowering boundary when shape
must be fixed. Invalid symbols can be rejected or mapped to the declared unknown token;
overflow can be rejected or explicitly truncated. Only the report makes the latter
scientifically auditable.

`reverse_complement` preserves masks. `TranslationPlan` requires an explicit frame,
strand, genetic code, ambiguity policy, incomplete-codon policy, and stop policy.
`translate` returns a protein batch and per-record ambiguity, incomplete-base, stop,
output-length, and validity evidence.

## Alignment and sequence models

`align_affine` implements global, local, and semiglobal three-state dynamic
programming with `AffineGapPenalties`. A full `AlignmentExecutionPlan` covers the
complete finite rectangle. A diagonal-band plan conditions paths to that band and
reports boundary contact and a suggested expansion; it is not a full-domain optimum.
Traceback capacity overflow and incomplete/score-inconsistent paths invalidate the
result.

The other public families make different claims:

- `pair_hmm_forward_backward` and its potential form provide full-lattice,
  checkpoint-labelled exact, or explicitly conditional banded inference according to
  `PairHMMExecutionPlan`.
- `profile_forward`, `profile_backward`, `profile_forward_backward`, and
  `profile_viterbi` operate on a declared finite `ProfileHMM`.
- `scan_motif` performs dense PWM evaluation without claiming motif discovery beyond
  the supplied `PositionWeightMatrix`.
- `progressive_multiple_alignment` is a guide-tree/profile heuristic and says so in
  `ProgressiveMSAEvidence`.
- `align_partial_order` is exact for the supplied, certified finite DAG and resource
  bounds. It does not discover a pangenome graph.

Discrete traceback, Viterbi paths, ties, motif calls, guide trees, and DAG topology are
nondifferentiable. Score derivatives do not make those choices smooth.

## References, coordinates, and annotation

`ReferenceGenome` keeps sequence bytes on the host and binds them to an ordered
`ReferenceDictionary` with canonical names, aliases, lengths, circularity, and
MD5/SHA-256/SHA-512t24u identity. It never fetches reference data. Numeric
`ReferenceSequence` values are lowered explicitly for kernels.

Genomic coordinates are zero-based and half-open. `LinearCoordinate`,
`LinearInterval`, `SourceAlleleCoordinate`, `TranscriptCoordinate`, `CDSCoordinate`,
`ProteinCoordinate`, `PhaseCoordinate`, and `GraphCoordinate` are different types.
`IntervalSet` is the canonical bounded interval collection. Exact intersection,
union, difference, containment, and overlap operations retain incompatibility,
disjoint components, and capacity status rather than collapsing them to ad hoc tuples.

`GenomicAnnotation` and `FeatureParentRelation` retain typed features and all sparse
parent routes. Overlap queries are exact on supplied half-open intervals.
`audit_feature_parents` diagnoses duplicate routes and cycles. GFF3/GTF/BED parsing is
host-only and loss-preserving; converting those records into a numeric annotation or a
`TranscriptModel` remains an explicit semantic mapping by the caller.

`TranscriptModel` stores exons in biological 5′→3′ order. `CDSModel` carries GFF phase
per segment. `splice_transcript`, `assemble_cds`, and `translate_cds` expose reference
bounds, phase consistency, capacity, ambiguous codons, partial codons, and stops.
Coordinate liftover calls report ambiguity and loss rather than selecting one route
silently.

## Reads, mapping evidence, and variation

`ReadLayout` and `ReadBatch` couple lowered sequence/quality data to flags, reference
identity, CIGAR, mate/read-group/UMI identity, and capacities. `CigarBatch` covers the
complete BAM operation set; `expand_alignment_events` rejects partial event overflow.
SAM/BAM/CRAM parsing is an interchange concern. CRAM requires an explicit reference
identity and filename.

The native mapping layer evaluates supplied candidates and reference-aware pileup
likelihoods. `MappingExecutionPlan` defines candidate capacity and MAPQ semantics;
`CandidateGenerationEvidence` records incompleteness or truncation. These routines do
not perform exhaustive seed/index candidate discovery. A high MAPQ under an incomplete
candidate set is not evidence of global mapping uniqueness.

Small-variant normalization is reference-aware and keeps normalized alleles separate
from host VCF fields. Genotype inference:

1. enumerates the complete unordered state space with `enumerate_genotype_states`;
2. obtains natural-log likelihoods from local reads, VCF GL, or VCF PL;
3. supplies a uniform or allele-frequency prior; and
4. calls `infer_genotype`, which returns a continuous posterior and a separate
   thresholded hard/no-call decision.

The local-read model marginalizes each read over chromosome copies and assumes the
specified candidate alleles are complete. Capacity overflow, omitted candidates, no
coverage, invalid likelihoods, and invalid priors remain no-calls. Population allele
frequencies are inputs to the random-mating prior, not estimated inside the call.

Phasing performs bounded exact max-product inference over declared copy permutations
and local-read evidence. Structural-variant, somatic, copy-number, variation-graph,
and assembly routines operate on explicit candidate/state/path sets. Supplied overlap
and assembly paths are not de novo search. Breakpoint aggregation is calibrated direct
aggregation, not an exhaustive discovery engine. Copy-number segmentation is exact for
its finite state chain; states above `maximum_states` are absent, not approximated.

Epigenomic contracts keep ATAC/ChIP peaks, controls, and blacklists distinct; methylation
keeps sequence context, coverage, and conversion controls; MM/ML modification calls keep
orientation, code vocabulary, and calibration explicit. Descriptive peak occupancy is
not inferential enrichment, and ChIP enrichment requires a declared distinct control.

## Fixed-tree phylogenetics

`tree_topology` validates one rooted parent array and constructs complete pre/postorder
traversals. Polytomies are retained only when child capacity is sufficient.
`tip_partials_from_sequence` converts canonical symbols to one-hot state sets,
ambiguity to multiple allowed states, and missing/gap tokens to all states.

`JC69`, `K80`, `HKY85`, `GTR`, and `general_substitution_model` construct finite-state
CTMCs with explicit equilibrium/generator diagnostics. `DiscreteRateMixture` separates
site-rate heterogeneity from substitution identity. `LikelihoodPartition` must cover
each pattern exactly once and may request exact variable-site ascertainment correction.
`felsenstein_pruning` is an exact fixed-tree likelihood for the declared finite model,
floating-point transitions, branches, partitions, tip partials, and pattern weights.
It does not infer topology.

`ancestral_marginals` uses that fixed model. Strict and relaxed clock evaluators retain
time/rate validity and the clock assumption. `bounded_nni_search`/`nni_topology_search`
perform deterministic bounded NNI search and therefore remain heuristic topology
search, not proof of a global maximum. Topology, traversal, ancestral hard calls, and
search decisions are nondifferentiable.

## Population genetics

`GenotypeCohort` keeps genotype/dosage calls, ploidy, sample and variant validity,
position/chromosome identity, and missingness explicit. Summaries include allele
counts, Hardy–Weinberg diagnostics, linkage disequilibrium, and kinship. Binary
association and LOCO kinship operate under their stated model and do not infer causal
variants.

`PiecewiseConstantDemography` supports bounded coalescent and expected-SFS models;
`coalescent_event_log_likelihood`, `expected_folded_sfs`, `expected_unfolded_sfs`, and
`sfs_log_likelihood` expose event/spectrum validity and assumptions. Recombination,
local ancestry, imputation, and pedigree inference are conditioned on the supplied
panel/candidates/map and declared capacities.

Numeric `NodeTable`, `EdgeTable`, `MutationTable`, `TreeSequence`, and `marginal_tree`
provide a native tree-sequence representation; ecosystem conversion and large storage
are separate host boundaries. Population rows are not independent when related.
Group-aware splits and kinship-aware analysis remain the caller's responsibility.

## Metagenomics

`count_kmers` is exact over the full declared alphabet k-mer space under
`KmerCountingPlan`; its space grows exponentially with `k`. `minhash_sketch` and
`compare_minhash` are stochastic/approximate set-similarity summaries whose hash bits,
seed, and sketch size are part of the evidence.

Taxonomic assignment is restricted to `SuppliedTaxonomicCandidates` and a versioned
`TaxonomyTree`. Ambiguity is represented rather than resolved by hidden tie-breaking.
Community abundance and functional profiles aggregate declared evidence and do not
create reference coverage. `supplied_contig_binning` and marker evaluation assess a
supplied binning; they do not discover bins or certify organism genomes.

## Resource and interpretation checklist

- Preflight all state, sequence, path, candidate, child, event, graph, and output
  capacities; never interpret invalid partial arrays.
- Keep reference, alphabet, annotation, taxonomy, substitution model, and tree identity
  with every result.
- State whether a result is full-domain, band-conditioned, supplied-candidate,
  bounded-search, approximate, or heuristic.
- Keep likelihood, prior, posterior, and hard call separate.
- Treat reference/annotation choice, sample relatedness, homologous families, and model
  selection as leakage risks.
- Do not infer clinical significance, organism identity, evolutionary causality, or
  global search optimality from these numerical primitives.
