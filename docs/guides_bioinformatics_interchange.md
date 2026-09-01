# Bioinformatics interchange

`phydrax.bioinformatics.interchange` owns host parsing, serialization, adapter
materialization, and explicit lowering into native numeric contracts. Parsers do not
enter JAX transformations. No adapter downloads data or discovers a provider.

## General boundary

A safe interchange workflow has four stages:

1. load bytes or external records on the host;
2. parse into an immutable host record that preserves source semantics;
3. validate identity, units, reference, capacities, and unsupported fields; then
4. lower into a domain object, keeping the loss/evidence result.

Paths, free-form strings, ordered attributes, raw source text, external objects, and
variable collections stay outside PyTrees. Lowered arrays carry numeric IDs, masks,
units, contracts, capacities, and fingerprints. Keep the host-to-numeric identity map
with your analysis artifact.

## FASTA and FASTQ

`parse_fasta` and `parse_fastq` accept text, paths, or line iterables. FASTQ requires an
explicit `PhredEncoding`; sequence and quality lengths must match. `FASTARecord` and
`FASTQRecord` are host-only.

`lower_fasta` never invents quality data. `lower_fastq` creates aligned
`SequenceBatch`/`QualityBatch` values and refuses mixed Phred encodings. Both use a
`SequenceLoweringPlan`; inspect `SequenceLoweringReport` for record overflow, truncated
symbols, mapped invalid symbols, and retained records. Formatting/writing returns
canonical text unless a destination is supplied.

## GFF3, GTF, and BED

`parse_gff3`, `parse_gtf`, and `parse_bed` retain directives/comments, raw line text,
newline style, attribute order, and native coordinate conventions. GFF3/GTF one-based
closed coordinates are exposed internally as zero-based half-open intervals; BED is
already half-open. Serialization can preserve original text.

`gff3_parent_relations` retains duplicate IDs, multi-parent routes, ambiguity, and
unresolved parent references. It does not choose a transcript model or coerce GFF/GTF
semantics into one annotation ontology. Numeric `GenomicAnnotation`, `TranscriptModel`,
and `CDSModel` construction remains an explicit mapping with a pinned reference and
feature vocabulary.

## VCF-like small variants

`parse_vcf` is a bounded germline small-variant host parser. `max_records`,
`max_samples`, and `max_alleles` are enforced before a partial result can be mistaken
for a complete callset. Host `VCFRecord` keeps strings and ordered sample fields.

`vcf_record_to_small_variant` requires the exact reference sequence and numeric
reference/contig identities, then normalizes into the native scientific site contract.
`vcf_record_from_small_variant` performs the inverse host mapping. Sample GL and PL are
converted with `vcf_sample_likelihoods` to natural-log `GenotypeLikelihoods`; likelihood,
prior, posterior, and hard call remain separate.

This boundary does not implement arbitrary VCF structural-variant schemas, clinical
annotation, phasing conventions beyond represented fields, or reference fetching.

## SAM, BAM, and CRAM

`SAMLikeRecord` is the dependency-free host record. `sam_like_from_pysam` and
`sam_like_records_from_pysam` copy from already-open pysam objects without retaining
those objects. `load_pysam_records` lazily imports pysam and materializes a local file.

CRAM requires both an explicit `reference_identity` and `reference_filename`; a decoder
filename is not treated as proof of reference identity. Mapping quality 255 is
normalized to unknown. `read_batch_from_sam_like` lowers reads only after reference,
read-group, record, sequence, quality, CIGAR, and layout capacities are declared.
`mapping_candidates_from_sam_like` reports incomplete bounded candidate sets; it is not
a mapper or candidate generator.

## PDBx/mmCIF

`parse_mmcif` and `load_mmcif` require entity, structural asymmetry, and atom-site
identity needed by `MacromolecularRecord`. The record retains label/auth identifiers,
chemical components, coordinate models, alternate locations, missingness, links, and
assembly operations. `dumps_mmcif`/`dump_mmcif` serialize the represented content.

`StructureLoweringPlan` and `lower_macromolecular_record` own the all-or-nothing numeric
and atomistic boundary. Parsing alone does not infer unresolved element identity,
chemistry, missing coordinates, protonation, or a force field.

## OME-NGFF metadata

`is_ome_ngff_metadata` distinguishes a declared OME-NGFF multiscale object from a plain
Zarr group. `ome_ngff_to_image_pyramid` requires host attributes plus array and chunk
shapes supplied by the storage layer. `image_pyramid_to_ome_ngff` serializes native
metadata. These calls do not open stores, read image arrays, or reinterpret generic
Zarr as NGFF. Pixel axes have no physical scale unless the metadata supplies one.

## SBML host AST

The SBML boundary is parser-neutral. `SBMLDocumentAST` and its compartment, species,
reaction, unit, kinetic-law, rule, event, and package nodes retain the host semantics
needed for validation. `validate_sbml_document` selects the explicit supported
Level/Version/package semantic profile. `lower_sbml_document` lowers only a fully
supported network and supported closed kinetic-law subset.

Unsupported rules, events, packages, units, kinetic constructs, and incomplete kinetics
produce semantic evidence or `SBMLSemanticError`; they are never ignored. The native
layer does not parse XML itself and does not execute arbitrary MathML expressions.

## mzML-like reads

`lower_mzml_record` and `lower_mzml_records` accept Pyteomics-compatible host mappings
and an `MzMLLoweringPlan` with explicit spectrum/point/precursor capacity and unit
defaults. They immediately copy into native `MassSpectrum`/`AcquisitionRun` contracts
and preserve read evidence. Generic mappings are not retained in JAX state.

`read_pyteomics_mzml` is the only built-in file reader for this route. It lazily imports
Pyteomics, reads a local mzML source, and lowers immediately. It is not a vendor raw
reader, downloader, peak picker, library search, or calibration procedure.

## Optional dependencies

Core parsers, host records, numeric lowerings, and all cookbooks use the normal Phydrax
dependencies. The implemented optional reader routes are:

| Install extra | Optional package | Public calls enabled | Boundary |
| --- | --- | --- | --- |
| `phydrax[bioinformatics-hts]` | `pysam` | `load_pysam_records` | local SAM/BAM/CRAM reading; CRAM reference remains explicit |
| `phydrax[bioinformatics-spectrometry]` | `pyteomics` | `read_pyteomics_mzml` | local mzML reading into native spectra |

Adapter functions that accept an already-open external object do not import or retain
that package. Other packaging extras do not, by their presence alone, imply a public
reader or converter. Install an extra only for the concrete public call that names it.

## Provenance checklist

Record the source path or URI supplied by the caller, a content digest, parser/adapter
name, reference and vocabulary identities, declared units, capacities, loss/status
evidence, optional package version, and the resulting native fingerprints. Never log a
path as if it were a digest. No qualification or production workflow should fetch a
missing reference, annotation, dictionary, model, ontology, or dataset automatically.
