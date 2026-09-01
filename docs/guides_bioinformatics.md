# Bioinformatics

`phydrax.bioinformatics` is the native biological-computation layer. It keeps file
interchange and unbounded host metadata outside JAX, lowers only validated numeric
batches, and composes existing Phydrax finite-state inference, PGM, sparse, graph,
transport, optimization, dynamics, topology, neural, artifact, and atomistic
substrates.

## Scientific claims

Every result binds a `BioinformaticsMethodContract` that separates:

- an exact finite model from an approximate model, relaxed objective, heuristic, or
  learned predictor;
- exact discrete, direct floating-point, tolerance-controlled iterative, and
  stochastic execution;
- exact AD, almost-everywhere, implicit, unrolled, stochastic-estimator, surrogate,
  and nondifferentiable behavior;
- conditioning, truncation, capacities, tolerances, assumptions, and
  nondifferentiable outputs.

Exact inference is exact only for the declared finite model and supplied state space.
A banded pair-HMM, supplied mapping-candidate set, fixed phylogenetic topology, or
bounded genotype state space does not establish correctness outside that boundary.

## Host and kernel boundary

Host adapters may retain strings, source headers, opaque attributes, sparse backed
objects, indexed readers, and complete file records. Numeric kernels accept only
fixed-capacity arrays and explicit masks. Lowering reports overflow, unknown content,
reference mismatch, coordinate conversion, and semantic loss. It never silently
truncates, densifies, fabricates quality, or retrieves a reference from the network.

Global genomic coordinates remain host-side 64-bit values until lowered into a
validated reference window. Device indices are bounded, relative numeric arrays.
Dataset labels and patient identifiers do not enter compiled PyTrees.

## Sequence analysis

`phydrax.bioinformatics.sequence` provides:

- DNA, RNA, and protein IUPAC alphabets with distinct ambiguity, gap, padding,
  unknown, missing, and soft-mask semantics;
- numeric `SequenceBatch`, `SequenceDistribution`, and `QualityBatch` contracts;
- reverse complement and policy-controlled translation;
- ambiguity-aware substitution matrices;
- full, checkpointed, and explicitly conditional banded affine alignment;
- normalized pair-HMM forward/backward, marginals, and Viterbi decoding;
- position-weight matrices, profile HMMs, supplied-DAG partial-order alignment, and
  heuristic progressive MSA.

Observed discrete sequences and relaxed sequence distributions are separate types.
Traceback, guide-tree construction, hard token selection, and MSA topology are
nondifferentiable.

## Genomics and population genetics

`genomics` owns reference identity, coordinate spaces, annotations, transcripts,
reads, CIGAR events, mapping evidence, small variants, phasing, structural variation,
copy number, somatic evidence, assembly graphs, and variation graphs.

Small-variant likelihood, posterior, dosage, hard call, and no-call are distinct.
Structural variants preserve oriented breakends and event links. Copy-number and
somatic models carry ploidy, sex/PAR, purity, contamination, LOH, and uncertainty
rather than reusing small-variant records.

`population` provides mixed-ploidy cohorts, genotype uncertainty, HWE/SFS/LD/kinship,
pedigree inference, recombination/local ancestry, bounded imputation, tree-sequence
tables, demographic likelihoods, and relatedness-aware quantitative and binary
association. Fine mapping, PRS, survival, and unqualified gene-burden analyses are
not part of the current surface.

## Phylogenetics

`phylogenetics` provides dynamic numeric tree topologies; JC69, K80, HKY85, GTR, and
general finite-state substitution models; rate mixtures; fixed-tree Felsenstein
pruning; ancestral marginals; bounded NNI search; and clock likelihoods.

The likelihood contract includes root law, ambiguity and missingness, site-pattern
weights, partitions, invariant/rate mixtures, ascertainment correction, scaling, and
branch units. Fixed-tree likelihood does not make topology search exact.

## Omics and experimental units

`foundation` owns feature identity, ontology relations, biospecimen lineage,
experimental units, exchangeability, biological grouping, split provenance, and
leakage audits. Statistical analyses therefore distinguish subject, specimen,
aliquot, library, run, section, cell, spot, and spectrum.

`omics` provides dense and fixed-sparse count/continuous assays while distinguishing
observed zero, structural absence, and missingness. The confirmatory count path is:

1. raw counts and explicit experimental design;
2. pseudobulk or declared technical aggregation;
3. normalization offsets;
4. feature dispersion, trend, and shrinkage;
5. negative-binomial GLM;
6. estimable contrasts and Wald/LRT evidence;
7. explicit-family BH or BY adjustment.

Cells are not biological replicates. Exploratory transport integration, multimodal
alignment, latent representations, RNA velocity, QC decisions, and feature-set tests
retain their separate heuristic, relaxed, approximate, or learned classifications and
cannot feed confirmatory p-values silently.

Epigenomics keeps chromatin fragments, bisulfite methylation, and native SAM MM/ML
base modifications as different observation families.

## Spatial and spectral biology

`spatial` binds assays to samples, coordinate frames, units, transforms, sections,
and fixed-capacity neighborhood graphs. Moran/Geary inference uses an explicit
exchangeability plan. Registration carries transport convergence and uncertainty.
OME-NGFF conversion requires actual NGFF axes and multiscale transforms; generic
Zarr is not treated as NGFF.

`spectrometry` distinguishes profile/centroid spectra, chromatograms, acquisition,
precursors, mass calibration, LC-MS feature matching, proteomic PSM/peptide/protein
error control, metabolite confidence, and replicate quantification. Missing peaks are
not zero intensity. A q-value does not establish metabolite identity.

## Metagenomics

`metagenomics` provides exact bounded k-mer counting, finite-bit MinHash evidence,
versioned taxonomies, ambiguous taxonomic assignment, community profiles with
unclassified mass, supplied contig-bin metrics, and functional profiles. Database
candidate retrieval, production assembly, bin discovery, strain resolution, and
taxonomic classification remain explicit heuristic or external boundaries.

## Structure and RNA

`structure` preserves mmCIF biological and chemical semantics before lowering into
native atomistic structures: author/label IDs, entities, chains, residues, altlocs,
models, assemblies, chemical components, bonds, modified residues, ligands, metals,
missing content, and connections. Lowering rejects unresolved required chemistry
instead of inventing connectivity or force-field parameters.

`rna` provides fingerprinted scoring/energy models, constraints, exact
pseudoknot-free MFE and log-space partition/inside-outside marginals for the declared
grammar, restricted pseudoknot heuristics, and tertiary-restraint lowering. No
unlicensed thermodynamic parameter table is bundled.

## Systems biology and learned models

`systems` provides typed stoichiometric networks, FBA/FVA with solver certificates,
kinetic reaction systems, regulatory PGMs, conservation, and identifiability. SBML
host records lower only when every used Level/Version/package semantic is supported.

`models` composes native recurrent, attention, graph, and equivariant layers with
biological batches. Foundation manifests bind model, tokenizer, alphabet, weights,
license, split, adapter, and pretraining-overlap identity. External runtimes remain
host-only. Inverse design reports relaxed and hard objectives, constraint violations,
relaxation gaps, diversity, and deterministic sampling evidence.

## Optional interoperability

Install only the needed extra:

- `bioinformatics-hts`: pysam/HTSlib;
- `bioinformatics-structure`: Gemmi;
- `bioinformatics-tree-sequence`: tskit;
- `bioinformatics-phylo-interop`: Biopython Newick/NEXUS;
- `bioinformatics-omics` and `bioinformatics-mudata`: assay objects/files;
- `bioinformatics-zarr` or `bioinformatics-ngff`: chunked or OME-NGFF data;
- `bioinformatics-spectrometry`: Pyteomics read-side mzML;
- `bioinformatics-sbml`: libSBML.

Importing Phydrax or any compiled scientific domain does not import these packages.
