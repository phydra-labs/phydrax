# Bind a native biological model artifact

This script creates one native alphabet-bound model artifact, hashes the exact bytes,
and binds it to caller-supplied tokenizer and reviewed license files. The license flags
in the script are assertions by the caller; set them only after review.

```python
from pathlib import Path
import argparse

import equinox as eqx
import jax

from phydrax.bioinformatics.models import (
    FoundationModelManifest,
    LicenseProvenance,
    PretrainingOverlapProvenance,
    SequenceEmbedding,
    TokenizerProvenance,
    bind_native_foundation_model,
    native_model_parameter_sha256,
    native_model_structure_fingerprint,
    sha256_file,
)
from phydrax.bioinformatics.sequence import PROTEIN_IUPAC

parser = argparse.ArgumentParser()
parser.add_argument("artifact", type=Path)
parser.add_argument("tokenizer", type=Path)
parser.add_argument("license", type=Path)
parser.add_argument("spdx_id")
parser.add_argument("evaluation_split_id")
parser.add_argument("homology_partition_id")
args = parser.parse_args()

model = SequenceEmbedding(PROTEIN_IUPAC, 32, key=jax.random.key(0))
args.artifact.parent.mkdir(parents=True, exist_ok=True)
eqx.tree_serialise_leaves(args.artifact, model)
artifact_sha256 = sha256_file(args.artifact)

tokenizer = TokenizerProvenance(
    "protein-iupac-native",
    sha256_file(args.tokenizer),
    PROTEIN_IUPAC,
    normalization="identity",
)
license_provenance = LicenseProvenance(
    args.spdx_id,
    sha256_file(args.license),
    status="verified",
    inference_allowed=True,
    adaptation_allowed=False,
    redistribution_allowed=False,
    attribution="See the digested license file.",
)
overlap = PretrainingOverlapProvenance(
    "unknown",
    evaluation_split_id=args.evaluation_split_id,
    homology_partition_id=args.homology_partition_id,
    search_method="",
    identity_threshold=0.30,
)
manifest = FoundationModelManifest(
    "cookbook-protein-embedding",
    "SequenceEmbedding/32",
    artifact_sha256,
    native_model_parameter_sha256(model),
    native_model_structure_fingerprint(model),
    tokenizer,
    license_provenance,
    overlap,
)
bound = bind_native_foundation_model(
    model,
    manifest,
    artifact_sha256=sha256_file(args.artifact),
    tokenizer_fingerprint=tokenizer.fingerprint,
    alphabet_fingerprint=PROTEIN_IUPAC.fingerprint,
    evaluation_split_id=args.evaluation_split_id,
    homology_partition_id=args.homology_partition_id,
)
print(manifest.fingerprint, int(bound.binding.status))
```

The resulting status records that pretraining overlap is unknown; it must not be
reported as “no detected overlap.” For a real pretrained model, load the already-native
callable by its documented constructor, hash its actual serialized bytes and numeric
parameters, and retain its exact split/family provenance. `bind_native_foundation_model`
rejects mismatched artifact, parameter, structure, tokenizer, alphabet, license, split,
homology partition, or base-model identity. It does not download weights, convert an
external framework, or establish biological validity. Low-rank artifacts additionally
require `LowRankAdapterProvenance` and exact base binding.
