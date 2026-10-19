# Lifecycle

`phydrax.lifecycle` owns immutable accepted revisions, transactional composition
changes, content-addressed artifacts, checkpoints, topology-aware restart, and
provenance. It does not infer compatibility from equal shapes, display names, or
an old object layout.

## Meshing epochs and atomic rebind

`MeshingAcceptedEpoch` binds one accepted `CellMeshingResult` to its complete
named physical field banks and optional mesh transition. It is a scientific
record, not a generic mapping wrapper. A topology-changing solver update stages
its target carrier, geometry, fields, materials, history, solver preparation,
and physical reanalysis through one `CompositionRebind`; a failed disposition
leaves the complete accepted composition unchanged.

Meshing source-closure codecs used by qualification and cold-restart campaigns
remain an internal, allowlisted boundary. They admit registered owner types and
complete field sets, reject stale or foreign source authority, and never
unpickle arbitrary Python objects. Their canonical record is the current
representation: there is no internal schema-generation field or compatibility
alias. Consequently, the intentional hierarchy PyTree/fingerprint cutover
changes affected recipe, model, receipt, and archive content identities.

`meshing_source_owner_reconstruction_scope()` bounds authenticated source-owner
reuse to one request. Archive and checkpoint operations open this scope
automatically; qualification workflows may span several related reads,
recertifications, and field preparations with one outer scope. Nested scopes
share only process-local earned validation and release retained owners when the
outer request ends. The scope does not bypass content, archive-limit, or source
binding checks.

Finite-element and finite-volume field declarations use
`MeshingFiniteElementFieldDeclaration` and `MeshingFiniteVolumeFieldDeclaration`,
respectively. `MeshingFieldDeclaration` is their abstract shared contract, not a
constructor. These concrete declaration types define the current persisted
recipe representation.

::: phydrax.lifecycle
    options:
      members:
        - Composition
        - CompositionDependency
        - CompositionEntry
        - CompositionFacet
        - CompositionRebind
        - CompositionRebindReceipt
        - CompositionRole
        - CompositionTransport
        - CompositionTransportKind
        - TransactionalCandidate
        - TransactionalCommit
        - commit_candidate
        - commit_composition_rebind

## Checkpoint and topology restart

An identity restart requires the same topology and complete logical array
content. A changed topology requires an explicit `TopologyRestartRelation` and
restorer; an accepted `CompositionRebindReceipt` may authorize that relation.
Direct restore validates complete, disjoint canonical byte coverage and writes
destination-owned ranges without a mandatory global payload gather. A semantic
change is not relabeled as topology migration.

`restore_meshing_checkpoint` returns a `MeshingCheckpointRestore` record:
`state` retains the admitted source-owner records, `arrays` contains the restored
scientific leaves, and `admission` carries the topology decision. Source recipe
binding lives in a downward lifecycle owner, so constructing a
`MeshingCheckpointState` does not import the checkpoint publisher. The checkpoint
state fingerprint payload and array publication order are unchanged.

::: phydrax.lifecycle
    options:
      members:
        - AddressableCheckpointShard
        - CanonicalRestartChunk
        - MeshingCheckpointRestore
        - MeshingCheckpointState
        - MeshingSourceBinding
        - DirectRestorePlan
        - RestartAdmission
        - RestartChunkMapping
        - RestartClass
        - RestartExecutionReport
        - TopologyRestartPolicy
        - TopologyRestartRelation
        - admit_topology_restart
        - admit_meshing_source_model
        - meshing_source_owner_reconstruction_scope
        - canonical_chunk_mapping
        - execute_direct_restore
        - prepare_direct_restore
        - assemble_distributed_checkpoint_from_repository
        - assemble_distributed_checkpoint_manifest
        - publish_process_checkpoint
        - restore_global_array_from_checkpoint
        - restore_meshing_checkpoint
        - snapshot_addressable_arrays

## Repositories and provenance

Repository commits, retained manifests, and build provenance are explicit
consumer boundaries. Runtime source, native provider, precision, topology, and
placement identities belong to evidence; a successful read does not qualify the
numerical method that produced the artifact.

Manifest decoding retains at most eight metadata-sized payloads per repository
instance. Payloads larger than 8 MiB are decoded without retention. Repeated reads
still revalidate durable bytes, the current artifact pointer, checksums, and
commit markers before reusing decoded records.

::: phydrax.lifecycle
    options:
      members:
        - ArtifactManifest
        - ArtifactRepository
        - BuildProvenance
        - LifecycleArchive
        - LifecycleRecord
        - POSIXArtifactRepository
        - RepositoryCorruptionError
        - RepositoryTransaction
        - RevisionLineage
        - RunRecord
        - RunStatus
        - create_build_provenance
        - generate_spdx_sbom
        - installed_packages
