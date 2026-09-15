# Black-hole artifact interchange

Host-only admission of checksum-pinned local field, image, visibility, waveform, and
inert numeric-model bytes with explicit producer/version/model/coverage, semantic
schema, resource limits, content-bound rights, requested-use policy and conversion
loss. `phydrax.service.ArtifactRights` is a separate delivery authorization record; it
does not replace `BlackHoleArtifactRights`. See the
[source and rights ledger](../black_hole_sources.md).

::: phydrax.interchange
    options:
      show_root_heading: true
      members:
        - BlackHoleArtifactKind
        - BlackHoleArtifactRights
        - BlackHoleArtifactUsePolicy
        - BlackHoleArtifactSchema
        - NeutralBlackHoleArtifact
        - map_black_hole_artifact
        - map_field_artifact
        - map_image_artifact
        - map_visibility_artifact
        - map_waveform_artifact
        - map_numeric_model_artifact
