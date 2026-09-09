# Domain decomposition

Subdomain covers separate fixed geometry and trace topology from functional training.
Public symbols are also re-exported from `phydrax.domain`.

::: phydrax.domain.decomposition.SubdomainPatch
    options:
        members:
            - __init__
            - lift
            - restrict

---

::: phydrax.domain.decomposition.PairedSupport
    options:
        members:
            - __init__
            - trace

---

::: phydrax.domain.decomposition.SubdomainCover
    options:
        members:
            - __init__
            - patch
            - pairing
            - structural_evidence
            - audit

---

::: phydrax.domain.decomposition.SubdomainCoverEvidence

---

::: phydrax.domain.decomposition.CartesianCoverPlan
    options:
        members:
            - __init__
            - build

---

::: phydrax.domain.decomposition.cartesian_subdomain_cover

---

::: phydrax.domain.decomposition.normalized_patch_coordinate

---

::: phydrax.domain.decomposition.LocalFieldFamily
    options:
        members:
            - __init__
            - field
            - field_name
            - solver_functions
            - lifted_fields

---

::: phydrax.domain.decomposition.partition_of_unity_field

---

::: phydrax.domain.decomposition.partition_of_unity_family

---

::: phydrax.domain.decomposition.BrokenField
    options:
        members:
            - local
            - trace
            - as_domain_function

---

::: phydrax.domain.decomposition.broken_field

---

::: phydrax.domain.decomposition.AxisPartition

---

::: phydrax.domain.decomposition.BoxPartition

---

::: phydrax.domain.decomposition.PairedSupportEvidence

---

::: phydrax.domain.decomposition.LocalFieldRef

---

::: phydrax.domain.decomposition.IntegrationOwnership

---

::: phydrax.domain.decomposition.IntegrationOwnershipEvidence

---

::: phydrax.domain.decomposition.cover_integration_ownership

---

::: phydrax.domain.decomposition.PreparedFieldRouting

---

::: phydrax.domain.decomposition.prepare_field_routing

---

::: phydrax.domain.decomposition.MappedCoverValidationPlan

---

::: phydrax.domain.decomposition.MappedCoverEvidence

---

::: phydrax.domain.decomposition.validate_mapped_cover

---

::: phydrax.domain.decomposition.SubdomainLevel

---

::: phydrax.domain.decomposition.SubdomainHierarchy

---

::: phydrax.domain.decomposition.CoverAdapterEvidence

---

::: phydrax.domain.decomposition.geometry_subdomain_patch

---

::: phydrax.domain.decomposition.validate_cell_partition_cover

---

::: phydrax.domain.decomposition.validate_atlas_cover_adapter
