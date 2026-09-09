# Qualification criteria and campaign causality

`phydrax.qualification` owns provider-neutral, content-addressed qualification
criteria and campaign boundary records. Applications supply their native campaign
specification, resolved-run, support-tuple, and raw-artifact identities; the
qualification layer does not introduce an application-specific campaign runner or
observation payload.

A `QualificationCriterion` fixes the exact support tuple, metric and unit,
comparison and target, aggregation, uncertainty treatment, applicability,
approval identity, issuance time, and optional validity deadline before execution.
Its content is canonical: there is no schema-version coordinate to reinterpret.
Changing any field creates a different `criterion_id`.

`CampaignStartRecord` binds that criterion to one campaign specification, resolved
run specification, and support tuple at the native execution boundary.
`CampaignObservationRecord` binds the corresponding raw artifacts to that exact
start. Both records are immutable and content-addressed. Their `from_record`
constructors recompute and verify serialized identities instead of trusting a
caller-supplied digest.

All persisted qualification times use UTC Unix-epoch nanoseconds. This applies
to criterion issuance and expiry, campaign scheduling, campaign start and
observation records, and qualification evidence issuance and expiry. A
campaign runner may use a separate monotonic clock to reject in-process clock
reversal or invalid elapsed ordering, but monotonic ticks are never serialized
or compared across processes, hosts, or boots.

`QualificationEvidence` retains campaign linkage through
`campaign_start_record_ids` and `campaign_observation_record_ids`. The fields are
empty for non-campaign evidence and may contain several records when one evidence
record covers a larger campaign. Qualification predicates can require either ID
without repurposing scientific `subject_ids`.

Call `validate_qualification_causality` before admitting campaign evidence. It
content-verifies every live record, requires exact criterion, support, campaign
specification, resolved-run, start, observation, and raw-artifact linkage, and
enforces

```
criterion.issued_at < start.started_at <= observation.observed_at <= evidence.issued_at
```

The criterion must be valid at campaign start. The validator preserves the
reported evidence outcome: it verifies identity and causality but does not turn a
failed or inconclusive result into a pass. Evidence currentness and supersession
remain the responsibility of `QualificationMatrix` and the release trust path.

## Scientific data roles versus governed campaign causality

`ScientificCampaign` freezes data-role membership, independent units,
preparations, batches, ancestry, preprocessing cases, and content-addressed
scientific metric criteria. It is the application-facing split contract; it does
not replace `CampaignStartRecord` or `CampaignObservationRecord`.

`ScientificMetricCriterion` describes the exact metric bounds consumed by a
`ScientificClaimProfile`. Before release, each bound must also be represented by
an approved `QualificationCriterion` for the same support tuple. A two-sided
scientific interval requires separate approved lower and upper criteria. The
approved criteria, resolved run, start, observations, and raw artifacts are then
validated with `validate_qualification_causality`.

Candidate or analytical workflows intentionally emit empty
`campaign_start_record_ids` and `campaign_observation_record_ids`. Even if their
internal numerical stages pass, that evidence is not causally admissible for
release until the approved pre-start records exist. This preserves the
distinction between a leakage-controlled retrospective analysis and a registered
prospective qualification campaign.

::: phydrax.qualification
    options:
      show_root_heading: true
      members:
        - ScientificCase
        - CampaignRole
        - ScientificCampaign
        - ScientificMetricCriterion
        - ScientificClaimProfile


::: phydrax.qualification.QualificationCriterion

::: phydrax.qualification.CampaignStartRecord

::: phydrax.qualification.CampaignObservationRecord

::: phydrax.qualification.validate_qualification_causality

::: phydrax.qualification.QualificationEvidence
