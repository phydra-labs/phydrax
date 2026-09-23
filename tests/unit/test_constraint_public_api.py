import phydrax as phx


LEGACY_TERMS = {
    "AbstractDomainSamplingTerm",
    "AbstractWeightedTerm",
    "ContinuousDirichletBoundaryTerm",
    "ContinuousFokkerPlanckTerm",
    "ContinuousInitialTerm",
    "ContinuousIntegralInteriorTerm",
    "ContinuousKolmogorovTerm",
    "ContinuousNeumannBoundaryTerm",
    "ContinuousODETerm",
    "ContinuousPointwiseInteriorTerm",
    "ContinuousRobinBoundaryTerm",
    "DiscreteDirichletBoundaryTerm",
    "DiscreteInitialTerm",
    "DiscreteInteriorDataTerm",
    "DiscreteNeumannBoundaryTerm",
    "DiscreteODETerm",
    "IntegralEqualityTerm",
    "PointSetTerm",
    "ResidualTerm",
}


def test_constraint_packages_are_the_only_root_constraint_surfaces():
    assert {"conditions", "terms", "enforcement"} <= set(phx.__all__)
    assert phx.conditions is not None
    assert phx.terms is not None
    assert phx.enforcement is not None
    assert "constraints" not in phx.__all__
    assert not hasattr(phx, "constraints")
    assert "objectives" not in phx.__all__
    assert not hasattr(phx, "objectives")


def test_conditions_expose_canonical_names_and_grouped_catalogs():
    required = {
        "Dirichlet",
        "Initial",
        "Moment",
        "Neumann",
        "Observation",
        "Residual",
        "Robin",
        "cfd",
        "conservation",
        "electromagnetics",
        "solids",
        "stochastic",
        "thermal",
    }
    legacy = {"MomentCondition", "ObservationCondition", "ResidualCondition"}

    assert required <= set(phx.conditions.__all__)
    assert all(getattr(phx.conditions, name) is not None for name in required)
    assert legacy.isdisjoint(phx.conditions.__all__)
    assert all(not hasattr(phx.conditions, name) for name in legacy)


def test_terms_public_exports_resolve_without_duplicates():
    exported = tuple(phx.terms.__all__)
    assert len(exported) == len(set(exported))
    assert all(getattr(phx.terms, name) is not None for name in exported)


def test_terms_do_not_expose_legacy_or_duplicate_physics_catalogs():
    assert LEGACY_TERMS.isdisjoint(phx.terms.__all__)
    assert all(not hasattr(phx.terms, name) for name in LEGACY_TERMS)
    assert not any(
        name.startswith(("Continuous", "Discrete")) for name in phx.terms.__all__
    )


def test_collocation_policy_attachment_is_not_a_parallel_term_surface():
    assert "with_collocation_policy" not in phx.sampling.collocation.__all__
    assert not hasattr(phx.sampling.collocation, "with_collocation_policy")
