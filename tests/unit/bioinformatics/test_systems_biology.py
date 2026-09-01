#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import diffrax as dfx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.bioinformatics.interchange._sbml import (
    lower_sbml_document,
    SBMLCompartmentAST,
    SBMLDocumentAST,
    SBMLEventAST,
    SBMLModelAST,
    SBMLPackageDeclaration,
    SBMLReactionAST,
    SBMLRuleAST,
    SBMLSemanticError,
    SBMLSemanticStatus,
    SBMLSpeciesAST,
    SBMLSpeciesReferenceAST,
    validate_sbml_document,
)
from phydrax.bioinformatics.systems._flux import (
    flux_balance_analysis,
    flux_variability_analysis,
    FluxStatus,
)
from phydrax.bioinformatics.systems._identifiability import (
    global_candidate_identifiability,
    GlobalIdentifiabilityStatus,
    local_identifiability,
    LocalIdentifiabilityStatus,
)
from phydrax.bioinformatics.systems._kinetics import (
    KineticReaction,
    KineticReactionSystem,
    KineticStatus,
    RateLawKind,
    simulate_kinetics,
)
from phydrax.bioinformatics.systems._network import (
    ChemicalComposition,
    Compartment,
    GeneReactionRule,
    Reaction,
    Species,
    StoichiometricNetwork,
    SUBSTANCE_FLUX,
    VOLUME,
)
from phydrax.bioinformatics.systems._regulatory import (
    DiscreteRegulatoryNetwork,
    RegulatoryRule,
    RegulatoryStatus,
)
from phydrax.bioinformatics.systems._stoichiometry import (
    audit_stoichiometry,
    conservation_analysis,
    ConservationStatus,
    StoichiometryStatus,
)


def _balanced_conversion_network(*, flux_unit=SUBSTANCE_FLUX):
    compartment = Compartment("cell", volume=1.0)
    hydrogen = ChemicalComposition({"H": 2}, charge=0)
    species = (
        Species("A", "cell", initial_amount=1.0, composition=hydrogen),
        Species("B", "cell", initial_amount=0.0, composition=hydrogen),
    )
    reaction = Reaction(
        "conversion",
        ("A", "B"),
        jnp.asarray([-1.0, 1.0]),
        lower_bound=-10.0,
        upper_bound=10.0,
        flux_unit=flux_unit,
    )
    return StoichiometricNetwork((compartment,), species, (reaction,))


def _flux_network(*, source_bounds=(0.0, 10.0), sink_bounds=(0.0, 10.0)):
    compartment = Compartment("cell")
    species = (Species("A", "cell", composition=ChemicalComposition({"C": 1})),)
    reactions = (
        Reaction(
            "source",
            ("A",),
            jnp.asarray([1.0]),
            lower_bound=source_bounds[0],
            upper_bound=source_bounds[1],
            exchange=True,
        ),
        Reaction(
            "sink_a",
            ("A",),
            jnp.asarray([-1.0]),
            lower_bound=sink_bounds[0],
            upper_bound=sink_bounds[1],
            objective_coefficient=1.0,
            exchange=True,
        ),
        Reaction(
            "sink_b",
            ("A",),
            jnp.asarray([-1.0]),
            lower_bound=sink_bounds[0],
            upper_bound=sink_bounds[1],
            objective_coefficient=1.0,
            exchange=True,
        ),
    )
    return StoichiometricNetwork((compartment,), species, reactions)


def test_mass_charge_unit_consistency_and_open_exchange_semantics():
    network = _balanced_conversion_network()
    audit = audit_stoichiometry(network)
    assert bool(audit.successful)
    np.testing.assert_allclose(audit.evidence.element_residuals, 0.0)
    np.testing.assert_allclose(audit.evidence.charge_residuals, 0.0)
    assert bool(jnp.all(audit.evidence.unit_consistent))
    assert bool(network.reactions[0].reversible)

    charged_species = (
        Species(
            "cation",
            "cell",
            composition=ChemicalComposition({"H": 1}, charge=1),
        ),
        Species(
            "neutral",
            "cell",
            composition=ChemicalComposition({"H": 1}, charge=0),
        ),
    )
    charged = StoichiometricNetwork(
        (Compartment("cell"),),
        charged_species,
        (
            Reaction(
                "charge-loss",
                ("cation", "neutral"),
                jnp.asarray([-1.0, 1.0]),
            ),
        ),
    )
    charged_audit = audit_stoichiometry(charged)
    assert int(charged_audit.status) == int(StoichiometryStatus.CHARGE_IMBALANCE)

    wrong_units = audit_stoichiometry(_balanced_conversion_network(flux_unit=VOLUME))
    assert int(wrong_units.status) == int(StoichiometryStatus.UNIT_MISMATCH)

    exchange = StoichiometricNetwork(
        (Compartment("cell"),),
        (Species("A", "cell", composition=ChemicalComposition({"C": 1})),),
        (
            Reaction(
                "EX_A",
                ("A",),
                jnp.asarray([1.0]),
                lower_bound=-5.0,
                upper_bound=5.0,
                exchange=True,
            ),
        ),
    )
    exchange_audit = audit_stoichiometry(exchange)
    assert bool(exchange_audit.successful)
    assert not bool(exchange_audit.evidence.balance_applicable[0])
    assert bool(exchange.reactions[0].reversible)


def test_complete_conservation_basis_has_small_residual():
    result = conservation_analysis(_balanced_conversion_network())
    assert bool(result.valid)
    assert int(result.status) == int(ConservationStatus.SUCCESS)
    assert result.num_conservation_laws == 1
    np.testing.assert_allclose(result.evidence.left_nullspace_residual, 0.0, atol=1.0e-10)
    law = np.asarray(result.conservation_laws[0])
    np.testing.assert_allclose(abs(law[0]), abs(law[1]), atol=1.0e-8)
    assert result.evidence.complete_basis
    assert not result.evidence.exact


def test_fba_and_fva_report_degenerate_alternate_optimal_face():
    network = _flux_network()
    result = flux_balance_analysis(network)
    assert bool(result.successful)
    np.testing.assert_allclose(result.objective_value, 10.0, atol=1.0e-5)
    np.testing.assert_allclose(result.evidence.mass_balance_residual, 0.0, atol=1.0e-6)
    assert bool(result.evidence.alternate.available)
    assert bool(result.evidence.alternate.alternate_optimum)
    assert result.evidence.native_result.equality_dual.shape == (1,)
    assert float(result.evidence.native_result.kkt_residual_norm) < 1.0e-4

    variability = flux_variability_analysis(network)
    assert bool(variability.valid)
    np.testing.assert_allclose(variability.minimum_fluxes[1:], 0.0, atol=1.0e-5)
    np.testing.assert_allclose(variability.maximum_fluxes[1:], 10.0, atol=1.0e-5)
    assert variability.evidence.complete
    assert len(variability.evidence.minimum_results) == network.num_reactions


def test_infeasible_and_unbounded_flux_keep_native_certificates():
    infeasible = _flux_network(source_bounds=(1.0, 1.0), sink_bounds=(0.0, 0.0))
    infeasible_result = flux_balance_analysis(infeasible)
    assert not bool(infeasible_result.valid)
    assert int(infeasible_result.status) == int(FluxStatus.INFEASIBLE)
    assert bool(infeasible_result.evidence.native_result.certificate.dual_ray_valid)

    unbounded = _flux_network(source_bounds=(0.0, jnp.inf), sink_bounds=(0.0, jnp.inf))
    unbounded_result = flux_balance_analysis(unbounded)
    assert not bool(unbounded_result.valid)
    assert int(unbounded_result.status) == int(FluxStatus.UNBOUNDED)
    assert bool(unbounded_result.evidence.native_result.certificate.primal_ray_valid)
    assert bool(jnp.isposinf(unbounded_result.objective_value))


def test_stiff_positive_kinetics_use_native_dynamics_and_conserve_amount():
    compartment = Compartment("cell", volume=1.0)
    composition = ChemicalComposition({"C": 1})
    species = (
        Species("A", "cell", initial_amount=1.0, composition=composition),
        Species("B", "cell", composition=composition),
        Species("C", "cell", composition=composition),
    )
    reactions = (
        Reaction("fast", ("A", "B"), jnp.asarray([-1.0, 1.0])),
        Reaction("slow", ("B", "C"), jnp.asarray([-1.0, 1.0])),
    )
    network = StoichiometricNetwork((compartment,), species, reactions)
    kinetics = KineticReactionSystem(
        network,
        (
            KineticReaction(
                0,
                jnp.asarray([0]),
                jnp.asarray([1.0]),
                jnp.asarray([1000.0]),
                rate_law=RateLawKind.MASS_ACTION,
                rate_unit=SUBSTANCE_FLUX,
                kinetic_id="fast-law",
            ),
            KineticReaction(
                1,
                jnp.asarray([1]),
                jnp.asarray([1.0]),
                jnp.asarray([1.0]),
                rate_law=RateLawKind.MASS_ACTION,
                rate_unit=SUBSTANCE_FLUX,
                kinetic_id="slow-law",
            ),
        ),
    )
    result = simulate_kinetics(
        kinetics,
        jnp.asarray([1.0, 0.0, 0.0]),
        jnp.asarray([0.0, 1.0e-3, 1.0e-2]),
        solver=dfx.Kvaerno5(),
        stepsize_controller=dfx.PIDController(rtol=1.0e-7, atol=1.0e-9),
        rtol=1.0e-7,
        atol=1.0e-9,
    )
    assert bool(result.valid)
    assert int(result.status) == int(KineticStatus.SUCCESS)
    assert bool(jnp.all(result.concentrations >= -1.0e-10))
    np.testing.assert_allclose(jnp.sum(result.concentrations, axis=1), 1.0, atol=2.0e-6)
    assert float(jnp.max(jnp.abs(result.evidence.conserved_pool_drift))) < 2.0e-6
    assert "DiffraxEvolution" in result.evidence.backend


def _supported_sbml_document(*, rules=(), events=()):
    return SBMLDocumentAST(
        3,
        2,
        SBMLModelAST(
            "toy",
            compartments=(SBMLCompartmentAST("cell"),),
            species=(
                SBMLSpeciesAST("A", "cell", elements=(("C", 1),)),
                SBMLSpeciesAST("B", "cell", elements=(("C", 1),)),
            ),
            reactions=(
                SBMLReactionAST(
                    "R",
                    reactants=(SBMLSpeciesReferenceAST("A"),),
                    products=(SBMLSpeciesReferenceAST("B"),),
                    reversible=True,
                    lower_bound=-10.0,
                    upper_bound=10.0,
                    objective_coefficient=1.0,
                    gpr_clauses=(("gene_a", "gene_b"), ("gene_c",)),
                ),
            ),
            rules=rules,
            events=events,
        ),
        packages=(SBMLPackageDeclaration("fbc", 2),),
        source_id="toy.xml",
    )


def test_sbml_supported_matrix_preserves_fbc_units_and_gpr():
    document = _supported_sbml_document()
    validation = validate_sbml_document(document)
    assert bool(validation.valid)
    assert validation.evidence.profile_id == "SBML-L3V2"
    assert validation.evidence.checked_before_lowering
    result = lower_sbml_document(document)
    assert bool(result.valid)
    assert result.evidence.lossless
    reaction = result.network.reactions[0]
    assert bool(reaction.reversible)
    assert reaction.gene_reaction_rule == GeneReactionRule(
        (("gene_a", "gene_b"), ("gene_c",))
    )
    assert reaction.gene_reaction_rule.evaluate(("gene_c",))
    np.testing.assert_allclose(result.evidence.reaction_scales, 1.0)


def test_sbml_rules_and_events_are_rejected_before_lowering_with_paths():
    rule_document = _supported_sbml_document(
        rules=(SBMLRuleAST("rule", "assignment", "A", "B"),)
    )
    validation = validate_sbml_document(rule_document)
    assert not bool(validation.valid)
    assert int(validation.status) == int(SBMLSemanticStatus.UNSUPPORTED_RULE)
    assert validation.evidence.rejected_paths == ("model.rules[0]",)
    with pytest.raises(SBMLSemanticError) as error:
        lower_sbml_document(rule_document)
    assert error.value.validation.evidence.checked_before_lowering

    event_document = _supported_sbml_document(
        events=(SBMLEventAST("event", "A > 0", (("B", "1"),)),)
    )
    validation = validate_sbml_document(event_document)
    assert int(validation.status) == int(SBMLSemanticStatus.UNSUPPORTED_EVENT)
    rejected = lower_sbml_document(event_document, reject_unsupported=False)
    assert not bool(rejected.valid)
    assert rejected.network is None
    assert not rejected.evidence.lossless


def test_regulatory_cycles_are_exact_synchronous_pgm_relations():
    network = DiscreteRegulatoryNetwork(
        ("a", "b"),
        (
            RegulatoryRule(0, jnp.asarray([1]), jnp.asarray([0, 1]), rule_id="b-to-a"),
            RegulatoryRule(1, jnp.asarray([0]), jnp.asarray([0, 1]), rule_id="a-to-b"),
        ),
    )
    assert network.has_cycles
    transition = network.step(jnp.asarray([1, 0]))
    assert bool(transition.valid)
    assert int(transition.status) == int(RegulatoryStatus.SUCCESS)
    np.testing.assert_array_equal(transition.state, [0, 1])
    lowering = network.to_factor_graph()
    assert bool(lowering.valid)
    assert lowering.evidence.exact
    assert lowering.evidence.complete
    assert bool(lowering.evidence.has_regulatory_cycles)
    assert len(lowering.factor_graph.factor_groups) == 2


def test_local_and_global_identifiability_claims_remain_distinct():
    local = local_identifiability(
        lambda parameters: jnp.stack(
            (parameters[0] + parameters[1], parameters[0] - parameters[1])
        ),
        jnp.asarray([1.0, 2.0]),
    )
    assert bool(local.valid)
    assert bool(local.locally_identifiable)
    assert int(local.status) == int(LocalIdentifiabilityStatus.LOCALLY_IDENTIFIABLE)
    assert local.evidence.rank_claim.startswith("local differential rank")

    underdetermined = local_identifiability(
        lambda parameters: parameters[0] + parameters[1],
        jnp.asarray([1.0, 2.0]),
    )
    assert not bool(underdetermined.locally_identifiable)
    assert int(underdetermined.status) == int(LocalIdentifiabilityStatus.UNDERDETERMINED)

    candidates = jnp.asarray([[-1.0], [1.0]])
    collision = global_candidate_identifiability(
        candidates, candidates**2, exhaustive=False
    )
    assert bool(collision.conclusive)
    assert not bool(collision.globally_identifiable)
    assert int(collision.status) == int(GlobalIdentifiabilityStatus.NOT_IDENTIFIABLE)
    assert int(collision.evidence.collision_count) == 1

    nonexhaustive = global_candidate_identifiability(
        jnp.asarray([[0.0], [1.0]]),
        jnp.asarray([[0.0], [1.0]]),
        exhaustive=False,
    )
    assert not bool(nonexhaustive.conclusive)
    assert int(nonexhaustive.status) == int(
        GlobalIdentifiabilityStatus.INCONCLUSIVE_NONEXHAUSTIVE
    )

    exhaustive = global_candidate_identifiability(
        jnp.asarray([[0.0], [1.0]]),
        jnp.asarray([[0.0], [1.0]]),
        exhaustive=True,
    )
    assert bool(exhaustive.conclusive)
    assert bool(exhaustive.globally_identifiable)
