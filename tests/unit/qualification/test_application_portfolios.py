#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from phydrax.qualification import application_promotion_portfolios


def test_application_portfolios_are_nonempty_honest_and_content_addressed() -> None:
    portfolios = application_promotion_portfolios()

    assert {portfolio.name for portfolio in portfolios} == {
        "flow-multiphysics",
        "atomistic-chemistry-polymer",
        "battery-energy",
        "nuclear-reactor-tokamak",
        "earth-atmosphere-ocean-climate",
        "gr-astrophysics-dark-qft",
        "hep-provider-composition",
        "biology-medical",
        "finance",
        "robotics",
    }
    assert all(portfolio.capability_ids for portfolio in portfolios)
    assert all(portfolio.portfolio_id for portfolio in portfolios)
    assert all(not portfolio.ready_for_release_review for portfolio in portfolios)
    assert all("release-authorization-absent" in portfolio.blockers for portfolio in portfolios)
