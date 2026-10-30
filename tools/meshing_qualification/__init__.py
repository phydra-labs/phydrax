"""Qualification CLI for executable native meshing workflows and explicit comparisons."""

import argparse

from ._registry import NATIVE_SCENARIOS, OPTIONAL_SCENARIOS

SCENARIOS = (*NATIVE_SCENARIOS, *OPTIONAL_SCENARIOS, "native-corpus")


def run(scenarios: tuple[str, ...], arguments: argparse.Namespace, /) -> dict[str, object]:
    from .cli import run as run_cli

    return run_cli(scenarios, arguments)

__all__ = ["NATIVE_SCENARIOS", "OPTIONAL_SCENARIOS", "SCENARIOS", "run"]
