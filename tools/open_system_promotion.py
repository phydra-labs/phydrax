#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json

from tools.open_system_campaign_matrix import run_campaign_matrix


def run_promotion(*, output_directory: str):
    """Run the sole artifact-driven open-system graduation path."""
    return run_campaign_matrix(output_directory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", required=True)
    arguments = parser.parse_args()
    summary = run_promotion(output_directory=arguments.output_directory)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
