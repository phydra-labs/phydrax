#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from phydrax.qualification import (
    builtin_source_absorption_ledger,
    validate_source_coverage,
)


def main():
    errors = validate_source_coverage(builtin_source_absorption_ledger())
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
