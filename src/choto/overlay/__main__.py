from __future__ import annotations

import sys


def main() -> int:
    from choto.overlay.app import main as run_app

    return run_app()


if __name__ == "__main__":
    sys.exit(main())
