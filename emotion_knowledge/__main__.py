"""Command line entry point for :mod:`emotion_knowledge`.

Provides a ``--reset-db`` flag that clears the persistent Chroma database
before delegating to :func:`emotion_knowledge.main`.

Example
-------
>>> python -m emotion_knowledge --reset-db <audiofile>
"""

import argparse
import sys

from . import main as pkg_main


def run() -> None:
    """Run the command line interface.

    ``--reset-db`` clears the saved ChromaDB collections before invoking the
    standard :func:`emotion_knowledge.main` workflow.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Reset the ChromaDB database before processing",
    )
    args, remaining = parser.parse_known_args()

    if args.reset_db:
        from .segment_saver import SegmentSaver

        SegmentSaver().reset_db()

    sys.argv = [sys.argv[0]] + remaining
    pkg_main()


if __name__ == "__main__":
    run()
