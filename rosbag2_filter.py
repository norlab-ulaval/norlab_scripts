#!/usr/bin/env python3.10

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING
from rosbags.rosbag1 import Reader

if TYPE_CHECKING:
    from typing import Callable, Tuple


def pathtype(exists: bool = True) -> Callable[[str], Path]:
    """Path argument for argparse.

    Args:
        exists: Path should exists in filesystem.

    Returns:
        Argparse type function.

    """

    def topath(pathname: str) -> Path:
        path = Path(pathname)
        if exists != path.exists():
            raise argparse.ArgumentTypeError(
                f"{path} should {'exist' if exists else 'not exist'}."
            )
        return path

    return topath


def parse_arguments():
    """Parse cli arguments"""
    parser = argparse.ArgumentParser(description="Filter topics from a ROS2 bag")
    parser.add_argument(
        "inbag",
        type=pathtype(),
        help="source path to read rosbag2 from",
    )
    parser.add_argument(
        "outbag",
        type=pathtype(exists=False),
        help="destination path for filtered rosbag",
    )
    parser.add_argument(
        "--topics",
        "--top",
        type=str,
        metavar="/topics",
        nargs="+",
        dest="topics",
        help="topics to copy in outbag",
        default=[],
    )
    args = parser.parse_args()
    return args


class TopicNotFoundError(Exception):
    """Topic is not in rosbag"""


def check_rosbag_topics(src_path: Path, topics: Tuple[str]) -> None:
    """Check that all topics are in src"""
    with Reader(src_path) as src_bag:
        src_topics = tuple(src_bag.topics.keys())

    unavailable_topics = tuple(top for top in topics if top not in src_topics)
    if len(unavailable_topics) > 0:
        raise TopicNotFoundError(f"Topics {unavailable_topics} are not in the rosbag")


def main() -> None:
    """Main function"""
    args = parse_arguments()
    check_rosbag_topics(args.inbag, args.topics)

    print(args.inbag, args.outbag, args.topics)


if __name__ == "__main__":
    main()
