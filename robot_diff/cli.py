from enum import Enum
from pathlib import Path
from typing import Annotated

import tyro

from .diff import CategoryType, compare_robots
from .formatters import CategoryFormatter, GitFormatter, StatusFormatter
from .parsers import IsaacUSDParser, MJCFParser, SDFParser, URDFParser


class Format(Enum):
    status = StatusFormatter
    git = GitFormatter
    category = CategoryFormatter


def _get_parser(path: Path):
    """Get parser based on file extension

    Args:
        path: Path to robot model file

    Returns:
        URDFParser, SDFParser, MJCFParser, or IsaacUSDParser instance

    Raises:
        ValueError: If file extension is not supported
    """
    if path.suffix == ".urdf":
        return URDFParser(path)
    if path.suffix == ".sdf":
        return SDFParser(path)
    elif path.suffix == ".xml":
        return MJCFParser(path)
    elif ".usd" in path.suffix:
        return IsaacUSDParser(path)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")


def main(
    old: Path,
    new: Path,
    /,
    format: Format = Format.status,
    exclude: Annotated[
        set[CategoryType] | None,
        tyro.conf.arg(metavar="[kinematic inertial collision visual]"),
    ] = None,
    float_tol: float = 1e-6,
) -> None:
    """Generate a human-readable diff between two robot model files.
    Supports URDF (.urdf), SDF (.sdf), MJCF (.xml), and USD (.usd) files.

    Args:
        old: Path to the original robot model file
        new: Path to the updated robot model file
        format: Output format of the diff
        exclude: Categories to exclude from the diff
        float_tol: Relative tolerance for float comparison

    """
    if not old.exists():
        raise FileNotFoundError(f"File not found: {old}")
    if not new.exists():
        raise FileNotFoundError(f"File not found: {new}")

    old_robot = _get_parser(old).parse()
    new_robot = _get_parser(new).parse()

    diff = compare_robots(old_robot, new_robot, exclude, float_tol)

    print(format.value(diff).format())


def tyro_cli():
    tyro.cli(main, prog="robot-diff")


if __name__ == "__main__":
    tyro_cli()
