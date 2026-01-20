from enum import Enum
from pathlib import Path
from typing import Annotated

import tyro
from tyro.extras import SubcommandApp

from .diff import CategoryType, compare_robots
from .formatters import CategoryFormatter, DetailedFormatter, GitFormatter, StatusFormatter, TreeFormatter
from .parsers import IsaacUSDParser, MJCFParser, SDFParser, URDFParser


class DiffFormat(Enum):
    status = StatusFormatter
    git = GitFormatter
    category = CategoryFormatter


class PrintFormat(Enum):
    tree = TreeFormatter
    detailed = DetailedFormatter


PARSER_MAP = {
    ".urdf": URDFParser,
    ".sdf": SDFParser,
    ".xml": MJCFParser,
    ".usd": IsaacUSDParser,
    ".usda": IsaacUSDParser,
    ".usdc": IsaacUSDParser,
}

app = SubcommandApp()


def _get_parser(path: Path):
    """Get parser based on file extension

    Args:
        path: Path to robot model file

    Returns:
        URDFParser, SDFParser, MJCFParser, or IsaacUSDParser instance

    Raises:
        ValueError: If file extension is not supported
    """
    parser_cls = PARSER_MAP.get(path.suffix)

    return parser_cls(path) if parser_cls else ValueError(f"Unsupported file extension: {path.suffix}")


@app.command
def diff(
    old: Path,
    new: Path,
    /,
    format: DiffFormat = DiffFormat.status,
    exclude: Annotated[
        set[CategoryType] | None,
        tyro.conf.arg(metavar="[kinematic inertial collision visual]"),
    ] = None,
    float_tol: float = 1e-6,
) -> None:
    """Generate a human-readable diff between two robot model files.

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


@app.command(name="print")
def print_robot(
    robot: Path,
    /,
    format: PrintFormat = PrintFormat.tree,
) -> None:
    """Print information about a robot model file.

    Args:
        robot: Path to the robot model file
        format: Output format of the information
    """
    if not robot.exists():
        raise FileNotFoundError(f"File not found: {robot}")

    robot = _get_parser(robot).parse()

    print(format.value(robot).format())


def tyro_cli():
    app.cli(
        prog="robot-diff",
        description=(
            "A diffing tool for robot models. Supports URDF (.urdf), SDF (.sdf), MJCF (.xml), and USD (.usd) files."
        ),
    )


if __name__ == "__main__":
    tyro_cli()
