import math
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Literal

from .model import Robot

CategoryType = Literal["kinematic", "inertial", "collision", "visual"]

__all__ = ["Change", "ItemDiff", "RobotDiff", "compare_robots"]


@dataclass
class Change:
    """Change in a single value of any type (int, float, str, list, tuple, Object, None, etc.)

    Attributes:
        old_value: Value before the change (None if added)
        new_value: Value after the change (None if removed)
        status: Status of the change ('added', 'removed', or 'modified')
    """

    old_value: Any
    new_value: Any

    @property
    def status(self) -> str:
        if self.old_value is None:
            return "added"
        elif self.new_value is None:
            return "removed"
        else:
            return "modified"


@dataclass
class ItemDiff:
    """Diff of a Link or Joint object

    Attributes:
        name: Name of the link or joint
        status: Status of the diff ('added', 'removed', or 'modified')
        changes: Dict mapping property paths to Change objects
    """

    name: str
    status: str
    changes: dict[str, Change] = field(default_factory=dict)


@dataclass
class RobotDiff:
    """Diff between two Robot objects

    Attributes:
        old_name: Name of the old robot
        new_name: Name of the new robot
        link_diffs: Dict mapping link names to ItemDiff objects
        joint_diffs: Dict mapping joint names to ItemDiff objects
    """

    old_name: str
    new_name: str
    link_diffs: dict[str, ItemDiff] = field(default_factory=dict)
    joint_diffs: dict[str, ItemDiff] = field(default_factory=dict)


def compare_values(old: Any, new: Any, path: str = "", float_tol: float = 1e-6) -> dict[str, Change]:
    """Recursively compare two values and return a dict of changes

    Args:
        old: Old value
        new: New value
        path: Current path in the object hierarchy
        float_tol: Relative tolerance for float comparison, defaults to 1e-6

    Returns:
        Dict mapping property paths to Change objects, empty dict if no changes
    """
    changes = {}

    if old is None or new is None:
        changes[path] = Change(old, new)
        return changes

    # handle floats
    if isinstance(old, float) and isinstance(new, float):
        if not math.isclose(old, new, rel_tol=float_tol):
            changes[path] = Change(old, new)
        return changes

    if old == new:
        return changes

    # handle dataclasses
    if is_dataclass(old) and is_dataclass(new):
        # incompatible classes
        if type(old) is not type(new):
            changes[path] = Change(old, new)
            return changes

        # same classes, recurse
        for field_name, field_info in old.__dataclass_fields__.items():
            if not field_info.compare:
                continue

            old_val = getattr(old, field_name)
            new_val = getattr(new, field_name)
            new_path = f"{path}.{field_name}" if path else field_name
            changes.update(compare_values(old_val, new_val, new_path, float_tol))

        return changes

    # handle lists
    if isinstance(old, list) and isinstance(new, list):
        # different lengths
        if len(old) != len(new):
            changes[path] = Change(len(old), len(new))

        # recurse along min length
        for i in range(min(len(old), len(new))):
            element_path = f"{path}[{i}]"
            changes.update(compare_values(old[i], new[i], element_path, float_tol))

        return changes

    # handle tuples
    if isinstance(old, tuple) and isinstance(new, tuple):
        if len(old) != len(new):
            changes[path] = Change(old, new)
            return changes

        # detect change with tolerance
        if any(compare_values(old_elem, new_elem, "", float_tol) for old_elem, new_elem in zip(old, new, strict=False)):
            changes[path] = Change(old, new)

        return changes

    # handle primitives or incompatible types
    changes[path] = Change(old, new)
    return changes


def _filter_diff_by_categories(diff: RobotDiff, excluded_categories: set[CategoryType]) -> RobotDiff:
    """Filter a robot diff by excluded categories

    Args:
        diff: Original RobotDiff
        excluded_categories: Set of categories to exclude ('kinematic', 'inertial', 'collision', 'visual')

    Returns:
        New robot diff with filtered changes
    """

    filtered_link_diffs = {}
    for name, link_diff in diff.link_diffs.items():
        if link_diff.status == "modified":
            filtered_changes = {
                path: change
                for path, change in link_diff.changes.items()
                if not any(category in path for category in excluded_categories)
            }
            if filtered_changes:
                filtered_link_diffs[name] = ItemDiff(
                    name=link_diff.name, status=link_diff.status, changes=filtered_changes
                )
        elif "kinematic" not in excluded_categories:
            filtered_link_diffs[name] = link_diff

    filtered_joint_diffs = {} if "kinematic" in excluded_categories else diff.joint_diffs

    return RobotDiff(
        old_name=diff.old_name, new_name=diff.new_name, link_diffs=filtered_link_diffs, joint_diffs=filtered_joint_diffs
    )


def compare_robots(
    old_robot: Robot, new_robot: Robot, excluded_categories: set[CategoryType] | None = None, float_tol: float = 1e-6
) -> RobotDiff:
    """Compare two robots and return a desired diff

    Args:
        old_robot: The original robot
        new_robot: The updated robot
        excluded_categories: Set of categories to exclude ('kinematic', 'inertial', 'collision', 'visual')
        float_tol: Relative tolerance for float comparison, defaults to 1e-6

    Returns:
        Diff containing all desired differences
    """
    diff = RobotDiff(old_robot.name, new_robot.name)

    all_link_names = set(old_robot.links.keys()) | set(new_robot.links.keys())
    for name in sorted(all_link_names):
        old_link = old_robot.links.get(name)
        new_link = new_robot.links.get(name)

        if old_link is None:
            diff.link_diffs[name] = ItemDiff(name, "added")
        elif new_link is None:
            diff.link_diffs[name] = ItemDiff(name, "removed")
        else:
            changes = compare_values(old_link, new_link, float_tol=float_tol)
            if changes:
                diff.link_diffs[name] = ItemDiff(name, "modified", changes)

    all_joint_names = set(old_robot.joints.keys()) | set(new_robot.joints.keys())
    for name in sorted(all_joint_names):
        old_joint = old_robot.joints.get(name)
        new_joint = new_robot.joints.get(name)

        if old_joint is None:
            diff.joint_diffs[name] = ItemDiff(name, "added")
        elif new_joint is None:
            diff.joint_diffs[name] = ItemDiff(name, "removed")
        else:
            changes = compare_values(old_joint, new_joint, float_tol=float_tol)
            if changes:
                diff.joint_diffs[name] = ItemDiff(name, "modified", changes)

    return _filter_diff_by_categories(diff, excluded_categories) if excluded_categories else diff
