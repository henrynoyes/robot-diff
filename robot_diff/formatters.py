from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable
from dataclasses import is_dataclass
from enum import Enum
from itertools import chain
from typing import Any

from .diff import Change, ItemDiff, RobotDiff
from .model import (
    Box,
    Collision,
    Cylinder,
    Geometry,
    Joint,
    Link,
    Material,
    Mesh,
    Robot,
    Sphere,
    Visual,
)

__all__ = ["StatusFormatter", "GitFormatter", "CategoryFormatter", "TreeFormatter", "DetailedFormatter"]


class Color(Enum):
    """ANSI color codes"""

    RED = "\033[31m"
    GREEN = "\033[32m"
    RESET = "\033[0m"

    def apply(self, text: str) -> str:
        """Apply ANSI color to text

        Args:
            text: Text to colorize

        Returns:
            Colorized string
        """
        return f"{self.value}{text}{Color.RESET.value}"


class StringFormatter(ABC):
    """Base class for all string formatters"""

    INDENT = "  "
    BULLET = "• "

    @abstractmethod
    def format(self) -> str:
        """Format the content"""
        pass

    def _format_value(self, value: Any, color: Color | None = None) -> str:
        """Format a single value for display

        Args:
            value: Value to format
            color: Optional color to apply

        Returns:
            Formatted string of the value
        """
        if value is None:
            formatted = "None"
        elif isinstance(value, str):
            formatted = value
        elif isinstance(value, (tuple, list)):
            formatted_items = ", ".join(str(v) for v in value)
            formatted = f"({formatted_items})"
        elif is_dataclass(value):
            formatted = f"{type(value).__name__}"
        else:
            formatted = str(value)

        return color.apply(formatted) if color else formatted

    def _wrap_bars(self, text: str, num_bars: int = 3) -> str:
        """Wrap text in horizontal bars (━)

        Args:
            text: Text to wrap
            num_bars: Number of bars on each side, defaults to 3

        Returns:
            Formatted string
        """
        bars = "━" * num_bars
        return f"{bars} {text} {bars}"


class RobotDiffFormatter(StringFormatter):
    """Base class for RobotDiff formatters

    Attributes:
        diff: RobotDiff object to format
    """

    def __init__(self, diff: RobotDiff):
        self.diff = diff

    def _format_tuple_with_diff(self, old_tuple: tuple, new_tuple: tuple) -> tuple[str, str]:
        """Format old and new tuples with colored diffs

        Args:
            old_tuple: Old tuple
            new_tuple: New tuple

        Returns:
            Tuple of (formatted_old, formatted_new) strings
        """
        if len(old_tuple) != len(new_tuple):
            return (self._format_value(old_tuple, Color.RED), self._format_value(new_tuple, Color.GREEN))

        old_parts, new_parts = [], []
        for old_val, new_val in zip(old_tuple, new_tuple, strict=False):
            if old_val == new_val:
                old_parts.append(str(old_val))
                new_parts.append(str(new_val))
            else:
                old_parts.append(Color.RED.apply(str(old_val)))
                new_parts.append(Color.GREEN.apply(str(new_val)))

        return f"({', '.join(old_parts)})", f"({', '.join(new_parts)})"

    def _filter_itemdiffs_by_status(self, item_diffs: Iterable[ItemDiff], status: str) -> list[ItemDiff]:
        """Filter item diffs by status

        Args:
            item_diffs: Iterable of ItemDiff objects
            status: Status to filter by ('added', 'removed', or 'modified')

        Returns:
            Sorted list of ItemDiff objects with matching status
        """
        return sorted((item_diff for item_diff in item_diffs if item_diff.status == status), key=lambda x: x.name)

    def _count_itemdiffs_by_status(self, item_diffs: Iterable[ItemDiff]) -> tuple[int, int, int]:
        """Count item diffs by status

        Args:
            item_diffs: Iterable of ItemDiff objects

        Returns:
            Tuple of (removed_count, added_count, modified_count)
        """
        counts = Counter(item_diff.status for item_diff in item_diffs)
        return counts["removed"], counts["added"], counts["modified"]


class StatusFormatter(RobotDiffFormatter):
    """Formatter that groups item diffs by status (removed, added, and modified)"""

    def format(self) -> str:
        """Format the diff

        Returns:
            Formatted string
        """
        lines = [self._wrap_bars("NAME"), ""]
        if self.diff.old_name != self.diff.new_name:
            lines.append(f"{Color.RED.apply(self.diff.old_name)} → {Color.GREEN.apply(self.diff.new_name)}")
        else:
            lines.append(f"{self.diff.old_name} → {self.diff.new_name}")
        lines.append("")

        all_item_diffs = chain(self.diff.link_diffs.values(), self.diff.joint_diffs.values())
        removed_count, added_count, modified_count = self._count_itemdiffs_by_status(all_item_diffs)
        lines.extend(
            [
                "═" * 45,
                f"SUMMARY: {removed_count} removed, {added_count} added, {modified_count} modified",
                "═" * 45,
                "",
            ]
        )

        lines.extend(self._format_simple_section("removed", "REMOVED", Color.RED))
        lines.extend(self._format_simple_section("added", "ADDED", Color.GREEN))
        lines.extend(self._format_modified_section("MODIFIED"))

        return "\n".join(lines).rstrip()

    def _format_simple_section(self, status: str, title: str, color: Color) -> list[str]:
        """Format a simple section (added or removed items)

        Args:
            status: Status to filter by
            title: Section title
            color: Color for item names

        Returns:
            List of formatted lines
        """
        link_diffs = self._filter_itemdiffs_by_status(self.diff.link_diffs.values(), status)
        joint_diffs = self._filter_itemdiffs_by_status(self.diff.joint_diffs.values(), status)

        if not (link_diffs or joint_diffs):
            return []

        return list(
            chain(
                [self._wrap_bars(title), ""],
                (f"Link: {color.apply(diff.name)}" for diff in link_diffs),
                (f"Joint: {color.apply(diff.name)}" for diff in joint_diffs),
                [""],
            )
        )

    def _format_modified_section(self, title: str) -> list[str]:
        """Format the modified section

        Args:
            title: Section title

        Returns:
            List of formatted lines
        """
        link_diffs = self._filter_itemdiffs_by_status(self.diff.link_diffs.values(), "modified")
        joint_diffs = self._filter_itemdiffs_by_status(self.diff.joint_diffs.values(), "modified")

        if not (link_diffs or joint_diffs):
            return []

        lines = [self._wrap_bars(title), ""]

        for item_type, item_diffs in [("Link", link_diffs), ("Joint", joint_diffs)]:
            for item_diff in item_diffs:
                lines.extend(
                    chain(
                        [f"{item_type}: {item_diff.name}"],
                        (
                            f"{self.INDENT}{self.BULLET}{path}: {self._format_change(change)}"
                            for path, change in sorted(item_diff.changes.items())
                        ),
                        [""],
                    )
                )

        return lines

    def _format_change(self, change: Change) -> str:
        """Format a change

        Args:
            change: Change to format

        Returns:
            Formatted string
        """
        old_str = self._format_value(change.old_value, Color.RED)
        new_str = self._format_value(change.new_value, Color.GREEN)

        if change.status == "removed":
            return old_str
        elif change.status == "added":
            return new_str

        # handle tuples
        if isinstance(change.old_value, tuple) and isinstance(change.new_value, tuple):
            old_str, new_str = self._format_tuple_with_diff(change.old_value, change.new_value)
            return f"{old_str} → {new_str}"

        return f"{old_str} → {new_str}"


class GitFormatter(RobotDiffFormatter):
    """Formatter that mimics the git style"""

    def format(self) -> str:
        """Format the diff in git style

        Returns:
            Formatted string
        """
        lines = ["@@ Name @@", ""]
        if self.diff.old_name != self.diff.new_name:
            lines.extend(
                [Color.RED.apply(f"-name: {self.diff.old_name}"), Color.GREEN.apply(f"+name: {self.diff.new_name}")]
            )
        lines.append("")

        link_removed_count, link_added_count, link_modified_count = self._count_itemdiffs_by_status(
            self.diff.link_diffs.values()
        )
        lines.extend(
            [
                (
                    f"@@ Links ({link_removed_count} removed, "
                    f"{link_added_count} added, {link_modified_count} modified) @@"
                ),
                "",
            ]
        )
        lines.extend(self._format_itemdiffs(self.diff.link_diffs.values(), "Link"))

        joint_removed_count, joint_added_count, joint_modified_count = self._count_itemdiffs_by_status(
            self.diff.joint_diffs.values()
        )
        lines.extend(
            [
                (
                    f"@@ Joints ({joint_removed_count} removed, "
                    f"{joint_added_count} added, {joint_modified_count} modified) @@"
                ),
                "",
            ]
        )
        lines.extend(self._format_itemdiffs(self.diff.joint_diffs.values(), "Joint"))

        return "\n".join(lines).rstrip()

    def _format_itemdiffs(self, item_diffs: Iterable[ItemDiff], item_type: str) -> list[str]:
        """Format item diffs in git style

        Args:
            item_diffs: Iterable of ItemDiff objects
            item_type: Type label ('Link' or 'Joint')

        Returns:
            List of formatted lines
        """
        lines = []
        for item_diff in sorted(item_diffs, key=lambda x: x.name):
            if item_diff.status == "removed":
                lines.append(Color.RED.apply(f"-{item_type} {item_diff.name}"))
            elif item_diff.status == "added":
                lines.append(Color.GREEN.apply(f"+{item_type} {item_diff.name}"))
            else:
                lines.append(f" {item_type} {item_diff.name}")
                for path, change in sorted(item_diff.changes.items()):
                    lines.extend(self._format_change(path, change))
            lines.append("")
        return lines

    def _format_change(self, path: str, change: Change) -> list[str]:
        """Format a change in git style

        Args:
            path: Property path
            change: Change to format

        Returns:
            List of formatted lines
        """
        if change.status == "removed":
            value_str = self._format_value(change.old_value)
            return [Color.RED.apply(f"-{self.INDENT}{path}: {value_str}")]

        if change.status == "added":
            value_str = self._format_value(change.new_value)
            return [Color.GREEN.apply(f"+{self.INDENT}{path}: {value_str}")]

        old_str = self._format_value(change.old_value)
        new_str = self._format_value(change.new_value)

        return [
            Color.RED.apply(f"-{self.INDENT}{path}: {old_str}"),
            Color.GREEN.apply(f"+{self.INDENT}{path}: {new_str}"),
        ]


class CategoryFormatter(RobotDiffFormatter):
    """Formatter that groups changes by category (kinematic, collision, inertia, visual)"""

    def format(self) -> str:
        """Format the diff grouped by change category

        Returns:
            Formatted string
        """
        lines = []

        if self.diff.old_name != self.diff.new_name:
            lines.extend(self._format_name_section())

        lines.extend(self._format_kinematics_section())
        lines.extend(self._format_category_section("collisions", "COLLISION"))
        lines.extend(self._format_category_section("inertial", "INERTIA"))
        lines.extend(self._format_category_section("visuals", "VISUAL"))

        return "\n".join(lines).rstrip()

    def _format_name_section(self) -> list[str]:
        """Format the name section"""
        old_name = Color.RED.apply(self.diff.old_name)
        new_name = Color.GREEN.apply(self.diff.new_name)
        return [self._wrap_bars("NAME"), "", f"{old_name} → {new_name}", ""]

    def _format_kinematics_section(self) -> list[str]:
        """Format the kinematics section"""
        lines = [self._wrap_bars("KINEMATIC"), ""]

        for status, color in {"removed": Color.RED, "added": Color.GREEN}.items():
            link_diffs = self._filter_itemdiffs_by_status(self.diff.link_diffs.values(), status)
            joint_diffs = self._filter_itemdiffs_by_status(self.diff.joint_diffs.values(), status)

            if link_diffs or joint_diffs:
                for link_diff in link_diffs:
                    lines.append(f"Link: {color.apply(link_diff.name)}")
                for joint_diff in joint_diffs:
                    lines.append(f"Joint: {color.apply(joint_diff.name)}")
                lines.append("")

        modified_joint_diffs = self._filter_itemdiffs_by_status(self.diff.joint_diffs.values(), "modified")
        for modified_joint_diff in modified_joint_diffs:
            lines.append(f"Joint: {modified_joint_diff.name}")
            for path, change in sorted(modified_joint_diff.changes.items()):
                lines.append(f"{self.INDENT}{self.BULLET}{path}: {self._format_change(change)}")
            lines.append("")

        return lines if len(lines) > 2 else []

    def _format_category_section(self, category: str, title: str) -> list[str]:
        """Format a category section (collision, inertia, or visual)

        Args:
            category: Category to filter changes by
            title: Section title

        Returns:
            List of formatted lines
        """
        lines = [self._wrap_bars(title), ""]

        modified_link_diffs = self._filter_itemdiffs_by_status(self.diff.link_diffs.values(), "modified")

        for modified_link_diff in modified_link_diffs:
            category_changes = {path: change for path, change in modified_link_diff.changes.items() if category in path}
            if category_changes:
                lines.append(f"Link: {modified_link_diff.name}")
                for path, change in sorted(category_changes.items()):
                    lines.append(f"{self.INDENT}{self.BULLET}{path}: {self._format_change(change)}")
                lines.append("")

        return lines if len(lines) > 2 else []

    def _format_change(self, change: Change) -> str:
        """Format a change

        Args:
            change: Change to format

        Returns:
            Formatted string
        """
        old_str = self._format_value(change.old_value, Color.RED)
        new_str = self._format_value(change.new_value, Color.GREEN)

        if change.status == "removed":
            return old_str
        elif change.status == "added":
            return new_str

        # handle tuples
        if isinstance(change.old_value, tuple) and isinstance(change.new_value, tuple):
            old_str, new_str = self._format_tuple_with_diff(change.old_value, change.new_value)
            return f"{old_str} → {new_str}"

        return f"{old_str} → {new_str}"


class RobotFormatter(StringFormatter):
    """Base class for Robot formatters

    Attributes:
        robot: Robot object to format
    """

    def __init__(self, robot: Robot):
        self.robot = robot

    def _format_header(self) -> list[str]:
        """Format the robot header

        Returns:
            List of formatted header lines
        """
        return [
            "═" * 45,
            f"{self.robot.name}: {len(self.robot.links)} Links, {len(self.robot.joints)} Joints",
            "═" * 45,
        ]


class TreeFormatter(RobotFormatter):
    """Formatter that displays the kinematic tree structure of a robot"""

    def format(self) -> str:
        """Format the robot as a tree

        Returns:
            Formatted string
        """

        self._children = {}
        self._joint_names = {}

        for joint in self.robot.joints.values():
            self._children.setdefault(joint.parent, []).append(joint.child)
            self._joint_names[joint.child] = joint.name

        root = next(link for link in self.robot.links if link not in self._joint_names)

        return "\n".join([*self._format_header(), root, *self._format_tree(root, [])])

    def _format_tree(self, parent, branch_state):
        """Recursively yield formatted tree lines

        Args:
            parent: Parent link name
            branch_state: List of booleans indicating if each ancestor is a last child

        Yields:
            Formatted tree lines
        """
        children = sorted(self._children.get(parent, []))

        for i, child in enumerate(children):
            is_last = i == len(children) - 1

            # build prefix from branch state
            prefix = "".join("    " if is_last_at_level else "│   " for is_last_at_level in branch_state)
            connector = "└── " if is_last else "├── "

            joint_name = self._joint_names.get(child)
            node = f"[{joint_name}] {child}" if joint_name else child
            yield f"{prefix}{connector}{node}"

            # recurse with update branch state
            yield from self._format_tree(child, [*branch_state, is_last])


class DetailedFormatter(RobotFormatter):
    """Formatter that displays detailed information of a robot"""

    def format(self) -> str:
        """Format the robot details

        Returns:
            Formatted string
        """
        lines = [*self._format_header(), "", self._wrap_bars("DETAILS"), ""]

        for _, link in sorted(self.robot.links.items()):
            lines.extend(self._format_link(link))

        for _, joint in sorted(self.robot.joints.items()):
            lines.extend(self._format_joint(joint))

        return "\n".join(lines).rstrip()

    def _format_link(self, link: Link) -> list[str]:
        """Format a link

        Args:
            link: Link to format

        Returns:
            List of formatted lines
        """
        lines = [f"Link: {link.name}"]

        if link.inertial:
            inertia = link.inertial.inertia
            inertia_vals = (inertia.ixx, inertia.ixy, inertia.ixz, inertia.iyy, inertia.iyz, inertia.izz)
            lines.extend(
                [
                    self._wrap_bars("INERTIAL", 1),
                    f"{self.INDENT}{self.BULLET}inertia: {inertia_vals}",
                    f"{self.INDENT}{self.BULLET}mass: {link.inertial.mass}",
                    f"{self.INDENT}{self.BULLET}pos: {self._format_value(link.inertial.origin.xyz)}",
                    f"{self.INDENT}{self.BULLET}quat: {self._format_value(link.inertial.origin.quat)}",
                ]
            )

        if link.collisions:
            lines.append(self._wrap_bars("COLLISION", 1))
            for collision in link.collisions:
                lines.extend(self._format_collision(collision))

        if link.visuals:
            lines.append(self._wrap_bars("VISUAL", 1))
            for visual in link.visuals:
                lines.extend(self._format_visual(visual))

        lines.append("")
        return lines

    def _format_collision(self, collision: Collision) -> list[str]:
        """Format a collision

        Args:
            collision: Collision to format

        Returns:
            List of formatted lines
        """
        if not collision.geometry:
            return []

        geom_type = type(collision.geometry).__name__.lower()
        return [
            f"{self.INDENT}{geom_type}: {collision.name}",
            *self._format_geometry(collision.geometry),
            f"{self.INDENT * 2}{self.BULLET}pos: {self._format_value(collision.origin.xyz)}",
            f"{self.INDENT * 2}{self.BULLET}quat: {self._format_value(collision.origin.quat)}",
        ]

    def _format_visual(self, visual: Visual) -> list[str]:
        """Format a visual

        Args:
            visual: Visual to format

        Returns:
            List of formatted lines
        """
        if not visual.geometry:
            return []

        geom_type = type(visual.geometry).__name__.lower()
        lines = [f"{self.INDENT}{geom_type}:", *self._format_geometry(visual.geometry)]

        if visual.material:
            lines.extend(self._format_material(visual.material))

        lines.extend(
            [
                f"{self.INDENT * 2}{self.BULLET}pos: {self._format_value(visual.origin.xyz)}",
                f"{self.INDENT * 2}{self.BULLET}quat: {self._format_value(visual.origin.quat)}",
            ]
        )

        return lines

    def _format_material(self, material: Material) -> list[str]:
        """Format material properties

        Args:
            material: Material object

        Returns:
            List of formatted lines
        """
        lines = []
        if material.name:
            lines.append(f"{self.INDENT * 2}{self.BULLET}material_name: {material.name}")
        if material.rgba:
            lines.append(f"{self.INDENT * 2}{self.BULLET}rgba: {self._format_value(material.rgba)}")
        if material.texture_filename:
            lines.append(f"{self.INDENT * 2}{self.BULLET}texture_filename: {material.texture_filename}")
        return lines

    def _format_geometry(self, geometry: Geometry) -> list[str]:  # ty: ignore[invalid-return-type]
        """Format geometry properties

        Args:
            geometry: Geometry object

        Returns:
            List of formatted lines
        """
        match geometry:
            case Box():
                return [f"{self.INDENT * 2}{self.BULLET}size: {self._format_value(geometry.size)}"]
            case Cylinder():
                return [
                    f"{self.INDENT * 2}{self.BULLET}radius: {geometry.radius}",
                    f"{self.INDENT * 2}{self.BULLET}length: {geometry.length}",
                ]
            case Sphere():
                return [f"{self.INDENT * 2}{self.BULLET}radius: {geometry.radius}"]
            case Mesh():
                return [
                    f"{self.INDENT * 2}{self.BULLET}filename: {geometry.filename}",
                    f"{self.INDENT * 2}{self.BULLET}scale: {self._format_value(geometry.scale)}",
                ]

    def _format_joint(self, joint: Joint) -> list[str]:
        """Format a joint

        Args:
            joint: Joint to format

        Returns:
            List of formatted lines
        """
        lines = [
            f"Joint: {joint.name}",
            f"{self.INDENT}{self.BULLET}type: {joint.type}",
            f"{self.INDENT}{self.BULLET}parent: {joint.parent}",
            f"{self.INDENT}{self.BULLET}child: {joint.child}",
            f"{self.INDENT}{self.BULLET}pos: {self._format_value(joint.origin.xyz)}",
            f"{self.INDENT}{self.BULLET}quat: {self._format_value(joint.origin.quat)}",
            f"{self.INDENT}{self.BULLET}axis: {self._format_value(joint.axis)}",
        ]

        if joint.limit:
            lines.extend(
                [
                    f"{self.INDENT}{self.BULLET}lower: {joint.limit.lower}",
                    f"{self.INDENT}{self.BULLET}upper: {joint.limit.upper}",
                    f"{self.INDENT}{self.BULLET}effort: {joint.limit.effort}",
                    f"{self.INDENT}{self.BULLET}velocity: {joint.limit.velocity}",
                ]
            )

        lines.append("")
        return lines
