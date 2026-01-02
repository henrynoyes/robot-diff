from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable
from dataclasses import is_dataclass
from itertools import chain
from typing import Any

from .diff import Change, ItemDiff, RobotDiff

__all__ = ["StringFormatter", "StatusFormatter", "GitFormatter", "CategoryFormatter"]

# ANSI color codes
RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"


class StringFormatter(ABC):
    """Base class for all string formatters

    Attributes:
        diff: RobotDiff object to format
    """

    def __init__(self, diff: RobotDiff):
        self.diff = diff

    @abstractmethod
    def format(self) -> str:
        """Format the diff"""
        pass

    def _colorize(self, text: str, color: str) -> str:
        """Apply ANSI color to text

        Args:
            text: Text to _colorize
            color: ANSI color code to apply

        Returns:
            Colorized string
        """
        return f"{color}{text}{RESET}"

    def _format_value(self, value: Any, color: str = "") -> str:
        """Format a single value for display

        Args:
            value: Value to format
            color: Optional ANSI color code to apply

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

        return self._colorize(formatted, color) if color else formatted

    def _format_tuple_with_diff(self, old_tuple: tuple, new_tuple: tuple) -> tuple[str, str]:
        """Format old and new tuples with colored diffs

        Args:
            old_tuple: Old tuple
            new_tuple: New tuple

        Returns:
            Tuple of (formatted_old, formatted_new) strings
        """
        if len(old_tuple) != len(new_tuple):
            return (self._format_value(old_tuple, RED), self._format_value(new_tuple, GREEN))

        old_parts, new_parts = [], []
        for old_val, new_val in zip(old_tuple, new_tuple, strict=False):
            if old_val == new_val:
                old_parts.append(str(old_val))
                new_parts.append(str(new_val))
            else:
                old_parts.append(self._colorize(str(old_val), RED))
                new_parts.append(self._colorize(str(new_val), GREEN))

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
            Tuple of (added_count, removed_count, modified_count)
        """

        counts = Counter(item_diff.status for item_diff in item_diffs)
        return counts["added"], counts["removed"], counts["modified"]

    def _wrap_bars(self, text: str) -> str:
        """Wrap text in horiztonal bars (━)

        Args:
            text: Text to wrap

        Returns:
            Formatted string
        """
        return f"━━━ {text} ━━━"


class StatusFormatter(StringFormatter):
    """Formatter that groups item diffs by status (removed, added, and modified)"""

    def format(self) -> str:
        """Format the diff

        Returns:
            Formatted string
        """
        lines = []
        lines.append(self._wrap_bars("NAME"))
        lines.append("")
        if self.diff.old_name != self.diff.new_name:
            lines.append(f"{self._colorize(self.diff.old_name, RED)} → {self._colorize(self.diff.new_name, GREEN)}")
        else:
            lines.append(f"{self.diff.old_name} → {self.diff.new_name}")
        lines.append("")

        all_item_diffs = chain(self.diff.link_diffs.values(), self.diff.joint_diffs.values())
        removed_count, added_count, modified_count = self._count_itemdiffs_by_status(all_item_diffs)
        lines.append("═" * 45)
        lines.append(f"SUMMARY: {removed_count} removed, {added_count} added, {modified_count} modified")
        lines.append("═" * 45)
        lines.append("")

        lines.extend(self._format_simple_section("removed", "REMOVED", RED))
        lines.extend(self._format_simple_section("added", "ADDED", GREEN))
        lines.extend(self._format_modified_section("MODIFIED"))

        return "\n".join(lines).rstrip()

    def _format_simple_section(self, status: str, title: str, color: str) -> list[str]:
        """Format a simple section (added or removed items)

        Args:
            status: Status to filter by
            title: Section title
            color: ANSI color code for item names

        Returns:
            List of formatted lines
        """
        link_diffs = self._filter_itemdiffs_by_status(self.diff.link_diffs.values(), status)
        joint_diffs = self._filter_itemdiffs_by_status(self.diff.joint_diffs.values(), status)

        if not (link_diffs or joint_diffs):
            return []

        lines = [self._wrap_bars(title), ""]

        for link_diff in link_diffs:
            lines.append(f"Link: {color}{link_diff.name}{RESET}")
        for joint_diff in joint_diffs:
            lines.append(f"Joint: {color}{joint_diff.name}{RESET}")

        lines.append("")
        return lines

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
                lines.append(f"{item_type}: {item_diff.name}")
                for path, change in sorted(item_diff.changes.items()):
                    lines.append(f"  • {path}: {self._format_change(change)}")
                lines.append("")

        return lines

    def _format_change(self, change: Change) -> str:
        """Format a change

        Args:
            change: Change to format

        Returns:
            Formatted string
        """
        if change.status == "added":
            return self._format_value("added", GREEN)
        elif change.status == "removed":
            return self._format_value("removed", RED)

        # handle tuples
        if isinstance(change.old_value, tuple) and isinstance(change.new_value, tuple):
            old_str, new_str = self._format_tuple_with_diff(change.old_value, change.new_value)
            return f"{old_str} → {new_str}"

        old_str = self._format_value(change.old_value, RED)
        new_str = self._format_value(change.new_value, GREEN)
        return f"{old_str} → {new_str}"


class GitFormatter(StringFormatter):
    """Formatter that mimics git-style diff"""

    def format(self) -> str:
        """Format the diff in git style

        Returns:
            Formatted string
        """
        lines = []

        lines.append("@@ Name @@")
        lines.append("")
        if self.diff.old_name != self.diff.new_name:
            lines.append(self._colorize(f"-name: {self.diff.old_name}", RED))
            lines.append(self._colorize(f"+name: {self.diff.new_name}", GREEN))
        lines.append("")

        link_removed_count, link_added_count, link_modified_count = self._count_itemdiffs_by_status(
            self.diff.link_diffs.values()
        )
        lines.append(
            f"@@ Links ({link_removed_count} removed, {link_added_count} added, {link_modified_count} modified) @@"
        )
        lines.append("")
        lines.extend(self._format_itemdiffs(self.diff.link_diffs.values(), "Link"))

        joint_removed_count, joint_added_count, joint_modified_count = self._count_itemdiffs_by_status(
            self.diff.joint_diffs.values()
        )
        lines.append(
            f"@@ Joints ({joint_removed_count} removed, {joint_added_count} added, {joint_modified_count} modified) @@"
        )
        lines.append("")
        lines.extend(self._format_itemdiffs(self.diff.joint_diffs.values(), "Joint"))

        return "\n".join(lines).rstrip()

    # ~~ change item_type var name?
    def _format_itemdiffs(self, item_diffs: Iterable[ItemDiff], item_type: str) -> list[str]:
        """Format item diffs in git style

        Args:
            item_diffs: Iterable of ItemDiff objects
            item_type: Type label ("Link" or "Joint")

        Returns:
            List of formatted lines
        """
        lines = []
        for item_diff in sorted(item_diffs, key=lambda x: x.name):
            if item_diff.status == "removed":
                lines.append(self._colorize(f"-{item_type} {item_diff.name}", RED))
            elif item_diff.status == "added":
                lines.append(self._colorize(f"+{item_type} {item_diff.name}", GREEN))
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
        indent = "  "

        if change.status == "added":
            value_str = self._format_value(change.new_value)
            return [self._colorize(f"+{indent}{path}: {value_str}", GREEN)]

        if change.status == "removed":
            value_str = self._format_value(change.old_value)
            return [self._colorize(f"-{indent}{path}: {value_str}", RED)]

        old_str = self._format_value(change.old_value)
        new_str = self._format_value(change.new_value)

        return [
            self._colorize(f"-{indent}{path}: {old_str}", RED),
            self._colorize(f"+{indent}{path}: {new_str}", GREEN),
        ]


class CategoryFormatter(StringFormatter):
    """Formatter that groups changes by category (kinematic, collision, inertia, visual)"""

    def format(self) -> str:
        """Format the diff grouped by change category

        Returns:
            Formatted string
        """
        lines = []

        if self.diff.old_name != self.diff.new_name:
            lines.extend(self._format_name_section())

        # Other sections
        lines.extend(self._format_kinematics_section())
        lines.extend(self._format_category_section("collisions", "COLLISION"))
        lines.extend(self._format_category_section("inertial", "INERTIA"))
        lines.extend(self._format_category_section("visuals", "VISUAL"))

        return "\n".join(lines).rstrip()

    def _format_name_section(self) -> list[str]:
        """Format the name section"""
        old_name = self._colorize(self.diff.old_name, RED)
        new_name = self._colorize(self.diff.new_name, GREEN)
        return [self._wrap_bars("NAME"), "", f"{old_name} → {new_name}", ""]

    # ~~ could potentially simplify by iterating over all link_diffs/joint_diffs in one loop
    def _format_kinematics_section(self) -> list[str]:
        """Format the kinematics section"""
        lines = [self._wrap_bars("KINEMATIC"), ""]

        for status, color in {"removed": RED, "added": GREEN}.items():
            link_diffs = self._filter_itemdiffs_by_status(self.diff.link_diffs.values(), status)
            joint_diffs = self._filter_itemdiffs_by_status(self.diff.joint_diffs.values(), status)

            if link_diffs or joint_diffs:
                for link_diff in link_diffs:
                    lines.append(f"Link: {self._colorize(link_diff.name, color)}")
                for joint_diff in joint_diffs:
                    lines.append(f"Joint: {self._colorize(joint_diff.name, color)}")
                lines.append("")

        modified_joint_diffs = self._filter_itemdiffs_by_status(self.diff.joint_diffs.values(), "modified")
        for modified_joint_diff in modified_joint_diffs:
            lines.append(f"Joint: {modified_joint_diff.name}")
            for path, change in sorted(modified_joint_diff.changes.items()):
                lines.append(f"  • {path}: {self._format_change(change)}")
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
                    lines.append(f"  • {path}: {self._format_change(change)}")
                lines.append("")

        return lines if len(lines) > 2 else []

    def _format_change(self, change: Change) -> str:
        """Format a change

        Args:
            change: Change to format

        Returns:
            Formatted string
        """
        if change.status == "added":
            return self._colorize("added", GREEN)
        elif change.status == "removed":
            return self._colorize("removed", RED)

        # handle tuples
        if isinstance(change.old_value, tuple) and isinstance(change.new_value, tuple):
            old_str, new_str = self._format_tuple_with_diff(change.old_value, change.new_value)
            return f"{old_str} → {new_str}"

        old_str = self._format_value(change.old_value, RED)
        new_str = self._format_value(change.new_value, GREEN)
        return f"{old_str} → {new_str}"
