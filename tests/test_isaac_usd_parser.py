from pathlib import Path

import pytest

from robot_diff.parsers import Box, Cylinder, IsaacUSDParser, Mesh, Sphere


@pytest.fixture
def data_dir() -> Path:
    """Path to test data directory"""
    return Path(__file__).parent / "isaac_usd_data"


def test_simple_robot(data_dir: Path) -> None:
    """Test parsing a simple valid robot"""
    parser = IsaacUSDParser(data_dir / "ver_simple_robot.usda")
    robot = parser.parse()

    assert robot.name == "simple_robot"
    assert len(robot.links) == 2
    assert len(robot.joints) == 1
    assert "base_link" in robot.links
    assert "end_link" in robot.links
    assert "joint1" in robot.joints


def test_missing_usd_file(data_dir: Path) -> None:
    """Test that missing USD file raises FileNotFoundError"""
    parser = IsaacUSDParser(data_dir / "nonexistent.usda")

    with pytest.raises(FileNotFoundError, match="USD file not found"):
        parser.parse()


def test_missing_default_prim(data_dir: Path) -> None:
    """Test that missing default prim raises ValueError"""
    parser = IsaacUSDParser(data_dir / "err_missing_default_prim.usda")

    with pytest.raises(ValueError, match="No default prim found"):
        parser.parse()


def test_source_metadata(data_dir: Path) -> None:
    """Test that source metadata is captured"""
    parser = IsaacUSDParser(data_dir / "ver_simple_robot.usda")
    robot = parser.parse()

    link = robot.links["base_link"]
    assert link._line_number is None
    assert link._source_path == "/simple_robot/base_link"
    assert link._source_file == str(data_dir / "ver_simple_robot.usda")


def test_inertial(data_dir: Path) -> None:
    """Test parsing inertial properties"""
    parser = IsaacUSDParser(data_dir / "ver_inertial.usda")
    robot = parser.parse()

    full_inertial_link = robot.links["full_inertial_link"]
    assert full_inertial_link.inertial is not None
    assert full_inertial_link.inertial.mass == 4.0
    assert full_inertial_link.inertial.origin.xyz == (1.0, 2.0, 3.0)
    assert full_inertial_link.inertial.origin.quat == pytest.approx((0.707107, 0.707107, 0.0, 0.0))
    assert full_inertial_link.inertial.inertia.ixx == 1.0
    assert full_inertial_link.inertial.inertia.ixy == 0.0
    assert full_inertial_link.inertial.inertia.ixz == 0.0
    assert full_inertial_link.inertial.inertia.iyy == 2.0
    assert full_inertial_link.inertial.inertia.iyz == 0.0
    assert full_inertial_link.inertial.inertia.izz == 3.0

    default_inertial_link = robot.links["default_inertial_link"]
    assert default_inertial_link.inertial is not None
    assert default_inertial_link.inertial.mass == 0.0
    assert default_inertial_link.inertial.origin.xyz == (0.0, 0.0, 0.0)
    assert default_inertial_link.inertial.origin.quat == (1.0, 0.0, 0.0, 0.0)
    assert default_inertial_link.inertial.inertia.ixx == 0.0
    assert default_inertial_link.inertial.inertia.ixy == 0.0
    assert default_inertial_link.inertial.inertia.ixz == 0.0
    assert default_inertial_link.inertial.inertia.iyy == 0.0
    assert default_inertial_link.inertial.inertia.iyz == 0.0
    assert default_inertial_link.inertial.inertia.izz == 0.0

    no_inertial_link = robot.links["no_inertial_link"]
    assert no_inertial_link.inertial is None


def test_collisions(data_dir: Path) -> None:
    """Test parsing collisions"""
    parser = IsaacUSDParser(data_dir / "ver_collisions.usda")
    robot = parser.parse()

    box_link = robot.links["box_link"]
    assert len(box_link.collisions) == 1
    assert isinstance(box_link.collisions[0].geometry, Box)
    assert box_link.collisions[0].geometry.size == (1.0, 2.0, 3.0)

    sphere_link = robot.links["sphere_link"]
    assert len(sphere_link.collisions) == 1
    assert isinstance(sphere_link.collisions[0].geometry, Sphere)
    assert sphere_link.collisions[0].geometry.radius == 2.0

    cylinder_link = robot.links["cylinder_link"]
    assert len(cylinder_link.collisions) == 1
    assert isinstance(cylinder_link.collisions[0].geometry, Cylinder)
    assert cylinder_link.collisions[0].geometry.radius == 2.0
    assert cylinder_link.collisions[0].geometry.length == 1.0

    mesh_link = robot.links["mesh_link"]
    assert len(mesh_link.collisions) == 1
    assert isinstance(mesh_link.collisions[0].geometry, Mesh)
    assert mesh_link.collisions[0].geometry.filename == "usd:mesh_collision/mesh"
    assert mesh_link.collisions[0].geometry.scale == (1.0, 2.0, 3.0)

    no_collisions_link = robot.links["no_collisions_link"]
    assert len(no_collisions_link.collisions) == 0


def test_visuals(data_dir: Path) -> None:
    """Test parsing visuals"""

    parser = IsaacUSDParser(data_dir / "ver_visuals.usda")
    robot = parser.parse()

    mesh_link = robot.links["mesh_link"]
    assert len(mesh_link.visuals) == 1
    assert isinstance(mesh_link.visuals[0].geometry, Mesh)
    assert mesh_link.visuals[0].geometry.filename == "usd:mesh_visual/mesh"
    assert mesh_link.visuals[0].geometry.scale == (1.0, 1.0, 1.0)
    assert mesh_link.visuals[0].material is None

    no_visuals_link = robot.links["no_visuals_link"]
    assert len(no_visuals_link.visuals) == 0


def test_joint_limits(data_dir: Path) -> None:
    """Test parsing joint limits"""
    parser = IsaacUSDParser(data_dir / "ver_joint_limits.usda")
    robot = parser.parse()

    revolute_joint = robot.joints["revolute_joint"]
    assert revolute_joint.type == "revolute"
    assert revolute_joint.limit is not None
    assert revolute_joint.limit.lower == pytest.approx(-1.570796)
    assert revolute_joint.limit.upper == pytest.approx(1.570796)
    assert revolute_joint.limit.effort == 100.0
    assert revolute_joint.limit.velocity == pytest.approx(3.141593)

    prismatic_joint = robot.joints["prismatic_joint"]
    assert prismatic_joint.type == "prismatic"
    assert prismatic_joint.limit is not None
    assert prismatic_joint.limit.lower == -0.5
    assert prismatic_joint.limit.upper == 0.5
    assert prismatic_joint.limit.effort == 50.0
    assert prismatic_joint.limit.velocity == 1.0

    continuous_joint = robot.joints["continuous_joint"]
    assert continuous_joint.type == "continuous"
    assert continuous_joint.limit is None

    fixed_joint = robot.joints["fixed_joint"]
    assert fixed_joint.type == "fixed"
    assert fixed_joint.limit is None


def test_cylinder_axes(data_dir: Path) -> None:
    """Test cylinder axis handling"""
    parser = IsaacUSDParser(data_dir / "ver_cylinder_axes.usda")
    robot = parser.parse()

    x_axis_link = robot.links["x_axis_link"]
    assert len(x_axis_link.collisions) == 1
    assert isinstance(x_axis_link.collisions[0].geometry, Cylinder)
    assert x_axis_link.collisions[0].geometry.radius == 1.0
    assert x_axis_link.collisions[0].geometry.length == 3.0

    y_axis_link = robot.links["y_axis_link"]
    assert len(y_axis_link.collisions) == 1
    assert isinstance(y_axis_link.collisions[0].geometry, Cylinder)
    assert y_axis_link.collisions[0].geometry.radius == 1.0
    assert y_axis_link.collisions[0].geometry.length == 3.0

    z_axis_link = robot.links["z_axis_link"]
    assert len(z_axis_link.collisions) == 1
    assert isinstance(z_axis_link.collisions[0].geometry, Cylinder)
    assert z_axis_link.collisions[0].geometry.radius == 1.0
    assert z_axis_link.collisions[0].geometry.length == 3.0
