from pathlib import Path

import pytest

from robot_diff.parsers import Box, Cylinder, Mesh, SDFParser, Sphere


@pytest.fixture
def data_dir() -> Path:
    """Path to test data directory"""
    return Path(__file__).parent / "sdf_data"


def test_simple_robot(data_dir: Path) -> None:
    """Test parsing a simple valid robot"""
    parser = SDFParser(data_dir / "ver_simple_robot.sdf")
    robot = parser.parse()

    assert robot.name == "simple_robot"
    assert len(robot.links) == 2
    assert len(robot.joints) == 1
    assert "base_link" in robot.links
    assert "end_link" in robot.links
    assert "joint1" in robot.joints


def test_complex_robot(data_dir: Path) -> None:
    """Test parsing a complex robot with multiple joint types"""
    parser = SDFParser(data_dir / "ver_complex_robot.sdf")
    robot = parser.parse()

    assert robot.name == "complex_robot"
    assert len(robot.links) == 4
    assert len(robot.joints) == 3

    assert robot.joints["fixed_joint"].type == "fixed"
    assert robot.joints["revolute_joint"].type == "revolute"
    assert robot.joints["prismatic_joint"].type == "prismatic"

    assert robot.joints["fixed_joint"].parent == "base_link"
    assert robot.joints["fixed_joint"].child == "link1"
    assert robot.joints["revolute_joint"].parent == "link1"
    assert robot.joints["revolute_joint"].child == "link2"


def test_duplicate_link_names(data_dir: Path) -> None:
    """Test that duplicate link names raise ValueError"""
    parser = SDFParser(data_dir / "err_duplicate_link_names.sdf")

    with pytest.raises(ValueError, match="Duplicate link name"):
        parser.parse()


def test_duplicate_joint_names(data_dir: Path) -> None:
    """Test that duplicate joint names raise ValueError"""
    parser = SDFParser(data_dir / "err_duplicate_joint_names.sdf")

    with pytest.raises(ValueError, match="Duplicate joint name"):
        parser.parse()


def test_inertial(data_dir: Path) -> None:
    """Test parsing inertial properties"""
    parser = SDFParser(data_dir / "ver_inertial.sdf")
    robot = parser.parse()

    full_inertia_link = robot.links["full_inertia_link"]
    assert full_inertia_link.inertial is not None
    assert full_inertia_link.inertial.mass == 10.0
    assert full_inertia_link.inertial.origin.xyz == (1.0, 2.0, 3.0)
    assert full_inertia_link.inertial.origin.quat == (1.0, 0.0, 0.0, 0.0)
    assert full_inertia_link.inertial.inertia.ixx == 1.0
    assert full_inertia_link.inertial.inertia.ixy == 0.0
    assert full_inertia_link.inertial.inertia.ixz == 0.0
    assert full_inertia_link.inertial.inertia.iyy == 1.0
    assert full_inertia_link.inertial.inertia.iyz == 0.0
    assert full_inertia_link.inertial.inertia.izz == 1.0

    no_inertia_link = robot.links["no_inertia_link"]
    assert no_inertia_link.inertial is not None
    assert no_inertia_link.inertial.mass == 0.0
    assert no_inertia_link.inertial.origin.xyz == (0.0, 0.0, 0.0)
    assert no_inertia_link.inertial.origin.quat == (1.0, 0.0, 0.0, 0.0)
    assert no_inertia_link.inertial.inertia.ixx == 0.0
    assert no_inertia_link.inertial.inertia.ixy == 0.0
    assert no_inertia_link.inertial.inertia.ixz == 0.0
    assert no_inertia_link.inertial.inertia.iyy == 0.0
    assert no_inertia_link.inertial.inertia.iyz == 0.0
    assert no_inertia_link.inertial.inertia.izz == 0.0

    empty_link = robot.links["empty_link"]
    assert empty_link.inertial is None


def test_collisions(data_dir: Path) -> None:
    """Test parsing collisions"""
    parser = SDFParser(data_dir / "ver_collisions.sdf")
    robot = parser.parse()

    box_link = robot.links["box_link"]
    assert len(box_link.collisions) == 1
    assert isinstance(box_link.collisions[0].geometry, Box)
    assert box_link.collisions[0].geometry.size == (1.0, 1.0, 1.0)

    cylinder_link = robot.links["cylinder_link"]
    assert len(cylinder_link.collisions) == 1
    assert isinstance(cylinder_link.collisions[0].geometry, Cylinder)
    assert cylinder_link.collisions[0].geometry.radius == 0.1
    assert cylinder_link.collisions[0].geometry.length == 0.5

    sphere_link = robot.links["sphere_link"]
    assert len(sphere_link.collisions) == 1
    assert isinstance(sphere_link.collisions[0].geometry, Sphere)
    assert sphere_link.collisions[0].geometry.radius == 0.2

    mesh_link = robot.links["mesh_link"]
    assert len(mesh_link.collisions) == 2
    assert isinstance(mesh_link.collisions[0].geometry, Mesh)
    assert mesh_link.collisions[0].geometry.filename == "assets/body.gltf"
    assert mesh_link.collisions[0].geometry.scale == (1.0, 1.0, 1.0)
    assert isinstance(mesh_link.collisions[1].geometry, Box)
    assert mesh_link.collisions[1].geometry.size == (0.2, 0.2, 0.2)

    empty_link = robot.links["empty_link"]
    assert len(empty_link.collisions) == 0


def test_visuals(data_dir: Path) -> None:
    """Test parsing visuals with materials"""
    parser = SDFParser(data_dir / "ver_visuals.sdf")
    robot = parser.parse()

    blue_link = robot.links["blue_link"]
    assert len(blue_link.visuals) == 1
    assert isinstance(blue_link.visuals[0].geometry, Box)
    assert blue_link.visuals[0].geometry.size == (1.0, 1.0, 1.0)
    assert blue_link.visuals[0].material is not None
    assert blue_link.visuals[0].material.name is None
    assert blue_link.visuals[0].material.rgba == (0.0, 0.0, 1.0, 1.0)
    assert blue_link.visuals[0].material.texture_filename is None

    metal_link = robot.links["metal_link"]
    assert len(metal_link.visuals) == 1
    assert isinstance(metal_link.visuals[0].geometry, Cylinder)
    assert metal_link.visuals[0].geometry.radius == 0.1
    assert metal_link.visuals[0].geometry.length == 0.5
    assert metal_link.visuals[0].material is not None
    assert metal_link.visuals[0].material.name == "metal_material"
    assert metal_link.visuals[0].material.rgba is None
    assert metal_link.visuals[0].material.texture_filename == "assets/metal.png"

    wood_link = robot.links["wood_link"]
    assert len(wood_link.visuals) == 1
    assert isinstance(wood_link.visuals[0].geometry, Sphere)
    assert wood_link.visuals[0].geometry.radius == 0.2
    assert wood_link.visuals[0].material is not None
    assert wood_link.visuals[0].material.name == "wood"
    assert wood_link.visuals[0].material.rgba == (0.0, 1.0, 0.0, 1.0)
    assert wood_link.visuals[0].material.texture_filename == "assets/wood.png"

    dual_visual_link = robot.links["dual_visual_link"]
    assert len(dual_visual_link.visuals) == 2
    assert isinstance(dual_visual_link.visuals[0].geometry, Mesh)
    assert dual_visual_link.visuals[0].geometry.filename == "assets/body.gltf"
    assert dual_visual_link.visuals[0].geometry.scale == (1.0, 1.0, 1.0)
    assert dual_visual_link.visuals[0].material is None
    assert isinstance(dual_visual_link.visuals[1].geometry, Box)
    assert dual_visual_link.visuals[1].geometry.size == (0.2, 0.2, 0.2)
    assert dual_visual_link.visuals[1].material is None

    empty_link = robot.links["empty_link"]
    assert len(empty_link.visuals) == 0


def test_rotations(data_dir: Path) -> None:
    """Test rotation representations"""
    parser = SDFParser(data_dir / "ver_rotations.sdf")
    robot = parser.parse()

    euler_joint = robot.joints["euler_joint"]
    assert euler_joint.origin.xyz == (0.0, 0.0, 0.0)
    assert euler_joint.origin.quat == pytest.approx((0.707107, -0.707107, 0.0, 0.0))

    quat_joint = robot.joints["quat_joint"]
    assert quat_joint.origin.xyz == (0.0, 0.0, 0.0)
    assert quat_joint.origin.quat == pytest.approx((0.707107, 0.0, 0.0, 0.707107))

    empty_pose_joint = robot.joints["empty_pose_joint"]
    assert empty_pose_joint.origin.xyz == (0.0, 0.0, 0.0)
    assert empty_pose_joint.origin.quat == (1.0, 0.0, 0.0, 0.0)


def test_joint_limits(data_dir: Path) -> None:
    """Test parsing joint limits"""
    parser = SDFParser(data_dir / "ver_joint_limits.sdf")
    robot = parser.parse()

    revolute_joint = robot.joints["revolute_joint"]
    assert revolute_joint.limit is not None
    assert revolute_joint.limit.lower == -1.57
    assert revolute_joint.limit.upper == 1.57
    assert revolute_joint.limit.effort == 100.0
    assert revolute_joint.limit.velocity == 2.0

    prismatic_joint = robot.joints["prismatic_joint"]
    assert prismatic_joint.limit is not None
    assert prismatic_joint.limit.lower == -0.5
    assert prismatic_joint.limit.upper == 0.5
    assert prismatic_joint.limit.effort == 0.0
    assert prismatic_joint.limit.velocity == 0.0

    fixed_joint = robot.joints["fixed_joint"]
    assert fixed_joint.limit is None
