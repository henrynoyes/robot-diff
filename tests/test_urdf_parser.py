from pathlib import Path

import pytest
from lxml import etree

from robot_diff.parsers import Box, Cylinder, Mesh, Sphere, URDFParser


@pytest.fixture
def data_dir() -> Path:
    """Path to test data directory"""
    return Path(__file__).parent / "urdf_data"


def test_simple_robot(data_dir: Path) -> None:
    """Test parsing a simple valid robot"""
    parser = URDFParser(data_dir / "ver_simple_robot.urdf")
    robot = parser.parse()

    assert robot.name == "simple_robot"
    assert len(robot.links) == 2
    assert len(robot.joints) == 1
    assert "base_link" in robot.links
    assert "end_link" in robot.links
    assert "joint1" in robot.joints


def test_complex_robot(data_dir: Path) -> None:
    """Test parsing a complex robot with multiple joint types"""
    parser = URDFParser(data_dir / "ver_complex_robot.urdf")
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


def test_missing_xml_file(data_dir: Path) -> None:
    """Test that missing XML file raises FileNotFoundError"""
    parser = URDFParser(data_dir / "nonexistent.urdf")

    with pytest.raises(FileNotFoundError, match="XML file not found"):
        parser.parse()


def test_missing_xsd_schema(data_dir: Path) -> None:
    """Test that missing XSD schema raises FileNotFoundError"""
    parser = URDFParser(data_dir / "ver_simple_robot.urdf", xsd_path=data_dir / "nonexistent.xsd")

    with pytest.raises(FileNotFoundError, match="XSD schema not found"):
        parser.parse()


def test_default_origin(data_dir: Path) -> None:
    """Test that joints without origin element use default values"""
    parser = URDFParser(data_dir / "ver_default_origin.urdf")
    robot = parser.parse()

    joint = robot.joints["joint1"]
    assert joint.origin.xyz == (0.0, 0.0, 0.0)
    assert joint.origin.quat == (1.0, 0.0, 0.0, 0.0)


def test_default_axis(data_dir: Path) -> None:
    """Test that joints without axis element use default values"""
    parser = URDFParser(data_dir / "ver_default_axis.urdf")
    robot = parser.parse()

    joint = robot.joints["joint1"]
    assert joint.axis == (1.0, 0.0, 0.0)


def test_partial_origin(data_dir: Path) -> None:
    """Test partial origin specification with defaults"""
    parser = URDFParser(data_dir / "ver_partial_origin.urdf")
    robot = parser.parse()

    joint1 = robot.joints["joint1"]
    assert joint1.origin.xyz == (1.0, 2.0, 3.0)
    assert joint1.origin.quat == (1.0, 0.0, 0.0, 0.0)

    joint2 = robot.joints["joint2"]
    assert joint2.origin.xyz == (0.0, 0.0, 0.0)
    assert joint2.origin.quat == (0, 1.0, 0.0, 0.0)


def test_duplicate_link_names(data_dir: Path) -> None:
    """Test that duplicate link names raise ValueError"""
    parser = URDFParser(data_dir / "err_duplicate_link_names.urdf")

    with pytest.raises(ValueError, match="Duplicate link name"):
        parser.parse()


def test_duplicate_joint_names(data_dir: Path) -> None:
    """Test that duplicate joint names raise ValueError"""
    parser = URDFParser(data_dir / "err_duplicate_joint_names.urdf")

    with pytest.raises(ValueError, match="Duplicate joint name"):
        parser.parse()


def test_duplicate_material_names(data_dir: Path) -> None:
    """Test that duplicate global material names raise ValueError"""
    parser = URDFParser(data_dir / "err_duplicate_material_names.urdf")

    with pytest.raises(ValueError, match="Duplicate material name"):
        parser.parse()


def test_invalid_vec3(data_dir: Path) -> None:
    """Test that invalid vector3 format raises ValueError"""
    parser = URDFParser(data_dir / "err_invalid_vec3.urdf")

    with pytest.raises(ValueError, match="Expected 3 space-separated values"):
        parser.parse()


def test_invalid_vec4(data_dir: Path) -> None:
    """Test that invalid vector4 format raises ValueError"""
    parser = URDFParser(data_dir / "err_invalid_vec4.urdf")

    with pytest.raises(ValueError, match="Expected 4 space-separated values"):
        parser.parse()


def test_invalid_xml(data_dir: Path) -> None:
    """Test that invalid XML raises XMLSyntaxError"""
    parser = URDFParser(data_dir / "err_invalid_xml.urdf")

    with pytest.raises(etree.XMLSyntaxError):
        parser.parse()


def test_source_metadata(data_dir: Path) -> None:
    """Test that source metadata is captured"""
    parser = URDFParser(data_dir / "ver_simple_robot.urdf")
    robot = parser.parse()

    link = robot.links["base_link"]
    assert link._line_number == 3
    assert link._source_path == "/robot/link[1]"
    assert link._source_file == str(data_dir / "ver_simple_robot.urdf")


def test_inertial(data_dir: Path) -> None:
    """Test parsing inertial properties"""
    parser = URDFParser(data_dir / "ver_inertial.urdf")
    robot = parser.parse()

    full_inertia_link = robot.links["full_inertia_link"]
    assert full_inertia_link.inertial is not None
    assert full_inertia_link.inertial.mass == 10.0
    assert full_inertia_link.inertial.inertia.ixx == 1.0
    assert full_inertia_link.inertial.inertia.ixy == 0.0
    assert full_inertia_link.inertial.inertia.ixz == 0.0
    assert full_inertia_link.inertial.inertia.iyy == 1.0
    assert full_inertia_link.inertial.inertia.iyz == 0.0
    assert full_inertia_link.inertial.inertia.izz == 1.0

    no_inertia_link = robot.links["no_inertia_link"]
    assert no_inertia_link.inertial is not None
    assert no_inertia_link.inertial.mass == 0.0
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
    parser = URDFParser(data_dir / "ver_collisions.urdf")
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
    assert mesh_link.collisions[0].geometry.filename == "assets/body.dae"
    assert mesh_link.collisions[0].geometry.scale == (1.0, 1.0, 1.0)
    assert isinstance(mesh_link.collisions[1].geometry, Box)
    assert mesh_link.collisions[1].geometry.size == (0.2, 0.2, 0.2)

    empty_link = robot.links["empty_link"]
    assert len(empty_link.collisions) == 0


def test_visuals(data_dir: Path) -> None:
    """Test parsing visuals with global and local materials"""
    parser = URDFParser(data_dir / "ver_visuals.urdf")
    robot = parser.parse()

    blue_link = robot.links["blue_link"]
    assert len(blue_link.visuals) == 1
    assert isinstance(blue_link.visuals[0].geometry, Box)
    assert blue_link.visuals[0].geometry.size == (1.0, 1.0, 1.0)
    assert blue_link.visuals[0].material is not None
    assert blue_link.visuals[0].material.name == "blue_material"
    assert blue_link.visuals[0].material.rgba == (0.0, 0.0, 1.0, 1.0)
    assert blue_link.visuals[0].material.texture_filename is None

    wood_link = robot.links["wood_link"]
    assert len(wood_link.visuals) == 1
    assert isinstance(wood_link.visuals[0].geometry, Cylinder)
    assert wood_link.visuals[0].geometry.radius == 0.1
    assert wood_link.visuals[0].geometry.length == 0.5
    assert wood_link.visuals[0].material is not None
    assert wood_link.visuals[0].material.name == "wood"
    assert wood_link.visuals[0].material.texture_filename == "assets/wood.png"

    metal_link = robot.links["metal_link"]
    assert len(metal_link.visuals) == 1
    assert isinstance(metal_link.visuals[0].geometry, Sphere)
    assert metal_link.visuals[0].geometry.radius == 0.2
    assert metal_link.visuals[0].material is not None
    assert metal_link.visuals[0].material.name == "metal_material"
    assert metal_link.visuals[0].material.rgba is None
    assert metal_link.visuals[0].material.texture_filename == "assets/metal.png"

    red_link = robot.links["red_link"]
    assert len(red_link.visuals) == 1
    assert isinstance(red_link.visuals[0].geometry, Sphere)
    assert red_link.visuals[0].geometry.radius == 0.3
    assert red_link.visuals[0].material is not None
    assert red_link.visuals[0].material.name == "red_material"
    assert red_link.visuals[0].material.rgba == (1.0, 0.0, 0.0, 1.0)
    assert red_link.visuals[0].material.texture_filename == "assets/red_texture.png"

    dual_visual_link = robot.links["dual_visual_link"]
    assert len(dual_visual_link.visuals) == 2
    assert isinstance(dual_visual_link.visuals[0].geometry, Mesh)
    assert dual_visual_link.visuals[0].geometry.filename == "assets/body.dae"
    assert dual_visual_link.visuals[0].geometry.scale == (1.0, 1.0, 1.0)
    assert dual_visual_link.visuals[0].material is None
    assert isinstance(dual_visual_link.visuals[1].geometry, Box)
    assert dual_visual_link.visuals[1].geometry.size == (0.2, 0.2, 0.2)
    assert dual_visual_link.visuals[1].material is None

    empty_link = robot.links["empty_link"]
    assert len(empty_link.visuals) == 0


def test_joint_limits(data_dir: Path) -> None:
    """Test parsing joint limits"""
    parser = URDFParser(data_dir / "ver_joint_limits.urdf")
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
