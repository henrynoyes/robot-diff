from pathlib import Path

import pytest
from lxml import etree

from robot_diff.parsers import Box, Cylinder, Mesh, MJCFParser, Sphere


@pytest.fixture
def data_dir() -> Path:
    """Path to test data directory"""
    return Path(__file__).parent / "mjcf_data"


def test_simple_robot(data_dir: Path) -> None:
    """Test parsing a simple valid robot"""
    parser = MJCFParser(data_dir / "ver_simple_robot.xml")
    robot = parser.parse()

    assert robot.name == "simple_robot"
    assert len(robot.links) == 2
    assert len(robot.joints) == 1
    joint = robot.joints["joint1"]
    assert joint.type == "revolute"
    assert joint.parent == "body1"
    assert joint.child == "body2"


def test_complex_robot(data_dir: Path) -> None:
    """Test parsing a complex robot with multiple joint types"""
    parser = MJCFParser(data_dir / "ver_complex_robot.xml")
    robot = parser.parse()

    assert robot.name == "complex_robot"
    assert len(robot.links) == 4
    assert len(robot.joints) == 3

    assert robot.joints["hinge_joint"].type == "revolute"
    assert robot.joints["slide_joint"].type == "prismatic"
    assert robot.joints["ball_joint"].type == "continuous"

    assert robot.joints["hinge_joint"].parent == "body1"
    assert robot.joints["hinge_joint"].child == "body2"
    assert robot.joints["slide_joint"].parent == "body2"
    assert robot.joints["slide_joint"].child == "body3"
    assert robot.joints["ball_joint"].parent == "body3"
    assert robot.joints["ball_joint"].child == "body4"


def test_missing_xml_file(data_dir: Path) -> None:
    """Test that missing XML file raises FileNotFoundError"""
    parser = MJCFParser(data_dir / "nonexistent.xml")

    with pytest.raises(FileNotFoundError, match="XML file not found"):
        parser.parse()


def test_missing_xsd_schema(data_dir: Path) -> None:
    """Test that missing XSD schema raises FileNotFoundError"""
    parser = MJCFParser(data_dir / "ver_simple_robot.xml", xsd_path=data_dir / "nonexistent.xsd")

    with pytest.raises(FileNotFoundError, match="XSD schema not found"):
        parser.parse()


def test_duplicate_material_names(data_dir: Path) -> None:
    """Test that duplicate global material names raise ValueError"""
    parser = MJCFParser(data_dir / "err_duplicate_material_names.xml")

    with pytest.raises(ValueError, match="Duplicate material name"):
        parser.parse()


def test_default_values(data_dir: Path) -> None:
    """Test that elements without explicit values use MJCF defaults"""
    parser = MJCFParser(data_dir / "ver_default_values.xml")
    robot = parser.parse()

    def_sphere = robot.links["def_sphere"]
    sphere_geom = def_sphere.collisions[0].geometry
    assert isinstance(sphere_geom, Sphere)
    assert sphere_geom.radius == 0.0

    def_box = robot.links["def_box"]
    box_geom = def_box.collisions[0].geometry
    assert isinstance(box_geom, Box)
    assert box_geom.size == (0.0, 0.0, 0.0)

    def_cylinder = robot.links["def_cylinder"]
    cylinder_geom = def_cylinder.collisions[0].geometry
    assert isinstance(cylinder_geom, Cylinder)
    assert cylinder_geom.radius == 0.0

    def_fixed_joint = robot.joints["def_fixedjoint_fixed"]
    assert def_fixed_joint.origin.xyz == (0.0, 0.0, 0.0)
    assert def_fixed_joint.origin.quat == (1.0, 0.0, 0.0, 0.0)
    assert def_fixed_joint.type == "fixed"

    hinge_joint = robot.joints["hinge_joint"]
    assert hinge_joint.type == "revolute"
    assert hinge_joint.axis == (0.0, 0.0, 1.0)
    assert hinge_joint.limit.lower == 0.0
    assert hinge_joint.limit.upper == 0.0


def test_freejoint(data_dir: Path) -> None:
    """Test parsing freejoint for floating base"""
    parser = MJCFParser(data_dir / "ver_freejoint.xml")
    robot = parser.parse()

    joint = robot.joints["free_joint"]
    assert joint.type == "floating"
    assert joint.parent == "body1"
    assert joint.child == "free_body"


def test_inertial(data_dir: Path) -> None:
    """Test parsing inertial properties with diagonal and full inertia"""
    parser = MJCFParser(data_dir / "ver_inertial.xml")
    robot = parser.parse()

    no_inertia_body = robot.links["no_inertia_body"]
    assert no_inertia_body.inertial is not None
    assert no_inertia_body.inertial.mass == 0.0
    assert no_inertia_body.inertial.inertia.ixx == 0.0
    assert no_inertia_body.inertial.inertia.ixy == 0.0
    assert no_inertia_body.inertial.inertia.ixz == 0.0
    assert no_inertia_body.inertial.inertia.iyy == 0.0
    assert no_inertia_body.inertial.inertia.iyz == 0.0
    assert no_inertia_body.inertial.inertia.izz == 0.0

    diag_inertia_body = robot.links["diag_inertia_body"]
    assert diag_inertia_body.inertial is not None
    assert diag_inertia_body.inertial.mass == 10.0
    assert diag_inertia_body.inertial.inertia.ixx == 1.0
    assert diag_inertia_body.inertial.inertia.iyy == 2.0
    assert diag_inertia_body.inertial.inertia.izz == 3.0
    assert diag_inertia_body.inertial.inertia.ixy == 0.0
    assert diag_inertia_body.inertial.inertia.ixz == 0.0
    assert diag_inertia_body.inertial.inertia.iyz == 0.0

    full_inertia_body = robot.links["full_inertia_body"]
    assert full_inertia_body.inertial is not None
    assert full_inertia_body.inertial.mass == 5.0
    assert full_inertia_body.inertial.inertia.ixx == 1.0
    assert full_inertia_body.inertial.inertia.ixy == 0.1
    assert full_inertia_body.inertial.inertia.ixz == 0.2
    assert full_inertia_body.inertial.inertia.iyy == 2.0
    assert full_inertia_body.inertial.inertia.iyz == 0.3
    assert full_inertia_body.inertial.inertia.izz == 3.0


def test_collisions(data_dir: Path) -> None:
    """Test parsing collision geometries"""
    parser = MJCFParser(data_dir / "ver_collisions.xml")
    robot = parser.parse()

    sphere_body = robot.links["sphere_body"]
    assert len(sphere_body.collisions) == 1
    assert isinstance(sphere_body.collisions[0].geometry, Sphere)
    assert sphere_body.collisions[0].geometry.radius == 0.2

    box_body = robot.links["box_body"]
    assert len(box_body.collisions) == 1
    assert isinstance(box_body.collisions[0].geometry, Box)
    assert box_body.collisions[0].geometry.size == (2.0, 2.0, 2.0)

    cylinder_body = robot.links["cylinder_body"]
    assert len(cylinder_body.collisions) == 1
    assert isinstance(cylinder_body.collisions[0].geometry, Cylinder)
    assert cylinder_body.collisions[0].geometry.radius == 0.1
    assert cylinder_body.collisions[0].geometry.length == 1.0

    mesh_body = robot.links["mesh_body"]
    assert len(mesh_body.collisions) == 1
    assert isinstance(mesh_body.collisions[0].geometry, Mesh)
    assert mesh_body.collisions[0].geometry.filename == "assets/body.stl"
    assert mesh_body.collisions[0].geometry.scale == (1.0, 2.0, 3.0)


def test_visuals(data_dir: Path) -> None:
    """Test parsing visual geometries"""
    parser = MJCFParser(data_dir / "ver_visuals.xml")
    robot = parser.parse()

    green_body = robot.links["green_body"]
    assert len(green_body.visuals) == 1
    assert isinstance(green_body.visuals[0].geometry, Box)
    assert green_body.visuals[0].material is not None
    assert green_body.visuals[0].material.name == "green_material"
    assert green_body.visuals[0].material.rgba == (0.0, 1.0, 0.0, 1.0)

    blue_body = robot.links["blue_body"]
    assert len(blue_body.visuals) == 1
    assert isinstance(blue_body.visuals[0].geometry, Mesh)
    assert blue_body.visuals[0].geometry.filename == "assets/body.obj"
    assert blue_body.visuals[0].material is not None
    assert blue_body.visuals[0].material.name == "blue_material"
    assert blue_body.visuals[0].material.rgba == (0.0, 0.0, 1.0, 1.0)

    metal_body = robot.links["metal_body"]
    assert len(metal_body.visuals) == 1
    assert metal_body.visuals[0].material is not None
    assert metal_body.visuals[0].material.texture_filename == "metal.png"


def test_classes(data_dir: Path) -> None:
    """Test parsing and applying default classes with inheritance"""
    parser = MJCFParser(data_dir / "ver_classes.xml")
    robot = parser.parse()

    body1 = robot.links["body1"]
    assert len(body1.visuals) == 1
    assert len(body1.collisions) == 0
    assert isinstance(body1.visuals[0].geometry, Box)
    assert body1.visuals[0].geometry.size == (1.0, 1.0, 1.0)

    body2 = robot.links["body2"]
    assert len(body2.visuals) == 0
    assert len(body2.collisions) == 1
    assert isinstance(body2.collisions[0].geometry, Sphere)
    assert body2.collisions[0].geometry.radius == 0.2

    body3 = robot.links["body3"]
    assert len(body3.collisions) == 1
    assert isinstance(body3.collisions[0].geometry, Sphere)
    assert body3.collisions[0].geometry.radius == 0.15

    joint = robot.joints["hinge_joint"]
    assert joint.type == "revolute"
    assert joint.axis == (0.0, 0.0, 1.0)
    assert joint.limit is not None
    assert joint.limit.lower == -1.57
    assert joint.limit.upper == 1.57


def test_joint_range(data_dir: Path) -> None:
    """Test parsing joint range (limits)"""
    parser = MJCFParser(data_dir / "ver_joint_range.xml")
    robot = parser.parse()

    joint = robot.joints["hinge_joint"]
    assert joint.type == "revolute"
    assert joint.limit is not None
    assert joint.limit.lower == -1.57
    assert joint.limit.upper == 1.57


def test_rotations(data_dir: Path) -> None:
    """Test rotation representations"""
    parser = MJCFParser(data_dir / "ver_rotations.xml")
    robot = parser.parse()

    euler_joint = robot.joints["euler_body_fixed"]
    assert euler_joint.origin.quat == (0.707107, -0.707107, 0.0, 0.0)

    quat_joint = robot.joints["quat_body_fixed"]
    assert quat_joint.origin.quat == (0.707107, 0, 0, 0.707107)


def test_invalid_xml(data_dir: Path) -> None:
    """Test that invalid XML raises XMLSyntaxError"""
    parser = MJCFParser(data_dir / "err_invalid_xml.xml")

    with pytest.raises(etree.XMLSyntaxError):
        parser.parse()


def test_source_metadata(data_dir: Path) -> None:
    """Test that source metadata is captured"""
    parser = MJCFParser(data_dir / "ver_simple_robot.xml")
    robot = parser.parse()

    link = robot.links["body1"]
    assert link._line_number == 3
    assert link._source_path == "/mujoco/worldbody/body"
    assert link._source_file == str(data_dir / "ver_simple_robot.xml")
