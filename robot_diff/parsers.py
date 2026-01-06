import math
from pathlib import Path

from lxml import etree
from pxr import Usd, UsdGeom, UsdPhysics

from .model import (
    Box,
    Collision,
    Cylinder,
    Geometry,
    Inertia,
    Inertial,
    Joint,
    Limit,
    Link,
    Material,
    Mesh,
    Pose,
    Robot,
    Sphere,
    Visual,
)

__all__ = ["XMLParser", "URDFParser", "SDFParser", "MJCFParser", "USDParser", "IsaacUSDParser"]

DEFAULT_SCHEMA_PATH = Path(__file__).parent / ("schemas")


class XMLParser:
    """Base class for XML parsers with XSD validation

    Attributes:
        xml_path: Path to XML file
        xsd_path: Path to XSD schema
        tree: Parsed XML tree (None until parsed)
    """

    def __init__(self, xml_path: Path | str, xsd_path: Path | str):
        """Initialize parser

        Args:
            xml_path: Path to XML file
            xsd_path: Path to XSD schema
        """
        self.xml_path = Path(xml_path)
        self.xsd_path = Path(xsd_path)
        self._tree = None

    def _load_and_validate(self):
        """Load XML file and validate against XSD schema

        Raises:
            FileNotFoundError: If XML or XSD file doesn't exist
            etree.XMLSyntaxError: If XML is malformed
            etree.DocumentInvalid: If validation fails
        """
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML file not found: {self.xml_path}")

        if not self.xsd_path.exists():
            raise FileNotFoundError(f"XSD schema not found: {self.xsd_path}")

        self._tree = etree.parse(str(self.xml_path))

        with open(self.xsd_path, "rb") as f:
            schema_doc = etree.parse(f)
        schema = etree.XMLSchema(schema_doc)
        schema.assertValid(self._tree)

    def _parse_vector(self, vector: str, length: int) -> tuple[float, ...]:
        """Parse space-separated string into tuple of floats

        Args:
            vector: Space-separated string of numbers
            length: Expected number of values

        Returns:
            Tuple of floats

        Raises:
            ValueError: If format is invalid
        """
        parts = vector.strip().split()
        if len(parts) != length:
            raise ValueError(f"Expected {length} space-separated values, got {len(parts)}: '{vector}'")

        return tuple(float(x) for x in parts)

    def _rpy_to_quat(self, rpy: tuple[float, float, float], precision: int = 6) -> tuple[float, float, float, float]:
        """Convert roll-pitch-yaw Euler angles to unit quaternion

        Args:
            rpy: Tuple of (roll, pitch, yaw) in radians
            precision: Number of decimal places to round to, defaults to 6

        Returns:
            (w, x, y, z) unit quaternion
        """
        roll, pitch, yaw = rpy
        cr, cp, cy = math.cos(0.5 * roll), math.cos(0.5 * pitch), math.cos(0.5 * yaw)
        sr, sp, sy = math.sin(0.5 * roll), math.sin(0.5 * pitch), math.sin(0.5 * yaw)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        # canonicalize
        if qw < 0:
            sign = -1
        else:
            sign = 1

        return tuple(round(sign * q, precision) for q in (qw, qx, qy, qz))

    def _get_source_metadata(self, elem: etree._Element) -> dict:
        """Get source tracking metadata for element

        Args:
            elem: Element to get metadata for

        Returns:
            Dict with _line_number, _source_path, _source_file
        """
        return {
            "_line_number": elem.sourceline,
            "_source_path": self._tree.getpath(elem),
            "_source_file": str(self.xml_path),
        }


class URDFParser(XMLParser):
    """Parser for URDF files"""

    def __init__(self, xml_path: Path | str, xsd_path: Path | str = DEFAULT_SCHEMA_PATH / "urdf.xsd"):
        """Initialize URDF parser"""
        super().__init__(xml_path, xsd_path)

    def parse(self) -> Robot:
        """Parse URDF file into Robot model

        Returns:
            Robot model
        """
        self._load_and_validate()
        root = self._tree.getroot()

        robot_name = root.get("name")
        self._global_materials = self._parse_global_materials(root)

        links = self._parse_links(root)
        joints = self._parse_joints(root)

        return Robot(name=robot_name, links=links, joints=joints)  # ty: ignore[invalid-argument-type]

    def _parse_global_materials(self, root: etree._Element) -> dict[str, Material]:
        """Parse all global material elements

        Args:
            root: Root element of URDF tree

        Returns:
            Dict mapping global material names to Material objects

        Raises:
            ValueError: If duplicate material names are found
        """

        materials = {}

        for material_elem in root.findall("material"):
            name = material_elem.get("name")

            if name in materials:
                raise ValueError(f"Duplicate material name: '{name}'")

            color_elem = material_elem.find("color")
            if color_elem is not None:
                rgba = self._parse_vector(color_elem.get("rgba", "0 0 0 0"), 4)
            else:
                rgba = None

            texture_elem = material_elem.find("texture")
            if texture_elem is not None:
                texture_filename = texture_elem.get("filename")
            else:
                texture_filename = None

            materials[name] = Material(
                name=name,
                rgba=rgba,  # ty: ignore[invalid-argument-type]
                texture_filename=texture_filename,
                **self._get_source_metadata(material_elem),
            )

        return materials

    def _parse_links(self, root: etree._Element) -> dict[str, Link]:
        """Parse all link elements

        Args:
            root: Root element of URDF tree

        Returns:
            Dict mapping link names to Link objects

        Raises
            ValueError: If duplicate link names are found
        """
        links = {}

        for link_elem in root.findall("link"):
            name = link_elem.get("name")

            if name in links:
                raise ValueError(f"Duplicate link name: '{name}'")

            inertial = self._parse_inertial(link_elem)
            collisions = self._parse_collisions(link_elem)
            visuals = self._parse_visuals(link_elem)

            links[name] = Link(
                name=name,
                inertial=inertial,
                collisions=collisions,
                visuals=visuals,
                **self._get_source_metadata(link_elem),
            )

        return links

    def _parse_inertial(self, link_elem: etree._Element) -> Inertial | None:
        """Parse inertial element

        Args:
            link_elem: Link element containing inertial element

        Returns:
            Inertial object, or None if no inertial is defined
        """
        inertial_elem = link_elem.find("inertial")
        if inertial_elem is None:
            return None

        origin = self._parse_origin(inertial_elem)

        mass_elem = inertial_elem.find("mass")
        if mass_elem is not None:
            mass = float(mass_elem.get("value", "0"))
        else:
            mass = 0.0

        inertia_elem = inertial_elem.find("inertia")
        if inertia_elem is not None:
            inertia = Inertia(
                ixx=float(inertia_elem.get("ixx", "0")),
                ixy=float(inertia_elem.get("ixy", "0")),
                ixz=float(inertia_elem.get("ixz", "0")),
                iyy=float(inertia_elem.get("iyy", "0")),
                iyz=float(inertia_elem.get("iyz", "0")),
                izz=float(inertia_elem.get("izz", "0")),
                **self._get_source_metadata(inertia_elem),
            )
        else:
            inertia = Inertia()

        return Inertial(origin=origin, mass=mass, inertia=inertia, **self._get_source_metadata(inertial_elem))

    def _parse_collisions(self, link_elem: etree._Element) -> list[Collision]:
        """Parse all collision elements

        Args:
            link_elem: Link element containing collision elements

        Returns:
            List of Collision objects
        """
        collisions = []

        for collision_elem in link_elem.findall("collision"):
            name = collision_elem.get("name")
            origin = self._parse_origin(collision_elem)
            geometry = self._parse_geometry(collision_elem)

            collisions.append(
                Collision(name=name, origin=origin, geometry=geometry, **self._get_source_metadata(collision_elem))
            )

        return collisions

    def _parse_geometry(self, parent_elem: etree._Element) -> Geometry | None:
        """Parse geometry element

        Args:
            parent_elem: Parent element containing geometry element

        Returns:
            Geometry object, or None if no geometry is defined
        """
        geometry_elem = parent_elem.find("geometry")
        shape_elem = geometry_elem[0]  # ty: ignore[not-subscriptable]

        if shape_elem.tag == "box":
            size = self._parse_vector(shape_elem.get("size", "0 0 0"), 3)
            return Box(size=size, **self._get_source_metadata(shape_elem))  # ty: ignore[invalid-argument-type]

        elif shape_elem.tag == "cylinder":
            radius = float(shape_elem.get("radius"))  # ty: ignore[invalid-argument-type]
            length = float(shape_elem.get("length"))  # ty: ignore[invalid-argument-type]
            return Cylinder(radius=radius, length=length, **self._get_source_metadata(shape_elem))  # ty: ignore[invalid-argument-type]

        elif shape_elem.tag == "sphere":
            radius = float(shape_elem.get("radius"))  # ty: ignore[invalid-argument-type]
            return Sphere(radius=radius, **self._get_source_metadata(shape_elem))  # ty: ignore[invalid-argument-type]

        elif shape_elem.tag == "mesh":
            filename = shape_elem.get("filename")
            scale = self._parse_vector(shape_elem.get("scale", "1 1 1"), 3)
            return Mesh(filename=filename, scale=scale, **self._get_source_metadata(shape_elem))  # ty: ignore[invalid-argument-type]

    def _parse_visuals(self, link_elem: etree._Element) -> list[Visual]:
        """Parse all visual elements

        Args:
            link_elem: Link element containing visual elements

        Returns:
            List of Visual objects
        """
        visuals = []

        for visual_elem in link_elem.findall("visual"):
            origin = self._parse_origin(visual_elem)
            geometry = self._parse_geometry(visual_elem)
            material = self._parse_material(visual_elem)

            visuals.append(
                Visual(origin=origin, geometry=geometry, material=material, **self._get_source_metadata(visual_elem))
            )

        return visuals

    def _parse_material(self, visual_elem: etree._Element) -> Material | None:
        """Parse material element

        Args:
            visual_elem: Visual element containing material

        Returns:
            Material object, or None if no material is defined
        """
        material_elem = visual_elem.find("material")
        if material_elem is None:
            return None

        name = material_elem.get("name")

        if name in self._global_materials:
            resolved = self._global_materials[name]
            rgba = resolved.rgba
            texture_filename = resolved.texture_filename
        else:
            rgba = None
            texture_filename = None

        color_elem = material_elem.find("color")
        if color_elem is not None:
            rgba = self._parse_vector(color_elem.get("rgba", "0 0 0 0"), 4)

        texture_elem = material_elem.find("texture")
        if texture_elem is not None:
            texture_filename = texture_elem.get("filename")

        return Material(
            name=name,
            rgba=rgba,  # ty: ignore[invalid-argument-type]
            texture_filename=texture_filename,
            **self._get_source_metadata(material_elem),
        )

    def _parse_joints(self, root: etree._Element) -> dict[str, Joint]:
        """Parse all joint elements

        Args:
            root: Root element of URDF tree

        Returns:
            Dict mapping joint names to Joint objects

        Raises
            ValueError: If duplicate joint names are found
        """
        joints = {}

        for joint_elem in root.findall("joint"):
            name = joint_elem.get("name")

            if name in joints:
                raise ValueError(f"Duplicate joint name: '{name}'")

            joint_type = joint_elem.get("type")

            parent = joint_elem.find("parent").get("link")
            child = joint_elem.find("child").get("link")

            origin = self._parse_origin(joint_elem)
            axis = self._parse_axis(joint_elem)
            limit = self._parse_limit(joint_elem)

            joints[name] = Joint(
                name=name,
                type=joint_type,  # ty: ignore[invalid-argument-type]
                parent=parent,  # ty: ignore[invalid-argument-type]
                child=child,  # ty: ignore[invalid-argument-type]
                origin=origin,
                axis=axis,
                limit=limit,
                **self._get_source_metadata(joint_elem),
            )

        return joints

    def _parse_origin(self, parent_elem: etree._Element) -> Pose:
        """Parse origin element

        Args:
            parent_elem: Parent element containing origin element

        Returns:
            Pose with xyz and quat, or default Pose if no origin is defined
        """
        origin_elem = parent_elem.find("origin")
        if origin_elem is None:
            return Pose()

        xyz = self._parse_vector(origin_elem.get("xyz", "0 0 0"), 3)
        rpy = self._parse_vector(origin_elem.get("rpy", "0 0 0"), 3)
        quat = self._rpy_to_quat(rpy)  # ty: ignore[invalid-argument-type]

        return Pose(xyz=xyz, quat=quat, **self._get_source_metadata(origin_elem))  # ty: ignore[invalid-argument-type]

    def _parse_axis(self, joint_elem: etree._Element) -> tuple[float, float, float]:
        """Parse axis element

        Args:
            joint_elem: Joint element containing axis element

        Returns:
            Axis as tuple of 3 floats, or (1.0, 0.0, 0.0) if no axis is defined
        """
        axis_elem = joint_elem.find("axis")
        if axis_elem is None:
            return (1.0, 0.0, 0.0)

        return self._parse_vector(axis_elem.get("xyz", "1 0 0"), 3)  # ty: ignore[invalid-return-type]

    def _parse_limit(self, joint_elem: etree._Element) -> Limit | None:
        """Parse limit element

        Args:
            joint_elem: Joint element containing limit element

        Returns:
            Limit object, or None if no limit is defined
        """
        limit_elem = joint_elem.find("limit")
        if limit_elem is None:
            return None

        lower = float(limit_elem.get("lower", "0"))
        upper = float(limit_elem.get("upper", "0"))
        effort = float(limit_elem.get("effort", "0"))
        velocity = float(limit_elem.get("velocity", "0"))

        return Limit(
            lower=lower, upper=upper, effort=effort, velocity=velocity, **self._get_source_metadata(limit_elem)
        )


class SDFParser(XMLParser):
    """Parser for SDF files"""

    def __init__(self, xml_path: Path | str, xsd_path: Path | str = DEFAULT_SCHEMA_PATH / "sdf.xsd"):
        """Initialize SDF parser"""
        super().__init__(xml_path, xsd_path)

    def parse(self) -> Robot:
        """Parse SDF file into Robot model

        Returns:
            Robot model
        """
        self._load_and_validate()
        root = self._tree.getroot()

        model_elem = root.find("model")
        robot_name = model_elem.get("name")

        joints = self._parse_joints(model_elem)  # ty: ignore[invalid-argument-type]
        links = self._parse_links(model_elem, joints)  # ty: ignore[invalid-argument-type]

        return Robot(name=robot_name, links=links, joints=joints)  # ty: ignore[invalid-argument-type]

    def _parse_joints(self, model_elem: etree._Element) -> dict[str, Joint]:
        """Parse all joint elements

        Args:
            model_elem: Model element containing joint elements

        Returns:
            Dict mapping joint names to Joint objects

        Raises
            ValueError: If duplicate joint names are found or pose relative_to doesn't match parent
        """
        joints = {}

        for joint_elem in model_elem.findall("joint"):
            name = joint_elem.get("name")

            if name in joints:
                raise ValueError(f"Duplicate joint name: '{name}'")

            joint_type = joint_elem.get("type")

            parent = joint_elem.findtext("parent")
            child = joint_elem.findtext("child")

            pose_elem = joint_elem.find("pose")
            if pose_elem is None or pose_elem.get("relative_to") != parent:
                raise ValueError(f"Joint '{name}' pose must have relative_to='{parent}'")

            origin = self._parse_pose(joint_elem)
            axis = self._parse_axis(joint_elem)
            limit = self._parse_limit(joint_elem)

            joints[name] = Joint(
                name=name,
                type=joint_type,  # ty: ignore[invalid-argument-type]
                parent=parent,  # ty: ignore[invalid-argument-type]
                child=child,  # ty: ignore[invalid-argument-type]
                origin=origin,
                axis=axis,
                limit=limit,
                **self._get_source_metadata(joint_elem),
            )

        return joints

    def _parse_links(self, model_elem: etree._Element, joints: dict[str, Joint]) -> dict[str, Link]:
        """Parse all link elements

        Args:
            model_elem: Model element containing link elements
            joints: Dict mapping joint names to Joint objects

        Returns:
            Dict mapping link names to Link objects

        Raises
            ValueError: If duplicate link names are found, pose relative_to doesn't match parent, or pose is non-trivial
        """
        links = {}

        for link_elem in model_elem.findall("link"):
            name = link_elem.get("name")

            if name in links:
                raise ValueError(f"Duplicate link name: '{name}'")

            parent_joint = next((joint_name for joint_name, joint in joints.items() if joint.child == name), None)
            if parent_joint is not None:
                pose_elem = link_elem.find("pose")
                if pose_elem is None or pose_elem.get("relative_to") != parent_joint:
                    raise ValueError(f"Link '{name}' pose must have relative_to='{parent_joint}'")

            pose = self._parse_pose(link_elem)
            if pose.xyz != (0.0, 0.0, 0.0) or pose.quat != (1.0, 0.0, 0.0, 0.0):
                raise ValueError(f"Link '{name}' pose must be the identity")

            inertial = self._parse_inertial(link_elem)
            collisions = self._parse_collisions(link_elem)
            visuals = self._parse_visuals(link_elem)

            links[name] = Link(
                name=name,
                inertial=inertial,
                collisions=collisions,
                visuals=visuals,
                **self._get_source_metadata(link_elem),
            )

        return links

    def _parse_inertial(self, link_elem: etree._Element) -> Inertial | None:
        """Parse inertial element

        Args:
            link_elem: Link element containing inertial element

        Returns:
            Inertial object, or None if no inertial is defined
        """
        inertial_elem = link_elem.find("inertial")
        if inertial_elem is None:
            return None

        origin = self._parse_pose(inertial_elem)

        mass = float(inertial_elem.findtext("mass", "0"))

        inertia_elem = inertial_elem.find("inertia")
        if inertia_elem is not None:
            inertia = Inertia(
                ixx=float(inertia_elem.findtext("ixx", "0")),
                ixy=float(inertia_elem.findtext("ixy", "0")),
                ixz=float(inertia_elem.findtext("ixz", "0")),
                iyy=float(inertia_elem.findtext("iyy", "0")),
                iyz=float(inertia_elem.findtext("iyz", "0")),
                izz=float(inertia_elem.findtext("izz", "0")),
                **self._get_source_metadata(inertia_elem),
            )
        else:
            inertia = Inertia()

        return Inertial(origin=origin, mass=mass, inertia=inertia, **self._get_source_metadata(inertial_elem))

    def _parse_collisions(self, link_elem: etree._Element) -> list[Collision]:
        """Parse all collision elements

        Args:
            link_elem: Link element containing collision elements

        Returns:
            List of Collision objects
        """
        collisions = []

        for collision_elem in link_elem.findall("collision"):
            name = collision_elem.get("name")
            origin = self._parse_pose(collision_elem)
            geometry = self._parse_geometry(collision_elem)

            collisions.append(
                Collision(name=name, origin=origin, geometry=geometry, **self._get_source_metadata(collision_elem))
            )

        return collisions

    def _parse_geometry(self, parent_elem: etree._Element) -> Geometry | None:
        """Parse geometry element

        Args:
            parent_elem: Parent element containing geometry element

        Returns:
            Geometry object, or None if no geometry is defined
        """
        geometry_elem = parent_elem.find("geometry")
        shape_elem = geometry_elem[0]  # ty: ignore[not-subscriptable]

        if shape_elem.tag == "box":
            size = self._parse_vector(shape_elem.findtext("size", "0 0 0"), 3)
            return Box(size=size, **self._get_source_metadata(shape_elem))  # ty: ignore[invalid-argument-type]

        elif shape_elem.tag == "cylinder":
            radius = float(shape_elem.findtext("radius"))  # ty: ignore[invalid-argument-type]
            length = float(shape_elem.findtext("length"))  # ty: ignore[invalid-argument-type]
            return Cylinder(radius=radius, length=length, **self._get_source_metadata(shape_elem))

        elif shape_elem.tag == "sphere":
            radius = float(shape_elem.findtext("radius"))  # ty: ignore[invalid-argument-type]
            return Sphere(radius=radius, **self._get_source_metadata(shape_elem))

        elif shape_elem.tag == "mesh":
            filename = shape_elem.findtext("uri")
            scale = self._parse_vector(shape_elem.findtext("scale", "1 1 1"), 3)
            return Mesh(filename=filename, scale=scale, **self._get_source_metadata(shape_elem))  # ty: ignore[invalid-argument-type]

    def _parse_visuals(self, link_elem: etree._Element) -> list[Visual]:
        """Parse all visual elements

        Args:
            link_elem: Link element containing visual elements

        Returns:
            List of Visual objects
        """
        visuals = []

        for visual_elem in link_elem.findall("visual"):
            origin = self._parse_pose(visual_elem)
            geometry = self._parse_geometry(visual_elem)
            material = self._parse_material(visual_elem)

            visuals.append(
                Visual(origin=origin, geometry=geometry, material=material, **self._get_source_metadata(visual_elem))
            )

        return visuals

    def _parse_material(self, visual_elem: etree._Element) -> Material | None:
        """Parse material element

        Args:
            visual_elem: Visual element containing material

        Returns:
            Material object, or None if no material is defined
        """
        material_elem = visual_elem.find("material")
        if material_elem is None:
            return None

        if diffuse_text := material_elem.findtext("diffuse"):
            rgba = self._parse_vector(diffuse_text, 4)
        else:
            rgba = None

        script_elem = material_elem.find("script")
        if script_elem is not None:
            name = script_elem.findtext("name")
            texture_filename = script_elem.findtext("uri")
        else:
            name = None
            texture_filename = None

        return Material(
            name=name,
            rgba=rgba,  # ty: ignore[invalid-argument-type]
            texture_filename=texture_filename,
            **self._get_source_metadata(material_elem),
        )

    def _parse_pose(self, parent_elem: etree._Element) -> Pose:
        """Parse origin element

        Args:
            parent_elem: Parent element containing origin element

        Returns:
            Pose with xyz and quat, or default Pose if no origin is defined
        """
        pose_elem = parent_elem.find("pose")

        if pose_elem is None:
            return Pose()

        pose_text = pose_elem.text
        if pose_text is None or pose_text.strip() == "":
            return Pose()

        rotation_format = pose_elem.get("rotation_format", "euler_rpy")

        if rotation_format == "euler_rpy":
            values = self._parse_vector(pose_text, 6)
            xyz = values[0:3]
            rpy = values[3:6]
            quat = self._rpy_to_quat(rpy)

        elif rotation_format == "quat_xyzw":
            values = self._parse_vector(pose_text, 7)
            xyz = values[0:3]
            quat = (values[6], values[3], values[4], values[5])

        return Pose(xyz=xyz, quat=quat, **self._get_source_metadata(pose_elem))

    def _parse_axis(self, joint_elem: etree._Element) -> tuple[float, float, float]:
        """Parse axis element

        Args:
            joint_elem: Joint element containing axis element

        Returns:
            Axis as tuple of 3 floats, or (1.0, 0.0, 0.0) if no axis is defined
        """
        axis_elem = joint_elem.find("axis")
        if axis_elem is None:
            return (1.0, 0.0, 0.0)

        return self._parse_vector(axis_elem.findtext("xyz", "1 0 0"), 3)  # ty: ignore[invalid-return-type]

    def _parse_limit(self, joint_elem: etree._Element) -> Limit | None:
        """Parse limit element

        Args:
            joint_elem: Joint element containing limit element

        Returns:
            Limit object, or None if no limit is defined
        """
        axis_elem = joint_elem.find("axis")
        if axis_elem is None:
            return None

        limit_elem = axis_elem.find("limit")
        if limit_elem is None:
            return None

        lower = float(limit_elem.findtext("lower", "0"))
        upper = float(limit_elem.findtext("upper", "0"))
        effort = float(limit_elem.findtext("effort", "0"))
        velocity = float(limit_elem.findtext("velocity", "0"))

        return Limit(
            lower=lower, upper=upper, effort=effort, velocity=velocity, **self._get_source_metadata(limit_elem)
        )


class MJCFParser(XMLParser):
    """Parser for MJCF files"""

    def __init__(self, xml_path: Path | str, xsd_path: Path | str = DEFAULT_SCHEMA_PATH / "mjcf.xsd"):
        """Initialize MJCF parser"""
        super().__init__(xml_path, xsd_path)

    def parse(self) -> Robot:
        """Parse MJCF file into Robot model

        Returns:
            Robot model
        """
        self._load_and_validate()
        root = self._tree.getroot()

        robot_name = root.get("model", "robot")

        self._global_materials = self._parse_global_materials(root)
        self._global_meshes = self._parse_global_meshes(root)
        self._classes = self._parse_classes(root)

        links = {}
        joints = {}

        worldbody = root.find("worldbody")
        self._parse_bodies(worldbody, None, links, joints)  # ty: ignore[invalid-argument-type]

        return Robot(name=robot_name, links=links, joints=joints)

    def _parse_classes(self, root: etree._Element) -> dict[str, dict]:
        """Parse all default classes

        Args:
            root: Root element of MJCF tree

        Returns:
            Dict mapping class names to class data
        """
        classes = {}

        default_root = root.find("default")
        if default_root is not None:
            for default_elem in default_root.findall("default"):
                self._parse_defaults(default_elem, None, classes)

        return classes

    def _parse_defaults(self, default_elem: etree._Element, parent_class: str | None, classes: dict):
        """Recursively parse default elements

        Args:
            default_elem: Default element to parse
            parent_class: Name of parent class
            classes: Dict to populate with class data
        """
        class_name = default_elem.get("class", "main")

        classes[class_name] = {"parent": parent_class}

        for elem_type in ["joint", "geom"]:
            elem = default_elem.find(elem_type)
            if elem is not None:
                classes[class_name][elem_type] = dict(elem.attrib)

        # recursively parse children
        for child_elem in default_elem.findall("default"):
            self._parse_defaults(child_elem, class_name, classes)

    def _get_class_defaults(self, class_name: str, element_type: str) -> dict:
        """Recursively get class defaults for an element type

        Args:
            class_name: Name of the class
            element_type: Type of element ('joint' or 'geom')

        Returns:
            Dict of default attributes
        """
        if not class_name or class_name not in self._classes:
            return {}

        class_data = self._classes[class_name]

        # recursively get parent defaults
        defaults = self._get_class_defaults(class_data.get("parent"), element_type)  # ty: ignore[invalid-argument-type]

        # stack child data
        if element_type in class_data:
            defaults.update(class_data[element_type])

        return defaults

    def _parse_global_materials(self, root: etree._Element) -> dict[str, Material]:
        """Parse all global material elements

        Args:
            root: Root element of MJCF tree

        Returns:
            Dict mapping global material names to Material objects

        Raises:
            ValueError: If duplicate material names are found
        """
        materials = {}

        assets = root.find("asset")
        if assets is None:
            return materials

        textures = self._parse_global_textures(assets)

        for material_elem in assets.findall("material"):
            name = material_elem.get("name")

            if name in materials:
                raise ValueError(f"Duplicate material name: '{name}'")

            if rgba_str := material_elem.get("rgba"):
                rgba = self._parse_vector(rgba_str, 4)
            else:
                rgba = None

            if texture_name := material_elem.get("texture"):
                texture_filename = textures.get(texture_name)
            else:
                texture_filename = None

            materials[name] = Material(
                name=name,
                rgba=rgba,  # ty: ignore[invalid-argument-type]
                texture_filename=texture_filename,
                **self._get_source_metadata(material_elem),
            )

        return materials

    def _parse_global_textures(self, assets: etree._Element) -> dict[str, str]:
        """Parse all global texture elements

        Args:
            assets: Asset element containing textures

        Returns:
            Dict mapping texture names to filenames
        """
        textures = {}

        for texture_elem in assets.findall("texture"):
            name = texture_elem.get("name")
            filename = texture_elem.get("file")

            if name and filename:
                textures[name] = filename

        return textures

    def _parse_global_meshes(self, root: etree._Element) -> dict[str, Mesh]:
        """Parse all global mesh elements

        Args:
            root: Root element of MJCF tree

        Returns:
            Dict mapping mesh names to Mesh objects
        """
        meshes = {}

        assets = root.find("asset")
        if assets is None:
            return meshes

        compiler_elem = root.find("compiler")
        if compiler_elem is not None:
            meshdir = compiler_elem.get("meshdir")
        else:
            meshdir = ""

        for mesh_elem in assets.findall("mesh"):
            name = mesh_elem.get("name")
            filename = mesh_elem.get("file")

            if filename is not None:
                if name is None:
                    name = Path(filename).stem

                if meshdir:
                    filename = str(Path(meshdir) / filename)

                scale_str = mesh_elem.get("scale", "1 1 1")
                scale = self._parse_vector(scale_str, 3)

                meshes[name] = Mesh(filename=filename, scale=scale, **self._get_source_metadata(mesh_elem))  # ty: ignore[invalid-argument-type]

        return meshes

    def _parse_bodies(self, parent_elem: etree._Element, parent_body_name: str | None, links: dict, joints: dict):
        """Recursively parse body hierarchy

        Args:
            parent_elem: Parent element (worldbody or body)
            parent_body_name: Name of parent body (None for worldbody)
            links: Dict mapping link names to Link objects, populate with recursion
            joints: Dict to mapping joint names to Joint objects, populate with recursion
        """
        for body_elem in parent_elem.findall("body"):
            body_name = body_elem.get("name")
            if body_name:
                link = self._create_link(body_elem)
                links[body_name] = link

                if parent_body_name is not None:
                    joint = self._create_joint(body_elem, parent_body_name, body_name)
                    if joint:
                        joints[joint.name] = joint

                self._parse_bodies(body_elem, body_name, links, joints)

    def _create_link(self, body_elem: etree._Element) -> Link:
        """Create Link object from body element

        Args:
            body_elem: Body element

        Returns:
            Link object
        """
        name = body_elem.get("name")
        inertial = self._parse_inertial(body_elem)
        collisions = self._parse_geoms(body_elem, "collision")
        visuals = self._parse_geoms(body_elem, "visual")

        return Link(
            name=name,  # ty: ignore[invalid-argument-type]
            inertial=inertial,
            collisions=collisions,
            visuals=visuals,
            **self._get_source_metadata(body_elem),
        )

    def _parse_inertial(self, body_elem: etree._Element) -> Inertial | None:
        """Parse inertial element

        Args:
            body_elem: Body element containing inertial element

        Returns:
            Inertial object, or None if no inertial defined
        """
        inertial_elem = body_elem.find("inertial")
        if inertial_elem is None:
            return None

        origin = self._parse_origin(inertial_elem)

        mass = float(inertial_elem.get("mass", "0"))

        if diaginertia := inertial_elem.get("diaginertia"):
            ixx, iyy, izz = self._parse_vector(diaginertia, 3)
            inertia = Inertia(ixx=ixx, iyy=iyy, izz=izz, **self._get_source_metadata(inertial_elem))
        elif fullinertia := inertial_elem.get("fullinertia"):
            ixx, ixy, ixz, iyy, iyz, izz = self._parse_vector(fullinertia, 6)
            inertia = Inertia(
                ixx=ixx, ixy=ixy, ixz=ixz, iyy=iyy, iyz=iyz, izz=izz, **self._get_source_metadata(inertial_elem)
            )
        else:
            inertia = Inertia()

        return Inertial(origin=origin, mass=mass, inertia=inertia, **self._get_source_metadata(inertial_elem))

    def _parse_geoms(self, body_elem: etree._Element, target_class: str) -> list:
        """Parse geom elements filtered by class

        Args:
            body_elem: Body element containing geom elements
            target_class: Class to filter by ('visual' or 'collision')

        Returns:
            List of Visual or Collision objects
        """

        def is_descendant(class_name: str) -> bool:
            """Check if class_name inherits from target_class"""
            current = class_name
            while current:
                if current == target_class:
                    return True
                current = self._classes.get(current, {}).get("parent")
            return False

        objects = []

        for geom_elem in body_elem.findall("geom"):
            geom_class = geom_elem.get("class")

            if geom_class and is_descendant(geom_class):
                class_defaults = self._get_class_defaults(geom_class, "geom")

                origin = self._parse_origin(geom_elem, class_defaults)
                geometry = self._parse_geometry(geom_elem, class_defaults)

                if target_class == "visual":
                    material = self._parse_material(geom_elem)
                    objects.append(
                        Visual(
                            origin=origin, geometry=geometry, material=material, **self._get_source_metadata(geom_elem)
                        )
                    )
                elif target_class == "collision":
                    name = geom_elem.get("name")
                    objects.append(
                        Collision(name=name, origin=origin, geometry=geometry, **self._get_source_metadata(geom_elem))
                    )

        return objects

    def _parse_geometry(self, geom_elem: etree._Element, class_defaults: dict | None = None) -> Geometry | None:
        """Parse geometry from geom element

        Args:
            geom_elem: Geom element
            class_defaults: Class defaults to apply

        Returns:
            Geometry object, or None if no geometry is defined
        """
        class_defaults = class_defaults or {}

        if mesh_name := geom_elem.get("mesh"):
            return self._global_meshes.get(mesh_name)

        geom_type = geom_elem.get("type") or class_defaults.get("type", "sphere")
        size_str = geom_elem.get("size") or class_defaults.get("size")

        if geom_type == "box":
            if size_str:
                size = tuple(2 * x for x in self._parse_vector(size_str, 3))  # MJCF uses half-extents
                return Box(size=size, **self._get_source_metadata(geom_elem))  # ty: ignore[invalid-argument-type]
            else:
                return Box(**self._get_source_metadata(geom_elem))

        elif geom_type == "cylinder":
            if size_str:
                size = self._parse_vector(size_str, 2)
                radius = float(size[0])
                length = 2 * float(size[1])  # MJCF uses half-length
                return Cylinder(radius=radius, length=length, **self._get_source_metadata(geom_elem))
            else:
                return Cylinder(**self._get_source_metadata(geom_elem))

        elif geom_type == "sphere":
            if size_str:
                size = self._parse_vector(size_str, 1)
                radius = float(size[0])
                return Sphere(radius=radius, **self._get_source_metadata(geom_elem))
            else:
                return Sphere(**self._get_source_metadata(geom_elem))
        return None

    def _parse_material(self, geom_elem: etree._Element) -> Material | None:
        """Parse material from geom element

        Args:
            geom_elem: Geom element

        Returns:
            Material object, or None if no material defined
        """
        material_name = geom_elem.get("material")
        if not material_name:
            return None

        if material_name in self._global_materials:
            resolved = self._global_materials[material_name]
            rgba = resolved.rgba
            texture_filename = resolved.texture_filename
        else:
            rgba = None
            texture_filename = None

        if rgba_str := geom_elem.get("rgba"):
            rgba = self._parse_vector(rgba_str, 4)

        return Material(
            name=material_name,
            rgba=rgba,  # ty: ignore[invalid-argument-type]
            texture_filename=texture_filename,
            **self._get_source_metadata(geom_elem),
        )

    def _create_joint(self, body_elem: etree._Element, parent_name: str, child_name: str) -> Joint | None:
        """Create Joint object from body element pair

        Args:
            body_elem: Body element containing joint
            parent_name: Name of parent body
            child_name: Name of this body (child)

        Returns:
            Joint object, or None if no joint defined
        """
        joint_types = {"hinge": "revolute", "slide": "prismatic", "ball": "continuous"}

        freejoint_elem = body_elem.find("freejoint")
        if freejoint_elem is not None:
            joint_name = freejoint_elem.get("name", f"{child_name}_freejoint")
            origin = self._parse_origin(body_elem)
            return Joint(
                name=joint_name,
                type="floating",
                parent=parent_name,
                child=child_name,
                origin=origin,
                **self._get_source_metadata(freejoint_elem),
            )

        joint_elem = body_elem.find("joint")
        if joint_elem is not None:
            if class_name := joint_elem.get("class"):
                class_defaults = self._get_class_defaults(class_name, "joint")
            else:
                class_defaults = {}

            joint_name = joint_elem.get("name", f"{child_name}_joint")

            type_str = joint_elem.get("type") or class_defaults.get("type", "hinge")
            joint_type = joint_types[type_str]

            origin = self._parse_origin(body_elem)

            axis_str = joint_elem.get("axis") or class_defaults.get("axis", "0 0 1")
            axis = self._parse_vector(axis_str, 3)

            range_str = joint_elem.get("range") or class_defaults.get("range", "0 0")
            range_vals = self._parse_vector(range_str, 2)
            lower = float(range_vals[0])
            upper = float(range_vals[1])
            limit = Limit(lower=lower, upper=upper, **self._get_source_metadata(joint_elem))

            return Joint(
                name=joint_name,
                type=joint_type,
                parent=parent_name,
                child=child_name,
                origin=origin,
                axis=axis,  # ty: ignore[invalid-argument-type]
                limit=limit,
                **self._get_source_metadata(joint_elem),
            )

        origin = self._parse_origin(body_elem)
        joint_name = f"{child_name}_fixed"
        return Joint(
            name=joint_name,
            type="fixed",
            parent=parent_name,
            child=child_name,
            origin=origin,
            **self._get_source_metadata(body_elem),
        )

    def _parse_origin(self, elem: etree._Element, class_defaults: dict | None = None) -> Pose:
        """Parse position and orientation from element

        Args:
            elem: Element containing pose information
            class_defaults: Class defaults to apply

        Returns:
            Pose object with xyz and quat
        """
        class_defaults = class_defaults or {}

        pos_str = elem.get("pos") or class_defaults.get("pos", "0 0 0")
        xyz = self._parse_vector(pos_str, 3)

        quat_str = elem.get("quat") or class_defaults.get("quat")
        euler_str = elem.get("euler") or class_defaults.get("euler")

        if quat_str:
            quat = self._parse_vector(quat_str, 4)
        elif euler_str:
            rpy = self._parse_vector(euler_str, 3)
            quat = self._rpy_to_quat(rpy)  # ty: ignore[invalid-argument-type]
        else:
            quat = (1.0, 0.0, 0.0, 0.0)

        return Pose(xyz=xyz, quat=quat, **self._get_source_metadata(elem))  # ty: ignore[invalid-argument-type]


class USDParser:
    """Base class for USD parser

    Attributes:
        usd_path: Path to USD file
        stage: Parsed USD stage (None until parsed)
    """

    def __init__(self, usd_path: Path | str):
        """Initialize USD parser

        Args:
            usd_path: Path to USD file
        """
        self.usd_path = Path(usd_path)
        self._stage = None

    def _get_source_metadata(self, prim: Usd.Prim) -> dict:
        """Get source tracking metadata for prim

        Args:
            prim: Prim to get metadata for

        Returns:
            Dict with _line_number, _source_path, _source_file
        """
        return {
            "_line_number": None,
            "_source_path": str(prim.GetPath()),
            "_source_file": str(self.usd_path),
        }


class IsaacUSDParser(USDParser):
    """Parser for Isaac-formatted USD files"""

    def __init__(self, usd_path: Path | str):
        """Initialize USD parser"""
        super().__init__(usd_path)

    def parse(self) -> Robot:
        """Parse USD file into Robot model

        Returns:
            Robot model

        Raises:
            FileNotFoundError: If USD file doesn't exist
            ValueError: If no default prim is found
        """
        if not self.usd_path.exists():
            raise FileNotFoundError(f"USD file not found: {self.usd_path}")

        self._stage = Usd.Stage.Open(str(self.usd_path))

        robot_prim = self._stage.GetDefaultPrim()
        if not robot_prim:
            raise ValueError("No default prim found in USD stage")

        robot_name = robot_prim.GetName()

        links = self._parse_links(robot_prim)
        joints = self._parse_joints(robot_prim)

        return Robot(name=robot_name, links=links, joints=joints)

    def _parse_links(self, robot_prim: Usd.Prim) -> dict[str, Link]:
        """Parse all link prims

        Args:
            robot_prim: Root robot prim

        Returns:
            Dict mapping link names to Link objects
        """
        links = {}

        robot_links_rel = robot_prim.GetRelationship("isaac:physics:robotLinks")
        if not robot_links_rel:
            return links

        link_targets = robot_links_rel.GetTargets()

        for link_path in link_targets:
            link_prim = self._stage.GetPrimAtPath(link_path)
            link_name = link_prim.GetName()

            inertial = self._parse_inertial(link_prim)
            collisions = self._parse_collisions(link_prim)
            visuals = self._parse_visuals(link_prim)

            link = Link(
                name=link_name,
                inertial=inertial,
                collisions=collisions,
                visuals=visuals,
                **self._get_source_metadata(link_prim),
            )

            links[link_name] = link

        return links

    def _parse_inertial(self, link_prim: Usd.Prim) -> Inertial | None:
        """Parse inertial information from a link prim

        Args:
            link_prim: Link prim

        Returns:
            Inertial object, or None if no inertial information is defined
        """
        if not link_prim.HasAPI("PhysicsMassAPI"):
            return None

        mass_api = UsdPhysics.MassAPI(link_prim)

        mass = mass_api.GetMassAttr().Get()
        xyz = tuple(mass_api.GetCenterOfMassAttr().Get())

        gf_quat = mass_api.GetPrincipalAxesAttr().Get().GetNormalized()
        quat = (gf_quat.GetReal(), *gf_quat.GetImaginary())

        origin = Pose(xyz=xyz, quat=quat)  # ty: ignore[invalid-argument-type]

        diag = mass_api.GetDiagonalInertiaAttr().Get()
        inertia = Inertia(ixx=diag[0], iyy=diag[1], izz=diag[2])

        return Inertial(origin=origin, mass=mass, inertia=inertia)

    def _parse_collisions(self, link_prim: Usd.Prim) -> list[Collision]:
        """Parse all collision xforms from a link prim

        Args:
            link_prim: Link prim containing collision xforms

        Returns:
            List of Collision objects
        """

        collisions = []

        collision_xform = link_prim.GetChild("collisions")
        if not collision_xform:
            return collisions

        for prim in Usd.PrimRange(collision_xform, Usd.TraverseInstanceProxies()):
            if prim.HasAPI("PhysicsCollisionAPI"):
                parent_prim = prim.GetParent()
                origin = self._extract_local_transform(parent_prim)

                scale = tuple(parent_prim.GetAttribute("xformOp:scale").Get())
                geometry = self._parse_geometry(prim, scale)

                collision = Collision(
                    name=parent_prim.GetName(),
                    origin=origin,
                    geometry=geometry,
                    **self._get_source_metadata(prim),
                )

                collisions.append(collision)

        return collisions

    def _extract_local_transform(self, prim: Usd.Prim) -> Pose:
        """Extract local transform from prim as Pose

        Args:
            prim: Prim to extract transform from

        Returns:
            Pose with xyz and quat
        """
        xform = UsdGeom.Xformable(prim)
        T = xform.GetLocalTransformation()
        trans = T.ExtractTranslation()
        rot = T.ExtractRotationQuat().GetNormalized()

        xyz = tuple(trans)  # ty: ignore[invalid-argument-type]
        quat = (rot.GetReal(), *rot.GetImaginary())

        return Pose(xyz=xyz, quat=quat)  # ty: ignore[invalid-argument-type]

    def _parse_geometry(self, geom_prim: Usd.Prim, scale: tuple[float, float, float]) -> Geometry | None:
        """Parse geometry from prim

        Args:
            geom_prim: Geometry prim
            scale: (x, y, z) scale from parent transform

        Returns:
            Geometry object, or None if type not supported
        """
        prim_type = geom_prim.GetTypeName()
        if prim_type == "Cylinder":
            cylinder = UsdGeom.Cylinder(geom_prim)

            radius = cylinder.GetRadiusAttr().Get()
            height = cylinder.GetHeightAttr().Get()
            axis = cylinder.GetAxisAttr().Get()

            if axis == "X":
                radius *= max(scale[1], scale[2])
                height *= scale[0]
            elif axis == "Y":
                radius *= max(scale[0], scale[2])
                height *= scale[1]
            else:
                radius *= max(scale[0], scale[1])
                height *= scale[2]

            return Cylinder(radius=radius, length=height, **self._get_source_metadata(geom_prim))

        elif prim_type == "Cube":
            cube = UsdGeom.Cube(geom_prim)
            size = cube.GetSizeAttr().Get()
            box_size = tuple(size * i for i in scale)

            return Box(size=box_size, **self._get_source_metadata(geom_prim))

        elif prim_type == "Sphere":
            sphere = UsdGeom.Sphere(geom_prim)
            radius = sphere.GetRadiusAttr().Get()
            radius *= max(scale)

            return Sphere(radius=radius, **self._get_source_metadata(geom_prim))

        elif prim_type == "Mesh":
            parent_name = geom_prim.GetParent().GetName()
            mesh_name = geom_prim.GetName()
            mesh_id = f"usd:{parent_name}/{mesh_name}"

            return Mesh(filename=mesh_id, scale=scale, **self._get_source_metadata(geom_prim))

        return None

    def _parse_visuals(self, link_prim: Usd.Prim) -> list[Visual]:
        """Parse all visual geometry from a link prim

        Args:
            link_prim: Link prim containing visuals xform

        Returns:
            List of Visual objects
        """
        visuals = []

        visual_xform = link_prim.GetChild("visuals")
        if not visual_xform:
            return visuals

        for prim in Usd.PrimRange(visual_xform, Usd.TraverseInstanceProxies()):
            if UsdGeom.Gprim(prim):
                parent_prim = prim.GetParent()
                origin = self._extract_local_transform(parent_prim)

                scale = tuple(parent_prim.GetAttribute("xformOp:scale").Get())
                geometry = self._parse_geometry(prim, scale)

                if geometry:
                    visual = Visual(origin=origin, geometry=geometry, material=None, **self._get_source_metadata(prim))
                    visuals.append(visual)

        return visuals

    def _parse_joints(self, robot_prim: Usd.Prim) -> dict[str, Joint]:
        """Parse all joint prims

        Args:
            robot_prim: Root robot prim

        Returns:
            Dict mapping joint names to Joint objects
        """
        axis_map = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}
        joints = {}
        has_limits = None

        robot_joints_rel = robot_prim.GetRelationship("isaac:physics:robotJoints")
        if not robot_joints_rel:
            return joints

        joint_targets = robot_joints_rel.GetTargets()

        for joint_path in joint_targets:
            joint_prim = self._stage.GetPrimAtPath(joint_path)
            joint_name = joint_prim.GetName()

            body0_rel = joint_prim.GetRelationship("physics:body0")
            body1_rel = joint_prim.GetRelationship("physics:body1")

            body0_targets = body0_rel.GetTargets()
            body1_targets = body1_rel.GetTargets()

            parent_prim = self._stage.GetPrimAtPath(body0_targets[0])
            child_prim = self._stage.GetPrimAtPath(body1_targets[0])

            parent_name = parent_prim.GetName()
            child_name = child_prim.GetName()

            axis_str = joint_prim.GetAttribute("physics:axis").Get()
            axis = axis_map.get(axis_str)

            pos = joint_prim.GetAttribute("physics:localPos0").Get()
            rot_quat = joint_prim.GetAttribute("physics:localRot0").Get().GetNormalized()

            origin = Pose(
                xyz=tuple(pos),
                quat=(rot_quat.GetReal(), *rot_quat.GetImaginary()),  # ty: ignore[invalid-argument-type]
            )

            type_str = joint_prim.GetTypeName()
            if type_str == "PhysicsRevoluteJoint":
                lower_attr = joint_prim.GetAttribute("physics:lowerLimit")
                upper_attr = joint_prim.GetAttribute("physics:upperLimit")

                has_limits = all((lower_attr, upper_attr, lower_attr.IsAuthored(), upper_attr.IsAuthored()))
                if has_limits:  # Isaac uses degrees
                    joint_type = "revolute"
                    lower = math.radians(lower_attr.Get())
                    upper = math.radians(upper_attr.Get())
                    effort = joint_prim.GetAttribute("drive:angular:physics:maxForce").Get()
                    velocity = math.radians(joint_prim.GetAttribute("physxJoint:maxJointVelocity").Get())
                    limit = Limit(lower=lower, upper=upper, effort=effort, velocity=velocity)
                else:
                    joint_type = "continuous"
                    limit = None

            elif type_str == "PhysicsPrismaticJoint":
                joint_type = "prismatic"
                lower_attr = joint_prim.GetAttribute("physics:lowerLimit")
                upper_attr = joint_prim.GetAttribute("physics:upperLimit")

                has_limits = all((lower_attr, upper_attr, lower_attr.IsAuthored(), upper_attr.IsAuthored()))

                if has_limits:
                    lower = lower_attr.Get()
                    upper = upper_attr.Get()
                    effort = joint_prim.GetAttribute("drive:linear:physics:maxForce").Get()
                    velocity = joint_prim.GetAttribute("physxJoint:maxJointVelocity").Get()
                    limit = Limit(lower=lower, upper=upper, effort=effort, velocity=velocity)
                else:
                    limit = None

            elif type_str == "PhysicsFixedJoint":
                joint_type = "fixed"
                limit = None
            else:
                continue

            joint = Joint(
                name=joint_name,
                type=joint_type,
                parent=parent_name,
                child=child_name,
                origin=origin,
                axis=axis,  # ty: ignore[invalid-argument-type]
                limit=limit,
                **self._get_source_metadata(joint_prim),
            )

            joints[joint_name] = joint

        return joints
