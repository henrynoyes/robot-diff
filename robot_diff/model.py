from dataclasses import dataclass, field

__all__ = [
    "Robot",
    "Link",
    "Joint",
    "Pose",
    "Base",
    "Geometry",
    "Box",
    "Cylinder",
    "Sphere",
    "Mesh",
    "Inertia",
    "Inertial",
    "Collision",
    "Material",
    "Visual",
    "Limit",
]


@dataclass
class Base:
    """Base class for objects with source tracking metadata

    Attributes:
        line_number: Line number of element in source file
        source_path: Hierarchical path to element in source format
        source_file: Path to source file
    """

    _line_number: int | None = field(default=None, repr=False, compare=False, kw_only=True)
    _source_path: str | None = field(default=None, repr=False, compare=False, kw_only=True)
    _source_file: str | None = field(default=None, repr=False, compare=False, kw_only=True)


@dataclass
class Pose(Base):
    """Position and orientation in SE(3)

    Attributes:
        xyz: (x, y, z) position in meters, defaults to (0.0, 0.0, 0.0)
        quat: (w, x, y, z) unit quaternion in radians, defaults to (1.0, 0.0, 0.0, 0.0)
    """

    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass
class Geometry(Base):
    """Base class for geometric shapes"""

    pass


@dataclass
class Box(Geometry):
    """Box geometry

    Attributes:
        size: (x, y, z) dimensions in meters
    """

    size: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Cylinder(Geometry):
    """Cylinder geometry

    Attributes:
        radius: Radius in meters
        length: Length in meters
    """

    radius: float = 0.0
    length: float = 0.0


@dataclass
class Sphere(Geometry):
    """Sphere geometry

    Attributes:
        radius: Radius in meters
    """

    radius: float = 0.0


@dataclass
class Mesh(Geometry):
    """Mesh geometry

    Attributes:
        filename: URI to mesh file
        scale: (x, y, z) scale factors, defaults to (1.0, 1.0, 1.0)
    """

    filename: str = ""
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class Inertia(Base):
    """Inertia tensor

    Attributes:
        ixx, ixy, ixz, iyy, iyz, izz: Components of the 3x3 symmetric inertia tensor in kg*m^2
    """

    ixx: float = 0.0
    ixy: float = 0.0
    ixz: float = 0.0
    iyy: float = 0.0
    iyz: float = 0.0
    izz: float = 0.0


@dataclass
class Inertial(Base):
    """Inertial properties of a link

    Attributes:
        origin: Pose of inertial frame w.r.t. link frame, defaults to the identity
        mass: Mass in kilograms
        inertia: Inertia tensor
    """

    origin: Pose = field(default_factory=Pose)
    mass: float = 0.0
    inertia: Inertia = field(default_factory=Inertia)


@dataclass
class Collision(Base):
    """Collision geometry of a link

    Attributes:
        name: Optional name of the collision element
        origin: Pose of collision geometry w.r.t. link frame, defaults to the identity
        geometry: Geometric shape for collision checking
    """

    name: str | None = None
    origin: Pose = field(default_factory=Pose)
    geometry: Geometry | None = None


@dataclass
class Material(Base):
    """Material properties for visual elements

    Attributes:
        name: Name of the material, defaults to None if not specified
        rgba: (r, g, b, a) color values from 0-1, defaults to None if not specified
        texture_filename: URI to texture file, defaults to None if not specified
    """

    name: str | None = None
    rgba: tuple[float, float, float, float] | None = None
    texture_filename: str | None = None


@dataclass
class Visual(Base):
    """Visual geometry of a link

    Attributes:
        origin: Pose of visual geometry w.r.t. link frame, defaults to the identity
        geometry: Geometric shape for visualization
        material: Material properties, defaults to None if not specified
    """

    origin: Pose = field(default_factory=Pose)
    geometry: Geometry | None = None
    material: Material | None = None


@dataclass
class Limit(Base):
    """Joint limits

    Attributes:
        lower: Lower joint limit (radians for revolute, meters for prismatic)
        upper: Upper joint limit (radians for revolute, meters for prismatic)
        effort: Maximum joint effort (torque for revolute, force for prismatic)
        velocity: Maximum joint velocity (rad/s for revolute, m/s for prismatic)
    """

    lower: float = 0.0
    upper: float = 0.0
    effort: float = 0.0
    velocity: float = 0.0


@dataclass
class Joint(Base):
    """Joint connecting two links

    Attributes:
        name: Name of the joint
        type: Type of joint (revolute, prismatic, fixed, continuous, floating, planar)
        parent: Name of the parent link
        child: Name of the child link
        origin: Pose of child link frame w.r.t. parent link frame, defaults to the identity
        axis: (x, y, z) axis of actuation expressed in the child link frame, defaults to (1.0, 0.0, 0.0)
    """

    name: str
    type: str
    parent: str
    child: str
    origin: Pose = field(default_factory=Pose)
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    limit: Limit | None = None


@dataclass
class Link(Base):
    """Robot link

    Attributes:
        name: Name of the link
        inertial: Inertial properties, defaults to None
        collisions: List of Collision objects, defaults to empty list
        visuals: List of Visual objects, defaults to empty list
    """

    name: str
    inertial: Inertial | None = None
    collisions: list[Collision] = field(default_factory=list)
    visuals: list[Visual] = field(default_factory=list)


@dataclass
class Robot:
    """Abstract robot representation

    Attributes:
        name: Name of the robot
        links: Dict mapping link names to Link objects, defaults to empty dict
        joints: Dict mapping joint names to Joint objects, defaults to empty dict
    """

    name: str
    links: dict[str, Link] = field(default_factory=dict)
    joints: dict[str, Joint] = field(default_factory=dict)
