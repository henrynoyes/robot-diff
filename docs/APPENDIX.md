## Thesis

In modern robotic simulation, robot models can be described in several formats. This includes the Unified Robot Description Format (URDF) and Simulation Description Format (SDF) from the OSRF, MuJoCo Format (MJCF) from Google DeepMind, and Universal Scene Description (USD) from Pixar (see [a brief history](#a-brief-history)).

These formats differ in several ways and each supports features that enable their targeted use case. For example, MJCF looks similar to URDF on the surface, but opts for a hierarchical XML tree structure and introduces additional fields like `<option>` and `<compiler>` that enable the fine-grained control over contact modeling that MuJoCo is known for.

Though these differences exist, there is a subset of core information that is common among the model formats. For a given robot, the kinematics, inertia, and collision primitives should be encoded equivalently in every format. Other information such as visual elements may match across formats as well, although it is more common for them to differ as mesh compatibility varies between simulators.

It is often advantageous, or even necessary, to import a robot model from a specified format. Many of these simulators provide support for importing multiple formats, though often recommend that one specific format is used for all development. In fact, [none of the modern robotics simulators currently support all four major formats](#support). Thus, it is common to see robot description repositories such as [unitree_ros](https://github.com/unitreerobotics/unitree_ros), [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco), and [unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main) that contain multiple formats of a given robot for development in different simulators.

Given this setup, one can imagine a scenario in the development of a robot in which a modification is made to the robot model, and it is desired that this change is reflected in all formats.

One option is to make this change in one format and rely on conversion tools to generate other formats. This *can work*, but it presents a few issues. First, the [current state of conversion tools is fairly messy](#converters). Since each format uses slightly different conventions and data structures, converting between any two formats is not a one-click process. And, in some cases, there does not exist a user-friendly way to convert between formats. Second, the use of a conversion tool abstracts the user from the process, risking a quiet error that can be accepted as correct. This is especially of note for properties such as inertial components, where discrepancies are less easy to spot with visualization tools. Third, the generated files from the conversion tools may not follow the desired conventions, requiring extra effort to "Frankenstein" the output into the correct format.

The other option is to perform the modifications manually, which bypasses the issues of tool configurations, but can require significant manual effort—and, is still prone to user error.

In both options, the key aspect that is lacking is **user verification**. Whether the conversion is performed manually or automatically generated, a layer of validation on the information between models would prove helpful in guaranteeing that the changes were migrated correctly. In the event of an error, this verification step would allow the user to detect the discrepancy before accepting the conversion and potentially branching the asset from its intended state. This is the justification for `robot-diff`.

`robot-diff` is a tool that provides human-readable diffs for various robot model formats, enabling the validation of robot models when developing across multiple robotic simulators. It functions by converting the core robot structure of a URDF/SDF/MJCF/USD file into an internal Pythonic representation. Then, the desired information from the representations is compared and the diffs are printed. Specialized fields such as physics properties, lighting, etc. are not included in the internal representation, and thus not checked for diffs.

### An aside

In my experience, today's LLMs have proven fairly capable at the task of converting between model formats. It is a task that naturally formulates as next-token prediction, and I have observed high success rates in converting robot descriptions—especially when a pair of example files are provided in-context (see [`conversion-examples/`](./conversion-examples)).

However, I am no foreigner to hallucinations and spotted some errors in these conversions on occasion. This led me to the idea for `robot-diff`. I was leveraging some conversion tools in my development, but found myself returning to the LLMs to patch things together in my desired format. Instead of using a one-size-fits-all conversion tool, what I really needed was a simple way to confirm that the converted files contained the correct representation of my robot. Without inspecting the models in a visualizer, I found it tedious to verify simple properties due to syntactic differences in the formats (MJCF uses half-extents for geometries, URDF uses rpy for orientations, etc.). By parsing these files into a common robot representation, I figured it would simplify the process of unifying my models.

### A brief history

The Unified Robot Description Format (URDF) is the oldest of the four major formats. First released in 2009 alongside ROS, URDF is an XML format consisting of simple elements like `<link>` and `<joint>` that define a robot's kinematics. Three years later, in 2012, the Simulation Description Format was released alongside Gazebo 1.0. SDF extends the XML style of URDF to include information about the entire simulation environment, with support for multiple robots, lighting, terrain, and other scene information. In the same year, the MuJoCo XML Format (MJCF) was introduced in the seminal MuJoCo paper at IROS 2012. Much like URDF and SDF, MJCF describes the kinematic tree of a robot with a hierarchy of `<body>` and `<joint>` XML elements, while also providing specialized elements and attributes which enable precise tuning of contact parameters—a core focus for the simulator. The Universal Scene Description (USD) format is the youngest of the four major formats, though it is also the most capable. USD is less of a file format, and more akin to a complete framework that enables non-destructive composition of 3D scenes through a sophisticated layering system. Originally developed by Pixar Animation Studios for internal film production pipelines, USD was open-sourced as OpenUSD in 2016 and has gradually entered the robotic simulation space over time. NVIDIA spearheaded this adoption by selecting USD as the foundational scene description framework in its Omniverse platform. Support for USD in robotics simulators is still in its early stages, as large organizations like NVIDIA and Google DeepMind work to expand its utility through the development of specialized schemas tailored to robotics applications.

## Support

Supported model formats in modern robotics simulators

|                                                                                                                                     | URDF | SDF | MJCF | USD                                                                  |
| ----------------------------------------------------------------------------------------------------------------------------------- | ---- | --- | ---- | -------------------------------------------------------------------- |
| [Gazebo](https://gazebosim.org/docs/latest/comparison/)                                                                             | ☑️   | ✅   | ❌    | ❌                                                                    |
| [MuJoCo](https://mujoco.readthedocs.io/en/stable/modeling.html)                                                                     | ☑️   | ❌   | ✅    | ⚠️ |
| [Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/importers_exporters.html)                           | ☑️   | ❌   | ☑️   | ✅                                                                    |
| [Drake](https://drake.mit.edu/doxygen_cxx/group__multibody__parsing.html)                                                           | ☑️   | ☑️  | ☑️   | ❌                                                                    |
| [Simple](https://github.com/stack-of-tasks/pinocchio?tab=readme-ov-file#pinocchio-main-features)                                    | ☑️   | ☑️  | ☑️   | ❌                                                                    |
| [Genesis](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/hello_genesis.html#load-objects-into-the-scene) | ☑️   | ❌   | ☑️   | ❌                                                                    |
| [Newton](https://newton-physics.github.io/newton/api/_generated/newton.ModelBuilder.html)                                           | ☑️   | ❌   | ☑️   | ☑️                                                                   |

- ✅ - Native format
- ☑️ - Full support
- ⚠️ - Experimental support
- ❌ - No support

## Converters
*: A non-exhaustive list*

### URDF to __

#### URDF to SDF

- [sdformat](https://github.com/gazebosim/sdformat) - `gz sdf -p robot.urdf > robot.sdf`

#### URDF to MJCF

- [mujoco](https://github.com/google-deepmind/mujoco) - `./compile robot_model.urdf robot_model.xml`

- [urdf2mjcf](https://github.com/kscalelabs/urdf2mjcf)

- [Wiki-GRx-MJCF](https://gitee.com/FourierIntelligence/wiki-grx-mjcf)

#### URDF to USD

- [`convert_urdf.py`](https://github.com/isaac-sim/IsaacLab/blob/main/scripts/tools/convert_urdf.py) from Isaac Lab

- [urdf-usd-converter](https://github.com/newton-physics/urdf-usd-converter) from Newton

### SDF to __

#### SDF to URDF

- [sdformat_urdf](https://github.com/ros/sdformat_urdf) from ROS

- `sdf2urdf` from [sdformat_tools](https://github.com/gezp/sdformat_tools)

- [sdf2urdf](https://github.com/theg4sh/sdf2urdf)

#### SDF to MJCF

- `sdf2mjcf` from [gz-mujoco](https://github.com/gazebosim/gz-mujoco/tree/main/sdformat_mjcf#tools-for-converting-sdformat-to-mjcf)

#### SDF to USD

- `sdf2usd` from [gz-usd](https://github.com/gazebosim/gz-usd/blob/main/tutorials/convert_sdf_to_usd.md)


### MJCF to __

#### MJCF to URDF

- [mjcf_urdf_simple_converter](https://github.com/Yasu31/mjcf_urdf_simple_converter)

- [mjcf2urdf](https://github.com/iory/mjcf2urdf)

#### MJCF to SDF

- `mjcf2sdf` from official [gz-mujoco](https://github.com/gazebosim/gz-mujoco/tree/main/sdformat_mjcf#tools-for-converting-mjcf-to-sdformat)

#### MJCF to USD

- [`convert_mjcf.py`](https://github.com/isaac-sim/IsaacLab/blob/main/scripts/tools/convert_mjcf.py) from Isaac Lab

- [mujoco-usd-converter](https://github.com/newton-physics/mujoco-usd-converter) from Newton

### USD to __

#### USD to URDF

- [USD to URDF Export Extension](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/importer_exporter/ext_omni_exporter_urdf.html) from Isaac Lab

#### USD to SDF

- `usd2sdf` from [gz-usd](https://github.com/gazebosim/gz-usd)

#### USD to MJCF

- ?
