# MayaMeshRetarget
MayaMeshRetarget is a tool for retargeting mesh deformations between 3D character models. It uses a 
combination of Radial Basis Function (RBF) interpolation and skin weight-based clustering to transfer 
complex deformations from a source mesh to a target mesh. The tool 
preserves the inherent rigidity of objects or regions where conventional retargeting methods often 
introduce distortions. In these cases, the method minimizes unwanted deformation—ensuring that rigid 
parts remain stable even when other regions undergo significant transformation. MayaMeshRetarget is 
designed for Autodesk Maya and leverages Maya's OpenMaya API along with scientific libraries 
(NumPy, SciPy, scikit-learn) for efficient computation.

---

## Features

- **RBF-Based Mesh Deformation**  
  Smoothly interpolates deformations from a source mesh to a target mesh—even for vertices that lack a direct counterpart—ensuring a seamless deformation field.

- **Skin Weights & Topology Clustering**  
  Groups vertices influenced by the same bones and similar topology, so rigid or semi-rigid areas (like buttons, jewelry, etc.) maintain their shape integrity during retargeting.

- **Scale Factor Adaptation**  
  Uses PCA-based scaling (both uniform and non-uniform) to handle differences in size or proportion between the source and target. This means larger or smaller limbs get appropriately scaled deformation effects.

- **Distance Matrix Inpainting**  
  Fills in missing or unreliable correspondences, ensuring smooth deformation continuity when certain parts of the mesh have no direct source counterpart.

- **Multiple Object Types**  
  Works with meshes, joints, and generic transform nodes—handling positional, rotational, and scale data when needed.

- **Advanced Registration(WIP)**
  Includes a dedicated registration module for establishing correspondences between meshes that have different vertex counts or topologies.

MayaMeshRetarget is released under the MIT License (see the [LICENSE](LICENSE) file for details). Contributions are welcome—see the **Contributing** section below.

---

## Installation
To get started with MayaMeshRetarget, you’ll need to install the tool and its dependencies. Below are two approaches: one for power users who manage Maya environments extensively, and a simpler one for beginners who just want to get it running quickly.

### **Quick Setup (Less Technical / Beginner Friendly)**

#### **1. Locate Your Maya ‘modules’ Folder**
- Windows:
    `C:\Users\<username>\Documents\maya\<MayaVersion>\modules\`
- macOS:
    `~/Library/Preferences/Autodesk/maya/<MayaVersion>/modules/`
- Linux:
    `~/maya/<MayaVersion>/modules/`

#### **2. Copy the ymt_mesh_retarget Folder**
- Download or clone the MayaMeshRetarget repository.
- Inside that repository, you’ll see a folder named ymt_mesh_retarget/.
- Copy this entire folder into the modules folder from step (1).

#### **3. Automatic Dependency Handling**
- By default, when you import or run ymt_mesh_retarget, the tool checks for required Python libraries:
  - `numpy`, `scipy`, `scikit-learn`, and `embreex`
- If any are missing, it attempts to install them into Maya’s Python environment via `mayapy -m pip install ....`
- If you prefer to manage Python packages manually, install them on your own. In any case, once satisfied, the tool is ready to run.

#### **4. Launch Maya**
- Inside Maya’s Script Editor (Python tab), run:
```python
import ymt_mesh_retarget
ymt_mesh_retarget.show_ui()
```
The MayaMeshRetarget UI should appear if everything is in place.

### **Advanced / Power-User Installation**
For TDs, TAs, riggers, or developers who prefer more control over Maya’s environment setup:

**1. Clone or Download the Repository**
```bash
git clone https://github.com/yamahigashi/MayaMeshRetarget.git
```
(Or download/unzip the latest release.)

**2. Set `MAYA_MODULE_PATH` or Use a Symlink**
- If you manage multiple Maya modules in a custom location, you can place ymt_mesh_retarget/ in any folder referenced by the `MAYA_MODULE_PATH` environment variable.
- Alternatively, create a symlink from ymt_mesh_retarget/ into your existing modules folder to keep it in version control.

**3. Manual Python Dependencies**
- For a fully controlled environment, install numpy, scipy, scikit-learn, and embreex into your Python environment.

**4. Verification**

Open Maya, run:
```python
import ymt_mesh_retarget
ymt_mesh_retarget.show_ui()
```
If the UI launches and no errors appear, you’re all set.

**Note:**
- This method is ideal if you have multiple Maya projects or pipelines and prefer a more explicit environment variable approach.
- You can also embed ymt_mesh_retarget inside your studio’s pipeline structure, referencing it from a central location.


After following either approach, MayaMeshRetarget should be available every time you start Maya. If you encounter issues (e.g., missing libraries, import errors), double-check that you:

- Installed the required Python packages
- Placed the ymt_mesh_retarget folder correctly
- Have no conflicting environment variables in Maya

With the installation complete, you can now begin retargeting deformations or exploring advanced mesh registration features.

---

## Basic Usage

1. **Launch the tool**: 
In Maya's Script Editor, run:
    ```python
    import ymt_mesh_retarget
    ymt_mesh_retarget.show_ui()
    ```
2. **Set objects**:
   - Source: The original, undeformed mesh
   - Target: The deformed version of the source mesh
   - To Retarget: The object(s) to apply the deformation to
3. **Configure settings**:
   - Maintain Rigid: Enable for articulated models to maintain shape integrity
   - Inpaint: Enable to fill in unconvinced vertex matches
   - Sampling Stride: Control vertex sampling density
4. **Execute**: Click "Execute!" to perform the retargeting

The tool will create new retargeted objects with the deformation applied.

---

## Advanced Options

### RBF Kernels

The tool supports multiple RBF kernels for different deformation characteristics:

- **Linear**: Direct linear interpolation
- **Gaussian**: Smooth, local deformations with rapid falloff
- **Thin-Plate**: Surface-like deformations with gradual transitions
- **Multi-Quadratic**: Balanced blend between local and global influence
- **Inverse Multi-Quadratic**: Inverse decay of distances
- **Beckert-Wendland**: Compact support for highly localized deformations

### Inpainting Settings

When inpaint is enabled, you can control:

- **Distance**: Threshold distance coefficient for determining confident matches
- **Angle**: Angle threshold for vertex normal comparison

### Object Handling

The tool automatically handles different object types:
- **Meshes**: Vertex-based deformation transfer
- **Joints**: Position, rotation, and scale transfer with hierarchy preservation
- **Transforms**: Position transfer with optional hierarchy preservation

---

## Troubleshooting

- **Different Vertex Counts**: Source and target meshes must have identical vertex counts
- **Missing Deformation**: Ensure source and target represent the same mesh in different states
- **Skinning Issues**: Both source and target should have proper skin clusters for best results
- **Performance**: For large meshes, increase the sampling stride to improve performance

## Frequently Asked Questions
1. *Can I retarget between meshes with different vertex counts?*

    Yes, but it requires an automatic matching step which can be slower and slightly less accurate. It’s recommended to prepare a “wrap” or base mesh if possible. If you must retarget drastically different meshes, use the registration module first to establish correspondences, then apply RBF.

2. *How can I run it without a GUI?*

    Simply import the relevant modules (e.g., logic.retarget, registration.main.find_correspondence_pairs) and build your script pipeline manually.

---

## Technical Details

### Requirements
- **Autodesk Maya 2022 (or later)**:  – MayaMeshRetarget requires Maya 2022+ (which uses Python 3.x). It has been tested with Maya 2022 and 2023. Earlier versions of Maya (with Python 2) are not supported due to the dependency on Python 3 libraries.
- **Operating System**: – Windows, macOS, or Linux (the tool is written in Python and uses Maya’s API, so it should work on any OS supported by Maya, as long as the dependencies can be installed).
- **Python Libraries**: – The Maya Python environment needs the following libraries:
    - **NumPy** - for efficient array and matrix operations.
    - **SciPy** - for RBF interpolation.
    - **scikit-learn** - for PCA and clustering.
    - **embreex** - for mesh registration.

### Mesh Retargeting Process

1. **Initialization**: Loads source, target, and objects to retarget
2. **RBF Calculation**: Creates RBF weights matrix between source and target
3. **Clustering**: Groups vertices based on skin weights and topology
4. **Transformation**: Applies RBF-based transformation to new objects
5. **Inpainting**: Fills in unconvinced vertex matches (if enabled)
6. **Hierarchy Handling**: Maintains scene hierarchy relationships (if enabled)

### Registration Module

For meshes with different topologies, the registration module can be used to establish correspondences:

```python
from ymt_mesh_retarget.registration import find_correspondence_pairs

# Find correspondence points between two meshes
source_points, target_points = find_correspondence_pairs(
    source_mesh="sourceModel",
    target_mesh="targetModel",
    visualize=True  # Optionally visualize the correspondences
)
```

## Logging

The tool includes a comprehensive logging system. You can adjust the log level:

```python
import ymt_mesh_retarget
ymt_mesh_retarget.set_log_level("DEBUG")  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Directory Structure
```plaintext
ymt_mesh_retarget/
├── __init__.py            # Package initialization
├── cluster.py             # Skin-weight + topology-based vertex clustering
├── inpaint.py             # Inpainting for filling missing data
├── logger.py              # Logging integration with Maya Script Editor
├── logic.py               # Core RBF-based mesh retargeting logic
├── objects/
│   ├── __init__.py
│   ├── base.py
│   ├── joint.py
│   ├── mesh.py
│   └── transform.py
├── registration/          # Correspondence-finding and advanced registration
│   ├── __init__.py
│   ├── alignment.py
│   ├── core.py
│   ├── geometry.py
│   ├── main.py
│   ├── mapping.py
│   ├── raycast.py
│   ├── scoring_components.py
│   ├── spatial.py
│   ├── users_guide.md
│   └── utils.py
├── types.py               # Shared type definitions
├── ui.py                  # PySide/Qt UI for Maya
└── util.py                # Maya OpenMaya and system helper functions
```

### Basic Retargeting

```python
import ymt_mesh_retarget

# Using the API directly
ymt_mesh_retarget.retarget(
    source="source_mesh",
    target="target_mesh",
    objects=["object_to_retarget"],
    kernel="gaussian",
    radius_coefficient=0.005,
    angle=180.0,
    apply_rigid_transform=True,
    inpaint=True
)
```

### Registration Module

For meshes with different topologies, the registration module can be used to establish correspondences:

```python
from ymt_mesh_retarget.registration import find_correspondence_pairs

# Find correspondence points between two meshes
source_points, target_points = find_correspondence_pairs(
    source_mesh="sourceModel",
    target_mesh="targetModel",
    visualize=True  # Optionally visualize the correspondences
)
```

### Custom Registration

```python
from ymt_mesh_retarget.registration import (
    MeshRegistration,
    get_default_registration_options
)

# Customize registration options
options = get_default_registration_options()
options.sample_count = 3000
options.sample_number = 32
options.sample_degree = 45.0

# Create registration object
registration = MeshRegistration("source_mesh", "target_mesh", options)

# Find correspondence pairs
source_points, target_points = registration.find_correspondence_pairs()

# Visualize correspondences
registration.visualize_correspondences()
```

## License

This tool is released under the MIT License.

## Contributing

We welcome contributions to MayaMeshRetarget! Please refer to the [Contribution Guidelines](CONTRIBUTING.md) for more information on how to get involved.
