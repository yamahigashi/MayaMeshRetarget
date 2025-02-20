# MayaMeshRetarget
MayaMeshRetarget is a tool for retargeting mesh deformations between 3D character models. It uses a 
combination of Radial Basis Function (RBF) interpolation and skin weight-based clustering to transfer 
complex deformations from a source mesh to a target mesh. This is especially useful when two characters 
share a similar rig (skeleton and skin weights) but have different proportions or topology. The tool 
preserves the inherent rigidity of objects or regions where conventional retargeting methods often 
introduce distortions. In these cases, the method minimizes unwanted deformation—ensuring that rigid 
parts remain stable even when other regions undergo significant transformation. MayaMeshRetarget is 
designed for Autodesk Maya and leverages Maya's OpenMaya API along with scientific libraries 
(NumPy, SciPy, scikit-learn) for efficient computation.


## Features
- **Mesh Deformation with RBF Interpolation**:  – Utilizes Radial Basis Functions to smoothly interpolate deformation from source to target, ensuring even vertices with no direct counterpart receive a plausible transformation (for example, wrinkles or muscle bulges are smoothly transferred). This yields a smooth deformation field across the target mesh.
- **Clustering Based on Skin Weights and Topology**: – Vertices are grouped into clusters based on their skin weights (and mesh topology). This means parts of the mesh influenced by the same bones (e.g., an upper arm, a face cheek, etc.) are clustered together. Clustering helps maintain rigid or semi-rigid behavior for those parts and prevents inappropriate mixing of unrelated regions during retargeting. Essentially, the tool knows which parts should move together and which are more flexible.
- **Scale Factor Adaptation**: –The tool computes uniform and non-uniform scale factors for each cluster using Principal Component Analysis (PCA). This allows it to handle characters of different sizes or proportions. For example, if the target character’s arm is thicker or longer than the source’s, MayaMeshRetarget calculates scale adjustments so that the deformation (like a muscle bulge or bend) is scaled appropriately for the larger or smaller limb. This helps maintain the correct look of the deformation on the target.
- **Distance Matrix Inpainting**: – If there are missing or unreliable correspondence data for some vertices, the tool uses an inpainting technique to fill those gaps. In practice, this means if some parts of the target mesh don’t have a clear counterpart on the source (or vice versa), the algorithm interpolates using neighboring information to ensure a continuous deformation. This leads to more consistent results and avoids holes or irregularities in the transferred deformation.

MayaMeshRetarget is released under the MIT License (see the LICENSE file for details). Contributions are welcome – see the Contributing Guide below.

## Installation
To get started with MayaMeshRetarget, you’ll need to install the tool and its dependencies. This section provides step-by-step instructions for setting up the environment.

### Requirements
- **Autodesk Maya 2022 (or later)**:  – MayaMeshRetarget requires Maya 2022+ (which uses Python 3.x). It has been tested with Maya 2022 and 2023. Earlier versions of Maya (with Python 2) are not supported due to the dependency on Python 3 libraries.
- **Operating System**: – Windows, macOS, or Linux (the tool is written in Python and uses Maya’s API, so it should work on any OS supported by Maya, as long as the dependencies can be installed).
- **Python Libraries**: – The Maya Python environment (a.k.a. mayapy) needs the following libraries:
    - **NumPy** - for efficient array and matrix operations.
    - **SciPy** - for RBF interpolation.
    - **scikit-learn** - for PCA and clustering.

### Installation steps
1. **Download the Repository**: – You can obtain the tool by cloning the GitHub repository or downloading a release zip:
    - **Download Zip**: Go to the [Releases]() page and download the latest release as a zip file. Extract the contents to a folder on your computer.
    - **Clone Repository**: If you have Git installed, you can clone the repository using the following command:
    ```bash
    git clone https://github.com/yamahigashi/MayaMeshRetarget.git
    ```

2. **Choose Installation Location**: - TODO write later


### Usage
TODO write later



### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


### Contributing
We welcome contributions to this project. Please follow the standard GitHub workflow for contributing to this repository.
