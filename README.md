# MayaMeshRetarget
This project provides a tool for retargeting mesh deformation using Radial Basis Functions (RBF) and skin weight-based clustering. It is primarily designed for use in Autodesk Maya, and uses both Maya's OpenMaya API and SciPy for efficient mesh manipulation.



## Features
- **Mesh Deformation with RBF Interpolation**: for smooth mesh deformation.
- **Clustering Based on Skin Weights and Topology**: Maintains rigid deformation for parts of the mesh that should remain stiff during transformation.
- **Scale Factor Calculation**: Uses PCA to compute both uniform and non-uniform scale factors for cluster deformation.
- **Distance Matrix Inpainting**: Fills missing or unreliable vertex data using inpainting techniques for consistent mesh deformation results.

## Installation

To use this tool, you need to have Autodesk Maya installed, as well as the following dependencies:

### Requirements
- Autodesk Maya 2022+
- SciPy
- NumPy
- scikit-learn (for PCA analysis)

### Installation


### Usage


### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


### Contributing
We welcome contributions to this project. Please follow the standard GitHub workflow for contributing to this repository.
