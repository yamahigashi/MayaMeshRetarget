# MayaMeshRetarget Development Guidelines

## Build & Development
- **Running in Maya**: Import `ymt_mesh_retarget` package and use `show_ui()` function
- **Lint**: `ruff python/`
- **Type Check**: `pyright python/`

## Code Style
- **Line Length**: 120 characters max
- **Formatting**: Use Black formatter (configured in pyproject.toml)
- **Type Hints**: Use Python type annotations where possible
- **Imports**: Group imports by stdlib, third-party, local; pyproject.toml disables import order warnings
- **Naming**: 
  - Use snake_case for variables, functions, methods
  - Use CamelCase for classes
- **Error Handling**: Use appropriate try/except blocks with meaningful error messages
- **Docstrings**: Include docstrings for public functions, classes, and methods
- **Comments**: Use comments for complex algorithms, especially in math-heavy sections

## Maya Specifics
- Use OpenMaya API (om) for mesh operations
- Properly handle DAG paths and dependency nodes
- Be mindful of Maya's reference counting and garbage collection
