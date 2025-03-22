# MayaMeshRetarget Registration Module Refactoring Plan

## Current Architecture

The registration module is responsible for finding correspondence between source and target meshes, essential for retargeting in Maya. Current structure:

- **alignment.py**: Alignment utilities between meshes and joint hierarchies
- **core.py**: Data classes for registration (CorrespondencePoint, MappingNode, etc.)
- **main.py**: Main registration functionality via MeshRegistration class
- **mapping.py**: Creates mapping points between meshes based on skeleton
- **raycast.py**: Raycasting operations using Embree or fallback methods

## Pain Points

1. **Code Organization**:
   - Duplicated functionality across files (joint matching logic)
   - Some methods are overly complex with multiple responsibilities

2. **Type System**:
   - Inconsistent type annotations (partially implemented)
   - Some legacy data structures using dictionaries instead of typed classes

3. **Error Handling**:
   - Limited graceful fallbacks for edge cases
   - Minimal user feedback on expected failures

4. **Performance**:
   - Potential optimizations for large meshes
   - Better batching for raycast operations

5. **API Design**:
   - MeshRegistration class interface could be more consistent
   - Lack of separation between public and internal methods

## Refactoring Goals

1. **Improve Code Structure**:
   - Establish clear boundaries between modules
   - Extract common utilities to reduce duplication
   - Follow single responsibility principle

2. **Enhance Type Safety**:
   - Complete type annotations for all functions and classes
   - Convert dictionary-based data to proper dataclasses
   - Use more precise types (e.g., np.ndarray[np.float64] vs generic np.ndarray)

3. **Better Error Handling**:
   - Add appropriate validation steps
   - Provide clearer error messages
   - Implement graceful fallbacks

4. **Performance Optimization**:
   - Profile and identify bottlenecks
   - Optimize critical paths (especially raycast)
   - Add progress reporting for long operations

5. **API Refinement**:
   - Clarify public interface vs implementation details
   - Streamline parameter lists
   - Add comprehensive docstrings with examples

## Implementation Plan

### Phase 1: Core Structure and Types

1. **Complete Type Annotations**:
   - Finish adding proper type hints to all functions
   - Use TypedDict where appropriate for dict parameters
   - Ensure consistent typing across modules

2. **Refactor Data Classes**:
   - Move all data classes to core.py
   - Convert remaining dict structures to proper classes
   - Ensure proper initialization and validation

### Phase 2: Module Reorganization

1. **Reduce Duplication**:
   - Move joint matching to a single location
   - Consolidate utility functions

2. **Clarify Responsibilities**:
   - Ensure each module has clear, focused purpose
   - Break down large functions into smaller, testable units
   - Establish clear interfaces between modules

### Phase 3: Performance and Robustness

1. **Performance Improvements**:
   - Profile and optimize critical paths
   - Implement batching for expensive operations
   - Add caching where appropriate

2. **Error Handling**:
   - Add input validation
   - Improve error messages and diagnostics
   - Add graceful fallbacks for common edge cases

### Phase 4: API and Documentation

1. **Clean Public Interface**:
   - Define and document public API
   - Ensure consistent parameter naming
   - Add appropriate defaults

2. **Documentation**:
   - Add examples to docstrings
   - Create higher-level documentation
   - Add visualization helpers for debugging

## Specific Tasks

1. **Core Module**:
   - Complete dataclass definitions with proper types
   - Add validation methods
   - Implement serialization/deserialization where needed

2. **Alignment Module**:
   - Refactor calculate_alignment_transform to use new type system
   - Extract joint-matching logic to separate function
   - Add visualization helpers for debugging

3. **Mapping Module**:
   - Refactor to use proper class structures
   - Add progress reporting for long operations
   - Optimize performance for large meshes

4. **Raycast Module**:
   - Improve Embree integration with better error handling
   - Optimize batch processing
   - Add fallback methods for all edge cases

5. **Main Module**:
   - Refactor MeshRegistration class for better API design
   - Add progress reporting
   - Implement validation and error handling

## Migration Strategy

To ensure compatibility with existing code:
1. Implement changes incrementally with thorough testing
2. Maintain backward compatibility where possible
3. Document breaking changes clearly
4. Add deprecation warnings for functions being refactored

This plan will improve the registration module's maintainability, performance, and user experience while preserving core functionality.