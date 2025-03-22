# -*- coding: utf-8 -*-
"""
Centralized logging module for ymt_mesh_retarget package.

This module provides a unified logging framework for the entire package,
ensuring consistent log formatting and control across all components.
"""

import os
import sys
import logging
from maya import cmds
from maya.api import OpenMaya as om

# Default logging level (can be overridden with environment variable)
DEFAULT_LOG_LEVEL = os.environ.get('YMT_LOG_LEVEL', 'INFO')

# Initialize logger
logger = logging.getLogger("ymt_mesh_retarget")

# Map string level names to logging constants
level_map = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}
logger.setLevel(level_map.get(DEFAULT_LOG_LEVEL, logging.INFO))

# Clear existing handlers (to prevent duplicate handlers when reloading)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Console output handler
console = logging.StreamHandler(sys.stdout)
console.setLevel(logger.level)

# Maya Script Editor handler
class MayaScriptEditorHandler(logging.Handler):
    """Maya Script Editor handler to display messages in Maya's script editor."""
    
    def emit(self, record):
        msg = self.format(record)
        if record.levelno >= logging.ERROR:
            om.MGlobal.displayError(msg)
        elif record.levelno >= logging.WARNING:
            om.MGlobal.displayWarning(msg)
        else:
            om.MGlobal.displayInfo(msg)

# Configure formatter
formatter = logging.Formatter('[%(levelname)s | %(name)s | %(asctime)s] %(message)s', 
                             datefmt='%H:%M:%S')
console.setFormatter(formatter)

# Create and configure Maya handler
maya_handler = MayaScriptEditorHandler()
maya_handler.setLevel(logger.level)
maya_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console)
logger.addHandler(maya_handler)

def set_log_level(level):
    """Dynamically change the logging level.
    
    Args:
        level: Either a string ('DEBUG', 'INFO', etc.) or a logging level constant
            (logging.DEBUG, logging.INFO, etc.)
            
    Returns:
        The numeric logging level that was set
    """
    if isinstance(level, str):
        level = level_map.get(level.upper(), logging.INFO)
    
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
    
    logger.info(f"Log level set to: {logging.getLevelName(level)}")
    return level